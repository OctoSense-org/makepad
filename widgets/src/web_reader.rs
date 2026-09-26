//! A web page inside a script app: the platform's own web view (WKWebView,
//! Android's WebView) glued to this widget's rect, driven from script.
//!
//! ```text
//! reader := WebReader{width: Fill height: Fill on_error: || show(ui.reader.error())}
//! ui.reader.open("https://example.org/story")   // true when it opened
//! ui.reader.close()
//! ui.reader.is_open()
//! ui.reader.error()                              // why the last page failed
//! ```
//!
//! The page gets no bridge into the app: it is a page, not part of the
//! program. What may be opened answers to the app's policy: a host on its
//! list, or with the `web` grant any public https page, followed through its
//! own links and redirects (a feed's story link is often a redirector). An app
//! without `web` gets a view that stays on the page it opened.
//!
//! The web view sits over the window, so it only moves or hides when this
//! widget draws or is told to. While a page is open a watchdog asks for a
//! redraw every half second and hides the overlay when the last one did not
//! draw this widget (its tile went off screen); the next draw puts it back.
//! Every web view an isolate opened is closed when the isolate is collected,
//! so an app that closes cannot leave a page on the window.
use crate::{
    makepad_derive_widget::*,
    makepad_draw::*,
    makepad_script::ScriptFnRef,
    view::View,
    widget::*,
    widget_async::{CxWidgetToScriptCallExt, ScriptAsyncResult},
};
use std::collections::HashMap;

const WATCHDOG_SECONDS: f64 = 0.5;

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.*

    mod.widgets.WebReaderBase = #(WebReader::register_widget(vm))
    mod.widgets.WebReader = set_type_default() do mod.widgets.WebReaderBase{
        width: Fill
        height: Fill
    }
}

thread_local! {
    /// heap key -> the web views that heap's readers spawned.
    static OPENED: std::cell::RefCell<HashMap<usize, Vec<SystemBrowserId>>> = Default::default();
}

/// Close every web view opened by readers in these (dead) heaps.
pub(crate) fn gc_web_readers(cx: &mut Cx, dead_heaps: &[usize]) {
    let ids: Vec<SystemBrowserId> = OPENED.with(|o| {
        let mut o = o.borrow_mut();
        dead_heaps.iter().filter_map(|heap| o.remove(heap)).flatten().collect()
    });
    for id in ids {
        let mut browser = cx.system_browser(id);
        browser.detach();
        browser.close();
    }
}

#[derive(Script, ScriptHook, Widget)]
pub struct WebReader {
    #[deref]
    view: View,
    /// Called when a page is refused or fails to load; `error()` says why.
    #[live]
    on_error: ScriptFnRef,

    #[rust]
    last_error: String,
    #[rust]
    url: Option<String>,
    #[rust]
    spawned: bool,
    #[rust]
    open: bool,
    #[rust]
    overlay_visible: bool,
    #[rust]
    timer: Option<Timer>,
    #[rust]
    drawn: bool,
    #[rust]
    failed: bool,
}

impl WebReader {
    /// Each reader owns its web view: its uid is unique in the process.
    fn browser_id(&self) -> SystemBrowserId {
        SystemBrowserId(LiveId(self.widget_uid().0))
    }

    fn heap_key(&self) -> usize {
        self.view.source.heap_key()
    }

    fn report(&mut self, cx: &mut Cx, why: String) {
        log!("web reader: {why}");
        self.last_error = why;
        if self.on_error.as_object() != ScriptObject::ZERO {
            cx.widget_to_script_call(self.widget_uid(), NIL, self.view.source.clone(), self.on_error.clone(), &[]);
        }
    }

    /// Open `url`, when this app may. Returns whether it opened.
    pub fn open(&mut self, cx: &mut Cx, url: &str) -> bool {
        let heap = self.heap_key();
        if !crate::splash_policy::page_allowed(heap, url) {
            self.report(cx, format!("refused {url}: not on this app's host list, and no `web` grant"));
            return false;
        }
        // Navigable only for an app that may reach any public page: its
        // links and redirects stay inside what it was granted.
        let navigable = !crate::splash_policy::is_enforced(heap)
            || crate::splash_policy::service_allowed(heap, "web.open").is_ok();
        let id = self.browser_id();
        if navigable {
            cx.system_browser(id).spawn_navigable(url);
        } else {
            cx.system_browser(id).spawn(url);
        }
        if !self.spawned {
            OPENED.with(|o| o.borrow_mut().entry(heap).or_default().push(id));
        }
        self.spawned = true;
        self.failed = false;
        self.url = Some(url.to_string());
        self.open = true;
        self.overlay_visible = true;
        self.drawn = true;
        if self.timer.is_none() {
            self.timer = Some(cx.start_interval(WATCHDOG_SECONDS));
        }
        self.view.redraw(cx);
        true
    }

    pub fn close(&mut self, cx: &mut Cx) {
        if !self.open {
            return;
        }
        self.open = false;
        self.hide_overlay(cx);
        if let Some(timer) = self.timer.take() {
            cx.stop_timer(timer);
        }
        self.view.redraw(cx);
    }

    fn hide_overlay(&mut self, cx: &mut Cx) {
        if self.spawned && self.overlay_visible {
            let area = self.view.area();
            cx.system_browser(self.browser_id()).update(area, false);
        }
        self.overlay_visible = false;
        self.drawn = false;
    }
}

impl Widget for WebReader {
    fn script_call(&mut self, vm: &mut ScriptVm, method: LiveId, args: ScriptValue) -> ScriptAsyncResult {
        if method == live_id!(open) {
            let mut url = None;
            if let Some(args_obj) = args.as_object() {
                let trap = vm.bx.threads.cur().trap.pass();
                let value = vm.bx.heap.vec_value(args_obj, 0, trap);
                url = vm.bx.heap.cast_to_owned_string(value, "copying a reader url");
            }
            let Some(url) = url else { return ScriptAsyncResult::Return(FALSE) };
            let opened = vm.with_cx_mut(|cx| self.open(cx, url.trim()));
            return ScriptAsyncResult::Return(opened.into());
        }
        if method == live_id!(close) {
            vm.with_cx_mut(|cx| self.close(cx));
            return ScriptAsyncResult::Return(NIL);
        }
        if method == live_id!(is_open) {
            return ScriptAsyncResult::Return(self.open.into());
        }
        if method == live_id!(error) {
            return ScriptAsyncResult::Return(vm.bx.heap.new_string_from_str(&self.last_error));
        }
        if method == live_id!(url) {
            let url = vm.bx.heap.new_string_from_str(self.url.as_deref().unwrap_or(""));
            return ScriptAsyncResult::Return(url);
        }
        self.view.script_call(vm, method, args)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        self.view.handle_event(cx, event, scope);
        if self.timer.is_some_and(|timer| timer.is_event(event).is_some()) {
            // Drawn since the last tick: ask for the redraw the next one
            // looks for. Not drawn: the tile is off screen; hide.
            if self.drawn {
                self.drawn = false;
                self.view.redraw(cx);
            } else {
                self.hide_overlay(cx);
            }
        }
        if let Event::Actions(actions) = event {
            for action in actions {
                if let Some(err) = action.downcast_ref::<crate::makepad_draw::makepad_platform::event::NativeSystemBrowserPageError>() {
                    if err.browser_id == self.browser_id().0.get_value() && self.open {
                        self.failed = true;
                        self.hide_overlay(cx);
                        let why = format!("could not load {}: {}", self.url.as_deref().unwrap_or(""), err.description);
                        self.report(cx, why);
                    }
                }
            }
        }
    }

    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        let step = self.view.draw_walk(cx, scope, walk);
        if step.is_done() && self.spawned && (self.open || self.overlay_visible) {
            let show = self.open && !self.failed;
            let area = self.view.area();
            cx.system_browser(self.browser_id()).update(area, show);
            if show {
                self.overlay_visible = true;
                self.drawn = true;
            }
        }
        step
    }
}
