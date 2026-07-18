// WebCard — renders an LLM-generated, self-contained HTML document ("web app
// card") in a native system WebView overlaid on the GL surface, via the
// SystemBrowser platform ops (WKWebView on macOS/iOS, android.webkit.WebView on
// Android).
//
// Streaming contract: like Splash, the markdown widget streams a growing
// ```runhtml fence by calling set_text() with the FULL body every reparse.
// Reloading a WebView per chunk would thrash it, so the card settles first: the
// document is (re)loaded only after the body has been stable for
// SETTLE_SECONDS. Until then the widget draws its placeholder background.
//
// Overlay lifecycle: ONE live web card at a time, under the well-known
// `web_card_browser_id()`. A new card (or a refinement) reuses the same
// overlay. Teardown is owned by the app shell (foreground switch / chat
// clear) — widgets dropped with their draw tree receive no further events
// and cannot clean up themselves, and "not drawn lately" is NOT a liveness
// signal in a retained-mode renderer.

use crate::{makepad_derive_widget::*, makepad_draw::*, widget::*};

/// The octos web-widget kit (the web counterpart of Splash `glass.*`), injected
/// into EVERY web card so cards compose from `octos.*` instead of hand-rolling.
/// Componentized into layers: `core` (domain-agnostic primitives every card
/// uses) + domain kits built on it (`media` = YouTube widgets, `finance` =
/// data-bound `octos.stock`). Load order matters — core defines the shared
/// theme/state/icons/http the domain kits reference. Bundled once (like MEMORY).
const OCTOS_CORE: &str = include_str!("octos_core.js");
const OCTOS_MEDIA: &str = include_str!("octos_media.js");
const OCTOS_FINANCE: &str = include_str!("octos_finance.js");
const OCTOS_WEATHER: &str = include_str!("octos_weather.js");

/// Insert the kit as `<script>`s at the start of the document head so
/// `window.octos` exists before the card's own script runs. Core first.
fn inject_widget_kit(html: &str) -> String {
    let tag = format!(
        "<script>{}</script><script>{}</script><script>{}</script><script>{}</script>",
        OCTOS_CORE, OCTOS_MEDIA, OCTOS_FINANCE, OCTOS_WEATHER
    );
    if let Some(i) = html.find("<head>") {
        let at = i + "<head>".len();
        let mut out = String::with_capacity(html.len() + tag.len());
        out.push_str(&html[..at]);
        out.push_str(&tag);
        out.push_str(&html[at..]);
        out
    } else if let Some(i) = html.find("<html>") {
        let at = i + "<html>".len();
        let mut out = String::with_capacity(html.len() + tag.len());
        out.push_str(&html[..at]);
        out.push_str(&tag);
        out.push_str(&html[at..]);
        out
    } else {
        format!("{}{}", tag, html)
    }
}

script_mod! {
    use mod.prelude.widgets_internal.*

    mod.widgets.WebCardBase = #(WebCard::register_widget(vm))

    mod.widgets.WebCard = set_type_default() do mod.widgets.WebCardBase{
        width: Fill
        height: Fill
        draw_bg +: {
            color: #x101418
        }
    }
}

/// The single shared overlay id for web app cards. The app shell can call
/// `cx.system_browser(web_card_browser_id()).detach()` when the web card
/// leaves the screen (app switch, chat clear).
pub fn web_card_browser_id() -> SystemBrowserId {
    SystemBrowserId(live_id!(octos_web_card))
}

const SETTLE_SECONDS: f64 = 0.35;

#[derive(Script, ScriptHook, Widget)]
pub struct WebCard {
    #[uid]
    uid: WidgetUid,
    #[source]
    source: ScriptObjectRef,
    #[walk]
    walk: Walk,
    #[redraw]
    #[live]
    draw_bg: DrawColor,
    /// Document origin for the inline HTML (relative fetches + embed referers
    /// resolve against this instead of about:blank).
    #[live]
    base_url: String,
    #[visible]
    #[live(true)]
    visible: bool,

    #[rust]
    html: String,
    #[rust]
    loaded_html: String,
    #[rust]
    spawned: bool,
    #[rust]
    settle_timer: Timer,
}

impl WebCard {
    fn browser_id(&self) -> SystemBrowserId {
        web_card_browser_id()
    }

    fn base_url_or_default(&self) -> &str {
        if self.base_url.is_empty() {
            "https://octos-one.app/"
        } else {
            &self.base_url
        }
    }

    fn load_settled(&mut self, cx: &mut Cx) {
        if self.html.is_empty() || self.html == self.loaded_html {
            return;
        }
        let id = self.browser_id();
        // DEBUG probe: a body of `URLTEST:<url>` navigates instead of loading
        // inline HTML — used to isolate loadHTMLString from the overlay path.
        if let Some(url) = self.html.trim().strip_prefix("URLTEST:") {
            let url = url.trim().to_string();
            if !self.spawned {
                cx.system_browser(id).spawn(&url);
                self.spawned = true;
            } else {
                cx.system_browser(id).set_url(&url, false);
            }
            self.loaded_html = self.html.clone();
            self.redraw(cx);
            return;
        }
        if !self.spawned {
            cx.system_browser(id).spawn("about:blank");
            self.spawned = true;
        }
        let html = inject_widget_kit(&self.html);
        let base = self.base_url_or_default().to_string();
        cx.system_browser(id).set_html(&html, &base);
        self.loaded_html = self.html.clone();
        self.redraw(cx);
    }
}

impl Widget for WebCard {
    fn handle_event(&mut self, cx: &mut Cx, event: &Event, _scope: &mut Scope) {
        if self.settle_timer.is_event(event).is_some() {
            self.load_settled(cx);
        }
        // NOTE: no draw-based liveness watchdog here — makepad draws on demand,
        // so "no draw since last frame" does NOT mean the widget left the
        // screen. Overlay teardown is owned by the app shell (foreground
        // switch / chat clear), which knows when the card actually goes away.
    }

    fn draw_walk(&mut self, cx: &mut Cx2d, _scope: &mut Scope, walk: Walk) -> DrawStep {
        if !self.visible {
            return DrawStep::done();
        }
        let rect = cx.walk_turtle(walk);
        self.draw_bg.draw_abs(cx, rect);

        // Keep the native overlay glued to this rect while we are drawn.
        let area = self.draw_bg.area();
        let has_doc = !self.loaded_html.is_empty();
        if self.spawned || has_doc {
            cx.system_browser(self.browser_id()).update(area, has_doc);
        }
        DrawStep::done()
    }

    fn text(&self) -> String {
        self.html.clone()
    }

    fn set_text(&mut self, cx: &mut Cx, v: &str) {
        if self.html == v {
            return;
        }
        self.html.clear();
        self.html.push_str(v);
        // Debounce (re)loads while the fence is still streaming: reload only
        // once the body has stopped changing for SETTLE_SECONDS.
        self.settle_timer = cx.start_timeout(SETTLE_SECONDS);
        self.redraw(cx);
    }
}
