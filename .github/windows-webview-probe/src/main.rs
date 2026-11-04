pub use makepad_widgets;
use makepad_widgets::*;
use makepad_widgets::makepad_platform::event::NativeSystemBrowserNavigation;

app_main!(App);
const BROWSER: LiveId = live_id!(windows_webview_smoke);
const LOCKED: LiveId = live_id!(windows_webview_locked_smoke);
const FAILED: LiveId = live_id!(windows_webview_failure_smoke);
const FIRST: &str = "data:text/html,%3Ctitle%3Ewv2-start%3C/title%3E%3Cp%3EFirst%3C/p%3E";
const SECOND: &str = "data:text/html,%3Ctitle%3Ewv2-second%3C/title%3E%3Cp%3ESecond%3C/p%3E";
const LOCKED_PAGE: &str = "data:text/html,%3Ctitle%3Ewv2-locked%3C/title%3E%3Cp%3ELocked%3C/p%3E";

script_mod! {
    use mod.prelude.widgets.*
    startup() do #(App::script_component(vm)){
        ui: Root {
            first_window := Window {
                window.inner_size: vec2(400, 280)
                body +: { first_surface := View {width: Fill height: Fill} }
            }
            second_window := Window {
                window.inner_size: vec2(400, 280)
                body +: { second_surface := View {width: Fill height: Fill} }
            }
        }
    }
}

#[derive(Script, ScriptHook)]
pub struct App {
    #[live] ui: WidgetRef,
    #[rust] phase: u8,
    #[rust] timeout: Timer,
    #[rust] reattach_timer: Timer,
    #[rust] locked_started: bool,
    #[rust] locked_ok: bool,
    #[rust] error_ok: bool,
    #[rust] finished: bool,
    #[rust] invalid_profile_file: Option<std::path::PathBuf>,
}

impl App {
    fn fail(&mut self, message: &str) {
        if !self.finished {
            self.finished = true;
            eprintln!("WV2_FAIL {message}");
        }
    }
    fn maybe_pass(&mut self, cx: &mut Cx) {
        if !self.finished && self.phase == 8 && self.locked_ok && self.error_ok {
            self.finished = true;
            println!("WV2_PASS reparenting retained state; back/forward, navigation policy, and initialization failure passed");
            cx.system_browser(BROWSER).close();
            cx.system_browser(LOCKED).close();
            cx.system_browser(FAILED).close();
            if let Some(path) = self.invalid_profile_file.take() { let _ = std::fs::remove_file(path); }
        }
    }
}

impl MatchEvent for App {
    fn handle_startup(&mut self, cx: &mut Cx) {
        self.phase = 1;
        self.timeout = cx.start_timeout(60.0);
        cx.system_browser(BROWSER).spawn_navigable(FIRST);
        cx.system_browser(LOCKED).spawn(LOCKED_PAGE);
        let file = std::env::temp_dir().join(format!("makepad-webview2-smoke-{}", std::process::id()));
        if let Err(error) = std::fs::write(&file, b"not a directory") {
            self.fail(&format!("cannot prepare invalid profile: {error}"));
            return;
        }
        cx.system_browser(FAILED).set_data_directory(file.join("profile"));
        cx.system_browser(FAILED).spawn_navigable(FIRST);
        self.invalid_profile_file = Some(file);
    }

    fn handle_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        for action in actions {
            let Some(nav) = action.downcast_ref::<NativeSystemBrowserNavigation>() else { continue };
            if nav.browser_id == FAILED.0 {
                if nav.error.is_some() { self.error_ok = true; }
            } else if nav.browser_id == LOCKED.0 {
                if let Some(error) = &nav.error {
                    self.fail(&format!("locked browser: {error}"));
                } else if !self.locked_started && !nav.loading && nav.title == "wv2-locked" {
                    self.locked_started = true;
                    cx.system_browser(LOCKED).eval_js("location.href='https://example.com/'; setTimeout(()=>document.title='wv2-locked-retained', 1000);");
                } else if nav.title == "wv2-locked-retained" {
                    self.locked_ok = true;
                }
            } else if nav.browser_id == BROWSER.0 {
                if let Some(error) = &nav.error {
                    self.fail(&format!("navigable browser: {error}"));
                } else if self.phase == 1 && !nav.loading && nav.title == "wv2-start" {
                    self.phase = 2;
                    cx.system_browser(BROWSER).eval_js("window.__makepad_test=41; document.title='wv2-ready';");
                } else if self.phase == 2 && nav.title == "wv2-ready" {
                    self.phase = 3;
                    cx.system_browser(BROWSER).detach();
                    self.ui.window(cx, ids!(first_window)).close(cx);
                    cx.redraw_all();
                    self.reattach_timer = cx.start_timeout(1.0);
                } else if self.phase == 4 && nav.title == "wv2-state-41" {
                    self.phase = 5;
                    cx.system_browser(BROWSER).set_url(SECOND, false);
                } else if self.phase == 5 && !nav.loading && nav.title == "wv2-second" {
                    self.phase = 6;
                    cx.system_browser(BROWSER).history_go(-1);
                } else if self.phase == 6 && !nav.loading && (nav.title == "wv2-start" || nav.title == "wv2-state-41") {
                    self.phase = 7;
                    cx.system_browser(BROWSER).history_go(1);
                } else if self.phase == 7 && !nav.loading && nav.title == "wv2-second" {
                    self.phase = 8;
                }
            }
            self.maybe_pass(cx);
        }
    }
}

impl AppMain for App {
    fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
        makepad_widgets::script_mod(vm);
        self::script_mod(vm)
    }
    fn handle_event(&mut self, cx: &mut Cx, event: &Event) {
        self.match_event(cx, event);
        self.ui.handle_event(cx, event, &mut Scope::empty());
        if self.timeout.is_event(event).is_some() && !self.finished {
            self.fail(&format!("timeout at phase {}, policy={}, error={}", self.phase, self.locked_ok, self.error_ok));
        }
        if self.reattach_timer.is_event(event).is_some() && self.phase == 3 {
            self.phase = 4;
            cx.system_browser(BROWSER).eval_js("document.title='wv2-state-'+window.__makepad_test;");
        }
        if matches!(event, Event::Draw(_)) && !self.finished {
            let surface = if self.phase < 3 { ids!(first_surface) } else { ids!(second_surface) };
            let area = self.ui.view(cx, surface).area();
            cx.system_browser(BROWSER).update(area, true);
        }
    }
}
