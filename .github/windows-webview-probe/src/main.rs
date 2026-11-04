pub use makepad_widgets;
use makepad_widgets::*;
use makepad_widgets::makepad_platform::event::NativeSystemBrowserNavigation;

app_main!(App);
const BROWSER: LiveId = live_id!(windows_webview_smoke);

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
}

impl MatchEvent for App {
    fn handle_startup(&mut self, cx: &mut Cx) {
        self.phase = 1;
        self.timeout = cx.start_timeout(45.0);
        cx.system_browser(BROWSER).spawn_navigable("data:text/html,%3Ctitle%3Ewv2-start%3C/title%3E%3Cp%3EWindows%20WebView2%3C/p%3E");
    }

    fn handle_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        for action in actions {
            let Some(navigation) = action.downcast_ref::<NativeSystemBrowserNavigation>() else { continue };
            if navigation.browser_id != BROWSER.0 { continue; }
            if let Some(error) = &navigation.error {
                eprintln!("WV2_FAIL navigation: {error}");
                self.phase = 6;
                return;
            }
            if self.phase == 1 && !navigation.loading && navigation.title == "wv2-start" {
                self.phase = 2;
                cx.system_browser(BROWSER).eval_js("window.__makepad_test=41; document.title='wv2-ready';");
            } else if self.phase == 2 && navigation.title == "wv2-ready" {
                self.phase = 3;
                cx.system_browser(BROWSER).detach();
                self.ui.window(cx, ids!(first_window)).close(cx);
                cx.redraw_all();
                self.reattach_timer = cx.start_timeout(1.0);
            } else if self.phase == 4 && navigation.title == "wv2-state-41" {
                self.phase = 5;
                println!("WV2_PASS initialized, detached, parent closed, and state retained after reattachment");
                cx.system_browser(BROWSER).close();
            }
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
        if self.timeout.is_event(event).is_some() && self.phase != 5 {
            eprintln!("WV2_FAIL timeout at phase {}", self.phase);
            self.phase = 6;
        }
        if self.reattach_timer.is_event(event).is_some() && self.phase == 3 {
            self.phase = 4;
            cx.system_browser(BROWSER).eval_js("document.title='wv2-state-'+window.__makepad_test;");
        }
        if matches!(event, Event::Draw(_)) {
            let surface = if self.phase < 3 { ids!(first_surface) } else { ids!(second_surface) };
            let area = self.ui.view(cx, surface).area();
            cx.system_browser(BROWSER).update(area, true);
        }
    }
}
