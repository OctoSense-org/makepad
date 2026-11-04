pub use makepad_widgets;
use makepad_widgets::*;
use makepad_widgets::makepad_platform::event::{NativeSystemBrowserInvoke, NativeSystemBrowserNavigation};
use std::io::{Read, Write};

app_main!(App);
const BROWSER: LiveId = live_id!(windows_webview_smoke);
const LOCKED: LiveId = live_id!(windows_webview_locked_smoke);
const FAILED: LiveId = live_id!(windows_webview_failure_smoke);
const CARD: LiveId = live_id!(windows_webview_card_smoke);
const CSP: LiveId = live_id!(windows_webview_csp_smoke);
const REDIRECT: LiveId = live_id!(windows_webview_redirect_smoke);
const FIRST: &str = "data:text/html,%3Ctitle%3Ewv2-start%3C/title%3E%3Cp%3EFirst%3C/p%3E";
const SECOND: &str = "data:text/html,%3Ctitle%3Ewv2-second%3C/title%3E%3Cp%3ESecond%3C/p%3E";
const LOCKED_PAGE: &str = "data:text/html,%3Ctitle%3Ewv2-locked%3C/title%3E%3Cp%3ELocked%3C/p%3E";
const CARD_HTML: &str = r#"<!doctype html><html><head><script>
document.title=document.baseURI==='https://example.org/root/'?'wv2-card-base-ok':'wv2-card-base-bad';
window.octos_native.invoke(42,'ping',JSON.stringify({probe:1}));
</script></head><body>Isolated card</body></html>"#;
const CSP_HTML: &str = r#"<!doctype html><html><head><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; base-uri 'none'"><script>
document.title=document.baseURI.startsWith('https://example.org/')?'wv2-csp-bad':'wv2-csp-ok';
</script></head><body>CSP</body></html>"#;
const REMOTE_HTML: &str = r#"<html><head><script>
if(window.octos_native){window.octos_native.invoke(99,'escape','{}');}
window.chrome.webview.postMessage({token:'forged',id:99,tool:'escape',args:'{}'});
document.title='wv2-remote';
</script></head></html>"#;

fn data_url(html: &str) -> String {
    let encoded: String = html.bytes().map(|byte| format!("%{byte:02X}")).collect();
    format!("data:text/html,{encoded}")
}

fn redirect_server() -> std::io::Result<String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let url = format!("http://{}/redirect", listener.local_addr()?);
    std::thread::spawn(move || {
        for stream in listener.incoming().take(2) {
            let Ok(mut stream) = stream else { break };
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
            let mut request = [0u8; 4096];
            let Ok(count) = stream.read(&mut request) else { break };
            let response = if request[..count].starts_with(b"GET /redirect ") {
                "HTTP/1.1 302 Found\r\nLocation: /destination\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string()
            } else {
                let body = "<html><head><title>wv2-redirect-ok</title></head><body>Redirected</body></html>";
                format!("HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body)
            };
            let _ = stream.write_all(response.as_bytes());
        }
    });
    Ok(url)
}

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
    #[rust] remote_timer: Timer,
    #[rust] locked_started: bool,
    #[rust] locked_ok: bool,
    #[rust] error_ok: bool,
    #[rust] card_invoke_ok: bool,
    #[rust] card_base_ok: bool,
    #[rust] card_remote_started: bool,
    #[rust] card_ok: bool,
    #[rust] csp_ok: bool,
    #[rust] redirect_ok: bool,
    #[rust] finished: bool,
    #[rust] invalid_profile_file: Option<std::path::PathBuf>,
}

impl App {
    fn maybe_navigate_card(&mut self, cx: &mut Cx) {
        if self.card_invoke_ok && self.card_base_ok && !self.card_remote_started {
            self.card_remote_started = true;
            cx.system_browser(CARD).set_url(&data_url(REMOTE_HTML), false);
        }
    }
    fn fail(&mut self, message: &str) {
        if !self.finished {
            self.finished = true;
            eprintln!("WV2_FAIL {message}");
        }
    }
    fn maybe_pass(&mut self, cx: &mut Cx) {
        if !self.finished && self.phase == 8 && self.locked_ok && self.error_ok && self.card_ok && self.csp_ok && self.redirect_ok {
            self.finished = true;
            println!("WV2_PASS reparenting, history, redirects, navigation policy, failure fallback, isolated card, CSP, and bridge isolation");
            cx.system_browser(BROWSER).close();
            cx.system_browser(LOCKED).close();
            cx.system_browser(FAILED).close();
            cx.system_browser(CARD).close();
            cx.system_browser(CSP).close();
            cx.system_browser(REDIRECT).close();
            if let Some(path) = self.invalid_profile_file.take() { let _ = std::fs::remove_file(path); }
        }
    }
}

impl MatchEvent for App {
    fn handle_startup(&mut self, cx: &mut Cx) {
        self.phase = 1;
        self.timeout = cx.start_timeout(75.0);
        cx.system_browser(BROWSER).spawn_navigable(FIRST);
        cx.system_browser(LOCKED).spawn(LOCKED_PAGE);
        cx.system_browser(CARD).spawn("about:blank");
        cx.system_browser(CARD).set_isolated_html(CARD_HTML, "https://example.org/root/");
        cx.system_browser(CSP).spawn("about:blank");
        cx.system_browser(CSP).set_isolated_html(CSP_HTML, "https://example.org/root/");
        match redirect_server() {
            Ok(url) => cx.system_browser(REDIRECT).spawn_navigable(&url),
            Err(error) => self.fail(&format!("cannot start redirect server: {error}")),
        }
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
            if let Some(invoke) = action.downcast_ref::<NativeSystemBrowserInvoke>() {
                if invoke.browser_id == CARD.0 && invoke.call_id == 42 && invoke.tool == "ping" && invoke.args == "{\"probe\":1}" {
                    self.card_invoke_ok = true;
                } else {
                    self.fail(&format!("unexpected bridge invocation: {invoke:?}"));
                }
                self.maybe_navigate_card(cx);
                self.maybe_pass(cx);
                continue;
            }
            let Some(nav) = action.downcast_ref::<NativeSystemBrowserNavigation>() else { continue };
            println!("WV2_EVENT id={} title={:?} loading={} error={:?}", nav.browser_id, nav.title, nav.loading, nav.error);
            if nav.browser_id == FAILED.0 {
                if nav.error.is_some() { self.error_ok = true; }
                else if !nav.loading && nav.title == "wv2-start" { self.fail("invalid profile unexpectedly loaded"); }
            } else if nav.browser_id == CARD.0 {
                if let Some(error) = &nav.error { self.fail(&format!("isolated card: {error}")); }
                else if nav.title == "wv2-card-base-ok" { self.card_base_ok = true; }
                else if nav.title == "wv2-card-base-bad" { self.fail("isolated card base URL did not resolve"); }
                else if nav.title == "wv2-remote" { self.remote_timer = cx.start_timeout(3.0); }
            } else if nav.browser_id == CSP.0 {
                if let Some(error) = &nav.error { self.fail(&format!("CSP card: {error}")); }
                else if nav.title == "wv2-csp-ok" { self.csp_ok = true; }
                else if nav.title == "wv2-csp-bad" { self.fail("CSP base-uri was bypassed"); }
            } else if nav.browser_id == REDIRECT.0 {
                if let Some(error) = &nav.error { self.fail(&format!("redirect: {error}")); }
                else if nav.title == "wv2-redirect-ok" { self.redirect_ok = true; }
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
            self.maybe_navigate_card(cx);
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
            self.fail(&format!("timeout at phase {}, policy={}, error={}, card={}, csp={}, redirect={}", self.phase, self.locked_ok, self.error_ok, self.card_ok, self.csp_ok, self.redirect_ok));
        }
        if self.remote_timer.is_event(event).is_some() && self.card_remote_started && !self.finished {
            self.card_ok = true;
            self.maybe_pass(cx);
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
