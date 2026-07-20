#[cfg(target_os = "macos")]
use crate::window::WindowId;
use crate::{
    makepad_math::Rect,
    os::apple::{apple_sys::*, apple_util::str_to_nsstring},
};
use makepad_objc_sys::{class, msg_send, sel};

#[link(name = "WebKit", kind = "framework")]
unsafe extern "C" {}

fn make_request(url: &str) -> Option<ObjcId> {
    unsafe {
        let url_string = str_to_nsstring(url);
        if url_string == nil {
            return None;
        }
        let ns_url: ObjcId = msg_send![class!(NSURL), URLWithString: url_string];
        if ns_url == nil {
            return None;
        }
        let request: ObjcId = msg_send![class!(NSURLRequest), requestWithURL: ns_url];
        if request == nil {
            None
        } else {
            Some(request)
        }
    }
}

fn history_go(web_view: ObjcId, delta: i32) {
    if web_view == nil || delta == 0 {
        return;
    }
    unsafe {
        for _ in 0..delta.unsigned_abs() {
            if delta < 0 {
                let can_go_back: BOOL = msg_send![web_view, canGoBack];
                if can_go_back == YES {
                    let () = msg_send![web_view, goBack];
                }
            } else {
                let can_go_forward: BOOL = msg_send![web_view, canGoForward];
                if can_go_forward == YES {
                    let () = msg_send![web_view, goForward];
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub(crate) struct MacosSystemBrowser {
    browser_id: u64,
    current_url: String,
    attached_window: Option<WindowId>,
    host_view: ObjcId,
    web_view: ObjcId,
    html_generation: u64,
}

#[cfg(target_os = "macos")]
impl MacosSystemBrowser {
    pub(crate) fn new(browser_id: crate::makepad_live_id::LiveId, url: &str) -> Self {
        let mut browser = Self {
            browser_id: browser_id.get_value(),
            current_url: String::new(),
            attached_window: None,
            host_view: nil,
            web_view: nil,
            html_generation: 0,
        };
        browser.ensure_web_view();
        browser.set_url(url, false);
        browser
    }

    fn ensure_web_view(&mut self) {
        if self.web_view != nil {
            return;
        }
        unsafe {
            let config: ObjcId = msg_send![class!(WKWebViewConfiguration), new];
            // Install the octos_native script-message handler: the JS→native half
            // of the bridge. octos.invoke posts here via
            // webkit.messageHandlers.octos_native.postMessage({id,tool,args}).
            let ucc: ObjcId = msg_send![config, userContentController];
            if ucc != nil {
                let handler_class = crate::os::apple::apple_classes::get_apple_class_global()
                    .octos_web_message_handler;
                let handler: ObjcId = msg_send![handler_class, alloc];
                let handler: ObjcId = msg_send![handler, init];
                if handler != nil {
                    (*handler).set_ivar::<u64>("octos_browser_id", self.browser_id);
                    let name = str_to_nsstring("octos_native");
                    let () = msg_send![ucc, addScriptMessageHandler: handler name: name];
                }
            }
            let web_view: ObjcId = msg_send![class!(WKWebView), alloc];
            let web_view: ObjcId = msg_send![web_view, initWithFrame: NSRect {
                origin: NSPoint { x: 0.0, y: 0.0 },
                size: NSSize { width: 1.0, height: 1.0 }
            } configuration: config];
            if web_view != nil {
                let () = msg_send![web_view, setHidden: YES];
                self.web_view = web_view;
            }
        }
    }

    fn ensure_host_view(&mut self) {
        if self.host_view != nil {
            return;
        }
        unsafe {
            let host_view: ObjcId = msg_send![class!(NSView), alloc];
            let host_view: ObjcId = msg_send![host_view, initWithFrame: NSRect {
                origin: NSPoint { x: 0.0, y: 0.0 },
                size: NSSize { width: 1.0, height: 1.0 }
            }];
            if host_view != nil {
                let () = msg_send![host_view, setWantsLayer: YES];
                let layer: ObjcId = msg_send![host_view, layer];
                if layer != nil {
                    let () = msg_send![layer, setMasksToBounds: YES];
                }
                let () = msg_send![host_view, setHidden: YES];
                self.host_view = host_view;
            }
        }
    }

    fn ensure_attached(&mut self, window_id: WindowId, parent_view: ObjcId) {
        self.ensure_web_view();
        self.ensure_host_view();
        if self.web_view == nil || self.host_view == nil {
            return;
        }
        unsafe {
            let host_super_view: ObjcId = msg_send![self.host_view, superview];
            if self.attached_window != Some(window_id) || host_super_view != parent_view {
                let () = msg_send![self.host_view, removeFromSuperview];
                let () = msg_send![parent_view, addSubview: self.host_view];
                self.attached_window = Some(window_id);
            }

            let web_super_view: ObjcId = msg_send![self.web_view, superview];
            if web_super_view != self.host_view {
                let () = msg_send![self.web_view, removeFromSuperview];
                let () = msg_send![self.host_view, addSubview: self.web_view];
            }
        }
    }

    pub(crate) fn update(
        &mut self,
        window_id: WindowId,
        parent_view: ObjcId,
        unclipped_rect: Rect,
        clipped_rect: Rect,
        visible: bool,
    ) {
        self.ensure_attached(window_id, parent_view);
        unsafe {
            if self.host_view == nil || self.web_view == nil {
                return;
            }
            let (host_rect, web_view_rect, is_visible) =
                clipped_browser_layout(unclipped_rect, clipped_rect, visible);
            let host_frame = NSRect {
                origin: NSPoint {
                    x: host_rect.pos.x,
                    y: host_rect.pos.y,
                },
                size: NSSize {
                    width: host_rect.size.x.max(0.0),
                    height: host_rect.size.y.max(0.0),
                },
            };
            let web_view_frame = NSRect {
                origin: NSPoint {
                    x: web_view_rect.pos.x,
                    y: web_view_rect.pos.y,
                },
                size: NSSize {
                    width: web_view_rect.size.x.max(0.0),
                    height: web_view_rect.size.y.max(0.0),
                },
            };
            let () = msg_send![self.host_view, setFrame: host_frame];
            let () = msg_send![self.web_view, setFrame: web_view_frame];
            let () = msg_send![self.web_view, setHidden: if is_visible { NO } else { YES }];
            let () = msg_send![self.host_view, setHidden: if is_visible { NO } else { YES }];
        }
    }

    pub(crate) fn detach(&mut self) {
        unsafe {
            if self.web_view != nil {
                let () = msg_send![self.web_view, removeFromSuperview];
                let () = msg_send![self.web_view, setHidden: YES];
            }
            if self.host_view != nil {
                let () = msg_send![self.host_view, removeFromSuperview];
                let () = msg_send![self.host_view, setHidden: YES];
            }
        }
        self.attached_window = None;
    }

    pub(crate) fn set_url(&mut self, url: &str, _replace: bool) {
        if self.current_url == url {
            return;
        }
        self.ensure_web_view();
        let Some(request) = make_request(url) else {
            return;
        };
        unsafe {
            if self.web_view != nil {
                let () = msg_send![self.web_view, loadRequest: request];
                self.current_url.clear();
                self.current_url.push_str(url);
            }
        }
    }

    pub(crate) fn set_html(&mut self, html: &str, base_url: &str) {
        self.ensure_web_view();
        if self.web_view != nil {
            self.html_generation += 1;
            load_html_document(self.web_view, html, base_url, self.html_generation);
            // Inline documents have no meaningful URL; make sure a later
            // set_url to the previous URL is not deduped away.
            self.current_url.clear();
        }
    }

    /// Native→card channel: run a snippet inside the document. Settles
    /// octos.invoke promises (octos._resolve) and pushes octos._event(...) for
    /// the events half of the bridge — the WKWebView twin of Android's
    /// evalSystemBrowserJs.
    pub(crate) fn eval_js(&self, js: &str) {
        if self.web_view == nil {
            return;
        }
        unsafe {
            let js_string = str_to_nsstring(js);
            let () = msg_send![self.web_view, evaluateJavaScript: js_string completionHandler: nil];
        }
    }

    pub(crate) fn history_go(&mut self, delta: i32) {
        history_go(self.web_view, delta);
    }

    pub(crate) fn cleanup(&mut self) {
        unsafe {
            if self.web_view != nil {
                let () = msg_send![self.web_view, stopLoading];
            }
        }
        self.detach();
    }
}

/// Load an inline HTML document. WKWebView in a non-bundled (plain cargo)
/// process renders network and file loads but silently never commits
/// loadHTMLString/data: documents (and ATS blocks plaintext-http loopback), so
/// the document is staged as a temp file and loaded via
/// loadFileURL:allowingReadAccessToURL: (generation-stamped file name defeats
/// the URL-keyed page cache on reload). Note: file:// documents send no
/// Referer, so referer-gated embeds (e.g. YouTube) show their error card on
/// macOS; on Android loadDataWithBaseURL supplies a real https origin.
fn load_html_document(web_view: ObjcId, html: &str, _base_url: &str, generation: u64) {
    if web_view == nil {
        return;
    }
    let dir = std::env::temp_dir().join("makepad_web_cards");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let file = dir.join(format!("card_{:x}_{}.html", web_view as usize, generation));
    if std::fs::write(&file, html).is_err() {
        return;
    }
    let prev = dir.join(format!(
        "card_{:x}_{}.html",
        web_view as usize,
        generation.wrapping_sub(1)
    ));
    let _ = std::fs::remove_file(prev);
    let Some(path) = file.to_str() else {
        return;
    };
    unsafe {
        let file_string = str_to_nsstring(path);
        let dir_string = str_to_nsstring(&dir.to_string_lossy());
        if file_string == nil || dir_string == nil {
            return;
        }
        let file_url: ObjcId = msg_send![class!(NSURL), fileURLWithPath: file_string];
        let dir_url: ObjcId = msg_send![class!(NSURL), fileURLWithPath: dir_string];
        if file_url == nil || dir_url == nil {
            return;
        }
        let () = msg_send![web_view, loadFileURL: file_url allowingReadAccessToURL: dir_url];
    }
}

/// The `octos_native` WKScriptMessageHandler class. octos.invoke in the card
/// posts `{id, tool, args}` here; we read the per-instance browser id (ivar) and
/// post an AndroidSystemBrowserInvoke action into the makepad loop — the exact
/// action the Android JavascriptInterface bridge posts, so the widget's dispatch
/// (web_card.rs handle_invoke) is identical on both platforms. Registered once in
/// AppleClasses.
#[cfg(target_os = "macos")]
pub fn define_octos_web_message_handler() -> *const Class {
    extern "C" fn did_receive_script_message(
        this: &Object,
        _: Sel,
        _content_controller: ObjcId,
        message: ObjcId,
    ) {
        unsafe {
            if message == nil {
                return;
            }
            let browser_id: u64 = *this.get_ivar("octos_browser_id");
            let body: ObjcId = msg_send![message, body];
            // The card's JS is untrusted — it can postMessage ANY value
            // (a string, a number, an array). Every msg_send below assumes
            // specific classes, and an unrecognized selector raises an
            // NSException that takes the whole app down. Validate first.
            if body == nil || !msg_send![body, isKindOfClass: class!(NSDictionary)] {
                return;
            }
            // body is the JS object {id:Number, tool:String, args:String(json)}.
            let id_obj: ObjcId = msg_send![body, objectForKey: str_to_nsstring("id")];
            let tool_obj: ObjcId = msg_send![body, objectForKey: str_to_nsstring("tool")];
            let args_obj: ObjcId = msg_send![body, objectForKey: str_to_nsstring("args")];
            let call_id: i64 = if id_obj != nil
                && msg_send![id_obj, isKindOfClass: class!(NSNumber)]
            {
                msg_send![id_obj, longLongValue]
            } else {
                0
            };
            let tool = if tool_obj != nil
                && msg_send![tool_obj, isKindOfClass: class!(NSString)]
            {
                crate::os::apple::apple_util::nsstring_to_string(tool_obj)
            } else {
                String::new()
            };
            if tool.is_empty() {
                return;
            }
            let args = if args_obj != nil
                && msg_send![args_obj, isKindOfClass: class!(NSString)]
            {
                crate::os::apple::apple_util::nsstring_to_string(args_obj)
            } else {
                String::new()
            };
            crate::Cx::post_action(crate::event::AndroidSystemBrowserInvoke {
                browser_id,
                call_id,
                tool,
                args,
            });
            crate::thread::SignalToUI::set_ui_signal();
        }
    }
    let superclass = class!(NSObject);
    let mut decl = ClassDecl::new("MakepadOctosWebMessageHandler", superclass).unwrap();
    unsafe {
        decl.add_method(
            sel!(userContentController: didReceiveScriptMessage:),
            did_receive_script_message as extern "C" fn(&Object, Sel, ObjcId, ObjcId),
        );
        if let Some(protocol) = Protocol::get("WKScriptMessageHandler") {
            decl.add_protocol(protocol);
        }
    }
    decl.add_ivar::<u64>("octos_browser_id");
    decl.register()
}

#[cfg(target_os = "macos")]
fn clipped_browser_layout(
    unclipped_rect: Rect,
    clipped_rect: Rect,
    visible: bool,
) -> (Rect, Rect, bool) {
    let web_view_rect = Rect {
        pos: unclipped_rect.pos - clipped_rect.pos,
        size: unclipped_rect.size,
    };
    let is_visible = visible && clipped_rect.size.x > 0.0 && clipped_rect.size.y > 0.0;
    (clipped_rect, web_view_rect, is_visible)
}

#[cfg(all(test, target_os = "macos"))]
mod macos_tests {
    use super::clipped_browser_layout;
    use crate::makepad_math::{dvec2, Rect};

    #[test]
    fn keeps_native_browser_anchored_when_top_is_clipped() {
        let unclipped_rect = Rect {
            pos: dvec2(26.0, 262.0),
            size: dvec2(298.0, 420.0),
        };
        let clipped_rect = Rect {
            pos: dvec2(26.0, 262.0),
            size: dvec2(298.0, 319.0),
        };

        let (host_rect, web_view_rect, is_visible) =
            clipped_browser_layout(unclipped_rect, clipped_rect, true);

        assert_eq!(host_rect, clipped_rect);
        assert_eq!(web_view_rect.pos, dvec2(0.0, 0.0));
        assert_eq!(web_view_rect.size, unclipped_rect.size);
        assert!(is_visible);
    }

    #[test]
    fn offsets_native_browser_inside_clip_when_bottom_is_clipped() {
        let unclipped_rect = Rect {
            pos: dvec2(26.0, 120.0),
            size: dvec2(298.0, 420.0),
        };
        let clipped_rect = Rect {
            pos: dvec2(26.0, 170.0),
            size: dvec2(298.0, 370.0),
        };

        let (_, web_view_rect, _) = clipped_browser_layout(unclipped_rect, clipped_rect, true);

        assert_eq!(web_view_rect.pos, dvec2(0.0, -50.0));
    }

    #[test]
    fn hides_when_clip_is_empty() {
        let unclipped_rect = Rect {
            pos: dvec2(26.0, 262.0),
            size: dvec2(298.0, 420.0),
        };
        let clipped_rect = Rect {
            pos: dvec2(26.0, 262.0),
            size: dvec2(0.0, 0.0),
        };

        let (_, _, is_visible) = clipped_browser_layout(unclipped_rect, clipped_rect, true);

        assert!(!is_visible);
    }
}

#[cfg(target_os = "ios")]
pub(crate) struct IosSystemBrowser {
    current_url: String,
    web_view: ObjcId,
    html_generation: u64,
}

#[cfg(target_os = "ios")]
impl IosSystemBrowser {
    pub(crate) fn new(url: &str) -> Self {
        let mut browser = Self {
            current_url: String::new(),
            web_view: nil,
            html_generation: 0,
        };
        browser.ensure_web_view();
        browser.set_url(url, false);
        browser
    }

    fn ensure_web_view(&mut self) {
        if self.web_view != nil {
            return;
        }
        unsafe {
            let config: ObjcId = msg_send![class!(WKWebViewConfiguration), new];
            let web_view: ObjcId = msg_send![class!(WKWebView), alloc];
            let web_view: ObjcId = msg_send![web_view, initWithFrame: NSRect {
                origin: NSPoint { x: 0.0, y: 0.0 },
                size: NSSize { width: 1.0, height: 1.0 }
            } configuration: config];
            if web_view != nil {
                let () = msg_send![web_view, setHidden: YES];
                self.web_view = web_view;
            }
        }
    }

    pub(crate) fn update(&mut self, parent_view: ObjcId, rect: Rect, visible: bool) {
        self.ensure_web_view();
        unsafe {
            if self.web_view == nil {
                return;
            }
            let super_view: ObjcId = msg_send![self.web_view, superview];
            if super_view != parent_view {
                let () = msg_send![self.web_view, removeFromSuperview];
                let () = msg_send![parent_view, addSubview: self.web_view];
            }
            let () = msg_send![parent_view, bringSubviewToFront: self.web_view];
            let frame = NSRect {
                origin: NSPoint {
                    x: rect.pos.x,
                    y: rect.pos.y,
                },
                size: NSSize {
                    width: rect.size.x.max(0.0),
                    height: rect.size.y.max(0.0),
                },
            };
            let () = msg_send![self.web_view, setFrame: frame];
            let () = msg_send![self.web_view, setHidden: if visible { NO } else { YES }];
        }
    }

    pub(crate) fn detach(&mut self) {
        unsafe {
            if self.web_view != nil {
                let () = msg_send![self.web_view, removeFromSuperview];
                let () = msg_send![self.web_view, setHidden: YES];
            }
        }
    }

    pub(crate) fn set_url(&mut self, url: &str, _replace: bool) {
        if self.current_url == url {
            return;
        }
        self.ensure_web_view();
        let Some(request) = make_request(url) else {
            return;
        };
        unsafe {
            if self.web_view != nil {
                let () = msg_send![self.web_view, loadRequest: request];
                self.current_url.clear();
                self.current_url.push_str(url);
            }
        }
    }

    pub(crate) fn set_html(&mut self, html: &str, base_url: &str) {
        self.ensure_web_view();
        if self.web_view != nil {
            self.html_generation += 1;
            load_html_document(self.web_view, html, base_url, self.html_generation);
            self.current_url.clear();
        }
    }

    pub(crate) fn history_go(&mut self, delta: i32) {
        history_go(self.web_view, delta);
    }

    pub(crate) fn cleanup(&mut self) {
        unsafe {
            if self.web_view != nil {
                let () = msg_send![self.web_view, stopLoading];
            }
        }
        self.detach();
    }
}
