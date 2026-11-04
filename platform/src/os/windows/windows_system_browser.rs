//! WebView2 system browsers. All COM objects and callbacks belong to the UI STA.
//!
//! Creation is asynchronous: callbacks enqueue work for the Makepad event loop
//! rather than running a nested message pump while Cx is borrowed. A hidden
//! parking window keeps the native child alive when a tab moves between windows.
use crate::{
    event::{NativeSystemBrowserInvoke, NativeSystemBrowserNavigation},
    makepad_live_id::LiveId,
    makepad_math::Rect,
    thread::SignalToUI,
    window::WindowId,
};
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    os::windows::ffi::OsStrExt,
    path::PathBuf,
    rc::{Rc, Weak},
    time::{Duration, Instant},
};
use webview2_com::{
    take_pwstr, CreateCoreWebView2ControllerCompletedHandler,
    CreateCoreWebView2EnvironmentCompletedHandler, DocumentTitleChangedEventHandler,
    Microsoft::Web::WebView2::Win32::*, NavigationCompletedEventHandler,
    NavigationStartingEventHandler, WebMessageReceivedEventHandler,
    WebResourceRequestedEventHandler,
};
use webview_windows::{
    core::{w, BOOL, PCWSTR, PWSTR},
    Win32::{
        Foundation::{HWND, RECT},
        UI::{
            Shell::SHCreateMemStream,
            WindowsAndMessaging::{
                CreateWindowExW, DestroyWindow, SetParent, SetWindowPos, ShowWindow,
                SWP_NOACTIVATE, SWP_NOZORDER, SW_HIDE, SW_SHOWNOACTIVATE, WINDOW_EX_STYLE,
                WS_CHILD, WS_CLIPCHILDREN, WS_CLIPSIBLINGS, WS_POPUP,
            },
        },
    },
};

type Queue = Rc<RefCell<VecDeque<BrowserEvent>>>;
type WeakQueue = Weak<RefCell<VecDeque<BrowserEvent>>>;

enum BrowserEvent {
    Environment(LiveId, u64, Result<ICoreWebView2Environment, String>),
    Controller(LiveId, u64, Result<ICoreWebView2Controller, String>),
    Navigation(LiveId, u64, NativeSystemBrowserNavigation),
    Invoke(LiveId, u64, String, NativeSystemBrowserInvoke),
}

pub(crate) enum BrowserAction {
    Navigation(NativeSystemBrowserNavigation),
    Invoke(NativeSystemBrowserInvoke),
}

struct InlineDocument {
    origin: String,
    url: String,
    token: String,
    html: Rc<str>,
}

type InlineState = Rc<RefCell<Option<InlineDocument>>>;
type AllowedNavigation = Rc<RefCell<Option<String>>>;

fn send(queue: &WeakQueue, event: BrowserEvent) {
    if let Some(queue) = queue.upgrade() {
        queue.borrow_mut().push_back(event);
        SignalToUI::set_internal_signal();
    }
}

fn wide(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(Some(0)).collect()
}

fn normalized_url(value: &str) -> String {
    url::Url::parse(value)
        .map(|url| url.to_string())
        .unwrap_or_else(|_| value.to_string())
}

fn isolated_document(html: String, base_url: String) -> InlineDocument {
    let origin = format!(
        "https://{}.makepad-card.invalid",
        uuid::Uuid::new_v4().simple()
    );
    let url = format!("{origin}/index.html");
    let token = uuid::Uuid::new_v4().simple().to_string();
    let bridge = format!(
        "<script>(()=>{{const token=\"{token}\";const post=window.chrome.webview.postMessage.bind(window.chrome.webview);Object.defineProperty(window,'octos_native',{{value:Object.freeze({{invoke:(id,tool,args)=>post({{token,id,tool,args}})}}),writable:false,configurable:false}});document.currentScript.remove();}})();</script>"
    );
    let escaped_base = base_url
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    let base = if escaped_base.is_empty() {
        String::new()
    } else {
        format!("<base href=\"{escaped_base}\">")
    };
    // Keep a card's CSP ahead of its base element. The host bridge itself is
    // inserted first and removes its script node before any card code runs.
    let mut result = html;
    if let Some(head) = result.to_ascii_lowercase().find("<head>") {
        let at = head + "<head>".len();
        let lower = result.to_ascii_lowercase();
        let end_head = lower[at..]
            .find("</head>")
            .map(|offset| at + offset)
            .unwrap_or(result.len());
        let csp = lower[at..end_head].find("<meta http-equiv=\"content-security-policy\"");
        let base_at = csp
            .and_then(|offset| {
                result[at + offset..]
                    .find('>')
                    .map(|end| at + offset + end + 1)
            })
            .unwrap_or(at);
        result.insert_str(base_at, &base);
        result.insert_str(at, &bridge);
    } else {
        result = format!("<head>{bridge}{base}</head>{result}");
    }
    InlineDocument {
        origin,
        url,
        token,
        html: result.into(),
    }
}

fn profile_directory() -> PathBuf {
    let root = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let exe = std::env::current_exe()
        .ok()
        .and_then(|path| path.file_stem().map(|name| name.to_os_string()))
        .unwrap_or_else(|| "makepad".into());
    root.join("Makepad").join(exe).join("WebView2")
}

#[derive(Clone)]
struct Layout {
    parent: HWND,
    window_id: WindowId,
    rect: Rect,
    clip: Rect,
    dpi: f64,
    visible: bool,
}

enum Pending {
    Url(String, bool),
    History(i32),
    Eval(String),
    HtmlUnsupported,
    IsolatedHtml(String, String),
}

struct Browser {
    generation: u64,
    started: Instant,
    host: HWND,
    parent: HWND,
    layout: Option<Layout>,
    controller: Option<ICoreWebView2Controller>,
    webview: Option<ICoreWebView2>,
    environment: Option<ICoreWebView2Environment>,
    allowed_navigation: AllowedNavigation,
    inline: InlineState,
    navigable: bool,
    pending: VecDeque<Pending>,
    url: String,
    failed: bool,
}

impl Browser {
    fn close(&mut self) {
        unsafe {
            if let Some(controller) = self.controller.take() {
                let _ = controller.Close();
            }
            self.webview = None;
            let _ = DestroyWindow(self.host);
        }
    }

    fn detach(&mut self, parking: HWND) {
        self.layout = None;
        unsafe {
            if let Some(controller) = &self.controller {
                let _ = controller.SetIsVisible(false);
            }
            let _ = ShowWindow(self.host, SW_HIDE);
            if self.parent != parking {
                if let Err(error) = SetParent(self.host, Some(parking)) {
                    crate::error!("Could not park system browser window: {}", error);
                } else {
                    self.parent = parking;
                }
            }
        }
    }

    fn update_layout(&mut self) {
        let Some(layout) = &self.layout else { return };
        let scale = |n: f64| (n * layout.dpi).round() as i32;
        let x = scale(layout.clip.pos.x);
        let y = scale(layout.clip.pos.y);
        let width = (scale(layout.clip.pos.x + layout.clip.size.x) - x).max(0);
        let height = (scale(layout.clip.pos.y + layout.clip.size.y) - y).max(0);
        let left = scale(layout.rect.pos.x) - x;
        let top = scale(layout.rect.pos.y) - y;
        let right = scale(layout.rect.pos.x + layout.rect.size.x) - x;
        let bottom = scale(layout.rect.pos.y + layout.rect.size.y) - y;
        let visible = layout.visible && width > 0 && height > 0 && right > left && bottom > top;
        unsafe {
            if self.parent != layout.parent {
                if let Err(error) = SetParent(self.host, Some(layout.parent)) {
                    crate::error!("Could not attach system browser window: {}", error);
                    let _ = ShowWindow(self.host, SW_HIDE);
                    return;
                }
                self.parent = layout.parent;
            }
            let _ = SetWindowPos(
                self.host,
                None,
                x,
                y,
                width,
                height,
                SWP_NOZORDER | SWP_NOACTIVATE,
            );
            if let Some(controller) = &self.controller {
                let _ = controller.SetBounds(RECT {
                    left,
                    top,
                    right,
                    bottom,
                });
                let _ = controller.NotifyParentWindowPositionChanged();
                let _ = controller.SetIsVisible(visible);
            }
            let _ = ShowWindow(self.host, if visible { SW_SHOWNOACTIVATE } else { SW_HIDE });
        }
    }
}

#[derive(Default)]
pub(crate) struct WindowsSystemBrowsers {
    parking: HWND,
    next_generation: u64,
    queue: Queue,
    browsers: HashMap<LiveId, Browser>,
    profiles: HashMap<LiveId, PathBuf>,
    failed_spawns: HashMap<LiveId, u64>,
}

impl WindowsSystemBrowsers {
    pub fn set_data_directory(&mut self, id: LiveId, path: PathBuf) {
        self.profiles.insert(id, path);
    }

    fn ensure_parking(&mut self) -> Result<HWND, String> {
        if self.parking.is_invalid() {
            self.parking = unsafe {
                CreateWindowExW(
                    WINDOW_EX_STYLE::default(),
                    w!("STATIC"),
                    w!(""),
                    WS_POPUP,
                    0,
                    0,
                    1,
                    1,
                    None,
                    None,
                    None,
                    None,
                )
            }
            .map_err(|error| error.to_string())?;
        }
        Ok(self.parking)
    }

    pub fn spawn(&mut self, id: LiveId, url: String, navigable: bool) {
        self.close_browser(id);
        self.failed_spawns.remove(&id);
        self.next_generation = self.next_generation.wrapping_add(1);
        let generation = self.next_generation;
        let parking = match self.ensure_parking() {
            Ok(hwnd) => hwnd,
            Err(error) => {
                self.failed_spawns.insert(id, generation);
                self.report(id, generation, url, error);
                return;
            }
        };
        let host = match unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                w!("STATIC"),
                w!(""),
                WS_CHILD | WS_CLIPCHILDREN | WS_CLIPSIBLINGS,
                0,
                0,
                1,
                1,
                Some(parking),
                None,
                None,
                None,
            )
        } {
            Ok(hwnd) => hwnd,
            Err(error) => {
                self.failed_spawns.insert(id, generation);
                self.report(id, generation, url, error.to_string());
                return;
            }
        };
        self.browsers.insert(
            id,
            Browser {
                generation,
                started: Instant::now(),
                host,
                parent: parking,
                layout: None,
                controller: None,
                webview: None,
                environment: None,
                allowed_navigation: Rc::new(RefCell::new(None)),
                inline: Rc::new(RefCell::new(None)),
                navigable,
                pending: VecDeque::from([Pending::Url(url.clone(), false)]),
                url: url.clone(),
                failed: false,
            },
        );
        let dir = self
            .profiles
            .get(&id)
            .cloned()
            .unwrap_or_else(profile_directory);
        let path: Vec<u16> = dir.as_os_str().encode_wide().chain(Some(0)).collect();
        let queue = Rc::downgrade(&self.queue);
        let handler = CreateCoreWebView2EnvironmentCompletedHandler::create(Box::new(
            move |result, environment| {
                let result = result.map_err(|error| error.to_string()).and_then(|()| {
                    environment.ok_or_else(|| "WebView2 returned no environment".to_string())
                });
                send(&queue, BrowserEvent::Environment(id, generation, result));
                Ok(())
            },
        ));
        let result = unsafe {
            CreateCoreWebView2EnvironmentWithOptions(
                PCWSTR::null(),
                PCWSTR(path.as_ptr()),
                None,
                &handler,
            )
        };
        if let Err(error) = result {
            self.fail(id, generation, format!("WebView2: {error}"));
        }
    }

    fn report(&self, id: LiveId, generation: u64, url: String, error: String) {
        self.queue.borrow_mut().push_back(BrowserEvent::Navigation(
            id,
            generation,
            NativeSystemBrowserNavigation {
                browser_id: id.0,
                url,
                title: String::new(),
                loading: false,
                error: Some(error),
            },
        ));
        SignalToUI::set_internal_signal();
    }

    fn fail(&mut self, id: LiveId, generation: u64, error: String) {
        if let Some(browser) = self
            .browsers
            .get_mut(&id)
            .filter(|browser| browser.generation == generation)
        {
            browser.failed = true;
            let url = browser.url.clone();
            self.report(id, generation, url, error);
        }
    }

    pub fn update(
        &mut self,
        id: LiveId,
        window_id: WindowId,
        parent: *mut std::ffi::c_void,
        rect: Rect,
        clip: Rect,
        dpi: f64,
        visible: bool,
    ) {
        if let Some(browser) = self.browsers.get_mut(&id) {
            browser.layout = Some(Layout {
                parent: HWND(parent),
                window_id,
                rect,
                clip,
                dpi,
                visible,
            });
            browser.update_layout();
        }
    }

    pub fn detach(&mut self, id: LiveId) {
        if let Some(browser) = self.browsers.get_mut(&id) {
            browser.detach(self.parking);
        }
    }

    pub fn detach_window(&mut self, window_id: WindowId) {
        for browser in self.browsers.values_mut() {
            if browser
                .layout
                .as_ref()
                .is_some_and(|layout| layout.window_id == window_id)
            {
                browser.detach(self.parking);
            }
        }
    }

    pub fn window_moved(&self, window_id: WindowId) {
        for browser in self.browsers.values() {
            if browser
                .layout
                .as_ref()
                .is_some_and(|layout| layout.window_id == window_id)
            {
                if let Some(controller) = &browser.controller {
                    unsafe {
                        let _ = controller.NotifyParentWindowPositionChanged();
                    }
                }
            }
        }
    }

    pub fn close(&mut self, id: LiveId) {
        self.close_browser(id);
        self.profiles.remove(&id);
        self.failed_spawns.remove(&id);
    }

    fn close_browser(&mut self, id: LiveId) {
        if let Some(mut browser) = self.browsers.remove(&id) {
            browser.close();
        }
    }

    pub fn set_url(&mut self, id: LiveId, url: String, replace: bool) {
        self.enqueue(id, Pending::Url(url, replace));
    }
    pub fn history(&mut self, id: LiveId, delta: i32) {
        self.enqueue(id, Pending::History(delta));
    }
    pub fn eval(&mut self, id: LiveId, js: String) {
        self.enqueue(id, Pending::Eval(js));
    }
    pub fn set_html(&mut self, id: LiveId) {
        self.enqueue(id, Pending::HtmlUnsupported);
    }
    pub fn set_isolated_html(&mut self, id: LiveId, html: String, base_url: String) {
        self.enqueue(id, Pending::IsolatedHtml(html, base_url));
    }

    fn enqueue(&mut self, id: LiveId, command: Pending) {
        if let Some(browser) = self.browsers.get_mut(&id) {
            browser.pending.push_back(command);
        }
        self.run_pending(id);
    }

    fn run_pending(&mut self, id: LiveId) {
        let mut failure = None;
        if let Some(browser) = self.browsers.get_mut(&id) {
            let Some(webview) = browser.webview.as_ref() else {
                return;
            };
            while let Some(command) = browser.pending.pop_front() {
                let result = unsafe {
                    match command {
                        Pending::Url(url, replace) => {
                            browser.url = url.clone();
                            *browser.inline.borrow_mut() = None;
                            *browser.allowed_navigation.borrow_mut() = Some(normalized_url(&url));
                            if replace {
                                let script = format!(
                                    "location.replace({})",
                                    serde_json::to_string(&url).unwrap_or_default()
                                );
                                let script = wide(&script);
                                webview.ExecuteScript(PCWSTR(script.as_ptr()), None)
                            } else {
                                let url = wide(&url);
                                webview.Navigate(PCWSTR(url.as_ptr()))
                            }
                        }
                        Pending::History(delta) => {
                            if delta == 0 || !browser.navigable {
                                Ok(())
                            } else {
                                // history.go also treats an out-of-range offset
                                // as a no-op, as browser back buttons should.
                                let script = wide(&format!("history.go({delta})"));
                                webview.ExecuteScript(PCWSTR(script.as_ptr()), None)
                            }
                        }
                        Pending::Eval(js) => {
                            let js = wide(&js);
                            webview.ExecuteScript(PCWSTR(js.as_ptr()), None)
                        }
                        Pending::HtmlUnsupported => {
                            failure = Some((browser.generation, "Inline HTML with a custom origin is not supported by the Windows browser".to_string()));
                            break;
                        }
                        Pending::IsolatedHtml(html, base_url) => {
                            let document = isolated_document(html, base_url);
                            let url = document.url.clone();
                            browser.url = url.clone();
                            *browser.inline.borrow_mut() = Some(document);
                            *browser.allowed_navigation.borrow_mut() = Some(normalized_url(&url));
                            let url = wide(&url);
                            webview.Navigate(PCWSTR(url.as_ptr()))
                        }
                    }
                };
                if let Err(error) = result {
                    failure = Some((browser.generation, error.to_string()));
                    break;
                }
            }
        }
        if let Some((generation, error)) = failure {
            self.fail(id, generation, error);
        }
    }

    pub fn drain(&mut self) -> Vec<BrowserAction> {
        // WebView2 can leave a creation request pending indefinitely (for
        // example if the user-data directory cannot be used). Keep the rest of
        // the UI responsive and provide an actionable fallback in that case.
        let expired: Vec<_> = self
            .browsers
            .iter()
            .filter(|(_, browser)| {
                !browser.failed
                    && browser.webview.is_none()
                    && browser.started.elapsed() >= Duration::from_secs(30)
            })
            .map(|(&id, browser)| (id, browser.generation))
            .collect();
        for (id, generation) in expired {
            self.fail(
                id,
                generation,
                "WebView2 initialization timed out".to_string(),
            );
        }
        let mut actions = Vec::new();
        loop {
            let event = self.queue.borrow_mut().pop_front();
            let Some(event) = event else { break };
            match event {
                BrowserEvent::Environment(id, generation, result) => {
                    let Some(browser) = self
                        .browsers
                        .get_mut(&id)
                        .filter(|b| b.generation == generation && !b.failed)
                    else {
                        continue;
                    };
                    match result {
                        Ok(environment) => {
                            let host = browser.host;
                            browser.environment = Some(environment.clone());
                            let queue = Rc::downgrade(&self.queue);
                            let handler = CreateCoreWebView2ControllerCompletedHandler::create(
                                Box::new(move |result, controller| {
                                    let result =
                                        result.map_err(|error| error.to_string()).and_then(|()| {
                                            controller.ok_or_else(|| {
                                                "WebView2 returned no controller".to_string()
                                            })
                                        });
                                    if queue.upgrade().is_none() {
                                        if let Ok(controller) = result {
                                            unsafe {
                                                let _ = controller.Close();
                                            }
                                        }
                                    } else {
                                        send(
                                            &queue,
                                            BrowserEvent::Controller(id, generation, result),
                                        );
                                    }
                                    Ok(())
                                }),
                            );
                            if let Err(error) =
                                unsafe { environment.CreateCoreWebView2Controller(host, &handler) }
                            {
                                self.fail(id, generation, format!("WebView2: {error}"));
                            }
                        }
                        Err(error) => self.fail(id, generation, format!("WebView2: {error}")),
                    }
                }
                BrowserEvent::Controller(id, generation, result) => {
                    if !self
                        .browsers
                        .get(&id)
                        .is_some_and(|b| b.generation == generation && !b.failed)
                    {
                        if let Ok(controller) = result {
                            unsafe {
                                let _ = controller.Close();
                            }
                        }
                        continue;
                    }
                    match result {
                        Ok(controller) => match unsafe { controller.CoreWebView2() } {
                            Ok(webview) => {
                                let browser = self.browsers.get_mut(&id).unwrap();
                                if let Err(error) = register_events(
                                    &webview,
                                    browser
                                        .environment
                                        .as_ref()
                                        .expect("environment precedes controller")
                                        .clone(),
                                    id,
                                    generation,
                                    browser.navigable,
                                    browser.allowed_navigation.clone(),
                                    browser.inline.clone(),
                                    Rc::downgrade(&self.queue),
                                ) {
                                    unsafe {
                                        let _ = controller.Close();
                                    }
                                    self.fail(id, generation, format!("WebView2: {error}"));
                                    continue;
                                }
                                browser.controller = Some(controller);
                                browser.webview = Some(webview);
                                browser.update_layout();
                                self.run_pending(id);
                            }
                            Err(error) => {
                                unsafe {
                                    let _ = controller.Close();
                                }
                                self.fail(id, generation, error.to_string());
                            }
                        },
                        Err(error) => self.fail(id, generation, format!("WebView2: {error}")),
                    }
                }
                BrowserEvent::Navigation(id, generation, action) => {
                    // A host creation failure has no Browser entry; still report it.
                    if self
                        .browsers
                        .get(&id)
                        .is_some_and(|b| b.generation == generation)
                        || self.failed_spawns.get(&id) == Some(&generation)
                    {
                        actions.push(BrowserAction::Navigation(action));
                    }
                }
                BrowserEvent::Invoke(id, generation, token, action) => {
                    if self.browsers.get(&id).is_some_and(|browser| {
                        browser.generation == generation
                            && browser
                                .inline
                                .borrow()
                                .as_ref()
                                .is_some_and(|doc| doc.token == token)
                    }) {
                        actions.push(BrowserAction::Invoke(action));
                    }
                }
            }
        }
        actions
    }
}

impl Drop for WindowsSystemBrowsers {
    fn drop(&mut self) {
        for (_, mut browser) in self.browsers.drain() {
            browser.close();
        }
        if !self.parking.is_invalid() {
            unsafe {
                let _ = DestroyWindow(self.parking);
            }
        }
    }
}

fn navigation(
    webview: Option<ICoreWebView2>,
    id: LiveId,
    loading: bool,
    error: Option<String>,
) -> NativeSystemBrowserNavigation {
    let mut url = PWSTR::null();
    let mut title = PWSTR::null();
    if let Some(webview) = webview {
        unsafe {
            let _ = webview.Source(&mut url);
            let _ = webview.DocumentTitle(&mut title);
        }
    }
    NativeSystemBrowserNavigation {
        browser_id: id.0,
        url: take_pwstr(url),
        title: take_pwstr(title),
        loading,
        error,
    }
}

fn register_events(
    webview: &ICoreWebView2,
    environment: ICoreWebView2Environment,
    id: LiveId,
    generation: u64,
    navigable: bool,
    allowed: AllowedNavigation,
    inline: InlineState,
    queue: WeakQueue,
) -> webview_windows::core::Result<()> {
    unsafe {
        let mut token = 0;
        let start_queue = queue.clone();
        let start_inline = inline.clone();
        webview.add_NavigationStarting(
            &NavigationStartingEventHandler::create(Box::new(move |sender, args| {
                let Some(args) = args else { return Ok(()) };
                let mut raw_url = PWSTR::null();
                if args.Uri(&mut raw_url).is_err() {
                    args.SetCancel(true)?;
                    return Ok(());
                }
                let target = normalized_url(&take_pwstr(raw_url));
                if !navigable {
                    if allowed.borrow().as_deref() != Some(target.as_str()) {
                        args.SetCancel(true)?;
                        return Ok(());
                    }
                    allowed.borrow_mut().take();
                }
                let mut doc = start_inline.borrow_mut();
                if doc.as_ref().is_some_and(|doc| {
                    let mut target_url = url::Url::parse(&target).ok();
                    if let Some(target_url) = target_url.as_mut() {
                        target_url.set_fragment(None);
                    }
                    target_url.map(|url| url.to_string()) != Some(doc.url.clone())
                }) {
                    *doc = None;
                }
                drop(doc);
                send(
                    &start_queue,
                    BrowserEvent::Navigation(id, generation, navigation(sender, id, true, None)),
                );
                Ok(())
            })),
            &mut token,
        )?;
        let end_queue = queue.clone();
        webview.add_NavigationCompleted(
            &NavigationCompletedEventHandler::create(Box::new(move |sender, args| {
                let mut success = BOOL::default();
                let mut status = COREWEBVIEW2_WEB_ERROR_STATUS::default();
                let error = if let Some(args) = args {
                    args.IsSuccess(&mut success)?;
                    if !success.as_bool() {
                        args.WebErrorStatus(&mut status)?;
                        // A superseded or deliberately cancelled navigation is not
                        // a failure of the document that remains visible.
                        if status == COREWEBVIEW2_WEB_ERROR_STATUS_OPERATION_CANCELED {
                            return Ok(());
                        }
                        Some(format!("WebView2 navigation error {}", status.0))
                    } else {
                        None
                    }
                } else {
                    Some("WebView2 navigation did not complete".to_string())
                };
                send(
                    &end_queue,
                    BrowserEvent::Navigation(id, generation, navigation(sender, id, false, error)),
                );
                Ok(())
            })),
            &mut token,
        )?;
        let title_queue = queue.clone();
        webview.add_DocumentTitleChanged(
            &DocumentTitleChangedEventHandler::create(Box::new(move |sender, _| {
                send(
                    &title_queue,
                    BrowserEvent::Navigation(id, generation, navigation(sender, id, false, None)),
                );
                Ok(())
            })),
            &mut token,
        )?;
        let filter = wide("https://*.makepad-card.invalid/*");
        webview.AddWebResourceRequestedFilter(
            PCWSTR(filter.as_ptr()),
            COREWEBVIEW2_WEB_RESOURCE_CONTEXT_ALL,
        )?;
        let resource_inline = inline.clone();
        webview.add_WebResourceRequested(
            &WebResourceRequestedEventHandler::create(Box::new(move |_, args| {
                let Some(args) = args else { return Ok(()) };
                let request = args.Request()?;
                let mut raw_url = PWSTR::null();
                request.Uri(&mut raw_url)?;
                let request_url = normalized_url(&take_pwstr(raw_url));
                let html = resource_inline.borrow().as_ref()
                    .filter(|doc| request_url == doc.url)
                    .map(|doc| doc.html.clone());
                let (body, code, reason, headers) = if let Some(html) = html {
                    (html, 200, w!("OK"), w!("Content-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff"))
                } else {
                    (Rc::<str>::from(""), 404, w!("Not Found"), w!("Content-Type: text/plain\r\nCache-Control: no-store"))
                };
                let stream = SHCreateMemStream(Some(body.as_bytes()));
                let response = environment.CreateWebResourceResponse(stream.as_ref(), code, reason, headers)?;
                args.SetResponse(&response)?;
                Ok(())
            })),
            &mut token,
        )?;
        webview.add_WebMessageReceived(
            &WebMessageReceivedEventHandler::create(Box::new(move |sender, args| {
                let (Some(sender), Some(args)) = (sender, args) else {
                    return Ok(());
                };
                let mut raw_source = PWSTR::null();
                let mut raw_current = PWSTR::null();
                args.Source(&mut raw_source)?;
                sender.Source(&mut raw_current)?;
                let source = take_pwstr(raw_source);
                let current = take_pwstr(raw_current);
                let same_origin = |value: &str, origin: &str| {
                    url::Url::parse(value)
                        .is_ok_and(|url| url.origin().ascii_serialization() == origin)
                };
                let (origin, secret) = match inline.borrow().as_ref() {
                    Some(doc) => (doc.origin.clone(), doc.token.clone()),
                    None => return Ok(()),
                };
                if !same_origin(&source, &origin) || !same_origin(&current, &origin) {
                    return Ok(());
                }
                let mut raw_json = PWSTR::null();
                args.WebMessageAsJson(&mut raw_json)?;
                let json = take_pwstr(raw_json);
                if json.len() > 1_048_576 {
                    return Ok(());
                }
                let Ok(message) = serde_json::from_str::<serde_json::Value>(&json) else {
                    return Ok(());
                };
                let (Some(message_token), Some(call_id), Some(tool), Some(args_json)) = (
                    message.get("token").and_then(|v| v.as_str()),
                    message.get("id").and_then(|v| v.as_i64()),
                    message.get("tool").and_then(|v| v.as_str()),
                    message.get("args").and_then(|v| v.as_str()),
                ) else {
                    return Ok(());
                };
                if message_token != secret
                    || call_id < 0
                    || tool.len() > 128
                    || serde_json::from_str::<serde_json::Value>(args_json).is_err()
                {
                    return Ok(());
                }
                send(
                    &queue,
                    BrowserEvent::Invoke(
                        id,
                        generation,
                        secret,
                        NativeSystemBrowserInvoke {
                            browser_id: id.0,
                            call_id,
                            tool: tool.to_string(),
                            args: args_json.to_string(),
                        },
                    ),
                );
                Ok(())
            })),
            &mut token,
        )?;
    }
    Ok(())
}
