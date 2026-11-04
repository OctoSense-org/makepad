//! WebView2 system browsers. All COM objects and callbacks belong to the UI STA.
//!
//! Creation is asynchronous: callbacks enqueue work for the Makepad event loop
//! rather than running a nested message pump while Cx is borrowed. A hidden
//! parking window keeps the native child alive when a tab moves between windows.
use crate::{
    event::NativeSystemBrowserNavigation, makepad_live_id::LiveId, makepad_math::Rect,
    thread::SignalToUI, window::WindowId,
};
use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, VecDeque},
    os::windows::ffi::OsStrExt,
    path::PathBuf,
    rc::{Rc, Weak},
};
use webview2_com::{
    take_pwstr, CreateCoreWebView2ControllerCompletedHandler,
    CreateCoreWebView2EnvironmentCompletedHandler, DocumentTitleChangedEventHandler,
    Microsoft::Web::WebView2::Win32::*, NavigationCompletedEventHandler,
    NavigationStartingEventHandler,
};
use webview_windows::{
    core::{w, BOOL, PCWSTR, PWSTR},
    Win32::{
        Foundation::{HWND, RECT},
        UI::WindowsAndMessaging::{
            CreateWindowExW, DestroyWindow, SetParent, SetWindowPos, ShowWindow, SWP_NOACTIVATE,
            SWP_NOZORDER, SW_HIDE, SW_SHOWNOACTIVATE, WINDOW_EX_STYLE, WS_CHILD, WS_CLIPCHILDREN,
            WS_CLIPSIBLINGS, WS_POPUP,
        },
    },
};

type Queue = Rc<RefCell<VecDeque<BrowserEvent>>>;
type WeakQueue = Weak<RefCell<VecDeque<BrowserEvent>>>;

enum BrowserEvent {
    Environment(LiveId, u64, Result<ICoreWebView2Environment, String>),
    Controller(LiveId, u64, Result<ICoreWebView2Controller, String>),
    Navigation(LiveId, u64, NativeSystemBrowserNavigation),
}

fn send(queue: &WeakQueue, event: BrowserEvent) {
    if let Some(queue) = queue.upgrade() {
        queue.borrow_mut().push_back(event);
        SignalToUI::set_internal_signal();
    }
}

fn wide(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(Some(0)).collect()
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
}

struct Browser {
    generation: u64,
    host: HWND,
    parent: HWND,
    layout: Option<Layout>,
    controller: Option<ICoreWebView2Controller>,
    webview: Option<ICoreWebView2>,
    allowed_navigation: Rc<Cell<bool>>,
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
                host,
                parent: parking,
                layout: None,
                controller: None,
                webview: None,
                allowed_navigation: Rc::new(Cell::new(false)),
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
                            browser.allowed_navigation.set(true);
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

    pub fn drain(&mut self) -> Vec<NativeSystemBrowserNavigation> {
        let mut actions = Vec::new();
        loop {
            let event = self.queue.borrow_mut().pop_front();
            let Some(event) = event else { break };
            match event {
                BrowserEvent::Environment(id, generation, result) => {
                    let Some(browser) = self
                        .browsers
                        .get(&id)
                        .filter(|b| b.generation == generation && !b.failed)
                    else {
                        continue;
                    };
                    match result {
                        Ok(environment) => {
                            let host = browser.host;
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
                                    id,
                                    generation,
                                    browser.navigable,
                                    browser.allowed_navigation.clone(),
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
                        actions.push(action);
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
    id: LiveId,
    generation: u64,
    navigable: bool,
    allowed: Rc<Cell<bool>>,
    queue: WeakQueue,
) -> webview_windows::core::Result<()> {
    unsafe {
        let mut token = 0;
        let start_queue = queue.clone();
        webview.add_NavigationStarting(
            &NavigationStartingEventHandler::create(Box::new(move |sender, args| {
                if !navigable && !allowed.replace(false) {
                    if let Some(args) = args {
                        args.SetCancel(true)?;
                    }
                    return Ok(());
                }
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
        webview.add_DocumentTitleChanged(
            &DocumentTitleChangedEventHandler::create(Box::new(move |sender, _| {
                send(
                    &queue,
                    BrowserEvent::Navigation(id, generation, navigation(sender, id, false, None)),
                );
                Ok(())
            })),
            &mut token,
        )?;
    }
    Ok(())
}
