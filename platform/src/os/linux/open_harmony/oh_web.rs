//! makepad's system browsers (`cx.system_browser(id)`) on OpenHarmony, with
//! ArkWeb.
//!
//! The NDK has no web node, so ArkTS declares one `Web` per browser, on top of
//! makepad's XComponent, from the list kept here; everything else stays out of
//! ArkTS where ArkWeb's NDK allows it (`web_shim.cpp`):
//!
//! - the page's `window.octos_native.invoke(callId, tool, args)` is a native
//!   proxy, delivered here as `NativeSystemBrowserInvoke`;
//! - `eval_js` (replies, events) is a native `runJavaScript`.
//!
//! What the NDK lacks stays a thin ArkTS step: loading a URL or HTML and
//! history are controller calls ArkTS takes from `web_view_take_commands`, and
//! navigation, errors and the navigation policy are `Web` events it forwards.
//!
//! Each browser's controller is named with its tag, `makepad_web_<id>`.

use super::oh_callbacks::{send_from_ohos_message, FromOhosMessage};
use crate::makepad_math::Rect;
use napi_derive_ohos::napi;
use std::collections::VecDeque;
use std::ffi::{c_char, CString};
use std::sync::Mutex;

extern "C" {
    fn makepad_arkweb_register_bridge(tag: *const c_char) -> i32;
    fn makepad_arkweb_run_js(tag: *const c_char, js: *const c_char, len: usize) -> i32;
}

/// What a browser wants ArkTS to do with its controller.
enum Command {
    Url(String),
    Html { html: String, base_url: String },
    HistoryGo(i32),
}

struct Browser {
    id: u64,
    navigable: bool,
    /// Logical points (vp), relative to the page.
    rect: Rect,
    visible: bool,
    /// The bridge is registered; the queued loads may run.
    attached: bool,
    /// URLs we asked to load: navigation to these is never blocked.
    requested: Vec<String>,
}

#[derive(Default)]
struct State {
    browsers: Vec<Browser>,
    commands: VecDeque<(u64, Command)>,
}

static STATE: Mutex<State> = Mutex::new(State { browsers: Vec::new(), commands: VecDeque::new() });

pub fn tag(id: u64) -> String {
    format!("makepad_web_{id}")
}

fn id_of(tag: &str) -> Option<u64> {
    tag.strip_prefix("makepad_web_")?.parse().ok()
}

fn with_browser<R>(id: u64, f: impl FnOnce(&mut Browser) -> R) -> Option<R> {
    STATE.lock().unwrap().browsers.iter_mut().find(|b| b.id == id).map(f)
}

// ---- Cx side (from CxOsOp) ----

/// Spawn (or reuse) a browser showing `url`. Returns whether ArkTS must
/// re-read the list.
pub fn spawn(id: u64, url: &str, navigable: bool) -> bool {
    let mut state = STATE.lock().unwrap();
    let new = !state.browsers.iter().any(|b| b.id == id);
    if new {
        state.browsers.push(Browser {
            id,
            navigable,
            rect: Rect::default(),
            visible: false,
            attached: false,
            requested: Vec::new(),
        });
    }
    if let Some(b) = state.browsers.iter_mut().find(|b| b.id == id) {
        b.navigable = navigable;
        if !url.is_empty() && url != "about:blank" {
            b.requested.push(url.to_string());
        }
    }
    if !url.is_empty() && url != "about:blank" {
        state.commands.push_back((id, Command::Url(url.to_string())));
    }
    true
}

/// Place the browser (logical points) and show or hide it. Returns whether
/// anything changed.
pub fn update(id: u64, rect: Rect, visible: bool) -> bool {
    with_browser(id, |b| {
        let changed = b.rect != rect || b.visible != visible;
        b.rect = rect;
        b.visible = visible;
        changed
    })
    .unwrap_or(false)
}

pub fn hide(id: u64) -> bool {
    with_browser(id, |b| std::mem::replace(&mut b.visible, false)).unwrap_or(false)
}

pub fn close(id: u64) -> bool {
    let mut state = STATE.lock().unwrap();
    let before = state.browsers.len();
    state.browsers.retain(|b| b.id != id);
    state.commands.retain(|(c, _)| *c != id);
    state.browsers.len() != before
}

pub fn set_url(id: u64, url: &str) {
    with_browser(id, |b| b.requested.push(url.to_string()));
    STATE.lock().unwrap().commands.push_back((id, Command::Url(url.to_string())));
}

pub fn set_html(id: u64, html: &str, base_url: &str) {
    let base_url = if base_url.is_empty() { "https://octos-one.app/" } else { base_url };
    with_browser(id, |b| b.requested.push(base_url.to_string()));
    STATE
        .lock()
        .unwrap()
        .commands
        .push_back((id, Command::Html { html: html.to_string(), base_url: base_url.to_string() }));
}

pub fn history_go(id: u64, delta: i32) {
    STATE.lock().unwrap().commands.push_back((id, Command::HistoryGo(delta)));
}

/// Evaluate `js` in the browser's document, natively. Runs on the ArkTS main
/// thread (ArkWeb's requirement).
pub fn eval_js(id: u64, js: &str) {
    let tag = tag(id);
    let js = js.to_string();
    super::oh_ime::run_on_main(move || {
        let Ok(tag) = CString::new(tag) else { return };
        if unsafe { makepad_arkweb_run_js(tag.as_ptr(), js.as_ptr() as *const c_char, js.len()) } != 0 {
            crate::error!("web: runJavaScript unavailable");
        }
    });
}

/// Whether loads or history steps are waiting for an attached browser (one
/// still attaching flushes its own when it attaches).
pub fn has_commands() -> bool {
    let state = STATE.lock().unwrap();
    state.commands.iter().any(|(id, _)| state.browsers.iter().any(|b| b.id == *id && b.attached))
}

// ---- ArkTS side (napi, on the ArkTS main thread) ----

/// The browsers to declare: `tag|x|y|w|h|visible`, in vp.
#[napi]
pub fn web_view_list() -> Vec<String> {
    STATE
        .lock()
        .unwrap()
        .browsers
        .iter()
        .map(|b| {
            format!(
                "{}|{}|{}|{}|{}|{}",
                tag(b.id),
                b.rect.pos.x,
                b.rect.pos.y,
                b.rect.size.x.max(1.0),
                b.rect.size.y.max(1.0),
                b.visible as u8
            )
        })
        .collect()
}

/// The controller calls that are due, for attached browsers only:
/// `tag|url|<url>`, `tag|html|<base>|<html>`, `tag|history|<delta>`.
#[napi]
pub fn web_view_take_commands() -> Vec<String> {
    let mut state = STATE.lock().unwrap();
    let attached: Vec<u64> = state.browsers.iter().filter(|b| b.attached).map(|b| b.id).collect();
    let mut out = Vec::new();
    let mut keep = VecDeque::new();
    while let Some((id, command)) = state.commands.pop_front() {
        if !attached.contains(&id) {
            keep.push_back((id, command));
            continue;
        }
        out.push(match command {
            Command::Url(url) => format!("{}|url|{}", tag(id), url),
            Command::Html { html, base_url } => format!("{}|html|{}|{}", tag(id), base_url, html),
            Command::HistoryGo(delta) => format!("{}|history|{}", tag(id), delta),
        });
    }
    state.commands = keep;
    out
}

/// The browser's controller is attached: expose `octos_native` to the
/// documents it loads from now on (natively).
#[napi]
pub fn web_view_attached(tag: String) {
    if let Ok(c_tag) = CString::new(tag) {
        if unsafe { makepad_arkweb_register_bridge(c_tag.as_ptr()) } != 0 {
            crate::error!("web: registerJavaScriptProxy unavailable; octos.invoke will not reach the app");
        }
    }
}

/// The browser finished its first (blank) page: its queued loads may run.
/// Loading earlier races that initial page, which aborts the load.
#[napi]
pub fn web_view_ready(tag: String) {
    let Some(id) = id_of(&tag) else { return };
    with_browser(id, |b| b.attached = true);
}

/// The navigation policy: a browser opened with `spawn` stays on the documents
/// the app loads; one opened with `spawn_navigable` follows links, scripts and
/// redirects.
///
/// A `spawn` browser may also move to another host of the same site as a page
/// the app asked for: ArkWeb on a 2-in-1 presents a desktop browser, so a
/// mobile host redirects (m.youtube.com -> www.youtube.com) where Android's
/// mobile WebView would not, and that hop is still the document the app
/// opened, not a link the person followed elsewhere.
#[napi]
pub fn web_view_may_navigate(tag: String, url: String) -> bool {
    let Some(id) = id_of(&tag) else { return true };
    with_browser(id, |b| {
        b.navigable
            || url.starts_with("about:")
            || url.starts_with("data:")
            || b.requested.iter().any(|r| {
                url == *r || url.trim_end_matches('/') == r.trim_end_matches('/') || same_site(&url, r)
            })
    })
    .unwrap_or(true)
}

/// Whether two https URLs are on the same site: the last two labels of the
/// host (youtube.com), enough for the mobile/desktop host split it serves.
fn same_site(a: &str, b: &str) -> bool {
    fn site(url: &str) -> Option<String> {
        let rest = url.strip_prefix("https://")?;
        let host = rest.split(['/', '?', '#', ':']).next()?.to_ascii_lowercase();
        let labels: Vec<&str> = host.split('.').filter(|l| !l.is_empty()).collect();
        (labels.len() >= 2).then(|| labels[labels.len() - 2..].join("."))
    }
    matches!((site(a), site(b)), (Some(x), Some(y)) if x == y)
}

#[napi]
pub fn web_view_navigation(tag: String, url: String, title: String, loading: bool) {
    let Some(id) = id_of(&tag) else { return };
    send_from_ohos_message(FromOhosMessage::WebNavigation { id, url, title, loading });
}

/// A main-frame load failed.
#[napi]
pub fn web_view_page_error(tag: String, code: i32, description: String, url: String) {
    let Some(id) = id_of(&tag) else { return };
    send_from_ohos_message(FromOhosMessage::WebPageError { id, code, description, url });
}

/// `octos_native.invoke(callId, tool, args)` from a page (web_shim.cpp).
#[no_mangle]
extern "C" fn makepad_ohos_web_invoke(
    tag: *const c_char,
    call_id: *const c_char,
    call_id_len: usize,
    tool: *const c_char,
    tool_len: usize,
    args: *const c_char,
    args_len: usize,
) {
    let text = |p: *const c_char, n: usize| {
        if p.is_null() {
            String::new()
        } else {
            let bytes = unsafe { std::slice::from_raw_parts(p as *const u8, n) };
            String::from_utf8_lossy(bytes).trim_end_matches('\0').to_string()
        }
    };
    let tag = if tag.is_null() { String::new() } else { unsafe { std::ffi::CStr::from_ptr(tag) }.to_string_lossy().into_owned() };
    let Some(id) = id_of(&tag) else { return };
    let call_id = text(call_id, call_id_len).trim().parse::<f64>().map(|v| v as i64).unwrap_or(0);
    send_from_ohos_message(FromOhosMessage::WebInvoke {
        id,
        call_id,
        tool: text(tool, tool_len),
        args: text(args, args_len),
    });
}
