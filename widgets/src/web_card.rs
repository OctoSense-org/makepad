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

use crate::{makepad_derive_widget::*, makepad_draw::*, makepad_micro_serde::*, widget::*};

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

/// Host allowlist for the `http.fetch` bridge tool — the ONLY hosts a card may
/// reach natively. This is the capability gate: the card's JS is untrusted, so
/// the trusted Rust side enforces it (a JS-side check is bypassable). ymote/Splash
/// `mod.tool` would formalize this with per-card leases + audit; until then it's a
/// curated static allowlist plus an SSRF guard (https-only, no private/loopback/
/// link-local/internal hosts). Keep the Piped hosts in sync with
/// `octos_media.js::O.ytSearchInstances`.
const HTTP_FETCH_ALLOWED_HOSTS: &[&str] = &[
    "googleapis.com",           // YouTube Data API
    "noembed.com",              // oEmbed
    "piped.private.coffee",     // Piped search instances ↓
    "pipedapi.r4fo.com",
    "pipedapi.orangenet.cc",
    "api.piped.yt",
    "pipedapi.adminforge.de",
    "query1.finance.yahoo.com", // stock card
    "query2.finance.yahoo.com",
];

/// Shared SSRF guard: require https and reject loopback / private / link-local /
/// internal hosts. Returns the lowercased host. Used by both `http.fetch` (which
/// adds the allowlist) and `download` (which doesn't — its bytes land in the
/// sandbox, so the SSRF block is the protection that matters).
fn check_https_public(url: &str) -> Result<String, String> {
    let rest = url
        .strip_prefix("https://")
        .ok_or_else(|| "only https:// URLs are allowed".to_string())?;
    let host = rest
        .split(|c| c == '/' || c == '?' || c == '#')
        .next()
        .unwrap_or("")
        .split('@')
        .next_back()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    if host.is_empty() {
        return Err("missing host".into());
    }
    let blocked_prefix = ["127.", "10.", "192.168.", "169.254.", "0."];
    let private_172 = host.starts_with("172.")
        && host
            .split('.')
            .nth(1)
            .and_then(|o| o.parse::<u8>().ok())
            .map(|o| (16..=31).contains(&o))
            .unwrap_or(false);
    if host == "localhost"
        || host == "::1"
        || host.ends_with(".localhost")
        || host.ends_with(".internal")
        || host.ends_with(".local")
        || blocked_prefix.iter().any(|p| host.starts_with(p))
        || private_172
    {
        return Err(format!("host not permitted (private/internal): {}", host));
    }
    Ok(host)
}

/// Enforce the `http.fetch` capability gate: SSRF guard + the host must be on the
/// allowlist (boundary-correct suffix match, so `evil-googleapis.com` does NOT
/// match `googleapis.com`).
fn fetch_host_allowed(url: &str) -> Result<(), String> {
    let host = check_https_public(url)?;
    let allowed = HTTP_FETCH_ALLOWED_HOSTS
        .iter()
        .any(|a| host == *a || host.ends_with(&format!(".{}", a)));
    if allowed {
        Ok(())
    } else {
        Err(format!("host not on allowlist: {}", host))
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
    /// In-flight `octos.invoke("http.fetch", ...)` calls: request_id → JS call_id,
    /// so the NetworkResponse can resolve the right card-side promise.
    #[rust]
    pending: Vec<(LiveId, i64)>,
    #[rust]
    req_seq: u64,
    /// In-flight downloads: call_id → (client download id, sandbox-relative dest).
    #[rust]
    downloads: Vec<(i64, String, String)>,
}

/// Args for the `http.fetch` bridge tool. The card sends headers as `[[k,v],…]`
/// (micro-serde-friendly) and a string body; method defaults to GET.
#[derive(DeJson)]
struct FetchArgs {
    url: String,
    method: Option<String>,
    headers: Option<Vec<Vec<String>>>,
    body: Option<String>,
}

/// Args for the `share` / `clipboard.write` bridge tools.
#[derive(DeJson)]
struct TextArg {
    text: String,
}

/// Args for the `notify` bridge tool.
#[derive(DeJson)]
struct NotifyArgs {
    title: String,
    body: Option<String>,
}

/// Args for the `fs.read`/`fs.list`/`fs.remove`/`fs.exists`/`fs.mkdir` tools.
#[derive(DeJson)]
struct PathArg {
    path: String,
}

/// Args for `fs.write`.
#[derive(DeJson)]
struct PathDataArg {
    path: String,
    data: String,
}

/// Args for `dialog.open` (optional MIME filter).
#[derive(DeJson)]
struct DialogArgs {
    mime: Option<String>,
}

/// Args for `download` — fetch `url` to the sandbox path `dest`; `id` is a
/// client-chosen download id echoed back in `download.progress` events.
#[derive(DeJson)]
struct DownloadArgs {
    url: String,
    dest: String,
    id: Option<String>,
}

/// Sandbox root for card fs — a dedicated subdir of the app's private storage.
/// A card can NEVER reach outside it (absolute paths and `..` are rejected in
/// `fs_resolve`), so it can't read the octos profile, other apps, or the system.
fn fs_sandbox_root() -> Result<std::path::PathBuf, String> {
    let home = std::env::var("HOME").map_err(|_| "no app storage available".to_string())?;
    Ok(std::path::PathBuf::from(home).join("card-fs"))
}

/// Resolve a card-relative path under the sandbox, rejecting any escape. Only
/// `Normal` components are allowed — absolute paths (`RootDir`/`Prefix`) and `..`
/// (`ParentDir`) are refused, so the result is always inside the sandbox even
/// before the file exists (no reliance on canonicalize).
fn fs_resolve(rel: &str) -> Result<std::path::PathBuf, String> {
    use std::path::Component;
    if rel.trim().is_empty() {
        return Err("empty path".into());
    }
    let mut safe = fs_sandbox_root()?;
    for comp in std::path::Path::new(rel).components() {
        match comp {
            Component::Normal(c) => safe.push(c),
            Component::CurDir => {}
            _ => return Err(format!("path not allowed (escapes sandbox): {}", rel)),
        }
    }
    Ok(safe)
}

/// List a sandbox directory as a JSON array of `{name, dir, size}`.
fn fs_list_json(dir: &std::path::Path) -> Result<String, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("list failed: {}", e))?;
    let mut items: Vec<String> = Vec::new();
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let md = entry.metadata().ok();
        let is_dir = md.as_ref().map(|m| m.is_dir()).unwrap_or(false);
        let size = md.as_ref().map(|m| m.len()).unwrap_or(0);
        items.push(format!(
            "{{\"name\":{},\"dir\":{},\"size\":{}}}",
            name.serialize_json(),
            is_dir,
            size
        ));
    }
    Ok(format!("[{}]", items.join(",")))
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

    fn next_seq(&mut self) -> u64 {
        self.req_seq = self.req_seq.wrapping_add(1);
        self.req_seq
    }

    /// Resolve the card-side promise for `call_id` with a raw JSON payload.
    fn resolve_raw(&mut self, cx: &mut Cx, call_id: i64, payload_json: &str) {
        let js = format!(
            "window.octos&&octos._resolve&&octos._resolve({},{})",
            call_id, payload_json
        );
        cx.system_browser(self.browser_id()).eval_js(&js);
    }

    fn reject(&mut self, cx: &mut Cx, call_id: i64, msg: &str) {
        let mj = msg.to_string().serialize_json();
        self.resolve_raw(cx, call_id, &format!("{{\"ok\":false,\"error\":{}}}", mj));
    }

    /// Dispatch one `octos.invoke(tool, args)`. `args` is a JSON string from the card.
    fn handle_invoke(&mut self, cx: &mut Cx, call_id: i64, tool: &str, args: &str) {
        match tool {
            // Round-trip probe: echo the args object back untouched.
            "ping" => {
                self.resolve_raw(cx, call_id, &format!("{{\"ok\":true,\"echo\":{}}}", args));
            }
            // Native HTTP (no browser CORS). Reuses the platform's http_request;
            // the response returns via Event::NetworkResponses (handle below).
            // GATED: the card's JS is untrusted, so the trusted Rust side enforces
            // a host allowlist + SSRF guard before any request leaves the device.
            "http.fetch" => match FetchArgs::deserialize_json(args) {
                Ok(a) => {
                    if let Err(why) = fetch_host_allowed(&a.url) {
                        log!("web_card http.fetch DENIED: {} ({})", a.url, why);
                        self.reject(cx, call_id, &format!("http.fetch denied: {}", why));
                    } else {
                        let method = match a.method.as_deref().unwrap_or("GET").to_ascii_uppercase().as_str() {
                            "POST" => HttpMethod::POST,
                            "PUT" => HttpMethod::PUT,
                            "DELETE" => HttpMethod::DELETE,
                            "HEAD" => HttpMethod::HEAD,
                            "PATCH" => HttpMethod::PATCH,
                            _ => HttpMethod::GET,
                        };
                        let mut req = HttpRequest::new(a.url.clone(), method);
                        if let Some(hs) = &a.headers {
                            for h in hs {
                                if h.len() == 2 {
                                    req.set_header(h[0].clone(), h[1].clone());
                                }
                            }
                        }
                        if let Some(b) = &a.body {
                            if !b.is_empty() {
                                req.set_string_body(b.clone());
                            }
                        }
                        let request_id = LiveId(0xF0C5_0000_0000_0000 ^ self.next_seq());
                        self.pending.push((request_id, call_id));
                        cx.http_request(request_id, req);
                    }
                }
                Err(e) => self.reject(cx, call_id, &format!("bad http.fetch args: {:?}", e)),
            },
            // Open the OS share sheet (Android ACTION_SEND). Fire-and-forget —
            // resolves immediately (no native round-trip needed).
            "share" => match TextArg::deserialize_json(args) {
                Ok(a) => {
                    cx.share_text(&a.text);
                    self.resolve_raw(cx, call_id, "{\"ok\":true}");
                }
                Err(e) => self.reject(cx, call_id, &format!("bad share args: {:?}", e)),
            },
            // Write text to the OS clipboard.
            "clipboard.write" => match TextArg::deserialize_json(args) {
                Ok(a) => {
                    cx.copy_to_clipboard(&a.text);
                    self.resolve_raw(cx, call_id, "{\"ok\":true}");
                }
                Err(e) => self.reject(cx, call_id, &format!("bad clipboard args: {:?}", e)),
            },
            // Post a system notification (Android NotificationManager).
            "notify" => match NotifyArgs::deserialize_json(args) {
                Ok(a) => {
                    cx.show_notification(&a.title, a.body.as_deref().unwrap_or(""));
                    self.resolve_raw(cx, call_id, "{\"ok\":true}");
                }
                Err(e) => self.reject(cx, call_id, &format!("bad notify args: {:?}", e)),
            },
            // ---- fs: sandboxed file storage (pure Rust std::fs under card-fs/) ----
            // Every path is confined to the sandbox by fs_resolve; a card can't
            // reach outside it. Synchronous (fs is fast) → resolves immediately.
            "fs.read" => match PathArg::deserialize_json(args) {
                Ok(a) => match fs_resolve(&a.path)
                    .and_then(|p| std::fs::read_to_string(&p).map_err(|e| format!("read failed: {}", e)))
                {
                    Ok(content) => self.resolve_raw(
                        cx,
                        call_id,
                        &format!("{{\"ok\":true,\"data\":{}}}", content.serialize_json()),
                    ),
                    Err(e) => self.reject(cx, call_id, &e),
                },
                Err(e) => self.reject(cx, call_id, &format!("bad fs.read args: {:?}", e)),
            },
            "fs.write" => match PathDataArg::deserialize_json(args) {
                Ok(a) => {
                    let r = fs_resolve(&a.path).and_then(|p| {
                        if let Some(parent) = p.parent() {
                            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir failed: {}", e))?;
                        }
                        std::fs::write(&p, a.data.as_bytes()).map_err(|e| format!("write failed: {}", e))
                    });
                    match r {
                        Ok(_) => self.resolve_raw(cx, call_id, "{\"ok\":true}"),
                        Err(e) => self.reject(cx, call_id, &e),
                    }
                }
                Err(e) => self.reject(cx, call_id, &format!("bad fs.write args: {:?}", e)),
            },
            "fs.list" => match PathArg::deserialize_json(args) {
                Ok(a) => match fs_resolve(&a.path).and_then(|p| fs_list_json(&p)) {
                    Ok(entries) => {
                        self.resolve_raw(cx, call_id, &format!("{{\"ok\":true,\"entries\":{}}}", entries))
                    }
                    Err(e) => self.reject(cx, call_id, &e),
                },
                Err(e) => self.reject(cx, call_id, &format!("bad fs.list args: {:?}", e)),
            },
            "fs.exists" => match PathArg::deserialize_json(args) {
                Ok(a) => match fs_resolve(&a.path) {
                    Ok(p) => self.resolve_raw(
                        cx,
                        call_id,
                        &format!("{{\"ok\":true,\"exists\":{}}}", p.exists()),
                    ),
                    Err(e) => self.reject(cx, call_id, &e),
                },
                Err(e) => self.reject(cx, call_id, &format!("bad fs.exists args: {:?}", e)),
            },
            "fs.remove" => match PathArg::deserialize_json(args) {
                Ok(a) => {
                    let r = fs_resolve(&a.path).and_then(|p| {
                        if p.is_dir() {
                            std::fs::remove_dir_all(&p)
                        } else {
                            std::fs::remove_file(&p)
                        }
                        .map_err(|e| format!("remove failed: {}", e))
                    });
                    match r {
                        Ok(_) => self.resolve_raw(cx, call_id, "{\"ok\":true}"),
                        Err(e) => self.reject(cx, call_id, &e),
                    }
                }
                Err(e) => self.reject(cx, call_id, &format!("bad fs.remove args: {:?}", e)),
            },
            "fs.mkdir" => match PathArg::deserialize_json(args) {
                Ok(a) => {
                    let r = fs_resolve(&a.path)
                        .and_then(|p| std::fs::create_dir_all(&p).map_err(|e| format!("mkdir failed: {}", e)));
                    match r {
                        Ok(_) => self.resolve_raw(cx, call_id, "{\"ok\":true}"),
                        Err(e) => self.reject(cx, call_id, &e),
                    }
                }
                Err(e) => self.reject(cx, call_id, &format!("bad fs.mkdir args: {:?}", e)),
            },
            // Native file picker (Storage Access Framework). Async: launch here,
            // resolve later when the AndroidDialogResult action arrives (handle_event).
            "dialog.open" => {
                let mime = DialogArgs::deserialize_json(args)
                    .ok()
                    .and_then(|a| a.mime)
                    .unwrap_or_else(|| "*/*".to_string());
                cx.open_file_dialog(call_id, &mime);
            }
            // Stream a URL to a sandbox file natively (large binary never touches
            // JS). Gated (https + SSRF, dest inside the sandbox); progress arrives as
            // download.progress events; resolves with {path} on complete.
            "download" => match DownloadArgs::deserialize_json(args) {
                Ok(a) => match check_https_public(&a.url).and_then(|_| fs_resolve(&a.dest)) {
                    Ok(abs) => {
                        let dest_abs = abs.to_string_lossy().into_owned();
                        self.downloads
                            .push((call_id, a.id.unwrap_or_default(), a.dest.clone()));
                        cx.download_file(call_id, &a.url, &dest_abs);
                    }
                    Err(e) => self.reject(cx, call_id, &format!("download denied: {}", e)),
                },
                Err(e) => self.reject(cx, call_id, &format!("bad download args: {:?}", e)),
            },
            // Default-deny: only registered tools are callable.
            other => self.reject(cx, call_id, &format!("unknown tool: {}", other)),
        }
    }

    fn handle_http_responses(&mut self, cx: &mut Cx, responses: &[NetworkResponse]) {
        if self.pending.is_empty() {
            return;
        }
        for response in responses {
            match response {
                NetworkResponse::HttpResponse { request_id, response } => {
                    if let Some(pos) = self.pending.iter().position(|(rid, _)| rid == request_id) {
                        let (_, call_id) = self.pending.remove(pos);
                        let status = response.status_code;
                        let body = response.get_string_body().unwrap_or_default();
                        let payload = format!(
                            "{{\"ok\":true,\"status\":{},\"body\":{}}}",
                            status,
                            body.serialize_json()
                        );
                        self.resolve_raw(cx, call_id, &payload);
                    }
                }
                NetworkResponse::HttpError { request_id, error } => {
                    if let Some(pos) = self.pending.iter().position(|(rid, _)| rid == request_id) {
                        let (_, call_id) = self.pending.remove(pos);
                        self.reject(cx, call_id, &format!("http error: {}", error.message));
                    }
                }
                _ => {}
            }
        }
    }
}

impl Widget for WebCard {
    fn handle_event(&mut self, cx: &mut Cx, event: &Event, _scope: &mut Scope) {
        if self.settle_timer.is_event(event).is_some() {
            self.load_settled(cx);
        }
        // JS→native bridge: a card called octos.invoke(tool, args) (posted from the
        // WebView's octos_native JavascriptInterface as an AndroidSystemBrowserInvoke
        // action). Dispatch only our own browser's calls.
        if let Event::Actions(actions) = event {
            for action in actions {
                if let Some(inv) = action
                    .downcast_ref::<crate::makepad_platform::event::AndroidSystemBrowserInvoke>()
                {
                    if inv.browser_id == self.browser_id().0.get_value() {
                        self.handle_invoke(cx, inv.call_id, &inv.tool, &inv.args);
                    }
                }
                // Native file-picker result → resolve the pending dialog.open promise.
                if let Some(dr) = action
                    .downcast_ref::<crate::makepad_platform::event::AndroidDialogResult>()
                {
                    let payload = if dr.cancelled {
                        "{\"ok\":true,\"cancelled\":true}".to_string()
                    } else if !dr.error.is_empty() {
                        format!("{{\"ok\":false,\"error\":{}}}", dr.error.serialize_json())
                    } else {
                        format!(
                            "{{\"ok\":true,\"name\":{},\"data\":{}}}",
                            dr.name.serialize_json(),
                            dr.content.serialize_json()
                        )
                    };
                    self.resolve_raw(cx, dr.call_id, &payload);
                }
                // Download progress → emit a download.progress event (keyed by the
                // client download id) so the card can render a progress bar.
                if let Some(p) = action
                    .downcast_ref::<crate::makepad_platform::event::AndroidDownloadProgress>()
                {
                    let dlid = self
                        .downloads
                        .iter()
                        .find(|(cid, _, _)| *cid == p.call_id)
                        .map(|(_, dlid, _)| dlid.clone());
                    if let Some(dlid) = dlid {
                        let payload = format!(
                            "{{\"id\":{},\"done\":{},\"total\":{}}}",
                            dlid.serialize_json(),
                            p.done,
                            p.total
                        );
                        cx.system_browser(self.browser_id()).emit("download.progress", &payload);
                    }
                }
                // Download complete → resolve the invoke with the saved sandbox path.
                if let Some(c) = action
                    .downcast_ref::<crate::makepad_platform::event::AndroidDownloadComplete>()
                {
                    if let Some(pos) = self.downloads.iter().position(|(cid, _, _)| *cid == c.call_id) {
                        let (_, _dlid, dest) = self.downloads.remove(pos);
                        let payload = if c.error.is_empty() {
                            format!("{{\"ok\":true,\"path\":{}}}", dest.serialize_json())
                        } else {
                            format!("{{\"ok\":false,\"error\":{}}}", c.error.serialize_json())
                        };
                        self.resolve_raw(cx, c.call_id, &payload);
                    }
                }
            }
        }
        // Native HTTP responses for our in-flight http.fetch calls.
        if let Event::NetworkResponses(responses) = event {
            self.handle_http_responses(cx, responses);
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
