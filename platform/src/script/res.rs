use crate::makepad_network::HttpRequest;
use crate::script::vm::*;
use crate::*;
use makepad_script::id;
use makepad_script::*;
use std::cell::RefCell;
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Read;
#[cfg(target_arch = "wasm32")]
use std::path::{Path, PathBuf};
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum CxScriptResourceData {
    NotLoaded,
    Loading,
    Loaded(Rc<Vec<u8>>),
    Error(String),
}

#[derive(Clone)]
pub struct CxScriptResource {
    pub abs_path: String,
    pub dependency_path: Option<String>,
    pub web_url: Option<String>,
    pub data: CxScriptResourceData,
    /// One handle per script heap that references this resource, WITH the
    /// heap it belongs to. Handle values are heap-local (a handle indexes
    /// its owning heap's handle table, and each heap's GC marks/sweeps it),
    /// so two isolates hand out equal values for different files: the pair
    /// is the identity, never the number alone (a clock module's alarm
    /// glyph once resolved to the host's speaker icon that way).
    pub handles: Vec<(usize, ScriptHandle)>,
}

impl CxScriptResource {
    /// Bytes held by this resource once loaded; 0 while pending or failed.
    pub fn loaded_len(&self) -> usize {
        match &self.data {
            CxScriptResourceData::Loaded(data) => data.len(),
            _ => 0,
        }
    }

    pub fn is_error(&self) -> bool {
        matches!(self.data, CxScriptResourceData::Error(_))
    }

    pub fn has_handle(&self, heap_key: usize, handle: ScriptHandle) -> bool {
        self.handles.contains(&(heap_key, handle))
    }
}

/// Tracks an in-flight HTTP request that will populate a resource
pub struct CxScriptHttpResource {
    pub request_id: LiveId,
    pub abs_path: String,
}

/// State of a script *data* fetch (sys.weather etc): a live JSON/text pull
/// keyed by URL. Unlike an image `http_resource`, no DSL value holds a handle
/// to it, so it lives in a plain URL-keyed side-table (not the GC'd handle
/// path) — the fetch persists for the card's lifetime without needing a root.
#[derive(Clone)]
pub enum DataFetch {
    /// Request in flight; carries the request_id so the response can be routed.
    Loading(LiveId),
    Loaded(Rc<Vec<u8>>),
    Error,
}

/// Bumped each time a script data fetch newly loads. A live-data-bound widget
/// bakes the "—" placeholder into a Label at eval time; a plain repaint won't
/// re-run the script, so it watches this epoch and re-evaluates once when it
/// changes, picking up the now-loaded value.
static DATA_FETCH_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Retries per data-fetch URL before the Error state sticks (initial attempt
/// not counted — 4 means up to 5 requests total). Retries are LAZY (issued on
/// the card's next evaluation, paced by the failing round-trips), so a larger
/// budget spreads over tens of seconds rather than hammering.
pub const DATA_FETCH_MAX_RETRIES: u8 = 4;

/// User-Agent for a script data fetch, chosen by target host. Yahoo endpoints
/// 429 requests without a browser-ish UA, so that stays the default — but
/// overpass-api.de's Apache rejects the bare "Mozilla/5.0" bot signature with
/// 406 (and OSM etiquette wants an identifying UA anyway), so Overpass gets a
/// descriptive one — as does Nominatim, which requires it outright.
pub fn data_fetch_user_agent(url: &str) -> &'static str {
    if url.contains("overpass") || url.contains("nominatim") {
        "octoscript-appcard/1.0 (+https://github.com/OctoSense-org/Octoscript-AppCard; live card data binding)"
    } else {
        "Mozilla/5.0"
    }
}

/// Bump the data-fetch epoch. Also fired on fetch FAILURE: live-data cards
/// re-evaluate on epoch change, which is what gives an errored URL its lazy
/// retry. Terminates: an exhausted URL fires no new request, so no new failure
/// bumps the epoch again.
pub fn bump_data_fetch_epoch() {
    DATA_FETCH_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The epoch's current value, for tests that assert something bumped it.
#[cfg(test)]
pub fn data_fetch_epoch_for_test() -> u64 {
    DATA_FETCH_EPOCH.load(std::sync::atomic::Ordering::Relaxed)
}

#[derive(Default)]
pub struct CxScriptResources {
    pub resources: Rc<RefCell<Vec<CxScriptResource>>>,
    /// Per-heap path cache: (heap_key, abs_path) → that heap's LOCAL handle.
    /// Never hand one heap's cached handle to another heap (see
    /// [`CxScriptResource::handles`]).
    pub handles_by_abs_path: Rc<RefCell<HashMap<(usize, String), ScriptHandle>>>,
    pub http_resources: Vec<CxScriptHttpResource>,
    /// Live data fetches for script data-binding, keyed by URL (see DataFetch).
    pub data_fetches: Rc<RefCell<HashMap<String, DataFetch>>>,
    /// Retry ledger for data fetches, keyed by URL: (attempts used, earliest
    /// next-retry Instant). Public data APIs shed load with transient 5xx and
    /// 429s can return in milliseconds — an immediate re-issue would burn the
    /// whole budget inside one rate-limit window, so retries back off
    /// exponentially (1s, 2s, 4s, …). Bounded by [`DATA_FETCH_MAX_RETRIES`];
    /// the entry is dropped when the URL finally loads.
    pub data_fetch_retries: Rc<RefCell<HashMap<String, (u8, std::time::Instant)>>>,
    /// Bumped whenever a resource is registered or its load state moves
    /// (a load attempted, an HTTP response or error). A consumer that
    /// caches a negative answer ("this resource cannot be read") keys it on
    /// this, so a resource that appears later is asked for again.
    pub generation: std::cell::Cell<u64>,
}

impl CxScriptResources {

    pub fn get_data_fetch(&self, url: &str) -> Option<DataFetch> {
        self.data_fetches.borrow().get(url).cloned()
    }

    /// Mark a URL as in flight under `request_id`.
    pub fn begin_data_fetch(&self, url: &str, request_id: LiveId) {
        self.data_fetches
            .borrow_mut()
            .insert(url.to_string(), DataFetch::Loading(request_id));
    }

    /// The URL a loading fetch was issued for — for failure logs, so a wire
    /// problem names the wire.
    pub fn data_fetch_url(&self, request_id: LiveId) -> Option<String> {
        let map = self.data_fetches.borrow();
        map.iter().find_map(|(url, fetch)| {
            matches!(fetch, DataFetch::Loading(id) if *id == request_id).then(|| url.clone())
        })
    }

    /// Does `request_id` belong to an in-flight data fetch?
    pub fn is_data_fetch(&self, request_id: LiveId) -> bool {
        self.data_fetches
            .borrow()
            .values()
            .any(|f| matches!(f, DataFetch::Loading(id) if *id == request_id))
    }

    /// Store loaded bytes for the in-flight fetch matching `request_id`.
    pub fn handle_data_fetch_response(&self, request_id: LiveId, data: Vec<u8>) -> bool {
        let mut map = self.data_fetches.borrow_mut();
        for (url, fetch) in map.iter_mut() {
            if matches!(fetch, DataFetch::Loading(id) if *id == request_id) {
                let url = url.clone();
                *fetch = DataFetch::Loaded(Rc::new(data));
                drop(map);
                // Success closes the retry ledger entry.
                self.data_fetch_retries.borrow_mut().remove(&url);
                DATA_FETCH_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return true;
            }
        }
        false
    }

    /// Mark the in-flight fetch matching `request_id` as errored.
    pub fn handle_data_fetch_error(&self, request_id: LiveId) -> bool {
        let mut map = self.data_fetches.borrow_mut();
        for fetch in map.values_mut() {
            if matches!(fetch, DataFetch::Loading(id) if *id == request_id) {
                *fetch = DataFetch::Error;
                return true;
            }
        }
        false
    }

    /// Lazy-retry bookkeeping: does `url` still have retry budget AND has its
    /// backoff window elapsed; if so, consume one unit.
    pub fn take_data_fetch_retry(&self, url: &str) -> bool {
        let now = std::time::Instant::now();
        let mut retries = self.data_fetch_retries.borrow_mut();
        let entry = retries.entry(url.to_string()).or_insert((0, now));
        if entry.0 >= DATA_FETCH_MAX_RETRIES || now < entry.1 {
            return false;
        }
        entry.0 += 1;
        entry.1 = now + std::time::Duration::from_secs(1u64 << (entry.0.min(6) - 1));
        true
    }

    /// Mark the fetch matching `request_id` as errored AND exhaust its retry
    /// budget — for permanent failures (404/403…) where re-issuing the same
    /// request can only fail identically.
    pub fn fail_data_fetch_terminally(&self, request_id: LiveId) -> bool {
        let url = {
            let map = self.data_fetches.borrow();
            map.iter().find_map(|(url, fetch)| {
                matches!(fetch, DataFetch::Loading(id) if *id == request_id).then(|| url.clone())
            })
        };
        let Some(url) = url else { return false };
        self.data_fetches
            .borrow_mut()
            .insert(url.clone(), DataFetch::Error);
        self.data_fetch_retries
            .borrow_mut()
            .insert(url, (DATA_FETCH_MAX_RETRIES, std::time::Instant::now()));
        true
    }

    /// Terminal-failure probe for widgets that render their own fetch: true
    /// once `url` is in the Error state with NO retry budget left — the signal
    /// to stop pumping frames and show a failure state.
    pub fn data_fetch_failed_terminally(&self, url: &str) -> bool {
        if !matches!(self.get_data_fetch(url), Some(DataFetch::Error)) {
            return false;
        }
        self.data_fetch_retries
            .borrow()
            .get(url)
            .is_some_and(|(used, _)| *used >= DATA_FETCH_MAX_RETRIES)
    }
    /// Resolve a heap-local resource handle before storing it in Cx-owned
    /// renderer state. Different script heaps can use the same handle value.
    pub fn path_for_handle(&self, heap_key: usize, handle: ScriptHandle) -> Option<String> {
        self.handles_by_abs_path.borrow().iter()
            .find(|((heap, _), value)| *heap == heap_key && **value == handle)
            .map(|((_, path), _)| path.clone())
    }
    pub fn get_handle_by_abs_path(&self, heap_key: usize, abs_path: &str) -> Option<ScriptHandle> {
        self.handles_by_abs_path
            .borrow()
            .get(&(heap_key, abs_path.to_string()))
            .copied()
    }

    /// Forget everything heaps in `dead` owned — call this the moment a script
    /// heap is dropped WHOLESALE, rather than collected.
    ///
    /// A `heap_key` is an allocation address, so a heap that dies frees its key
    /// for the next heap to land on. The per-handle [`CxScriptResourceGc`] only
    /// runs when the owning heap's own GC sweeps that handle, which never
    /// happens for a heap that is simply dropped — so its `(heap_key, path)`
    /// entries outlive it, and the NEXT heap allocated at that address is
    /// handed a dead heap's handle index for a path it asks about. That index
    /// means nothing in the new heap's own (usually smaller) handle table, and
    /// nothing notices at the time: the value sits in a font object until that
    /// heap's next GC walks it and indexes out of bounds, in code that did
    /// nothing wrong.
    pub fn gc_heaps(&self, dead: &[usize]) {
        if dead.is_empty() {
            return;
        }
        let mut handles = self.handles_by_abs_path.borrow_mut();
        handles.retain(|(heap, _), _| !dead.contains(heap));
        prune_resource_handles(&mut self.resources.borrow_mut(), &handles);
    }

    pub fn insert_resource(&self, heap_key: usize, resource: CxScriptResource) {
        self.handles_by_abs_path.borrow_mut().insert(
            (heap_key, resource.abs_path.clone()),
            resource.handles[0].1,
        );
        self.resources.borrow_mut().push(resource);
        self.bump_generation();
    }

    /// See [`Self::generation`].
    pub fn bump_generation(&self) {
        self.generation.set(self.generation.get().wrapping_add(1));
    }

    /// Attach an additional heap's local handle to an existing resource entry
    /// (by path). Returns true if the entry existed.
    pub fn attach_handle_for_path(
        &self,
        heap_key: usize,
        abs_path: &str,
        handle: ScriptHandle,
    ) -> bool {
        let mut resources = self.resources.borrow_mut();
        if let Some(res) = resources.iter_mut().find(|v| v.abs_path == abs_path) {
            res.handles.push((heap_key, handle));
            self.handles_by_abs_path
                .borrow_mut()
                .insert((heap_key, abs_path.to_string()), handle);
            true
        } else {
            false
        }
    }

    /// Get the data for a resource by its owning heap and handle.
    pub fn get_data(&self, heap_key: usize, handle: ScriptHandle) -> Option<Rc<Vec<u8>>> {
        let resources = self.resources.borrow();
        if let Some(res) = resources.iter().find(|v| v.has_handle(heap_key, handle)) {
            if let CxScriptResourceData::Loaded(data) = &res.data {
                return Some(data.clone());
            }
        }
        None
    }

    /// Store HTTP response data into a resource by request_id.
    /// Returns true if a matching resource was found and updated.
    pub fn handle_http_response(&mut self, request_id: LiveId, data: Vec<u8>) -> bool {
        if let Some(idx) = self
            .http_resources
            .iter()
            .position(|r| r.request_id == request_id)
        {
            let path = self.http_resources.remove(idx).abs_path;
            let mut resources = self.resources.borrow_mut();
            if let Some(res) = resources.iter_mut().find(|r| r.abs_path == path) {
                res.data = CxScriptResourceData::Loaded(Rc::new(data));
                self.bump_generation();
                return true;
            }
        }
        false
    }

    /// Store HTTP error into a resource by request_id.
    /// Returns true if a matching resource was found and updated.
    pub fn handle_http_error(&mut self, request_id: LiveId, error: String) -> bool {
        if let Some(idx) = self
            .http_resources
            .iter()
            .position(|r| r.request_id == request_id)
        {
            let path = self.http_resources.remove(idx).abs_path;
            let mut resources = self.resources.borrow_mut();
            if let Some(res) = resources.iter_mut().find(|r| r.abs_path == path) {
                res.data = CxScriptResourceData::Error(error);
                self.bump_generation();
                return true;
            }
        }
        false
    }

    /// Check if a request_id belongs to an http_resource
    pub fn is_http_resource(&self, request_id: LiveId) -> bool {
        self.http_resources
            .iter()
            .any(|r| r.request_id == request_id)
    }
}

// ---------------------------------------------------------------------------
// Platform-specific resource loading
//
// Loading order depends on whether we are packaged or not:
//   - Desktop unpackaged (package_root is None):
//       1. dependency table (populated at init)
//       2. direct filesystem via abs_path
//       3. error
//   - Desktop packaged (package_root is Some):
//       1. dependency table
//       2. packaged path: package_root/dep_path on filesystem
//       3. error
//   - iOS/tvOS (always packaged, package_root = "makepad"):
//       1. dependency table
//       2. apple bundle: NSBundle.resourcePath / package_root / dep_path
//       3. error
//   - Android (always packaged, package_root = "makepad"):
//       1. dependency table (get_dependency calls JNI asset manager)
//       2. error
//   - Wasm (always packaged, package_root = ""):
//       1. dependency table (may have pre-loaded deps)
//       2. HTTP fetch via web_url (async)
//       3. error
// ---------------------------------------------------------------------------

/// Try to load a resource from the packaged location on Apple platforms
/// using NSBundle to resolve the app bundle's resource path.
#[cfg(any(
    target_os = "ios",
    target_os = "tvos",
    all(target_os = "macos", apple_bundle)
))]
fn load_packaged_resource(cx: &Cx, dep_path: &str) -> Option<Rc<Vec<u8>>> {
    let bundle_path = if let Some(root) = cx.package_root.as_deref() {
        format!("{}/{}", root, dep_path)
    } else {
        dep_path.to_string()
    };
    cx.apple_bundle_load_file(&bundle_path).ok()
}

/// Try to load a resource from the packaged location on desktop.
/// Returns None when not in packaged mode (package_root is None).
///
/// A relative `package_root` (the desktop packagers use `.` beside the executable) is searched
/// both from the working directory and from the executable's own directory, because a launcher
/// is free to start the process anywhere — see `crate::os::cx_native::exe_relative_path`.
#[cfg(all(
    not(target_arch = "wasm32"),
    not(any(target_os = "android", target_os = "ios", target_os = "tvos")),
    not(all(target_os = "macos", apple_bundle))
))]
fn load_packaged_resource(cx: &Cx, dep_path: &str) -> Option<Rc<Vec<u8>>> {
    let root = cx.package_root.as_deref()?;
    let full_path = format!("{}/{}", root, dep_path);
    crate::os::cx_native::read_file_cwd_or_exe_relative(&full_path).map(Rc::new)
}

/// Load a file directly from the filesystem (desktop/mobile only, not wasm).
#[cfg(not(target_arch = "wasm32"))]
fn load_file_direct(abs_path: &str) -> Option<Result<Rc<Vec<u8>>, String>> {
    let mut file = File::open(abs_path).ok()?;
    let mut data = Vec::new();
    match file.read_to_end(&mut data) {
        Ok(_) => Some(Ok(Rc::new(data))),
        Err(e) => Some(Err(format!("Failed to read file: {}", e))),
    }
}

#[cfg(target_arch = "wasm32")]
fn web_resource_request_path(cx: &Cx, dep_path: &str) -> String {
    let mut base_path = String::new();
    if let crate::cx::OsType::Web(params) = &cx.os_type {
        base_path = web_resource_base_path(&params.pathname);
    }
    if base_path.is_empty() {
        dep_path.to_string()
    } else {
        format!(
            "{}/{}",
            base_path.trim_end_matches('/'),
            dep_path.trim_start_matches('/')
        )
    }
}

#[cfg(any(test, target_arch = "wasm32"))]
fn web_resource_base_path(pathname: &str) -> String {
    let pathname = pathname.trim_end_matches('/');
    if pathname.is_empty() {
        return String::new();
    }

    let last_segment = pathname.rsplit('/').next().unwrap_or_default();
    let base_path = if last_segment.contains('.') {
        pathname.rsplit_once('/').map_or("", |(base, _)| base)
    } else {
        pathname
    };
    base_path.trim_start_matches('/').to_string()
}

fn register_crate_resource_parts(
    vm: &mut ScriptVm,
    res_type: ScriptHandleType,
    crate_part: &str,
    file_path: &str,
) -> ScriptValue {
    let Some((abs_path, dependency_path, web_url)) =
        resolve_crate_resource_paths(vm, crate_part, file_path)
    else {
        return NIL;
    };
    let heap_key = vm.bx.heap.heap_key();
    let cx = vm.host.cx_mut();
    if let Some(existing) = cx
        .script_data
        .resources
        .get_handle_by_abs_path(heap_key, &abs_path)
    {
        return existing.into();
    }

    let handle_gc = CxScriptResourceGc {
        resources: cx.script_data.resources.resources.clone(),
        handles_by_abs_path: cx.script_data.resources.handles_by_abs_path.clone(),
        handle: ScriptHandle::ZERO,
        heap_key,
    };
    let handle = vm.bx.heap.new_handle(res_type, Box::new(handle_gc));

    if cx
        .script_data
        .resources
        .attach_handle_for_path(heap_key, &abs_path, handle)
    {
        return handle.into();
    }

    cx.script_data.resources.insert_resource(
        heap_key,
        CxScriptResource {
            abs_path,
            dependency_path,
            web_url,
            data: CxScriptResourceData::NotLoaded,
            handles: vec![(heap_key, handle)],
        },
    );
    handle.into()
}

/// Register a stable logical `crate_name/path` without evaluating a second
/// script branch. FontPolicy uses this to turn only its selected members into
/// resource handles.
pub fn register_crate_resource_path(vm: &mut ScriptVm, logical_path: &str) -> ScriptValue {
    let Some((crate_part, file_path)) = logical_path.split_once('/') else {
        return NIL;
    };
    let res_type = vm.handle_type(id_lut!(res));
    register_crate_resource_parts(vm, res_type, crate_part, file_path)
}

fn font_policy_declares_path(font_set: crate::FontSet, dependency_path: Option<&str>) -> bool {
    let Some(dependency_path) = dependency_path else {
        return false;
    };
    font_set.policy().declares_asset_path(dependency_path)
}

impl Cx {

    /// Get-or-fetch a live data resource (JSON/text) by URL, for script
    /// data-binding helpers like `sys.weather`. Returns the loaded bytes when
    /// ready; otherwise fires the request once (deduped by URL) and returns
    /// None while it loads. This lets generated DSL bind live data instead of
    /// the model hardcoding numbers.
    pub fn script_data_fetch(&mut self, url: &str) -> Option<Rc<Vec<u8>>> {
        // The data behind `sys.*` is fetched on behalf of whichever isolate is
        // running right now; it answers to that isolate's allowlist.
        let heap_key = self.script_vm.as_ref().map(|vm| vm.heap.heap_key()).unwrap_or(0);
        if !makepad_script_std::script_url_allowed(heap_key, url) {
            crate::log!("Script data fetch refused by the host's allowlist: {url}");
            return None;
        }
        match self.script_data.resources.get_data_fetch(url) {
            Some(DataFetch::Loaded(bytes)) => return Some(bytes),
            Some(DataFetch::Loading(_)) => return None,
            Some(DataFetch::Error) => {
                // Lazy retry: re-fire on this evaluation if budget remains,
                // else the Error is terminal.
                if !self.script_data.resources.take_data_fetch_retry(url) {
                    return None;
                }
                crate::log!("Script data fetch retrying: {}", url);
            }
            None => {}
        }
        let request_id = LiveId::unique();
        self.script_data.resources.begin_data_fetch(url, request_id);
        crate::log!("Script data fetch: issuing {url}");
        let mut req = HttpRequest::new(url.to_string(), Default::default());
        // Host-appropriate UA — Yahoo 429s without a browser-ish one, Overpass
        // 406s ON the bare browser signature.
        req.set_header(
            "User-Agent".to_string(),
            data_fetch_user_agent(url).to_string(),
        );
        self.http_request(request_id, req);
        None
    }

    /// Placeholder for a `sys.*` binding whose fetch is unresolved: the loading
    /// glyph while it may still arrive, a visibly distinct "n/a" once the retry
    /// budget is spent. Without the distinction a permanently unreachable source
    /// renders identically to "still loading", forever.
    pub fn script_data_placeholder(&self, url: &str) -> String {
        if self.script_data.resources.data_fetch_failed_terminally(url) {
            "n/a".to_string()
        } else {
            "—".to_string()
        }
    }

    /// Monotonic counter bumped whenever any script data fetch newly loads. A
    /// live-data-bound widget re-evaluates when this changes.
    pub fn script_data_fetch_epoch(&self) -> u64 {
        DATA_FETCH_EPOCH.load(std::sync::atomic::Ordering::Relaxed)
    }
    fn load_script_resource_impl(
        &mut self,
        path: &str,
        #[cfg(target_arch = "wasm32")] crate_manifests: &HashMap<String, String>,
    ) {
        // On wasm, skip loading if we haven't received ToWasmInit yet (os_type is Unknown).
        #[cfg(target_arch = "wasm32")]
        if matches!(self.os_type(), crate::cx::OsType::Unknown) {
            return;
        }

        let is_packaged = self.package_root.is_some();

        #[cfg(target_arch = "wasm32")]
        let mut pending_http = None::<(LiveId, String)>;

        {
            let mut resources = self.script_data.resources.resources.borrow_mut();
            let Some(res) = resources.iter_mut().find(|res| res.abs_path == path) else {
                return;
            };

            if !matches!(res.data, CxScriptResourceData::NotLoaded) {
                return;
            }
            // Every path below leaves NotLoaded (Loaded, Loading or Error).
            self.script_data.resources.bump_generation();

            #[cfg(target_arch = "wasm32")]
            if res.dependency_path.is_none() {
                if let Some(dep_path) = resolve_dependency_path_from_manifests(
                    &res.abs_path,
                    None,
                    None,
                    crate_manifests,
                ) {
                    res.dependency_path = Some(dep_path);
                }
            }

            #[cfg(target_arch = "wasm32")]
            if let Some(dep_path) = res.dependency_path.as_deref() {
                res.web_url = Some(format!("/{}", web_resource_request_path(self, dep_path)));
            } else {
                res.web_url = None;
            }

            if let Some(dep_path) = res.dependency_path.as_deref() {
                if let Ok(data) = self.get_dependency(dep_path) {
                    res.data = CxScriptResourceData::Loaded(data);
                    return;
                }
            }

            #[cfg(target_arch = "wasm32")]
            {
                if let Some(url) = res.web_url.clone() {
                    let request_id = LiveId::unique();
                    res.data = CxScriptResourceData::Loading;
                    pending_http = Some((request_id, url));
                } else {
                    res.data = CxScriptResourceData::Error(format!(
                        "Failed to load resource: {} (dep: {:?}, packaged: {})",
                        res.abs_path, res.dependency_path, is_packaged,
                    ));
                }
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                if is_packaged {
                    #[cfg(not(target_os = "android"))]
                    if let Some(dep_path) = res.dependency_path.as_deref() {
                        if let Some(data) = load_packaged_resource(self, dep_path) {
                            res.data = CxScriptResourceData::Loaded(data);
                            return;
                        }
                    }
                } else if let Some(result) = load_file_direct(&res.abs_path) {
                    res.data = match result {
                        Ok(data) => CxScriptResourceData::Loaded(data),
                        Err(e) => CxScriptResourceData::Error(e),
                    };
                    return;
                }
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                res.data = CxScriptResourceData::Error(format!(
                    "Failed to load resource: {} (dep: {:?}, packaged: {})",
                    res.abs_path, res.dependency_path, is_packaged,
                ));
            }
        }

        #[cfg(target_arch = "wasm32")]
        if let Some((request_id, url)) = pending_http {
            self.script_data
                .resources
                .http_resources
                .push(CxScriptHttpResource { request_id, abs_path: path.to_string() });
            self.http_request(request_id, HttpRequest::new(url, Default::default()));
        }
    }

    pub fn load_script_resource(&mut self, heap_key: usize, handle: ScriptHandle) {
        let Some(path) = self.get_resource_abs_path(heap_key, handle) else { return };
        self.load_script_resource_by_path(&path);
    }

    /// Load a resource whose identity was resolved in its owning script heap.
    pub fn load_script_resource_by_path(&mut self, path: &str) {
        #[cfg(target_arch = "wasm32")]
        let crate_manifests = self.script_data.crate_manifests.borrow().clone();

        self.load_script_resource_impl(
            path,
            #[cfg(target_arch = "wasm32")]
            &crate_manifests,
        );
    }

    /// Start every resource declared by the selected application font set.
    /// This changes timing only: it never adds resources beyond FontPolicy.
    pub fn preload_font_set(&mut self) {
        let policy = self.font_set().policy();
        let paths = {
            let resources = self.script_data.resources.resources.borrow();
            policy
                .assets
                .iter()
                .filter_map(|asset| {
                    resources
                        .iter()
                        .find(|resource| {
                            resource.dependency_path.as_deref() == Some(asset.resource_path)
                        })
                        .map(|resource| resource.abs_path.clone())
                })
                .collect::<Vec<_>>()
        };
        for path in paths {
            self.load_script_resource_by_path(&path);
        }
    }

    /// Load all script resources that are still pending.
    ///
    /// Each platform uses a different loading strategy:
    /// - Desktop unpackaged: dependency table, then direct filesystem
    /// - Desktop packaged: dependency table, then package_root-relative file
    /// - iOS/tvOS: dependency table, then apple bundle
    /// - Android: dependency table only (JNI asset manager is inside get_dependency)
    /// - Wasm: dependency table, then async HTTP fetch
    pub fn load_all_script_resources(&mut self) {
        // On wasm, skip loading if we haven't received ToWasmInit yet (os_type is Unknown).
        #[cfg(target_arch = "wasm32")]
        if matches!(self.os_type(), crate::cx::OsType::Unknown) {
            return;
        }

        // On wasm, resolve web_url for resources that don't have one yet.
        #[cfg(target_arch = "wasm32")]
        let crate_manifests = self.script_data.crate_manifests.borrow().clone();

        let font_set = self.font_set();
        let paths = {
            let resources = self.script_data.resources.resources.borrow();
            resources
                .iter()
                .filter(|resource| {
                    !font_policy_declares_path(font_set, resource.dependency_path.as_deref())
                })
                .map(|res| res.abs_path.clone())
                .collect::<Vec<_>>()
        };

        let _mp_t0 = self.seconds_since_app_start();
        let _mp_n = paths.len();
        for path in paths {
            self.load_script_resource_impl(
                &path,
                #[cfg(target_arch = "wasm32")]
                &crate_manifests,
            );
        }
        if crate::startup_trace_enabled() {
            let bytes: usize = self
                .script_data
                .resources
                .resources
                .borrow()
                .iter()
                .filter_map(|r| match &r.data {
                    CxScriptResourceData::Loaded(d) => Some(d.len()),
                    _ => None,
                })
                .sum();
            crate::startup_trace(&format!(
                "load_all_script_resources ({} files, {:.2} MB, {:.2} ms)",
                _mp_n,
                bytes as f64 / (1024.0 * 1024.0),
                (self.seconds_since_app_start() - _mp_t0).max(0.0) * 1000.0
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{font_policy_declares_path, web_resource_base_path};

    #[test]
    fn heap_local_resource_handles_do_not_alias_font_bytes_or_http_responses() {
        use super::*;
        let mut cx = Cx::new(Box::new(|_, _| {}));
        let handle = ScriptHandle::ZERO;
        for (heap, path, bytes) in [(1, "test://icon.svg", b"<svg>".to_vec()), (2, "test://font.ttf", b"font".to_vec())] {
            cx.script_data.resources.insert_resource(heap, CxScriptResource {
                abs_path: path.into(), dependency_path: None, web_url: None,
                data: CxScriptResourceData::Loaded(Rc::new(bytes)), handles: vec![(heap, handle)],
            });
        }
        let path = cx.script_data.resources.path_for_handle(2, handle).unwrap();
        assert_eq!(cx.get_resource_font_bytes_by_path(&path).unwrap().as_slice(), b"font");
        let request_id = LiveId::unique();
        cx.script_data.resources.http_resources.push(CxScriptHttpResource { request_id, abs_path: path });
        assert!(cx.script_data.resources.handle_http_response(request_id, b"new font".to_vec()));
        assert_eq!(cx.get_resource_font_bytes_by_path("test://font.ttf").unwrap().as_slice(), b"new font");
        assert_eq!(cx.get_resource_font_bytes_by_path("test://icon.svg").unwrap().as_slice(), b"<svg>");
    }

    #[test]
    fn equal_handle_values_in_two_heaps_name_their_own_resources() {
        use super::*;
        // Two isolates mint handle value ZERO for different files: the host's
        // speaker icon and a module's alarm glyph. Each heap gets its own bytes.
        let cx = Cx::new(Box::new(|_, _| {}));
        let handle = ScriptHandle::ZERO;
        for (heap, path, bytes) in [(1, "apps/wm/resources/icons/volume-0.svg", b"<svg speaker>".to_vec()), (2, "apps/clock/resources/icons/alarm.svg", b"<svg bell>".to_vec())] {
            cx.script_data.resources.insert_resource(heap, CxScriptResource {
                abs_path: path.into(), dependency_path: None, web_url: None,
                data: CxScriptResourceData::Loaded(Rc::new(bytes)), handles: vec![(heap, handle)],
            });
        }
        assert_eq!(cx.get_resource(1, handle).unwrap().as_slice(), b"<svg speaker>");
        assert_eq!(cx.get_resource(2, handle).unwrap().as_slice(), b"<svg bell>");
        assert_eq!(cx.get_resource_abs_path(2, handle).as_deref(), Some("apps/clock/resources/icons/alarm.svg"));
        assert!(cx.get_resource(3, handle).is_none(), "a heap that never registered it sees nothing");
        // The same file from a third heap shares the entry, still by pair.
        assert!(cx.script_data.resources.attach_handle_for_path(3, "apps/clock/resources/icons/alarm.svg", handle));
        assert_eq!(cx.get_resource(3, handle).unwrap().as_slice(), b"<svg bell>");
    }

    #[test]
    fn collecting_a_heap_keeps_other_heaps_equal_resource_handles() {
        use super::*;
        let resources = CxScriptResources::default();
        let handle = ScriptHandle::ZERO;
        for (heap, path) in [(1, "icon.svg"), (2, "font.ttf")] {
            resources.insert_resource(heap, CxScriptResource {
                abs_path: path.into(), dependency_path: None, web_url: None,
                data: CxScriptResourceData::NotLoaded, handles: vec![(heap, handle)],
            });
        }
        resources.attach_handle_for_path(3, "font.ttf", handle);
        let mut gc = CxScriptResourceGc {
            resources: resources.resources.clone(), handles_by_abs_path: resources.handles_by_abs_path.clone(),
            heap_key: 1, handle,
        };
        gc.gc();
        assert_eq!(resources.resources.borrow().len(), 1);
        assert_eq!(resources.path_for_handle(2, handle).as_deref(), Some("font.ttf"));
        resources.gc_heaps(&[2]);
        assert_eq!(resources.resources.borrow().len(), 1);
        assert_eq!(resources.path_for_handle(3, handle).as_deref(), Some("font.ttf"));
        resources.gc_heaps(&[3]);
        assert!(resources.resources.borrow().is_empty());
    }

    #[test]
    fn eager_resource_loading_defers_the_selected_font_policy() {
        assert!(font_policy_declares_path(
            crate::FontSet::Latin,
            Some("makepad_widgets/resources/IBMPlexSans-Text.ttf")
        ));
        assert!(font_policy_declares_path(
            crate::FontSet::Latin,
            Some("makepad_widgets/resources/LXGWWenKaiRegular.ttf")
        ));
        assert!(font_policy_declares_path(
            crate::FontSet::International,
            Some("makepad_widgets/resources/LXGWWenKaiRegular.ttf")
        ));
    }

    #[test]
    fn derives_web_resource_base_path_from_browser_pathname() {
        assert_eq!(web_resource_base_path("/"), "");
        assert_eq!(web_resource_base_path("/index.html"), "");
        assert_eq!(
            web_resource_base_path("/makepad-example-splash/"),
            "makepad-example-splash"
        );
        assert_eq!(
            web_resource_base_path("/makepad-example-splash/index.html"),
            "makepad-example-splash"
        );
        assert_eq!(
            web_resource_base_path("/examples/splash/index.html"),
            "examples/splash"
        );
    }
}

pub struct CxScriptResourceGc {
    pub resources: Rc<RefCell<Vec<CxScriptResource>>>,
    pub handles_by_abs_path: Rc<RefCell<HashMap<(usize, String), ScriptHandle>>>,
    pub handle: ScriptHandle,
    /// Heap identity of the VM this handle was minted in — the Gc must only
    /// detach ITS heap's handle/cache entry, never the whole shared entry.
    pub heap_key: usize,
}

/// Keep only references still owned by a living heap, grouped by resource
/// path so equal local handles in different heaps cannot delete each other.
fn prune_resource_handles(
    resources: &mut Vec<CxScriptResource>,
    handles: &HashMap<(usize, String), ScriptHandle>,
) {
    let mut live: HashMap<&str, Vec<(usize, ScriptHandle)>> = HashMap::new();
    for ((heap, path), handle) in handles {
        live.entry(path.as_str()).or_default().push((*heap, *handle));
    }
    resources.retain_mut(|resource| {
        let Some(owners) = live.get(resource.abs_path.as_str()) else { return false };
        resource.handles.retain(|handle| owners.contains(handle));
        !resource.handles.is_empty()
    });
}

impl ScriptHandleGc for CxScriptResourceGc {
    fn gc(&mut self) {
        // The numeric handle is only meaningful in its owning heap. Removing
        // every matching number can erase another isolate's fonts or icons.
        let mut handles = self.handles_by_abs_path.borrow_mut();
        handles.retain(|(heap, _), handle| *heap != self.heap_key || *handle != self.handle);
        prune_resource_handles(&mut self.resources.borrow_mut(), &handles);
    }

    fn set_handle(&mut self, handle: ScriptHandle) {
        self.handle = handle
    }
}

// ---------------------------------------------------------------------------
// Crate path resolution
// ---------------------------------------------------------------------------

/// Parses a crate path like "self:resources/file.jpg" or "other_crate:path/file.ext"
/// Returns (crate_part, file_path)
fn parse_crate_path(path: &str) -> Option<(&str, &str)> {
    let mut split = path.splitn(2, ':');
    let crate_part = split.next()?;
    let file_path = split.next()?;
    Some((crate_part, file_path))
}

/// Accept both `self:path` and `self://path` syntax.
fn strip_crate_resource_leading_slashes(file_path: &str) -> &str {
    file_path.trim_start_matches('/')
}

fn normalize_dependency_file_path(path: &str) -> Option<String> {
    let mut stack: Vec<&str> = Vec::new();
    let normalized = path.replace('\\', "/");
    for part in normalized.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if stack.pop().is_none() {
                    return None;
                }
            }
            other => stack.push(other),
        }
    }
    Some(stack.join("/"))
}

#[cfg(target_arch = "wasm32")]
fn normalize_path(path: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            std::path::Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            std::path::Component::RootDir => out.push(comp.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !out.pop() {
                    return None;
                }
            }
            std::path::Component::Normal(part) => out.push(part),
        }
    }
    Some(out)
}

#[cfg(target_arch = "wasm32")]
fn normalize_manifest_relative_path(path: &Path) -> Option<String> {
    normalize_dependency_file_path(&path.to_string_lossy().replace('\\', "/"))
}

#[cfg(target_arch = "wasm32")]
fn resolve_dependency_path_from_manifests(
    abs_path: &str,
    default_crate_name: Option<&str>,
    default_manifest_path: Option<&str>,
    manifests: &HashMap<String, String>,
) -> Option<String> {
    let abs_norm = normalize_path(Path::new(abs_path))?;
    let mut best: Option<(usize, String)> = None;

    let mut candidates = Vec::<(String, String)>::new();
    if let (Some(crate_name), Some(manifest_path)) = (default_crate_name, default_manifest_path) {
        candidates.push((crate_name.to_string(), manifest_path.to_string()));
    }
    for (crate_name, manifest_path) in manifests {
        candidates.push((crate_name.clone(), manifest_path.clone()));
    }

    for (crate_name, manifest_path) in candidates {
        let Some(manifest_norm) = normalize_path(Path::new(&manifest_path)) else {
            continue;
        };
        let Ok(rel) = abs_norm.strip_prefix(&manifest_norm) else {
            continue;
        };
        let Some(rel_norm) = normalize_manifest_relative_path(rel) else {
            continue;
        };
        let dep_path = format!("{}/{}", crate_name, rel_norm);
        let manifest_len = manifest_norm.to_string_lossy().len();
        match &best {
            Some((best_len, _)) if *best_len >= manifest_len => {}
            _ => best = Some((manifest_len, dep_path)),
        }
    }

    best.map(|(_, dep_path)| dep_path)
}

// ---------------------------------------------------------------------------
// Crate resource path resolution (wasm vs native)
// ---------------------------------------------------------------------------

/// On wasm, resolve crate resource paths and compute the web_url for HTTP fetching.
/// The abs_path uses normalized Path joining to handle .. segments correctly.
#[cfg(target_arch = "wasm32")]
fn resolve_crate_resource_paths(
    vm: &mut ScriptVm,
    crate_part: &str,
    file_path: &str,
) -> Option<(String, Option<String>, Option<String>)> {
    let file_path = strip_crate_resource_leading_slashes(file_path);
    let manifests = vm.bx.code.crate_manifests.borrow().clone();
    let (abs_path, default_crate_name, default_manifest_path) = if crate_part == "self" {
        let bodies = vm.bx.code.bodies.borrow();
        let body_id = vm.thread().trap.ip.body as usize;
        let body = bodies.get(body_id)?;
        let script_mod = match &body.source {
            ScriptSource::Mod(script_mod) => script_mod,
            _ => return None,
        };
        let abs_path = normalize_path(&Path::new(&script_mod.cargo_manifest_path).join(file_path))
            .map(|path| path.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|| {
                let mut fallback = script_mod.cargo_manifest_path.clone();
                fallback.push('/');
                fallback.push_str(file_path);
                fallback
            });
        let crate_name = script_mod
            .module_path
            .split("::")
            .next()
            .unwrap_or("")
            .replace('-', "_");
        (
            abs_path,
            Some(crate_name),
            Some(script_mod.cargo_manifest_path.clone()),
        )
    } else {
        let crate_name = crate_part.replace('-', "_");
        let manifest_path = manifests.get(&crate_name)?.clone();
        let abs_path = normalize_path(&Path::new(&manifest_path).join(file_path))
            .map(|path| path.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|| {
                let mut fallback = manifest_path.clone();
                fallback.push('/');
                fallback.push_str(file_path);
                fallback
            });
        (abs_path, Some(crate_name), Some(manifest_path))
    };

    let dependency_path = resolve_dependency_path_from_manifests(
        &abs_path,
        default_crate_name.as_deref(),
        default_manifest_path.as_deref(),
        &manifests,
    );
    let web_url = dependency_path.as_ref().map(|path| format!("/{}", path));
    if web_url.is_none() {
        crate::log!(
            "crate_resource_unmapped crate_part={} file_path={} abs_path={}",
            crate_part,
            file_path,
            abs_path
        );
    }
    Some((abs_path, dependency_path, web_url))
}

/// On native platforms, resolve crate resource paths.
/// Returns (abs_path, dependency_path). web_url is always None on native.
#[cfg(not(target_arch = "wasm32"))]
fn resolve_crate_resource_paths(
    vm: &mut ScriptVm,
    crate_part: &str,
    file_path: &str,
) -> Option<(String, Option<String>, Option<String>)> {
    let file_path = strip_crate_resource_leading_slashes(file_path);
    let (abs_path, crate_name) = if crate_part == "self" {
        let bodies = vm.bx.code.bodies.borrow();
        let body_id = vm.thread().trap.ip.body as usize;
        let body = bodies.get(body_id)?;
        let script_mod = match &body.source {
            ScriptSource::Mod(script_mod) => script_mod,
            _ => return None,
        };
        let mut abs_path = script_mod.cargo_manifest_path.clone();
        abs_path.push('/');
        abs_path.push_str(file_path);
        let crate_name = script_mod
            .module_path
            .split("::")
            .next()
            .unwrap_or("")
            .replace('-', "_");
        (abs_path, crate_name)
    } else {
        let crate_name = crate_part.replace('-', "_");
        let manifests = vm.bx.code.crate_manifests.borrow();
        let manifest_path = manifests.get(&crate_name)?.clone();
        let mut abs_path = manifest_path;
        abs_path.push('/');
        abs_path.push_str(file_path);
        (abs_path, crate_name)
    };

    let dependency_path = normalize_dependency_file_path(file_path)
        .map(|file_path| format!("{}/{}", crate_name, file_path));
    Some((abs_path, dependency_path, None))
}

fn script_value_to_u8_bytes(vm: &mut ScriptVm, value: ScriptValue) -> Option<Vec<u8>> {
    if let Some(array) = value.as_array() {
        if let ScriptArrayStorage::U8(data) = vm.bx.heap.array_storage(array) {
            return Some(data.clone());
        }
    }
    None
}

pub fn script_mod(vm: &mut ScriptVm) {
    let res = vm.new_module(id!(res));
    let res_type = vm.new_handle_type(id_lut!(res));

    // Get the path of the resource
    vm.set_handle_getter(res_type, |vm, pself, prop| {
        if let Some(handle) = pself.as_handle() {
            let heap_key = vm.bx.heap.heap_key();
            let cx = vm.host.cx_mut();
            let resources = cx.script_data.resources.resources.borrow();
            if let Some(res) = resources.iter().find(|v| v.has_handle(heap_key, handle)) {
                match prop {
                    _ if prop == id!(path) => {
                        let path = res.abs_path.clone();
                        drop(resources);
                        return vm
                            .new_string_with(|_vm, s| {
                                s.push_str(&path);
                            })
                            .into();
                    }
                    _ if prop == id!(is_loaded) => {
                        return matches!(res.data, CxScriptResourceData::Loaded(_)).into()
                    }
                    _ if prop == id!(is_error) => {
                        return matches!(res.data, CxScriptResourceData::Error(_)).into()
                    }
                    _ if prop == id!(error) => {
                        if let CxScriptResourceData::Error(ref e) = res.data {
                            let err = e.clone();
                            drop(resources);
                            return vm
                                .new_string_with(|_vm, s| {
                                    s.push_str(&err);
                                })
                                .into();
                        }
                        return NIL;
                    }
                    _ if prop == id!(data) => {
                        if let CxScriptResourceData::Loaded(ref data) = res.data {
                            let data: Vec<u8> = (**data).clone();
                            drop(resources);
                            return vm.bx.heap.new_array_from_vec_u8(data).into();
                        }
                        return NIL;
                    }
                    _ => {}
                }
            }
        }
        script_err_not_found!(vm.trap(), "invalid res prop")
    });

    // res.load_all() - loads all pending resources via the active platform backend
    vm.add_method(
        res,
        id_lut!(load_all_resources),
        script_args_def!(value = NIL),
        move |vm, args| {
            let value = script_value!(vm, args.value);
            let cx = vm.host.cx_mut();
            cx.load_all_script_resources();
            value
        },
    );

    // res.file("/absolute/path/to/file")
    // Uses an absolute file path directly
    vm.add_method(
        res,
        id_lut!(file_resource),
        script_args_def!(path = NIL),
        move |vm, args| {
            let path = script_value!(vm, args.path);
            if !path.is_string_like() {
                return script_err_type_mismatch!(vm.trap(), "invalid res arg type");
            }

            if let Some(abs_path) = vm.string_with(path, |_vm, s| s.to_string()) {
                let heap_key = vm.bx.heap.heap_key();
                let cx = vm.host.cx_mut();
                if let Some(existing) = cx
                    .script_data
                    .resources
                    .get_handle_by_abs_path(heap_key, &abs_path)
                {
                    return existing.into();
                }

                let handle_gc = CxScriptResourceGc {
                    resources: cx.script_data.resources.resources.clone(),
                    handles_by_abs_path: cx.script_data.resources.handles_by_abs_path.clone(),
                    handle: ScriptHandle::ZERO,
                    heap_key,
                };
                let handle = vm.bx.heap.new_handle(res_type, Box::new(handle_gc));

                // Another heap may already track this path — attach our local
                // handle to the shared entry instead of duplicating it.
                if cx
                    .script_data
                    .resources
                    .attach_handle_for_path(heap_key, &abs_path, handle)
                {
                    return handle.into();
                }

                cx.script_data.resources.insert_resource(
                    heap_key,
                    CxScriptResource {
                        abs_path,
                        dependency_path: None,
                        web_url: None,
                        data: CxScriptResourceData::NotLoaded,
                        handles: vec![(heap_key, handle)],
                    },
                );

                return handle.into();
            }

            script_err_type_mismatch!(vm.trap(), "invalid res arg type")
        },
    );

    // res.crate("self:path/to/file") or res.crate("crate_name:path/to/file")
    // Resolves a crate-relative path to an absolute path
    vm.add_method(
        res,
        id_lut!(crate_resource),
        script_args_def!(path = NIL),
        move |vm, args| {
            let path = script_value!(vm, args.path);
            if !path.is_string_like() {
                return script_err_type_mismatch!(vm.trap(), "invalid res arg type");
            }

            let path_string = vm.string_with(path, |_vm, s| s.to_string());

            if let Some(path_string) = path_string {
                if let Some((crate_part, file_path)) = parse_crate_path(&path_string) {
                    return register_crate_resource_parts(vm, res_type, crate_part, file_path);
                }
            }

            script_err_type_mismatch!(vm.trap(), "invalid res arg type")
        },
    );

    // res.http_resource("https://example.com/file.svg")
    // Loads a resource from an HTTP URL asynchronously
    vm.add_method(
        res,
        id_lut!(http_resource),
        script_args_def!(url = NIL),
        move |vm, args| {
            let url = script_value!(vm, args.url);
            if !url.is_string_like() {
                return script_err_type_mismatch!(vm.trap(), "invalid res arg type");
            }

            if let Some(url_string) = vm.string_with(url, |_vm, s| s.to_string()) {
                let heap_key = vm.bx.heap.heap_key();
                // Artwork is a way out of the isolate too: a card with no
                // network grant was seen fetching nine images through here.
                if !makepad_script_std::script_media_url_allowed(heap_key, &url_string) {
                    // Logged as well as raised: the widget that asked draws
                    // nothing, and a silent blank is the hardest bug to find.
                    crate::log!("Script resource refused by the host's allowlist: {url_string}");
                    return script_err_io!(vm.trap(), "this app may not load {}", url_string);
                }
                let cx = vm.host.cx_mut();
                if let Some(existing) = cx
                    .script_data
                    .resources
                    .get_handle_by_abs_path(heap_key, &url_string)
                {
                    return existing.into();
                }
                let handle_gc = CxScriptResourceGc {
                    resources: cx.script_data.resources.resources.clone(),
                    handles_by_abs_path: cx.script_data.resources.handles_by_abs_path.clone(),
                    handle: ScriptHandle::ZERO,
                    heap_key,
                };
                let handle = vm.bx.heap.new_handle(res_type, Box::new(handle_gc));

                if cx
                    .script_data
                    .resources
                    .attach_handle_for_path(heap_key, &url_string, handle)
                {
                    return handle.into();
                }

                // Create the resource in Loading state
                cx.script_data.resources.insert_resource(
                    heap_key,
                    CxScriptResource {
                        abs_path: url_string.clone(),
                        dependency_path: None,
                        web_url: None,
                        data: CxScriptResourceData::Loading,
                        handles: vec![(heap_key, handle)],
                    },
                );

                // Fire the HTTP request
                let request_id = LiveId::unique();
                cx.script_data
                    .resources
                    .http_resources
                    .push(CxScriptHttpResource { request_id, abs_path: url_string.clone() });
                cx.http_request(request_id, HttpRequest::new(url_string, Default::default()));

                return handle.into();
            }

            script_err_type_mismatch!(vm.trap(), "invalid res arg type")
        },
    );

    // res.binary_resource(bytes_u8_array)
    // Creates an in-memory resource directly from bytes.
    vm.add_method(
        res,
        id_lut!(binary_resource),
        script_args_def!(data = NIL),
        move |vm, args| {
            let data = script_value!(vm, args.data);
            let Some(bytes) = script_value_to_u8_bytes(vm, data) else {
                return script_err_type_mismatch!(
                    vm.trap(),
                    "binary_resource expects a U8 byte array"
                );
            };

            let heap_key = vm.bx.heap.heap_key();
            let cx = vm.host.cx_mut();
            let handle_gc = CxScriptResourceGc {
                resources: cx.script_data.resources.resources.clone(),
                handles_by_abs_path: cx.script_data.resources.handles_by_abs_path.clone(),
                handle: ScriptHandle::ZERO,
                heap_key,
            };
            let handle = vm.bx.heap.new_handle(res_type, Box::new(handle_gc));

            cx.script_data.resources.insert_resource(
                heap_key,
                CxScriptResource {
                    abs_path: format!("binary://{}", LiveId::unique().0),
                    dependency_path: None,
                    web_url: None,
                    data: CxScriptResourceData::Loaded(Rc::new(bytes)),
                    handles: vec![(heap_key, handle)],
                },
            );

            handle.into()
        },
    );
}
