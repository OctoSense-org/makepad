//! The entry points ArkTS and the XComponent call. They arrive on the UI thread (the vsync on a
//! thread of its own) and only write the shell's shared state.
use super::{shell::{lock, OhosParams, Shared}, surface::Window, sys::*};
use crate::{buffers::Wire, clock_ns, input, lifecycle, TouchPoint};
use std::ffi::{c_void, CStr, CString};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static VSYNC: AtomicUsize = AtomicUsize::new(0);
static FRAME_POSTED: AtomicBool = AtomicBool::new(false);
/// ArkTS's `(name, value)` function as a thread-safe one: how the shell asks the ability for
/// what only ArkTS can do (system bars, the clipboard).
static CALL: AtomicUsize = AtomicUsize::new(0);

static mut MODULE: napi_module = napi_module {
    nm_version: 1, nm_flags: 0, nm_filename: std::ptr::null(), nm_register_func: Some(module_init),
    nm_modname: c"makepad".as_ptr(), nm_priv: null_mut(), reserved: [null_mut(); 4],
};
static mut CALLBACKS: OH_NativeXComponent_Callback = OH_NativeXComponent_Callback {
    OnSurfaceCreated: Some(surface_created), OnSurfaceChanged: Some(surface_changed),
    OnSurfaceDestroyed: Some(surface_destroyed), DispatchTouchEvent: Some(dispatch_touch),
};

/// Registers `libmakepad.so` as the Node-API module `makepad`.
/// # Safety
/// Called once, as the library loads.
pub(super) unsafe fn register_module() { napi_module_register(std::ptr::addr_of_mut!(MODULE)); }

/// Runs for the ability's `import` and again for each XComponent that names the library, which
/// then finds itself in the exports.
unsafe extern "C" fn module_init(env: napi_env, exports: napi_value) -> napi_value {
    let methods: [(&CStr, napi_callback); 9] = [
        (c"start", start), (c"foreground", foreground), (c"background", background), (c"focus", focus), (c"destroy", destroy),
        (c"avoidArea", avoid_area), (c"keyboard", keyboard), (c"backPress", back_press), (c"nightMode", night_mode),
    ];
    let properties: Vec<napi_property_descriptor> = methods.iter().map(|(name, method)| napi_property_descriptor {
        utf8name: name.as_ptr(), name: null_mut(), method: Some(*method), getter: None, setter: None, value: null_mut(), attributes: 0, data: null_mut(),
    }).collect();
    napi_define_properties(env, exports, properties.len(), properties.as_ptr());
    let mut has = false;
    if napi_has_named_property(env, exports, OH_NATIVE_XCOMPONENT_OBJ.as_ptr(), &mut has) == NAPI_OK && has {
        let (mut object, mut component) = (null_mut(), null_mut());
        if napi_get_named_property(env, exports, OH_NATIVE_XCOMPONENT_OBJ.as_ptr(), &mut object) == NAPI_OK
            && napi_unwrap(env, object, &mut component) == NAPI_OK && !component.is_null() {
            if OH_NativeXComponent_RegisterCallback(component.cast(), std::ptr::addr_of_mut!(CALLBACKS)) != 0 {
                eprintln!("[hal-ohos] the XComponent refused its callbacks");
            }
        }
    }
    exports
}

unsafe fn args<const N: usize>(env: napi_env, info: napi_callback_info) -> [napi_value; N] {
    let mut argc = N;
    let mut argv = [null_mut(); N];
    napi_get_cb_info(env, info, &mut argc, argv.as_mut_ptr(), null_mut(), null_mut());
    argv
}
unsafe fn string(env: napi_env, value: napi_value) -> String {
    let mut len = 0;
    if value.is_null() || napi_get_value_string_utf8(env, value, null_mut(), 0, &mut len) != NAPI_OK { return String::new(); }
    let mut buf = vec![0u8; len + 1];
    napi_get_value_string_utf8(env, value, buf.as_mut_ptr().cast(), buf.len(), &mut len);
    buf.truncate(len);
    String::from_utf8_lossy(&buf).into_owned()
}
unsafe fn number(env: napi_env, value: napi_value) -> f64 {
    let mut out = 0.;
    if !value.is_null() { napi_get_value_double(env, value, &mut out); }
    out
}
unsafe fn boolean(env: napi_env, value: napi_value) -> bool {
    let mut out = false;
    if !value.is_null() { napi_get_value_bool(env, value, &mut out); }
    out
}
unsafe fn undefined(env: napi_env) -> napi_value {
    let mut out = null_mut();
    napi_get_undefined(env, &mut out);
    out
}

/// `start(filesDir, cacheDir, density, osVersion, appBuild, nightMode, resourceManager, call, launch)`:
/// the ability's `onCreate`. The app starts here, on a thread of its own.
///
/// `launch` is the Want's `makepad.*` parameters as `NAME=value` lines, which become the app's
/// environment (an ability gets none of its own): `aa start ... --ps makepad.HAL_LOG 1` sets
/// `MAKEPAD_HAL_LOG=1`, as `cargo makepad android run` passes its extras.
unsafe extern "C" fn start(env: napi_env, info: napi_callback_info) -> napi_value {
    let a = args::<9>(env, info);
    for line in string(env, a[8]).lines() {
        if let Some((name, value)) = line.split_once('=').filter(|(name, _)| !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')) {
            std::env::set_var(format!("MAKEPAD_{name}"), value);
        }
    }
    let density = number(env, a[2]) as f32;
    let params = OhosParams {
        data_path: string(env, a[0]), cache_path: string(env, a[1]), density: if density.is_finite() && density > 0. { density } else { 1. },
        os_version: string(env, a[3]), app_build: string(env, a[4]), night_mode: boolean(env, a[5]),
    };
    // The process starts without a usable HOME or TMPDIR: the ability's own directories are both.
    if std::env::var_os("HOME").is_none_or(|home| home.is_empty() || home == "/") { std::env::set_var("HOME", &params.data_path); }
    if std::env::var_os("TMPDIR").is_none_or(|dir| dir.is_empty()) { std::env::set_var("TMPDIR", &params.cache_path); }
    if crate::backend::log_enabled() { eprintln!("[hal-ohos] start: {params:?}"); }
    lock(&Shared::global().state).params = params;
    let manager = OH_ResourceManager_InitNativeResourceManager(env, a[6]);
    if manager.is_null() { eprintln!("[hal-ohos] no native resource manager: the HAP's raw files cannot be read"); }
    else { super::assets::set_manager(manager); }
    if CALL.load(Ordering::Acquire) == 0 {
        let (mut name, mut function) = (null_mut(), null_mut());
        napi_create_string_utf8(env, c"makepad".as_ptr(), NAPI_AUTO_LENGTH, &mut name);
        if napi_create_threadsafe_function(env, a[7], null_mut(), name, 0, 1, null_mut(), null_mut(), null_mut(), Some(call_js), &mut function) == NAPI_OK {
            CALL.store(function as usize, Ordering::Release);
        }
    }
    if VSYNC.load(Ordering::Acquire) == 0 {
        let vsync = OH_NativeVSync_Create(c"makepad".as_ptr(), 7);
        VSYNC.store(vsync as usize, Ordering::Release);
        note_period();
    }
    Shared::global().display();
    super::standalone::start();
    undefined(env)
}

unsafe extern "C" fn call_js(env: napi_env, function: napi_value, _: *mut c_void, data: *mut c_void) {
    let call = Box::from_raw(data as *mut (CString, CString));
    if env.is_null() || function.is_null() { return; }
    let mut argv = [null_mut(); 2];
    napi_create_string_utf8(env, call.0.as_ptr(), NAPI_AUTO_LENGTH, &mut argv[0]);
    napi_create_string_utf8(env, call.1.as_ptr(), NAPI_AUTO_LENGTH, &mut argv[1]);
    napi_call_function(env, undefined(env), function, 2, argv.as_ptr(), null_mut());
}
/// Asks the ability to do `name` with `value`, on its UI thread, later. False before `start`.
pub(super) fn call(name: &str, value: &str) -> bool {
    let function = CALL.load(Ordering::Acquire);
    let (Ok(name), Ok(value)) = (CString::new(name), CString::new(value)) else { return false };
    if function == 0 { return false; }
    let data = Box::into_raw(Box::new((name, value)));
    unsafe {
        if napi_call_threadsafe_function(function as _, data.cast(), NAPI_TSFN_NONBLOCKING) != NAPI_OK {
            drop(Box::from_raw(data));
            return false;
        }
    }
    true
}

fn emit_lifecycle(kind: u32) { Shared::global().emit(clock_ns(), |q, _| q.lifecycle(kind)); }
unsafe extern "C" fn foreground(env: napi_env, _: napi_callback_info) -> napi_value {
    if crate::backend::log_enabled() { eprintln!("[hal-ohos] foreground"); }
    lock(&Shared::global().state).active = true;
    emit_lifecycle(lifecycle::FOREGROUND);
    emit_lifecycle(lifecycle::RESUME);
    request_frame();
    undefined(env)
}
unsafe extern "C" fn background(env: napi_env, _: napi_callback_info) -> napi_value {
    if crate::backend::log_enabled() { eprintln!("[hal-ohos] background"); }
    lock(&Shared::global().state).active = false;
    emit_lifecycle(lifecycle::PAUSE);
    emit_lifecycle(lifecycle::BACKGROUND);
    crate::background_hook();
    undefined(env)
}
unsafe extern "C" fn destroy(env: napi_env, _: napi_callback_info) -> napi_value {
    lock(&Shared::global().state).active = false;
    emit_lifecycle(lifecycle::DESTROYED);
    undefined(env)
}
unsafe extern "C" fn focus(env: napi_env, info: napi_callback_info) -> napi_value {
    let on = boolean(env, args::<1>(env, info)[0]);
    lock(&Shared::global().state).focused = on;
    Shared::global().emit(clock_ns(), |q, id| { q.focus(id, on); q.lifecycle(if on { lifecycle::FOCUS_GAINED } else { lifecycle::FOCUS_LOST }); });
    undefined(env)
}
/// `avoidArea(top, right, bottom, left)` in vp: what the system bars and the cutout cover.
unsafe extern "C" fn avoid_area(env: napi_env, info: napi_callback_info) -> napi_value {
    let a = args::<4>(env, info);
    lock(&Shared::global().state).insets = [number(env, a[0]) as f32, number(env, a[1]) as f32, number(env, a[2]) as f32, number(env, a[3]) as f32];
    Shared::global().display();
    undefined(env)
}
/// `keyboard(height, open)`: the soft keyboard's height in pixels.
unsafe extern "C" fn keyboard(env: napi_env, info: napi_callback_info) -> napi_value {
    let a = args::<2>(env, info);
    let (height, open) = (number(env, a[0]).max(0.) as i32, boolean(env, a[1]));
    lock(&Shared::global().state).ime = Some((height, open));
    let height = height as f64 / Shared::global().dpi() as f64;
    Shared::global().emit(clock_ns(), |q, id| q.ime_resize(id, height, open));
    undefined(env)
}
unsafe extern "C" fn back_press(env: napi_env, _: napi_callback_info) -> napi_value {
    Shared::global().emit(clock_ns(), |q, id| q.back_pressed(id));
    undefined(env)
}
unsafe extern "C" fn night_mode(env: napi_env, info: napi_callback_info) -> napi_value {
    lock(&Shared::global().state).params.night_mode = boolean(env, args::<1>(env, info)[0]);
    undefined(env)
}

unsafe fn component_size(component: *mut OH_NativeXComponent, window: *mut c_void) -> [u32; 2] {
    let (mut w, mut h) = (0u64, 0u64);
    OH_NativeXComponent_GetXComponentSize(component, window, &mut w, &mut h);
    [w as u32, h as u32]
}
unsafe fn set_surface(component: *mut OH_NativeXComponent, window: *mut c_void, created: bool) {
    if window.is_null() { return surface_destroyed(component, window); }
    let size = component_size(component, window);
    if crate::backend::log_enabled() { eprintln!("[hal-ohos] surface {} {size:?}", if created { "created" } else { "changed" }); }
    let shared = Shared::global();
    let old = {
        let mut r = lock(&shared.registry);
        if created { r.state.created(window as usize, size); crate::backend::shell::window_made(); } else { r.state.changed(window as usize, size); }
        r.window.replace(Window::lease(window))
    };
    drop(old);
    emit_lifecycle(lifecycle::SURFACE_AVAILABLE);
    shared.display();
    request_frame();
}
unsafe extern "C" fn surface_created(component: *mut OH_NativeXComponent, window: *mut c_void) { set_surface(component, window, true); }
unsafe extern "C" fn surface_changed(component: *mut OH_NativeXComponent, window: *mut c_void) { set_surface(component, window, false); }
unsafe extern "C" fn surface_destroyed(_: *mut OH_NativeXComponent, _: *mut c_void) {
    if crate::backend::log_enabled() { eprintln!("[hal-ohos] surface destroyed"); }
    let shared = Shared::global();
    let old = { let mut r = lock(&shared.registry); r.state.destroyed(); r.window.take() };
    if let Some(release) = &*lock(&shared.release_window) { release(); }
    drop(old);
    emit_lifecycle(lifecycle::SURFACE_LOST);
    shared.display();
}

/// The phase of one point of an event of `kind`: the event names the pointer it is about.
fn phase(kind: i32, acting: bool) -> u32 {
    match kind {
        touch_type::MOVE => input::touch::MOVE,
        touch_type::DOWN if acting => input::touch::START,
        touch_type::UP | touch_type::CANCEL if acting => input::touch::STOP,
        touch_type::CANCEL => input::touch::STOP,
        _ => input::touch::STABLE,
    }
}
/// Positions are the XComponent's physical pixels; HAL input uses window points.
unsafe extern "C" fn dispatch_touch(component: *mut OH_NativeXComponent, window: *mut c_void) {
    let mut event = std::mem::MaybeUninit::<OH_NativeXComponent_TouchEvent>::zeroed();
    let got = OH_NativeXComponent_GetTouchEvent(component, window, event.as_mut_ptr());
    if got != 0 {
        if crate::backend::log_enabled() { eprintln!("[hal-ohos] OH_NativeXComponent_GetTouchEvent: {got}"); }
        return;
    }
    let event = event.assume_init();
    if crate::backend::log_enabled() {
        static SEEN: AtomicUsize = AtomicUsize::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 12 {
            eprintln!("[hal-ohos] touch type={} id={} at {},{} points={} stamp={} clock={}", event.type_, event.id, event.x, event.y, event.numPoints, event.timeStamp, clock_ns());
        }
    }
    if !(touch_type::DOWN..=touch_type::CANCEL).contains(&event.type_) { return; }
    let shared = Shared::global();
    let dpi = shared.dpi();
    let listed = &event.touchPoints[..(event.numPoints as usize).min(OH_NATIVE_XCOMPONENT_MAX_TOUCH_POINTS_NUMBER)];
    let mut points: Vec<TouchPoint> = listed.iter().map(|p| TouchPoint { id: p.id as u32, phase: phase(event.type_, p.id == event.id), x: p.x / dpi, y: p.y / dpi, force: p.force }).collect();
    // a release whose pointer the list of pressed points already left out
    if event.type_ != touch_type::MOVE && !listed.iter().any(|p| p.id == event.id) {
        points.push(TouchPoint { id: event.id as u32, phase: phase(event.type_, true), x: event.x / dpi, y: event.y / dpi, force: event.force });
    }
    if points.is_empty() { return; }
    // the event's own time when it is on the HAL's clock (nanoseconds since boot), else now
    let now = clock_ns();
    let stamp = event.timeStamp.max(0) as u64;
    let time = if stamp <= now && now - stamp < 5_000_000_000 { stamp } else { now };
    let mut bytes = Vec::with_capacity(points.len() * TouchPoint::SIZE);
    for p in &points { p.write(&mut bytes); }
    shared.emit(time, |q, id| {
        if event.type_ == touch_type::CANCEL { q.touch_cancel(id, points.len() as u32, &bytes); }
        else { q.touch(id, points.len() as u32, &bytes); }
    });
}

fn can_show() -> bool {
    let shared = Shared::global();
    lock(&shared.registry).window.is_some() && lock(&shared.state).active
}
/// While the ability is in the background or has no surface its frame requests wait here, and
/// `foreground` and the next surface post them, so the app needn't ask again meanwhile.
pub fn frame_requests_held() -> bool { !can_show() }
pub(super) fn request_frame() {
    if !lock(&Shared::global().state).pending.is_empty() { post_frame_callback(); }
}
fn post_frame_callback() {
    let vsync = VSYNC.load(Ordering::Acquire);
    if vsync != 0 && can_show() && !FRAME_POSTED.swap(true, Ordering::AcqRel) {
        if unsafe { OH_NativeVSync_RequestFrame(vsync as _, frame_callback, null_mut()) } != 0 { FRAME_POSTED.store(false, Ordering::Release); }
    }
}
/// The display's period for this app, which changes with its refresh mode.
fn note_period() {
    let vsync = VSYNC.load(Ordering::Acquire);
    let mut period = 0i64;
    if vsync != 0 && unsafe { OH_NativeVSync_GetPeriod(vsync as _, &mut period) } == 0 && period > 0 {
        lock(&Shared::global().state).refresh = 1e9 / period as f32;
    }
}
unsafe extern "C" fn frame_callback(time: i64, _: *mut c_void) {
    FRAME_POSTED.store(false, Ordering::Release);
    note_period();
    if crate::backend::log_enabled() {
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| eprintln!("[hal-ohos] first vsync at {time} (clock {}), {} Hz", clock_ns(), lock(&Shared::global().state).refresh));
    }
    let shared = Shared::global();
    let asked = !lock(&shared.state).pending.is_empty();
    shared.frame(time.max(0) as u64);
    // After a frame we post the next callback right away, so the app's ask for its next frame still
    // gets the next vsync when it comes late in this one. A callback with nothing asked posts no more.
    if asked { post_frame_callback(); }
}

/// A frame was presented. (Android lifts its Activity's cover here; this shell has none.)
pub(crate) fn surface_presented() {}
