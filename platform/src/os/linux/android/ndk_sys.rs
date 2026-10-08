#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use makepad_jni_sys as jni_sys;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ANativeWindow {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct AHardwareBuffer {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct AAssetManager {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct AAsset {
    _unused: [u8; 0],
}

pub const AASSET_MODE_BUFFER: ::std::os::raw::c_uint = 3;

#[link(name = "android")]
extern "C" {
    pub fn AAssetManager_open(
        mgr: *mut AAssetManager,
        filename: *const ::std::os::raw::c_char,
        mode: ::std::os::raw::c_int,
    ) -> *mut AAsset;
    pub fn AAsset_getLength64(asset: *mut AAsset) -> i64;
    pub fn ANativeWindow_release(window: *mut ANativeWindow);
    pub fn ANativeWindow_fromSurface(
        env: *mut jni_sys::JNIEnv,
        surface: jni_sys::jobject,
    ) -> *mut ANativeWindow;
    pub fn AAsset_read(
        asset: *mut AAsset,
        buf: *mut ::std::os::raw::c_void,
        count: usize,
    ) -> ::std::os::raw::c_int;
    pub fn AAsset_close(asset: *mut AAsset);
    pub fn AAssetManager_fromJava(
        env: *mut jni_sys::JNIEnv,
        assetManager: jni_sys::jobject,
    ) -> *mut AAssetManager;

    // NOTE: `ANativeWindow_setFrameRate` is deliberately NOT declared here.
    // It is API 30+; a plain `extern "C"` strong reference to it would break
    // `dlopen` of libmakepad.so on API 26-29 devices. If surface frame-rate
    // control is wanted, resolve it via `dlsym(libandroid.so, ...)` gated on
    // `sdk_version >= 30`, rather than linking newer symbols at the minSdk floor.

    // AHardwareBuffer_acquire / _release are API 26 — safe at the minSdk floor.
    pub fn AHardwareBuffer_acquire(buffer: *mut AHardwareBuffer);
    pub fn AHardwareBuffer_release(buffer: *mut AHardwareBuffer);
}

pub const AHARDWAREBUFFER_USAGE_CPU_READ_RARELY: u64 = 2;
pub const AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN: u64 = 3;
pub const AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE: u64 = 1 << 8;

#[repr(C)]
pub struct ANativeActivity {
    pub callbacks: *mut ANativeActivityCallbacks,
    pub vm: *mut jni_sys::JavaVM,
    pub env: *mut jni_sys::JNIEnv,
    pub clazz: jni_sys::jobject,
    pub internalDataPath: *const ::std::os::raw::c_char,
    pub externalDataPath: *const ::std::os::raw::c_char,
    pub sdkVersion: i32,
    pub instance: *mut ::std::os::raw::c_void,
    pub assetManager: *mut AAssetManager,
    pub obbPath: *const ::std::os::raw::c_char,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ANativeActivityCallbacks {
    pub onStart: ::std::option::Option<unsafe extern "C" fn(activity: *mut ANativeActivity)>,
    pub onResume: ::std::option::Option<unsafe extern "C" fn(activity: *mut ANativeActivity)>,
    pub onSaveInstanceState: ::std::option::Option<
        unsafe extern "C" fn(
            activity: *mut ANativeActivity,
            outSize: *mut usize,
        ) -> *mut ::std::os::raw::c_void,
    >,
    pub onPause: ::std::option::Option<unsafe extern "C" fn(activity: *mut ANativeActivity)>,
    pub onStop: ::std::option::Option<unsafe extern "C" fn(activity: *mut ANativeActivity)>,
    pub onDestroy: ::std::option::Option<unsafe extern "C" fn(activity: *mut ANativeActivity)>,
    pub onWindowFocusChanged: ::std::option::Option<
        unsafe extern "C" fn(activity: *mut ANativeActivity, hasFocus: ::std::os::raw::c_int),
    >,
    pub onNativeWindowCreated: ::std::option::Option<
        unsafe extern "C" fn(activity: *mut ANativeActivity, window: *mut ANativeWindow),
    >,
    pub onNativeWindowResized: ::std::option::Option<
        unsafe extern "C" fn(activity: *mut ANativeActivity, window: *mut ANativeWindow),
    >,
    pub onNativeWindowRedrawNeeded: ::std::option::Option<
        unsafe extern "C" fn(activity: *mut ANativeActivity, window: *mut ANativeWindow),
    >,
    pub onNativeWindowDestroyed: ::std::option::Option<
        unsafe extern "C" fn(activity: *mut ANativeActivity, window: *mut ANativeWindow),
    >,
    pub onInputQueueCreated: ::std::option::Option<
        unsafe extern "C" fn(activity: *mut ANativeActivity, queue: *mut AInputQueue),
    >,
    pub onInputQueueDestroyed: ::std::option::Option<
        unsafe extern "C" fn(activity: *mut ANativeActivity, queue: *mut AInputQueue),
    >,
    pub onContentRectChanged: ::std::option::Option<
        unsafe extern "C" fn(activity: *mut ANativeActivity, rect: *const ARect),
    >,
    pub onConfigurationChanged:
        ::std::option::Option<unsafe extern "C" fn(activity: *mut ANativeActivity)>,
    pub onLowMemory: ::std::option::Option<unsafe extern "C" fn(activity: *mut ANativeActivity)>,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct AInputQueue {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ARect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}
