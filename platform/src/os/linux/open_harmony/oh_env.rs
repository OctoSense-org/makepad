//! What the app needs to know about its device and itself, from the NDK
//! instead of ArkTS: device type and OS name (`deviceinfo`) and display
//! density (`display_manager`). The application-level files, cache and temp
//! directories (`ability_runtime`, API 16) are only a fallback: the app's
//! data lives in its module's directories, which only ArkTS reports.

use std::ffi::{c_char, c_int, CStr};

#[link(name = "deviceinfo_ndk.z")]
extern "C" {
    fn OH_GetDeviceType() -> *const c_char;
    fn OH_GetOSFullName() -> *const c_char;
}

#[link(name = "native_display_manager")]
extern "C" {
    fn OH_NativeDisplayManager_GetDefaultDisplayDensityPixels(density: *mut f32) -> c_int;
}

#[link(name = "ability_runtime")]
extern "C" {
    fn OH_AbilityRuntime_ApplicationContextGetFilesDir(buffer: *mut c_char, size: i32, written: *mut i32) -> c_int;
    fn OH_AbilityRuntime_ApplicationContextGetCacheDir(buffer: *mut c_char, size: i32, written: *mut i32) -> c_int;
    fn OH_AbilityRuntime_ApplicationContextGetTempDir(buffer: *mut c_char, size: i32, written: *mut i32) -> c_int;
}

fn c_string(ptr: *const c_char) -> Option<String> {
    (!ptr.is_null()).then(|| unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()).filter(|s| !s.is_empty())
}

/// "phone", "tablet", "2in1", ...
pub fn device_type() -> String {
    c_string(unsafe { OH_GetDeviceType() }).unwrap_or_else(|| "phone".into())
}

pub fn os_full_name() -> String {
    c_string(unsafe { OH_GetOSFullName() }).unwrap_or_else(|| "OpenHarmony".into())
}

/// Physical pixels per vp (densityDPI / 160).
pub fn display_density() -> f64 {
    let mut density = 0.0f32;
    let ok = unsafe { OH_NativeDisplayManager_GetDefaultDisplayDensityPixels(&mut density) } == 0;
    if ok && density > 0.0 { density as f64 } else { 3.25 }
}

fn context_dir(get: unsafe extern "C" fn(*mut c_char, i32, *mut i32) -> c_int) -> String {
    let mut buffer = [0 as c_char; 1024];
    let mut written = 0i32;
    if unsafe { get(buffer.as_mut_ptr(), buffer.len() as i32, &mut written) } != 0 {
        crate::error!("ohos: application context directory unavailable");
        return String::new();
    }
    c_string(buffer.as_ptr()).unwrap_or_default()
}

pub fn files_dir() -> String {
    context_dir(OH_AbilityRuntime_ApplicationContextGetFilesDir)
}

pub fn cache_dir() -> String {
    context_dir(OH_AbilityRuntime_ApplicationContextGetCacheDir)
}

pub fn temp_dir() -> String {
    context_dir(OH_AbilityRuntime_ApplicationContextGetTempDir)
}
