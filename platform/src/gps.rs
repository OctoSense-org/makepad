//! Last-known device GPS fix.
//!
//! Written by the Android `LocationListener` through the JNI `onLocation`
//! receiver (`os/linux/android/android_jni.rs`) and read SYNCHRONOUSLY by the
//! Splash `sys.gps(...)` helper (in the widgets crate). It lives here in
//! `makepad-platform` because that is the one crate both the JNI writer and the
//! `sys.*` reader (widgets -> draw -> platform) can reach. On non-Android
//! targets nothing writes it, so `last_gps_fix()` simply stays `None`.
use std::sync::Mutex;

#[derive(Clone, Copy, Debug)]
pub struct GpsFix {
    pub lat: f64,
    pub lon: f64,
    /// horizontal accuracy in metres
    pub acc: f32,
}

static LAST_GPS_FIX: Mutex<Option<GpsFix>> = Mutex::new(None);

/// Store a fresh fix (called from the Android JNI `onLocation` receiver).
pub fn set_gps_fix(lat: f64, lon: f64, acc: f32) {
    if let Ok(mut g) = LAST_GPS_FIX.lock() {
        *g = Some(GpsFix { lat, lon, acc });
    }
}

/// The most recent fix, or `None` if the device has not produced one yet.
pub fn last_gps_fix() -> Option<GpsFix> {
    LAST_GPS_FIX.lock().ok().and_then(|g| *g)
}
