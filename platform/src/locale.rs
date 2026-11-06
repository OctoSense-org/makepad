//! What the device's region says about units.

/// Whether the device's region measures distance in metric. `None` where the
/// platform can't say.
pub fn uses_metric_system() -> Option<bool> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        use makepad_apple_sys::*;
        unsafe {
            let locale: ObjcId = msg_send![class!(NSLocale), currentLocale];
            let metric: BOOL = msg_send![locale, usesMetricSystem];
            Some(metric != NO)
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        None
    }
}
