//! Input provenance for host-owned approval controls.
//!
//! Only a native touchscreen path can establish trust (the fork's Android
//! backend does; no HAL backend here does yet). Other platforms, remote input
//! and synthetic dispatch default to false. This is an app-level boundary, not
//! attestation against a compromised OS/root process that can impersonate a
//! physical input device.
use std::cell::Cell;

thread_local! {
    static TRUSTED: Cell<bool> = const { Cell::new(false) };
}

/// True only during a verified native input's synchronous application handler.
/// This is not a persistent approval token; hosts must bind the actual gesture
/// and reviewed operation. Script tasks and subsequent actions are untrusted.
pub fn trusted_user_input() -> bool {
    TRUSTED.with(Cell::get)
}

/// Run synthetic/direct-widget input without inheriting a surrounding native
/// gesture. This API can remove trust but cannot create it.
pub fn with_untrusted_input<R>(f: impl FnOnce() -> R) -> R {
    let _guard = InputTrustGuard::enter(false);
    f()
}

pub(crate) struct InputTrustGuard(bool);
impl InputTrustGuard {
    pub(crate) fn enter(trusted: bool) -> Self {
        Self(TRUSTED.with(|value| value.replace(trusted)))
    }
}
impl Drop for InputTrustGuard {
    fn drop(&mut self) {
        TRUSTED.with(|value| value.set(self.0));
    }
}

/// Android MotionEvent metadata. Missing/virtual devices, non-touch sources,
/// obscured windows and cancellation never establish approval provenance.
#[cfg(any(target_os = "android", test))]
pub(crate) fn android_touch_is_trusted(device_id: i32, physical_device: bool, source: i32, flags: i32, action: i32) -> bool {
    const SOURCE_TOUCHSCREEN: i32 = 0x1002;
    const OBSCURED: i32 = 0x1 | 0x2;
    device_id > 0
        && physical_device
        && source & SOURCE_TOUCHSCREEN == SOURCE_TOUCHSCREEN
        && flags & OBSCURED == 0
        && matches!(action, 0 | 1 | 2 | 5 | 6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_nested_dispatch_cannot_inherit_native_trust() {
        assert!(!trusted_user_input());
        {
            let _native = InputTrustGuard::enter(true);
            assert!(trusted_user_input());
            with_untrusted_input(|| {
                assert!(!trusted_user_input());
                with_untrusted_input(|| assert!(!trusted_user_input()));
            });
            assert!(trusted_user_input());
        }
        assert!(!trusted_user_input());
    }

    #[test]
    fn unwind_restores_previous_scope_and_threads_do_not_share_trust() {
        let _native = InputTrustGuard::enter(true);
        assert!(!std::thread::spawn(trusted_user_input).join().unwrap());
        let result = std::panic::catch_unwind(|| with_untrusted_input(|| panic!("test")));
        assert!(result.is_err());
        assert!(trusted_user_input());
    }

    #[test]
    fn android_requires_physical_unobscured_touch_and_never_cancel() {
        for action in [0, 1, 2, 5, 6] {
            assert!(android_touch_is_trusted(4, true, 0x1002, 0, action));
        }
        for (id, physical, source, flags, action) in [
            (0, true, 0x1002, 0, 1),
            (-1, true, 0x1002, 0, 1),
            (4, false, 0x1002, 0, 1),
            (4, true, 0x2002, 0, 1),
            (4, true, 0x1002, 1, 1),
            (4, true, 0x1002, 2, 1),
            (4, true, 0x1002, 0, 3),
        ] {
            assert!(!android_touch_is_trusted(id, physical, source, flags, action));
        }
    }
}
