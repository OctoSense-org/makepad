use super::shell::{lock, Shared};
use super::sys;
use crate::backend::android::state;
use crate::backend::gl::surface as egl;
use crate::Id;
use std::sync::Arc;

/// Each lease owns a reference to the XComponent's `OHNativeWindow`, the registry's current
/// window too. Replacing the registry's never drops the one an EGL surface still presents into.
pub(super) struct Window(pub usize);
impl Window {
    /// # Safety
    /// `window` is the live `OHNativeWindow` an XComponent callback handed over.
    pub unsafe fn lease(window: *mut std::ffi::c_void) -> Self {
        sys::OH_NativeWindow_NativeObjectReference(window);
        Self(window as usize)
    }
}
impl Clone for Window {
    fn clone(&self) -> Self { unsafe { sys::OH_NativeWindow_NativeObjectReference(self.0 as _); } Self(self.0) }
}
impl Drop for Window {
    fn drop(&mut self) { unsafe { sys::OH_NativeWindow_NativeObjectUnreference(self.0 as _); } }
}

#[derive(Default)]
pub(super) struct Registry {
    pub host: Option<Id>,
    pub state: state::SurfaceState,
    pub window: Option<Window>,
}

/// The GLES core's view of the XComponent's surface: a lease on its `OHNativeWindow`, and only
/// render targets when the windows are offscreen.
pub struct OhosEglSurfaces { pub offscreen: bool }
impl egl::EglSurfaceSource for OhosEglSurfaces {
    fn display(&self) -> Result<egl::Display, String> {
        // the default display, as on Android
        Ok(egl::Display { platform: egl::PLATFORM_ANDROID, native: 0, visual: 0 })
    }
    fn state(&mut self, sid: u32, id: Id) -> egl::SurfaceState {
        if self.offscreen { return egl::SurfaceState::Offscreen; }
        let r = lock(&Shared::global().registry);
        if sid != crate::session::HOST || r.host != Some(id) { return egl::SurfaceState::Lost; }
        match (&r.window, r.state.status()) {
            (Some(window), state::SurfaceStatus::Native { generation, size }) =>
                egl::SurfaceState::Native(egl::NativeSurface { generation, window: window.0, size, keep: Arc::new(window.clone()) }),
            _ => egl::SurfaceState::Lost,
        }
    }
}
