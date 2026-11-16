//! OpenHarmony's ability and its XComponent are ArkTS's; what they report arrives through
//! Node-API and the XComponent's native callbacks, which never enter the HAL's world lock.
//! The system refuses memory an app wrote and then makes executable, so an app here is always
//! compiled ahead (`standalone`): there is no makepad run host on this system.
pub mod sys;
mod shell;
mod surface;
pub mod napi;
pub mod assets;
pub mod media;
pub mod standalone;

pub use shell::{open, OhosBackend, OhosParams};
pub use surface::OhosEglSurfaces;
pub use napi::frame_requests_held;
pub(crate) use napi::surface_presented;
