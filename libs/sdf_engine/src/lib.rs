//! The realtime SDF engine (shared by Fab and Sandbox).
//!
//! - [`program`]: a `csg_sdf` shape lowered to a flat f32 program;
//! - [`twin`]: the CPU interpreter of that program, bit-exact with the GPU;
//! - [`volume`]: an object's sparse brick volume (lattice, allocation,
//!   per-brick culled programs, dirty re-builds);
//! - [`kernel`]: Splash SDF kernels compiled into the bake shader;
//! - [`audit`]: a frame's baked bricks read back and compared with the exact fields;
//! - [`budget`]: GPU safety (measured frame costs, the bake's steering, caps);
//! - [`bake`]: the GPU bake into persistent R16F/R8 storage atlases by compute;
//! - [`march`]: the coarse pass, the march (depth + G-buffer) and the shade;
//! - [`pick`]: one ray against the exact field, naming the feature hit;
//! - [`splat`]: the splat look (surface splats from the bricks).
//!
//! Buffer-backed compute shaders are available in [`compute`]; frame dispatch
//! remains with the host integration.

pub use makepad_csg_sdf;
pub use makepad_draw;

pub mod audit;
pub mod bake;
pub mod budget;
pub mod compute;
pub mod kernel;
pub mod march;
pub mod mesh;
pub mod pick;
pub mod program;
pub mod splat;
pub mod twin;
pub mod trace;
pub mod volume;

use makepad_draw::*;

pub fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
    crate::compute::script_mod(vm);
    crate::trace::script_mod(vm);
    // the environment lookups the shade lights with, as the model lanes do
    makepad_render_material::builtin::register(vm);
    crate::march::script_mod(vm);
    crate::splat::script_mod(vm)
}
