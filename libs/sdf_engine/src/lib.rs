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
//! - [`splat`]: the splat look (surface splats from the bricks);
//! - [`plane_abi`]: the GPU records handed between the slicer's stages;
//! - [`plane`]: plane tracing (coarse grid, nodes, crossings, spans, the outline FeatureSet);
//! - [`list_rank`]: roots, ranks and ids of successor lists (cycles and open chains);
//! - [`chain`]: pieces to loops and open chains, their metrics and containment;
//! - [`rings`]: chained wall pieces to the layer plan's rings, the thick test and the readback;
//! - [`feature_set`]: the FeatureSet for contours traced on the cores;
//! - [`feature_field`]: a FeatureSet's band, nearest features and candidate lists;
//! - [`wall_pieces`]: the level curves of a feature field as keyed pieces per fine cell;
//! - [`thin`]: thin walls, the skeleton of a FeatureSet's sites and its bead pieces;
//! - [`regions_compose`]: a round of the class program, compositions of FeatureSets into keyed pieces;
//! - [`regions_label`]: packed class labels at fine nodes, with overlap counts;
//! - [`regions_program`]: the class program as rounds over a batch of planes;
//! - [`regions_classes`]: the class rounds of a batch, from the outline's and the fill's fields;
//! - [`regions_concentric`]: S1b, the concentric solid fill's rounds over a region's own field levels;
//! - [`regions_overhang`]: S3, the overhang labels of a batch's wall loops against the layer below;
//! - [`regions_set`]: a region's FeatureSet from a compose round's output and its chain;
//! - [`regions_keep`]: a Keep round's per-component decision over a chain.
//!
//! Buffer-backed compute shaders are available in [`compute`]; frame dispatch
//! remains with the host integration.

pub use makepad_csg_sdf;
pub use makepad_draw;

pub mod audit;
pub mod bake;
pub mod budget;
pub mod chain;
pub mod compute;
pub mod feature_field;
pub mod feature_set;
pub mod gpu_jobs;
pub mod kernel;
pub mod list_rank;
pub mod march;
pub mod mesh;
pub mod pick;
pub mod plane;
pub mod plane_abi;
pub mod program;
pub mod regions_classes;
pub mod regions_compose;
pub mod rings;
pub mod regions_label;
pub mod regions_concentric;
pub mod regions_overhang;
pub mod regions_keep;
pub mod regions_program;
pub mod regions_set;
pub mod splat;
pub mod thin;
pub mod twin;
pub mod trace;
pub mod volume;
pub mod wall_pieces;

use makepad_draw::*;

pub fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
    crate::compute::script_mod(vm);
    crate::gpu_jobs::script_mod(vm);
    crate::trace::script_mod(vm);
    crate::plane::script_mod(vm);
    crate::list_rank::script_mod(vm);
    crate::chain::script_mod(vm);
    crate::rings::script_mod(vm);
    crate::feature_field::script_mod(vm);
    crate::regions_compose::script_mod(vm);
    crate::regions_concentric::script_mod(vm);
    crate::regions_overhang::script_mod(vm);
    crate::regions_set::script_mod(vm);
    crate::regions_keep::script_mod(vm);
    crate::wall_pieces::script_mod(vm);
    crate::regions_label::script_mod(vm);
    crate::thin::script_mod(vm);
    // the environment lookups the shade lights with, as the model lanes do
    makepad_render_material::builtin::register(vm);
    crate::march::script_mod(vm);
    crate::splat::script_mod(vm)
}
