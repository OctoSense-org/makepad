//! Splash SDF kernels on the GPU.
//!
//! A part's custom distance function (`m.sdf({field: fn(p) {..}})`) is a
//! kernel compiled by the Splash kernel compiler (`makepad_script_compute`)
//! into AIR, the structured IR whose ops are exactly specified. Here that
//! AIR becomes a function of the bake shader, built as shader IR by the
//! shader compiler's host hook (`makepad_script_compute::shader_route`, inside
//! the compiler: the AIR and the shader IR are its values): the program
//! moves as data, never as source text, and every AIR op maps to the shader
//! op with the same IEEE meaning (min/max as selects, integer ops wrapping,
//! saturating conversions written out). The bake shader is `precise_math`
//! (no contraction, correctly rounded division and square root), so a baked
//! brick equals the CPU kernel's distances bit for bit.
//!
//! - The document's numbers the function reads (`vase_height`, ...) are
//!   kernel parameters ([`makepad_csg_sdf::SdfSplashExpr`] lifts them), so a
//!   parameter drag is the same compiled kernel: its values travel in the
//!   program texels (op `KERNEL`) and only the touched bricks bake again.
//! - A volume's kernels have slots ([`KernelSlot`], by program identity); a
//!   bake variant ([`KernelBake`]) holds the dispatch over them. A new
//!   program (a shape-structure change) is a new variant, compiled while the
//!   old bricks keep drawing.
//! - Limits: a kernel's AIR is bounded by construction (every loop has a
//!   cap); one whose worst case exceeds [`MAX_KERNEL_COST`] ops per sample,
//!   reads more than [`MAX_KERNEL_PARAMS`] parameters, uses f64, host calls,
//!   tables or more than [`MAX_FRAME_WORDS`] words of local arrays has no GPU
//!   route and is rejected before baking.


use crate::march::DrawSdfMarch;
use makepad_csg_sdf::{SdfKernel, SdfNode, SdfSplashExpr};
use makepad_draw::*;
use makepad_script_compute::shader_route::{self, ShaderRoute};
use std::sync::Arc;

/// Worst-case AIR ops of one sample accepted by the route (a
/// bake quad samples 512 points, a march step one; GPU watchdogs kill
/// draws of seconds).
pub const MAX_KERNEL_COST: u64 = 20_000;
/// Parameters a kernel call passes (four texels).
pub const MAX_KERNEL_PARAMS: usize = 16;
/// Words of local arrays (the AIR frame) a GPU kernel may use.
pub const MAX_FRAME_WORDS: u32 = 64;

/// A kernel's GPU route ([`ShaderRoute`]: its AIR checked and translated
/// into shader IR inside the compiler), with what the volume reads of it.
pub struct KernelAir {
    /// The program's identity ([`SdfSplashExpr::structure_key`]).
    pub key: u64,
    pub params: usize,
    /// Worst-case AIR ops per sample.
    pub cost: u64,
    route: ShaderRoute,
}

impl KernelAir {
    /// The GPU route of a Splash field, or why it has none.
    pub fn new(splash: &SdfSplashExpr) -> Result<KernelAir, String> {
        let key = splash.structure_key();
        let route = ShaderRoute::new(splash.kernel(), key, MAX_KERNEL_COST, MAX_KERNEL_PARAMS as u32, MAX_FRAME_WORDS);
        let info = route.info();
        if !info.error.is_empty() {
            return Err(info.error.to_string());
        }
        let (params, cost) = (info.params as usize, info.cost);
        Ok(KernelAir { key, params, cost, route })
    }
}

/// One kernel of a volume (by program identity).
#[derive(Clone)]
pub struct KernelSlot {
    /// The program's identity (or the field's key when it is no Splash program).
    pub key: u64,
    route: FieldRoute,
}

#[derive(Clone)]
enum FieldRoute {
    Splash { field: Arc<SdfSplashExpr>, air: Arc<KernelAir> },
    Mesh(Arc<crate::mesh::MeshAsset>),
}

fn structure_key(k: &SdfKernel) -> u64 {
    k.splash.as_ref().map_or(k.key, |s| s.structure_key())
}

impl KernelSlot {
    pub fn new(k: &SdfKernel) -> Result<KernelSlot, crate::program::LowerError> {
        let field = k.splash.clone().ok_or(crate::program::LowerError::Unsupported("SDF kernels require a Splash field"))?;
        let air = Arc::new(KernelAir::new(&field).map_err(crate::program::LowerError::Kernel)?);
        Ok(KernelSlot { key: structure_key(k), route: FieldRoute::Splash { field, air } })
    }

    pub fn mesh(m: &Arc<makepad_csg_sdf::mesh_field::MeshField>) -> KernelSlot {
        KernelSlot { key: m.key(), route: FieldRoute::Mesh(Arc::new(crate::mesh::MeshAsset::new(m))) }
    }

    pub fn mesh_asset(&self) -> Option<&Arc<crate::mesh::MeshAsset>> {
        match &self.route { FieldRoute::Mesh(m) => Some(m), _ => None }
    }

    pub fn air(&self) -> Option<&Arc<KernelAir>> {
        match &self.route { FieldRoute::Splash { air, .. } => Some(air), _ => None }
    }

    pub fn is_mesh(&self) -> bool { self.mesh_asset().is_some() }

    /// Whether `k` runs this slot's program.
    pub fn matches(&self, k: &SdfKernel) -> bool {
        !self.is_mesh() && self.key == structure_key(k)
    }

    /// [`KernelSlot::distance`] at many points (one kernel call).
    pub fn distance_many(&self, params: &[f32], ps: &[[f32; 3]]) -> Vec<f32> {
        match &self.route {
            FieldRoute::Splash { field, .. } => {
                let xyz: Vec<f32> = ps.iter().flatten().copied().collect();
                let mut out = vec![0.0f32; ps.len()];
                field.distance_batch_with(params, &xyz, &mut out);
                out
            }
            FieldRoute::Mesh(m) => ps.iter().map(|p| m.signed(*p)).collect(),
        }
    }

    /// The distance at `p` with parameters `params` (by parameter index).
    pub fn distance(&self, params: &[f32], p: [f32; 3]) -> f32 {
        match &self.route {
            FieldRoute::Splash { field, .. } => {
                let mut out = [0.0f32];
                field.distance_batch_with(params, &p, &mut out);
                out[0]
            }
            FieldRoute::Mesh(m) => m.signed(p),
        }
    }

}

/// Whether `n` calls any kernel.
pub fn has_kernel(n: &SdfNode) -> bool {
    match n {
        SdfNode::Kernel(_) | SdfNode::Mesh(_) => true,
        SdfNode::Combine { of, .. } => of.iter().any(has_kernel),
        SdfNode::Shell { of, .. } | SdfNode::Round { of, .. } | SdfNode::Transform { of, .. } | SdfNode::Instance { of, .. } | SdfNode::Bone { of, .. } | SdfNode::Deform { of, .. } | SdfNode::Id { of, .. } | SdfNode::Sculpt { of, .. } => has_kernel(of),
        SdfNode::Material { within, of, .. } => within.as_deref().is_some_and(has_kernel) || has_kernel(of),
        _ => false,
    }
}

/// Whether `n` holds a primitive whose field is no true distance (the ellipsoid's estimate,
/// the gyroid's): steeper than 1 in places, so a volume measures it ([`crate::volume`]).
pub fn has_estimate(n: &SdfNode) -> bool {
    match n {
        SdfNode::Ellipsoid { .. } | SdfNode::Gyroid { .. } => true,
        SdfNode::Combine { of, .. } => of.iter().any(has_estimate),
        SdfNode::Shell { of, .. } | SdfNode::Round { of, .. } | SdfNode::Transform { of, .. } | SdfNode::Instance { of, .. } | SdfNode::Bone { of, .. } | SdfNode::Deform { of, .. } | SdfNode::Id { of, .. } | SdfNode::Sculpt { of, .. } => has_estimate(of),
        SdfNode::Material { within, of, .. } => within.as_deref().is_some_and(has_estimate) || has_estimate(of),
        _ => false,
    }
}

/// Adds the kernels of `n` missing from `slots` (their order is stable:
/// a slot keeps its index while its volume lives).
pub fn collect(n: &SdfNode, slots: &mut Vec<KernelSlot>) -> Result<(), crate::program::LowerError> {
    match n {
        SdfNode::Kernel(k) => {
            if !slots.iter().any(|s| s.matches(k)) {
                slots.push(KernelSlot::new(k)?);
            }
        }
        SdfNode::Mesh(m) => {
            if !slots.iter().any(|s| s.is_mesh() && s.key == m.key()) {
                slots.push(KernelSlot::mesh(m));
            }
        }
        SdfNode::Combine { of, .. } => for c in of { collect(c, slots)?; },
        SdfNode::Shell { of, .. } | SdfNode::Round { of, .. } | SdfNode::Transform { of, .. } | SdfNode::Instance { of, .. } | SdfNode::Bone { of, .. } | SdfNode::Deform { of, .. } | SdfNode::Id { of, .. } | SdfNode::Sculpt { of, .. } => collect(of, slots)?,
        SdfNode::Material { within, of, .. } => {
            if let Some(w) = within {
                collect(w, slots)?;
            }
            collect(of, slots)?;
        }
        _ => {}
    }
    Ok(())
}

/// A volume's shaders with its kernels compiled in: the bake's, and the
/// march's (made from the march its host draws with, once asked for).
pub struct KernelBake {
    /// The kernels' program identities (slot order) it was made for.
    pub keys: Vec<u64>,
    pub compute: ComputeKernel,
    source: ComputeBake,
    /// The march with the kernels (exact evaluation near the surface), and
    /// the shader it was made from.
    march: Option<(DrawShaderId, DrawSdfMarch)>,
    /// Every GPU kernel is cheap enough to evaluate in the march.
    pub march_ok: bool,
}

/// Worst-case ops per sample of a kernel the march evaluates exactly (a ray
/// may evaluate it at every step in the band and four times for its
/// normal); costlier kernels march their baked copy.
pub const MAX_MARCH_KERNEL_COST: u64 = 2_000;

/// `base` with the kernels' dispatch `own` and the fetching kernel_at.
fn variant(vm: &mut ScriptVm, base: ScriptObject, own: ScriptObject) -> Option<ScriptObject> {
    let fetching = vm.value(base, id!(kernel_at_fetch)).as_object()?;
    let obj = vm.new_with_proto(base);
    vm.set_value(obj, id!(kernel_call), own);
    vm.set_value(obj, id!(kernel_at), fetching);
    Some(obj)
}

/// A buffer-backed bake object with one dispatch splice per kernel set.
/// Compile once for the selected device; this object never dispatches work.
/// Keep it alive while compiling derived draw variants, then call release.
pub struct ComputeBake {
    pub object: ScriptObject,
    pub keys: Vec<u64>,
    dispatch: Option<ScriptObject>,
    mesh: Option<ScriptObject>,
    mesh_slots: Vec<usize>,
    refs: Vec<ScriptObjectRef>,
}

impl ComputeBake {
    pub fn new(vm: &mut ScriptVm, slots: &[KernelSlot]) -> Option<Self> {
        let module = vm.heap().module(id!(sdf));
        let base = vm.value(module, id!(Bake)).as_object()?;
        let keys: Vec<_> = slots.iter().map(|s| s.key).collect();
        let mut refs = Vec::new();
        let dispatch = if slots.iter().any(|s| s.air().is_some()) {
            let key = slots.iter().fold(0xcbf2_9ce4_8422_2325u64, |key, s| {
                (key ^ s.key ^ s.air().is_some() as u64).wrapping_mul(0x100_0000_01b3)
            });
            let stub = vm.value(base, id!(kernel_call)).as_object()?;
            let routes: Vec<_> = slots.iter().map(|s| s.air().map(|g| &g.route)).collect();
            let own = shader_route::dispatch_fn(vm, stub, key, &routes)?;
            refs.push(vm.new_object_ref(own));
            Some(own)
        } else { None };
        let object = match dispatch {
            Some(own) => variant(vm, base, own)?,
            None => vm.new_with_proto(base),
        };
        let mesh_slots: Vec<_> = slots.iter().enumerate().filter_map(|(i, s)| s.is_mesh().then_some(i)).collect();
        let mesh = if mesh_slots.is_empty() { None } else {
            // One binding per asset; choose the binding before each buffer read.
            // Object instances share the uploaded buffers, including across bakers.
            let mut code = String::from("use mod.pod.*\nuse mod.shader\n{\n");
            for i in &mesh_slots { code += &format!("mesh_{i}: shader.buffer(vec4u)\n"); }
            code += "mesh_read: fn(slot: u32, at: u32) -> vec4u {\n";
            for i in &mesh_slots { code += &format!("if slot == {i}u {{ return self.mesh_{i}[at] }}\n"); }
            code += "return vec4u(0u)\n}\n}";
            let mesh = vm.eval(ScriptMod::new("", "", "sdf_mesh_bindings", 0, 0, &code, &[])).as_object()?;
            refs.push(vm.new_object_ref(mesh));
            Self::attach_mesh(vm, object, mesh, &mesh_slots);
            Some(mesh)
        };
        refs.push(vm.new_object_ref(object));
        Some(Self { object, keys, dispatch, mesh, mesh_slots, refs })
    }

    fn attach_mesh(vm: &mut ScriptVm, object: ScriptObject, mesh: ScriptObject, slots: &[usize]) {
        for name in slots.iter().map(|i| crate::mesh::binding(*i)).chain([id!(mesh_read)]) {
            let value = vm.value(mesh, name);
            vm.set_value(object, name, value);
        }
    }

    /// Compute uses IrFnCompiler::compile_shader_def, the same lowering as
    /// draw. shader_lower_calls consults shader_ir_fns before compiling a
    /// script body, so dispatch_fn's AIR splice also runs for compute.
    pub fn compile(&self, vm: &mut ScriptVm, target: makepad_draw::makepad_script::shader_ir_compute::ComputeTarget)
        -> Result<makepad_draw::makepad_script::shader_ir_compute::CompiledKernel, String>
    {
        makepad_draw::makepad_script::shader_ir_compute::compile_compute_kernel(vm, self.object, target)
    }

    /// Apply the same dispatch to mod.draw.SdfMarch (or a host's derived shader).
    pub fn draw_variant(&mut self, vm: &mut ScriptVm, base: ScriptObject) -> Option<ScriptObject> {
        let object = match self.dispatch {
            Some(own) => variant(vm, base, own)?,
            None => vm.new_with_proto(base),
        };
        if let Some(mesh) = self.mesh { Self::attach_mesh(vm, object, mesh, &self.mesh_slots); }
        self.refs.push(vm.new_object_ref(object));
        Some(object)
    }

    pub fn release(self, vm: &mut ScriptVm) {
        if let Some(own) = self.dispatch { shader_route::release_dispatch(vm, own); }
    }
}

impl KernelBake {
    /// The bake variant for this kernel set, including the empty primitive set.
    pub fn new(cx: &mut Cx, slots: &[KernelSlot]) -> Option<KernelBake> {
        let march_ok = slots.iter().all(|s| s.is_mesh() || s.air().is_some_and(|g| g.cost <= MAX_MARCH_KERNEL_COST));
        cx.with_vm(|vm| {
            let mut source = ComputeBake::new(vm, slots)?;
            if let Err(error) = crate::trace::lower_spliced(vm, &mut source) {
                log!("sdf trace lowering: {error}");
                source.release(vm);
                return None;
            }
            let compute = match ComputeKernel::new(vm, source.object, "sdf bake") {
                Ok(kernel) => kernel,
                Err(error) => {
                    log!("sdf compute: {error}");
                    source.release(vm);
                    return None;
                }
            };
            Some(KernelBake { keys: source.keys.clone(), compute, source, march: None, march_ok })
        })
    }

    /// The march `base` with the kernels compiled in (made once per base
    /// shader); `None` when a kernel is too costly to evaluate per step.
    pub fn march(&mut self, cx: &mut Cx, base: &DrawSdfMarch) -> Option<&mut DrawSdfMarch> {
        if !self.march_ok {
            return None;
        }
        let id = base.draw_super.draw_vars.draw_shader_id?;
        if self.march.as_ref().is_none_or(|(b, _)| *b != id) {
            let source = cx.draw_shaders.shaders.get(id.index)?.mapping.source.as_object();
            let made = cx.with_vm(|vm| {
                let obj = self.source.draw_variant(vm, source)?;
                let mut draw = DrawSdfMarch::script_from_value(vm, obj.into());
                draw.kernels = true;
                Some(draw)
            });
            let draw = made?;
            self.march = Some((id, draw));
        }
        self.march.as_mut().map(|(_, d)| d)
    }

    /// Whether this variant runs `slots`' kernels.
    pub fn serves(&self, slots: &[KernelSlot]) -> bool {
        self.keys.len() == slots.len() && self.keys.iter().zip(slots).all(|(k, s)| *k == s.key)
    }

    pub fn release(self, cx: &mut Cx) {
        cx.with_vm(|vm| self.source.release(vm));
    }
}

