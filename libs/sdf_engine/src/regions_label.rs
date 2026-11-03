//! Class labels at every fine node of a plane batch, in `csg_sdf::regions::CLASSES` order.
//! Each present term contributes one bit per byte to a packed mask, using exactly
//! ComposeTermSide's low-node side rule. A final pass picks the lowest class and counts nodes
//! inside multiple classes. One thread owns four nodes, so neither pass races on packed bytes.
use crate::gpu_jobs::{label, BufRef, CountWord, Elements, JobError, Pass, Pool, Step};
use crate::regions_compose::{plane_offset, Term, PLANE_BIAS};
use makepad_draw::*;

/// A node inside no class.
pub const NO_CLASS: u8 = 255;
/// u32 words of the label buffer: four u8 labels per word, planes × (10 nx) × (10 ny) nodes.
pub fn label_words(nx: u32, ny: u32, planes: u32) -> u32 {
    25 * nx * ny * planes
}

/// The class label kernels, made once.
#[derive(Clone)]
pub struct LabelKernels {
    class: ComputeKernel,
    finish: ComputeKernel,
    empty: ComputeKernel,
}
impl LabelKernels {
    pub fn new(cx: &mut Cx) -> Result<Self, String> {
        cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            let mut kernel = |name: LiveId, label: &str| -> Result<ComputeKernel, String> {
                let object = vm.value(module, name).as_object().ok_or_else(|| format!("{label} kernel missing"))?;
                ComputeKernel::new(vm, object, label)
            };
            Ok(Self {
                class: kernel(id!(LabelClass), "regions label class")?,
                finish: kernel(id!(LabelFinish), "regions label finish")?,
                empty: kernel(id!(LabelEmpty), "regions label empty")?,
            })
        })
    }
}

/// The class label of every fine node (gx, gy), 0 <= gx < 10 nx, 0 <= gy < 10 ny, of every plane:
/// the index c of the class term whose side is inside there (classes[c] = None: that class is empty
/// on every plane), or NO_CLASS. A node inside two or more classes takes the lowest index and adds 1
/// to `conflicts`. Node (gx, gy) of plane p is byte (gx % 4) (little-endian) of word
/// (p * 100 nx ny + gy * 10 nx + gx) / 4 of the returned buffer.
pub fn steps(k: &LabelKernels, pool: &mut Pool, nx: u32, ny: u32, planes: u32, classes: &[Option<Term>], conflicts: CountWord) -> Result<(BufRef, Vec<Step>), JobError> {
    if classes.len() > crate::makepad_csg_sdf::regions::CLASSES.len() {
        return Err(JobError::Failed("at most seven region classes".into()));
    }
    nx.checked_mul(ny).and_then(|n| n.checked_mul(planes)).and_then(|n| n.checked_mul(100))
        .ok_or_else(|| JobError::Failed("label node address overflow".into()))?;
    for t in classes.iter().flatten() {
        if t.plane_offset.unsigned_abs() >= PLANE_BIAS {
            return Err(JobError::Failed("label term plane offset outside its bias".into()));
        }
        if t.levels.as_ref().is_some_and(|levels| levels.len() != planes as usize) {
            return Err(JobError::Failed("a label term level per plane".into()));
        }
    }
    let words = label_words(nx, ny, planes);
    let output = pool.buffer("regions label", words as u64 * 4);
    let mask = pool.buffer("regions label mask", words as u64 * 4);
    let pass = |kernel: &ComputeKernel, name: &str, t: Option<&Term>, params: BufRef, z: u32, w: u32| {
        let mut missing = None;
        let mut p = Pass::new(kernel, label(name), Elements::Fixed(words), 4, 1 << 20, |b| {
            match (b, t) {
                (id!(counts), _) => BufRef::COUNTS,
                (id!(label_mask), _) => mask,
                (id!(labels), _) => output,
                (id!(params), _) => params,
                (id!(band_slot), Some(t)) => t.band_slot,
                (id!(cover), Some(t)) => t.cover,
                (id!(node_side), Some(t)) => t.node_side,
                _ => { missing = Some(b); BufRef::RUNTIME }
            }
        }).zw(z, w);
        if let Some(b) = missing { return Err(JobError::Failed(format!("{name}: no buffer for binding {b}"))); }
        if kernel.layout().slots.iter().any(|s| s.name == id!(grid)) {
            p = p.param_u32x4(id!(grid), [nx, ny, planes, 0]).map_err(JobError::Failed)?;
        }
        Ok(Step::Pass(p))
    };
    let mut out = Vec::new();
    for (c, t) in classes.iter().enumerate() {
        let Some(t) = t else { continue };
        // Offset (biased), side base, cover base, has levels, then the reading planes' levels.
        let mut params = vec![plane_offset(t.plane_offset), t.side_base, t.cover_base, t.levels.is_some() as u32];
        if let Some(levels) = &t.levels { params.extend(levels.iter().map(|l| l.to_bits())); }
        let buf = pool.buffer("regions label params", params.len() as u64 * 4);
        pool.upload(buf, 0, params.iter().flat_map(|w| w.to_le_bytes()).collect());
        out.push(pass(&k.class, "regions label class", Some(t), buf, c as u32, u32::from(!out.is_empty()))?);
    }
    if out.is_empty() {
        out.push(pass(&k.empty, "regions label empty", None, BufRef::RUNTIME, 0, 0)?);
    } else {
        out.push(pass(&k.finish, "regions label finish", None, BufRef::RUNTIME, conflicts.0, 0)?);
    }
    Ok((output, out))
}

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.LabelBase = {
        precise_math: true
        workgroup_size: 64
        dispatch_range: shader.uniform(vec4u)
    }
    // dispatch_range.z is the class index; w is 0 for the first present class, 1 thereafter.
    mod.sdf.LabelClass = {
        ..mod.sdf.LabelBase
        grid: shader.uniform(vec4u)
        params: shader.buffer(u32)
        band_slot: shader.buffer(u32)
        cover: shader.buffer(u32)
        node_side: shader.buffer(u32)
        label_mask: shader.buffer_rw(u32)
        main: fn() {
            if global_id().x >= self.dispatch_range.y { return }
            let e = self.dispatch_range.x + global_id().x
            let cells = self.grid.x * self.grid.y
            let width = self.grid.x * 10u
            var v = 0u
            if self.dispatch_range.w != 0u { v = self.label_mask[e] }
            for b in 0u..4u {
                let i = e * 4u + b
                let p = i / (cells * 100u)
                let gx = i % width
                let gy = (i / width) % (self.grid.y * 10u)
                let q = p + self.params[0u]
                var s = 0u
                if q >= 16u && q < self.grid.z + 16u {
                    let c = (q - 16u) * cells + (gy / 10u) * self.grid.x + gx / 10u
                    let band = self.band_slot[c]
                    if band == 4294967295u { s = self.cover[self.params[2u] + c] & 1u }
                    else {
                        let node = band * 121u + (gy % 10u) * 11u + gx % 10u
                        if self.params[3u] == 0u { s = self.node_side[self.params[1u] + node] & 1u }
                        else if asfloat(self.node_side[node * 2u + 1u]) <= asfloat(self.params[4u + p]) { s = 1u }
                    }
                }
                if s != 0u { v = v | (1u << (b * 8u + self.dispatch_range.z)) }
            }
            self.label_mask[e] = v
        }
    }
    // dispatch_range.z is the conflicts count word. Count each conflicting node once.
    mod.sdf.LabelFinish = {
        ..mod.sdf.LabelBase
        counts: shader.buffer_rw(u32)
        label_mask: shader.buffer(u32)
        labels: shader.buffer_rw(u32)
        main: fn() {
            if global_id().x >= self.dispatch_range.y { return }
            let e = self.dispatch_range.x + global_id().x
            let mask = self.label_mask[e]
            var v = 0u
            for b in 0u..4u {
                let m = (mask >> (b * 8u)) & 255u
                var c = 255u
                for t in 0u..7u {
                    if c == 255u && (m & (1u << t)) != 0u { c = t }
                }
                if m != 0u {
                    if (m & (m - 1u)) != 0u { atomic_add(self.counts[self.dispatch_range.z], 1u) }
                }
                v = v | (c << (b * 8u))
            }
            self.labels[e] = v
        }
    }
    mod.sdf.LabelEmpty = {
        ..mod.sdf.LabelBase
        labels: shader.buffer_rw(u32)
        main: fn() {
            if global_id().x >= self.dispatch_range.y { return }
            self.labels[self.dispatch_range.x + global_id().x] = 4294967295u
        }
    }
    mod.sdf
}
