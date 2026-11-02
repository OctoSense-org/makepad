//! The brick volume of one object: which bricks of its lattice hold surface,
//! the program each one bakes from, and where each one lives in the atlas.
//!
//! Lattice (csg_sculpt's): sample `g` sits at `g * cell` in the object frame;
//! brick `b` holds samples `8b .. 8b + 7` per axis (no shared borders).
//! Values are clamped to `±band`. A brick is allocated when the distance at
//! the centre of its region (its samples and the gap to the next brick's
//! first) is within the region's half diagonal plus the band (the field is
//! 1-Lipschitz, so no sample of an unallocated brick lies inside the band).
//!
//! Work is per block of 4^3 bricks: one culled program decides a block that
//! the surface does not come near (all its bricks empty, one signed skip
//! distance), otherwise each brick is culled, lowered and tested. A brick's
//! program is its node culled to its box grown by the band; equal programs
//! are stored once. Blocks are classified in parallel, then applied in
//! lattice order so program ids and atlas slots stay stable. Re-baking after
//! an edit redoes the blocks the edit's changed region touches, the same
//! computation as a full build, so a dirty re-bake equals a full bake.

use crate::kernel::KernelSlot;
use crate::program::{lower_with, LowerError, Program};
use crate::twin;
use makepad_csg_math::thread_pool::{parallel_for, parallel_map, thread_count};
use makepad_csg_sdf::{changed, cull, node_bounds, Bounds, SdfNode};
use std::collections::HashMap;
use std::sync::Arc;

/// Samples per brick edge.
pub const BS: i32 = 8;
/// Bricks per block edge.
pub const BLOCK: i32 = 4;
/// Truth-atlas slots per row: a slot is 64 x 8 texels (8 z slices of 8 x 8).
pub const ATLAS_SLOTS_X: usize = 64;
pub const ATLAS_W: usize = ATLAS_SLOTS_X * 64;

/// What one brick of the volume is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Brick {
    /// Holds surface: its atlas slot and its program.
    Baked { slot: u32, program: u32 },
    /// No sample within the band: a signed lower bound of the distance over it
    /// (negative inside the solid).
    Empty { skip: f32 },
}

/// One object's brick volume.
#[derive(Clone)]
pub struct Volume {
    pub cell: f32,
    pub band: f32,
    /// The bricks the volume covers: `min .. min + dim` per axis.
    pub min: [i32; 3],
    pub dim: [i32; 3],
    pub bricks: Vec<Brick>,
    /// Distinct programs; `Brick::Baked::program` indexes them.
    pub programs: Vec<Program>,
    /// The compiled Splash kernels its programs call (a KERNEL op's slot
    /// indexes them; a slot keeps its index while the volume lives).
    pub kernels: Vec<KernelSlot>,
    /// Multiplies its kernels' distances: 1 / the steepest gradient the
    /// build measured near their surface (a kernel need not be a true
    /// distance; this keeps the field 1-Lipschitz for the brick tests and
    /// the march). Only shrinks while the volume lives.
    pub kernel_scale: f32,
    /// Multiplies the whole field, at its root, when a primitive in it is an estimate (the
    /// ellipsoid's): 1 / the steepest gradient measured near the surface. At the root, so a
    /// shell or a blend over it keeps its exact offsets. Only shrinks while the volume lives.
    pub field_scale: f32,
    program_ids: HashMap<Vec<u32>, u32>,
    free_slots: Vec<u32>,
    pub slot_count: u32,
    /// Bricks whose texels must be (re)baked.
    pub dirty: Vec<usize>,
    /// Bricks the CPU twin baked (a program the GPU cannot run: a mesh, a kernel past the GPU
    /// limits): their texels, which the baker uploads instead of baking.
    pub cpu: HashMap<usize, Arc<CpuBrick>>,
    /// Bricks a build classified again (baked or not): their neighbours'
    /// render-copy aprons may have changed.
    pub touched: Vec<usize>,
}

/// Build statistics.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuildStats {
    pub blocks: usize,
    pub blocks_skipped: usize,
    pub bricks_tested: usize,
    pub bricks_baked: usize,
    pub programs: usize,
    pub ms: f64,
    /// Of `ms`: the kernel scale's measure, the blocks (parallel), applying them.
    pub scale_ms: f64,
    pub blocks_ms: f64,
    pub apply_ms: f64,
}

/// A brick's samples baked on the CPU twin: 64 x 8 slot texels (z slices along x), each a
/// distance clamped to the band, and their material slots.
pub struct CpuBrick {
    pub d: Vec<f32>,
    pub slots: Vec<u8>,
}

enum BrickResult {
    Empty { skip: f32 },
    /// `cpu`: its samples, when the GPU cannot run its program.
    Bake { program: Program, cpu: Option<CpuBrick> },
}

/// The samples of brick `b` (lattice coordinates) of `program`, as the GPU bake lays them out.
fn bake_cpu(program: &[[f32; 4]], b: [i32; 3], cell: f32, band: f32, kernels: &[KernelSlot]) -> CpuBrick {
    let mut qs = Vec::with_capacity(512);
    for y in 0..8 {
        for tx in 0..64 {
            let (z, x) = (tx / 8, tx % 8);
            let g = [b[0] * BS + x, b[1] * BS + y, b[2] * BS + z];
            qs.push(g.map(|v| v as f32 * cell));
        }
    }
    let vals = twin::eval_many(program, &qs, kernels);
    CpuBrick { d: vals.iter().map(|(v, _)| v.max(-band).min(band) + 0.0).collect(), slots: vals.iter().map(|(_, s)| *s).collect() }
}

struct BlockResult {
    block_skip: Option<f64>,
    // Keep brick errors in order so release and touched still precede them.
    bricks: Vec<(usize, Result<BrickResult, LowerError>)>,
}

fn grow(b: Bounds, r: f64) -> Bounds {
    (b.0.map(|x| x - r), b.1.map(|x| x + r))
}

/// The steepest gradient of `root`'s field near its surface inside `bounds`
/// (`n` samples per axis: a volume's bricks, the resolution its brick tests work
/// at; forward differences of a cell), with a tenth more for what the samples
/// missed; within 1..8.
fn lipschitz(root: &SdfNode, kernels: &[KernelSlot], bounds: Bounds, band: f64, cell: f64, n: [i32; 3]) -> f32 {
    let Ok(p) = lower_with(root, kernels, 1.0) else { return 1.0 };
    let n = n.map(|v| v.max(1) as usize);
    let h = cell as f32;
    let texels = Arc::new(p.texels);
    let kernels = Arc::new(kernels.to_vec());
    let slices: Vec<_> = (0..n[2]).collect();
    let l = parallel_map(&slices, move |chunk| {
        let mut pts = Vec::with_capacity(n[0] * n[1] * chunk.len() * 4);
        for &z in chunk {
            for y in 0..n[1] {
                for x in 0..n[0] {
                    let i = [x, y, z];
                    let t: [f64; 3] = std::array::from_fn(|k| (i[k] as f64 + 0.5) / n[k] as f64);
                    let q: [f32; 3] = std::array::from_fn(|k| (bounds.0[k] + (bounds.1[k] - bounds.0[k]) * t[k]) as f32);
                    pts.push(q);
                    pts.push([q[0] + h, q[1], q[2]]);
                    pts.push([q[0], q[1] + h, q[2]]);
                    pts.push([q[0], q[1], q[2] + h]);
                }
            }
        }
        let d = twin::eval_many(&texels, &pts, &kernels);
        let mut l = 1.0f32;
        for c in d.chunks(4) {
            if (c[0].0.abs() as f64) > band * 4.0 + cell {
                continue;
            }
            let g = [c[1].0 - c[0].0, c[2].0 - c[0].0, c[3].0 - c[0].0].map(|v| v / h);
            let m = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            if m.is_finite() {
                l = l.max(m);
            }
        }
        vec![l]
    })
    .into_iter()
    .fold(1.0f32, f32::max);
    // a true distance (gradient 1, within the differences' error) stays as it is
    if l <= 1.05 {
        return 1.0;
    }
    (l * 1.1).clamp(1.0, 8.0)
}

impl Volume {
    /// An empty volume over `bounds` (the object's solid) at `cell`.
    pub fn new(bounds: Bounds, cell: f32, band: f32) -> Volume {
        let reach = band as f64 + cell as f64;
        let lo = bounds.0.map(|x| ((x - reach) / cell as f64 / BS as f64).floor() as i32);
        let hi = bounds.1.map(|x| ((x + reach) / cell as f64 / BS as f64).floor() as i32);
        // whole blocks
        let dim: [i32; 3] = std::array::from_fn(|i| ((hi[i] - lo[i] + 1 + BLOCK - 1) / BLOCK) * BLOCK);
        let n = (dim[0] * dim[1] * dim[2]) as usize;
        Volume {
            cell,
            band,
            min: lo,
            dim,
            bricks: vec![Brick::Empty { skip: f32::MAX }; n],
            programs: Vec::new(),
            kernels: Vec::new(),
            kernel_scale: 1.0,
            field_scale: 1.0,
            program_ids: HashMap::new(),
            free_slots: Vec::new(),
            slot_count: 0,
            dirty: Vec::new(),
            cpu: HashMap::new(),
            touched: Vec::new(),
        }
    }

    pub fn index(&self, b: [i32; 3]) -> usize {
        let l = [b[0] - self.min[0], b[1] - self.min[1], b[2] - self.min[2]];
        ((l[2] * self.dim[1] + l[1]) * self.dim[0] + l[0]) as usize
    }

    pub fn brick_of(&self, i: usize) -> [i32; 3] {
        let i = i as i32;
        let (x, y, z) = (i % self.dim[0], (i / self.dim[0]) % self.dim[1], i / (self.dim[0] * self.dim[1]));
        [x + self.min[0], y + self.min[1], z + self.min[2]]
    }

    /// The region brick `b` answers for: its samples and the gap to the next
    /// brick's first sample, `[8b, 8b + 8]` cells (object frame).
    pub fn brick_region(&self, b: [i32; 3]) -> Bounds {
        let c = self.cell as f64;
        (b.map(|v| (v * BS) as f64 * c), b.map(|v| (v * BS + BS) as f64 * c))
    }

    /// The box of brick `b`'s samples (object frame).
    pub fn brick_box(&self, b: [i32; 3]) -> Bounds {
        let c = self.cell as f64;
        (b.map(|v| (v * BS) as f64 * c), b.map(|v| (v * BS + BS - 1) as f64 * c))
    }

    fn program_id(&mut self, p: Program) -> u32 {
        let key: Vec<u32> = p.texels.iter().flat_map(|t| t.map(f32::to_bits)).collect();
        if let Some(id) = self.program_ids.get(&key) {
            return *id;
        }
        let id = self.programs.len() as u32;
        self.programs.push(p);
        self.program_ids.insert(key, id);
        id
    }

    fn alloc_slot(&mut self) -> u32 {
        self.free_slots.pop().unwrap_or_else(|| {
            self.slot_count += 1;
            self.slot_count - 1
        })
    }

    fn release(&mut self, i: usize) {
        if let Brick::Baked { slot, .. } = self.bricks[i] {
            self.free_slots.push(slot);
        }
        self.cpu.remove(&i);
    }

    /// Classifies and programs every brick of the blocks overlapping `region`
    /// (everything when `None`), queueing the baked ones for (re)baking.
    pub fn build(&mut self, root: &SdfNode, region: Option<Bounds>) -> Result<BuildStats, LowerError> {
        let t0 = std::time::Instant::now();
        let mut st = BuildStats::default();
        let band = self.band as f64;
        let c = self.cell as f64;
        let min = self.min;
        let dim = self.dim;
        crate::kernel::collect(root, &mut self.kernels);
        let mut region = region;
        let estimate = crate::kernel::has_estimate(root);
        if !self.kernels.is_empty() || estimate {
            let lo: [f64; 3] = std::array::from_fn(|i| (self.min[i] * BS) as f64 * c);
            let hi: [f64; 3] = std::array::from_fn(|i| ((self.min[i] + self.dim[i]) * BS) as f64 * c);
            let t = std::time::Instant::now();
            let scale = 1.0 / lipschitz(root, &self.kernels, (lo, hi), band, c, self.dim);
            st.scale_ms = t.elapsed().as_secs_f64() * 1e3;
            // steeper than before: every brick again, at one scale (an estimate's field
            // scales at its root, kernels' at each kernel)
            let held = if estimate { &mut self.field_scale } else { &mut self.kernel_scale };
            if scale < *held {
                *held = scale;
                region = None;
            }
        }
        let brick_region = move |b: [i32; 3]| {
            (b.map(|v| (v * BS) as f64 * c), b.map(|v| (v * BS + BS) as f64 * c))
        };
        // A brick's region is what the march looks it up for: [8b, 8b + 8) cells
        // (its samples and the gap to the next brick's first); the tests cover it whole.
        let half_diag = move |n: f64| 0.5 * n * BS as f64 * c * 3f64.sqrt();
        let nb: [i32; 3] = std::array::from_fn(|i| dim[i] / BLOCK);
        let mut blocks: Vec<[i32; 3]> = Vec::new();
        for bz in 0..nb[2] {
            for by in 0..nb[1] {
                for bx in 0..nb[0] {
                    let b0 = [min[0] + bx * BLOCK, min[1] + by * BLOCK, min[2] + bz * BLOCK];
                    let bb = (brick_region(b0).0, brick_region([b0[0] + BLOCK - 1, b0[1] + BLOCK - 1, b0[2] + BLOCK - 1]).1);
                    if let Some(r) = &region {
                        let g = grow(*r, band + c);
                        if (0..3).any(|i| bb.1[i] < g.0[i] || bb.0[i] > g.1[i]) {
                            continue;
                        }
                    }
                    blocks.push(b0);
                }
            }
        }
        let t_blocks = std::time::Instant::now();
        let kernel_scale = self.kernel_scale;
        let field_scale = self.field_scale;
        let kernels = Arc::new(self.kernels.clone());
        let root = Arc::new(root.clone());
        // Small tasks let workers take more blocks as they finish.
        let chunk_size = (blocks.len() / (thread_count() * 4)).max(1);
        let tasks: Vec<_> = blocks
            .chunks(chunk_size)
            .map(|chunk| {
                let chunk = chunk.to_vec();
                let kernels = kernels.clone();
                let root = root.clone();
                move || {
                    chunk
                        .iter()
                        .map(|&b0| -> Result<BlockResult, LowerError> {
                            let bb = (brick_region(b0).0, brick_region([b0[0] + BLOCK - 1, b0[1] + BLOCK - 1, b0[2] + BLOCK - 1]).1);
                            // the block as a whole: one test at its centre
                            let rb = half_diag(BLOCK as f64);
                            let centre: [f64; 3] = std::array::from_fn(|i| 0.5 * (bb.0[i] + bb.1[i]));
                            // A culled program is exact only up to the cull's margin (band + c)
                            // around the region: what it dropped can be that near, so a skip
                            // distance never goes past the margin.
                            let margin = band + c;
                            let skip_of = |d: f64, r: f64| if d < 0.0 { -(-d - r).min(margin) } else { (d - r).min(margin) };
                            let mut block_prog = None;
                            let block_skip = match cull(&root, &grow(bb, margin)) {
                                None => Some(margin),
                                Some(n) => {
                                    let p = lower_with(&n, &kernels, kernel_scale)?.scaled(field_scale);
                                    let (d, _) = twin::eval_with(&p.texels, centre.map(|v| v as f32), &kernels);
                                    let d = d as f64;
                                    block_prog = Some(p);
                                    (d.abs() > rb + band).then(|| skip_of(d, rb))
                                }
                            };
                            // A block whose program calls kernels: its bricks' centre
                            // tests in one batch on the block's program (exact over the
                            // block, so over each brick), one kernel call per op.
                            let batched: Option<Vec<f32>> = match (&block_prog, block_skip) {
                                (Some(p), None) if p.texels.iter().any(|t| t[0] == crate::program::op::KERNEL) => {
                                    let mut ctrs = Vec::with_capacity((BLOCK * BLOCK * BLOCK) as usize);
                                    for z in 0..BLOCK {
                                        for y in 0..BLOCK {
                                            for x in 0..BLOCK {
                                                let bx = brick_region([b0[0] + x, b0[1] + y, b0[2] + z]);
                                                ctrs.push(std::array::from_fn(|k| (0.5 * (bx.0[k] + bx.1[k])) as f32));
                                            }
                                        }
                                    }
                                    Some(twin::eval_many(&p.texels, &ctrs, &kernels).into_iter().map(|(d, _)| d).collect())
                                }
                                _ => None,
                            };
                            let mut bricks = Vec::with_capacity((BLOCK * BLOCK * BLOCK) as usize);
                            for z in 0..BLOCK {
                                for y in 0..BLOCK {
                                    for x in 0..BLOCK {
                                        let b = [b0[0] + x, b0[1] + y, b0[2] + z];
                                        let l = [b[0] - min[0], b[1] - min[1], b[2] - min[2]];
                                        let i = ((l[2] * dim[1] + l[1]) * dim[0] + l[0]) as usize;
                                        if let Some(skip) = block_skip {
                                            bricks.push((i, Ok(BrickResult::Empty { skip: skip as f32 })));
                                            continue;
                                        }
                                        let bx = brick_region(b);
                                        let r = half_diag(1.0);
                                        let ctr: [f64; 3] = std::array::from_fn(|k| 0.5 * (bx.0[k] + bx.1[k]));
                                        let k = ((z * BLOCK + y) * BLOCK + x) as usize;
                                        let brick = match cull(&root, &grow(bx, band + c)) {
                                            None => Ok(BrickResult::Empty { skip: margin as f32 }),
                                            Some(n) => lower_with(&n, &kernels, kernel_scale).map(|p| p.scaled(field_scale)).map(|p| {
                                                let d = match &batched {
                                                    Some(ds) => ds[k],
                                                    None => twin::eval_with(&p.texels, ctr.map(|v| v as f32), &kernels).0,
                                                };
                                                let d = d as f64;
                                                if d.abs() > r + band {
                                                    BrickResult::Empty { skip: skip_of(d, r) as f32 }
                                                } else {
                                                    let cpu = (!p.fits_gpu()).then(|| bake_cpu(&p.texels, b, c as f32, band as f32, &kernels));
                                                    BrickResult::Bake { program: p, cpu }
                                                }
                                            }),
                                        };
                                        bricks.push((i, brick));
                                    }
                                }
                            }
                            Ok(BlockResult { block_skip, bricks })
                        })
                        .collect::<Vec<_>>()
                }
            })
            .collect();
        let results: Vec<_> = parallel_for(tasks).into_iter().flatten().collect();
        st.blocks_ms = t_blocks.elapsed().as_secs_f64() * 1e3;
        let t_apply = std::time::Instant::now();
        // Apply in lattice order: slot reuse and program ids depend on it.
        for result in results {
            st.blocks += 1;
            let block = result?;
            for (i, brick) in block.bricks {
                self.release(i);
                self.touched.push(i);
                if block.block_skip.is_none() {
                    st.bricks_tested += 1;
                }
                self.bricks[i] = match brick? {
                    BrickResult::Empty { skip } => Brick::Empty { skip },
                    BrickResult::Bake { program: p, cpu } => {
                        let program = self.program_id(p);
                        let slot = self.alloc_slot();
                        st.bricks_baked += 1;
                        if let Some(cpu) = cpu {
                            self.cpu.insert(i, Arc::new(cpu));
                        }
                        self.dirty.push(i);
                        Brick::Baked { slot, program }
                    }
                };
            }
            if block.block_skip.is_some() {
                st.blocks_skipped += 1;
            }
        }
        st.programs = self.programs.len();
        st.apply_ms = t_apply.elapsed().as_secs_f64() * 1e3;
        st.ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(st)
    }

    /// The field's distances per true distance (its kernels' and its root's scales): a march's
    /// footprint and a pick's step take it.
    pub fn step_scale(&self) -> f32 {
        self.kernel_scale * self.field_scale
    }

    /// Whether the volume's bricks cover `bounds` grown by the band.
    pub fn covers(&self, bounds: &Bounds) -> bool {
        let lo = self.brick_box(self.min).0;
        let hi = self.brick_box([self.min[0] + self.dim[0] - 1, self.min[1] + self.dim[1] - 1, self.min[2] + self.dim[2] - 1]).1;
        let g = grow(*bounds, self.band as f64 + self.cell as f64);
        (0..3).all(|i| g.0[i] >= lo[i] && g.1[i] <= hi[i])
    }

    /// Re-builds after `old` became `new`: the blocks the change can reach.
    /// `Ok(None)` when the new solid leaves the volume (the caller makes a new
    /// one).
    pub fn edit(&mut self, old: &SdfNode, new: &SdfNode) -> Result<Option<BuildStats>, LowerError> {
        if !node_bounds(new).is_some_and(|b| self.covers(&b)) {
            return Ok(None);
        }
        match changed(old, new) {
            None => Ok(Some(BuildStats::default())),
            Some(r) => self.build(new, Some(r)).map(Some),
        }
    }

    /// The index the GPU reads per brick (x + dim.x (y + dim.y z)): its slot
    /// (-1 when empty) and, for an empty brick, its signed skip distance.
    pub fn index_texels(&self) -> Vec<[f32; 4]> {
        self.bricks
            .iter()
            .map(|b| match *b {
                Brick::Baked { slot, .. } => [slot as f32, 0.0, 0.0, 0.0],
                Brick::Empty { skip } => [-1.0, skip, 0.0, 0.0],
            })
            .collect()
    }

    /// The 26 neighbours of brick `i` inside the volume, and itself.
    pub fn around(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let b = self.brick_of(i);
        (0..27).filter_map(move |k| {
            let n = [b[0] + k % 3 - 1, b[1] + (k / 3) % 3 - 1, b[2] + k / 9 - 1];
            let inside = (0..3).all(|a| n[a] >= self.min[a] && n[a] < self.min[a] + self.dim[a]);
            inside.then(|| self.index(n))
        })
    }

    /// The baked value of every sample of brick `i` on the CPU twin (64 x 8
    /// slot texels, z slices along x), and its material slots.
    pub fn bake_brick_cpu(&self, i: usize) -> Option<(Vec<f32>, Vec<u8>)> {
        let Brick::Baked { program, .. } = self.bricks[i] else { return None };
        let b = bake_cpu(&self.programs[program as usize].texels, self.brick_of(i), self.cell, self.band, &self.kernels);
        Some((b.d, b.slots))
    }

    /// The distance and material slot of `root` at `points`, all inside `region`: its program
    /// culled to the region, as a brick's. `f32::MAX` everywhere when nothing of the solid is
    /// near it, or it cannot be lowered. For a client that asks about far more space than one
    /// brick, at its own resolution.
    pub fn sample_region(root: &SdfNode, kernels: &[KernelSlot], kernel_scale: f32, region: Bounds, points: &[[f32; 3]], out: &mut Vec<(f32, u8)>) {
        let program = cull(root, &region).and_then(|n| lower_with(&n, kernels, kernel_scale).ok());
        match program {
            Some(p) => out.extend(twin::eval_many(&p.texels, points, kernels)),
            None => out.extend(points.iter().map(|_| (f32::MAX, 0))),
        }
    }

    /// The node's bounds (the volume a new object needs).
    pub fn bounds_of(root: &SdfNode) -> Option<Bounds> {
        node_bounds(root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use makepad_csg_sdf::SdfOp;

    fn at(p: [f64; 3], of: SdfNode) -> SdfNode {
        SdfNode::Transform { at: p, rotate: [0.0; 3], scale: 1.0, of: Box::new(of) }
    }

    /// Fab's demo vase (`vase.cad.splash`) at a height and a belly radius (mm).
    fn vase(height: f64, belly: f64) -> SdfNode {
        let (neck, wall, blend) = (9.5, 1.6, 9.0);
        let body = SdfNode::Combine {
            op: SdfOp::Union,
            blend,
            of: vec![
                at([0.0, height * 0.34, 0.0], SdfNode::Ellipsoid { radii: [belly, height * 0.36, belly] }),
                at([0.0, height * 0.74, 0.0], SdfNode::Cylinder { radius: neck, top_radius: neck, height: height * 0.42 }),
            ],
        };
        let footed = SdfNode::Combine { op: SdfOp::Difference, blend: 0.0, of: vec![body, at([0.0, -7.0, 0.0], SdfNode::Box { size: [4.0 * belly, 20.0, 4.0 * belly], radius: 0.0 })] };
        let hollow = SdfNode::Shell { thickness: wall, of: Box::new(footed) };
        let rimmed = SdfNode::Combine { op: SdfOp::Union, blend: 1.5, of: vec![hollow, at([0.0, height * 0.94, 0.0], SdfNode::Torus { radius: neck + 1.2, tube: 1.8 })] };
        let (low, high) = (height * 0.87 - wall, height + 20.0);
        let mouth = at([0.0, (low + high) / 2.0, 0.0], SdfNode::Cylinder { radius: neck - wall, top_radius: neck - wall, height: high - low });
        SdfNode::Combine { op: SdfOp::Difference, blend: 0.0, of: vec![rimmed, mouth] }
    }

    /// The samples of empty brick `i` that lie inside the band (they should not exist).
    fn inside_band(vol: &Volume, root: &SdfNode, i: usize) -> Vec<([f32; 3], f32)> {
        let p = crate::program::lower_with(root, &[], 1.0).expect("lowers").scaled(vol.field_scale).texels;
        let b = vol.brick_of(i);
        let mut out = Vec::new();
        for z in 0..BS {
            for y in 0..BS {
                for x in 0..BS {
                    let q = [b[0] * BS + x, b[1] * BS + y, b[2] * BS + z].map(|v| v as f32 * vol.cell);
                    let (d, _) = twin::eval(&p, q);
                    if d.abs() < vol.band {
                        out.push((q, d));
                    }
                }
            }
        }
        out
    }

    /// An edited volume equals a volume built at the new value (at its scale), and no brick it
    /// calls empty has a sample inside the band (the march would step through the surface
    /// there): the vase's belly is an ellipsoid, whose field is an estimate.
    #[test]
    fn a_dragged_vase_keeps_every_surface_brick() {
        let steps: Vec<(f64, f64)> = (0..=12).map(|k| (70.0, 21.0 + k as f64 * 2.5)).chain((0..=12).map(|k| (70.0 + k as f64 * 6.0, 51.0))).collect();
        let first = vase(steps[0].0, steps[0].1);
        let bounds = Volume::bounds_of(&first).unwrap();
        let pad: [f64; 3] = std::array::from_fn(|i| (bounds.1[i] - bounds.0[i]) * 0.6);
        let room = (std::array::from_fn(|i| bounds.0[i] - pad[i]), std::array::from_fn(|i| bounds.1[i] + pad[i] * 2.0));
        let cell = 0.6f32;
        let mut edited = Volume::new(room, cell, cell * 4.0);
        edited.build(&first, None).unwrap();
        let mut prev = first;
        let mut misses = Vec::new();
        for &(h, belly) in &steps[1..] {
            let next = vase(h, belly);
            if edited.edit(&prev, &next).unwrap().is_none() {
                continue;
            }
            // at the scale the edited volume holds (a scale only shrinks while a volume lives)
            let mut full = Volume::new(room, cell, cell * 4.0);
            full.field_scale = edited.field_scale;
            full.kernel_scale = edited.kernel_scale;
            full.build(&next, None).unwrap();
            let kind = |v: &Volume| v.bricks.iter().map(|b| matches!(b, Brick::Baked { .. })).collect::<Vec<_>>();
            let (ke, kf) = (kind(&edited), kind(&full));
            let differ = (0..ke.len()).filter(|i| ke[*i] != kf[*i]).count();
            let holes: Vec<usize> = (0..full.bricks.len()).filter(|i| !kf[*i] && !inside_band(&full, &next, *i).is_empty()).collect();
            if differ > 0 || !holes.is_empty() {
                let example = holes.first().map(|i| (full.brick_of(*i), inside_band(&full, &next, *i)[0]));
                misses.push(format!("height {h} belly {belly}: {differ} bricks differ from a full build, {} empty bricks hold surface (e.g. {example:?})", holes.len()));
            }
            prev = next;
        }
        assert!(misses.is_empty(), "{}", misses.join("\n"));
    }
}
