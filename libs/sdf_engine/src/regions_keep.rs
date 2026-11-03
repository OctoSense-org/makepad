//! A Keep round's per-component decision on the GPU (S1+S2, GPU-EXEC §9): which components of a
//! region a Keep keeps, from one chain over a compose round whose output 0 is the region R and
//! whose other outputs are its queries (R ∧ S@dz for a Meets test, R's level set at -r for a
//! Thin test), with each query level's reference level R.
//!
//! A component is one of R's contours with its holes (a hole's parent at R's level). Per plane:
//! - its area is the contour's signed area plus its holes' (chain's shoelace areas; the holes'
//!   are summed with integer atomics in 1/64 grid units², so the sum never depends on order);
//! - a query holds for it where some loop of that query lies in it (its reference contour): a
//!   hit word per contour and query output, set with atomic_max;
//! - AreaOver(a): area > a; Thin: no loop of the eroded query (the planner's offset by -r is
//!   empty); Meets: some loop of the meeting query.
//!
//! The plane's mode in its flag state picks: nothing, every component, or those whose tests all
//! hold (or not all). A hole takes its contour's decision. The answer is a word per chain loop,
//! 1 where it is kept, for regions_set's `keep`.
//!
//! Passes over the chain's loops: `KeepClear` (the capacity), `KeepGather` (hole areas and query
//! hits), `KeepDecide` (contours) and `KeepHoles`.
use makepad_draw::*;
use crate::chain::Chain;
use crate::gpu_jobs::{label, BufRef, Elements, JobError, Pass, Pool, Step};
use crate::regions_compose::{self as rc, MAX_FLAGS, MAX_OUTPUTS, STATES};
use crate::regions_program::Mode;

const CEILING: u32 = 1 << 20;
/// Tests one Keep holds.
pub const MAX_TESTS: usize = 4;
/// Hole areas are summed in units of 1 / AREA_SCALE grid units².
const AREA_SCALE: f32 = 64.0;
/// Grid units² per mm².
const PER_MM2: f32 = 100.0;
const FLAG_AT: u32 = 4;
const TEST_AT: u32 = FLAG_AT + MAX_FLAGS as u32 * 4;
const PLANES_AT: u32 = TEST_AT + MAX_TESTS as u32 * 4;
// the kernels read these offsets as literals
const _: () = assert!(FLAG_AT == 4 && TEST_AT == 12 && PLANES_AT == 28 && MAX_TESTS + STATES == 8 && AREA_SCALE == 64.0 && MAX_OUTPUTS == 8);

/// What a test reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestKind {
    /// The component's area over `value[p]` mm².
    AreaOver,
    /// No loop of query output `output` in the component.
    Thin,
    /// Some loop of query output `output` in the component.
    Meets,
}

#[derive(Clone, Debug)]
pub struct KeepTest {
    pub kind: TestKind,
    /// The query's output in the chained round (unused by AreaOver).
    pub output: u32,
    /// Per plane: the AreaOver threshold (mm²); unused by the others.
    pub value: Vec<f32>,
}

/// A Keep over the chain of a round at pieces' level `level` (output o at `level + o`).
#[derive(Clone, Debug)]
pub struct Keep {
    pub planes: u32,
    pub level: u32,
    pub tests: Vec<KeepTest>,
    /// Per plane and flag state.
    pub modes: Vec<[Mode; STATES]>,
    pub flags: Vec<rc::Flag>,
}

/// The Keep's kernels, made once per Cx.
#[derive(Clone)]
pub struct KeepKernels {
    clear: ComputeKernel,
    gather: ComputeKernel,
    decide: ComputeKernel,
    holes: ComputeKernel,
}
impl KeepKernels {
    pub fn new(cx: &mut Cx) -> Result<Self, String> {
        cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            let mut kernel = |name: LiveId, label: &str| -> Result<ComputeKernel, String> {
                let object = vm.value(module, name).as_object().ok_or_else(|| format!("{label} kernel missing"))?;
                ComputeKernel::new(vm, object, label)
            };
            Ok(Self {
                clear: kernel(id!(KeepClear), "regions keep clear")?,
                gather: kernel(id!(KeepGather), "regions keep gather")?,
                decide: kernel(id!(KeepDecide), "regions keep decide")?,
                holes: kernel(id!(KeepHoles), "regions keep holes")?,
            })
        })
    }
}

fn mode_code(m: Mode) -> u32 {
    match m {
        Mode::Nothing => 0,
        Mode::All => 1,
        Mode::Hold(true) => 2,
        Mode::Hold(false) => 3,
    }
}

/// The params buffer's words: header (planes, tests, level, 0), the flags, the tests (kind,
/// output, 0, 0), then per plane the tests' values and the modes per state.
fn words(keep: &Keep) -> Result<Vec<u32>, String> {
    let (planes, tests) = (keep.planes as usize, keep.tests.len());
    if tests > MAX_TESTS { return Err(format!("a Keep holds at most {MAX_TESTS} tests, not {tests}")); }
    if keep.flags.len() > MAX_FLAGS { return Err(format!("a Keep reads at most {MAX_FLAGS} flags")); }
    if keep.modes.len() != planes { return Err(format!("{} modes for {planes} planes", keep.modes.len())); }
    let mut w = vec![0u32; PLANES_AT as usize + planes * (MAX_TESTS + STATES)];
    w[..4].copy_from_slice(&[keep.planes, tests as u32, keep.level, 0]);
    for (i, f) in keep.flags.iter().enumerate() {
        let at = (FLAG_AT + i as u32 * 4) as usize;
        w[at..at + 4].copy_from_slice(&[f.any.0 + f.output, f.outputs, rc::plane_offset(f.dz), 1]);
    }
    for (i, t) in keep.tests.iter().enumerate() {
        let at = (TEST_AT + i as u32 * 4) as usize;
        let kind = match t.kind { TestKind::AreaOver => 0, TestKind::Thin => 1, TestKind::Meets => 2 };
        if t.kind != TestKind::AreaOver && (t.output == 0 || t.output >= MAX_OUTPUTS) { return Err(format!("test {i}: query output {}", t.output)); }
        w[at..at + 4].copy_from_slice(&[kind, t.output, 0, 0]);
        if t.kind == TestKind::AreaOver {
            if t.value.len() != planes { return Err(format!("test {i}: {} values for {planes} planes", t.value.len())); }
            for p in 0..planes { w[PLANES_AT as usize + p * (MAX_TESTS + STATES) + i] = (t.value[p] * PER_MM2).to_bits(); }
        }
    }
    for (p, modes) in keep.modes.iter().enumerate() {
        for (s, &m) in modes.iter().enumerate() { w[PLANES_AT as usize + p * (MAX_TESTS + STATES) + MAX_TESTS + s] = mode_code(m); }
    }
    Ok(w)
}

/// The Keep's decision over `chain`: a word per chain loop, 1 where it is kept.
pub fn steps(k: &KeepKernels, pool: &mut Pool, chain: &Chain, keep: &Keep) -> Result<(BufRef, Vec<Step>), JobError> {
    let w = words(keep).map_err(JobError::Failed)?;
    let params = pool.buffer("regions keep params", w.len() as u64 * 4);
    pool.upload(params, 0, w.iter().flat_map(|w| w.to_le_bytes()).collect());
    let lcap = chain.loop_capacity.max(1);
    let (acc, hits, kept) = (pool.buffer("regions keep holes", lcap as u64 * 4), pool.buffer("regions keep hits", lcap as u64 * MAX_OUTPUTS as u64 * 4), pool.buffer("regions keep", lcap as u64 * 4));
    let bind = |b: LiveId| -> BufRef {
        match b {
            id!(counts) => BufRef::COUNTS,
            id!(params) => params, id!(loop_meta) => chain.loop_meta, id!(loop_metric) => chain.loop_metric, id!(loop_tree) => chain.loop_tree,
            id!(acc) => acc, id!(hits) => hits, id!(kept) => kept,
            _ => BufRef::RUNTIME,
        }
    };
    let lw = chain.loops_word();
    let pass = |kernel: &ComputeKernel, name: &str, elements: Elements| -> Result<Step, JobError> {
        let mut p = Pass::new(kernel, label(name), elements, 4, CEILING, |b| bind(b));
        // a kernel that never reads the loop count has no `words`
        if kernel.layout().slots.iter().any(|s| s.name == id!(words)) { p = p.param_u32x4(id!(words), [lw.0, 0, 0, 0]).map_err(JobError::Failed)?; }
        Ok(Step::Pass(p))
    };
    let counted = Elements::Counted { word: lw, bound: lcap };
    let steps = vec![
        pass(&k.clear, "regions keep clear", Elements::Fixed(lcap))?,
        pass(&k.gather, "regions keep gather", counted)?,
        pass(&k.decide, "regions keep decide", counted)?,
        pass(&k.holes, "regions keep holes", counted)?,
    ];
    Ok((kept, steps))
}

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    // words.x: the chain's loops count word. params: see `words` (FLAG_AT 4, TEST_AT 12,
    // PLANES_AT 28, per plane MAX_TESTS + STATES = 8 words).
    mod.sdf.Keep = {
        precise_math: true
        workgroup_size: 64
        dispatch_range: shader.uniform(vec4u)
        words: shader.uniform(vec4u)
        counts: shader.buffer(u32)
        params: shader.buffer(u32)
        loop_meta: shader.buffer(vec4u)
        element: fn() -> u32 {
            if global_id().x >= self.dispatch_range.y { return 4294967295u }
            let l = self.dispatch_range.x + global_id().x
            if l >= self.counts[self.words.x] { return 4294967295u }
            return l
        }
        // A closed loop at the kept region's level (output 0).
        own: fn(l: u32) -> bool {
            let meta = self.loop_meta[l]
            return meta.y == self.params[2u] && (meta.w & 2u) == 0u
        }
        is_hole: fn(l: u32) -> bool { return (self.loop_meta[l].w & 1u) != 0u }
    }

    mod.sdf.KeepClear = {
        ..mod.sdf.Keep
        acc: shader.buffer_rw(u32)
        hits: shader.buffer_rw(u32)
        kept: shader.buffer_rw(u32)
        main: fn() {
            if global_id().x >= self.dispatch_range.y { return }
            let l = self.dispatch_range.x + global_id().x
            self.acc[l] = 0u
            for o in 0u..8u { self.hits[l * 8u + o] = 0u }
            self.kept[l] = 0u
        }
    }

    // A hole of the region adds its area to its contour; a query loop marks its reference
    // contour's hit word for its output.
    mod.sdf.KeepGather = {
        ..mod.sdf.Keep
        loop_metric: shader.buffer(vec4f)
        loop_tree: shader.buffer(vec4u)
        acc: shader.buffer_rw(u32)
        hits: shader.buffer_rw(u32)
        main: fn() {
            let l = self.element()
            if l == 4294967295u { return }
            let meta = self.loop_meta[l]
            if (meta.w & 2u) != 0u { return }
            let level = self.params[2u]
            let tree = self.loop_tree[l]
            if meta.y == level {
                if self.is_hole(l) && tree.x < self.counts[self.words.x] {
                    let a = 0.0 - self.loop_metric[l * 2u].x
                    atomic_add(self.acc[tree.x], u32(a * 64.0 + 0.5))
                }
            } else if meta.y > level && meta.y < level + 8u && tree.z < self.counts[self.words.x] {
                atomic_max(self.hits[tree.z * 8u + meta.y - level], 1u)
            }
        }
    }

    mod.sdf.KeepDecide = {
        ..mod.sdf.Keep
        loop_metric: shader.buffer(vec4f)
        acc: shader.buffer(u32)
        hits: shader.buffer(u32)
        kept: shader.buffer_rw(u32)
        // Plane p's flag state: bit i where flag i's output has a piece on plane p + dz.
        state: fn(p: u32) -> u32 {
            var s = 0u
            for i in 0u..2u {
                let at = 4u + i * 4u
                if self.params[at + 3u] != 0u {
                    let q = p + self.params[at + 2u]
                    if q >= 16u && q < self.params[0u] + 16u {
                        if self.counts[self.params[at] + (q - 16u) * self.params[at + 1u]] != 0u { s = s | (1u << i) }
                    }
                }
            }
            return s
        }
        main: fn() {
            let l = self.element()
            // no || guards: the shader's || and && do not short-circuit
            if l == 4294967295u { return }
            if !self.own(l) || self.is_hole(l) { return }
            let p = self.loop_meta[l].x
            let at = 28u + p * 8u
            let mode = self.params[at + 4u + self.state(p)]
            var all = true
            for t in 0u..self.params[1u] {
                let kind = self.params[12u + t * 4u]
                let bit = self.hits[l * 8u + self.params[12u + t * 4u + 1u]]
                var holds = true
                if kind == 0u {
                    let area = self.loop_metric[l * 2u].x - f32(self.acc[l]) / 64.0
                    holds = area > asfloat(self.params[at + t])
                } else if kind == 1u { holds = bit == 0u }
                else { holds = bit != 0u }
                all = all && holds
            }
            var keep = 0u
            if mode == 1u || (mode == 2u && all) || (mode == 3u && !all) { keep = 1u }
            self.kept[l] = keep
        }
    }

    // A hole takes its contour's decision.
    mod.sdf.KeepHoles = {
        ..mod.sdf.Keep
        loop_tree: shader.buffer(vec4u)
        kept: shader.buffer_rw(u32)
        main: fn() {
            let l = self.element()
            if l == 4294967295u { return }
            if !self.own(l) || !self.is_hole(l) { return }
            let parent = self.loop_tree[l].x
            if parent < self.counts[self.words.x] { self.kept[l] = self.kept[parent] }
        }
    }
    mod.sdf
}
