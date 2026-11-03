//! S1b on the GPU (GPU-EXEC §9 S1b, §10.3 "S1b concentric"): the concentric solid fill of a
//! region's components, with the CPU's concentric_solid semantics, decided over the loops of
//! the region's own field levels (extract, then the chain with its reference table).
//!
//! The CPU fills a component C round by round. A wide C gets the rings of offset(C, -s/2), and
//! each component of offset(C, -s) is filled next; any other C ends with one bead, two beads
//! sharing its width, or a centre line. Offsets compose, so round r's components are the
//! components of the level B_r = -r s0 of the region's field (Miter(3)), and their loops are
//! the level L_r = -(r + 1/2) s0. s0 is the plane's concentric spacing (the first layer's own).
//!
//! The levels, per plane (d' in grid units), and the reference level each resolves against in
//! the chain (a loop's `loop_tree.z` is the innermost contour of that level holding it):
//! - B_0 = 0: the region's own boundary (the caller's boundary slot); no reference;
//! - B_r = -r s0 for r = 1..=R: reference B_{r-1};
//! - L_r = -(r + 1/2) s0 for r < R: reference B_r;
//! - T_r = -((r + 5/4) s0 + [`THICK_MARGIN`]) for r < R: reference B_r (the wide test);
//! - N = -(the narrow depth) when the run has the narrow test: reference B_0.
//!
//! Wide. On the CPU a component is wide when short >= 2.5 s. short is the least hull width
//! across C's edge directions. With holes it is the least of that and the thickness t, the
//! deepest inset that leaves anything (bisected to 0.005 mm from below). A T_r loop inside C
//! proves t >= 2.5 s0 + 0.006 mm, which has two consequences:
//! - the CPU's bisected thickness is past 2.5 s0;
//! - so is the hull width: it is at least twice the Euclidean inradius, and the miter inset
//!   never exceeds that.
//! So every component the GPU calls wide is wide on the CPU. Every other active component goes
//! whole to the CPU's own concentric_solid (a handoff): the terminal components (short <
//! 2.5 s0), the rare thin but hull-wide ones, and the components past round R. The CPU then
//! decides as it always does, so the loop counts are the CPU's by construction. Handoffs run
//! on the CPU pool beside the GPU batch.
//!
//! A B_r contour is active when every component above it is wide (its reference chain up to
//! B_0). Round 0 also applies the run's rules:
//! - with the narrow test, a component holding an N loop is not narrow (the CPU fills it with
//!   lines);
//! - with holed components skipped (floating-shell bands), a component with a hole stays on
//!   the CPU's Round loop.
//!
//! Passes (elements: the loop bound):
//! 1. `S1bClear`: the per-loop marks;
//! 2. `S1bMark`: each T loop marks its component wide, each N loop marks it deep, and each
//!    B_0 hole marks its contour holed (atomic adds, so the order does not matter);
//! 3. `S1bState`: each loop's record and whether it is selected, by a walk up its reference
//!    chain (at most R links);
//! 4. the exclusive scan of the selection;
//! 5. `S1bFill`: the selected records in loop order, and the header.
//!
//! `s1b_out` (u32 words) starts with [`OUT_HEADER`] words: records, loop bound, rounds, flags.
//! Then come 4 words per record: loop id, `kind | state << 2 | round << 8 | plane << 16`, link,
//! source. The records are:
//! - a component (kind 0): an active B_r contour, with its state (wide, handoff, not narrow,
//!   holed). Its link is the component above it (NONE at round 0). At round 0 its source is
//!   the FeatureSet seg of its root piece, which names the region component it came from;
//! - a loop (kind 1): an L_r ring (contour or hole) of a wide component; its link is that
//!   component;
//! - a hole (kind 2): a B_r hole of a handoff component; its link is that component's contour.
//!
//! Ring points come from the rings stage by loop id. Like the CPU, the consumer drops a ring
//! with fewer than 3 points.
use crate::gpu_jobs::{label, BufRef, Elements, JobError, Pass, Pool, ScanKernels, Step};
use crate::plane_abi::{concentric_level, LEVEL_CONCENTRIC, LEVEL_UNASSIGNED, NONE};
use crate::regions_overhang::Size;
use makepad_draw::*;
use std::collections::BTreeMap;

/// Rounds a run may decide on the GPU (3 R + 2 level slots, within extract's 64).
pub const ROUNDS_MAX: u32 = 20;
/// L_r = LOOP_BASE + r (concentric_level(r)), B_r = BOUNDARY_BASE + r - 1 for r >= 1,
/// T_r = THICK_BASE + r, and N.
pub const LOOP_BASE: u32 = LEVEL_CONCENTRIC;
pub const BOUNDARY_BASE: u32 = LEVEL_CONCENTRIC + ROUNDS_MAX;
pub const THICK_BASE: u32 = LEVEL_CONCENTRIC + 2 * ROUNDS_MAX;
pub const NARROW_LEVEL: u32 = LEVEL_CONCENTRIC + 3 * ROUNDS_MAX;
const _: () = assert!(NARROW_LEVEL < LEVEL_UNASSIGNED);
/// mm added to the wide test's half thickness: past the CPU's bisection step (0.005 mm on the
/// thickness) and the field's error against the CPU's offsets.
pub const THICK_MARGIN: f32 = 0.003;
/// Grid units the field reaches past the deepest level.
pub const DEPTH_MARGIN: f32 = 1.5;
/// Words of the `s1b` parameter header.
pub const HEADER: u32 = 12;
/// Words of the `s1b_out` header.
pub const OUT_HEADER: u32 = 4;
/// Grid units per mm.
const GRID: f32 = 10.0;
const FLAG_NARROW: u32 = 1;
const FLAG_SKIP_HOLED: u32 = 2;

/// The kernels in pass order.
pub const KERNELS: [&str; 4] = ["S1bClear", "S1bMark", "S1bState", "S1bFill"];

/// Loop `r` (0-based): the level -(r + 1/2) s0.
pub fn loop_level(r: u32) -> u32 {
    debug_assert!(r < ROUNDS_MAX);
    concentric_level(r)
}
/// Round `r`'s component boundary -r s0 (`boundary` for r = 0: the region's own).
pub fn boundary_level(r: u32, boundary: u32) -> u32 {
    debug_assert!(r <= ROUNDS_MAX);
    if r == 0 { boundary } else { BOUNDARY_BASE + r - 1 }
}
/// Round `r`'s wide test -((r + 5/4) s0 + THICK_MARGIN).
pub fn thick_level(r: u32) -> u32 {
    debug_assert!(r < ROUNDS_MAX);
    THICK_BASE + r
}

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    // s1b[0..12]: loops (known), the loops' count word (NONE: known), the loop bound, the
    // boundary level, rounds R, flags (1 the narrow test, 2 holed components skipped), the
    // selection's total word, the loop, boundary (r = 1) and thick level bases, the narrow
    // level, 0.
    mod.sdf.S1b = {
        workgroup_size: 64
        s1b: shader.buffer(u32)
        loops: shader.buffer(vec4u)
        loop_meta: shader.buffer(vec4u)
        loop_tree: shader.buffer(vec4u)
        piece_info: shader.buffer(vec4u)
        s1b_mark: shader.buffer(u32)
        s1b_state: shader.buffer(vec4u)
        s1b_count: shader.buffer(u32)
        s1b_scan: shader.buffer(u32)
        s1b_out: shader.buffer(u32)
        counts: shader.buffer(u32)
        dispatch_range: shader.uniform(vec4u)

        index: fn() -> u32 {
            let i = global_id().x
            if i >= self.dispatch_range.y { return 4294967295u }
            return self.dispatch_range.x + i
        }
        bound: fn() -> u32 { return self.s1b[2u] }
        rounds: fn() -> u32 { return self.s1b[4u] }
        n_loops: fn() -> u32 {
            let w = self.s1b[1u]
            if w != 4294967295u { return min(self.counts[w], self.bound()) }
            return min(self.s1b[0u], self.bound())
        }
        // A level's role and round: x = 0 a boundary B_r, 1 a loop L_r, 2 a wide test T_r,
        // 3 the narrow test, 4 none of S1b's.
        role: fn(level: u32) -> vec2u {
            let r = self.rounds()
            if level == self.s1b[3u] { return vec2u(0u, 0u) }
            if level >= self.s1b[7u] && level < self.s1b[7u] + r { return vec2u(1u, level - self.s1b[7u]) }
            if level >= self.s1b[8u] && level < self.s1b[8u] + r { return vec2u(0u, level - self.s1b[8u] + 1u) }
            if level >= self.s1b[9u] && level < self.s1b[9u] + r { return vec2u(2u, level - self.s1b[9u]) }
            if level == self.s1b[10u] { return vec2u(3u, 0u) }
            return vec2u(4u, 0u)
        }
        mark: fn(c: u32, k: u32) -> u32 { return self.s1b_mark[c * 3u + k] }
        // B_r contour c: 0 wide, 1 handoff, 2 not narrow, 3 holed (both at round 0 only), or 4
        // inactive (a component above it is not wide, or round 0 is skipped).
        state: fn(c: u32, r: u32) -> u32 {
            let n = self.bound()
            if c >= n { return 4u }
            var x = c
            var i = 0u
            while i < r {
                x = self.loop_tree[x].z
                if x >= n { return 4u }
                if self.mark(x, 0u) == 0u { return 4u }
                i += 1u
            }
            let flags = self.s1b[5u]
            var skip = 4u
            if (flags & 1u) != 0u && self.mark(x, 1u) != 0u { skip = 2u }
            if (flags & 2u) != 0u && self.mark(x, 2u) != 0u { skip = 3u }
            if skip != 4u {
                if r == 0u { return skip }
                return 4u
            }
            if r < self.rounds() && self.mark(c, 0u) != 0u { return 0u }
            return 1u
        }
    }

    mod.sdf.S1bClear = {
        ..mod.sdf.S1b
        s1b_mark: shader.buffer_rw(u32)
        main: fn() {
            let l = self.index()
            if l == 4294967295u { return }
            self.s1b_mark[l * 3u] = 0u
            self.s1b_mark[l * 3u + 1u] = 0u
            self.s1b_mark[l * 3u + 2u] = 0u
        }
    }

    mod.sdf.S1bMark = {
        ..mod.sdf.S1b
        s1b_mark: shader.buffer_rw(u32)
        main: fn() {
            let l = self.index()
            if l == 4294967295u { return }
            if l >= self.n_loops() { return }
            let meta = self.loop_meta[l]
            if (meta.w & 2u) != 0u { return }
            let role = self.role(meta.y)
            let tree = self.loop_tree[l]
            let n = self.bound()
            if role.x == 2u && tree.z < n { atomic_add(self.s1b_mark[tree.z * 3u], 1u) }
            if role.x == 3u && tree.z < n { atomic_add(self.s1b_mark[tree.z * 3u + 1u], 1u) }
            if role.x == 0u && role.y == 0u && (meta.w & 1u) != 0u && tree.x < n { atomic_add(self.s1b_mark[tree.x * 3u + 2u], 1u) }
        }
    }

    mod.sdf.S1bState = {
        ..mod.sdf.S1b
        s1b_state: shader.buffer_rw(vec4u)
        s1b_count: shader.buffer_rw(u32)
        main: fn() {
            let l = self.index()
            if l == 4294967295u { return }
            var sel = 0u
            var word = 0u
            var link = 4294967295u
            var source = 4294967295u
            if l < self.n_loops() {
                let meta = self.loop_meta[l]
                let role = self.role(meta.y)
                let tree = self.loop_tree[l]
                let r = role.y
                if (meta.w & 2u) == 0u {
                    if role.x == 0u {
                        if (meta.w & 1u) == 0u {
                            let st = self.state(l, r)
                            if st != 4u {
                                sel = 1u
                                word = st << 2u
                                if r > 0u { link = tree.z }
                                else { source = self.piece_info[self.loops[l].x].z & 536870911u }
                            }
                        } else {
                            if self.state(tree.x, r) == 1u {
                                sel = 1u
                                word = 2u
                                link = tree.x
                            }
                        }
                    }
                    if role.x == 1u {
                        if self.state(tree.z, r) == 0u {
                            sel = 1u
                            word = 1u
                            link = tree.z
                        }
                    }
                }
                word = word | (r << 8u) | (meta.x << 16u)
            }
            self.s1b_state[l] = vec4u(word, link, source, 0u)
            self.s1b_count[l] = sel
        }
    }

    mod.sdf.S1bFill = {
        ..mod.sdf.S1b
        s1b_out: shader.buffer_rw(u32)
        main: fn() {
            let l = self.index()
            if l == 4294967295u { return }
            if l == 0u {
                self.s1b_out[0u] = self.counts[self.s1b[6u]]
                self.s1b_out[1u] = self.bound()
                self.s1b_out[2u] = self.rounds()
                self.s1b_out[3u] = self.s1b[5u]
            }
            if self.s1b_count[l] == 0u { return }
            let st = self.s1b_state[l]
            let at = 4u + self.s1b_scan[l] * 4u
            self.s1b_out[at] = l
            self.s1b_out[at + 1u] = st.x
            self.s1b_out[at + 2u] = st.y
            self.s1b_out[at + 3u] = st.z
        }
    }

    mod.sdf
}

/// S1b's kernels, made once per Cx (after gpu_jobs' modules are registered).
#[derive(Clone)]
pub struct S1bKernels { kernels: Vec<ComputeKernel> }
impl S1bKernels {
    pub fn new(cx: &mut Cx) -> Result<Self, String> {
        cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            let kernels = KERNELS.iter().map(|name| {
                let object = vm.value(module, LiveId::from_str(name)).as_object().ok_or_else(|| format!("{name} kernel missing"))?;
                ComputeKernel::new(vm, object, name)
            }).collect::<Result<Vec<_>, String>>()?;
            Ok(Self { kernels })
        })
    }
    fn get(&self, name: &str) -> &ComputeKernel { &self.kernels[KERNELS.iter().position(|k| *k == name).unwrap()] }
}

/// What S1b decides over, and the run's rules.
#[derive(Clone, Debug)]
pub struct S1bSpec {
    /// Per plane: s0 in mm (spacing(w_first, h) on a first layer, else spacing(w_solid, h));
    /// NaN where the plane has no region of this run.
    pub spacing: Vec<f32>,
    /// Per plane: the narrow test's depth in mm (6 spacing(w_solid, h)): a component holding
    /// that level is not narrow. Empty: no test (vertical shells, bands).
    pub narrow: Vec<f32>,
    /// Round-0 components with a hole stay on the CPU (floating-shell bands: the Round loop).
    pub skip_holed: bool,
    /// R: the rounds decided here. A component wide at round R - 1 hands its children over.
    pub rounds: u32,
    /// The level id of the region's own boundary pieces (extract's d' = 0 slot).
    pub boundary: u32,
    /// The chain's loops.
    pub loops: Size,
}

/// The rounds after which no component of a region narrower than `narrow` is wide (it would
/// need depth (R - 1 + 5/4) s0 < narrow): ceil(narrow / s0 - 1/4), at least 1.
pub fn rounds_for(spacing: f32, narrow: f32) -> u32 {
    ((narrow / spacing - 0.25).ceil().max(1.0) as u32).min(ROUNDS_MAX)
}

/// Extract's level slots for a run: ids, and values per plane (planes x slots, plane-major,
/// d' grid units; NaN where a plane has no region). The boundary slot (value 0) comes first.
#[derive(Clone, Debug, PartialEq)]
pub struct S1bLevels { pub ids: Vec<u32>, pub values: Vec<f32> }

impl S1bSpec {
    fn check(&self) -> Result<(), String> {
        if self.rounds == 0 || self.rounds > ROUNDS_MAX { return Err(format!("{} rounds (1..={ROUNDS_MAX})", self.rounds)); }
        if !self.narrow.is_empty() && self.narrow.len() != self.spacing.len() { return Err(format!("{} planes, {} narrow depths", self.spacing.len(), self.narrow.len())); }
        if (LEVEL_CONCENTRIC..LEVEL_UNASSIGNED).contains(&self.boundary) { return Err(format!("boundary level {} inside S1b's", self.boundary)); }
        if let Some(s) = self.spacing.iter().find(|s| !(s.is_nan() || **s > 0.0)) { return Err(format!("spacing {s}")); }
        Ok(())
    }

    /// The level slots in order: the boundary, B_1..=B_R, L_0..L_R, T_0..T_R, then N.
    pub fn levels(&self) -> Result<S1bLevels, String> {
        self.check()?;
        let r = self.rounds;
        let mut ids = vec![self.boundary];
        ids.extend((1..=r).map(|k| boundary_level(k, self.boundary)));
        ids.extend((0..r).map(loop_level));
        ids.extend((0..r).map(thick_level));
        if !self.narrow.is_empty() { ids.push(NARROW_LEVEL); }
        let mut values = Vec::with_capacity(ids.len() * self.spacing.len());
        for (p, &s) in self.spacing.iter().enumerate() {
            values.push(if s.is_nan() { f32::NAN } else { 0.0 });
            values.extend((1..=r).map(|k| -(k as f32) * s * GRID));
            values.extend((0..r).map(|k| -(k as f32 + 0.5) * s * GRID));
            values.extend((0..r).map(|k| -((k as f32 + 1.25) * s + THICK_MARGIN) * GRID));
            if let Some(&n) = self.narrow.get(p) { values.push(if s.is_nan() { f32::NAN } else { -n * GRID }); }
        }
        Ok(S1bLevels { ids, values })
    }

    /// (level, reference level) for the chain's reference table.
    pub fn reference(&self) -> Vec<(u32, u32)> {
        let b = |k| boundary_level(k, self.boundary);
        let mut out: Vec<(u32, u32)> = (1..=self.rounds).map(|k| (b(k), b(k - 1))).collect();
        out.extend((0..self.rounds).flat_map(|k| [(loop_level(k), b(k)), (thick_level(k), b(k))]));
        if !self.narrow.is_empty() { out.push((NARROW_LEVEL, self.boundary)); }
        out
    }

    /// The inside depth the field needs (grid units): the deepest level plus [`DEPTH_MARGIN`].
    pub fn field_depth(&self) -> f32 {
        let r = self.rounds as f32;
        let deepest = self.spacing.iter().enumerate().filter(|(_, s)| !s.is_nan()).map(|(p, &s)| {
            let n = self.narrow.get(p).copied().unwrap_or(0.0);
            (r * s).max((r + 0.25) * s + THICK_MARGIN).max(n)
        }).fold(0.0f32, f32::max);
        deepest * GRID + DEPTH_MARGIN
    }
}

/// The chain buffers S1b binds by name.
#[derive(Clone, Copy, Debug)]
pub struct S1bInputs {
    /// (root piece, pieces, -, -) per loop.
    pub loops: BufRef,
    /// (plane, level, island, flags: LOOP_HOLE, LOOP_OPEN).
    pub loop_meta: BufRef,
    /// (parent at the same level, island, the reference contour, -).
    pub loop_tree: BufRef,
    /// (plane, level, feature, cell) per piece.
    pub piece_info: BufRef,
}

/// S1b's output: the readback buffer and its bytes (the item's [`Step::Output`]).
#[derive(Clone, Copy, Debug)]
pub struct S1bOut { pub buf: BufRef, pub bytes: u64 }

/// S1b's steps over a chain of the run's levels, and its readback. Records are placed by a
/// prefix sum in loop order, so the bytes never depend on job sizes.
pub fn steps(k: &S1bKernels, scans: &ScanKernels, pool: &mut Pool, inputs: &S1bInputs, spec: &S1bSpec) -> Result<(Vec<Step>, S1bOut), JobError> {
    spec.check().map_err(|e| JobError::Failed(format!("S1b: {e}")))?;
    let n = spec.loops.bound().max(1);
    let total = pool.claim("s1b.total", 1)?;
    let (known, word) = spec.loops.words();
    let flags = if spec.narrow.is_empty() { 0 } else { FLAG_NARROW } | if spec.skip_holed { FLAG_SKIP_HOLED } else { 0 };
    let header: [u32; HEADER as usize] = [known, word, n, spec.boundary, spec.rounds, flags, total.0, LOOP_BASE, BOUNDARY_BASE, THICK_BASE, NARROW_LEVEL, 0];
    let params = pool.buffer("s1b params", HEADER as u64 * 4);
    pool.upload(params, 0, header.iter().flat_map(|w| w.to_le_bytes()).collect());
    let mark = pool.buffer("s1b mark", n as u64 * 12);
    let state = pool.buffer("s1b state", n as u64 * 16);
    let count = pool.buffer("s1b count", n as u64 * 4);
    let scan = pool.buffer("s1b scan", n as u64 * 4);
    let out_words = OUT_HEADER as u64 + n as u64 * 4;
    let out = pool.readback("s1b out", out_words * 4);
    let bind = |name: LiveId| -> BufRef {
        [(id!(s1b), params), (id!(s1b_mark), mark), (id!(s1b_state), state), (id!(s1b_count), count), (id!(s1b_scan), scan),
            (id!(s1b_out), out), (id!(loops), inputs.loops), (id!(loop_meta), inputs.loop_meta), (id!(loop_tree), inputs.loop_tree),
            (id!(piece_info), inputs.piece_info), (id!(counts), BufRef::COUNTS)]
            .into_iter().find(|b| b.0 == name).map_or(BufRef::RUNTIME, |b| b.1)
    };
    let pass = |name: &str, cost: u32| Step::Pass(Pass::new(k.get(name), label(name), Elements::Fixed(n), cost, 1 << 20, bind));
    let mut steps = vec![pass("S1bClear", 1), pass("S1bMark", 2), pass("S1bState", 4 + 2 * spec.rounds)];
    steps.extend(scans.exclusive(pool, count, scan, n, total));
    steps.push(pass("S1bFill", 1));
    Ok((steps, S1bOut { buf: out, bytes: out_words * 4 }))
}

/// A component's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum State {
    /// Its loops are on the GPU, and its children are filled next.
    Wide,
    /// The CPU's concentric_solid fills it (contour and holes).
    Handoff,
    /// Round 0 with the narrow test: it holds the narrow level, so it is not concentric.
    NotNarrow,
    /// Round 0 of a band run with a hole: the CPU's Round loop.
    Holed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind { Component(State), Loop, Hole }

/// One `s1b_out` record (loop ids are the chain's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Record { pub loop_id: u32, pub kind: Kind, pub round: u32, pub plane: u32, pub link: u32, pub source: u32 }

/// The records of an `s1b_out` readback, in loop order.
pub fn decode(words: &[u32]) -> Result<Vec<Record>, String> {
    let head = words.get(..OUT_HEADER as usize).ok_or_else(|| format!("s1b_out: {} words", words.len()))?;
    let (n, bound) = (head[0] as usize, head[1] as usize);
    if n > bound { return Err(format!("s1b_out: {n} records over the loop bound {bound}")); }
    let body = words.get(OUT_HEADER as usize..OUT_HEADER as usize + n * 4).ok_or("s1b_out: records past the buffer")?;
    body.chunks_exact(4).map(|w| {
        let state = [State::Wide, State::Handoff, State::NotNarrow, State::Holed][(w[1] >> 2 & 3) as usize];
        let kind = match w[1] & 3 {
            0 => Kind::Component(state),
            1 => Kind::Loop,
            2 => Kind::Hole,
            k => return Err(format!("s1b_out: loop {} has kind {k}", w[0])),
        };
        Ok(Record { loop_id: w[0], kind, round: w[1] >> 8 & 0xff, plane: w[1] >> 16, link: w[2], source: w[3] })
    }).collect()
}

/// One step of a component's concentric fill, in fill order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    /// A loop at -(round + 1/2) s0 (the bead of width_for_spacing(s0)).
    Loop { loop_id: u32, round: u32 },
    /// A component the CPU's concentric_solid fills: its contour and holes (chain loop ids).
    Handoff { contour: u32, holes: Vec<u32>, round: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fill { Concentric(Vec<Item>), NotNarrow, Holed }

/// A region component (round 0): its contour loop, plane, FeatureSet source seg, and fill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region { pub contour: u32, pub plane: u32, pub source: u32, pub fill: Fill }

/// The region components of decoded records, by contour loop id. A wide component's fill is
/// its loops (in loop order), then each child's fill depth first (children in loop order), as
/// the CPU fills a component and then its pending children. Links are checked: every record
/// must hang off a component of the right state, round and plane.
pub fn assemble(records: &[Record]) -> Result<Vec<Region>, String> {
    struct Comp { state: State, round: u32, plane: u32, source: u32, loops: Vec<(u32, u32)>, holes: Vec<u32>, children: Vec<u32> }
    let mut comps: BTreeMap<u32, Comp> = BTreeMap::new();
    for r in records {
        if let Kind::Component(state) = r.kind {
            let c = Comp { state, round: r.round, plane: r.plane, source: r.source, loops: vec![], holes: vec![], children: vec![] };
            if comps.insert(r.loop_id, c).is_some() { return Err(format!("loop {} recorded twice", r.loop_id)); }
        }
    }
    let mut roots = vec![];
    for r in records {
        let at = |comps: &BTreeMap<u32, Comp>, want: State, round: u32| -> Result<(), String> {
            match comps.get(&r.link) {
                Some(c) if c.state == want && c.round == round && c.plane == r.plane => Ok(()),
                _ => Err(format!("{r:?}: link {} is no {want:?} component of round {round}, plane {}", r.link, r.plane)),
            }
        };
        match r.kind {
            Kind::Component(_) if r.round == 0 => {
                if r.link != NONE { return Err(format!("{r:?}: a round-0 component with a link")); }
                roots.push(r.loop_id);
            }
            Kind::Component(state) => {
                if matches!(state, State::NotNarrow | State::Holed) { return Err(format!("{r:?}: {state:?} past round 0")); }
                at(&comps, State::Wide, r.round - 1)?;
                comps.get_mut(&r.link).unwrap().children.push(r.loop_id);
            }
            Kind::Loop => {
                at(&comps, State::Wide, r.round)?;
                comps.get_mut(&r.link).unwrap().loops.push((r.loop_id, r.round));
            }
            Kind::Hole => {
                at(&comps, State::Handoff, r.round)?;
                comps.get_mut(&r.link).unwrap().holes.push(r.loop_id);
            }
        }
    }
    fn walk(comps: &BTreeMap<u32, Comp>, c: u32, out: &mut Vec<Item>) {
        let comp = &comps[&c];
        match comp.state {
            State::Wide => {
                out.extend(comp.loops.iter().map(|&(loop_id, round)| Item::Loop { loop_id, round }));
                for &child in &comp.children { walk(comps, child, out); }
            }
            _ => out.push(Item::Handoff { contour: c, holes: comp.holes.clone(), round: comp.round }),
        }
    }
    Ok(roots.into_iter().map(|c| {
        let comp = &comps[&c];
        let fill = match comp.state {
            State::NotNarrow => Fill::NotNarrow,
            State::Holed => Fill::Holed,
            _ => { let mut items = vec![]; walk(&comps, c, &mut items); Fill::Concentric(items) }
        };
        Region { contour: c, plane: comp.plane, source: comp.source, fill }
    }).collect())
}
