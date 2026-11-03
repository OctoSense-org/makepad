//! The nearest-feature field of a FeatureSet (GPU-EXEC §10.3), generic over
//! the outline and region loop sets.
//!
//! Features are a set's segs (perpendicular distance inside their slab) and
//! the start vertex of every seg that is reflex for one side (the wedge
//! between its two normals): `VERTEX_BIT | s` names seg `s`'s start vertex
//! as a whole (sub 0 of `plane_abi::vertex_feature`; pieces name the join
//! part). The metric per field is the wall join's:
//! - Round: Euclidean distance to the vertex;
//! - Miter(limit): max(l1, l2) over the two edge lines. Past the miter limit
//!   (1 + cos θ <= 2 / limit²) the corner is squared:
//!   max(l1, l2, u·(p − v)), with u the unit bisector.
//!
//! A point's side comes from the set itself, never from a program value:
//! coarse-centre parity along each centre row, then parity along an L path
//! inside the coarse cell. Every query point is perturbed to (x + ε, y + ε²),
//! so a vertex on a grid line has one consistent side. Inside points take
//! only inside features and outside points only outside ones. The value is
//! v = m + b inside and v = m − b outside (b = the loop's bias, + grows), and
//! the biased signed distance is d' = −v inside, +v outside.
//!
//! Passes, in order, with their element counts (P planes, n = seg count):
//! 1. `FieldClear` P·nx·ny: marks, coverage, this stage's count words;
//! 2. `FieldSegMeta` n: per seg `seg_geo` = (a, unit direction), (length, bias, cos of the
//!    turn at its start, 0) and `seg_meta` = (plane | flags << 24, next, gx | gy << 16, prev);
//! 3. `FieldDomain` n twice (`dispatch_range.z` = 0 segs, 1 vertices): marks the coarse cells near
//!    each feature's region, and sums the cand bound;
//! 4. the deterministic exclusive scan of `band_mark` into `band_slot`;
//! 5. `FieldBandList` P·nx·ny: `band`, `band_off`, `band_slot` (NONE off the band) and
//!    the band count words;
//! 6. `FieldCoverCount` n and 7. `FieldCoverRows` P·ny: `cover`, inside per coarse centre;
//! — FIELD checkpoint: band slots and the cand bound size `nf`, `node_side`, `cand`;
//! 8. `FieldNodeSide` slots·121: every band node's side; clears its nf;
//! 9. `FieldNearestDist` n twice: atomic_min of the value's monotone u32 key (all bits
//!    flipped if negative, else only the sign bit), into every band copy of each node;
//! 10. `FieldNearestId` n twice: atomic_min of the feature id among exact ties;
//! 11. `FieldNearestFinish` slots·121: `nf` = (feature, d'), or (NONE, ±BIG) where no
//!     feature reaches;
//! 12. `FieldCellBound` slots·100: per fine cell the cand threshold T and its side
//!     (0 outside, 1 inside, 2 both);
//! 13. `FieldCandCount` n twice: per cell counts in `cand_cursor`; 14. the exclusive scan
//!     of `cand_cursor` (slots·100 + 1 values, the last 0) into `cand_off`;
//! 15. `FieldCandFill` n twice, counting `cand_cursor` back down; 16. `FieldCandSort`
//!     slots·100: each cell's list ascending, so its bytes never depend on the fill's order.
//!
//! [`steps`] lays these out as gpu_jobs steps: the scans are the runtime's, and one host
//! step at the FIELD checkpoint sizes the band's buffers from the counts.
//!
//! cand(cell) = {f : lb(centre, f) <= T}, with lb the lower bound of f's value
//! over the disk of radius √2/2 around the centre. T = min over the corners'
//! v + √2 + 1e-3 (the cell's diagonal and the slack), or
//! max(that min, max |bias|) + √2 + 1e-3 where the outline crosses the cell.
//! So two neighbouring cells see the same minimal features on their shared
//! edge, each from its own list. nf, cand and field_at are valid where
//! |d'| <= depth.
use makepad_draw::*;
use crate::feature_set::FeatureSet;
use crate::gpu_jobs::{label, BufRef, CountWord, Counts, Elements, JobError, Pass, Pool, ScanKernels, Step};

/// The field's count words, from `field[2].x` in the batch's counts buffer.
pub const WORD_CAND_BOUND: u32 = 0;
/// Nodes inside a feature's region with no band copy (a marking defect; must stay 0).
pub const WORD_ORPHAN_NODES: u32 = 1;
/// Cand entries past the allocated capacity (must stay 0).
pub const WORD_CAND_OVERFLOW: u32 = 2;
pub const WORD_BAND_SLOTS: u32 = 3;
pub const WORD_BAND_NODES: u32 = 4;
pub const WORD_BAND_CELLS: u32 = 5;
/// The cand entries the lists hold (the scan's total).
pub const WORD_CAND_TOTAL: u32 = 6;
pub const WORDS: u32 = 7;

/// The distance stored where no feature reaches a band node (with its side's sign).
pub const BIG: f32 = 1.0e9;
/// Overlap of neighbouring features' domains, in grid units, so no node falls between a
/// slab and the vertex wedge beside it by rounding.
pub const DOMAIN_SLACK: f32 = 1.0e-4;
/// The cand radius: half a fine cell's diagonal, rounded up.
pub const CELL_RADIUS: f32 = 0.7072;
/// The cand slack, in grid units of the field's metric.
pub const CAND_EPS: f32 = 1.0e-3;

/// The kernels in pass order.
pub const KERNELS: [&str; 15] = [
    "FieldClear", "FieldSegMeta", "FieldDomain", "FieldBandList", "FieldCoverCount", "FieldCoverRows",
    "FieldNodeSide", "FieldNearestDist", "FieldNearestId", "FieldNearestFinish", "FieldCellBound",
    "FieldCandCount", "FieldCandFill", "FieldCandSort", "FieldProbe",
];

/// The join metric a field measures in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldJoin { Round, Miter(f32) }

/// What every field pass reads from its `field` buffer, but the batch's words.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldParams {
    pub nx: u32,
    pub ny: u32,
    pub planes: u32,
    /// Grid units (0.1 mm = 1).
    pub depth_in: f32,
    pub depth_out: f32,
    pub join: FieldJoin,
    /// The largest |loop bias| in the set, grid units.
    pub bias_max: f32,
}

impl FieldParams {
    pub fn grid(&self) -> [u32; 4] { [self.nx, self.ny, self.planes, 0] }
    pub fn depth(&self) -> [f32; 4] {
        let limit = match self.join { FieldJoin::Round => 0.0, FieldJoin::Miter(l) => l };
        [self.depth_in, self.depth_out, limit, self.bias_max]
    }
    /// The `field` buffer: grid, depth (f32 bits), then the counts base word and
    /// the cand capacity.
    pub fn words(&self, counts_base: u32, cand_capacity: u32) -> [u32; 12] {
        let g = self.grid();
        let d = self.depth().map(f32::to_bits);
        [g[0], g[1], g[2], g[3], d[0], d[1], d[2], d[3], counts_base, cand_capacity, 0, 0]
    }
}

/// The field's kernels (with the runtime's scan), made once per Cx.
#[derive(Clone)]
pub struct FieldKernels { kernels: Vec<ComputeKernel>, scan: ScanKernels }

#[derive(Clone, Copy)]
enum K {
    Clear, SegMeta, Domain, BandList, CoverCount, CoverRows, NodeSide, NearestDist, NearestId,
    NearestFinish, CellBound, CandCount, CandFill, CandSort, Probe,
}

impl FieldKernels {
    pub fn new(cx: &mut Cx) -> Result<Self, String> {
        let kernels = cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            KERNELS.iter().map(|name| {
                let object = vm.value(module, LiveId::from_str(name)).as_object().ok_or_else(|| format!("{name} kernel missing"))?;
                ComputeKernel::new(vm, object, name)
            }).collect::<Result<Vec<_>, String>>()
        })?;
        Ok(Self { kernels, scan: ScanKernels::new(cx)? })
    }
    fn kernel(&self, k: K) -> &ComputeKernel { &self.kernels[k as usize] }
}

/// Per pass: its label, work units per element (the calibration's first estimate) and its job
/// ceiling. A feature's walk visits at most its region at the asked depth, about (2 · reach)²
/// nodes for a vertex and 2 · reach per unit of seg length, so 8192 features stay within 50 ms
/// for depths up to 64 grid units; a node, cell or row element does a fixed small walk.
fn shape(k: K, which: u32) -> (&'static str, u32, u32) {
    let vertices = which == 1;
    match k {
        K::Clear => ("field clear", 2, 1 << 20),
        K::SegMeta => ("field seg meta", 96, 1 << 18),
        K::Domain => (if vertices { "field domain vertices" } else { "field domain segs" }, 256, 8192),
        K::BandList => ("field band list", 4, 1 << 20),
        K::CoverCount => ("field cover count", 16, 1 << 18),
        K::CoverRows => ("field cover rows", 512, 1 << 16),
        K::NodeSide => ("field node side", 64, 1 << 18),
        K::NearestDist => (if vertices { "field nearest dist vertices" } else { "field nearest dist segs" }, 1024, 8192),
        K::NearestId => (if vertices { "field nearest id vertices" } else { "field nearest id segs" }, 1024, 8192),
        K::NearestFinish => ("field nearest finish", 4, 1 << 20),
        K::CellBound => ("field cell bound", 48, 1 << 18),
        K::CandCount => (if vertices { "field cand count vertices" } else { "field cand count segs" }, 512, 8192),
        K::CandFill => (if vertices { "field cand fill vertices" } else { "field cand fill segs" }, 512, 8192),
        K::CandSort => ("field cand sort", 256, 1 << 16),
        K::Probe => ("field probe", 256, 1 << 16),
    }
}

fn le(words: impl IntoIterator<Item = u32>) -> Vec<u8> { words.into_iter().flat_map(u32::to_le_bytes).collect() }

/// The field buffer: the params' words, then field[2].z the seg count word (NONE when fixed) and
/// field[2].w the seg count (the clear pass copies a counted one in).
fn field_words(params: FieldParams, words: CountWord, cand_capacity: u32, segs: SegCount, count: u32) -> [u32; 12] {
    let mut w = params.words(words.0, cand_capacity);
    (w[10], w[11]) = match segs { SegCount::Fixed(n) => (u32::MAX, n), SegCount::Counted { word, .. } => (word.0, count) };
    w
}

/// A FeatureSet in an item's pool, and its seg count.
#[derive(Clone, Copy, Debug)]
pub struct FieldInput {
    pub seg: BufRef,
    pub seg_info: BufRef,
    pub seg_key: BufRef,
    pub loop_off: BufRef,
    pub loop_info: BufRef,
    /// Per loop: its bias as f32 bits (grid units, + grows the part).
    pub loop_bias: BufRef,
    pub set_slot: BufRef,
    pub set_kept: BufRef,
    pub set_kept_off: BufRef,
    pub seg_off: BufRef,
    pub segs: SegCount,
}

/// A set's seg count: fixed (a set traced on the cores), or a count word with the capacity its
/// buffers hold (a set made on the GPU earlier in the item).
#[derive(Clone, Copy, Debug)]
pub enum SegCount {
    Fixed(u32),
    Counted { word: CountWord, capacity: u32 },
}

impl SegCount {
    /// The segs the set's buffers hold room for.
    pub fn capacity(&self) -> u32 { match *self { Self::Fixed(n) => n, Self::Counted { capacity, .. } => capacity } }
    fn elements(&self) -> Elements {
        match *self { Self::Fixed(n) => Elements::Fixed(n), Self::Counted { word, capacity } => Elements::Counted { word, bound: capacity } }
    }
}

impl FieldInput {
    /// A set traced on the cores, uploaded into `pool`, with a bias per loop.
    pub fn upload(pool: &mut Pool, set: &FeatureSet, loop_bias: &[f32]) -> Result<Self, JobError> {
        if loop_bias.len() != set.loop_info.len() { return Err(JobError::Failed(format!("{} biases for {} loops", loop_bias.len(), set.loop_info.len()))); }
        let mut put = |name: &'static str, bytes: Vec<u8>| {
            let buf = pool.buffer(name, bytes.len() as u64);
            if !bytes.is_empty() { pool.upload(buf, 0, bytes); }
            buf
        };
        Ok(Self {
            seg: put("set seg", le(set.seg.iter().flatten().map(|v| v.to_bits()))),
            seg_info: put("set seg_info", le(set.seg_info.iter().flatten().copied())),
            seg_key: put("set seg_key", le(set.seg_key.iter().copied())),
            loop_off: put("set loop_off", le(set.loop_off.iter().copied())),
            loop_info: put("set loop_info", le(set.loop_info.iter().flatten().copied())),
            loop_bias: put("set loop_bias", le(loop_bias.iter().map(|b| b.to_bits()))),
            set_slot: put("set slot", le(set.set_slot.iter().copied())),
            set_kept: put("set kept", le(set.set_kept.iter().copied())),
            set_kept_off: put("set kept_off", le(set.set_kept_off.iter().copied())),
            seg_off: put("set seg_off", le(set.seg_off.iter().copied())),
            segs: SegCount::Fixed(set.seg.len() as u32),
        })
    }
}

/// The field's buffers once its steps ran, for the stages that read it (wall pieces, the
/// region stages), and its sizes.
#[derive(Clone, Copy, Debug)]
pub struct FieldOut {
    pub input: FieldInput,
    pub params: FieldParams,
    pub field: BufRef,
    pub seg_geo: BufRef,
    pub seg_meta: BufRef,
    pub band_mark: BufRef,
    pub band_slot: BufRef,
    pub band: BufRef,
    pub band_off: BufRef,
    pub cover: BufRef,
    pub node_side: BufRef,
    pub nf_key: BufRef,
    pub nf: BufRef,
    pub cell_t: BufRef,
    pub cell_side: BufRef,
    pub cand_off: BufRef,
    pub cand_cursor: BufRef,
    pub cand: BufRef,
    /// The first of the field's [`WORDS`] counts words.
    pub words: CountWord,
    /// Band coarse cells over all planes, and the cand entries allocated.
    pub slots: u32,
    pub cand_bound: u32,
}

impl FieldOut {
    /// The buffer a field kernel's binding names, if the field or its set has it.
    pub fn bind(&self, name: LiveId) -> Option<BufRef> {
        let i = &self.input;
        [(id!(seg), i.seg), (id!(seg_info), i.seg_info), (id!(seg_key), i.seg_key), (id!(loop_off), i.loop_off), (id!(loop_info), i.loop_info),
            (id!(loop_bias), i.loop_bias), (id!(set_slot), i.set_slot), (id!(set_kept), i.set_kept), (id!(set_kept_off), i.set_kept_off),
            (id!(seg_off), i.seg_off), (id!(field), self.field), (id!(seg_geo), self.seg_geo), (id!(seg_meta), self.seg_meta),
            (id!(band_mark), self.band_mark), (id!(band_slot), self.band_slot), (id!(band), self.band), (id!(band_off), self.band_off),
            (id!(cover), self.cover), (id!(node_side), self.node_side), (id!(nf_key), self.nf_key), (id!(nf), self.nf),
            (id!(cell_t), self.cell_t), (id!(cell_side), self.cell_side), (id!(cand_off), self.cand_off),
            (id!(cand_cursor), self.cand_cursor), (id!(cand), self.cand), (id!(counts), BufRef::COUNTS)]
            .into_iter().find(|b| b.0 == name).map(|b| b.1)
    }
    /// The field's invariants in counts read after its last pass.
    pub fn verify(&self, counts: &Counts) -> Result<(), JobError> {
        let word = |w: u32| counts.word(CountWord(self.words.0 + w));
        if word(WORD_ORPHAN_NODES) != 0 { return Err(JobError::Failed(format!("feature field: {} region nodes without a band copy", word(WORD_ORPHAN_NODES)))); }
        if word(WORD_CAND_OVERFLOW) != 0 { return Err(JobError::Failed(format!("feature field: {} cand entries past {}", word(WORD_CAND_OVERFLOW), self.cand_bound))); }
        if word(WORD_CAND_TOTAL) > self.cand_bound { return Err(JobError::Failed(format!("feature field: cand total {} over its bound {}", word(WORD_CAND_TOTAL), self.cand_bound))); }
        Ok(())
    }
    /// `field_feature` at `count` points of `probes` (vec4f x, y, plane bits, 0) into
    /// `probe_out` ((feature, d' bits) per point).
    pub fn probe(&self, k: &FieldKernels, probes: BufRef, probe_out: BufRef, count: u32) -> Step {
        let (name, cost, ceiling) = shape(K::Probe, 0);
        Step::Pass(Pass::new(k.kernel(K::Probe), label(name), Elements::Fixed(count), cost, ceiling,
            |b| if b == id!(probes) { probes } else if b == id!(probe_out) { probe_out } else { self.bind(b).unwrap_or(BufRef::RUNTIME) }))
    }
}

/// The field's counts words, claimed before its item starts.
pub fn claim(pool: &mut Pool) -> Result<CountWord, JobError> { pool.claim("feature field", WORDS) }

/// The steps of `input`'s field in `pool`, counting in `words` (from [`claim`]): the passes up
/// to the FIELD checkpoint, a host step that sizes the band's buffers from their counts, the
/// rest of the passes, then `then`'s steps (the field's readers, given its buffers and the
/// FIELD counts).
pub fn steps(k: &FieldKernels, pool: &mut Pool, words: CountWord, input: FieldInput, params: FieldParams,
    then: impl FnOnce(&Counts, &mut Pool, &FieldOut) -> Result<Vec<Step>, JobError> + 'static) -> Result<Vec<Step>, JobError> {
    let cells = params.nx as u64 * params.ny as u64 * params.planes as u64;
    if cells == 0 || cells * 4 > u32::MAX as u64 { return Err(JobError::Failed(format!("feature field: {cells} coarse cells"))); }
    let (cells, n, segs) = (cells as u32, input.segs.capacity(), input.segs.elements());
    let field = pool.buffer("field params", 48);
    pool.upload(field, 0, le(field_words(params, words, 0, input.segs, n)));
    let none = BufRef::RUNTIME;
    let mut out = FieldOut {
        input, params, field, words, slots: 0, cand_bound: 0,
        seg_geo: pool.buffer("field seg_geo", n as u64 * 32),
        seg_meta: pool.buffer("field seg_meta", n as u64 * 16),
        band_mark: pool.buffer("field band_mark", cells as u64 * 4),
        band_slot: pool.buffer("field band_slot", cells as u64 * 4),
        band_off: pool.buffer("field band_off", (params.planes as u64 + 1) * 4),
        cover: pool.buffer("field cover", cells as u64 * 4),
        band: none, node_side: none, nf_key: none, nf: none, cell_t: none, cell_side: none, cand_off: none, cand_cursor: none, cand: none,
    };
    let pass = |k: &FieldKernels, out: &FieldOut, kind: K, elements: Elements, which: u32| {
        let (name, cost, ceiling) = shape(kind, which);
        Step::Pass(Pass::new(k.kernel(kind), label(name), elements, cost, ceiling, |b| out.bind(b).unwrap_or(BufRef::RUNTIME)).zw(which, 0))
    };
    let mut steps = vec![
        pass(k, &out, K::Clear, Elements::Fixed(cells), 0),
        pass(k, &out, K::SegMeta, segs, 0),
        pass(k, &out, K::Domain, segs, 0),
        pass(k, &out, K::Domain, segs, 1),
        pass(k, &out, K::CoverCount, segs, 0),
        pass(k, &out, K::CoverRows, Elements::Fixed(params.ny * params.planes), 0),
    ];
    steps.extend(k.scan.exclusive(pool, out.band_mark, out.band_slot, cells, CountWord(words.0 + WORD_BAND_SLOTS)));
    let k = k.clone();
    steps.push(Step::Host(Box::new(move |counts: &Counts, pool: &mut Pool| {
        let slots = counts.word(CountWord(words.0 + WORD_BAND_SLOTS));
        let cand_bound = counts.word(CountWord(words.0 + WORD_CAND_BOUND));
        if slots > cells { return Err(JobError::Failed(format!("feature field: {slots} band slots of {cells} cells"))); }
        let (s, fine) = (slots as u64, slots as u64 * 100);
        out.slots = slots;
        out.cand_bound = cand_bound;
        out.band = pool.buffer("field band", s * 4);
        out.node_side = pool.buffer("field node_side", s * 121 * 4);
        out.nf_key = pool.buffer("field nf_key", s * 121 * 4);
        out.nf = pool.buffer("field nf", s * 121 * 8);
        out.cell_t = pool.buffer("field cell_t", fine * 4);
        out.cell_side = pool.buffer("field cell_side", fine * 4);
        out.cand_off = pool.buffer("field cand_off", (fine + 1) * 4);
        out.cand_cursor = pool.buffer("field cand_cursor", (fine + 1) * 4);
        out.cand = pool.buffer("field cand", cand_bound as u64 * 4);
        let count = match input.segs { SegCount::Fixed(n) => n, SegCount::Counted { word, .. } => counts.word(word) };
        if count > n { return Err(JobError::Failed(format!("feature field: {count} segs over their capacity {n}"))); }
        pool.upload(out.field, 0, le(field_words(params, words, cand_bound, input.segs, count)));
        let (nodes, fine) = (slots * 121, slots * 100);
        let mut steps = vec![
            pass(&k, &out, K::BandList, Elements::Fixed(cells), 0),
            pass(&k, &out, K::NodeSide, Elements::Fixed(nodes), 0),
            pass(&k, &out, K::NearestDist, segs, 0),
            pass(&k, &out, K::NearestDist, segs, 1),
            pass(&k, &out, K::NearestId, segs, 0),
            pass(&k, &out, K::NearestId, segs, 1),
            pass(&k, &out, K::NearestFinish, Elements::Fixed(nodes), 0),
            pass(&k, &out, K::CellBound, Elements::Fixed(fine), 0),
            pass(&k, &out, K::CandCount, segs, 0),
            pass(&k, &out, K::CandCount, segs, 1),
        ];
        steps.extend(k.scan.exclusive(pool, out.cand_cursor, out.cand_off, fine + 1, CountWord(words.0 + WORD_CAND_TOTAL)));
        steps.push(pass(&k, &out, K::CandFill, segs, 0));
        steps.push(pass(&k, &out, K::CandFill, segs, 1));
        steps.push(pass(&k, &out, K::CandSort, Elements::Fixed(fine), 0));
        steps.extend(then(counts, pool, &out)?);
        Ok(steps)
    })));
    Ok(steps)
}

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.Field = {
        // Every pass that counts and the pass that fills evaluate the same
        // expressions; a precise module fuses and reorders nothing.
        precise_math: true
        workgroup_size: 64
        seg: shader.buffer(vec4f)
        seg_info: shader.buffer(vec4u)
        loop_off: shader.buffer(u32)
        loop_info: shader.buffer(vec4u)
        loop_bias: shader.buffer(u32)
        set_slot: shader.buffer(u32)
        set_kept: shader.buffer(u32)
        set_kept_off: shader.buffer(u32)
        seg_off: shader.buffer(u32)
        seg_geo: shader.buffer(vec4f)
        seg_meta: shader.buffer(vec4u)
        band_mark: shader.buffer(u32)
        band_slot: shader.buffer(u32)
        band: shader.buffer(u32)
        band_off: shader.buffer(u32)
        cover: shader.buffer(u32)
        node_side: shader.buffer(u32)
        nf_key: shader.buffer(u32)
        nf: shader.buffer(u32)
        cell_t: shader.buffer(u32)
        cell_side: shader.buffer(u32)
        cand_off: shader.buffer(u32)
        cand_cursor: shader.buffer(u32)
        cand: shader.buffer(u32)
        // field[0]: nx, ny, planes, 0; [1]: f32 bits of depth in, out, miter limit
        // (0 = round), max |bias|; [2]: counts base word, cand capacity, the seg count word
        // (NONE: fixed), the seg count.
        // A buffer, not params: the push block holds 8 bytes per buffer and 128 in all.
        field: shader.buffer(vec4u)
        // .z: 0 for segs, 1 for vertices
        dispatch_range: shader.uniform(vec4u)
        index: fn() -> u32 {
            let i = global_id().x
            if i >= self.dispatch_range.y { return 4294967295u }
            return self.dispatch_range.x + i
        }
        cells: fn() -> u32 { return self.field[0u].x * self.field[0u].y }
        // The set's seg count (a counted set's, copied in by the clear pass).
        segs: fn() -> u32 { return self.field[2u].w }
        nodes_x: fn() -> u32 { return self.field[0u].x * 10u + 1u }
        nodes_y: fn() -> u32 { return self.field[0u].y * 10u + 1u }
        miter: fn() -> bool { return asfloat(self.field[1u].z) > 0.0 }
        left: fn(e: vec2) -> vec2 { return vec2(0.0 - e.y, e.x) }
        direction: fn(a: vec2, b: vec2) -> vec2 {
            let l = length(b - a)
            if l > 0.0 { return (b - a) / l }
            return vec2(1.0, 0.0)
        }
        // Largest i in [0, n) with list[i] <= v (0 when none).
        search: fn(v: u32, n: u32, which: u32) -> u32 {
            var lo = 0u
            var hi = n
            for bit in 0..32u {
                if hi - lo <= 1u { break }
                let mid = lo + (hi - lo) / 2u
                var at = 0u
                if which == 0u { at = self.seg_off[mid] }
                else if which == 1u { at = self.set_kept_off[mid] }
                else if which == 2u { at = self.loop_info[mid].x }
                else { at = self.band_off[mid] }
                if at <= v { lo = mid } else { hi = mid }
            }
            return lo
        }
        band_plane: fn(slot: u32) -> u32 { return self.search(slot, self.field[0u].z, 3u) }
        // The set's segs in fine cell (fx, fy) of plane p, as [first, end).
        cell_segs: fn(p: u32, fx: u32, fy: u32) -> vec2u {
            if fx + 1u >= self.nodes_x() || fy + 1u >= self.nodes_y() { return vec2u(0u, 0u) }
            let slot = self.set_slot[p * self.cells() + (fy / 10u) * self.field[0u].x + fx / 10u]
            if slot == 4294967295u { return vec2u(0u, 0u) }
            let at = slot * 100u + (fy % 10u) * 10u + fx % 10u
            return vec2u(self.seg_off[at], self.seg_off[at + 1u])
        }
        seg_a: fn(s: u32) -> vec2 { return self.seg_geo[s * 2u].xy }
        seg_e: fn(s: u32) -> vec2 { return self.seg_geo[s * 2u].zw }
        seg_len: fn(s: u32) -> f32 { return self.seg_geo[s * 2u + 1u].x }
        seg_bias: fn(s: u32) -> f32 { return self.seg_geo[s * 2u + 1u].y }
        seg_cos: fn(s: u32) -> f32 { return self.seg_geo[s * 2u + 1u].z }
        seg_plane: fn(s: u32) -> u32 { return self.seg_meta[s].x & 16777215u }
        // flags: 1 start vertex reflex inside, 2 reflex outside, 4 squared, 8 nonzero length.
        seg_flags: fn(s: u32) -> u32 { return self.seg_meta[s].x >> 24u }
        seg_prev: fn(s: u32) -> u32 { return self.seg_meta[s].w }
        seg_next: fn(s: u32) -> u32 { return self.seg_meta[s].y }
        vertex_inside: fn(s: u32) -> bool { return (self.seg_flags(s) & 1u) != 0u }
        // The two edge normals pointing into the vertex's reflex side.
        vertex_m1: fn(s: u32) -> vec2 {
            let m = self.left(self.seg_e(self.seg_prev(s)))
            if self.vertex_inside(s) { return m }
            return vec2(0.0) - m
        }
        vertex_m2: fn(s: u32) -> vec2 {
            let m = self.left(self.seg_e(s))
            if self.vertex_inside(s) { return m }
            return vec2(0.0) - m
        }
        // The unit bisector of the wedge, stable up to a full turn back.
        vertex_u: fn(s: u32) -> vec2 { return normalize(self.seg_e(self.seg_prev(s)) - self.seg_e(s)) }
        // The farthest point of the vertex's region at metric depth d.
        vertex_reach: fn(s: u32, d: f32) -> f32 {
            if !self.miter() { return d }
            let c = sqrt(max(0.5 * (1.0 + self.seg_cos(s)), 0.0))
            return d / max(c, 1.0 / asfloat(self.field[1u].z))
        }
        vertex_depth: fn(s: u32, din: f32, dout: f32) -> f32 {
            if self.vertex_inside(s) { return din }
            return dout
        }
        vertex_metric: fn(s: u32, d: vec2) -> f32 {
            if !self.miter() { return length(d) }
            var m = max(dot(d, self.vertex_m1(s)), dot(d, self.vertex_m2(s)))
            if (self.seg_flags(s) & 4u) != 0u { m = max(m, dot(d, self.vertex_u(s))) }
            return m
        }
        // The half-plane w . (dx, dy) <= k on the row at dy, on a span of dx.
        clip: fn(span: vec2, w: vec2, k: f32, dy: f32) -> vec2 {
            let rest = k - w.y * dy
            var out = span
            if w.x == 0.0 {
                if rest < 0.0 { out = vec2(1.0, 0.0) }
                return out
            }
            let bound = rest / w.x
            if w.x > 0.0 { out.y = min(out.y, bound) } else { out.x = max(out.x, bound) }
            return out
        }
        // A feature's region: the seg's slab t in [-g, len + g], h in [-(dout + g), din + g],
        // or the vertex's wedge grown by g, within metric depth + g and its reach.
        region_rows: fn(s: u32, vertex: bool, din: f32, dout: f32, g: f32) -> vec2 {
            let a = self.seg_a(s)
            if vertex {
                let r = self.vertex_reach(s, self.vertex_depth(s, din, dout)) + g
                return vec2(a.y - r, a.y + r)
            }
            let e = self.seg_e(s)
            let n = self.left(e)
            let t0 = (0.0 - g) * e.y
            let t1 = (self.seg_len(s) + g) * e.y
            let h0 = (0.0 - (dout + g)) * n.y
            let h1 = (din + g) * n.y
            let lo = min(min(t0 + h0, t0 + h1), min(t1 + h0, t1 + h1))
            let hi = max(max(t0 + h0, t0 + h1), max(t1 + h0, t1 + h1))
            return vec2(a.y + lo, a.y + hi)
        }
        region_span: fn(s: u32, vertex: bool, y: f32, din: f32, dout: f32, g: f32) -> vec2 {
            let a = self.seg_a(s)
            let dy = y - a.y
            var sp = vec2(0.0 - 1.0e30, 1.0e30)
            if vertex {
                let d = self.vertex_depth(s, din, dout)
                let r = self.vertex_reach(s, d) + g
                if abs(dy) > r { return vec2(1.0, 0.0) }
                let half = sqrt(r * r - dy * dy)
                sp = vec2(0.0 - half, half)
                sp = self.clip(sp, vec2(0.0) - self.seg_e(self.seg_prev(s)), g, dy)
                sp = self.clip(sp, self.seg_e(s), g, dy)
                if self.miter() {
                    sp = self.clip(sp, self.vertex_m1(s), d + g, dy)
                    sp = self.clip(sp, self.vertex_m2(s), d + g, dy)
                    if (self.seg_flags(s) & 4u) != 0u { sp = self.clip(sp, self.vertex_u(s), d + g, dy) }
                }
            } else {
                let e = self.seg_e(s)
                let n = self.left(e)
                sp = self.clip(sp, vec2(0.0) - e, g, dy)
                sp = self.clip(sp, e, self.seg_len(s) + g, dy)
                sp = self.clip(sp, n, din + g, dy)
                sp = self.clip(sp, vec2(0.0) - n, dout + g, dy)
            }
            return vec2(a.x + sp.x, a.x + sp.y)
        }
        // Integer index ranges [lo, hi) of a region's rows and of one row's span, over
        // nodes (offset 0) or fine-cell centres (offset 0.5), clamped to the grid.
        range: fn(lo: f32, hi: f32, offset: f32, last: u32) -> vec2u {
            let a = max(ceil(lo - offset), 0.0)
            let b = min(floor(hi - offset), f32(last))
            if b < a { return vec2u(0u, 0u) }
            return vec2u(u32(a), u32(b) + 1u)
        }
        // The three regions every feature pass walks: 0 marks (cell centres), 1 nodes,
        // 2 cand (cell centres).
        region_din: fn(kind: u32) -> f32 {
            var d = asfloat(self.field[1u].x) + asfloat(self.field[1u].w)
            if kind == 2u { d += 3.0 * 0.7072 + 0.001 }
            return d
        }
        region_dout: fn(kind: u32) -> f32 {
            var d = asfloat(self.field[1u].y) + asfloat(self.field[1u].w)
            if kind == 2u { d += 3.0 * 0.7072 + 0.001 }
            return d
        }
        region_grow: fn(kind: u32) -> f32 {
            if kind == 0u { return 0.75 }
            if kind == 1u { return 0.001 }
            return 0.7072 + 0.001
        }
        region_offset: fn(kind: u32) -> f32 {
            if kind == 1u { return 0.0 }
            return 0.5
        }
        region_last_x: fn(kind: u32) -> u32 {
            if kind == 1u { return self.nodes_x() - 1u }
            return self.nodes_x() - 2u
        }
        region_last_y: fn(kind: u32) -> u32 {
            if kind == 1u { return self.nodes_y() - 1u }
            return self.nodes_y() - 2u
        }
        rows_of: fn(s: u32, vertex: bool, kind: u32) -> vec2u {
            let r = self.region_rows(s, vertex, self.region_din(kind), self.region_dout(kind), self.region_grow(kind))
            return self.range(r.x, r.y, self.region_offset(kind), self.region_last_y(kind))
        }
        span_of: fn(s: u32, vertex: bool, kind: u32, row: u32) -> vec2u {
            let o = self.region_offset(kind)
            let sp = self.region_span(s, vertex, f32(row) + o, self.region_din(kind), self.region_dout(kind), self.region_grow(kind))
            return self.range(sp.x, sp.y, o, self.region_last_x(kind))
        }
        // The thread's feature: (seg, 1 for its start vertex), or NONE when it has none.
        feature_of: fn(i: u32) -> vec2u {
            if i >= self.segs() { return vec2u(4294967295u, 0u) }
            if (self.dispatch_range.z & 1u) != 0u {
                if (self.seg_flags(i) & 3u) == 0u { return vec2u(4294967295u, 0u) }
                return vec2u(i, 1u)
            }
            if (self.seg_flags(i) & 8u) == 0u { return vec2u(4294967295u, 0u) }
            return vec2u(i, 0u)
        }
        feature_id: fn(s: u32, vertex: bool) -> u32 {
            if vertex { return s | 2147483648u }
            return s
        }
        // A feature's value at a point on the given side, or 1e30 outside its domain.
        value: fn(s: u32, vertex: bool, p: vec2, inside: bool) -> f32 {
            let d = p - self.seg_a(s)
            if vertex {
                if self.vertex_inside(s) != inside { return 1.0e30 }
                if dot(d, self.seg_e(self.seg_prev(s))) < 0.0 - 0.0001 || dot(d, self.seg_e(s)) > 0.0001 { return 1.0e30 }
                let m = max(self.vertex_metric(s, d), 0.0)
                if inside { return m + self.seg_bias(s) }
                return m - self.seg_bias(s)
            }
            let e = self.seg_e(s)
            let t = dot(d, e)
            if t < 0.0 - 0.0001 || t > self.seg_len(s) + 0.0001 { return 1.0e30 }
            let h = dot(d, self.left(e))
            if inside {
                if h < 0.0 - 0.0001 { return 1.0e30 }
                return max(h, 0.0) + self.seg_bias(s)
            }
            if h > 0.0001 { return 1.0e30 }
            return max(0.0 - h, 0.0) - self.seg_bias(s)
        }
        // The lower bound of a feature's value over the disk of radius √2/2 around c, on
        // side 0 (outside), 1 (inside) or 2 (both).
        lower_bound: fn(s: u32, vertex: bool, c: vec2, side: u32) -> f32 {
            let r = 0.7072
            let d = c - self.seg_a(s)
            if vertex {
                let vin = self.vertex_inside(s)
                if side == 1u && !vin { return 1.0e30 }
                if side == 0u && vin { return 1.0e30 }
                if dot(d, self.seg_e(self.seg_prev(s))) < 0.0 - r || dot(d, self.seg_e(s)) > r { return 1.0e30 }
                let lb = max(self.vertex_metric(s, d) - r, 0.0)
                if vin { return lb + self.seg_bias(s) }
                return lb - self.seg_bias(s)
            }
            let e = self.seg_e(s)
            let t = dot(d, e)
            if t < 0.0 - r || t > self.seg_len(s) + r { return 1.0e30 }
            let h = dot(d, self.left(e))
            var best = 1.0e30
            if side != 0u && h >= 0.0 - r { best = min(best, max(h - r, 0.0) + self.seg_bias(s)) }
            if side != 1u && h <= r { best = min(best, max(0.0 - h - r, 0.0) - self.seg_bias(s)) }
            return best
        }
        // Monotone u32 keys of f32 values for atomic_min.
        key: fn(v: f32) -> u32 {
            let bits = asuint(v)
            if (bits & 2147483648u) != 0u { return bits ^ 4294967295u }
            return bits ^ 2147483648u
        }
        unkey: fn(k: u32) -> f32 {
            if (k & 2147483648u) != 0u { return asfloat(k ^ 2147483648u) }
            return asfloat(k ^ 4294967295u)
        }
        // The band copy of node (gx, gy) of plane p in coarse cell (cx, cy), or NONE.
        node_copy: fn(p: u32, gx: u32, gy: u32, cx: u32, cy: u32) -> u32 {
            let slot = self.band_slot[p * self.cells() + cy * self.field[0u].x + cx]
            if slot == 4294967295u { return 4294967295u }
            return slot * 121u + (gy - cy * 10u) * 11u + (gx - cx * 10u)
        }
        // The coarse cells holding node (gx, gy): [lo, hi] per axis.
        node_cells: fn(gx: u32, gy: u32) -> vec4u {
            var cx0 = gx / 10u
            var cy0 = gy / 10u
            if gx % 10u == 0u && gx > 0u { cx0 -= 1u }
            if gy % 10u == 0u && gy > 0u { cy0 -= 1u }
            return vec4u(cx0, cy0, min(gx / 10u, self.field[0u].x - 1u), min(gy / 10u, self.field[0u].y - 1u))
        }
        // Parity of the set's crossings on the horizontal leg y = row + ε² over x in (x0, x1],
        // and on the vertical leg x = col + ε over y in (y0, y1]. Vertices on the line count
        // as below / left of it, so only the fine row above / column right holds crossings.
        row_parity: fn(p: u32, row: u32, x0: u32, x1: u32) -> u32 {
            var parity = 0u
            let y = f32(row)
            for fx in x0..(x1 + 1u) {
                let r = self.cell_segs(p, fx, row)
                for s in r.x..r.y {
                    let q = self.seg[s]
                    if (q.y > y) != (q.w > y) {
                        var xc = q.x
                        if q.w == y { xc = q.z }
                        if xc > f32(x0) && xc <= f32(x1) { parity = parity ^ 1u }
                    }
                }
            }
            return parity
        }
        col_parity: fn(p: u32, col: u32, y0: u32, y1: u32) -> u32 {
            var parity = 0u
            let x = f32(col)
            for fy in y0..(y1 + 1u) {
                let r = self.cell_segs(p, col, fy)
                for s in r.x..r.y {
                    let q = self.seg[s]
                    if (q.x > x) != (q.z > x) {
                        var yc = q.y
                        if q.z == x { yc = q.w }
                        if yc > f32(y0) && yc <= f32(y1) { parity = parity ^ 1u }
                    }
                }
            }
            return parity
        }
        // The side (1 inside) of an arbitrary point in band fine cell (fx, fy) of slot.
        point_side: fn(p: u32, slot: u32, fx: u32, fy: u32, q: vec2) -> u32 {
            let lx = fx % 10u
            let ly = fy % 10u
            var side = self.node_side[slot * 121u + ly * 11u + lx]
            let r = self.cell_segs(p, fx, fy)
            if r.y == r.x { return side }
            let x = f32(fx)
            let y0 = f32(fy)
            for s in r.x..r.y {
                let g = self.seg[s]
                // up the cell's left edge from its low node, then along the point's row
                if (g.x > x) != (g.z > x) {
                    var yc = g.y
                    if g.z == x { yc = g.w }
                    if yc > y0 && yc <= q.y { side = side ^ 1u }
                }
                if (g.y > q.y) != (g.w > q.y) {
                    let xc = g.x + (q.y - g.y) * (g.z - g.x) / (g.w - g.y)
                    if xc > x && xc <= q.x { side = side ^ 1u }
                }
            }
            return side
        }
        // The nearest feature and the bits of the biased signed distance d' at q in plane p,
        // from the cand list of q's fine cell; (NONE, ±BIG) off the band, by coverage.
        field_feature: fn(p: u32, q: vec2) -> vec2u {
            let fx = min(u32(max(floor(q.x), 0.0)), self.nodes_x() - 2u)
            let fy = min(u32(max(floor(q.y), 0.0)), self.nodes_y() - 2u)
            let coarse = (fy / 10u) * self.field[0u].x + fx / 10u
            let slot = self.band_slot[p * self.cells() + coarse]
            if slot == 4294967295u {
                if self.cover[p * self.cells() + coarse] != 0u { return vec2u(4294967295u, asuint(0.0 - 1.0e9)) }
                return vec2u(4294967295u, asuint(1.0e9))
            }
            let inside = self.point_side(p, slot, fx, fy, q) != 0u
            let cell = slot * 100u + (fy % 10u) * 10u + fx % 10u
            var best = 1.0e30
            var feature = 4294967295u
            for k in self.cand_off[cell]..self.cand_off[cell + 1u] {
                let f = self.cand[k]
                let v = self.value(f & 2147483647u, (f & 2147483648u) != 0u, q, inside)
                if v < best { best = v; feature = f }
            }
            if feature == 4294967295u { best = 1.0e9 }
            if inside { return vec2u(feature, asuint(0.0 - best)) }
            return vec2u(feature, asuint(best))
        }
        field_at: fn(p: u32, q: vec2) -> f32 { return asfloat(self.field_feature(p, q).y) }
    }

    mod.sdf.FieldClear = {
        ..mod.sdf.Field
        field: shader.buffer_rw(vec4u)
        band_mark: shader.buffer_rw(u32)
        cover: shader.buffer_rw(u32)
        counts: shader.buffer_rw(u32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            self.band_mark[i] = 0u
            self.cover[i] = 0u
            if i == 0u {
                for k in 0..7u { self.counts[self.field[2u].x + k] = 0u }
                // a set made on the GPU: its seg count word into field[2].w
                if self.field[2u].z != 4294967295u { self.field[2u].w = self.counts[self.field[2u].z] }
            }
        }
    }

    mod.sdf.FieldSegMeta = {
        ..mod.sdf.Field
        seg_geo: shader.buffer_rw(vec4f)
        seg_meta: shader.buffer_rw(vec4u)
        main: fn() {
            let s = self.index()
            if s == 4294967295u || s >= self.segs() { return }
            let slots = self.set_kept_off[self.field[0u].z]
            let fine = self.search(s, slots * 100u, 0u)
            let slot = fine / 100u
            let local = fine % 100u
            let coarse = self.set_kept[slot]
            let p = self.search(slot, self.field[0u].z, 1u)
            let gx = (coarse % self.field[0u].x) * 10u + local % 10u
            let gy = (coarse / self.field[0u].x) * 10u + local / 10u
            let info = self.seg_info[s]
            let l = self.search(info.x, self.loop_off[self.field[0u].z], 2u)
            let bias = asfloat(self.loop_bias[l])
            let q = self.seg[s]
            let len = length(q.zw - q.xy)
            let e = self.direction(q.xy, q.zw)
            let pq = self.seg[info.z]
            let e1 = self.direction(pq.xy, pq.zw)
            let turn = e1.x * e.y - e1.y * e.x
            let cosine = dot(e1, e)
            var flags = 0u
            if (info.w & 1u) != 0u { flags = 1u } else if turn > 0.0 { flags = 2u }
            if flags != 0u && self.miter() && 1.0 + cosine <= 2.0 / (asfloat(self.field[1u].z) * asfloat(self.field[1u].z)) { flags = flags | 4u }
            if len > 0.0 { flags = flags | 8u }
            self.seg_geo[s * 2u] = vec4f(q.x, q.y, e.x, e.y)
            self.seg_geo[s * 2u + 1u] = vec4f(len, bias, cosine, 0.0)
            self.seg_meta[s] = vec4u(p | (flags << 24u), info.y, gx | (gy << 16u), info.z)
        }
    }

    mod.sdf.FieldDomain = {
        ..mod.sdf.Field
        band_mark: shader.buffer_rw(u32)
        counts: shader.buffer_rw(u32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let f = self.feature_of(i)
            if f.x == 4294967295u { return }
            let s = f.x
            let vertex = f.y != 0u
            let base = self.seg_plane(s) * self.cells()
            // Marks: the coarse cells of the fine cells whose centres lie in the region
            // grown by 0.75, so every node of the node region has a band copy.
            let rows = self.rows_of(s, vertex, 0u)
            var lo = 4294967295u
            var hi = 0u
            for fy in rows.x..rows.y {
                let sp = self.span_of(s, vertex, 0u, fy)
                if sp.y > sp.x {
                    lo = min(lo, sp.x / 10u)
                    hi = max(hi, (sp.y - 1u) / 10u)
                }
                if fy + 1u == rows.y || (fy + 1u) % 10u == 0u {
                    for cx in lo..(hi + 1u) { self.band_mark[base + (fy / 10u) * self.field[0u].x + cx] = 1u }
                    lo = 4294967295u
                    hi = 0u
                }
            }
            var bound = 0u
            let crows = self.rows_of(s, vertex, 2u)
            for fy in crows.x..crows.y {
                let sp = self.span_of(s, vertex, 2u, fy)
                bound += sp.y - sp.x
            }
            if bound != 0u { atomic_add(self.counts[self.field[2u].x], bound) }
        }
    }

    mod.sdf.FieldBandList = {
        ..mod.sdf.Field
        band_slot: shader.buffer_rw(u32)
        band: shader.buffer_rw(u32)
        band_off: shader.buffer_rw(u32)
        counts: shader.buffer_rw(u32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let at = self.band_slot[i]
            let marked = self.band_mark[i] != 0u
            if i % self.cells() == 0u { self.band_off[i / self.cells()] = at }
            if i + 1u == self.field[0u].z * self.cells() {
                var total = at
                if marked { total += 1u }
                self.band_off[self.field[0u].z] = total
                self.counts[self.field[2u].x + 3u] = total
                self.counts[self.field[2u].x + 4u] = total * 121u
                self.counts[self.field[2u].x + 5u] = total * 100u
            }
            if marked { self.band[at] = i % self.cells() } else { self.band_slot[i] = 4294967295u }
        }
    }

    mod.sdf.FieldCoverCount = {
        ..mod.sdf.Field
        cover: shader.buffer_rw(u32)
        main: fn() {
            let s = self.index()
            if s == 4294967295u || s >= self.segs() { return }
            let cell = self.seg_meta[s].z
            let gy = cell >> 16u
            if gy % 10u != 5u { return }
            let y = f32(gy)
            let q = self.seg[s]
            if (q.y > y) == (q.w > y) { return }
            // A seg of the fine row above the centre row crosses it only at an endpoint.
            var xc = q.x
            if q.w == y { xc = q.z }
            var cx = u32(max(floor((xc - 5.0) / 10.0), 0.0))
            for k in 0..3u { if cx < self.field[0u].x && f32(cx * 10u + 5u) < xc { cx += 1u } }
            for k in 0..3u { if cx > 0u && f32(cx * 10u - 5u) >= xc { cx -= 1u } }
            if cx < self.field[0u].x {
                atomic_add(self.cover[self.seg_plane(s) * self.cells() + (gy / 10u) * self.field[0u].x + cx], 1u)
            }
        }
    }

    mod.sdf.FieldCoverRows = {
        ..mod.sdf.Field
        cover: shader.buffer_rw(u32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let at = (i / self.field[0u].y) * self.cells() + (i % self.field[0u].y) * self.field[0u].x
            var run = 0u
            for cx in 0..self.field[0u].x {
                run += self.cover[at + cx]
                self.cover[at + cx] = run & 1u
            }
        }
    }

    mod.sdf.FieldNodeSide = {
        ..mod.sdf.Field
        node_side: shader.buffer_rw(u32)
        nf_key: shader.buffer_rw(u32)
        nf: shader.buffer_rw(u32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let slot = i / 121u
            let local = i % 121u
            let p = self.band_plane(slot)
            let coarse = self.band[slot]
            let cx = coarse % self.field[0u].x
            let cy = coarse / self.field[0u].x
            let gx = cx * 10u + local % 11u
            let gy = cy * 10u + local / 11u
            let xc = cx * 10u + 5u
            let yc = cy * 10u + 5u
            var side = self.cover[p * self.cells() + coarse]
            side = side ^ self.row_parity(p, yc, min(xc, gx), max(xc, gx))
            side = side ^ self.col_parity(p, gx, min(yc, gy), max(yc, gy))
            self.node_side[i] = side
            self.nf_key[i] = 4294967295u
            self.nf[i * 2u] = 4294967295u
            self.nf[i * 2u + 1u] = 0u
        }
    }

    mod.sdf.FieldNearestDist = {
        ..mod.sdf.Field
        nf_key: shader.buffer_rw(u32)
        counts: shader.buffer_rw(u32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let f = self.feature_of(i)
            if f.x == 4294967295u { return }
            let s = f.x
            let vertex = f.y != 0u
            let p = self.seg_plane(s)
            let rows = self.rows_of(s, vertex, 1u)
            for gy in rows.x..rows.y {
                let sp = self.span_of(s, vertex, 1u, gy)
                for gx in sp.x..sp.y {
                    let c = self.node_cells(gx, gy)
                    var first = 4294967295u
                    for cy in c.y..(c.w + 1u) {
                        for cx in c.x..(c.z + 1u) {
                            let at = self.node_copy(p, gx, gy, cx, cy)
                            if at != 4294967295u && first == 4294967295u { first = at }
                        }
                    }
                    if first == 4294967295u {
                        atomic_add(self.counts[self.field[2u].x + 1u], 1u)
                    } else {
                        let v = self.value(s, vertex, vec2(f32(gx), f32(gy)), self.node_side[first] != 0u)
                        if v < 1.0e29 {
                            let k = self.key(v)
                            for cy in c.y..(c.w + 1u) {
                                for cx in c.x..(c.z + 1u) {
                                    let at = self.node_copy(p, gx, gy, cx, cy)
                                    if at != 4294967295u { atomic_min(self.nf_key[at], k) }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    mod.sdf.FieldNearestId = {
        ..mod.sdf.Field
        nf: shader.buffer_rw(u32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let f = self.feature_of(i)
            if f.x == 4294967295u { return }
            let s = f.x
            let vertex = f.y != 0u
            let id = self.feature_id(s, vertex)
            let p = self.seg_plane(s)
            let rows = self.rows_of(s, vertex, 1u)
            for gy in rows.x..rows.y {
                let sp = self.span_of(s, vertex, 1u, gy)
                for gx in sp.x..sp.y {
                    let c = self.node_cells(gx, gy)
                    for cy in c.y..(c.w + 1u) {
                        for cx in c.x..(c.z + 1u) {
                            let at = self.node_copy(p, gx, gy, cx, cy)
                            if at != 4294967295u {
                                let v = self.value(s, vertex, vec2(f32(gx), f32(gy)), self.node_side[at] != 0u)
                                if v < 1.0e29 && self.key(v) == self.nf_key[at] { atomic_min(self.nf[at * 2u], id) }
                            }
                        }
                    }
                }
            }
        }
    }

    mod.sdf.FieldNearestFinish = {
        ..mod.sdf.Field
        nf: shader.buffer_rw(u32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let k = self.nf_key[i]
            var v = 1.0e9
            if k != 4294967295u { v = self.unkey(k) }
            if self.node_side[i] != 0u { v = 0.0 - v }
            self.nf[i * 2u + 1u] = asuint(v)
        }
    }

    mod.sdf.FieldCellBound = {
        ..mod.sdf.Field
        cell_t: shader.buffer_rw(u32)
        cell_side: shader.buffer_rw(u32)
        cand_cursor: shader.buffer_rw(u32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let slot = i / 100u
            let lx = (i % 100u) % 10u
            let ly = (i % 100u) / 10u
            var lo = 1.0e30
            var sides = 0u
            for k in 0..4u {
                let node = slot * 121u + (ly + k / 2u) * 11u + lx + k % 2u
                let d = asfloat(self.nf[node * 2u + 1u])
                if self.node_side[node] != 0u { lo = min(lo, 0.0 - d); sides = sides | 1u }
                else { lo = min(lo, d); sides = sides | 2u }
            }
            let coarse = self.band[slot]
            let r = self.cell_segs(self.band_plane(slot), (coarse % self.field[0u].x) * 10u + lx, (coarse / self.field[0u].x) * 10u + ly)
            var side = 0u
            if sides == 1u { side = 1u }
            var t = lo
            if sides == 3u || r.y > r.x { side = 2u; t = max(lo, asfloat(self.field[1u].w)) }
            self.cell_t[i] = asuint(t + 1.4143 + 0.001)
            self.cell_side[i] = side
            self.cand_cursor[i] = 0u
            // the scan's extra value, so cand_off ends with the total
            if i == 0u { self.cand_cursor[self.band_off[self.field[0u].z] * 100u] = 0u }
        }
    }

    // Count (the cand capacity unused) and fill share the walk over the feature's cand region.
    mod.sdf.FieldCandWalk = {
        ..mod.sdf.Field
        visit: fn(cell: u32, id: u32) {}
        walk: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let f = self.feature_of(i)
            if f.x == 4294967295u { return }
            let s = f.x
            let vertex = f.y != 0u
            let id = self.feature_id(s, vertex)
            let base = self.seg_plane(s) * self.cells()
            let rows = self.rows_of(s, vertex, 2u)
            for fy in rows.x..rows.y {
                let sp = self.span_of(s, vertex, 2u, fy)
                for fx in sp.x..sp.y {
                    let slot = self.band_slot[base + (fy / 10u) * self.field[0u].x + fx / 10u]
                    if slot != 4294967295u {
                        let cell = slot * 100u + (fy % 10u) * 10u + fx % 10u
                        let c = vec2(f32(fx) + 0.5, f32(fy) + 0.5)
                        if self.lower_bound(s, vertex, c, self.cell_side[cell]) <= asfloat(self.cell_t[cell]) {
                            self.visit(cell, id)
                        }
                    }
                }
            }
        }
    }

    mod.sdf.FieldCandCount = {
        ..mod.sdf.FieldCandWalk
        cand_cursor: shader.buffer_rw(u32)
        visit: fn(cell: u32, id: u32) { atomic_add(self.cand_cursor[cell], 1u) }
        main: fn() { self.walk() }
    }

    mod.sdf.FieldCandFill = {
        ..mod.sdf.FieldCandWalk
        cand_cursor: shader.buffer_rw(u32)
        cand: shader.buffer_rw(u32)
        counts: shader.buffer_rw(u32)
        // the cursor holds the cell's count: each visit takes the next slot down
        visit: fn(cell: u32, id: u32) {
            let at = self.cand_off[cell] + atomic_add(self.cand_cursor[cell], 4294967295u) - 1u
            if at < self.field[2u].y { self.cand[at] = id }
            else { atomic_add(self.counts[self.field[2u].x + 2u], 1u) }
        }
        main: fn() { self.walk() }
    }

    mod.sdf.FieldCandSort = {
        ..mod.sdf.Field
        cand: shader.buffer_rw(u32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let lo = self.cand_off[i]
            let hi = min(self.cand_off[i + 1u], self.field[2u].y)
            for k in (lo + 1u)..hi {
                let v = self.cand[k]
                var j = k
                while j > lo {
                    let w = self.cand[j - 1u]
                    if w <= v { break }
                    self.cand[j] = w
                    j -= 1u
                }
                self.cand[j] = v
            }
        }
    }

    // field_feature at given points (x, y, plane bits, 0): (feature, d' bits).
    mod.sdf.FieldProbe = {
        ..mod.sdf.Field
        probes: shader.buffer(vec4f)
        probe_out: shader.buffer_rw(u32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let q = self.probes[i]
            let r = self.field_feature(asuint(q.z), q.xy)
            self.probe_out[i * 2u] = r.x
            self.probe_out[i * 2u + 1u] = r.y
        }
    }

    mod.sdf
}
