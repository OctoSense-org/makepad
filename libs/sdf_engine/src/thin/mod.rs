//! W1b: thin walls on the GPU (GPU-EXEC §10.3). A FeatureSet's segs become the
//! thin site set ([`sites`]); the exact inner skeleton of those sites, clipped
//! to the band's fine cells, becomes a node graph whose central paths carry
//! bead counts and layouts; each skeleton edge's faces become bead pieces with
//! plane_abi keys at levels `thin_level(j)` (the fill boundary at
//! `thin_level(cap + 1)`), and where a face climbs past the reach its slots run
//! on along the site at the fixed layout (reach.rs).
//!
//! [`steps`] lays the stage out on the gpu_jobs runtime: the nearest-feature
//! field of the set (Round join, no bias) for the cell site lists, then three
//! host steps that size everything from exact counts: band slots (the field's),
//! the pair total, and the span total. Everything later is bounded by spans.
pub mod sites;
mod skeleton;
mod graph;
mod layout;
mod faces;
mod reach;

use crate::feature_field::{self, FieldInput, FieldJoin, FieldKernels, FieldOut, FieldParams};
use crate::feature_set::FeatureSet;
use crate::gpu_jobs::{label, BufRef, CountWord, Counts, Elements, JobError, Pass, Pool, ScanKernels, Step};
use makepad_csg_sdf::plan::{ArachneSpec, Join, PlanSpec};
use makepad_draw::*;

pub use sites::{HARD_TURN, SITE_TOL};

/// The thin stage's count words, from [`ThinOut::words`].
pub const WORD_CHUNKS: u32 = 0;
pub const WORD_PAIRS: u32 = 1;
pub const WORD_SPANS: u32 = 2;
/// Spans past the pair sweep's total (must stay 0).
pub const WORD_SPAN_OVERFLOW: u32 = 3;
/// Node incidences past the tie capacity (must stay 0).
pub const WORD_TIE_OVERFLOW: u32 = 4;
pub const WORD_SAMPLES: u32 = 5;
pub const WORD_EDGES: u32 = 6;
pub const WORD_EVENTS: u32 = 7;
/// Monotone owner paths (a diagnostic).
pub const WORD_OWNER_PATHS: u32 = 8;
pub const WORD_ARMS: u32 = 9;
pub const WORD_PIECES: u32 = 10;
/// Piece keys past plane_abi's thin ordinals (must stay 0).
pub const WORD_KEY_OVERFLOW: u32 = 11;
/// Site walks that never closed their loop (a FeatureSet defect; must stay 0).
pub const WORD_WALK_OVERFLOW: u32 = 12;
/// The longest cell site list: the tie capacity of a node.
pub const WORD_LIST_MAX: u32 = 13;
/// Cut transitions in the per-site lists (reach.rs).
pub const WORD_TRANSITIONS: u32 = 14;
/// The largest cut end + 1 (0: none): the fixed layout of a site with no cut.
pub const WORD_REACH_NODE: u32 = 15;
/// Run ends at the reach (the pair count sweep): the bound of the cuts.
pub const WORD_CUTS: u32 = 16;
/// Reach faces with no continuation along their site (must stay 0).
pub const WORD_REACH_DEFECT: u32 = 17;
/// Reflex point sites (ThinSiteLinks).
pub const WORD_POINTS: u32 = 18;
/// Node ends that see another node than their own (ThinSelectPaths; must stay 0).
pub const WORD_NODE_SPLIT: u32 = 19;
pub const WORDS: u32 = 20;
/// Piece slots per band fine cell in `piece_off`: the faces' pieces, then the reach faces'.
pub const PIECE_SLOTS: u32 = 2;

/// The chord tolerance of a skeleton parabola, grid units (0.00025 mm).
pub const CURVE_ERROR: f32 = 0.0025;
/// The band past the reach: a fine cell's diagonal (√2) rounded up, so every
/// cell holding a skeleton point within reach has a whole site list.
pub const BAND_MARGIN: f32 = 2.0;
/// The field's outside depth: the band needs only the inside, and one fine
/// cell outside keeps the outline's own cells whole.
pub const DEPTH_OUT: f32 = 1.0;

/// The thin request of one Arachne wall set, in grid units (0.1 mm = 1.0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThinSpec {
    /// The `parameters` rows (layout.rs).
    pub parameters: [u32; 12],
    /// 2 · loops: the most beads a layout has before the fill.
    pub cap: u32,
    /// The deepest inside distance the skeleton and its layouts read.
    pub reach: f32,
    /// sin(transition angle / 2): an edge is central where |Δr| <= len · this.
    pub central_sine: f32,
}

impl ThinSpec {
    /// The thin request of `spec`'s wall set `set`: None for a plan without
    /// Arachne walls of Round join there, or with no loops.
    pub fn for_set(spec: &PlanSpec, set: usize) -> Result<Option<Self>, String> {
        if !matches!(spec.join, Join::Round) { return Ok(None); }
        let Some(sets) = &spec.arachne else { return Ok(None) };
        if sets.len() != spec.wall_sets.len() { return Err("Arachne settings must match the plan's wall sets".into()); }
        let s = sets.get(set).ok_or_else(|| format!("no Arachne settings for wall set {set}"))?;
        Self::new(s)
    }

    pub fn new(s: &ArachneSpec) -> Result<Option<Self>, String> {
        if s.loops == 0 { return Ok(None); }
        let cap = s.loops.checked_mul(2).filter(|&c| c + 2 < 32).ok_or("Arachne loop count past the thin levels")?;
        // The fixed nominal layout, with the fill marker, begins at count cap + 1;
        // at count cap the CPU still distributes the thickness.
        let cap_thickness = 2.0 * s.s_outer + (cap - 1) as f64 * s.s_inner - 0.01;
        if !cap_thickness.is_finite() || cap_thickness <= 0.0 || !s.transition_length.is_finite()
            || s.transition_length <= 0.0 || !s.transition_filter_deviation.is_finite()
            || s.transition_filter_deviation < 0.0 || !s.transition_angle.is_finite() || s.transition_angle <= 0.0 {
            return Err("invalid Arachne reach parameters".into());
        }
        let shrink = (s.w_outer - s.s_outer) * if s.precise { 1.0 } else { 0.5 };
        let need = cap_thickness * 0.5 + s.transition_length + s.transition_filter_deviation + shrink.abs();
        let reach = round_up(need * 10.0)?;
        let g = |mm: f64| (mm * 10.0) as f32;
        let parameters = [
            g(s.s_outer).to_bits(), g(s.s_inner).to_bits(), g(s.w_outer).to_bits(), g(s.w_inner).to_bits(),
            g(s.min_bead_width).to_bits(), g(s.min_feature_size).to_bits(), g(s.transition_length).to_bits(), g(s.transition_filter_deviation).to_bits(),
            (s.transition_angle as f32).to_bits(), s.distribution_count as u32, cap as u32, u32::from(s.precise),
        ];
        Ok(Some(Self { parameters, cap: cap as u32, reach, central_sine: (s.transition_angle * 0.5).sin() as f32 }))
    }

    /// The field depth the thin stage reads, grid units.
    pub fn depth_in(&self) -> f32 { self.reach + BAND_MARGIN }
}

/// The f32 at or above a non-negative `value`: directed rounding, not padding.
fn round_up(value: f64) -> Result<f32, String> {
    let rounded = value as f32;
    if !rounded.is_finite() || rounded < 0.0 { return Err("Arachne reach overflow".into()); }
    Ok(f32::from_bits(rounded.to_bits() + u32::from((rounded as f64) < value)))
}

/// Registers the thin kernels (after gpu_jobs and the draw modules).
pub fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
    sites::script_mod(vm);
    skeleton::script_mod(vm);
    graph::script_mod(vm);
    layout::script_mod(vm);
    faces::script_mod(vm);
    reach::script_mod(vm)
}

macro_rules! kernels {
    ($($k:ident),* $(,)?) => {
        #[derive(Clone, Copy, Debug)]
        enum K { $($k),* }
        const KERNELS: &[&str] = &[$(concat!("Thin", stringify!($k))),*];
    };
}
kernels!(
    LoopClear, SiteBreaks, ChunkStarts, ChunkList, Runs, SiteLinks,
    SlotInfo, SiteLists, PairCount, Pairs, CellSpans, Keys, JumpInit, Jump, Finish,
    NodeInfo, SelectPaths, PathRankInit, PathRankJump, PathRankFinish, PathSizes, PathPack,
    AdjCount, AdjScatter, OwnerPaths, OwnerPathJump, OwnerPathCounts, OwnerDagInit, OwnerDagJump, OwnerFinish,
    CountEvents, EventFlags, EventScatter, EventPaths, Gaps, PathRanges, TransitionFilter, Ramps, PathCounts,
    LayoutSamples, LayoutQueries, BeadWeights, BeadScan, BeadRaw, BeadNormalize, BeadClassify, BeadHalf, BeadBlend,
    NodeOrdinal, FaceCount, FaceGeometry, FaceKeys,
    ReachClear, ReachMark, ReachScatter, ReachExits, ReachStarts, ReachCount, ReachGeometry, ReachKeys,
);

/// Per pass: its label, work units per element (the calibration's first
/// estimate) and its job ceiling, the most elements one job may take within
/// 50 ms. A cell or pair element visits its 3 x 3 cells' spans or its cell's
/// sites (a few dozen at the band's density); a node element its incidences.
fn shape(k: K) -> (&'static str, u32, u32) {
    use K::*;
    match k {
        LoopClear | ChunkStarts | ChunkList | SlotInfo | CellSpans | JumpInit | Jump | Finish
        | PathRankInit | PathRankJump | PathRankFinish | PathSizes | AdjCount | AdjScatter
        | OwnerPathJump | OwnerPathCounts | OwnerDagJump | EventFlags | PathRanges | BeadWeights
        | BeadScan | BeadRaw | BeadNormalize | BeadClassify | BeadHalf | BeadBlend | ReachClear | ReachMark => (KERNELS[k as usize], 4, 1 << 20),
        SiteBreaks | SiteLinks | CountEvents | EventScatter | PathCounts | LayoutQueries | ReachScatter => (KERNELS[k as usize], 32, 1 << 18),
        // a site's transition list per element
        ReachExits | ReachStarts => (KERNELS[k as usize], 64, 1 << 16),
        PathPack | EventPaths | Gaps | OwnerPaths | OwnerDagInit | OwnerFinish | LayoutSamples => (KERNELS[k as usize], 64, 1 << 18),
        // one walk per chunk; the cost is set from the mean loop length
        Runs => (KERNELS[k as usize], 64, 1 << 12),
        SiteLists | PairCount => (KERNELS[k as usize], 256, 1 << 16),
        Keys | NodeInfo | SelectPaths | NodeOrdinal => (KERNELS[k as usize], 512, 1 << 16),
        // a path's events in order on one thread
        TransitionFilter | Ramps => (KERNELS[k as usize], 256, 1 << 14),
        Pairs => (KERNELS[k as usize], 8192, 1 << 12),
        FaceCount | FaceGeometry | FaceKeys | ReachCount | ReachGeometry | ReachKeys => (KERNELS[k as usize], 4096, 1 << 14),
    }
}

/// The thin kernels with the field's and the runtime's scan, made once per Cx.
#[derive(Clone)]
pub struct ThinKernels { kernels: Vec<ComputeKernel>, field: FieldKernels, scan: ScanKernels }

impl ThinKernels {
    pub fn new(cx: &mut Cx) -> Result<Self, String> {
        let kernels = cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            KERNELS.iter().map(|name| {
                let object = vm.value(module, LiveId::from_str(name)).as_object().ok_or_else(|| format!("{name} kernel missing"))?;
                ComputeKernel::new(vm, object, name)
            }).collect::<Result<Vec<_>, String>>()
        })?;
        Ok(Self { kernels, field: FieldKernels::new(cx)?, scan: ScanKernels::new(cx)? })
    }
    fn kernel(&self, k: K) -> &ComputeKernel { &self.kernels[k as usize] }
}

/// The stage's buffers once its steps ran, and its sizes. Per span 4 `output`
/// records (its two ends: x, y, radius, flags; its sites, cell, pair; its tags
/// and plane); per end two `node_info`, two `ranked` and cap / 2 + 1 `layouts`
/// records; per band cell and [`PIECE_SLOTS`] slot its first piece in `piece_off`
/// (cells · 2 + 1 entries, ending with the total).
#[derive(Clone, Copy, Debug)]
pub struct ThinOut {
    pub field: FieldOut,
    /// The first of [`WORDS`] counts words.
    pub words: CountWord,
    pub cfg: BufRef,
    pub cap: u32,
    pub cells: u32,
    pub segs: u32,
    pub pairs: u32,
    pub spans: u32,
    /// The tie capacity: the longest cell site list.
    pub ties: u32,
    pub site: BufRef,
    pub site_info: BufRef,
    pub site_of: BufRef,
    pub site_lists: BufRef,
    pub site_list: BufRef,
    pub output: BufRef,
    pub thin: BufRef,
    pub node_info: BufRef,
    pub ranked: BufRef,
    pub paths: BufRef,
    pub samples: BufRef,
    pub fractional: BufRef,
    pub owners: BufRef,
    pub queries: BufRef,
    pub layouts: BufRef,
    /// Per site index its span flag, then the cut transitions (reach.rs).
    pub transitions: BufRef,
    /// Per end, then per site index, a reach face's run (reach.rs).
    pub reach_runs: BufRef,
    pub piece_off: BufRef,
    pub piece: BufRef,
    pub piece_geo: BufRef,
    pub piece_width: BufRef,
    pub piece_key: BufRef,
    pub piece_info: BufRef,
    /// Pieces allocated: per layout slot two faces per span, and one reach face
    /// per cut and two per reflex point site.
    pub piece_bound: u32,
}

impl ThinOut {
    pub fn word(&self, w: u32) -> CountWord { CountWord(self.words.0 + w) }
    /// Bead slots per endpoint layout (the last is the fill marker's).
    pub fn slots(&self) -> u32 { self.cap / 2 + 1 }
    /// The stage's invariants in counts read after its last pass.
    pub fn verify(&self, counts: &Counts) -> Result<(), JobError> {
        for (w, what) in [(WORD_SPAN_OVERFLOW, "spans past the pair sweep's total"), (WORD_TIE_OVERFLOW, "node incidences past the tie capacity"),
            (WORD_KEY_OVERFLOW, "piece keys past the thin ordinals"), (WORD_WALK_OVERFLOW, "site walks that never closed"),
            (WORD_REACH_DEFECT, "reach faces with no continuation"), (WORD_NODE_SPLIT, "node ends that see another node")] {
            let n = counts.word(self.word(w));
            if n != 0 { return Err(JobError::Failed(format!("thin walls: {n} {what}"))); }
        }
        let pieces = counts.word(self.word(WORD_PIECES));
        if pieces > self.piece_bound { return Err(JobError::Failed(format!("thin walls: {pieces} pieces past {}", self.piece_bound))); }
        Ok(())
    }
}

/// Named buffers a pass binds by its kernel's binding names.
#[derive(Clone)]
struct Binds(Vec<(LiveId, BufRef)>);
impl Binds {
    fn set(&mut self, name: LiveId, buf: BufRef) {
        match self.0.iter_mut().find(|b| b.0 == name) { Some(b) => b.1 = buf, None => self.0.push((name, buf)) }
    }
    fn get(&self, name: LiveId) -> Option<BufRef> {
        if name == id!(counts) { return Some(BufRef::RUNTIME); }
        self.0.iter().find(|b| b.0 == name).map(|b| b.1)
    }
}

fn le(words: impl IntoIterator<Item = u32>) -> Vec<u8> { words.into_iter().flat_map(u32::to_le_bytes).collect() }
/// Pointer-jumping rounds that cover chains of up to n links.
fn rounds(n: u32) -> u32 { 32 - n.leading_zeros() }

/// The stage's kernels and the buffers bound so far, carried from host step to host step.
struct Build { k: ThinKernels, binds: Binds }
impl Build {
    fn pass(&self, k: K, elements: Elements, zw: [u32; 2], cost: Option<u32>) -> Result<Pass, JobError> {
        let (name, base, ceiling) = shape(k);
        let kernel = self.k.kernel(k);
        let missing: Vec<String> = kernel.layout().buffers.iter().filter(|b| self.binds.get(b.name).is_none()).map(|b| b.name.to_string()).collect();
        if !missing.is_empty() { return Err(JobError::Failed(format!("{name}: unbound {missing:?}"))); }
        Ok(Pass::new(kernel, label(name), elements, cost.unwrap_or(base), ceiling, |b| self.binds.get(b).unwrap()).zw(zw[0], zw[1]))
    }
    fn fixed(&self, k: K, n: u32) -> Result<Step, JobError> { self.fixed_w(k, n, 0) }
    fn fixed_w(&self, k: K, n: u32, w: u32) -> Result<Step, JobError> { Ok(Step::Pass(self.pass(k, Elements::Fixed(n), [0, w], None)?)) }
    fn stepped(&self, k: K, n: u32, step: [u32; 4]) -> Result<Step, JobError> {
        Ok(Step::Pass(self.pass(k, Elements::Fixed(n), [0, 0], None)?.param_u32x4(id!(step), step).map_err(JobError::Failed)?))
    }
    fn repeat(&self, k: K, n: u32, rounds: u32, w: u32) -> Result<Step, JobError> {
        Ok(Step::Repeat(self.pass(k, Elements::Fixed(n), [0, w], None)?, rounds))
    }
    fn get(&self, name: &str) -> BufRef { self.binds.get(LiveId::from_str(name)).expect("a bound buffer") }
    fn scan(&self, pool: &mut Pool, input: &str, output: &str, n: u32, total: CountWord) -> Vec<Step> {
        self.k.scan.exclusive(pool, self.get(input), self.get(output), n, total)
    }
    fn buffer(&mut self, pool: &mut Pool, name: &'static str, bytes: u64) -> BufRef {
        let b = pool.buffer(name, bytes);
        self.binds.set(LiveId::from_str(name), b);
        b
    }
}

/// The thin stage's steps over `set` in `pool`: its field, sites, skeleton,
/// layouts and pieces, then `then`'s steps (the pieces' readers, given the
/// stage's buffers and the counts at the span total).
pub fn steps(k: &ThinKernels, pool: &mut Pool, set: &FeatureSet, spec: ThinSpec,
    then: impl FnOnce(&Counts, &mut Pool, &ThinOut) -> Result<Vec<Step>, JobError> + 'static) -> Result<Vec<Step>, JobError> {
    if set.seg.is_empty() { return Err(JobError::Failed("thin walls: an empty FeatureSet".into())); }
    let input = FieldInput::upload(pool, set, &vec![0.0; set.loop_info.len()])?;
    let field_words = feature_field::claim(pool)?;
    let words = pool.claim("thin walls", WORDS)?;
    let params = FieldParams { nx: set.nx, ny: set.ny, planes: set.planes, depth_in: spec.depth_in(), depth_out: DEPTH_OUT, join: FieldJoin::Round, bias_max: 0.0 };
    let loops = set.loop_info.len() as u32;
    let build = Build { k: k.clone(), binds: Binds(Vec::new()) };
    feature_field::steps(&k.field, pool, field_words, input, params, move |counts, pool, field| {
        field.verify(counts)?;
        sites_and_pairs(build, pool, field, spec, words, loops, then)
    })
}

/// What the host steps carry between the pair and span totals.
#[derive(Clone, Copy)]
struct Shape { field: FieldOut, words: CountWord, cfg: BufRef, cells: u32, segs: u32, pairs: u32, ties: u32, cap: u32, points: u32 }

fn sites_and_pairs(mut b: Build, pool: &mut Pool, field: &FieldOut, spec: ThinSpec, words: CountWord, loops: u32,
    then: impl FnOnce(&Counts, &mut Pool, &ThinOut) -> Result<Vec<Step>, JobError> + 'static) -> Result<Vec<Step>, JobError> {
    let (segs, slots, p) = (field.input.segs.capacity(), field.slots, field.params);
    let cells = slots.checked_mul(100).ok_or_else(|| JobError::Failed("thin walls: band cell overflow".into()))?;
    for (name, buf) in [(id!(seg), field.input.seg), (id!(seg_info), field.input.seg_info), (id!(band), field.band),
        (id!(band_off), field.band_off), (id!(band_slot), field.band_slot), (id!(cand_off), field.cand_off), (id!(cand), field.cand), (id!(cell_t), field.cell_t)] {
        b.binds.set(name, buf);
    }
    let cfg = b.buffer(pool, "cfg", (4 + slots as u64) * 16);
    pool.upload(cfg, 0, le([
        p.nx, p.ny, p.planes, cells,
        spec.reach.to_bits(), spec.central_sine.to_bits(), CURVE_ERROR.to_bits(), 0,
        words.0, segs, 0, spec.cap,
        0, 0, 0, 0,
    ]));
    b.buffer(pool, "parameters", 48);
    pool.upload(b.get("parameters"), 0, le(spec.parameters));
    for (name, bytes) in [("site_flag", 4), ("loop_hard", 4), ("chunk_flag", 4), ("chunk_off", 4), ("chunks", 4), ("site_of", 4), ("site", 16), ("site_info", 16)] {
        b.buffer(pool, name, segs as u64 * bytes);
    }
    b.buffer(pool, "site_lists", cells as u64 * 8);
    // A cand entry brings at most three sites (ThinSiteLists).
    b.buffer(pool, "site_list", field.cand_bound as u64 * 20);
    b.buffer(pool, "pair_count", cells as u64 * 4);
    b.buffer(pool, "pair_off", cells as u64 * 4);
    let word = move |w: u32| CountWord(words.0 + w);
    let mut steps = vec![b.fixed(K::LoopClear, segs)?, b.fixed(K::SiteBreaks, segs)?, b.fixed(K::ChunkStarts, segs)?];
    steps.extend(b.scan(pool, "chunk_flag", "chunk_off", segs, word(WORD_CHUNKS)));
    steps.push(b.fixed(K::ChunkList, segs)?);
    // A chunk walks at most its loop: its cost is the mean loop length.
    let walk = shape(K::Runs).1 * segs.div_ceil(loops.max(1));
    steps.push(Step::Pass(b.pass(K::Runs, Elements::Counted { word: word(WORD_CHUNKS), bound: segs }, [0, 0], Some(walk))?));
    steps.push(b.fixed(K::SiteLinks, segs)?);
    steps.push(b.fixed(K::SlotInfo, slots)?);
    steps.push(b.fixed(K::SiteLists, cells)?);
    steps.push(b.fixed(K::PairCount, cells)?);
    steps.extend(b.scan(pool, "pair_count", "pair_off", cells, word(WORD_PAIRS)));
    let field = *field;
    steps.push(Step::Host(Box::new(move |counts: &Counts, pool: &mut Pool| {
        let walks = counts.word(word(WORD_WALK_OVERFLOW));
        if walks != 0 { return Err(JobError::Failed(format!("thin walls: {walks} site walks never closed"))); }
        let pairs = counts.word(word(WORD_PAIRS));
        let ties = counts.word(word(WORD_LIST_MAX)).max(1);
        pool.upload(cfg, 28, le([ties]));
        b.buffer(pool, "pair_spans", pairs as u64 * 4);
        b.buffer(pool, "span_off", pairs as u64 * 4);
        // The count sweep writes no spans: its output stays a stand-in until
        // the span total sizes the real one.
        b.buffer(pool, "output", 16);
        let mut steps = vec![b.fixed_w(K::Pairs, pairs, 0)?];
        steps.extend(b.scan(pool, "pair_spans", "span_off", pairs, word(WORD_SPANS)));
        let points = counts.word(word(WORD_POINTS));
        let shape = Shape { field, words, cfg, cells, segs, pairs, ties, cap: spec.cap, points };
        steps.push(Step::Host(Box::new(move |counts: &Counts, pool: &mut Pool| skeleton_and_beads(b, pool, counts, shape, then))));
        Ok(steps)
    })));
    Ok(steps)
}

fn skeleton_and_beads(mut b: Build, pool: &mut Pool, counts: &Counts, s: Shape,
    then: impl FnOnce(&Counts, &mut Pool, &ThinOut) -> Result<Vec<Step>, JobError>) -> Result<Vec<Step>, JobError> {
    let word = |w: u32| CountWord(s.words.0 + w);
    let spans = counts.word(word(WORD_SPANS));
    let ends = spans.checked_mul(2).ok_or_else(|| JobError::Failed("thin walls: span overflow".into()))?;
    pool.upload(s.cfg, 40, le([spans]));
    let (cells, cap, ties) = (s.cells, s.cap, s.ties);
    let (stride, row, slots) = (cap + 1, cap + 2, cap / 2 + 1);
    let (e, n) = (ends as u64, spans as u64);
    let entries = ends.checked_mul(2 * row).ok_or_else(|| JobError::Failed("thin walls: layout overflow".into()))?;
    let thin_words = cells as u64 * 2 + e * 12 + e * ties as u64;
    if thin_words > u32::MAX as u64 { return Err(JobError::Failed(format!("thin walls: {thin_words} graph words"))); }
    let cuts = counts.word(word(WORD_CUTS));
    let sources = cuts as u64 + 2 * s.points as u64;
    let piece_bound = u32::try_from((n * 2 + sources) * slots as u64).map_err(|_| JobError::Failed("thin walls: piece overflow".into()))?;
    pool.upload(s.cfg, 48, le([piece_bound]));
    let sites = s.segs as u64 * 2;
    for (name, bytes) in [
        ("output", n * 64), ("thin", thin_words * 4), ("node_info", e * 32), ("ranked", e * 32), ("path_work", e * 32),
        ("path_samples", n * 4), ("path_edges", n * 4), ("sample_off", n * 4), ("edge_off", n * 4),
        ("paths", n * 16), ("samples", e * 16), ("edges", n * 16),
        ("events", n * stride as u64 * 16), ("event_flag", n * stride as u64 * 4), ("event_off", n * stride as u64 * 4),
        ("candidates", n * stride as u64 * 16), ("path_rules", n * 16), ("ranges", rounds(ends) as u64 * e * 8),
        ("filter_state", n * 16), ("kept", n * stride as u64 * 4), ("ramps", n * stride as u64 * 16), ("fractional", e * 4),
        ("adj_count", e * 4), ("adj_off", e * 4), ("adjacency", e * 8), ("owner_paths", e * 32), ("owners", e * 48),
        ("query_sources", e * 4), ("queries", e * 16),
        ("bead_work", entries as u64 * 32), ("bead_values", entries as u64 * 16), ("halves", entries as u64 * 16), ("layouts", e * slots as u64 * 16),
        ("reach_count", sites * 4), ("reach_off", sites * 4), ("transitions", (sites + cuts as u64 * 4) * 4),
        ("reach", (e + sites) * 64),
        ("piece_count", (cells as u64 * 2 + 1) * 4), ("piece_off", (cells as u64 * 2 + 1) * 4),
        ("piece", piece_bound as u64 * 16), ("piece_geo", piece_bound as u64 * 16), ("piece_width", piece_bound as u64 * 8),
        ("piece_key", piece_bound as u64 * 16), ("piece_info", piece_bound as u64 * 16),
    ] {
        b.buffer(pool, name, bytes);
    }
    let rounds = rounds(ends);
    let bank = rounds % 2;
    let mut steps = vec![b.fixed_w(K::Pairs, s.pairs, 1)?, b.fixed(K::CellSpans, cells)?];
    if spans != 0 {
        steps.push(b.fixed(K::Keys, ends)?);
        // Non-central paths first (their gaps may become central), then central ones.
        for central in [0, 1] {
            if central == 1 { steps.push(b.fixed(K::Gaps, spans)?); }
            steps.extend([
                b.fixed(K::NodeInfo, ends)?, b.fixed_w(K::SelectPaths, ends, central)?,
                b.fixed(K::JumpInit, ends)?, b.repeat(K::Jump, ends, rounds, 0)?, b.fixed_w(K::Finish, ends, bank)?,
                b.stepped(K::PathRankInit, ends, [0, 0, 0, 0])?, b.repeat(K::PathRankJump, ends, rounds, 0)?,
                b.stepped(K::PathRankInit, ends, [1, bank, 0, 0])?, b.repeat(K::PathRankJump, ends, rounds, 0)?,
                b.stepped(K::PathRankFinish, ends, [bank, 0, 0, 0])?,
            ]);
        }
        steps.push(b.fixed(K::PathSizes, spans)?);
        steps.extend(b.scan(pool, "path_samples", "sample_off", spans, word(WORD_SAMPLES)));
        steps.extend(b.scan(pool, "path_edges", "edge_off", spans, word(WORD_EDGES)));
        steps.extend([b.fixed_w(K::PathPack, spans, 0)?, b.fixed_w(K::PathPack, spans, 1)?]);
        // Count events, their transitions and the fractional counts along central paths.
        steps.extend([b.fixed(K::CountEvents, spans * stride)?, b.fixed(K::EventFlags, spans * stride)?]);
        steps.extend(b.scan(pool, "event_flag", "event_off", spans * stride, word(WORD_EVENTS)));
        steps.extend([
            b.fixed(K::EventScatter, spans * stride)?, b.fixed(K::EventPaths, spans)?, b.repeat(K::PathRanges, ends, rounds, 0)?,
            b.fixed(K::TransitionFilter, spans)?, b.fixed(K::Ramps, spans)?, b.fixed(K::PathCounts, ends)?,
        ]);
        // Layout owners: each node's arms, monotone paths, the downhill DAG.
        steps.push(b.fixed(K::AdjCount, ends)?);
        steps.extend(b.scan(pool, "adj_count", "adj_off", ends, word(WORD_ARMS)));
        let dag = rounds.min(32 - spans.leading_zeros());
        steps.extend([
            b.fixed(K::AdjScatter, ends)?, b.fixed(K::OwnerPaths, ends)?, b.repeat(K::OwnerPathJump, ends, rounds, 0)?,
            b.stepped(K::OwnerPathCounts, ends, [bank, 0, 0, 0])?, b.stepped(K::OwnerDagInit, ends, [bank, 0, 0, 0])?,
            b.repeat(K::OwnerDagJump, ends, dag, 0)?, b.stepped(K::OwnerFinish, ends, [bank, 0, dag % 2, 0])?,
            b.fixed(K::LayoutSamples, ends)?, b.fixed(K::LayoutQueries, ends)?,
        ]);
        // Bead layouts: weights, raw widths, normalised widths, locations, each
        // followed by its row scan, then the halves and their blend.
        let scan_rounds = 32 - (row - 1).leading_zeros();
        let mut at = 0;
        steps.push(b.fixed(K::BeadWeights, entries)?);
        for (kernel, kind) in [(K::BeadRaw, 0), (K::BeadNormalize, 0), (K::BeadClassify, 2)] {
            steps.push(b.repeat(K::BeadScan, entries, scan_rounds, at)?);
            at = (at + scan_rounds) % 2;
            steps.push(b.fixed_w(kernel, entries, at)?);
            at = 1 - at;
            if kind == 2 {
                steps.push(b.repeat(K::BeadScan, entries, scan_rounds, at | 2)?);
                at = (at + scan_rounds) % 2;
            }
        }
        steps.extend([b.fixed_w(K::BeadHalf, entries, at)?, b.fixed(K::BeadBlend, ends * slots)?, b.fixed(K::NodeOrdinal, ends)?]);
        // Reach faces: per site its cuts in a list, then each exit's and
        // each vertex start's run along its site.
        let sites = s.segs * 2;
        steps.extend([b.fixed(K::ReachClear, sites)?, b.fixed(K::ReachMark, ends)?]);
        steps.extend(b.scan(pool, "reach_count", "reach_off", sites, word(WORD_TRANSITIONS)));
        steps.extend([
            b.fixed(K::ReachScatter, ends)?,
            Step::Pass(b.pass(K::ReachExits, Elements::Counted { word: word(WORD_TRANSITIONS), bound: cuts * 2 }, [0, 0], None)?),
            b.fixed(K::ReachStarts, sites)?,
        ]);
    }
    // Pieces per band cell and slot (faces, reach faces): counts, their scan,
    // then geometry and keys at each slot's offset.
    steps.push(b.fixed(K::FaceCount, cells)?);
    if spans != 0 { steps.push(b.fixed(K::ReachCount, cells)?); }
    steps.extend(b.scan(pool, "piece_count", "piece_off", cells * 2 + 1, word(WORD_PIECES)));
    if spans != 0 {
        steps.extend([b.fixed(K::FaceGeometry, cells)?, b.fixed(K::FaceKeys, cells)?, b.fixed(K::ReachGeometry, cells)?, b.fixed(K::ReachKeys, cells)?]);
    }
    let out = ThinOut {
        field: s.field, words: s.words, cfg: s.cfg, cap, cells, segs: s.segs, pairs: s.pairs, spans, ties,
        site: b.get("site"), site_info: b.get("site_info"), site_of: b.get("site_of"), site_lists: b.get("site_lists"), site_list: b.get("site_list"),
        output: b.get("output"), thin: b.get("thin"), node_info: b.get("node_info"), ranked: b.get("ranked"),
        paths: b.get("paths"), samples: b.get("samples"), fractional: b.get("fractional"), owners: b.get("owners"),
        queries: b.get("queries"), layouts: b.get("layouts"), transitions: b.get("transitions"), reach_runs: b.get("reach"), piece_off: b.get("piece_off"), piece: b.get("piece"),
        piece_geo: b.get("piece_geo"), piece_width: b.get("piece_width"), piece_key: b.get("piece_key"), piece_info: b.get("piece_info"),
        piece_bound,
    };
    steps.extend(then(counts, pool, &out)?);
    Ok(steps)
}
