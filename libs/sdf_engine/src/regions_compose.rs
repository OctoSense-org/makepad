//! Region compositions on the GPU (GPU-EXEC §9, §10.3): one round of the class program
//! (csg_sdf::regions) over every plane of a batch. A round's terms are segs per fine cell (the
//! inside on their left: a FeatureSet, an extract's level, an earlier round's output) with their
//! fields' node sides, each read at a plane offset (the layer above or below). Its outputs are
//! compositions of those terms, one truth table each per plane and flag state; each output's
//! boundary comes out as pieces per fine cell, with positional endpoint keys, for chain.
//!
//! - Cuts. A term's seg is cut where another term's seg crosses or touches it, and where a
//!   collinear overlap starts or ends. A crossing point is computed once per pair of segs, from the
//!   lower merged index, and an endpoint is copied exactly. So every point that two output pieces
//!   share is the same f32 pair.
//! - Classification. Each sub-piece is classified on both sides at its midpoint. Its own term is
//!   inside on the left and outside on the right. Another term counts by its side: its field's side
//!   at the cell's low node, then parity along the L path, as feature_field's `point_side` measures
//!   it. Where the sub-piece lies on one of that term's segs, the term counts as coincident instead
//!   (inside on the same side when they run the same way).
//! - Output. A sub-piece goes to every output whose composition differs across it, turned so its
//!   inside is on its left. A coincident stretch is kept only by its lowest term.
//! - Flags. A plane's tables are picked by up to two flags: an earlier round's output has a piece
//!   on plane p + dz (its any-words, which every round writes per plane and output).
//!
//! Keys (PieceKey) take one of three forms:
//! - a point on a node: (its +x edge, key_lo(level, NODE_ORDINAL));
//! - a point on a grid line: (that fine edge, key_lo(level, rank)), where rank is its rank among
//!   the distinct points of that output on the edge, from both cells beside it;
//! - a point inside a cell: (CELL_BIT | cell, key_lo(level, rank)), where rank is its rank among
//!   that output's distinct interior points in the cell, by x then y.
//!
//! Passes. T terms, O outputs, P planes, C = nx * ny coarse cells. The round's coarse cells are
//! the R cells holding any term's segs. Round fine cell e = slot * 100 + local.
//! 1. `ComposeMark` per term, P·C: `rmark` = 1 on the cells where the term has segs;
//! 2. exclusive scan of `rmark` into `rslot`, total R;
//! 3. `ComposeSlots` P·C: `rinfo[slot]` = (plane, cx, cy, plane cell), `rslot` = NONE off the
//!    round, and the count words R·100 and R·100·T;
//! 4. `ComposeTermCount` per term over the capacity: `rcnt[e·T + t]`, the term's segs in e;
//! 5. exclusive scan of `rcnt` into `roff`, cell-major and term-minor (deterministic, no atomics);
//! 6. `ComposeTermFill` per term, R·100: `mp` (the segs) and `mterm` (t | seg << 3);
//! 7. `ComposeTermSide` per term, R·100: bit t of `side[e]`;
//! 8. `ComposeCount` over the capacity: `ocnt[e·O + o]`;
//! 9. exclusive scan of `ocnt` into `ooff` (total: the outputs);
//! 10. `ComposeFill` R·100: `out_piece` and `out_info` (plane, level + o, t | seg << 3, cell);
//! 11. `ComposeKeys` R·100: `out_key`, `out_geo`, per output `out_side` (the side at each low
//!     node), and the any-words;
//! and `ComposeCover` P·C after the marks: per output `out_cover`, its side at every coarse centre
//! (the marks gather each term's cover bit). Passes 4 and 8 write 0 past the round's cells, so
//! the scans' totals hold whatever buffers they reuse.
//!
//! Every kernel's `grid` uniform is (nx, ny, P, level); the round buffer holds the term records,
//! flags and tables. For the other uniforms, see each kernel. Outputs past the capacity, and cuts
//! or points past a thread's fixed arrays, count into the overflow word.
use makepad_draw::*;
use crate::gpu_jobs::{label, BufRef, CountWord, Elements, JobError, Pass, Pool, ScanKernels, Step};

/// Cuts per seg, and points per edge or cell, that one thread holds.
pub const CUTS: u32 = 16;
pub const POINTS: u32 = 32;
/// Terms per round. A truth table has 2^T entries, in two words.
pub const MAX_TERMS: u32 = 6;
/// Outputs per round: tables over the same terms, sharing the cuts.
pub const MAX_OUTPUTS: u32 = 8;
/// Flags per round, and the flag states that pick a plane's tables (flag i in bit i).
pub const MAX_FLAGS: usize = 2;
pub const STATES: usize = 1 << MAX_FLAGS;
/// A plane offset rides with this bias.
pub const PLANE_BIAS: u32 = 16;
/// The round buffer (u32 words): a header (outputs, terms, cell capacity, plane cells), a record
/// per term (plane offset biased, seg stride, seg at, side base, cover base, sides by level, its
/// levels' first word), a record per flag (word, stride, plane offset biased, on), the tables:
/// ((plane * STATES + state) * outputs + output) * 2, low word first, then the levels (f32 bits
/// per plane) of each term whose sides come from a distance field.
const TERM_AT: u32 = 8;
const TERM_WORDS: u32 = 8;
const FLAG_AT: u32 = TERM_AT + MAX_TERMS * TERM_WORDS;
const TABLE_AT: u32 = FLAG_AT + MAX_FLAGS as u32 * 4;

/// The truth table of a composition over `terms` terms: bit b is set where `inside(b)`, with
/// term t inside when bit t of b is set. Returns the low word, then the high word.
pub fn table(terms: u32, inside: impl Fn(u32) -> bool) -> [u32; 2] {
    assert!(terms <= MAX_TERMS, "a round composes at most {MAX_TERMS} terms");
    let mut t = 0u64;
    for b in 0..1u32 << terms {
        if inside(b) {
            t |= 1 << b;
        }
    }
    [t as u32, (t >> 32) as u32]
}

/// A plane offset with its bias.
pub fn plane_offset(offset: i32) -> u32 {
    assert!(offset.unsigned_abs() < PLANE_BIAS, "plane offset {offset}");
    (PLANE_BIAS as i32 + offset) as u32
}

/// One element bound per job for each pass shape: the round's simple passes touch a few words
/// per element; Compose cuts and classifies at most CUTS·T sub-pieces against T·CUTS segs.
const SIMPLE_CEILING: u32 = 1 << 20;
const COMPOSE_CEILING: u32 = 1 << 16;

/// The round's kernels, made once.
#[derive(Clone)]
pub struct ComposeKernels {
    mark: ComputeKernel,
    slots: ComputeKernel,
    term_count: ComputeKernel,
    term_fill: ComputeKernel,
    term_side: ComputeKernel,
    count: ComputeKernel,
    fill: ComputeKernel,
    keys: ComputeKernel,
    cover: ComputeKernel,
    scan: ScanKernels,
}
impl ComposeKernels {
    pub fn new(cx: &mut Cx) -> Result<Self, String> {
        let scan = ScanKernels::new(cx)?;
        cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            let mut kernel = |name: LiveId, label: &str| -> Result<ComputeKernel, String> {
                let object = vm.value(module, name).as_object().ok_or_else(|| format!("{label} kernel missing"))?;
                ComputeKernel::new(vm, object, label)
            };
            Ok(Self {
                mark: kernel(id!(ComposeMark), "regions mark")?,
                slots: kernel(id!(ComposeSlots), "regions slots")?,
                term_count: kernel(id!(ComposeTermCount), "regions term count")?,
                term_fill: kernel(id!(ComposeTermFill), "regions term fill")?,
                term_side: kernel(id!(ComposeTermSide), "regions term side")?,
                count: kernel(id!(ComposeCount), "regions compose count")?,
                fill: kernel(id!(ComposeFill), "regions compose fill")?,
                keys: kernel(id!(ComposeKeys), "regions keys")?,
                cover: kernel(id!(ComposeCover), "regions cover")?,
                scan,
            })
        })
    }
}

/// A term of a round: segs per fine cell (`set_slot`, `seg_off`, `seg`) with their field's sides
/// (`band_slot`, `cover`, `node_side`), read at a plane offset (+1 = the plane above).
/// `seg_off` holds `seg_stride` ranges per fine cell, cell-major, the term's at `seg_at` (a
/// multi-output round's output, an extract's level); its sides start at `side_base` in
/// `node_side` and `cover_base` in `cover`.
///
/// With `levels`, the term is a level set of a distance field: `node_side` is the field's `nf`
/// ((feature, d' bits) per band node) and a node is inside where d' <= `levels[p]` (grid units)
/// for the reading plane p. Off the band the field's cover holds (|level| within its depth).
#[derive(Clone, Debug)]
pub struct Term {
    pub set_slot: BufRef,
    pub seg_off: BufRef,
    pub seg: BufRef,
    pub band_slot: BufRef,
    pub cover: BufRef,
    pub node_side: BufRef,
    pub plane_offset: i32,
    pub seg_stride: u32,
    pub seg_at: u32,
    pub side_base: u32,
    pub cover_base: u32,
    pub levels: Option<Vec<f32>>,
}
impl Term {
    /// A FeatureSet with its field's sides.
    pub fn set(set_slot: BufRef, seg_off: BufRef, seg: BufRef, band_slot: BufRef, cover: BufRef, node_side: BufRef, plane_offset: i32) -> Self {
        Self { set_slot, seg_off, seg, band_slot, cover, node_side, plane_offset, seg_stride: 1, seg_at: 0, side_base: 0, cover_base: 0, levels: None }
    }
    /// Slot `slot` of an extract's pieces (`piece_off` over band fine cell × `slots` + slot,
    /// `piece`) on a field (`band_slot`, `cover`, `nf`): the field's level set at `levels[p]`
    /// (grid units) for reading plane p.
    #[allow(clippy::too_many_arguments)]
    pub fn level(band_slot: BufRef, cover: BufRef, nf: BufRef, piece_off: BufRef, piece: BufRef, slots: u32, slot: u32, plane_offset: i32, levels: Vec<f32>) -> Self {
        Self { set_slot: band_slot, seg_off: piece_off, seg: piece, band_slot, cover, node_side: nf, plane_offset, seg_stride: slots, seg_at: slot, side_base: 0, cover_base: 0, levels: Some(levels) }
    }
}

/// A flag: output `output` of an earlier round (its words' `any`, `outputs` per plane) has a
/// piece on plane p + `dz`.
#[derive(Clone, Copy, Debug)]
pub struct Flag {
    pub any: CountWord,
    pub outputs: u32,
    pub output: u32,
    pub dz: i32,
}

/// One round over `terms`: per output, plane and flag state a truth table (from [`table`]);
/// output o's pieces are at `level + o`. Capacities: `cells` coarse cells holding any term's segs
/// (all planes), `segs` of the terms' segs in them, and `outputs` pieces. Past one, the round
/// writes what fits and counts the rest in its overflow word; the caller runs it again with more.
#[derive(Clone, Debug)]
pub struct Round {
    pub nx: u32,
    pub ny: u32,
    pub planes: u32,
    pub level: u32,
    pub terms: Vec<Term>,
    pub flags: Vec<Flag>,
    pub tables: Vec<Vec<[[u32; 2]; STATES]>>,
    pub cells: u32,
    pub segs: u32,
    pub outputs: u32,
}
impl Round {
    /// Tables the same on every plane and flag state, one per output.
    pub fn uniform(tables: &[[u32; 2]], planes: u32) -> Vec<Vec<[[u32; 2]; STATES]>> {
        tables.iter().map(|&t| vec![[t; STATES]; planes as usize]).collect()
    }
}

/// The round's count words.
#[derive(Clone, Copy, Debug)]
pub struct RoundWords {
    /// R, the round's coarse cells (before the capacity).
    pub cells: CountWord,
    /// R·100 and R·100·T, within the capacity.
    pub fine: CountWord,
    pub fine_terms: CountWord,
    pub merged: CountWord,
    /// The output pieces (before the capacity).
    pub outputs: CountWord,
    /// Elements past a capacity or a thread's fixed arrays; nonzero = run again with more.
    pub overflow: CountWord,
    /// Per plane and output (plane-major): nonzero when the output has a piece there.
    pub any: CountWord,
}
impl RoundWords {
    pub fn claim(pool: &mut Pool, name: &'static str, planes: u32, outputs: u32) -> Result<Self, JobError> {
        let at = pool.claim(name, 6)?.0;
        let any = pool.claim(name, planes * outputs)?;
        Ok(Self { cells: CountWord(at), fine: CountWord(at + 1), fine_terms: CountWord(at + 2), merged: CountWord(at + 3),
            outputs: CountWord(at + 4), overflow: CountWord(at + 5), any })
    }
}

/// The round's output, GPU-resident: §10.3 pieces with their sides, per output.
/// - `rslot` (P·C -> round slot or NONE) and `rinfo` (slot -> plane, cx, cy, plane cell) index it;
/// - `ooff` is the CSR of pieces over (slot·100 + local fine cell, output), cell-major;
/// - `piece` vec4f, `key` vec4u (PieceKey), `info` vec4u (PieceInfo: plane, level + output,
///   term | seg << 3, cell), `geo` vec4f (PieceGeo: lines, all 0);
/// - `side` holds per output the composition's side at each round fine cell's low node (node_side
///   layout), `cover` per output its side at every coarse centre.
/// So a later round reads an output as a term ([`RoundOut::term`]) without chaining it first, and
/// chain reads all outputs as one piece set.
#[derive(Clone, Copy, Debug)]
pub struct RoundOut {
    pub rslot: BufRef,
    pub rinfo: BufRef,
    pub ooff: BufRef,
    pub piece: BufRef,
    pub key: BufRef,
    pub info: BufRef,
    pub geo: BufRef,
    pub side: BufRef,
    pub cover: BufRef,
    pub any: CountWord,
    pub outputs: u32,
    pub cells: u32,
    pub plane_cells: u32,
}
impl RoundOut {
    pub fn term(&self, output: u32, plane_offset: i32) -> Term {
        Term { set_slot: self.rslot, seg_off: self.ooff, seg: self.piece, band_slot: self.rslot, cover: self.cover, node_side: self.side, plane_offset,
            seg_stride: self.outputs, seg_at: output, side_base: output * self.cells * 121, cover_base: output * self.plane_cells, levels: None }
    }
    pub fn flag(&self, output: u32, dz: i32) -> Flag { Flag { any: self.any, outputs: self.outputs, output, dz } }
}

/// The round buffer's words.
fn round_words(round: &Round, cells: u32, plane_cells: u32) -> Result<Vec<u32>, String> {
    let (terms, outputs) = (round.terms.len() as u32, round.tables.len() as u32);
    if terms == 0 || terms > MAX_TERMS { return Err(format!("a round composes 1 to {MAX_TERMS} terms, not {terms}")); }
    if outputs == 0 || outputs > MAX_OUTPUTS { return Err(format!("a round has 1 to {MAX_OUTPUTS} outputs, not {outputs}")); }
    if round.flags.len() > MAX_FLAGS { return Err(format!("a round reads at most {MAX_FLAGS} flags")); }
    if round.tables.iter().any(|t| t.len() != round.planes as usize) { return Err("a table per plane".into()); }
    let mut w = vec![0u32; (TABLE_AT + round.planes * STATES as u32 * outputs * 2) as usize];
    w[..4].copy_from_slice(&[outputs, terms, cells, plane_cells]);
    for (i, t) in round.terms.iter().enumerate() {
        let at = (TERM_AT + i as u32 * TERM_WORDS) as usize;
        let level_at = w.len() as u32;
        if let Some(levels) = &t.levels {
            if levels.len() != round.planes as usize { return Err(format!("term {i}: {} levels for {} planes", levels.len(), round.planes)); }
            w.extend(levels.iter().map(|l| l.to_bits()));
        }
        w[at..at + 7].copy_from_slice(&[plane_offset(t.plane_offset), t.seg_stride, t.seg_at, t.side_base, t.cover_base, t.levels.is_some() as u32, level_at]);
    }
    for (i, f) in round.flags.iter().enumerate() {
        let at = (FLAG_AT + i as u32 * 4) as usize;
        w[at..at + 4].copy_from_slice(&[f.any.0 + f.output, f.outputs, plane_offset(f.dz), 1]);
    }
    for (o, per_plane) in round.tables.iter().enumerate() {
        for (p, states) in per_plane.iter().enumerate() {
            for (s, t) in states.iter().enumerate() {
                let at = (TABLE_AT as usize) + ((p * STATES + s) * outputs as usize + o) * 2;
                w[at..at + 2].copy_from_slice(t);
            }
        }
    }
    Ok(w)
}

/// The steps of one round, its buffers in `pool`.
pub fn steps(k: &ComposeKernels, pool: &mut Pool, round: &Round, words: RoundWords) -> Result<(Vec<Step>, RoundOut), String> {
    let terms = round.terms.len() as u32;
    let no = round.tables.len() as u32;
    let plane_cells = round.planes * round.nx * round.ny;
    let cells = round.cells.max(1);
    let fine = cells * 100;
    let fine_terms = fine * terms;
    let segs = round.segs.max(1);
    let outputs = round.outputs.max(1);
    let desc = round_words(round, cells, plane_cells)?;
    let round_buf = pool.buffer("regions round", desc.len() as u64 * 4);
    pool.upload(round_buf, 0, desc.iter().flat_map(|w| w.to_le_bytes()).collect());
    let rmark = pool.buffer("regions rmark", plane_cells as u64 * 4);
    let rslot = pool.buffer("regions rslot", plane_cells as u64 * 4);
    let rinfo = pool.buffer("regions rinfo", cells as u64 * 16);
    let rcnt = pool.buffer("regions rcnt", (fine_terms + 1) as u64 * 4);
    let roff = pool.buffer("regions roff", (fine_terms + 1) as u64 * 4);
    let mp = pool.buffer("regions merged", segs as u64 * 16);
    let mterm = pool.buffer("regions merged term", segs as u64 * 4);
    let side = pool.buffer("regions side", fine as u64 * 4);
    let ocnt = pool.buffer("regions ocnt", (fine * no + 1) as u64 * 4);
    let ooff = pool.buffer("regions ooff", (fine * no + 1) as u64 * 4);
    let piece = pool.buffer("regions piece", outputs as u64 * 16);
    let key = pool.buffer("regions key", outputs as u64 * 16);
    let info = pool.buffer("regions info", outputs as u64 * 16);
    let geo = pool.buffer("regions geo", outputs as u64 * 16);
    let cbits = pool.buffer("regions cover bits", plane_cells as u64 * 4);
    let out_cover = pool.buffer("regions cover", (plane_cells * no) as u64 * 4);
    let out_side = pool.buffer("regions node side", (cells * no) as u64 * 121 * 4);
    let grid = [round.nx, round.ny, round.planes, round.level];
    let counted = Elements::Counted { word: words.fine, bound: fine };
    // every binding by its name: the round's own buffers, then the term's; `offs` is the
    // outputs' counts for Count and their offsets for Fill
    let bind = |b: LiveId, t: Option<&Term>, offs: BufRef| -> Option<BufRef> {
        Some(match (b, t) {
            (id!(counts), _) => BufRef::COUNTS,
            (id!(round), _) => round_buf,
            (id!(rmark), _) => rmark, (id!(rslot), _) => rslot, (id!(rinfo), _) => rinfo, (id!(rcnt), _) => rcnt, (id!(roff), _) => roff,
            (id!(mp), _) => mp, (id!(mterm), _) => mterm, (id!(side), _) => side, (id!(offs), _) => offs, (id!(ooff), _) => ooff,
            (id!(out_piece), _) => piece, (id!(out_key), _) => key, (id!(out_info), _) => info, (id!(out_geo), _) => geo,
            (id!(cbits), _) => cbits, (id!(out_cover), _) => out_cover, (id!(out_side), _) => out_side,
            (id!(set_slot), Some(t)) => t.set_slot, (id!(seg_off), Some(t)) => t.seg_off, (id!(seg), Some(t)) => t.seg,
            (id!(band_slot), Some(t)) => t.band_slot, (id!(cover), Some(t)) => t.cover, (id!(node_side), Some(t)) => t.node_side,
            _ => return None,
        })
    };
    let pass = |kernel: &ComputeKernel, name: &str, elements: Elements, cost: u32, ceiling: u32, w: [u32; 4], zw: [u32; 2], t: Option<&Term>, offs: BufRef| {
        let mut missing = None;
        let p = Pass::new(kernel, label(name), elements, cost, ceiling, |b| bind(b, t, offs).unwrap_or_else(|| { missing = Some(b); BufRef::RUNTIME }));
        if let Some(b) = missing { return Err(format!("{name}: no buffer for binding {b}")); }
        // a kernel that never reads a parameter has none in its layout
        let mut p = p.zw(zw[0], zw[1]);
        for (param, v) in [(id!(words), w), (id!(grid), grid)] {
            if kernel.layout().slots.iter().any(|s| s.name == param) { p = p.param_u32x4(param, v)?; }
        }
        Ok(Step::Pass(p))
    };
    let none = BufRef::RUNTIME;
    let mut out = Vec::new();
    for (i, t) in round.terms.iter().enumerate() {
        out.push(pass(&k.mark, "regions mark", Elements::Fixed(plane_cells), 1, SIMPLE_CEILING, [0; 4], [i as u32, terms], Some(t), none)?);
    }
    out.push(pass(&k.cover, "regions cover", Elements::Fixed(plane_cells), no, SIMPLE_CEILING, [0; 4], [0, 0], None, none)?);
    out.extend(k.scan.exclusive(pool, rmark, rslot, plane_cells, words.cells));
    out.push(pass(&k.slots, "regions slots", Elements::Fixed(plane_cells), 1, SIMPLE_CEILING,
        [words.cells.0, words.fine.0, words.fine_terms.0, terms], [cells, words.overflow.0], None, none)?);
    // the counts run over the capacity: past the round's cells they write 0 for the scans
    let per_term = [words.fine.0, fine, words.overflow.0, segs];
    for (i, t) in round.terms.iter().enumerate() {
        out.push(pass(&k.term_count, "regions term count", Elements::Fixed(fine), 2, SIMPLE_CEILING, per_term, [i as u32, terms], Some(t), none)?);
    }
    out.extend(k.scan.exclusive(pool, rcnt, roff, fine_terms + 1, words.merged));
    for (i, t) in round.terms.iter().enumerate() {
        out.push(pass(&k.term_fill, "regions term fill", counted, 4, SIMPLE_CEILING, per_term, [i as u32, terms], Some(t), none)?);
        out.push(pass(&k.term_side, "regions term side", counted, 2, SIMPLE_CEILING, per_term, [i as u32, terms], Some(t), none)?);
    }
    let compose = [words.fine.0, outputs, words.overflow.0, fine];
    out.push(pass(&k.count, "regions compose count", Elements::Fixed(fine), 64, COMPOSE_CEILING, compose, [terms, 0], None, ocnt)?);
    out.extend(k.scan.exclusive(pool, ocnt, ooff, fine * no + 1, words.outputs));
    out.push(pass(&k.fill, "regions compose fill", counted, 64, COMPOSE_CEILING, compose, [terms, 0], None, ooff)?);
    out.push(pass(&k.keys, "regions keys", counted, 32 * no, COMPOSE_CEILING, [words.fine.0, words.any.0, words.overflow.0, 0], [0, 0], None, none)?);
    Ok((out, RoundOut { rslot, rinfo, ooff, piece, key, info, geo, side: out_side, cover: out_cover, any: words.any, outputs: no, cells, plane_cells }))
}

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    // The round buffer (see TERM_AT, FLAG_AT, TABLE_AT): header (outputs, terms, cell capacity,
    // plane cells), term records from 8, flags from 56, tables from 64.
    mod.sdf.ComposeBase = {
        precise_math: true
        workgroup_size: 64
        dispatch_range: shader.uniform(vec4u)
        words: shader.uniform(vec4u)
        grid: shader.uniform(vec4u)
        counts: shader.buffer_rw(u32)
        round: shader.buffer(u32)
        element: fn() -> u32 {
            if global_id().x >= self.dispatch_range.y { return 4294967295u }
            return self.dispatch_range.x + global_id().x
        }
        cells: fn() -> u32 { return self.grid.x * self.grid.y }
        nodes_x: fn() -> u32 { return self.grid.x * 10u + 1u }
        outputs: fn() -> u32 { return self.round[0u] }
        // Word i of term t's record: 0 plane offset (biased), 1 seg stride, 2 seg at, 3 side
        // base, 4 cover base, 5 sides by level, 6 its levels' first word.
        term_word: fn(t: u32, i: u32) -> u32 { return self.round[8u + t * 8u + i] }
        // Entry `bits` of a truth table (lo, hi): 1 where the composition is inside.
        table_bit: fn(bits: u32, lo: u32, hi: u32) -> u32 {
            if bits < 32u { return (lo >> bits) & 1u }
            return (hi >> (bits - 32u)) & 1u
        }
        // Plane p's flag state: bit i where flag i's output has a piece on plane p + dz.
        state: fn(p: u32) -> u32 {
            var s = 0u
            for i in 0u..2u {
                let at = 56u + i * 4u
                if self.round[at + 3u] != 0u {
                    let q = p + self.round[at + 2u]
                    if q >= 16u && q < self.grid.z + 16u {
                        if self.counts[self.round[at] + (q - 16u) * self.round[at + 1u]] != 0u { s = s | (1u << i) }
                    }
                }
            }
            return s
        }
        // Output o's table on plane p in flag state s, as (lo, hi).
        table: fn(p: u32, s: u32, o: u32) -> vec2u {
            let at = 64u + ((p * 4u + s) * self.outputs() + o) * 2u
            return vec2u(self.round[at], self.round[at + 1u])
        }
        // Round plane cell e (plane * C + coarse) in term t's set at its plane offset, or NONE
        // when that plane is outside the batch.
        term_cell: fn(t: u32, e: u32) -> u32 {
            let q = e / self.cells() + self.term_word(t, 0u)
            if q < 16u || q >= self.grid.z + 16u { return 4294967295u }
            return (q - 16u) * self.cells() + e % self.cells()
        }
    }

    // dispatch_range.z = the term. A cell is marked where the term has a seg in it (in its seg
    // slot), and its cover bit is the term's side at the coarse centre (node 60 of a level set's
    // band cell). Term 0 writes every mark and cover bit, the others add theirs.
    mod.sdf.ComposeMark = {
        ..mod.sdf.ComposeBase
        set_slot: shader.buffer(u32)
        seg_off: shader.buffer(u32)
        band_slot: shader.buffer(u32)
        cover: shader.buffer(u32)
        node_side: shader.buffer(u32)
        rmark: shader.buffer_rw(u32)
        cbits: shader.buffer_rw(u32)
        main: fn() {
            let e = self.element()
            if e == 4294967295u || e >= self.cells() * self.grid.z { return }
            let t = self.dispatch_range.z
            let c = self.term_cell(t, e)
            var m = 0u
            var cb = 0u
            if c != 4294967295u {
                let slot = self.set_slot[c]
                if slot != 4294967295u {
                    let stride = self.term_word(t, 1u)
                    if stride == 1u {
                        if self.seg_off[slot * 100u + 100u] > self.seg_off[slot * 100u] { m = 1u }
                    } else {
                        for f in 0u..100u {
                            let at = (slot * 100u + f) * stride + self.term_word(t, 2u)
                            if self.seg_off[at + 1u] > self.seg_off[at] { m = 1u }
                        }
                    }
                }
                var side = self.cover[self.term_word(t, 4u) + c] & 1u
                if self.term_word(t, 5u) != 0u {
                    let band = self.band_slot[c]
                    if band != 4294967295u {
                        side = 0u
                        if asfloat(self.node_side[(band * 121u + 60u) * 2u + 1u]) <= asfloat(self.round[self.term_word(t, 6u) + e / self.cells()]) { side = 1u }
                    }
                }
                cb = side << t
            }
            if t == 0u {
                self.rmark[e] = m
                self.cbits[e] = cb
            } else {
                if m != 0u { self.rmark[e] = 1u }
                self.cbits[e] = self.cbits[e] | cb
            }
        }
    }

    // Each output's cover (its side at every coarse centre), output-major.
    mod.sdf.ComposeCover = {
        ..mod.sdf.ComposeBase
        cbits: shader.buffer(u32)
        out_cover: shader.buffer_rw(u32)
        main: fn() {
            let e = self.element()
            let pc = self.cells() * self.grid.z
            if e == 4294967295u || e >= pc { return }
            let p = e / self.cells()
            let s = self.state(p)
            for o in 0u..self.outputs() {
                let t = self.table(p, s, o)
                self.out_cover[o * pc + e] = self.table_bit(self.cbits[e], t.x, t.y)
            }
        }
    }

    // words: x the count word of R, y the count word of R·100, z the count word of R·100·T,
    // w = T; dispatch_range: z the round's cell capacity, w the overflow word.
    mod.sdf.ComposeSlots = {
        ..mod.sdf.ComposeBase
        rmark: shader.buffer(u32)
        rslot: shader.buffer_rw(u32)
        rinfo: shader.buffer_rw(vec4u)
        main: fn() {
            let e = self.element()
            if e == 4294967295u || e >= self.cells() * self.grid.z { return }
            if e == 0u {
                var r = self.counts[self.words.x]
                if r > self.dispatch_range.z {
                    atomic_add(self.counts[self.dispatch_range.w], r - self.dispatch_range.z)
                    r = self.dispatch_range.z
                }
                self.counts[self.words.y] = r * 100u
                self.counts[self.words.z] = r * 100u * self.words.w
            }
            if self.rmark[e] == 0u {
                self.rslot[e] = 4294967295u
                return
            }
            let slot = self.rslot[e]
            if slot < self.dispatch_range.z {
                let c = e % self.cells()
                self.rinfo[slot] = vec4u(e / self.cells(), c % self.grid.x, c / self.grid.x, e)
            } else {
                self.rslot[e] = 4294967295u
            }
        }
    }

    // The round's fine cells. words: x the count word of R·100, y R·100's capacity, z the
    // overflow word, w the merged capacity; dispatch_range: z the term, w = T.
    mod.sdf.ComposeCells = {
        ..mod.sdf.ComposeBase
        rinfo: shader.buffer(vec4u)
        round_cell: fn() -> u32 {
            let e = self.element()
            if e == 4294967295u || e >= self.counts[self.words.x] { return 4294967295u }
            return e
        }
    }

    mod.sdf.ComposeTermRange = {
        ..mod.sdf.ComposeCells
        set_slot: shader.buffer(u32)
        seg_off: shader.buffer(u32)
        // The term's segs in round fine cell e, as [first, end).
        term_segs: fn(e: u32) -> vec2u {
            let t = self.dispatch_range.z
            let c = self.term_cell(t, self.rinfo[e / 100u].w)
            if c == 4294967295u { return vec2u(0u, 0u) }
            let slot = self.set_slot[c]
            if slot == 4294967295u { return vec2u(0u, 0u) }
            let at = (slot * 100u + e % 100u) * self.term_word(t, 1u) + self.term_word(t, 2u)
            return vec2u(self.seg_off[at], self.seg_off[at + 1u])
        }
    }

    // Over the capacity: past the round's cells, 0 for the scan (and the closing entry).
    mod.sdf.ComposeTermCount = {
        ..mod.sdf.ComposeTermRange
        rcnt: shader.buffer_rw(u32)
        main: fn() {
            let e = self.element()
            if e == 4294967295u || e >= self.words.y { return }
            let tt = self.dispatch_range.w
            if e == 0u && self.dispatch_range.z == 0u { self.rcnt[self.words.y * tt] = 0u }
            var n = 0u
            if e < self.counts[self.words.x] {
                let r = self.term_segs(e)
                n = r.y - r.x
            }
            self.rcnt[e * tt + self.dispatch_range.z] = n
        }
    }

    mod.sdf.ComposeTermFill = {
        ..mod.sdf.ComposeTermRange
        seg: shader.buffer(vec4f)
        roff: shader.buffer(u32)
        mp: shader.buffer_rw(vec4f)
        mterm: shader.buffer_rw(u32)
        main: fn() {
            let e = self.round_cell()
            if e == 4294967295u { return }
            let r = self.term_segs(e)
            let t = self.dispatch_range.z
            let at = self.roff[e * self.dispatch_range.w + t]
            for k in 0u..(r.y - r.x) {
                if at + k < self.words.w {
                    self.mp[at + k] = self.seg[r.x + k]
                    self.mterm[at + k] = t | ((r.x + k) << 3u)
                } else {
                    atomic_add(self.counts[self.words.z], 1u)
                }
            }
        }
    }

    // The term's side at e's low node: its field's band node (its side bit, or for a level set
    // d' <= the reading plane's level), else its coarse cell's cover (a cell off the band holds
    // none of the term's segs, so one side covers it).
    mod.sdf.ComposeTermSide = {
        ..mod.sdf.ComposeCells
        band_slot: shader.buffer(u32)
        cover: shader.buffer(u32)
        node_side: shader.buffer(u32)
        side: shader.buffer_rw(u32)
        main: fn() {
            let e = self.round_cell()
            if e == 4294967295u { return }
            let t = self.dispatch_range.z
            let c = self.term_cell(t, self.rinfo[e / 100u].w)
            var s = 0u
            if c != 4294967295u {
                let band = self.band_slot[c]
                let local = e % 100u
                if band == 4294967295u { s = self.cover[self.term_word(t, 4u) + c] & 1u }
                else {
                    let node = band * 121u + (local / 10u) * 11u + local % 10u
                    if self.term_word(t, 5u) == 0u { s = self.node_side[self.term_word(t, 3u) + node] & 1u }
                    else if asfloat(self.node_side[node * 2u + 1u]) <= asfloat(self.round[self.term_word(t, 6u) + self.rinfo[e / 100u].x]) { s = 1u }
                }
            }
            // term 0 writes the word, the others add their bit
            var v = 0u
            if t != 0u { v = self.side[e] }
            if s != 0u { v = v | (1u << t) }
            self.side[e] = v
        }
    }

    // words: x the count word of R·100, y the output capacity, z the overflow word, w R·100's
    // capacity; dispatch_range.z = T. `offs`: the counts per (cell, output) for Count, their
    // offsets for Fill.
    mod.sdf.Compose = {
        ..mod.sdf.ComposeBase
        rinfo: shader.buffer(vec4u)
        roff: shader.buffer(u32)
        mp: shader.buffer(vec4f)
        mterm: shader.buffer(u32)
        side: shader.buffer(u32)
        offs: shader.buffer_rw(u32)
        out_piece: shader.buffer_rw(vec4f)
        out_info: shader.buffer_rw(vec4u)
        terms: fn() -> u32 { return self.dispatch_range.z }
        cross2: fn(a: vec2, b: vec2) -> f32 { return a.x * b.y - a.y * b.x }
        overflow: fn() { atomic_add(self.counts[self.words.z], 1u) }
        // The side of term u at q in round fine cell e with low node (fx, fy): feature_field's
        // point_side over the term's segs in the cell.
        side_of: fn(e: u32, u: u32, fx: u32, fy: u32, q: vec2) -> u32 {
            var s = (self.side[e] >> u) & 1u
            let x = f32(fx)
            let y0 = f32(fy)
            for k in self.roff[e * self.terms() + u]..self.roff[e * self.terms() + u + 1u] {
                let g = self.mp[k]
                // up the cell's left edge from its low node, then along the point's row
                if (g.x > x) != (g.z > x) {
                    var yc = g.y
                    if g.z == x { yc = g.w }
                    if yc > y0 && yc <= q.y { s = s ^ 1u }
                }
                if (g.y > q.y) != (g.w > q.y) {
                    let xc = g.x + (q.y - g.y) * (g.z - g.x) / (g.w - g.y)
                    if xc > x && xc <= q.x { s = s ^ 1u }
                }
            }
            return s
        }
        // 1 when q lies on a seg of term u running along unit d, 2 on one running against it,
        // else 0.
        coincident: fn(e: u32, u: u32, q: vec2, d: vec2, eps: f32) -> u32 {
            var out = 0u
            for k in self.roff[e * self.terms() + u]..self.roff[e * self.terms() + u + 1u] {
                let g = self.mp[k]
                let r = g.zw - g.xy
                let l2 = dot(r, r)
                if out == 0u && l2 > 0.0 {
                    let l = sqrt(l2)
                    let t = dot(q - g.xy, r) / l
                    if t >= 0.0 - eps && t <= l + eps && abs(self.cross2(r, q - g.xy)) <= eps * l && abs(self.cross2(r, d)) <= 0.0001 * l {
                        if dot(r, d) > 0.0 { out = 1u } else { out = 2u }
                    }
                }
            }
            return out
        }
        // The kept sub-pieces of round fine cell e per output: counted into offs[e·O + o], or
        // written from those offsets when `write`.
        run: fn(e: u32, write: bool) {
            let info = self.rinfo[e / 100u]
            let local = e % 100u
            let fx = info.y * 10u + local % 10u
            let fy = info.z * 10u + local / 10u
            let tt = self.terms()
            let no = self.outputs()
            let st = self.state(info.x)
            var tlo = array(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
            var thi = array(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
            var at = array(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
            var cnt = array(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
            for o in 0u..no {
                let t = self.table(info.x, st, o)
                tlo[o] = t.x
                thi[o] = t.y
                if write { at[o] = self.offs[e * no + o] }
            }
            let lo = self.roff[e * tt]
            let hi = self.roff[e * tt + tt]
            for i in lo..hi {
                let g = self.mp[i]
                let ti = self.mterm[i] & 7u
                let a = g.xy
                let b = g.zw
                let r = b - a
                let l2 = dot(r, r)
                if l2 > 0.0 {
                    let l = sqrt(l2)
                    let m = max(max(max(abs(a.x), abs(a.y)), max(abs(b.x), abs(b.y))), 1.0)
                    let eps = 0.0000005 * m
                    var ts = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
                    var px = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
                    var py = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
                    px[0] = a.x
                    py[0] = a.y
                    ts[1] = 1.0
                    px[1] = b.x
                    py[1] = b.y
                    var n = 2u
                    for j in lo..hi {
                        if (self.mterm[j] & 7u) != ti {
                            let h = self.mp[j]
                            let c = h.xy
                            let w = h.zw - c
                            let lw = length(w)
                            let ac = c - a
                            let den = self.cross2(r, w)
                            var t1 = 0.0 - 1.0
                            var q1 = a
                            var t2 = 0.0 - 1.0
                            var q2 = a
                            if abs(den) > 0.000001 * l * lw {
                                let s = self.cross2(ac, w) / den
                                let v = self.cross2(ac, r) / den
                                if s * l >= 0.0 - eps && (1.0 - s) * l >= 0.0 - eps && v * lw >= 0.0 - eps && (1.0 - v) * lw >= 0.0 - eps {
                                    // the other seg's endpoint exactly, else the pair's point from the lower index
                                    if v * lw <= eps { q1 = c }
                                    else if (1.0 - v) * lw <= eps { q1 = h.zw }
                                    else if i < j { q1 = a + r * s }
                                    else { q1 = c + w * v }
                                    t1 = dot(q1 - a, r) / l2
                                }
                            } else if abs(self.cross2(r, ac)) <= eps * l {
                                // collinear: the other seg's endpoints that lie on this one
                                q1 = c
                                t1 = dot(c - a, r) / l2
                                q2 = h.zw
                                t2 = dot(h.zw - a, r) / l2
                            }
                            if t1 * l > eps && (1.0 - t1) * l > eps {
                                if n < 16u { ts[n] = t1; px[n] = q1.x; py[n] = q1.y; n += 1u } else { self.overflow() }
                            }
                            if t2 * l > eps && (1.0 - t2) * l > eps {
                                if n < 16u { ts[n] = t2; px[n] = q2.x; py[n] = q2.y; n += 1u } else { self.overflow() }
                            }
                        }
                    }
                    // sort the cuts by t (the start stays first, the end goes last), then merge
                    // points within eps, keeping the seg's own ends
                    for k in 2u..n {
                        var z = k
                        for step in 0u..16u {
                            if z > 1u && ts[z - 1u] > ts[z] {
                                let sw = ts[z]
                                ts[z] = ts[z - 1u]
                                ts[z - 1u] = sw
                                let sx = px[z]
                                px[z] = px[z - 1u]
                                px[z - 1u] = sx
                                let sy = py[z]
                                py[z] = py[z - 1u]
                                py[z - 1u] = sy
                                z -= 1u
                            }
                        }
                    }
                    var kept = 1u
                    for k in 1u..n {
                        if abs(px[k] - px[kept - 1u]) <= eps && abs(py[k] - py[kept - 1u]) <= eps {
                            if ts[k] == 1.0 { px[kept - 1u] = px[k]; py[kept - 1u] = py[k]; ts[kept - 1u] = 1.0 }
                        } else {
                            ts[kept] = ts[k]
                            px[kept] = px[k]
                            py[kept] = py[k]
                            kept += 1u
                        }
                    }
                    let dir = r / l
                    for k in 1u..kept {
                        let p0 = vec2(px[k - 1u], py[k - 1u])
                        let p1 = vec2(px[k], py[k])
                        let mid = (p0 + p1) * 0.5
                        var bl = 1u << ti
                        var br = 0u
                        var lower = false
                        for u in 0u..tt {
                            if u != ti {
                                let co = self.coincident(e, u, mid, dir, eps)
                                if co == 1u {
                                    bl = bl | (1u << u)
                                    if u < ti { lower = true }
                                } else if co == 2u {
                                    br = br | (1u << u)
                                    if u < ti { lower = true }
                                } else if self.side_of(e, u, fx, fy, mid) != 0u {
                                    bl = bl | (1u << u)
                                    br = br | (1u << u)
                                }
                            }
                        }
                        if !lower {
                            for o in 0u..no {
                                let il = self.table_bit(bl, tlo[o], thi[o])
                                if il != self.table_bit(br, tlo[o], thi[o]) {
                                    if write {
                                        let w = at[o] + cnt[o]
                                        if w < self.words.y {
                                            if il != 0u { self.out_piece[w] = vec4(p0.x, p0.y, p1.x, p1.y) }
                                            else { self.out_piece[w] = vec4(p1.x, p1.y, p0.x, p0.y) }
                                            self.out_info[w] = vec4u(info.x, self.grid.w + o, self.mterm[i], fy * self.nodes_x() + fx)
                                        } else {
                                            self.overflow()
                                        }
                                    }
                                    cnt[o] += 1u
                                }
                            }
                        }
                    }
                }
            }
            if !write {
                for o in 0u..no { self.offs[e * no + o] = cnt[o] }
            }
        }
    }

    // Over the capacity: past the round's cells, 0 for the scan (and the closing entry).
    mod.sdf.ComposeCount = {
        ..mod.sdf.Compose
        main: fn() {
            let e = self.element()
            if e == 4294967295u || e >= self.words.w { return }
            let no = self.outputs()
            if e == 0u { self.offs[self.words.w * no] = 0u }
            if e < self.counts[self.words.x] { self.run(e, false) }
            else { for o in 0u..no { self.offs[e * no + o] = 0u } }
        }
    }

    mod.sdf.ComposeFill = {
        ..mod.sdf.Compose
        main: fn() {
            let e = self.element()
            if e == 4294967295u || e >= self.counts[self.words.x] { return }
            self.run(e, true)
        }
    }

    // The keys and geo of every output piece, each output's side at every round fine cell's low
    // node (`out_side`, per output feature_field's node_side layout: slot * 121 + ly * 11 + lx,
    // valid at the low nodes), and the any-words. words: x the count word of R·100, y the first
    // any-word, z the overflow word.
    mod.sdf.ComposeKeys = {
        ..mod.sdf.ComposeBase
        rinfo: shader.buffer(vec4u)
        rslot: shader.buffer(u32)
        side: shader.buffer(u32)
        ooff: shader.buffer(u32)
        out_piece: shader.buffer(vec4f)
        out_key: shader.buffer_rw(vec4u)
        out_geo: shader.buffer_rw(vec4f)
        out_side: shader.buffer_rw(u32)
        overflow: fn() { atomic_add(self.counts[self.words.z], 1u) }
        // Output o's pieces in fine cell (gx, gy) of plane p, as [first, end); empty off the round.
        outs: fn(p: u32, gx: u32, gy: u32, o: u32) -> vec2u {
            if gx >= self.grid.x * 10u || gy >= self.grid.y * 10u { return vec2u(0u, 0u) }
            let rs = self.rslot[p * self.cells() + (gy / 10u) * self.grid.x + gx / 10u]
            if rs == 4294967295u { return vec2u(0u, 0u) }
            let at = (rs * 100u + (gy % 10u) * 10u + gx % 10u) * self.outputs() + o
            return vec2u(self.ooff[at], self.ooff[at + 1u])
        }
        end: fn(i: u32, k: u32) -> vec2 {
            let g = self.out_piece[i]
            if k == 0u { return g.xy }
            return g.zw
        }
        // The rank of v among output o's distinct points inside the fine edge on grid line `at`
        // (x = at for axis 1, y = at for axis 0), span `span`, from the two cells beside it.
        edge_rank: fn(p: u32, o: u32, axis: u32, at: u32, span: u32, v: f32) -> u32 {
            var vals = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var nv = 0u
            let line = f32(at)
            for beside in 0u..2u {
                if beside == 1u || at > 0u {
                    var r = vec2u(0u, 0u)
                    if axis == 1u { r = self.outs(p, at + beside - 1u, span, o) } else { r = self.outs(p, span, at + beside - 1u, o) }
                    for i in r.x..r.y {
                        for k in 0u..2u {
                            let q = self.end(i, k)
                            var on = false
                            var w = q.x
                            if axis == 1u {
                                on = q.x == line && floor(q.y) != q.y && u32(floor(q.y)) == span
                                w = q.y
                            } else {
                                on = q.y == line && floor(q.x) != q.x && u32(floor(q.x)) == span
                            }
                            if on {
                                if nv < 32u { vals[nv] = w; nv += 1u } else { self.overflow() }
                            }
                        }
                    }
                }
            }
            var rank = 0u
            for k in 0u..nv {
                let a = vals[k]
                if a < v {
                    var dup = false
                    for j in 0u..k { if vals[j] == a { dup = true } }
                    if !dup { rank += 1u }
                }
            }
            return rank
        }
        // The rank of q among the distinct interior points of pieces [r.x, r.y), by x then y.
        cell_rank: fn(r: vec2u, q: vec2) -> u32 {
            var xs = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var ys = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var nv = 0u
            for i in r.x..r.y {
                for k in 0u..2u {
                    let p = self.end(i, k)
                    if floor(p.x) != p.x && floor(p.y) != p.y {
                        if nv < 32u { xs[nv] = p.x; ys[nv] = p.y; nv += 1u } else { self.overflow() }
                    }
                }
            }
            var rank = 0u
            for k in 0u..nv {
                if xs[k] < q.x || (xs[k] == q.x && ys[k] < q.y) {
                    var dup = false
                    for j in 0u..k { if xs[j] == xs[k] && ys[j] == ys[k] { dup = true } }
                    if !dup { rank += 1u }
                }
            }
            return rank
        }
        key: fn(p: u32, o: u32, fx: u32, fy: u32, r: vec2u, q: vec2) -> vec2u {
            let lv = (self.grid.w + o) << 24u
            let nx = self.nodes_x()
            let xi = floor(q.x) == q.x
            let yi = floor(q.y) == q.y
            if xi && yi { return vec2u(2u * (u32(q.y) * nx + u32(q.x)), lv | 16777215u) }
            if xi {
                let x = u32(q.x)
                let span = u32(floor(q.y))
                return vec2u(2u * (span * nx + x) + 1u, lv | self.edge_rank(p, o, 1u, x, span, q.y))
            }
            if yi {
                let y = u32(q.y)
                let span = u32(floor(q.x))
                return vec2u(2u * (y * nx + span), lv | self.edge_rank(p, o, 0u, y, span, q.x))
            }
            return vec2u(2147483648u | (fy * nx + fx), lv | self.cell_rank(r, q))
        }
        main: fn() {
            let e = self.element()
            if e == 4294967295u || e >= self.counts[self.words.x] { return }
            let info = self.rinfo[e / 100u]
            let local = e % 100u
            let fx = info.y * 10u + local % 10u
            let fy = info.z * 10u + local / 10u
            let no = self.outputs()
            let st = self.state(info.x)
            let node = (e / 100u) * 121u + (local / 10u) * 11u + local % 10u
            for o in 0u..no {
                let t = self.table(info.x, st, o)
                self.out_side[o * self.round[2u] * 121u + node] = self.table_bit(self.side[e], t.x, t.y)
                let r = vec2u(self.ooff[e * no + o], self.ooff[e * no + o + 1u])
                if r.y > r.x { atomic_max(self.counts[self.words.y + info.x * no + o], 1u) }
                for i in r.x..r.y {
                    let g = self.out_piece[i]
                    let from = self.key(info.x, o, fx, fy, r, g.xy)
                    let to = self.key(info.x, o, fx, fy, r, g.zw)
                    self.out_key[i] = vec4u(from.x, from.y, to.x, to.y)
                    self.out_geo[i] = vec4(0.0, 0.0, 0.0, 0.0)
                }
            }
        }
    }
    mod.sdf
}

#[cfg(test)]
mod tests {
    #[test]
    fn tables_index_terms_by_bit() {
        // A & !B over two terms: only bits = 0b01
        assert_eq!(super::table(2, |b| b & 1 != 0 && b & 2 == 0), [0b0010, 0]);
        // any of six terms: every entry but 0
        assert_eq!(super::table(6, |b| b != 0), [u32::MAX - 1, u32::MAX]);
        assert_eq!(super::plane_offset(-1), 15);
    }
}
