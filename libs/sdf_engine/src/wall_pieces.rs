//! Level pieces of a feature field (GPU-EXEC §10.4): the curves d' = λ of a FeatureSet's
//! biased signed distance, cut into closed-form pieces per band fine cell and level slot, keyed
//! so the chain stage links them. Generic over the levels: walls, the base, region terms.
//!
//! In a cell, d' on side σ is −min v (inside) or +min v (outside) over the cell's cand list, so
//! the level curve is the boundary of the union of the features' regions {v_g < ρ_g}, each
//! bounded by the feature's own curve m_g = ρ_g, with ρ = −λ − b inside and λ + b outside (b
//! its loop's bias): a seg's offset line, a reflex vertex's arc (Round) or its miter lines and
//! square cut (Miter). One thread per band fine cell walks its level slots:
//! 1. reject: |d'(centre) − λ| > the cell's half diagonal (d' is 1-Lipschitz);
//! 2. crossings: along each edge (in +axis order) every part's cuts with the edge are the
//!    candidate zeros of d' − λ; one is a crossing where d' <= λ differs at the points beside
//!    it (midpoints to the next candidates, or the edge's nodes). A point on the level counts
//!    below. Both cells of an edge list every feature nearest anywhere on it and evaluate the
//!    same points, so they agree on every crossing, its coordinate and its rank. A crossing's
//!    coordinate is a node's only where d'(node) = λ exactly; it then lies just inside its edge;
//! 3. strands: from each crossing where the curve enters the cell (keeping {d' <= λ} on its
//!    left), follow the part through it that heads in (or runs along the edge) and is on the
//!    level a step on. Its next event is the earliest of leaving the cell (snapped to that edge's
//!    crossing, or at a node to the node's other edge), its domain end (the join's next part, or
//!    the next seg's line across a straight join), or entering another part's region (a
//!    junction). Ties go exit, domain end, junction. A strand that grazes a corner node leaves at
//!    once. A vertex's own parts, and a seg's line and its vertices' parts, pass at domain ends
//!    only (they meet tangent or collinear, where f32 junctions are noise); a vertex whose f32
//!    turn is under 1e-3 joins straight (its parts differ from that by under r·turn²/2).
//!    Tolerances grow with the cell's coordinates (a few f32 ulps).
//!
//! A point's side is the field's own (the crossing parity from the cell's low node).
//! A slot at exactly 0 gives the FeatureSet's own loops: each seg is a piece, keyed by the set's
//! seg keys and ordinals, whatever the bias. A vertex on a grid node keeps the set's key (the
//! node's +x edge) with the node ordinal 0xFFFFFF, the convention regions_compose keys node
//! points by: a loop through a node can link diagonal cells, which share no edge, and chain
//! searches a node key's four cells. A level piece's edge keys are always its own cell's edges
//! (a crossing at a node belongs to whichever of the node's edges has it), never node keys. A
//! closed curve that never leaves one fine cell has no crossing and gives no pieces.
//!
//! Pieces: `piece` (a, b), `piece_geo` (centre, r, signed sweep, CCW positive; r = 0 for a
//! line), `piece_key` (from hi, lo, to hi, lo): an edge crossing is (edge id, key_lo(level,
//! rank)), a junction (CELL_BIT | cell id, key_lo(level, j)) with j in the cell's discovery
//! order; `piece_info` (plane, level, feature, cell id), the feature a seg or
//! plane_abi::vertex_feature. Grouped by `piece_off` over (band fine cell · slots + slot).
use makepad_draw::*;
use crate::feature_field::FieldOut;
use crate::gpu_jobs::{label, BufRef, CountWord, Counts, Elements, JobError, Pass, Pool, ScanKernels, Step};

/// The extract's count words: the pieces' total, then its invariants (each must stay 0).
pub const WORD_PIECES: u32 = 0;
/// Strands that hit the step cap.
pub const WORD_STEP_CAP: u32 = 1;
/// Exits with no crossing on their edge.
pub const WORD_SNAP: u32 = 2;
/// Domain ends at a corner that is neither reflex on the strand's side nor straight.
pub const WORD_CONVEX_END: u32 = 3;
/// Entering crossings with no part to follow.
pub const WORD_NO_PART: u32 = 4;
/// Cell slots whose fill emits other than the count's pieces.
pub const WORD_FILL: u32 = 5;
pub const WORDS: u32 = 6;
/// Level slots one extract takes.
pub const MAX_LEVELS: usize = 64;

/// The extract's levels: a level id per slot, and per plane each slot's d' (grid units, planes ×
/// slots, plane-major), NaN where the slot is unused on that plane.
#[derive(Clone, Copy, Debug)]
pub struct Levels<'a> {
    pub ids: &'a [u32],
    pub values: &'a [f32],
}

/// The extract's kernels and the scan, made once per Cx.
#[derive(Clone)]
pub struct PieceKernels { cells: ComputeKernel, count: ComputeKernel, fill: ComputeKernel, outline: ComputeKernel, scan: ScanKernels }
impl PieceKernels {
    pub fn new(cx: &mut Cx) -> Result<Self, String> {
        let [cells, count, fill, outline] = cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            let mut kernel = |name: LiveId, label: &str| -> Result<ComputeKernel, String> {
                let object = vm.value(module, name).as_object().ok_or_else(|| format!("{label} kernel missing"))?;
                ComputeKernel::new(vm, object, label)
            };
            Ok::<_, String>([kernel(id!(PiecesCells), "pieces cells")?, kernel(id!(PiecesCount), "pieces count")?,
                kernel(id!(PiecesFill), "pieces fill")?, kernel(id!(PiecesOutline), "pieces outline")?])
        })?;
        Ok(Self { cells, count, fill, outline, scan: ScanKernels::new(cx)? })
    }
}

/// The extract's buffers once its steps ran, for the chain stage.
#[derive(Clone, Copy, Debug)]
pub struct PiecesOut {
    pub piece: BufRef,
    pub piece_geo: BufRef,
    pub piece_key: BufRef,
    pub piece_info: BufRef,
    /// (band fine cell · slots + slot) → first piece; one more entry holds the total.
    pub piece_off: BufRef,
    /// Per band fine cell: (fx | fy << 16, plane | low node side << 31, its set segs [first, end)).
    pub cell_info: BufRef,
    /// The pieces' count word, and the pieces allocated.
    pub count: CountWord,
    pub capacity: u32,
    /// Level slots per band fine cell in `piece_off`.
    pub levels_per_cell: u32,
    /// Band fine cells (band slots · 100).
    pub cells: u32,
    /// The first of the extract's [`WORDS`] counts words.
    pub words: CountWord,
}

impl PiecesOut {
    /// The pieces as the chain stage binds them, over the field they were extracted from.
    pub fn chain_input(&self, field: &FieldOut) -> crate::chain::Pieces {
        crate::chain::Pieces { piece: self.piece, piece_geo: self.piece_geo, piece_key: self.piece_key, piece_info: self.piece_info,
            piece_off: self.piece_off, band_slot: field.band_slot, count: self.count, capacity: self.capacity, levels_per_cell: self.levels_per_cell }
    }
    /// The extract's invariants in counts read after its fill.
    pub fn verify(&self, counts: &Counts) -> Result<(), JobError> {
        for (w, what) in [(WORD_STEP_CAP, "strands over the step cap"), (WORD_SNAP, "exits without a crossing"),
            (WORD_CONVEX_END, "domain ends at convex corners"), (WORD_NO_PART, "entries without a part"), (WORD_FILL, "cell slots the fill and count disagree on")] {
            let n = counts.word(CountWord(self.words.0 + w));
            if n != 0 { return Err(JobError::Failed(format!("wall pieces: {n} {what}"))); }
        }
        Ok(())
    }
}

/// The extract's counts words, claimed before its item starts.
pub fn claim(pool: &mut Pool) -> Result<CountWord, JobError> { pool.claim("wall pieces", WORDS) }

fn le(words: impl IntoIterator<Item = u32>) -> Vec<u8> { words.into_iter().flat_map(u32::to_le_bytes).collect() }

/// The extract's steps over a field whose band is sized (run them from the field's `then`):
/// the cells' info, the count, the scan, a host step at the pieces checkpoint that sizes the
/// pieces, the fills, then `then`'s steps. The levels lie within the field's depths.
pub fn steps(k: &PieceKernels, pool: &mut Pool, field: &FieldOut, words: CountWord, levels: &Levels,
    then: impl FnOnce(&Counts, &mut Pool, &PiecesOut) -> Result<Vec<Step>, JobError> + 'static) -> Result<Vec<Step>, JobError> {
    let fail = |what: String| Err(JobError::Failed(format!("wall pieces: {what}")));
    let (slots, planes) = (levels.ids.len() as u32, field.params.planes);
    if slots == 0 || slots as usize > MAX_LEVELS { return fail(format!("{slots} level slots (1 to {MAX_LEVELS})")); }
    if levels.values.len() != (planes * slots) as usize { return fail(format!("{} level values for {planes} planes × {slots} slots", levels.values.len())); }
    if let Some(id) = levels.ids.iter().find(|&&id| id > 255) { return fail(format!("level id {id} over 255")); }
    if field.params.nx * 10 >= 1 << 16 || field.params.ny * 10 >= 1 << 16 { return fail(format!("grid {}x{} over 6553 coarse cells", field.params.nx, field.params.ny)); }
    let cells = field.slots * 100;
    let entries = cells as u64 * slots as u64 + 1;
    if entries > (u32::MAX / 4) as u64 { return fail(format!("{entries} cell slots")); }
    let level_buf = pool.buffer("pieces levels", (4 + slots as u64 * (1 + planes as u64)) * 4);
    pool.upload(level_buf, 0, le([slots, words.0, planes, cells].into_iter().chain(levels.ids.iter().copied()).chain(levels.values.iter().map(|v| v.to_bits()))));
    let cell_info = pool.buffer("pieces cell info", cells.max(1) as u64 * 16);
    let piece_count = pool.buffer("piece count", entries * 4);
    let piece_off = pool.buffer("piece off", entries * 4);
    let f = *field;
    let bind = move |b: LiveId, extra: &[(LiveId, BufRef)]| -> BufRef {
        if b == id!(levels) { return level_buf; }
        if b == id!(cell_info) { return cell_info; }
        if let Some(e) = extra.iter().find(|e| e.0 == b) { return e.1; }
        f.bind(b).unwrap_or(BufRef::RUNTIME)
    };
    let mut steps = vec![
        Step::Pass(Pass::new(&k.cells, label("pieces cells"), Elements::Fixed(cells), 16, 1 << 18, |b| bind(b, &[]))),
        Step::Pass(Pass::new(&k.count, label("pieces count"), Elements::Fixed(cells), 64 * slots, 8192, |b| bind(b, &[(id!(piece_count), piece_count)]))),
    ];
    steps.extend(k.scan.exclusive(pool, piece_count, piece_off, entries as u32, CountWord(words.0 + WORD_PIECES)));
    let (fill, outline) = (k.fill.clone(), k.outline.clone());
    steps.push(Step::Host(Box::new(move |counts: &Counts, pool: &mut Pool| {
        let pieces = counts.word(CountWord(words.0 + WORD_PIECES));
        let bytes = pieces.max(1) as u64 * 16;
        let out = PiecesOut {
            piece: pool.buffer("piece", bytes), piece_geo: pool.buffer("piece geo", bytes),
            piece_key: pool.buffer("piece key", bytes), piece_info: pool.buffer("piece info", bytes),
            piece_off, cell_info, count: CountWord(words.0 + WORD_PIECES), capacity: pieces, levels_per_cell: slots, cells, words,
        };
        let extra = [(id!(piece_off), piece_off), (id!(piece), out.piece), (id!(piece_geo), out.piece_geo), (id!(piece_key), out.piece_key), (id!(piece_info), out.piece_info)];
        let mut steps = vec![
            Step::Pass(Pass::new(&fill, label("pieces fill"), Elements::Fixed(cells), 64 * slots, 8192, |b| bind(b, &extra))),
            Step::Pass(Pass::new(&outline, label("pieces outline"), Elements::Fixed(cells), 4 * slots, 1 << 16, |b| bind(b, &extra))),
        ];
        steps.extend(then(counts, pool, &out)?);
        Ok(steps)
    })));
    Ok(steps)
}

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.Pieces = {
        ..mod.sdf.Field
        // [0] slots, [1] counts base word, [2] planes, [3] band fine cells; [4 + l] slot l's level
        // id; [4 + slots + plane · slots + l] the f32 bits of its d' on that plane (NaN: unused).
        levels: shader.buffer(u32)
        // Per band fine cell: (fx | fy << 16, plane | low node side << 31, its set segs [first, end)).
        cell_info: shader.buffer(vec4u)
        // Per piece (count: nothing; fill: write), per defect (count: tally), per cell slot.
        emit: fn(at: u32, k: u32, a: vec2, b: vec2, geo: vec4, key: vec4u, info: vec4u) {}
        defect: fn(w: u32) {}
        level_done: fn(at: u32, n: u32) {}
        outline: fn(at: u32, n: u32) {}
        // The plane of band slot `slot`: the last plane whose first slot is at most it.
        slot_plane: fn(slot: u32) -> u32 {
            var lo = 0u
            var hi = self.field[0u].z
            for bit in 0..32u {
                if hi - lo <= 1u { break }
                let mid = lo + (hi - lo) / 2u
                if self.band_off[mid] <= slot { lo = mid } else { hi = mid }
            }
            return lo
        }
        cell_fx: fn(ci: vec4u) -> u32 { return ci.x & 65535u }
        cell_fy: fn(ci: vec4u) -> u32 { return ci.x >> 16u }
        cell_plane: fn(ci: vec4u) -> u32 { return ci.y & 2147483647u }
        // A few ulps of the cell's coordinates: f32 positions there are no finer.
        slack: fn(ci: vec4u) -> f32 { return 0.0000005 * f32(max(self.cell_fx(ci), self.cell_fy(ci)) + 1u) }
        // How far a parameter or coordinate may stray past its range and still count.
        tol: fn(ci: vec4u) -> f32 { return 0.0001 + self.slack(ci) }
        // Whether parts of `side` can lie in the cell: both sides where the outline passes.
        side_ok: fn(ci: vec4u, side: u32) -> bool { return ci.w > ci.z || side == (ci.y >> 31u) }
        // The side (1 inside) of q in the cell, as the field's point_side finds it.
        side_of: fn(ci: vec4u, q: vec2) -> u32 {
            var side = ci.y >> 31u
            let x = f32(self.cell_fx(ci))
            let y0 = f32(self.cell_fy(ci))
            for s in ci.z..ci.w {
                let g = self.seg[s]
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
        // d' at q on its side, from the cell's cand list.
        dval: fn(cr: vec2u, q: vec2, side: u32) -> f32 {
            var best = 1.0e30
            for k in cr.x..cr.y {
                let f = self.cand[k]
                best = min(best, self.value(f & 2147483647u, (f & 2147483648u) != 0u, q, side == 1u))
            }
            if side == 1u { return 0.0 - best }
            return best
        }
        // A point on the level counts below.
        below: fn(ci: vec4u, cr: vec2u, q: vec2, lam: f32) -> bool {
            return self.dval(cr, q, self.side_of(ci, q)) <= lam
        }
        squared: fn(s: u32) -> bool { return (self.seg_flags(s) & 4u) != 0u }
        // A vertex whose f32 turn is too small for its parts (or disagrees with its reflex side)
        // joins its segs' lines straight: they differ there by under r·turn²/2.
        vertex_flat: fn(s: u32) -> bool {
            let a = self.seg_e(self.seg_prev(s))
            let b = self.seg_e(s)
            let c = a.x * b.y - a.y * b.x
            if self.vertex_inside(s) { return c > 0.0 - 0.001 }
            return c < 0.001
        }
        // Parts the walk passes between at domain ends, never at junctions: a vertex's own
        // parts, and a seg's line with its two vertices' parts (they meet tangent or collinear).
        adjacent: fn(s: u32, kind: u32, g: u32, gkind: u32) -> bool {
            if kind != 0u && gkind != 0u { return s == g }
            if kind == 0u && gkind != 0u { return g == s || g == self.seg_next(s) }
            if kind != 0u && gkind == 0u { return g == s || g == self.seg_prev(s) }
            return false
        }
        // ρ of cand feature f's part `kind` (0 seg line, 1 arc, 2 line 1, 3 line 2, 4 square
        // cut) on a side at level lam, or 0 where that part does not exist.
        part_rho: fn(f: u32, kind: u32, side: u32, lam: f32) -> f32 {
            let s = f & 2147483647u
            var r = lam + self.seg_bias(s)
            if side == 1u { r = 0.0 - lam - self.seg_bias(s) }
            if r <= 0.0001 { return 0.0 }
            if (f & 2147483648u) == 0u {
                if kind == 0u && (self.seg_flags(s) & 8u) != 0u { return r }
                return 0.0
            }
            if kind == 0u || (self.seg_flags(s) & 3u) == 0u || self.vertex_inside(s) != (side == 1u) || self.vertex_flat(s) { return 0.0 }
            if self.miter() {
                if kind == 2u || kind == 3u || (kind == 4u && self.squared(s)) { return r }
                return 0.0
            }
            if kind == 1u { return r }
            return 0.0
        }
        // A line part (kinds 0, 2, 3, 4): (its point at parameter 0, unit direction of travel).
        part_line: fn(s: u32, kind: u32, side: u32, r: f32) -> vec4 {
            if kind == 0u {
                let e = self.seg_e(s)
                var w = self.left(e)
                if side == 0u { w = vec2(0.0) - w }
                let p = self.seg_a(s) + w * r
                return vec4(p.x, p.y, e.x, e.y)
            }
            let v = self.seg_a(s)
            if kind == 2u {
                let p = v + self.vertex_m1(s) * r
                let d = self.seg_e(self.seg_prev(s))
                return vec4(p.x, p.y, d.x, d.y)
            }
            if kind == 3u {
                let p = v + self.vertex_m2(s) * r
                let d = self.seg_e(s)
                return vec4(p.x, p.y, d.x, d.y)
            }
            let u = self.vertex_u(s)
            let p = v + u * r
            if side == 1u { return vec4(p.x, p.y, u.y, 0.0 - u.x) }
            return vec4(p.x, p.y, 0.0 - u.y, u.x)
        }
        // The direction a line part's feature value grows in.
        part_normal: fn(s: u32, kind: u32, side: u32) -> vec2 {
            if kind == 0u {
                if side == 1u { return self.left(self.seg_e(s)) }
                return vec2(0.0) - self.left(self.seg_e(s))
            }
            if kind == 2u { return self.vertex_m1(s) }
            if kind == 3u { return self.vertex_m2(s) }
            return self.vertex_u(s)
        }
        // A part's parameter range: a line's travel from its point, an arc's angle from m1.
        part_span: fn(s: u32, kind: u32, side: u32, r: f32) -> vec2 {
            if kind == 0u { return vec2(0.0, self.seg_len(s)) }
            let m1 = self.vertex_m1(s)
            let m2 = self.vertex_m2(s)
            if kind == 1u { return vec2(0.0, acos(clamp(dot(m1, m2), 0.0 - 1.0, 1.0))) }
            let ep = self.seg_e(self.seg_prev(s))
            let es = self.seg_e(s)
            let u = self.vertex_u(s)
            // line 1 ends at the miter point (or the square cut), line 2 starts there
            var n1 = m2
            var n2 = m1
            if self.squared(s) { n1 = u; n2 = u }
            let d1 = dot(n1, ep)
            let d2 = dot(n2, es)
            var l1 = 0.0
            var l2 = 0.0
            if abs(d1) > 1.0e-12 { l1 = r * (1.0 - dot(n1, m1)) / d1 }
            if abs(d2) > 1.0e-12 { l2 = r * (1.0 - dot(n2, m2)) / d2 }
            if kind == 2u { return vec2(0.0, l1) }
            if kind == 3u { return vec2(l2, 0.0) }
            let q = self.part_line(s, 4u, side, r)
            let end1 = self.seg_a(s) + m1 * r + ep * l1
            let start2 = self.seg_a(s) + m2 * r + es * l2
            return vec2(dot(end1 - q.xy, q.zw), dot(start2 - q.xy, q.zw))
        }
        arc_dir: fn(side: u32) -> f32 {
            if side == 1u { return 0.0 - 1.0 }
            return 1.0
        }
        // An angle on vertex s's arc as its parameter, in [-π, π) from m1 along the travel.
        arc_param: fn(s: u32, side: u32, a: f32) -> f32 {
            let m1 = self.vertex_m1(s)
            let d = self.arc_dir(side) * (a - atan2(m1.y, m1.x))
            return d - 6.2831855 * floor((d + 3.1415927) / 6.2831855)
        }
        // The point and unit tangent of a part at parameter t.
        part_at: fn(s: u32, kind: u32, side: u32, r: f32, t: f32) -> vec4 {
            if kind == 1u {
                let m1 = self.vertex_m1(s)
                let dir = self.arc_dir(side)
                let a = atan2(m1.y, m1.x) + dir * t
                let c = self.seg_a(s)
                return vec4(c.x + r * cos(a), c.y + r * sin(a), 0.0 - dir * sin(a), dir * cos(a))
            }
            let l = self.part_line(s, kind, side, r)
            return vec4(l.x + t * l.z, l.y + t * l.w, l.z, l.w)
        }
        // A part's cuts with the grid line {p[ax] = c}: (parameter, coordinate along the line)
        // twice, 1e30 where none.
        part_cut: fn(s: u32, kind: u32, side: u32, r: f32, ax: u32, c: f32) -> vec4 {
            if kind == 1u {
                let o = self.seg_a(s)
                var oc = o.y
                var oa = o.x
                if ax == 1u { oc = o.x; oa = o.y }
                let h = c - oc
                if abs(h) > r { return vec4(1.0e30) }
                let w = sqrt(max(r * r - h * h, 0.0))
                var a1 = atan2(h, w)
                var a2 = atan2(h, 0.0 - w)
                if ax == 1u { a1 = atan2(w, h); a2 = atan2(0.0 - w, h) }
                return vec4(self.arc_param(s, side, a1), oa + w, self.arc_param(s, side, a2), oa - w)
            }
            let l = self.part_line(s, kind, side, r)
            var pc = l.y
            var pa = l.x
            var dc = l.w
            var da = l.z
            if ax == 1u { pc = l.x; pa = l.y; dc = l.z; da = l.w }
            if abs(dc) < 1.0e-12 { return vec4(1.0e30) }
            let t = (c - pc) / dc
            return vec4(t, pa + t * da, 1.0e30, 1.0e30)
        }
        // The part through q the level goes on along: of the parts through q heading across
        // `inward` or along it (any, for a zero vector), the least |d' − λ| a step along, where
        // another feature's region covers a part that is not on the level; the first in cand
        // order on ties, and a part ending at q only when none goes on. (seg or vertex, kind, its
        // parameter at q, the f32 bits of its |d' − λ|), NONE when no part passes q.
        pick: fn(ci: vec4u, cr: vec2u, lam: f32, side: u32, q: vec2, inward: vec2) -> vec4u {
            let tol = self.tol(ci)
            var best = vec4u(4294967295u, 0u, 0u, 0u)
            var bm = 1.0e30
            for k in cr.x..cr.y {
                let f = self.cand[k]
                let fs = f & 2147483647u
                for kind in 0..5u {
                    let r = self.part_rho(f, kind, side, lam)
                    if r > 0.0 {
                        let span = self.part_span(fs, kind, side, r)
                        let pn = self.part_near(fs, kind, side, r, q)
                        if pn.y <= 0.00002 + self.slack(ci) && pn.x >= span.x - tol && pn.x <= span.y + tol {
                            if dot(self.part_at(fs, kind, side, r, pn.x).zw, inward) >= 0.0 - 0.000001 {
                                let m = self.ahead(ci, cr, lam, side, fs, kind, pn.x)
                                if m < bm { bm = m; best = vec4u(fs, kind, asuint(pn.x), asuint(m)) }
                            }
                        }
                    }
                }
            }
            return best
        }
        // |d' − λ| a step along a part from t (1e20 where it ends within the step).
        ahead: fn(ci: vec4u, cr: vec2u, lam: f32, side: u32, s: u32, kind: u32, t: f32) -> f32 {
            let r = self.part_rho(self.cand_feature(s, kind), kind, side, lam)
            let span = self.part_span(s, kind, side, r)
            var step = 0.001
            if kind == 1u { step = 0.001 / r }
            step = min(step, span.y - t)
            if step <= 0.000001 { return 1.0e20 }
            let q = self.part_at(s, kind, side, r, t + step).xy
            return abs(self.dval(cr, q, self.side_of(ci, q)) - lam)
        }
        // The parameter of q's nearest point on a part's curve, and q's distance from the curve.
        part_near: fn(s: u32, kind: u32, side: u32, r: f32, q: vec2) -> vec2 {
            if kind == 1u {
                let v = q - self.seg_a(s)
                return vec2(self.arc_param(s, side, atan2(v.y, v.x)), abs(length(v) - r))
            }
            let l = self.part_line(s, kind, side, r)
            let v = q - l.xy
            return vec2(dot(v, l.zw), abs(v.x * l.w - v.y * l.z))
        }
        // Two parts' meeting points on one side: (F parameter, G parameter) twice, 1e30 where none.
        part_meet: fn(s: u32, kind: u32, side: u32, r: f32, g: u32, gkind: u32, rg: f32) -> vec4 {
            if kind != 1u && gkind != 1u {
                let lf = self.part_line(s, kind, side, r)
                let lg = self.part_line(g, gkind, side, rg)
                // nearly parallel lines meet nowhere reliable in f32: a straight join or the
                // join's own parts carry the curve across instead
                let den = lf.z * lg.w - lf.w * lg.z
                if abs(den) < 0.001 { return vec4(1.0e30) }
                let q = lg.xy - lf.xy
                return vec4((q.x * lg.w - q.y * lg.z) / den, (q.x * lf.w - q.y * lf.z) / den, 1.0e30, 1.0e30)
            }
            if gkind == 1u && kind != 1u {
                let lf = self.part_line(s, kind, side, r)
                let c = self.seg_a(g)
                let m = lf.xy - c
                let b = dot(lf.zw, m)
                let disc = b * b - (dot(m, m) - rg * rg)
                if disc < 0.0 { return vec4(1.0e30) }
                let t1 = 0.0 - b - sqrt(disc)
                let t2 = 0.0 - b + sqrt(disc)
                let p1 = m + lf.zw * t1
                let p2 = m + lf.zw * t2
                return vec4(t1, self.arc_param(g, side, atan2(p1.y, p1.x)), t2, self.arc_param(g, side, atan2(p2.y, p2.x)))
            }
            if kind == 1u && gkind != 1u {
                let lg = self.part_line(g, gkind, side, rg)
                let c = self.seg_a(s)
                let m = lg.xy - c
                let b = dot(lg.zw, m)
                let disc = b * b - (dot(m, m) - r * r)
                if disc < 0.0 { return vec4(1.0e30) }
                let u1 = 0.0 - b - sqrt(disc)
                let u2 = 0.0 - b + sqrt(disc)
                let p1 = m + lg.zw * u1
                let p2 = m + lg.zw * u2
                return vec4(self.arc_param(s, side, atan2(p1.y, p1.x)), u1, self.arc_param(s, side, atan2(p2.y, p2.x)), u2)
            }
            let c1 = self.seg_a(s)
            let c2 = self.seg_a(g)
            let d = c2 - c1
            let dd = length(d)
            if dd < 1.0e-9 || dd > r + rg || dd < abs(r - rg) { return vec4(1.0e30) }
            let a = (r * r - rg * rg + dd * dd) / (2.0 * dd)
            let h = sqrt(max(r * r - a * a, 0.0))
            let m = c1 + d * (a / dd)
            let n = vec2(0.0 - d.y, d.x) / dd
            let p1 = m + n * h
            let p2 = m - n * h
            return vec4(self.arc_param(s, side, atan2(p1.y - c1.y, p1.x - c1.x)), self.arc_param(g, side, atan2(p1.y - c2.y, p1.x - c2.x)),
                self.arc_param(s, side, atan2(p2.y - c1.y, p2.x - c1.x)), self.arc_param(g, side, atan2(p2.y - c2.y, p2.x - c2.x)))
        }
        // Edge e of the cell (0 bottom, 1 right, 2 top, 3 left, each in +axis order): (axis of
        // its line, the line's coordinate, where it starts along the line).
        edge_geo: fn(ci: vec4u, e: u32) -> vec3 {
            let fx = self.cell_fx(ci)
            let fy = self.cell_fy(ci)
            if e == 0u { return vec3(0.0, f32(fy), f32(fx)) }
            if e == 1u { return vec3(1.0, f32(fx + 1u), f32(fy)) }
            if e == 2u { return vec3(0.0, f32(fy + 1u), f32(fx)) }
            return vec3(1.0, f32(fx), f32(fy))
        }
        edge_key: fn(ci: vec4u, e: u32) -> u32 {
            var gx = self.cell_fx(ci)
            var gy = self.cell_fy(ci)
            var axis = 0u
            if e == 1u { gx += 1u; axis = 1u }
            if e == 2u { gy += 1u }
            if e == 3u { axis = 1u }
            return 2u * (gy * self.nodes_x() + gx) + axis
        }
        edge_point: fn(g: vec3, along: f32) -> vec2 {
            if g.x == 0.0 { return vec2(along, g.y) }
            return vec2(g.y, along)
        }
        // The cell's outward normal across edge e.
        edge_out: fn(e: u32) -> vec2 {
            if e == 0u { return vec2(0.0, 0.0 - 1.0) }
            if e == 1u { return vec2(1.0, 0.0) }
            if e == 2u { return vec2(0.0, 1.0) }
            return vec2(0.0 - 1.0, 0.0)
        }
        // The smallest cut of a cand part with edge g past `after` (1e30 when none).
        next_along: fn(ci: vec4u, cr: vec2u, lam: f32, g: vec3, after: f32) -> f32 {
            let tol = self.tol(ci)
            var best = 1.0e30
            let ax = u32(g.x)
            for k in cr.x..cr.y {
                let f = self.cand[k]
                let s = f & 2147483647u
                for side in 0..2u {
                    if self.side_ok(ci, side) {
                        for kind in 0..5u {
                            let r = self.part_rho(f, kind, side, lam)
                            if r > 0.0 {
                                let span = self.part_span(s, kind, side, r)
                                let cut = self.part_cut(s, kind, side, r, ax, g.y)
                                if cut.x >= span.x - tol && cut.x <= span.y + tol && cut.y >= g.z && cut.y <= g.z + 1.0 && cut.y > after { best = min(best, cut.y) }
                                if cut.z >= span.x - tol && cut.z <= span.y + tol && cut.w >= g.z && cut.w <= g.z + 1.0 && cut.w > after { best = min(best, cut.w) }
                            }
                        }
                    }
                }
            }
            return best
        }
        // Whether d' is below the level just before the first candidate u on edge g.
        before_start: fn(ci: vec4u, cr: vec2u, lam: f32, g: vec3, u: f32) -> bool {
            if u == g.z { return self.below(ci, cr, self.edge_point(g, g.z), lam) }
            return self.below(ci, cr, self.edge_point(g, (g.z + min(u, g.z + 1.0)) * 0.5), lam)
        }
        // Whether d' is below the level just past candidate u on edge g (next: the candidate after).
        after_at: fn(ci: vec4u, cr: vec2u, lam: f32, g: vec3, u: f32, next: f32) -> bool {
            if u == g.z + 1.0 { return self.below(ci, cr, self.edge_point(g, u), lam) }
            return self.below(ci, cr, self.edge_point(g, (u + min(next, g.z + 1.0)) * 0.5), lam)
        }
        // The rank and coordinate bits of the leaving crossing on edge e within 1e-4 of `along`,
        // rank NONE when there is none.
        exit_rank: fn(ci: vec4u, cr: vec2u, lam: f32, e: u32, along: f32) -> vec2u {
            let g = self.edge_geo(ci, e)
            var rank = 0u
            var best = vec2u(4294967295u, 0u)
            var dist = self.tol(ci)
            var u = self.next_along(ci, cr, lam, g, g.z - 1.0)
            var before = self.before_start(ci, cr, lam, g, u)
            for it in 0..256u {
                if u > 1.0e29 { break }
                let next = self.next_along(ci, cr, lam, g, u)
                let after = self.after_at(ci, cr, lam, g, u, next)
                if before != after {
                    // leaving: below to above on the bottom and right edges, above to below on the others
                    if (e < 2u) != (before && !after) && abs(u - along) <= dist { dist = abs(u - along); best = vec2u(rank, asuint(u)) }
                    rank += 1u
                }
                before = after
                u = next
            }
            return best
        }
        key_lo: fn(level: u32, ordinal: u32) -> u32 { return (level << 24u) | ordinal }
        // A strand piece's from key: its entry crossing until the strand's first junction (j0 its
        // first junction's number), then the junction before j.
        from_hi: fn(ci: vec4u, e0: u32, cell: u32, j: u32, j0: u32) -> u32 {
            if j == j0 { return self.edge_key(ci, e0) }
            return 2147483648u | cell
        }
        from_lo: fn(lid: u32, rank0: u32, j: u32, j0: u32) -> u32 {
            if j == j0 { return self.key_lo(lid, rank0) }
            return self.key_lo(lid, j - 1u)
        }
        piece_feature: fn(s: u32, kind: u32) -> u32 {
            if kind == 0u { return s }
            return 2147483648u | ((kind - 1u) << 29u) | s
        }
        cand_feature: fn(s: u32, kind: u32) -> u32 {
            if kind == 0u { return s }
            return s | 2147483648u
        }
        // The pieces of one strand entering at `along` on edge e0 (its rank0-th crossing):
        // returns the cell's (junctions, pieces) so far, counted on from jk.
        strand: fn(ci: vec4u, cr: vec2u, lam: f32, lid: u32, e0: u32, along: f32, rank0: u32, at: u32, jk: vec2u) -> vec2u {
            let g0 = self.edge_geo(ci, e0)
            let x0 = self.edge_point(g0, along)
            let side = self.side_of(ci, x0)
            let inward = vec2(0.0) - self.edge_out(e0)
            let tol = self.tol(ci)
            // the part through the crossing heading into the cell (or along its edge) that is on
            // the level past it
            let first = self.pick(ci, cr, lam, side, x0, inward)
            var s = first.x
            var kd = first.y
            var t = asfloat(first.z)
            if s == 4294967295u { self.defect(4u); return jk }
            let plane = self.cell_plane(ci)
            let cell = self.cell_fy(ci) * self.nodes_x() + self.cell_fx(ci)
            // a step this short is no progress (f32 at the grid's coordinates, ~1e-5)
            let eps = 0.00001 + 0.25 * self.slack(ci)
            var j = jk.x
            var n = jk.y
            var a = x0
            // a piece's from key is the strand's entry, then the junction the last piece ended at
            // (derived, not carried: a key var carried through the loop lost its first value on
            // one Vulkan driver)
            let j0 = jk.x
            for step in 0..64u {
                let r = self.part_rho(self.cand_feature(s, kd), kd, side, lam)
                let span = self.part_span(s, kd, side, r)
                // leaving the cell: a cut ahead (or at the part's start, as after a join just
                // across an edge or at a grazed corner) heading out, within the part and the edge
                var t_exit = 1.0e30
                var e_exit = 4u
                var a_exit = 0.0
                for e in 0..4u {
                    let g = self.edge_geo(ci, e)
                    let cut = self.part_cut(s, kd, side, r, u32(g.x), g.y)
                    for root in 0..2u {
                        var tc = cut.x
                        var ac = cut.y
                        if root == 1u { tc = cut.z; ac = cut.w }
                        if tc >= t - tol && tc < t_exit && tc <= span.y + tol && ac >= g.z - tol && ac <= g.z + 1.0 + tol {
                            if dot(self.part_at(s, kd, side, r, tc).zw, self.edge_out(e)) > 0.000001 { t_exit = tc; e_exit = e; a_exit = ac }
                        }
                    }
                }
                // entering another part's region
                var t_in = 1.0e30
                var gs = 0u
                var gk = 0u
                var gt = 0.0
                for k in cr.x..cr.y {
                    let gf = self.cand[k]
                    let gseg = gf & 2147483647u
                    for gkind in 0..5u {
                        let rg = self.part_rho(gf, gkind, side, lam)
                        if rg > 0.0 && !(gf == self.cand_feature(s, kd) && gkind == kd) && !self.adjacent(s, kd, gseg, gkind) {
                            let gspan = self.part_span(gseg, gkind, side, rg)
                            let meet = self.part_meet(s, kd, side, r, gseg, gkind, rg)
                            for root in 0..2u {
                                var tf = meet.x
                                var tg = meet.y
                                if root == 1u { tf = meet.z; tg = meet.w }
                                if tf > t + eps && tf < t_in && tg >= gspan.x - tol && tg <= gspan.y + tol {
                                    let pf = self.part_at(s, kd, side, r, tf)
                                    var grow = vec2(0.0)
                                    if gkind == 1u { grow = normalize(pf.xy - self.seg_a(gseg)) }
                                    else { grow = self.part_normal(gseg, gkind, side) }
                                    if dot(pf.zw, grow) < 0.0 - 0.000001 { t_in = tf; gs = gseg; gk = gkind; gt = tg }
                                }
                            }
                        }
                    }
                }
                var geo = vec4(0.0)
                if t_exit < 1.0e29 && t_exit <= t_in + eps {
                    var snap = self.exit_rank(ci, cr, lam, e_exit, a_exit)
                    if snap.x == 4294967295u {
                        // at a node the level leaves by whichever of its two edges has the crossing
                        let p = self.edge_point(self.edge_geo(ci, e_exit), a_exit)
                        for e in 0..4u {
                            let g = self.edge_geo(ci, e)
                            var across = p.y
                            var on = p.x
                            if g.x == 1.0 { across = p.x; on = p.y }
                            if snap.x == 4294967295u && e != e_exit && abs(across - g.y) <= tol && on >= g.z - tol && on <= g.z + 1.0 + tol {
                                let other = self.exit_rank(ci, cr, lam, e, on)
                                if other.x != 4294967295u { snap = other; e_exit = e }
                            }
                        }
                    }
                    if snap.x == 4294967295u { self.defect(2u); return vec2u(j, n) }
                    let b = self.edge_point(self.edge_geo(ci, e_exit), asfloat(snap.y))
                    if kd == 1u { geo = vec4(self.seg_a(s).x, self.seg_a(s).y, r, self.arc_dir(side) * (t_exit - t)) }
                    self.emit(at, n, a, b, geo, vec4u(self.from_hi(ci, e0, cell, j, j0), self.from_lo(lid, rank0, j, j0), self.edge_key(ci, e_exit), self.key_lo(lid, snap.x)),
                        vec4u(plane, lid, self.piece_feature(s, kd), cell))
                    return vec2u(j, n + 1u)
                }
                let ends = span.y <= t_in + eps
                var t_to = t_in
                if ends { t_to = span.y }
                let b = self.part_at(s, kd, side, r, t_to).xy
                if t_to > t + eps {
                    if kd == 1u { geo = vec4(self.seg_a(s).x, self.seg_a(s).y, r, self.arc_dir(side) * (t_to - t)) }
                    let to = vec2u(2147483648u | cell, self.key_lo(lid, j))
                    self.emit(at, n, a, b, geo, vec4u(self.from_hi(ci, e0, cell, j, j0), self.from_lo(lid, rank0, j, j0), to.x, to.y), vec4u(plane, lid, self.piece_feature(s, kd), cell))
                    j += 1u
                    n += 1u
                    a = b
                }
                if ends {
                    if kd == 0u {
                        // the seg's end: the next vertex's join where it is reflex on this side,
                        // else across a straight join onto the next seg's line
                        let nx = self.seg_next(s)
                        if (self.seg_flags(nx) & 3u) != 0u && self.vertex_inside(nx) == (side == 1u) && !self.vertex_flat(nx) {
                            s = nx
                            kd = 2u
                            if !self.miter() { kd = 1u }
                            t = 0.0
                        } else {
                            let rn = self.part_rho(nx, 0u, side, lam)
                            if rn <= 0.0 { self.defect(3u); return vec2u(j, n) }
                            let l = self.part_line(nx, 0u, side, rn)
                            let tn = dot(b - l.xy, l.zw)
                            let off = b - l.xy - l.zw * tn
                            if dot(off, off) > (0.001 + self.slack(ci)) * (0.001 + self.slack(ci)) { self.defect(3u); return vec2u(j, n) }
                            s = nx
                            kd = 0u
                            t = tn
                        }
                    } else if kd == 1u || kd == 3u {
                        kd = 0u
                        t = 0.0
                    } else {
                        var next = 3u
                        if kd == 2u && self.squared(s) { next = 4u }
                        kd = next
                        t = self.part_span(s, next, side, self.part_rho(s | 2147483648u, next, side, lam)).x
                    }
                    // the next part can be covered from its start (another vertex's collinear
                    // miter, a neighbour's region): the level then goes on along the part through
                    // b that is on it
                    let m = self.ahead(ci, cr, lam, side, s, kd, t)
                    if m > tol && m < 1.0e19 {
                        let other = self.pick(ci, cr, lam, side, b, vec2(0.0))
                        if other.x != 4294967295u && asfloat(other.w) < m { s = other.x; kd = other.y; t = asfloat(other.z) }
                    }
                } else {
                    s = gs
                    kd = gk
                    t = gt
                }
            }
            self.defect(1u)
            return vec2u(j, n)
        }
        // A cell's pieces at one level: every strand from the cell's entering crossings.
        level_pieces: fn(ci: vec4u, cr: vec2u, lam: f32, lid: u32, at: u32) -> u32 {
            let centre = vec2(f32(self.cell_fx(ci)) + 0.5, f32(self.cell_fy(ci)) + 0.5)
            if abs(self.dval(cr, centre, self.side_of(ci, centre)) - lam) > 0.7072 + 0.001 { return 0u }
            var jk = vec2u(0u, 0u)
            for e in 0..4u {
                let g = self.edge_geo(ci, e)
                var rank = 0u
                var u = self.next_along(ci, cr, lam, g, g.z - 1.0)
                var before = self.before_start(ci, cr, lam, g, u)
                for it in 0..256u {
                    if u > 1.0e29 { break }
                    let next = self.next_along(ci, cr, lam, g, u)
                    let after = self.after_at(ci, cr, lam, g, u, next)
                    if before != after {
                        // entering: below to above on the bottom and right edges, above to below on the others
                        if (e < 2u) == (before && !after) { jk = self.strand(ci, cr, lam, lid, e, u, rank, at, jk) }
                        rank += 1u
                    }
                    before = after
                    u = next
                }
            }
            return jk.y
        }
        walk: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let ci = self.cell_info[i]
            let cr = vec2u(self.cand_off[i], self.cand_off[i + 1u])
            let slots = self.levels[0u]
            let base = 4u + slots + self.cell_plane(ci) * slots
            for l in 0..slots {
                let bits = self.levels[base + l]
                let at = i * slots + l
                if (bits & 2147483647u) > 2139095040u { self.level_done(at, 0u) }
                else if (bits & 2147483647u) == 0u { self.outline(at, ci.w - ci.z) }
                else { self.level_done(at, self.level_pieces(ci, cr, asfloat(bits), self.levels[4u + l], at)) }
            }
        }
    }

    mod.sdf.PiecesCells = {
        ..mod.sdf.Pieces
        cell_info: shader.buffer_rw(vec4u)
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let slot = i / 100u
            let lx = (i % 100u) % 10u
            let ly = (i % 100u) / 10u
            let p = self.slot_plane(slot)
            let coarse = self.band[slot]
            let fx = (coarse % self.field[0u].x) * 10u + lx
            let fy = (coarse / self.field[0u].x) * 10u + ly
            let r = self.cell_segs(p, fx, fy)
            let side = self.node_side[slot * 121u + ly * 11u + lx]
            self.cell_info[i] = vec4u(fx | (fy << 16u), p | (side << 31u), r.x, r.y)
        }
    }

    mod.sdf.PiecesCount = {
        ..mod.sdf.Pieces
        piece_count: shader.buffer_rw(u32)
        counts: shader.buffer_rw(u32)
        defect: fn(w: u32) { atomic_add(self.counts[self.levels[1u] + w], 1u) }
        level_done: fn(at: u32, n: u32) { self.piece_count[at] = n }
        outline: fn(at: u32, n: u32) { self.piece_count[at] = n }
        main: fn() {
            // the scan's extra value, so piece_off ends with the total
            if self.index() == 0u { self.piece_count[self.levels[3u] * self.levels[0u]] = 0u }
            self.walk()
        }
    }

    mod.sdf.PiecesFill = {
        ..mod.sdf.Pieces
        piece_off: shader.buffer(u32)
        piece: shader.buffer_rw(vec4f)
        piece_geo: shader.buffer_rw(vec4f)
        piece_key: shader.buffer_rw(vec4u)
        piece_info: shader.buffer_rw(vec4u)
        counts: shader.buffer_rw(u32)
        level_done: fn(at: u32, n: u32) {
            if n != self.piece_off[at + 1u] - self.piece_off[at] { atomic_add(self.counts[self.levels[1u] + 5u], 1u) }
        }
        emit: fn(at: u32, k: u32, a: vec2, b: vec2, geo: vec4, key: vec4u, info: vec4u) {
            let i = self.piece_off[at] + k
            if i < self.piece_off[at + 1u] {
                self.piece[i] = vec4f(a.x, a.y, b.x, b.y)
                self.piece_geo[i] = geo
                self.piece_key[i] = key
                self.piece_info[i] = info
            }
        }
        main: fn() { self.walk() }
    }

    // The slots at 0: the cell's set segs as pieces, keyed by their seg keys and ordinals.
    mod.sdf.PiecesOutline = {
        ..mod.sdf.Pieces
        piece_off: shader.buffer(u32)
        piece: shader.buffer_rw(vec4f)
        piece_geo: shader.buffer_rw(vec4f)
        piece_key: shader.buffer_rw(vec4u)
        piece_info: shader.buffer_rw(vec4u)
        seg_key: shader.buffer(u32)
        // The set's ordinal of seg s's start vertex; a vertex on a node keeps the set's key (the
        // node's +x edge) with the node ordinal, so chain searches the node's four cells.
        ordinal: fn(s: u32) -> u32 {
            let a = self.seg[s].xy
            if fract(a.x) == 0.0 && fract(a.y) == 0.0 { return 16777215u }
            return (self.seg_info[s].w >> 8u) & 255u
        }
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            let ci = self.cell_info[i]
            let slots = self.levels[0u]
            let base = 4u + slots + self.cell_plane(ci) * slots
            let cell = self.cell_fy(ci) * self.nodes_x() + self.cell_fx(ci)
            for l in 0..slots {
                if (self.levels[base + l] & 2147483647u) == 0u {
                    let lid = self.levels[4u + l]
                    let first = self.piece_off[i * slots + l]
                    for s in ci.z..ci.w {
                        let next = self.seg_info[s].y
                        let k = first + s - ci.z
                        self.piece[k] = self.seg[s]
                        self.piece_geo[k] = vec4f(0.0)
                        self.piece_key[k] = vec4u(self.seg_key[s], self.key_lo(lid, self.ordinal(s)), self.seg_key[next], self.key_lo(lid, self.ordinal(next)))
                        self.piece_info[k] = vec4u(self.cell_plane(ci), lid, s, cell)
                    }
                }
            }
        }
    }

    mod.sdf
}
