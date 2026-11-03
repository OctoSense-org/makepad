//! Rings (GPU-EXEC §10.3, DESIGN W1): from the chained wall pieces of a batch to each layer's
//! `plan::LayerPlan`, in walls::rings' shape. Per base piece, wall k exists when level e_k has a
//! loop in it; the count stops at the first missing k and is capped by the plan's rules. Wall k's
//! lines are the centre level d_k's loops of the piece, and the fill is level e_c's loops (the
//! piece itself when it has no walls). The thick test sends an island whose walls the plain
//! levels cannot print to the thin stage: its walls are None.
//!
//! Passes, after the chain (P = the pieces' count word, L = the chain's loops):
//! 1. `RingClear` over the loop and position capacities.
//! 1a. With a cover (top one wall: a base piece more than 99 % uncovered by the next layer's
//!    outline gets one wall), over the cover chain (CP its pieces, CL its loops): `RingTopClear`,
//!    `RingTopPick` CP (a cover loop's base piece from its pieces cut from the base),
//!    `RingTopRay` CL (the rest by a ray in the wall chain), `RingTopArea` CL (each closed cover
//!    loop's signed area into its base piece) and `RingTopBase` L (each base piece's own area,
//!    holes negative).
//! 2. `RingCount` L: each base piece's highest edge level e_k holding a contour. The levels are
//!    nested erosions, so the highest k is also the first missing one, less one.
//! 3. `RingCap` L: the plan's count rules per base piece (the layer's walls; one wall where top
//!    one wall holds), and for Round the bead count rule of the thick test.
//! 3a. `RingVertCount` P, the scan of its counts and `RingVertFill` P: the vertex table of the
//!    pieces the thick test walks (wall centres and their margin levels), each one's lattice
//!    points and the vertex after its end, made once for the batch.
//! 4. `RingThick` P and `RingThickFull` over its list: every point of each existing wall's centre
//!    level d_k must lie within s_k / 2 + m of the margin level e_k + m under the Miter metric
//!    (DESIGN W1; m = the thin margin, Miter: 0, so the margin level is e_k itself): then d_k is
//!    walls::rings' opening. Each centre polyline segment is covered by the strips and
//!    convex-join wedges of the base piece's margin segments nearby: first those around the
//!    segment moved in by the reach, and only for a piece where they fall short (corners sharper
//!    than 90 degrees, thin spots) all within the Miter reach.
//! 5. `RingSelect` L: the roles (outline, wall k line, fill) and the thin islands.
//! 6. Scans: the selected loops (their table rows) and the thin islands (their list).
//! 7. `RingEmit` P: each selected piece's ring point count at its chain position.
//! 8. The scan of the counts over positions, then 9. `RingLoops` L: each loop's first point,
//!    point count and rotation (`loops`.zw).
//! 10. `RingPoints` P: `ring_pt` (grid units, GPU-resident) and the nm points of the readback,
//!     and the loops' nm bboxes.
//! 11. `RingThinList` L and 12. `RingTable` L: the thin island list and the table rows.
//!
//! Readback `ring_out`: the points (i32 nm, x then y, relative to the grid origin), then one
//! TABLE_WORDS row per selected loop in loop id order. Its used length is counts word USED.
use crate::chain::{Chain, ChainGrid, Pieces};
use crate::gpu_jobs::{label, BufRef, Counts, CountWord, Elements, JobError, Output, Pass, Pool, ScanKernels, Step};
use crate::plane_abi::{wall_level, WallLevel, HALO, LEVEL_THIN, MAX_WALLS, NONE};
use makepad_csg_sdf::plan::{Join, LayerPlan, IslandPlan, Loop, PlanSpec, WallLine, WallPlan};
use makepad_draw::*;

/// The rings' count words, from their claimed base.
pub const WORD_TABLE: u32 = 0;
pub const WORD_POINTS: u32 = 1;
/// Words of `ring_out` in use: the Output's length.
pub const WORD_USED: u32 = 2;
/// Points and vertex table entries past their capacities (the batch re-runs with K doubled).
pub const WORD_OVERFLOW: u32 = 3;
/// Islands whose walls failed the thick test (their outline loop ids are in `thin_list`).
pub const WORD_THIN: u32 = 4;
/// Centre pieces the near thick test left to the far one (`RingThickFull`'s list).
pub const WORD_THICK: u32 = 5;
/// The thick test's vertex table entries.
pub const WORD_VERTS: u32 = 6;
pub const WORDS: u32 = 7;

/// Roles of a selected loop (a loop can be the outline and the fill at once).
pub const ROLE_OUTLINE: u32 = 1;
pub const ROLE_WALL: u32 = 2;
pub const ROLE_FILL: u32 = 4;
/// Table flags.
pub const FLAG_HOLE: u32 = 1;
pub const FLAG_THIN: u32 = 2;
pub const FLAG_OPEN: u32 = 4;
/// A table row: loop id, plane, island, base piece, parent, level | role << 8 | wall << 12 |
/// flags << 16, points, first point, area (mm², f32), its base piece's area (mm², f32), bbox nm.
pub const TABLE_WORDS: u32 = 14;
/// Ring points per piece at K = 1.
pub const POINTS_PER_PIECE: u32 = 1;
/// Vertex table entries per piece at K = 1 (a line takes one: the vertex after its end).
pub const VERTS_PER_PIECE: u32 = 2;

/// The plan of one plane, as rings applies it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RingPlane {
    /// The layer index (HALO: a halo plane, never emitted).
    pub layer: u32,
    /// The layer's wall count and wall set.
    pub walls: u32,
    pub set: u32,
    /// Top one wall applies to this layer (never layer 0), and the last layer always has one wall.
    pub top: bool,
    pub last: bool,
}

/// The plan's walls in grid units, per wall set: (centre d, spacing s, width w, reach s/2 + m).
#[derive(Clone, Debug, PartialEq)]
pub struct RingSpec {
    pub round: bool,
    /// The thick test's Miter metric limit (Round plans: 3, walls::rings' opening join).
    pub miter_limit: f32,
    pub sets: Vec<Vec<[f32; 4]>>,
    pub planes: Vec<RingPlane>,
}

/// The rings' constants for a batch whose plane p samples layer `plane_layer[p]` (or HALO).
pub fn ring_spec(spec: &PlanSpec, plane_layer: &[u32]) -> RingSpec {
    let round = matches!(spec.join, Join::Round);
    let m = if round { spec.thin_margin } else { 0.0 };
    let last = spec.zs.len().saturating_sub(1) as u32;
    let sets = spec.wall_sets.iter().map(|set| set.iter().take(MAX_WALLS as usize)
        .map(|w| [w.inset * 10.0, w.spacing * 10.0, w.width * 10.0, (w.spacing * 0.5 + m) * 10.0]).collect()).collect();
    let planes = plane_layer.iter().map(|&layer| {
        if layer == HALO { return RingPlane { layer, walls: 0, set: 0, top: false, last: false } }
        let set = spec.layer_set[layer as usize] as u32;
        RingPlane { layer, walls: (spec.wall_sets[set as usize].len() as u32).min(MAX_WALLS), set,
            top: spec.top_one_wall && layer != 0, last: spec.top_one_wall && layer != 0 && layer == last }
    }).collect();
    let miter_limit = match spec.join { Join::Miter(limit) => limit, Join::Round => 3.0 };
    RingSpec { round, miter_limit, sets, planes }
}

/// The `ring` constants buffer: three globals rows, a row per plane, MAX_WALLS rows per wall
/// set, then the level slot of each wall level (below LEVEL_THIN) in the pieces' CSR, four a
/// row. `levels` are the pieces' level ids by slot. Base pieces are the contours of the walls'
/// reference level (the base, or the outline when the plan has no base level), and that level
/// resolves against itself.
pub fn params_words(spec: &RingSpec, words: CountWord, points_cap: u32, verts_cap: u32, loops: u32, positions: u32, grid: &ChainGrid, levels: &[u32]) -> Vec<u32> {
    let base = grid.reference[wall_level(1, WallLevel::Centre) as usize];
    debug_assert!(base != NONE && grid.reference[base as usize] == base, "walls resolve against a base level resolving against itself");
    let mut w = vec![words.0, points_cap, loops, positions, grid.outline_level, base, spec.planes.len() as u32, verts_cap,
        spec.round as u32, spec.miter_limit.to_bits(), 0, 0];
    for p in &spec.planes {
        let flags = (p.layer != HALO) as u32 | (p.top as u32) << 1 | (p.last as u32) << 2;
        w.extend([p.walls, flags, p.set, p.layer]);
    }
    for set in &spec.sets {
        for k in 0..MAX_WALLS as usize {
            w.extend(set.get(k).copied().unwrap_or([0.0; 4]).map(f32::to_bits));
        }
    }
    w[10] = w.len() as u32 / 4;
    w.extend((0..LEVEL_THIN).map(|level| levels.iter().position(|&l| l == level).map_or(NONE, |s| s as u32)));
    w
}

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.Ring = {
        ..mod.sdf.Chain
        // ring[0]: count words base, points capacity, loop capacity, position capacity;
        // [1]: outline level, base level, planes, vertex table capacity; [2]: round, the thick
        // test's miter limit (f32 bits), the level slot rows' first row, 0; then per plane
        // (walls, flags: 1 emit, 2 top one wall, 4 last layer; wall set, layer); then per wall
        // set and wall (d, s, w, reach) as f32 bits; then the wall levels' slots.
        ring: shader.buffer(vec4u)
        // per wall loop: a base piece's area and its area covered (1/16 grid units², summed
        // with integer atomics: the same in any order)
        top_cover: shader.buffer(i32)
        // the cover chain's piece_info, piece_loop, loop_meta and loop_metric, and per cover
        // loop its base piece
        top_info: shader.buffer(vec4u)
        top_loop: shader.buffer(u32)
        top_meta: shader.buffer(vec4u)
        top_metric: shader.buffer(vec4f)
        top_base: shader.buffer(u32)
        ring_cnt: shader.buffer(u32)
        ring_thin: shader.buffer(u32)
        ring_role: shader.buffer(u32)
        ring_sel: shader.buffer(u32)
        ring_tsel: shader.buffer(u32)
        ring_tab: shader.buffer(u32)
        ring_toff: shader.buffer(u32)
        ring_lp: shader.buffer(vec4u)
        ring_box: shader.buffer(i32)
        pos_n: shader.buffer(u32)
        pos_off: shader.buffer(u32)
        ring_pt: shader.buffer(vec2f)
        ring_out: shader.buffer(u32)
        thin_list: shader.buffer(u32)
        // the thick test's vertex table: entries per piece, their offsets, the entries
        vert_n: shader.buffer(u32)
        vert_off: shader.buffer(u32)
        vert: shader.buffer(vec2f)

        rword: fn(w: u32) -> u32 { return self.ring[0u].x + w }
        // The element of a counted pass over the cover chain (its count word in dispatch_range.w).
        top_item: fn() -> u32 { return self.item(self.dispatch_range.w) }
        outline_level: fn() -> u32 { return self.ring[1u].x }
        base_level: fn() -> u32 { return self.ring[1u].y }
        round: fn() -> bool { return self.ring[2u].x != 0u }
        plane_row: fn(p: u32) -> vec4u { return self.ring[3u + p] }
        // Wall k (1-based) of wall set `set`: (d, s, w, reach) in grid units.
        wall_row: fn(set: u32, k: u32) -> vec4 {
            let r = self.ring[3u + self.ring[1u].z + set * 10u + k - 1u]
            return vec4(asfloat(r.x), asfloat(r.y), asfloat(r.z), asfloat(r.w))
        }
        // Wall levels (below 32): wall k at its centre (0), edge (1) or margin (2).
        wall_of: fn(level: u32) -> u32 {
            if level < 2u || level >= 32u { return 0u }
            return (level - 2u) / 3u + 1u
        }
        which: fn(level: u32) -> u32 { return (level - 2u) % 3u }
        // A level the thick test walks: a wall's centre, or the margin level its test reaches
        // (Miter: the edge, Round: the margin).
        walked: fn(level: u32) -> bool {
            if self.wall_of(level) == 0u { return false }
            let w = self.which(level)
            if w == 0u { return true }
            if self.round() { return w == 2u }
            return w == 1u
        }
        // A piece's ring point count: a head gives its start, every piece its lattice points
        // (from its start, when that is on the lattice and it is no head), an open tail its end.
        emits: fn(k: u32, head: bool) -> vec4 {
            return self.lattice_points(k, !head)
        }
    }

    mod.sdf.RingClear = {
        ..mod.sdf.Ring
        ring_cnt: shader.buffer_rw(u32)
        ring_thin: shader.buffer_rw(u32)
        ring_sel: shader.buffer_rw(u32)
        ring_tsel: shader.buffer_rw(u32)
        ring_box: shader.buffer_rw(i32)
        pos_n: shader.buffer_rw(u32)
        vert_n: shader.buffer_rw(u32)
        top_cover: shader.buffer_rw(i32)
        main: fn() {
            let i = self.fixed()
            if i == 4294967295u { return }
            if i < self.ring[0u].z {
                self.ring_cnt[i] = 0u
                self.ring_thin[i] = 0u
                self.top_cover[i * 2u] = 0
                self.top_cover[i * 2u + 1u] = 0
                // the scans run over the loop capacity: loops past the count select nothing
                self.ring_sel[i] = 0u
                self.ring_tsel[i] = 0u
                self.ring_box[i * 4u] = 2147483647
                self.ring_box[i * 4u + 1u] = 2147483647
                self.ring_box[i * 4u + 2u] = 0 - 2147483647
                self.ring_box[i * 4u + 3u] = 0 - 2147483647
            }
            if i <= self.ring[0u].w {
                self.pos_n[i] = 0u
                self.vert_n[i] = 0u
            }
        }
    }

    // Top one wall's cover: SDFREGIONS' round of the base (term 0, a level term over the wall
    // pieces, so a seg is a wall piece) and the next plane's outline (term 1), chained. Each
    // closed cover loop lies in one base piece.
    mod.sdf.RingTopClear = {
        ..mod.sdf.Ring
        top_base: shader.buffer_rw(u32)
        main: fn() {
            let i = self.fixed()
            if i == 4294967295u { return }
            self.top_base[i] = 4294967295u
        }
    }

    // A cover loop's base piece from its pieces cut from the base (all name the same one).
    mod.sdf.RingTopPick = {
        ..mod.sdf.Ring
        top_base: shader.buffer_rw(u32)
        main: fn() {
            let k = self.top_item()
            if k == 4294967295u { return }
            let l = self.top_loop[k]
            let info = self.top_info[k].z
            if l == 4294967295u || (info & 7u) != 0u { return }
            let wl = self.piece_loop[info >> 3u]
            if wl == 4294967295u { return }
            let b = self.loop_tree[wl].z
            if b != 4294967295u { atomic_min(self.top_base[l], b) }
        }
    }

    // A cover loop with no piece of the base lies strictly inside a base piece: the ray from
    // its rightmost vertex against the wall chain's base level (dispatch_range.z) has no ties.
    mod.sdf.RingTopRay = {
        ..mod.sdf.Ring
        top_base: shader.buffer_rw(u32)
        main: fn() {
            let l = self.top_item()
            if l == 4294967295u { return }
            let meta = self.top_meta[l]
            if (meta.w & 2u) != 0u || self.top_base[l] != 4294967295u { return }
            let base = self.dispatch_range.z
            let px = self.top_metric[l * 2u].y
            let py = self.top_metric[l * 2u].z
            var hit = self.origin_hit(4294967295u, meta.x, base, px, py)
            if hit == 4294967295u { hit = self.ray(4294967295u, meta.x, base, px, py) }
            if hit == 4294967295u { return }
            let h = hit & 2147483647u
            // inside a contour, or outside a hole: in that loop's base piece
            if ((hit & 2147483648u) == 0u) != self.is_hole(h) { self.top_base[l] = self.loop_tree[h].z }
        }
    }

    // Each closed cover loop's signed area (holes negative) into its base piece's cover.
    mod.sdf.RingTopArea = {
        ..mod.sdf.Ring
        top_cover: shader.buffer_rw(i32)
        main: fn() {
            let l = self.top_item()
            if l == 4294967295u { return }
            let b = self.top_base[l]
            // (an atomic's old value would be the body's value: the if ends it)
            if (self.top_meta[l].w & 2u) == 0u && b != 4294967295u {
                atomic_add(self.top_cover[b * 2u + 1u], i32(round(self.top_metric[l * 2u].x * 16.0)))
            }
        }
    }

    // Each base piece's area: its base level loops' signed areas.
    mod.sdf.RingTopBase = {
        ..mod.sdf.Ring
        top_cover: shader.buffer_rw(i32)
        main: fn() {
            let l = self.item(self.loops_word())
            if l == 4294967295u { return }
            let b = self.loop_tree[l].z
            if self.loop_meta[l].y == self.base_level() && !self.is_open(l) && b != 4294967295u {
                atomic_add(self.top_cover[b * 2u], i32(round(self.loop_metric[l * 2u].x * 16.0)))
            }
        }
    }

    mod.sdf.RingCount = {
        ..mod.sdf.Ring
        ring_cnt: shader.buffer_rw(u32)
        main: fn() {
            let l = self.item(self.loops_word())
            if l == 4294967295u { return }
            let level = self.loop_meta[l].y
            let k = self.wall_of(level)
            let b = self.loop_tree[l].z
            if k != 0u && self.which(level) == 1u && !self.is_hole(l) && !self.is_open(l) && b != 4294967295u {
                atomic_max(self.ring_cnt[b], k)
            }
        }
    }

    mod.sdf.RingCap = {
        ..mod.sdf.Ring
        ring_cnt: shader.buffer_rw(u32)
        ring_thin: shader.buffer_rw(u32)
        main: fn() {
            let l = self.item(self.loops_word())
            if l == 4294967295u { return }
            let meta = self.loop_meta[l]
            let tree = self.loop_tree[l]
            // base pieces: the base level's contours
            if meta.y != self.base_level() || self.is_hole(l) || tree.z != l { return }
            let row = self.plane_row(meta.x)
            var rule = row.x
            // top one wall: area minus covered above 0.99 of the area
            let area = f32(self.top_cover[l * 2u])
            let top = area - f32(self.top_cover[l * 2u + 1u]) > 0.99 * area
            if ((row.y & 2u) != 0u && top) || (row.y & 4u) != 0u { rule = min(rule, 1u) }
            let raw = self.ring_cnt[l]
            self.ring_cnt[l] = min(raw, rule)
            // Round: Arachne adds or widens beads where a piece falls short of its count
            if self.round() && raw < rule && tree.y != 4294967295u { atomic_max(self.ring_thin[tree.y], 1u) }
        }
    }

    // The vertex table: a walked piece's entries are its lattice points, then the vertex after
    // its end (its successor's first; an open end's own end).
    mod.sdf.RingVertCount = {
        ..mod.sdf.Ring
        vert_n: shader.buffer_rw(u32)
        main: fn() {
            let k = self.item(self.pieces_word())
            if k == 4294967295u { return }
            if self.walked(self.piece_info[k].y) { self.vert_n[k] = u32(self.lattice(k).y) + 1u }
        }
    }

    mod.sdf.RingVertFill = {
        ..mod.sdf.Ring
        vert: shader.buffer_rw(vec2f)
        main: fn() {
            let k = self.item(self.pieces_word())
            if k == 4294967295u { return }
            let vo = self.vert_off[k]
            let vn = self.vert_off[k + 1u] - vo
            // past the capacity the batch re-runs (WORD_OVERFLOW)
            if vn == 0u || vo + vn > self.ring[1u].w { return }
            let lat = self.lattice(k)
            for j in 1u..vn {
                self.vert[vo + j - 1u] = self.vertex(k, j, lat)
            }
            let nm = self.succ[k]
            var after = self.piece[k].zw
            if nm != 4294967295u { after = self.vertex(nm, 1u, self.lattice(nm)) }
            self.vert[vo + vn - 1u] = after
        }
    }

    mod.sdf.RingThick = {
        ..mod.sdf.Ring
        ring_thin: shader.buffer_rw(u32)
        thick_list: shader.buffer_rw(u32)
        counts: shader.buffer_rw(u32)
        // The part of [0, 1] where a + (b - a) t stays <= limit, cut into `iv`.
        clip: fn(iv: vec2, a: f32, b: f32, limit: f32) -> vec2 {
            var out = iv
            if a > limit && b > limit { return vec2(1.0, 0.0) }
            if a > limit { out.x = max(out.x, (limit - a) / (b - a)) }
            if b > limit { out.y = min(out.y, (limit - a) / (b - a)) }
            return out
        }
        // How far along p -> q, from `end`, the Miter reach of margin segment a -> b covers: its
        // strip, and the wedge of its join to b -> c where that turns left (a convex corner of
        // the material on the left).
        cover: fn(p: vec2, q: vec2, a: vec2, b: vec2, c: vec2, end: f32, reach: f32, err: f32, tol: f32, limit: f32) -> f32 {
            var out = end
            let ab = b - a
            let aa = dot(ab, ab)
            if aa <= 0.0 { return out }
            let n1 = vec2(ab.y, 0.0 - ab.x) / sqrt(aa)
            var iv = vec2(0.0, 1.0)
            iv = self.clip(iv, 0.0 - dot(p - a, ab), 0.0 - dot(q - a, ab), 0.0)
            iv = self.clip(iv, dot(p - a, ab), dot(q - a, ab), aa)
            iv = self.clip(iv, dot(p - a, n1), dot(q - a, n1), reach + err)
            iv = self.clip(iv, 0.0 - dot(p - a, n1), 0.0 - dot(q - a, n1), reach + err)
            if iv.x <= out + tol && iv.y >= iv.x { out = max(out, iv.y) }
            let bc = c - b
            let cc = dot(bc, bc)
            if cc > 0.0 && ab.x * bc.y - ab.y * bc.x > 0.0 {
                let n2 = vec2(bc.y, 0.0 - bc.x) / sqrt(cc)
                // a short span's end error turns its support planes: widen them by that
                let e1 = err * (1.0 + 2.0 * max(max(dot(p - b, ab), dot(q - b, ab)), 0.0) / aa)
                let e2 = err * (1.0 + 2.0 * max(max(0.0 - dot(p - b, bc), 0.0 - dot(q - b, bc)), 0.0) / cc)
                var w = vec2(0.0, 1.0)
                w = self.clip(w, 0.0 - dot(p - b, ab), 0.0 - dot(q - b, ab), 0.0)
                w = self.clip(w, dot(p - b, bc), dot(q - b, bc), 0.0)
                w = self.clip(w, dot(p - b, n1), dot(q - b, n1), reach + e1)
                w = self.clip(w, dot(p - b, n2), dot(q - b, n2), reach + e2)
                // past the miter limit, the square cut
                if 1.0 + dot(n1, n2) <= 2.0 / (limit * limit) {
                    let bis = normalize(n1 + n2)
                    w = self.clip(w, dot(p - b, bis), dot(q - b, bis), reach + err)
                }
                if w.x <= out + tol && w.y >= w.x { out = max(out, w.y) }
            }
            return out
        }
        // The level slot of wall level `level` in the pieces' CSR (NONE: no such slot).
        level_slot: fn(level: u32) -> u32 {
            let r = self.ring[self.ring[2u].z + level / 4u]
            let c = level % 4u
            if c == 0u { return r.x }
            if c == 1u { return r.y }
            if c == 2u { return r.z }
            return r.w
        }
        // The pieces of level slot `slot` in plane p's fine cell (fx, fy) as [first, end).
        slot_range: fn(p: u32, fx: u32, fy: u32, slot: u32) -> vec2u {
            if slot == 4294967295u || fx >= self.cells_x() || fy >= self.chain[0u].y * 10u { return vec2u(0u, 0u) }
            let nx = self.chain[0u].x
            let band = self.band_slot[p * nx * self.chain[0u].y + (fy / 10u) * nx + fx / 10u]
            if band == 4294967295u { return vec2u(0u, 0u) }
            let at = (band * 100u + (fy % 10u) * 10u + fx % 10u) * self.chain[0u].w + slot
            return vec2u(self.piece_off[at], self.piece_off[at + 1u])
        }
        // Vertex j of walked piece k, its table entries from vo with n lattice points: 0 its
        // start, 1..=n its lattice points, n + 1 its end, n + 2 the vertex after it.
        pv: fn(k: u32, vo: u32, n: u32, j: u32) -> vec2 {
            if j == 0u { return self.piece[k].xy }
            if j <= n { return self.vert[vo + j - 1u] }
            if j == n + 1u { return self.piece[k].zw }
            return self.vert[vo + n]
        }
        // How far along p -> q, from `end`, the margin pieces (level slot `slot`) of base piece
        // `base` in the cells of [lo, hi] cover.
        scan: fn(p: vec2, q: vec2, plane: u32, slot: u32, base: u32, lo: vec2u, hi: vec2u, end: f32, reach: f32, err: f32, tol: f32, limit: f32) -> f32 {
            var out = end
            let cap = self.ring[1u].w
            for cy in lo.y..(hi.y + 1u) {
                for cx in lo.x..(hi.x + 1u) {
                    let range = self.slot_range(plane, cx, cy, slot)
                    for m in range.x..range.y {
                        let ml = self.piece_loop[m]
                        if ml != 4294967295u {
                            let vo = self.vert_off[m]
                            let ve = self.vert_off[m + 1u]
                            if self.loop_tree[ml].z == base && ve > vo && ve <= cap {
                                let n = ve - vo - 1u
                                var a = self.piece[m].xy
                                var b = self.pv(m, vo, n, 1u)
                                for j in 1u..(n + 2u) {
                                    let c = self.pv(m, vo, n, j + 1u)
                                    out = self.cover(p, q, a, b, c, out, reach, err, tol, limit)
                                    a = b
                                    b = c
                                }
                            }
                        }
                    }
                }
            }
            return out
        }
        // Whether the margin pieces (level slot `slot`) of base piece `base` in the cells of
        // [lo, hi] (grid units) cover all of p -> q. Each round takes at least one more
        // interval, so the rounds end.
        covers: fn(p: vec2, q: vec2, plane: u32, slot: u32, base: u32, lo: vec2, hi: vec2, reach: f32, err: f32, limit: f32) -> bool {
            let cells = vec2(f32(self.cells_x() - 1u), f32(self.chain[0u].y * 10u - 1u))
            let a = clamp(floor(lo), vec2(0.0), cells)
            let b = clamp(floor(hi), vec2(0.0), cells)
            let tol = err / max(length(q - p), err)
            var covered = 0.0
            var more = true
            while more {
                let end = self.scan(p, q, plane, slot, base, vec2u(u32(a.x), u32(a.y)), vec2u(u32(b.x), u32(b.y)), covered, reach, err, tol, limit)
                if end > covered { covered = end } else { more = false }
                if covered >= 1.0 { more = false }
            }
            return covered >= 1.0
        }
        // The slack of p -> q: an arc chord (CHORD_TOL) on each level, and f32 coordinates.
        slack: fn(p: vec2, q: vec2) -> f32 {
            return 0.1 + 0.0008 + max(max(abs(p.x), abs(p.y)), max(abs(q.x), abs(q.y))) * 0.0000005
        }
        // A centre piece of an existing wall: its margin target level, or NONE.
        thick_target: fn(k: u32) -> u32 {
            let l = self.piece_loop[k]
            if l == 4294967295u { return l }
            let info = self.piece_info[k]
            let kk = self.wall_of(info.y)
            if kk == 0u || self.which(info.y) != 0u { return 4294967295u }
            let tree = self.loop_tree[l]
            if tree.z == 4294967295u || tree.y == 4294967295u { return 4294967295u }
            if kk > self.ring_cnt[tree.z] { return 4294967295u }
            if self.round() { return info.y + 2u }
            return info.y + 1u
        }
        // Near: every segment covered by the margin pieces around it moved in by the reach (the
        // material is on the left): straight runs and corners of 90 degrees or more. A piece
        // that is not goes to RingThickFull's list.
        main: fn() {
            let k = self.item(self.pieces_word())
            if k == 4294967295u { return }
            let target = self.thick_target(k)
            if target == 4294967295u { return }
            let info = self.piece_info[k]
            let base = self.loop_tree[self.piece_loop[k]].z
            let reach = self.wall_row(self.plane_row(info.x).z, self.wall_of(info.y)).w
            let limit = max(asfloat(self.ring[2u].y), 1.0)
            let slot = self.level_slot(target)
            let vo = self.vert_off[k]
            let ve = self.vert_off[k + 1u]
            // past the vertex capacity the batch re-runs
            var far = ve > self.ring[1u].w
            if !far {
                let n = ve - vo - 1u
                var p = self.piece[k].xy
                for j in 1u..(n + 2u) {
                    let q = self.pv(k, vo, n, j)
                    if !far {
                        let d = q - p
                        let len = length(d)
                        if len <= 0.0 { far = true }
                        else {
                            let err = self.slack(p, q)
                            let inward = vec2(0.0 - d.y, d.x) * (reach / len)
                            let r = reach + err
                            if !self.covers(p, q, info.x, slot, base, min(p, q) + inward - r, max(p, q) + inward + r, reach, err, limit) { far = true }
                        }
                    }
                    p = q
                }
            }
            if far {
                let at = atomic_add(self.counts[self.rword(5u)], 1u)
                self.thick_list[at] = k
            }
        }
    }

    // Far: a listed piece's segments against every margin piece within the Miter reach (sharp
    // corners and thin spots). One uncovered sends its island to the thin stage.
    mod.sdf.RingThickFull = {
        ..mod.sdf.RingThick
        main: fn() {
            let i = self.item(self.rword(5u))
            if i == 4294967295u { return }
            let k = self.thick_list[i]
            let target = self.thick_target(k)
            let info = self.piece_info[k]
            let tree = self.loop_tree[self.piece_loop[k]]
            let reach = self.wall_row(self.plane_row(info.x).z, self.wall_of(info.y)).w
            let limit = max(asfloat(self.ring[2u].y), 1.0)
            let slot = self.level_slot(target)
            let vo = self.vert_off[k]
            let ve = self.vert_off[k + 1u]
            var fail = ve > self.ring[1u].w
            if !fail {
                let n = ve - vo - 1u
                var p = self.piece[k].xy
                for j in 1u..(n + 2u) {
                    let q = self.pv(k, vo, n, j)
                    if !fail {
                        let err = self.slack(p, q)
                        let rad = (reach + err) * limit
                        if !self.covers(p, q, info.x, slot, tree.z, min(p, q) - rad, max(p, q) + rad, reach, err, limit) { fail = true }
                    }
                    p = q
                }
            }
            if fail { atomic_max(self.ring_thin[tree.y], 1u) }
        }
    }

    mod.sdf.RingSelect = {
        ..mod.sdf.Ring
        ring_role: shader.buffer_rw(u32)
        ring_sel: shader.buffer_rw(u32)
        ring_tsel: shader.buffer_rw(u32)
        main: fn() {
            let l = self.item(self.loops_word())
            if l == 4294967295u { return }
            let meta = self.loop_meta[l]
            let tree = self.loop_tree[l]
            var role = 0u
            var wall = 0u
            var thin = 0u
            if (self.plane_row(meta.x).y & 1u) != 0u && !self.is_open(l) && tree.y != 4294967295u {
                if self.ring_thin[tree.y] != 0u { thin = 1u }
                if meta.y == self.outline_level() { role = role | 1u }
                if thin == 0u && tree.z != 4294967295u {
                    let c = self.ring_cnt[tree.z]
                    let k = self.wall_of(meta.y)
                    if k != 0u && k <= c && self.which(meta.y) == 0u { role = role | 2u; wall = k }
                    if k != 0u && k == c && self.which(meta.y) == 1u { role = role | 4u }
                    // a piece is its own fill only when its layer asks no walls (walls::rings
                    // with walls asked and none fitting has no fill)
                    if c == 0u && meta.y == self.base_level() && self.plane_row(meta.x).x == 0u { role = role | 4u }
                }
            }
            self.ring_role[l] = role | wall << 8u | thin << 16u
            var sel = 0u
            if role != 0u { sel = 1u }
            self.ring_sel[l] = sel
            var tsel = 0u
            if thin != 0u && tree.y == l { tsel = 1u }
            self.ring_tsel[l] = tsel
        }
    }

    mod.sdf.RingEmit = {
        ..mod.sdf.Ring
        pos_n: shader.buffer_rw(u32)
        main: fn() {
            let k = self.item(self.pieces_word())
            if k == 4294967295u { return }
            let l = self.piece_loop[k]
            if l == 4294967295u { return }
            if (self.ring_role[l] & 255u) == 0u { return }
            let pp = self.piece_pos[k]
            let head = (pp & 2147483648u) != 0u
            let lat = self.emits(k, head)
            var n = u32(lat.y)
            if head { n += 1u }
            if self.succ[k] == 4294967295u { n += 1u }
            self.pos_n[pp & 2147483647u] = n
        }
    }

    mod.sdf.RingLoops = {
        ..mod.sdf.Ring
        ring_lp: shader.buffer_rw(vec4u)
        loops: shader.buffer_rw(vec4u)
        main: fn() {
            let l = self.item(self.loops_word())
            if l == 4294967295u { return }
            let rec = self.loops[l]
            let p0 = self.loop_piece_off[l]
            let first = self.pos_off[p0]
            let n = self.pos_off[p0 + rec.y] - first
            // a loop with no head starts at lattice index 0
            var shift = 0u
            if !self.is_open(l) && self.loop_acc[l * 9u] == 0u && n != 0u { shift = self.loop_acc[l * 9u + 8u] % n }
            self.ring_lp[l] = vec4u(first, n, shift, 0u)
            self.loops[l] = vec4u(rec.x, rec.y, n, first)
        }
    }

    mod.sdf.RingPoints = {
        ..mod.sdf.Ring
        ring_pt: shader.buffer_rw(vec2f)
        ring_out: shader.buffer_rw(u32)
        ring_box: shader.buffer_rw(i32)
        main: fn() {
            let k = self.item(self.pieces_word())
            if k == 4294967295u { return }
            let l = self.piece_loop[k]
            if l == 4294967295u { return }
            let pp = self.piece_pos[k]
            let pos = pp & 2147483647u
            let g0 = self.pos_off[pos]
            let n = self.pos_off[pos + 1u] - g0
            let head = (pp & 2147483648u) != 0u
            let lp = self.ring_lp[l]
            let cap = self.ring[0u].y
            let lat = self.emits(k, head)
            let count = u32(lat.y)
            let q = self.piece[k]
            var h = 0u
            if head { h = 1u }
            var bx0 = 2147483647
            var by0 = 2147483647
            var bx1 = 0 - 2147483647
            var by1 = 0 - 2147483647
            for j in 0u..n {
                var p = q.zw
                if head && j == 0u { p = q.xy }
                else if j - h < count { p = self.lattice_point(k, lat.x + lat.z * f32(j - h), lat.w) }
                let local = g0 + j - lp.x
                let out = lp.x + (local + lp.y - lp.z) % lp.y
                let nx = i32(round(p.x * 100000.0))
                let ny = i32(round(p.y * 100000.0))
                bx0 = min(bx0, nx)
                by0 = min(by0, ny)
                bx1 = max(bx1, nx)
                by1 = max(by1, ny)
                if out < cap {
                    self.ring_pt[out] = p
                    self.ring_out[out * 2u] = u32(nx)
                    self.ring_out[out * 2u + 1u] = u32(ny)
                }
            }
            if n != 0u {
                atomic_min(self.ring_box[l * 4u], bx0)
                atomic_min(self.ring_box[l * 4u + 1u], by0)
                atomic_max(self.ring_box[l * 4u + 2u], bx1)
                atomic_max(self.ring_box[l * 4u + 3u], by1)
            }
        }
    }

    mod.sdf.RingThinList = {
        ..mod.sdf.Ring
        thin_list: shader.buffer_rw(u32)
        main: fn() {
            let l = self.item(self.loops_word())
            if l == 4294967295u { return }
            if self.ring_tsel[l] != 0u { self.thin_list[self.ring_toff[l]] = l }
        }
    }

    mod.sdf.RingTable = {
        ..mod.sdf.Ring
        ring_out: shader.buffer_rw(u32)
        counts: shader.buffer_rw(u32)
        main: fn() {
            let l = self.item(self.loops_word())
            if l == 4294967295u { return }
            let cap = self.ring[0u].y
            let points = self.counts[self.rword(1u)]
            let used = min(points, cap)
            if l == 0u {
                self.counts[self.rword(2u)] = used * 2u + self.counts[self.rword(0u)] * 14u
                let verts = self.counts[self.rword(6u)]
                let vcap = self.ring[1u].w
                var over = 0u
                if points > cap { over = points - cap }
                if verts > vcap { over += verts - vcap }
                if over != 0u { self.counts[self.rword(3u)] = over }
            }
            let rr = self.ring_role[l]
            if (rr & 255u) == 0u { return }
            let meta = self.loop_meta[l]
            let tree = self.loop_tree[l]
            let lp = self.ring_lp[l]
            var flags = 0u
            if self.is_hole(l) { flags = flags | 1u }
            if ((rr >> 16u) & 1u) != 0u { flags = flags | 2u }
            if self.is_open(l) { flags = flags | 4u }
            var base_area = 0.0
            if tree.z != 4294967295u { base_area = abs(self.loop_metric[tree.z * 2u].x) * 0.01 }
            let at = used * 2u + self.ring_tab[l] * 14u
            self.ring_out[at] = l
            self.ring_out[at + 1u] = meta.x
            self.ring_out[at + 2u] = tree.y
            self.ring_out[at + 3u] = tree.z
            self.ring_out[at + 4u] = tree.x
            self.ring_out[at + 5u] = meta.y | (rr & 255u) << 8u | ((rr >> 8u) & 255u) << 12u | flags << 16u
            self.ring_out[at + 6u] = lp.y
            self.ring_out[at + 7u] = lp.x
            self.ring_out[at + 8u] = asuint(abs(self.loop_metric[l * 2u].x) * 0.01)
            self.ring_out[at + 9u] = asuint(base_area)
            self.ring_out[at + 10u] = u32(self.ring_box[l * 4u])
            self.ring_out[at + 11u] = u32(self.ring_box[l * 4u + 1u])
            self.ring_out[at + 12u] = u32(self.ring_box[l * 4u + 2u])
            self.ring_out[at + 13u] = u32(self.ring_box[l * 4u + 3u])
        }
    }

    mod.sdf
}

/// The kernels in pass order.
pub const KERNELS: [&str; 18] = [
    "RingClear", "RingTopClear", "RingTopPick", "RingTopRay", "RingTopArea", "RingTopBase", "RingCount", "RingCap", "RingVertCount",
    "RingVertFill", "RingThick", "RingThickFull", "RingSelect", "RingEmit", "RingLoops", "RingPoints", "RingThinList", "RingTable",
];

/// The rings' kernels with the scan kernels they run, made once.
#[derive(Clone)]
pub struct RingKernels { kernels: Vec<ComputeKernel>, scan: ScanKernels }
impl RingKernels {
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
    fn kernel(&self, name: &str) -> &ComputeKernel { &self.kernels[KERNELS.iter().position(|k| *k == name).unwrap()] }
}

/// The rings' outputs: `ring_pt` and `thin_list` stay on the GPU for the batch; `ring_out` is read back.
#[derive(Clone, Copy, Debug)]
pub struct Rings {
    pub words: CountWord,
    pub points_capacity: u32,
    /// The thick test's vertex table entries allocated.
    pub verts_capacity: u32,
    pub ring_pt: BufRef,
    pub ring_out: BufRef,
    pub out_words: u32,
    pub thin_list: BufRef,
}
impl Rings {
    pub fn word(&self, w: u32) -> CountWord { CountWord(self.words.0 + w) }
    /// The readback of `ring_out`'s used words (counts word USED).
    pub fn output(&self) -> Step { Output::counted(self.ring_out, self.word(WORD_USED), 4, self.out_words) }
}

/// Top one wall's cover: SDFREGIONS' compose round of the base (term 0: `Term::level` over the
/// wall pieces' base level slot, so an output's seg is a wall piece) and the outline one plane
/// up (term 1), chained with `ChainGrid::new(nx, ny, planes, NONE, &[])`.
#[derive(Clone, Copy)]
pub struct Cover<'a> {
    pub chain: &'a Chain,
    pub pieces: &'a Pieces,
}

/// The rings' count words, claimed when the item is built (a stage in a continuation cannot
/// claim them once its item has started).
pub fn claim(pool: &mut Pool) -> Result<CountWord, JobError> { pool.claim("rings", WORDS) }

/// The rings' steps over a chain of wall pieces at re-run factor `k`, counting into `words`
/// ([`claim`]). `levels` are the pieces' level ids by slot. A plan with top one wall takes the
/// cover of its planes.
pub fn steps(kernels: &RingKernels, pool: &mut Pool, words: CountWord, chain: &Chain, pieces: &Pieces, levels: &[u32], grid: &ChainGrid, spec: &RingSpec, cover: Option<Cover>, k: u32) -> Result<(Rings, Vec<Step>), JobError> {
    if cover.is_none() && spec.planes.iter().any(|p| p.top) {
        return Err(JobError::Failed("rings: top one wall without its cover".into()));
    }
    if levels.len() != pieces.levels_per_cell as usize {
        return Err(JobError::Failed(format!("rings: {} level ids for {} level slots", levels.len(), pieces.levels_per_cell)));
    }
    let (nl, np) = (chain.loop_capacity, chain.piece_capacity);
    let capacity = |per_piece: u32| (np as u64 * per_piece as u64 * k.max(1) as u64).min(u32::MAX as u64 / 4) as u32;
    let (points_capacity, verts_capacity) = (capacity(POINTS_PER_PIECE), capacity(VERTS_PER_PIECE));
    let params = params_words(spec, words, points_capacity, verts_capacity, nl, np, grid, levels);
    let ring = pool.buffer("ring params", params.len() as u64 * 4);
    pool.upload(ring, 0, params.iter().flat_map(|w| w.to_le_bytes()).collect());
    let out_words = points_capacity * 2 + nl * TABLE_WORDS;
    let b = |pool: &mut Pool, name: &'static str, n: u32, size: u64| pool.buffer(name, n as u64 * size);
    let (ring_cnt, ring_thin, ring_role, ring_sel, ring_tsel) = (b(pool, "ring cnt", nl, 4), b(pool, "ring thin", nl, 4), b(pool, "ring role", nl, 4), b(pool, "ring sel", nl, 4), b(pool, "ring tsel", nl, 4));
    let (ring_tab, ring_toff, ring_lp, ring_box) = (b(pool, "ring tab", nl, 4), b(pool, "ring toff", nl, 4), b(pool, "ring lp", nl, 16), b(pool, "ring box", nl, 16));
    let (pos_n, pos_off) = (b(pool, "ring pos n", np + 1, 4), b(pool, "ring pos off", np + 1, 4));
    let (vert_n, vert_off, vert) = (b(pool, "ring vert n", np + 1, 4), b(pool, "ring vert off", np + 1, 4), b(pool, "ring vert", verts_capacity.max(1), 8));
    let top_cover = b(pool, "ring top cover", nl, 8);
    let thick_list = b(pool, "ring thick list", np, 4);
    let top_base = cover.map(|c| b(pool, "ring top base", c.chain.loop_capacity, 4));
    let rings = Rings {
        words, points_capacity, verts_capacity, out_words,
        ring_pt: b(pool, "ring pt", points_capacity.max(1), 8),
        ring_out: pool.readback("ring out", out_words as u64 * 4),
        thin_list: b(pool, "thin list", nl, 4),
    };
    let bind = |name: LiveId| -> BufRef {
        match name {
            n if n == id!(piece) => pieces.piece,
            n if n == id!(piece_geo) => pieces.piece_geo,
            n if n == id!(piece_info) => pieces.piece_info,
            n if n == id!(piece_off) => pieces.piece_off,
            n if n == id!(band_slot) => pieces.band_slot,
            n if n == id!(succ) => chain.succ,
            n if n == id!(rank_pred) => chain.ranked.pred,
            n if n == id!(piece_loop) => chain.piece_loop,
            n if n == id!(piece_pos) => chain.piece_pos,
            n if n == id!(loops) => chain.loops,
            n if n == id!(loop_meta) => chain.loop_meta,
            n if n == id!(loop_metric) => chain.loop_metric,
            n if n == id!(loop_acc) => chain.loop_acc,
            n if n == id!(loop_piece_off) => chain.loop_piece_off,
            n if n == id!(loop_tree) => chain.loop_tree,
            n if n == id!(chain) => chain.params,
            n if n == id!(ring) => ring,
            n if n == id!(top_cover) => top_cover,
            n if n == id!(thick_list) => thick_list,
            n if n == id!(top_info) => cover.expect("a top pass has its cover").pieces.piece_info,
            n if n == id!(top_loop) => cover.expect("a top pass has its cover").chain.piece_loop,
            n if n == id!(top_meta) => cover.expect("a top pass has its cover").chain.loop_meta,
            n if n == id!(top_metric) => cover.expect("a top pass has its cover").chain.loop_metric,
            n if n == id!(top_base) => top_base.expect("a top pass has its cover"),
            n if n == id!(ring_cnt) => ring_cnt,
            n if n == id!(ring_thin) => ring_thin,
            n if n == id!(ring_role) => ring_role,
            n if n == id!(ring_sel) => ring_sel,
            n if n == id!(ring_tsel) => ring_tsel,
            n if n == id!(ring_tab) => ring_tab,
            n if n == id!(ring_toff) => ring_toff,
            n if n == id!(ring_lp) => ring_lp,
            n if n == id!(ring_box) => ring_box,
            n if n == id!(pos_n) => pos_n,
            n if n == id!(pos_off) => pos_off,
            n if n == id!(vert_n) => vert_n,
            n if n == id!(vert_off) => vert_off,
            n if n == id!(vert) => vert,
            n if n == id!(ring_pt) => rings.ring_pt,
            n if n == id!(ring_out) => rings.ring_out,
            n if n == id!(thin_list) => rings.thin_list,
            n if n == id!(counts) => BufRef::RUNTIME,
            n => panic!("rings bind no buffer {n:?}"),
        }
    };
    let on_pieces = Elements::Counted { word: pieces.count, bound: np };
    let on_loops = Elements::Counted { word: chain.loops_word(), bound: nl };
    // Ceilings: within 50 ms on the slowest supported GPU. The near thick test scans the cells
    // around a segment moved in by the reach, the far one all within the Miter reach; a piece's
    // points are its lattice points in one cell.
    let pass = |name: &str, elements: Elements, cost: u32, ceiling: u32| Pass::new(kernels.kernel(name), label(name), elements, cost, ceiling, bind);
    let mut steps = vec![
        Step::Pass(pass("RingClear", Elements::Fixed(nl.max(np + 1)), 4, 1 << 20)),
    ];
    if let Some(c) = cover {
        // the cover's ray walks the wall chain's cells as ChainRay does
        let (cp, cl, base) = (c.pieces.count, c.chain.loops_word(), grid.reference[wall_level(1, WallLevel::Centre) as usize]);
        steps.extend([
            Step::Pass(pass("RingTopClear", Elements::Fixed(c.chain.loop_capacity), 4, 1 << 20)),
            Step::Pass(pass("RingTopPick", Elements::Counted { word: cp, bound: c.chain.piece_capacity }, 8, 1 << 20).zw(0, cp.0)),
            Step::Pass(pass("RingTopRay", Elements::Counted { word: cl, bound: c.chain.loop_capacity }, 4096, 1 << 13).zw(base, cl.0)),
            Step::Pass(pass("RingTopArea", Elements::Counted { word: cl, bound: c.chain.loop_capacity }, 4, 1 << 20).zw(0, cl.0)),
            Step::Pass(pass("RingTopBase", on_loops, 8, 1 << 20)),
        ]);
    }
    steps.extend([
        Step::Pass(pass("RingCount", on_loops, 8, 1 << 20)),
        Step::Pass(pass("RingCap", on_loops, 8, 1 << 20)),
        Step::Pass(pass("RingVertCount", on_pieces, 16, 1 << 20)),
    ]);
    steps.extend(kernels.scan.exclusive(pool, vert_n, vert_off, np + 1, rings.word(WORD_VERTS)));
    steps.extend([
        Step::Pass(pass("RingVertFill", on_pieces, 64, 1 << 18)),
        Step::Pass(pass("RingThick", on_pieces, 8192, 1 << 14)),
        Step::Pass(pass("RingThickFull", Elements::Counted { word: rings.word(WORD_THICK), bound: np }, 65536, 1 << 10)),
        Step::Pass(pass("RingSelect", on_loops, 8, 1 << 20)),
    ]);
    steps.extend(kernels.scan.exclusive(pool, ring_sel, ring_tab, nl, rings.word(WORD_TABLE)));
    steps.extend(kernels.scan.exclusive(pool, ring_tsel, ring_toff, nl, rings.word(WORD_THIN)));
    steps.push(Step::Pass(pass("RingEmit", on_pieces, 16, 1 << 20)));
    steps.extend(kernels.scan.exclusive(pool, pos_n, pos_off, np + 1, rings.word(WORD_POINTS)));
    steps.push(Step::Pass(pass("RingLoops", on_loops, 8, 1 << 20)));
    steps.push(Step::Pass(pass("RingPoints", on_pieces, 64, 1 << 18)));
    steps.push(Step::Pass(pass("RingThinList", on_loops, 4, 1 << 20)));
    steps.push(Step::Pass(pass("RingTable", on_loops, 16, 1 << 20)));
    Ok((rings, steps))
}

/// A table row of the readback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    pub loop_id: u32,
    pub plane: u32,
    pub island: u32,
    pub base: u32,
    pub parent: u32,
    pub level: u32,
    pub role: u32,
    pub wall: u32,
    pub flags: u32,
    pub n_points: u32,
    pub first_point: u32,
    pub area_mm2: f32,
    pub base_area_mm2: f32,
    pub bbox: [i32; 4],
}
impl Row {
    pub fn hole(&self) -> bool { self.flags & FLAG_HOLE != 0 }
}

/// The rows of a readback with `points` points and `rows` table rows.
pub fn rows(words: &[u32], points: u32, rows: u32) -> Vec<Row> {
    let at = points as usize * 2;
    (0..rows as usize).map(|i| {
        let w = &words[at + i * TABLE_WORDS as usize..at + (i + 1) * TABLE_WORDS as usize];
        Row { loop_id: w[0], plane: w[1], island: w[2], base: w[3], parent: w[4], level: w[5] & 255, role: (w[5] >> 8) & 15,
            wall: (w[5] >> 12) & 15, flags: w[5] >> 16, n_points: w[6], first_point: w[7], area_mm2: f32::from_bits(w[8]),
            base_area_mm2: f32::from_bits(w[9]), bbox: [w[10] as i32, w[11] as i32, w[12] as i32, w[13] as i32] }
    }).collect()
}

/// The order walls::rings gives the holes of one contour. walls::rings takes each piece as
/// `std::iter::once(e.contour).chain(e.holes)`, and fab's clip.rs groups a region's holes in the
/// order it links their rings: by each ring's smallest undirected edge (a, b), points ordered
/// by (x, y). That is the ring's least vertex, then the lesser of its two neighbours.
fn hole_key(pts: &[[i32; 2]]) -> ([i32; 2], [i32; 2]) {
    let n = pts.len();
    let key = |p: [i32; 2]| (p[0], p[1]);
    let i = (0..n).min_by_key(|&i| key(pts[i])).unwrap_or(0);
    let (a, b) = (pts[(i + n - 1) % n], pts[(i + 1) % n]);
    (pts[i], if key(a) < key(b) { a } else { b })
}

/// The layers' plans from a rings readback: `counts` are the item's counts, `plane_layer` the
/// layer (or HALO) of each plane, `origin_nm` the grid origin in printer nanometres.
pub fn decode(words: &[u32], counts: &Counts, rings: &Rings, spec: &PlanSpec, plane_layer: &[u32], origin_nm: [i64; 2]) -> Result<Vec<(u32, LayerPlan)>, String> {
    let points = counts.word(rings.word(WORD_POINTS));
    if points > rings.points_capacity { return Err(format!("{points} ring points past the capacity {}", rings.points_capacity)); }
    let verts = counts.word(rings.word(WORD_VERTS));
    if verts > rings.verts_capacity { return Err(format!("{verts} thick test vertices past the capacity {}", rings.verts_capacity)); }
    let rows = rows(words, points, counts.word(rings.word(WORD_TABLE)));
    let pts = |r: &Row| -> Vec<[i32; 2]> {
        (r.first_point as usize..(r.first_point + r.n_points) as usize)
            .map(|i| [(words[2 * i] as i32 as i64 + origin_nm[0]) as i32, (words[2 * i + 1] as i32 as i64 + origin_nm[1]) as i32]).collect()
    };
    let ring = |r: &Row| Loop { pts: pts(r), area_mm2: r.area_mm2, hole: r.hole(),
        bbox: [(r.bbox[0] as i64 + origin_nm[0]) as i32, (r.bbox[1] as i64 + origin_nm[1]) as i32, (r.bbox[2] as i64 + origin_nm[0]) as i32, (r.bbox[3] as i64 + origin_nm[1]) as i32] };
    // descending area, ties by loop id (the chain numbers loops by their roots)
    let by_area = |v: &mut Vec<&Row>| v.sort_by(|a, b| b.area_mm2.total_cmp(&a.area_mm2).then(a.loop_id.cmp(&b.loop_id)));
    // a contour, then its holes (rows with this parent and `with`) in walls::rings' order
    let with_holes = |out: &mut Vec<Loop>, c: &Row, holes: &[&Row]| {
        out.push(ring(c));
        let mut h: Vec<(([i32; 2], [i32; 2]), Loop)> = holes.iter().filter(|h| h.parent == c.loop_id).map(|h| { let l = ring(h); (hole_key(&l.pts), l) }).collect();
        h.sort_by(|a, b| (a.0 .0[0], a.0 .0[1], a.0 .1[0], a.0 .1[1]).cmp(&(b.0 .0[0], b.0 .0[1], b.0 .1[0], b.0 .1[1])));
        out.extend(h.into_iter().map(|h| h.1));
    };
    let mut layers = Vec::new();
    for (plane, &layer) in plane_layer.iter().enumerate() {
        if layer == HALO { continue; }
        let on: Vec<&Row> = rows.iter().filter(|r| r.plane == plane as u32).collect();
        let mut islands: Vec<&Row> = on.iter().copied().filter(|r| r.role & ROLE_OUTLINE != 0 && !r.hole()).collect();
        by_area(&mut islands);
        let set = &spec.wall_sets[spec.layer_set[layer as usize]];
        let mut plan = LayerPlan::default();
        for isle in islands {
            let mine: Vec<&Row> = on.iter().copied().filter(|r| r.island == isle.loop_id).collect();
            let mut outline = Vec::new();
            let outline_holes: Vec<&Row> = mine.iter().copied().filter(|r| r.role & ROLE_OUTLINE != 0 && r.hole()).collect();
            with_holes(&mut outline, isle, &outline_holes);
            let walls = (isle.flags & FLAG_THIN == 0).then(|| {
                // base pieces by descending area, ties by loop id
                let mut bases: Vec<(f32, u32)> = mine.iter().filter(|r| r.role & (ROLE_WALL | ROLE_FILL) != 0).map(|r| (r.base_area_mm2, r.base)).collect();
                bases.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                bases.dedup_by_key(|b| b.1);
                let part = |base: u32, role: u32, wall: u32, hole: bool| -> Vec<&Row> {
                    let mut v: Vec<&Row> = mine.iter().copied().filter(|r| r.base == base && r.role & role != 0 && (role != ROLE_WALL || r.wall == wall) && r.hole() == hole).collect();
                    by_area(&mut v);
                    v
                };
                let mut lines: Vec<Vec<WallLine>> = Vec::new();
                for k in 1..=set.len() as u32 {
                    let mut wall = Vec::new();
                    for &(_, base) in &bases {
                        let holes = part(base, ROLE_WALL, k, true);
                        for c in part(base, ROLE_WALL, k, false) {
                            let mut loops = Vec::new();
                            with_holes(&mut loops, c, &holes);
                            // Round plans supply Arachne beads: a bead line stores its spacing
                            let width = match spec.join { Join::Round => set[k as usize - 1].spacing, Join::Miter(_) => set[k as usize - 1].width };
                            wall.extend(loops.into_iter().map(|l| WallLine { closed: true, odd: false, widths: vec![width; l.pts.len()], pts: l.pts }));
                        }
                    }
                    if wall.is_empty() { break; }
                    lines.push(wall);
                }
                let mut fill = Vec::new();
                for &(_, base) in &bases {
                    let holes = part(base, ROLE_FILL, 0, true);
                    for c in part(base, ROLE_FILL, 0, false) { with_holes(&mut fill, c, &holes); }
                }
                WallPlan { lines, fill }
            });
            plan.islands.push(IslandPlan { outline, walls, regions: None });
        }
        layers.push((layer, plan));
    }
    Ok(layers)
}

#[cfg(test)]
mod tests {
    #[test]
    fn hole_order_is_the_least_edge() {
        assert_eq!(super::hole_key(&[[5, 5], [3, 7], [1, 2], [4, 1]]), ([1, 2], [3, 7]));
    }
}
