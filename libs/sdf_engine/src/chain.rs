//! Chaining (GPU-EXEC §10.3): the pieces of any producer (wall levels, region compositions,
//! thin lines) to loops and open chains, their metrics, and containment. Rings and the region
//! stages run it on their own pieces.
//!
//! Input: `piece`, `piece_geo`, `piece_key` and `piece_info`, grouped per band fine cell by
//! `piece_off` (cell-major, then the producer's level slots), and the band's `band_slot` map.
//! A piece runs from its from key to its to key, and its successor is the piece whose from key
//! is its to key. An edge key is looked up in the two cells beside the edge and a CELL_BIT key
//! in its own cell, so chaining needs no sort or hash.
//!
//! Passes, in order (P = the pieces' count word, L = the loops' count word):
//! 1. `ChainClear` over the loop capacity: clears the per-loop records.
//! 2. `ChainSucc` P: writes `succ` (NONE where nothing continues: the tail of an open chain)
//!    and counts open ends and duplicate keys.
//! 3. List ranking (list_rank.rs): writes `rank_root`, `rank`, `rank_pred`, `loop_scan` and L.
//! 4. `ChainLoopId` P: writes `piece_loop`. The tail of each loop writes `loops` (root, pieces)
//!    and `loop_meta` (plane, level, -, LOOP_OPEN).
//! 5. `ChainExtent` P, 6. `ChainRightmost` P and 7. `ChainStart` P: each loop's bbox and
//!    rightmost vertex (the max x, then the min y), and its start piece. These use atomic
//!    min/max of monotone keys.
//! 8. The exclusive scan of `loop_n` into `loop_piece_off`.
//! 9. `ChainOrder` P: each loop's pieces in rank order from its start piece (`chain_order`, and
//!    `piece_pos` with the HEAD bit).
//! 10. `ChainAreaBlocks` over blocks of 256 positions, then 11. `ChainAreaLoops` L: the signed
//!     area (shoelace in position order, fixed blocks), the hole flag, and `loop_metric`.
//! 12. `ChainRay` L three times (z = 0 the loop's own level, 1 the outline level, 2 the loop's
//!     reference level): the first crossing right of the rightmost vertex.
//! 13. `ChainJump` L, ceil(log2 loop capacity) rounds: a loop outside its first hit takes that
//!     loop's parent.
//! 14. `ChainResolve` L: writes `loop_tree` (parent at the same level, island, reference
//!     contour) and `loop_meta.island`.
//!
//! Geometry. A piece is a line, or an arc about `piece_geo.xy` (r > 0, signed sweep). Arcs are
//! tessellated on one lattice per circle: the pseudo-angle of the diamond |x| + |y| = 1, with
//! M = ceil(sqrt(r / 2 tol)) steps per quadrant, so no step turns more than 2 / M rad. A piece's
//! polyline is its start, the lattice points strictly inside its sweep, and its end. Lattice
//! indices are exact integers in f32: a piece's range comes from its own endpoints'
//! coordinates, and the unwrap adds whole turns. Area, extent and containment use the polyline.
//!
//! Runs. Consecutive pieces of one feature form a run, and a run's head is the piece where the
//! feature changes. Below LEVEL_THIN and from LEVEL_REGION on, only a head contributes its start
//! to the ring (collinear pieces merge, and an arc's lattice continues across cells): another
//! piece contributes its start's lattice point when that start lies on the lattice. Thin lines
//! (32..63) carry widths per point and never merge: each piece is a head. A ring therefore has
//! the same points however its curves are split across cells. It starts at the head whose start
//! is rightmost (the max x, then the min y; the lower rank on a tie). A loop with no head is one
//! whole circle, and it starts at lattice index 0. Its start piece's points before that index go
//! last (`loop_acc` word 8 counts them). An open chain starts at its head.
//!
//! Containment. A ray runs from a loop's rightmost vertex to +x along its fine cell row, and the
//! first crossing among the target level's closed loops decides. A crossing piece running up
//! has a counter-clockwise loop on its left. Crossings use the half-open rule: a vertex exactly
//! on the ray counts as below it. A tie at one point goes to the edge that crosses first once
//! that vertex moves just below the ray. If the loop is inside the hit loop B, B is its parent;
//! if outside, its parent is B's parent, resolved by pointer jumping. A pending link always
//! names a loop whose rightmost vertex lies further right, so the links have no cycles. A
//! loop's island is the innermost contour of the outline level that holds it, and its reference
//! contour the innermost contour of its level's reference level (`ChainGrid::reference`): walls
//! resolve against the base (their base piece), region queries against the region they are cut
//! from. A loop at its reference level is its own (a hole its parent's).
//!
//! Closed material. A query region lies inside its reference region, and the two share pieces
//! where a region keeps its input's boundary. When a loop's rightmost vertex lies on a piece of
//! the outline or reference level (within a few ulps of its coordinates: shared vertices are
//! bitwise equal, shared stretches computed apart may differ in the last bits), that piece
//! decides: inside it if its loop is a contour, outside if a hole. In the reference's material
//! next to a contour lies that contour's region, and next to a hole its parent's, so this holds
//! whatever the half-open rule would give.
use crate::gpu_jobs::{label, BufRef, CountWord, Elements, JobError, Pass, Pool, ScanKernels, Step};
use crate::list_rank::{self, RankKernels, Ranked, Succ};
use makepad_draw::*;

/// The chain's count words, from its claimed base.
pub const WORD_LOOPS: u32 = 0;
/// Loops past the loop capacity (the batch re-runs with K doubled).
pub const WORD_OVERFLOW: u32 = 1;
/// To keys nothing continues, and from keys nothing enters (each open chain has one of each).
pub const WORD_OPEN_TAILS: u32 = 2;
pub const WORD_OPEN_HEADS: u32 = 3;
/// Keys matched more than once (a producer defect; must stay 0).
pub const WORD_DUPLICATES: u32 = 4;
/// Pieces placed in loop order (the scan total of the loops' piece counts).
pub const WORD_POSITIONS: u32 = 5;
/// Loops whose island or base piece is not a contour holding them (must stay 0).
pub const WORD_DEFECTS: u32 = 6;
pub const WORDS: u32 = 7;

/// A loop hit or parent link: the loop lies outside the loop named in the low bits.
pub const OUTSIDE: u32 = 1 << 31;
/// A final link with no parent.
pub const NO_PARENT: u32 = OUTSIDE - 1;
/// `piece_pos`: the piece heads a run (its start point is a ring point).
pub const HEAD: u32 = 1 << 31;
/// Words per loop of `loop_acc`: [0] the rightmost head start's x key (max), [1] its y key
/// (min), [2] the start piece's rank (min), [3] the rightmost vertex's x key (max), [4] its y
/// key (min), [5..8] the bbox's min x, min y and max y keys, [8] the start piece's points before
/// lattice index 0 (a loop with no head).
pub const ACC: u32 = 9;
/// The arc chord tolerance, in grid units (0.005 mm).
pub const CHORD_TOL: f32 = 0.05;
/// Positions per area block.
pub const AREA_BLOCK: u32 = 256;
/// Pieces per loop at K = 1: the loop capacity is ceil(K * pieces / PIECES_PER_LOOP), at most the pieces.
pub const PIECES_PER_LOOP: u32 = 8;
/// The largest K: every piece its own loop.
pub const K_MAX: u32 = PIECES_PER_LOOP;

/// Rounds of pointer jumping that cover `n` elements.
pub fn rounds(n: u32) -> u32 { 32 - n.max(1).saturating_sub(1).leading_zeros() }

/// The loop capacity for `pieces` at re-run factor `k`.
pub fn loop_capacity(pieces: u32, k: u32) -> u32 {
    ((pieces as u64 * k.clamp(1, K_MAX) as u64).div_ceil(PIECES_PER_LOOP as u64) as u32).clamp(1, pieces.max(1))
}

/// Whether pieces of `level` merge into runs (every level but thin lines).
pub fn merges(level: u32) -> bool { !(crate::plane_abi::LEVEL_THIN..crate::plane_abi::LEVEL_REGION).contains(&level) }

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.Chain = {
        precise_math: true
        workgroup_size: 64
        piece: shader.buffer(vec4f)
        piece_geo: shader.buffer(vec4f)
        piece_key: shader.buffer(vec4u)
        piece_info: shader.buffer(vec4u)
        piece_off: shader.buffer(u32)
        band_slot: shader.buffer(u32)
        succ: shader.buffer(u32)
        rank_root: shader.buffer(u32)
        rank: shader.buffer(u32)
        rank_pred: shader.buffer(u32)
        loop_scan: shader.buffer(u32)
        piece_loop: shader.buffer(u32)
        piece_pos: shader.buffer(u32)
        chain_order: shader.buffer(u32)
        loops: shader.buffer(vec4u)
        loop_meta: shader.buffer(vec4u)
        loop_metric: shader.buffer(vec4f)
        loop_acc: shader.buffer(u32)
        loop_n: shader.buffer(u32)
        loop_piece_off: shader.buffer(u32)
        loop_area: shader.buffer(u32)
        area_part: shader.buffer(vec4u)
        loop_hit: shader.buffer(u32)
        loop_link0: shader.buffer(u32)
        loop_link1: shader.buffer(u32)
        loop_tree: shader.buffer(vec4u)
        // chain[0]: nx, ny, planes, level slots per cell; [1]: the pieces' count word, this
        // stage's first count word, loop capacity, piece capacity; [2]: outline level (NONE:
        // none), 0, the buffer holding the final links, chord tolerance (f32 bits); [3..67]: the
        // reference level of each level, four a row.
        chain: shader.buffer(vec4u)
        counts: shader.buffer(u32)
        dispatch_range: shader.uniform(vec4u)

        fixed: fn() -> u32 {
            let i = global_id().x
            if i >= self.dispatch_range.y { return 4294967295u }
            return self.dispatch_range.x + i
        }
        // The element of a counted pass over counts[word], or NONE.
        item: fn(word: u32) -> u32 {
            let i = global_id().x
            if i >= self.dispatch_range.y { return 4294967295u }
            let e = self.dispatch_range.x + i
            if e >= self.counts[word] { return 4294967295u }
            return e
        }
        word: fn(w: u32) -> u32 { return self.chain[1u].y + w }
        // The level whose contours a loop of `level` resolves against (NONE: none).
        reference: fn(level: u32) -> u32 {
            if level >= 256u { return 4294967295u }
            let r = self.chain[3u + level / 4u]
            let c = level % 4u
            if c == 0u { return r.x }
            if c == 1u { return r.y }
            if c == 2u { return r.z }
            return r.w
        }
        pieces_word: fn() -> u32 { return self.chain[1u].x }
        loops_word: fn() -> u32 { return self.chain[1u].y }
        // Fine cells across, and the node row stride (a cell is the node id of its low corner).
        cells_x: fn() -> u32 { return self.chain[0u].x * 10u }
        stride: fn() -> u32 { return self.chain[0u].x * 10u + 1u }
        // The pieces of plane p's fine cell (fx, fy) as [first, end): none off the band or grid.
        cell_range: fn(p: u32, fx: u32, fy: u32) -> vec2u {
            if fx >= self.cells_x() || fy >= self.chain[0u].y * 10u { return vec2u(0u, 0u) }
            let nx = self.chain[0u].x
            let slot = self.band_slot[p * nx * self.chain[0u].y + (fy / 10u) * nx + fx / 10u]
            if slot == 4294967295u { return vec2u(0u, 0u) }
            let levels = self.chain[0u].w
            let at = (slot * 100u + (fy % 10u) * 10u + fx % 10u) * levels
            return vec2u(self.piece_off[at], self.piece_off[at + levels])
        }
        // The cells that can hold the other piece of an endpoint key: (fx0, fy0, fx1, fy1). The
        // second cell is NONE for a key inside a cell.
        key_cells: fn(hi: u32) -> vec4u {
            if (hi & 2147483648u) != 0u {
                let c = hi & 2147483647u
                return vec4u(c % self.stride(), c / self.stride(), 4294967295u, 4294967295u)
            }
            let g = hi >> 1u
            let gx = g % self.stride()
            let gy = g / self.stride()
            // a +x edge: the cells below and above; a +y edge: left and right (wrapping past
            // 0 is off the grid)
            if (hi & 1u) == 0u { return vec4u(gx, gy - 1u, gx, gy) }
            return vec4u(gx - 1u, gy, gx, gy)
        }
        // The first piece of plane p whose from key (which 0) or to key (which 1) is (hi, lo),
        // and how many there are. A point on a node is keyed by its +x edge with the node
        // ordinal 0xFFFFFF (regions_compose); its other piece can lie in any of the node's four
        // cells, so the two left of the node are searched too.
        find: fn(p: u32, hi: u32, lo: u32, which: u32) -> vec2u {
            let c = self.key_cells(hi)
            var sides = 2u
            if (hi & 2147483648u) == 0u && (lo & 16777215u) == 16777215u { sides = 4u }
            var first = 4294967295u
            var n = 0u
            for side in 0u..sides {
                var fx = c.x
                var fy = c.y
                if side == 1u { fx = c.z; fy = c.w }
                // left of the node (wrapping past 0 is off the grid)
                if side == 2u { fx = c.x - 1u }
                if side == 3u { fx = c.z - 1u; fy = c.w }
                if fx != 4294967295u {
                    let r = self.cell_range(p, fx, fy)
                    for m in r.x..r.y {
                        let key = self.piece_key[m]
                        var mhi = key.x
                        var mlo = key.y
                        if which == 1u { mhi = key.z; mlo = key.w }
                        if mhi == hi && mlo == lo {
                            if first == 4294967295u { first = m }
                            n += 1u
                        }
                    }
                }
            }
            return vec2u(first, n)
        }
        // Thin lines (32..63) never merge runs.
        merge_level: fn(level: u32) -> bool { return level < 32u || level >= 64u }
        // Monotone u32 keys of f32 values, for atomic min/max.
        key: fn(v: f32) -> u32 {
            let bits = asuint(v)
            if (bits & 2147483648u) != 0u { return bits ^ 4294967295u }
            return bits ^ 2147483648u
        }
        unkey: fn(k: u32) -> f32 {
            if (k & 2147483648u) != 0u { return asfloat(k ^ 2147483648u) }
            return asfloat(k ^ 4294967295u)
        }
        // The pseudo-angle of d in [0, 4): the diamond |x| + |y| = 1, one unit per quadrant.
        diamond: fn(d: vec2) -> f32 {
            let s = abs(d.x) + abs(d.y)
            if s == 0.0 { return 0.0 }
            if d.x > 0.0 && d.y >= 0.0 { return d.y / s }
            if d.x <= 0.0 && d.y > 0.0 { return 1.0 + (0.0 - d.x) / s }
            if d.x < 0.0 && d.y <= 0.0 { return 2.0 + (0.0 - d.y) / s }
            return 3.0 + d.x / s
        }
        // Lattice steps per quadrant of a circle of radius r.
        lattice_m: fn(r: f32) -> f32 { return max(ceil(sqrt(r / (2.0 * asfloat(self.chain[2u].w)))), 1.0) }
        // A point's lattice coordinate in [0, 4m).
        ring_u: fn(d: vec2, m: f32) -> f32 {
            var u = self.diamond(d) * m
            if u >= 4.0 * m { u -= 4.0 * m }
            return u
        }
        // An arc piece's start and end coordinates in [0, 4m), the whole turns its end adds
        // (-1, 0 or 1: the unwrap nearest its sweep), and the sign of its travel (0: none).
        arc_ends: fn(k: u32, m: f32) -> vec4 {
            let q = self.piece[k]
            let g = self.piece_geo[k]
            let a = self.ring_u(q.xy - g.xy, m)
            if q.x == q.z && q.y == q.w { return vec4(a, a, 0.0, 0.0) }
            let b = self.ring_u(q.zw - g.xy, m)
            let n4 = 4.0 * m
            let expect = g.w * (2.0 * m / 3.14159265)
            let d0 = b - a
            var turns = 0.0
            if abs(d0 + n4 - expect) < abs(d0 - expect) { turns = 1.0 }
            if abs(d0 - n4 - expect) < abs(d0 + turns * n4 - expect) { turns = 0.0 - 1.0 }
            let travel = d0 + turns * n4
            var sign = 0.0
            if travel > 0.0 { sign = 1.0 }
            if travel < 0.0 { sign = 0.0 - 1.0 }
            return vec4(a, b, turns, sign)
        }
        // A piece's lattice points as (first index, count, step, m), the indices unwrapped
        // along its travel. They lie strictly inside its sweep; with `from_start`, the start
        // too when it lies on the lattice. A line has none.
        lattice_points: fn(k: u32, from_start: bool) -> vec4 {
            let g = self.piece_geo[k]
            if g.z <= 0.0 { return vec4(0.0, 0.0, 1.0, 1.0) }
            let m = self.lattice_m(g.z)
            let e = self.arc_ends(k, m)
            let n4 = 4.0 * m
            if e.w > 0.0 {
                var first = floor(e.x) + 1.0
                if from_start { first = ceil(e.x) }
                let last = ceil(e.y) - 1.0 + e.z * n4
                return vec4(first, max(last - first + 1.0, 0.0), 1.0, m)
            }
            if e.w < 0.0 {
                var first = ceil(e.x) - 1.0
                if from_start { first = floor(e.x) }
                let last = floor(e.y) + 1.0 + e.z * n4
                return vec4(first, max(first - last + 1.0, 0.0), 0.0 - 1.0, m)
            }
            return vec4(0.0, 0.0, 1.0, m)
        }
        lattice: fn(k: u32) -> vec4 { return self.lattice_points(k, false) }
        // The unit direction of lattice index i (any integer, taken mod 4m).
        lattice_dir: fn(i: f32, m: f32) -> vec2 {
            let n4 = 4.0 * m
            let im = i - n4 * floor(i / n4)
            let q = floor(im / m)
            let f = (im - q * m) / m
            var d = vec2(1.0 - f, f)
            if q == 1.0 { d = vec2(0.0 - f, 1.0 - f) }
            else if q == 2.0 { d = vec2(f - 1.0, 0.0 - f) }
            else if q == 3.0 { d = vec2(f, f - 1.0) }
            return d / sqrt(d.x * d.x + d.y * d.y)
        }
        // The point of lattice index i on piece k. A merging level keeps the circle's radius,
        // so all pieces of one circle share their points. A thin spiral's radius runs linearly
        // in angle between its ends.
        lattice_point: fn(k: u32, i: f32, m: f32) -> vec2 {
            let g = self.piece_geo[k]
            let d = self.lattice_dir(i, m)
            if self.merge_level(self.piece_info[k].y) { return g.xy + d * g.z }
            let q = self.piece[k]
            let r0 = length(q.xy - g.xy)
            let r1 = length(q.zw - g.xy)
            let s = q.xy - g.xy
            // the angle from the start, unwrapped along the sweep
            var t = atan2(s.x * d.y - s.y * d.x, s.x * d.x + s.y * d.y)
            if g.w > 0.0 && t < 0.0 { t += 6.28318531 }
            if g.w < 0.0 && t > 0.0 { t -= 6.28318531 }
            var f = 0.0
            if g.w != 0.0 { f = clamp(t / g.w, 0.0, 1.0) }
            return g.xy + d * (r0 + (r1 - r0) * f)
        }
        // Vertex j of piece k's polyline: 0 its start, 1..=count its lattice points, then its end.
        vertex: fn(k: u32, j: u32, lat: vec4) -> vec2 {
            let q = self.piece[k]
            if j == 0u { return q.xy }
            if f32(j) > lat.y { return q.zw }
            return self.lattice_point(k, lat.x + lat.z * f32(j - 1u), lat.w)
        }
        // A piece starts a run: it has no predecessor, its predecessor has another feature, or
        // it is thin.
        natural_head: fn(k: u32) -> bool {
            let pred = self.rank_pred[k]
            if pred == 4294967295u { return true }
            let info = self.piece_info[k]
            if !self.merge_level(info.y) { return true }
            return self.piece_info[pred].z != info.z
        }
        is_open: fn(l: u32) -> bool { return (self.loop_meta[l].w & 2u) != 0u }
        is_hole: fn(l: u32) -> bool { return (self.loop_meta[l].w & 1u) != 0u }
        // Twice the signed area piece k adds about its loop's reference point (the root's start).
        area_term: fn(k: u32, l: u32) -> f32 {
            if self.is_open(l) { return 0.0 }
            let r0 = self.piece[self.loops[l].x].xy
            let lat = self.lattice(k)
            let n = u32(lat.y)
            var v = self.piece[k].xy - r0
            var s = 0.0
            for j in 1u..(n + 2u) {
                let w = self.vertex(k, j, lat) - r0
                s += v.x * w.y - v.y * w.x
                v = w
            }
            return s
        }
        // A loop's final parent (NONE for none).
        parent_of: fn(l: u32) -> u32 {
            var link = self.loop_link0[l]
            if self.chain[2u].z == 1u { link = self.loop_link1[l] }
            if link == 2147483647u || (link & 2147483648u) != 0u { return 4294967295u }
            return link
        }
        // The closed loop of the target level of plane p with a piece through (px, py), within
        // eps: the least loop id, with OUTSIDE for a hole; NONE for none. The cells within eps
        // of the point hold every such piece.
        origin_hit: fn(a: u32, p: u32, target: u32, px: f32, py: f32) -> u32 {
            let eps = 0.000004 * max(max(abs(px), abs(py)), 1.0)
            let o = vec2(px, py)
            var best = 4294967295u
            let fx0 = u32(max(floor(px - eps), 0.0))
            let fy0 = u32(max(floor(py - eps), 0.0))
            // at most two cells a side
            let fx1 = min(u32(max(floor(px + eps), 0.0)), fx0 + 1u)
            let fy1 = min(u32(max(floor(py + eps), 0.0)), fy0 + 1u)
            for fy in fy0..(fy1 + 1u) {
                for fx in fx0..(fx1 + 1u) {
                    let r = self.cell_range(p, fx, fy)
                    for m in r.x..r.y {
                        let lm = self.piece_loop[m]
                        if self.piece_info[m].y == target && lm != a && lm != 4294967295u && lm < best {
                            if !self.is_open(lm) {
                                let lat = self.lattice(m)
                                let n = u32(lat.y)
                                var v = self.piece[m].xy
                                var on = false
                                for j in 1u..(n + 2u) {
                                    let w = self.vertex(m, j, lat)
                                    let d = w - v
                                    let dd = dot(d, d)
                                    var t = 0.0
                                    if dd > 0.0 { t = clamp(dot(o - v, d) / dd, 0.0, 1.0) }
                                    let e = o - (v + d * t)
                                    if abs(e.x) <= eps && abs(e.y) <= eps { on = true }
                                    v = w
                                }
                                if on { best = lm }
                            }
                        }
                    }
                }
            }
            if best == 4294967295u { return best }
            if !self.is_hole(best) { return best }
            return best | 2147483648u
        }
        // The first crossing right of (px, py) among the target level's closed loops of plane p
        // but loop a: the hit loop, with OUTSIDE when the point lies outside it; NONE for none.
        ray: fn(a: u32, p: u32, target: u32, px: f32, py: f32) -> u32 {
            let cells = self.cells_x()
            let nx = self.chain[0u].x
            let gy = min(u32(max(floor(py), 0.0)), self.chain[0u].y * 10u - 1u)
            var gx = min(u32(max(floor(px), 0.0)), cells - 1u)
            var bx = 1.0e30
            var bkey = 1.0e30
            var bk = 4294967295u
            var bup = false
            while gx < cells {
                let slot = self.band_slot[p * nx * self.chain[0u].y + (gy / 10u) * nx + gx / 10u]
                if slot == 4294967295u {
                    if bk != 4294967295u { break }
                    gx = (gx / 10u + 1u) * 10u
                } else {
                    let r = self.cell_range(p, gx, gy)
                    for m in r.x..r.y {
                        let lm = self.piece_loop[m]
                        if self.piece_info[m].y == target && lm != a && lm != 4294967295u {
                            if !self.is_open(lm) {
                                let lat = self.lattice(m)
                                let n = u32(lat.y)
                                var v = self.piece[m].xy
                                for j in 1u..(n + 2u) {
                                    let w = self.vertex(m, j, lat)
                                    if (v.y > py) != (w.y > py) {
                                        // a vertex on the ray counts as below it, and ties go by
                                        // the edge's dx/dy up from that vertex
                                        var x = 0.0
                                        var tie = 0.0
                                        if v.y == py { x = v.x; tie = (w.x - v.x) / (w.y - v.y) }
                                        else if w.y == py { x = w.x; tie = (v.x - w.x) / (v.y - w.y) }
                                        else { x = v.x + (py - v.y) * (w.x - v.x) / (w.y - v.y) }
                                        if x > px && (x < bx || (x == bx && tie < bkey)) {
                                            bx = x
                                            bkey = tie
                                            bk = m
                                            bup = w.y > v.y
                                        }
                                    }
                                    v = w
                                }
                            }
                        }
                    }
                    // a hit on the cell's right edge may tie with one in the next cell
                    if bk != 4294967295u && bx < f32(gx + 1u) { break }
                    gx += 1u
                }
            }
            if bk == 4294967295u { return 4294967295u }
            let b = self.piece_loop[bk]
            // a crossing running up has a counter-clockwise loop's inside on its left
            if bup == (self.loop_metric[b * 2u].x > 0.0) { return b }
            return b | 2147483648u
        }
        // The innermost contour of a ray's target level holding the loop, from its hit, and 1
        // when the hit puts the loop in no material of that level (a defect).
        contour_of: fn(hit: u32) -> vec2u {
            if hit == 4294967295u { return vec2u(hit, 1u) }
            let b = hit & 2147483647u
            let inside = (hit & 2147483648u) == 0u
            // inside a contour, or outside a hole: in its material
            var defect = 0u
            if inside == self.is_hole(b) { defect = 1u }
            var x = b
            if !inside { x = self.parent_of(b) }
            if x == 4294967295u { return vec2u(x, 1u) }
            if self.is_hole(x) { x = self.parent_of(x) }
            return vec2u(x, defect)
        }
    }

    mod.sdf.ChainClear = {
        ..mod.sdf.Chain
        loop_acc: shader.buffer_rw(u32)
        loop_n: shader.buffer_rw(u32)
        loop_area: shader.buffer_rw(u32)
        loops: shader.buffer_rw(vec4u)
        loop_meta: shader.buffer_rw(vec4u)
        main: fn() {
            let i = self.fixed()
            if i == 4294967295u { return }
            let a = i * 9u
            self.loop_acc[a] = 0u
            self.loop_acc[a + 1u] = 4294967295u
            self.loop_acc[a + 2u] = 4294967295u
            self.loop_acc[a + 3u] = 0u
            self.loop_acc[a + 4u] = 4294967295u
            self.loop_acc[a + 5u] = 4294967295u
            self.loop_acc[a + 6u] = 4294967295u
            self.loop_acc[a + 7u] = 0u
            self.loop_acc[a + 8u] = 0u
            self.loop_n[i] = 0u
            self.loop_area[i] = 0u
            self.loops[i] = vec4u(4294967295u, 0u, 0u, 0u)
            self.loop_meta[i] = vec4u(4294967295u, 4294967295u, 4294967295u, 0u)
        }
    }

    mod.sdf.ChainSucc = {
        ..mod.sdf.Chain
        succ: shader.buffer_rw(u32)
        counts: shader.buffer_rw(u32)
        main: fn() {
            let k = self.item(self.pieces_word())
            if k == 4294967295u { return }
            let key = self.piece_key[k]
            let p = self.piece_info[k].x
            let s = self.find(p, key.z, key.w, 0u)
            let q = self.find(p, key.x, key.y, 1u)
            self.succ[k] = s.x
            if s.y == 0u { atomic_add(self.counts[self.word(2u)], 1u) }
            if q.y == 0u { atomic_add(self.counts[self.word(3u)], 1u) }
            if s.y > 1u || q.y > 1u { atomic_add(self.counts[self.word(4u)], 1u) }
        }
    }

    mod.sdf.ChainLoopId = {
        ..mod.sdf.Chain
        piece_loop: shader.buffer_rw(u32)
        loops: shader.buffer_rw(vec4u)
        loop_meta: shader.buffer_rw(vec4u)
        loop_n: shader.buffer_rw(u32)
        counts: shader.buffer_rw(u32)
        main: fn() {
            let k = self.item(self.pieces_word())
            if k == 4294967295u { return }
            let root = self.rank_root[k]
            let l = self.loop_scan[root]
            if l >= self.chain[1u].z {
                self.piece_loop[k] = 4294967295u
                if root == k { atomic_add(self.counts[self.word(1u)], 1u) }
                return
            }
            self.piece_loop[k] = l
            let s = self.succ[k]
            if s == 4294967295u || s == root {
                let n = self.rank[k] + 1u
                let info = self.piece_info[root]
                var flags = 0u
                if s == 4294967295u { flags = 2u }
                self.loops[l] = vec4u(root, n, 0u, 0u)
                self.loop_n[l] = n
                self.loop_meta[l] = vec4u(info.x, info.y, 4294967295u, flags)
            }
        }
    }

    mod.sdf.ChainExtent = {
        ..mod.sdf.Chain
        loop_acc: shader.buffer_rw(u32)
        main: fn() {
            let k = self.item(self.pieces_word())
            if k == 4294967295u { return }
            let l = self.piece_loop[k]
            if l == 4294967295u { return }
            let a = l * 9u
            let lat = self.lattice(k)
            var n = u32(lat.y) + 1u
            // an open chain's tail also owns its end
            if self.succ[k] == 4294967295u { n += 1u }
            var xmax = 0u
            var xmin = 4294967295u
            var ymin = 4294967295u
            var ymax = 0u
            for j in 0u..n {
                let v = self.vertex(k, j, lat)
                xmax = max(xmax, self.key(v.x))
                xmin = min(xmin, self.key(v.x))
                ymin = min(ymin, self.key(v.y))
                ymax = max(ymax, self.key(v.y))
            }
            atomic_max(self.loop_acc[a + 3u], xmax)
            atomic_min(self.loop_acc[a + 5u], xmin)
            atomic_min(self.loop_acc[a + 6u], ymin)
            atomic_max(self.loop_acc[a + 7u], ymax)
            // (an atomic's old value would be the body's value: the if ends it)
            if !self.is_open(l) && self.natural_head(k) { atomic_max(self.loop_acc[a], self.key(self.piece[k].x)) }
        }
    }

    mod.sdf.ChainRightmost = {
        ..mod.sdf.Chain
        loop_acc: shader.buffer_rw(u32)
        main: fn() {
            let k = self.item(self.pieces_word())
            if k == 4294967295u { return }
            let l = self.piece_loop[k]
            if l == 4294967295u { return }
            let a = l * 9u
            let start = self.piece[k].xy
            if !self.is_open(l) && self.natural_head(k) && self.key(start.x) == self.loop_acc[a] {
                atomic_min(self.loop_acc[a + 1u], self.key(start.y))
            }
            let xmax = self.loop_acc[a + 3u]
            let lat = self.lattice(k)
            var n = u32(lat.y) + 1u
            if self.succ[k] == 4294967295u { n += 1u }
            var y = 4294967295u
            for j in 0u..n {
                let v = self.vertex(k, j, lat)
                if self.key(v.x) == xmax { y = min(y, self.key(v.y)) }
            }
            if y != 4294967295u { atomic_min(self.loop_acc[a + 4u], y) }
        }
    }

    // The start piece: the rightmost head start, or for a loop with no head the piece holding
    // lattice index 0 (it also counts its points before that index).
    mod.sdf.ChainStart = {
        ..mod.sdf.Chain
        loop_acc: shader.buffer_rw(u32)
        main: fn() {
            let k = self.item(self.pieces_word())
            if k == 4294967295u { return }
            let l = self.piece_loop[k]
            if l == 4294967295u { return }
            if self.is_open(l) { return }
            let a = l * 9u
            if self.loop_acc[a] != 0u {
                let start = self.piece[k].xy
                if self.natural_head(k) && self.key(start.x) == self.loop_acc[a] && self.key(start.y) == self.loop_acc[a + 1u] {
                    atomic_min(self.loop_acc[a + 2u], self.rank[k])
                }
                return
            }
            let lat = self.lattice_points(k, true)
            if lat.y < 1.0 { return }
            let n4 = 4.0 * lat.w
            let last = lat.x + lat.z * (lat.y - 1.0)
            var zero = ceil(lat.x / n4) * n4
            var before = zero - lat.x
            if lat.z < 0.0 {
                zero = floor(lat.x / n4) * n4
                before = lat.x - zero
            }
            if (lat.z > 0.0 && zero <= last) || (lat.z < 0.0 && zero >= last) {
                self.loop_acc[a + 2u] = self.rank[k]
                self.loop_acc[a + 8u] = u32(before)
            }
        }
    }

    mod.sdf.ChainOrder = {
        ..mod.sdf.Chain
        piece_pos: shader.buffer_rw(u32)
        chain_order: shader.buffer_rw(u32)
        main: fn() {
            let k = self.item(self.pieces_word())
            if k == 4294967295u { return }
            let l = self.piece_loop[k]
            if l == 4294967295u { return }
            let n = self.loops[l].y
            let a = l * 9u
            var start = self.loop_acc[a + 2u]
            var head = self.natural_head(k)
            // an open chain starts at its head; a loop with no head has no head piece
            if self.is_open(l) { start = 0u }
            else if self.loop_acc[a] == 0u { head = false }
            if start == 4294967295u { start = 0u }
            let pos = self.loop_piece_off[l] + (self.rank[k] + n - start) % n
            self.chain_order[pos] = k
            var bit = 0u
            if head { bit = 2147483648u }
            self.piece_pos[k] = pos | bit
        }
    }

    mod.sdf.ChainAreaBlocks = {
        ..mod.sdf.Chain
        loop_area: shader.buffer_rw(u32)
        area_part: shader.buffer_rw(vec4u)
        // A loop within one block gets its area here. A longer loop's part in this block is the
        // block's first segment or its last.
        multi: fn(l: u32) -> bool {
            let off = self.loop_piece_off[l]
            return off / 256u != (off + self.loop_n[l] - 1u) / 256u
        }
        main: fn() {
            let b = self.fixed()
            if b == 4294967295u { return }
            let total = self.counts[self.word(5u)]
            let lo = b * 256u
            let hi = max(min(lo + 256u, total), lo)
            var first_loop = 4294967295u
            var first_sum = 0.0
            var last_loop = 4294967295u
            var last_sum = 0.0
            var cur = 4294967295u
            var sum = 0.0
            var segments = 0u
            for q in lo..(hi + 1u) {
                var l = 4294967295u
                var k = 0u
                if q < hi {
                    k = self.chain_order[q]
                    l = self.piece_loop[k]
                }
                if l != cur || q == hi {
                    if cur != 4294967295u {
                        if !self.multi(cur) { self.loop_area[cur] = asuint(sum * 0.5) }
                        else if segments == 1u { first_loop = cur; first_sum = sum }
                        else { last_loop = cur; last_sum = sum }
                    }
                    cur = l
                    sum = 0.0
                    segments += 1u
                }
                if q < hi { sum += self.area_term(k, l) }
            }
            self.area_part[b] = vec4u(first_loop, asuint(first_sum), last_loop, asuint(last_sum))
        }
    }

    mod.sdf.ChainAreaLoops = {
        ..mod.sdf.Chain
        loop_metric: shader.buffer_rw(vec4f)
        loop_meta: shader.buffer_rw(vec4u)
        main: fn() {
            let l = self.item(self.loops_word())
            if l == 4294967295u { return }
            var area = asfloat(self.loop_area[l])
            let off = self.loop_piece_off[l]
            let b0 = off / 256u
            let b1 = (off + self.loop_n[l] - 1u) / 256u
            if b0 != b1 {
                var s = 0.0
                for b in b0..(b1 + 1u) {
                    let part = self.area_part[b]
                    if part.x == l { s += asfloat(part.y) } else if part.z == l { s += asfloat(part.w) }
                }
                area = s * 0.5
            }
            let meta = self.loop_meta[l]
            var flags = meta.w
            if area < 0.0 { flags = flags | 1u }
            self.loop_meta[l] = vec4u(meta.x, meta.y, meta.z, flags)
            let a = l * 9u
            let rx = self.unkey(self.loop_acc[a + 3u])
            self.loop_metric[l * 2u] = vec4f(area, rx, self.unkey(self.loop_acc[a + 4u]), 0.0)
            self.loop_metric[l * 2u + 1u] = vec4f(self.unkey(self.loop_acc[a + 5u]), self.unkey(self.loop_acc[a + 6u]), rx, self.unkey(self.loop_acc[a + 7u]))
        }
    }

    mod.sdf.ChainRay = {
        ..mod.sdf.Chain
        loop_hit: shader.buffer_rw(u32)
        loop_link0: shader.buffer_rw(u32)
        main: fn() {
            let a = self.item(self.loops_word())
            if a == 4294967295u { return }
            let z = self.dispatch_range.z
            let meta = self.loop_meta[a]
            let outline = self.chain[2u].x
            let reference = self.reference(meta.y)
            var target = meta.y
            if z == 1u { target = outline }
            if z == 2u { target = reference }
            // the outline's ray serves a reference at the outline level
            var skip = false
            if z == 1u && (outline == 4294967295u || meta.y == outline) { skip = true }
            if z == 2u && (reference == 4294967295u || reference == meta.y || reference == outline) { skip = true }
            // at the outline and reference levels, a piece through the vertex decides first
            // (closed material)
            let px = self.loop_metric[a * 2u].y
            let py = self.loop_metric[a * 2u].z
            var hit = 4294967295u
            if !skip && z != 0u { hit = self.origin_hit(a, meta.x, target, px, py) }
            if !skip && hit == 4294967295u { hit = self.ray(a, meta.x, target, px, py) }
            self.loop_hit[a * 4u + z] = hit
            if z == 0u {
                var link = hit
                if hit == 4294967295u { link = 2147483647u }
                self.loop_link0[a] = link
            }
        }
    }

    // Round z: a pending link (outside its loop B) takes B's link from the other buffer.
    mod.sdf.ChainJump = {
        ..mod.sdf.Chain
        loop_link0: shader.buffer_rw(u32)
        loop_link1: shader.buffer_rw(u32)
        main: fn() {
            let l = self.item(self.loops_word())
            if l == 4294967295u { return }
            if self.dispatch_range.z % 2u == 0u {
                let link = self.loop_link0[l]
                var next = link
                if link != 2147483647u && (link & 2147483648u) != 0u { next = self.loop_link0[link & 2147483647u] }
                self.loop_link1[l] = next
            } else {
                let link = self.loop_link1[l]
                var next = link
                if link != 2147483647u && (link & 2147483648u) != 0u { next = self.loop_link1[link & 2147483647u] }
                self.loop_link0[l] = next
            }
        }
    }

    mod.sdf.ChainResolve = {
        ..mod.sdf.Chain
        loop_tree: shader.buffer_rw(vec4u)
        loop_meta: shader.buffer_rw(vec4u)
        counts: shader.buffer_rw(u32)
        main: fn() {
            let l = self.item(self.loops_word())
            if l == 4294967295u { return }
            let meta = self.loop_meta[l]
            let parent = self.parent_of(l)
            let outline = self.chain[2u].x
            let reference = self.reference(meta.y)
            // at its own level, a contour is its own region and a hole is its parent's
            var own = l
            var defects = 0u
            if self.is_hole(l) {
                own = parent
                if parent == 4294967295u && (meta.y == outline || meta.y == reference) { defects = 1u }
            }
            var island = own
            if meta.y != outline {
                if outline == 4294967295u { island = 4294967295u }
                else {
                    let c = self.contour_of(self.loop_hit[l * 4u + 1u])
                    island = c.x
                    defects += c.y
                }
            }
            var base = own
            if reference == 4294967295u { base = 4294967295u }
            else if reference != meta.y {
                if reference == outline { base = island }
                else {
                    let c = self.contour_of(self.loop_hit[l * 4u + 2u])
                    base = c.x
                    defects += c.y
                }
            }
            // an open chain bounds no region: it is no island or reference contour of its own
            if self.is_open(l) && meta.y == outline { island = 4294967295u }
            if self.is_open(l) && meta.y == reference { base = 4294967295u }
            if defects != 0u { atomic_add(self.counts[self.word(6u)], 1u) }
            self.loop_tree[l] = vec4u(parent, island, base, 0u)
            self.loop_meta[l] = vec4u(meta.x, meta.y, island, meta.w)
        }
    }

    mod.sdf
}

/// The kernels in pass order.
pub const KERNELS: [&str; 12] = [
    "ChainClear", "ChainSucc", "ChainLoopId", "ChainExtent", "ChainRightmost", "ChainStart", "ChainOrder",
    "ChainAreaBlocks", "ChainAreaLoops", "ChainRay", "ChainJump", "ChainResolve",
];

/// The chain's kernels with the ranking and scan kernels it runs, made once.
#[derive(Clone)]
pub struct ChainKernels { kernels: Vec<ComputeKernel>, rank: RankKernels, scan: ScanKernels }

impl ChainKernels {
    pub fn new(cx: &mut Cx) -> Result<Self, String> {
        let kernels = cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            KERNELS.iter().map(|name| {
                let object = vm.value(module, LiveId::from_str(name)).as_object().ok_or_else(|| format!("{name} kernel missing"))?;
                ComputeKernel::new(vm, object, name)
            }).collect::<Result<Vec<_>, String>>()
        })?;
        Ok(Self { kernels, rank: RankKernels::new(cx)?, scan: ScanKernels::new(cx)? })
    }
    fn kernel(&self, name: &str) -> &ComputeKernel { &self.kernels[KERNELS.iter().position(|k| *k == name).unwrap()] }
}

/// A producer's pieces (GPU-EXEC §10.3), as the chain binds them.
#[derive(Clone, Copy, Debug)]
pub struct Pieces {
    pub piece: BufRef,
    pub piece_geo: BufRef,
    pub piece_key: BufRef,
    pub piece_info: BufRef,
    /// The CSR over (band fine cell, level slot), cell-major.
    pub piece_off: BufRef,
    pub band_slot: BufRef,
    /// The pieces' count word and capacity.
    pub count: CountWord,
    pub capacity: u32,
    /// Level slots per band fine cell in `piece_off`.
    pub levels_per_cell: u32,
}

/// Levels (a level is a u8).
pub const LEVELS: usize = 256;

/// The grid, and the levels containment resolves against.
#[derive(Clone, Copy, Debug)]
pub struct ChainGrid {
    pub nx: u32,
    pub ny: u32,
    pub planes: u32,
    /// Islands are the outline level's contours (NONE: no islands).
    pub outline_level: u32,
    /// Per level, the level whose innermost contour holding a loop is its `loop_tree.z` (NONE:
    /// none). Walls name the base (the outline without one), a region query the region it is
    /// cut from. Both levels share the producer's piece CSR, and the query's region must lie
    /// inside the reference's.
    pub reference: [u32; LEVELS],
}
impl ChainGrid {
    /// A grid whose `levels` are (level, its reference level) pairs; other levels have none.
    pub fn new(nx: u32, ny: u32, planes: u32, outline_level: u32, levels: &[(u32, u32)]) -> Self {
        let mut reference = [crate::plane_abi::NONE; LEVELS];
        for &(level, to) in levels { reference[level as usize] = to; }
        Self { nx, ny, planes, outline_level, reference }
    }
}

/// The chain's outputs, GPU-resident for the batch.
#[derive(Clone, Copy, Debug)]
pub struct Chain {
    pub words: CountWord,
    pub loop_capacity: u32,
    pub piece_capacity: u32,
    /// The `chain` constants (rings binds them too).
    pub params: BufRef,
    pub succ: BufRef,
    /// The list ranking of `succ`: each piece's loop root (a cycle's smallest index, an open
    /// chain's head), rank from it and predecessor, and each root's loop id.
    pub ranked: Ranked,
    /// Per piece: its loop (NONE past the loop capacity), and its position in `chain_order` with HEAD.
    pub piece_loop: BufRef,
    pub piece_pos: BufRef,
    /// The pieces of every loop in order from its start, loops in id order.
    pub chain_order: BufRef,
    /// Per loop: LoopRec (root, pieces; rings writes the points), LoopMeta, LoopArea (area,
    /// rightmost x, rightmost y) then LoopBox, `loop_acc` (ACC words), its pieces' count and
    /// first position, and `loop_tree` (parent, island, reference contour, 0).
    pub loops: BufRef,
    pub loop_meta: BufRef,
    pub loop_metric: BufRef,
    pub loop_acc: BufRef,
    pub loop_n: BufRef,
    pub loop_piece_off: BufRef,
    pub loop_tree: BufRef,
}
impl Chain {
    pub fn word(&self, w: u32) -> CountWord { CountWord(self.words.0 + w) }
    /// The loops' count word (list ranking's total): counted loop passes take it.
    pub fn loops_word(&self) -> CountWord { self.word(WORD_LOOPS) }
}

/// The `chain` constants buffer.
pub fn params_words(grid: &ChainGrid, pieces: &Pieces, words: CountWord, loop_capacity: u32, final_link: u32) -> Vec<u32> {
    let mut w = vec![grid.nx, grid.ny, grid.planes, pieces.levels_per_cell,
        pieces.count.0, words.0, loop_capacity, pieces.capacity,
        grid.outline_level, 0, final_link, CHORD_TOL.to_bits()];
    w.extend(grid.reference);
    w
}

/// The chain's counts words, claimed before its item starts.
pub fn claim(pool: &mut Pool) -> Result<CountWord, JobError> { pool.claim("chain", WORDS) }

/// The chain's steps over `pieces` at re-run factor `k`, counting in `words` (from [`claim`]).
pub fn steps(kernels: &ChainKernels, pool: &mut Pool, words: CountWord, pieces: &Pieces, grid: &ChainGrid, k: u32) -> Result<(Chain, Vec<Step>), JobError> {
    let np = pieces.capacity.max(1);
    let nl = loop_capacity(np, k);
    let rounds = rounds(nl);
    let params_words = params_words(grid, pieces, words, nl, rounds % 2);
    let params = pool.buffer("chain params", params_words.len() as u64 * 4);
    pool.upload(params, 0, params_words.iter().flat_map(|w| w.to_le_bytes()).collect());
    let succ = pool.buffer("chain succ", np as u64 * 4);
    let (rank_steps, ranked) = list_rank::steps(&kernels.rank, &kernels.scan, pool, pieces.count, np, Succ { buffer: succ, stride: 1, at: 0 },
        CountWord(words.0 + WORD_LOOPS))?;
    let chain = Chain {
        words, loop_capacity: nl, piece_capacity: np, params, succ, ranked,
        piece_loop: pool.buffer("chain piece loop", np as u64 * 4),
        piece_pos: pool.buffer("chain piece pos", np as u64 * 4),
        chain_order: pool.buffer("chain order", np as u64 * 4),
        loops: pool.buffer("chain loops", nl as u64 * 16),
        loop_meta: pool.buffer("chain loop meta", nl as u64 * 16),
        loop_metric: pool.buffer("chain loop metric", nl as u64 * 32),
        loop_acc: pool.buffer("chain loop acc", nl as u64 * ACC as u64 * 4),
        loop_n: pool.buffer("chain loop n", nl as u64 * 4),
        loop_piece_off: pool.buffer("chain loop piece off", nl as u64 * 4),
        loop_tree: pool.buffer("chain loop tree", nl as u64 * 16),
    };
    let blocks = np.div_ceil(AREA_BLOCK);
    let loop_area = pool.buffer("chain loop area", nl as u64 * 4);
    let area_part = pool.buffer("chain area part", blocks as u64 * 16);
    let loop_hit = pool.buffer("chain loop hit", nl as u64 * 16);
    let link0 = pool.buffer("chain link0", nl as u64 * 4);
    let link1 = pool.buffer("chain link1", nl as u64 * 4);
    let bind = |name: LiveId| -> BufRef {
        match name {
            n if n == id!(piece) => pieces.piece,
            n if n == id!(piece_geo) => pieces.piece_geo,
            n if n == id!(piece_key) => pieces.piece_key,
            n if n == id!(piece_info) => pieces.piece_info,
            n if n == id!(piece_off) => pieces.piece_off,
            n if n == id!(band_slot) => pieces.band_slot,
            n if n == id!(succ) => chain.succ,
            n if n == id!(rank_root) => chain.ranked.root,
            n if n == id!(rank) => chain.ranked.rank,
            n if n == id!(rank_pred) => chain.ranked.pred,
            n if n == id!(loop_scan) => chain.ranked.list_scan,
            n if n == id!(piece_loop) => chain.piece_loop,
            n if n == id!(piece_pos) => chain.piece_pos,
            n if n == id!(chain_order) => chain.chain_order,
            n if n == id!(loops) => chain.loops,
            n if n == id!(loop_meta) => chain.loop_meta,
            n if n == id!(loop_metric) => chain.loop_metric,
            n if n == id!(loop_acc) => chain.loop_acc,
            n if n == id!(loop_n) => chain.loop_n,
            n if n == id!(loop_piece_off) => chain.loop_piece_off,
            n if n == id!(loop_area) => loop_area,
            n if n == id!(area_part) => area_part,
            n if n == id!(loop_hit) => loop_hit,
            n if n == id!(loop_link0) => link0,
            n if n == id!(loop_link1) => link1,
            n if n == id!(loop_tree) => chain.loop_tree,
            n if n == id!(chain) => chain.params,
            n if n == id!(counts) => BufRef::RUNTIME,
            n => panic!("the chain binds no buffer {n:?}"),
        }
    };
    let on_pieces = Elements::Counted { word: pieces.count, bound: np };
    let on_loops = Elements::Counted { word: chain.loops_word(), bound: nl };
    // Ceilings: the most elements of one job within 50 ms on the slowest supported GPU. A piece
    // reads two cells' pieces or walks its own polyline; a ray walks at most one cell row.
    let pass = |name: &str, elements: Elements, cost: u32, ceiling: u32| Pass::new(kernels.kernel(name), label(name), elements, cost, ceiling, bind);
    let mut steps = vec![
        Step::Pass(pass("ChainClear", Elements::Fixed(nl), 8, 1 << 20)),
        Step::Pass(pass("ChainSucc", on_pieces, 64, 1 << 18)),
    ];
    steps.extend(rank_steps);
    steps.push(Step::Pass(pass("ChainLoopId", on_pieces, 8, 1 << 20)));
    steps.push(Step::Pass(pass("ChainExtent", on_pieces, 48, 1 << 18)));
    steps.push(Step::Pass(pass("ChainRightmost", on_pieces, 48, 1 << 18)));
    steps.push(Step::Pass(pass("ChainStart", on_pieces, 16, 1 << 20)));
    steps.extend(kernels.scan.exclusive(pool, chain.loop_n, chain.loop_piece_off, nl, chain.word(WORD_POSITIONS)));
    steps.push(Step::Pass(pass("ChainOrder", on_pieces, 16, 1 << 20)));
    steps.push(Step::Pass(pass("ChainAreaBlocks", Elements::Fixed(blocks), AREA_BLOCK * 24, 1 << 12)));
    steps.push(Step::Pass(pass("ChainAreaLoops", on_loops, 16, 1 << 18)));
    for target in 0..3 {
        steps.push(Step::Pass(pass("ChainRay", on_loops, 4096, 1 << 13).zw(target, 0)));
    }
    if rounds > 0 { steps.push(Step::Repeat(pass("ChainJump", on_loops, 4, 1 << 20), rounds)); }
    steps.push(Step::Pass(pass("ChainResolve", on_loops, 16, 1 << 18)));
    Ok((chain, steps))
}

#[cfg(test)]
mod tests {
    #[test]
    fn capacities_and_rounds() {
        assert_eq!(super::rounds(1), 0);
        assert_eq!(super::rounds(2), 1);
        assert_eq!(super::rounds(5), 3);
        assert_eq!(super::loop_capacity(100, 1), 13);
        assert_eq!(super::loop_capacity(100, 2), 25);
        assert_eq!(super::loop_capacity(100, super::K_MAX), 100);
        assert_eq!(super::loop_capacity(0, 1), 1);
        assert!(super::merges(0) && super::merges(31) && !super::merges(32) && !super::merges(63) && super::merges(64));
    }
}
