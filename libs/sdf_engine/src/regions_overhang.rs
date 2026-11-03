//! S3 on the GPU (GPU-EXEC §9 S3 RULING, §10.3 "S3 labels"): label_ring for every wall loop of
//! a batch against the layer below, with the semantics of the CPU's walls::label_line.
//!
//! A loop's line runs from its point 0 (the closing segment included for a closed loop). Two
//! levels of the layer below's biased field d' (plane k-1's feature_field, Miter(3)) cut it:
//! - support: d' = n/2 (half a nozzle outside the lower outline);
//! - full support: d' = lo = -w/2 + (w + n)/20, w the loop's bead width.
//! Each segment is cut on its own. Between consecutive cuts a span is supported where d' <= n/2
//! at its middle, and partial where also d' > lo there. With slowdown a partial span splits into
//! degree samples every 2 mm. Each span or sample is one [`S3Label`] at its start (the loop's
//! point index from 0, and t along that point's segment): supported (flag bit 0) at degree(Euclidean signed distance to the
//! lower outline at its middle), or over air at degree 5. Full-support cuts are the seeds
//! ([`S3Seed`], where speed ramps start). No arc length is summed on the GPU: [`LoopLabels::to_mm`]
//! measures the read-back points in f64.
//!
//! Cuts are exact for the field the cand lists define. Each segment is walked fine cell by fine
//! cell. A cell is skipped when its two ends' values rule a crossing out (d' is 1-Lipschitz; the
//! field's depth keeps a cell diagonal past each level, so both ends' values are exact wherever
//! a crossing could be). Elsewhere, within each stretch on one side of the raw outline, {d' <= c}
//! is the union over the cell's cand features of one interval each: a feature's metric is linear
//! along a line in its domain, a round vertex's quadratic. The union's ends are the cuts. Off the
//! band a point takes the field's cover side: deep inside is supported, far outside is not.
//!
//! The degree's distance is exact where it matters. walls::degree is 0 for d <= lo and 5 for
//! d >= n/2, so only |d| < R0 = max(n/2, |lo|) needs the true distance. The scan visits every
//! fine cell meeting the box of half-width R = R0 + max|bias| + 0.01 around the point, and the
//! FeatureSet lists each seg in the fine cell that holds it, so every seg whose biased distance
//! is below R0 is seen. A seg not seen has a biased distance above R0 and cannot be the
//! minimum below R0; when the minimum is above R0 the scan's value is too, on the same side, and
//! the degree saturates the same way. The minimum is order free, so the value is deterministic.
//! A span at or past full support (d' <= lo <= 0) has degree 0 with no scan: the miter metric
//! never exceeds the Euclidean distance on the same side, so d <= d' <= lo.
//!
//! Passes, in order (elements):
//! 1. `S3Prep` (P·nx·ny coarse cells): the band word (slot, or the cover side) and set slot;
//! 2. `S3Loops` (loop bound): per loop its points, the plane below and lo, or unlabelled (also
//!    when its points run past the point count: an overflowed rings run, which is void);
//! 3. `S3Count` (point bound): labels and seeds per segment;
//! 4. the exclusive scan of both counts ([`ScanKernels::exclusive`]);
//! 5. `S3Fill` (point bound): the labels and seeds at their offsets;
//! 6. `S3LoopOff` (loop bound + 1): the per-loop CSRs and the header of `s3_out`.
//!
//! `s3_out` (the readback, u32 words): [`OUT_HEADER`] words (loops, labels, seeds, label
//! capacity, seed capacity, label base, seed base, loop bound), `s3_label_off` (loop bound + 1
//! words), `s3_seed_off` (loop bound + 1), the labels (4 words each, from the label base) and
//! the seeds (2 words each, from the seed base).
use crate::gpu_jobs::{label, BufRef, CountWord, Counts, Elements, JobError, Pass, Pool, ScanKernels, Step};
use crate::plane_abi::{S3Label, S3Seed, LABEL_SUPPORTED, LEVEL_REGION, NONE};
use makepad_draw::*;

/// Words of the `s3_work` header; the tables and sections follow at the offsets it holds.
pub const HEADER: u32 = 24;
/// Words of the `s3_out` header.
pub const OUT_HEADER: u32 = 8;
/// Labels and seeds room per point unless a re-run doubles it (the K rule, §10.2).
pub const K_DEFAULT: u32 = 2;
/// Grid units per mm.
const GRID: f32 = 10.0;
/// nm per mm (the read-back points).
const NM: f64 = 1.0e6;
/// Levels a loop may carry and still be labelled (walls below 32, thin beads 32..63).
pub const LABEL_LEVELS: u32 = LEVEL_REGION;
/// Margin the field's depth keeps past each level (grid units): more than a fine cell's
/// diagonal, so a cell's end values are exact wherever the cell could hold a crossing.
pub const DEPTH_MARGIN: f32 = 1.5;

// s3_work header words.
const H_POINTS: usize = 0;
const H_LOOPS: usize = 1;
const H_FLAGS: usize = 2;
const H_POINTS_WORD: usize = 3;
const H_LOOPS_WORD: usize = 4;
const H_POINT_BOUND: usize = 5;
const H_LOOP_BOUND: usize = 6;
const H_C1: usize = 7;
const H_WINDOW: usize = 8;
const H_LABEL_CAP: usize = 9;
const H_SEED_CAP: usize = 10;
const H_NX: usize = 11;
const H_NY: usize = 12;
const H_PLANES: usize = 13;
const H_MITER: usize = 14;
const H_OVERFLOW: usize = 15;
const H_LEVELS: usize = 16;
const H_BELOW: usize = 17;
const H_COARSE: usize = 18;
const H_LOOPREC: usize = 19;
const H_FAILED: usize = 20;
const H_TOTAL: usize = 21;
const H_OUT_LABELS: usize = 22;
const H_OUT_SEEDS: usize = 23;
/// H_FLAGS: degree samples every 2 mm along partial spans.
const FLAG_SLOWDOWN: u32 = 1;

/// Every pass's kernel, in pass order.
pub const KERNELS: [&str; 5] = ["S3Prep", "S3Loops", "S3Count", "S3Fill", "S3LoopOff"];

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    // The field accessors read the s3_work header instead of `field`, and a coarse cell's band
    // word and set slot from its section, so a walk binds 12 buffers (push block: 128 bytes).
    mod.sdf.S3 = {
        ..mod.sdf.Field
        s3_work: shader.buffer(u32)
        ring_pt: shader.buffer(u32)
        loops: shader.buffer(vec4u)
        loop_meta: shader.buffer(vec4u)
        s3_count: shader.buffer(u32)
        s3_scan: shader.buffer(u32)
        s3_out: shader.buffer(u32)
        counts: shader.buffer_rw(u32)
        hw: fn(k: u32) -> u32 { return self.s3_work[k] }
        hf: fn(k: u32) -> f32 { return asfloat(self.s3_work[k]) }
        cells: fn() -> u32 { return self.hw(11u) * self.hw(12u) }
        nodes_x: fn() -> u32 { return self.hw(11u) * 10u + 1u }
        nodes_y: fn() -> u32 { return self.hw(12u) * 10u + 1u }
        miter: fn() -> bool { return self.hf(14u) > 0.0 }
        n_points: fn() -> u32 {
            let w = self.hw(3u)
            if w != 4294967295u { return min(self.counts[w], self.hw(5u)) }
            return self.hw(0u)
        }
        n_loops: fn() -> u32 {
            let w = self.hw(4u)
            if w != 4294967295u { return min(self.counts[w], self.hw(6u)) }
            return self.hw(1u)
        }
        fail: fn() { atomic_add(self.counts[self.hw(20u)], 1u) }
        overflow: fn() { atomic_add(self.counts[self.hw(15u)], 1u) }
        coarse_at: fn(p: u32, fx: u32, fy: u32) -> u32 {
            return self.hw(18u) + (p * self.cells() + (fy / 10u) * self.hw(11u) + fx / 10u) * 2u
        }
        // The band slot of fine cell (fx, fy)'s coarse cell, or NONE outside / NONE - 1 inside.
        band_word: fn(p: u32, fx: u32, fy: u32) -> u32 { return self.s3_work[self.coarse_at(p, fx, fy)] }
        cell_segs: fn(p: u32, fx: u32, fy: u32) -> vec2u {
            if fx + 1u >= self.nodes_x() || fy + 1u >= self.nodes_y() { return vec2u(0u, 0u) }
            let slot = self.s3_work[self.coarse_at(p, fx, fy) + 1u]
            if slot == 4294967295u { return vec2u(0u, 0u) }
            let at = slot * 100u + (fy % 10u) * 10u + fx % 10u
            return vec2u(self.seg_off[at], self.seg_off[at + 1u])
        }
        fine_x: fn(x: f32) -> u32 { return min(u32(max(floor(x), 0.0)), self.nodes_x() - 2u) }
        fine_y: fn(y: f32) -> u32 { return min(u32(max(floor(y), 0.0)), self.nodes_y() - 2u) }
        // d' at q from band fine cell (fx, fy)'s cand list (q in that closed cell).
        cell_eval: fn(p: u32, slot: u32, fx: u32, fy: u32, q: vec2) -> f32 {
            let inside = self.point_side(p, slot, fx, fy, q) != 0u
            let cell = slot * 100u + (fy % 10u) * 10u + fx % 10u
            var best = 1.0e30
            for k in self.cand_off[cell]..self.cand_off[cell + 1u] {
                let f = self.cand[k]
                best = min(best, self.value(f & 2147483647u, (f & 2147483648u) != 0u, q, inside))
            }
            if best > 1.0e29 { best = 1.0e9 }
            if inside { return 0.0 - best }
            return best
        }
        // d' at q in plane p, by cover off the band.
        s3_at: fn(p: u32, q: vec2) -> f32 {
            let fx = self.fine_x(q.x)
            let fy = self.fine_y(q.y)
            let w = self.band_word(p, fx, fy)
            if w == 4294967295u { return 1.0e9 }
            if w == 4294967294u { return 0.0 - 1.0e9 }
            return self.cell_eval(p, w, fx, fy, q)
        }
        // The Euclidean signed distance to plane p's raw outline less the seg's bias: exact
        // below the window radius less the largest bias, beyond it past that on the same side.
        euclid: fn(p: u32, q: vec2) -> f32 {
            let r = self.hf(8u)
            let fx = self.fine_x(q.x)
            let fy = self.fine_y(q.y)
            let w = self.band_word(p, fx, fy)
            var inside = w == 4294967294u
            if w < 4294967294u { inside = self.point_side(p, w, fx, fy, q) != 0u }
            let x0 = self.fine_x(q.x - r)
            let x1 = self.fine_x(q.x + r)
            let y0 = self.fine_y(q.y - r)
            let y1 = self.fine_y(q.y + r)
            var best = 1.0e9
            for gy in y0..(y1 + 1u) {
                for gx in x0..(x1 + 1u) {
                    let sr = self.cell_segs(p, gx, gy)
                    for s in sr.x..sr.y {
                        let g = self.seg[s]
                        let e = g.zw - g.xy
                        let ee = dot(e, e)
                        var t = 0.0
                        if ee > 0.0 { t = min(max(dot(q - g.xy, e) / ee, 0.0), 1.0) }
                        let dist = length(q - (g.xy + e * t))
                        if inside { best = min(best, dist + self.seg_bias(s)) }
                        else { best = min(best, dist - self.seg_bias(s)) }
                    }
                }
            }
            if inside { return 0.0 - best }
            return best
        }
        // walls::degree: 0 at or past full support, then 1 + 4 x the share of the way to n/2, in tenths.
        degree: fn(d: f32, lo: f32, c1: f32) -> f32 {
            let fraction = (d - lo) / (c1 - lo)
            if fraction <= 0.0 { return 0.0 }
            return floor(min(max(1.0 + 4.0 * fraction, 1.0), 5.0) * 10.0) / 10.0
        }
        // g0 + u gd <= k, on the parameter range r.
        span_clip: fn(r: vec2, g0: f32, gd: f32, k: f32) -> vec2 {
            var out = r
            if gd == 0.0 {
                if g0 > k { out = vec2(1.0, 0.0) }
                return out
            }
            let b = (k - g0) / gd
            if gd > 0.0 { out.y = min(out.y, b) } else { out.x = max(out.x, b) }
            return out
        }
        // The parameters u of the line a + u d where feature f's metric is at most k, within f's
        // domain on the given side (as `value` defines both); empty when x >= y.
        feature_span: fn(f: u32, a: vec2, d: vec2, inside: bool, k: f32) -> vec2 {
            if k < 0.0 { return vec2(1.0, 0.0) }
            let s = f & 2147483647u
            let w = a - self.seg_a(s)
            var r = vec2(0.0 - 1.0e30, 1.0e30)
            if (f & 2147483648u) != 0u {
                if self.vertex_inside(s) != inside { return vec2(1.0, 0.0) }
                let pe = self.seg_e(self.seg_prev(s))
                let e = self.seg_e(s)
                r = self.span_clip(r, 0.0 - dot(w, pe), 0.0 - dot(d, pe), 0.0001)
                r = self.span_clip(r, dot(w, e), dot(d, e), 0.0001)
                if self.miter() {
                    let m1 = self.vertex_m1(s)
                    let m2 = self.vertex_m2(s)
                    r = self.span_clip(r, dot(w, m1), dot(d, m1), k)
                    r = self.span_clip(r, dot(w, m2), dot(d, m2), k)
                    if (self.seg_flags(s) & 4u) != 0u {
                        let ub = self.vertex_u(s)
                        r = self.span_clip(r, dot(w, ub), dot(d, ub), k)
                    }
                } else {
                    let qa = dot(d, d)
                    let qb = dot(w, d)
                    let qc = dot(w, w) - k * k
                    if qa <= 0.0 {
                        if qc > 0.0 { r = vec2(1.0, 0.0) }
                    } else {
                        let disc = qb * qb - qa * qc
                        if disc < 0.0 { return vec2(1.0, 0.0) }
                        let root = sqrt(disc)
                        r = vec2(max(r.x, (0.0 - qb - root) / qa), min(r.y, (0.0 - qb + root) / qa))
                    }
                }
                return r
            }
            if (self.seg_flags(s) & 8u) == 0u { return vec2(1.0, 0.0) }
            let e = self.seg_e(s)
            let n = self.left(e)
            r = self.span_clip(r, 0.0 - dot(w, e), 0.0 - dot(d, e), 0.0001)
            r = self.span_clip(r, dot(w, e), dot(d, e), self.seg_len(s) + 0.0001)
            if inside {
                r = self.span_clip(r, 0.0 - dot(w, n), 0.0 - dot(d, n), 0.0001)
                r = self.span_clip(r, dot(w, n), dot(d, n), k)
            } else {
                r = self.span_clip(r, dot(w, n), dot(d, n), 0.0001)
                r = self.span_clip(r, 0.0 - dot(w, n), 0.0 - dot(d, n), k)
            }
            return r
        }
        // Feature f's span for level c on a side: outside, where m - b <= c; inside, where
        // m + b <= -c (the complement of {d' <= c} up to its ends, which are the same).
        level_span: fn(f: u32, a: vec2, d: vec2, inside: bool, c: f32) -> vec2 {
            let b = self.seg_bias(f & 2147483647u)
            if inside { return self.feature_span(f, a, d, true, 0.0 - c - b) }
            return self.feature_span(f, a, d, false, c + b)
        }
        // Whether x, an end (0 start, 1 end) of cand entry j's span, is an end of the union of
        // the cell's cand spans: no other span covers it from that side.
        span_end: fn(cell: u32, j: u32, x: f32, end: u32, a: vec2, d: vec2, inside: bool, c: f32) -> bool {
            for k in self.cand_off[cell]..self.cand_off[cell + 1u] {
                if k == j { continue }
                let r = self.level_span(self.cand[k], a, d, inside, c)
                if r.x >= r.y { continue }
                if end == 0u && r.x < x && x <= r.y { return false }
                if end == 1u && r.x <= x && x < r.y { return false }
            }
            return true
        }
        loop_rec: fn(l: u32) -> vec4u {
            let at = self.hw(19u) + l * 4u
            return vec4u(self.s3_work[at], self.s3_work[at + 1u], self.s3_work[at + 2u], self.s3_work[at + 3u])
        }
        loop_first: fn(l: u32) -> u32 { return self.s3_work[self.hw(19u) + l * 4u] }
        // The loop holding point i: the last loop whose first point is at most i.
        loop_of: fn(i: u32) -> u32 {
            var lo = 0u
            var hi = self.n_loops()
            for bit in 0..32u {
                if hi - lo <= 1u { break }
                let mid = lo + (hi - lo) / 2u
                if self.loop_first(mid) <= i { lo = mid } else { hi = mid }
            }
            return lo
        }
        pt: fn(i: u32) -> vec2 { return vec2(asfloat(self.ring_pt[i * 2u]), asfloat(self.ring_pt[i * 2u + 1u])) }
        // The point after i on its loop, or NONE where i is past its loop, ends an open loop,
        // or the loop is unlabelled.
        seg_end: fn(i: u32, rec: vec4u) -> u32 {
            if rec.z == 4294967295u { return 4294967295u }
            let n = rec.y & 2147483647u
            if i < rec.x || i >= rec.x + n { return 4294967295u }
            if i + 1u < rec.x + n { return i + 1u }
            if (rec.y & 2147483648u) != 0u { return 4294967295u }
            return rec.x
        }
        // The fill writes labels, seeds and degrees; the count counts.
        fills: fn() -> bool { return false }
        counted: fn(i: u32, labels: u32, seeds: u32) {}
        emit_label: fn(i: u32, k: u32, v: u32, t: f32, flags: u32, degree: f32) {}
        emit_seed: fn(i: u32, k: u32, v: u32, t: f32) {}
        // One span [t0, t1] of segment i (its loop's point v): its labels from k0 on; returns how many.
        piece: fn(i: u32, v: u32, k0: u32, a: vec2, d: vec2, len: f32, below: u32, lo: f32, c1: f32, t0: f32, t1: f32) -> u32 {
            let value = self.s3_at(below, a + d * (0.5 * (t0 + t1)))
            let supported = value <= c1
            let partial = supported && value > lo
            var kn = 1u
            if partial && (self.hw(2u) & 1u) != 0u { kn = max(u32(ceil(len * (t1 - t0) / 20.0)), 1u) }
            if self.fills() {
                for k in 0..kn {
                    let s0 = t0 + (t1 - t0) * f32(k) / f32(kn)
                    var s1 = t1
                    if k + 1u < kn { s1 = t0 + (t1 - t0) * f32(k + 1u) / f32(kn) }
                    var flags = 0u
                    var degree = 5.0
                    if supported {
                        flags = 1u
                        degree = 0.0
                        if partial || lo > 0.0 { degree = self.degree(self.euclid(below, a + d * (0.5 * (s0 + s1))), lo, c1) }
                    }
                    self.emit_label(i, k0 + k, v, s0, flags, degree)
                }
            }
            return kn
        }
        // Segment i's labels and seeds, in order along it.
        s3_walk: fn(i: u32) {
            if i >= self.n_points() { self.counted(i, 0u, 0u); return }
            let rec = self.loop_rec(self.loop_of(i))
            let j = self.seg_end(i, rec)
            if j == 4294967295u { self.counted(i, 0u, 0u); return }
            let a = self.pt(i)
            let d = self.pt(j) - a
            let len = length(d)
            if len < 0.00001 { self.counted(i, 0u, 0u); return }
            let below = rec.z
            let lo = asfloat(rec.w)
            let c1 = self.hf(7u)
            let tiny = 0.00001 / len
            var labels = 0u
            var seeds = 0u
            var prev = 0.0
            var last_seed = 0.0 - 1.0
            // the next grid lines the segment crosses, per axis
            var kx = 0.0
            var ky = 0.0
            var sx = 0.0
            var sy = 0.0
            if d.x > 0.0 { kx = floor(a.x) + 1.0; sx = 1.0 } else if d.x < 0.0 { kx = ceil(a.x) - 1.0; sx = 0.0 - 1.0 }
            if d.y > 0.0 { ky = floor(a.y) + 1.0; sy = 1.0 } else if d.y < 0.0 { ky = ceil(a.y) - 1.0; sy = 0.0 - 1.0 }
            let b = a + d
            let steps = u32(abs(floor(b.x) - floor(a.x)) + abs(floor(b.y) - floor(a.y))) + 3u
            var cut_x = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var cut_level = array(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
            var sp = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var t = 0.0
            for step in 0..steps {
                if t >= 1.0 { break }
                var tx = 2.0
                if sx != 0.0 { tx = (kx - a.x) / d.x }
                var ty = 2.0
                if sy != 0.0 { ty = (ky - a.y) / d.y }
                let tn = min(min(tx, ty), 1.0)
                if tn > t {
                    let q = a + d * (0.5 * (t + tn))
                    let fx = self.fine_x(q.x)
                    let fy = self.fine_y(q.y)
                    let slot = self.band_word(below, fx, fy)
                    var nc = 0u
                    if slot < 4294967294u {
                        let cell = slot * 100u + (fy % 10u) * 10u + fx % 10u
                        let v0 = self.cell_eval(below, slot, fx, fy, a + d * t)
                        let v1 = self.cell_eval(below, slot, fx, fy, a + d * tn)
                        let reach = len * (tn - t) + 0.001
                        let near1 = abs(v0 - c1) + abs(v1 - c1) <= reach
                        let near2 = abs(v0 - lo) + abs(v1 - lo) <= reach
                        if near1 || near2 {
                            // stretches on one side: split where the raw outline crosses
                            var ns = 2u
                            sp[0] = t
                            sp[1] = tn
                            let sr = self.cell_segs(below, fx, fy)
                            for s in sr.x..sr.y {
                                let g = self.seg[s]
                                let e = g.zw - g.xy
                                let den = d.x * e.y - d.y * e.x
                                if den == 0.0 { continue }
                                let w = g.xy - a
                                let u = (w.x * e.y - w.y * e.x) / den
                                let v = (w.x * d.y - w.y * d.x) / den
                                if v < 0.0 || v > 1.0 || u <= t || u >= tn { continue }
                                if ns >= 10u { self.fail(); continue }
                                // sp[ns - 1] holds tn: insert u before it in order, then tn after
                                var at = ns - 1u
                                for z in 0..10u {
                                    if at == 0u { break }
                                    if sp[at - 1u] <= u { break }
                                    sp[at] = sp[at - 1u]
                                    at -= 1u
                                }
                                sp[at] = u
                                ns += 1u
                                sp[ns - 1u] = tn
                            }
                            for part in 0..9u {
                                if part + 1u >= ns { break }
                                let u0 = sp[part]
                                let u1 = sp[part + 1u]
                                if u1 <= u0 { continue }
                                let inside = self.point_side(below, slot, fx, fy, a + d * (0.5 * (u0 + u1))) != 0u
                                for level in 0..2u {
                                    if level == 0u && !near1 { continue }
                                    if level == 1u && !near2 { continue }
                                    var c = c1
                                    if level == 1u { c = lo }
                                    for k in self.cand_off[cell]..self.cand_off[cell + 1u] {
                                        let r = self.level_span(self.cand[k], a, d, inside, c)
                                        if r.x >= r.y { continue }
                                        for end in 0..2u {
                                            var x = r.x
                                            if end == 1u { x = r.y }
                                            // a part owns (u0, u1]
                                            if x <= u0 || x > u1 || x >= 1.0 { continue }
                                            if !self.span_end(cell, k, x, end, a, d, inside, c) { continue }
                                            var dup = false
                                            for z in 0..16u {
                                                if z >= nc { break }
                                                if cut_x[z] == x && cut_level[z] == level { dup = true }
                                            }
                                            if dup { continue }
                                            if nc >= 16u { self.fail(); continue }
                                            var at = nc
                                            for z in 0..16u {
                                                if at == 0u { break }
                                                if cut_x[at - 1u] <= x { break }
                                                cut_x[at] = cut_x[at - 1u]
                                                cut_level[at] = cut_level[at - 1u]
                                                at -= 1u
                                            }
                                            cut_x[at] = x
                                            cut_level[at] = level
                                            nc += 1u
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for z in 0..16u {
                        if z >= nc { break }
                        let x = cut_x[z]
                        if x < tiny || 1.0 - x < tiny { continue }
                        if cut_level[z] == 1u && x - last_seed >= tiny {
                            self.emit_seed(i, seeds, i - rec.x, x)
                            seeds += 1u
                            last_seed = x
                        }
                        if x - prev < tiny { continue }
                        labels += self.piece(i, i - rec.x, labels, a, d, len, below, lo, c1, prev, x)
                        prev = x
                    }
                }
                if tx <= tn { kx += sx }
                if ty <= tn { ky += sy }
                t = tn
            }
            if t < 1.0 { self.fail() }
            labels += self.piece(i, i - rec.x, labels, a, d, len, below, lo, c1, prev, 1.0)
            self.counted(i, labels, seeds)
        }
    }

    mod.sdf.S3Prep = {
        ..mod.sdf.S3
        s3_work: shader.buffer_rw(u32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            var w = self.band_slot[i]
            if w == 4294967295u { w = 4294967295u - self.cover[i] }
            let at = self.hw(18u) + i * 2u
            self.s3_work[at] = w
            self.s3_work[at + 1u] = self.set_slot[i]
        }
    }

    mod.sdf.S3Loops = {
        ..mod.sdf.S3
        s3_work: shader.buffer_rw(u32)
        main: fn() {
            let l = self.index()
            if l == 4294967295u { return }
            let at = self.hw(19u) + l * 4u
            if l >= self.n_loops() {
                self.s3_work[at] = self.n_points()
                self.s3_work[at + 1u] = 0u
                self.s3_work[at + 2u] = 4294967295u
                self.s3_work[at + 3u] = 0u
                return
            }
            let lp = self.loops[l]
            let meta = self.loop_meta[l]
            var below = 4294967295u
            var lo = 4294967295u
            if meta.y < 64u && meta.x < self.hw(13u) {
                lo = self.s3_work[self.hw(16u) + meta.y]
                below = self.s3_work[self.hw(17u) + meta.x]
            }
            // A loop whose points run past the stored ones (the rings overflowed: a void run) stays unlabelled.
            if lo == 4294967295u || lp.z < 2u || lp.w > self.n_points() || lp.z > self.n_points() - lp.w { below = 4294967295u }
            var n = lp.z
            if (meta.w & 2u) != 0u { n = n | 2147483648u }
            self.s3_work[at] = lp.w
            self.s3_work[at + 1u] = n
            self.s3_work[at + 2u] = below
            self.s3_work[at + 3u] = lo
        }
    }

    mod.sdf.S3Count = {
        ..mod.sdf.S3
        s3_count: shader.buffer_rw(u32)
        counted: fn(i: u32, labels: u32, seeds: u32) {
            self.s3_count[i] = labels
            self.s3_count[self.hw(5u) + i] = seeds
        }
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            self.s3_walk(i)
        }
    }

    mod.sdf.S3Fill = {
        ..mod.sdf.S3
        s3_out: shader.buffer_rw(u32)
        fills: fn() -> bool { return true }
        emit_label: fn(i: u32, k: u32, v: u32, t: f32, flags: u32, degree: f32) {
            let at = self.s3_scan[i] + k
            if at >= self.hw(9u) { self.overflow(); return }
            let o = self.hw(22u) + at * 4u
            self.s3_out[o] = v
            self.s3_out[o + 1u] = asuint(t)
            self.s3_out[o + 2u] = asuint(degree)
            self.s3_out[o + 3u] = flags
        }
        emit_seed: fn(i: u32, k: u32, v: u32, t: f32) {
            let bound = self.hw(5u)
            let s = self.s3_scan[bound + i] - self.s3_scan[bound] + k
            if s >= self.hw(10u) { self.overflow(); return }
            let o = self.hw(23u) + s * 2u
            self.s3_out[o] = v
            self.s3_out[o + 1u] = asuint(t)
        }
        main: fn() {
            let i = self.index()
            if i == 4294967295u { return }
            self.s3_walk(i)
        }
    }

    mod.sdf.S3LoopOff = {
        ..mod.sdf.S3
        s3_out: shader.buffer_rw(u32)
        main: fn() {
            let l = self.index()
            if l == 4294967295u { return }
            let nl = self.n_loops()
            if l > nl { return }
            let bound = self.hw(5u)
            let lb = self.hw(6u)
            let labels = self.s3_scan[bound]
            let seeds = self.counts[self.hw(21u)] - labels
            var label_at = labels
            var seed_at = seeds
            if l < nl {
                let fp = min(self.loop_first(l), self.n_points())
                if fp < bound {
                    label_at = self.s3_scan[fp]
                    seed_at = self.s3_scan[bound + fp] - labels
                }
            }
            self.s3_out[8u + l] = label_at
            self.s3_out[9u + lb + l] = seed_at
            if l == 0u {
                self.s3_out[0u] = nl
                self.s3_out[1u] = labels
                self.s3_out[2u] = seeds
                self.s3_out[3u] = self.hw(9u)
                self.s3_out[4u] = self.hw(10u)
                self.s3_out[5u] = self.hw(22u)
                self.s3_out[6u] = self.hw(23u)
                self.s3_out[7u] = lb
                if labels > self.hw(9u) || seeds > self.hw(10u) { self.overflow() }
            }
        }
    }

    mod.sdf
}

/// S3's kernels, made once per Cx (after feature_field's and gpu_jobs' modules are registered).
#[derive(Clone)]
pub struct S3Kernels { kernels: Vec<ComputeKernel> }
impl S3Kernels {
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

/// A count S3 (and S1b) is given exactly, or as a counts word an earlier stage of the batch writes
/// (at most its bound).
#[derive(Clone, Copy, Debug)]
pub enum Size { Known(u32), Counted { word: CountWord, bound: u32 } }
impl Size {
    pub(crate) fn bound(&self) -> u32 { match *self { Size::Known(n) => n, Size::Counted { bound, .. } => bound } }
    pub(crate) fn words(&self) -> (u32, u32) { match *self { Size::Known(n) => (n, NONE), Size::Counted { word, .. } => (0, word.0) } }
}

/// What S3 labels, and against what.
#[derive(Clone, Debug)]
pub struct S3Spec {
    /// The part revision's coarse grid and the batch's planes.
    pub nx: u32,
    pub ny: u32,
    pub planes: u32,
    /// The lower field's miter limit (3 for the CPU's offsets; 0 = round) and its largest
    /// |bias|, grid units.
    pub miter: f32,
    pub bias_max: f32,
    /// Nozzle diameter, mm.
    pub nozzle: f32,
    /// Degree samples every 2 mm along partial spans (walls::OverhangSpeeds::slowdown).
    pub slowdown: bool,
    /// Every labelled level (below [`LABEL_LEVELS`]) with its nominal bead width in mm.
    pub widths: Vec<(u32, f32)>,
    /// Per plane, the plane of the layer below, or NONE (no labels: the first layer, the halo).
    pub below: Vec<u32>,
    pub points: Size,
    pub loops: Size,
    /// Labels and seeds room per point ([`K_DEFAULT`], doubled by an overflow re-run).
    pub k: u32,
}

/// walls::supported_offset in grid units: the full-support level for a bead of width `w` mm.
pub fn full_support_level(w: f32, nozzle: f32) -> f32 { (-w * 0.5 + (w + nozzle) * 0.05) * GRID }

/// The depths (in, out; grid units) S3 needs of the layer below's field: each level it cuts
/// at, plus [`DEPTH_MARGIN`].
pub fn field_depth(spec: &S3Spec) -> (f32, f32) {
    let lo = spec.widths.iter().map(|&(_, w)| full_support_level(w, spec.nozzle).abs()).fold(0.0f32, f32::max);
    (lo + DEPTH_MARGIN, spec.nozzle * 0.5 * GRID + DEPTH_MARGIN)
}

/// The counts words S3 writes, claimed before the batch starts.
#[derive(Clone, Copy, Debug)]
pub struct S3Words {
    /// The scan total: labels plus seeds.
    pub total: CountWord,
    /// Labels or seeds past their capacity: re-run with K doubled.
    pub overflow: CountWord,
    /// A per-cell limit (16 cuts, 8 outline crossings) or the cell walk was exceeded.
    pub failed: CountWord,
}
impl S3Words {
    pub fn claim(pool: &mut Pool) -> Result<Self, JobError> {
        Ok(Self { total: pool.claim("s3.total", 1)?, overflow: pool.claim("s3.overflow", 1)?, failed: pool.claim("s3.failed", 1)? })
    }
    /// Ok(true) when the labels fit, Ok(false) when the batch re-runs with K doubled.
    pub fn check(&self, counts: &Counts) -> Result<bool, JobError> {
        let failed = counts.word(self.failed);
        if failed != 0 { return Err(JobError::Failed(format!("S3: {failed} segment cells past the walk's limits"))); }
        Ok(counts.word(self.overflow) == 0)
    }
}

/// The batch buffers S3 binds by name: plane k-1's outline FeatureSet and its feature_field,
/// and the rings.
#[derive(Clone, Copy, Debug)]
pub struct S3Inputs {
    pub seg: BufRef,
    pub seg_off: BufRef,
    pub set_slot: BufRef,
    pub seg_geo: BufRef,
    pub seg_meta: BufRef,
    pub band_slot: BufRef,
    pub cover: BufRef,
    pub node_side: BufRef,
    pub cand_off: BufRef,
    pub cand: BufRef,
    pub loops: BufRef,
    pub loop_meta: BufRef,
    pub ring_pt: BufRef,
}

/// S3's output: the readback buffer and its bytes (the item's [`Step::Output`]).
#[derive(Clone, Copy, Debug)]
pub struct S3Out { pub buf: BufRef, pub bytes: u64 }

/// The `s3_work` header with its tables, the work and output sizes.
struct Layout { upload: Vec<u32>, work_words: u32, out_words: u32, points: u32, loops: u32 }

fn layout(spec: &S3Spec, words: &S3Words) -> Result<Layout, JobError> {
    let fail = |e: String| JobError::Failed(format!("S3: {e}"));
    if spec.below.len() != spec.planes as usize { return Err(fail(format!("{} planes, {} below entries", spec.planes, spec.below.len()))); }
    if !(spec.nozzle > 0.0) { return Err(fail(format!("nozzle {}", spec.nozzle))); }
    let points = spec.points.bound().max(1);
    let loops = spec.loops.bound().max(1);
    let cells = spec.nx as u64 * spec.ny as u64 * spec.planes as u64;
    let cap = points as u64 * spec.k.max(1) as u64;
    let work = HEADER as u64 + LABEL_LEVELS as u64 + spec.planes as u64 + cells * 2 + loops as u64 * 4;
    let out = (OUT_HEADER as u64 + 2 * (loops as u64 + 1)).next_multiple_of(4) + cap * 6;
    if work >= u32::MAX as u64 / 4 || out >= u32::MAX as u64 / 4 { return Err(fail(format!("{work} work words, {out} output words"))); }
    let cap = cap as u32;
    let mut h = vec![0u32; HEADER as usize];
    let ((p, pw), (l, lw)) = (spec.points.words(), spec.loops.words());
    h[H_POINTS] = p;
    h[H_LOOPS] = l;
    h[H_POINTS_WORD] = pw;
    h[H_LOOPS_WORD] = lw;
    h[H_FLAGS] = if spec.slowdown { FLAG_SLOWDOWN } else { 0 };
    h[H_POINT_BOUND] = points;
    h[H_LOOP_BOUND] = loops;
    let c1 = spec.nozzle * 0.5 * GRID;
    let lo_max = field_depth(spec).0 - DEPTH_MARGIN;
    h[H_C1] = c1.to_bits();
    h[H_WINDOW] = (c1.max(lo_max) + spec.bias_max + 0.01).to_bits();
    h[H_LABEL_CAP] = cap;
    h[H_SEED_CAP] = cap;
    h[H_NX] = spec.nx;
    h[H_NY] = spec.ny;
    h[H_PLANES] = spec.planes;
    h[H_MITER] = spec.miter.to_bits();
    h[H_OVERFLOW] = words.overflow.0;
    h[H_FAILED] = words.failed.0;
    h[H_TOTAL] = words.total.0;
    let mut at = HEADER;
    let mut section = |n: u32| { let s = at; at += n; s };
    h[H_LEVELS] = section(LABEL_LEVELS);
    h[H_BELOW] = section(spec.planes);
    h[H_COARSE] = section(cells as u32 * 2);
    h[H_LOOPREC] = section(loops * 4);
    let work_words = at;
    let label_base = (OUT_HEADER + 2 * (loops + 1)).next_multiple_of(4);
    h[H_OUT_LABELS] = label_base;
    h[H_OUT_SEEDS] = label_base + cap * 4;
    let out_words = label_base + cap * 6;
    let mut levels = vec![NONE; LABEL_LEVELS as usize];
    for &(level, w) in &spec.widths {
        if level >= LABEL_LEVELS || !(w > 0.0) { return Err(fail(format!("labelled level {level} width {w}"))); }
        levels[level as usize] = full_support_level(w, spec.nozzle).to_bits();
    }
    let mut upload = h;
    upload.extend(levels);
    upload.extend(&spec.below);
    Ok(Layout { upload, work_words, out_words, points, loops })
}

/// The 50 ms element bound of a walk pass: a segment crosses at most 10 (nx + ny) + 3 fine cells,
/// each about 2 x 10^4 operations at worst (two cand evaluations, then two levels over a few
/// side stretches, each end tested against every other cand span), plus the degree scans; the
/// slowest supported GPU sustains about 0.5 x 10^12 simple operations per second.
fn walk_ceiling(spec: &S3Spec) -> u32 {
    let cells = 10 * (spec.nx + spec.ny) as u64 + 3;
    let per = cells * 20_000 + 4_000;
    (25_000_000_000u64 / per).clamp(64, 1 << 16) as u32
}

/// S3's steps over the batch's buffers, and its readback. Every count is placed by prefix sums
/// in point order, so the bytes never depend on job sizes.
pub fn steps(k: &S3Kernels, scans: &ScanKernels, pool: &mut Pool, words: &S3Words, inputs: &S3Inputs, spec: &S3Spec) -> Result<(Vec<Step>, S3Out), JobError> {
    let lay = layout(spec, words)?;
    let (points, loops) = (lay.points, lay.loops);
    let work = pool.buffer("s3 work", lay.work_words as u64 * 4);
    pool.upload(work, 0, lay.upload.iter().flat_map(|w| w.to_le_bytes()).collect());
    let count = pool.buffer("s3 count", points as u64 * 8);
    let scan = pool.buffer("s3 scan", points as u64 * 8);
    let out = pool.readback("s3 out", lay.out_words as u64 * 4);
    let bind = |name: LiveId| -> BufRef {
        [(id!(s3_work), work), (id!(s3_count), count), (id!(s3_scan), scan), (id!(s3_out), out),
            (id!(seg), inputs.seg), (id!(seg_off), inputs.seg_off), (id!(set_slot), inputs.set_slot),
            (id!(seg_geo), inputs.seg_geo), (id!(seg_meta), inputs.seg_meta), (id!(band_slot), inputs.band_slot),
            (id!(cover), inputs.cover), (id!(node_side), inputs.node_side), (id!(cand_off), inputs.cand_off),
            (id!(cand), inputs.cand), (id!(loops), inputs.loops), (id!(loop_meta), inputs.loop_meta),
            (id!(ring_pt), inputs.ring_pt), (id!(counts), BufRef::COUNTS)]
            .into_iter().find(|b| b.0 == name).map_or(BufRef::RUNTIME, |b| b.1)
    };
    let walk = walk_ceiling(spec);
    let pass = |name: &str, n: u32, cost: u32, ceiling: u32| Step::Pass(Pass::new(k.get(name), label(name), Elements::Fixed(n), cost, ceiling, bind));
    let mut steps = vec![
        pass("S3Prep", spec.nx * spec.ny * spec.planes, 1, 1 << 20),
        pass("S3Loops", loops, 1, 1 << 20),
        pass("S3Count", points, 64, walk),
    ];
    steps.extend(scans.exclusive(pool, count, scan, points * 2, words.total));
    steps.push(pass("S3Fill", points, 96, walk));
    steps.push(pass("S3LoopOff", loops + 1, 1, 1 << 20));
    Ok((steps, S3Out { buf: out, bytes: lay.out_words as u64 * 4 }))
}

/// One loop's labels and seeds as read back.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoopLabels { pub labels: Vec<S3Label>, pub seeds: Vec<S3Seed> }

/// A loop's line as read back: its points in nm (the points S3 labelled), and whether it is open.
#[derive(Clone, Copy, Debug)]
pub struct Line<'a> { pub points: &'a [[i64; 2]], pub open: bool }

/// walls::Labels' content in mm: pieces (from, to, supported, degree), seeds, and the line's
/// length; all zero for a loop without labels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LabelsMm { pub pieces: Vec<(f64, f64, bool, f64)>, pub seeds: Vec<f64>, pub length: f64 }

impl LoopLabels {
    /// Positions in mm of arc from point 0, from f64 lengths of the line's segments in order.
    pub fn to_mm(&self, line: Line<'_>) -> Result<LabelsMm, String> {
        if self.labels.is_empty() {
            if !self.seeds.is_empty() { return Err(format!("{} seeds without labels", self.seeds.len())); }
            return Ok(LabelsMm::default());
        }
        let n = line.points.len();
        let segs = if line.open { n.saturating_sub(1) } else { n };
        let len = |v: usize| {
            let (a, b) = (line.points[v], line.points[(v + 1) % n]);
            ((b[0] - a[0]) as f64).hypot((b[1] - a[1]) as f64) / NM
        };
        let mut arc = Vec::with_capacity(segs + 1);
        arc.push(0.0f64);
        for v in 0..segs { arc.push(arc[v] + len(v)); }
        let at = |vertex: u32, t: f32| -> Result<f64, String> {
            let v = vertex as usize;
            if v >= segs || !(0.0..=1.0).contains(&t) { return Err(format!("a position at point {vertex} + {t} on a line of {segs} segments")); }
            Ok(arc[v] + len(v) * t as f64)
        };
        let starts = self.labels.iter().map(|l| at(l.vertex, l.t)).collect::<Result<Vec<_>, _>>()?;
        let length = arc[segs];
        let mut pieces = Vec::with_capacity(starts.len());
        for (k, l) in self.labels.iter().enumerate() {
            let to = starts.get(k + 1).copied().unwrap_or(length);
            if to < starts[k] { return Err(format!("label {k} ends at {to} mm before it starts at {} mm", starts[k])); }
            pieces.push((starts[k], to, l.flags & LABEL_SUPPORTED != 0, l.degree as f64));
        }
        let seeds = self.seeds.iter().map(|s| at(s.vertex, s.t)).collect::<Result<Vec<_>, _>>()?;
        Ok(LabelsMm { pieces, seeds, length })
    }
}

/// The per-loop labels and seeds of an `s3_out` readback, in loop-table order.
pub fn decode(words: &[u32]) -> Result<Vec<LoopLabels>, String> {
    let get = |i: usize| words.get(i).copied().map(|w| w as usize).ok_or_else(|| format!("s3_out: {} words, wanted word {i}", words.len()));
    let (loops, labels, seeds) = (get(0)?, get(1)?, get(2)?);
    let (label_cap, seed_cap, label_base, seed_base, loop_bound) = (get(3)?, get(4)?, get(5)?, get(6)?, get(7)?);
    if labels > label_cap || seeds > seed_cap { return Err(format!("s3_out: {labels} labels / {seeds} seeds over capacity {label_cap} / {seed_cap}")); }
    if loops > loop_bound { return Err(format!("s3_out: {loops} loops over the bound {loop_bound}")); }
    let (label_off, seed_off) = (OUT_HEADER as usize, OUT_HEADER as usize + loop_bound + 1);
    let mut out = Vec::with_capacity(loops);
    for l in 0..loops {
        let (l0, l1) = (get(label_off + l)?, get(label_off + l + 1)?);
        let (s0, s1) = (get(seed_off + l)?, get(seed_off + l + 1)?);
        if l1 < l0 || l1 > labels || s1 < s0 || s1 > seeds { return Err(format!("s3_out: loop {l} spans labels {l0}..{l1}, seeds {s0}..{s1}")); }
        let lw = words.get(label_base + l0 * 4..label_base + l1 * 4).ok_or("s3_out: labels past the buffer")?;
        let sw = words.get(seed_base + s0 * 2..seed_base + s1 * 2).ok_or("s3_out: seeds past the buffer")?;
        out.push(LoopLabels { labels: S3Label::from_words(lw), seeds: S3Seed::from_words(sw) });
    }
    Ok(out)
}
