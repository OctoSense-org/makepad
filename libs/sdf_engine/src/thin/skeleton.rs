//! The exact inner skeleton of the thin site set, clipped to band fine cells,
//! and its node graph. No field evaluator is involved: each cell sweeps the
//! bisector of every pair of its listed sites in closed form.
//!
//! An edge is central where its radius changes slower than sin(θ / 2) per unit
//! of length (θ the transition angle). Per bisector kind the slope is closed
//! form in the bisector's own parameter t: constant on a line-line bisector,
//! |t| / sqrt(d² + t²) on a point-point one (d half the points' distance), and
//! |t| / sqrt(h² + t²) on a point-line parabola (h the point's height over the
//! line). So a run splits at ±k = ±d · tan(θ / 2) or ±h · tan(θ / 2), the
//! parameters where the slope reaches the sine, and each piece takes the flag
//! of its midpoint parameter. The split depends on the two sites alone: no
//! cell clip, no cross-cell reduction, and the same under any motion.
use makepad_draw::*;

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.ThinVoronoi = {
        ..mod.sdf.Thin
        // Site records by first seg (sites.rs): chord, and (last seg, next
        // site, previous site, flags: 1 reflex start vertex, 2 a site).
        site: shader.buffer(vec4f)
        site_info: shader.buffer(vec4u)
        // Per band cell (slot * 100 + local): first list entry, site count.
        site_lists: shader.buffer(vec2u)
        site_list: shader.buffer(u32)
        band_slot: shader.buffer(u32)
        output: shader.buffer_rw(vec4u)
        // Per band cell its span range (two words), then the graph regions.
        thin: shader.buffer_rw(u32)
        site_list_of: fn(cell: u32) -> vec2u { return self.site_lists[cell] }
        // The plane fine cell of band cell `cell`.
        cell_grid: fn(cell: u32) -> vec2u {
            let local = cell % 100u
            let coarse = self.slot(cell / 100u).x
            return vec2u((coarse % self.nx()) * 10u + local % 10u, (coarse / self.nx()) * 10u + local / 10u)
        }
        // Site x: segment 2r (the chord of run r) or point 2r + 1 (its start vertex).
        sa: fn(x: u32) -> vec2 { return self.site[x / 2u].xy }
        sb: fn(x: u32) -> vec2 { return self.site[x / 2u].zw }
        // The start of the chord before point site x's vertex.
        s_in: fn(x: u32) -> vec2 { return self.site[self.site_info[x / 2u].z].xy }
        site_point: fn(s: u32) -> vec2 { return self.sa(s) }
        tangent: fn(s: u32) -> vec2 { return normalize(self.sb(s) - self.sa(s)) }
        normal: fn(s: u32) -> vec2 { let t = self.tangent(s); return vec2(0.0 - t.y, t.x) }
        site_less: fn(a: u32, b: u32) -> bool {
            if b == 4294967295u { return true }
            return a < b
        }
        domain: fn(s: u32, p: vec2) -> bool {
            let a = self.sa(s); let b = self.sb(s)
            if s % 2u == 0u {
                let t = dot(p - a, b - a)
                return t >= 0.0 && t <= dot(b - a, b - a)
            }
            return dot(p - a, a - self.s_in(s)) >= 0.0 && dot(p - a, b - a) <= 0.0
        }
        distance2: fn(s: u32, p: vec2) -> f32 {
            if s % 2u != 0u { let d = p - self.site_point(s); return dot(d, d) }
            let d = dot(p - self.sa(s), self.normal(s))
            return d * d
        }
        // A segment site and a point site at one of its ends: the normal ray
        // between them is not skeleton.
        excluded: fn(a: u32, b: u32) -> bool {
            if a % 2u == b % 2u { return false }
            var segment = a; var point = b
            if a % 2u != 0u { segment = b; point = a }
            let r = segment / 2u
            let v = point / 2u
            return v == r || v == self.site_info[r].y
        }
        point: fn(p0: vec2, p1: vec2, p2: vec2, t: f32) -> vec2 { return p0 + t * (p1 + t * p2) }
        polynomial: fn(p0: vec2, p1: vec2, p2: vec2, n: vec2, c: f32) -> vec3 {
            return vec3(dot(p0, n) - c, dot(p1, n), dot(p2, n))
        }
        root_after: fn(poly: vec3, after: f32, limit: f32) -> f32 {
            var result = limit
            if poly.z == 0.0 {
                if poly.y != 0.0 { let t = (0.0 - poly.x) / poly.y; if t > after && t < result { result = t } }
            } else {
                let d = poly.y * poly.y - 4.0 * poly.z * poly.x
                if d >= 0.0 {
                    var sign = 1.0
                    if poly.y < 0.0 { sign = 0.0 - 1.0 }
                    let q = (0.0 - 0.5) * (poly.y + sign * sqrt(d))
                    if q != 0.0 {
                        let a = q / poly.z; let b = poly.x / q
                        if a > after && a < result { result = a }
                        if b > after && b < result { result = b }
                    } else {
                        let t = (0.0 - poly.y) / (2.0 * poly.z)
                        if t > after && t < result { result = t }
                    }
                }
            }
            return result
        }
        // Smallest root after t of site s's two domain boundaries.
        domain_next: fn(s: u32, origin: vec2, p0: vec2, p1: vec2, p2: vec2, t: f32, limit: f32) -> f32 {
            let a = self.sa(s) - origin; let b = self.sb(s) - origin
            var first = self.polynomial(p0, p1, p2, b - a, dot(a, b - a))
            var second = self.polynomial(p0, p1, p2, b - a, dot(b, b - a))
            if s % 2u != 0u {
                let u = a - (self.s_in(s) - origin)
                first = self.polynomial(p0, p1, p2, u, dot(a, u))
                second = self.polynomial(p0, p1, p2, b - a, dot(a, b - a))
            }
            return self.root_after(second, t, self.root_after(first, t, limit))
        }
        // Equidistance of a boundary point v with the pair: against the
        // pair's point site when it has one (linear in p), else against the
        // line distance of two segments (then p and the radius are linear).
        point_poly: fn(v: vec2, reference: u32, origin: vec2, p0: vec2, p1: vec2, p2: vec2, radius: vec3) -> vec3 {
            if reference % 2u != 0u {
                let u = self.site_point(reference) - origin
                return self.polynomial(p0, p1, p2, 2.0 * (u - v), dot(u, u) - dot(v, v))
            }
            let d = p0 - v
            return vec3(dot(d, d) - radius.x * radius.x, 2.0 * (dot(d, p1) - radius.x * radius.y), dot(p1, p1) - radius.y * radius.y)
        }
        // A boundary point of the pair's own sites (a point site, or a
        // segment's endpoint) is never strictly nearer than the pair inside
        // their domains; testing it would only let rounding block the pair.
        own_point: fn(v: vec2, a: u32, b: u32) -> bool {
            var own = false
            for k in 0..2u {
                var s = a
                if k != 0u { s = b }
                if s % 2u != 0u {
                    let w = self.site_point(s)
                    if v.x == w.x && v.y == w.y { own = true }
                } else {
                    let x = self.sa(s); let y = self.sb(s)
                    if (v.x == x.x && v.y == x.y) || (v.x == y.x && v.y == y.y) { own = true }
                }
            }
            return own
        }
        // The next parameter after t where a predicate of pair (a, b) can
        // change: the cell box, the reach cap and zero radius, both domains,
        // and per other candidate its domain and equidistance, plus the
        // equidistance of a segment's two endpoints.
        pair_next: fn(list: vec2u, a: u32, b: u32, reference: u32, origin: vec2, lo: vec2, hi: vec2, p0: vec2, p1: vec2, p2: vec2,
                radius: vec3, cap: f32, t: f32, high: f32) -> f32 {
            let pp = a % 2u != 0u && b % 2u != 0u
            let x = vec3(p0.x, p1.x, p2.x); let y = vec3(p0.y, p1.y, p2.y)
            var next = self.root_after(x - vec3(lo.x - origin.x, 0.0, 0.0), t, high)
            next = self.root_after(x - vec3(hi.x - origin.x, 0.0, 0.0), t, next)
            next = self.root_after(y - vec3(lo.y - origin.y, 0.0, 0.0), t, next)
            next = self.root_after(y - vec3(hi.y - origin.y, 0.0, 0.0), t, next)
            next = self.root_after(radius - vec3(cap, 0.0, 0.0), t, next)
            next = self.root_after(radius, t, next)
            next = self.domain_next(a, origin, p0, p1, p2, t, next)
            next = self.domain_next(b, origin, p0, p1, p2, t, next)
            for si in 0..list.y {
                let s = self.site_list[list.x + si]
                if s == a || s == b { continue }
                next = self.domain_next(s, origin, p0, p1, p2, t, next)
                if s % 2u == 0u {
                    let n = self.normal(s)
                    let d = self.polynomial(p0, p1, p2, n, dot(self.sa(s) - origin, n))
                    if pp { next = self.root_after(vec3(d.x * d.x, 2.0 * d.x * d.y, d.y * d.y) - radius, t, next) }
                    else { next = self.root_after(d - radius, t, next); next = self.root_after(d + radius, t, next) }
                    for end in 0..2u {
                        var v = self.sa(s)
                        if end != 0u { v = self.sb(s) }
                        if !self.own_point(v, a, b) {
                            next = self.root_after(self.point_poly(v - origin, reference, origin, p0, p1, p2, radius), t, next)
                        }
                    }
                } else {
                    next = self.root_after(self.point_poly(self.site_point(s) - origin, reference, origin, p0, p1, p2, radius), t, next)
                }
            }
            return next
        }
        // The pair's bisector at t is skeleton: inside the half-open cell,
        // within the reach, in both domains, and no candidate site and no
        // segment endpoint strictly nearer. The endpoints block every outside
        // point whose nearest boundary point is a vertex; a point nearest to
        // an open segment's inner side is material, so no inside test runs.
        pair_valid: fn(list: vec2u, a: u32, b: u32, origin: vec2, lo: vec2, hi: vec2, p0: vec2, p1: vec2, p2: vec2,
                radius: vec3, cap: f32, t: f32) -> bool {
            let p = origin + self.point(p0, p1, p2, t)
            let value = radius.x + t * (radius.y + t * radius.z)
            if p.x < lo.x || p.y < lo.y || p.x >= hi.x || p.y >= hi.y || value < 0.0 || value > cap
                || !self.domain(a, p) || !self.domain(b, p) { return false }
            var r2 = value * value
            if a % 2u != 0u && b % 2u != 0u { r2 = value }
            var valid = true
            for si in 0..list.y {
                let s = self.site_list[list.x + si]
                if s == a || s == b { continue }
                if self.domain(s, p) && self.distance2(s, p) < r2 { valid = false }
                if s % 2u == 0u {
                    for end in 0..2u {
                        var v = self.sa(s)
                        if end != 0u { v = self.sb(s) }
                        let e = p - v
                        if !self.own_point(v, a, b) && dot(e, e) < r2 { valid = false }
                    }
                }
            }
            return valid
        }
        // Chords of one curve range within the curve error: none for an empty range.
        run_pieces: fn(p2: vec2, s0: f32, s1: f32) -> u32 {
            if s1 <= s0 { return 0u }
            return max(1u, u32(ceil(sqrt(length(p2) * (s1 - s0) * (s1 - s0) / (4.0 * self.curve_error())))))
        }
        emit: fn(at: u32, sa: u32, sb: u32, cell: u32, pair: u32, tag: u32, cuts: u32, plane: u32, a: vec2, b: vec2, ra: f32, rb: f32, central: bool) {
            if at >= self.spans() { atomic_add(self.counts[self.cfg[2u].x + 3u], 1u); return }
            var ca = cuts & 1u; var cb = (cuts >> 1u) & 1u
            if central { ca |= 2u; cb |= 2u }
            self.output[at * 4u] = vec4u(asuint(a.x), asuint(a.y), asuint(ra), ca)
            self.output[at * 4u + 1u] = vec4u(asuint(b.x), asuint(b.y), asuint(rb), cb)
            self.output[at * 4u + 2u] = vec4u(sa, sb, cell, pair)
            self.output[at * 4u + 3u] = vec4u(tag, tag + 1u, plane, 0u)
        }
    }

    mod.sdf.ThinSlotInfo = {
        ..mod.sdf.ThinVoronoi
        band: shader.buffer(u32)
        band_off: shader.buffer(u32)
        cfg: shader.buffer_rw(vec4u)
        main: fn() {
            let slot = self.index()
            if slot == 4294967295u || slot * 100u >= self.cells() { return }
            var plane = 0u
            for p in 0..self.cfg[0u].z { if self.band_off[p] <= slot { plane = p } }
            self.cfg[4u + slot] = vec4u(self.band[slot], plane, 0u, 0u)
        }
    }

    // A band cell's sites: the chords of its cand features' runs, and at each
    // end joint of such a run the run across it and the joint's point site
    // (if the chords turn reflex there), each where it passes the field's
    // cand test for the cell (feature_field's lower bound over the cell's
    // disk, within the cell's threshold). The field measures segs, not
    // chords: a chord's slab reaches past its end seg's by the angle between
    // them, and a joint reflex between chords can be convex between its two
    // segs, so near a joint the field may name only the run across it. A
    // cand entry adds at most five sites, so each list lives at five times
    // its cell's cand offset.
    mod.sdf.ThinSiteLists = {
        ..mod.sdf.ThinVoronoi
        cand_off: shader.buffer(u32)
        cand: shader.buffer(u32)
        cell_t: shader.buffer(u32)
        site_of: shader.buffer(u32)
        site_list: shader.buffer_rw(u32)
        site_lists: shader.buffer_rw(vec2u)
        insert: fn(at: u32, size: u32, s: u32) -> u32 {
            var pos = 0u
            while pos < size && self.site_less(self.site_list[at + pos], s) { pos += 1u }
            if pos == size || self.site_list[at + pos] != s {
                var j = size
                while j > pos { self.site_list[at + j] = self.site_list[at + j - 1u]; j -= 1u }
                self.site_list[at + pos] = s
                return size + 1u
            }
            return size
        }
        // The field's cand tests (its CELL_RADIUS disk about the centre c):
        // chord site 2r by its slab, point site x by its wedge.
        chord_near: fn(c: vec2, r: u32, threshold: f32) -> bool {
            let a = self.site[r].xy
            let len = length(self.site[r].zw - a)
            if len == 0.0 { return false }
            let u = (self.site[r].zw - a) / len
            let t = dot(c - a, u)
            if t < 0.0 - 0.7072 || t > len + 0.7072 { return false }
            return abs(self.cross(u, c - a)) - 0.7072 <= threshold
        }
        point_near: fn(c: vec2, x: u32, threshold: f32) -> bool {
            let a = self.sa(x)
            let d = c - a
            if dot(d, normalize(a - self.s_in(x))) < 0.0 - 0.7072 || dot(d, normalize(self.sb(x) - a)) > 0.7072 { return false }
            return length(d) - 0.7072 <= threshold
        }
        main: fn() {
            let cell = self.index()
            if cell == 4294967295u || cell >= self.cells() { return }
            let g = self.cell_grid(cell)
            let c = vec2(f32(g.x) + 0.5, f32(g.y) + 0.5)
            let threshold = asfloat(self.cell_t[cell])
            let at = self.cand_off[cell] * 5u
            var size = 0u
            for k in self.cand_off[cell]..self.cand_off[cell + 1u] {
                let r = self.site_of[self.cand[k] & 2147483647u]
                let info = self.site_info[r]
                size = self.insert(at, size, r * 2u)
                // The joint at r's start (the run across: the previous) and at its end (the next).
                for end in 0..2u {
                    var joint = r; var across = info.z
                    if end == 1u { joint = info.y; across = info.y }
                    if (self.site_info[joint].w & 1u) != 0u && self.point_near(c, joint * 2u + 1u, threshold) { size = self.insert(at, size, joint * 2u + 1u) }
                    if self.chord_near(c, across, threshold) { size = self.insert(at, size, across * 2u) }
                }
            }
            self.site_lists[cell] = vec2u(at, size)
            // The longest list bounds the sites one node can tie.
            if size != 0u { atomic_max(self.counts[self.cfg[2u].x + 13u], size) }
        }
    }

    mod.sdf.ThinPairCount = {
        ..mod.sdf.ThinVoronoi
        pair_count: shader.buffer_rw(u32)
        main: fn() {
            let cell = self.index()
            if cell == 4294967295u || cell >= self.cells() { return }
            let n = self.site_lists[cell].y
            var pairs = 0u
            if n >= 2u { pairs = n * (n - 1u) / 2u }
            self.pair_count[cell] = pairs
        }
    }

    mod.sdf.ThinPairs = {
        ..mod.sdf.ThinVoronoi
        // Per cell its first pair (exclusive scan); per pair its span count
        // (the count sweep) and first span (exclusive scan).
        pair_off: shader.buffer(u32)
        pair_spans: shader.buffer_rw(u32)
        span_off: shader.buffer(u32)
        pairs: fn() -> u32 { return self.word(1u) }
        // One element per candidate pair of a skeleton cell. The pair's
        // bisector is swept in order of its predicate roots; each maximal
        // valid run becomes chords. Count (dispatch_range.w = 0) and write
        // (1) sweeps are identical.
        main: fn() {
            let q = self.index()
            if q == 4294967295u || q >= self.pairs() { return }
            let write = self.dispatch_range.w != 0u
            // The pair's cell: the last whose first pair is at or before q
            // (empty cells share their successor's offset and come first).
            var lo_cell = 0u; var hi_cell = self.cells()
            for probe in 0..32u {
                if hi_cell - lo_cell <= 1u { break }
                let middle = lo_cell + (hi_cell - lo_cell) / 2u
                if self.pair_off[middle] <= q { lo_cell = middle } else { hi_cell = middle }
            }
            let cell = lo_cell
            var j = q - self.pair_off[cell]
            let pair = j
            let list = self.site_list_of(cell)
            var ai = 0u
            for row in 0..list.y {
                if j < list.y - 1u - ai { break }
                j -= list.y - 1u - ai; ai += 1u
            }
            let bi = ai + 1u + j
            let g = self.cell_grid(cell)
            let lo = vec2(f32(g.x), f32(g.y)); let hi = lo + vec2(1.0)
            let a = self.site_list[list.x + ai]
            let b = self.site_list[list.x + bi]
            // All cells evaluate exactly the same curve expression.
            var origin = self.sa(a)
            if a % 2u != 0u {
                origin = self.site_point(a)
                if b % 2u == 0u { origin = self.sa(b) }
            }
            var valid = !self.excluded(a, b)
            var p0 = vec2(0.0); var p1 = vec2(0.0); var p2 = vec2(0.0)
            var radius = vec3(0.0)
            let pp = a % 2u != 0u && b % 2u != 0u
            if a % 2u == b % 2u {
                var n = vec2(0.0); var c = 0.0
                if pp {
                    let u = self.site_point(a) - origin; let v = self.site_point(b) - origin
                    n = 2.0 * (v - u); c = dot(v, v) - dot(u, u)
                } else {
                    let na = self.normal(a); let nb = self.normal(b)
                    n = na - nb; c = dot(self.sa(a) - origin, na) - dot(self.sa(b) - origin, nb)
                }
                let norm = dot(n, n)
                if norm == 0.0 { valid = false }
                else {
                    p0 = n * (c / norm); p1 = vec2(0.0 - n.y, n.x) / sqrt(norm)
                    if pp { let d = p0 - (self.site_point(a) - origin); radius = vec3(dot(d, d), 2.0 * dot(d, p1), dot(p1, p1)) }
                    else { let n = self.normal(a); radius = vec3(dot(p0 - (self.sa(a) - origin), n), dot(p1, n), 0.0) }
                }
            } else {
                var line = a; var point = b
                if a % 2u != 0u { line = b; point = a }
                let n = self.normal(line); let u = self.tangent(line)
                let focus = self.site_point(point) - origin
                let h = dot(focus - (self.sa(line) - origin), n)
                if h <= 0.0 { valid = false }
                else {
                    p0 = focus - n * (h * 0.5); p1 = u; p2 = n / (2.0 * h)
                    radius = vec3(h * 0.5, 0.0, 1.0 / (2.0 * h))
                }
            }
            var count = 0u
            var base = 0u
            if write { base = self.span_off[q] }
            if valid {
                // The parameter is the tangent coordinate, so the box
                // projects to centre +- extent. That rounded window only
                // bounds the root search: it is taken twice as wide, so the
                // cell's four boundary roots alone clip the curve and
                // neighbouring cells share each boundary point bit for bit.
                let centre = dot((lo + hi) * 0.5 - origin - p0, p1)
                let extent = dot((hi - lo) * 0.5, abs(p1))
                let low = centre - 2.0 * extent; let high = centre + 2.0 * extent
                var cap = self.reach()
                if pp { cap *= cap }
                let cap_poly = radius - vec3(cap, 0.0, 0.0)
                let infinity = asfloat(2139095040u)
                let cut0 = self.root_after(cap_poly, 0.0 - infinity, infinity)
                let cut1 = self.root_after(cap_poly, cut0, infinity)
                // The central bound: along a point-point bisector (r² = d² + t²)
                // and a point-line parabola (r = h / 2 + t² / (2 h)) the radius
                // slope is sin(θ / 2) at |t| = d · tan(θ / 2), h · tan(θ / 2);
                // along a line-line bisector it is the constant radius.y.
                let sine = self.central_sine()
                let tangent = sine / sqrt(max(1.0 - sine * sine, 0.0))
                let curved = pp || a % 2u != b % 2u
                var bound = 2.0 * radius.x * tangent
                if pp { bound = sqrt(radius.x) * tangent }
                var reference = a
                if a % 2u == 0u { reference = b }
                // Each predicate polynomial is at most quadratic, so the
                // sweep visits at most two roots per polynomial.
                let steps = 2u * (10u + 6u * list.y) + 2u
                var t = low
                var run = false
                // A run's end waits for the next valid interval: a gap within
                // the roundoff of its point is a tie at one point in exact
                // arithmetic, so the run goes on through it.
                var pending = false
                var start = 0.0
                var end = 0.0
                var tag = 0u
                for step in 0..(steps + 1u) {
                    var next = high
                    var here = false
                    if t < high {
                        next = self.pair_next(list, a, b, reference, origin, lo, hi, p0, p1, p2, radius, cap, t, high)
                        let middle = t + (next - t) * 0.5
                        here = middle > t && self.pair_valid(list, a, b, origin, lo, hi, p0, p1, p2, radius, cap, middle)
                    }
                    var close = false
                    var reopen = false
                    if here && pending {
                        let pe = origin + self.point(p0, p1, p2, end)
                        let pt = origin + self.point(p0, p1, p2, t)
                        if length(pt - pe) > self.roundoff(pe) { close = true; reopen = true }
                        pending = false
                    } else if here && !run { run = true; start = t
                    } else if !here && run && !pending { pending = true; end = t }
                    if pending && t >= high { close = true }
                    if close {
                        // The run splits where its radius slope along the curve
                        // reaches sin(θ / 2), so every piece is central or not
                        // as a whole: [start, b1], [b1, b2] (central), [b2, end].
                        var b1 = start; var b2 = end
                        if curved { b1 = clamp(0.0 - bound, start, end); b2 = clamp(bound, start, end) }
                        var n0 = self.run_pieces(p2, start, b1)
                        var n1 = self.run_pieces(p2, b1, b2)
                        let n2 = self.run_pieces(p2, b2, end)
                        if n0 + n1 + n2 == 0u { b1 = start; b2 = end; n1 = 1u }
                        // Run ends at the reach: they bound the cuts, the reach faces' sources.
                        if !write {
                            var at_reach = 0u
                            if start == cut0 || start == cut1 { at_reach += 1u }
                            if end == cut0 || end == cut1 { at_reach += 1u }
                            if at_reach != 0u { atomic_add(self.counts[self.cfg[2u].x + 16u], at_reach) }
                        }
                        if write {
                            for k in 0..(n0 + n1 + n2) {
                                var s0 = start; var s1 = b1; var n = n0; var j = k
                                if k >= n0 { s0 = b1; s1 = b2; n = n1; j = k - n0 }
                                if k >= n0 + n1 { s0 = b2; s1 = end; n = n2; j = k - n0 - n1 }
                                var u0 = s0 + (s1 - s0) * f32(j) / f32(n)
                                var u1 = s0 + (s1 - s0) * f32(j + 1u) / f32(n)
                                if j == 0u { u0 = s0 }
                                if j + 1u == n { u1 = s1 }
                                var ra = radius.x + u0 * (radius.y + u0 * radius.z)
                                var rb = radius.x + u1 * (radius.y + u1 * radius.z)
                                if pp { ra = sqrt(max(ra, 0.0)); rb = sqrt(max(rb, 0.0)) }
                                // No piece straddles a bound: its midpoint decides.
                                var central = abs(radius.y) <= sine
                                if curved { central = abs((u0 + u1) * 0.5) <= bound }
                                var cuts = 0u
                                if u0 == cut0 || u0 == cut1 { cuts |= 1u }
                                if u1 == cut0 || u1 == cut1 { cuts |= 2u }
                                self.emit(base + count + k, a, b, cell, pair, tag + k, cuts, self.slot(cell / 100u).y,
                                    origin + self.point(p0, p1, p2, u0), origin + self.point(p0, p1, p2, u1), ra, rb, central)
                            }
                        }
                        count += n0 + n1 + n2; tag += n0 + n1 + n2 + 1u
                        if reopen { start = t } else { run = false; pending = false }
                    }
                    if t >= high { break }
                    t = next
                }
            }
            if !write { self.pair_spans[q] = count }
        }
    }

    // Per band cell its span range, read by cell-local node incidence: the
    // spans of its pairs, which the write sweep stored contiguously.
    mod.sdf.ThinCellSpans = {
        ..mod.sdf.ThinVoronoi
        pair_count: shader.buffer(u32)
        pair_off: shader.buffer(u32)
        span_off: shader.buffer(u32)
        span_at: fn(q: u32) -> u32 {
            if q >= self.word(1u) { return self.spans() }
            return self.span_off[q]
        }
        main: fn() {
            let cell = self.index()
            if cell == 4294967295u || cell >= self.cells() { return }
            let first = self.pair_off[cell]
            self.thin[cell * 2u] = self.span_at(first)
            self.thin[cell * 2u + 1u] = self.span_at(first + self.pair_count[cell])
        }
    }

    mod.sdf.ThinGraph = {
        ..mod.sdf.ThinVoronoi
        // The plane fine cell of end e's span.
        span_grid: fn(e: u32) -> vec2u { return self.cell_grid(self.output[(e / 2u) * 4u + 2u].z) }
        // The fine cells next to end e's span cell, columns x0..x1, rows
        // y0..y1. A node is its key, and its ends' spans lie in neighbouring
        // cells: ends of one vertex solved from different pairs of near
        // parallel sites can lie well past roundoff of each other, so no
        // window around e's own point. Neighbourhood is symmetric, so every
        // end of a node sees the same ends (ThinSelectPaths checks it).
        node_cells: fn(e: u32) -> vec4u {
            let g = self.span_grid(e)
            return vec4u(max(g.x, 1u) - 1u, min(g.x + 1u, self.nx() * 10u - 1u), max(g.y, 1u) - 1u, min(g.y + 1u, self.ny() * 10u - 1u))
        }
        plane_of: fn(e: u32) -> u32 { return self.output[(e / 2u) * 4u + 3u].z }
        // The band cell of plane fine cell (gx, gy), or NONE off the band.
        band_cell: fn(plane: u32, gx: u32, gy: u32) -> u32 {
            let slot = self.band_slot[(plane * self.ny() + gy / 10u) * self.nx() + gx / 10u]
            if slot == 4294967295u { return 4294967295u }
            return slot * 100u + (gy % 10u) * 10u + gx % 10u
        }
        node_cell: fn(e: u32, gx: u32, gy: u32) -> u32 { return self.band_cell(self.plane_of(e), gx, gy) }
        // A cell's spans, as the pair sweep wrote them.
        cell_spans: fn(cell: u32) -> vec2u {
            if cell == 4294967295u { return vec2u(0u, 0u) }
            return vec2u(self.thin[cell * 2u], self.thin[cell * 2u + 1u])
        }
        endpoint: fn(e: u32) -> vec4u { return self.output[(e / 2u) * 4u + e % 2u] }
        key: fn(e: u32) -> vec4u {
            let at = self.keys() + e * 4u
            return vec4u(self.thin[at], self.thin[at + 1u], self.thin[at + 2u], self.thin[at + 3u])
        }
        key_for: fn(e: u32) -> vec4u {
            let value = self.endpoint(e)
            let p = self.xy(value)
            let r = asfloat(value.z)
            let pair = self.output[(e / 2u) * 4u + 2u]
            let cell = pair.z
            let plane = self.plane_of(e)
            let list = self.site_list_of(cell)
            // Use the f32 roundoff bound to identify incidence, never arrival
            // order or a new geometric welding tolerance.
            let error = self.roundoff(p)
            var sites = vec3u(4294967295u)
            var ties = 0u
            var fingerprint = 2166136261u
            for si in 0..list.y {
                let s = self.site_list[list.x + si]
                var foot = self.site_point(s)
                var in_domain = true
                if s % 2u == 0u {
                    let a = self.sa(s); let u = self.sb(s) - a
                    let t = dot(p - a, u) / dot(u, u)
                    in_domain = t * length(u) >= (0.0 - error) && (t - 1.0) * length(u) <= error
                    foot = a + u * clamp(t, 0.0, 1.0)
                } else {
                    in_domain = dot(p - foot, normalize(foot - self.s_in(s))) >= (0.0 - error)
                        && dot(p - foot, normalize(self.sb(s) - foot)) <= error
                }
                if in_domain && abs(length(p - foot) - r) <= error {
                    if ties == 0u { sites.x = s }
                    if ties == 1u { sites.y = s }
                    if ties == 2u { sites.z = s }
                    if ties < self.ties() { self.thin[self.incidence() + e * self.ties() + ties] = s }
                    else { atomic_add(self.counts[self.cfg[2u].x + 4u], 1u) }
                    fingerprint = (fingerprint ^ s) * 16777619u
                    ties += 1u
                }
            }
            if ties >= 3u { return vec4u(1u, ties, self.vertex_root(sites, p), fingerprint) }
            if (value.w & 1u) != 0u {
                var branch = 0u
                var point = pair.x; var line = pair.y
                if point % 2u == 0u { point = pair.y; line = pair.x }
                if point % 2u != 0u && line % 2u == 0u {
                    if dot(p - self.site_point(point), self.tangent(line)) < 0.0 { branch = 1u }
                } else if point % 2u != 0u {
                    let d = self.site_point(pair.y) - self.site_point(pair.x)
                    let v = p - (self.site_point(pair.y) + self.site_point(pair.x)) * 0.5
                    if self.cross(d, v) < 0.0 { branch = 1u }
                }
                return vec4u(2u, pair.x, pair.y, branch)
            }
            if r <= error { return vec4u(0u, pair.x, pair.y, 0u) }
            let nearest = round(p)
            let on_x = abs(p.x - nearest.x) <= error
            let on_y = abs(p.y - nearest.y) <= error
            let nx = self.nx() * 10u + 1u
            let ny = self.ny() * 10u + 1u
            if on_x && on_y { return vec4u(5u, pair.x, pair.y, (plane * ny + u32(nearest.y)) * nx + u32(nearest.x)) }
            if on_x || on_y {
                var line_index = u32(nearest.y)
                var axis = 1u
                if on_x { line_index = u32(nearest.x); axis = 0u }
                var branch = 0u
                if pair.x % 2u != pair.y % 2u {
                    var point = pair.x; var line = pair.y
                    if point % 2u == 0u { point = pair.y; line = pair.x }
                    let n = self.normal(line); let u = self.tangent(line)
                    let h = dot(self.site_point(point) - self.sa(line), n)
                    let t = dot(p - self.site_point(point), u)
                    let derivative = u + n * (t / h)
                    if (on_x && derivative.x < 0.0) || (!on_x && derivative.y < 0.0) { branch = 1u }
                }
                return vec4u(3u, pair.x, pair.y, ((plane * (nx + ny) + line_index + axis * nx) * 2u) + branch)
            }
            let tags = self.output[(e / 2u) * 4u + 3u]
            var tag = tags.x
            if e % 2u != 0u { tag = tags.y }
            return vec4u(4u, cell, pair.w, tag)
        }
        vertex_root: fn(sites: vec3u, p: vec2) -> u32 {
            // Three lines or three points have at most one inner solution.
            let kinds = sites.x % 2u + sites.y % 2u + sites.z % 2u
            if kinds == 0u || kinds == 3u { return 0u }
            // A segment and its own endpoint are equidistant only on the
            // normal ray. Its intersection is a repeated root, not two nodes.
            if self.excluded(sites.x, sites.y) || self.excluded(sites.x, sites.z) || self.excluded(sites.y, sites.z) { return 0u }
            var a = sites.x; var b = sites.y; var c = sites.z
            if a % 2u != b % 2u {
                if a % 2u == c % 2u { b = sites.z; c = sites.y }
                else { a = sites.y; b = sites.z; c = sites.x }
            }
            var origin = self.sa(a)
            var n = self.normal(a) - self.normal(b)
            var constant = dot(self.sa(a) - origin, self.normal(a)) - dot(self.sa(b) - origin, self.normal(b))
            if kinds == 2u {
                origin = self.site_point(a)
                let v = self.site_point(b) - origin
                n = 2.0 * v; constant = dot(v, v)
            }
            let p0 = n * (constant / dot(n, n))
            let p1 = normalize(vec2(0.0 - n.y, n.x))
            var line = a; var point = c
            if kinds == 2u { line = c; point = a }
            let normal = self.normal(line)
            let d = p0 - (self.site_point(point) - origin)
            let r0 = dot(p0 - (self.sa(line) - origin), normal)
            let r1 = dot(p1, normal)
            let qa = dot(p1, p1) - r1 * r1
            let qb = 2.0 * (dot(d, p1) - r0 * r1)
            let qc = dot(d, d) - r0 * r0
            if qb * qb - 4.0 * qa * qc <= 0.0 { return 0u }
            if qa != 0.0 && dot(p - origin - p0, p1) > (0.0 - qb) / (2.0 * qa) { return 1u }
            return 0u
        }
        same_node: fn(a: u32, b: u32) -> bool {
            let key = self.key(a)
            if !self.equal(key, self.key(b)) { return false }
            if key.x == 1u {
                for k in 0..min(key.y, self.ties()) {
                    if self.thin[self.incidence() + a * self.ties() + k] != self.thin[self.incidence() + b * self.ties() + k] { return false }
                }
            }
            return true
        }
        edge: fn(e: u32) -> bool {
            return !self.same_node(e, e ^ 1u)
        }
    }
    mod.sdf.ThinKeys = {
        ..mod.sdf.ThinGraph
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            let key = self.key_for(e)
            let at = self.keys() + e * 4u
            self.thin[at] = key.x; self.thin[at + 1u] = key.y; self.thin[at + 2u] = key.z; self.thin[at + 3u] = key.w
        }
    }
    mod.sdf.ThinJumpInit = {
        ..mod.sdf.ThinGraph
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            let opposite = self.graph() + (e ^ 1u) * 4u
            var next = 4294967295u
            if self.thin[opposite + 2u] == 2u { next = self.thin[opposite] }
            self.thin[self.jump_a() + e * 2u] = next
            self.thin[self.jump_a() + e * 2u + 1u] = e / 2u
            // Only a degree-two peer is read above, and only other ends unlink.
            if self.thin[self.graph() + e * 4u + 2u] != 2u { self.thin[self.graph() + e * 4u] = 4294967295u }
        }
    }
    // Pointer jumping over the component's arms: dispatch_range.z is the round.
    mod.sdf.ThinJump = {
        ..mod.sdf.ThinGraph
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            var src = self.jump_a(); var dst = self.jump_b()
            if self.dispatch_range.z % 2u != 0u { src = self.jump_b(); dst = self.jump_a() }
            var next = self.thin[src + e * 2u]
            var root = self.thin[src + e * 2u + 1u]
            if next != 4294967295u { root = min(root, self.thin[src + next * 2u + 1u]); next = self.thin[src + next * 2u] }
            self.thin[dst + e * 2u] = next; self.thin[dst + e * 2u + 1u] = root
        }
    }
    // dispatch_range.w: the bank the last jump round wrote (0 a, 1 b).
    mod.sdf.ThinFinish = {
        ..mod.sdf.ThinGraph
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            var src = self.jump_a()
            if self.dispatch_range.w != 0u { src = self.jump_b() }
            self.thin[self.graph() + e * 4u + 3u] = min(self.thin[src + e * 2u + 1u], self.thin[src + (e ^ 1u) * 2u + 1u])
        }
    }

    mod.sdf
}
