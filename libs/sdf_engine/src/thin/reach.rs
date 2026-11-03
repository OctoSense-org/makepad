//! Reach faces: the bead slots where a site's face climbs past the reach.
//!
//! The skeleton stops at the reach: a run that climbs to it ends at a cut, a
//! node of degree one. Past it the layout is the fixed nominal one, so each
//! slot runs on at its constant inset along the site (a line on a segment
//! site, an arc on a point site) until the face comes back within reach at
//! the next cut along the site, or reaches the site's far end, where the
//! vertex's normal ray hands it to the next site.
//!
//! A face's boundary height over its site is one function of the site
//! parameter (the chord parameter of a segment, the clockwise angle from the
//! incoming normal of a point), so a site's cuts in parameter order alternate:
//! an exit (the cut ends the face piece along the site) then an entry. Each
//! cut is an exit on one side of its edge and an entry on the other, since
//! the two faces run along the edge in opposite directions.
//!
//! Passes: per site its span flag and cut count (ThinReachMark), their scan,
//! the per-site transition lists (ThinReachScatter), per exit and per site
//! start the resolved run (ThinReachExits, ThinReachStarts), then per band
//! cell its reach pieces (count, geometry, keys) in the cell's second piece
//! slot. A run that starts at a vertex belongs to the cell of the vertex's
//! seg; its key is `thin_ray_ordinal(vertex, ray)`.
use makepad_draw::*;

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.ThinReach = {
        ..mod.sdf.ThinFaces
        seg: shader.buffer(vec4f)
        // Per site index (2 · segs): its transition count, decremented to zero
        // again by the scatter.
        reach_count: shader.buffer_rw(u32)
        reach_off: shader.buffer(u32)
        // Per site index its span flag, then per transition (end · 4 + side · 2
        // + exit, parameter bits).
        transitions: shader.buffer_rw(u32)
        // Per end, then per site index, a resolved run: (from foot, from
        // node), (to foot, to node) as f32 bits, (from key, from ordinal, to
        // key, to ordinal), (from layout node, valid, to layout node, site).
        reach: shader.buffer_rw(vec4u)
        site_count: fn() -> u32 { return self.segs() * 2u }
        entries_at: fn() -> u32 { return self.site_count() }
        defect: fn() { atomic_add(self.counts[self.cfg[2u].x + 17u], 1u) }
        key_overflow: fn() { atomic_add(self.counts[self.cfg[2u].x + 11u], 1u) }
        // A canonical node of degree one at the reach.
        is_cut: fn(e: u32) -> bool {
            let info = self.node_info[e * 2u]
            if info.x != e || info.y != 1u { return false }
            return self.node_info[e * 2u + 1u].x != 0u
        }
        side_site: fn(e: u32, side: u32) -> u32 {
            let pair = self.pair_of(e)
            if side == 0u { return pair.x }
            return pair.y
        }
        // The place of p along site x.
        param: fn(x: u32, p: vec2) -> f32 {
            if x % 2u == 0u { return dot(p - self.sa(x), self.sb(x) - self.sa(x)) }
            let d = p - self.site_point(x)
            let n0 = self.normal(self.site_info[x / 2u].z * 2u)
            return atan2(self.cross(d, n0), dot(d, n0))
        }
        // The cut end e is the end of x's face piece along its edge.
        exit_on: fn(e: u32, x: u32) -> bool {
            var start = 0u
            if !self.forward(e / 2u, x) { start = 1u }
            return e % 2u != start
        }
        list_of: fn(x: u32) -> vec2u {
            var end = self.word(14u)
            if x + 1u < self.site_count() { end = self.reach_off[x + 1u] }
            return vec2u(self.reach_off[x], end)
        }
        // The first transition of x after (p, e) in (parameter, end) order, or NONE.
        after: fn(x: u32, p: f32, e: u32, any: bool) -> u32 {
            let list = self.list_of(x)
            var best = 4294967295u
            var best_p = 0.0
            var best_e = 0u
            for j in list.x..list.y {
                let w = self.transitions[self.entries_at() + j * 2u]
                let q = asfloat(self.transitions[self.entries_at() + j * 2u + 1u])
                let f = w / 4u
                var later = any || q > p
                if !any && q == p { later = f > e }
                if later {
                    var better = best == 4294967295u
                    if !better {
                        better = q < best_p
                        if q == best_p { better = f < best_e }
                    }
                    if better { best = j; best_p = q; best_e = f }
                }
            }
            return best
        }
        f2: fn(a: vec2, b: vec2) -> vec4u { return vec4u(asuint(a.x), asuint(a.y), asuint(b.x), asuint(b.y)) }
        // The plane fine cell of seg r (its midpoint's), as an endpoint key.
        ray_cell: fn(r: u32) -> u32 {
            let q = self.seg[r]
            let top = vec2(f32(self.nx() * 10u - 1u), f32(self.ny() * 10u - 1u))
            let g = clamp(floor((q.xy + q.zw) * 0.5), vec2(0.0), top)
            return 2147483648u | (u32(g.y) * (self.nx() * 10u + 1u) + u32(g.x))
        }
        // plane_abi's thin_ray_ordinal: THIN_RAY_BIT | vertex · 2 + ray.
        ray_ordinal: fn(r: u32, ray: u32) -> u32 {
            if r >= 4194304u { self.key_overflow() }
            return 8388608u | ((r * 2u + ray) & 8388607u)
        }
        rib_key: fn(n: u32, x: u32) -> vec2u {
            let key = self.node_key(n); let rank = self.sector_rank(n, x)
            if rank >= 8u || key.y >= 524288u { self.key_overflow() }
            return vec2u(key.x, self.thin_rib_ordinal(key.y, rank))
        }
        // The vertex rays of a site: x's start (ray 0 of a point site, ray 1
        // of a segment site after a reflex vertex) and its far end (ray 1 of
        // a point site, ray 0 of the next vertex after a segment site), each
        // as (foot, foot + normal) and its key. A missing ray has key 0.
        start_ray: fn(x: u32) -> vec4u {
            let foot = self.sa(x)
            var n = self.normal(x)
            if x % 2u != 0u { n = self.normal(self.site_info[x / 2u].z * 2u) }
            return self.f2(foot, foot + n)
        }
        start_key: fn(x: u32) -> vec2u {
            let r = x / 2u
            if (self.site_info[r].w & 1u) == 0u { return vec2u(0u) }
            var ray = 1u
            if x % 2u != 0u { ray = 0u }
            return vec2u(self.ray_cell(r), self.ray_ordinal(r, ray))
        }
        end_ray: fn(x: u32) -> vec4u {
            if x % 2u != 0u {
                let foot = self.sa(x)
                return self.f2(foot, foot + self.normal(x))
            }
            let foot = self.sb(x)
            return self.f2(foot, foot + self.normal(x))
        }
        end_key: fn(x: u32) -> vec2u {
            let r = x / 2u
            if x % 2u != 0u { return vec2u(self.ray_cell(r), self.ray_ordinal(r, 1u)) }
            let next = self.site_info[r].y
            if (self.site_info[next].w & 1u) == 0u { return vec2u(0u) }
            return vec2u(self.ray_cell(next), self.ray_ordinal(next, 0u))
        }
        rib_of: fn(n: u32, x: u32) -> vec4u { return self.f2(self.foot(n, x), self.xy(self.endpoint(n))) }
    }

    mod.sdf.ThinReachClear = {
        ..mod.sdf.ThinReach
        main: fn() {
            let x = self.index()
            if x == 4294967295u || x >= self.site_count() { return }
            self.reach_count[x] = 0u
            self.transitions[x] = 0u
        }
    }

    // Per end: its sites' span flags (once per span) and, at a cut, a
    // transition on each side; the largest cut lends its fixed layout to
    // sites with no cut at all.
    mod.sdf.ThinReachMark = {
        ..mod.sdf.ThinReach
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            let pair = self.pair_of(e)
            if e % 2u == 0u && self.node_info[e * 2u].y != 0u {
                atomic_max(self.transitions[pair.x], 1u)
                atomic_max(self.transitions[pair.y], 1u)
            }
            if self.is_cut(e) {
                atomic_add(self.reach_count[pair.x], 1u)
                atomic_add(self.reach_count[pair.y], 1u)
                atomic_max(self.counts[self.cfg[2u].x + 15u], e + 1u)
            }
        }
    }

    // Per cut: its two transitions into its sites' lists (in arrival order;
    // every reader orders them by parameter), and its run record cleared.
    mod.sdf.ThinReachScatter = {
        ..mod.sdf.ThinReach
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            if !self.is_cut(e) { return }
            self.reach[e * 4u + 3u] = vec4u(0u)
            var exits = 0u
            for side in 0..2u {
                let x = self.side_site(e, side)
                var exit = 0u
                if self.exit_on(e, x) { exit = 1u }
                exits += exit
                let slot = atomic_add(self.reach_count[x], 4294967295u) - 1u
                let at = self.entries_at() + (self.reach_off[x] + slot) * 2u
                self.transitions[at] = e * 4u + side * 2u + exit
                self.transitions[at + 1u] = asuint(self.param(x, self.xy(self.endpoint(e))))
            }
            if exits != 1u { self.defect() }
        }
    }

    // Per exit: its run to the next transition along the site (an entry) or
    // to the site's far ray.
    mod.sdf.ThinReachExits = {
        ..mod.sdf.ThinReach
        main: fn() {
            let k = self.index()
            if k == 4294967295u || k >= self.word(14u) { return }
            let w = self.transitions[self.entries_at() + k * 2u]
            if (w & 1u) == 0u { return }
            let c = w / 4u
            let x = self.side_site(c, (w / 2u) % 2u)
            let p = asfloat(self.transitions[self.entries_at() + k * 2u + 1u])
            let next = self.after(x, p, c, false)
            var to = self.end_ray(x)
            var to_key = self.end_key(x)
            var to_layout = c
            if next != 4294967295u {
                let n = self.transitions[self.entries_at() + next * 2u]
                if (n & 1u) != 0u { self.defect(); return }
                to_layout = n / 4u
                to = self.rib_of(to_layout, x)
                to_key = self.rib_key(to_layout, x)
            } else if to_key.x == 0u { self.defect(); return }
            let from_key = self.rib_key(c, x)
            self.reach[c * 4u] = self.rib_of(c, x)
            self.reach[c * 4u + 1u] = to
            self.reach[c * 4u + 2u] = vec4u(from_key.x, from_key.y, to_key.x, to_key.y)
            self.reach[c * 4u + 3u] = vec4u(c, 1u, to_layout, x)
        }
    }

    // Per site whose start is a vertex ray beyond reach (its first
    // transition is an entry, or it has no transition and no span): its run
    // from that ray.
    mod.sdf.ThinReachStarts = {
        ..mod.sdf.ThinReach
        site_of: shader.buffer(u32)
        main: fn() {
            let x = self.index()
            if x == 4294967295u || x >= self.site_count() { return }
            let rec = (self.ends() + x) * 4u
            self.reach[rec + 3u] = vec4u(0u)
            let r = x / 2u
            if self.site_of[r] != r { return }
            if (self.site_info[r].w & 1u) == 0u { return }
            let first = self.after(x, 0.0, 0u, true)
            var to = self.end_ray(x)
            var to_key = self.end_key(x)
            var layout = 4294967295u
            if first == 4294967295u {
                if self.transitions[x] != 0u { return }
                let largest = self.word(15u)
                if largest == 0u || to_key.x == 0u { self.defect(); return }
                layout = largest - 1u
            } else {
                let w = self.transitions[self.entries_at() + first * 2u]
                if (w & 1u) != 0u { return }
                layout = w / 4u
                to = self.rib_of(layout, x)
                to_key = self.rib_key(layout, x)
            }
            let from_key = self.start_key(x)
            self.reach[rec] = self.start_ray(x)
            self.reach[rec + 1u] = to
            self.reach[rec + 2u] = vec4u(from_key.x, from_key.y, to_key.x, to_key.y)
            self.reach[rec + 3u] = vec4u(layout, 1u, layout, x)
        }
    }

    // Per band cell its reach pieces, in the cell's second piece slot: per
    // level, the exits at its cut nodes in span order, then the starts at
    // its vertices in site order.
    mod.sdf.ThinReachFaces = {
        ..mod.sdf.ThinFaces
        seg: shader.buffer(vec4f)
        reach: shader.buffer(vec4u)
        is_cut: fn(e: u32) -> bool {
            let info = self.node_info[e * 2u]
            if info.x != e || info.y != 1u { return false }
            return self.node_info[e * 2u + 1u].x != 0u
        }
        seg_cell: fn(r: u32) -> vec2u {
            let q = self.seg[r]
            let top = vec2(f32(self.nx() * 10u - 1u), f32(self.ny() * 10u - 1u))
            let g = clamp(floor((q.xy + q.zw) * 0.5), vec2(0.0), top)
            return vec2u(u32(g.x), u32(g.y))
        }
        on_reach: fn(at: u32, rec: u32, slot: u32, level: u32, plane: u32) {}
        take: fn(at: u32, rec: u32, slot: u32, level: u32, plane: u32) -> u32 {
            let flags = self.reach[rec + 3u]
            if flags.y == 0u { return 0u }
            if !self.inside(flags.x, slot) { return 0u }
            self.on_reach(at, rec, slot, level, plane)
            return 1u
        }
        visit_reach: fn(cell: u32, first: u32) -> u32 {
            let plane = self.slot(cell / 100u).y
            let g = self.cell_grid(cell)
            let slots = self.slots()
            let list = self.site_lists[cell]
            // A cut is a canonical end, so its node's cell is its own span's.
            let range = self.cell_spans(cell)
            var count = 0u
            for slot in 0..slots {
                var level = 32u + slot
                if slot + 1u == slots { level = 33u + self.cap() }
                for span in range.x..range.y {
                    for end in 0..2u {
                        let arm = span * 2u + end
                        if self.is_cut(arm) { count += self.take(first + count, arm * 4u, slot, level, plane) }
                    }
                }
                for i in 0..list.y {
                    let s = self.site_list[list.x + i]
                    if s % 2u != 0u {
                        let h = self.seg_cell(s / 2u)
                        if h.x == g.x && h.y == g.y {
                            for k in 0..2u { count += self.take(first + count, (self.ends() + s - k) * 4u, slot, level, plane) }
                        }
                    }
                }
            }
            return count
        }
    }
    mod.sdf.ThinReachCount = {
        ..mod.sdf.ThinReachFaces
        piece_count: shader.buffer_rw(u32)
        main: fn() {
            let cell = self.index()
            if cell == 4294967295u || cell >= self.cells() { return }
            self.piece_count[cell * 2u + 1u] = self.visit_reach(cell, 0u)
        }
    }
    mod.sdf.ThinReachGeometry = {
        ..mod.sdf.ThinReachFaces
        piece_off: shader.buffer(u32)
        piece: shader.buffer_rw(vec4)
        piece_geo: shader.buffer_rw(vec4)
        piece_width: shader.buffer_rw(vec2)
        rib: fn(v: vec4u, l: f32) -> vec2 {
            let foot = vec2(asfloat(v.x), asfloat(v.y))
            let p = vec2(asfloat(v.z), asfloat(v.w))
            let d = length(p - foot)
            if d > 0.0 { return foot + (p - foot) * (l / d) }
            return foot
        }
        on_reach: fn(at: u32, rec: u32, slot: u32, level: u32, plane: u32) {
            // The bound is the stage's piece capacity; a count past it fails verify.
            if at >= self.cfg[3u].x { return }
            let flags = self.reach[rec + 3u]
            let l0 = self.loc(flags.x, slot); let l1 = self.loc(flags.z, slot)
            let p0 = self.rib(self.reach[rec], l0.x)
            let p1 = self.rib(self.reach[rec + 1u], l1.x)
            var geo = vec4(0.0)
            let x = flags.w
            if x % 2u != 0u {
                let v = self.seg[x / 2u].xy
                var sweep = atan2(p0.y - v.y, p0.x - v.x) - atan2(p1.y - v.y, p1.x - v.x)
                if sweep < 0.0 { sweep += 6.283185307 }
                if sweep > 0.0 && sweep < 3.141592654 {
                    geo = vec4(v.x, v.y, max(length(p0 - v), length(p1 - v)), 0.0 - sweep)
                }
            }
            self.piece[at] = vec4(p0.x, p0.y, p1.x, p1.y)
            self.piece_geo[at] = geo
            self.piece_width[at] = vec2(l0.y, l1.y)
        }
        main: fn() {
            let cell = self.index()
            if cell == 4294967295u || cell >= self.cells() { return }
            let count = self.visit_reach(cell, self.piece_off[cell * 2u + 1u])
        }
    }
    mod.sdf.ThinReachKeys = {
        ..mod.sdf.ThinReachFaces
        piece_off: shader.buffer(u32)
        piece_key: shader.buffer_rw(vec4u)
        piece_info: shader.buffer_rw(vec4u)
        on_reach: fn(at: u32, rec: u32, slot: u32, level: u32, plane: u32) {
            if at >= self.cfg[3u].x { return }
            let keys = self.reach[rec + 2u]
            let x = self.reach[rec + 3u].w
            self.piece_key[at] = vec4u(keys.x, (level << 24u) | keys.y, keys.z, (level << 24u) | keys.w)
            var feature = x / 2u
            if x % 2u != 0u { feature = 2147483648u | feature }
            self.piece_info[at] = vec4u(plane, level, feature, keys.x & 2147483647u)
        }
        main: fn() {
            let cell = self.index()
            if cell == 4294967295u || cell >= self.cells() { return }
            let count = self.visit_reach(cell, self.piece_off[cell * 2u + 1u])
        }
    }

    mod.sdf
}
