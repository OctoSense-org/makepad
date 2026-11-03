//! Bead lines as per-face pieces (GPU-EXEC §10.3 records). A face is one side
//! site of one skeleton edge. Its bead (or fill marker) slot runs from the rib
//! at the face's start node to the rib at its end node, or ends where its
//! location climbs past the skeleton and joins the other side's bead there.
//! Keys are CELL_BIT | the node's fine cell id and key_lo(level, ordinal):
//! thin_rib_ordinal(node, sector rank among the node's sites), or
//! thin_edge_ordinal(node, arm rank) for a join. A band cell's face pieces
//! take its first piece slot; reach.rs fills the second.
use makepad_draw::*;

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.ThinNodeOrdinal = {
        ..mod.sdf.ThinGraph
        node_info: shader.buffer_rw(vec4u)
        canonical: fn(e: u32) -> u32 { return self.node_info[e * 2u].x }
        // The fine cell of a canonical node: its span's band cell, which
        // holds its pieces. Every end of the node is a span end next to it.
        grid_of: fn(n: u32) -> vec2u { return self.span_grid(n) }
        // Geometric key: position, then the canonical endpoint.
        node_less: fn(a: u32, b: u32) -> bool {
            let p = self.xy(self.endpoint(a)); let q = self.xy(self.endpoint(b))
            if p.x != q.x { return p.x < q.x }
            if p.y != q.y { return p.y < q.y }
            return a < b
        }
        main: fn() {
            let n = self.index()
            if n == 4294967295u || n >= self.ends() { return }
            let info = self.node_info[n * 2u]
            if info.x != n || info.y == 0u { return }
            // The nodes of this fine cell are ends of its own spans.
            var ordinal = 0u
            let range = self.cell_spans(self.output[(n / 2u) * 4u + 2u].z)
            for span in range.x..range.y {
                for end in 0..2u {
                    let m = span * 2u + end
                    let other = self.node_info[m * 2u]
                    if other.x == m && other.y != 0u && self.node_less(m, n) { ordinal += 1u }
                }
            }
            let tail = self.node_info[n * 2u + 1u]
            self.node_info[n * 2u + 1u] = vec4u(tail.x, ordinal, tail.z, tail.w)
        }
    }
    // The face passes share one visit of a band cell's pieces in their stored
    // order; counts, geometry and keys are separate kernels so that each
    // binds few enough buffers for the 128-byte push block.
    mod.sdf.ThinFaces = {
        ..mod.sdf.ThinNodeOrdinal
        node_info: shader.buffer(vec4u)
        adj_off: shader.buffer(u32)
        adjacency: shader.buffer(u32)
        layouts: shader.buffer(vec4)
        slots: fn() -> u32 { return self.cap() / 2u + 1u }
        // A slot's (location, width) at a node; an absent bead or marker sits
        // on the skeleton.
        loc: fn(n: u32, slot: u32) -> vec2 {
            let slots = self.slots()
            let r = asfloat(self.endpoint(n).z)
            let info = self.layouts[n * slots + slots - 1u]
            if slot + 1u == slots {
                if info.z == info.z { return vec2(info.z, 0.0) }
                return vec2(r, 0.0)
            }
            if f32(slot) < info.x { return self.layouts[n * slots + slot].xy }
            return vec2(r, 0.0)
        }
        inside: fn(n: u32, slot: u32) -> bool { return self.loc(n, slot).x < asfloat(self.endpoint(n).z) }
        // A canonical node's arms: first word, count. The list order is the
        // scatter's arrival order; everything here is order independent.
        arms: fn(n: u32) -> vec2u { return vec2u(self.ends() + self.adj_off[n], self.node_info[n * 2u].y) }
        pair_of: fn(arm: u32) -> vec4u { return self.output[(arm / 2u) * 4u + 2u] }
        // The face of site x runs from end 0 to end 1 of its edge when x
        // lies on its right (the outline keeps the inside on its left): when
        // the edge's direction agrees with segment x's tangent, or turns
        // clockwise about point x. Ends follow the pair's curve parameter
        // (ThinPairs' tangent p1), along which that sign is fixed: a mixed
        // pair's segment runs forward and its point back, and of two
        // segments or two points the pair's second runs forward. The ends'
        // positions would decide a short edge by roundoff.
        forward: fn(span: u32, x: u32) -> bool {
            let pair = self.output[span * 4u + 2u]
            if pair.x % 2u != pair.y % 2u { return x % 2u == 0u }
            return x == pair.y
        }
        // x's own segment or reflex endpoint at node n: the normal ray
        // between them ends at n, so they share one rib sector.
        partner: fn(n: u32, x: u32) -> u32 {
            let arms = self.arms(n)
            var partner = 4294967295u
            for k in 0..arms.y {
                let pair = self.pair_of(self.adjacency[arms.x + k])
                if pair.x != x && self.excluded(x, pair.x) { partner = min(partner, pair.x) }
                if pair.y != x && self.excluded(x, pair.y) { partner = min(partner, pair.y) }
            }
            return partner
        }
        // Rank of x's sector (named by its lower site) among the node's sites.
        sector_rank: fn(n: u32, x: u32) -> u32 {
            let sector = min(x, self.partner(n, x))
            let arms = self.arms(n)
            var rank = 0u
            for k in 0..arms.y {
                let pair = self.pair_of(self.adjacency[arms.x + k])
                for side in 0..2u {
                    var t = pair.x
                    if side == 1u { t = pair.y }
                    if t >= sector || (side == 1u && t == pair.x) { continue }
                    var seen = false
                    for j in 0..k {
                        let earlier = self.pair_of(self.adjacency[arms.x + j])
                        if earlier.x == t || earlier.y == t { seen = true }
                    }
                    if !seen { rank += 1u }
                }
            }
            return rank
        }
        arm_rank: fn(n: u32, arm: u32) -> u32 {
            let arms = self.arms(n)
            var rank = 0u
            for k in 0..arms.y { if self.adjacency[arms.x + k] < arm { rank += 1u } }
            return rank
        }
        // plane_abi's thin ordinals: THIN_NODE_STRIDE = 16, THIN_RIB_SECTORS = 8.
        thin_rib_ordinal: fn(node: u32, sector: u32) -> u32 { return node * 16u + sector }
        thin_edge_ordinal: fn(node: u32, rank: u32) -> u32 { return node * 16u + 8u + rank }
        // Endpoint key: CELL_BIT | the node's fine cell id, and its ordinal.
        node_key: fn(n: u32) -> vec2u {
            let g = self.grid_of(n)
            return vec2u(2147483648u | (g.y * (self.nx() * 10u + 1u) + g.x), self.node_info[n * 2u + 1u].y)
        }
        foot: fn(n: u32, x: u32) -> vec2 {
            var point = x
            if x % 2u == 0u { point = self.partner(n, x) }
            if point != 4294967295u { return self.site_point(point) }
            let p = self.xy(self.endpoint(n))
            let a = self.sa(x); let u = self.sb(x) - a
            return a + u * clamp(dot(p - a, u) / dot(u, u), 0.0, 1.0)
        }
        // The point at location l along the rib from x's foot to node n.
        rib_point: fn(n: u32, x: u32, l: f32) -> vec2 {
            let foot = self.foot(n, x)
            let p = self.xy(self.endpoint(n))
            let d = length(p - foot)
            if d > 0.0 { return foot + (p - foot) * (l / d) }
            return foot
        }
        // Where a slot's location meets the skeleton along the edge, in the
        // edge's own end order so that both faces get the same bits.
        join: fn(span: u32, slot: u32) -> vec3 {
            let a = self.canonical(span * 2u); let b = self.canonical(span * 2u + 1u)
            let la = self.loc(a, slot); let lb = self.loc(b, slot)
            let ga = asfloat(self.endpoint(a).z) - la.x
            let gb = asfloat(self.endpoint(b).z) - lb.x
            var t = 0.5
            if ga != gb { t = clamp(ga / (ga - gb), 0.0, 1.0) }
            let p = self.xy(self.endpoint(a)); let q = self.xy(self.endpoint(b))
            return vec3(p + (q - p) * t, la.y + (lb.y - la.y) * t)
        }
        on_piece: fn(at: u32, span: u32, x: u32, start: u32, slot: u32, level: u32, plane: u32) {}
        // Visits this band cell's pieces from piece index `first`: level (slots
        // ascend to the fill marker), then span order. Returns the count.
        visit: fn(cell: u32, first: u32) -> u32 {
            let plane = self.slot(cell / 100u).y
            let g = self.cell_grid(cell)
            let slots = self.slots()
            var count = 0u
            for slot in 0..slots {
                var level = 32u + slot
                if slot + 1u == slots { level = 33u + self.cap() }
                for gy in (max(g.y, 1u) - 1u)..min(g.y + 2u, self.ny() * 10u) {
                    for gx in (max(g.x, 1u) - 1u)..min(g.x + 2u, self.nx() * 10u) {
                        let range = self.cell_spans(self.band_cell(plane, gx, gy))
                        for span in range.x..range.y {
                            for end in 0..2u {
                                let arm = span * 2u + end
                                // Zero-length spans join no nodes and carry no degree.
                                if self.node_info[arm * 2u].y == 0u { continue }
                                let h = self.grid_of(self.canonical(arm))
                                if h.x != g.x || h.y != g.y { continue }
                                let pair = self.pair_of(arm)
                                for side in 0..2u {
                                    var x = pair.x
                                    if side == 1u { x = pair.y }
                                    var start = 0u
                                    if !self.forward(span, x) { start = 1u }
                                    let s_in = self.inside(self.canonical(span * 2u + start), slot)
                                    let t_in = self.inside(self.canonical(span * 2u + 1u - start), slot)
                                    // A piece belongs to its from key's node: the
                                    // start rib, or end 0 for a join.
                                    if (!s_in && !t_in) || (s_in && end != start) || (!s_in && end != 0u) { continue }
                                    self.on_piece(first + count, span, x, start, slot, level, plane)
                                    count += 1u
                                }
                            }
                        }
                    }
                }
            }
            return count
        }
    }
    mod.sdf.ThinFaceCount = {
        ..mod.sdf.ThinFaces
        piece_count: shader.buffer_rw(u32)
        main: fn() {
            let cell = self.index()
            if cell == 4294967295u || cell >= self.cells() { return }
            // the scan's extra value, so piece_off ends with the total
            if cell == 0u { self.piece_count[self.cells() * 2u] = 0u }
            self.piece_count[cell * 2u] = self.visit(cell, 0u)
            // the reach faces' slot, which ThinReachCount fills after this
            self.piece_count[cell * 2u + 1u] = 0u
        }
    }
    mod.sdf.ThinFaceGeometry = {
        ..mod.sdf.ThinFaces
        piece_off: shader.buffer(u32)
        piece: shader.buffer_rw(vec4)
        piece_geo: shader.buffer_rw(vec4)
        piece_width: shader.buffer_rw(vec2)
        on_piece: fn(at: u32, span: u32, x: u32, start: u32, slot: u32, level: u32, plane: u32) {
            let s = self.canonical(span * 2u + start)
            let t = self.canonical(span * 2u + 1u - start)
            let ls = self.loc(s, slot); let lt = self.loc(t, slot)
            let join = self.join(span, slot)
            var p0 = join.xy; var w0 = join.z
            var p1 = p0; var w1 = w0
            if ls.x < asfloat(self.endpoint(s).z) { p0 = self.rib_point(s, x, ls.x); w0 = ls.y }
            if lt.x < asfloat(self.endpoint(t).z) { p1 = self.rib_point(t, x, lt.x); w1 = lt.y }
            // Round a reflex point site the bead turns clockwise; its radius
            // follows the two ends (a spiral where the location changes).
            var geo = vec4(0.0)
            if x % 2u != 0u {
                let v = self.site_point(x)
                var sweep = atan2(p0.y - v.y, p0.x - v.x) - atan2(p1.y - v.y, p1.x - v.x)
                if sweep < 0.0 { sweep += 6.283185307 }
                if sweep > 0.0 && sweep < 3.141592654 {
                    geo = vec4(v.x, v.y, max(length(p0 - v), length(p1 - v)), 0.0 - sweep)
                }
            }
            self.piece[at] = vec4(p0.x, p0.y, p1.x, p1.y)
            self.piece_geo[at] = geo
            self.piece_width[at] = vec2(w0, w1)
        }
        main: fn() {
            let cell = self.index()
            if cell == 4294967295u || cell >= self.cells() { return }
            let count = self.visit(cell, self.piece_off[cell * 2u])
        }
    }
    mod.sdf.ThinFaceKeys = {
        ..mod.sdf.ThinFaces
        piece_off: shader.buffer(u32)
        piece_key: shader.buffer_rw(vec4u)
        piece_info: shader.buffer_rw(vec4u)
        on_piece: fn(at: u32, span: u32, x: u32, start: u32, slot: u32, level: u32, plane: u32) {
            let s = self.canonical(span * 2u + start)
            let t = self.canonical(span * 2u + 1u - start)
            let a = self.canonical(span * 2u)
            let join_key = self.node_key(a)
            let join_rank = self.arm_rank(a, span * 2u)
            var k0 = vec2u(join_key.x, (level << 24u) | self.thin_edge_ordinal(join_key.y, join_rank))
            var k1 = k0
            // Ordinals stay below THIN_RAY_BIT: nodes below 2^19 per cell.
            var overflow = join_rank >= 8u || join_key.y >= 524288u
            if self.inside(s, slot) {
                let key = self.node_key(s); let rank = self.sector_rank(s, x)
                overflow = overflow || rank >= 8u || key.y >= 524288u
                k0 = vec2u(key.x, (level << 24u) | self.thin_rib_ordinal(key.y, rank))
            }
            if self.inside(t, slot) {
                let key = self.node_key(t); let rank = self.sector_rank(t, x)
                overflow = overflow || rank >= 8u || key.y >= 524288u
                k1 = vec2u(key.x, (level << 24u) | self.thin_rib_ordinal(key.y, rank))
            }
            self.piece_key[at] = vec4u(k0.x, k0.y, k1.x, k1.y)
            // A site is named by its run's first seg: a segment site is that
            // seg's feature, a point site its start vertex.
            var feature = x / 2u
            if x % 2u != 0u { feature = 2147483648u | feature }
            self.piece_info[at] = vec4u(plane, level, feature, k0.x & 2147483647u)
            if overflow { atomic_add(self.counts[self.cfg[2u].x + 11u], 1u) }
        }
        main: fn() {
            let cell = self.index()
            if cell == 4294967295u || cell >= self.cells() { return }
            let count = self.visit(cell, self.piece_off[cell * 2u])
        }
    }

    mod.sdf
}
