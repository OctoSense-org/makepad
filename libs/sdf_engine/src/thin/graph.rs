//! Skeleton nodes, paths and layout owners over the span graph.
use makepad_draw::*;

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.ThinNodeInfo = {
        ..mod.sdf.ThinGraph
        // Two records per endpoint: representative, full/central degrees,
        // maximum adjacent radius; then cut-anchor presence, the node's
        // ordinal in its fine cell (faces.rs) and reserved words.
        node_info: shader.buffer_rw(vec4u)
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            var info = vec4u(e, 0u, 0u, 0u)
            var cut = 0u
            if self.edge(e) {
                let near = self.node_cells(e)
                for gy in near.z..(near.w + 1u) {
                    for gx in near.x..(near.y + 1u) {
                        let range = self.cell_spans(self.node_cell(e, gx, gy))
                        for span in range.x..range.y {
                            for end in 0..2u {
                                let other = span * 2u + end
                                if self.edge(other) && self.same_node(e, other) {
                                    info.x = min(info.x, other); info.y += 1u
                                    let value = self.output[(other / 2u) * 4u + other % 2u]
                                    if (value.w & 6u) != 0u { info.z += 1u }
                                    cut |= value.w & 1u
                                    let opposite = other ^ 1u
                                    // A convex terminal can round slightly below zero.
                                    // Compare radii as floats: unsigned bit order would
                                    // make that terminal the largest adjacent radius.
                                    info.w = asuint(max(asfloat(info.w), asfloat(self.endpoint(opposite).z)))
                                }
                            }
                        }
                    }
                }
            }
            self.node_info[e * 2u] = info
            self.node_info[e * 2u + 1u] = vec4u(cut, 0u, 0u, 0u)
        }
    }
    // dispatch_range.w: 0 links the non-central graph, 1 the central one.
    mod.sdf.ThinSelectPaths = {
        ..mod.sdf.ThinNodeInfo
        node_info: shader.buffer(vec4u)
        selected: fn(e: u32) -> bool {
            let central = (self.output[(e / 2u) * 4u + e % 2u].w & 6u) != 0u
            if self.dispatch_range.w == 0u { return !central }
            return central
        }
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            let at = self.graph() + e * 4u
            let info = self.node_info[e * 2u]
            var degree = 0u; var peer = 4294967295u
            if self.edge(e) && self.selected(e) {
                let near = self.node_cells(e)
                for gy in near.z..(near.w + 1u) {
                    for gx in near.x..(near.y + 1u) {
                        let range = self.cell_spans(self.node_cell(e, gx, gy))
                        for span in range.x..range.y {
                            for end in 0..2u {
                                let other = span * 2u + end
                                if self.edge(other) && self.same_node(e, other) {
                                    // Each end checks its node once, in the pass that selects it.
                                    let seen = self.node_info[other * 2u]
                                    if seen.x != info.x || seen.y != info.y { atomic_add(self.counts[self.cfg[2u].x + 19u], 1u) }
                                    if self.selected(other) {
                                        degree += 1u
                                        if other != e { peer = other }
                                    }
                                }
                            }
                        }
                    }
                }
                // Non-central paths stop at any central incidence, even if
                // precisely two non-central arms meet there.
                if self.dispatch_range.w == 0u && (info.z != 0u || info.y != 2u) { peer = 4294967295u }
            }
            self.thin[at] = peer; self.thin[at + 1u] = info.x
            self.thin[at + 2u] = degree; self.thin[at + 3u] = e / 2u
        }
    }

    // Path ranks. step.x: 0 for the first pass, 1 for the second, which cuts
    // closed components at their canonical root; step.y: the bank of the first
    // pass's result. The jump's round is dispatch_range.z.
    mod.sdf.ThinPathRankInit = {
        ..mod.sdf.ThinGraph
        path_work: shader.buffer_rw(vec4u)
        ranked: shader.buffer_rw(vec4u)
        step: shader.uniform(vec4u)
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            if !self.edge(e) { self.path_work[e] = vec4u(4294967295u, e, 0u, 0u); return }
            let opposite = e ^ 1u
            var next = self.thin[self.graph() + opposite * 4u]
            var closed = 0u
            if self.step.x != 0u {
                // The first rank pass identifies components without terminals.
                // Cut a cycle at its canonical path root, never at an arrival.
                if self.path_work[self.step.y * self.ends() + e].y == 4294967295u {
                    closed = 1u
                    let root = self.thin[self.graph() + e * 4u + 3u] * 2u
                    let node = self.thin[self.graph() + root * 4u + 1u]
                    if self.thin[self.graph() + opposite * 4u + 1u] == node { next = 4294967295u }
                }
                self.ranked[e * 2u + 1u] = vec4u(0u, 0u, 0u, closed)
            }
            var terminal = 4294967295u
            if next == 4294967295u { terminal = opposite }
            let distance = length(self.xy(self.endpoint(e)) - self.xy(self.endpoint(opposite)))
            self.path_work[e] = vec4u(next, terminal, asuint(distance), 1u)
        }
    }
    mod.sdf.ThinPathRankJump = {
        ..mod.sdf.ThinPathRankInit
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            let source = (self.dispatch_range.z % 2u) * self.ends()
            let destination = (1u - self.dispatch_range.z % 2u) * self.ends()
            var value = self.path_work[source + e]
            if value.x != 4294967295u {
                let next = self.path_work[source + value.x]
                value = vec4u(next.x, next.y, asuint(asfloat(value.z) + asfloat(next.z)), value.w + next.w)
            }
            self.path_work[destination + e] = value
        }
    }
    // step.x: the final jump bank.
    mod.sdf.ThinPathRankFinish = {
        ..mod.sdf.ThinPathRankInit
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() || !self.edge(e) { return }
            let source = self.step.x * self.ends()
            let forward = self.path_work[source + e]
            let backward = self.path_work[source + (e ^ 1u)]
            let head = min(forward.y, backward.y)
            let tail = max(forward.y, backward.y)
            let distance = length(self.xy(self.endpoint(e)) - self.xy(self.endpoint(e ^ 1u)))
            var position = asfloat(forward.z)
            var ordinal = forward.w
            var direction = 0u
            if backward.y == head {
                position = asfloat(backward.z) - distance
                ordinal = backward.w - 1u
                direction = 1u
            }
            let closed = self.ranked[e * 2u + 1u].w
            self.ranked[e * 2u] = vec4u(head, ordinal, forward.w + backward.w - 1u, direction)
            self.ranked[e * 2u + 1u] = vec4u(asuint(position), asuint(asfloat(forward.z) + asfloat(backward.z) - distance), tail, closed)
        }
    }
    // Per central path root its sample and edge counts (the scans' inputs).
    mod.sdf.ThinPathSizes = {
        ..mod.sdf.ThinGraph
        ranked: shader.buffer(vec4u)
        path_samples: shader.buffer_rw(u32)
        path_edges: shader.buffer_rw(u32)
        main: fn() {
            let edge = self.index()
            if edge == 4294967295u || edge >= self.spans() { return }
            let e = edge * 2u
            var size = vec2u(0u)
            if self.thin[self.graph() + e * 4u + 2u] != 0u && (self.endpoint(e).w & 6u) != 0u
                && self.thin[self.graph() + e * 4u + 3u] == edge {
                let count = self.ranked[e * 2u].z
                size = vec2u(count + 1u, count)
            }
            self.path_samples[edge] = size.x
            self.path_edges[edge] = size.y
        }
    }
    // dispatch_range.w: 0 writes the path records and samples, 1 the edges.
    mod.sdf.ThinPathPack = {
        ..mod.sdf.ThinPathSizes
        path_samples: shader.buffer(u32)
        path_edges: shader.buffer(u32)
        sample_off: shader.buffer(u32)
        edge_off: shader.buffer(u32)
        node_info: shader.buffer(vec4u)
        paths: shader.buffer_rw(vec4u)
        samples: shader.buffer_rw(vec4u)
        edges: shader.buffer_rw(vec4)
        main: fn() {
            let edge = self.index()
            if edge == 4294967295u || edge >= self.spans() { return }
            let edges_pass = self.dispatch_range.w != 0u
            var e = edge * 2u
            if !edges_pass { self.paths[edge] = vec4u(0u) }
            if self.thin[self.graph() + e * 4u + 2u] == 0u || (self.endpoint(e).w & 6u) == 0u { return }
            let root = self.thin[self.graph() + e * 4u + 3u]
            let first = vec2u(self.sample_off[root], self.edge_off[root])
            let size = vec2u(self.path_samples[root], self.path_edges[root])
            if root == edge && !edges_pass { self.paths[edge] = vec4u(first.x, size.x, first.y, size.y) }
            if self.ranked[e * 2u].w == 0u { e ^= 1u }
            let rank = self.ranked[e * 2u]
            if edges_pass {
                let a = self.samples[first.x + rank.y]; let b = self.samples[first.x + rank.y + 1u]
                self.edges[first.y + rank.y] = vec4(asfloat(a.y), asfloat(b.y), asfloat(a.x), asfloat(b.x))
                return
            }
            let node = self.node_info[e * 2u].x
            let next = self.node_info[(e ^ 1u) * 2u].x
            let t0 = 2.0 * asfloat(self.endpoint(node).z)
            let t1 = 2.0 * asfloat(self.endpoint(next).z)
            let s0 = asfloat(self.ranked[e * 2u + 1u].x)
            let s1 = asfloat(self.ranked[(e ^ 1u) * 2u + 1u].x)
            self.samples[first.x + rank.y] = vec4u(asuint(s0), asuint(t0), root, node)
            if rank.y + 1u == size.y {
                self.samples[first.x + rank.y + 1u] = vec4u(asuint(s1), asuint(t1), root, next)
            }
        }
    }

    // Layout owners: per canonical node its arms (a CSR over endpoints),
    // monotone path suffixes, and the downhill DAG to the nearest seed.
    mod.sdf.ThinOwnerBase = {
        ..mod.sdf.ThinGraph
        node_info: shader.buffer(vec4u)
        // Per canonical node: its first arm (exclusive scan of its degree).
        adj_off: shader.buffer(u32)
        // Scatter cursors per endpoint, then the arms.
        adjacency: shader.buffer_rw(u32)
        owners: shader.buffer_rw(vec4u)
        owner_paths: shader.buffer_rw(vec4u)
        step: shader.uniform(vec4u)
        arm: fn(node: u32, k: u32) -> u32 { return self.adjacency[self.ends() + self.adj_off[node] + k] }
        owner_seed: fn(node: u32) -> bool {
            let info = self.node_info[node * 2u]
            return info.z != 0u || self.node_info[node * 2u + 1u].x != 0u
                || asfloat(self.endpoint(node).z) >= asfloat(info.w)
        }
    }
    mod.sdf.ThinAdjCount = {
        ..mod.sdf.ThinOwnerBase
        adj_count: shader.buffer_rw(u32)
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            let node = self.node_info[e * 2u]
            var count = 0u
            if node.x == e { count = node.y }
            self.adj_count[e] = count
            self.adjacency[e] = 0u
        }
    }
    mod.sdf.ThinAdjScatter = {
        ..mod.sdf.ThinOwnerBase
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            let node = self.node_info[e * 2u]
            if node.y == 0u { return }
            let slot = atomic_add(self.adjacency[node.x], 1u)
            self.adjacency[self.ends() + self.adj_off[node.x] + slot] = e
        }
    }
    mod.sdf.ThinOwnerPaths = {
        ..mod.sdf.ThinOwnerBase
        // Directed monotone path suffixes: next arm, terminal node,
        // cumulative distance, edge count. Two banks for segmented doubling.
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            let a = self.node_info[e * 2u]
            let b = self.node_info[(e ^ 1u) * 2u]
            var next = 4294967295u
            let p = self.endpoint(a.x); let q = self.endpoint(b.x)
            if a.y == 0u { self.owner_paths[e] = vec4u(next, e, 0u, 0u); return }
            if b.y == 2u && !self.owner_seed(b.x) {
                let arm0 = self.arm(b.x, 0u)
                let arm1 = self.arm(b.x, 1u)
                var peer = arm0
                if peer == (e ^ 1u) { peer = arm1 }
                let after = self.node_info[(peer ^ 1u) * 2u].x
                let r = asfloat(self.endpoint(after).z)
                // Extrema and flat runs are explicit path boundaries. No
                // coordinate tolerance decides connectivity or direction.
                if (asfloat(p.z) < asfloat(q.z) && asfloat(q.z) < r)
                    || (asfloat(p.z) > asfloat(q.z) && asfloat(q.z) > r) { next = peer }
            }
            self.owner_paths[e] = vec4u(next, b.x, asuint(length(self.xy(p) - self.xy(q))), 1u)
        }
    }
    mod.sdf.ThinOwnerPathJump = {
        ..mod.sdf.ThinOwnerBase
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            let source = (self.dispatch_range.z % 2u) * self.ends()
            let destination = (1u - self.dispatch_range.z % 2u) * self.ends()
            var value = self.owner_paths[source + e]
            if value.x != 4294967295u {
                let next = self.owner_paths[source + value.x]
                value = vec4u(next.x, next.y, asuint(asfloat(value.z) + asfloat(next.z)), value.w + next.w)
            }
            self.owner_paths[destination + e] = value
        }
    }
    // The number of monotone paths (a diagnostic word). step.x: the final bank.
    mod.sdf.ThinOwnerPathCounts = {
        ..mod.sdf.ThinOwnerBase
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            let source = self.step.x * self.ends()
            let path = self.owner_paths[source + e]
            let reverse = self.owner_paths[source + (e ^ 1u)]
            if self.node_info[e * 2u].y != 0u && reverse.w == 1u && self.node_info[e * 2u].x < path.y {
                atomic_add(self.counts[self.cfg[2u].x + 8u], 1u)
            }
        }
    }
    // step.x: the final owner path bank.
    mod.sdf.ThinOwnerDagInit = {
        ..mod.sdf.ThinOwnerBase
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            let node = self.node_info[e * 2u]
            if node.x != e || node.y == 0u { return }
            var value = vec4u(4294967295u, 0u, 0u, 0u)
            if self.owner_seed(e) { self.owners[e] = vec4u(e, 0u, 1u, 0u); return }
            for k in 0..node.y {
                let arm = self.arm(e, k)
                let other = self.node_info[(arm ^ 1u) * 2u].x
                if asfloat(self.endpoint(other).z) <= asfloat(self.endpoint(e).z) { continue }
                let path = self.owner_paths[self.step.x * self.ends() + arm]
                value.w += 1u
                value.x = path.y; value.y = path.z
            }
            // A minimum can receive two uphill layouts. It is a sink of the
            // downhill DAG: resolve its candidates in the final gather.
            if value.w > 1u { value.x = 4294967295u }
            self.owners[e] = value
        }
    }
    mod.sdf.ThinOwnerDagJump = {
        ..mod.sdf.ThinOwnerBase
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            let node = self.node_info[e * 2u]
            if node.x != e || node.y == 0u { return }
            let source = (self.dispatch_range.z % 2u) * self.ends()
            let destination = (1u - self.dispatch_range.z % 2u) * self.ends()
            var value = self.owners[source + e]
            if value.z == 0u && value.x != 4294967295u {
                let next = self.owners[source + value.x]
                value = vec4u(next.x, asuint(asfloat(value.y) + asfloat(next.y)), next.z, value.w)
            }
            self.owners[destination + e] = value
        }
    }
    // step.x: the final owner path bank; step.z: the final DAG bank. Writes bank two.
    mod.sdf.ThinOwnerFinish = {
        ..mod.sdf.ThinOwnerBase
        owner_less: fn(a: u32, b: u32) -> bool {
            if b == 4294967295u { return true }
            let p = self.endpoint(a); let q = self.endpoint(b)
            if asfloat(p.x) != asfloat(q.x) { return asfloat(p.x) < asfloat(q.x) }
            if asfloat(p.y) != asfloat(q.y) { return asfloat(p.y) < asfloat(q.y) }
            if asfloat(p.z) != asfloat(q.z) { return asfloat(p.z) < asfloat(q.z) }
            return a < b
        }
        main: fn() {
            let e = self.index()
            if e == 4294967295u || e >= self.ends() { return }
            let node = self.node_info[e * 2u]
            let destination = self.ends() * 2u + e
            var best = vec4u(4294967295u, 2139095040u, 0u, 0u)
            if node.x != e || node.y == 0u { self.owners[destination] = best; return }
            if self.owner_seed(e) { self.owners[destination] = vec4u(e, 0u, 1u, 0u); return }
            for k in 0..node.y {
                let arm = self.arm(e, k)
                let other = self.node_info[(arm ^ 1u) * 2u].x
                if asfloat(self.endpoint(other).z) <= asfloat(self.endpoint(e).z) { continue }
                let path = self.owner_paths[self.step.x * self.ends() + arm]
                let candidate = self.owners[self.step.z * self.ends() + path.y]
                if candidate.z == 0u { best.w += 1u; continue }
                let distance = asfloat(path.z) + asfloat(candidate.y)
                if distance < asfloat(best.y) || (distance == asfloat(best.y) && self.owner_less(candidate.x, best.x)) {
                    best = vec4u(candidate.x, asuint(distance), 1u, best.w)
                }
            }
            self.owners[destination] = best
        }
    }

    mod.sdf
}
