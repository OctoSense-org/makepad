//! Bead counts along the central paths, their transitions and the bead layouts
//! at the skeleton's nodes. All lengths are grid units (0.1 mm = 1.0); the
//! constants the CPU strategy writes in mm are scaled here (0.01 mm = 0.1).
//!
//! `parameters` (vec4u rows, f32 bits in grid units unless noted):
//! 0: s_outer, s_inner, w_outer, w_inner;
//! 1: min bead width, min feature size, transition length, transition filter deviation;
//! 2: transition angle (radians), distribution count, cap (2 · loops), precise.
use makepad_draw::*;

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.ThinLayout = {
        ..mod.sdf.Thin
        parameters: shader.buffer(vec4u)
        threshold_fraction: fn(n: u32) -> f32 {
            let widths = self.parameters[0u]
            let minimum = asfloat(self.parameters[1u].x)
            if n % 2u == 0u { return clamp(minimum / asfloat(widths.w), 0.01, 0.99) }
            return clamp(2.0 * minimum / asfloat(widths.z) - 1.0, 0.01, 0.99)
        }
        threshold: fn(n: u32) -> f32 {
            let widths = self.parameters[0u]
            let outer = asfloat(widths.x); let inner = asfloat(widths.y)
            let cap = self.parameters[2u].z
            if n == 0u { return asfloat(self.parameters[1u].y) }
            // The strategy's cap to cap + 1 boundary: nominal(cap + 1) - 0.01 mm.
            if n == cap { return 2.0 * outer + f32(cap - 1u) * inner - 0.1 }
            if n == 1u { return outer * (1.0 + self.threshold_fraction(1u)) }
            return 2.0 * outer + (f32(n) - 2.0 + self.threshold_fraction(n)) * inner
        }
        count: fn(thickness: f32) -> u32 {
            if !(thickness - thickness == 0.0) || thickness < 0.0 { return 0u }
            let widths = self.parameters[0u]
            let outer = asfloat(widths.x); let inner = asfloat(widths.y)
            let cap = self.parameters[2u].z
            if cap == 0u { return 0u }
            if thickness < inner {
                if thickness >= asfloat(self.parameters[1u].y) { return 1u }
                return 0u
            }
            if thickness < outer * 0.5 { return 0u }
            if thickness <= 2.0 * outer {
                if thickness > self.threshold(1u) { return 2u }
                return 1u
            }
            if thickness >= self.threshold(cap) { return cap + 1u }
            let q = (thickness - 2.0 * outer) / inner
            let n = min(u32(floor(q)), cap)
            var extra = 0u
            if q - f32(n) >= self.threshold_fraction(n) { extra = 1u }
            return min(2u + n + extra, cap)
        }
        stride: fn() -> u32 { return self.parameters[2u].z + 1u }
        // Packed central paths, samples and edges (graph.rs ThinPathPack).
        sample_total: fn() -> u32 { return self.word(5u) }
        edge_total: fn() -> u32 { return self.word(6u) }
    }

    // One element per (edge, lower count): long paths and large caps add
    // elements, not thread work.
    mod.sdf.ThinCountEvents = {
        ..mod.sdf.ThinLayout
        edges: shader.buffer(vec4)
        events: shader.buffer_rw(vec4u)
        main: fn() {
            let index = self.index()
            if index == 4294967295u || index >= self.edge_total() * self.stride() { return }
            let stride = self.stride()
            let edge = self.edges[index / stride]
            let lower = index % stride
            let a = self.count(edge.x); let b = self.count(edge.y)
            var position = edge.z
            var flags = 0u
            if lower >= min(a, b) && lower < max(a, b) {
                var fraction = 0.5
                if edge.x != edge.y { fraction = clamp((self.threshold(lower) - edge.x) / (edge.y - edge.x), 0.0, 1.0) }
                position += (edge.w - edge.z) * fraction
                flags = 1u
                if b > a { flags |= 2u }
            }
            self.events[index] = vec4u(a, b, asuint(position), flags)
        }
    }
    // Per event slot in path order (a falling edge's thresholds descend): its
    // flag, the exclusive scan's input. Slots past the edge total are 0.
    mod.sdf.ThinEventFlags = {
        ..mod.sdf.ThinLayout
        edges: shader.buffer(vec4)
        events: shader.buffer(vec4u)
        event_flag: shader.buffer_rw(u32)
        ordered_event: fn(i: u32) -> u32 {
            let stride = self.stride()
            let edge = i / stride
            var lower = i % stride
            if self.edges[edge].x > self.edges[edge].y { lower = stride - 1u - lower }
            return edge * stride + lower
        }
        main: fn() {
            let i = self.index()
            if i == 4294967295u || i >= self.spans() * self.stride() { return }
            var flag = 0u
            if i < self.edge_total() * self.stride() { flag = self.events[self.ordered_event(i)].w & 1u }
            self.event_flag[i] = flag
        }
    }
    // candidates: event distance, preceding sample index, lower count, and
    // the path root in the bits above the direction bit.
    mod.sdf.ThinEventScatter = {
        ..mod.sdf.ThinEventFlags
        event_flag: shader.buffer(u32)
        event_off: shader.buffer(u32)
        sample_off: shader.buffer(u32)
        edge_off: shader.buffer(u32)
        candidates: shader.buffer_rw(vec4u)
        main: fn() {
            let i = self.index()
            if i == 4294967295u || i >= self.edge_total() * self.stride() || self.event_flag[i] == 0u { return }
            let original = self.ordered_event(i)
            let event = self.events[original]
            let edge = i / self.stride()
            // The root owning this packed edge: the last whose first edge is
            // at or before it. An empty root shares its successor's offset.
            var lo = 0u; var hi = self.spans()
            for bit in 0..32u {
                if hi - lo <= 1u { break }
                let mid = lo + (hi - lo) / 2u
                if self.edge_off[mid] <= edge { lo = mid } else { hi = mid }
            }
            let sample = self.sample_off[lo] + edge - self.edge_off[lo]
            self.candidates[self.event_off[i]] = vec4u(event.z, sample, original % self.stride(), (lo << 1u) | ((event.w >> 1u) & 1u))
        }
    }
    // Per path root: its event range, and its rules (free start, free end,
    // initial count).
    mod.sdf.ThinEventPaths = {
        ..mod.sdf.ThinLayout
        paths: shader.buffer_rw(vec4u)
        samples: shader.buffer(vec4u)
        node_info: shader.buffer(vec4u)
        ranked: shader.buffer(vec4u)
        event_off: shader.buffer(u32)
        path_rules: shader.buffer_rw(vec4u)
        event_at: fn(slot: u32) -> u32 {
            if slot >= self.spans() * self.stride() { return self.word(7u) }
            return self.event_off[slot]
        }
        main: fn() {
            let root = self.index()
            if root == 4294967295u || root >= self.spans() { return }
            let path = self.paths[root]
            if path.y == 0u { self.path_rules[root] = vec4u(0u); return }
            let stride = self.stride()
            let first = self.event_at(path.z * stride)
            let last = self.event_at((path.z + path.w) * stride)
            self.paths[root] = vec4u(path.x, path.y, first, last - first)
            let rank = self.ranked[root * 4u]; let metric = self.ranked[root * 4u + 1u]
            var free_start = 0u; var free_end = 0u
            if metric.w == 0u {
                if self.node_info[rank.x * 2u].z == 1u { free_start = 1u }
                if self.node_info[metric.z * 2u].z == 1u { free_end = 1u }
            }
            self.path_rules[root] = vec4u(free_start, free_end, self.count(asfloat(self.samples[path.x].y)), 0u)
        }
    }
    // A short non-central path between two central nodes whose counts differ
    // by at most one becomes central (the CPU strategy's gap rule).
    mod.sdf.ThinGaps = {
        ..mod.sdf.ThinLayout
        thin: shader.buffer(u32)
        output: shader.buffer_rw(vec4u)
        node_info: shader.buffer(vec4u)
        ranked: shader.buffer(vec4u)
        main: fn() {
            let edge = self.index()
            let e = edge * 2u
            if edge == 4294967295u || edge >= self.spans() || self.thin[self.graph() + e * 4u + 2u] == 0u { return }
            let rank = self.ranked[e * 2u]
            let metric = self.ranked[e * 2u + 1u]
            let a = rank.x; let b = metric.z
            if metric.w != 0u { return }
            if self.node_info[a * 2u].z == 0u || self.node_info[b * 2u].z == 0u { return }
            if asfloat(metric.y) >= asfloat(self.parameters[1u].z) { return }
            let ra = self.output[(a / 2u) * 4u + a % 2u].z
            let rb = self.output[(b / 2u) * 4u + b % 2u].z
            let ca = self.count(2.0 * asfloat(ra)); let cb = self.count(2.0 * asfloat(rb))
            if max(ca, cb) - min(ca, cb) > 1u { return }
            for end in 0..2u {
                let at = edge * 4u + end
                let value = self.output[at]
                self.output[at] = vec4u(value.xyz, value.w | 4u)
            }
        }
    }
    // A sparse table of each path's (min, max) thickness: level dispatch_range.z
    // covers 2^level samples. Its rows are ends() samples apart (the sample bound).
    mod.sdf.ThinPathRanges = {
        ..mod.sdf.ThinLayout
        // samples: distance, thickness, path root, node (all packed).
        // paths: first sample, sample count, first event, event count.
        samples: shader.buffer(vec4u)
        paths: shader.buffer(vec4u)
        ranges: shader.buffer_rw(vec2)
        main: fn() {
            let i = self.index()
            if i == 4294967295u || i >= self.sample_total() { return }
            let sample = self.samples[i]
            let level = self.dispatch_range.z
            let total = self.ends()
            var value = vec2(asfloat(sample.y))
            if level != 0u {
                let previous = (level - 1u) * total
                value = self.ranges[previous + i]
                let other = i + (1u << (level - 1u))
                let path = self.paths[sample.z]
                if other < path.x + path.y {
                    let right = self.ranges[previous + other]
                    value = vec2(min(value.x, right.x), max(value.y, right.y))
                }
            }
            self.ranges[level * total + i] = value
        }
    }
    mod.sdf.ThinTransitionFilter = {
        ..mod.sdf.ThinPathRanges
        ranges: shader.buffer(vec2)
        candidates: shader.buffer(vec4u)
        filter_state: shader.buffer_rw(vec4u)
        kept: shader.buffer_rw(u32)
        main: fn() {
            let root = self.index()
            if root == 4294967295u || root >= self.spans() { return }
            let path = self.paths[root]
            // One element filters its path's events in order: the stack is
            // sequential, and its cost is the path's event count.
            var depth = 0u
            for current in path.z..path.z + path.w {
                let event = self.candidates[current]
                var remove = false
                if depth != 0u {
                    let previous = self.candidates[self.kept[path.z + depth - 1u]]
                    // Opposite transitions of one count within 100 mm.
                    if previous.z == event.z && (previous.w & 1u) != (event.w & 1u)
                        && asfloat(event.x) - asfloat(previous.x) <= 1000.0 {
                        let first = previous.y + 1u
                        let last = event.y
                        remove = true
                        if first <= last {
                            let length = last - first + 1u
                            var level = 0u
                            for bit in 1..32u { if (1u << bit) <= length { level = bit } }
                            let a = self.ranges[level * self.ends() + first]
                            let b = self.ranges[level * self.ends() + last + 1u - (1u << level)]
                            let threshold = self.threshold(event.z)
                            let deviation = asfloat(self.parameters[1u].w)
                            remove = abs(min(a.x, b.x) - threshold) <= deviation && abs(max(a.y, b.y) - threshold) <= deviation
                        }
                    }
                }
                if remove { depth -= 1u }
                else { self.kept[path.z + depth] = current; depth += 1u }
            }
            self.filter_state[root] = vec4u(path.w, depth, 0u, 0u)
        }
    }
    mod.sdf.ThinRamps = {
        ..mod.sdf.ThinTransitionFilter
        filter_state: shader.buffer(vec4u)
        kept: shader.buffer(u32)
        // Per path: free start, free end, initial count, reserved.
        path_rules: shader.buffer(vec4u)
        // Per event slot: lo, hi, direction, adjustment to the initial count.
        ramps: shader.buffer_rw(vec4)
        main: fn() {
            let root = self.index()
            if root == 4294967295u || root >= self.spans() { return }
            let path = self.paths[root]
            if path.w == 0u { return }
            let kept = self.filter_state[root].y
            let rule = self.path_rules[root]
            let total = asfloat(self.samples[path.x + path.y - 1u].x)
            for slot in 0..path.w {
                let i = path.z + slot
                if slot >= kept { self.ramps[i] = vec4(0.0); continue }
                let event = self.candidates[self.kept[i]]
                let up = (event.w & 1u) != 0u
                var length = asfloat(self.parameters[1u].z)
                // The 0 to 1 transition ramps over 0.01 mm.
                if event.z == 0u { length = 0.1 }
                var anchor = self.threshold_fraction(event.z)
                var direction = 0.0 - 1.0
                if up { anchor = 1.0 - anchor; direction = 1.0 }
                var lo = asfloat(event.x) - anchor * length
                var hi = lo + length
                if lo < 0.0 {
                    if rule.x != 0u { self.ramps[i] = vec4(0.0, 0.0, 0.0, direction); continue }
                    hi -= lo; lo = 0.0
                }
                if hi > total {
                    if rule.y != 0u { self.ramps[i] = vec4(0.0); continue }
                    lo = max(0.0, lo - (hi - total)); hi = total
                }
                self.ramps[i] = vec4(lo, hi, direction, 0.0)
            }
        }
    }
    mod.sdf.ThinPathCounts = {
        ..mod.sdf.ThinRamps
        ramps: shader.buffer(vec4)
        fractional: shader.buffer_rw(f32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u || i >= self.sample_total() { return }
            let sample = self.samples[i]
            let path = self.paths[sample.z]
            let rules = self.path_rules[sample.z]
            var value = f32(rules.z)
            // The canonical ramp order fixes floating-point accumulation order.
            for event in 0..self.filter_state[sample.z].y {
                let ramp = self.ramps[path.z + event]
                var fraction = 0.0
                if ramp.y > ramp.x { fraction = clamp((asfloat(sample.x) - ramp.x) / (ramp.y - ramp.x), 0.0, 1.0) }
                else if asfloat(sample.x) >= ramp.x { fraction = 1.0 }
                value += ramp.w + ramp.z * fraction
            }
            if (i == path.x && rules.x == 0u) || (i + 1u == path.x + path.y && rules.y == 0u) { value = f32(self.count(asfloat(sample.y))) }
            self.fractional[i] = value
        }
    }

    // Per canonical node: the lowest packed sample of its incident central
    // paths (one writer per node), then each node's layout query from its owner.
    mod.sdf.ThinLayoutSamples = {
        ..mod.sdf.ThinLayout
        output: shader.buffer(vec4u)
        thin: shader.buffer(u32)
        node_info: shader.buffer(vec4u)
        adj_off: shader.buffer(u32)
        adjacency: shader.buffer(u32)
        ranked: shader.buffer(vec4u)
        paths: shader.buffer(vec4u)
        query_sources: shader.buffer_rw(u32)
        main: fn() {
            let i = self.index()
            if i == 4294967295u || i >= self.ends() { return }
            let node = self.node_info[i * 2u]
            var sample = 4294967295u
            if node.x == i {
                for k in 0..node.y {
                    let arm = self.adjacency[self.ends() + self.adj_off[i] + k]
                    if (self.output[(arm / 2u) * 4u + arm % 2u].w & 6u) == 0u { continue }
                    let root = self.thin[self.graph() + arm * 4u + 3u]
                    sample = min(sample, self.paths[root].x + self.ranked[arm * 2u].y)
                }
            }
            self.query_sources[i] = sample
        }
    }
    // queries: thickness and fractional count per endpoint.
    mod.sdf.ThinLayoutQueries = {
        ..mod.sdf.ThinLayout
        output: shader.buffer(vec4u)
        node_info: shader.buffer(vec4u)
        owners: shader.buffer(vec4u)
        query_sources: shader.buffer(u32)
        fractional: shader.buffer(f32)
        queries: shader.buffer_rw(vec4)
        main: fn() {
            let i = self.index()
            if i == 4294967295u || i >= self.ends() { return }
            let node = self.node_info[i * 2u]
            var query = vec4(0.0)
            if node.y != 0u {
                let owner = self.owners[self.ends() * 2u + node.x].x
                let thickness = 2.0 * asfloat(self.output[(owner / 2u) * 4u + owner % 2u].z)
                var q = f32(self.count(thickness))
                let sample = self.query_sources[owner]
                if sample != 4294967295u { q = self.fractional[sample] }
                if self.node_info[owner * 2u + 1u].x != 0u { q = f32(self.parameters[2u].z + 1u) }
                query = vec4(thickness, q, 0.0, 0.0)
            }
            self.queries[i] = query
        }
    }

    // Bead layouts. Each query has two integer layouts (floor and ceiling of
    // its fractional count), each a row of cap + 2 slots (with the possible
    // fill markers). Entries: ends · 2 · (cap + 2). bead_work holds two
    // banks; dispatch_range.w is the source bank (and the scan's reduction
    // kind in bit 1), a scan's round is dispatch_range.z.
    mod.sdf.ThinBeadWeights = {
        ..mod.sdf.ThinLayout
        queries: shader.buffer(vec4)
        bead_work: shader.buffer_rw(vec4)
        bead_values: shader.buffer_rw(vec4)
        row: fn() -> u32 { return self.parameters[2u].z + 2u }
        entries: fn() -> u32 { return self.ends() * 2u * self.row() }
        source: fn() -> u32 { return self.entries() * (self.dispatch_range.w & 1u) }
        destination: fn() -> u32 { return self.entries() * (1u - (self.dispatch_range.w & 1u)) }
        description: fn(i: u32) -> vec4u {
            let stride = self.row()
            let row = i / stride
            let query = self.queries[row / 2u]
            if query.x <= 0.0 { return vec4u(0u) }
            let cap = self.parameters[2u].z
            let n = min(u32(floor(clamp(query.y, 0.0, f32(cap + 1u)))) + row % 2u, cap + 1u)
            let slot = i % stride
            var bead = slot
            var kind = 1u
            if n == cap {
                if slot >= cap + 1u { kind = 0u }
                else if slot == cap / 2u { kind = 2u }
                else if slot > cap / 2u { bead -= 1u }
            } else if n > cap {
                if slot == cap / 2u { kind = 2u }
                else if slot == cap / 2u + 1u { kind = 3u }
                else if slot > cap / 2u + 1u { bead -= 2u }
            } else if slot >= n { kind = 0u }
            return vec4u(n, min(n, cap), bead, kind)
        }
        dimensions: fn(i: u32, description: vec4u) -> vec4 {
            let widths = self.parameters[0u]
            let outer = asfloat(widths.x); let inner = asfloat(widths.y)
            var thickness = self.queries[i / (self.row() * 2u)].x
            if description.x > description.y { thickness = 2.0 * outer + (f32(description.y) - 2.0) * inner }
            let outside = min(thickness * 0.5, outer)
            return vec4(thickness, outside, max(0.0, thickness - 2.0 * outside), max(0.0, f32(description.y) - 2.0))
        }
        weight: fn(description: vec4u) -> f32 {
            if description.w != 1u || description.y <= 2u || description.z == 0u || description.z + 1u == description.y { return 0.0 }
            let inner_count = description.y - 2u
            if inner_count <= 2u { return 1.0 }
            let centre = f32(inner_count - 1u) * 0.5
            let radius = max(1.0, f32(self.parameters[2u].y) - 1.0)
            let d = (f32(description.z - 1u) - centre) / radius
            return max(0.0, 1.0 - d * d)
        }
        row_total: fn(i: u32) -> vec4 { return self.bead_work[self.source() + (i / self.row() + 1u) * self.row() - 1u] }
        main: fn() {
            let i = self.index()
            if i == 4294967295u || i >= self.entries() { return }
            self.bead_work[i] = vec4(self.weight(self.description(i)), 0.0, 0.0, 0.0)
        }
    }
    // An inclusive segmented scan over each row: round z adds the value 2^z
    // slots back. Reduction kind 0 sums; kind 1 sums x, takes max y, min z.
    mod.sdf.ThinBeadScan = {
        ..mod.sdf.ThinBeadWeights
        main: fn() {
            let i = self.index()
            if i == 4294967295u || i >= self.entries() { return }
            let distance = 1u << self.dispatch_range.z
            let bank = (self.dispatch_range.w + self.dispatch_range.z) & 1u
            let from = self.entries() * bank
            let into = self.entries() * (1u - bank)
            var value = self.bead_work[from + i]
            if i % self.row() >= distance {
                let other = self.bead_work[from + i - distance]
                if (self.dispatch_range.w & 2u) == 0u { value += other }
                else { value = vec4(value.x + other.x, max(value.y, other.y), min(value.z, other.z), 0.0) }
            }
            self.bead_work[into + i] = value
        }
    }
    mod.sdf.ThinBeadRaw = {
        ..mod.sdf.ThinBeadWeights
        main: fn() {
            let i = self.index()
            if i == 4294967295u || i >= self.entries() { return }
            let description = self.description(i)
            let dimensions = self.dimensions(i, description)
            var width = 0.0
            var interior = 0.0
            if description.w == 1u {
                if description.y <= 2u {
                    width = dimensions.x / f32(description.y)
                    if description.y == 1u && dimensions.x < asfloat(self.parameters[0u].y) {
                        width = 0.0
                        if dimensions.x >= asfloat(self.parameters[1u].y) { width = max(dimensions.x, asfloat(self.parameters[1u].x)) }
                    }
                } else if description.z == 0u || description.z + 1u == description.y { width = dimensions.y }
                else {
                    let inner = asfloat(self.parameters[0u].y)
                    let total = self.row_total(i).x
                    width = max(0.0, inner + (dimensions.z - dimensions.w * inner) * self.weight(description) / total)
                    interior = width
                }
            }
            self.bead_values[i] = vec4(width, 0.0, 0.0, 0.0)
            self.bead_work[self.destination() + i] = vec4(interior, width, 0.0, 0.0)
        }
    }
    mod.sdf.ThinBeadNormalize = {
        ..mod.sdf.ThinBeadWeights
        main: fn() {
            let i = self.index()
            if i == 4294967295u || i >= self.entries() { return }
            let description = self.description(i)
            var width = self.bead_values[i].x
            if description.w == 1u && description.y > 2u && description.z > 0u && description.z + 1u < description.y {
                let dimensions = self.dimensions(i, description)
                let actual = self.row_total(i).x
                // The strategy's 1e-12 mm.
                if actual > dimensions.z + 0.00000000001 { width *= dimensions.z / actual }
                else if description.z + 2u == description.y { width += dimensions.z - actual }
            }
            self.bead_values[i] = vec4(width, 0.0, 0.0, 0.0)
            self.bead_work[self.destination() + i] = vec4(width, 0.0, 0.0, 0.0)
        }
    }
    mod.sdf.ThinBeadClassify = {
        ..mod.sdf.ThinBeadWeights
        main: fn() {
            let i = self.index()
            if i == 4294967295u || i >= self.entries() { return }
            let description = self.description(i)
            let query = self.queries[i / (self.row() * 2u)]
            let centre = query.x * 0.5
            let width = self.bead_values[i].x
            let prefix = self.bead_work[self.source() + i].x
            var location = prefix - width * 0.5
            if description.x > description.y {
                if description.w == 3u { location = query.x - prefix }
                else if description.w == 1u && description.z >= description.y / 2u {
                    location += query.x - self.row_total(i).x
                }
            } else if description.w == 2u || (description.w == 1u && width == 0.0)
                || (description.y % 2u == 1u && description.z == description.y / 2u) {
                // Symmetry fixes the middle exactly; accumulated f32 sums must
                // not turn it into a side bead or a fill marker.
                location = centre
            }
            var side = 0.0; var middle = 0.0
            var marker = asfloat(2139095040u)
            if description.w != 0u {
                // The strategy's 1e-9 mm.
                if width <= 0.0 {
                    if location < centre - 0.00000001 { marker = location }
                } else if abs(location - centre) <= 0.00000001 { middle = width }
                else if location < centre { side = 1.0 }
            }
            self.bead_values[i] = vec4(location, width, side, 0.0)
            self.bead_work[self.destination() + i] = vec4(side, middle, marker, 0.0)
        }
    }
    // Per row: its side beads in order, and in its last slot (side count,
    // middle width, marker or NaN).
    mod.sdf.ThinBeadHalf = {
        ..mod.sdf.ThinBeadWeights
        bead_work: shader.buffer(vec4)
        bead_values: shader.buffer(vec4)
        halves: shader.buffer_rw(vec4)
        main: fn() {
            let i = self.index()
            if i == 4294967295u || i >= self.entries() { return }
            let row = (i / self.row()) * self.row()
            let total = self.bead_work[self.source() + row + self.row() - 1u]
            let bead = self.bead_values[i]
            if bead.z != 0.0 {
                let ordinal = u32(self.bead_work[self.source() + i].x) - 1u
                var location = bead.x
                if ordinal == 0u && self.parameters[2u].w != 0u {
                    let widths = self.parameters[0u]
                    location = max(0.0, location - 0.5 * (asfloat(widths.z) - asfloat(widths.x)))
                }
                self.halves[row + ordinal] = vec4(location, bead.y, 0.0, 0.0)
            }
            if i % self.row() + 1u == self.row() {
                var marker = total.z
                if marker == asfloat(2139095040u) { marker = asfloat(2143289344u) }
                self.halves[i] = vec4(total.x, total.y, marker, 0.0)
            }
        }
    }
    // Per endpoint, cap / 2 + 1 slots: its side beads (location, width) and
    // last (count, middle width, marker), blended between its two rows.
    mod.sdf.ThinBeadBlend = {
        ..mod.sdf.ThinLayout
        queries: shader.buffer(vec4)
        halves: shader.buffer(vec4)
        layouts: shader.buffer_rw(vec4)
        main: fn() {
            let i = self.index()
            let cap = self.parameters[2u].z
            let slots = cap / 2u + 1u
            if i == 4294967295u || i >= self.ends() * slots { return }
            let row = cap + 2u
            let query = self.queries[i / slots]
            let centre = query.x * 0.5
            let low = (i / slots) * row * 2u
            let high = low + row
            let a = self.halves[low + row - 1u]
            let b = self.halves[high + row - 1u]
            let q = clamp(query.y, 0.0, f32(cap + 1u))
            var t = q - floor(q)
            if t <= 0.000000001 { t = 0.0 }
            var count = a.x
            if t != 0.0 { count = max(count, b.x) }
            let slot = i % slots
            if slot + 1u == slots {
                var marker = a.z
                if t != 0.0 && (a.z == a.z || b.z == b.z) {
                    var first = a.z; var second = b.z
                    if !(first == first) { first = centre }
                    if !(second == second) { second = centre }
                    marker = first + (second - first) * t
                }
                self.layouts[i] = vec4(count, a.y + (b.y - a.y) * t, marker, 0.0)
            } else {
                var first = vec2(centre, 0.0); var second = first
                if f32(slot) < a.x { first = self.halves[low + slot].xy }
                if f32(slot) < b.x { second = self.halves[high + slot].xy }
                self.layouts[i] = vec4(first.x + (second.x - first.x) * t, first.y + (second.y - first.y) * t, 0.0, 0.0)
            }
        }
    }

    mod.sdf
}
