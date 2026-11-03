//! The thin site set: the skeleton's sites, built from a FeatureSet's segs.
//!
//! Every grid crossing is a seg endpoint (GPU-EXEC §10.3), so a straight wall
//! arrives as a chain of nearly collinear segs kinked only by trace noise. The
//! exact skeleton of those segs would branch at every convex kink with an
//! ill-conditioned bisector and grow a point site at every reflex one, and a
//! sub-cell move would flip kinks between the two. So consecutive segs merge
//! into one segment site while every vertex stays within [`SITE_TOL`] of the
//! run's chord: the CPU oracle's simplify_ring tolerance (0.0025 mm, in
//! fab/toolpath arachne prepare), applied before its own skeleton.
//!
//! Runs never cross a hard break, a vertex that is a crease or turns by more
//! than [`HARD_TURN`] between two segs of at least [`SITE_TOL`]; each chunk
//! between hard breaks is walked from its own start, so on a polygon the sites
//! are its edges at any translation. A loop without a hard break (a circle, a
//! smooth ring) is one chunk from its root seg, and its last run merges into
//! its first when the first still fits the last run's cone.
//!
//! A site is named by its first seg `r`: segment site `2r`, and point site
//! `2r + 1` for its start vertex when the turn from the previous site's chord
//! is reflex. `site_of[s]` is the run holding seg s.
//!
//! The cell site lists stay conservative: cand(cell) holds every feature whose
//! lower bound over the cell's disk is within 2r + eps of the nearest (r the
//! cell radius 0.7072), and a site's chord is within SITE_TOL of each of its
//! segs, so a site that is nearest anywhere in the cell has a seg within
//! 2 * SITE_TOL = 0.05 of the nearest, far inside that margin.
use makepad_draw::*;

/// The run tolerance in grid units (0.0025 mm, the oracle's simplify_ring).
pub const SITE_TOL: f32 = 0.025;
/// A turn of more than this (radians) between two segs of at least SITE_TOL is a hard break.
pub const HARD_TURN: f32 = 0.1;

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.Thin = {
        // The count and write sweeps evaluate the same expressions: a precise
        // module fuses and reorders nothing.
        precise_math: true
        workgroup_size: 64
        counts: shader.buffer_rw(u32)
        // cfg rows (vec4u):
        // 0: nx, ny, planes, band cells (band slots * 100);
        // 1: f32 bits of the reach cap, central sine, curve error (grid units), tie capacity;
        // 2: the thin counts base word, segs, spans (written once the span
        //    scan's total is known), the bead cap;
        // 3: reserved; 4 + slot: per band slot its coarse cell and plane
        //    (ThinSlotInfo).
        // One buffer, not params or one per table: the push block holds 8
        // bytes per referenced buffer and 128 in all.
        cfg: shader.buffer(vec4u)
        dispatch_range: shader.uniform(vec4u)
        index: fn() -> u32 {
            let i = global_id().x
            if i >= self.dispatch_range.y { return 4294967295u }
            return self.dispatch_range.x + i
        }
        word: fn(k: u32) -> u32 { return self.counts[self.cfg[2u].x + k] }
        segs: fn() -> u32 { return self.cfg[2u].y }
        cross: fn(a: vec2, b: vec2) -> f32 { return a.x * b.y - a.y * b.x }
        xy: fn(v: vec4u) -> vec2 { return vec2(asfloat(v.x), asfloat(v.y)) }
        equal: fn(a: vec4u, b: vec4u) -> bool { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w }
        // The f32 rounding bound of a position in grid units.
        roundoff: fn(p: vec2) -> f32 { return 0.00002 + max(abs(p.x), abs(p.y)) * 0.00000024 }
        nx: fn() -> u32 { return self.cfg[0u].x }
        ny: fn() -> u32 { return self.cfg[0u].y }
        cells: fn() -> u32 { return self.cfg[0u].w }
        reach: fn() -> f32 { return asfloat(self.cfg[1u].x) }
        central_sine: fn() -> f32 { return asfloat(self.cfg[1u].y) }
        curve_error: fn() -> f32 { return asfloat(self.cfg[1u].z) }
        ties: fn() -> u32 { return self.cfg[1u].w }
        // Spans and their ends (the pair sweep's exact total).
        spans: fn() -> u32 { return self.cfg[2u].z }
        cap: fn() -> u32 { return self.cfg[2u].w }
        slot: fn(s: u32) -> vec2u { return self.cfg[4u + s].xy }
        ends: fn() -> u32 { return self.spans() * 2u }
        // Regions of the thin scratch buffer: per band cell its span range,
        // then per end its key, graph link and two jump banks, then its ties.
        keys: fn() -> u32 { return self.cells() * 2u }
        graph: fn() -> u32 { return self.keys() + self.ends() * 4u }
        jump_a: fn() -> u32 { return self.graph() + self.ends() * 4u }
        jump_b: fn() -> u32 { return self.jump_a() + self.ends() * 2u }
        incidence: fn() -> u32 { return self.jump_b() + self.ends() * 2u }
    }

    mod.sdf.ThinSiteBase = {
        ..mod.sdf.Thin
        seg: shader.buffer(vec4f)
        seg_info: shader.buffer(vec4u)
        // Per seg: 1 when its start vertex is a hard break.
        site_flag: shader.buffer_rw(u32)
        // Per loop root seg (seg_info.x): 1 when any vertex of its loop is a hard break.
        loop_hard: shader.buffer_rw(u32)
        // Per seg: 1 when a chunk starts at it (the scan's input).
        chunk_flag: shader.buffer_rw(u32)
        tol: fn() -> f32 { return 0.025 }
        seg_len: fn(s: u32) -> f32 { let q = self.seg[s]; return length(q.zw - q.xy) }
        seg_dir: fn(s: u32) -> vec2 {
            let q = self.seg[s]
            let l = length(q.zw - q.xy)
            if l > 0.0 { return (q.zw - q.xy) / l }
            return vec2(1.0, 0.0)
        }
    }

    mod.sdf.ThinLoopClear = {
        ..mod.sdf.ThinSiteBase
        main: fn() {
            let s = self.index()
            if s == 4294967295u || s >= self.segs() { return }
            self.loop_hard[s] = 0u
        }
    }

    mod.sdf.ThinSiteBreaks = {
        ..mod.sdf.ThinSiteBase
        main: fn() {
            let s = self.index()
            if s == 4294967295u || s >= self.segs() { return }
            let info = self.seg_info[s]
            let prev = info.z
            var hard = (info.w & 2u) != 0u
            if !hard && self.seg_len(s) >= self.tol() && self.seg_len(prev) >= self.tol() {
                let a = self.seg_dir(prev); let b = self.seg_dir(s)
                hard = abs(atan2(self.cross(a, b), dot(a, b))) > 0.1
            }
            var flag = 0u
            if hard { flag = 1u; atomic_max(self.loop_hard[info.x], 1u) }
            self.site_flag[s] = flag
        }
    }

    mod.sdf.ThinChunkStarts = {
        ..mod.sdf.ThinSiteBase
        main: fn() {
            let s = self.index()
            if s == 4294967295u || s >= self.segs() { return }
            let root = self.seg_info[s].x
            var start = self.site_flag[s] != 0u
            if self.loop_hard[root] == 0u && root == s { start = true }
            var flag = 0u
            if start { flag = 1u }
            self.chunk_flag[s] = flag
        }
    }

    mod.sdf.ThinChunkList = {
        ..mod.sdf.ThinSiteBase
        chunk_off: shader.buffer(u32)
        chunks: shader.buffer_rw(u32)
        main: fn() {
            let s = self.index()
            if s == 4294967295u || s >= self.segs() { return }
            if self.chunk_flag[s] != 0u { self.chunks[self.chunk_off[s]] = s }
        }
    }

    // One thread per chunk walks its segs in loop order. A run's cone holds
    // the chord directions from its start that keep every vertex so far
    // within tol; the next vertex extends the run when its direction lies in
    // the cone and it does not double back, else a new run starts at the
    // previous vertex.
    mod.sdf.ThinRuns = {
        ..mod.sdf.ThinSiteBase
        chunk_off: shader.buffer(u32)
        chunks: shader.buffer(u32)
        site_of: shader.buffer_rw(u32)
        site: shader.buffer_rw(vec4f)
        site_info: shader.buffer_rw(vec4u)
        // The cone of q seen from p0: its direction turned by -+asin(tol / l).
        side: fn(u: vec2, s: f32, sign: f32) -> vec2 {
            let c = sqrt(max(1.0 - s * s, 0.0))
            return vec2(u.x * c - sign * u.y * s, sign * u.x * s + u.y * c)
        }
        // cone: (lo, hi) unit directions; state: (open, farthest distance).
        fits: fn(p0: vec2, q: vec2, cone: vec4, state: vec2) -> bool {
            let d = q - p0
            let l = length(d)
            if l <= self.tol() { return true }
            if l < state.y - self.tol() { return false }
            if state.x == 0.0 { return true }
            let u = d / l
            return self.cross(cone.xy, u) >= 0.0 && self.cross(u, cone.zw) >= 0.0
        }
        narrow: fn(p0: vec2, q: vec2, cone: vec4, state: vec2) -> vec4 {
            let d = q - p0
            let l = length(d)
            if l <= self.tol() { return cone }
            let u = d / l
            let s = self.tol() / l
            let lo = self.side(u, s, 0.0 - 1.0)
            let hi = self.side(u, s, 1.0)
            if state.x == 0.0 { return vec4(lo, hi) }
            var out = cone
            if self.cross(cone.xy, lo) > 0.0 { out = vec4(lo, out.zw) }
            if self.cross(hi, cone.zw) > 0.0 { out = vec4(out.xy, hi) }
            return out
        }
        grow: fn(p0: vec2, q: vec2, state: vec2) -> vec2 {
            let l = length(q - p0)
            if l <= self.tol() { return state }
            return vec2(1.0, max(state.y, l))
        }
        close: fn(r: u32, p0: vec2, last: u32) {
            self.site[r] = vec4f(p0.x, p0.y, self.seg[last].z, self.seg[last].w)
            self.site_info[r] = vec4u(last, 0u, 0u, 0u)
        }
        main: fn() {
            let i = self.index()
            if i == 4294967295u || i >= self.word(0u) { return }
            let first = self.chunks[i]
            var r = first
            var p0 = self.seg[first].xy
            var cone = vec4(0.0)
            var state = vec2(0.0)
            var s = first
            var last = first
            var walking = true
            var steps = 0u
            while walking {
                // A chain that never closes is a FeatureSet defect: stop and count it.
                steps += 1u
                if steps > self.segs() { atomic_add(self.counts[self.cfg[2u].x + 12u], 1u); break }
                let q = self.seg[s].zw
                if s != r && !self.fits(p0, q, cone, state) {
                    self.close(r, p0, last)
                    r = s; p0 = self.seg[s].xy; cone = vec4(0.0); state = vec2(0.0)
                }
                cone = self.narrow(p0, q, cone, state)
                state = self.grow(p0, q, state)
                self.site_of[s] = r
                last = s
                let n = self.seg_info[s].y
                if n == first || self.chunk_flag[n] != 0u { walking = false }
                s = n
            }
            // A loop without hard breaks is one chunk from its root: its last
            // run takes in the first one when the first still fits its cone.
            let wraps = self.seg_info[last].y == first && self.site_flag[first] == 0u
            if wraps && r != first {
                let end = self.site_info[first].x
                var fit = true
                var c = cone
                var st = state
                var t = first
                var going = true
                var guard = 0u
                while going {
                    guard += 1u
                    if guard > self.segs() { fit = false; break }
                    let q = self.seg[t].zw
                    if !self.fits(p0, q, c, st) { fit = false; going = false }
                    else {
                        c = self.narrow(p0, q, c, st)
                        st = self.grow(p0, q, st)
                        if t == end { going = false }
                        t = self.seg_info[t].y
                    }
                }
                if fit {
                    t = first
                    going = true
                    while going {
                        self.site_of[t] = r
                        if t == end { going = false }
                        t = self.seg_info[t].y
                    }
                    last = end
                }
            }
            self.close(r, p0, last)
        }
    }

    // Per run start: the next and previous sites, and whether the start
    // vertex is a reflex point site (the turn between the two chords).
    mod.sdf.ThinSiteLinks = {
        ..mod.sdf.ThinSiteBase
        site_of: shader.buffer(u32)
        site: shader.buffer(vec4f)
        site_info: shader.buffer_rw(vec4u)
        main: fn() {
            let s = self.index()
            if s == 4294967295u || s >= self.segs() || self.site_of[s] != s { return }
            let last = self.site_info[s].x
            let next = self.site_of[self.seg_info[last].y]
            let prev = self.site_of[self.seg_info[s].z]
            let a = self.site[prev]; let b = self.site[s]
            var flags = 2u
            if self.cross(a.zw - a.xy, b.zw - b.xy) < 0.0 {
                flags = 3u
                atomic_add(self.counts[self.cfg[2u].x + 18u], 1u)
            }
            self.site_info[s] = vec4u(last, next, prev, flags)
        }
    }

    mod.sdf
}
