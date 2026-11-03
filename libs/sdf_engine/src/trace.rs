//! Exact layer tracing in the printer grid. Edge identities and endpoint-only
//! refinement make adjacent workgroups agree without a GPU deduplication pass.
use makepad_draw::*;

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.Trace = {
        ..mod.sdf.Jobs
        precise_math: true
        workgroup_size: 64
        // Grid-to-shape frame rows, printer grid origin + Lipschitz bound, then layer z's.
        inputs: shader.buffer(vec4f)
        kept: shader.buffer_rw(vec4u)
        segments: shader.buffer_rw(vec4u)
        grid: shader.uniform(vec4u) // nx, ny, layers, kept capacity
        capacity: shader.uniform(u32)
        values: shader.buffer_rw(u32)
        // One BVH node per step; validated leaves hold at most eight triangles.
        mesh_step: fn(task: u32, slot: u32, p: vec3) -> vec2 {
            let h = self.mesh_read(slot, 0u)
            let state = task + 144u
            var phase = self.continuations[state]
            var count = self.continuations[state + 1u]
            var best = asfloat(self.continuations[state + 2u])
            var winding = asfloat(self.continuations[state + 3u])
            if count == 0u {
                if phase == 1u {
                    if winding > 0.5 { best = asfloat(asuint(best) ^ 2147483648u) }
                    return vec2(1.0, best)
                }
                phase = 1u; count = 1u; self.continuations[task + 160u] = 0u
            }
            if h.x == 0u { return vec2(1.0, best) }
            count -= 1u
            let node = self.continuations[task + 160u + count]
            let at = 1u + node * 4u
            var visit = true
            if phase == 0u {
                if self.mesh_box(slot, node, p) >= best * best { visit = false }
            } else {
                let d = self.mesh_xyz(slot, at + 3u) - p
                let dl = self.len3(d.x, d.y, d.z)
                let radius = asfloat(self.mesh_read(slot, at + 3u).w)
                if dl > 2.0 * radius && dl > 0.0 {
                    winding += self.mesh_dot(self.mesh_xyz(slot, at + 2u), d) / (12.566370964050293 * dl * dl * dl)
                    visit = false
                }
            }
            if visit {
                let left = self.mesh_read(slot, at).w
                let right = self.mesh_read(slot, at + 1u).w
                if self.mesh_read(slot, at + 2u).w != 0u {
                    for triangle in 0..8u {
                        let i = left + triangle
                        if i >= right { break }
                        let t = self.mesh_read(slot, h.z + self.mesh_read(slot, h.w + i).x)
                        let a = self.mesh_xyz(slot, h.y + t.x)
                        let b = self.mesh_xyz(slot, h.y + t.y)
                        let c = self.mesh_xyz(slot, h.y + t.z)
                        if phase == 0u {
                            let d = p - self.mesh_closest(p, a, b, c)
                            let dl = self.len3(d.x, d.y, d.z)
                            if dl < best { best = dl }
                        } else { winding += self.mesh_angle(p, a, b, c) }
                    }
                } else {
                    var a = left; var b = right
                    if phase == 0u {
                        if self.mesh_box(slot, left, p) < self.mesh_box(slot, right, p) { a = right; b = left }
                    }
                    self.continuations[task + 160u + count] = a
                    self.continuations[task + 161u + count] = b
                    count += 2u
                }
            }
            self.continuations[state] = phase; self.continuations[state + 1u] = count
            self.continuations[state + 2u] = asuint(best); self.continuations[state + 3u] = asuint(winding)
            return vec2(0.0)
        }
        links_step: fn(task: u32, base: u32, q: vec3) -> vec4 {
            let h = self.program[base]
            let r = h.x; let reach = h.y
            let origin = self.program[base + 4u].xyz
            let dim_words = self.program[base + 5u]
            let dims = vec3u(asuint(dim_words.x), asuint(dim_words.y), asuint(dim_words.z))
            let offset_words = self.program[base + 6u]
            let offsets = vec3u(asuint(offset_words.x), asuint(offset_words.y), asuint(offset_words.z))
            let lo = max(floor((q - reach - r - origin) / reach), vec3(0.0))
            let hi = min(floor((q + reach + r - origin) / reach), vec3(f32(dims.x - 1u), f32(dims.y - 1u), f32(dims.z - 1u)))
            let state = task + 192u
            var cell = self.continuations[state]; var member = self.continuations[state + 1u]
            var best = asfloat(self.continuations[state + 2u]); var slot = asfloat(self.continuations[state + 3u])
            if cell == 64u {
                let out = max(max(self.program[base + 2u].xyz - q, q - self.program[base + 3u].xyz), vec3(0.0))
                let d_box = self.len3(out.x, out.y, out.z)
                var d = min(best, reach)
                if d_box > 0.0 { d = max(d_box, d) }
                if h.z != 0.0 {
                    let n = self.program[base + 1u]
                    d = max(d, n.w - ((n.x * q.x + n.y * q.y) + n.z * q.z))
                }
                return vec4(1.0, d, slot, 0.0)
            }
            let x = lo.x + f32(cell % 4u)
            let y = lo.y + f32((cell / 4u) % 4u)
            let z = lo.z + f32(cell / 16u)
            if x > hi.x || y > hi.y || z > hi.z { cell += 1u; member = 0u }
            else {
                let c = (u32(z) * dims.y + u32(y)) * dims.x + u32(x)
                let start = self.links_bits(base, offsets.x + c)
                let end = self.links_bits(base, offsets.x + c + 1u)
                if member >= asuint(h.w) || start + member >= end { cell += 1u; member = 0u }
                else {
                    let at = offsets.z + self.links_bits(base, offsets.y + start + member) * 7u
                    let a = vec3(self.links_word(base, at), self.links_word(base, at + 1u), self.links_word(base, at + 2u))
                    let b = vec3(self.links_word(base, at + 3u), self.links_word(base, at + 4u), self.links_word(base, at + 5u))
                    let ab = b - a; let ap = q - a
                    let l2 = (ab.x * ab.x + ab.y * ab.y) + ab.z * ab.z
                    var t = 0.0
                    if l2 > 0.0 { t = self.clamp01(((ap.x * ab.x + ap.y * ab.y) + ap.z * ab.z) / l2) }
                    let d = self.len3(ap.x - ab.x * t, ap.y - ab.y * t, ap.z - ab.z * t) - r
                    if d < best { best = d; slot = f32(self.links_bits(base, at + 6u)) }
                    member += 1u
                }
            }
            self.continuations[state] = cell; self.continuations[state + 1u] = member
            self.continuations[state + 2u] = asuint(best); self.continuations[state + 3u] = asuint(slot)
            return vec4(0.0)
        }
        // Each call shares this invocation's count budget. A completed sample
        // clears only its interpreter state; the parent's sample cursor moves
        // once. An incomplete sample resumes without repeating side effects.
        field: fn(p: vec2, z: f32) -> vec2 {
            let task = self.task()
            let a = self.inputs[0u]; let b = self.inputs[1u]; let c = self.inputs[2u]
            let x = ((a.x * p.x + a.y * p.y) + a.z * z) + a.w
            let y = ((b.x * p.x + b.y * p.y) + b.z * z) + b.w
            let zz = ((c.x * p.x + c.y * p.y) + c.z * z) + c.w
            let base = 0.0; let masks = 0u; let cell = 0.0
            var known = 0u
            var ones = 0u
            var px = x
            var py = y
            var pz = zz
            var vs = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var ss = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var sx8 = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var sy8 = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var sz8 = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var ms = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var vn = 0i
            var pn = 0i
            var mn = 0i
            let len = self.fetch(0.0).x
            var pc = 1.0
            var phase = self.continuations[task + 64u]
            if phase == 0u { phase = 1u }
            else {
                pc = asfloat(self.continuations[task + 65u])
                px = asfloat(self.continuations[task + 66u]); py = asfloat(self.continuations[task + 67u]); pz = asfloat(self.continuations[task + 68u])
                vn = i32(self.continuations[task + 69u]); pn = i32(self.continuations[task + 70u]); mn = i32(self.continuations[task + 71u])
                for i in 0..16u { vs[i] = asfloat(self.continuations[task + 80u + i]); ss[i] = asfloat(self.continuations[task + 96u + i]) }
                for i in 0..8u {
                    sx8[i] = asfloat(self.continuations[task + 112u + i]); sy8[i] = asfloat(self.continuations[task + 120u + i])
                    sz8[i] = asfloat(self.continuations[task + 128u + i]); ms[i] = asfloat(self.continuations[task + 136u + i])
                }
            }
            var budget = self.continuations[task + 63u]
            for instruction in 0..64u {
                if pc >= len { break }
                let h = self.fetch(base + pc)
                let code = h.x
                // AIR's accepted per-kernel cost is fixed by KernelAir::new.
                // All other steps are one opcode, one <=8-triangle BVH node,
                // or one link candidate, with fixed arithmetic bounds.
                var cost = 512u
                if phase == 2u { cost = 8192u }
                else if phase == 1u && code >= 10.5 && code < 11.5 { cost = #(crate::kernel::MAX_KERNEL_COST as u32) }
                if budget < cost { break }
                budget -= cost
                if phase == 2u {
                    let value = self.mesh_step(task, u32(h.y), vec3(px, py, pz))
                    if value.x != 0.0 {
                        vs[vn] = value.y
                        if mn > 0i { ss[vn] = ms[mn - 1i] } else { ss[vn] = 0.0 }
                        vn += 1i; pc += 1.0; phase = 1u
                    }
                    continue
                }
                if phase == 3u {
                    let value = self.links_step(task, u32(base) + asuint(h.y), vec3(px, py, pz))
                    if value.x != 0.0 {
                        vs[vn] = value.y
                        if value.z != 0.0 { ss[vn] = value.z }
                        else if mn > 0i { ss[vn] = ms[mn - 1i] } else { ss[vn] = 0.0 }
                        vn += 1i; pc += 1.0; phase = 1u
                    }
                    continue
                }
                if code < 0.5 {
                    vs[vn] = self.prim(base + pc, px, py, pz)
                    if mn > 0i { ss[vn] = ms[mn - 1i] } else { ss[vn] = 0.0 }
                    vn = vn + 1i
                    pc = pc + 3.0
                } else if code < 1.5 {
                    sx8[pn] = px
                    sy8[pn] = py
                    sz8[pn] = pz
                    pn = pn + 1i
                    let r0 = self.fetch(base + pc + 1.0)
                    let r1 = self.fetch(base + pc + 2.0)
                    let r2 = self.fetch(base + pc + 3.0)
                    let nx = ((r0.x * px + r0.y * py) + r0.z * pz) + r0.w
                    let ny = ((r1.x * px + r1.y * py) + r1.z * pz) + r1.w
                    let nz = ((r2.x * px + r2.y * py) + r2.z * pz) + r2.w
                    px = nx
                    py = ny
                    pz = nz
                    pc = pc + 4.0
                } else if code < 2.5 {
                    pn = pn - 1i
                    px = sx8[pn]
                    py = sy8[pn]
                    pz = sz8[pn]
                    vs[vn - 1i] = vs[vn - 1i] * h.y
                    pc = pc + 1.0
                } else if code < 3.5 {
                    sx8[pn] = px
                    sy8[pn] = py
                    sz8[pn] = pz
                    pn = pn + 1i
                    let x = px
                    let y = py
                    let z = pz
                    if h.y < 0.5 {
                        let e = self.psc(h.z * y)
                        px = e.y * x + e.x * z
                        pz = e.y * z - e.x * x
                    } else if h.y < 1.5 {
                        let e = self.psc(h.z * x)
                        px = e.y * x + e.x * y
                        py = e.y * y - e.x * x
                    } else if h.y < 2.5 {
                        let k = 1.0 / max(1.0 + h.z * y, 0.001)
                        px = x * k
                        pz = z * k
                    } else if h.w < 0.5 {
                        px = abs(x)
                    } else if h.w < 1.5 {
                        py = abs(y)
                    } else {
                        pz = abs(z)
                    }
                    pc = pc + 1.0
                } else if code < 4.5 {
                    pn = pn - 1i
                    px = sx8[pn]
                    py = sy8[pn]
                    pz = sz8[pn]
                    vs[vn - 1i] = vs[vn - 1i] / h.y
                    pc = pc + 1.0
                } else if code < 5.5 {
                    let e = vs[vn - 1i]
                    let se = ss[vn - 1i]
                    let d = vs[vn - 2i]
                    let sd = ss[vn - 2i]
                    vn = vn - 1i
                    var v = 0.0
                    var s = sd
                    if h.y < 0.5 {
                        v = self.smin(d, e, h.z)
                        if e < d { s = se }
                    } else if h.y < 1.5 {
                        v = 0.0 - self.smin(0.0 - d, e, h.z)
                    } else {
                        v = 0.0 - self.smin(0.0 - d, 0.0 - e, h.z)
                        if e > d { s = se }
                    }
                    vs[vn - 1i] = v
                    ss[vn - 1i] = s
                    pc = pc + 1.0
                } else if code < 6.5 {
                    vs[vn - 1i] = abs(vs[vn - 1i]) - h.y
                    pc = pc + 1.0
                } else if code < 7.5 {
                    vs[vn - 1i] = vs[vn - 1i] - h.y
                    pc = pc + 1.0
                } else if code < 8.5 {
                    ms[mn] = h.y
                    mn = mn + 1i
                    pc = pc + 1.0
                } else if code < 9.5 {
                    mn = mn - 1i
                    pc = pc + 1.0
                } else if code < 10.5 {
                    // MAT_WITHIN: the region's distance is on top
                    vn = vn - 1i
                    if vs[vn] > 0.0 {
                        if mn > 0i { ms[mn] = ms[mn - 1i] } else { ms[mn] = 0.0 }
                    } else {
                        ms[mn] = h.y
                    }
                    mn = mn + 1i
                    pc = pc + 1.0
                } else if code < 11.5 {
                    // KERNEL: slot h.y, h.z parameter texels
                    vs[vn] = self.kernel_at(base + pc, h.y, px, py, pz)
                    if mn > 0i { ss[vn] = ms[mn - 1i] } else { ss[vn] = 0.0 }
                    vn = vn + 1i
                    pc = pc + 1.0 + h.z
                } else if code < 12.5 {
                    // SCALE
                    vs[vn - 1i] = vs[vn - 1i] * h.y
                    pc = pc + 1.0
                } else if code < 14.5 {
                    phase = 2u
                    self.continuations[task + 144u] = 0u; self.continuations[task + 145u] = 1u
                    self.continuations[task + 146u] = 2139095040u; self.continuations[task + 147u] = 0u
                    self.continuations[task + 160u] = 0u
                } else if code < 15.5 {
                    phase = 3u
                    self.continuations[task + 192u] = 0u; self.continuations[task + 193u] = 0u
                    self.continuations[task + 194u] = 2139095040u; self.continuations[task + 195u] = 0u
                } else if code < 16.5 {
                    // TEX: the pattern's values on top (3 for triplanar, else 1), the source's
                    // value below them and, with the gradient, its four corners below that.
                    // Known masks: the mask is exactly the 1 or 0 the ramps give there, and no
                    // corners are on the stack (GRAD skipped them) nor, masked off, a pattern
                    // (PAT skipped it).
                    let a = self.fetch(base + pc + 1.0)
                    let m = self.fetch(base + pc + 2.0)
                    var np = 1i
                    if h.y > 0.5 { np = 3i }
                    // the source's value, and where the result goes
                    var si = vn - np - 1i
                    var at = si
                    let o = u32(a.z)
                    var mm = 0.0
                    var t = 0.0
                    if o < 16u && ((masks >> o) & 1u) != 0u {
                        mm = f32((masks >> (o + 16u)) & 1u)
                        if mm > 0.5 { t = vs[vn - 1i] } else { si = vn - 1i }
                        at = si
                    } else {
                        var gx = 0.0
                        var gy = 0.0
                        var gz = 0.0
                        if h.z > 0.5 {
                            let g0 = vs[si - 4i]
                            let g1 = vs[si - 3i]
                            let g2 = vs[si - 2i]
                            let g3 = vs[si - 1i]
                            gx = (((g0 - g1) - g2) + g3) * a.y
                            gy = (((0.0 - g0) - g1) + g2 + g3) * a.y
                            gz = (((0.0 - g0) + g1) - g2 + g3) * a.y
                            at = si - 4i
                        }
                        let mt0 = self.clamp01((m.y - gy) / (m.y - m.x))
                        let mb0 = self.clamp01((m.w + gy) / (m.w - m.z))
                        let mt = (mt0 * mt0) * (3.0 - 2.0 * mt0)
                        let mb = (mb0 * mb0) * (3.0 - 2.0 * mb0)
                        mm = mt * mb
                        if cell > 0.0 && o < 16u && np == 1i && h.z > 0.5 {
                            // The gradient's Y changes at most 4 a.y per unit (a 1-Lipschitz
                            // source; a quarter more for estimates): within half a cell's
                            // diagonal of here it stays this far from the ramps' ends.
                            let reach = (4.33 * cell) * a.y + 0.0001
                            let one = gy >= reach - m.z && gy <= m.x - reach
                            if one || gy >= m.y + reach || gy <= (0.0 - m.w) - reach {
                                known = known | (1u << o)
                                if one { ones = ones | (1u << o) }
                            }
                        }
                        t = vs[vn - 1i]
                        if np == 3i {
                            let n = max((gx * gx + gy * gy) + gz * gz, 1.0)
                            t = (((gx * gx) / n) * vs[vn - 3i] + ((gy * gy) / n) * vs[vn - 2i]) + ((gz * gz) / n) * vs[vn - 1i]
                        }
                    }
                    let sd = ss[si]
                    vs[at] = (vs[si] + h.w * (mm * t)) * a.x
                    ss[at] = sd
                    vn = at + 1i
                    pc = pc + 3.0
                } else {
                    // GRAD (corners) or PAT (pattern, where the masks are 0): h.y texels follow
                    // that a texture with known masks skips (its TEX knows they are not there)
                    let o = u32(h.z)
                    var skip = false
                    if o < 16u && ((masks >> o) & 1u) != 0u {
                        skip = code < 17.5 || ((masks >> (o + 16u)) & 1u) == 0u
                    }
                    pc = pc + 1.0
                    if skip { pc = pc + h.y }
                }
            }
            self.continuations[task + 63u] = budget
            if pc >= len { self.continuations[task + 64u] = 0u; return vec2(1.0, vs[0u]) }
            self.continuations[task + 64u] = phase
            self.continuations[task + 65u] = asuint(pc)
            self.continuations[task + 66u] = asuint(px); self.continuations[task + 67u] = asuint(py); self.continuations[task + 68u] = asuint(pz)
            self.continuations[task + 69u] = u32(vn); self.continuations[task + 70u] = u32(pn); self.continuations[task + 71u] = u32(mn)
            for i in 0..16u { self.continuations[task + 80u + i] = asuint(vs[i]); self.continuations[task + 96u + i] = asuint(ss[i]) }
            for i in 0..8u {
                self.continuations[task + 112u + i] = asuint(sx8[i]); self.continuations[task + 120u + i] = asuint(sy8[i])
                self.continuations[task + 128u + i] = asuint(sz8[i]); self.continuations[task + 136u + i] = asuint(ms[i])
            }
            return vec2(0.0)
        }

        node: fn(i: u32, j: u32) -> vec2 {
            // Refine in grid-local millimetres. Large printer coordinates must
            // not round a near-face sample onto the face before evaluation.
            return vec2(f32(i) * 0.1, f32(j) * 0.1)
        }
        span: fn(ea: vec4u, eb: vec4u, a: vec2, b: vec2, entering: bool, middle: vec2, has_middle: u32) {
            let index = atomic_add(self.counts[2u], 1u)
            if index >= self.capacity { atomic_add(self.counts[3u], 1u); return }
            let at = index * 4u
            let pa = a + self.inputs[3u].xy
            let pb = b + self.inputs[3u].xy
            let pm = middle + self.inputs[3u].xy
            // The boundary signs, not a possibly degenerate gradient, fix
            // direction. Every shared edge then has one incoming span.
            if entering {
                self.segments[at] = ea
                self.segments[at + 1u] = eb
                self.segments[at + 2u] = vec4u(asuint(pa.x), asuint(pa.y), asuint(pb.x), asuint(pb.y))
            } else {
                self.segments[at] = eb
                self.segments[at + 1u] = ea
                self.segments[at + 2u] = vec4u(asuint(pb.x), asuint(pb.y), asuint(pa.x), asuint(pa.y))
            }
            self.segments[at + 3u] = vec4u(asuint(pm.x), asuint(pm.y), has_middle, 0u)
        }
    }
    mod.sdf.TraceCoarse = {
        ..mod.sdf.Trace
        main: fn() {
            if self.control() { return }
            let index = self.index()
            if index >= self.grid.x * self.grid.y * self.grid.z { return }
            if !self.begin() { return }
            let task = self.task()
            let x = index % (self.grid.x * self.grid.y)
            let layer = index / (self.grid.x * self.grid.y)
            let i = x % self.grid.x
            let j = x / self.grid.x
            let z = self.inputs[4u + layer].x
            var values = array(0f, 0f, 0f, 0f)
            var corner = 0u
            if self.continuations[task + 61u] != 0u {
                corner = self.continuations[task + 1u]
                values[0u] = asfloat(self.continuations[task + 2u])
                values[1u] = asfloat(self.continuations[task + 3u])
                values[2u] = asfloat(self.continuations[task + 4u])
                values[3u] = asfloat(self.continuations[task + 5u])
            }
            for sample in 0..4u {
                if corner == 4u { break }
                let value = self.field(self.node((i + corner % 2u) * 10u, (j + corner / 2u) * 10u), z)
                if value.x == 0.0 {
                    self.continuations[task + 1u] = corner
                    self.continuations[task + 2u] = asuint(values[0u])
                    self.continuations[task + 3u] = asuint(values[1u])
                    self.continuations[task + 4u] = asuint(values[2u])
                    self.continuations[task + 5u] = asuint(values[3u])
                    self.pending(); return
                }
                values[corner] = value.y
                corner += 1u
            }
            let a = values[0u]
            let b = values[1u]
            let c = values[2u]
            let d = values[3u]
            let low = min(min(a, b), min(c, d))
            let high = max(max(a, b), max(c, d))
            let near = min(min(abs(a), abs(b)), min(abs(c), abs(d)))
            if (low < 0.0 && high >= 0.0) || near <= self.inputs[3u].z * 1.4142135623730951 {
                let at = atomic_add(self.counts[0u], 1u)
                atomic_add(self.counts[32u + layer * 3u], 1u)
                if at < self.grid.w { self.kept[at] = vec4u(i, j, layer, 0u) }
                else { atomic_add(self.counts[1u], 1u) }
            }
            self.done()
        }
    }
    mod.sdf.TraceCell = {
        ..mod.sdf.Trace
        // One flat step per dispatch, chosen by `mode`, with no workgroup memory (31 KB of
        // it held an Apple GPU core to one workgroup) and one field call site per step:
        //   1 a thread per kept cell node
        //   2 a thread per kept cell: its sign-changing edges become crossings
        //   4 a thread per crossing: bracket refinement, then the gradient
        //   5 a thread per fine cell: its spans (a saddle takes one centre sample)
        //   7 a thread per span: crease vertex, curve projection, then the segment
        // After its `grid.w` cells the kept buffer holds two words per cell: its first
        // crossing and the mask of its sign-changing edges. Crossings and spans take two
        // words each: identity, then position (and normal, or middle).
        mode: shader.uniform(u32)
        cross: shader.buffer_rw(vec4u)
        spans: shader.buffer_rw(vec4u)
        edge_nodes: fn(edge: u32) -> vec4u {
            if edge < 110u { return vec4u(edge % 10u, edge / 10u, 0u, 1u) }
            return vec4u((edge - 110u) % 11u, (edge - 110u) / 11u, 1u, 11u)
        }
        grown: fn(q: vec2, origin: vec2) -> bool {
            return q.x >= origin.x - 0.05 && q.x <= origin.x + 0.15 && q.y >= origin.y - 0.05 && q.y <= origin.y + 0.15
        }
        delta: fn(sample: u32) -> vec2 {
            if sample == 0u { return vec2(0.001, 0.0) }
            if sample == 1u { return vec2(0.0 - 0.001, 0.0) }
            if sample == 2u { return vec2(0.0, 0.001) }
            if sample == 3u { return vec2(0.0, 0.0 - 0.001) }
            return vec2(0.0)
        }
        seam_delta: fn(j: u32) -> vec2 {
            if j == 4u { return vec2(0.001, 0.001) }
            if j == 5u { return vec2(0.001, 0.0 - 0.001) }
            if j == 6u { return vec2(0.0 - 0.001, 0.001) }
            if j == 7u { return vec2(0.0 - 0.001, 0.0 - 0.001) }
            return self.delta(j)
        }
        fraction: fn(bracket: vec4, round: u32) -> f32 {
            let lo = bracket.x
            let hi = bracket.y
            if round % 2u == 0u {
                return clamp(lo + (hi - lo) * bracket.z / (bracket.z - bracket.w), lo + (hi - lo) * 0.4, hi - (hi - lo) * 0.4)
            }
            return (lo + hi) * 0.5
        }
        bits: fn(x: u32) -> u32 {
            let a = x - ((x >> 1u) & 1431655765u)
            let b = (a & 858993459u) + ((a >> 2u) & 858993459u)
            let c = (b + (b >> 4u)) & 252645135u
            return (c * 16843009u) >> 24u
        }
        meta: fn(c: u32, w: u32) -> u32 {
            let v = self.kept[self.grid.w + c * 2u + w / 4u]
            let k = w % 4u
            if k == 0u { return v.x }
            if k == 1u { return v.y }
            if k == 2u { return v.z }
            return v.w
        }
        // The crossing of kept cell `c`'s sign-changing `edge`: its first crossing plus the
        // set mask bits below the edge.
        crossing: fn(c: u32, edge: u32) -> u32 {
            let word = edge / 32u
            var below = self.meta(c, 0u)
            for w in 0..word { below += self.bits(self.meta(c, 1u + w)) }
            return below + self.bits(self.meta(c, 1u + word) & ((1u << (edge % 32u)) - 1u))
        }
        ends: fn(bi: u32, bj: u32, edge: u32) -> vec4 {
            let e = self.edge_nodes(edge)
            let a = self.node(bi + e.x, bj + e.y)
            var b = self.node(bi + e.x + 1u, bj + e.y)
            if e.z == 1u { b = self.node(bi + e.x, bj + e.y + 1u) }
            return vec4(a.x, a.y, b.x, b.y)
        }
        point: fn(c: vec4u) -> vec4 {
            return vec4(asfloat(c.x), asfloat(c.y), asfloat(c.z), asfloat(c.w))
        }
        main: fn() {
            if self.control() { return }
            let t = self.index()
            if t == 4294967295u { return }
            if !self.begin() { return }
            let task = self.task()
            let kept = min(self.counts[0u], self.grid.w)
            let mode = self.mode
            // A count/allocation invariant failure stops evaluation and fails the request.
            if self.counts[1u] != 0u || self.counts[3u] != 0u { self.done(); return }
            if mode == 1u {
                if t >= kept * 121u { return }
                let c = t / 121u
                let n = t % 121u
                let cell = self.kept[c]
                let z = self.inputs[4u + cell.z].x
                let q = self.node(cell.x * 10u + n % 11u, cell.y * 10u + n / 11u)
                // An exact zero takes eight more samples: one with nothing positive around
                // it is an internal material seam, not a positive/negative border. Every
                // cell holding the node samples it the same way.
                var value = 0.0
                var lo = 0.0
                var hi = 0.0
                var j = 0u
                if self.continuations[task + 61u] != 0u {
                    value = asfloat(self.continuations[task + 1u])
                    lo = asfloat(self.continuations[task + 2u])
                    hi = asfloat(self.continuations[task + 3u])
                    j = self.continuations[task + 4u]
                }
                for sample in 0..9u {
                    if j == 9u { break }
                    var s = q
                    if j > 0u { s = q + self.seam_delta(j - 1u) }
                    let sampled = self.field(s, z)
                    if sampled.x == 0.0 {
                        self.continuations[task + 1u] = asuint(value)
                        self.continuations[task + 2u] = asuint(lo)
                        self.continuations[task + 3u] = asuint(hi)
                        self.continuations[task + 4u] = j
                        self.pending(); return
                    }
                    let f = sampled.y
                    if j == 0u { value = f }
                    else { lo = min(lo, f); hi = max(hi, f) }
                    if value != 0.0 { break }
                    j += 1u
                }
                if value == 0.0 {
                    atomic_add(self.counts[32u + cell.z * 3u + 2u], 8u)
                    if lo < 0.0 && hi <= 0.0 { value = 0.0 - asfloat(8388608u) }
                }
                self.values[t] = asuint(value)
                self.done()
                return
            }
            if mode == 2u {
                if t >= kept { return }
                let cell = self.kept[t]
                let v = t * 121u
                var masks = array(0u, 0u, 0u, 0u, 0u, 0u, 0u)
                var count = 0u
                for edge in 0..220u {
                    let e = self.edge_nodes(edge)
                    let at = v + e.y * 11u + e.x
                    if (asfloat(self.values[at]) < 0.0) != (asfloat(self.values[at + e.w]) < 0.0) {
                        masks[edge / 32u] = masks[edge / 32u] | (1u << (edge % 32u))
                        count += 1u
                    }
                }
                let base = atomic_add(self.counts[4u], count)
                atomic_add(self.counts[32u + cell.z * 3u + 1u], count)
                atomic_add(self.counts[32u + cell.z * 3u + 2u], 121u)
                self.kept[self.grid.w + t * 2u] = vec4u(base, masks[0u], masks[1u], masks[2u])
                self.kept[self.grid.w + t * 2u + 1u] = vec4u(masks[3u], masks[4u], masks[5u], masks[6u])
                self.done()
                return
            }
            if mode == 3u {
                if t >= kept { self.done(); return }
                let a = self.kept[self.grid.w + t * 2u]
                let b = self.kept[self.grid.w + t * 2u + 1u]
                let masks = array(a.y, a.z, a.w, b.x, b.y, b.z, b.w)
                var k = a.x
                for edge in 0..220u {
                    if (masks[edge / 32u] & (1u << (edge % 32u))) != 0u {
                        self.cross[k * 2u] = vec4u(t, edge, 0u, 0u)
                        k += 1u
                    }
                }
                self.done()
                return
            }
            if mode == 4u {
                if t >= min(self.counts[4u], self.capacity) { return }
                let id = self.cross[t * 2u]
                let c = id.x
                let edge = id.y
                let cell = self.kept[c]
                let z = self.inputs[4u + cell.z].x
                let ab = self.ends(cell.x * 10u, cell.y * 10u, edge)
                let a = ab.xy
                let b = ab.zw
                let e = self.edge_nodes(edge)
                let at = c * 121u + e.y * 11u + e.x
                let fa = asfloat(self.values[at])
                let fb = asfloat(self.values[at + e.w])
                var q = a + (b - a) * 0.5
                var done = false
                if fa == 0.0 { q = a; done = true }
                else if fb == 0.0 { q = b; done = true }
                var bracket = vec4(0.0, 1.0, fa, fb)
                var g = vec4(0.0)
                var round = 0u
                var evaluations = 0u
                // Twelve bracket rounds (regula falsi and bisection alternating) until the
                // bracket is under a micron, then four gradient samples: one call site.
                var iteration = 0u
                if self.continuations[task + 61u] != 0u {
                    q = vec2(asfloat(self.continuations[task + 1u]), asfloat(self.continuations[task + 2u]))
                    done = self.continuations[task + 3u] != 0u
                    bracket = vec4(asfloat(self.continuations[task + 4u]), asfloat(self.continuations[task + 5u]), asfloat(self.continuations[task + 6u]), asfloat(self.continuations[task + 7u]))
                    g = vec4(asfloat(self.continuations[task + 8u]), asfloat(self.continuations[task + 9u]), asfloat(self.continuations[task + 10u]), asfloat(self.continuations[task + 11u]))
                    round = self.continuations[task + 12u]
                    evaluations = self.continuations[task + 13u]
                    iteration = self.continuations[task + 14u]
                }
                for step in 0..16u {
                    if iteration == 16u { break }
                    if round < 12u && (done || (bracket.y - bracket.x) * 0.1 <= 0.0001) { done = true; round = 12u }
                    var s = q
                    var ft = 0.0
                    if round < 12u {
                        ft = self.fraction(bracket, round)
                        s = a + (b - a) * ft
                    } else { s = q + self.delta(round - 12u) }
                    let sampled = self.field(s, z)
                    if sampled.x == 0.0 {
                        self.continuations[task + 1u] = asuint(q.x); self.continuations[task + 2u] = asuint(q.y)
                        self.continuations[task + 3u] = 0u; if done { self.continuations[task + 3u] = 1u }
                        self.continuations[task + 4u] = asuint(bracket.x); self.continuations[task + 5u] = asuint(bracket.y); self.continuations[task + 6u] = asuint(bracket.z); self.continuations[task + 7u] = asuint(bracket.w)
                        self.continuations[task + 8u] = asuint(g.x); self.continuations[task + 9u] = asuint(g.y); self.continuations[task + 10u] = asuint(g.z); self.continuations[task + 11u] = asuint(g.w)
                        self.continuations[task + 12u] = round
                        self.continuations[task + 13u] = evaluations
                        self.continuations[task + 14u] = iteration
                        self.pending(); return
                    }
                    let f = sampled.y
                    iteration += 1u
                    evaluations += 1u
                    if round < 12u {
                        if f == 0.0 { q = s; done = true }
                        else {
                            if (f < 0.0) == (bracket.z < 0.0) { bracket = vec4(ft, bracket.y, f, bracket.w) }
                            else { bracket = vec4(bracket.x, ft, bracket.z, f) }
                            q = a + (b - a) * ((bracket.x + bracket.y) * 0.5)
                        }
                    }
                    else if round == 12u { g.x = f }
                    else if round == 13u { g.y = f }
                    else if round == 14u { g.z = f }
                    else { g.w = f }
                    round += 1u
                    if round == 16u { break }
                }
                let gradient = vec2(g.x - g.y, g.z - g.w) * 500.0
                let normal = gradient / max(length(gradient), 0.000000000001)
                self.cross[t * 2u + 1u] = vec4u(asuint(q.x), asuint(q.y), asuint(normal.x), asuint(normal.y))
                atomic_add(self.counts[32u + cell.z * 3u + 2u], evaluations)
                self.done()
                return
            }
            if mode == 5u {
                if t >= kept * 100u { return }
                let c = t / 100u
                let n = t % 100u
                let cell = self.kept[c]
                let at = (n / 10u) * 11u + n % 10u
                let v = c * 121u + at
                let a = asfloat(self.values[v])
                let b = asfloat(self.values[v + 1u])
                let cc = asfloat(self.values[v + 12u])
                let d = asfloat(self.values[v + 11u])
                var ids = array(0u, 0u, 0u, 0u)
                var enters = array(false, false, false, false)
                var count = 0u
                for edge in 0..4u {
                    var sample = (n / 10u) * 10u + n % 10u
                    var fa = a
                    var fb = b
                    if edge == 1u { fa = b; fb = cc; sample = 110u + at + 1u }
                    if edge == 2u { fa = d; fb = cc; sample = sample + 10u }
                    if edge == 3u { fa = a; fb = d; sample = 110u + at }
                    if (fa < 0.0) != (fb < 0.0) {
                        ids[count] = sample
                        enters[count] = fa < 0.0
                        if edge >= 2u { enters[count] = fb < 0.0 }
                        count += 1u
                    }
                }
                if count == 0u { self.done(); return }
                var alternate = false
                if count == 4u {
                    let z = self.inputs[4u + cell.z].x
                    let sampled = self.field(self.node(cell.x * 10u + n % 10u, cell.y * 10u + n / 10u) + vec2(0.05), z)
                    if sampled.x == 0.0 { self.pending(); return }
                    let f = sampled.y
                    atomic_add(self.counts[32u + cell.z * 3u + 2u], 1u)
                    alternate = (f < 0.0) != (a < 0.0)
                }
                for pair in 0..2u {
                    if (pair == 0u && count >= 2u) || (pair == 1u && count == 4u) {
                        var ia = pair * 2u
                        var ib = ia + 1u
                        if alternate { ia = (ia + 3u) % 4u; ib = (ib + 3u) % 4u }
                        let xa = self.crossing(c, ids[ia])
                        let xb = self.crossing(c, ids[ib])
                        let s = atomic_add(self.counts[6u], 1u)
                        if s < self.capacity && xa < self.capacity && xb < self.capacity {
                            let ca = self.point(self.cross[xa * 2u + 1u])
                            let cb = self.point(self.cross[xb * 2u + 1u])
                            var entering = 0u
                            if enters[ia] { entering = 1u }
                            let middle = (ca.xy + cb.xy) * 0.5
                            self.spans[s * 2u] = vec4u(xa, xb, c * 128u + n, entering)
                            self.spans[s * 2u + 1u] = vec4u(asuint(middle.x), asuint(middle.y), asuint(dot(ca.zw, cb.zw)), asuint(0.0))
                        } else { atomic_add(self.counts[3u], 1u) }
                    }
                }
                self.done()
                return
            }
            if mode == 7u {
                if t >= min(self.counts[6u], self.capacity) { return }
                let span = self.spans[t * 2u]
                var mid = self.point(self.spans[t * 2u + 1u])
                let c = span.z / 128u
                let n = span.z % 128u
                let cell = self.kept[c]
                let bi = cell.x * 10u
                let bj = cell.y * 10u
                let z = self.inputs[4u + cell.z].x
                let origin = self.node(bi + n % 10u, bj + n / 10u)
                let ca = self.point(self.cross[span.x * 2u + 1u])
                let cb = self.point(self.cross[span.y * 2u + 1u])
                // A sharp turn's tangent intersection is a crease candidate (3), else the
                // crease is rejected (4) and the span stays a curve.
                var sides = vec4(0.0)
                if mid.z < 0.9063078 {
                    let det = ca.z * cb.w - ca.w * cb.z
                    mid.w = 4.0
                    if abs(det) > 0.000000001 {
                        let k = dot(cb.zw, cb.xy - ca.xy)
                        let x = ca.xy + vec2(0.0 - ca.w, ca.z) * (k / det)
                        if self.grown(x, origin) {
                            mid = vec4(x.x, x.y, mid.z, 3.0)
                            sides = vec4(ca.x, ca.y, cb.x, cb.y)
                        }
                    }
                }
                // One evaluator call site: each pass takes one sample of the current stage;
                // a stage's consumer runs after its last sample. A crease candidate runs up
                // to four cycles of seven stages (sides projected, their normals, the new
                // intersection, its value); then a curved span takes up to four Newton
                // projections of its middle.
                var part = 0u
                if mid.w != 3.0 { part = 1u }
                var stage = 0u
                var cycle = 0u
                var curve = 0u
                var j = 0u
                var r = array(0f, 0f, 0f, 0f, 0f)
                var na = vec2(0.0)
                var evaluations = 0u
                var iteration = 0u
                if self.continuations[task + 61u] != 0u {
                    mid = vec4(asfloat(self.continuations[task + 1u]), asfloat(self.continuations[task + 2u]), asfloat(self.continuations[task + 3u]), asfloat(self.continuations[task + 4u]))
                    sides = vec4(asfloat(self.continuations[task + 5u]), asfloat(self.continuations[task + 6u]), asfloat(self.continuations[task + 7u]), asfloat(self.continuations[task + 8u]))
                    part = self.continuations[task + 9u]
                    stage = self.continuations[task + 10u]
                    cycle = self.continuations[task + 11u]
                    curve = self.continuations[task + 12u]
                    j = self.continuations[task + 13u]
                    na = vec2(asfloat(self.continuations[task + 14u]), asfloat(self.continuations[task + 15u]))
                    evaluations = self.continuations[task + 16u]
                    iteration = self.continuations[task + 17u]
                    r[0u] = asfloat(self.continuations[task + 18u])
                    r[1u] = asfloat(self.continuations[task + 19u])
                    r[2u] = asfloat(self.continuations[task + 20u])
                    r[3u] = asfloat(self.continuations[task + 21u])
                    r[4u] = asfloat(self.continuations[task + 22u])
                }
                for step in 0..140u {
                    if iteration == 140u { break }
                    if part == 1u && curve == 0u && j == 0u && mid.w >= 3.0 {
                        // A crease that did not resolve keeps its span, as a curve.
                        let m = (ca.xy + cb.xy) * 0.5
                        mid = vec4(m.x, m.y, 0.0, 0.0)
                    }
                    if part == 1u && !(mid.w == 0.0 && mid.z < 0.9975641) { part = 2u }
                    if part == 2u { break }
                    var q = mid.xy
                    var number = 5u
                    if part == 0u {
                        if stage == 0u && j == 0u && self.continuations[task + 64u] == 0u { sides = vec4((sides.xy + mid.xy) * 0.5, (sides.zw + mid.xy) * 0.5) }
                        q = sides.xy
                        if stage >= 3u { q = sides.zw }
                        if stage == 2u || stage == 5u { number = 4u }
                        if stage == 6u { q = mid.xy; number = 1u }
                    }
                    var s = q
                    if number != 1u && j < 4u { s = q + self.delta(j) }
                    let sampled = self.field(s, z)
                    if sampled.x == 0.0 {
                        self.continuations[task + 1u] = asuint(mid.x); self.continuations[task + 2u] = asuint(mid.y); self.continuations[task + 3u] = asuint(mid.z); self.continuations[task + 4u] = asuint(mid.w)
                        self.continuations[task + 5u] = asuint(sides.x); self.continuations[task + 6u] = asuint(sides.y); self.continuations[task + 7u] = asuint(sides.z); self.continuations[task + 8u] = asuint(sides.w)
                        self.continuations[task + 9u] = part
                        self.continuations[task + 10u] = stage
                        self.continuations[task + 11u] = cycle
                        self.continuations[task + 12u] = curve
                        self.continuations[task + 13u] = j
                        self.continuations[task + 14u] = asuint(na.x); self.continuations[task + 15u] = asuint(na.y)
                        self.continuations[task + 16u] = evaluations
                        self.continuations[task + 17u] = iteration
                        self.continuations[task + 18u] = asuint(r[0u])
                        self.continuations[task + 19u] = asuint(r[1u])
                        self.continuations[task + 20u] = asuint(r[2u])
                        self.continuations[task + 21u] = asuint(r[3u])
                        self.continuations[task + 22u] = asuint(r[4u])
                        self.pending(); return
                    }
                    let f = sampled.y
                    iteration += 1u
                    evaluations += 1u
                    r[j] = f
                    j += 1u
                    if j == number {
                        j = 0u
                        if part == 0u {
                            if stage == 6u {
                                if abs(r[0u]) <= 0.0001 * self.inputs[3u].z && mid.z < 0.001 { mid.w = 2.0 }
                                else if cycle == 3u { mid.w = 4.0 }
                            } else {
                                let g = vec2(r[0u] - r[1u], r[2u] - r[3u]) * 500.0
                                if stage == 2u { na = g / max(length(g), 0.000000000001) }
                                else if stage == 5u {
                                    let nb = g / max(length(g), 0.000000000001)
                                    let det = na.x * nb.y - na.y * nb.x
                                    mid.w = 4.0
                                    // A smooth turn ceases to have a crease as samples
                                    // approach. Keep the span without a crease vertex.
                                    if abs(det) > 0.000000001 && dot(na, nb) < 0.9063078 {
                                        let k = dot(nb, sides.zw - sides.xy)
                                        let p = sides.xy + vec2(0.0 - na.y, na.x) * (k / det)
                                        if self.grown(p, origin) { mid = vec4(p.x, p.y, length(p - mid.xy), 3.0) }
                                    }
                                } else {
                                    var p = sides.xy
                                    if stage >= 3u { p = sides.zw }
                                    p = p - g * (r[4u] / max(dot(g, g), 0.000000000001))
                                    if stage < 3u { sides = vec4(p.x, p.y, sides.z, sides.w) }
                                    else { sides = vec4(sides.x, sides.y, p.x, p.y) }
                                }
                            }
                            stage += 1u
                            if stage == 7u { stage = 0u; cycle += 1u }
                            if mid.w != 3.0 { part = 1u }
                        } else {
                            let g = vec2(r[0u] - r[1u], r[2u] - r[3u]) * 500.0
                            let value = r[4u]
                            let g2 = dot(g, g)
                            let settled = abs(value) < 0.0000001 || g2 < 0.000000000000000001
                            var p = mid.xy
                            if !settled { p = p - g * (value / g2) }
                            mid.x = p.x
                            mid.y = p.y
                            if settled || curve == 3u {
                                if p.x >= origin.x - 0.1 && p.x <= origin.x + 0.2 && p.y >= origin.y - 0.1 && p.y <= origin.y + 0.2 { mid.w = 1.0 }
                                else { mid.z = 1.0 }
                            }
                            curve += 1u
                        }
                    }
                }
                atomic_add(self.counts[32u + cell.z * 3u + 2u], evaluations)
                // A rejected crease rejects only its vertex: the two sign-changing edges
                // still own a span.
                var has_middle = 0u
                if mid.w <= 2.0 { has_middle = u32(mid.w) }
                let ea = self.edge_nodes(self.cross[span.x * 2u].y)
                let eb = self.edge_nodes(self.cross[span.y * 2u].y)
                self.span(vec4u(bi + ea.x, bj + ea.y, ea.z, cell.z), vec4u(bi + eb.x, bj + eb.y, eb.z, cell.z),
                    ca.xy, cb.xy, span.w != 0u, mid.xy, has_middle)
                self.done()
            }
        }
    }
    mod.sdf
}

use crate::{gpu_jobs::{BufRef, CountWord, Counts, DROPS_IN_A_ROW, Elements, Finished, ItemId, ItemStats, ItemTag, JobClass, JobError, Jobs, LONG_MS, Output, OutputBytes, Pass, Pool, Refusal, Step, WorkItem}, kernel::{ComputeBake, KernelSlot}, program::Program};
pub(crate) use crate::gpu_jobs::Allocation;
use makepad_draw::makepad_platform::thread::{Class, TaskPool, Ticket};
use std::{collections::{BTreeMap, HashMap, HashSet, VecDeque}, rc::Rc, sync::{Arc, Weak, atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering}}, time::Instant};

/// Host-independent request, prepared on the app pool, in focus order. `sink` runs on that pool
/// (pipeline handoff v1): once per batch of layers as the GPU left it, spawned where its
/// readbacks land (never through the UI thread) with this request's ticket, or once with the
/// error that ends the request.
#[derive(Clone)]
pub struct TraceRequest {
    pub requested: Instant,
    pub part: u64,
    /// The shape's content key: requests of one part and revision share uploads and a lead.
    pub revision: u64,
    /// The engine revision whose run asked (reports only: [`ItemTag::revision`]).
    pub input: u64,
    pub program: Arc<Program>,
    pub kernels: Arc<[KernelSlot]>,
    pub to_shape: [[f32; 4]; 3],
    pub bounds: ([f64; 2], [f64; 2]),
    pub lipschitz: f32,
    pub layers: Vec<(usize, f64)>,
    pub class: Class,
    pub ticket: Ticket,
    pub sink: BatchSink,
}

/// A batch of a request's layers as the GPU left it (pipeline handoff v1): owned, Send, and
/// decoded on any thread ([`decode`]).
pub struct GpuBatch {
    pub part: u64,
    pub revision: u64,
    /// Its ordinal among its request's batches.
    pub batch: u32,
    /// (index, z) of its planes, in plane order: the request's layers it traced.
    pub layers: Arc<[(usize, f64)]>,
    pub counts: Counts,
    pub output: OutputBytes,
    /// Through the submissions before its last.
    pub stats: ItemStats,
    /// When its final readbacks were checked.
    pub landed: Instant,
}
pub type BatchSink = Arc<dyn Fn(Result<GpuBatch, JobError>) + Send + Sync>;

/// A batch's layers: its count invariants, then its segments chained into loops, a layer per
/// plane. Pure, on any thread.
pub fn decode(batch: &GpuBatch) -> Result<Vec<TracedLayer>, String> {
    let c = &batch.counts;
    if c.at(1) != 0 || c.at(3) != 0 {
        return Err(format!("GPU trace count/allocation invariant failed: kept={} segments={} trace_overflow={}",
            c.at(0), c.at(2).max(c.at(4)).max(c.at(6)), c.at(1) + c.at(3)));
    }
    chain_layers(&batch.output, &batch.layers)
}

/// A sink handing each batch on decoded ([`decode`]), on the batch's pool thread.
pub fn decoded(f: impl Fn(Result<Vec<TracedLayer>, JobError>) + Send + Sync + 'static) -> BatchSink {
    Arc::new(move |batch: Result<GpuBatch, JobError>| f(batch.and_then(|b| decode(&b).map_err(JobError::Failed))))
}

#[derive(Clone, Debug)]
pub struct TracedLayer {
    pub index: usize,
    pub z: f64,
    /// Closed, directed printer-mm loops, canonical by their smallest edge id.
    pub loops: Vec<Vec<[f32; 2]>>,
    /// Accepted GPU tangent intersections, separate from projected curve points.
    pub creases: Vec<[f32; 2]>,
}

#[derive(Default)]
struct TracePipelines {
    shaders: HashMap<Vec<u64>, (ComputeKernel, ComputeKernel)>,
    pending: Vec<(Vec<u64>, Instant)>,
    warm: Option<Allocation>,
}

/// Standalone trace users lower through the same splice as the bake.
pub fn lower(vm: &mut ScriptVm, kernels: &[KernelSlot]) -> Result<(), String> {
    let mut splice = ComputeBake::new(vm, kernels).ok_or("trace kernel splice")?;
    let result = lower_spliced(vm, &mut splice);
    splice.release(vm);
    result
}

/// Called while the bake lowers its kernel set with the legitimately borrowed
/// VM and the same mesh/AIR dispatch. Warming only receives compiled objects.
pub(crate) fn lower_spliced(vm: &mut ScriptVm, splice: &mut ComputeBake) -> Result<(), String> {
    let keys = splice.keys.clone();
    if vm.host.cx_mut().global::<TracePipelines>().shaders.contains_key(&keys) { return Ok(()); }
    let started = Instant::now();
    let result = (|| {
        let module = vm.heap().module(id!(sdf));
        let mut compile = |id, name| {
            let started = Instant::now();
            let base = vm.value(module, id).as_object().ok_or("trace shader missing")?;
            let object = splice.draw_variant(vm, base).ok_or("trace shader variant")?;
            let shader = ComputeKernel::new(vm, object, name)?;
            log!("sdf trace lower: {} {:.3} ms", name, started.elapsed().as_secs_f64() * 1000.0);
            Ok::<_, String>(shader)
        };
        Ok::<_, String>((compile(id!(TraceCoarse), "trace_coarse")?, compile(id!(TraceCell), "trace_cell")?))
    })();
    let shaders = result?;
    let cache = vm.host.cx_mut().global::<TracePipelines>();
    cache.shaders.insert(keys.clone(), shaders);
    cache.pending.push((keys, started));
    SignalToUI::set_ui_signal();
    Ok(())
}

/// Start native compilation for already lowered kernels. No Splash operations,
/// VM borrow, or field evaluation occurs here.
pub fn warm(cx: &mut Cx) -> Result<(), String> {
    while let Some((keys, started)) = cx.global::<TracePipelines>().pending.last().cloned() {
        let shaders = cx.global::<TracePipelines>().shaders[&keys].clone();
        if cx.global::<TracePipelines>().warm.is_none() {
            let document = Jobs::document(cx);
            let allocation = Allocation::new(cx, &document, 64, GpuBufferFlags { readback: true, indirect: false })?;
            allocation.buffer.update(cx, 0, &[0; 64])?;
            cx.global::<TracePipelines>().warm = Some(allocation);
        }
        let buffer = cx.global::<TracePipelines>().warm.as_ref().unwrap().buffer.clone();
        let mut pass = ComputePass::new("sdf trace warm", None);
        for kernel in [&shaders.0, &shaders.1] {
            let params = kernel.params();
            let buffers = vec![buffer.clone(); kernel.layout().buffers.len()];
            pass.dispatch(kernel, [0, 1, 1], &params, &buffers, &[])?;
        }
        pass.submit(cx);
        cx.read_gpu_buffer(&buffer, 0, 4, move |result| {
            log!("sdf trace warm: pipelines available after {:.3} ms, readback={:?}", started.elapsed().as_secs_f64() * 1000.0, result.as_ref().map(|_| ()));
            SignalToUI::set_ui_signal();
        }).map_err(|e| format!("{e:?}"))?;
        cx.global::<TracePipelines>().pending.pop();
    }
    Ok(())
}

/// Lowers `kernels` here unless the bake (or an earlier request) did: the bake lowers only the
/// shapes the view shows, and a request's shape may be one it skipped.
fn lower_for(cx: &mut Cx, kernels: &[KernelSlot]) -> Result<(), String> {
    if prepared(cx, kernels).is_ok() {
        return Ok(());
    }
    cx.with_vm(|vm| lower(vm, kernels))
}

fn prepared(cx: &mut Cx, kernels: &[KernelSlot]) -> Result<(ComputeKernel, ComputeKernel), String> {
    let keys: Vec<_> = kernels.iter().map(|s| s.key).collect();
    cx.global::<TracePipelines>().shaders.get(&keys).cloned().ok_or_else(|| "trace kernels have not been lowered yet".into())
}

/// A part revision's uploaded inputs, on the device generation they were made on.
struct Prepared {
    generation: u64,
    program: Allocation,
    meshes: Vec<(LiveId, Rc<crate::mesh::MeshGpu>)>,
}

/// One completion/error state per enqueue, not per part/revision: independent
/// continuations can share both keys. A loss also cancels work already handed
/// to the CPU pool so it cannot publish a later successful fragment.
struct PendingRequest {
    request: TraceRequest,
    // Zero is complete; usize::MAX is failed; otherwise layers still owed.
    remaining: AtomicUsize,
    /// Batches started.
    batches: AtomicU32,
}
impl std::ops::Deref for PendingRequest {
    type Target = TraceRequest;
    fn deref(&self) -> &TraceRequest { &self.request }
}
impl PendingRequest {
    fn failed(&self) -> bool { self.remaining.load(Ordering::Acquire) == usize::MAX }
    fn deliver(&self, result: Result<GpuBatch, JobError>) {
        // Reserve this delivery atomically against replacement and other CPU
        // decodes. A fully delivered request cannot receive a late error.
        let mut remaining = self.remaining.load(Ordering::Acquire);
        loop {
            if remaining == 0 || remaining == usize::MAX { return; }
            let next = match &result {
                Err(_) => usize::MAX,
                Ok(batch) => remaining.checked_sub(batch.layers.len()).expect("trace delivers each requested layer once"),
            };
            match self.remaining.compare_exchange_weak(remaining, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break,
                Err(now) => remaining = now,
            }
        }
        (self.request.sink)(result);
    }
}

impl Drop for PendingRequest {
    fn drop(&mut self) {
        // a batch the pool skipped for its stale ticket ends a stale request: Cancelled
        self.deliver(Err(if self.ticket.is_stale() { JobError::Cancelled }
            else { JobError::Failed("GPU trace request abandoned before all layers were delivered".into()) }));
    }
}

/// Layers of a request not started yet. `seq` is the enqueue order: the GPU traces requests in
/// the order the slicer asked for them. `most` caps a batch after a refusal split it.
struct Queued {
    request: Arc<PendingRequest>,
    first: usize,
    end: usize,
    seq: u64,
    most: usize,
    /// This batch's dropped attempts in a row.
    drops: u32,
}
/// A request's layers in the runtime.
struct Running {
    request: Arc<PendingRequest>,
    first: usize,
    count: usize,
    seq: u64,
    class: Class,
    trace_layers: Vec<(usize, f64)>,
    started: Instant,
    drops: u32,
}

/// Batches that may run at once per class and above: the GPU always has another batch's jobs
/// while one waits on its counts.
const ACTIVE_PER_CLASS: usize = 3;
pub(crate) const HEADER: usize = 32;
/// Layers of a part's first chunk, before its bytes a plane are measured.
const FIRST_CHUNK: usize = 8;
/// Grid cells (1 mm) of margin around a part's bounds.
const PADDING: f64 = 1.0;
/// Most slots of one resumed trace stage: each thread holds 1 KiB of resume state and its round
/// is bounded by its 32768-unit work budget. A round of this many slots, every one spending
/// its whole budget, is the dispatch `gpu_full_budget_dispatch` proves within the bound (about
/// 6 ms on a laptop GPU). It is low occupancy, ~1k threads where a 10-core GPU keeps 10-20k in
/// flight: the remedy is plane kernels of bounded samples per element with ceilings proven per
/// cost, not more slots of this interpreter, so it does not grow.
pub const TRACE_SLOTS: u32 = 1024;
pub const TRACE_BUDGET: u32 = 32768;
/// How far below a layer's plane (mm) its field is sampled: a flat face lying exactly on the
/// plane belongs to the solid under it, as a mesh's cut ((min z, max z] of a triangle) has it,
/// instead of a zero field across the face. The f64 contours sample at the same height
/// (`SdfSource::BELOW_PLANE` in Fab's slicer).
pub const BELOW_PLANE: f64 = 1e-4;

static OWNERS: AtomicU64 = AtomicU64::new(1);

/// One owner on Cx, submitting through the GPU job runtime ([`Jobs`]). It runs on every frame
/// (`pump`) and every signal or next-frame event (`signal`): readbacks drive it, so a slice
/// keeps going whether or not the UI redraws. Requests are traced in the order they came
/// (within a class, each request's viewed layer first). When nothing is queued or running, no
/// GPU memory is held.
pub struct GpuTracer {
    owner: u64,
    seq: u64,
    pool: TaskPool,
    queue: VecDeque<Queued>,
    requests: Vec<Weak<PendingRequest>>,
    /// Per (part, revision) queued or running: two shapes of a part can trace at once.
    prepared: HashMap<(u64, u64), Prepared>,
    logged_programs: HashMap<u64, u64>,
    running: HashMap<ItemId, Running>,
    interactive: bool,
    misfit: HashSet<u64>,
    /// Per part, a plane of its last batch: the GPU bytes it held (the batch's peak over its
    /// planes) and its GPU ms.
    plane_cost: HashMap<u64, (u64, f32)>,
}

impl GpuTracer {
    pub fn new(cx: &mut Cx) -> Self {
        Self { owner: OWNERS.fetch_add(1, Ordering::Relaxed), seq: 0, pool: cx.task_pool(), queue: VecDeque::new(), requests: Vec::new(),
            prepared: HashMap::new(), logged_programs: HashMap::new(), running: HashMap::new(), interactive: false, misfit: HashSet::new(),
            plane_cost: HashMap::new() }
    }
    /// The runtime answers every running batch DeviceLost; their layers trace again on the
    /// new device, in their order. Nothing uploaded on the old device is used again.
    pub fn device_lost(&mut self) {
        self.prepared.clear();
        // the runtime answers on the next pump, whether or not a readback still comes
        SignalToUI::set_ui_signal();
    }
    pub fn device_reset(&mut self, _cx: &mut Cx) {
        self.prepared.clear();
        // the queued batches start on the next signal: no frame is involved
        SignalToUI::set_ui_signal();
    }
    /// A test hook: this part's batches answer as a misfit.
    pub fn force_misfit(&mut self, part: Option<u64>) {
        self.misfit.clear();
        self.misfit.extend(part);
    }
    /// A request ends only by its ticket: one whose run was superseded goes stale and answers
    /// Cancelled (`signal`), while an older shape's request whose ticket stays live (an edited
    /// part's viewed layers, which a newer run of that shape may adopt) traces on beside the
    /// newer shape's.
    pub fn enqueue(&mut self, _cx: &mut Cx, request: TraceRequest) {
        let end = request.layers.len();
        // no layers, no batches: nothing to answer
        if end == 0 { return; }
        self.requests.retain(|r| r.strong_count() != 0);
        let request = Arc::new(PendingRequest { request, remaining: AtomicUsize::new(end), batches: AtomicU32::new(0) });
        self.requests.push(Arc::downgrade(&request));
        self.seq += 1;
        self.queue.push_back(Queued { request, first: 0, end, seq: self.seq, most: usize::MAX, drops: 0 });
        SignalToUI::set_ui_signal();
    }
    pub fn has_work(&self) -> bool { !self.running.is_empty() || !self.queue.is_empty() }

    /// A drag or touch runs: every submission stays short.
    pub fn set_interactive(&mut self, interactive: bool) { self.interactive = interactive; }

    /// Called once per frame.
    pub fn pump(&mut self, cx: &mut Cx) { self.signal(cx) }

    /// Called on signals (readbacks arriving) and next-frame events: the slice goes on without
    /// a redraw.
    pub fn signal(&mut self, cx: &mut Cx) {
        Jobs::set_interactive(cx, self.interactive);
        // stale work stops at its next submission boundary
        let stale: Vec<ItemId> = self.running.iter().filter(|(_, r)| r.request.failed() || r.request.ticket.is_stale()).map(|(id, _)| *id).collect();
        for id in stale {
            Jobs::abort(cx, id);
            let r = self.running.remove(&id).unwrap();
            r.request.deliver(Err(JobError::Cancelled));
        }
        Jobs::pump(cx);
        for finished in Jobs::finished(cx, self.owner) { self.finished(finished); }
        self.memory(cx);
        if !cx.gpu_device_lost() { self.start_queued(cx); }
        else if !cx.gpu_device_recovers() {
            // nothing will run again on this session: every request is answered now
            let error = "the GPU device was lost and this GPU backend cannot recover it; restart the app";
            for q in self.queue.drain(..) { q.request.deliver(Err(JobError::Failed(error.into()))); }
            for (id, r) in self.running.drain() {
                Jobs::abort(cx, id);
                r.request.deliver(Err(JobError::Failed(error.into())));
            }
        }
        let (queue, running) = (&self.queue, &self.running);
        self.prepared.retain(|&(part, revision), _| queue.iter().map(|q| &q.request).chain(running.values().map(|r| &r.request))
            .any(|r| r.part == part && r.revision == revision));
    }

    fn requeue(&mut self, r: Running, most: usize) {
        let q = Queued { request: r.request, first: r.first, end: r.first + r.count, seq: r.seq, most, drops: r.drops };
        let at = self.queue.iter().position(|o| o.seq > q.seq || (o.seq == q.seq && o.first > q.first)).unwrap_or(self.queue.len());
        self.queue.insert(at, q);
    }

    fn finished(&mut self, finished: Finished) {
        match finished {
            Finished::Failed { id, error } => {
                let Some(r) = self.running.remove(&id) else { return };
                match error {
                    JobError::DeviceLost => {
                        log!("sdf trace: part {} revision {} layers {}..{} trace again after the device reset", r.request.part, r.request.revision, r.first, r.first + r.count);
                        self.requeue(r, usize::MAX);
                    }
                    // a transient: the batch runs again from its start, and fails visibly only
                    // when it keeps being dropped
                    JobError::Dropped if r.drops + 1 < DROPS_IN_A_ROW => {
                        log!("sdf trace: part {} revision {} layers {}..{}: a submission was dropped; the batch runs again", r.request.part, r.request.revision, r.first, r.first + r.count);
                        let most = r.count;
                        let r = Running { drops: r.drops + 1, ..r };
                        self.requeue(r, most);
                    }
                    JobError::Dropped => {
                        let error = format!("GPU submissions dropped {DROPS_IN_A_ROW} times in a row on layers {}..{}", r.first, r.first + r.count);
                        log!("sdf trace failed: part {} revision {}: {}", r.request.part, r.request.revision, error);
                        r.request.deliver(Err(JobError::Failed(error)));
                    }
                    error => {
                        if !matches!(error, JobError::Cancelled) { log!("sdf trace failed: part {} revision {}: {}", r.request.part, r.request.revision, error); }
                        r.request.deliver(Err(error));
                    }
                }
            }
            Finished::Done { id, counts, stats, .. } => {
                let Some(r) = self.running.remove(&id) else { return };
                let wall_ms = r.started.elapsed().as_secs_f32() * 1000.0;
                let mut kernels = String::new();
                for (label, elements, ms, jobs) in &stats.kernels {
                    kernels.push_str(&format!(" {label}={elements}/{ms:.3}ms/{jobs}j"));
                }
                log!("sdf jobs batch: part {} revision {} layers {}..{} gpu_ms={:.3} wall_ms={:.3} per_layer_gpu_ms={:.3} per_layer_wall_ms={:.3} submissions={} jobs={} max_job_ms={:.3} ({}) readback_ms={:.3} max_readback_ms={:.3} build_ms={:.3} wait_ms={:.3} gap_ms={:.3} turn_ms={:.3} peak_mib={:.1} ends:[{}] kernels:{}",
                    r.request.part, r.request.revision, r.first, r.first + r.count, stats.gpu_ms, wall_ms, stats.gpu_ms / r.count as f32, wall_ms / r.count as f32,
                    stats.submissions, stats.jobs, stats.max_job_ms, stats.max_job, stats.readback_ms / stats.submissions.max(1) as f32, stats.max_readback_ms,
                    stats.build_ms, stats.wait_ms, stats.gap_ms, stats.turn_ms, stats.peak_bytes as f64 / (1u64 << 20) as f64, stats.ends, kernels);
                for n in 0..r.trace_layers.len().min(r.count) {
                    log!("sdf trace work: part {} revision {} layer {} kept={} crossings={} cell_evaluations={}",
                        r.request.part, r.request.revision, r.trace_layers[n].0,
                        counts.at((HEADER + n * 3) as u32), counts.at((HEADER + 1 + n * 3) as u32), counts.at((HEADER + 2 + n * 3) as u32));
                }
                // its output went on where it landed (`start`): what a plane held and took sizes the next chunks
                let planes = r.count.max(1);
                self.plane_cost.insert(r.request.part, (stats.peak_bytes / planes as u64, stats.gpu_ms / planes as f32));
            }
        }
    }

    /// Refused reservations: wait while our other batches hold memory; when every batch of
    /// ours waits (none can free memory for another), the lowest one goes back to the queue
    /// at half its layers, and one layer alone that still does not fit is a misfit.
    fn memory(&mut self, cx: &mut Cx) {
        let waiting = Jobs::waiting(cx, self.owner);
        if waiting.is_empty() || waiting.len() < self.running.len() { return; }
        let Some((id, refusal)) = waiting.into_iter().min_by_key(|(id, _)| {
            let r = &self.running[id];
            (r.class, std::cmp::Reverse(r.seq), std::cmp::Reverse(r.first))
        }) else { return };
        Jobs::abort(cx, id);
        let r = self.running.remove(&id).unwrap();
        if r.count > 1 || !self.running.is_empty() {
            log!("sdf trace admission: part {} layers {}..{} wait at {} layers a batch: {}", r.request.part, r.first, r.first + r.count, (r.count / 2).max(1), refusal.reason);
            let most = (r.count / 2).max(1);
            self.requeue(r, most);
        } else {
            self.misfit(r.request, &refusal);
        }
    }

    fn misfit(&mut self, request: Arc<PendingRequest>, refusal: &Refusal) {
        let mib = |b: u64| b.div_ceil(1 << 20);
        log!("sdf trace misfit: part {} revision {}: {} MiB wanted, {} MiB available: {}", request.part, request.revision, mib(refusal.wanted), mib(refusal.available), refusal.reason);
        self.queue.retain(|q| !Arc::ptr_eq(&q.request, &request));
        request.deliver(Err(JobError::Misfit { wanted_mib: mib(refusal.wanted), available_mib: mib(refusal.available), reason: refusal.reason.clone() }));
    }

    fn class(&self, class: Class) -> JobClass {
        match class {
            Class::Interactive if self.interactive => JobClass::Interactive,
            Class::Interactive | Class::Visible => JobClass::Visible,
            Class::Background => JobClass::Background,
        }
    }

    fn start_queued(&mut self, cx: &mut Cx) {
        loop {
            self.queue.retain(|q| {
                if q.request.ticket.is_stale() { q.request.deliver(Err(JobError::Cancelled)); }
                !q.request.failed() && q.first < q.end
            });
            // Within a class, a part revision's lead (the first layer of its first request, the
            // viewed one) precedes the rest; then requests in the order they came.
            let prepared = &self.prepared;
            let lead = |q: &Queued| q.first == 0 && !prepared.contains_key(&(q.request.part, q.request.revision));
            let Some(i) = self.queue.iter().enumerate()
                .min_by_key(|(_, q)| (std::cmp::Reverse(q.request.class), !lead(q), q.seq, q.first)).map(|(i, _)| i) else { break };
            let class = self.queue[i].request.class;
            let lead = lead(&self.queue[i]);
            // per class: a background batch never keeps a visible one from starting
            let active = self.running.values().filter(|r| r.class >= class).count();
            if active >= ACTIVE_PER_CLASS + usize::from(lead) { break }
            let q = self.queue.remove(i).unwrap();
            if self.misfit.contains(&q.request.part) {
                let refusal = Refusal { wanted: 1 << 40, available: 0, reason: "forced by a test".into() };
                self.misfit(q.request, &refusal);
                continue;
            }
            // A shape the bake never lowered (an edit the view skipped on its way to a newer
            // one) is lowered here: a request never waits for a bake that will not come.
            if let Err(e) = lower_for(cx, &q.request.kernels) {
                log!("sdf trace: part {} revision {} does not lower: {}", q.request.part, q.request.revision, e);
                q.request.deliver(Err(JobError::Failed(format!("its shape does not lower for GPU slicing: {e}"))));
                continue;
            }
            let count = if lead { 1 } else { self.chunk(cx, &q) };
            match self.start(cx, &q, count, lead) {
                Ok(Ok(())) => {
                    if q.first + count < q.end {
                        self.queue.insert(i.min(self.queue.len()), Queued { request: q.request, first: q.first + count, end: q.end, seq: q.seq, most: q.most, drops: 0 });
                    }
                }
                Ok(Err(refusal)) => {
                    if !self.running.is_empty() {
                        // our running batches hold memory: this one waits for them
                        self.queue.insert(i.min(self.queue.len()), q);
                        break;
                    }
                    if count > 1 {
                        log!("sdf trace admission: {}; {} layers a batch", refusal.reason, (count / 2).max(1));
                        self.queue.insert(i.min(self.queue.len()), Queued { most: (count / 2).max(1), ..q });
                        continue;
                    }
                    self.misfit(q.request, &refusal);
                }
                Err(e) => {
                    log!("sdf trace refused: {}", e);
                    q.request.deliver(Err(JobError::Failed(e)));
                }
            }
        }
    }

    /// Layers of the next chunk of `q`, at what a plane of the part's last batch held and took
    /// ([`FIRST_CHUNK`] before one was measured): within half the device's headroom, and about
    /// [`LONG_MS`] of GPU time, so the cores plan a chunk while the GPU traces the next; within
    /// what a refusal left it (`most`).
    fn chunk(&self, cx: &mut Cx, q: &Queued) -> usize {
        let n = match self.plane_cost.get(&q.request.part) {
            Some(&(bytes, ms)) => {
                let fits = Jobs::headroom(cx) / 2 / bytes.max(1);
                let paced = (LONG_MS / ms.max(1e-3)) as u64;
                fits.min(paced).min(usize::MAX as u64) as usize
            }
            None => FIRST_CHUNK,
        };
        n.clamp(1, q.end - q.first).min(q.most)
    }

    /// Starts layers `q.first..q.first + count` as one work item. Ok(Err): the runtime refused
    /// its first buffers.
    fn start(&mut self, cx: &mut Cx, q: &Queued, count: usize, lead: bool) -> Result<Result<(), Refusal>, String> {
        let r = &q.request;
        let (coarse, cell) = prepared(cx, &r.kernels)?;
        if self.logged_programs.insert(r.part, r.revision) != Some(r.revision) {
            let mut pc = 1;
            let mut ops = 0;
            let mut kernel_cost = 0;
            while pc < r.program.texels[0][0] as usize {
                let h = r.program.texels[pc];
                if h[0] == crate::program::op::KERNEL {
                    kernel_cost += r.kernels[h[1] as usize].air().map_or(0, |g| g.cost);
                } else if h[0] == crate::program::op::MESH {
                    kernel_cost += r.kernels[h[1] as usize].mesh_asset().map_or(0, |m| m.cost());
                }
                pc += crate::program::op_len(h);
                ops += 1;
            }
            log!("sdf trace program: part {} revision {} ops_per_sample={} kernel_cost_per_sample={} texels={}",
                r.part, r.revision, ops, kernel_cost, r.program.texels.len());
        }
        let generation = cx.gpu_device_generation();
        if self.prepared.get(&(r.part, r.revision)).is_none_or(|p| p.generation != generation) {
            let document = Jobs::document(cx);
            let bytes: Vec<_> = r.program.texels.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
            let program = match Allocation::new(cx, &document, bytes.len() as u64, GpuBufferFlags::default()) {
                Ok(p) => p,
                Err(reason) => return Ok(Err(Refusal { wanted: bytes.len() as u64, available: 0, reason })),
            };
            program.buffer.update(cx, 0, &bytes)?;
            let meshes = r.kernels.iter().enumerate().filter_map(|(i, slot)| slot.mesh_asset().map(|asset| (i, asset)))
                .map(|(i, asset)| crate::mesh::upload(cx, &document, asset).map(|gpu| (crate::mesh::binding(i), gpu)))
                .collect::<Result<Vec<_>, _>>();
            let meshes = match meshes { Ok(m) => m, Err(reason) => return Ok(Err(Refusal { wanted: 1, available: 0, reason })) };
            self.prepared.insert((r.part, r.revision), Prepared { generation, program, meshes });
        }
        let origin = [(r.bounds.0[0] - PADDING) as f32, (r.bounds.0[1] - PADDING) as f32];
        let nx = (r.bounds.1[0] - r.bounds.0[0] + PADDING * 2.0).ceil() as u32;
        let ny = (r.bounds.1[1] - r.bounds.0[1] + PADDING * 2.0).ceil() as u32;
        let trace_layers = r.layers[q.first..q.first + count].to_vec();
        let traced = trace_layers.len() as u32;
        // Compose the grid origin into the inverse frame before rounding to
        // f32. The shader samples/refines locally and adds the printer origin
        // only when emitting vertices. No field evaluation runs on the host.
        let base = [r.bounds.0[0] - PADDING, r.bounds.0[1] - PADDING];
        let frame = r.to_shape.map(|mut row| {
            row[3] = ((row[0] as f64 * base[0] + row[1] as f64 * base[1]) + row[3] as f64) as f32;
            row
        });
        let mut inputs: Vec<u8> = frame.into_iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        for v in [origin[0], origin[1], r.lipschitz, 0.0] { inputs.extend(v.to_le_bytes()); }
        for &(_, z) in &trace_layers {
            for v in [(z - BELOW_PLANE) as f32, 0.0, 0.0, 0.0] { inputs.extend(v.to_le_bytes()); }
        }
        let prepared = &self.prepared[&(r.part, r.revision)];
        let mut pool = Pool::new(HEADER as u32 + traced * 3 + 32);
        let program = pool.external("sdf program", prepared.program.buffer.clone());
        let meshes: Vec<(LiveId, BufRef)> = prepared.meshes.iter().map(|(name, gpu)| (*name, pool.external("sdf mesh", gpu.buffer.clone()))).collect();
        let input = pool.buffer("trace inputs", inputs.len() as u64);
        pool.upload(input, 0, inputs);
        // Every coarse cell has at most one kept entry, then two words of crossing meta per
        // kept cell: an exact grid bound, before the first count is known.
        let kept_capacity = nx.checked_mul(ny).and_then(|n| n.checked_mul(traced)).ok_or("trace grid address overflow")?.max(1);
        let kept = pool.buffer("trace kept", kept_capacity as u64 * 48);
        let trace = Rc::new(TraceStages { coarse, cell, grid: [nx, ny, traced, kept_capacity], program, input, kept, meshes });
        let steps = trace.steps()?;
        let class = self.class(r.class);
        let order = (u64::from(!lead) << 48) | q.seq;
        let label = format!("sdf trace part {} revision {} layers {}..{}", r.part, r.revision, q.first, q.first + count);
        match Jobs::start(cx, WorkItem { owner: self.owner, label, class, order, pool, steps }) {
            Ok(id) => {
                let (lo, hi) = trace_layers.iter().fold((u32::MAX, 0), |(lo, hi), &(k, _)| (lo.min(k as u32), hi.max(k as u32 + 1)));
                Jobs::tag(cx, id, ItemTag { part: r.part, revision: r.input, layers: (lo, hi) });
                // The batch goes to its sink from where its readbacks land, on the pool: the lead
                // and a viewed request's chunks at Visible, a background request's at Background.
                let class = if lead || r.class > Class::Background { Class::Visible } else { Class::Background };
                let (pool, request, layers) = (self.pool.clone(), r.clone(), Arc::<[(usize, f64)]>::from(&trace_layers[..]));
                let (part, revision, batch) = (r.part, r.revision, r.batches.fetch_add(1, Ordering::Relaxed));
                Jobs::on_landed(cx, id, Box::new(move |counts, output, stats, landed| {
                    let batch = GpuBatch { part, revision, batch, layers, counts, output, stats, landed };
                    let deliver = request.clone();
                    match pool.submit_at(class, "sdf batch", request.ticket.clone(), move || deliver.deliver(Ok(batch))) {
                        Ok(job) => job.detach(),
                        // a refused spawn also answers the caller: a protected slice never waits on a stopped pool
                        Err(e) => request.deliver(Err(JobError::Failed(format!("GPU trace batch could not be scheduled: {e:?}")))),
                    }
                }));
                log!("sdf trace submit: part {} key {} layers {}..{} running={} request_ms={:.3}", r.part, r.revision, q.first, q.first + count,
                    self.running.len() + 1, r.requested.elapsed().as_secs_f64() * 1000.0);
                self.running.insert(id, Running { request: r.clone(), first: q.first, count, seq: q.seq, class: r.class, trace_layers,
                    started: Instant::now(), drops: q.drops });
                Ok(Ok(()))
            }
            Err(refusal) => Ok(Err(refusal)),
        }
    }
}

/// Today's trace kernels as steps: each stage resumes until its elements are done, and the
/// next stage's elements (and its buffers) come from the counts the previous one wrote.
struct TraceStages {
    coarse: ComputeKernel,
    cell: ComputeKernel,
    grid: [u32; 4],
    program: BufRef,
    input: BufRef,
    kept: BufRef,
    meshes: Vec<(LiveId, BufRef)>,
}
struct Stage { mode: u32, label: &'static str, n: u32 }
#[derive(Clone, Copy)]
struct Bound { values: BufRef, cross: BufRef, spans: BufRef, segments: BufRef, capacity: u32 }
impl TraceStages {
    fn pass(&self, stage: Stage, b: Bound) -> Result<Step, JobError> {
        let kernel = if stage.mode == 0 { &self.coarse } else { &self.cell };
        let bind = |name: LiveId| {
            if name == id!(program) { self.program }
            else if name == id!(inputs) { self.input }
            else if name == id!(kept) { self.kept }
            else if name == id!(values) { b.values }
            else if name == id!(cross) { b.cross }
            else if name == id!(spans) { b.spans }
            else if name == id!(segments) { b.segments }
            else if name == id!(counts) || name == id!(continuations) { BufRef::RUNTIME }
            else { self.meshes.iter().find(|(n, _)| *n == name).map_or(BufRef::RUNTIME, |m| m.1) }
        };
        let mut pass = Pass::new(kernel, crate::gpu_jobs::label(stage.label), Elements::Fixed(stage.n), TRACE_BUDGET, TRACE_SLOTS, bind)
            .param_u32x4(id!(grid), self.grid).map_err(JobError::Failed)?;
        if stage.mode != 0 {
            pass = pass.param_u32(id!(capacity), b.capacity).map_err(JobError::Failed)?.param_u32(id!(mode), stage.mode).map_err(JobError::Failed)?;
        }
        Ok(Step::Resume { pass, slots_ceiling: TRACE_SLOTS })
    }

    fn steps(self: Rc<Self>) -> Result<Vec<Step>, String> {
        // the coarse stage binds no values, crossings or spans yet
        let none = Bound { values: self.kept, cross: self.kept, spans: self.kept, segments: self.kept, capacity: 1 };
        let [nx, ny, traced, kept_capacity] = self.grid;
        let n = nx.checked_mul(ny).and_then(|n| n.checked_mul(traced)).ok_or("trace dispatch address overflow")?;
        let coarse = self.pass(Stage { mode: 0, label: "trace coarse", n }, none).map_err(|e| e.to_string())?;
        let me = self.clone();
        let after_coarse: crate::gpu_jobs::HostStep = Box::new(move |counts, pool| {
            me.check(counts)?;
            let kept = counts.at(0).min(kept_capacity);
            let values = pool.buffer("trace values", kept as u64 * 121 * 4);
            let b = Bound { values, ..none };
            let nodes = kept.checked_mul(121).ok_or_else(|| JobError::Failed("trace dispatch address overflow".into()))?;
            let me2 = me.clone();
            let after_edges: crate::gpu_jobs::HostStep = Box::new(move |counts, pool| {
                me2.check(counts)?;
                // Every span consumes two crossing uses, and a crossing belongs to at most two
                // fine cells: spans <= crossings. Sized from the exact count before scattering
                // or refining any crossing; this layer never starts over.
                let capacity = counts.at(4).max(1);
                let cross = pool.buffer("trace crossings", capacity as u64 * 32);
                let spans = pool.buffer("trace spans", capacity as u64 * 32);
                let segments = pool.readback("trace output", capacity as u64 * 64);
                let b = Bound { values, cross, spans, segments, capacity };
                let crossings = counts.at(4).min(capacity);
                let cells = kept.checked_mul(100).ok_or_else(|| JobError::Failed("trace dispatch address overflow".into()))?;
                let me3 = me2.clone();
                let after_spans: crate::gpu_jobs::HostStep = Box::new(move |counts, _pool| {
                    me3.check(counts)?;
                    let spans = counts.at(6).min(capacity);
                    Ok(vec![me3.pass(Stage { mode: 7, label: "trace segments", n: spans }, b)?, Output::counted(segments, CountWord(2), 64, capacity)])
                });
                Ok(vec![
                    me2.pass(Stage { mode: 3, label: "trace scatter", n: kept }, b)?,
                    me2.pass(Stage { mode: 4, label: "trace crossings", n: crossings }, b)?,
                    me2.pass(Stage { mode: 5, label: "trace spans", n: cells }, b)?,
                    Step::Checkpoint,
                    Step::Host(after_spans),
                ])
            });
            Ok(vec![
                me.pass(Stage { mode: 1, label: "trace nodes", n: nodes }, b)?,
                me.pass(Stage { mode: 2, label: "trace edges", n: kept }, b)?,
                Step::Checkpoint,
                Step::Host(after_edges),
            ])
        });
        Ok(vec![coarse, Step::Checkpoint, Step::Host(after_coarse)])
    }

    /// A count past its allocation stops the batch (the kernels stop evaluating too).
    fn check(&self, counts: &Counts) -> Result<(), JobError> {
        if counts.at(1) != 0 || counts.at(3) != 0 {
            return Err(JobError::Failed(format!("GPU trace count/allocation invariant failed: kept={} trace_overflow={}", counts.at(0), counts.at(1) + counts.at(3))));
        }
        Ok(())
    }
}

pub(crate) fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at * 4..at * 4 + 4].try_into().unwrap())
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct Edge(u32, u32, u32);

/// Linear hash chaining, with deterministic loop start/order independent of
/// atomic append order. An open chain is an error, never silently repaired.
fn chain_layers(bytes: &[u8], layers: &[(usize, f64)]) -> Result<Vec<TracedLayer>, String> {
    let mut by_layer = vec![HashMap::new(); layers.len()];
    for s in 0..bytes.len() / 64 {
        let at = s * 16;
        let layer = word(bytes, at + 3) as usize;
        let a = Edge(word(bytes, at), word(bytes, at + 1), word(bytes, at + 2));
        if layer >= layers.len() { return Err("trace segment has an invalid layer".into()) }
        if by_layer[layer].insert(a, s).is_some() { return Err(format!("trace layer {} z={} has duplicate outgoing edge {a:?}", layers[layer].0, layers[layer].1)) }
    }
    let mut out = Vec::with_capacity(layers.len());
    for ((index, z), mut edges) in layers.iter().copied().zip(by_layer) {
        let mut creases = Vec::new();
        let mut loops = BTreeMap::new();
        while let Some((&start, _)) = edges.iter().next() {
            let mut edge = start;
            let mut path = Vec::new();
            let mut starts = Vec::new();
            let mut expected = None;
            loop {
                let s = edges.remove(&edge).ok_or_else(|| format!("open trace layer {index} z={z} chain at {edge:?}"))?;
                let at = s * 16;
                let point = [word(bytes, at + 8), word(bytes, at + 9)];
                if expected.is_some_and(|p| p != point) { return Err(format!("shared trace layer {index} z={z} edge {edge:?} has different crossing bits")); }
                expected = Some([word(bytes, at + 10), word(bytes, at + 11)]);
                starts.push((edge, path.len()));
                path.push(point.map(f32::from_bits));
                if word(bytes, at + 14) == 2 { creases.push([f32::from_bits(word(bytes, at + 12)), f32::from_bits(word(bytes, at + 13))]); }
                if word(bytes, at + 14) != 0 { path.push([f32::from_bits(word(bytes, at + 12)), f32::from_bits(word(bytes, at + 13))]); }
                edge = Edge(word(bytes, at + 4), word(bytes, at + 5), word(bytes, at + 6));
                if edge == start {
                    if expected != Some(path[0].map(f32::to_bits)) { return Err("trace closing crossing differs".into()); }
                    break
                }
            }
            let (minimum, offset) = starts.into_iter().min_by_key(|s| s.0).unwrap();
            path.rotate_left(offset);
            loops.insert(minimum, path);
        }
        out.push(TracedLayer { index, z, loops: loops.into_values().collect(), creases });
    }
    Ok(out)
}

impl Drop for GpuTracer {
    fn drop(&mut self) {
        for request in self.requests.drain(..).filter_map(|r| r.upgrade()) {
            request.deliver(Err(JobError::Failed("GPU tracer dropped before the request completed".into())));
        }
    }
}
