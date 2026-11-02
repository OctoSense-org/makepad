//! Buffer-backed SDF evaluation and compute bake; no resource creation or dispatch.
//!
//! SHADER LAW: unary minus has the lowest precedence; use `0.0 - x`.
//! Programs retain the old interpreter's f32 operation order. The interpreter
//! executes at most the header's texel count (each op advances at least one).

use makepad_draw::*;

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.Eval = {
        program: shader.buffer(vec4f)
        fetch: fn(i: f32) -> vec4 {
            return self.program[u32(i)]
        }
        len2: fn(x: f32, y: f32) -> f32 {
            return sqrt(x * x + y * y)
        }
        len3: fn(x: f32, y: f32, z: f32) -> f32 {
            return sqrt((x * x + y * y) + z * z)
        }
        clamp01: fn(x: f32) -> f32 {
            return min(max(x, 0.0), 1.0)
        }
        // Select packed words as integers. FXC's dynamic float-vector indexing
        // flushes the subnormal float patterns used by small u32 indices.
        links_bits: fn(base: u32, i: u32) -> u32 {
            let v = self.program[base + i / 4u]
            let bits = vec4u(asuint(v.x), asuint(v.y), asuint(v.z), asuint(v.w))
            return bits[i % 4u]
        }
        links_word: fn(base: u32, i: u32) -> f32 {
            return asfloat(self.links_bits(base, i))
        }
        links_at: fn(base: u32, q: vec3) -> vec2 {
            let h = self.program[base]
            let r = h.x
            let reach = h.y
            let origin = self.program[base + 4u].xyz
            let dim_words = self.program[base + 5u]
            let dims = vec3u(asuint(dim_words.x), asuint(dim_words.y), asuint(dim_words.z))
            let offset_words = self.program[base + 6u]
            let offsets = vec3u(asuint(offset_words.x), asuint(offset_words.y), asuint(offset_words.z))
            let lo = max(floor((q - reach - r - origin) / reach), vec3(0.0))
            let hi = min(floor((q + reach + r - origin) / reach), vec3(f32(dims.x - 1u), f32(dims.y - 1u), f32(dims.z - 1u)))
            var best = asfloat(2139095040u)
            var slot = 0.0
            for dz in 0..4u { for dy in 0..4u { for dx in 0..4u {
                let x = lo.x + f32(dx)
                let y = lo.y + f32(dy)
                let z = lo.z + f32(dz)
                if x > hi.x || y > hi.y || z > hi.z { continue }
                let c = (u32(z) * dims.y + u32(y)) * dims.x + u32(x)
                let start = self.links_bits(base, offsets.x + c)
                let end = self.links_bits(base, offsets.x + c + 1u)
                for i in 0..asuint(h.w) {
                    if start + i >= end { break }
                    let at = offsets.z + self.links_bits(base, offsets.y + start + i) * 7u
                    let a = vec3(self.links_word(base, at), self.links_word(base, at + 1u), self.links_word(base, at + 2u))
                    let b = vec3(self.links_word(base, at + 3u), self.links_word(base, at + 4u), self.links_word(base, at + 5u))
                    let ab = b - a
                    let ap = q - a
                    let l2 = (ab.x * ab.x + ab.y * ab.y) + ab.z * ab.z
                    var t = 0.0
                    if l2 > 0.0 { t = self.clamp01(((ap.x * ab.x + ap.y * ab.y) + ap.z * ab.z) / l2) }
                    let d = self.len3(ap.x - ab.x * t, ap.y - ab.y * t, ap.z - ab.z * t) - r
                    if d < best { best = d; slot = f32(self.links_bits(base, at + 6u)) }
                }
            } } }
            let out = max(max(self.program[base + 2u].xyz - q, q - self.program[base + 3u].xyz), vec3(0.0))
            let d_box = self.len3(out.x, out.y, out.z)
            var d = min(best, reach)
            if d_box > 0.0 { d = max(d_box, d) }
            if h.z != 0.0 {
                let n = self.program[base + 1u]
                d = max(d, n.w - ((n.x * q.x + n.y * q.y) + n.z * q.z))
            }
            return vec2(d, slot)
        }
        // Replaced by a binding dispatch for the object's immutable mesh assets.
        mesh_read: fn(slot: u32, at: u32) -> vec4u { return vec4u(0u) }
        mesh_xyz: fn(slot: u32, at: u32) -> vec3 {
            let w = self.mesh_read(slot, at)
            return vec3(asfloat(w.x), asfloat(w.y), asfloat(w.z))
        }
        mesh_dot: fn(a: vec3, b: vec3) -> f32 {
            return (a.x * b.x + a.y * b.y) + a.z * b.z
        }
        mesh_box: fn(slot: u32, node: u32, p: vec3) -> f32 {
            let lo = self.mesh_xyz(slot, 1u + node * 4u)
            let hi = self.mesh_xyz(slot, 2u + node * 4u)
            let e = max(max(lo - p, vec3(0.0)), p - hi)
            return ((0.0 + e.x * e.x) + e.y * e.y) + e.z * e.z
        }
        // MeshField::closest_on_tri, preserving each Voronoi test and operation.
        mesh_closest: fn(p: vec3, a: vec3, b: vec3, c: vec3) -> vec3 {
            let ab = b - a
            let ac = c - a
            let ap = p - a
            let d1 = self.mesh_dot(ab, ap)
            let d2 = self.mesh_dot(ac, ap)
            if d1 <= 0.0 && d2 <= 0.0 { return a }
            let bp = p - b
            let d3 = self.mesh_dot(ab, bp)
            let d4 = self.mesh_dot(ac, bp)
            if d3 >= 0.0 && d4 <= d3 { return b }
            let vc = d1 * d4 - d3 * d2
            if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 { return a + ab * (d1 / (d1 - d3)) }
            let cp = p - c
            let d5 = self.mesh_dot(ab, cp)
            let d6 = self.mesh_dot(ac, cp)
            if d6 >= 0.0 && d5 <= d6 { return c }
            let vb = d5 * d2 - d1 * d6
            if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 { return a + ac * (d2 / (d2 - d6)) }
            let va = d3 * d6 - d5 * d4
            if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
                return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
            }
            let denom = 1.0 / (va + vb + vc)
            let v = vb * denom
            let w = vc * denom
            return (a + ab * v) + ac * w
        }
        // mesh_field::mesh_atan2, including the sign of zero on each axis.
        mesh_atan2: fn(y: f32, x: f32) -> f32 {
            let yn = y < 0.0 || (y == 0.0 && 1.0 / y < 0.0)
            let xn = x < 0.0 || (x == 0.0 && 1.0 / x < 0.0)
            if y == 0.0 {
                if xn {
                    if yn { return 0.0 - 3.1415927410125732 }
                    return 3.1415927410125732
                }
                return y
            }
            if x == 0.0 {
                if yn { return 0.0 - 1.5707963705062866 }
                return 1.5707963705062866
            }
            var ax = x
            var ay = y
            if xn { ax = 0.0 - x }
            if yn { ay = 0.0 - y }
            let swap = ay > ax
            var r = 1.0
            if ax != ay {
                if swap { r = ax / ay } else { r = ay / ax }
            }
            var base = 0.0
            if r > 0.4142135679721832 {
                r = (r - 1.0) / (r + 1.0)
                base = 0.7853981852531433
            }
            let z = r * r
            let p = (0.0 - 0.3333333333333333) + z * (0.2 + z * ((0.0 - 0.14285714285714285) + z * (0.1111111111111111 + z * ((0.0 - 0.09090909090909091) + z * (0.07692307692307693 + z * ((0.0 - 0.06666666666666667) + z * 0.058823529411764705))))))
            var a = base + (r + (r * z) * p)
            if swap { a = 1.5707963705062866 - a }
            if xn { a = 3.1415927410125732 - a }
            if yn { return a * (0.0 - 1.0) }
            return a
        }
        mesh_angle: fn(p: vec3, a: vec3, b: vec3, c: vec3) -> f32 {
            let x = a - p
            let y = b - p
            let z = c - p
            let lx = self.len3(x.x, x.y, x.z)
            let ly = self.len3(y.x, y.y, y.z)
            let lz = self.len3(z.x, z.y, z.z)
            let yz = vec3(y.y * z.z - y.z * z.y, y.z * z.x - y.x * z.z, y.x * z.y - y.y * z.x)
            let det = self.mesh_dot(x, yz)
            let div = ((lx * ly * lz + self.mesh_dot(x, y) * lz) + self.mesh_dot(x, z) * ly) + self.mesh_dot(y, z) * lx
            return self.mesh_atan2(det, div) * 0.15915493667125702
        }
        mesh_at: fn(slot: u32, p: vec3) -> f32 {
            let h = self.mesh_read(slot, 0u)
            var best = asfloat(2139095040u)
            if h.x == 0u { return best }
            // Median split, leaves <= 8: depth <= 30 for u32 triangle counts.
            // query_words asserts the actual depth before upload. Both walks
            // visit at most the node count; leaf loops visit at most 8 triangles.
            var stack = array(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u)
            var count = 1u
            for visit in 0..h.x {
                if count == 0u { break }
                count -= 1u
                let node = stack[count]
                if self.mesh_box(slot, node, p) >= best * best { continue }
                let at = 1u + node * 4u
                let left = self.mesh_read(slot, at).w
                let right = self.mesh_read(slot, at + 1u).w
                if self.mesh_read(slot, at + 2u).w != 0u {
                    for i in left..right {
                        let t = self.mesh_read(slot, h.z + self.mesh_read(slot, h.w + i).x)
                        let a = self.mesh_xyz(slot, h.y + t.x)
                        let b = self.mesh_xyz(slot, h.y + t.y)
                        let c = self.mesh_xyz(slot, h.y + t.z)
                        let d = p - self.mesh_closest(p, a, b, c)
                        let dl = self.len3(d.x, d.y, d.z)
                        if dl < best { best = dl }
                    }
                } else {
                    if self.mesh_box(slot, left, p) < self.mesh_box(slot, right, p) {
                        stack[count] = right
                        stack[count + 1u] = left
                    } else {
                        stack[count] = left
                        stack[count + 1u] = right
                    }
                    count += 2u
                }
            }
            let winding = self.mesh_winding(slot, p)
            if winding > 0.5 { return asfloat(asuint(best) ^ 2147483648u) }
            return best
        }
        mesh_winding: fn(slot: u32, p: vec3) -> f32 {
            let h = self.mesh_read(slot, 0u)
            var stack = array(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u)
            var count = 1u
            var winding = 0.0
            for visit in 0..h.x {
                if count == 0u { break }
                count -= 1u
                let at = 1u + stack[count] * 4u
                let d = self.mesh_xyz(slot, at + 3u) - p
                let dl = self.len3(d.x, d.y, d.z)
                let radius = asfloat(self.mesh_read(slot, at + 3u).w)
                if dl > 2.0 * radius && dl > 0.0 {
                    winding += self.mesh_dot(self.mesh_xyz(slot, at + 2u), d) / (12.566370964050293 * dl * dl * dl)
                    continue
                }
                let left = self.mesh_read(slot, at).w
                let right = self.mesh_read(slot, at + 1u).w
                if self.mesh_read(slot, at + 2u).w != 0u {
                    for i in left..right {
                        let t = self.mesh_read(slot, h.z + self.mesh_read(slot, h.w + i).x)
                        let a = self.mesh_xyz(slot, h.y + t.x)
                        let b = self.mesh_xyz(slot, h.y + t.y)
                        let c = self.mesh_xyz(slot, h.y + t.z)
                        winding += self.mesh_angle(p, a, b, c)
                    }
                } else {
                    stack[count] = left
                    stack[count + 1u] = right
                    count += 2u
                }
            }
            return winding
        }
        // sin and cos (twin::psin_pcos)
        psc: fn(x: f32) -> vec2 {
            let q = floor(x * 0.6366198 + 0.5)
            let r = (x - q * 1.5707964) - q * (0.0 - 0.00000004371139)
            let r2 = r * r
            let s = r + r * r2 * ((0.0 - 0.16666667) + r2 * (0.008333331 + r2 * ((0.0 - 0.00019840874) + r2 * 0.0000027525562)))
            let c = 1.0 + r2 * ((0.0 - 0.5) + r2 * (0.041666638 + r2 * ((0.0 - 0.0013888378) + r2 * 0.000024760495)))
            let n = q - floor(q * 0.25) * 4.0
            if n < 0.5 { return vec2(s, c) }
            if n < 1.5 { return vec2(c, 0.0 - s) }
            if n < 2.5 { return vec2(0.0 - s, 0.0 - c) }
            return vec2(0.0 - c, s)
        }
        smin: fn(a: f32, b: f32, k: f32) -> f32 {
            if k <= 0.0 { return min(a, b) }
            let h = self.clamp01(0.5 + (0.5 * (b - a)) / k)
            return (b + (a - b) * h) - (k * h) * (1.0 - h)
        }
        prim: fn(pc: f32, px: f32, py: f32, pz: f32) -> f32 {
            let t0 = self.fetch(pc)
            let t1 = self.fetch(pc + 1.0)
            let t2 = self.fetch(pc + 2.0)
            let kind = t0.y
            let a = t0.z
            let b = t0.w
            let c = t1.x
            let d = t1.y
            if kind < 0.5 {
                return self.len3(px, py, pz) - a
            }
            if kind < 1.5 {
                let qx = abs(px) - a
                let qy = abs(py) - b
                let qz = abs(pz) - c
                return self.len3(max(qx, 0.0), max(qy, 0.0), max(qz, 0.0)) + min(max(qx, max(qy, qz)), 0.0) - d
            }
            if kind < 2.5 {
                let qx = self.len2(px, pz)
                let dx = qx - a
                let dy = abs(py) - b
                let mx = max(dx, 0.0)
                let my = max(dy, 0.0)
                return sqrt(mx * mx + my * my) + min(max(dx, dy), 0.0)
            }
            if kind < 3.5 {
                let qx = self.len2(px, pz)
                let k2x = b - a
                let k2y = c * 2.0
                var rr = b
                if py < 0.0 { rr = a }
                let cax = qx - min(qx, rr)
                let cay = abs(py) - c
                let t = self.clamp01(((b - qx) * k2x + (c - py) * k2y) / (k2x * k2x + k2y * k2y))
                let cbx = (qx - b) + k2x * t
                let cby = (py - c) + k2y * t
                var s = 1.0
                if cbx < 0.0 && cay < 0.0 { s = 0.0 - 1.0 }
                return s * sqrt(min(cax * cax + cay * cay, cbx * cbx + cby * cby))
            }
            if kind < 4.5 {
                let q = self.len2(px, pz) - a
                return self.len2(q, py) - b
            }
            if kind < 5.5 {
                let pax = px - a
                let pay = py - b
                let paz = pz - c
                let bx = d
                let by = t1.z
                let bz = t1.w
                let h = self.clamp01(((pax * bx + pay * by) + paz * bz) / ((bx * bx + by * by) + bz * bz))
                return self.len3(pax - bx * h, pay - by * h, paz - bz * h) - t2.x
            }
            if kind < 6.5 {
                let k0 = self.len3(px / a, py / b, pz / c)
                let k1 = self.len3(px / (a * a), py / (b * b), pz / (c * c))
                return (k0 * (k0 - 1.0)) / max(k1, 0.000000000001)
            }
            let ex = self.psc(px * a)
            let ey = self.psc(py * a)
            let ez = self.psc(pz * a)
            let gv = (ex.x * ey.y + ey.x * ez.y) + ez.x * ex.y
            return abs(gv) / a - b
        }

        // A KERNEL op at program texel `at` (kernel slot `slot`): the plain
        // shaders have none (a volume with kernels bakes with a variant,
        // `kernel::KernelBake`, whose kernel_at is kernel_at_fetch and whose
        // kernel_call the host builds from the kernels' compiled programs).
        kernel_at: fn(at: f32, slot: f32, x: f32, y: f32, z: f32) -> f32 {
            return 1000000000.0
        }
        kernel_at_fetch: fn(at: f32, slot: f32, x: f32, y: f32, z: f32) -> f32 {
            let n = self.fetch(at).z
            var a = vec4(0.0)
            var b = vec4(0.0)
            var c = vec4(0.0)
            var d = vec4(0.0)
            if n > 0.0 { a = self.fetch(at + 1.0) }
            if n > 1.0 { b = self.fetch(at + 2.0) }
            if n > 2.0 { c = self.fetch(at + 3.0) }
            if n > 3.0 { d = self.fetch(at + 4.0) }
            return self.kernel_call(slot, vec3(x, y, z), a, b, c, d)
        }
        kernel_call: fn(slot: f32, p: vec3, a: vec4, b: vec4, c: vec4, d: vec4) -> f32 {
            return 1000000000.0
        }

        eval_program: fn(base: f32, x: f32, y: f32, z: f32) -> vec2 {
            var px = x
            var py = y
            var pz = z
            var vs = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var ss = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var sx8 = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var sy8 = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var sz8 = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var ms = array(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            var vn = 0i
            var pn = 0i
            var mn = 0i
            let len = self.fetch(base).x
            var pc = 1.0
            for instruction in 0..u32(len) {
                if pc >= len { break }
                let h = self.fetch(base + pc)
                let code = h.x
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
                    vs[vn] = self.mesh_at(u32(h.y), vec3(px, py, pz))
                    if mn > 0i { ss[vn] = ms[mn - 1i] } else { ss[vn] = 0.0 }
                    vn = vn + 1i
                    pc = pc + 1.0
                } else {
                    let e = self.links_at(u32(base) + asuint(h.y), vec3(px, py, pz))
                    vs[vn] = e.x
                    if e.y != 0.0 { ss[vn] = e.y }
                    else if mn > 0i { ss[vn] = ms[mn - 1i] } else { ss[vn] = 0.0 }
                    vn = vn + 1i
                    pc = pc + 1.0
                }
            }
            // + 0.0: a zero leaves as +0 (min/max of signed zeros differ between GPUs)
            return vec2(vs[0], ss[0])
        }
    }

    mod.sdf.Bake = {
        ..mod.sdf.Eval
        precise_math: true
        workgroup_size: 64
        // Two uint4s per entry: signed xyz bits + slot, then program base + reserved.
        bricks: shader.buffer(vec4u)
        atlas: shader.texture_2d_rw(@r16f)
        materials: shader.texture_2d_rw(@r8unorm)
        cell: shader.uniform(f32)
        band: shader.uniform(f32)
        first: shader.uniform(u32)
        count: shader.uniform(u32)

        // `v` rounded to the nearest f16 (ties to even), as an f32: the
        // R16F write then stores it exactly. Render-target conversion modes
        // differ between GPUs (a GPU-B truncates).
        to_half: fn(v: f32) -> f32 {
            let a = abs(v)
            if a < 0.00006103515625 {
                // the f16 subnormals: multiples of 2^-24
                let q = a * 16777216.0
                var r = floor(q)
                let f = q - r
                if f > 0.5 || (f == 0.5 && r - floor(r * 0.5) * 2.0 > 0.5) { r = r + 1.0 }
                let h = r * 0.000000059604644775390625
                if v < 0.0 { return 0.0 - h }
                return h
            }
            let b = asuint(v)
            let odd = (b >> 13u) & 1u
            return asfloat((b + 4095u + odd) & 4294959104u)
        }

        main: fn() {
            let entry = group_id().x
            if entry >= self.count { return }
            let at = (self.first + entry) * 2u
            let brick = self.bricks[at]
            let base = f32(self.bricks[at + 1u].x)
            let bx = i32(brick.x)
            let by = i32(brick.y)
            let bz = i32(brick.z)
            let col = brick.w % 50u
            let row = brick.w / 50u
            for k in 0..12u {
                let sample = local_index() + k * 64u
                if sample < 729u {
                    let x = sample % 9u
                    let z = (sample / 9u) % 9u
                    let y = sample / 81u
                    let e = self.eval_program(base,
                        f32(bx * 8i + i32(x)) * self.cell,
                        f32(by * 8i + i32(y)) * self.cell,
                        f32(bz * 8i + i32(z)) * self.cell)
                    let d = min(max(e.x, 0.0 - self.band), self.band) + 0.0
                    let texel = vec2u(col * 81u + z * 9u + x, row * 9u + y)
                    self.atlas.write(texel, self.to_half(d))
                    self.materials.write(texel, e.y / 255.0)
                }
            }
        }
    }
    // Force module assignments to commit before returning.
    mod.sdf
}
