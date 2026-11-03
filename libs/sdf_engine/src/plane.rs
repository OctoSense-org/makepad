//! Plane tracing: one part revision's batch of planes through its coarse grid (§10 of the
//! slicer's GPU execution model). Every SDF sample goes through one resumable sampler, so the
//! plane passes are flat, fixed-work kernels.
//!
//! - coarse: one sample per coarse node (`coarse_d`); a coarse cell is kept when its corners
//!   change sign or one is within the Lipschitz reach of a zero (`kept_off`, `kept`, `coarse_slot`);
//! - nodes: every kept slot samples its 121 fine nodes (`node_d`, slot*121 + j*11 + i); a node two
//!   slots share is one point through one program, so both read the same bits;
//! - crossings: one per sign-changing fine edge, numbered two-level: a slot's base (the scan of
//!   the slots' counts) plus its local number in edge order (`slot_cross_base`, `edge_local`); at
//!   most 12 bracket samples and 4 gradient samples (`cross_off`, `cross`, `cross_key`, `cross_g`);
//! - spans: one per crossing, numbered by its start crossing: a fine cell's crossings paired with
//!   the inside on the left (a saddle takes one centre sample); a turning span gets a curve middle
//!   (at most 4 samples) or a crease vertex (at most 3 cycles of 4 samples, then the curve), at
//!   most 16 samples in all;
//! - the outline FeatureSet: `seg` (a span's first at `span_seg`), `seg_info`, `seg_key`,
//!   `seg_plane_off`, `loop_off`, `loop_info`;
//! - [`item`]: the stage, then the loops packed into one readback, which [`decode_plane`] reads
//!   plane by plane.
//!
//! Units: grid units (0.1 mm = 1.0) relative to the part revision's origin. Ids and record
//! layouts are `plane_abi`'s. The sampler charges real costs (see [`cost`]): an op its ALU work,
//! a kernel its AIR cost, a mesh step the node or triangles it visits.
//!
//! The stage runs on the GPU job runtime (`gpu_jobs`) with two checkpoints, coarse and
//! crossings, where the next buffers are sized from exact counts. A bounded program (no mesh or
//! links walk, its cost within one round's budget) samples in plain counted passes; any other
//! resumes under `mod.sdf.Jobs`.
use makepad_draw::*;
use crate::{gpu_jobs::{label, BufRef, CountWord, Counts, Elements, JobError, Output, Pass, Pool, ScanKernels, Step},
    kernel::{ComputeBake, KernelSlot}, list_rank::{self, RankKernels, Succ}, plane_abi::NONE, program::{op, op_len, Program}};
use std::{collections::HashMap, rc::Rc};

/// `mod.sdf.Jobs`' work budget per slot and round.
pub const ROUND_BUDGET: u32 = 32768;
/// Slots of one resume round: the full-budget round proven within the 50 ms bound.
pub const RESUME_SLOTS: u32 = 1024;
/// Work units of one proven 50 ms job: a full-budget round of [`RESUME_SLOTS`].
const PROVEN_UNITS: u64 = RESUME_SLOTS as u64 * ROUND_BUDGET as u64;
/// Ops one sampler call steps through (its loop bound).
pub const CALL_OPS: u32 = 256;
/// Elements of one flat plane job: fixed work of a few dozen loads and stores each, far within
/// the 50 ms bound at this size.
pub const CEILING: u32 = 1 << 20;

/// Static work units (ALU-scale ops) the sampler charges, shared with its shader.
pub mod cost {
    pub const PRIM: u32 = 32;
    pub const XFORM_PUSH: u32 = 24;
    pub const POP: u32 = 4;
    pub const DEFORM_PUSH: u32 = 24;
    pub const FOLD: u32 = 16;
    pub const SMALL: u32 = 2;
    pub const START: u32 = 4;
    pub const TEX: u32 = 32;
    /// A BVH node whose box or far field ends the visit.
    pub const MESH_PRUNED: u32 = 16;
    /// An inner BVH node visited.
    pub const MESH_INNER: u32 = 24;
    /// Per triangle of a visited leaf (closest point, or solid angle).
    pub const MESH_TRIANGLE: u32 = 40;
    /// One links candidate.
    pub const LINKS_STEP: u32 = 32;
}

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    // One SDF sample per element: a point (x, y, z bits, dest), grid units x/y and printer mm z,
    // through the frame rows to shape space; values[dest] = the program's value, an exact zero
    // as -min normal (d <= 0 is inside). words = (the element count word, the source, a, b):
    // source 0 a request list, req[e]; 1 the coarse nodes, plane e / b at site[plane].x, node
    // e % b in rows of a; 2 the fine nodes, slot e / 121 at site[slot] = (plane, cx, cy, z bits),
    // node e % 121 in rows of 11. A bounded program runs in plain passes (dispatch_range.w = 1,
    // element first + i, no continuation); any other resumes under mod.sdf.Jobs, its slots
    // seeded up to the bound (past the count, done).
    mod.sdf.PlaneSample = {
        ..mod.sdf.Jobs
        precise_math: true
        workgroup_size: 64
        words: shader.uniform(vec4u)
        req: shader.buffer(vec4u)
        site: shader.buffer(vec4u)
        values: shader.buffer_rw(f32)
        frame: shader.buffer(vec4f)
        slot_costs: shader.buffer(u32)
        op_cost: fn(code: f32, h: vec4) -> u32 {
            if code < 0.5 { return u32(#(crate::plane::cost::PRIM)) }
            if code < 1.5 { return u32(#(crate::plane::cost::XFORM_PUSH)) }
            if code < 2.5 { return u32(#(crate::plane::cost::POP)) }
            if code < 3.5 { return u32(#(crate::plane::cost::DEFORM_PUSH)) }
            if code < 4.5 { return u32(#(crate::plane::cost::POP)) }
            if code < 5.5 { return u32(#(crate::plane::cost::FOLD)) }
            if code < 10.5 { return u32(#(crate::plane::cost::SMALL)) }
            if code < 11.5 { return max(self.slot_costs[u32(h.y)], 1u) }
            if code < 12.5 { return u32(#(crate::plane::cost::SMALL)) }
            if code < 15.5 { return u32(#(crate::plane::cost::START)) }
            if code < 16.5 { return u32(#(crate::plane::cost::TEX)) }
            return u32(#(crate::plane::cost::SMALL))
        }
        // One BVH node per step (validated leaves hold at most eight triangles):
        // (finished, value, cost, 0).
        mesh_step: fn(task: u32, slot: u32, p: vec3) -> vec4 {
            let h = self.mesh_read(slot, 0u)
            let state = task + 144u
            var phase = self.continuations[state]
            var count = self.continuations[state + 1u]
            var best = asfloat(self.continuations[state + 2u])
            var winding = asfloat(self.continuations[state + 3u])
            if count == 0u {
                if phase == 1u {
                    if winding > 0.5 { best = asfloat(asuint(best) ^ 2147483648u) }
                    return vec4(1.0, best, f32(#(crate::plane::cost::START)), 0.0)
                }
                phase = 1u; count = 1u; self.continuations[task + 160u] = 0u
            }
            if h.x == 0u { return vec4(1.0, best, f32(#(crate::plane::cost::START)), 0.0) }
            count -= 1u
            let node = self.continuations[task + 160u + count]
            let at = 1u + node * 4u
            var visit = true
            var spent = u32(#(crate::plane::cost::MESH_PRUNED))
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
                        spent += u32(#(crate::plane::cost::MESH_TRIANGLE))
                    }
                } else {
                    var a = left; var b = right
                    if phase == 0u {
                        if self.mesh_box(slot, left, p) < self.mesh_box(slot, right, p) { a = right; b = left }
                    }
                    self.continuations[task + 160u + count] = a
                    self.continuations[task + 161u + count] = b
                    count += 2u
                    spent = u32(#(crate::plane::cost::MESH_INNER))
                }
            }
            self.continuations[state] = phase; self.continuations[state + 1u] = count
            self.continuations[state + 2u] = asuint(best); self.continuations[state + 3u] = asuint(winding)
            return vec4(0.0, 0.0, f32(spent), 0.0)
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
        // The program at (x, y, z) under this invocation's budget: (finished, value). An
        // unfinished sample keeps its interpreter state in the continuation at `task` and
        // resumes there; a plain pass (task NONE) runs a bounded program to its end.
        field: fn(task: u32, x: f32, y: f32, zz: f32) -> vec2 {
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
            var phase = 1u
            var budget = 4294967295u
            if task != 4294967295u {
                phase = self.continuations[task + 64u]
                budget = self.continuations[task + 63u]
            }
            if phase == 0u { phase = 1u }
            else if task != 4294967295u {
                pc = asfloat(self.continuations[task + 65u])
                px = asfloat(self.continuations[task + 66u]); py = asfloat(self.continuations[task + 67u]); pz = asfloat(self.continuations[task + 68u])
                vn = i32(self.continuations[task + 69u]); pn = i32(self.continuations[task + 70u]); mn = i32(self.continuations[task + 71u])
                for i in 0..16u { vs[i] = asfloat(self.continuations[task + 80u + i]); ss[i] = asfloat(self.continuations[task + 96u + i]) }
                for i in 0..8u {
                    sx8[i] = asfloat(self.continuations[task + 112u + i]); sy8[i] = asfloat(self.continuations[task + 120u + i])
                    sz8[i] = asfloat(self.continuations[task + 128u + i]); ms[i] = asfloat(self.continuations[task + 136u + i])
                }
            }
            for instruction in 0..u32(#(crate::plane::CALL_OPS)) {
                if pc >= len || budget == 0u { break }
                let h = self.fetch(base + pc)
                let code = h.x
                var spent = 0u
                if phase == 2u {
                    let value = self.mesh_step(task, u32(h.y), vec3(px, py, pz))
                    spent = u32(value.z)
                    if value.x != 0.0 {
                        vs[vn] = value.y
                        if mn > 0i { ss[vn] = ms[mn - 1i] } else { ss[vn] = 0.0 }
                        vn += 1i; pc += 1.0; phase = 1u
                    }
                } else if phase == 3u {
                    let value = self.links_step(task, u32(base) + asuint(h.y), vec3(px, py, pz))
                    spent = u32(#(crate::plane::cost::LINKS_STEP))
                    if value.x != 0.0 {
                        vs[vn] = value.y
                        if value.z != 0.0 { ss[vn] = value.z }
                        else if mn > 0i { ss[vn] = ms[mn - 1i] } else { ss[vn] = 0.0 }
                        vn += 1i; pc += 1.0; phase = 1u
                    }
                } else {
                    spent = self.op_cost(code, h)
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
                        let x0 = px
                        let y0 = py
                        let z0 = pz
                        if h.y < 0.5 {
                            let e = self.psc(h.z * y0)
                            px = e.y * x0 + e.x * z0
                            pz = e.y * z0 - e.x * x0
                        } else if h.y < 1.5 {
                            let e = self.psc(h.z * x0)
                            px = e.y * x0 + e.x * y0
                            py = e.y * y0 - e.x * x0
                        } else if h.y < 2.5 {
                            let k = 1.0 / max(1.0 + h.z * y0, 0.001)
                            px = x0 * k
                            pz = z0 * k
                        } else if h.w < 0.5 {
                            px = abs(x0)
                        } else if h.w < 1.5 {
                            py = abs(y0)
                        } else {
                            pz = abs(z0)
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
                        let a = self.fetch(base + pc + 1.0)
                        let m = self.fetch(base + pc + 2.0)
                        var np = 1i
                        if h.y > 0.5 { np = 3i }
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
                        // GRAD or PAT: h.y texels follow that known masks skip
                        let o = u32(h.z)
                        var skip = false
                        if o < 16u && ((masks >> o) & 1u) != 0u {
                            skip = code < 17.5 || ((masks >> (o + 16u)) & 1u) == 0u
                        }
                        pc = pc + 1.0
                        if skip { pc = pc + h.y }
                    }
                }
                budget -= min(spent, budget)
            }
            if task == 4294967295u {
                if pc >= len { return vec2(1.0, vs[0u]) }
                return vec2(0.0)
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
        // Request e at continuation `task` (NONE in a plain pass): whether its value is written.
        point: fn(e: u32) -> vec4u {
            if self.words.y == 0u { return self.req[e] }
            if self.words.y == 1u {
                let k = e % self.words.w
                return vec4u(asuint(f32((k % self.words.z) * 10u)), asuint(f32((k / self.words.z) * 10u)), self.site[e / self.words.w].x, e)
            }
            let si = self.site[e / 121u]
            let n = e % 121u
            return vec4u(asuint(f32(si.y * 10u + n % 11u)), asuint(f32(si.z * 10u + n / 11u)), si.w, e)
        }
        sample: fn(e: u32, task: u32) -> bool {
            let q = self.point(e)
            let gx = asfloat(q.x); let gy = asfloat(q.y); let z = asfloat(q.z)
            let a = self.frame[0u]; let b = self.frame[1u]; let c = self.frame[2u]
            let x = ((a.x * gx + a.y * gy) + a.z * z) + a.w
            let y = ((b.x * gx + b.y * gy) + b.z * z) + b.w
            let zz = ((c.x * gx + c.y * gy) + c.z * z) + c.w
            let v = self.field(task, x, y, zz)
            if v.x == 0.0 { return false }
            var d = v.y
            if d == 0.0 { d = 0.0 - asfloat(8388608u) }
            self.values[q.w] = d
            return true
        }
        main: fn() {
            if self.dispatch_range.w == 1u {
                if global_id().x >= self.dispatch_range.y { return }
                let e = self.dispatch_range.x + global_id().x
                if e >= self.counts[self.words.x] { return }
                if self.sample(e, 4294967295u) { return }
                return
            }
            if self.control() { return }
            let e = self.index()
            if e == 4294967295u { return }
            if !self.begin() { return }
            if e >= self.counts[self.words.x] { self.done(); return }
            if self.sample(e, self.task()) { self.done() }
        }
    }

    // The plane passes' common frame. dispatch_range = (first, count, z, w); batch = (nx, ny,
    // planes, 0); words = the count words a pass reads or writes.
    mod.sdf.Plane = {
        precise_math: true
        workgroup_size: 64
        dispatch_range: shader.uniform(vec4u)
        batch: shader.uniform(vec4u)
        words: shader.uniform(vec4u)
        counts: shader.buffer_rw(u32)
        // Per kept slot: (plane, coarse cx, cy, z bits).
        slot_info: shader.buffer(vec4u)
        element: fn() -> u32 {
            if global_id().x >= self.dispatch_range.y { return 4294967295u }
            return self.dispatch_range.x + global_id().x
        }
        // The element of a pass whose count is a count word, or NONE.
        counted: fn(word: u32) -> u32 {
            let e = self.element()
            if e == 4294967295u || e >= self.counts[word] { return 4294967295u }
            return e
        }
        // batch.w: the kept count word.
        slots: fn() -> u32 { return self.counts[self.batch.w] }
        slot_z: fn(slot: u32) -> f32 { return asfloat(self.slot_info[slot].w) }
        // A span's base seg count by its final mode: straight 1, middle, vertex or refined middle
        // 2, zero length 0 (a refined middle takes 4 where its plane has the room, `seg_n`).
        seg_count: fn(mode: f32) -> u32 {
            if mode < 10.5 { return 1u }
            if mode > 12.5 && mode < 13.5 { return 0u }
            return 2u
        }
        // Local edge `le`: (i, j, axis, local node of its low end); the other end is +1 or +11.
        edge_nodes: fn(le: u32) -> vec4u {
            if le < 110u { return vec4u(le % 10u, le / 10u, 0u, (le / 10u) * 11u + le % 10u) }
            let m = le - 110u
            return vec4u(m % 11u, m / 11u, 1u, m)
        }
        // plane_abi's fine-grid ids (§10.1): a node gy * NX + gx, a cell the node id of its low
        // corner, an edge 2 * node + axis.
        node_id: fn(gx: u32, gy: u32) -> u32 { return gy * (self.batch.x * 10u + 1u) + gx }
        cell_id: fn(fx: u32, fy: u32) -> u32 { return self.node_id(fx, fy) }
        edge_id: fn(gx: u32, gy: u32, axis: u32) -> u32 { return 2u * self.node_id(gx, gy) + axis }
        // A point at parameter t along the fine edge from node (gx, gy) on `axis`.
        edge_point: fn(gx: u32, gy: u32, axis: u32, t: f32) -> vec2 {
            if axis == 0u { return vec2(f32(gx) + t, f32(gy)) }
            return vec2(f32(gx), f32(gy) + t)
        }
        // The regula falsi point of a bracket (lo, hi, f(lo), f(hi)), kept off its ends.
        secant: fn(br: vec4) -> f32 {
            let w = br.y - br.x
            let t = br.x + w * (br.z / (br.z - br.w))
            return clamp(t, br.x + 0.001 * w, br.y - 0.001 * w)
        }
        cross2: fn(a: vec2, b: vec2) -> f32 { return a.x * b.y - a.y * b.x }
    }

    // The kernels that resolve shared nodes and edges to their owner slots.
    mod.sdf.PlaneOwned = {
        ..mod.sdf.Plane
        coarse_slot: shader.buffer(u32)
        node_d: shader.buffer(f32)
        // Whether local edge `le` of `slot` joins an inside node to an outside one.
        edge_sign_change: fn(slot: u32, le: u32) -> bool {
            let en = self.edge_nodes(le)
            let a = self.node_d[slot * 121u + en.w]
            var b = self.node_d[slot * 121u + en.w + 1u]
            if en.z == 1u { b = self.node_d[slot * 121u + en.w + 11u] }
            return (a <= 0.0) != (b <= 0.0)
        }
        slot_at: fn(p: u32, cx: u32, cy: u32) -> u32 {
            return self.coarse_slot[(p * self.batch.y + cy) * self.batch.x + cx]
        }
            // (slot, local edge) of the kept cell with the smallest cell id holding local edge `le`.
        edge_owner: fn(p: u32, cx: u32, cy: u32, slot: u32, le: u32) -> vec4u {
            if le < 10u && cy > 0u {
                let s0 = self.slot_at(p, cx, cy - 1u)
                if s0 != 4294967295u { return vec4u(s0, 100u + le, 0u, 0u) }
            }
            if le >= 110u && (le - 110u) % 11u == 0u && cx > 0u {
                let s1 = self.slot_at(p, cx - 1u, cy)
                if s1 != 4294967295u { return vec4u(s1, le + 10u, 0u, 0u) }
            }
            return vec4u(slot, le, 0u, 0u)
        }
    }

    // The crossings by edge, two-level: a slot's owned sign-changing edges are numbered from its
    // base (slot_cross_base, the scan of the slots' counts); edge_local holds each local edge's
    // number + 1 in a byte (0: no crossing it owns), 55 words a slot.
    mod.sdf.PlaneEdges = {
        ..mod.sdf.PlaneOwned
        edge_local: shader.buffer(u32)
        slot_cross_base: shader.buffer(u32)
        local_of: fn(slot: u32, le: u32) -> u32 { return (self.edge_local[slot * 55u + le / 4u] >> ((le % 4u) * 8u)) & 255u }
        // The crossing on local edge `le` of `slot`, or NONE.
        cross_of: fn(slot: u32, le: u32) -> u32 {
            let si = self.slot_info[slot]
            let o = self.edge_owner(si.x, si.y, si.z, slot, le)
            let n = self.local_of(o.x, o.y)
            if n == 0u { return 4294967295u }
            return self.slot_cross_base[o.x] + n - 1u
        }
    }

    mod.sdf.PlaneKeep = {
        ..mod.sdf.Plane
        coarse_d: shader.buffer(f32)
        frame: shader.buffer(vec4f)
        flags: shader.buffer_rw(u32)
        main: fn() {
            let e = self.element()
            let nx = self.batch.x
            let cells = nx * self.batch.y
            if e == 4294967295u || e >= cells * self.batch.z { return }
            let c = e % cells
            let at = (e / cells) * (nx + 1u) * (self.batch.y + 1u) + (c / nx) * (nx + 1u) + c % nx
            let a = self.coarse_d[at]
            let b = self.coarse_d[at + 1u]
            let d = self.coarse_d[at + nx + 1u]
            let f = self.coarse_d[at + nx + 2u]
            let lo = min(min(a, b), min(d, f))
            let hi = max(max(a, b), max(d, f))
            let near = min(min(abs(a), abs(b)), min(abs(d), abs(f)))
            // Every point of the cell is within half its diagonal (sqrt(50) units) of a corner.
            var keep = 0u
            if (lo <= 0.0 && hi > 0.0) || near <= self.frame[3u].x * 7.08 { keep = 1u }
            self.flags[e] = keep
        }
    }

    mod.sdf.PlaneKept = {
        ..mod.sdf.Plane
        flags: shader.buffer(u32)
        offs: shader.buffer(u32)
        planes: shader.buffer(vec4f)
        kept: shader.buffer_rw(u32)
        coarse_slot: shader.buffer_rw(u32)
        kept_off: shader.buffer_rw(u32)
        slot_info: shader.buffer_rw(vec4u)
        // words: x the kept total, y the node count (121 a slot)
        main: fn() {
            let e = self.element()
            let cells = self.batch.x * self.batch.y
            if e == 4294967295u || e >= cells * self.batch.z { return }
            let o = self.offs[e]
            let c = e % cells
            if self.flags[e] != 0u {
                self.kept[o] = c
                self.coarse_slot[e] = o
                self.slot_info[o] = vec4u(e / cells, c % self.batch.x, c / self.batch.x, asuint(self.planes[e / cells].x))
            }
            else { self.coarse_slot[e] = 4294967295u }
            if e % cells == 0u { self.kept_off[e / cells] = o }
            if e == 0u {
                self.kept_off[self.batch.z] = self.counts[self.words.x]
                self.counts[self.words.y] = self.counts[self.words.x] * 121u
            }
        }
    }

    // A slot's owned sign-changing edges, numbered in local edge order (edge_local), and their
    // count (slot_cross). Every slot samples all its 121 nodes: a node two slots share reads the
    // same bits in both (one point, one program), so its edges agree whichever slot reads them.
    mod.sdf.PlaneEdgeCount = {
        ..mod.sdf.PlaneOwned
        edge_local: shader.buffer_rw(u32)
        slot_cross: shader.buffer_rw(u32)
        main: fn() {
            let slot = self.element()
            if slot == 4294967295u || slot >= self.slots() { return }
            let si = self.slot_info[slot]
            var n = 0u
            var word = 0u
            for le in 0..220u {
                if self.edge_owner(si.x, si.y, si.z, slot, le).x == slot && self.edge_sign_change(slot, le) {
                    n += 1u
                    word = word | (n << ((le % 4u) * 8u))
                }
                if le % 4u == 3u {
                    self.edge_local[slot * 55u + le / 4u] = word
                    word = 0u
                }
            }
            self.slot_cross[slot] = n
        }
    }

    // A slot's crossings: cross_bs[2x] = the bracket (lo, hi, f(lo), f(hi)), Illinois-scaled;
    // cross_bs[2x + 1] = the state (t, last side, state, |f| sum of the edge's nodes); cross_geo
    // = (gx, gy, axis, slot); the first bracket round's request x for every crossing x.
    mod.sdf.PlaneCrossInit = {
        ..mod.sdf.PlaneEdges
        req: shader.buffer_rw(vec4u)
        cross_geo: shader.buffer_rw(vec4u)
        cross_bs: shader.buffer_rw(vec4f)
        // words: x crossings, y the first bracket round's request count
        main: fn() {
            let slot = self.element()
            if slot == 4294967295u || slot >= self.slots() { return }
            if slot == 0u { self.counts[self.words.y] = self.counts[self.words.x] }
            let si = self.slot_info[slot]
            let base = self.slot_cross_base[slot]
            let zero = 0.0 - asfloat(8388608u)
            for le in 0..220u {
                let n = self.local_of(slot, le)
                if n != 0u {
                    let x = base + n - 1u
                    let en = self.edge_nodes(le)
                    let gx = si.y * 10u + en.x
                    let gy = si.z * 10u + en.y
                    self.cross_geo[x] = vec4u(gx, gy, en.z, slot)
                    let a = self.node_d[slot * 121u + en.w]
                    var b = self.node_d[slot * 121u + en.w + 1u]
                    if en.z == 1u { b = self.node_d[slot * 121u + en.w + 11u] }
                    let br = vec4(0.0, 1.0, a, b)
                    self.cross_bs[2u * x] = br
                    var st = vec4(self.secant(br), 0.0, 0.0, abs(a) + abs(b))
                    // A node exactly on the zero (-min normal) first probes 0.001 inside the edge:
                    // state 2 with the node's end in y.
                    if a == zero { st = vec4(0.001, 0.0, 2.0, st.w) }
                    if b == zero { st = vec4(0.999, 1.0, 2.0, st.w) }
                    self.cross_bs[2u * x + 1u] = st
                    let q = self.edge_point(gx, gy, en.z, st.x)
                    self.req[x] = vec4u(asuint(q.x), asuint(q.y), si.w, x)
                }
            }
        }
    }

    // One bracket round: the sample at the state's t narrows the bracket (Illinois), then the next
    // request, until the bracket is under 1e-3 units. A value within 1e-4 of the edge's ends no
    // search by itself: a face in the plane leaves the field flat at zero over an area, where any
    // point reads near zero, and the crossing is where the inside starts. So a near-zero value is
    // checked by a probe 5e-4 toward the other end (state 4): a sign change there closes the
    // bracket; none means the field is flat here, and the bracket halves from then on (state 3).
    // st = (t, side of the last sample, state, |f| scale): state 0 secant, 1 done, 2 the probe
    // beside an exact-zero node, 3 halving, 4 the near-zero probe.
    mod.sdf.PlaneCrossStep = {
        ..mod.sdf.Plane
        cross_geo: shader.buffer(vec4u)
        cross_v: shader.buffer(f32)
        req: shader.buffer_rw(vec4u)
        cross_bs: shader.buffer_rw(vec4f)
        // words: x crossings, y the next round's request count (NONE on the last round)
        main: fn() {
            let x = self.counted(self.words.x)
            if x == 4294967295u { return }
            var st = self.cross_bs[2u * x + 1u]
            if st.z > 0.5 && st.z < 1.5 { return }
            let v = self.cross_v[x]
            var br = self.cross_bs[2u * x]
            let last = self.words.y == 4294967295u
            let near = abs(v) <= 0.0001 * st.w
            if st.z > 1.5 && st.z < 2.5 {
                // The probe beside an exact-zero node: outside, and the node alone is the crossing;
                // inside, and the crossing lies between the probe and the outside end.
                if v > 0.0 { st = vec4(st.y, 0.0, 1.0, st.w) }
                else {
                    if st.y > 0.5 { br.y = st.x; br.w = v } else { br.x = st.x; br.z = v }
                    st = vec4(self.secant(br), 0.0, 0.0, st.w)
                    if near { st = vec4((br.x + br.y) * 0.5, 0.0, 3.0, st.w) }
                    if last { st.x = (br.x + br.y) * 0.5; st.z = 1.0 }
                }
            } else {
                var hi = false
                if (v <= 0.0) == (br.z <= 0.0) {
                    br.x = st.x; br.z = v
                    if st.y < 0.0 { br.w = br.w * 0.5 }
                    st.y = 0.0 - 1.0
                } else {
                    br.y = st.x; br.w = v
                    if st.y > 0.0 { br.z = br.z * 0.5 }
                    st.y = 1.0
                    hi = true
                }
                if br.y - br.x <= 0.001 || last { st.x = (br.x + br.y) * 0.5; st.z = 1.0 }
                else if st.z > 2.5 { st.x = (br.x + br.y) * 0.5; st.z = 3.0 }
                else if near {
                    if hi { st.x = br.y - 0.0005 } else { st.x = br.x + 0.0005 }
                    st.z = 4.0
                }
                else { st.x = self.secant(br) }
            }
            self.cross_bs[2u * x] = br
            self.cross_bs[2u * x + 1u] = st
            if st.z < 0.5 || st.z > 1.5 {
                let g = self.cross_geo[x]
                let q = self.edge_point(g.x, g.y, g.z, st.x)
                let r = atomic_add(self.counts[self.words.y], 1u)
                self.req[r] = vec4u(asuint(q.x), asuint(q.y), self.slot_info[g.w].w, x)
            }
        }
    }

    // Four central-difference samples around each crossing (requests 4x..4x+3).
    mod.sdf.PlaneCrossGrad = {
        ..mod.sdf.Plane
        cross_geo: shader.buffer(vec4u)
        cross_bs: shader.buffer(vec4f)
        req: shader.buffer_rw(vec4u)
        // words: x crossings, y the request count
        main: fn() {
            let x = self.counted(self.words.x)
            if x == 4294967295u { return }
            if x == 0u { self.counts[self.words.y] = 4u * self.counts[self.words.x] }
            let g = self.cross_geo[x]
            let q = self.edge_point(g.x, g.y, g.z, self.cross_bs[2u * x + 1u].x)
            let z = self.slot_info[g.w].w
            self.req[4u * x] = vec4u(asuint(q.x + 0.01), asuint(q.y), z, 4u * x)
            self.req[4u * x + 1u] = vec4u(asuint(q.x - 0.01), asuint(q.y), z, 4u * x + 1u)
            self.req[4u * x + 2u] = vec4u(asuint(q.x), asuint(q.y + 0.01), z, 4u * x + 2u)
            self.req[4u * x + 3u] = vec4u(asuint(q.x), asuint(q.y - 0.01), z, 4u * x + 3u)
        }
    }

    // cross = (x, y, unit outward gradient); cross_g = |gradient| in d per grid unit; cross_key =
    // the edge's key (2 * node + axis).
    mod.sdf.PlaneCross = {
        ..mod.sdf.Plane
        cross_geo: shader.buffer(vec4u)
        cross_bs: shader.buffer(vec4f)
        cross_gv: shader.buffer(f32)
        cross: shader.buffer_rw(vec4f)
        cross_g: shader.buffer_rw(f32)
        cross_key: shader.buffer_rw(u32)
        main: fn() {
            let x = self.counted(self.words.x)
            if x == 4294967295u { return }
            let g = self.cross_geo[x]
            self.cross_key[x] = self.edge_id(g.x, g.y, g.z)
            let q = self.edge_point(g.x, g.y, g.z, self.cross_bs[2u * x + 1u].x)
            // The offsets the samples actually took, exactly (Sterbenz).
            let dx = (q.x + 0.01 - q.x) + (q.x - (q.x - 0.01))
            let dy = (q.y + 0.01 - q.y) + (q.y - (q.y - 0.01))
            let grad = vec2((self.cross_gv[4u * x] - self.cross_gv[4u * x + 1u]) / dx, (self.cross_gv[4u * x + 2u] - self.cross_gv[4u * x + 3u]) / dy)
            let gl = length(grad)
            var n = vec2(0.0)
            if gl > 0.000000000000000001 { n = grad / gl }
            else {
                // A flat field here: outward along the edge, from its inside end.
                var s = 1.0
                if self.cross_bs[2u * x].z > 0.0 { s = 0.0 - 1.0 }
                if g.z == 0u { n = vec2(s, 0.0) } else { n = vec2(0.0, s) }
            }
            self.cross[x] = vec4(q.x, q.y, n.x, n.y)
            self.cross_g[x] = gl
        }
    }

    // A saddle fine cell (four crossings) requests its centre, dest its bottom edge's crossing
    // (a cell's own: the cell below holds that edge as its top).
    mod.sdf.PlaneCellSpans = {
        ..mod.sdf.PlaneEdges
        req: shader.buffer_rw(vec4u)
        // words.x: the centre request count
        main: fn() {
            let e = self.element()
            if e == 4294967295u || e >= self.slots() * 100u { return }
            let slot = e / 100u
            let n = e % 100u
            let i = n % 10u
            let j = n / 10u
            let bottom = self.cross_of(slot, j * 10u + i)
            if bottom == 4294967295u || self.cross_of(slot, (j + 1u) * 10u + i) == 4294967295u { return }
            if self.cross_of(slot, 110u + j * 11u + i) == 4294967295u || self.cross_of(slot, 110u + j * 11u + i + 1u) == 4294967295u { return }
            let si = self.slot_info[slot]
            let x = f32(si.y * 10u + i) + 0.5
            let y = f32(si.z * 10u + j) + 0.5
            let r = atomic_add(self.counts[self.words.x], 1u)
            self.req[r] = vec4u(asuint(x), asuint(y), si.w, bottom)
        }
    }

    // A span per crossing, numbered by its start crossing: span_src = (fine cell, start crossing,
    // end crossing, pair | start edge << 8 | end edge << 16); cross_end = the span a crossing ends.
    // Walking the cell boundary counterclockwise (bottom, right, top, left), a span starts where
    // the walk leaves the inside and ends where it re-enters: the inside is on the span's left.
    // A saddle pairs a start with the next end when its centre is inside, else the previous one.
    mod.sdf.PlaneSpanSrc = {
        ..mod.sdf.PlaneEdges
        cross_c: shader.buffer(f32)
        span_src: shader.buffer_rw(vec4u)
        cross_end: shader.buffer_rw(u32)
        put: fn(src: vec4u) {
            self.span_src[src.y] = src
            self.cross_end[src.z] = src.y
        }
        main: fn() {
            let e = self.element()
            if e == 4294967295u || e >= self.slots() * 100u { return }
            let slot = e / 100u
            let n = e % 100u
            let i = n % 10u
            let j = n / 10u
            let v = slot * 121u + j * 11u + i
            let corner = array(self.node_d[v], self.node_d[v + 1u], self.node_d[v + 12u], self.node_d[v + 11u])
            let ids = array(self.cross_of(slot, j * 10u + i), self.cross_of(slot, 110u + j * 11u + i + 1u),
                self.cross_of(slot, (j + 1u) * 10u + i), self.cross_of(slot, 110u + j * 11u + i))
            var m = 0u
            var first_start = 4u
            var first_end = 4u
            for k in 0..4u {
                if ids[k] != 4294967295u {
                    m += 1u
                    let starts = corner[k] <= 0.0 && corner[(k + 1u) % 4u] > 0.0
                    if starts && first_start == 4u { first_start = k }
                    if !starts && first_end == 4u { first_end = k }
                }
            }
            if m == 0u { return }
            if m == 2u {
                self.put(vec4u(e, ids[first_start], ids[first_end], (first_start << 8u) | (first_end << 16u)))
                return
            }
            // A saddle: starts at k and k + 2.
            var step = 3u
            if self.cross_c[ids[0]] <= 0.0 { step = 1u }
            let s0 = first_start
            let s1 = (first_start + 2u) % 4u
            let e0 = (s0 + step) % 4u
            let e1 = (s1 + step) % 4u
            if e0 < e1 {
                self.put(vec4u(e, ids[s0], ids[e0], (s0 << 8u) | (e0 << 16u)))
                self.put(vec4u(e, ids[s1], ids[e1], 1u | (s1 << 8u) | (e1 << 16u)))
            } else {
                self.put(vec4u(e, ids[s1], ids[e1], (s1 << 8u) | (e1 << 16u)))
                self.put(vec4u(e, ids[s0], ids[e0], 1u | (s0 << 8u) | (e0 << 16u)))
            }
        }
    }

    // One round of every span: consume the last round's samples (span_v, 4 per span), then
    // request this round's. span_m = (vertex x, y, mode, cycle or iteration); modes: 1 curve,
    // 2 crease, 5 and 6 the halves; final: 10 straight, 11 curve middle, 12 crease vertex,
    // 13 zero length, 14 curve middle with its halves' middles. span_a: curve (u before, f before,
    // u now, samples before the curve); crease (lambda A, lambda B, 0, 0); halves 6 (f, u of the
    // left half's middle, f, u of the right's); final 14 (left middle x, y, right middle x, y).
    mod.sdf.PlaneSpanStep = {
        ..mod.sdf.Plane
        span_src: shader.buffer(vec4u)
        cross: shader.buffer(vec4f)
        cross_g: shader.buffer(f32)
        span_v: shader.buffer(f32)
        req: shader.buffer_rw(vec4u)
        span_m: shader.buffer_rw(vec4f)
        span_a: shader.buffer_rw(vec4f)
        // words: x spans, y this round's request count (NONE on the last round), z 1 on the first round
        ask: fn(s: u32, k: u32, q: vec2, z: f32) {
            let r = atomic_add(self.counts[self.words.y], 1u)
            self.req[r] = vec4u(asuint(q.x), asuint(q.y), asuint(z), 4u * s + k)
        }
        finish: fn(s: u32, m: vec2, mode: f32) { self.span_m[s] = vec4(m.x, m.y, mode, 0.0) }
        // The fine cell's square, grown by `by` units.
        inside_cell: fn(f: u32, q: vec2, by: f32) -> bool {
            let si = self.slot_info[f / 100u]
            let x = f32(si.y * 10u + (f % 100u) % 10u)
            let y = f32(si.z * 10u + (f % 100u) / 10u)
            return q.x >= x - by && q.x <= x + 1.0 + by && q.y >= y - by && q.y <= y + 1.0 + by
        }
        // Where a crease vertex may lie: up to three cells outside its span's cell. Where a face
        // runs along a grid line its nodes read zero, and a narrow wedge crosses an edge twice,
        // so the run to the vertex (a corner, a notch's or wedge's tip) crosses no sign change.
        in_reach: fn(f: u32, q: vec2) -> bool { return self.inside_cell(f, q, 3.0) }
        // Start the curve middle: the circular arc's sagitta on the chord's bisector (none for a
        // straight span, whose middle sample checks that nothing narrower than a cell, a notch
        // between two edges, leaves the chord). span_m.xy = the bracket: where f was last seen
        // outside and inside along the bisector (+-1e6 while unseen).
        // The circular arc's middle on the chord's bisector (toward the inside): from the
        // tangents' turn, the arc through both crossings with those tangents.
        sagitta: fn(a: vec4, b: vec4) -> f32 {
            let ta = vec2(0.0 - a.w, a.z)
            let tb = vec2(0.0 - b.w, b.z)
            let t2 = self.cross2(ta, tb) / max(1.0 + dot(ta, tb), 0.000001)
            let t4 = t2 / (1.0 + sqrt(1.0 + t2 * t2))
            return 0.0 - 0.5 * length(b.xy - a.xy) * t4
        }
        curve: fn(s: u32, a: vec4, b: vec4, z: f32, used: f32) {
            let c = b.xy - a.xy
            let nu = vec2(0.0 - c.y, c.x) / length(c)
            let u = self.sagitta(a, b)
            let m = (a.xy + b.xy) * 0.5
            self.span_m[s] = vec4(0.0 - 1000000.0, 1000000.0, 1.0, 0.0)
            self.span_a[s] = vec4(0.0, 0.0, u, used)
            if self.words.y == 4294967295u { self.finish(s, m, 10.0); return }
            self.ask(s, 0u, m + nu * u, z)
        }
        // One Newton step from f along a normal where the field's slope is `slope` (negative
        // inward; at least a quarter of |grad| `g`), within `lim` units.
        newton: fn(f: f32, slope: f32, g: f32, lim: f32) -> f32 {
            if abs(f) <= 0.001 * g { return 0.0 }
            let u = (0.0 - f) / min(slope, 0.0 - 0.25 * g)
            return clamp(u, 0.0 - lim, lim)
        }
        // The secant step from (0, f0) and (u1, f1), within `lim` units; u1 when f1 is the zero.
        secant_u: fn(f0: f32, u1: f32, f1: f32, g: f32, lim: f32) -> f32 {
            if abs(f1) <= 0.001 * g || f1 == f0 { return u1 }
            return clamp(u1 - f1 * u1 / (f1 - f0), 0.0 - lim, lim)
        }
        // A side's curvature from its probe at `lambda` along the tangent: f there (f0) and 0.05
        // units outward (f1) give the zero's offset off the tangent by the field's own slope (a
        // field that is no distance has its own |grad| there), and the circle through the
        // crossing with that tangent and that zero (exact for circles). An offset under 0.002
        // units is the crossings' and the field's noise: a straight side.
        side: fn(f0: f32, f1: f32, g: f32, lambda: f32) -> f32 {
            var slope = (f1 - f0) / 0.05
            if slope < 0.25 * g { slope = g }
            let delta = f0 / slope
            if abs(delta) < 0.002 { return 0.0 }
            return 2.0 * delta / (lambda * lambda + delta * delta)
        }
        // Where side A (through a, tangent ta, curvature ka) meets side B (through b, tangent tb
        // ahead, curvature kb), nearest `near`: (x, y, 1) or (0, 0, 0). Positive curvature turns left.
        meet: fn(a: vec2, ta: vec2, ka: f32, b: vec2, tb: vec2, kb: f32, near: vec2) -> vec3 {
            let pa = a - near
            let pb = b - near
            if ka == 0.0 && kb == 0.0 {
                let det = self.cross2(ta, tb)
                if abs(det) < 0.000001 { return vec3(0.0) }
                let al = self.cross2(pb - pa, tb) / det
                let q = pa + ta * al
                return vec3(q.x + near.x, q.y + near.y, 1.0)
            }
            if ka == 0.0 || kb == 0.0 {
                // The line through p (direction t) and the circle (centre o, radius r).
                var p = pa; var t = ta; var o = pb + vec2(0.0 - tb.y, tb.x) / kb; var r = abs(1.0 / kb)
                if kb == 0.0 { p = pb; t = tb; o = pa + vec2(0.0 - ta.y, ta.x) / ka; r = abs(1.0 / ka) }
                let w = p - o
                let hb = dot(t, w)
                let disc = hb * hb - (dot(w, w) - r * r)
                if disc < 0.0 { return vec3(0.0) }
                let root = sqrt(disc)
                let q0 = p + t * (0.0 - hb - root)
                let q1 = p + t * (0.0 - hb + root)
                var q = q0
                if dot(q1, q1) < dot(q0, q0) { q = q1 }
                return vec3(q.x + near.x, q.y + near.y, 1.0)
            }
            let oa = pa + vec2(0.0 - ta.y, ta.x) / ka
            let ob = pb + vec2(0.0 - tb.y, tb.x) / kb
            let ra = abs(1.0 / ka)
            let rb = abs(1.0 / kb)
            let dv = ob - oa
            let d = length(dv)
            if d < 0.000001 || d > ra + rb || d < abs(ra - rb) { return vec3(0.0) }
            let u = dv / d
            let along = (ra * ra - rb * rb + d * d) / (2.0 * d)
            let h = sqrt(max(ra * ra - along * along, 0.0))
            let mid = oa + u * along
            let perp = vec2(0.0 - u.y, u.x)
            let q0 = mid + perp * h
            let q1 = mid - perp * h
            var q = q0
            if dot(q1, q1) < dot(q0, q0) { q = q1 }
            return vec3(q.x + near.x, q.y + near.y, 1.0)
        }
        // Request a crease cycle toward vertex v: on each side the point halfway along its
        // tangent to v, and 0.05 units outward of it.
        crease: fn(s: u32, a: vec4, b: vec4, v: vec2, cycle: f32, z: f32) -> bool {
            let ta = vec2(0.0 - a.w, a.z)
            let tb = vec2(0.0 - b.w, b.z)
            let da = dot(v - a.xy, ta)
            let db = dot(b.xy - v, tb)
            if da <= 0.0 || db <= 0.0 || self.words.y == 4294967295u { return false }
            let la = 0.5 * da
            let lb = 0.5 * db
            self.span_m[s] = vec4(v.x, v.y, 2.0, cycle)
            self.span_a[s] = vec4(la, lb, 0.0, 0.0)
            let pa = a.xy + ta * la
            let pb = b.xy - tb * lb
            self.ask(s, 0u, pa, z)
            self.ask(s, 1u, pa + a.zw * 0.05, z)
            self.ask(s, 2u, pb, z)
            self.ask(s, 3u, pb + b.zw * 0.05, z)
            return true
        }
        main: fn() {
            let s = self.counted(self.words.x)
            if s == 4294967295u { return }
            let src = self.span_src[s]
            let a = self.cross[src.y]
            let b = self.cross[src.z]
            let ga = self.cross_g[src.y]
            let gb = self.cross_g[src.z]
            let g = max(0.5 * (ga + gb), 0.000000000000000001)
            let z = self.slot_z(src.x / 100u)
            let ta = vec2(0.0 - a.w, a.z)
            let tb = vec2(0.0 - b.w, b.z)
            if self.words.z == 1u {
                if a.x == b.x && a.y == b.y { self.finish(s, a.xy, 13.0); return }
                let co = dot(ta, tb)
                // Under 25 degrees of turn: the curve middle (the chord's, for a straight span).
                if co >= 0.9063078 { self.curve(s, a, b, z, 0.0); return }
                // 25 degrees or more: a crease candidate at the tangents' intersection.
                let w = self.meet(a.xy, ta, 0.0, b.xy, tb, 0.0, a.xy)
                if w.z == 0.0 || !self.in_reach(src.x, w.xy) || !self.crease(s, a, b, w.xy, 0.0, z) { self.curve(s, a, b, z, 0.0) }
                return
            }
            let st = self.span_m[s]
            if st.z > 9.5 { return }
            let last = self.words.y == 4294967295u
            if st.z > 4.5 {
                // The halves a -> q and q -> b: their middles' samples take a Newton step along
                // each half's inside normal, then a secant step from both samples.
                let q = st.xy
                let cl = q - a.xy
                let cr = b.xy - q
                let ll = length(cl)
                let lr = length(cr)
                let nl = vec2(0.0 - cl.y, cl.x) / ll
                let nr = vec2(0.0 - cr.y, cr.x) / lr
                let ml = (a.xy + q) * 0.5
                let mr = (q + b.xy) * 0.5
                let fl = self.span_v[4u * s]
                let fr = self.span_v[4u * s + 1u]
                if st.z < 5.5 {
                    if last { self.finish(s, q, 11.0); return }
                    let ul = self.newton(fl, ga * dot(a.zw, nl), g, 0.5 * ll)
                    let ur = self.newton(fr, gb * dot(b.zw, nr), g, 0.5 * lr)
                    self.span_m[s] = vec4(q.x, q.y, 6.0, 0.0)
                    self.span_a[s] = vec4(fl, ul, fr, ur)
                    self.ask(s, 0u, ml + nl * ul, z)
                    self.ask(s, 1u, mr + nr * ur, z)
                    return
                }
                let h = self.span_a[s]
                let pl = ml + nl * self.secant_u(h.x, h.y, fl, g, 0.5 * ll)
                let pr = mr + nr * self.secant_u(h.z, h.w, fr, g, 0.5 * lr)
                self.span_m[s] = vec4(q.x, q.y, 14.0, 0.0)
                self.span_a[s] = vec4(pl.x, pl.y, pr.x, pr.y)
                return
            }
            if st.z > 1.5 {
                // The sides as circles through their crossings and probes' zeros; where they meet
                // is the vertex once two cycles agree within 0.02 units (0.002 mm). Three cycles,
                // then within 0.05 units, else the curve.
                let la = self.span_a[s].x
                let lb = self.span_a[s].y
                let ka = self.side(self.span_v[4u * s], self.span_v[4u * s + 1u], ga, la)
                let kb = self.side(self.span_v[4u * s + 2u], self.span_v[4u * s + 3u], gb, lb)
                let w = self.meet(a.xy, ta, ka, b.xy, tb, kb, st.xy)
                let found = w.z != 0.0 && self.in_reach(src.x, w.xy)
                let off = length(w.xy - st.xy)
                if found && off < 0.02 { self.finish(s, w.xy, 12.0); return }
                if !found || st.w > 1.5 || !self.crease(s, a, b, w.xy, st.w + 1.0, z) {
                    if found && off < 0.05 { self.finish(s, w.xy, 12.0) }
                    else if last { self.finish(s, a.xy, 10.0) } else { self.curve(s, a, b, z, 4.0 * (st.w + 1.0)) }
                }
                return
            }
            // The curve middle: secant steps on the bisector (nu, toward the inside), Newton first,
            // halving the bracket when a step leaves it.
            let c = b.xy - a.xy
            let nu = vec2(0.0 - c.y, c.x) / length(c)
            let m = (a.xy + b.xy) * 0.5
            let ua = self.span_a[s]
            let u = ua.z
            let f = self.span_v[4u * s]
            var u_out = st.x
            var u_in = st.y
            if f > 0.0 { u_out = u } else { u_in = u }
            if abs(f) <= 0.001 * g || st.w > 2.5 || last {
                // The middle when within 0.01 units of the zero set, or nearer it than the chord
                // (whose middle is about |u| off).
                let q = m + nu * u
                let e = abs(f) / g
                if abs(u) < 0.001 || (e > 0.01 && e >= 0.5 * abs(u)) || !self.inside_cell(src.x, q, 0.5) { self.finish(s, a.xy, 10.0); return }
                // A middle 0.05 units off the arc the tangents give holds a feature narrower than
                // the chord (a notch, a bump): its halves' middles too, within 16 samples.
                if !last && abs(u - self.sagitta(a, b)) > 0.05 && ua.w + st.w + 5.0 <= 16.0 {
                    self.span_m[s] = vec4(q.x, q.y, 5.0, 0.0)
                    self.ask(s, 0u, (a.xy + q) * 0.5, z)
                    self.ask(s, 1u, (q + b.xy) * 0.5, z)
                    return
                }
                self.finish(s, q, 11.0)
                return
            }
            var slope = g * dot(normalize(a.zw + b.zw), nu)
            if st.w > 0.5 && f != ua.y { slope = (f - ua.y) / (u - ua.x) }
            var next = u
            if abs(slope) > 0.000000000000000001 { next = u - f / slope }
            let bracketed = abs(u_out) < 100000.0 && abs(u_in) < 100000.0
            if bracketed && !(next > min(u_out, u_in) && next < max(u_out, u_in)) { next = 0.5 * (u_out + u_in) }
            if abs(next) > length(c) { self.finish(s, a.xy, 10.0); return }
            self.span_m[s] = vec4(u_out, u_in, 1.0, st.w + 1.0)
            self.span_a[s] = vec4(u, f, next, ua.w)
            self.ask(s, 0u, m + nu * next, z)
        }
    }

    // A plane's seg room: segs are at most two a span, so a refined middle's four take what its
    // plane's straight and zero-length spans leave. seg_room = (spare segs, refined middles) per
    // plane: words.y 0 clears them (an element a plane), 1 adds every span's.
    mod.sdf.PlaneSegRoom = {
        ..mod.sdf.Plane
        span_src: shader.buffer(vec4u)
        span_m: shader.buffer(vec4f)
        seg_room: shader.buffer_rw(u32)
        // words.x: spans, y: 0 clear or 1 add
        main: fn() {
            if self.words.y == 0u {
                let p = self.element()
                if p == 4294967295u { return }
                self.seg_room[2u * p] = 0u
                self.seg_room[2u * p + 1u] = 0u
                return
            }
            let s = self.counted(self.words.x)
            if s == 4294967295u { return }
            let mode = self.span_m[s].z
            let p = self.slot_info[self.span_src[s].x / 100u].x
            let n = self.seg_count(mode)
            if n < 2u { atomic_add(self.seg_room[2u * p], 2u - n) }
            if mode > 13.5 { atomic_add(self.seg_room[2u * p + 1u], 1u) }
        }
    }

    // A span's seg count (its first seg is the counts' scan, span_seg): a refined middle's four
    // when its plane has room for all of them, else two; and loop_len cleared below the span
    // count for the loops' lengths. The same whatever planes share the batch.
    mod.sdf.PlaneSegCount = {
        ..mod.sdf.Plane
        span_src: shader.buffer(vec4u)
        span_m: shader.buffer(vec4f)
        seg_room: shader.buffer(u32)
        seg_n: shader.buffer_rw(u32)
        loop_len: shader.buffer_rw(u32)
        // words.x: spans
        main: fn() {
            let s = self.counted(self.words.x)
            if s == 4294967295u { return }
            let mode = self.span_m[s].z
            var n = self.seg_count(mode)
            if mode > 13.5 {
                let p = self.slot_info[self.span_src[s].x / 100u].x
                if 2u * self.seg_room[2u * p + 1u] <= self.seg_room[2u * p] { n = 4u }
            }
            self.seg_n[s] = n
            self.loop_len[s] = 0u
        }
    }

    mod.sdf.PlaneSegs = {
        ..mod.sdf.Plane
        span_src: shader.buffer(vec4u)
        span_m: shader.buffer(vec4f)
        span_a: shader.buffer(vec4f)
        span_seg: shader.buffer(u32)
        cross: shader.buffer(vec4f)
        cross_key: shader.buffer(u32)
        seg: shader.buffer_rw(vec4f)
        seg_key: shader.buffer_rw(u32)
        // words.x: spans, y: segs
        main: fn() {
            let s = self.counted(self.words.x)
            if s == 4294967295u { return }
            let m = self.span_m[s]
            // its segs: up to the next span's first (the segs' total after the last span)
            let first = self.span_seg[s]
            var end = self.counts[self.words.y]
            if s + 1u < self.counts[self.words.x] { end = self.span_seg[s + 1u] }
            let n = end - first
            if n == 0u { return }
            let src = self.span_src[s]
            let a = self.cross[src.y].xy
            let b = self.cross[src.z].xy
            self.seg_key[first] = self.cross_key[src.y]
            if n == 1u { self.seg[first] = vec4(a.x, a.y, b.x, b.y); return }
            let f = src.x
            let si = self.slot_info[f / 100u]
            let gx = si.y * 10u + (f % 100u) % 10u
            let gy = si.z * 10u + (f % 100u) / 10u
            let cell = u32(#(crate::plane_abi::CELL_BIT)) | self.cell_id(gx, gy)
            if n == 2u {
                self.seg[first] = vec4(a.x, a.y, m.x, m.y)
                self.seg[first + 1u] = vec4(m.x, m.y, b.x, b.y)
                self.seg_key[first + 1u] = cell
                return
            }
            // a refined middle: a, the left half's middle, the middle, the right half's, b
            let h = self.span_a[s]
            self.seg[first] = vec4(a.x, a.y, h.x, h.y)
            self.seg[first + 1u] = vec4(h.x, h.y, m.x, m.y)
            self.seg[first + 2u] = vec4(m.x, m.y, h.z, h.w)
            self.seg[first + 3u] = vec4(h.z, h.w, b.x, b.y)
            self.seg_key[first + 1u] = cell
            self.seg_key[first + 2u] = cell
            self.seg_key[first + 3u] = cell
        }
    }

    // A span's segs' links and flags, gathered: the next span with segs after its end crossing
    // (the span that crossing starts), the previous one before its start (zero-length spans at
    // one node pass through: at most four around it). seg_info = (NONE, next, prev, flags) until
    // the roots are known.
    mod.sdf.PlaneSegLink = {
        ..mod.sdf.Plane
        span_src: shader.buffer(vec4u)
        span_m: shader.buffer(vec4f)
        span_seg: shader.buffer(u32)
        cross_end: shader.buffer(u32)
        seg_n: shader.buffer(u32)
        seg_info: shader.buffer_rw(vec4u)
        segs_of: fn(s: u32) -> u32 { return self.seg_n[s] }
        // words.x: spans
        main: fn() {
            let s = self.counted(self.words.x)
            if s == 4294967295u { return }
            let n = self.segs_of(s)
            if n == 0u { return }
            let first = self.span_seg[s]
            let src = self.span_src[s]
            let m = self.span_m[s]
            var t = src.z
            for step in 0..4u {
                if self.segs_of(t) != 0u { break }
                t = self.span_src[t].z
            }
            var next = first
            if self.segs_of(t) != 0u { next = self.span_seg[t] }
            var r = self.cross_end[src.y]
            for step in 0..4u {
                if self.segs_of(r) != 0u { break }
                r = self.cross_end[self.span_src[r].y]
            }
            var prev = first + n - 1u
            if self.segs_of(r) != 0u { prev = self.span_seg[r] + self.segs_of(r) - 1u }
            // The vertices inside the cell: the middle or crease vertex at the span's pair ordinal,
            // a refined middle's halves' middles at the pair + 2 and + 4 (distinct in the cell).
            let pair = src.w & 255u
            for i in 0..4u {
                if i >= n { break }
                var after = first + i + 1u
                if i + 1u == n { after = next }
                var before = prev
                if i > 0u { before = first + i - 1u }
                var flags = 0u
                if i > 0u {
                    var ordinal = pair
                    if n == 4u && i == 1u { ordinal = pair + 2u }
                    if n == 4u && i == 3u { ordinal = pair + 4u }
                    flags = ordinal << u32(#(crate::plane_abi::SEG_ORDINAL_SHIFT))
                    if m.z > 11.5 && m.z < 12.5 { flags = flags | u32(#(crate::plane_abi::SEG_CREASE)) }
                }
                self.seg_info[first + i] = vec4u(4294967295u, after, before, flags)
            }
        }
    }

    // seg_info = (loop root, next, prev, flags); loop_info per loop root.
    mod.sdf.PlaneSegInfo = {
        ..mod.sdf.Plane
        seg: shader.buffer(vec4f)
        rank_root: shader.buffer(u32)
        rank: shader.buffer(u32)
        loop_scan: shader.buffer(u32)
        seg_plane_off: shader.buffer(u32)
        seg_info: shader.buffer_rw(vec4u)
        loop_info: shader.buffer_rw(vec4u)
        loop_len: shader.buffer_rw(u32)
        // words.x: segs
        main: fn() {
            let k = self.counted(self.words.x)
            if k == 4294967295u { return }
            let info = self.seg_info[k]
            let next = info.y
            let prev = info.z
            let root = self.rank_root[k]
            let a = self.seg[prev]
            let b = self.seg[k]
            var flags = info.w
            if self.cross2(a.zw - a.xy, b.zw - b.xy) < 0.0 { flags = flags | u32(#(crate::plane_abi::SEG_REFLEX)) }
            self.seg_info[k] = vec4u(root, next, prev, flags)
            if root == k {
                var lo = 0u
                var hi = self.batch.z
                for step in 0..32u {
                    if hi - lo <= 1u { break }
                    let mid = (lo + hi) / 2u
                    if self.seg_plane_off[mid] <= k { lo = mid } else { hi = mid }
                }
                self.loop_info[self.loop_scan[k]] = vec4u(k, self.rank[prev] + 1u, lo, 0u)
                self.loop_len[self.loop_scan[k]] = self.rank[prev] + 1u
            }
        }
    }

    // off[p] = scan[base_off[p] * stride] (z = stride), or the total (words.x) past the end.
    mod.sdf.PlaneOffsets = {
        ..mod.sdf.Plane
        base_off: shader.buffer(u32)
        scan: shader.buffer(u32)
        off: shader.buffer_rw(u32)
        main: fn() {
            let p = self.element()
            if p == 4294967295u || p > self.batch.z { return }
            let at = self.base_off[p] * self.dispatch_range.z
            if at < self.base_off[self.batch.z] * self.dispatch_range.z { self.off[p] = self.scan[at] }
            else { self.off[p] = self.counts[self.words.x] }
        }
    }
    // The batch's loops packed for a reader (u32 words): (planes, loops, segs, 0), each plane's
    // first loop (planes + 1, the loops last), its first point (planes + 1, the segs last), each
    // loop's length, then the points (x, y grid-unit f32 bits) loop by loop from each root along
    // next. A plane's loops are its own, in root order, and its points start at its first
    // point. words: x segs, y loops, z the packed word count.
    mod.sdf.PlanePack = {
        ..mod.sdf.Plane
        seg: shader.buffer(vec4f)
        seg_info: shader.buffer(vec4u)
        rank: shader.buffer(u32)
        loop_scan: shader.buffer(u32)
        loop_start: shader.buffer(u32)
        loop_len: shader.buffer(u32)
        loop_off: shader.buffer(u32)
        seg_plane_off: shader.buffer(u32)
        out: shader.buffer_rw(u32)
        main: fn() {
            let e = self.element()
            if e == 4294967295u { return }
            let planes = self.batch.z
            let segs = self.counts[self.words.x]
            let loops = self.counts[self.words.y]
            let head = 2u * planes + 6u
            if e == 0u {
                self.out[0] = planes
                self.out[1] = loops
                self.out[2] = segs
                self.out[3] = 0u
                self.counts[self.words.z] = head + loops + 2u * segs
            }
            if e <= planes {
                self.out[4u + e] = self.loop_off[e]
                self.out[5u + planes + e] = self.seg_plane_off[e]
            }
            if e < loops { self.out[head + e] = self.loop_len[e] }
            if e < segs {
                let at = head + loops + 2u * (self.loop_start[self.loop_scan[self.seg_info[e].x]] + self.rank[e])
                let q = self.seg[e]
                self.out[at] = asuint(q.x)
                self.out[at + 1u] = asuint(q.y)
            }
        }
    }
    mod.sdf
}

/// The grid's offset off the 1 mm lattice (0.371 fine cells). Designs put faces at round
/// coordinates; on a grid line a face's nodes read zero, so no edge along it changes sign and
/// a corner or a notch on it falls between cells. Off the lattice those faces cross cells.
pub const GRID_PHASE: f64 = 0.0371;

/// The part revision's grid: origin in printer mm (the 1 mm lattice less [`GRID_PHASE`]),
/// coarse cells across.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneGrid { pub origin: [f64; 2], pub nx: u32, pub ny: u32 }

impl PlaneGrid {
    /// The coarse grid around `bounds` (printer mm) with one coarse cell of margin, so no zero
    /// touches the grid's border.
    pub fn around(bounds: ([f64; 2], [f64; 2])) -> PlaneGrid {
        let origin = [(bounds.0[0] - 1.0).floor() - GRID_PHASE, (bounds.0[1] - 1.0).floor() - GRID_PHASE];
        let nx = ((bounds.1[0] + 1.0 - origin[0]).ceil() as u32).max(1);
        let ny = ((bounds.1[1] + 1.0 - origin[1]).ceil() as u32).max(1);
        PlaneGrid { origin, nx, ny }
    }
    /// Fine nodes across (the edge key's row stride).
    pub fn node_stride(&self) -> u32 { self.nx * 10 + 1 }
}

/// The `frame` buffer: rows 0..2 map grid units (gx, gy) and z in printer mm to shape space,
/// composed in f64 with the origin and the 0.1 mm grid; row 3 = (lipschitz per grid unit, 0, 0, 0).
pub fn frame_rows(to_shape: [[f64; 4]; 3], grid: &PlaneGrid, lipschitz: f64) -> [[f32; 4]; 4] {
    let mut rows = [[0.0f32; 4]; 4];
    for (r, m) in to_shape.iter().enumerate() {
        rows[r] = [(m[0] * 0.1) as f32, (m[1] * 0.1) as f32, m[2] as f32, (m[0] * grid.origin[0] + m[1] * grid.origin[1] + m[3]) as f32];
    }
    rows[3] = [(lipschitz * 0.1) as f32, 0.0, 0.0, 0.0];
    rows
}

/// The AIR cost of each kernel slot (0 for meshes, which charge per node).
pub fn slot_costs(slots: &[KernelSlot]) -> Vec<u32> {
    slots.iter().map(|s| s.air().map_or(0, |a| a.cost.min(u32::MAX as u64) as u32)).collect()
}

/// Work units of one sample of `program`: its ops at their costs, each kernel its AIR cost, and
/// a mesh at an estimate (both walks down and back through about four nodes per level, four
/// full leaves each); the sampler's per-round budget bounds the real worst case.
pub fn program_cost(program: &Program, slots: &[KernelSlot]) -> u32 {
    let texels = &program.texels;
    let end = texels.first().map_or(0, |h| h[0] as usize).min(texels.len());
    let mut pc = 1;
    let mut total = 0u64;
    while pc < end {
        let h = texels[pc];
        let code = h[0];
        total += if code == op::PRIM { cost::PRIM }
            else if code == op::XFORM_PUSH { cost::XFORM_PUSH }
            else if code == op::DEFORM_PUSH { cost::DEFORM_PUSH }
            else if code == op::XFORM_POP || code == op::DEFORM_POP { cost::POP }
            else if code == op::FOLD { cost::FOLD }
            else if code == op::KERNEL { slots.get(h[1] as usize).and_then(|s| s.air()).map_or(1, |a| a.cost.min(u32::MAX as u64) as u32) }
            else if code == op::MESH {
                let nodes = slots.get(h[1] as usize).and_then(|s| s.mesh_asset()).map_or(1, |m| m.words[0][0].max(1));
                let depth = 32 - nodes.leading_zeros();
                cost::START + 2 * (4 * depth * cost::MESH_INNER + 4 * (cost::MESH_PRUNED + 8 * cost::MESH_TRIANGLE))
            }
            else if code == op::LINKS { cost::START + 64 * cost::LINKS_STEP }
            else if code == op::TEX { cost::TEX }
            else if code == op::SCALE { cost::SMALL }
            else { cost::SMALL } as u64;
        pc += op_len(h);
    }
    total.min(u32::MAX as u64) as u32
}

/// The count words this stage claims in the batch's counts buffer.
#[derive(Clone, Copy, Debug)]
pub struct Words {
    /// The coarse nodes (set at the start).
    pub coarse_req: u32,
    pub kept: u32,
    /// The fine nodes, 121 a kept slot.
    pub node_req: u32,
    /// The crossings, which are the spans: every crossing starts one.
    pub cross: u32,
    /// The bracket rounds' request counts (12).
    pub cross_req: u32,
    pub grad_req: u32,
    pub centre_req: u32,
    /// The span rounds' request counts.
    pub span_req: u32,
    pub segs: u32,
    pub loops: u32,
    /// The loops' points (the segs again, as the loop lengths' scan total).
    pub points: u32,
    /// The packed readback's words ([`item`]).
    pub packed: u32,
}
/// Bracket rounds: at most 12 samples per crossing.
pub const CROSS_ROUNDS: u32 = 12;
/// Span sample rounds: 3 crease cycles, then 4 curve steps, and the round that ends them.
pub const SPAN_ROUNDS: u32 = 8;

impl Words {
    /// The words from `first` on ([`Words::COUNT`] in all).
    pub fn from(first: u32) -> Words {
        let cross_req = first + 4;
        let span_req = cross_req + CROSS_ROUNDS + 2;
        let segs = span_req + SPAN_ROUNDS;
        Words { coarse_req: first, kept: first + 1, node_req: first + 2, cross: first + 3, cross_req,
            grad_req: cross_req + CROSS_ROUNDS, centre_req: cross_req + CROSS_ROUNDS + 1, span_req,
            segs, loops: segs + 1, points: segs + 2, packed: segs + 3 }
    }
    pub const COUNT: u32 = 4 + CROSS_ROUNDS + 2 + SPAN_ROUNDS + 4;
}

/// Whether every sample of `program` finishes in one sampler call: no mesh or links walk, at
/// most [`CALL_OPS`] ops, and its cost (`program_cost`) within one round's budget. A bounded
/// program samples in plain passes; any other resumes.
pub fn bounded(program: &Program, cost: u32) -> bool {
    let texels = &program.texels;
    let end = texels.first().map_or(0, |h| h[0] as usize).min(texels.len());
    let (mut pc, mut ops) = (1, 0u32);
    while pc < end {
        let h = texels[pc];
        if h[0] == op::MESH || h[0] == op::LINKS { return false; }
        ops += 1;
        pc += op_len(h);
    }
    ops <= CALL_OPS && cost <= ROUND_BUDGET
}

/// What a batch traces: `planes` of one part revision, its program's per-sample cost and mode.
#[derive(Clone, Copy, Debug)]
pub struct PlaneBatch { pub grid: PlaneGrid, pub planes: u32, pub sample_cost: u32, pub bounded: bool }

impl PlaneBatch {
    pub fn new(grid: PlaneGrid, planes: u32, program: &Program, slots: &[KernelSlot]) -> Self {
        let sample_cost = program_cost(program, slots).max(1);
        Self { grid, planes, sample_cost, bounded: bounded(program, sample_cost) }
    }
    /// The kernels' `batch` uniform: (coarse cells across, down, planes, the kept count word).
    pub fn uniform(&self, w: &Words) -> [u32; 4] { [self.grid.nx, self.grid.ny, self.planes, w.kept] }
}

/// The plane kernels: the flat passes, the sampler spliced with one kernel set (its program's
/// kernels and meshes), and the shared ranking and scan.
#[derive(Clone)]
pub struct PlaneKernels { flat: HashMap<LiveId, ComputeKernel>, sampler: ComputeKernel, rank: RankKernels, scan: ScanKernels }

impl PlaneKernels {
    pub fn new(cx: &mut Cx, slots: &[KernelSlot]) -> Result<Self, String> {
        let rank = RankKernels::new(cx)?;
        let scan = ScanKernels::new(cx)?;
        cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            let mut flat = HashMap::new();
            for id in KERNELS {
                let object = vm.value(module, id).as_object().ok_or_else(|| format!("plane kernel {id} missing"))?;
                flat.insert(id, ComputeKernel::new(vm, object, &format!("{id}"))?);
            }
            let base = vm.value(module, SAMPLER).as_object().ok_or("plane sampler missing")?;
            let mut splice = ComputeBake::new(vm, slots).ok_or("plane sampler splice")?;
            let sampler = splice.draw_variant(vm, base).ok_or_else(|| "plane sampler variant".to_string())
                .and_then(|object| ComputeKernel::new(vm, object, "plane sample"));
            splice.release(vm);
            Ok(Self { flat, sampler: sampler?, rank, scan })
        })
    }
}

/// The part's prepared inputs and the batch's `planes` ((z bits, plane, 0, 0) per plane), bound
/// by name: `program`, the meshes (`mesh::binding(slot)`), `slot_costs`, `frame` ([`frame_rows`]).
#[derive(Clone)]
pub struct PlaneInputs { pub program: BufRef, pub meshes: Vec<(LiveId, BufRef)>, pub slot_costs: BufRef, pub frame: BufRef, pub planes: BufRef }

/// §10.3's outputs by name (coarse_d, kept_off, kept, coarse_slot, node_d, edge_local,
/// slot_cross_base, cross_off, cross, cross_key, cross_g, span_src, span_m, span_a, span_v,
/// span_seg, seg, seg_info, seg_key, seg_plane_off, loop_off, loop_info, loop_len) and the count
/// words that hold their lengths ([`Words`]: kept, cross, segs, loops).
pub struct PlaneOutputs { pub bufs: HashMap<LiveId, BufRef>, pub words: Words }
impl PlaneOutputs {
    pub fn get(&self, name: LiveId) -> Result<BufRef, JobError> {
        self.bufs.get(&name).copied().ok_or_else(|| JobError::Failed(format!("plane output {name} missing")))
    }
}

/// The steps downstream of the stage, built at its crossings checkpoint from its outputs and
/// appended after its own (no host step of their own).
pub type Next = Box<dyn FnOnce(&PlaneOutputs, &Counts, &mut Pool) -> Result<Vec<Step>, JobError>>;

struct Stage { k: Rc<PlaneKernels>, b: PlaneBatch, w: Words, bufs: HashMap<LiveId, BufRef>, next: Next }

/// `p` with uniform `name` set, if its kernel reads it.
fn uniform(p: Pass, name: LiveId, v: [u32; 4]) -> Result<Pass, JobError> {
    if p.kernel.layout().slots.iter().any(|s| s.name == name) { p.param_u32x4(name, v).map_err(JobError::Failed) } else { Ok(p) }
}

/// The buffers sized at the checkpoints: the coarse one (kept slots), then the crossings one.
const AT_COARSE: [&str; 4] = ["node_d", "edge_local", "slot_cross", "slot_cross_base"];
const AT_CROSSINGS: [&str; 23] = ["req", "cross_geo", "cross_bs", "cross_v", "cross_gv", "cross", "cross_g", "cross_key", "cross_c", "cross_off",
    "span_src", "cross_end", "span_m", "span_a", "span_v", "seg_n", "span_seg", "loop_len", "seg", "seg_key", "seg_info", "seg_plane_off", "seg_room"];
/// Bytes of each buffer sized at the crossings checkpoint, per crossing (segs are at most two a
/// span, loops one), then per plane.
const CROSS_BYTES: u64 = 4 * 16 + 16 + 32 + 4 + 16 + 16 + 4 + 4 + 4 + 16 + 4 + 16 + 16 + 16 + 4 + 4 + 4 + 2 * (16 + 4 + 16) + 16 + 4;

impl Stage {
    fn id(name: &str) -> LiveId { LiveId::from_str(name) }
    /// A buffer sized now.
    fn buffer(&mut self, pool: &mut Pool, name: &'static str, bytes: u64) { self.bufs.insert(Self::id(name), pool.buffer(name, bytes.max(4))); }
    /// A deferred buffer's size: `n` elements of `bytes` (at least one).
    fn size(&self, pool: &mut Pool, name: &str, n: u32, bytes: u64) -> Result<(), JobError> { pool.size(self.bufs[&Self::id(name)], n.max(1) as u64 * bytes) }
    fn bind<'a>(&'a self, renames: &'a [(LiveId, LiveId)]) -> impl FnMut(LiveId) -> BufRef + 'a {
        move |name| {
            let name = renames.iter().find(|r| r.0 == name).map_or(name, |r| r.1);
            self.bufs.get(&name).copied().unwrap_or(BufRef::RUNTIME)
        }
    }
    fn pass(&self, kernel: LiveId, name: &str, elements: Elements, cost: u32, words: [u32; 4], renames: &[(LiveId, LiveId)]) -> Result<Pass, JobError> {
        let k = self.k.flat.get(&kernel).ok_or_else(|| JobError::Failed(format!("plane kernel {kernel} not lowered")))?;
        let p = Pass::new(k, label(name), elements, cost, CEILING, self.bind(renames));
        uniform(uniform(p, id!(batch), self.b.uniform(&self.w))?, id!(words), words)
    }
    fn step(&self, kernel: LiveId, name: &str, elements: Elements, cost: u32, words: [u32; 4], renames: &[(LiveId, LiveId)]) -> Result<Step, JobError> {
        Ok(Step::Pass(self.pass(kernel, name, elements, cost, words, renames)?))
    }
    /// The per-plane offsets `off` from `scan` at `base_off` x `stride`, the total in `total`.
    fn offsets(&self, name: &str, total: u32, renames: &[(LiveId, LiveId)], stride: u32) -> Result<Step, JobError> {
        Ok(Step::Pass(self.pass(id!(PlaneOffsets), name, Elements::Fixed(self.b.planes + 1), 4, [total, NONE, NONE, NONE], renames)?.zw(stride, 0)))
    }
    /// The elements counted in `word` (at most `bound`) sampled into `values`: the requests in
    /// `req` (source 0), or points from `site` (sources 1 and 2, see `PlaneSample`).
    fn sample(&self, word: u32, bound: u32, values: LiveId, source: [u32; 3], site: LiveId) -> Result<Step, JobError> {
        // the binding a source does not read takes the other one's buffer
        let req = if source[0] == 0 { id!(req) } else { site };
        let renames = [(id!(values), values), (id!(site), site), (id!(req), req)];
        sample_step(&self.k, &self.b, CountWord(word), bound, source, self.bind(&renames))
    }
    /// The exclusive scan of `n` values of `input` into `output`, the total into `total`.
    fn scan(&self, pool: &mut Pool, input: LiveId, output: LiveId, n: u32, total: u32) -> Vec<Step> {
        self.k.scan.exclusive(pool, self.bufs[&input], self.bufs[&output], n, CountWord(total))
    }
}

/// The step that samples the elements counted in `word` (at most `bound`) into `values`:
/// `source` = (0, _, _) the requests `req` = (x, y, z bits, dest); (1, nx + 1, coarse nodes a
/// plane) the coarse nodes, `site` the planes; (2, _, _) the kept slots' fine nodes, `site` their
/// slot_info. The program's bindings beside them, `continuations` any 16-byte buffer (a resume
/// binds the runtime's).
pub fn sample_step(k: &PlaneKernels, b: &PlaneBatch, word: CountWord, bound: u32, source: [u32; 3], bind: impl FnMut(LiveId) -> BufRef) -> Result<Step, JobError> {
    let elements = Elements::Counted { word, bound: bound.max(1) };
    let words = [word.0, source[0], source[1], source[2]];
    if b.bounded {
        let ceiling = (PROVEN_UNITS / b.sample_cost.max(1) as u64).clamp(64, CEILING as u64) as u32;
        let p = Pass::new(&k.sampler, label("plane sample"), elements, b.sample_cost.max(1), ceiling, bind).zw(0, 1);
        Ok(Step::Pass(uniform(p, id!(words), words)?))
    } else {
        let p = Pass::new(&k.sampler, label("plane sample resume"), elements, ROUND_BUDGET, RESUME_SLOTS, bind);
        Ok(Step::Resume { pass: uniform(p, id!(words), words)?, slots_ceiling: RESUME_SLOTS })
    }
}

/// Bytes a batch of `planes` reserves on grid `nx` x `ny` with `kept` slots and `crossings`:
/// the fixed coarse buffers, then what the two checkpoints size (the scans' block sums and the
/// list ranking's buffers estimated at a word per element and four per seg).
pub fn bytes(nx: u32, ny: u32, planes: u32, kept: u32, crossings: u32) -> u64 {
    let (planes, kept, c) = (planes as u64, kept as u64, crossings as u64);
    let cells = nx as u64 * ny as u64 * planes;
    let coarse = (nx as u64 + 1) * (ny as u64 + 1) * planes * 4 + cells * (4 * 4 + 16) + cells / 64 * 4 + (planes + 1) * 4;
    let slots = kept * (121 * 4 + 55 * 4 + 4 + 4 + 2);
    coarse + slots + c * CROSS_BYTES + 2 * c * 4 * 4 + (planes + 1) * 20
}

/// An estimate of one plane's bytes on grid `nx` x `ny` before any batch ran: kept slots along
/// an outline about four times the grid's side (a lattice keeps more), 30 crossings a slot.
/// A host corrects it from a batch's measured peak.
pub fn bytes_per_plane(nx: u32, ny: u32) -> u64 {
    let kept = (4 * (nx + ny)).min(nx * ny);
    bytes(nx, ny, 1, kept, 30 * kept)
}

/// The stage's steps for batch `b`: its count words claimed in `pool`, its buffers there (the
/// checkpoints' deferred, sized from exact counts), then `next`'s steps after its own.
pub fn stage(k: Rc<PlaneKernels>, b: PlaneBatch, inputs: &PlaneInputs, pool: &mut Pool, next: Next) -> Result<Vec<Step>, JobError> {
    let w = Words::from(pool.claim("plane", Words::COUNT)?.0);
    let mut s = Stage { k, b, w, bufs: HashMap::new(), next };
    for (name, buf) in [(id!(program), inputs.program), (id!(slot_costs), inputs.slot_costs), (id!(frame), inputs.frame), (id!(planes), inputs.planes)]
        .into_iter().chain(inputs.meshes.iter().copied()) {
        s.bufs.insert(name, buf);
    }
    let (nx, ny, planes) = (b.grid.nx, b.grid.ny, b.planes);
    let cells = nx * ny * planes;
    let per = (nx + 1) * (ny + 1);
    let coarse_nodes = per * planes;
    // The sampler's plain passes bind a continuation they never touch; a resume binds the runtime's.
    s.buffer(pool, "continuations", 16);
    s.buffer(pool, "coarse_d", coarse_nodes as u64 * 4);
    for (name, bytes) in [("flags", 4), ("offs", 4), ("coarse_slot", 4), ("kept", 4), ("slot_info", 16)] { s.buffer(pool, name, cells as u64 * bytes); }
    s.buffer(pool, "kept_off", (planes + 1) as u64 * 4);
    for name in AT_COARSE.into_iter().chain(AT_CROSSINGS).chain(["loop_info", "loop_off"]) { s.bufs.insert(Stage::id(name), pool.deferred(name)); }
    pool.upload(BufRef::COUNTS, w.coarse_req as u64 * 4, coarse_nodes.to_le_bytes().to_vec());
    let mut out = vec![
        s.sample(w.coarse_req, coarse_nodes, id!(coarse_d), [1, nx + 1, per], id!(planes))?,
        s.step(id!(PlaneKeep), "plane keep", Elements::Fixed(cells), 16, [NONE; 4], &[])?,
    ];
    out.extend(s.scan(pool, id!(flags), id!(offs), cells, w.kept));
    out.push(s.step(id!(PlaneKept), "plane kept", Elements::Fixed(cells), 8, [w.kept, w.node_req, NONE, NONE], &[])?);
    out.push(Step::Host(Box::new(move |counts, pool| nodes_and_crossings(s, counts, pool))));
    Ok(out)
}

/// At the coarse checkpoint: each kept slot's 121 nodes, then its owned sign-changing edges.
fn nodes_and_crossings(s: Stage, counts: &Counts, pool: &mut Pool) -> Result<Vec<Step>, JobError> {
    let w = s.w;
    let kept = counts.at(w.kept);
    for (name, n, bytes) in [("node_d", kept * 121, 4), ("edge_local", kept * 55, 4), ("slot_cross", kept, 4), ("slot_cross_base", kept, 4)] { s.size(pool, name, n, bytes)?; }
    let mut out = vec![
        s.sample(w.node_req, kept * 121, id!(node_d), [2, 0, 0], id!(slot_info))?,
        s.step(id!(PlaneEdgeCount), "plane edge counts", Elements::Fixed(kept), 220 * 8, [NONE; 4], &[])?,
    ];
    out.extend(s.scan(pool, id!(slot_cross), id!(slot_cross_base), kept, w.cross));
    out.push(Step::Host(Box::new(move |counts, pool| crossings_and_spans(s, kept, counts, pool))));
    Ok(out)
}

/// At the crossings checkpoint: every crossing starts one span and ends one, so spans equal
/// crossings and segs are at most twice that.
fn crossings_and_spans(mut s: Stage, kept: u32, counts: &Counts, pool: &mut Pool) -> Result<Vec<Step>, JobError> {
    let w = s.w;
    let fine = kept * 100;
    // buffers hold at least one element; scans and bounds take the exact count
    let exact = counts.at(w.cross);
    let c = exact.max(1);
    let planes = s.b.planes;
    let sized: [(&str, u32, u64); 25] = [
        ("req", 4 * c, 16), ("cross_geo", c, 16), ("cross_bs", c, 32), ("cross_v", c, 4), ("cross_gv", c, 16), ("cross", c, 16), ("cross_g", c, 4),
        ("cross_key", c, 4), ("cross_c", c, 4), ("cross_off", planes + 1, 4), ("span_src", c, 16), ("cross_end", c, 4), ("span_m", c, 16),
        ("span_a", c, 16), ("span_v", c, 16), ("seg_n", c, 4), ("span_seg", c, 4), ("loop_len", c, 4), ("seg", 2 * c, 16), ("seg_key", 2 * c, 4),
        ("seg_info", 2 * c, 16), ("seg_plane_off", planes + 1, 4), ("loop_info", c, 16), ("loop_off", planes + 1, 4), ("seg_room", 2 * planes, 4),
    ];
    for (name, n, bytes) in sized { s.size(pool, name, n, bytes)?; }
    let cross = Elements::Counted { word: CountWord(w.cross), bound: c };
    let mut out = vec![
        s.step(id!(PlaneCrossInit), "plane crossings", Elements::Fixed(kept), 220 * 4, [w.cross, w.cross_req, NONE, NONE], &[])?,
        s.offsets("plane crossing offsets", w.cross, &[(id!(base_off), id!(kept_off)), (id!(scan), id!(slot_cross_base)), (id!(off), id!(cross_off))], 1)?,
    ];
    for k in 0..CROSS_ROUNDS {
        let next = if k + 1 < CROSS_ROUNDS { w.cross_req + k + 1 } else { NONE };
        out.push(s.sample(w.cross_req + k, c, id!(cross_v), [0, 0, 0], id!(req))?);
        out.push(s.step(id!(PlaneCrossStep), "plane bracket", cross, 32, [w.cross, next, NONE, NONE], &[])?);
    }
    out.extend([
        s.step(id!(PlaneCrossGrad), "plane gradient requests", cross, 16, [w.cross, w.grad_req, NONE, NONE], &[])?,
        s.sample(w.grad_req, 4 * c, id!(cross_gv), [0, 0, 0], id!(req))?,
        s.step(id!(PlaneCross), "plane crossing records", cross, 24, [w.cross, NONE, NONE, NONE], &[])?,
        s.step(id!(PlaneCellSpans), "plane saddles", Elements::Fixed(fine), 24, [w.centre_req, NONE, NONE, NONE], &[])?,
        s.sample(w.centre_req, c, id!(cross_c), [0, 0, 0], id!(req))?,
        s.step(id!(PlaneSpanSrc), "plane spans", Elements::Fixed(fine), 48, [NONE; 4], &[])?,
    ]);
    for k in 0..=SPAN_ROUNDS {
        let round = if k < SPAN_ROUNDS { w.span_req + k } else { NONE };
        out.push(s.step(id!(PlaneSpanStep), "plane span round", cross, 96, [w.cross, round, (k == 0) as u32, NONE], &[])?);
        if k < SPAN_ROUNDS { out.push(s.sample(w.span_req + k, 4 * c, id!(span_v), [0, 0, 0], id!(req))?); }
    }
    out.extend([
        s.step(id!(PlaneSegRoom), "plane seg room", Elements::Fixed(planes), 2, [NONE, 0, NONE, NONE], &[])?,
        s.step(id!(PlaneSegRoom), "plane seg room", cross, 8, [w.cross, 1, NONE, NONE], &[])?,
        s.step(id!(PlaneSegCount), "plane seg counts", cross, 8, [w.cross, NONE, NONE, NONE], &[])?,
    ]);
    out.extend(s.scan(pool, id!(seg_n), id!(span_seg), exact, w.segs));
    out.extend([
        s.step(id!(PlaneSegs), "plane segs", cross, 32, [w.cross, w.segs, NONE, NONE], &[])?,
        s.step(id!(PlaneSegLink), "plane seg links", cross, 24, [w.cross, NONE, NONE, NONE], &[])?,
    ]);
    // The successors are seg_info's next words.
    let segs = 2 * c;
    let succ = Succ { buffer: s.bufs[&id!(seg_info)], stride: 4, at: 1 };
    let (rank, ranked) = list_rank::steps(&s.k.rank, &s.k.scan, pool, CountWord(w.segs), segs, succ, CountWord(w.loops))?;
    out.extend(rank);
    for (name, buf) in [(id!(rank_root), ranked.root), (id!(rank), ranked.rank), (id!(loop_scan), ranked.list_scan)] { s.bufs.insert(name, buf); }
    out.extend([
        s.offsets("plane seg offsets", w.segs, &[(id!(base_off), id!(cross_off)), (id!(scan), id!(span_seg)), (id!(off), id!(seg_plane_off))], 1)?,
        s.step(id!(PlaneSegInfo), "plane seg records", Elements::Counted { word: CountWord(w.segs), bound: segs }, 24, [w.segs, NONE, NONE, NONE], &[])?,
        s.offsets("plane loop offsets", w.loops, &[(id!(base_off), id!(seg_plane_off)), (id!(scan), id!(loop_scan)), (id!(off), id!(loop_off))], 1)?,
    ]);
    let Stage { bufs, next, .. } = s;
    out.extend(next(&PlaneOutputs { bufs, words: w }, counts, pool)?);
    Ok(out)
}

/// One part revision's batch of planes as one work item: the stage, then its loops packed into
/// one readback (`PlanePack`'s layout, read by [`decode_plane`]). The planes may come in any
/// order: each plane's records are its own, the same whatever batch it is traced in.
pub fn item(k: Rc<PlaneKernels>, b: PlaneBatch, inputs: &PlaneInputs, pool: &mut Pool) -> Result<Vec<Step>, JobError> {
    let kernels = k.clone();
    stage(k, b, inputs, pool, Box::new(move |outs, counts, pool| pack(&kernels, &b, outs, counts, pool)))
}

/// The packed readback's words for `crossings` (segs at most two a crossing, loops one) on
/// `planes`: its capacity.
pub fn packed_words(planes: u32, crossings: u32) -> u32 { 2 * planes + 6 + 5 * crossings.max(1) }

fn pack(k: &PlaneKernels, b: &PlaneBatch, outs: &PlaneOutputs, counts: &Counts, pool: &mut Pool) -> Result<Vec<Step>, JobError> {
    let w = outs.words;
    let exact = counts.at(w.cross);
    let c = exact.max(1);
    let loop_start = pool.buffer("loop_start", c as u64 * 4);
    let words = packed_words(b.planes, c);
    let out = pool.readback("plane loops", words as u64 * 4);
    let mut steps = k.scan.exclusive(pool, outs.get(id!(loop_len))?, loop_start, exact, CountWord(w.points));
    let kernel = k.flat.get(&id!(PlanePack)).ok_or_else(|| JobError::Failed("plane kernel PlanePack not lowered".into()))?;
    let bind = |name: LiveId| if name == id!(loop_start) { loop_start } else if name == id!(out) { out } else { outs.bufs.get(&name).copied().unwrap_or(BufRef::RUNTIME) };
    let p = Pass::new(kernel, label("plane pack"), Elements::Fixed((2 * c).max(b.planes + 1)), 16, CEILING, bind);
    steps.push(Step::Pass(uniform(uniform(p, id!(batch), b.uniform(&w))?, id!(words), [w.segs, w.loops, w.packed, NONE])?));
    steps.push(Output::counted(out, CountWord(w.packed), 4, words));
    Ok(steps)
}

fn word(bytes: &[u8], at: usize) -> Result<u32, String> {
    bytes.get(at * 4..at * 4 + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(|| format!("the packed loops end before word {at}"))
}

/// Plane `i`'s loops from a packed readback ([`item`]'s output) on `grid`: printer-mm points,
/// its loops in root order, each from its root along next. Pure: planes decode independently.
pub fn decode_plane(grid: &PlaneGrid, bytes: &[u8], i: usize) -> Result<Vec<Vec<[f32; 2]>>, String> {
    let planes = word(bytes, 0)? as usize;
    let loops = word(bytes, 1)? as usize;
    if i >= planes { return Err(format!("plane {i} of a batch of {planes}")); }
    let head = 2 * planes + 6;
    let (first, end) = (word(bytes, 4 + i)? as usize, word(bytes, 5 + i)? as usize);
    let mut point = word(bytes, 5 + planes + i)? as usize;
    if first > end || end > loops { return Err(format!("plane {i}: loops {first}..{end} of {loops}")); }
    let mm = |q: u32, o: f64| (o + f32::from_bits(q) as f64 * 0.1) as f32;
    (first..end).map(|l| {
        let n = word(bytes, head + l)? as usize;
        let at = head + loops + 2 * point;
        point += n;
        (0..n).map(|j| Ok([mm(word(bytes, at + 2 * j)?, grid.origin[0]), mm(word(bytes, at + 2 * j + 1)?, grid.origin[1])])).collect()
    }).collect()
}

/// Every plane's loops ([`decode_plane`] per plane), in the batch's plane order.
pub fn decode(grid: &PlaneGrid, bytes: &[u8]) -> Result<Vec<Vec<Vec<[f32; 2]>>>, String> {
    (0..word(bytes, 0)? as usize).map(|i| decode_plane(grid, bytes, i)).collect()
}

/// The plane kernels' script objects (mod.sdf), in the order a host compiles them. The sampler
/// is spliced per kernel set (its program's kernels and meshes); the rest compile once.
pub const KERNELS: [LiveId; 17] = [id!(PlaneKeep), id!(PlaneKept), id!(PlaneEdgeCount), id!(PlaneCrossInit), id!(PlaneCrossStep),
    id!(PlaneCrossGrad), id!(PlaneCross), id!(PlaneCellSpans), id!(PlaneSpanSrc), id!(PlaneSpanStep), id!(PlaneSegRoom), id!(PlaneSegCount), id!(PlaneSegs),
    id!(PlaneSegLink), id!(PlaneSegInfo), id!(PlaneOffsets), id!(PlanePack)];
pub const SAMPLER: LiveId = id!(PlaneSample);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grid_frame_and_words() {
        let g = PlaneGrid::around(([10.2, -3.5], [20.0, 4.0]));
        assert_eq!(g, PlaneGrid { origin: [9.0 - GRID_PHASE, -5.0 - GRID_PHASE], nx: 13, ny: 11 });
        let rows = frame_rows([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]], &g, 1.0);
        assert_eq!(rows[0], [0.1, 0.0, 0.0, (9.0 - GRID_PHASE) as f32]);
        assert_eq!(rows[1], [0.0, 0.1, 0.0, (-5.0 - GRID_PHASE) as f32]);
        assert_eq!(rows[3][0], 0.1);
        let w = Words::from(0);
        assert_eq!(w.packed + 1, Words::COUNT);
        assert_eq!(w.span_req, w.centre_req + 1);
    }
}
