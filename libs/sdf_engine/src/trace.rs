//! Exact layer tracing in the printer grid. Edge identities and endpoint-only
//! refinement make adjacent workgroups agree without a GPU deduplication pass.
use makepad_draw::*;

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    mod.sdf.Trace = {
        ..mod.sdf.Eval
        precise_math: true
        workgroup_size: 64
        // Inverse frame rows, grid origin + Lipschitz bound, then layer z's.
        inputs: shader.buffer(vec4f)
        kept: shader.buffer_rw(vec4u)
        counts: shader.buffer_rw(u32)
        segments: shader.buffer_rw(vec4u)
        grid: shader.uniform(vec4u) // nx, ny, layers, kept capacity
        capacity: shader.uniform(u32)
        field: fn(p: vec2, z: f32) -> f32 {
            let a = self.inputs[0u]
            let b = self.inputs[1u]
            let c = self.inputs[2u]
            return self.eval_program(0.0,
                ((a.x * p.x + a.y * p.y) + a.z * z) + a.w,
                ((b.x * p.x + b.y * p.y) + b.z * z) + b.w,
                ((c.x * p.x + c.y * p.y) + c.z * z) + c.w).x
        }
        node: fn(i: u32, j: u32) -> vec2 {
            return self.inputs[3u].xy + vec2(f32(i) * 0.1, f32(j) * 0.1)
        }
        span: fn(ea: vec4u, eb: vec4u, a: vec2, b: vec2, entering: bool, middle: vec2, has_middle: u32) {
            let index = atomic_add(self.counts[2u], 1u)
            if index >= self.capacity { atomic_add(self.counts[3u], 1u); return }
            let at = index * 4u
            // The boundary signs, not a possibly degenerate gradient, fix
            // direction. Every shared edge then has one incoming span.
            if entering {
                self.segments[at] = ea
                self.segments[at + 1u] = eb
                self.segments[at + 2u] = vec4u(asuint(a.x), asuint(a.y), asuint(b.x), asuint(b.y))
            } else {
                self.segments[at] = eb
                self.segments[at + 1u] = ea
                self.segments[at + 2u] = vec4u(asuint(b.x), asuint(b.y), asuint(a.x), asuint(a.y))
            }
            self.segments[at + 3u] = vec4u(asuint(middle.x), asuint(middle.y), has_middle, 0u)
        }
    }
    mod.sdf.TraceCoarse = {
        ..mod.sdf.Trace
        main: fn() {
            let x = global_id().x
            let layer = group_id().y
            if x >= self.grid.x * self.grid.y || layer >= self.grid.z { return }
            let i = x % self.grid.x
            let j = x / self.grid.x
            let z = self.inputs[4u + layer].x
            var values = array(0f, 0f, 0f, 0f)
            for corner in 0..4u {
                values[corner] = self.field(self.node((i + corner % 2u) * 10u, (j + corner / 2u) * 10u), z)
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
                atomic_add(self.counts[16u + layer * 3u], 1u)
                if at < self.grid.w {
                    self.kept[at] = vec4u(i, j, layer, 0u)
                    if at < 65535u { atomic_add(self.counts[8u], 1u); }
                    if at != 0u && at % 65535u == 0u { atomic_add(self.counts[9u], 1u); }
                }
                else { atomic_add(self.counts[1u], 1u) }
            }
        }
    }
    mod.sdf.TraceCell = {
        ..mod.sdf.Trace
        values: shader.workgroup(f32, 121)
        crossings: shader.workgroup(vec4f, 220)
        edge_ids: shader.workgroup(u32, 220)
        edge_count: shader.workgroup(u32, 1)
        // Brackets during refinement; sample offsets after refinement.
        state: shader.workgroup(vec4f, 220)
        spans: shader.workgroup(vec4u, 200)
        middles: shader.workgroup(vec4f, 200)
        // At most two spans per fine cell, with five Newton samples each.
        // Shared payload is 30,812 bytes, below 32 KiB including alignment.
        samples: shader.workgroup(vec2f, 1000)
        results: shader.workgroup(f32, 1000)
        sample_ids: shader.workgroup(u32, 1000)
        sample_count: shader.workgroup(u32, 1)
        edge_nodes: fn(edge: u32) -> vec4u {
            if edge < 110u { return vec4u(edge % 10u, edge / 10u, 0u, 1u) }
            return vec4u((edge - 110u) % 11u, (edge - 110u) / 11u, 1u, 11u)
        }
        delta: fn(sample: u32) -> vec2 {
            if sample == 0u { return vec2(0.001, 0.0) }
            if sample == 1u { return vec2(0.0 - 0.001, 0.0) }
            if sample == 2u { return vec2(0.0, 0.001) }
            if sample == 3u { return vec2(0.0, 0.0 - 0.001) }
            return vec2(0.0)
        }
        fraction: fn(bracket: vec4, round: u32) -> f32 {
            let lo = bracket.x
            let hi = bracket.y
            if round % 2u == 0u {
                return clamp(lo + (hi - lo) * bracket.z / (bracket.z - bracket.w), lo + (hi - lo) * 0.4, hi - (hi - lo) * 0.4)
            }
            return (lo + hi) * 0.5
        }
        main: fn() {
            let index = group_id().x + group_id().y * min(self.counts[0u], min(self.grid.w, 65535u))
            if index >= min(self.counts[0u], self.grid.w) { return }
            let cell = self.kept[index]
            let bi = cell.x * 10u
            let bj = cell.y * 10u
            let z = self.inputs[4u + cell.z].x
            // Every phase produces its points, all lanes run the SAME evaluator
            // site, then consumers use the results. No per-point interpreter is
            // inlined into refinement, gradients, saddle or projection code.
            var evaluations = 0u
            for phase in 0..18u {
                var candidates = 1000u
                if phase == 0u { candidates = 121u }
                else if phase <= 12u { candidates = self.edge_count[0u] }
                else if phase == 13u { candidates = self.edge_count[0u] * 4u }
                else if phase == 14u { candidates = 100u }
                else if phase == 15u { candidates = 200u }
                for k in 0..((candidates + 63u) / 64u) {
                    let n = local_index() + k * 64u
                    if n < candidates { self.sample_ids[n] = 0u }
                }
                barrier()
                if phase == 0u {
                    for k in 0..2u {
                        let n = local_index() + k * 64u
                        if n < 121u {
                            let q = self.node(bi + n % 11u, bj + n / 11u)
                            self.samples[n] = q
                            self.sample_ids[n] = 1u
                        }
                    }
                } else if phase <= 12u {
                    for k in 0..((self.edge_count[0u] + 63u) / 64u) {
                        let n = local_index() + k * 64u
                        if n < self.edge_count[0u] {
                            let edge = self.edge_ids[n]
                            let bracket = self.state[edge]
                            let crossing = self.crossings[edge]
                            if crossing.w == 0.0 && (bracket.y - bracket.x) * 0.1 > 0.0001 {
                                let e = self.edge_nodes(edge)
                                let a = self.node(bi + e.x, bj + e.y)
                                var b = self.node(bi + e.x + 1u, bj + e.y)
                                if e.z == 1u { b = self.node(bi + e.x, bj + e.y + 1u) }
                                let q = a + (b - a) * self.fraction(bracket, phase - 1u)
                                let at = n
                                self.samples[at] = q
                                self.sample_ids[at] = 1u
                                self.crossings[edge] = vec4(q.x, q.y, f32(at), 0.0)
                            } else {
                                self.crossings[edge] = vec4(crossing.x, crossing.y, 0.0, 1.0)
                            }
                        }
                    }
                } else if phase == 13u {
                    for k in 0..((self.edge_count[0u] * 4u + 63u) / 64u) {
                        let n = local_index() + k * 64u
                        if n < self.edge_count[0u] * 4u {
                            let q = self.crossings[self.edge_ids[n / 4u]].xy + self.delta(n % 4u)
                            self.samples[n] = q
                            self.sample_ids[n] = 1u
                        }
                    }
                } else if phase == 14u {
                    for k in 0..2u {
                        let n = local_index() + k * 64u
                        if n < 100u {
                            let at = (n / 10u) * 11u + n % 10u
                            let a = self.values[at] < 0.0
                            let b = self.values[at + 1u] < 0.0
                            let c = self.values[at + 12u] < 0.0
                            let d = self.values[at + 11u] < 0.0
                            if a == c && b == d && a != b {
                                let q = self.node(bi + n % 10u, bj + n / 10u) + vec2(0.05)
                                let sample = n
                                self.samples[sample] = q
                                self.sample_ids[sample] = 1u
                                self.state[n] = vec4(f32(sample), 0.0, 0.0, 0.0)
                            }
                        }
                    }
                } else {
                    for k in 0..4u {
                        let n = local_index() + k * 64u
                        if n < 200u {
                            let mid = self.middles[n]
                            self.state[n] = vec4(0.0 - 1.0, 0.0, 0.0, 0.0)
                            if mid.w == 0.0 {
                                let span = self.spans[n]
                                let a = self.crossings[span.x]
                                let b = self.crossings[span.y]
                                let origin = self.node(bi + span.z % 10u, bj + span.z / 10u)
                                if phase == 15u {
                                    if mid.z < 0.9063078 {
                                        let na = a.zw
                                        let nb = b.zw
                                        let det = na.x * nb.y - na.y * nb.x
                                        if abs(det) > 0.000000001 {
                                            let c = dot(nb, b.xy - a.xy)
                                            let q = a.xy + vec2((0.0 - na.y) * c, na.x * c) / det
                                            if q.x >= origin.x - 0.05 && q.x <= origin.x + 0.15 && q.y >= origin.y - 0.05 && q.y <= origin.y + 0.15 {
                                                let sample = n
                                                self.samples[sample] = q
                                                self.sample_ids[sample] = 1u
                                                self.state[n] = vec4(f32(sample), 0.0, 0.0, 0.0)
                                            }
                                        }
                                    }
                                } else if mid.z < 0.9975641 {
                                    let sample = n * 5u
                                    self.state[n] = vec4(f32(sample), 0.0, 0.0, 0.0)
                                    for j in 0..5u {
                                        var q = mid.xy
                                        if j < 4u { q = q + self.delta(j) }
                                        self.samples[sample + j] = q
                                        self.sample_ids[sample + j] = 1u
                                    }
                                }
                            }
                        }
                    }
                }
                barrier()
                // Compact only indices. Producers own fixed slots; results go
                // back to those slots, so no sample or result scatter races.
                if local_index() == 0u {
                    var count = 0u
                    for n in 0..candidates {
                        if self.sample_ids[n] != 0u { self.sample_ids[count] = n; count += 1u }
                    }
                    self.sample_count[0u] = count
                }
                barrier()
                let samples = self.sample_count[0u]
                for k in 0..((samples + 63u) / 64u) {
                    let n = local_index() + k * 64u
                    if n < samples {
                        let at = self.sample_ids[n]
                        self.results[at] = self.field(self.samples[at], z)
                    }
                }
                barrier()
                evaluations += samples
                if phase == 0u {
                    for k in 0..2u {
                        let n = local_index() + k * 64u
                        if n < 121u { self.values[n] = self.results[n] }
                    }
                    barrier()
                    for k in 0..4u {
                        let edge = local_index() + k * 64u
                        if edge < 220u {
                            let e = self.edge_nodes(edge)
                            let at = e.y * 11u + e.x
                            let fa = self.values[at]
                            let fb = self.values[at + e.w]
                            self.edge_ids[edge] = 0u
                            if (fa < 0.0) != (fb < 0.0) {
                                self.edge_ids[edge] = 1u
                                let a = self.node(bi + e.x, bj + e.y)
                                var b = self.node(bi + e.x + 1u, bj + e.y)
                                if e.z == 1u { b = self.node(bi + e.x, bj + e.y + 1u) }
                                var q = a + (b - a) * 0.5
                                var done = 0.0
                                if fa == 0.0 { q = a; done = 1.0 }
                                else if fb == 0.0 { q = b; done = 1.0 }
                                self.crossings[edge] = vec4(q.x, q.y, 0.0, done)
                                self.state[edge] = vec4(0.0, 1.0, fa, fb)
                            }
                        }
                    }
                    barrier()
                    if local_index() == 0u {
                        var count = 0u
                        for edge in 0..220u {
                            if self.edge_ids[edge] != 0u { self.edge_ids[count] = edge; count += 1u }
                        }
                        self.edge_count[0u] = count
                        atomic_add(self.counts[16u + cell.z * 3u + 1u], count)
                    }
                } else if phase <= 12u {
                    for k in 0..((self.edge_count[0u] + 63u) / 64u) {
                        let n = local_index() + k * 64u
                        if n < self.edge_count[0u] {
                            let edge = self.edge_ids[n]
                            let crossing = self.crossings[edge]
                            if crossing.w == 0.0 {
                                let f = self.results[u32(crossing.z)]
                                if f == 0.0 { self.crossings[edge] = vec4(crossing.x, crossing.y, 0.0, 1.0) }
                                else {
                                    var bracket = self.state[edge]
                                    let t = self.fraction(bracket, phase - 1u)
                                    if (f < 0.0) == (bracket.z < 0.0) { bracket = vec4(t, bracket.y, f, bracket.w) }
                                    else { bracket = vec4(bracket.x, t, bracket.z, f) }
                                    self.state[edge] = bracket
                                    let e = self.edge_nodes(edge)
                                    let a = self.node(bi + e.x, bj + e.y)
                                    var b = self.node(bi + e.x + 1u, bj + e.y)
                                    if e.z == 1u { b = self.node(bi + e.x, bj + e.y + 1u) }
                                    let q = a + (b - a) * ((bracket.x + bracket.y) * 0.5)
                                    self.crossings[edge] = vec4(q.x, q.y, 0.0, 0.0)
                                }
                            }
                        }
                    }
                } else if phase == 13u {
                    for k in 0..((self.edge_count[0u] + 63u) / 64u) {
                        let n = local_index() + k * 64u
                        if n < self.edge_count[0u] {
                            let at = n * 4u
                            let g = vec2(self.results[at] - self.results[at + 1u], self.results[at + 2u] - self.results[at + 3u]) * 500.0
                            let normal = g / max(length(g), 0.000000000001)
                            let edge = self.edge_ids[n]
                            let q = self.crossings[edge].xy
                            self.crossings[edge] = vec4(q.x, q.y, normal.x, normal.y)
                        }
                    }
                } else if phase == 14u {
                    for k in 0..2u {
                        let n = local_index() + k * 64u
                        if n < 100u {
                            let at = (n / 10u) * 11u + n % 10u
                            let a = self.values[at]
                            let b = self.values[at + 1u]
                            let c = self.values[at + 12u]
                            let d = self.values[at + 11u]
                            var ids = array(0u, 0u, 0u, 0u)
                            var enters = array(false, false, false, false)
                            var count = 0u
                            for edge in 0..4u {
                                var sample = (n / 10u) * 10u + n % 10u
                                var fa = a
                                var fb = b
                                if edge == 1u { fa = b; fb = c; sample = 110u + at + 1u }
                                if edge == 2u { fa = d; fb = c; sample = sample + 10u }
                                if edge == 3u { fa = a; fb = d; sample = 110u + at }
                                if (fa < 0.0) != (fb < 0.0) {
                                    ids[count] = sample
                                    enters[count] = fa < 0.0
                                    if edge >= 2u { enters[count] = fb < 0.0 }
                                    count += 1u
                                }
                            }
                            var alternate = false
                            if count == 4u { alternate = (self.results[u32(self.state[n].x)] < 0.0) != (a < 0.0) }
                            for pair in 0..2u {
                                let span = n * 2u + pair
                                self.middles[span] = vec4(0.0, 0.0, 0.0, 0.0 - 1.0)
                                if (pair == 0u && count >= 2u) || (pair == 1u && count == 4u) {
                                    var ia = pair * 2u
                                    var ib = ia + 1u
                                    if alternate { ia = (ia + 3u) % 4u; ib = (ib + 3u) % 4u }
                                    let ca = self.crossings[ids[ia]]
                                    let cb = self.crossings[ids[ib]]
                                    var entering = 0u
                                    if enters[ia] { entering = 1u }
                                    self.spans[span] = vec4u(ids[ia], ids[ib], n, entering)
                                    let middle = (ca.xy + cb.xy) * 0.5
                                    self.middles[span] = vec4(middle.x, middle.y, dot(ca.zw, cb.zw), 0.0)
                                }
                            }
                        }
                    }
                } else {
                    for k in 0..4u {
                        let n = local_index() + k * 64u
                        if n < 200u {
                            if self.state[n].x >= 0.0 {
                                let sample = u32(self.state[n].x)
                                let mid = self.middles[n]
                                if phase == 15u {
                                    let q = self.samples[sample]
                                    if abs(self.results[sample]) <= 0.001 { self.middles[n] = vec4(q.x, q.y, mid.z, 1.0) }
                                } else {
                                    let g = vec2(self.results[sample] - self.results[sample + 1u], self.results[sample + 2u] - self.results[sample + 3u]) * 500.0
                                    var q = mid.xy - g * (self.results[sample + 4u] / max(dot(g, g), 0.000000000001))
                                    var has_middle = 0.0
                                    if phase == 17u {
                                        let span = self.spans[n]
                                        let origin = self.node(bi + span.z % 10u, bj + span.z / 10u)
                                        if q.x >= origin.x - 0.1 && q.x <= origin.x + 0.2 && q.y >= origin.y - 0.1 && q.y <= origin.y + 0.2 { has_middle = 1.0 }
                                        else { q = (self.crossings[span.x].xy + self.crossings[span.y].xy) * 0.5 }
                                    }
                                    self.middles[n] = vec4(q.x, q.y, mid.z, has_middle)
                                }
                            }
                        }
                    }
                }
                barrier()
                // The coarse bound keeps nearby empty cells too. Once their
                // fine grid has no crossings, every later queue is empty.
                // This condition is shared by the entire workgroup.
                if phase == 0u && self.edge_count[0u] == 0u {
                    if local_index() == 0u { atomic_add(self.counts[16u + cell.z * 3u + 2u], evaluations) }
                    return
                }
            }
            if local_index() == 0u { atomic_add(self.counts[16u + cell.z * 3u + 2u], evaluations) }
            for k in 0..4u {
                let n = local_index() + k * 64u
                if n < 200u {
                    let mid = self.middles[n]
                    if mid.w >= 0.0 {
                        let span = self.spans[n]
                        let a = self.edge_nodes(span.x)
                        let b = self.edge_nodes(span.y)
                        self.span(vec4u(bi + a.x, bj + a.y, a.z, cell.z), vec4u(bi + b.x, bj + b.y, b.z, cell.z),
                            self.crossings[span.x].xy, self.crossings[span.y].xy, span.w != 0u, mid.xy, u32(mid.w))
                    }
                }
            }
        }
    }
    mod.sdf
}

use crate::{kernel::{ComputeBake, KernelSlot}, program::Program};
use makepad_draw::makepad_platform::{gpu_admission::{GpuDocument, GpuReservation}, gpu_frame_timer, thread::{Class, TaskPool, Ticket}};
use std::{collections::{BTreeMap, HashMap, VecDeque}, rc::Rc, sync::{Arc, mpsc}, time::Instant};

/// Host-independent request, prepared on the app pool, in focus order.
/// `done` also runs on that pool, at `class` with this request's ticket.
#[derive(Clone)]
pub struct TraceRequest {
    pub requested: Instant,
    pub part: u64,
    pub revision: u64,
    pub program: Arc<Program>,
    pub kernels: Arc<[KernelSlot]>,
    pub to_shape: [[f32; 4]; 3],
    pub bounds: ([f64; 2], [f64; 2]),
    pub lipschitz: f32,
    pub layers: Vec<(usize, f64)>,
    pub class: Class,
    pub ticket: Ticket,
    pub done: Arc<dyn Fn(Result<Vec<TracedLayer>, String>) + Send + Sync>,
}

#[derive(Clone, Debug)]
pub struct TracedLayer {
    pub index: usize,
    pub z: f64,
    /// Closed, directed printer-mm loops, canonical by their smallest edge id.
    pub loops: Vec<Vec<[f32; 2]>>,
}

struct Allocation {
    buffer: GpuBuffer,
    _reservation: GpuReservation,
}
impl Allocation {
    fn new(cx: &mut Cx, doc: &GpuDocument, size: u64, readback: bool, indirect: bool) -> Result<Self, String> {
        let reservation = doc.admit(size).map_err(|e| e.to_string())?;
        let buffer = GpuBuffer::new(cx, size, GpuBufferFlags { readback, indirect })?;
        Ok(Self { buffer, _reservation: reservation })
    }
}

#[derive(Default)]
struct TracePipelines {
    shaders: HashMap<Vec<u64>, (ComputeKernel, ComputeKernel)>,
    pending: Vec<(Vec<u64>, Instant)>,
    warm: Option<(Allocation, GpuDocument)>,
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
            let document = cx.gpu_admission().register("sdf trace pipelines");
            let allocation = Allocation::new(cx, &document, 64, true, false)?;
            allocation.buffer.update(cx, 0, &[0; 64])?;
            cx.global::<TracePipelines>().warm = Some((allocation, document));
        }
        let buffer = cx.global::<TracePipelines>().warm.as_ref().unwrap().0.buffer.clone();
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

fn prepared(cx: &mut Cx, kernels: &[KernelSlot]) -> Result<(ComputeKernel, ComputeKernel), String> {
    let keys: Vec<_> = kernels.iter().map(|s| s.key).collect();
    cx.global::<TracePipelines>().shaders.get(&keys).cloned().ok_or_else(|| "trace kernels have not been lowered by the bake yet".into())
}

struct Pools {
    inputs: Allocation,
    kept: Allocation,
    counts: Allocation,
    segments: Allocation,
    layers: usize,
    kept_capacity: u32,
    segment_capacity: u32,
}
struct Prepared {
    revision: u64,
    program: Allocation,
    meshes: Vec<(LiveId, Rc<crate::mesh::MeshGpu>)>,
}
struct Queued {
    request: Arc<TraceRequest>,
    first: usize,
}
struct Batch {
    serial: u64,
    request: Arc<TraceRequest>,
    first: usize,
    count: usize,
    started: Instant,
    coarse_evaluations: u64,
    readback_at: Option<Instant>,
    counts: Option<Arc<[u8]>>,
    segments: Option<Arc<[u8]>>,
}
struct Chain {
    request: Arc<TraceRequest>,
    layers: Vec<(usize, f64)>,
    segments: Arc<[u8]>,
    count: usize,
    started: Instant,
}
type Readback = (u64, bool, Instant, Result<Arc<[u8]>, String>);

/// One owner on Cx. `pump` is called once per frame and never waits for GPU
/// completion or pool admission. No per-frame work is requested when idle.
pub struct GpuTracer {
    serial: u64,
    submitted_before_draw: bool,
    last_error: Option<String>,
    document: GpuDocument,
    pool: TaskPool,
    queue: VecDeque<Queued>,
    prepared: HashMap<u64, Prepared>,
    pools: Option<Pools>,
    active: Option<Batch>,
    chains: VecDeque<Chain>,
    tx: mpsc::Sender<Readback>,
    rx: mpsc::Receiver<Readback>,
    timed_layers: VecDeque<(u64, u64, usize, usize)>,
    cost_per_layer: HashMap<u64, f32>,
    coarse_times: VecDeque<f32>,
    cell_times: VecDeque<f32>,
    kept_per_layer: u32,
    segments_per_layer: u32,
}

impl GpuTracer {
    pub fn new(cx: &mut Cx) -> Self {
        let (tx, rx) = mpsc::channel();
        gpu_frame_timer::watch("sdf trace coarse");
        gpu_frame_timer::watch("sdf trace cell");
        Self { serial: 0, submitted_before_draw: false, last_error: None, document: cx.gpu_admission().register("sdf trace"), pool: cx.task_pool(), queue: VecDeque::new(),
            prepared: HashMap::new(), pools: None, active: None, chains: VecDeque::new(), tx, rx,
            timed_layers: VecDeque::new(), cost_per_layer: HashMap::new(), coarse_times: VecDeque::new(), cell_times: VecDeque::new(), kept_per_layer: 0, segments_per_layer: 0 }
    }
    pub fn enqueue(&mut self, cx: &mut Cx, request: TraceRequest) {
        self.queue.retain(|q| q.request.part != request.part || q.request.revision == request.revision);
        self.queue.push_front(Queued { request: Arc::new(request), first: 0 });
        cx.redraw_all();
    }
    pub fn has_work(&self) -> bool { self.active.is_some() || !self.queue.is_empty() || !self.chains.is_empty() }

    /// Completion signals drain readbacks and submit chaining immediately. This
    /// does not start background GPU work; that remains paced by `pump`.
    pub fn poll(&mut self, cx: &mut Cx) {
        self.coarse_times.extend(gpu_frame_timer::take_watched("sdf trace coarse"));
        self.cell_times.extend(gpu_frame_timer::take_watched("sdf trace cell"));
        while !self.coarse_times.is_empty() && !self.cell_times.is_empty() {
            let coarse = self.coarse_times.pop_front().unwrap();
            let cell = self.cell_times.pop_front().unwrap();
            if let Some((part, revision, first, n)) = self.timed_layers.pop_front() {
                let per_layer = (coarse + cell) / n as f32;
                let cost = self.cost_per_layer.entry(part).or_insert(per_layer);
                *cost = *cost * 0.75 + per_layer * 0.25;
                log!("sdf trace GPU: part {} revision {} layers {}..{} {:.3} ms/frame, {:.3} ms/layer, {} layers coarse_ms={:.3} cell_ms={:.3}",
                    part, revision, first, first + n, coarse + cell, per_layer, n, coarse / n as f32, cell / n as f32);
            }
        }
        while let Ok((serial, counts, arrived, result)) = self.rx.try_recv() {
            if let Some(batch) = self.active.as_mut().filter(|b| b.serial == serial) {
                batch.readback_at = Some(arrived);
                match result {
                    Ok(bytes) => if counts { batch.counts = Some(bytes) } else { batch.segments = Some(bytes) },
                    Err(e) => {
                        log!("sdf trace readback: {}; retrying batch", e);
                        let batch = self.active.take().unwrap();
                        self.queue.push_front(Queued { request: batch.request, first: batch.first });
                    }
                }
            }
        }
        if self.active.as_ref().is_some_and(|b| b.counts.is_some() && b.segments.is_some()) {
            let batch = self.active.take().unwrap();
            let counts = batch.counts.unwrap();
            let kept = word(&counts, 0);
            let segments = word(&counts, 2);
            for n in 0..batch.count {
                log!("sdf trace work: part {} revision {} layer {} kept={} crossings={} coarse_evaluations={} cell_evaluations={}",
                    batch.request.part, batch.request.revision, batch.request.layers[batch.first + n].0,
                    word(&counts, 16 + n * 3), word(&counts, 17 + n * 3), batch.coarse_evaluations, word(&counts, 18 + n * 3));
            }
            if word(&counts, 1) != 0 || word(&counts, 3) != 0 {
                self.kept_per_layer = self.kept_per_layer.max(kept.div_ceil(batch.count as u32).saturating_mul(2));
                self.segments_per_layer = self.segments_per_layer.max(segments.div_ceil(batch.count as u32).saturating_mul(2));
                log!("sdf trace pool overflow: growing to {} cells / {} segments and retrying", kept, segments);
                self.queue.push_front(Queued { request: batch.request, first: batch.first });
            } else {
                let arrived = batch.readback_at.unwrap();
                log!("sdf trace readback: part {} revision {} layers {}..{} {:.3} ms after GPU submit, signal_ms={:.3}", batch.request.part, batch.request.revision,
                    batch.first, batch.first + batch.count, arrived.duration_since(batch.started).as_secs_f64() * 1000.0, arrived.elapsed().as_secs_f64() * 1000.0);
                self.chains.push_back(Chain { layers: batch.request.layers[batch.first..batch.first + batch.count].to_vec(),
                    request: batch.request.clone(), segments: batch.segments.unwrap(), count: segments as usize, started: batch.started });
                let first = batch.first + batch.count;
                if first < batch.request.layers.len() && !batch.request.ticket.is_stale() {
                    self.queue.push_back(Queued { request: batch.request, first });
                }
            }
        }
        // Submission may refuse a full pool. Keep the Arc payload and retry on
        // the next frame; never block the UI or discard a live request.
        while let Some(chain) = self.chains.front() {
            if chain.request.ticket.is_stale() { self.chains.pop_front(); continue }
            let bytes = chain.segments.clone();
            let layers = chain.layers.clone();
            let count = chain.count;
            let request = chain.request.clone();
            let started = chain.started;
            let queued = Instant::now();
            match self.pool.submit_at(request.class, "sdf chain", request.ticket.clone(), move || {
                let work = Instant::now();
                let chained = chain_layers(&bytes, count, &layers);
                log!("sdf trace chain: part {} revision {} {:.3} ms after GPU submit, pool_ms={:.3} chain_ms={:.3}",
                    request.part, request.revision, started.elapsed().as_secs_f64() * 1000.0,
                    work.duration_since(queued).as_secs_f64() * 1000.0, work.elapsed().as_secs_f64() * 1000.0);
                (request.done)(chained);
            }) {
                Ok(job) => { job.detach(); self.chains.pop_front(); }
                Err(e) => { log!("sdf trace chain queued: {:?}", e); break }
            }
        }
        if self.has_work() { cx.redraw_all(); }
    }

    /// One submission per draw. A new focus always runs alone, before a batch.
    pub fn pump(&mut self, cx: &mut Cx) {
        self.poll(cx);
        if !self.submitted_before_draw { self.start_queued(cx, false); }
        self.submitted_before_draw = false;
    }

    /// Handle focus requests in the event turn that receives them. Background
    /// continuations still wait for the next draw's budget.
    pub fn signal(&mut self, cx: &mut Cx) {
        self.poll(cx);
        if !self.submitted_before_draw { self.start_queued(cx, true); }
    }

    fn start_queued(&mut self, cx: &mut Cx, focus_only: bool) {
        if self.active.is_none() {
            self.queue.retain(|q| !q.request.ticket.is_stale() && q.first < q.request.layers.len());
            // Visible edits precede background printability work. Within a
            // class, newly queued focus windows precede the remaining layers.
            let next = self.queue.iter().enumerate().filter(|(_, q)| !focus_only || q.first == 0)
                .min_by_key(|(i, q)| (std::cmp::Reverse(q.request.class), q.first != 0, *i)).map(|(i, _)| i);
            if let Some(q) = next.and_then(|i| self.queue.remove(i)) {
                if let Err(e) = self.start(cx, &q) {
                    if self.last_error.as_ref() != Some(&e) { log!("sdf trace deferred: {}", e); }
                    self.last_error = Some(e);
                    self.queue.push_front(q);
                }
            }
        }
        if self.has_work() { cx.redraw_all(); }
    }

    fn start(&mut self, cx: &mut Cx, q: &Queued) -> Result<(), String> {
        let r = &q.request;
        let (coarse, cell) = prepared(cx, &r.kernels)?;
        if self.prepared.get(&r.part).is_none_or(|p| p.revision != r.revision) {
            let mut pc = 1;
            let mut ops = 0;
            let mut kernel_cost = 0;
            while pc < r.program.texels.len() {
                let h = r.program.texels[pc];
                if h[0] == crate::program::op::KERNEL {
                    kernel_cost += r.kernels[h[1] as usize].air().map_or(0, |g| g.cost);
                } else if h[0] == crate::program::op::MESH {
                    kernel_cost += r.kernels[h[1] as usize].mesh_asset().map_or(0, |m| m.cost());
                }
                pc += if h[0] == crate::program::op::PRIM { 3 } else if h[0] == crate::program::op::XFORM_PUSH { 4 }
                    else if h[0] == crate::program::op::KERNEL { 1 + h[2] as usize } else { 1 };
                ops += 1;
            }
            log!("sdf trace program: part {} revision {} ops_per_sample={} kernel_cost_per_sample={} texels={}",
                r.part, r.revision, ops, kernel_cost, r.program.texels.len());
            let bytes: Vec<_> = r.program.texels.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
            let program = Allocation::new(cx, &self.document, bytes.len() as u64, false, false)?;
            program.buffer.update(cx, 0, &bytes)?;
            let meshes = r.kernels.iter().enumerate().filter_map(|(i, slot)| slot.mesh_asset().map(|asset| (i, asset)))
                .map(|(i, asset)| crate::mesh::upload(cx, &self.document, asset).map(|gpu| (crate::mesh::binding(i), gpu)))
                .collect::<Result<Vec<_>, _>>()?;
            self.prepared.insert(r.part, Prepared { revision: r.revision, program, meshes });
        }
        let nx = (r.bounds.1[0] - r.bounds.0[0] + 2.0).ceil() as u32;
        let ny = (r.bounds.1[1] - r.bounds.0[1] + 2.0).ceil() as u32;
        let remaining = r.layers.len() - q.first;
        let cost = self.cost_per_layer.get(&r.part).copied().unwrap_or(1.5);
        let mut count = if q.first == 0 { 1 } else { ((1.5 / cost.max(0.001)).floor() as usize).max(1).min(remaining) };
        loop {
            let kept = self.kept_per_layer.max((nx + ny).saturating_mul(8)).saturating_mul(count as u32).max(1);
            let segments = self.segments_per_layer.max((nx + ny).saturating_mul(40)).saturating_mul(count as u32).max(1);
            let fits = self.pools.as_ref().is_some_and(|p| p.layers >= count && p.kept_capacity >= kept && p.segment_capacity >= segments);
            if fits { break }
            // Release old buffers before admitting their replacements (there is
            // no in-flight batch here). Admission precedes every allocation.
            self.pools = None;
            let allocated = (|| {
                Ok::<_, String>(Pools {
                    inputs: Allocation::new(cx, &self.document, (4 + count) as u64 * 16, false, false)?,
                    kept: Allocation::new(cx, &self.document, kept as u64 * 16, false, false)?,
                    counts: Allocation::new(cx, &self.document, 64 + count as u64 * 12, true, true)?,
                    segments: Allocation::new(cx, &self.document, segments as u64 * 64, true, false)?,
                    layers: count, kept_capacity: kept, segment_capacity: segments,
                })
            })();
            match allocated {
                Ok(p) => { self.pools = Some(p); break }
                Err(e) if count > 1 => { log!("sdf trace admission: {}; reducing batch", e); count = (count / 2).max(1); }
                Err(e) => return Err(e),
            }
        }
        let p = self.pools.as_ref().unwrap();
        // Retain allocation capacity across frames, but bound this dispatch and
        // readback by the current batch. A former large batch must not make a
        // one-layer focus read back its entire peak allocation.
        let kept_capacity = self.kept_per_layer.max((nx + ny).saturating_mul(8)).saturating_mul(count as u32).max(1).min(p.kept_capacity);
        let segment_capacity = self.segments_per_layer.max((nx + ny).saturating_mul(40)).saturating_mul(count as u32).max(1).min(p.segment_capacity);
        let mut inputs: Vec<_> = r.to_shape.into_iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        for v in [(r.bounds.0[0] - 1.0) as f32, (r.bounds.0[1] - 1.0) as f32, r.lipschitz, 0.0] { inputs.extend(v.to_le_bytes()); }
        for &(_, z) in &r.layers[q.first..q.first + count] {
            for v in [z as f32, 0.0, 0.0, 0.0] { inputs.extend(v.to_le_bytes()); }
        }
        p.inputs.buffer.update(cx, 0, &inputs)?;
        let mut counters = vec![0u8; 64 + count * 12];
        counters[9 * 4..10 * 4].copy_from_slice(&1u32.to_le_bytes());
        counters[10 * 4..11 * 4].copy_from_slice(&1u32.to_le_bytes());
        p.counts.buffer.update(cx, 0, &counters)?;
        for (kernel, name, groups) in [(&coarse, "sdf trace coarse", [(nx * ny).div_ceil(64), count as u32, 1]),
            (&cell, "sdf trace cell", [kept_capacity.min(65535), kept_capacity.div_ceil(65535), 1])] {
            let mut pass = ComputePass::new(name, None);
            let mut params = kernel.params();
            params.set(id!(grid), &[nx, ny, count as u32, kept_capacity].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>())?;
            if kernel.layout().slots.iter().any(|s| s.name == id!(capacity)) {
                params.set_u32(id!(capacity), segment_capacity)?;
            }
            let buffers = kernel.layout().buffers.iter().map(|b| {
                if b.name == id!(program) { self.prepared[&r.part].program.buffer.clone() }
                else if b.name == id!(inputs) { p.inputs.buffer.clone() }
                else if b.name == id!(kept) { p.kept.buffer.clone() }
                else if b.name == id!(counts) { p.counts.buffer.clone() }
                else if b.name == id!(segments) { p.segments.buffer.clone() }
                else { self.prepared[&r.part].meshes.iter().find(|(name, _)| *name == b.name).expect("mesh binding").1.buffer.clone() }
            }).collect::<Vec<_>>();
            if name == "sdf trace cell" {
                pass.dispatch_indirect(kernel, &p.counts.buffer, 8 * 4, &params, &buffers, &[])?;
            } else {
                pass.dispatch(kernel, groups, &params, &buffers, &[])?;
            }
            pass.submit(cx);
        }
        log!("sdf trace submit: part {} key {} layers {}..{} request_ms={:.3}", r.part, r.revision, q.first, q.first + count, r.requested.elapsed().as_secs_f64() * 1000.0);
        self.timed_layers.push_back((r.part, r.revision, q.first, count));
        self.last_error = None;
        self.serial += 1;
        self.submitted_before_draw = true;
        let serial = self.serial;
        self.active = Some(Batch { serial, request: r.clone(), first: q.first, count, started: Instant::now(), coarse_evaluations: nx as u64 * ny as u64 * 4, readback_at: None, counts: None, segments: None });
        for (counts, buffer) in [(true, &p.counts.buffer), (false, &p.segments.buffer)] {
            let tx = self.tx.clone();
            let size = if counts { 64 + count as u64 * 12 } else { segment_capacity as u64 * 64 };
            let readback = cx.read_gpu_buffer(buffer, 0, size, move |result| {
                let _ = tx.send((serial, counts, Instant::now(), result.map_err(|e| format!("{e:?}"))));
                SignalToUI::set_ui_signal();
            });
            if let Err(e) = readback {
                self.active = None;
                return Err(format!("{e:?}"));
            }
        }
        Ok(())
    }
}

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at * 4..at * 4 + 4].try_into().unwrap())
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct Edge(u32, u32, u32);

/// Linear hash chaining, with deterministic loop start/order independent of
/// atomic append order. An open chain is an error, never silently repaired.
fn chain_layers(bytes: &[u8], count: usize, layers: &[(usize, f64)]) -> Result<Vec<TracedLayer>, String> {
    let mut by_layer = vec![HashMap::new(); layers.len()];
    for s in 0..count {
        let at = s * 16;
        let layer = word(bytes, at + 3) as usize;
        let a = Edge(word(bytes, at), word(bytes, at + 1), word(bytes, at + 2));
        if layer >= layers.len() { return Err("trace segment has an invalid layer".into()) }
        if by_layer[layer].insert(a, s).is_some() { return Err(format!("trace layer {} z={} has duplicate outgoing edge {a:?}", layers[layer].0, layers[layer].1)) }
    }
    let mut out = Vec::with_capacity(layers.len());
    for ((index, z), mut edges) in layers.iter().copied().zip(by_layer) {
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
        out.push(TracedLayer { index, z, loops: loops.into_values().collect() });
    }
    Ok(out)
}
