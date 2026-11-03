//! Shared production Eval compute parity harness. One program call site per kernel.
use makepad_sdf_engine::{kernel::{self, ComputeBake}, program, twin};
use makepad_sdf_engine::makepad_csg_sdf::SdfNode;
use makepad_sdf_engine::makepad_draw::{self as makepad_draw, *, makepad_platform::makepad_hal as hal};
use makepad_draw::makepad_script::shader_ir_compute::*;
use hal::{codec::ArenaSink, *};
use std::time::{Duration, Instant};

fn until(hal: &Hal, session: u32, mut done: impl FnMut(HalEvent<'_>) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut events = EventStore::new(session, 256, 1024 * 1024);
    loop {
        assert!(Instant::now() < deadline, "GPU completion deadline");
        let mut out = events.buf(); let n = hal.pump(10_000_000, &mut out);
        for (_, e) in events.reader(n, out.arena_len) {
            match &e {
                HalEvent::PipelineFailed { log, .. } => panic!("pipeline: {}", String::from_utf8_lossy(log)),
                HalEvent::DeviceLost { .. } => panic!("device lost"), _ => {}
            }
            if done(e) { return; }
        }
    }
}

/// A native GPU session for the program kernels.
struct Gpu {
    hal: std::sync::Arc<Hal>,
    info: HalInfo,
    target: ComputeTarget,
    cx: Cx,
}

impl Gpu {
    fn new() -> Gpu {
        let hal = Hal::new();
        let backend = if cfg!(target_os = "macos") { backend_id::METAL } else if cfg!(windows) { backend_id::D3D11 } else { backend_id::VULKAN };
        let info = hal.open(&OpenReq { backend, flags: open_flags::OFFSCREEN_WINDOWS, ..Default::default() });
        assert_eq!(info.backend, backend, "requires native GPU");
        assert_ne!(info.flags & info_flags::COMPUTE, 0);
        let target = match info.dialect { dialect::MSL => ComputeTarget::Msl, dialect::HLSL => ComputeTarget::Hlsl, _ => ComputeTarget::Spirv };
        Gpu { hal, info, target, cx: Cx::new(Box::new(|_, _| {})) }
    }

    /// Runs `body` (Splash statements writing `self.result[i]` for result `i`, reading
    /// `self.samples`) for results 0..count over the program of `node` (its kernels compiled
    /// in, its mesh assets bound): the results' words.
    fn run(&mut self, index: usize, node: &SdfNode, samples: &[[f32; 4]], count: usize, body: &str) -> Vec<[u32; 4]> {
        let (info, gpu) = (&self.info, &self.hal);
        let mut slots = Vec::new(); kernel::collect(node, &mut slots).unwrap();
        let prog = program::lower_with(node, &slots, 1.0).unwrap();
        let target = self.target;
        let compiled = self.cx.with_vm(|vm| {
            vm.capture_errors();
            makepad_draw::script_mod(vm);
            makepad_sdf_engine::compute::script_mod(vm);
            let source = format!("use mod.pod.*\nuse mod.math.*\nuse mod.shader\n{{ ..mod.sdf.Eval\nprecise_math: true\nworkgroup_size: 64\nsamples: shader.buffer(vec4f)\nresult: shader.buffer_rw(vec4f)\nmain: fn() {{ for k in 0..12u {{ let i = group_id().x * 768u + local_index() + k * 64u\nif i < {count}u {{ {body} }} }} }} }}");
            let base = vm.eval(ScriptMod::new("", "", "mesh_eval", 0, 0, &source, &[])).as_object().unwrap();
            let _keep = vm.new_object_ref(base);
            let mut route = ComputeBake::new(vm, &slots).unwrap();
            let object = route.draw_variant(vm, base).unwrap();
            let out = compile_compute_kernel(vm, object, target).unwrap();
            assert!(vm.take_errors().is_empty());
            route.release(vm);
            out
        });
        let pid = Id::new(index as u32 + 1, 1);
        let mut data = vec![
            prog.texels.iter().flatten().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>(),
            samples.iter().flatten().flat_map(|f| f.to_le_bytes()).collect(), vec![0; count * 16],
        ];
        // A mesh slot's words are a buffer; a Splash slot is code in the variant.
        for asset in slots.iter().filter_map(|s| s.mesh_asset()) {
            data.push(asset.words.iter().flatten().flat_map(|w| w.to_le_bytes()).collect());
        }
        let ids: Vec<_> = (0..data.len() as u32).map(|i| Id::new(index as u32 * 4 + i + 1, 1)).collect();
        let mut w = CmdWriter::new(info.session);
        for (id, bytes) in ids.iter().zip(&data) {
            w.buffer_create(*id, buffer_kind::STORAGE | buffer_kind::READBACK, bytes.len() as u64, Attr::default());
            w.buffer_update(*id, 0, bytes);
        }
        let l = &compiled.layout;
        let code = match &compiled.code { KernelCode::Spirv(words) => words.iter().flat_map(|w| w.to_le_bytes()).collect(), KernelCode::Msl(s) | KernelCode::Hlsl(s) => s.as_bytes().to_vec() };
        let bindings: Vec<_> = l.buffers.iter().map(|b| StorageBinding { index: b.index, offset: b.offset, writable: b.writable as u32 }).collect();
        let blob = CmdWriter::compute_pipeline_blob(PipelineDesc::default(), &code, ComputeDesc {
            push_size: l.push_size, param_binding: l.param_binding.unwrap_or(u32::MAX), param_binding_size: l.param_binding_size,
            group_count_offset: l.group_count_offset.unwrap_or(u32::MAX), workgroup: l.workgroup, precise_math: l.precise_math as u32, ..Default::default()
        }, &bindings, &[]);
        w.pipeline_create(pid, info.dialect, Attr::default(), &blob);
        gpu.submit(&w.buf());
        until(gpu, info.session, |e| matches!(e, HalEvent::PipelineReady { pipeline } if pipeline == pid));
        w.clear();
        let bound: Vec<_> = l.buffers.iter().map(|b| match b.name { n if n == id!(program) => ids[0], n if n == id!(samples) => ids[1], n if n == id!(result) => ids[2], _ => ids[3] }).collect();
        w.frame_begin(index as u64 + 1, FrameMeta::default());
        let buffers = w.put_ids(&bound);
        let dispatch = Dispatch { pipeline: pid, timer: 0, groups: [count.div_ceil(768) as u32, 1, 1], indirect_buffer: Id::NONE, indirect_offset: 0,
            params: w.put_bytes(&vec![0; l.push_size as usize]), buffers: Range { off: buffers.off * 8, len: buffers.len * 8 }, textures: Range::default() };
        w.compute_pass(0, 0, &[dispatch]);
        w.frame_end(Id::NONE, target::COMPUTE);
        w.buffer_readback(ids[2], 1, 0, (count * 16) as u64);
        gpu.submit(&w.buf());
        let mut result = Vec::new();
        until(gpu, info.session, |e| {
            if let HalEvent::Readback { req: 1, status, pixels, .. } = e { assert_eq!(status, readback::OK); result = pixels.to_vec(); true } else { false }
        });
        result.chunks_exact(16).map(|c| std::array::from_fn(|k| u32::from_le_bytes(c[k * 4..k * 4 + 4].try_into().unwrap()))).collect()
    }

    fn close(self) {
        assert_eq!(self.hal.with_world(|w| w.session(self.info.session).unwrap().stats.bad_args), 0);
        self.hal.close(self.info.session);
    }
}

fn report(failures: Vec<String>) {
    assert!(failures.is_empty(), "{} mismatches; first:\n{}", failures.len(), failures.iter().take(8).cloned().collect::<Vec<_>>().join("\n"));
}

pub fn check(cases: Vec<(String, SdfNode, Vec<[f32; 3]>)>) {
    let mut gpu = Gpu::new();
    let mut failures = Vec::new();
    for (index, (name, node, ps)) in cases.into_iter().enumerate() {
        let mesh = match &node { SdfNode::Mesh(m) => Some(m), _ => None };
        let windings: Vec<_> = ps.iter().map(|p| mesh.map_or(0.0, |m| m.winding(*p))).collect();
        let mut slots = Vec::new(); kernel::collect(&node, &mut slots).unwrap();
        let prog = program::lower_with(&node, &slots, 1.0).unwrap();
        let expected = twin::eval_many(&prog.texels, &ps, &slots);
        let diagnostics = if mesh.is_some() { "self.mesh_winding(0u, p.xyz), self.mesh_atan2(p.y, p.x)" } else { "0.0, 0.0" };
        let samples: Vec<_> = ps.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        let result = gpu.run(index, &node, &samples, ps.len(), &format!("let p = self.samples[i]\nself.result[i] = vec4(self.eval_program(0.0, p.x, p.y, p.z), {diagnostics})"));
        for (i, (want, slot)) in expected.iter().enumerate() {
            let [got, got_slot, winding, angle] = result[i];
            let got_slot = f32::from_bits(got_slot);
            let reference = makepad_sdf_engine::makepad_csg_sdf::mesh_field::mesh_atan2(ps[i][1], ps[i][0]);
            if mesh.is_some() { assert_eq!(angle, reference.to_bits(), "{name} atan2 {:?}", ps[i]); }
            if got != want.to_bits() || got_slot != *slot as f32 {
                failures.push(format!("{name} {:?}: GPU {} slot {got_slot} twin {want} slot {slot}; winding GPU {:?} twin {:?}", ps[i], f32::from_bits(got), f32::from_bits(winding), windings[i]));
            }
        }
        println!("FIELD GPU {:?} {name}: {} eval_program samples, {} mismatches so far", gpu.target, ps.len(), failures.len());
    }
    gpu.close();
    report(failures);
}

/// The known masks (`run_program`'s): per case, pairs (p, q) with q within half a cell's
/// diagonal of p. The program at p with the bake's `cell` names the textures whose masks it
/// proves constant out to there; told those, the program at q gives the twin's value at q bit
/// for bit. Returns per case the pairs proven (masks 1, masks 0).
pub fn check_known_masks(cases: Vec<(String, SdfNode, Vec<([f32; 3], [f32; 3])>)>, cell: f32) -> Vec<(usize, usize)> {
    let mut gpu = Gpu::new();
    let mut failures = Vec::new();
    let mut proven = Vec::new();
    for (index, (name, node, pairs)) in cases.into_iter().enumerate() {
        let mut slots = Vec::new(); kernel::collect(&node, &mut slots).unwrap();
        let prog = program::lower_with(&node, &slots, 1.0).unwrap();
        let ps: Vec<_> = pairs.iter().map(|(p, _)| *p).collect();
        let qs: Vec<_> = pairs.iter().map(|(_, q)| *q).collect();
        let at_p = twin::eval_many(&prog.texels, &ps, &slots);
        let at_q = twin::eval_many(&prog.texels, &qs, &slots);
        let samples: Vec<_> = pairs.iter().flat_map(|(p, q)| [[p[0], p[1], p[2], 0.0], [q[0], q[1], q[2], 0.0]]).collect();
        let result = gpu.run(index, &node, &samples, pairs.len(), &format!("let p = self.samples[i * 2u]\nlet q = self.samples[i * 2u + 1u]\n\
            let a = self.run_program(0.0, p.x, p.y, p.z, 0u, {cell:?})\n\
            let b = self.run_program(0.0, q.x, q.y, q.z, u32(a.z) | (u32(a.w) << 16u), 0.0)\n\
            self.result[i] = vec4(a.x, b.x, a.z, a.w)"));
        let (mut ones, mut zeros) = (0, 0);
        for (i, (p, q)) in pairs.iter().enumerate() {
            let [a, b, known, one] = result[i].map(f32::from_bits);
            if a.to_bits() != at_p[i].0.to_bits() {
                failures.push(format!("{name} {p:?}: GPU {a} twin {}", at_p[i].0));
            }
            if known == 0.0 { continue; }
            if one == known { ones += 1 } else { zeros += 1 }
            if b.to_bits() != at_q[i].0.to_bits() {
                failures.push(format!("{name} {p:?} -> {q:?}: masks {known}/{one} give {b}, twin {}", at_q[i].0));
            }
        }
        println!("KNOWN MASKS GPU {:?} {name}: {} pairs, proven 1 at {ones}, 0 at {zeros}, {} mismatches so far", gpu.target, pairs.len(), failures.len());
        proven.push((ones, zeros));
    }
    gpu.close();
    report(failures);
    proven
}

