//! Shared production Eval compute parity harness. One eval_program shader call site.
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

pub fn check(cases: Vec<(String, SdfNode, Vec<[f32; 3]>)>) {
    let gpu = Hal::new();
    let backend = if cfg!(target_os = "macos") { backend_id::METAL } else if cfg!(windows) { backend_id::D3D11 } else { backend_id::VULKAN };
    let info = gpu.open(&OpenReq { backend, flags: open_flags::OFFSCREEN_WINDOWS, ..Default::default() });
    assert_eq!(info.backend, backend, "requires native GPU");
    assert_ne!(info.flags & info_flags::COMPUTE, 0);
    let target = match info.dialect { dialect::MSL => ComputeTarget::Msl, dialect::HLSL => ComputeTarget::Hlsl, _ => ComputeTarget::Spirv };
    let mut cx = Cx::new(Box::new(|_, _| {}));
    let mut failures = Vec::new();
    for (index, (name, node, ps)) in cases.into_iter().enumerate() {
        let point_bytes: Vec<_> = ps.iter().flat_map(|p| [p[0], p[1], p[2], 0.0]).flat_map(f32::to_le_bytes).collect();
        let mesh = match &node { SdfNode::Mesh(m) => Some(m), _ => None };
        let windings: Vec<_> = ps.iter().map(|p| mesh.map_or(0.0, |m| m.winding(*p))).collect();
        let mut slots = Vec::new(); kernel::collect(&node, &mut slots).unwrap();
        let prog = program::lower_with(&node, &slots, 1.0).unwrap();
        let expected = twin::eval_many(&prog.texels, &ps, &slots);
        let compiled = cx.with_vm(|vm| {
            vm.capture_errors();
            makepad_draw::script_mod(vm);
            makepad_sdf_engine::compute::script_mod(vm);
            let diagnostics = if mesh.is_some() { "self.mesh_winding(0u, p.xyz), self.mesh_atan2(p.y, p.x)" } else { "0.0, 0.0" };
            let source = format!("use mod.pod.*\nuse mod.math.*\nuse mod.shader\n{{ ..mod.sdf.Eval\nprecise_math: true\nworkgroup_size: 64\nsamples: shader.buffer(vec4f)\nresult: shader.buffer_rw(vec4f)\nmain: fn() {{ for k in 0..12u {{ let i = group_id().x * 768u + local_index() + k * 64u\nif i < {}u {{ let p = self.samples[i]\nself.result[i] = vec4(self.eval_program(0.0, p.x, p.y, p.z), {diagnostics}) }} }} }} }}", ps.len());
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
            point_bytes.clone(), vec![0; ps.len() * 16],
        ];
        for slot in &slots {
            data.push(slot.mesh_asset().unwrap().words.iter().flatten().flat_map(|w| w.to_le_bytes()).collect());
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
        until(&gpu, info.session, |e| matches!(e, HalEvent::PipelineReady { pipeline } if pipeline == pid));
        w.clear();
        let bound: Vec<_> = l.buffers.iter().map(|b| match b.name { n if n == id!(program) => ids[0], n if n == id!(samples) => ids[1], n if n == id!(result) => ids[2], _ => ids[3] }).collect();
        w.frame_begin(index as u64 + 1, FrameMeta::default());
        let buffers = w.put_ids(&bound);
        let dispatch = Dispatch { pipeline: pid, groups: [ps.len().div_ceil(768) as u32, 1, 1], indirect_buffer: Id::NONE, indirect_offset: 0,
            params: w.put_bytes(&vec![0; l.push_size as usize]), buffers: Range { off: buffers.off * 8, len: buffers.len * 8 }, textures: Range::default() };
        w.compute_pass(0, 0, &[dispatch]);
        w.frame_end(Id::NONE, target::COMPUTE);
        w.buffer_readback(ids[2], 1, 0, (ps.len() * 16) as u64);
        gpu.submit(&w.buf());
        let mut result = Vec::new();
        until(&gpu, info.session, |e| {
            if let HalEvent::Readback { req: 1, status, pixels, .. } = e { assert_eq!(status, readback::OK); result = pixels.to_vec(); true } else { false }
        });
        for (i, (want, slot)) in expected.iter().enumerate() {
            let got_slot = f32::from_le_bytes(result[i * 16 + 4..i * 16 + 8].try_into().unwrap());
            let got = u32::from_le_bytes(result[i * 16..i * 16 + 4].try_into().unwrap());
            let angle = u32::from_le_bytes(result[i * 16 + 12..i * 16 + 16].try_into().unwrap());
            let reference = makepad_sdf_engine::makepad_csg_sdf::mesh_field::mesh_atan2(ps[i][1], ps[i][0]);
            if mesh.is_some() { assert_eq!(angle, reference.to_bits(), "{name} atan2 {:?}", ps[i]); }
            if got != want.to_bits() || got_slot != *slot as f32 {
                let winding = f32::from_le_bytes(result[i * 16 + 8..i * 16 + 12].try_into().unwrap());
                failures.push(format!("{name} {:?}: GPU {} slot {got_slot} twin {want} slot {slot}; winding GPU {winding:?} twin {:?}", ps[i], f32::from_bits(got), windings[i]));
            }
        }
        println!("FIELD GPU {target:?} {name}: {} eval_program samples, {} mismatches so far", ps.len(), failures.len());
    }
    assert_eq!(gpu.with_world(|w| w.session(info.session).unwrap().stats.bad_args), 0);
    gpu.close(info.session);
    assert!(failures.is_empty(), "{} mismatches; first:\n{}", failures.len(), failures.iter().take(8).cloned().collect::<Vec<_>>().join("\n"));
}

