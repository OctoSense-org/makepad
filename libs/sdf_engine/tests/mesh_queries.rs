use makepad_sdf_engine::{kernel::{self, ComputeBake}, mesh::MeshAsset, program, twin};
use makepad_sdf_engine::makepad_csg_sdf::SdfNode;
use makepad_sdf_engine::makepad_draw::{*, makepad_platform::makepad_hal as hal};
use makepad_draw::makepad_script::shader_ir_compute::*;
use hal::{codec::ArenaSink, *};
use std::{sync::Arc, time::{Duration, Instant}};

#[path = "support/meshes.rs"]
mod meshes;
use meshes::corpus;

static GPU_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn mesh_atan2_matches_f64_dense() {
    use makepad_sdf_engine::makepad_csg_sdf::mesh_field::mesh_atan2;
    let mut worst = 0.0f64;
    let mut count = 0;
    let mut check = |y: f32, x: f32| {
        let got = mesh_atan2(y, x);
        let expected = (y as f64).atan2(x as f64);
        let error = (got as f64 - expected).abs();
        worst = worst.max(error);
        count += 1;
        assert!(error <= 1e-6, "atan2({y:?}, {x:?})={got:?}, f64={expected:?}, error={error}");
        if y == 0.0 || x == 0.0 {
            assert_eq!(got.to_bits(), (expected as f32).to_bits(), "axis/zero atan2({y:?}, {x:?})");
        }
    };
    for y in -128..=128 { for x in -128..=128 { check(y as f32 / 128.0, x as f32 / 128.0); } }
    for i in 0..16384 {
        let a = i as f64 * std::f64::consts::TAU / 16384.0;
        for scale in [1e-30f32, 1e-10, 1.0, 1e30] { check(a.sin() as f32 * scale, a.cos() as f32 * scale); }
    }
    for y in [-f32::INFINITY, -1.0, -0.0, 0.0, 1.0, f32::INFINITY] {
        for x in [-f32::INFINITY, -1.0, -0.0, 0.0, 1.0, f32::INFINITY] { check(y, x); }
    }
    println!("MESH atan2: {count} samples, worst {worst:e} rad against f64");
}

fn points() -> Vec<[f32; 3]> {
    let mut ps = Vec::new();
    // Faces, edges, vertices, both sides of the surface, interior and far dipoles.
    for a in 0..3 { for side in [-1.0, 1.0] { for offset in [-0.001, 0.0, 0.001] {
        for u in [-1.0, -0.37, 0.0, 0.63, 1.0] { for v in [-1.0, -0.29, 0.0, 0.71, 1.0] {
            let mut p = [0.0; 3]; p[a] = side + offset; p[(a + 1) % 3] = u; p[(a + 2) % 3] = v;
            ps.push(p);
        } }
    } } }
    ps.extend([[0.0; 3], [0.5, 0.25, -0.1], [8.0, -9.0, 12.0]]);
    // Also query exactly the lattice exercised by the production bake test.
    ps.extend((0..729).map(|i| [(i % 9) as f32 * 0.25, (i / 81) as f32 * 0.25, ((i / 9) % 9) as f32 * 0.25]));
    for x in [-1.0, -0.0, 0.0, 1.0] { for y in [-1.0, -0.0, 0.0, 1.0] { ps.push([x, y, 0.3]); } }
    ps
}

#[test]
fn mesh_twin_matches_signed_and_signed64() {
    let ps = points();
    for (name, mesh) in corpus() {
        let asset = MeshAsset::new(&mesh);
        if name == "cup" {
            let p = [-0.29, 1.0, -0.37];
            // The missing lid lies exactly on the winding threshold. The fixed
            // arithmetic chooses the inside sign here on every backend.
            assert_eq!(mesh.signed(p).to_bits(), (-0.63f32).to_bits());
            println!("MESH cup threshold regression: distance {:?}, winding {:?}", mesh.signed(p), mesh.winding(p));
        }
        for p in &ps {
            let got = asset.signed(*p);
            assert_eq!(got.to_bits(), mesh.signed(*p).to_bits(), "{name} {p:?}: packed traversal");
            let truth = mesh.signed64(makepad_csg_math::Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64));
            assert!((got as f64 - truth).abs() < 1e-5, "{name} {p:?}: f64 error {}", (got as f64 - truth).abs());
        }
        let node = SdfNode::Mesh(Arc::new(mesh));
        let mut slots = Vec::new(); kernel::collect(&node, &mut slots).unwrap();
        let prog = program::lower_with(&node, &slots, 1.0).unwrap();
        assert_eq!(prog.texels[1][0], program::op::MESH);
        let many = twin::eval_many(&prog.texels, &ps, &slots);
        for (p, got) in ps.iter().zip(many) {
            assert_eq!(got.0.to_bits(), asset.signed(*p).to_bits(), "{name} batch {p:?}");
            assert_eq!(twin::eval_with(&prog.texels, *p, &slots).0.to_bits(), got.0.to_bits());
        }
        println!("MESH twin {name}: {} points bit-exact signed, signed64 error < 1e-5 mm", ps.len());
    }
}

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

#[path = "support/eval_program.rs"]
mod eval_program;

#[test]
fn gpu_eval_program_matches_mesh_twin() {
    let _gpu_test = GPU_TEST.lock().unwrap_or_else(|p| p.into_inner());
    eval_program::check(corpus().into_iter().map(|(name, mesh)|
        (name.into(), SdfNode::Mesh(Arc::new(mesh)), points())).collect());
}

/// Exercise the actual 9^3 bake, including its apron and half conversion. The
/// timings are observations, never test thresholds; exclusive runs use devhub.
#[test]
fn gpu_mesh_brick_matches_twin() {
    let _gpu_test = GPU_TEST.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = Hal::new();
    let backend = if cfg!(target_os = "macos") { backend_id::METAL } else if cfg!(windows) { backend_id::D3D11 } else { backend_id::VULKAN };
    let info = gpu.open(&OpenReq { backend, flags: open_flags::OFFSCREEN_WINDOWS, ..Default::default() });
    assert_eq!(info.backend, backend);
    let target = match info.dialect { dialect::MSL => ComputeTarget::Msl, dialect::HLSL => ComputeTarget::Hlsl, _ => ComputeTarget::Spirv };
    let mesh = corpus().remove(2).1;
    let node = SdfNode::Mesh(Arc::new(mesh.clone()));
    let mut slots = Vec::new(); kernel::collect(&node, &mut slots).unwrap();
    let prog = program::lower_with(&node, &slots, 1.0).unwrap();
    let mut cx = Cx::new(Box::new(|_, _| {}));
    let compiled = cx.with_vm(|vm| {
        makepad_draw::script_mod(vm); makepad_sdf_engine::compute::script_mod(vm);
        let bake = ComputeBake::new(vm, &slots).unwrap();
        let out = bake.compile(vm, target).unwrap(); bake.release(vm); out
    });
    let l = &compiled.layout;
    let program = Id::new(1, 1); let bricks = Id::new(2, 1); let asset = Id::new(3, 1); let slopes = Id::new(4, 1);
    let atlas = Id::new(1, 1); let materials = Id::new(2, 1); let pipeline = Id::new(1, 1);
    let mut w = CmdWriter::new(info.session);
    // HAL upload commands borrow these bytes until submit; retain every input.
    let inputs = [
        (program, prog.texels.iter().flatten().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>()),
        (bricks, [0u32; 8].into_iter().flat_map(u32::to_le_bytes).collect()),
        (asset, slots[0].mesh_asset().unwrap().words.iter().flatten().flat_map(|w| w.to_le_bytes()).collect()),
        // the one slot's slope bound and known masks, which the bake writes
        (slopes, vec![0u8; 16]),
    ];
    for (id, bytes) in &inputs {
        w.buffer_create(*id, buffer_kind::STORAGE, bytes.len() as u64, Attr::default()); w.buffer_update(*id, 0, bytes);
    }
    w.texture_create(atlas, usage::STORAGE | usage::SAMPLED, format::R16F, 81, 9, [1, 1], Attr::default());
    w.texture_create(materials, usage::STORAGE | usage::SAMPLED, format::R8, 81, 9, [1, 1], Attr::default());
    let code = match &compiled.code { KernelCode::Spirv(words) => words.iter().flat_map(|w| w.to_le_bytes()).collect(), KernelCode::Msl(s) | KernelCode::Hlsl(s) => s.as_bytes().to_vec() };
    let buffers: Vec<_> = l.buffers.iter().map(|b| StorageBinding { index: b.index, offset: b.offset, writable: b.writable as u32 }).collect();
    let images: Vec<_> = l.storage_textures.iter().map(|t| StorageImageBinding { index: t.index, format: if t.name == id!(atlas) { format::R16F } else { format::R8 } }).collect();
    let blob = CmdWriter::compute_pipeline_blob(PipelineDesc::default(), &code, ComputeDesc {
        push_size: l.push_size, param_binding: l.param_binding.unwrap_or(u32::MAX), param_binding_size: l.param_binding_size,
        group_count_offset: l.group_count_offset.unwrap_or(u32::MAX), workgroup: l.workgroup, precise_math: l.precise_math as u32, ..Default::default()
    }, &buffers, &images);
    w.pipeline_create(pipeline, info.dialect, Attr::default(), &blob);
    gpu.submit(&w.buf());
    until(&gpu, info.session, |e| matches!(e, HalEvent::PipelineReady { .. }));
    let truth = twin::bake_brick(&prog.texels, [0; 3], 0.25, 4.0, &slots);
    let mut params = vec![0u8; l.push_size as usize];
    for slot in &l.slots {
        let value = match slot.name { n if n == id!(cell) => 0.25f32.to_bits(), n if n == id!(band) => 4.0f32.to_bits(), n if n == id!(count) => 1, _ => 0u32 };
        params[slot.offset as usize..slot.offset as usize + 4].copy_from_slice(&value.to_le_bytes());
    }
    let bound: Vec<_> = l.buffers.iter().map(|b| match b.name { n if n == id!(program) => program, n if n == id!(bricks) => bricks, n if n == id!(slopes) => slopes, _ => asset }).collect();
    let textures: Vec<_> = l.storage_textures.iter().map(|t| if t.name == id!(atlas) { atlas } else { materials }).collect();
    assert_eq!(gpu.with_world(|w| w.session(info.session).unwrap().stats.bad_args), 0, "bake setup");
    let points: Vec<_> = (0..729).map(|i| [(i % 9) as f32 * 0.25, (i / 81) as f32 * 0.25, ((i / 9) % 9) as f32 * 0.25]).collect();
    for run in 0..3 {
        w.clear(); w.frame_begin(run + 1, FrameMeta::default());
        let b = w.put_ids(&bound); let t = w.put_ids(&textures);
        let dispatch = Dispatch { pipeline, timer: 0, groups: [1, 1, 1], indirect_buffer: Id::NONE, indirect_offset: 0, params: w.put_bytes(&params),
            buffers: Range { off: b.off * 8, len: b.len * 8 }, textures: Range { off: t.off * 8, len: t.len * 8 } };
        w.compute_pass(0, 0, &[dispatch]); w.frame_end(Id::NONE, target::COMPUTE);
        let start = Instant::now(); gpu.submit(&w.buf());
        let mut time = 0;
        until(&gpu, info.session, |e| {
            if let HalEvent::FrameDone { gpu_ns, .. } = e { time = gpu_ns; true } else { false }
        });
        println!("MESH bake completed {target:?}: run={run}, gpu_ns={time}");
        w.clear();
        w.texture_readback(atlas, 1, [0, 0, 81, 9]); w.texture_readback(materials, 2, [0, 0, 81, 9]);
        gpu.submit(&w.buf());
        let mut distance = None; let mut material = None;
        until(&gpu, info.session, |e| {
            match e {
                HalEvent::Readback { req, status, pixels, .. } => { assert_eq!(status, readback::NATIVE); if req == 1 { distance = Some(pixels.to_vec()); } else { material = Some(pixels.to_vec()); } }
                _ => {}
            }
            distance.is_some() && material.is_some()
        });
        let wall = start.elapsed();
        let bytes = distance.unwrap(); assert_eq!(bytes.len(), 729 * 2);
        for (i, bytes) in bytes.chunks_exact(2).enumerate() {
            let got = makepad_draw::makepad_math::f16_bits_to_f32(u16::from_le_bytes(bytes.try_into().unwrap()));
            assert_eq!(got.to_bits(), truth.d[i].to_bits(), "baked apron sample {i}");
        }
        assert_eq!(material.unwrap(), truth.slots);
        // Observe the mesh batch oracle with the current deterministic angle.
        // A full old-twin baseline must run at the pre-MESH revision separately.
        let cpu = Instant::now();
        let batch: Vec<_> = mesh.signed_many(&points).into_iter().map(|d| twin::to_half(d.max(-4.0).min(4.0) + 0.0)).collect();
        let cpu_wall = cpu.elapsed(); std::hint::black_box(batch);
        println!("MESH brick {target:?} run={run} triangles={} samples=729 gpu_ns={time} submit_readback_ns={} signed_many_ns={}", mesh.data.triangles.len(), wall.as_nanos(), cpu_wall.as_nanos());
    }
    assert_eq!(gpu.with_world(|w| w.session(info.session).unwrap().stats.bad_args), 0);
    gpu.close(info.session);
}
