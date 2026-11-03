//! A gridded kernel (a Splash field reading a grid through the kernel prelude's `sdf_grid3`):
//! its grid travels as its program's payload, and the GPU's eval_program reads it there bit for
//! bit as the CPU twin does: moved and scaled inside a union, and as a brick's culled program
//! holding only its piece of the grid (which reads the same bits as the whole); a root scale
//! (the payload one texel on) keeps the offset right. And the prelude's hashes,
//! whose integer multiply-adds overflow and must wrap alike on every backend.
use makepad_sdf_engine::makepad_csg_sdf::{cull, SdfGrid, SdfKernel, SdfNode, SdfOp, SdfSplashExpr, SplashPoint};
use makepad_sdf_engine::makepad_draw::{self as makepad_draw, *};
use makepad_sdf_engine::{kernel, program, twin};

#[path = "support/eval_program.rs"]
mod eval_program;

const DIMS: [u32; 3] = [12, 9, 14];

/// A kernel over a 12 x 9 x 14 grid of uneven values (a hash), 0.5 apart from (-3, -2, -3.5).
fn gridded() -> SdfKernel {
    let data: Vec<f32> = (0..DIMS.iter().product::<u32>()).map(|i| (i.wrapping_mul(2654435761) % 1000) as f32 / 400.0 - 0.6).collect();
    let grid = SdfGrid::new([-3.0, -2.0, -3.5], 0.5, DIMS, 0.0, data.into());
    let mut cx = Cx::new(Box::new(|_, _| {}));
    cx.with_vm(|vm| {
        vm.capture_errors();
        makepad_draw::script_mod(vm);
        let f = vm.eval(ScriptMod::new("", "", "grid", 0, 0, "use mod.math.*\nlet f = |p| sdf_grid3(p) + 0.1 * p.x\n(f)", &[])).as_object().unwrap();
        let _keep = vm.new_object_ref(f);
        let field = SdfSplashExpr::compile_gridded(vm, f, SplashPoint::Vec3, &[], grid).unwrap();
        assert!(vm.take_errors().is_empty());
        SdfKernel::splash(field, None)
    })
}

/// A kernel of the prelude's lattice hash and an i32 multiply-add that overflows.
fn hashed() -> SdfKernel {
    let src = "use mod.math.*\nlet f = |p| length(p) - 2.0 + lattice3(floor(p.x * 3.0), floor(p.y * 3.0), floor(p.z * 3.0)) * 0.1 + float(((int(p.x * 100000.0) * 1664525 + 1013904223) >> 8) & 255) * 0.0001\n(f)";
    let mut cx = Cx::new(Box::new(|_, _| {}));
    cx.with_vm(|vm| {
        vm.capture_errors();
        makepad_draw::script_mod(vm);
        let f = vm.eval(ScriptMod::new("", "", "hashed", 0, 0, src, &[])).as_object().unwrap();
        let _keep = vm.new_object_ref(f);
        let field = SdfSplashExpr::compile(vm, f, SplashPoint::Vec3, &[]).unwrap();
        assert!(vm.take_errors().is_empty());
        SdfKernel::splash(field, None)
    })
}

/// Points over the grid and past its edges.
fn points() -> Vec<[f32; 3]> {
    let mut s = 4242u32;
    let mut r = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 8) as f32 / 16777216.0
    };
    (0..256).map(|_| [r() * 8.0 - 4.0, r() * 6.0 - 3.0, r() * 9.0 - 4.5]).collect()
}

#[test]
fn gpu_reads_a_kernels_grid_and_wraps_its_hashes_as_the_twin_does() {
    let k = SdfNode::Kernel(gridded());
    let moved = SdfNode::Combine {
        op: SdfOp::Union,
        blend: 0.3,
        of: vec![SdfNode::Sphere { radius: 0.8 }, SdfNode::Transform { at: [0.4, -0.2, 0.1], rotate: [0.0, 30.0, 0.0], scale: 1.5, of: Box::new(k.clone()) }],
    };
    let region = ([-1.0, -0.5, -1.0], [0.5, 1.0, 0.75]);
    let culled = cull(&k, &region).expect("the kernel reaches the region");
    let SdfNode::Kernel(piece) = &culled else { panic!("{culled:?}") };
    assert!(piece.splash.as_ref().and_then(|s| s.grid()).unwrap().data.len() * 4 < DIMS.iter().product::<u32>() as usize, "a brick's piece");
    let inside: Vec<[f32; 3]> = points().iter().map(|p| std::array::from_fn(|a| region.0[a] as f32 + (p[a] + 4.5) / 9.0 * (region.1[a] - region.0[a]) as f32)).collect();
    // the twin of the culled program reads the piece as the whole program reads the grid
    let lowered = |n: &SdfNode| {
        let mut slots = Vec::new();
        kernel::collect(n, &mut slots).unwrap();
        (program::lower_with(n, &slots, 1.0).unwrap(), slots)
    };
    let ((whole, ws), (part, ps)) = (lowered(&k), lowered(&culled));
    let (a, b) = (twin::eval_many(&whole.texels, &inside, &ws), twin::eval_many(&part.texels, &inside, &ps));
    for (i, p) in inside.iter().enumerate() {
        assert_eq!(a[i].0.to_bits(), b[i].0.to_bits(), "at {p:?}: whole {} piece {}", a[i].0, b[i].0);
    }
    // a root scale inserts a texel before the payload: the op's offset follows it
    let scaled = whole.clone().scaled(0.5);
    let c = twin::eval_many(&scaled.texels, &inside, &ws);
    for i in 0..inside.len() {
        assert_eq!(c[i].0.to_bits(), (a[i].0 * 0.5).to_bits(), "scaled at {:?}", inside[i]);
    }
    eval_program::check(vec![("moved in a union".into(), moved, points()), ("a brick's piece".into(), culled, inside), ("hashes".into(), SdfNode::Kernel(hashed()), points())]);
}
