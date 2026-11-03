//! A knurled cylinder (a texture node over a cylinder: a diamond knurl round Y, as m.texture
//! lowers one): the program's CPU twin against csg_sdf's f64 field, and the GPU's eval_program
//! against the twin bit for bit, with the masks' gradient, without it, triplanar and a turned
//! projection axis.
use makepad_sdf_engine::makepad_csg_sdf::{self as sdf, texture::angle_mask, SdfKernel, SdfNode, SdfSplashExpr, SdfTexture, SplashPoint, TexProject};
use makepad_sdf_engine::makepad_draw::{self as makepad_draw, *};
use makepad_sdf_engine::{kernel, program, twin};
use std::sync::Arc;

#[path = "support/eval_program.rs"]
mod eval_program;

/// The knurl through its cylindrical mapping, in metres (94 cells round a 15mm radius, cells
/// 1mm / tan 30deg up, flat tops of a quarter, faded out inside 14.2mm down to 7.1mm).
const KNURL: &str = "use mod.math.*
let tex_nu = 94.0
let tex_tv = 577.3503
let knurl_flat = 0.25
let knurl_gain = 1.3333334
let tex_r0 = 0.0071
let tex_r1 = 0.0142
let knurl = fn(p) {
    let r = sqrt(p.x * p.x + p.z * p.z)
    let u = tex_nu * (atan2(p.z, p.x) / TAU)
    let v = tex_tv * p.y
    let a = 2.0 * abs(fract(u + v) - 0.5)
    let b = 2.0 * abs(fract(u - v) - 0.5)
    return clamp((r - tex_r0) / (tex_r1 - tex_r0), 0.0, 1.0) * (max(max(a, b) - knurl_flat, 0.0) * knurl_gain)
}
(knurl)";

/// The cases: (name, a textured cylinder 15mm in radius and 20mm high).
fn cases() -> Vec<(String, SdfNode)> {
    let mut cx = Cx::new(Box::new(|_, _| {}));
    let kernel = cx.with_vm(|vm| {
        vm.capture_errors();
        makepad_draw::script_mod(vm);
        let f = vm.eval(ScriptMod::new("", "", "knurl", 0, 0, KNURL, &[])).as_object().unwrap();
        let _keep = vm.new_object_ref(f);
        let field = SdfSplashExpr::compile(vm, f, SplashPoint::Vec3, &[]).unwrap();
        assert!(vm.take_errors().is_empty());
        SdfKernel::splash(field, None)
    });
    let m = angle_mask(20f64.to_radians(), 15f64.to_radians());
    let cylinder = || Box::new(SdfNode::Cylinder { radius: 15.0, top_radius: 15.0, height: 20.0 });
    let tex = |project, masks: bool, depth| Arc::new(SdfTexture::new(kernel.clone(), project, 0.001, depth, 1.0, masks.then_some(m).flatten(), masks.then_some(m).flatten(), 4.2, 0.0));
    vec![
        ("masked".into(), SdfNode::Texture { tex: tex(TexProject::Axis(1), true, 0.4), of: cylinder() }),
        ("unmasked".into(), SdfNode::Texture { tex: tex(TexProject::Axis(1), false, 0.4), of: cylinder() }),
        ("triplanar raised".into(), SdfNode::Texture { tex: tex(TexProject::Triplanar, false, -0.3), of: cylinder() }),
        ("about x".into(), SdfNode::Transform { at: [1.0, 2.0, 0.0], rotate: [0.0, 0.0, 90.0], scale: 1.0, of: Box::new(SdfNode::Texture { tex: tex(TexProject::Axis(0), true, 0.4), of: cylinder() }) }),
    ]
}

/// Round the side (at, inside and outside the surface, the seam's half turn included), over
/// the masked rims and the top face.
fn points() -> Vec<[f32; 3]> {
    let mut ps = Vec::new();
    for i in 0..96 {
        let a = i as f64 * 0.0731 + if i % 8 == 0 { std::f64::consts::PI - 1e-4 } else { 0.0 };
        for r in [14.5, 14.8, 15.0, 15.2] {
            for y in [-10.6, -9.3, -4.0, 0.0, 3.7, 9.1, 10.4] {
                ps.push([(r * a.cos()) as f32, y as f32, (r * a.sin()) as f32]);
            }
        }
    }
    for x in [0.0f32, 4.0, 9.5, 13.0] {
        for y in [9.8f32, 10.0, 10.3] {
            ps.push([x, y, 2.5]);
        }
    }
    ps
}

#[test]
fn texture_twin_matches_f64() {
    let ps = points();
    for (name, n) in cases() {
        let mut slots = Vec::new();
        kernel::collect(&n, &mut slots).unwrap();
        let p = program::lower_with(&n, &slots, 1.0).unwrap();
        assert!(p.value_depth <= program::MAX_VALUE_DEPTH, "{name}: depth {}", p.value_depth);
        let got = twin::eval_many(&p.texels, &ps, &slots);
        let mut worst = 0.0f64;
        for (q, (d, slot)) in ps.iter().zip(&got) {
            assert_eq!((*d, *slot), twin::eval_with(&p.texels, *q, &slots), "{name}: batched {q:?}");
            let want = sdf::distance(&n, makepad_csg_math::Vec3d::new(q[0] as f64, q[1] as f64, q[2] as f64));
            worst = worst.max((*d as f64 - want).abs());
        }
        assert!(worst < 2e-4, "{name}: twin against f64 {worst}");
        println!("TEXTURE twin {name}: {} samples, worst {worst:e} mm against f64", ps.len());
    }
}

#[test]
fn gpu_eval_program_matches_texture_twin() {
    eval_program::check(cases().into_iter().map(|(name, n)| (name, n, points())).collect());
}

/// The masks a bake proves constant (the side's 1, the top's 0, none at the rims): told them,
/// the program skips the gradient's corners (and, masked off, the pattern) and still gives the
/// twin's value bit for bit, anywhere within half a cell's diagonal of the proving sample.
#[test]
fn gpu_known_masks_match_texture_twin() {
    // fine enough that the top's ramp (it ends at cos 15 degrees) leaves the flat top room
    const CELL: f32 = 0.02;
    let reach = CELL * 0.86;
    let step = [[1.0f32, 1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]];
    let pairs: Vec<_> = points().into_iter().enumerate()
        .map(|(i, p)| (p, std::array::from_fn(|k| p[k] + step[i % 4][k] * reach / 3f32.sqrt())))
        .collect();
    let proven = eval_program::check_known_masks(cases().into_iter().map(|(name, n)| (name, n, pairs.clone())).collect(), CELL);
    // the masked single projections prove both, on the side and the top; no masks (no
    // gradient) and triplanar nothing
    for (name, (ones, zeros)) in ["masked", "unmasked", "triplanar raised", "about x"].iter().zip(&proven) {
        match *name {
            "masked" | "about x" => assert!(*ones > 100 && *zeros > 0, "{name}: {ones} {zeros}"),
            _ => assert_eq!((*ones, *zeros), (0, 0), "{name}"),
        }
    }
}
