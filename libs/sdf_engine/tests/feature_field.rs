//! feature_field::steps on the gpu_jobs runtime against its f64 mirror on analytic
//! FeatureSets (feature_set::from_loops), and byte for byte at every job size.
use makepad_sdf_engine::{feature_field::*, feature_set::{from_loops, FeatureSet, Loop}, gpu_jobs::{self, BufRef, Elements, Finished, JobClass, Output, Pass, Pool, Step}};
use makepad_sdf_engine::plane_abi::{NONE, SEG_REFLEX, VERTEX_BIT};
use makepad_sdf_engine::makepad_draw::{self as makepad_draw, *};
use std::{cell::RefCell, collections::HashMap, rc::Rc, time::Instant};

#[path = "support/gpu_rig.rs"]
mod gpu_rig;
#[path = "support/loops.rs"]
mod loops;
#[path = "support/field_mirror.rs"]
mod field_mirror;
use field_mirror::Mirror;
use loops::{circle, polygon, rotated_square, Ring};

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    // Copies one buffer's words into the packed readback at dispatch_range.z.
    mod.sdf.TestPack = {
        workgroup_size: 64
        source: shader.buffer(u32)
        packed: shader.buffer_rw(u32)
        dispatch_range: shader.uniform(vec4u)
        main: fn() {
            let i = global_id().x
            if i >= self.dispatch_range.y { return }
            let k = self.dispatch_range.x + i
            self.packed[self.dispatch_range.z + k] = self.source[k]
        }
    }
    // Writes dispatch_range.w into count word dispatch_range.z (a set made on the GPU).
    mod.sdf.TestSegCount = {
        workgroup_size: 64
        counts: shader.buffer_rw(u32)
        dispatch_range: shader.uniform(vec4u)
        main: fn() {
            if global_id().x == 0u { self.counts[self.dispatch_range.z] = self.dispatch_range.w }
        }
    }
    mod.sdf
}

/// A FeatureSet with its per-loop bias.
struct Set { fs: FeatureSet, loop_bias: Vec<f32> }
impl std::ops::Deref for Set { type Target = FeatureSet; fn deref(&self) -> &FeatureSet { &self.fs } }
fn build_set(n: u32, planes: &[Vec<Ring>], bias: (f32, f32)) -> Set {
    let planes: Vec<Vec<Loop>> = planes.iter().map(|p| p.iter().map(|r| Loop { points: &r.points, creases: &r.creases, hole: r.hole }).collect()).collect();
    let fs = from_loops(n, n, &planes).unwrap();
    let loop_bias = fs.loop_bias(bias.0, bias.1);
    Set { fs, loop_bias }
}

struct Kernels { field: FieldKernels, pack: ComputeKernel, seg_count: ComputeKernel }
impl Kernels {
    fn new(cx: &mut Cx) -> Kernels {
        let (pack, seg_count) = cx.with_vm(|vm| {
            script_mod(vm);
            let module = vm.heap().module(id!(sdf));
            let pack = vm.value(module, id!(TestPack)).as_object().unwrap();
            let pack = ComputeKernel::new(vm, pack, "test pack").unwrap();
            let count = vm.value(module, id!(TestSegCount)).as_object().unwrap();
            (pack, ComputeKernel::new(vm, count, "test seg count").unwrap())
        });
        Kernels { field: FieldKernels::new(cx).unwrap(), pack, seg_count }
    }
}

/// The field's outputs as read back.
#[derive(PartialEq, Debug)]
struct Field {
    band: Vec<u32>,
    band_off: Vec<u32>,
    band_slot: Vec<u32>,
    cover: Vec<u32>,
    node_side: Vec<u32>,
    nf: Vec<u32>,
    cand_off: Vec<u32>,
    cand: Vec<u32>,
    words: Vec<u32>,
    probe_out: Vec<u32>,
    cand_bound: u32,
    submissions: u32,
    gpu_ms: f32,
}

/// Probe points: every 7th fine cell around the set's segs, jittered deterministically.
fn probe_points(set: &Set, params: FieldParams) -> Vec<[f32; 4]> {
    let mut out = Vec::new();
    let mut h = 0x9e3779b9u32;
    let reach = params.depth_in.max(params.depth_out) as f64 + params.bias_max as f64 + 2.0;
    for p in 0..set.planes {
        let segs: Vec<&[f32; 4]> = (0..set.seg.len()).filter(|&s| {
            let l = set.loop_info.iter().position(|li| li[0] == set.seg_info[s][0]).unwrap();
            set.loop_info[l][2] == p
        }).map(|s| &set.seg[s]).collect();
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for q in segs { for k in 0..2 { lo[k] = lo[k].min(q[k].min(q[k + 2]) as f64); hi[k] = hi[k].max(q[k].max(q[k + 2]) as f64); } }
        let w = set.nx * 10;
        let x0 = (lo[0] - reach).floor().max(0.0) as u32;
        let x1 = ((hi[0] + reach).ceil() as u32).min(w);
        let y0 = (lo[1] - reach).floor().max(0.0) as u32;
        let y1 = ((hi[1] + reach).ceil() as u32).min(set.ny * 10);
        for fy in y0..y1 {
            for fx in x0..x1 {
                if (fy * w + fx) % 7 != 0 { continue; }
                h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                let jx = (h >> 8) as f32 / (1 << 24) as f32;
                h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                let jy = (h >> 8) as f32 / (1 << 24) as f32;
                out.push([fx as f32 + jx, fy as f32 + jy, f32::from_bits(p), 0.0]);
            }
        }
    }
    out
}

/// The field's steps, then the probes, then every output packed into one readback. `counted`:
/// the seg count comes from a count word a pass writes (as for a set made on the GPU), with
/// that much spare capacity.
fn run_field(cx: &mut Cx, k: &Kernels, set: &Set, params: FieldParams) -> Field { run_field_counted(cx, k, set, params, None) }
fn run_field_counted(cx: &mut Cx, k: &Kernels, set: &Set, params: FieldParams, counted: Option<u32>) -> Field {
    let mut pool = Pool::new(0);
    let claimed = claim(&mut pool).unwrap();
    let mut input = FieldInput::upload(&mut pool, &set.fs, &set.loop_bias).unwrap();
    let mut prefix = Vec::new();
    if let Some(spare) = counted {
        let word = pool.claim("test seg count", 1).unwrap();
        let n = set.seg.len() as u32;
        prefix.push(Step::Pass(Pass::new(&k.seg_count, gpu_jobs::label("test seg count"), Elements::Fixed(1), 1, 1 << 10, |_| BufRef::COUNTS).zw(word.0, n)));
        input.segs = SegCount::Counted { word, capacity: n + spare };
    }
    let probes = probe_points(set, params);
    let layout: Rc<RefCell<Option<(FieldOut, Vec<(&'static str, u32, u32)>, BufRef)>>> = Rc::new(RefCell::new(None));
    let (seen, pack) = (layout.clone(), k.pack.clone());
    let field = k.field.clone();
    let cells = params.nx * params.ny * params.planes;
    let steps = steps(&k.field, &mut pool, claimed, input, params, move |_, pool, out| {
        let probe_in = pool.buffer("test probes", probes.len() as u64 * 16);
        if !probes.is_empty() { pool.upload(probe_in, 0, probes.iter().flatten().flat_map(|v| v.to_bits().to_le_bytes()).collect()); }
        let probe_out = pool.buffer("test probe_out", probes.len() as u64 * 8);
        let mut steps = vec![out.probe(&field, probe_in, probe_out, probes.len() as u32)];
        let parts: Vec<(&'static str, BufRef, u32)> = vec![
            ("band", out.band, out.slots), ("band_off", out.band_off, params.planes + 1), ("band_slot", out.band_slot, cells),
            ("cover", out.cover, cells), ("node_side", out.node_side, out.slots * 121), ("nf", out.nf, out.slots * 242),
            ("cand_off", out.cand_off, out.slots * 100 + 1), ("cand", out.cand, out.cand_bound), ("probe_out", probe_out, probes.len() as u32 * 2),
        ];
        let total: u32 = parts.iter().map(|p| p.2).sum();
        let packed = pool.readback("test packed", total as u64 * 4);
        let mut at = 0;
        let mut offsets = Vec::new();
        for (name, buf, n) in parts {
            steps.push(Step::Pass(Pass::new(&pack, gpu_jobs::label("test pack"), Elements::Fixed(n), 1, 1 << 20,
                |b| if b == id!(source) { buf } else { packed }).zw(at, 0)));
            offsets.push((name, at, n));
            at += n;
        }
        steps.push(Output::fixed(packed, total as u64 * 4));
        *seen.borrow_mut() = Some((*out, offsets, packed));
        Ok(steps)
    }).unwrap();
    prefix.extend(steps);
    let finished = gpu_rig::run_steps(cx, pool, prefix, JobClass::Visible);
    let (counts, output, stats) = match finished {
        Finished::Done { counts, output, stats, .. } => (counts, output, stats),
        Finished::Failed { error, .. } => panic!("feature field failed: {error}"),
    };
    let (out, offsets, _) = layout.borrow_mut().take().unwrap();
    out.verify(&counts).unwrap();
    let words = gpu_rig::words(&output.unwrap());
    let part = |name: &str| { let &(_, at, n) = offsets.iter().find(|o| o.0 == name).unwrap(); words[at as usize..(at + n) as usize].to_vec() };
    let base = out.words.0;
    let field_words = (0..WORDS).map(|w| counts.at(base + w)).collect::<Vec<_>>();
    let total = field_words[WORD_CAND_TOTAL as usize];
    let mut cand = part("cand");
    cand.truncate(total as usize);
    Field {
        band: part("band"), band_off: part("band_off"), band_slot: part("band_slot"), cover: part("cover"),
        node_side: part("node_side"), nf: part("nf"), cand_off: part("cand_off"), cand, words: field_words,
        probe_out: part("probe_out"), cand_bound: out.cand_bound, submissions: stats.submissions, gpu_ms: stats.gpu_ms,
    }
}

fn slot_plane(field: &Field, slot: u32) -> u32 { (0..field.band_off.len() as u32 - 1).rev().find(|&p| field.band_off[p as usize] <= slot).unwrap() }

struct Report { nodes: usize, cells: usize, probes: usize, cand: u32, max_err: f64 }

fn check(name: &str, set: &Set, params: FieldParams, field: &Field, probes: &[[f32; 4]]) -> Report {
    let m = Mirror::new(&set.fs, &set.loop_bias, params);
    let (nx, cells) = (set.nx, set.nx * set.ny);
    assert_eq!(field.words[WORD_ORPHAN_NODES as usize], 0, "{name}: domain nodes without a band copy");
    assert_eq!(field.words[WORD_CAND_OVERFLOW as usize], 0, "{name}: cand overflow");
    assert_eq!(field.words[WORD_BAND_SLOTS as usize] as usize, field.band.len(), "{name}: band count word");
    assert_eq!(*field.cand_off.last().unwrap(), field.words[WORD_CAND_TOTAL as usize], "{name}: cand_off ends with the total");
    // coverage at every coarse centre
    for p in 0..set.planes {
        for c in 0..cells {
            let centre = [(c % nx * 10 + 5) as f64, (c / nx * 10 + 5) as f64];
            assert_eq!(field.cover[(p * cells + c) as usize] != 0, m.inside(p, centre), "{name}: cover plane {p} cell {c}");
        }
    }
    // the band map
    for (i, &slot) in field.band_slot.iter().enumerate() {
        if slot != NONE { assert_eq!(field.band[slot as usize], i as u32 % cells, "{name}: band_slot {i}"); }
    }
    // the field is defined where |d'| is within the asked depth
    let within = |d: f64, inside: bool| if inside { -d <= params.depth_in as f64 } else { d <= params.depth_out as f64 };
    let mut max_err = 0.0f64;
    let mut nodes = 0;
    let mut copies: HashMap<(u32, u32, u32), (u32, u32, u32)> = HashMap::new();
    for slot in 0..field.band.len() as u32 {
        let p = slot_plane(field, slot);
        let c = field.band[slot as usize];
        for local in 0..121u32 {
            let (gx, gy) = (c % nx * 10 + local % 11, c / nx * 10 + local / 11);
            let i = (slot * 121 + local) as usize;
            let (feature, dist) = (field.nf[i * 2], f32::from_bits(field.nf[i * 2 + 1]));
            let entry = (field.node_side[i], feature, dist.to_bits());
            if let Some(prev) = copies.insert((p, gx, gy), entry) { assert_eq!(prev, entry, "{name}: node copies differ at plane {p} ({gx}, {gy})"); }
            let q = [gx as f64, gy as f64];
            let (d, inside) = m.field(p, q);
            assert_eq!(field.node_side[i] != 0, inside, "{name}: node side plane {p} ({gx}, {gy})");
            if !within(d, inside) { continue; }
            nodes += 1;
            let err = (dist as f64 - d).abs();
            max_err = max_err.max(err);
            assert!(err <= 5e-4, "{name}: nf plane {p} ({gx}, {gy}) gpu {dist} mirror {d}");
            assert!(feature != NONE, "{name}: no feature at plane {p} ({gx}, {gy})");
            let v = m.value(feature, q, inside);
            assert!((v - if inside { -d } else { d }).abs() <= 5e-4, "{name}: nf feature {feature:#x} at ({gx}, {gy}) gives {v}, min {d}");
        }
    }
    // cand: ascending, and an argmin of every sample point within depth
    let mut checked = 0;
    for slot in 0..field.band.len() as u32 {
        let p = slot_plane(field, slot);
        let c = field.band[slot as usize];
        for local in 0..100u32 {
            let cell = (slot * 100 + local) as usize;
            let list = &field.cand[field.cand_off[cell] as usize..field.cand_off[cell + 1] as usize];
            assert!(list.windows(2).all(|w| w[0] < w[1]), "{name}: cand not ascending in cell {cell}");
            let (fx, fy) = (c % nx * 10 + local % 10, c / nx * 10 + local / 10);
            for (sx, sy) in [(0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.0), (0.0, 0.5), (0.13, 0.71), (0.87, 0.29)] {
                let q = [fx as f64 + sx, fy as f64 + sy];
                let (d, inside) = m.field(p, q);
                if !within(d, inside) { continue; }
                checked += 1;
                let best = list.iter().map(|&f| m.value(f, q, inside)).fold(f64::INFINITY, f64::min);
                let want = if inside { -d } else { d };
                assert!((best - want).abs() <= 1e-9, "{name}: cand of cell ({fx}, {fy}) misses the argmin at {q:?}: {best} vs {want}, list {list:x?}");
            }
        }
    }
    // field_at probes, on and off the band
    let mut probed = 0;
    for (k, q) in probes.iter().enumerate() {
        let p = q[2].to_bits();
        let at = [q[0] as f64, q[1] as f64];
        let (d, inside) = m.field(p, at);
        if !within(d, inside) { continue; }
        probed += 1;
        let got = f32::from_bits(field.probe_out[k * 2 + 1]) as f64;
        max_err = max_err.max((got - d).abs());
        assert!((got - d).abs() <= 5e-4, "{name}: field_at plane {p} {at:?}: gpu {got} mirror {d}");
    }
    Report { nodes, cells: checked, probes: probed, cand: *field.cand_off.last().unwrap(), max_err }
}

fn cases() -> Vec<(&'static str, u32, Vec<Vec<Ring>>, (f32, f32))> {
    let disk = vec![circle([40.37, 39.71], 24.3, 720, false)];
    let square = vec![rotated_square([40.21, 40.43], 22.0, 17.0)];
    let l = vec![polygon(&[[12.3, 12.6], [60.7, 12.6], [60.7, 30.2], [31.4, 30.2], [31.4, 66.9], [12.3, 66.9]])];
    let wedge = vec![polygon(&[[14.2, 20.3], [66.8, 33.1], [14.9, 46.7]])];
    let notch = vec![polygon(&[[12.5, 12.5], [67.5, 12.5], [67.5, 67.5], [44.1, 67.5], [40.3, 30.2], [36.2, 67.5], [12.5, 67.5]])];
    let boxed = vec![polygon(&[[14.3, 15.2], [65.9, 15.2], [65.9, 63.7], [14.3, 63.7]]), circle([40.6, 39.4], 9.3, 6, true)];
    let far = vec![rotated_square([1951.37, 1960.21], 18.0, 23.0)];
    vec![
        ("disk", 8, vec![disk.clone()], (0.0, 0.0)),
        ("square", 8, vec![square], (0.0, 0.0)),
        ("L", 8, vec![l], (0.0, 0.0)),
        ("wedge", 8, vec![wedge], (0.0, 0.0)),
        ("notch", 8, vec![notch], (0.4, 0.0)),
        ("box-hole biased", 8, vec![boxed.clone()], (0.7, -0.4)),
        ("two planes", 8, vec![disk, boxed], (-0.3, 0.2)),
        ("far from the origin", 200, vec![far], (0.0, 0.0)),
    ]
}

fn params_of(n: u32, planes: u32, join: FieldJoin, bias: (f32, f32)) -> FieldParams {
    FieldParams { nx: n, ny: n, planes, depth_in: 15.0, depth_out: 3.0, join, bias_max: bias.0.abs().max(bias.1.abs()) }
}

#[test]
fn field_steps_match_their_mirror() {
    gpu_rig::with_gpu(|cx| {
        let k = Kernels::new(cx);
        let start = Instant::now();
        for (name, n, shapes, bias) in cases() {
            let set = build_set(n, &shapes, bias);
            for join in [FieldJoin::Miter(3.0), FieldJoin::Round] {
                let params = params_of(n, shapes.len() as u32, join, bias);
                let field = run_field(cx, &k, &set, params);
                let label = format!("{name} {join:?}");
                let r = check(&label, &set, params, &field, &probe_points(&set, params));
                eprintln!("{label}: segs {} nodes {} samples {} probes {} cand {} bound {} ({:.2}x) max err {:.2e}; {} submissions, {:.2} GPU ms",
                    set.seg.len(), r.nodes, r.cells, r.probes, r.cand, field.cand_bound, field.cand_bound as f64 / r.cand.max(1) as f64, r.max_err,
                    field.submissions, field.gpu_ms);
            }
        }
        eprintln!("field cases: {:.0} ms", start.elapsed().as_secs_f64() * 1000.0);
    });
}

#[test]
fn field_steps_are_the_same_at_every_job_size() {
    gpu_rig::with_gpu(|cx| {
        let k = Kernels::new(cx);
        let (_, n, shapes, bias) = cases().into_iter().find(|c| c.0 == "two planes").unwrap();
        let set = build_set(n, &shapes, bias);
        let params = params_of(n, 2, FieldJoin::Miter(3.0), bias);
        let mut runs = Vec::new();
        for elements in [Some(64), Some(1000), None] {
            gpu_rig::force_elements(cx, elements);
            runs.push(run_field(cx, &k, &set, params));
        }
        gpu_rig::force_elements(cx, None);
        // cells whose lists several threads fill (one feature each)
        let shared = runs[0].cand_off.windows(2).filter(|w| w[1] - w[0] >= 2).count();
        let longest = runs[0].cand_off.windows(2).map(|w| w[1] - w[0]).max().unwrap();
        assert!(shared > 0 && longest >= 4, "cells with several cand writers: {shared}, longest list {longest}");
        eprintln!("cells with several cand writers: {shared}, longest list {longest}");
        for r in &mut runs { (r.submissions, r.gpu_ms) = (0, 0.0); }
        for r in &runs[1..] { assert!(*r == runs[0], "the field differs between job sizes"); }
    });
}

#[test]
fn field_steps_take_a_counted_seg_count() {
    gpu_rig::with_gpu(|cx| {
        let k = Kernels::new(cx);
        let (_, n, shapes, bias) = cases().into_iter().find(|c| c.0 == "two planes").unwrap();
        let set = build_set(n, &shapes, bias);
        let params = params_of(n, 2, FieldJoin::Miter(3.0), bias);
        let mut fixed = run_field(cx, &k, &set, params);
        let mut counted = run_field_counted(cx, &k, &set, params, Some(37));
        for r in [&mut fixed, &mut counted] { (r.submissions, r.gpu_ms) = (0, 0.0); }
        assert!(counted == fixed, "the field of a counted seg count differs from the fixed one");
    });
}
