//! S3 (regions_overhang) on the GPU through gpu_jobs Steps. Off the lower field's band every
//! span takes the cover side, so with no band each segment is one label by its middle's coarse
//! cell: these tests check the loop table, counts, scan, CSRs and the decode to mm against an
//! f64 reference, byte-identical at three job sizes, with given and counted sizes.
//! Field crossings also use the real lower feature_field and its f64 mirror.
#[path = "support/field_mirror.rs"]
mod field_mirror;
#[path = "support/gpu_rig.rs"]
mod rig;
use field_mirror::Mirror;
use makepad_sdf_engine::{
    feature_field::{self, FieldInput, FieldJoin, FieldKernels, FieldParams, BIG, DOMAIN_SLACK},
    feature_set::{self, FeatureSet, Loop},
    gpu_jobs::{BufRef, Finished, JobClass, Output, Pool, ScanKernels},
    makepad_draw::Cx,
    plane_abi::{concentric_level, thin_level, wall_level, S3Label, WallLevel, LABEL_SUPPORTED, LOOP_OPEN, NONE, SEG_REFLEX, VERTEX_BIT},
    regions_overhang::{self as s3, S3Inputs, S3Kernels, S3Spec, S3Words, Size},
};

const NX: u32 = 4;
const NY: u32 = 4;
const PLANES: u32 = 2;
const NOZZLE: f32 = 0.4;

struct Ring { points: Vec<[f32; 2]>, plane: u32, level: u32, open: bool }

fn rings() -> Vec<Ring> {
    let wall = wall_level(1, WallLevel::Centre);
    let ring = |points: Vec<[f32; 2]>, plane, level, open| Ring { points, plane, level, open };
    let circle: Vec<[f32; 2]> = (0..600).map(|i| {
        let a = i as f64 / 600.0 * std::f64::consts::TAU + 0.3;
        [(20.0 + 12.0 * a.cos()) as f32, (20.0 + 12.0 * a.sin()) as f32]
    }).collect();
    vec![
        // inside the lower part's cover, with a repeated point (a zero-length segment)
        ring(vec![[3.0, 3.0], [17.0, 3.0], [17.0, 3.0], [17.0, 13.0], [3.0, 13.0]], 1, wall, false),
        // a thin bead from over the part to over air
        ring(vec![[12.0, 22.0], [25.0, 22.0], [25.0, 35.0]], 1, thin_level(0), true),
        // not a wall: no labels
        ring(vec![[5.0, 5.0], [8.0, 5.0], [8.0, 8.0]], 1, concentric_level(0), false),
        // the first layer: nothing below
        ring(vec![[5.0, 5.0], [8.0, 5.0], [8.0, 8.0], [5.0, 8.0]], 0, wall, false),
        // across both sides and three arc blocks
        ring(circle, 1, wall, false),
        ring(vec![[30.0, 30.0], [33.0, 31.0]], 1, wall, false),
        ring(vec![[30.0, 30.0]], 1, wall, false),
    ]
}

fn spec(points: Size, loops: Size) -> S3Spec {
    S3Spec {
        nx: NX, ny: NY, planes: PLANES, miter: 3.0, bias_max: 0.0, nozzle: NOZZLE, slowdown: true,
        widths: vec![(wall_level(1, WallLevel::Centre), 0.45), (thin_level(0), 0.3)],
        below: vec![NONE, 0], points, loops, k: s3::K_DEFAULT,
    }
}

/// The lower part covers the coarse columns x < 2 of plane 0.
fn covered(p: [f32; 2]) -> bool { (p[0].floor().clamp(0.0, (NX * 10 - 1) as f32) as u32) / 10 < 2 }

/// Per loop: (vertex, supported) of each label, one per segment longer than 1 nm.
fn reference(spec: &S3Spec) -> Vec<Vec<(u32, bool)>> {
    rings().iter().map(|r| {
        let labelled = spec.below[r.plane as usize] != NONE && spec.widths.iter().any(|w| w.0 == r.level) && r.points.len() >= 2;
        if !labelled { return vec![]; }
        let n = r.points.len();
        let segs = if r.open { n - 1 } else { n };
        (0..segs).filter_map(|i| {
            let (a, b) = (r.points[i], r.points[(i + 1) % n]);
            let len = ((b[0] - a[0]) as f64).hypot((b[1] - a[1]) as f64);
            (len >= 1.0e-5).then(|| (i as u32, covered([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5])))
        }).collect()
    }).collect()
}

/// A ring's points in nm, as the rings readback rounds them.
fn nm(r: &Ring) -> Vec<[i64; 2]> { r.points.iter().map(|p| [(p[0] as f64 * 1.0e5).round() as i64, (p[1] as f64 * 1.0e5).round() as i64]).collect() }

fn le(words: impl IntoIterator<Item = u32>) -> Vec<u8> { words.into_iter().flat_map(u32::to_le_bytes).collect() }

/// One S3 run over `rings()` with no band, its counts sizes given or counted; the s3_out words.
/// `counted`: sizes from count words, the points `short` of all (a rings run past its capacity).
fn run(cx: &mut makepad_sdf_engine::makepad_draw::Cx, kernels: &S3Kernels, scans: &ScanKernels, counted: Option<u32>) -> Vec<u32> {
    let rings = rings();
    let mut pool = Pool::new(32);
    let words = S3Words::claim(&mut pool).unwrap();
    let put = |pool: &mut Pool, name: &'static str, bytes: Vec<u8>| {
        let buf = pool.buffer(name, bytes.len().max(16) as u64);
        if !bytes.is_empty() { pool.upload(buf, 0, bytes); }
        buf
    };
    let cells = (NX * NY * PLANES) as usize;
    let cover: Vec<u32> = (0..cells).map(|i| (i < (NX * NY) as usize && (i as u32 % NX) < 2) as u32).collect();
    let (mut points, mut loops, mut meta) = (vec![], vec![], vec![]);
    for r in &rings {
        loops.extend([0, 0, r.points.len() as u32, (points.len() / 2) as u32]);
        meta.extend([r.plane, r.level, 0, if r.open { LOOP_OPEN } else { 0 }]);
        points.extend(r.points.iter().flat_map(|p| [p[0].to_bits(), p[1].to_bits()]));
    }
    let (n_points, n_loops) = ((points.len() / 2) as u32, rings.len() as u32);
    let none = put(&mut pool, "s3 test empty", le([0u32; 4]));
    let inputs = S3Inputs {
        seg: none, seg_off: none, seg_geo: none, seg_meta: none, node_side: none, cand_off: none, cand: none,
        set_slot: put(&mut pool, "s3 test set slot", le(vec![NONE; cells])),
        band_slot: put(&mut pool, "s3 test band slot", le(vec![NONE; cells])),
        cover: put(&mut pool, "s3 test cover", le(cover)),
        loops: put(&mut pool, "s3 test loops", le(loops)),
        loop_meta: put(&mut pool, "s3 test loop meta", le(meta)),
        ring_pt: put(&mut pool, "s3 test ring pt", le(points)),
    };
    let mut steps = vec![];
    let sizes = if let Some(short) = counted {
        // counts words an earlier stage writes: the totals of scans of ones, under looser bounds
        let mut count = |pool: &mut Pool, name: &'static str, n: u32| {
            let word = pool.claim(name, 1).unwrap();
            let ones = put(pool, "s3 test ones", le(vec![1u32; n as usize]));
            let sink = pool.buffer("s3 test sink", n as u64 * 4);
            steps.extend(scans.exclusive(pool, ones, sink, n, word));
            word
        };
        let (pw, lw) = (count(&mut pool, "s3 test points", n_points - short), count(&mut pool, "s3 test loops", n_loops));
        (Size::Counted { word: pw, bound: n_points + 7 }, Size::Counted { word: lw, bound: n_loops + 3 })
    } else {
        (Size::Known(n_points), Size::Known(n_loops))
    };
    let (s3_steps, out) = s3::steps(kernels, scans, &mut pool, &words, &inputs, &spec(sizes.0, sizes.1)).unwrap();
    steps.extend(s3_steps);
    steps.push(Output::fixed(out.buf, out.bytes));
    assert_ne!(out.buf, BufRef::RUNTIME);
    match rig::run_steps(cx, pool, steps, JobClass::Visible) {
        Finished::Done { counts, output, .. } => {
            assert!(words.check(&counts).unwrap(), "S3 labels past their capacity");
            rig::words(&output.expect("s3_out"))
        }
        Finished::Failed { error, .. } => panic!("S3 run (counted {counted:?}): {error}"),
    }
}

fn check(got: &[s3::LoopLabels], spec: &S3Spec) {
    let want = reference(spec);
    let rings = rings();
    assert_eq!(got.len(), want.len(), "loops");
    for (l, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(g.seeds.is_empty(), "loop {l}: seeds {:?}", g.seeds);
        let records: Vec<(u32, bool)> = g.labels.iter().map(|x| (x.vertex, x.flags & LABEL_SUPPORTED != 0)).collect();
        assert_eq!(&records, w, "loop {l}: labels {:?}", g.labels);
        for (k, x) in g.labels.iter().enumerate() {
            assert_eq!((x.t, x.degree), (0.0, if x.flags & LABEL_SUPPORTED != 0 { 0.0 } else { 5.0 }), "loop {l} label {k}: {x:?}");
        }
        // in mm: each label from its segment's start to the next label's, the last to the end
        let points = nm(&rings[l]);
        let mm = g.to_mm(s3::Line { points: &points, open: rings[l].open }).unwrap();
        if g.labels.is_empty() { assert_eq!(mm, s3::LabelsMm::default(), "loop {l}"); continue; }
        let n = points.len();
        let mut arc = vec![0.0f64];
        for i in 0..n { let (a, b) = (points[i], points[(i + 1) % n]); arc.push(arc[i] + ((b[0] - a[0]) as f64).hypot((b[1] - a[1]) as f64) / 1.0e6); }
        let length = arc[if rings[l].open { n - 1 } else { n }];
        assert_eq!(mm.length, length, "loop {l} length");
        for (k, piece) in mm.pieces.iter().enumerate() {
            let to = w.get(k + 1).map_or(length, |next| arc[next.0 as usize]);
            assert_eq!((piece.0, piece.1, piece.2), (arc[w[k].0 as usize], to, w[k].1), "loop {l} piece {k}");
        }
    }
}

#[test]
fn labels_by_cover_match_the_reference_at_three_job_sizes() {
    rig::with_gpu(|cx| {
        let kernels = S3Kernels::new(cx).unwrap();
        let scans = ScanKernels::new(cx).unwrap();
        let mut first: Option<Vec<u32>> = None;
        for forced in [Some(3), Some(64), None] {
            rig::force_elements(cx, forced);
            let out = run(cx, &kernels, &scans, None);
            check(&s3::decode(&out).unwrap(), &spec(Size::Known(0), Size::Known(0)));
            match &first {
                Some(f) => assert!(*f == out, "s3_out differs at jobs of {forced:?}"),
                None => first = Some(out),
            }
        }
        rig::force_elements(cx, Some(5));
        let counted = run(cx, &kernels, &scans, Some(0));
        // Two points short: the 2-point wall ring runs past the stored points and stays unlabelled.
        let short = run(cx, &kernels, &scans, Some(2));
        rig::force_elements(cx, None);
        let all = s3::decode(first.as_ref().unwrap()).unwrap();
        assert_eq!(s3::decode(&counted).unwrap(), all, "counted sizes");
        let mut cut = all.clone();
        assert!(!cut[5].labels.is_empty());
        cut[5] = s3::LoopLabels::default();
        assert_eq!(s3::decode(&short).unwrap(), cut, "points past the count");
    });
}

#[test]
fn field_depth_covers_both_levels() {
    let spec = spec(Size::Known(1), Size::Known(1));
    // w = 0.45 mm: lo = -0.225 + 0.0425 mm; the support level n/2 = 0.2 mm
    assert!((s3::full_support_level(0.45, NOZZLE) + 1.825).abs() < 1.0e-5);
    let (din, dout) = s3::field_depth(&spec);
    assert!((din - (1.825 + s3::DEPTH_MARGIN)).abs() < 1.0e-5 && (dout - (2.0 + s3::DEPTH_MARGIN)).abs() < 1.0e-5, "{din} {dout}");
    assert!(s3::decode(&[1, 0, 0, 0, 0, 12, 12, 0]).is_err(), "a loop past the loop bound");
    // a label past the line's segments is an error, not a guess
    let bad = s3::LoopLabels { labels: vec![S3Label { vertex: 3, t: 0.0, degree: 0.0, flags: 1 }], seeds: vec![] };
    assert!(bad.to_mm(s3::Line { points: &[[0, 0], [1_000_000, 0], [0, 1_000_000]], open: true }).is_err());
}

const FNX: u32 = 5;
const FNY: u32 = 5;
const W_WALL: f32 = 0.45;
const W_THIN: f32 = 0.3;

/// The layer below (plane 0): a part with a convex spike (miter join, not squared) and a hole.
fn lower() -> Vec<Vec<[f64; 2]>> {
    vec![
        vec![[3.0, 5.0], [30.0, 5.0], [30.0, 15.0], [40.0, 20.0], [30.0, 25.0], [3.0, 25.0]],
        vec![[10.0, 10.0], [10.0, 16.0], [16.0, 16.0], [16.0, 10.0]],
    ]
}

/// Walls on plane 1 (grid units).
fn field_rings() -> Vec<Ring> {
    let wall = wall_level(1, WallLevel::Centre);
    let thin = thin_level(0);
    let ring = |points: Vec<[f32; 2]>, level, open| Ring { points, plane: 1, level, open };
    let circle: Vec<[f32; 2]> = (0..160).map(|i| {
        let a = i as f64 / 160.0 * std::f64::consts::TAU + 0.05;
        [(20.0 + 9.5 * a.cos()) as f32, (15.0 + 9.5 * a.sin()) as f32]
    }).collect();
    vec![
        // full support, partial and air across the right edge; past the spike's miter tip
        // (supported by the miter offset, Euclidean distance past n/2: degree 5); back over the top edge
        ring(vec![[22.0, 8.0], [43.0, 8.0], [43.0, 32.0], [22.0, 32.0]], wall, false),
        // a thin bead bridging the hole
        ring(vec![[8.0, 13.0], [18.0, 13.0], [18.0, 20.0]], thin, true),
        // oblique crossings all round
        ring(circle, wall, false),
        // near the spike's tip, inside the degree window: an exact Euclidean degree
        ring(vec![[41.5, 12.0], [41.5, 28.0]], thin, true),
        // a long partial span: two degree samples per direction
        ring(vec![[4.0, 4.0], [29.0, 4.0]], wall, false),
    ]
}

fn field_spec(points: u32, loops: u32, k: u32) -> S3Spec {
    S3Spec {
        nx: FNX, ny: FNY, planes: 2, miter: 3.0, bias_max: 0.0, nozzle: NOZZLE, slowdown: true,
        widths: vec![(wall_level(1, WallLevel::Centre), W_WALL), (thin_level(0), W_THIN)],
        below: vec![NONE, 0], points: Size::Known(points), loops: Size::Known(loops), k,
    }
}

/// label_line's runs in f64 on the mirror of the lower field: per loop (vertex, t, supported,
/// degree, at a tenth boundary) and seeds (vertex, t).
struct RefLoop { labels: Vec<(u32, f64, bool, f64, bool)>, seeds: Vec<(u32, f64)> }

// Keep whether the unrounded degree is on a tenth boundary: only these samples
// may choose adjacent tenths when the GPU evaluates the distance in f32.
fn degree(d: f64, lo: f64, c1: f64) -> (f64, bool) {
    let f = (d - lo) / (c1 - lo);
    if f <= 0.0 { return (0.0, false); }
    let tenths = (1.0 + 4.0 * f).clamp(1.0, 5.0) * 10.0;
    (tenths.floor() / 10.0, (tenths - tenths.round()).abs() <= 1.0e-5)
}

fn reference_field(set: &FeatureSet, mirror: &Mirror, spec: &S3Spec) -> Vec<RefLoop> {
    let c1 = (NOZZLE * 0.5 * 10.0) as f64;
    // the Euclidean signed distance to plane 0's outline, over every seg
    let euclid = |p: [f64; 2]| {
        let mut best = f64::INFINITY;
        for (s, q) in set.seg.iter().enumerate() {
            if set.loop_info.iter().find(|li| li[0] == set.seg_info[s][0]).unwrap()[2] != 0 { continue; }
            let (a, b) = ([q[0] as f64, q[1] as f64], [q[2] as f64, q[3] as f64]);
            let e = [b[0] - a[0], b[1] - a[1]];
            let ee = e[0] * e[0] + e[1] * e[1];
            let t = if ee > 0.0 { (((p[0] - a[0]) * e[0] + (p[1] - a[1]) * e[1]) / ee).clamp(0.0, 1.0) } else { 0.0 };
            best = best.min((p[0] - a[0] - e[0] * t).hypot(p[1] - a[1] - e[1] * t));
        }
        if mirror.inside(0, p) { -best } else { best }
    };
    field_rings().iter().map(|r| {
        let lo = s3::full_support_level(spec.widths.iter().find(|w| w.0 == r.level).unwrap().1, NOZZLE) as f64;
        let n = r.points.len();
        let segs = if r.open { n - 1 } else { n };
        let mut out = RefLoop { labels: vec![], seeds: vec![] };
        for i in 0..segs {
            let (a, b) = (r.points[i], r.points[(i + 1) % n]);
            let (a, d) = ([a[0] as f64, a[1] as f64], [b[0] as f64 - a[0] as f64, b[1] as f64 - a[1] as f64]);
            let len = d[0].hypot(d[1]);
            if len < 1.0e-5 { continue; }
            let at = |t: f64| [a[0] + d[0] * t, a[1] + d[1] * t];
            let value = |t: f64| mirror.field(0, at(t)).0;
            // crossings of both levels: sign changes on a 0.01 grid unit lattice, bisected
            let steps = (len / 0.01).ceil() as usize;
            let mut cuts: Vec<(f64, u32)> = vec![];
            for (level, c) in [(0u32, c1), (1, lo)] {
                let mut prev = value(0.0) - c;
                for k in 1..=steps {
                    let t = k as f64 / steps as f64;
                    let v = value(t) - c;
                    if (prev <= 0.0) != (v <= 0.0) {
                        let (mut l, mut h) = ((k - 1) as f64 / steps as f64, t);
                        for _ in 0..60 {
                            let m = 0.5 * (l + h);
                            if (value(m) - c <= 0.0) == (prev <= 0.0) { l = m } else { h = m }
                        }
                        cuts.push((0.5 * (l + h), level));
                    }
                    prev = v;
                }
            }
            cuts.sort_by(|x, y| x.0.total_cmp(&y.0));
            let tiny = 1.0e-5 / len;
            let (mut prev, mut last_seed) = (0.0f64, -1.0f64);
            let piece = |out: &mut RefLoop, t0: f64, t1: f64| {
                let v = value(0.5 * (t0 + t1));
                let (supported, partial) = (v <= c1, v <= c1 && v > lo);
                let kn = if partial { ((len * (t1 - t0) / 20.0).ceil() as u32).max(1) } else { 1 };
                for k in 0..kn {
                    let s0 = t0 + (t1 - t0) * k as f64 / kn as f64;
                    let s1 = if k + 1 < kn { t0 + (t1 - t0) * (k + 1) as f64 / kn as f64 } else { t1 };
                    let (deg, boundary) = if !supported { (5.0, false) } else if partial || lo > 0.0 { degree(euclid(at(0.5 * (s0 + s1))), lo, c1) } else { (0.0, false) };
                    out.labels.push((i as u32, s0, supported, deg, boundary));
                }
            };
            for &(x, level) in &cuts {
                if x < tiny || 1.0 - x < tiny { continue; }
                if level == 1 && x - last_seed >= tiny { out.seeds.push((i as u32, x)); last_seed = x; }
                if x - prev < tiny { continue; }
                piece(&mut out, prev, x);
                prev = x;
            }
            piece(&mut out, prev, 1.0);
        }
        out
    }).collect()
}

/// One field + S3 run; (s3_out words, S3 fits), with the field defects checked.
fn run_field(cx: &mut Cx, fk: &FieldKernels, kernels: &S3Kernels, scans: &ScanKernels, set: &FeatureSet, spec: &S3Spec) -> (Vec<u32>, bool) {
    let rings = field_rings();
    let mut pool = Pool::new(32);
    let words = S3Words::claim(&mut pool).unwrap();
    let field_words = feature_field::claim(&mut pool).unwrap();
    let input = FieldInput::upload(&mut pool, set, &set.loop_bias(0.0, 0.0)).unwrap();
    let put = |pool: &mut Pool, name: &'static str, bytes: Vec<u8>| {
        let buf = pool.buffer(name, bytes.len().max(16) as u64);
        if !bytes.is_empty() { pool.upload(buf, 0, bytes); }
        buf
    };
    let (mut points, mut loops, mut meta) = (vec![], vec![], vec![]);
    for r in &rings {
        loops.extend([0, 0, r.points.len() as u32, (points.len() / 2) as u32]);
        meta.extend([r.plane, r.level, 0, if r.open { LOOP_OPEN } else { 0 }]);
        points.extend(r.points.iter().flat_map(|p| [p[0].to_bits(), p[1].to_bits()]));
    }
    let (loops, loop_meta, ring_pt) = (put(&mut pool, "s3 test loops", le(loops)), put(&mut pool, "s3 test loop meta", le(meta)), put(&mut pool, "s3 test ring pt", le(points)));
    let (din, dout) = s3::field_depth(spec);
    let params = FieldParams { nx: FNX, ny: FNY, planes: 2, depth_in: din, depth_out: dout, join: FieldJoin::Miter(3.0), bias_max: 0.0 };
    let (k3, sc, spec2) = (kernels.clone(), scans.clone(), spec.clone());
    let steps = feature_field::steps(fk, &mut pool, field_words, input, params, move |_counts, pool, out| {
        let inputs = S3Inputs {
            seg: out.input.seg, seg_off: out.input.seg_off, set_slot: out.input.set_slot, seg_geo: out.seg_geo, seg_meta: out.seg_meta,
            band_slot: out.band_slot, cover: out.cover, node_side: out.node_side, cand_off: out.cand_off, cand: out.cand,
            loops, loop_meta, ring_pt,
        };
        let (mut steps, o) = s3::steps(&k3, &sc, pool, &words, &inputs, &spec2)?;
        steps.push(Output::fixed(o.buf, o.bytes));
        Ok(steps)
    }).unwrap();
    match rig::run_steps(cx, pool, steps, JobClass::Visible) {
        Finished::Done { counts, output, .. } => {
            let field = counts.named("feature field");
            assert_eq!((field[1], field[2]), (0, 0), "field orphan nodes / cand overflow");
            let fits = words.check(&counts).unwrap();
            (rig::words(&output.expect("s3_out")), fits)
        }
        Finished::Failed { error, .. } => panic!("field + S3: {error}"),
    }
}

#[test]
fn labels_on_the_lower_field_match_the_f64_reference() {
    let lower = lower();
    let loops: Vec<Loop> = lower.iter().map(|p| Loop { points: p, creases: &[], hole: p.iter().zip(p.iter().cycle().skip(1)).map(|(a, b)| a[0] * b[1] - a[1] * b[0]).sum::<f64>() * 0.5 < 0.0 }).collect();
    let set = feature_set::from_loops(FNX, FNY, &[loops, vec![]]).unwrap();
    let n_points = field_rings().iter().map(|r| r.points.len() as u32).sum::<u32>();
    let n_loops = field_rings().len() as u32;
    let spec = field_spec(n_points, n_loops, s3::K_DEFAULT);
    let (din, dout) = s3::field_depth(&spec);
    let params = FieldParams { nx: FNX, ny: FNY, planes: 2, depth_in: din, depth_out: dout, join: FieldJoin::Miter(3.0), bias_max: 0.0 };
    let mirror = Mirror::new(&set, &set.loop_bias(0.0, 0.0), params);
    let want = reference_field(&set, &mirror, &spec);
    rig::with_gpu(|cx| {
        let fk = FieldKernels::new(cx).unwrap();
        let kernels = S3Kernels::new(cx).unwrap();
        let scans = ScanKernels::new(cx).unwrap();
        // K = 1 has room for one label per point: these rings need more
        let (_, fits) = run_field(cx, &fk, &kernels, &scans, &set, &field_spec(n_points, n_loops, 1));
        assert!(!fits, "K = 1 overflows");
        let mut first: Option<Vec<u32>> = None;
        for forced in [Some(3), Some(64), None] {
            rig::force_elements(cx, forced);
            let (out, fits) = run_field(cx, &fk, &kernels, &scans, &set, &spec);
            assert!(fits);
            match &first { Some(f) => assert!(*f == out, "s3_out differs at jobs of {forced:?}"), None => first = Some(out) }
        }
        rig::force_elements(cx, None);
        let got = s3::decode(first.as_ref().unwrap()).unwrap();
        let rings = field_rings();
        assert_eq!(got.len(), want.len(), "field loops");
        for (l, (g, w)) in got.iter().zip(&want).enumerate() {
            let n = rings[l].points.len();
            let segment = |v: u32| (rings[l].points[v as usize], rings[l].points[(v as usize + 1) % n]);
            let len = |v: u32| { let (a, b) = segment(v); (b[0] as f64 - a[0] as f64).hypot(b[1] as f64 - a[1] as f64) };
            assert_eq!(g.labels.len(), w.labels.len(), "loop {l}: labels {:?} vs {:?}, points {:?}", g.labels, w.labels, rings[l].points);
            for (k, (x, y)) in g.labels.iter().zip(&w.labels).enumerate() {
                assert_eq!((x.vertex, x.flags & LABEL_SUPPORTED != 0), (y.0, y.2), "loop {l} label {k}: {x:?} vs {y:?}, segment {:?}", segment(y.0));
                assert!((x.t as f64 - y.1).abs() * len(x.vertex) <= 1.0e-3, "loop {l} label {k} starts {x:?} vs {y:?}, segment {:?}", segment(y.0));
                // a degree may differ by a tenth only where the reference sits on a tenth step
                if (x.degree as f64 - y.3).abs() > 1.0e-6 {
                    assert!(y.4 && (x.degree as f64 - y.3).abs() <= 0.1 + 1.0e-6, "loop {l} label {k} degree {x:?} vs {y:?}, segment {:?}", segment(y.0));
                }
            }
            assert_eq!(g.seeds.len(), w.seeds.len(), "loop {l} seeds {:?} vs {:?}", g.seeds, w.seeds);
            for (x, y) in g.seeds.iter().zip(&w.seeds) {
                assert!(x.vertex == y.0 && (x.t as f64 - y.1).abs() * len(x.vertex) <= 1.0e-3, "loop {l} seed {x:?} vs {y:?}, segment {:?}", segment(y.0));
            }
        }
        // The long partial segment has two degree samples in each direction.
        assert_eq!(got[4].labels.len(), 4, "long partial span: {:?}", got[4].labels);
        for vertex in 0..2 {
            assert_eq!(got[4].labels.iter().filter(|x| x.vertex == vertex && x.flags & LABEL_SUPPORTED != 0 && x.degree > 0.0 && x.degree < 5.0).count(), 2, "long partial segment {vertex}");
        }
        // the spike: past the miter tip, supported at degree 5
        let spike = &got[0].labels;
        assert!(spike.iter().any(|x| x.vertex == 1 && x.flags & LABEL_SUPPORTED != 0 && x.degree == 5.0), "{spike:?}");
        // inside the window near the tip: an exact degree below 5
        assert!(got[3].labels.iter().any(|x| x.flags & LABEL_SUPPORTED != 0 && x.degree > 0.0 && x.degree < 5.0), "{:?}", got[3].labels);
    });
}
