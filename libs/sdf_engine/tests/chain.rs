//! Chaining on the GPU against an f64 mirror: synthetic pieces of nested contours and holes,
//! rounded corners, whole circles and an open thin chain, clipped to fine cells.
#[path = "support/gpu_rig.rs"]
mod rig;
#[path = "support/chain_pieces.rs"]
mod chain_pieces;
#[path = "support/pack.rs"]
mod pack;
use chain_pieces::{circle, poly, rounded_rect, scene, Feature, LoopSpec, Scene};
use makepad_sdf_engine::chain::{self, ChainGrid, ChainKernels, Pieces, HEAD};
use makepad_sdf_engine::gpu_jobs::{BufRef, Finished, JobClass, Pool};
use makepad_sdf_engine::makepad_draw::{ComputeKernel, Cx};
use makepad_sdf_engine::plane_abi::{thin_level, wall_level, WallLevel, LEVEL_BASE, LEVEL_OUTLINE, LOOP_HOLE, LOOP_OPEN, NONE};

const W1: fn() -> u32 = || wall_level(1, WallLevel::Centre);

fn spec(plane: u32, level: u32, features: Vec<Feature>, hole: bool, parent: Option<usize>, island: Option<usize>) -> LoopSpec {
    LoopSpec { plane, level, features, closed: true, hole, parent, island, base: island }
}

/// Plane 0: a rounded contour with a round hole holding a square island, and a small square
/// hole; their first walls. Plane 1: a whole circle, its wall, and an open thin line.
fn nested() -> Vec<LoopSpec> {
    let w = W1();
    let mut s = vec![
        spec(0, LEVEL_OUTLINE, rounded_rect(5.37, 4.63, 94.21, 95.17, 6.3, true), false, None, Some(0)),
        spec(0, LEVEL_OUTLINE, circle([50.13, 50.29], 33.7, 0.3, false), true, Some(0), Some(0)),
        spec(0, LEVEL_OUTLINE, rounded_rect(41.27, 42.11, 58.71, 57.83, 0.0, true), false, Some(1), Some(2)),
        spec(0, LEVEL_OUTLINE, rounded_rect(10.33, 10.41, 20.62, 20.27, 0.0, false), true, Some(0), Some(0)),
        spec(0, w, rounded_rect(5.82, 5.08, 93.76, 94.72, 5.85, true), false, None, Some(0)),
        spec(0, w, circle([50.13, 50.29], 34.15, 1.1, false), true, Some(4), Some(0)),
        spec(0, w, rounded_rect(41.72, 42.56, 58.26, 57.38, 0.0, true), false, Some(5), Some(2)),
        spec(0, w, rounded_rect(9.88, 9.96, 21.07, 20.72, 0.45, false), true, Some(4), Some(0)),
        spec(1, LEVEL_OUTLINE, circle([30.41, 28.77], 17.3, 2.0, true), false, None, Some(8)),
        spec(1, w, circle([30.41, 28.77], 16.85, 4.4, true), false, None, Some(8)),
    ];
    let thin = thin_level(0);
    s.push(LoopSpec { plane: 1, level: thin, closed: false, hole: false, parent: None, island: Some(8), base: Some(8), features: vec![
        Feature::Line { a: [22.13, 20.37], b: [36.71, 23.29] },
        Feature::Arc { c: [36.71, 27.29], r: 4.0, a0: -std::f64::consts::FRAC_PI_2, sweep: 2.1 },
        Feature::Line { a: [36.71 + 4.0 * (2.1f64 - std::f64::consts::FRAC_PI_2).cos(), 27.29 + 4.0 * (2.1f64 - std::f64::consts::FRAC_PI_2).sin()], b: [27.53, 37.19] },
    ] });
    // the arc's ends meet its lines
    if let Feature::Arc { c, r, a0, sweep } = s[10].features[1] {
        let end = [c[0] + r * (a0 + sweep).cos(), c[1] + r * (a0 + sweep).sin()];
        s[10].features[2] = Feature::Line { a: end, b: [27.53, 37.19] };
    }
    s
}

/// Many whole circles of two pieces each (each crosses one grid line): more loops than K = 1 holds.
fn crowded() -> Vec<LoopSpec> {
    let mut s = Vec::new();
    for j in 0..6 {
        for i in 0..6 {
            s.push(spec(0, LEVEL_OUTLINE, circle([2.13 + i as f64, 2.57 + j as f64], 0.3, 0.7, true), false, None, Some(s.len())));
        }
    }
    s
}

/// An outline, a base contour with a hole, and two first-wall loops resolving against the base:
/// one shares a stretch of the contour (its rightmost vertex inside a contour piece), the other
/// a stretch of the hole (its rightmost vertex a vertex of the hole's pieces).
fn coincident() -> Vec<LoopSpec> {
    let w = W1();
    let at = |level, features, hole, parent, base| LoopSpec { plane: 0, level, features, closed: true, hole, parent, island: Some(0), base };
    vec![
        at(LEVEL_OUTLINE, rounded_rect(3.3, 3.2, 96.7, 96.8, 0.0, true), false, None, None),
        at(LEVEL_BASE, rounded_rect(10.3, 10.2, 89.7, 89.8, 0.0, true), false, None, Some(1)),
        at(LEVEL_BASE, rounded_rect(40.3, 40.2, 59.7, 59.8, 0.0, false), true, Some(1), Some(1)),
        at(w, rounded_rect(80.4, 20.55, 89.7, 30.35, 0.0, true), false, None, Some(1)),
        at(w, poly(&[[30.45, 46.37], [40.3, 45.0], [40.3, 54.85], [30.45, 54.85]]), false, None, Some(1)),
    ]
}

/// The chain's records, packed into one readback, and its count words.
struct Run { words: Vec<u32>, packed: pack::Packed, counts: Vec<u32>, loops: u32, nl: usize }
impl Run {
    fn get(&self, name: &str) -> &[u32] { self.packed.get(&self.words, name) }
    fn rec(&self, name: &str, i: usize, n: usize) -> &[u32] { &self.get(name)[i * n..i * n + n] }
    fn word(&self, w: u32) -> u32 { self.counts[w as usize] }
}

/// The records `check` reads.
const RECORDS: [&str; 10] = ["succ", "root", "rank", "piece_loop", "piece_pos", "chain_order", "loops", "loop_meta", "loop_metric", "loop_tree"];

struct Kernels { chain: ChainKernels, copy: ComputeKernel }

/// Every level resolving against the outline (walls without a base).
fn on_outline() -> Vec<(u32, u32)> { [LEVEL_OUTLINE, W1(), thin_level(0)].map(|l| (l, LEVEL_OUTLINE)).to_vec() }

/// The chain over `scene` at re-run factor k, with each level's reference level, read back.
fn run_with(cx: &mut Cx, kernels: &Kernels, scene: &Scene, k: u32, references: &[(u32, u32)]) -> Run {
    let np = scene.pieces.len() as u32;
    let mut pool = Pool::new(32);
    let upload = |pool: &mut Pool, name: &'static str, words: Vec<u32>| {
        let buf = pool.buffer(name, words.len() as u64 * 4);
        pool.upload(buf, 0, rig::bytes(&words));
        buf
    };
    let count = pool.claim("pieces", 1).unwrap();
    // the producer writes the pieces' count
    pool.upload(BufRef::COUNTS, count.0 as u64 * 4, rig::bytes(&[np]));
    let pieces = Pieces {
        piece: upload(&mut pool, "piece", scene.piece_words()),
        piece_geo: upload(&mut pool, "piece geo", scene.geo_words()),
        piece_key: upload(&mut pool, "piece key", scene.key_words()),
        piece_info: upload(&mut pool, "piece info", scene.info_words()),
        piece_off: upload(&mut pool, "piece off", scene.piece_off.clone()),
        band_slot: upload(&mut pool, "band slot", scene.band_slot.clone()),
        count, capacity: np, levels_per_cell: scene.levels.len() as u32,
    };
    let grid = ChainGrid::new(scene.nx, scene.ny, scene.planes, LEVEL_OUTLINE, references);
    let chain_words = chain::claim(&mut pool).unwrap();
    let (c, mut steps) = chain::steps(&kernels.chain, &mut pool, chain_words, &pieces, &grid, k).unwrap();
    let bufs = [c.succ, c.ranked.root, c.ranked.rank, c.piece_loop, c.piece_pos, c.chain_order, c.loops, c.loop_meta, c.loop_metric, c.loop_tree];
    let named: Vec<(&'static str, BufRef)> = RECORDS.iter().copied().zip(bufs).chain([("loop_acc", c.loop_acc)]).collect();
    let (copy, packed) = pack::steps(&kernels.copy, &mut pool, &named);
    steps.extend(copy);
    let (counts, output) = match rig::run_steps(cx, pool, steps, JobClass::Visible) {
        Finished::Done { counts, output, .. } => (counts, output.expect("the chain's packed output")),
        Finished::Failed { error, .. } => panic!("chain: {error}"),
    };
    let words = (0..chain::WORDS).map(|w| counts.word(c.word(w))).collect();
    Run { words: rig::words(&output), packed, counts: words, loops: counts.word(c.loops_word()), nl: c.loop_capacity as usize }
}

fn run(cx: &mut Cx, kernels: &Kernels, scene: &Scene, k: u32) -> Run { run_with(cx, kernels, scene, k, &on_outline()) }

// --- the f64 mirror of the polylines ---

fn diamond(d: [f64; 2]) -> f64 {
    let s = d[0].abs() + d[1].abs();
    if s == 0.0 { 0.0 } else if d[0] > 0.0 && d[1] >= 0.0 { d[1] / s } else if d[0] <= 0.0 && d[1] > 0.0 { 1.0 - d[0] / s }
    else if d[0] < 0.0 && d[1] <= 0.0 { 2.0 - d[1] / s } else { 3.0 + d[0] / s }
}

/// A piece's polyline (its start, the lattice points inside its sweep, its end), and with
/// `from_start` the lattice indices from its start when that is on the lattice: (indices, m).
fn polyline(p: &chain_pieces::GenPiece, from_start: bool) -> (Vec<[f64; 2]>, Vec<f64>, f64) {
    let q = p.p.map(|v| v as f64);
    let (a, b) = ([q[0], q[1]], [q[2], q[3]]);
    let g = p.geo.map(|v| v as f64);
    if g[2] <= 0.0 { return (vec![a, b], vec![], 1.0) }
    let m = (g[2] / (2.0 * chain::CHORD_TOL as f64)).sqrt().ceil().max(1.0);
    let n4 = 4.0 * m;
    let u = |d: [f64; 2]| { let v = diamond(d) * m; if v >= n4 { v - n4 } else { v } };
    let (ua, ub) = (u([a[0] - g[0], a[1] - g[1]]), u([b[0] - g[0], b[1] - g[1]]));
    let expect = g[3] * 2.0 * m / std::f64::consts::PI;
    let d0 = ub - ua;
    let mut turns = 0.0;
    if (d0 + n4 - expect).abs() < (d0 - expect).abs() { turns = 1.0 }
    if (d0 - n4 - expect).abs() < (d0 + turns * n4 - expect).abs() { turns = -1.0 }
    let travel = d0 + turns * n4;
    let idx: Vec<f64> = if travel > 0.0 {
        let first = if from_start { ua.ceil() } else { ua.floor() + 1.0 };
        let last = ub.ceil() - 1.0 + turns * n4;
        (0..((last - first + 1.0).max(0.0) as i64)).map(|i| first + i as f64).collect()
    } else if travel < 0.0 {
        let first = if from_start { ua.floor() } else { ua.ceil() - 1.0 };
        let last = ub.floor() + 1.0 + turns * n4;
        (0..((first - last + 1.0).max(0.0) as i64)).map(|i| first - i as f64).collect()
    } else { vec![] };
    let mut pts = vec![a];
    for &i in idx.iter().filter(|&&i| !from_start || i != ua) {
        let im = i - n4 * (i / n4).floor();
        let qd = (im / m).floor();
        let f = (im - qd * m) / m;
        let d = match qd as i64 { 0 => [1.0 - f, f], 1 => [-f, 1.0 - f], 2 => [f - 1.0, -f], _ => [f, f - 1.0] };
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
        pts.push([g[0] + d[0] / l * g[2], g[1] + d[1] / l * g[2]]);
    }
    pts.push(b);
    (pts, idx, m)
}

struct Mirror { area: f64, rightmost: [f64; 2], bbox: [f64; 4], start: usize, before: u32, heads: Vec<bool> }

/// One spec's loop, mirrored: its signed area, rightmost vertex, bbox, start (index in its
/// chain), points before lattice zero, and which pieces head runs.
fn mirror(scene: &Scene, s: usize) -> Mirror {
    let spec = &scene.specs[s];
    let chain = &scene.chains[s];
    let n = chain.len();
    let piece = |i: usize| &scene.pieces[chain[i] as usize];
    let merges = chain::merges(spec.level);
    let heads: Vec<bool> = (0..n).map(|i| {
        if !spec.closed && i == 0 { return true }
        !merges || piece(i).info[2] != piece((i + n - 1) % n).info[2]
    }).collect();
    let mut verts = Vec::new();
    for i in 0..n {
        let (pts, _, _) = polyline(piece(i), false);
        let take = if !spec.closed && i + 1 == n { pts.len() } else { pts.len() - 1 };
        verts.extend_from_slice(&pts[..take]);
    }
    let mut area = 0.0;
    if spec.closed {
        let o = verts[0];
        for i in 0..verts.len() {
            let (v, w) = (verts[i], verts[(i + 1) % verts.len()]);
            area += (v[0] - o[0]) * (w[1] - o[1]) - (v[1] - o[1]) * (w[0] - o[0]);
        }
        area *= 0.5;
    }
    let xmax = verts.iter().map(|v| v[0]).fold(f64::MIN, f64::max);
    let rightmost = [xmax, verts.iter().filter(|v| v[0] == xmax).map(|v| v[1]).fold(f64::MAX, f64::min)];
    let bbox = [verts.iter().map(|v| v[0]).fold(f64::MAX, f64::min), verts.iter().map(|v| v[1]).fold(f64::MAX, f64::min), xmax,
        verts.iter().map(|v| v[1]).fold(f64::MIN, f64::max)];
    // the start: the rightmost head start (then the lowest, then the least rank), or lattice zero
    let mut start = 0;
    let mut before = 0;
    if spec.closed {
        let any_head = heads.iter().any(|&h| h);
        if any_head {
            let key = |i: usize| { let p = piece(i).p; (p[0], -p[1]) };
            let best = (0..n).filter(|&i| heads[i]).map(key).fold((f32::MIN, f32::MIN), |a, b| if b.0 > a.0 || (b.0 == a.0 && b.1 > a.1) { b } else { a });
            let r0 = chain.iter().position(|&k| k == *chain.iter().min().unwrap()).unwrap();
            start = (0..n).filter(|&i| heads[i] && key(i) == best).min_by_key(|&i| (i + n - r0) % n).unwrap();
        } else {
            for i in 0..n {
                let (_, idx, m) = polyline(piece(i), true);
                if let Some(j) = idx.iter().position(|&x| (x / (4.0 * m)).fract() == 0.0) { start = i; before = j as u32; }
            }
        }
    }
    let heads = if spec.closed && !heads.iter().any(|&h| h) { vec![false; n] } else { heads };
    Mirror { area, rightmost, bbox, start, before, heads }
}

fn close(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() <= tol * (1.0 + b.abs()) }

/// Every record of the chain against the mirror.
fn check(scene: &Scene, r: &Run) {
    let ids = scene.loop_ids();
    let loop_of: Vec<u32> = (0..scene.specs.len()).map(|s| ids.iter().position(|&x| x == s).unwrap() as u32).collect();
    let open = scene.specs.iter().filter(|s| !s.closed).count() as u32;
    assert_eq!(r.loops, scene.specs.len() as u32, "loops");
    assert_eq!([r.word(chain::WORD_OVERFLOW), r.word(chain::WORD_OPEN_TAILS), r.word(chain::WORD_OPEN_HEADS), r.word(chain::WORD_DUPLICATES)],
        [0, open, open, 0], "overflow, open tails, open heads, duplicates");
    assert_eq!(r.word(chain::WORD_POSITIONS), scene.pieces.len() as u32, "positions");
    assert_eq!(r.word(chain::WORD_DEFECTS), 0, "defects");
    let succ = scene.succ();
    for k in 0..scene.pieces.len() {
        assert_eq!(r.get("succ")[k], succ[k], "succ of piece {k}");
    }
    let mut first = 0u32;
    for (lid, &s) in ids.iter().enumerate() {
        let spec = &scene.specs[s];
        let chain = &scene.chains[s];
        let n = chain.len();
        let m = mirror(scene, s);
        let what = format!("loop {lid} (spec {s})");
        let rec = r.rec("loops", lid, 4);
        assert_eq!(rec[1], n as u32, "{what} pieces");
        let meta = r.rec("loop_meta", lid, 4);
        let flags = if spec.hole { LOOP_HOLE } else { 0 } | if spec.closed { 0 } else { LOOP_OPEN };
        let id = |x: Option<usize>| x.map_or(NONE, |x| loop_of[x]);
        assert_eq!(meta, [spec.plane, spec.level, id(spec.island), flags], "{what} meta (plane, level, island, flags)");
        let tree = r.rec("loop_tree", lid, 4);
        assert_eq!(&tree[..3], [id(spec.parent), id(spec.island), id(spec.base)], "{what} tree (parent, island, base)");
        let metric: Vec<f64> = r.rec("loop_metric", lid, 8).iter().map(|&w| f32::from_bits(w) as f64).collect();
        assert!(close(metric[0], m.area, 1e-5), "{what} area {} != {}", metric[0], m.area);
        assert!(close(metric[1], m.rightmost[0], 1e-6) && close(metric[2], m.rightmost[1], 1e-6), "{what} rightmost {:?} != {:?}", &metric[1..3], m.rightmost);
        for i in 0..4 { assert!(close(metric[4 + i], m.bbox[i], 1e-6), "{what} bbox {:?} != {:?}", &metric[4..8], m.bbox); }
        assert_eq!(r.rec("loop_acc", lid, chain::ACC as usize)[8], m.before, "{what} points before lattice zero");
        // the pieces in order from the start
        for i in 0..n {
            let k = chain[(m.start + i) % n] as usize;
            assert_eq!(r.get("piece_loop")[k], lid as u32, "{what} piece {k} loop");
            assert_eq!(r.get("chain_order")[first as usize + i], k as u32, "{what} position {i}");
            let head = m.heads[(m.start + i) % n];
            assert_eq!(r.get("piece_pos")[k], (first + i as u32) | if head { HEAD } else { 0 }, "{what} piece {k} position and head");
        }
        first += n as u32;
    }
}

fn with_chain(test: impl FnOnce(&mut Cx, &Kernels)) {
    rig::with_gpu(|cx| {
        let kernels = Kernels { chain: ChainKernels::new(cx).unwrap(), copy: pack::kernel(cx) };
        test(cx, &kernels);
        rig::force_elements(cx, None);
    });
}

#[test]
fn chains_nested_loops_holes_circles_and_an_open_line() {
    with_chain(|cx, kernels| {
        let scene = scene(10, 10, 2, nested(), false);
        // the round hole spans more than one area block
        assert!(scene.chains[1].len() > chain::AREA_BLOCK as usize);
        let r = run(cx, kernels, &scene, 2);
        check(&scene, &r);
    });
}

#[test]
fn chain_bytes_match_at_three_job_sizes() {
    with_chain(|cx, kernels| {
        let scene = scene(10, 10, 2, nested(), false);
        let mut first: Option<Vec<u32>> = None;
        for forced in [Some(1 << 20), Some(64), Some(1000)] {
            rig::force_elements(cx, forced);
            let r = run(cx, kernels, &scene, 2);
            match &first {
                None => first = Some(r.words),
                Some(f) => assert!(*f == r.words, "jobs of {forced:?} change the chain's bytes"),
            }
        }
    });
}

#[test]
fn chain_k1_matches_k2_and_counts_overflow() {
    with_chain(|cx, kernels| {
        let sc = scene(10, 10, 2, nested(), false);
        let (a, b) = (run(cx, kernels, &sc, 1), run(cx, kernels, &sc, 2));
        check(&sc, &a);
        let np = sc.pieces.len();
        let loops = a.loops as usize;
        for (name, n) in [("succ", np), ("root", np), ("rank", np), ("piece_loop", np), ("piece_pos", np), ("chain_order", np),
            ("loops", 4 * loops), ("loop_meta", 4 * loops), ("loop_metric", 8 * loops), ("loop_tree", 4 * loops)] {
            assert_eq!(a.get(name)[..n], b.get(name)[..n], "K = 1 and K = 2 differ in {name}");
        }
        // two pieces a loop: K = 1 holds an eighth of the pieces as loops, K = 4 half
        let crowded = scene(2, 2, 1, crowded(), false);
        assert!(crowded.chains.iter().all(|c| c.len() == 2));
        let over = run(cx, kernels, &crowded, 1);
        assert_eq!(over.loops, 36, "loops counted past the capacity");
        assert_eq!(over.word(chain::WORD_OVERFLOW), 36 - over.nl as u32, "overflowed loops");
        let fits = run(cx, kernels, &crowded, 4);
        assert_eq!(fits.word(chain::WORD_OVERFLOW), 0);
        check(&crowded, &fits);
    });
}

#[test]
fn the_same_loops_split_another_way_chain_alike() {
    with_chain(|cx, kernels| {
        let (a, b) = (scene(10, 10, 2, nested(), false), scene(10, 10, 2, nested(), true));
        let (ra, rb) = (run(cx, kernels, &a, 2), run(cx, kernels, &b, 2));
        check(&a, &ra);
        check(&b, &rb);
        // the same start points and holes, by spec
        let (ia, ib) = (a.loop_ids(), b.loop_ids());
        for s in 0..a.specs.len() {
            let (ma, mb) = (mirror(&a, s), mirror(&b, s));
            let (pa, pb) = (a.pieces[a.chains[s][ma.start] as usize].p, b.pieces[b.chains[s][mb.start] as usize].p);
            if a.specs[s].closed && ma.heads.iter().any(|&h| h) { assert_eq!(pa[..2], pb[..2], "spec {s} starts elsewhere"); }
            let (la, lb) = (ia.iter().position(|&x| x == s).unwrap(), ib.iter().position(|&x| x == s).unwrap());
            assert_eq!(ra.rec("loop_meta", la, 4)[3], rb.rec("loop_meta", lb, 4)[3], "spec {s} flags");
        }
    });
}

#[test]
fn loops_sharing_their_reference_boundary_resolve_as_closed_material() {
    with_chain(|cx, kernels| {
        let sc = scene(10, 10, 1, coincident(), false);
        let r = run_with(cx, kernels, &sc, 2, &[(LEVEL_BASE, LEVEL_BASE), (W1(), LEVEL_BASE)]);
        check(&sc, &r);
    });
}
