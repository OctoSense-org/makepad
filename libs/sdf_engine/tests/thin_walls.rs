//! The thin stage on a vase wall: one thin ring and its copy, rotated and moved,
//! in one FeatureSet. Per level the pieces chain by their keys into closed loops;
//! the copies have as many loops at every level, and each copy's beads lie on the
//! other's once the motion is undone, within the skeleton's chord tolerance. A
//! second, calibrated run gives the same bytes.
#[path = "support/gpu_rig.rs"]
mod gpu_rig;
#[path = "support/loops.rs"]
mod loops;
#[path = "support/pack.rs"]
mod pack;
use makepad_sdf_engine::{feature_set::{from_loops, FeatureSet, Loop}, gpu_jobs::{Counts, Finished, ItemStats, JobClass, Pool},
    makepad_draw::{ComputeKernel, Cx},
    makepad_csg_sdf::plan::ArachneSpec, thin::{self, ThinKernels, ThinOut, ThinSpec, CURVE_ERROR, WORD_PIECES, WORD_SPANS}};
use std::{cell::RefCell, collections::{BTreeMap, HashMap}, rc::Rc};

const NX: u32 = 23;
const NY: u32 = 13;
/// The ring's centre, and its copy's, in grid units (0.1 mm).
const A: [f64; 2] = [60.0, 60.0];
const B: [f64; 2] = [163.37, 67.21];
const TURN_DEG: f64 = 37.0;

/// A point of the ring at A to its copy at B.
fn moved(p: [f64; 2]) -> [f64; 2] {
    let (s, c) = TURN_DEG.to_radians().sin_cos();
    let (x, y) = (p[0] - A[0], p[1] - A[1]);
    [B[0] + c * x - s * y, B[1] + s * x + c * y]
}

/// A point of the copy back to the ring.
fn unmoved(p: [f64; 2]) -> [f64; 2] {
    let (s, c) = TURN_DEG.to_radians().sin_cos();
    let (x, y) = (p[0] - B[0], p[1] - B[1]);
    [A[0] + c * x + s * y, A[1] - s * x + c * y]
}

/// A bead piece: its points in order (an arc piece sampled), its level and its two keys.
struct Piece { points: Vec<[f64; 2]>, level: u32, from: (u32, u32), to: (u32, u32) }

fn pieces(n: usize, piece: &[u32], geo: &[u32], key: &[u32], info: &[u32]) -> Vec<Piece> {
    let f = |w: &[u32], i: usize| f32::from_bits(w[i]) as f64;
    (0..n).map(|i| {
        let (p0, p1) = ([f(piece, i * 4), f(piece, i * 4 + 1)], [f(piece, i * 4 + 2), f(piece, i * 4 + 3)]);
        let mut points = vec![p0];
        // A round piece turns clockwise about (x, y) by -w, its radius following the ends.
        if f(geo, i * 4 + 2) > 0.0 {
            let v = [f(geo, i * 4), f(geo, i * 4 + 1)];
            let sweep = -f(geo, i * 4 + 3);
            let (a0, r0, r1) = ((p0[1] - v[1]).atan2(p0[0] - v[0]), (p0[0] - v[0]).hypot(p0[1] - v[1]), (p1[0] - v[0]).hypot(p1[1] - v[1]));
            for k in 1..32 {
                let t = k as f64 / 32.0;
                let (a, r) = (a0 - sweep * t, r0 + (r1 - r0) * t);
                points.push([v[0] + r * a.cos(), v[1] + r * a.sin()]);
            }
        }
        points.push(p1);
        Piece { points, level: info[i * 4 + 1], from: (key[i * 4], key[i * 4 + 1]), to: (key[i * 4 + 2], key[i * 4 + 3]) }
    }).collect()
}

/// Per level, the pieces chained by their keys into closed loops of points.
fn chain(pieces: &[Piece]) -> BTreeMap<u32, Vec<Vec<[f64; 2]>>> {
    let mut from: HashMap<(u32, (u32, u32)), usize> = HashMap::new();
    for (i, p) in pieces.iter().enumerate() {
        assert!(from.insert((p.level, p.from), i).is_none(), "two pieces leave key {:?} at level {}", p.from, p.level);
    }
    let mut seen = vec![false; pieces.len()];
    let mut loops: BTreeMap<u32, Vec<Vec<[f64; 2]>>> = BTreeMap::new();
    for start in 0..pieces.len() {
        if seen[start] { continue; }
        let mut ring = Vec::new();
        let mut at = start;
        loop {
            seen[at] = true;
            let p = &pieces[at];
            ring.extend_from_slice(&p.points[..p.points.len() - 1]);
            let next = *from.get(&(p.level, p.to)).unwrap_or_else(|| panic!("no piece leaves key {:?} at level {}", p.to, p.level));
            let (end, begin) = (p.points[p.points.len() - 1], pieces[next].points[0]);
            assert!((end[0] - begin[0]).hypot(end[1] - begin[1]) <= 1e-3, "pieces {at} and {next} meet {end:?} vs {begin:?}");
            if next == start { break; }
            assert!(!seen[next], "piece {next} joins two chains at level {}", p.level);
            at = next;
        }
        loops.entry(pieces[start].level).or_default().push(ring);
    }
    loops
}

/// The farthest point of `a` from the closed polyline `b`.
fn farthest(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    a.iter().map(|p| (0..b.len()).map(|i| {
        let (s, e) = (b[i], b[(i + 1) % b.len()]);
        let (ux, uy) = (e[0] - s[0], e[1] - s[1]);
        let len2 = ux * ux + uy * uy;
        let t = if len2 > 0.0 { (((p[0] - s[0]) * ux + (p[1] - s[1]) * uy) / len2).clamp(0.0, 1.0) } else { 0.0 };
        (p[0] - s[0] - ux * t).hypot(p[1] - s[1] - uy * t)
    }).fold(f64::INFINITY, f64::min)).fold(0.0, f64::max)
}

/// The stage over `set`, its outputs packed into one readback.
fn run(cx: &mut Cx, kernels: &ThinKernels, copy: &ComputeKernel, set: &FeatureSet, spec: ThinSpec) -> (Counts, ThinOut, pack::Packed, Vec<u8>, ItemStats) {
    let mut pool = Pool::new(0);
    let layout: Rc<RefCell<Option<(ThinOut, pack::Packed)>>> = Rc::new(RefCell::new(None));
    let (keep, copy) = (layout.clone(), copy.clone());
    let steps = thin::steps(kernels, &mut pool, set, spec, move |_, pool, out| {
        let (steps, packed) = pack::steps(&copy, pool, &[("piece", out.piece), ("piece_geo", out.piece_geo),
            ("piece_key", out.piece_key), ("piece_info", out.piece_info)]);
        *keep.borrow_mut() = Some((*out, packed));
        Ok(steps)
    }).unwrap();
    let (counts, output, stats) = match gpu_rig::run_steps(cx, pool, steps, JobClass::Visible) {
        Finished::Done { counts, output, stats, .. } => (counts, output.unwrap(), stats),
        Finished::Failed { error, .. } => panic!("thin walls failed: {error}"),
    };
    let (out, packed) = layout.borrow_mut().take().unwrap();
    out.verify(&counts).unwrap();
    (counts, out, packed, output.to_vec(), stats)
}

#[test]
fn thin_ring_moved_and_turned() {
    // A 0.9 mm vase wall: two beads of a two-loop wall set (cap 4).
    let outer = loops::circle(A, 40.0, 48, false);
    let inner = loops::circle(A, 31.0, 48, true);
    let copy = |r: &loops::Ring| loops::Ring { points: r.points.iter().map(|&p| moved(p)).collect(), ..r.clone() };
    let rings = [outer.clone(), inner.clone(), copy(&outer), copy(&inner)];
    let plane: Vec<Loop> = rings.iter().map(|r| Loop { points: &r.points, creases: &r.creases, hole: r.hole }).collect();
    let set = from_loops(NX, NY, &[plane]).unwrap();
    let spec = ThinSpec::new(&ArachneSpec { loops: 2, w_outer: 0.42, w_inner: 0.45, s_outer: 0.4, s_inner: 0.45, min_bead_width: 0.34,
        min_feature_size: 0.1, transition_length: 0.4, transition_filter_deviation: 0.025, transition_angle: 0.17,
        distribution_count: 1, precise: false }).unwrap().unwrap();
    gpu_rig::with_gpu(|cx| {
        let kernels = ThinKernels::new(cx).unwrap();
        let copy_kernel = pack::kernel(cx);
        // The second run is calibrated: its submissions are the batch's, and its bytes the first's.
        let (counts, out, packed, output, _) = run(cx, &kernels, &copy_kernel, &set, spec);
        let (_, _, _, again, stats) = run(cx, &kernels, &copy_kernel, &set, spec);
        assert!(output == again, "a second run differs");
        let words = gpu_rig::words(&output);
        let n = counts.word(out.word(WORD_PIECES)) as usize;
        let all = pieces(n, packed.get(&words, "piece"), packed.get(&words, "piece_geo"), packed.get(&words, "piece_key"), packed.get(&words, "piece_info"));
        let loops = chain(&all);
        let split = (A[0] + B[0]) * 0.5;
        let centre_x = |ring: &Vec<[f64; 2]>| ring.iter().map(|p| p[0]).sum::<f64>() / ring.len() as f64;
        // Both copies' beads, chained and closed, within two chord tolerances of each other.
        let tolerance = 2.0 * CURVE_ERROR as f64 + 1e-4;
        let mut worst = 0.0f64;
        for (&level, rings) in &loops {
            let (here, there): (Vec<_>, Vec<_>) = rings.iter().partition(|r| centre_x(r) < split);
            assert_eq!(here.len(), there.len(), "level {level}: {} loops vs {} on the copy", here.len(), there.len());
            let back: Vec<Vec<[f64; 2]>> = there.iter().map(|r| r.iter().map(|&p| unmoved(p)).collect()).collect();
            let mut taken = vec![false; back.len()];
            for ring in &here {
                let (best, d) = back.iter().enumerate().map(|(j, b)| (j, farthest(ring, b).max(farthest(b, ring))))
                    .min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
                assert!(!taken[best] && d <= tolerance, "level {level}: a loop of {} points is {d} from the copy's", ring.len());
                taken[best] = true;
                worst = worst.max(d);
            }
        }
        assert!(loops.values().any(|r| !r.is_empty()), "no bead loops");
        let per_level: Vec<(u32, usize)> = loops.iter().map(|(&l, r)| (l, r.len())).collect();
        println!("thin ring: {} spans, {n} pieces, loops per level {per_level:?}, worst {worst:.6} grid; calibrated: {} submissions, {:.2} ms GPU",
            counts.word(out.word(WORD_SPANS)), stats.submissions, stats.gpu_ms);
    });
}
