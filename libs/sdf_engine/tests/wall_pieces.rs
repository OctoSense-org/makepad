//! wall_pieces::steps on the gpu_jobs runtime against the f64 field mirror: every piece on its
//! level and in its cell with {d' <= λ} on its left, every chain closed bitwise through its keys,
//! every crossing the mirror sees on a fine edge keyed, the outline slot the set's own loops,
//! and byte for byte at every job size.
use makepad_sdf_engine::{feature_field::{self, *}, feature_set::{from_loops, FeatureSet, Loop}, gpu_jobs::{self, BufRef, Elements, Finished, JobClass, Output, Pass, Pool, Step}};
use makepad_sdf_engine::plane_abi::{wall_level, WallLevel, CELL_BIT, LEVEL_BASE, LEVEL_OUTLINE, SEG_REFLEX, VERTEX_BIT};
use makepad_sdf_engine::wall_pieces::{self, Levels, PieceKernels, PiecesOut};
use makepad_sdf_engine::makepad_draw::{self as makepad_draw, *};
use std::{cell::RefCell, collections::{HashMap, HashSet}, rc::Rc, time::Instant};

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
    mod.sdf
}

struct Set { fs: FeatureSet, loop_bias: Vec<f32> }
fn build_set(n: u32, planes: &[Vec<Ring>], bias: (f32, f32)) -> Set {
    let planes: Vec<Vec<Loop>> = planes.iter().map(|p| p.iter().map(|r| Loop { points: &r.points, creases: &r.creases, hole: r.hole }).collect()).collect();
    let fs = from_loops(n, n, &planes).unwrap();
    let loop_bias = fs.loop_bias(bias.0, bias.1);
    Set { fs, loop_bias }
}

struct Kernels { field: FieldKernels, pieces: PieceKernels, pack: ComputeKernel }
impl Kernels {
    fn new(cx: &mut Cx) -> Kernels {
        let pack = cx.with_vm(|vm| {
            script_mod(vm);
            let module = vm.heap().module(id!(sdf));
            let object = vm.value(module, id!(TestPack)).as_object().unwrap();
            ComputeKernel::new(vm, object, "test pack").unwrap()
        });
        Kernels { field: FieldKernels::new(cx).unwrap(), pieces: PieceKernels::new(cx).unwrap(), pack }
    }
}

/// The extract's outputs as read back.
#[derive(PartialEq, Debug)]
struct Pieces {
    piece: Vec<[u32; 4]>,
    geo: Vec<[u32; 4]>,
    key: Vec<[u32; 4]>,
    info: Vec<[u32; 4]>,
    piece_off: Vec<u32>,
    cell_info: Vec<[u32; 4]>,
    words: Vec<u32>,
    submissions: u32,
    gpu_ms: f32,
    /// The extract's own kernels' GPU ms (cells, count, fill, outline; not its scan).
    pieces_ms: f32,
}

fn quads(w: &[u32]) -> Vec<[u32; 4]> { w.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect() }

/// The field's steps, the extract's from its `then`, then the extract's outputs packed into one readback.
fn run(cx: &mut Cx, k: &Kernels, set: &Set, params: FieldParams, ids: &[u32], values: &[f32]) -> Pieces {
    let mut pool = Pool::new(0);
    let field_words = feature_field::claim(&mut pool).unwrap();
    let words = wall_pieces::claim(&mut pool).unwrap();
    let input = FieldInput::upload(&mut pool, &set.fs, &set.loop_bias).unwrap();
    let layout: Rc<RefCell<Option<(PiecesOut, Vec<(&'static str, u32, u32)>)>>> = Rc::new(RefCell::new(None));
    let (seen, pack, pk) = (layout.clone(), k.pack.clone(), k.pieces.clone());
    let (ids, values) = (ids.to_vec(), values.to_vec());
    let steps = feature_field::steps(&k.field, &mut pool, field_words, input, params, move |_, pool, field| {
        wall_pieces::steps(&pk, pool, field, words, &Levels { ids: &ids, values: &values }, move |_, pool, out| {
            let n = out.capacity;
            let parts: Vec<(&'static str, BufRef, u32)> = vec![
                ("piece", out.piece, n * 4), ("geo", out.piece_geo, n * 4), ("key", out.piece_key, n * 4), ("info", out.piece_info, n * 4),
                ("piece_off", out.piece_off, out.cells * out.levels_per_cell + 1), ("cell_info", out.cell_info, out.cells * 4),
            ];
            let total: u32 = parts.iter().map(|p| p.2).sum();
            let packed = pool.readback("test packed", total.max(1) as u64 * 4);
            let (mut steps, mut at, mut offsets) = (Vec::new(), 0, Vec::new());
            for (name, buf, n) in parts {
                if n > 0 {
                    steps.push(Step::Pass(Pass::new(&pack, gpu_jobs::label("test pack"), Elements::Fixed(n), 1, 1 << 20,
                        |b| if b == id!(source) { buf } else { packed }).zw(at, 0)));
                }
                offsets.push((name, at, n));
                at += n;
            }
            steps.push(Output::fixed(packed, total.max(1) as u64 * 4));
            *seen.borrow_mut() = Some((*out, offsets));
            Ok(steps)
        })
    }).unwrap();
    let finished = gpu_rig::run_steps(cx, pool, steps, JobClass::Visible);
    let (counts, output, stats) = match finished {
        Finished::Done { counts, output, stats, .. } => (counts, output, stats),
        Finished::Failed { error, .. } => panic!("wall pieces failed: {error}"),
    };
    let (_, offsets) = layout.borrow_mut().take().unwrap();
    let w = gpu_rig::words(&output.unwrap());
    let part = |name: &str| { let &(_, at, n) = offsets.iter().find(|o| o.0 == name).unwrap(); w[at as usize..(at + n) as usize].to_vec() };
    Pieces {
        piece: quads(&part("piece")), geo: quads(&part("geo")), key: quads(&part("key")), info: quads(&part("info")),
        piece_off: part("piece_off"), cell_info: quads(&part("cell_info")),
        words: (0..wall_pieces::WORDS).map(|i| counts.at(words.0 + i)).collect(),
        submissions: stats.submissions, gpu_ms: stats.gpu_ms,
        pieces_ms: stats.kernels.iter().filter(|k| ["pieces cells", "pieces count", "pieces fill", "pieces outline"].iter().any(|&n| gpu_jobs::label(n) == k.0)).map(|k| k.2).sum(),
    }
}

fn f(v: u32) -> f64 { f32::from_bits(v) as f64 }

/// A piece's point at fraction u of its way (lines straight, arcs by their sweep).
fn piece_at(p: &[u32; 4], g: &[u32; 4], u: f64) -> [f64; 2] {
    let (a, b) = ([f(p[0]), f(p[1])], [f(p[2]), f(p[3])]);
    if f(g[2]) == 0.0 { return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u]; }
    let (c, r, sweep) = ([f(g[0]), f(g[1])], f(g[2]), f(g[3]));
    let alpha = (a[1] - c[1]).atan2(a[0] - c[0]) + sweep * u;
    [c[0] + r * alpha.cos(), c[1] + r * alpha.sin()]
}

struct Report { pieces: usize, junctions: usize, crossings: usize, arcs: usize, max_err: f64 }

fn check(name: &str, set: &Set, params: FieldParams, ids: &[u32], values: &[f32], got: &Pieces) -> Report {
    let m = Mirror::new(&set.fs, &set.loop_bias, params);
    if got.words[1..].iter().any(|&w| w != 0) { eprintln!("{name}: defect words {:?} (step cap, snap, convex end, no part, fill)", &got.words[1..]); }
    let slots = ids.len();
    let nxn = set.fs.nx * 10 + 1;
    assert_eq!(*got.piece_off.last().unwrap(), got.words[0], "{name}: piece_off ends with the total");
    assert_eq!(got.words[0] as usize, got.piece.len(), "{name}: pieces word");
    let within = |d: f64, inside: bool| if inside { -d <= params.depth_in as f64 } else { d <= params.depth_out as f64 };
    let (mut max_err, mut junctions, mut arcs) = (0.0f64, 0, 0);
    let mut bad: Vec<String> = Vec::new();
    // per (plane, slot): from key → piece, and every edge crossing key
    let mut by_from: HashMap<(u32, usize, [u32; 2]), usize> = HashMap::new();
    let mut edge_keys: HashMap<(u32, usize, u32), HashSet<u32>> = HashMap::new();
    for (cell, ci) in got.cell_info.iter().enumerate() {
        let (fx, fy, plane) = (ci[0] & 0xffff, ci[0] >> 16, ci[1] & 0x7fff_ffff);
        for l in 0..slots {
            let lam = values[plane as usize * slots + l];
            let at = cell * slots + l;
            let range = got.piece_off[at] as usize..got.piece_off[at + 1] as usize;
            if lam.is_nan() { assert!(range.is_empty(), "{name}: pieces at an unused slot"); continue; }
            for i in range {
                let (p, g, key, info) = (&got.piece[i], &got.geo[i], &got.key[i], &got.info[i]);
                assert_eq!(*info, [plane, ids[l], info[2], fy * nxn + fx], "{name}: piece {i} info");
                let tol = |q: [f64; 2]| 2e-3 + 8.0 * f32::EPSILON as f64 * q[0].abs().max(q[1].abs());
                for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let q = piece_at(p, g, u);
                    if !(q[0] >= fx as f64 - 1e-4 && q[0] <= fx as f64 + 1.0 + 1e-4 && q[1] >= fy as f64 - 1e-4 && q[1] <= fy as f64 + 1.0 + 1e-4) {
                        bad.push(format!("piece {i} ({info:?}, key {key:x?}, {:?}, geo {:?}) leaves its cell ({fx}, {fy}) at {q:?}", p.map(|v| f32::from_bits(v)), g.map(|v| f32::from_bits(v))));
                        break;
                    }
                    if lam == 0.0 { continue; }
                    let (d, _) = m.field(plane, q);
                    max_err = max_err.max((d - lam as f64).abs());
                    if (d - lam as f64).abs() > tol(q) { bad.push(format!("piece {i} ({info:?}, key {key:x?}, {:?}, geo {:?}) at {u}: d' {d} off its level {lam}", p.map(|v| f32::from_bits(v)), g.map(|v| f32::from_bits(v)))); }
                }
                if lam == 0.0 {
                    let s = info[2] as usize;
                    assert_eq!(*p, set.fs.seg[s].map(f32::to_bits), "{name}: outline piece {i} is its seg");
                } else {
                    // {d' <= λ} on the left, a step off the piece (a smaller step where the
                    // region beside it is a sliver narrower than the first)
                    let (q0, q1) = (piece_at(p, g, 0.45), piece_at(p, g, 0.55));
                    let (t, len) = ([q1[0] - q0[0], q1[1] - q0[1]], ((q1[0] - q0[0]).powi(2) + (q1[1] - q0[1]).powi(2)).sqrt());
                    if len > 0.005 {
                        let mid = piece_at(p, g, 0.5);
                        let n = [-t[1] / len, t[0] / len];
                        let side = |h: f64| (m.field(plane, [mid[0] + h * n[0], mid[1] + h * n[1]]), m.field(plane, [mid[0] - h * n[0], mid[1] - h * n[1]]));
                        let ((dl, il), (dr, ir)) = side(0.01);
                        let fine = side(0.002);
                        let left_below = |((dl, _), (dr, _)): ((f64, bool), (f64, bool))| dl < lam as f64 && dr > lam as f64;
                        if within(dl, il) && within(dr, ir) {
                            if !left_below(((dl, il), (dr, ir))) && !left_below(fine) {
                                let s = (info[2] & 0x1fff_ffff) as usize;
                                bad.push(format!("piece {i} ({info:?}, key {key:x?}, {:?}, geo {:?}, seg flags {:#x}) has d' {dl} on its left, {dr} on its right of {lam}",
                                    p.map(|v| f32::from_bits(v)), g.map(|v| f32::from_bits(v)), set.fs.seg_info[s][3]));
                            }
                        }
                    }
                }
                if f(g[2]) != 0.0 { arcs += 1; }
                if key[1] >> 24 != ids[l] || key[3] >> 24 != ids[l] {
                    bad.push(format!("piece {i} ({info:?}, key {key:x?}, {:?}, geo {:?}) in cell {cell} slot {l} of [{}, {}): key level", p.map(|v| f32::from_bits(v)),
                        g.map(|v| f32::from_bits(v)), got.piece_off[at], got.piece_off[at + 1]));
                    continue;
                }
                for (hi, lo) in [(key[0], key[1]), (key[2], key[3])] {
                    if hi & CELL_BIT != 0 {
                        assert_eq!(hi & !CELL_BIT, fy * nxn + fx, "{name}: piece {i} junction key in another cell");
                    } else {
                        edge_keys.entry((plane, l, hi)).or_default().insert(lo);
                    }
                }
                if key[2] & CELL_BIT != 0 { junctions += 1; }
                assert!(by_from.insert((plane, l, [key[0], key[1]]), i).is_none(), "{name}: from key {:x?} twice", [key[0], key[1]]);
            }
        }
    }
    for b in bad.iter().take(16) { eprintln!("{name}: BAD {b}"); }
    assert!(bad.is_empty(), "{name}: {} pieces off their level or side (words {:?})", bad.len(), got.words);
    // chains: every to key is one piece's from key, continuing bitwise; the links permute the pieces
    let mut taken = vec![false; got.piece.len()];
    for ((plane, l, _), &i) in &by_from {
        let key = got.key[i];
        let Some(&j) = by_from.get(&(*plane, *l, [key[2], key[3]])) else { panic!("{name}: piece {i} ({:?}, {:?}) leads to key {:x?} nobody starts from", got.info[i], got.piece[i].map(f32::from_bits), [key[2], key[3]]) };
        assert!(!taken[j], "{name}: piece {j} follows two pieces");
        taken[j] = true;
        assert_eq!([got.piece[i][2], got.piece[i][3]], [got.piece[j][0], got.piece[j][1]], "{name}: piece {i} ends where {j} does not start");
    }
    // every fine edge of the band whose ends the mirror puts on either side of a level is crossed
    let crossings = edge_keys.values().map(|s| s.len()).sum();
    let on_loop = |plane: u32, q: [f64; 2]| (0..set.fs.loop_info.len()).filter(|&l| set.fs.loop_info[l][2] == plane).any(|l| {
        let (root, len) = (set.fs.loop_info[l][0] as usize, set.fs.loop_info[l][1]);
        let mut s = root;
        (0..len).any(|_| {
            let g = set.fs.seg[s].map(|v| v as f64);
            s = set.fs.seg_info[s][1] as usize;
            let (d, e) = ([q[0] - g[0], q[1] - g[1]], [g[2] - g[0], g[3] - g[1]]);
            let t = ((d[0] * e[0] + d[1] * e[1]) / (e[0] * e[0] + e[1] * e[1])).clamp(0.0, 1.0);
            (d[0] - t * e[0]).abs() < 1e-9 && (d[1] - t * e[1]).abs() < 1e-9
        })
    });
    for ci in &got.cell_info {
        let (fx, fy, plane) = (ci[0] & 0xffff, ci[0] >> 16, ci[1] & 0x7fff_ffff);
        for l in 0..slots {
            let lam = values[plane as usize * slots + l] as f64;
            if lam.is_nan() { continue; }
            for (gx, gy, axis) in [(fx, fy, 0u32), (fx, fy, 1)] {
                let a = [gx as f64, gy as f64];
                let b = if axis == 0 { [a[0] + 1.0, a[1]] } else { [a[0], a[1] + 1.0] };
                let ((da, ia), (db, ib)) = (m.field(plane, a), m.field(plane, b));
                let edge = 2 * (gy * nxn + gx) + axis;
                let n = edge_keys.get(&(plane, l, edge)).map_or(0, |s| s.len());
                if lam == 0.0 {
                    // the outline slot is the set's loops whatever the bias: its parity is the
                    // sides', where neither end lies on a loop (the set keys a loop through a node
                    // on the node's +x edge, and a loop along an edge crosses none)
                    if (n % 2 == 1) != (ia != ib) && !on_loop(plane, a) && !on_loop(plane, b) {
                        panic!("{name}: plane {plane} outline edge ({gx}, {gy}, {axis}): {n} crossings, sides {ia} and {ib}");
                    }
                    continue;
                }
                if !within(da, ia) || !within(db, ib) || (da - lam).abs() < 1e-3 || (db - lam).abs() < 1e-3 { continue; }
                assert_eq!(n % 2 == 1, (da <= lam) != (db <= lam), "{name}: plane {plane} level {lam} edge ({gx}, {gy}, {axis}): {n} crossings, d' {da} and {db}");
            }
        }
    }
    // the extract's own invariants last, so a failing run still shows the geometry checks above
    for (w, v) in got.words.iter().enumerate().skip(1) { assert_eq!(*v, 0, "{name}: wall pieces word {w} (1 step cap, 2 snap, 3 convex end, 4 no part, 5 fill)"); }
    Report { pieces: got.piece.len(), junctions, crossings, arcs, max_err }
}

/// The defect words of each level slot alone, for a failing case.
fn defects_by_slot(cx: &mut Cx, k: &Kernels, set: &Set, params: FieldParams, ids: &[u32], values: &[f32]) -> Vec<(usize, Vec<u32>)> {
    (0..ids.len()).map(|l| {
        let only: Vec<f32> = values.iter().enumerate().map(|(i, &v)| if i % ids.len() == l { v } else { f32::NAN }).collect();
        (l, run(cx, k, set, params, ids, &only).words)
    }).collect()
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

/// The outline, three walls' centres and edges at 0.45 mm in grid units, and an outside level;
/// the second plane leaves one slot unused and moves the rest.
fn levels(planes: u32) -> (Vec<u32>, Vec<f32>) {
    let ids = vec![LEVEL_OUTLINE, wall_level(1, WallLevel::Centre), wall_level(1, WallLevel::Edge), wall_level(2, WallLevel::Centre),
        wall_level(3, WallLevel::Edge), LEVEL_BASE];
    let mut values = vec![0.0, -2.25, -4.5, -6.75, -13.5, 1.7];
    if planes > 1 { values.extend([0.0, -2.1, f32::NAN, -5.3, -11.9, 2.6]); }
    (ids, values)
}

#[test]
fn wall_pieces_match_the_field_mirror() {
    gpu_rig::with_gpu(|cx| {
        let k = Kernels::new(cx);
        let start = Instant::now();
        for (name, n, shapes, bias) in cases() {
            let set = build_set(n, &shapes, bias);
            let (ids, values) = levels(shapes.len() as u32);
            for join in [FieldJoin::Miter(3.0), FieldJoin::Round] {
                let params = params_of(n, shapes.len() as u32, join, bias);
                let got = run(cx, &k, &set, params, &ids, &values);
                let label = format!("{name} {join:?}");
                if got.words[1..].iter().any(|&w| w != 0) {
                    eprintln!("{label}: defects {:?}; by slot {:?}", &got.words[1..], defects_by_slot(cx, &k, &set, params, &ids, &values));
                }
                let r = check(&label, &set, params, &ids, &values, &got);
                eprintln!("{label}: segs {} pieces {} arcs {} junctions {} crossings {} max err {:.2e}; {} submissions, {:.2} GPU ms, wall_pieces {:.2} ms",
                    set.fs.seg.len(), r.pieces, r.arcs, r.junctions, r.crossings, r.max_err, got.submissions, got.gpu_ms, got.pieces_ms);
            }
        }
        eprintln!("wall pieces cases: {:.0} ms", start.elapsed().as_secs_f64() * 1000.0);
    });
}

#[test]
fn wall_pieces_are_the_same_at_every_job_size() {
    gpu_rig::with_gpu(|cx| {
        let k = Kernels::new(cx);
        let (_, n, shapes, bias) = cases().into_iter().find(|c| c.0 == "two planes").unwrap();
        let set = build_set(n, &shapes, bias);
        let params = params_of(n, 2, FieldJoin::Miter(3.0), bias);
        let (ids, values) = levels(2);
        let mut runs = Vec::new();
        for elements in [Some(64), Some(1000), None] {
            gpu_rig::force_elements(cx, elements);
            runs.push(run(cx, &k, &set, params, &ids, &values));
        }
        gpu_rig::force_elements(cx, None);
        assert!(runs[0].piece.len() > 1000, "pieces: {}", runs[0].piece.len());
        for r in &mut runs { (r.submissions, r.gpu_ms, r.pieces_ms) = (0, 0.0, 0.0); }
        for (i, r) in runs.iter().enumerate().skip(1) {
            if *r != runs[0] {
                let diff = |what: &str, a: &[[u32; 4]], b: &[[u32; 4]]| match a.iter().zip(b).position(|(x, y)| x != y) {
                    Some(j) => format!("{what} {j}: {:x?} vs {:x?}", a[j], b[j]),
                    None if a.len() != b.len() => format!("{what}: {} vs {}", a.len(), b.len()),
                    None => String::new(),
                };
                let off = runs[0].piece_off.iter().zip(&r.piece_off).position(|(x, y)| x != y);
                panic!("run {i}: the pieces differ between job sizes: words {:?} vs {:?}; {}; {}; {}; {}; piece_off at {off:?}", runs[0].words, r.words,
                    diff("piece", &runs[0].piece, &r.piece), diff("geo", &runs[0].geo, &r.geo), diff("key", &runs[0].key, &r.key), diff("info", &runs[0].info, &r.info));
            }
        }
    });
}

/// Levels through grid nodes: a rect with whole-unit corners and a triangle whose diagonal runs
/// through nodes, with the outline and walls at whole grid units. Every level piece's endpoint
/// key is its own cell's junction or one of its cell's four edges, never a node (ordinal
/// 0xFFFFFF), so chain's search of a key's two cells finds every link; the outline's node
/// vertices keep the node key (compose's convention), which chain searches four cells for.
#[test]
fn levels_through_nodes_key_edges_of_their_own_cells() {
    gpu_rig::with_gpu(|cx| {
        let k = Kernels::new(cx);
        let shapes = vec![vec![polygon(&[[12.0, 12.0], [60.0, 12.0], [60.0, 40.0], [12.0, 40.0]])], vec![polygon(&[[12.0, 12.0], [60.0, 12.0], [12.0, 60.0]])]];
        let set = build_set(8, &shapes, (0.0, 0.0));
        let ids = vec![LEVEL_OUTLINE, wall_level(1, WallLevel::Centre), wall_level(1, WallLevel::Edge)];
        let values = vec![0.0, -2.0, -4.0, 0.0, -2.0, -4.0];
        let params = params_of(8, 2, FieldJoin::Miter(3.0), (0.0, 0.0));
        let got = run(cx, &k, &set, params, &ids, &values);
        let r = check("through nodes", &set, params, &ids, &values, &got);
        let nxn = set.fs.nx * 10 + 1;
        for (i, (key, info)) in got.key.iter().zip(&got.info).enumerate() {
            let (fx, fy) = (info[3] % nxn, info[3] / nxn);
            let own = [2 * (fy * nxn + fx), 2 * ((fy + 1) * nxn + fx), 2 * (fy * nxn + fx) + 1, 2 * (fy * nxn + fx + 1) + 1];
            for (hi, lo) in [(key[0], key[1]), (key[2], key[3])] {
                if info[1] == LEVEL_OUTLINE {
                    // the outline is the set's segs: a loop through a node can link diagonal
                    // cells, so its node vertices keep the node key chain searches four cells for
                    if hi & CELL_BIT == 0 && !own.contains(&hi) { assert_eq!(lo & 0xff_ffff, 0xff_ffff, "piece {i} ({info:?}) in cell ({fx}, {fy}) keyed by edge {hi:#x}, not one of its own nor a node"); }
                    continue;
                }
                assert_ne!(lo & 0xff_ffff, 0xff_ffff, "piece {i} ({info:?}): a node key");
                if hi & CELL_BIT == 0 { assert!(own.contains(&hi), "piece {i} ({info:?}) in cell ({fx}, {fy}) keyed by edge {hi:#x}, not one of its own"); }
            }
        }
        eprintln!("through nodes: {} pieces, {} crossings", r.pieces, r.crossings);
    });
}

/// The extract's kernels as SPIR-V pass the Vulkan validator (on Linux, where it is installed).
#[test]
fn wall_pieces_kernels_pass_spirv_val() {
    use makepad_draw::makepad_script::shader_ir_compute::{compile_compute_kernel, ComputeTarget, KernelCode};
    gpu_rig::with_gpu(|cx| {
        cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            for name in ["PiecesCells", "PiecesCount", "PiecesFill", "PiecesOutline"] {
                let object = vm.value(module, LiveId::from_str(name)).as_object().unwrap();
                let k = compile_compute_kernel(vm, object, ComputeTarget::Spirv).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert!(k.layout.push_size <= 128, "{name}: push block {} bytes", k.layout.push_size);
                let KernelCode::Spirv(words) = &k.code else { panic!("{name}: not SPIR-V") };
                #[cfg(target_os = "linux")]
                {
                    let file = std::env::temp_dir().join(format!("wall-pieces-{name}-{}.spv", std::process::id()));
                    std::fs::write(&file, words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
                    let out = std::process::Command::new("spirv-val").args(["--target-env", "vulkan1.2"]).arg(&file).output().expect("spirv-val required");
                    std::fs::remove_file(file).unwrap();
                    eprintln!("{name}: {} words, {} buffers, spirv-val {}", words.len(), k.layout.buffers.len(), if out.status.success() { "ok".into() } else {
                        format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)) });
                }
                #[cfg(not(target_os = "linux"))]
                assert!(!words.is_empty());
            }
        });
    });
}

/// Miter tips of 90° reflex vertices, and two reflex vertices' collinear miters (the regions
/// stage's real path): a holed band, an L, and a dumbbell whose neck miters overlap.
#[test]
fn miter_tips_and_collinear_miters_close() {
    let s = |p: &[[f64; 2]]| polygon(&p.iter().map(|&[x, y]| [x + 0.37, y + 0.21]).collect::<Vec<_>>());
    let band = vec![s(&[[30., 2.], [61., 2.], [61., 33.], [30., 33.]]), s(&[[41., 13.], [41., 22.], [50., 22.], [50., 13.]])];
    let l = vec![s(&[[64., 36.], [95., 36.], [95., 49.], [77., 49.], [77., 78.], [64., 78.]])];
    let dumbbell = vec![s(&[[65., 2.], [80., 2.], [80., 7.], [85., 7.], [85., 2.], [100., 2.], [100., 17.], [85., 17.], [85., 12.], [80., 12.],
        [80., 17.], [65., 17.]])];
    gpu_rig::with_gpu(|cx| {
        let k = Kernels::new(cx);
        for (name, shape, levels) in [("holed band", band, [-5.03f32, -6.0]), ("L", l, [-5.03, -6.0]), ("dumbbell", dumbbell, [-4.0, -5.03])] {
            let set = build_set(12, &[shape], (0.0, 0.0));
            let params = FieldParams { nx: 12, ny: 12, planes: 1, depth_in: 14.53, depth_out: 1.5, join: FieldJoin::Miter(3.0), bias_max: 0.0 };
            let ids = [168, 169];
            let got = run(cx, &k, &set, params, &ids, &levels);
            let r = check(name, &set, params, &ids, &levels, &got);
            eprintln!("{name} at {levels:?}: {} pieces, {} junctions, {} crossings, max err {:.2e}", r.pieces, r.junctions, r.crossings, r.max_err);
        }
    });
}
