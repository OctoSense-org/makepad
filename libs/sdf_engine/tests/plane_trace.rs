//! Plane tracing stages on the GPU job runtime against the CPU twin (bit for bit) and an f64
//! oracle (0.01 mm): coarse and node samples, kept cells, crossings, and the outline
//! FeatureSet's loops. Plain and resumed sampling, and three job sizes, give the same bytes.
use makepad_sdf_engine::{kernel, plane, plane_abi::{self, CELL_BIT, NONE}, twin};
use makepad_sdf_engine::makepad_csg_sdf::{self as sdf, SdfNode, SdfOp};
use makepad_csg_math::Vec3d;

#[path = "support/gpu_rig.rs"]
mod gpu_rig;
#[path = "support/pack.rs"]
mod pack;
#[path = "support/plane_driver.rs"]
mod plane_driver;
use plane_driver::{sample_points, trace, Part, Traced};

type Ring = Vec<[f64; 2]>;
const IDENTITY: [[f64; 4]; 3] = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]];
const ZERO: f32 = -f32::MIN_POSITIVE;

fn transform(of: SdfNode, at: [f64; 3], rotate: [f64; 3]) -> SdfNode { SdfNode::Transform { at, rotate, scale: 1., of: Box::new(of) } }
fn combine(op: SdfOp, of: Vec<SdfNode>) -> SdfNode { SdfNode::Combine { op, blend: 0., of } }
fn rect(x: f64, y: f64) -> Ring { vec![[-x, -y], [x, -y], [x, y], [-x, y]] }
// The intersection of the polygon's supporting half planes (boxes 200 mm deep).
fn polygon(r: &Ring) -> SdfNode {
    let planes = r.iter().zip(r.iter().cycle().skip(1)).take(r.len()).map(|(a, b)| {
        let d = [b[0] - a[0], b[1] - a[1]];
        let len = d[0].hypot(d[1]);
        let n = [d[1] / len, -d[0] / len];
        transform(SdfNode::Box { size: [400., 400., 4.], radius: 0. },
            [(a[0] + b[0]) * 0.5 - n[0] * 200., (a[1] + b[1]) * 0.5 - n[1] * 200., 0.], [0., 0., n[1].atan2(n[0]).to_degrees()])
    }).collect();
    combine(SdfOp::Intersection, planes)
}
struct Case { name: &'static str, node: SdfNode, rings: Option<Vec<Ring>>, bounds: ([f64; 2], [f64; 2]), loops: usize, creases: bool }

fn cases() -> Vec<Case> {
    let a = 17f64.to_radians();
    let boxed: Ring = rect(2., 1.).into_iter().map(|p| [p[0] * a.cos() - p[1] * a.sin(), p[0] * a.sin() + p[1] * a.cos()]).collect();
    let rhombus: Ring = { let t = 30f64.to_radians(); vec![[0., 0.], [2., 0.], [2. + 2. * t.cos(), 2. * t.sin()], [2. * t.cos(), 2. * t.sin()]] };
    let mut hole = rect(0.7, 0.7); hole.reverse();
    vec![
        Case { name: "box 17deg", node: transform(SdfNode::Box { size: [4., 2., 4.], radius: 0. }, [0.; 3], [0., 0., 17.]), rings: Some(vec![boxed]),
            bounds: ([-2.5, -2.], [2.5, 2.]), loops: 1, creases: true },
        Case { name: "rhombus 30deg", node: polygon(&rhombus), rings: Some(vec![rhombus.clone()]), bounds: ([0., 0.], [3.8, 1.1]), loops: 1, creases: true },
        Case { name: "square hole", node: combine(SdfOp::Difference, vec![SdfNode::Box { size: [4., 4., 4.], radius: 0. }, SdfNode::Box { size: [1.4, 1.4, 6.], radius: 0. }]),
            rings: Some(vec![rect(2., 2.), hole]), bounds: ([-2., -2.], [2., 2.]), loops: 2, creases: true },
        Case { name: "disk", node: transform(SdfNode::Cylinder { radius: 1.3, top_radius: 1.3, height: 4. }, [0.; 3], [90., 0., 0.]), rings: None,
            bounds: ([-1.3, -1.3], [1.3, 1.3]), loops: 1, creases: false },
        Case { name: "sphere blend", node: SdfNode::Combine { op: SdfOp::Union, blend: 0.6, of: vec![SdfNode::Sphere { radius: 1.3 }, transform(SdfNode::Sphere { radius: 1.3 }, [1., 0., 0.], [0.; 3])] },
            rings: None, bounds: ([-1.4, -1.4], [2.4, 1.4]), loops: 1, creases: false },
    ]
}

fn segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0. { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0., 1.) } else { 0. };
    (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy)
}
fn ring_distance(p: [f64; 2], rings: &[Ring]) -> f64 {
    rings.iter().flat_map(|r| r.iter().zip(r.iter().cycle().skip(1)).take(r.len())).map(|(a, b)| segment_distance(p, *a, *b)).fold(f64::INFINITY, f64::min)
}

/// The field's in-plane distance to its zero, to first order: |d| over the in-plane gradient.
fn in_plane(oracle: &impl Fn([f64; 2], f64) -> f64, q: [f64; 2], z: f64) -> f64 {
    let h = 1e-6;
    let gx = (oracle([q[0] + h, q[1]], z) - oracle([q[0] - h, q[1]], z)) / (2. * h);
    let gy = (oracle([q[0], q[1] + h], z) - oracle([q[0], q[1] - h], z)) / (2. * h);
    oracle(q, z).abs() / gx.hypot(gy).max(1e-3)
}

/// Every check of one plane; returns the worst crossing and outline errors in mm.
fn check(t: &Traced, c: &Case, node: &SdfNode, slots: &[kernel::KernelSlot], texels: &[[f32; 4]], fails: &mut Vec<String>) -> (f64, f64) {
    let g = t.grid;
    let (nx, ny) = (g.nx as usize, g.ny as usize);
    let planes = t.zs.len();
    let eval = |ps: &[[f32; 3]]| -> Vec<f32> { twin::eval_many(texels, ps, slots).into_iter().map(|(v, _)| if v == 0.0 { ZERO } else { v }).collect() };
    // Coarse samples and kept cells.
    let mut coarse = Vec::new();
    for p in 0..planes { for cy in 0..=ny { for cx in 0..=nx { coarse.push(t.point((cx * 10) as f32, (cy * 10) as f32, p)); } } }
    let want = eval(&coarse);
    let bad: Vec<usize> = (0..want.len()).filter(|i| want[*i].to_bits() != t.coarse_d.get(*i).map_or(u32::MAX, |v| v.to_bits())).collect();
    if let Some(&i) = bad.first() {
        fails.push(format!("{}: {} coarse samples differ from the twin, first {i} at {:?}: {:?} for {}", c.name, bad.len(), coarse[i], t.coarse_d.get(i), want[i]));
    }
    let mut kept = Vec::new();
    for p in 0..planes { for cy in 0..ny { for cx in 0..nx {
        let at = p * (nx + 1) * (ny + 1) + cy * (nx + 1) + cx;
        let v = [t.coarse_d[at], t.coarse_d[at + 1], t.coarse_d[at + nx + 1], t.coarse_d[at + nx + 2]];
        let lo = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let near = v.iter().map(|x| x.abs()).fold(f32::INFINITY, f32::min);
        if (lo <= 0.0 && hi > 0.0) || near <= t.frame[3][0] * 7.08 { kept.push((p, cy * nx + cx)); }
    } } }
    let got: Vec<(usize, usize)> = (0..planes).flat_map(|p| t.kept[t.kept_off[p] as usize..t.kept_off[p + 1] as usize].iter().map(move |c| (p, *c as usize))).collect();
    if got != kept { fails.push(format!("{}: kept cells {} differ from {} expected", c.name, got.len(), kept.len())); return (f64::INFINITY, f64::INFINITY); }
    for (slot, (p, cell)) in kept.iter().enumerate() {
        if t.coarse_slot[p * nx * ny + cell] != slot as u32 { fails.push(format!("{}: coarse_slot of plane {p} cell {cell}", c.name)); }
    }
    // Every slot's 121 nodes, bit for bit (shared nodes included).
    let nodes: Vec<[f32; 3]> = kept.iter().flat_map(|(p, cell)| { let (cx, cy) = (cell % nx, cell / nx);
        (0..121).map(move |n| t.point((cx * 10 + n % 11) as f32, (cy * 10 + n / 11) as f32, *p)) }).collect::<Vec<_>>();
    let want = eval(&nodes);
    let bad: Vec<usize> = (0..want.len()).filter(|i| want[*i].to_bits() != t.node_d.get(*i).map_or(u32::MAX, |v| v.to_bits())).collect();
    if let Some(&i) = bad.first() {
        fails.push(format!("{}: {} of {} node values differ from the twin, first {i} at {:?}: {:?} for {}", c.name, bad.len(), want.len(), nodes[i], t.node_d.get(i), want[i]));
    }
    // Crossings against the f64 roots along their edges: each owned sign-changing edge has its
    // crossing at its slot's base plus its local number, keyed by the edge.
    let oracle = |q: [f64; 2], z: f64| sdf::distance(node, Vec3d::new(q[0], q[1], z));
    let mut worst_cross = 0f64;
    let mut worst_at = String::new();
    let mut expected = 0usize;
    let mut per_plane = vec![0u32; planes + 1];
    for (slot, (p, cell)) in kept.iter().enumerate() {
        let (cx, cy) = (cell % nx, cell / nx);
        let mut local = 0u32;
        for le in 0..220usize {
            let (i, j, axis) = if le < 110 { (le % 10, le / 10, 0) } else { ((le - 110) % 11, (le - 110) / 11, 1) };
            let a = j * 11 + i;
            let b = if axis == 0 { a + 1 } else { a + 11 };
            let sign = (t.node_d[slot * 121 + a] <= 0.0) != (t.node_d[slot * 121 + b] <= 0.0);
            let owner = !((le < 10 && cy > 0 && t.coarse_slot[p * nx * ny + cell - nx] != NONE) || (le >= 110 && i == 0 && cx > 0 && t.coarse_slot[p * nx * ny + cell - 1] != NONE));
            let byte = (t.edge_local[slot * 55 + le / 4] >> ((le % 4) * 8)) & 255;
            if !(owner && sign) {
                if byte != 0 { fails.push(format!("{}: slot {slot} edge {le} numbered {byte} without an owned sign change", c.name)); }
                continue
            }
            local += 1;
            if byte != local { fails.push(format!("{}: slot {slot} edge {le} numbered {byte}, {local} expected", c.name)); continue }
            expected += 1;
            per_plane[p + 1] += 1;
            let x = (t.slot_cross_base[slot] + local - 1) as usize;
            let (gx, gy) = ((cx * 10 + i) as u32, (cy * 10 + j) as u32);
            if t.cross_key.get(x) != Some(&plane_abi::edge_id(gx, gy, axis as u32, g.nx)) { fails.push(format!("{}: crossing {x} key", c.name)); continue }
            let z = t.zs[*p] as f64;
            let q = t.cross[x];
            let on_edge = if axis == 0 { q[1] == gy as f32 && q[0] >= gx as f32 && q[0] <= gx as f32 + 1. } else { q[0] == gx as f32 && q[1] >= gy as f32 && q[1] <= gy as f32 + 1. };
            if !on_edge { fails.push(format!("{}: crossing {x} at {q:?} is off its edge ({gx}, {gy}) axis {axis}", c.name)); }
            // The distance to the f64 surface: an edge along the boundary or grazing it has many
            // roots, and any of them is the edge's crossing.
            let got = t.mm([q[0], q[1]]);
            let e = in_plane(&oracle, got, z);
            if e > worst_cross {
                worst_cross = e;
                worst_at = format!("slot {slot} edge {le} ({gx}, {gy}) axis {axis}: crossing {x} at {:?}, nodes {} {}",
                    got, t.node_d[slot * 121 + a], t.node_d[slot * 121 + b]);
            }
        }
    }
    let crossings = t.n(t.words.cross) as usize;
    // Per-plane crossing offsets: the owned crossings of each plane's slots, in plane order.
    for p in 0..planes { per_plane[p + 1] += per_plane[p]; }
    if t.cross_off != per_plane { fails.push(format!("{}: cross_off {:?}, the crossings' planes give {per_plane:?}", c.name, t.cross_off)); }
    // Each crossing's gradient: within the Lipschitz bound, its normal pointing outward.
    for (x, q) in t.cross.iter().enumerate() {
        let (g, n) = (t.cross_g[x], [q[2] as f64, q[3] as f64]);
        let p = t.mm([q[0], q[1]]);
        let z = t.zs[(0..planes).rev().find(|p| t.cross_off[*p] as usize <= x).unwrap()] as f64;
        let out = oracle([p[0] + n[0] * 0.01, p[1] + n[1] * 0.01], z) - oracle([p[0] - n[0] * 0.01, p[1] - n[1] * 0.01], z);
        if !(g > 0.0 && g <= 0.1415) || out <= 0.0 { fails.push(format!("{}: crossing {x} gradient {g}, normal {n:?} (outward difference {out})", c.name)); break }
    }
    if crossings != expected { fails.push(format!("{}: {crossings} crossings, {expected} sign-changing edges", c.name)); }
    if worst_cross > 1e-4 { fails.push(format!("{}: a crossing is {worst_cross:.2e} mm off the f64 surface: {worst_at}", c.name)); }
    // The FeatureSet: links, roots, bitwise closure, keys, and the outline against the oracle.
    let segs = t.seg.len();
    for k in 0..segs {
        let [root, next, prev, flags] = t.seg_info[k];
        if t.seg_info[next as usize][2] != k as u32 || t.seg_info[prev as usize][1] != k as u32 { fails.push(format!("{}: seg {k} links", c.name)); break }
        if t.seg[k][2].to_bits() != t.seg[next as usize][0].to_bits() || t.seg[k][3].to_bits() != t.seg[next as usize][1].to_bits() { fails.push(format!("{}: seg {k} does not close bitwise", c.name)); break }
        if root > k as u32 || t.seg_info[root as usize][0] != root { fails.push(format!("{}: seg {k} root {root}", c.name)); break }
        if flags & plane_abi::SEG_CREASE != 0 && !c.creases { fails.push(format!("{}: seg {k} has a false crease", c.name)); }
        let key = t.seg_key[k];
        if key & CELL_BIT == 0 && !t.cross_key.contains(&key) { fails.push(format!("{}: seg {k} key {key} is no crossing", c.name)); }
        // A middle's key is its span's fine cell (a crease vertex may lie just outside it): the
        // span's first seg, just before it, starts at a crossing on one of that cell's edges.
        if key & CELL_BIT != 0 {
            let s = g.node_stride();
            let (cell, edge) = (key & !CELL_BIT, t.seg_key[k.wrapping_sub(1).min(segs - 1)]);
            let (fx, fy, gx, gy, axis) = (cell % s, cell / s, edge / 2 % s, edge / 2 / s, edge % 2);
            let sides = [plane_abi::edge_id(fx, fy, 0, g.nx), plane_abi::edge_id(fx, fy + 1, 0, g.nx), plane_abi::edge_id(fx, fy, 1, g.nx), plane_abi::edge_id(fx + 1, fy, 1, g.nx)];
            if k == 0 || plane_abi::cell_id(fx, fy, g.nx) != cell || !sides.contains(&edge) {
                fails.push(format!("{}: seg {k} middle keyed to cell ({fx}, {fy}), its span starts on edge ({gx}, {gy}) axis {axis}", c.name));
            }
        }
    }
    // Per span seg offsets: nondecreasing, each span's segs before the next's, the seg total
    // after the last; per plane, the plane's first span's.
    let spans = t.span_seg.len();
    // by mode: straight 1, middle or vertex 2, zero length 0, a refined middle 2 or 4
    let by_mode = |x: usize| match t.span_m[x][2] { m if m < 10.5 => t.seg_n[x] == 1, m if m > 12.5 && m < 13.5 => t.seg_n[x] == 0,
        m if m > 13.5 => t.seg_n[x] == 2 || t.seg_n[x] == 4, _ => t.seg_n[x] == 2 };
    if let Some(x) = (0..spans).find(|&x| !by_mode(x)) {
        fails.push(format!("{}: span {x} mode {} has {} segs", c.name, t.span_m[x][2], t.seg_n[x]));
    }
    if (0..spans).any(|x| t.span_seg[x] + t.seg_n[x] != if x + 1 < spans { t.span_seg[x + 1] } else { segs as u32 }) {
        fails.push(format!("{}: span_seg is not the spans' seg scan", c.name));
    }
    let want: Vec<u32> = (0..=planes).map(|p| t.span_seg.get(t.cross_off[p] as usize).copied().unwrap_or(segs as u32)).collect();
    if t.seg_plane_off != want { fails.push(format!("{}: seg_plane_off {:?}, the spans give {want:?}", c.name, t.seg_plane_off)); }
    let mut worst_outline = 0f64;
    let mut outline_at = String::new();
    let mut worst_q = (0usize, [0f64; 2]);
    for p in 0..planes {
        let rings = t.rings(p);
        if rings.len() != c.loops { fails.push(format!("{}: plane {p}: {} loops, {} expected", c.name, rings.len(), c.loops)); }
        let z = t.zs[p] as f64;
        let chain: Vec<([f64; 2], [f64; 2])> = rings.iter().flat_map(|r| r.iter().zip(r.iter().cycle().skip(1)).take(r.len()).map(|(a, b)| (*a, *b))).collect();
        let to_chain = |q: [f64; 2]| chain.iter().map(|(a, b)| segment_distance(q, *a, *b)).fold(f64::INFINITY, f64::min);
        for (a, b) in &chain {
            for s in [0., 0.25, 0.5, 0.75] {
                let q = [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s];
                let e = match &c.rings { Some(r) => ring_distance(q, r), None => in_plane(&oracle, q, z) };
                if e > worst_outline { worst_outline = e; outline_at = format!("chain point {q:?} of plane {p}"); worst_q = (p, q); }
            }
        }
        // And the other way: the true outline is near the chain.
        if let Some(rs) = &c.rings {
            for r in rs { for (a, b) in r.iter().zip(r.iter().cycle().skip(1)).take(r.len()) {
                for s in 0..=40 {
                    let q = [a[0] + (b[0] - a[0]) * s as f64 / 40., a[1] + (b[1] - a[1]) * s as f64 / 40.];
                    let e = to_chain(q);
                    if e > worst_outline { worst_outline = e; outline_at = format!("outline point {q:?} of plane {p}"); worst_q = (p, q); }
                }
            } }
        }
    }
    if worst_outline > 0.01 {
        // the spans whose crossings are within 0.3 mm of the point: ends, normals, final state
        let (p, q) = worst_q;
        let near: Vec<String> = (t.cross_off[p] as usize..t.cross_off[p + 1] as usize).filter(|&x| {
            let a = t.mm([t.cross[x][0], t.cross[x][1]]);
            (a[0] - q[0]).hypot(a[1] - q[1]) < 0.3
        }).map(|x| {
            let [_, start, end, _] = t.span_src[x];
            let (a, b) = (t.cross[start as usize], t.cross[end as usize]);
            let (m, st) = (t.span_m[x], t.span_a[x]);
            let v = t.mm([m[0], m[1]]);
            format!("  span {x}: {:.4?} n {:.3?} |g| {:.4} -> {:.4?} n {:.3?} |g| {:.4}: mode {} cycle {} at ({:.4}, {:.4}), state {:.4?}, samples {}",
                t.mm([a[0], a[1]]), [a[2], a[3]], t.cross_g[start as usize], t.mm([b[0], b[1]]), [b[2], b[3]], t.cross_g[end as usize],
                m[2], m[3], v[0], v[1], st, t.span_v[4 * x..4 * x + 4].iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>().join(" "))
        }).collect();
        fails.push(format!("{}: the outline is {worst_outline:.4} mm off the oracle at {outline_at}\n{}", c.name, near.join("\n")));
    }
    (worst_cross, worst_outline)
}

#[test]
fn plane_stages_match_twin_and_oracle() {
    gpu_rig::with_gpu(|cx| {
        let mut fails = Vec::new();
        for c in cases() {
            let part = Part::new(cx, &c.node);
            // The sampler alone, plain and resumed, bit for bit against the twin.
            let points: Vec<[f32; 3]> = (0..200).map(|i| [((i * 37) % 61) as f32 * 0.1 - 3.0, ((i * 53) % 47) as f32 * 0.1 - 2.3, 0.137]).collect();
            let want: Vec<f32> = twin::eval_many(&part.program.texels, &points, &part.slots).into_iter().map(|(v, _)| if v == 0.0 { ZERO } else { v }).collect();
            for bounded in [true, false] {
                let got = sample_points(cx, &part, &points, bounded);
                if let Some(i) = (0..points.len()).find(|i| got[*i].to_bits() != want[*i].to_bits()) {
                    fails.push(format!("{} (bounded {bounded}): sampler at {:?}: {} for {}", c.name, points[i], got[i], want[i]));
                }
            }
            let zs = [0.137, -0.3];
            let t = trace(cx, &part, IDENTITY, c.bounds, &zs, 1.0, true);
            let (cross, outline) = check(&t, &c, &c.node, &part.slots, &part.program.texels, &mut fails);
            // The same bytes resumed, and at very different job sizes.
            let bytes = t.bytes();
            for (bounded, force) in [(false, None), (true, Some(64)), (true, Some(1 << 16))] {
                gpu_rig::force_elements(cx, force);
                if trace(cx, &part, IDENTITY, c.bounds, &zs, 1.0, bounded).bytes() != bytes {
                    fails.push(format!("{}: outputs differ when bounded {bounded}, jobs {force:?}", c.name));
                }
            }
            gpu_rig::force_elements(cx, None);
            // The packed item: each plane's loops as the FeatureSet walks them, and the same bits
            // whatever batch the plane is traced in, in either order.
            let item = plane_driver::time(cx, &part, IDENTITY, c.bounds, &zs, 1.0, true);
            for p in 0..zs.len() {
                let walked = t.rings(p);
                let packed = &item.loops[p];
                let same = walked.len() == packed.len() && walked.iter().zip(packed).all(|(a, b)| a.len() == b.len()
                    && a.iter().zip(b).all(|(u, v)| (u[0] - v[0] as f64).abs() < 1e-5 && (u[1] - v[1] as f64).abs() < 1e-5));
                if !same { fails.push(format!("{}: plane {p}: the packed loops differ from the FeatureSet's", c.name)); }
            }
            let bits = |l: &Vec<Vec<[f32; 2]>>| l.iter().flatten().flat_map(|q| [q[0].to_bits(), q[1].to_bits()]).collect::<Vec<_>>();
            let alone = plane_driver::time(cx, &part, IDENTITY, c.bounds, &zs[1..], 1.0, true);
            let reversed = plane_driver::time(cx, &part, IDENTITY, c.bounds, &[zs[1], zs[0]], 1.0, true);
            if bits(&alone.loops[0]) != bits(&item.loops[1]) || bits(&reversed.loops[0]) != bits(&item.loops[1]) || bits(&reversed.loops[1]) != bits(&item.loops[0]) {
                fails.push(format!("{}: a plane's packed loops change with its batch", c.name));
            }
            let (w, n) = (t.words, |i: u32| t.n(i));
            let bracket: u32 = (0..plane::CROSS_ROUNDS).map(|k| n(w.cross_req + k)).sum();
            let span: u32 = (0..plane::SPAN_ROUNDS).map(|k| n(w.span_req + k)).sum();
            let creases = t.seg_info.iter().filter(|i| i[3] & plane_abi::SEG_CREASE != 0).count();
            let middles = t.seg_key.iter().filter(|k| **k & CELL_BIT != 0).count();
            let refined = t.seg_n.iter().filter(|n| **n == 4).count();
            println!("PLANE {}: kept {} nodes {} (sampled {}) crossings {} (bracket {bracket}, gradient {}) centres {} spans {} (samples {span}) segs {} middles {middles} (refined {refined}) creases {creases} loops {}; worst crossing {cross:.1e} mm, outline {outline:.4} mm; GPU {:.3} ms, wall {:.1} ms, {} submissions, {} jobs",
                c.name, n(w.kept), n(w.kept) * 121, n(w.node_req), n(w.cross), n(w.grad_req), n(w.centre_req), n(w.cross), n(w.segs), n(w.loops),
                t.stats.gpu_ms, t.wall_ms, t.stats.submissions, t.stats.jobs);
        }
        assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.join("\n"));
    });
}
