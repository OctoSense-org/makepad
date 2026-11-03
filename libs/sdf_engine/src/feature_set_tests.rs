//! from_loops against its definition: every seg in its fine cell on the input polyline,
//! loops closed bitwise, slot-major order, keys and ordinals as plane_abi defines them.
use super::*;

#[path = "../tests/support/loops.rs"]
mod loops;
use loops::{circle, polygon, rotated_square, Ring};

fn set_of(nx: u32, ny: u32, planes: &[Vec<Ring>]) -> Result<FeatureSet, String> {
    let planes: Vec<Vec<Loop>> = planes.iter().map(|p| p.iter().map(|r| Loop { points: &r.points, creases: &r.creases, hole: r.hole }).collect()).collect();
    from_loops(nx, ny, &planes)
}

fn dist_to_seg(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let l2 = d[0] * d[0] + d[1] * d[1];
    let t = if l2 > 0.0 { (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / l2).clamp(0.0, 1.0) } else { 0.0 };
    ((p[0] - a[0] - t * d[0]).powi(2) + (p[1] - a[1] - t * d[1]).powi(2)).sqrt()
}

/// Every invariant of the layout, and the geometry against the input rings.
fn check(name: &str, set: &FeatureSet, planes: &[Vec<Ring>]) {
    let n = set.seg.len();
    let nxn = set.nx * 10 + 1;
    let at = |s: usize| [set.seg[s][0] as f64, set.seg[s][1] as f64];
    let to = |s: usize| [set.seg[s][2] as f64, set.seg[s][3] as f64];
    let cells = (set.nx * set.ny) as usize;
    // the cell index and slot-major order
    assert_eq!(set.seg_off.len(), set.set_kept.len() * 100 + 1, "{name}: seg_off size");
    assert_eq!(*set.seg_off.last().unwrap() as usize, n, "{name}: seg_off total");
    let mut plane_of = vec![0u32; n];
    for p in 0..set.planes {
        for slot in set.set_kept_off[p as usize]..set.set_kept_off[p as usize + 1] {
            let coarse = set.set_kept[slot as usize];
            assert_eq!(set.set_slot[p as usize * cells + coarse as usize], slot, "{name}: set_slot");
            if slot > set.set_kept_off[p as usize] { assert!(set.set_kept[slot as usize - 1] < coarse, "{name}: kept ascending"); }
            for local in 0..100u32 {
                let (fx, fy) = (coarse % set.nx * 10 + local % 10, coarse / set.nx * 10 + local / 10);
                let k = (slot * 100 + local) as usize;
                for s in set.seg_off[k] as usize..set.seg_off[k + 1] as usize {
                    plane_of[s] = p;
                    let (a, b) = (at(s), to(s));
                    let m = [((a[0] + b[0]) * 0.5).floor() as u32, ((a[1] + b[1]) * 0.5).floor() as u32];
                    assert_eq!(m, [fx, fy], "{name}: seg {s} listed in fine cell ({fx}, {fy})");
                    for q in [a, b] {
                        assert!(q[0] >= fx as f64 && q[0] <= fx as f64 + 1.0 && q[1] >= fy as f64 && q[1] <= fy as f64 + 1.0,
                            "{name}: seg {s} leaves its cell at {q:?}");
                    }
                    assert!(a != b, "{name}: zero-length seg {s}");
                }
            }
        }
    }
    assert_eq!(set.set_slot.iter().filter(|&&s| s != NONE).count(), set.set_kept.len(), "{name}: slots");
    // loops: links, roots, bitwise closure
    for s in 0..n {
        let [root, next, prev, _] = set.seg_info[s];
        assert_eq!(set.seg_info[next as usize][2] as usize, s, "{name}: next/prev");
        assert_eq!(set.seg_info[prev as usize][1] as usize, s, "{name}: prev/next");
        assert_eq!(set.seg[s][2].to_bits(), set.seg[next as usize][0].to_bits(), "{name}: open at seg {s}");
        assert_eq!(set.seg[s][3].to_bits(), set.seg[next as usize][1].to_bits(), "{name}: open at seg {s}");
        assert_eq!(set.seg_info[root as usize][0], root, "{name}: root");
        assert!(root as usize <= s);
    }
    let mut seen = vec![false; n];
    assert_eq!(set.loop_off.len(), set.planes as usize + 1);
    assert_eq!(*set.loop_off.last().unwrap() as usize, set.loop_info.len());
    let mut area_by_plane = vec![0.0f64; set.planes as usize];
    for (l, info) in set.loop_info.iter().enumerate() {
        let [root, len, plane, _] = *info;
        assert!(set.loop_off[plane as usize] as usize <= l && l < set.loop_off[plane as usize + 1] as usize, "{name}: loop_off");
        if l > 0 { assert!(set.loop_info[l - 1][0] < root, "{name}: loops ascending by root"); }
        let mut s = root as usize;
        let mut pts = Vec::new();
        for _ in 0..len {
            assert!(!seen[s], "{name}: seg {s} in two loops");
            seen[s] = true;
            assert_eq!(set.seg_info[s][0], root);
            assert_eq!(plane_of[s], plane, "{name}: seg {s} plane");
            assert!(s as u32 >= root);
            pts.push(at(s));
            s = set.seg_info[s][1] as usize;
        }
        assert_eq!(s, root as usize, "{name}: loop {l} has {len} segs");
        let a = loops::area(&pts);
        assert_eq!(a < 0.0, set.loop_hole[l], "{name}: loop {l} orientation vs its hole flag");
        area_by_plane[plane as usize] += a;
    }
    assert!(seen.iter().all(|&s| s), "{name}: segs outside every loop");
    // reflex flags, keys and ordinals
    let mut under: HashMap<(u32, u32), Vec<(u32, usize)>> = HashMap::new();
    for s in 0..n {
        let flags = set.seg_info[s][3];
        let p = set.seg_info[s][2] as usize;
        let (d0, d1) = ([at(s)[0] - at(p)[0], at(s)[1] - at(p)[1]], [to(s)[0] - at(s)[0], to(s)[1] - at(s)[1]]);
        assert_eq!(flags & SEG_REFLEX != 0, d0[0] * d1[1] - d0[1] * d1[0] < 0.0, "{name}: reflex flag of seg {s}");
        let key = set.seg_key[s];
        let (x, y) = (at(s)[0], at(s)[1]);
        if key & CELL_BIT != 0 {
            let c = key & !CELL_BIT;
            let (fx, fy) = ((c % nxn) as f64, (c / nxn) as f64);
            assert!(x > fx && x < fx + 1.0 && y > fy && y < fy + 1.0, "{name}: seg {s} cell key at ({x}, {y})");
        } else {
            let (g, axis) = (key / 2, key % 2);
            let (gx, gy) = ((g % nxn) as f64, (g / nxn) as f64);
            if axis == 0 { assert!(y == gy && x >= gx && x < gx + 1.0, "{name}: seg {s} +x edge key at ({x}, {y})"); }
            else { assert!(x == gx && y > gy && y < gy + 1.0, "{name}: seg {s} +y edge key at ({x}, {y})"); }
        }
        under.entry((plane_of[s], key)).or_default().push(((flags >> SEG_ORDINAL_SHIFT) & 0xff, s));
    }
    for ((_, key), mut list) in under {
        list.sort();
        for (k, &(ordinal, s)) in list.iter().enumerate() {
            assert_eq!(ordinal as usize, k, "{name}: ordinals under key {key:#x}");
            if k > 0 && key & CELL_BIT == 0 {
                let (prev, axis) = (list[k - 1].1, (key % 2) as usize);
                assert!(at(prev)[axis] < at(s)[axis], "{name}: edge ordinals not in +axis order under {key:#x}");
            }
        }
    }
    // geometry: vertices on the input, areas kept
    for p in 0..set.planes as usize {
        let rings = &planes[p];
        for s in (0..n).filter(|&s| plane_of[s] == p as u32) {
            let q = at(s);
            let d = rings.iter().flat_map(|r| (0..r.points.len()).map(move |i| dist_to_seg(q, r.points[i], r.points[(i + 1) % r.points.len()])))
                .fold(f64::INFINITY, f64::min);
            assert!(d <= 2.0 * SNAP, "{name}: vertex {q:?} is {d} off the input");
        }
        let want: f64 = rings.iter().map(|r| { let a = loops::area(&r.points); if r.hole { -a.abs() } else { a.abs() } }).sum();
        assert!((area_by_plane[p] - want).abs() <= 1e-3 * want.abs().max(1.0), "{name}: plane {p} area {} vs {want}", area_by_plane[p]);
    }
}

#[test]
fn feature_sets_from_loops_keep_the_layout() {
    let disk = circle([40.37, 39.71], 24.3, 720, false);
    let l = polygon(&[[12.3, 12.6], [60.7, 12.6], [60.7, 30.2], [31.4, 30.2], [31.4, 66.9], [12.3, 66.9]]);
    let boxed = vec![polygon(&[[14.3, 15.2], [65.9, 15.2], [65.9, 63.7], [14.3, 63.7]]), circle([40.6, 39.4], 9.3, 6, true)];
    // on grid lines and nodes, and within the snap of them
    let aligned = polygon(&[[10.0, 10.0], [50.0, 10.0], [50.0, 30.0], [30.00004, 30.0], [29.99997, 50.00003], [10.0, 50.0]]);
    // a spike crossing the same fine edges twice, and a loop turning inside one fine cell
    let spike = polygon(&[[10.2, 5.3], [10.8, 5.3], [10.5, 8.7]]);
    let speck = polygon(&[[20.1, 20.1], [20.9, 20.1], [20.9, 20.9], [20.1, 20.9]]);
    // a contour given clockwise
    let mut reversed = rotated_square([40.0, 40.0], 12.0, 23.0);
    reversed.points.reverse();
    reversed.hole = false;
    let cases: Vec<(&str, u32, Vec<Vec<Ring>>)> = vec![
        ("disk", 8, vec![vec![disk.clone()]]),
        ("L", 8, vec![vec![l]]),
        ("box-hole", 8, vec![boxed.clone()]),
        ("aligned", 6, vec![vec![aligned]]),
        ("spike and speck", 3, vec![vec![spike, speck]]),
        ("reversed contour", 8, vec![vec![reversed]]),
        ("two planes", 8, vec![vec![disk], boxed]),
        ("far from the origin", 200, vec![vec![rotated_square([1951.37, 1960.21], 18.0, 23.0)]]),
    ];
    for (name, n, planes) in &cases {
        let set = set_of(*n, *n, planes).unwrap_or_else(|e| panic!("{name}: {e}"));
        check(name, &set, planes);
    }
    // the cases above reach every key kind and ordinals past 0
    let set = set_of(3, 3, &cases[4].2).unwrap();
    let ordinals = |cell: bool| set.seg_info.iter().zip(&set.seg_key)
        .filter(|(_, &k)| (k & CELL_BIT != 0) == cell).map(|(i, _)| (i[3] >> SEG_ORDINAL_SHIFT) & 0xff).max().unwrap();
    assert_eq!(ordinals(false), 1, "two crossings on one fine edge");
    assert_eq!(ordinals(true), 3, "four corners in one fine cell");
    let aligned = set_of(6, 6, &cases[3].2).unwrap();
    assert!(aligned.seg.iter().flatten().all(|v| v.fract() == 0.0), "snapped onto the grid lines");
}

#[test]
fn feature_sets_refuse_points_off_the_grid() {
    let off = polygon(&[[-0.5, 1.0], [5.0, 1.0], [5.0, 5.0]]);
    assert!(set_of(1, 1, &[vec![off]]).is_err());
    let far_edge = polygon(&[[1.0, 1.0], [10.0, 1.0], [10.0, 5.0]]);
    assert!(set_of(1, 1, &[vec![far_edge]]).is_err(), "a seg on x = 10 nx belongs to no cell");
}
