//! The FeatureSet for contours traced on the cores (GPU-EXEC §10.3): closed loops laid out
//! as plane tracing lays out its outline, so the field, wall and region stages read either.
//!
//! - A point within [`SNAP`] of a fine grid line is moved onto it, and every fine grid line a
//!   loop crosses inserts a vertex with that coordinate exact, so each seg lies in one fine
//!   cell (the cell of its midpoint; a seg on a grid line belongs to the cell above or right);
//! - segs are directed with the inside on the left (a loop whose signed area disagrees with
//!   its hole flag is reversed) and ordered slot-major: (plane, coarse cell, local fine cell,
//!   loop, position). A zero-length seg is dropped, so loops close bitwise;
//! - `seg_info` = (root, next, prev, flags): the loop's root (its smallest seg index), the
//!   neighbours, `SEG_REFLEX` (cross(prev dir, dir) < 0), `SEG_CREASE`, and in bits 8..15 the
//!   start vertex's ordinal under its key: its rank along its fine edge in +axis order, or
//!   among the inner vertices of its cell in seg order;
//! - `seg_key` = the start vertex's key: its fine edge (`plane_abi::edge_id`; a vertex on a
//!   node takes the node's +x edge), or `CELL_BIT | plane_abi::cell_id` inside a cell;
//! - `loop_off` per plane over `loop_info` = (root, n, plane, 0), ascending by root;
//! - the cell index: `set_slot` (plane * nx * ny + coarse cell → slot or NONE), `set_kept`
//!   (slot → coarse cell) with `set_kept_off` per plane, and `seg_off` over slot * 100 + local.
//!
//! Units: grid units (0.1 mm = 1.0) relative to the part's origin O. The part's grid is
//! nx × ny coarse cells of 1 mm, so points lie in [0, 10 nx] × [0, 10 ny].
use crate::plane_abi::{cell_id, edge_id, CELL_BIT, NONE, SEG_CREASE, SEG_ORDINAL_MASK, SEG_ORDINAL_SHIFT, SEG_REFLEX};
use std::collections::HashMap;

/// Points this close to a fine grid line (grid units, 10 nm) are moved onto it.
pub const SNAP: f64 = 1.0e-4;

/// One closed loop, in grid units relative to the part's origin.
#[derive(Clone, Copy, Debug)]
pub struct Loop<'a> {
    pub points: &'a [[f64; 2]],
    /// Per point: the vertex is a crease (empty: none is).
    pub creases: &'a [bool],
    pub hole: bool,
}

/// A point in printer mm, in grid units relative to the origin O (mm).
pub fn grid_from_mm(p: [f64; 2], origin_mm: [f64; 2]) -> [f64; 2] { [(p[0] - origin_mm[0]) * 10.0, (p[1] - origin_mm[1]) * 10.0] }
/// A point in nm, in grid units relative to the origin O (nm).
pub fn grid_from_nm(p: [i64; 2], origin_nm: [i64; 2]) -> [f64; 2] { [(p[0] - origin_nm[0]) as f64 / 1.0e5, (p[1] - origin_nm[1]) as f64 / 1.0e5] }

/// A FeatureSet's buffers, record by record (plane_abi layouts).
#[derive(Clone, Debug, Default)]
pub struct FeatureSet {
    pub nx: u32,
    pub ny: u32,
    pub planes: u32,
    pub seg: Vec<[f32; 4]>,
    pub seg_info: Vec<[u32; 4]>,
    pub seg_key: Vec<u32>,
    pub loop_off: Vec<u32>,
    pub loop_info: Vec<[u32; 4]>,
    /// Per loop, in `loop_info` order: a hole.
    pub loop_hole: Vec<bool>,
    pub set_slot: Vec<u32>,
    pub set_kept: Vec<u32>,
    pub set_kept_off: Vec<u32>,
    pub seg_off: Vec<u32>,
}

impl FeatureSet {
    /// Per loop: the contour or hole bias (grid units, + grows the part).
    pub fn loop_bias(&self, contour: f32, hole: f32) -> Vec<f32> { self.loop_hole.iter().map(|&h| if h { hole } else { contour }).collect() }
}

fn snap(v: f64) -> f64 {
    let r = v.round();
    if (v - r).abs() <= SNAP { r } else { v }
}
fn round(p: [f64; 2]) -> [f64; 2] { [snap(p[0]) as f32 as f64, snap(p[1]) as f32 as f64] }
fn area(points: &[[f64; 2]]) -> f64 {
    (0..points.len()).map(|i| { let (a, b) = (points[i], points[(i + 1) % points.len()]); a[0] * b[1] - a[1] * b[0] }).sum::<f64>() * 0.5
}

/// The FeatureSet of `planes[p]`'s loops on plane p of an nx × ny coarse grid.
pub fn from_loops(nx: u32, ny: u32, planes: &[Vec<Loop<'_>>]) -> Result<FeatureSet, String> {
    if nx == 0 || ny == 0 || (nx as u64 * 10 + 1) * (ny as u64 * 10 + 1) * 2 >= CELL_BIT as u64 {
        return Err(format!("a {nx} x {ny} grid has no FeatureSet ids"));
    }
    let (w, h) = (nx as f64 * 10.0, ny as f64 * 10.0);
    struct Raw { plane: u32, lp: u32, pos: u32, a: [f64; 2], b: [f64; 2], crease: bool, cell: [u32; 2] }
    let mut raws = Vec::new();
    let mut loop_meta = Vec::new();
    for (p, loops) in planes.iter().enumerate() {
        for l in loops {
            if !l.creases.is_empty() && l.creases.len() != l.points.len() { return Err("a loop's crease flags do not match its points".into()); }
            let mut points: Vec<([f64; 2], bool)> = l.points.iter().enumerate()
                .map(|(i, &q)| (round(q), l.creases.get(i).copied().unwrap_or(false))).collect();
            if let Some(q) = points.iter().find(|q| !(q.0[0] >= 0.0 && q.0[0] <= w && q.0[1] >= 0.0 && q.0[1] <= h)) {
                return Err(format!("loop point {:?} outside the {nx} x {ny} grid", q.0));
            }
            let pts: Vec<[f64; 2]> = points.iter().map(|q| q.0).collect();
            if (area(&pts) < 0.0) != l.hole { points.reverse(); }
            // the loop with a vertex on every fine grid line it crosses
            let mut verts: Vec<([f64; 2], bool)> = Vec::new();
            for i in 0..points.len() {
                let (a, crease) = points[i];
                let b = points[(i + 1) % points.len()].0;
                verts.push((a, crease));
                let mut ts = Vec::new();
                for axis in 0..2 {
                    let (lo, hi) = (a[axis].min(b[axis]), a[axis].max(b[axis]));
                    let mut k = lo.floor() + 1.0;
                    while k < hi {
                        let t = (k - a[axis]) / (b[axis] - a[axis]);
                        let mut q = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
                        q[axis] = k;
                        ts.push((t, round(q)));
                        k += 1.0;
                    }
                }
                ts.sort_by(|x, y| x.0.total_cmp(&y.0));
                verts.extend(ts.into_iter().map(|(_, q)| (q, false)));
            }
            verts.dedup_by(|x, y| x.0 == y.0);
            while verts.len() > 1 && verts.first().unwrap().0 == verts.last().unwrap().0 { verts.pop(); }
            if verts.len() < 3 { continue; }
            let lp = loop_meta.len() as u32;
            loop_meta.push((p as u32, l.hole));
            for i in 0..verts.len() {
                let (a, crease) = verts[i];
                let b = verts[(i + 1) % verts.len()].0;
                let cell = [((a[0] + b[0]) * 0.5).floor(), ((a[1] + b[1]) * 0.5).floor()];
                if cell[0] >= w || cell[1] >= h { return Err(format!("seg {a:?} -> {b:?} on the grid's far edge")); }
                let cell = [cell[0] as u32, cell[1] as u32];
                let inside = |q: [f64; 2]| q[0] >= cell[0] as f64 && q[0] <= cell[0] as f64 + 1.0 && q[1] >= cell[1] as f64 && q[1] <= cell[1] as f64 + 1.0;
                if !inside(a) || !inside(b) { return Err(format!("seg {a:?} -> {b:?} leaves fine cell {cell:?}")); }
                raws.push(Raw { plane: p as u32, lp, pos: i as u32, a, b, crease, cell });
            }
        }
    }
    let coarse_of = |c: [u32; 2]| (c[1] / 10) * nx + c[0] / 10;
    let local_of = |c: [u32; 2]| (c[1] % 10) * 10 + c[0] % 10;
    let mut order: Vec<usize> = (0..raws.len()).collect();
    order.sort_by_key(|&i| { let r = &raws[i]; (r.plane, coarse_of(r.cell), local_of(r.cell), r.lp, r.pos) });
    let mut index_of = vec![0u32; raws.len()];
    for (k, &i) in order.iter().enumerate() { index_of[i] = k as u32; }
    // raws are loop-contiguous in position order
    let mut first = 0;
    let n = raws.len();
    let mut next = vec![0u32; n];
    let mut prev = vec![0u32; n];
    let mut root = vec![0u32; n];
    let mut loops = Vec::new();
    while first < n {
        let mut end = first;
        while end < n && raws[end].lp == raws[first].lp { end += 1; }
        let len = end - first;
        let r = (first..end).map(|i| index_of[i]).min().unwrap();
        for i in first..end {
            next[i] = index_of[first + (i - first + 1) % len];
            prev[i] = index_of[first + (i - first + len - 1) % len];
            root[i] = r;
        }
        let (plane, hole) = loop_meta[raws[first].lp as usize];
        loops.push((r, len as u32, plane, hole));
        first = end;
    }
    let mut set = FeatureSet { nx, ny, planes: planes.len() as u32, ..Default::default() };
    set.seg = vec![[0.0; 4]; n];
    set.seg_info = vec![[0; 4]; n];
    set.seg_key = vec![0; n];
    // per key: (position along the edge, seg) for the ordinals
    let mut under_key: HashMap<(u32, u32), Vec<(f64, u32)>> = HashMap::new();
    for (i, r) in raws.iter().enumerate() {
        let k = index_of[i];
        set.seg[k as usize] = [r.a[0] as f32, r.a[1] as f32, r.b[0] as f32, r.b[1] as f32];
        let pa = raws[order[prev[i] as usize]].a;
        let (d0, d1) = ([r.a[0] - pa[0], r.a[1] - pa[1]], [r.b[0] - r.a[0], r.b[1] - r.a[1]]);
        let mut flags = 0;
        if d0[0] * d1[1] - d0[1] * d1[0] < 0.0 { flags |= SEG_REFLEX; }
        if r.crease { flags |= SEG_CREASE; }
        set.seg_info[k as usize] = [root[i], next[i], prev[i], flags];
        let (x, y) = (r.a[0], r.a[1]);
        let (key, along) = if y == y.floor() { (edge_id(x.floor() as u32, y as u32, 0, nx), x) }
            else if x == x.floor() { (edge_id(x as u32, y.floor() as u32, 1, nx), y) }
            else { (CELL_BIT | cell_id(r.cell[0], r.cell[1], nx), 0.0) };
        set.seg_key[k as usize] = key;
        under_key.entry((r.plane, key)).or_default().push((along, k));
    }
    for list in under_key.values_mut() {
        if list.len() > SEG_ORDINAL_MASK as usize + 1 { return Err(format!("{} vertices under one key (at most 256)", list.len())); }
        list.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (ordinal, &(_, s)) in list.iter().enumerate() { set.seg_info[s as usize][3] |= (ordinal as u32) << SEG_ORDINAL_SHIFT; }
    }
    loops.sort();
    set.loop_off = vec![0; planes.len() + 1];
    for &(r, len, plane, hole) in &loops {
        set.loop_info.push([r, len, plane, 0]);
        set.loop_hole.push(hole);
        set.loop_off[plane as usize + 1] += 1;
    }
    for p in 0..planes.len() { set.loop_off[p + 1] += set.loop_off[p]; }
    let cells = (nx * ny) as usize;
    set.set_slot = vec![NONE; planes.len() * cells];
    set.set_kept_off = vec![0; planes.len() + 1];
    for &i in &order {
        let r = &raws[i];
        let at = r.plane as usize * cells + coarse_of(r.cell) as usize;
        if set.set_slot[at] == NONE {
            set.set_slot[at] = set.set_kept.len() as u32;
            set.set_kept.push(coarse_of(r.cell));
            set.set_kept_off[r.plane as usize + 1] = set.set_kept.len() as u32;
        }
    }
    for p in 0..planes.len() { set.set_kept_off[p + 1] = set.set_kept_off[p + 1].max(set.set_kept_off[p]); }
    set.seg_off = vec![0; set.set_kept.len() * 100 + 1];
    for &i in &order {
        let r = &raws[i];
        let slot = set.set_slot[r.plane as usize * cells + coarse_of(r.cell) as usize];
        set.seg_off[(slot * 100 + local_of(r.cell)) as usize + 1] += 1;
    }
    for k in 0..set.seg_off.len() - 1 { set.seg_off[k + 1] += set.seg_off[k]; }
    Ok(set)
}

#[cfg(test)]
#[path = "feature_set_tests.rs"]
mod tests;
