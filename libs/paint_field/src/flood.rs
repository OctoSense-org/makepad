//! Smooth-region fill: from a seed point the fill spreads over a part's surface while each
//! step's normal stays within a maximum angle of the one it came from, so it stops at creases and
//! edges sharper than that. The angle is the surface's turn over the last [`SPAN`] of the fill's
//! path (a step's normal against every one behind it that close), so a rounding counts as the
//! edge it rounds (a 1 mm edge radius stops a 30 degree fill as the sharp edge would), a sharp
//! crease of 40 degrees stops a 30 degree fill and not a 45 degree one, and the result does not
//! depend on how finely the surface is tessellated or sampled. At 30 degrees a fill crosses
//! roundings wider than about 4 mm radius.
//!
//! - A mesh (CSG, B-rep, imported): [`Patch::flood`] walks its triangles across shared edges,
//!   a normal per triangle.
//! - A distance field: [`Patch::flood_field`] walks the surface cells of a fine lattice (the
//!   cells the zero level passes), each cell's normal the field's gradient; nothing is meshed.
//!
//! The result is a [`Patch`]: the surface sites the walk saw (triangles, or the field's surface
//! points), in bounding-volume trees for flooded and other sites. A point is inside when its
//! nearest site is flooded (a field's point first goes to its surface point along the gradient); the distance
//! is half the gap between the nearest flooded and the nearest other site (the bisector between
//! them), so the edge sits on the crease at any depth through the wall.

use crate::program::P3;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// [`Patch::distance`] is exact this far from the surface (mm): a brick's reach.
pub const REACH_MM: f64 = 4.0;

/// The surface length the maximum angle is measured over (mm).
pub const SPAN: f64 = 2.0;

/// Whether a step to (`at`, `n`) from walk node `from` turns within `max_angle` of every node
/// on the path behind it within [`SPAN`] of `at` (`nodes`: the site, its normal and the node it
/// was reached from, the root its own; a site is a triangle, by its nearest point, so a strip of
/// slivers along a rounded edge counts by its width).
fn smooth(nodes: &[([P3; 3], P3, u32)], from: u32, at: P3, n: P3, max_angle: f64) -> bool {
    let cos = max_angle.cos();
    let mut k = from;
    loop {
        let (t, nk, parent) = &nodes[k as usize];
        if dot(*nk, n) < cos {
            return false;
        }
        if *parent == k || tri_distance(at, t) >= SPAN {
            return true;
        }
        k = *parent;
    }
}

/// A part's distance field in the program's frame (mm, negative inside).
pub type Field = Arc<dyn Fn(P3) -> f64 + Send + Sync>;

pub struct Patch {
    /// The sites: triangles, or a field's surface points (degenerate triangles).
    tris: Vec<[P3; 3]>,
    inside: Vec<bool>,
    /// A field patch: the field (a point goes to its surface first) and the walk's pitch.
    surface: Option<(Field, f64)>,
    /// A field patch's site normals (the gradient at each).
    normals: Vec<P3>,
    /// Flat trees for flooded and other sites, sharing the leaf site indices.
    nodes: Vec<SiteNode>,
    sites: Vec<u32>,
    /// Flooded root first, then the other sites; an empty kind has no root.
    roots: [Option<usize>; 2],
    /// The flooded sites' box.
    bounds: (P3, P3),
    /// What was flooded: seed, angle, site count and flooded count (the paint key).
    summary: (P3, f64, usize, usize),
}

impl std::fmt::Debug for Patch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (seed, angle, n, inside) = self.summary;
        write!(f, "Patch {{ seed: {seed:?}, angle: {angle}, sites: {n}, flooded: {inside}, box: {:?} }}", self.bounds)
    }
}

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) struct SiteNode {
    pub(crate) min: P3,
    pub(crate) max: P3,
    pub(crate) kind: SiteNodeKind,
}

#[derive(Clone, Copy)]
pub(crate) enum SiteNodeKind {
    Leaf { start: usize, len: usize },
    Branch { left: usize, right: usize },
}

impl SiteNode {
    /// Builds a balanced tree over a nonempty slice; leaf ranges refer to the shared site array.
    pub(crate) fn build<T: Copy + Into<f64>>(tris: &[[[T; 3]; 3]], sites: &mut [u32], start: usize, nodes: &mut Vec<SiteNode>) -> usize {
        let (mut min, mut max) = ([f64::MAX; 3], [f64::MIN; 3]);
        for &i in sites.iter() {
            for p in &tris[i as usize] {
                for k in 0..3 {
                    min[k] = min[k].min(p[k].into());
                    max[k] = max[k].max(p[k].into());
                }
            }
        }
        let node = nodes.len();
        nodes.push(SiteNode { min, max, kind: SiteNodeKind::Leaf { start, len: sites.len() } });
        if sites.len() > 8 {
            let mut axis = 0;
            for k in 1..3 {
                if max[k] - min[k] > max[axis] - min[axis] {
                    axis = k;
                }
            }
            let centre = |i: u32| {
                let t = &tris[i as usize];
                (t[0][axis].into() + t[1][axis].into() + t[2][axis].into()) / 3.0
            };
            let mid = sites.len() / 2;
            sites.select_nth_unstable_by(mid, |&a, &b| centre(a).total_cmp(&centre(b)));
            let (a, b) = sites.split_at_mut(mid);
            let left = Self::build(tris, a, start, nodes);
            let right = Self::build(tris, b, start + mid, nodes);
            nodes[node].kind = SiteNodeKind::Branch { left, right };
        }
        node
    }

    /// Visits the nearer child first, tightening the radius in `best` without allocating.
    pub(crate) fn nearest<T: Copy + Into<f64>>(tris: &[[[T; 3]; 3]], sites: &[u32], nodes: &[SiteNode], p: P3, i: usize, lower: f64, best: &mut (Option<u32>, f64)) {
        // Include sites exactly at the initial radius; once found, ties cannot improve it.
        if lower > best.1 || (best.0.is_some() && lower >= best.1) {
            return;
        }
        match nodes[i].kind {
            SiteNodeKind::Leaf { start, len } => {
                for &i in &sites[start..start + len] {
                    let t = tris[i as usize].map(|p| p.map(Into::into));
                    let d = tri_distance(p, &t);
                    if d < best.1 || (best.0.is_none() && d <= best.1) {
                        *best = (Some(i), d);
                    }
                }
            }
            SiteNodeKind::Branch { left, right } => {
                let a = box_distance(p, &nodes[left]);
                let b = box_distance(p, &nodes[right]);
                if a <= b {
                    Self::nearest(tris, sites, nodes, p, left, a, best);
                    Self::nearest(tris, sites, nodes, p, right, b, best);
                } else {
                    Self::nearest(tris, sites, nodes, p, right, b, best);
                    Self::nearest(tris, sites, nodes, p, left, a, best);
                }
            }
        }
    }
}

/// A lower bound on the distance to any triangle in the node, including its corners.
pub(crate) fn box_distance(p: P3, node: &SiteNode) -> f64 {
    let d: P3 = std::array::from_fn(|k| (node.min[k] - p[k]).max(p[k] - node.max[k]).max(0.0));
    dot(d, d).sqrt()
}

/// The distance from `p` to triangle `t`.
pub(crate) fn tri_distance(p: P3, t: &[P3; 3]) -> f64 {
    let (a, b, c) = (t[0], t[1], t[2]);
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    let q = if d1 <= 0.0 && d2 <= 0.0 {
        a
    } else {
        let bp = sub(p, b);
        let (d3, d4) = (dot(ab, bp), dot(ac, bp));
        if d3 >= 0.0 && d4 <= d3 {
            b
        } else {
            let cp = sub(p, c);
            let (d5, d6) = (dot(ab, cp), dot(ac, cp));
            if d6 >= 0.0 && d5 <= d6 {
                c
            } else {
                let vc = d1 * d4 - d3 * d2;
                let vb = d5 * d2 - d1 * d6;
                let va = d3 * d6 - d5 * d4;
                if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                    let v = d1 / (d1 - d3);
                    [a[0] + ab[0] * v, a[1] + ab[1] * v, a[2] + ab[2] * v]
                } else if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                    let w = d2 / (d2 - d6);
                    [a[0] + ac[0] * w, a[1] + ac[1] * w, a[2] + ac[2] * w]
                } else if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
                    let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                    [b[0] + (c[0] - b[0]) * w, b[1] + (c[1] - b[1]) * w, b[2] + (c[2] - b[2]) * w]
                } else {
                    let den = 1.0 / (va + vb + vc);
                    let (v, w) = (vb * den, vc * den);
                    [a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w]
                }
            }
        }
    };
    let d = sub(p, q);
    dot(d, d).sqrt()
}

/// The field's unit gradient at `p` (central differences of step `e`).
fn gradient(field: &Field, p: P3, e: f64) -> P3 {
    let g: P3 = std::array::from_fn(|k| {
        let (mut a, mut b) = (p, p);
        a[k] += e;
        b[k] -= e;
        field(a) - field(b)
    });
    let l = dot(g, g).sqrt();
    if l > 0.0 {
        g.map(|v| v / l)
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// `p` moved to the field's surface along the gradient, in `steps` steps (one is enough for a
/// true distance near its surface).
fn project(field: &Field, p: P3, e: f64, steps: usize) -> P3 {
    let mut q = p;
    for _ in 0..steps {
        let d = field(q);
        if d.abs() < 1e-4 {
            break;
        }
        let n = gradient(field, q, e);
        q = std::array::from_fn(|k| q[k] - n[k] * d);
    }
    q
}

type Cells = HashMap<[i32; 3], (f64, Option<(P3, P3)>)>;

impl Patch {
    fn new(tris: Vec<[P3; 3]>, inside: Vec<bool>, surface: Option<(Field, f64)>, seed: P3, max_angle: f64) -> Patch {
        let mut sites: Vec<u32> = (0..tris.len()).filter(|&i| inside.get(i) == Some(&true)).map(|i| i as u32).collect();
        let flooded = sites.len();
        sites.extend((0..tris.len()).filter(|&i| inside.get(i) != Some(&true)).map(|i| i as u32));
        let mut nodes = Vec::new();
        let (a, b) = sites.split_at_mut(flooded);
        let roots = [
            (!a.is_empty()).then(|| SiteNode::build(&tris, a, 0, &mut nodes)),
            (!b.is_empty()).then(|| SiteNode::build(&tris, b, flooded, &mut nodes)),
        ];
        let bounds = roots[0].map_or(([f64::MAX; 3], [f64::MIN; 3]), |i| (nodes[i].min, nodes[i].max));
        let summary = (seed, max_angle, tris.len(), flooded);
        Patch { tris, inside, surface, normals: Vec::new(), nodes, sites, roots, bounds, summary }
    }

    /// Floods `tris` (with `normals`, one per triangle, unit) from the triangle nearest `seed`
    /// across shared edges while the surface turns less than `max_angle` (radians) over [`SPAN`]. `None` when no
    /// triangle lies within `near` of the seed (the seed is on another part).
    pub fn flood(tris: Vec<[P3; 3]>, normals: &[P3], seed: P3, max_angle: f64, near: f64) -> Option<Patch> {
        let patch = Patch::new(tris, Vec::new(), None, seed, max_angle);
        let (start, d) = patch.nearest(seed)?;
        if d > near {
            return None;
        }
        // shared edges by welded corners (a mesh's seams split its vertices)
        let key = |p: P3| -> [i64; 3] { std::array::from_fn(|k| (p[k] * 1e4).round() as i64) };
        let edge = |a: [i64; 3], b: [i64; 3]| if a < b { (a, b) } else { (b, a) };
        let mut edges: HashMap<([i64; 3], [i64; 3]), Vec<u32>> = HashMap::new();
        for (i, t) in patch.tris.iter().enumerate() {
            for e in 0..3 {
                edges.entry(edge(key(t[e]), key(t[(e + 1) % 3]))).or_default().push(i as u32);
            }
        }
        let centre = |t: &[P3; 3]| -> P3 { std::array::from_fn(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0) };
        // the walk: per triangle its centre, normal and the triangle it was reached from
        let mut nodes: Vec<([P3; 3], P3, u32)> = patch.tris.iter().zip(normals).enumerate().map(|(i, (t, n))| (*t, *n, i as u32)).collect();
        let mut inside = vec![false; patch.tris.len()];
        inside[start as usize] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(i) = queue.pop_front() {
            let t = patch.tris[i as usize];
            for e in 0..3 {
                for &j in &edges[&edge(key(t[e]), key(t[(e + 1) % 3]))] {
                    let (tj, n, _) = nodes[j as usize];
                    if !inside[j as usize] && smooth(&nodes, i, centre(&tj), n, max_angle) {
                        inside[j as usize] = true;
                        nodes[j as usize].2 = i;
                        queue.push_back(j);
                    }
                }
            }
        }
        Some(Patch::new(patch.tris, inside, None, seed, max_angle))
    }

    /// Floods a distance field's surface from the surface point nearest `seed`, over the cells
    /// of a lattice of `pitch` the zero level passes (a cell's normal the field's gradient at
    /// it), across the 26 neighbours while the surface turns less than `max_angle` (radians) over [`SPAN`]. `None` when
    /// the seed is further than `near` from the surface.
    pub fn flood_field(field: Field, seed: P3, max_angle: f64, near: f64, pitch: f64) -> Option<Patch> {
        let d0 = field(seed);
        if !d0.is_finite() || d0.abs() > near {
            return None;
        }
        let h = pitch;
        // a cell: the distance at its centre, and its surface site and normal when the zero
        // level passes it
        let cell = |c: [i32; 3], cells: &mut Cells| -> Option<(P3, P3)> {
            if let Some(v) = cells.get(&c) {
                return v.1;
            }
            let p: P3 = std::array::from_fn(|k| (c[k] as f64 + 0.5) * h);
            let d = field(p);
            let site = (d.abs() <= 0.87 * h).then(|| {
                let n = gradient(&field, p, 0.25 * h);
                (std::array::from_fn(|k| p[k] - n[k] * d), n)
            });
            cells.insert(c, (d, site));
            site
        };
        let mut cells: Cells = HashMap::new();
        let q0 = project(&field, seed, 0.25 * h, 3);
        let c0: [i32; 3] = std::array::from_fn(|k| (q0[k] / h).floor() as i32);
        // the start: the surface cell round the seed's surface point nearest the zero level
        let mut start: Option<([i32; 3], f64)> = None;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let c = [c0[0] + dx, c0[1] + dy, c0[2] + dz];
                    if let Some((site, _)) = cell(c, &mut cells) {
                        let d = dot(sub(site, q0), sub(site, q0));
                        if start.is_none_or(|(_, b)| d < b) {
                            start = Some((c, d));
                        }
                    }
                }
            }
        }
        let (start, _) = start?;
        // true: flooded; false: a surface cell it reached and did not cross into
        let mut state: HashMap<[i32; 3], bool> = HashMap::from([(start, true)]);
        // the flooded cells' walk nodes (site, normal, the node it was reached from)
        let (s0, n0) = cells[&start].1.expect("the start is on the surface");
        let mut nodes: Vec<([P3; 3], P3, u32)> = vec![([s0; 3], n0, 0)];
        let mut node_of: HashMap<[i32; 3], u32> = HashMap::from([(start, 0)]);
        let mut queue = VecDeque::from([start]);
        while let Some(c) = queue.pop_front() {
            let i = node_of[&c];
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let j = [c[0] + dx, c[1] + dy, c[2] + dz];
                        if state.get(&j) == Some(&true) {
                            continue;
                        }
                        let Some((sj, nj)) = cell(j, &mut cells) else { continue };
                        let crosses = smooth(&nodes, i, sj, nj, max_angle);
                        state.insert(j, crosses);
                        if crosses {
                            node_of.insert(j, nodes.len() as u32);
                            nodes.push(([sj; 3], nj, i));
                            queue.push_back(j);
                        }
                    }
                }
            }
        }
        // in lattice order: the same field and seed give the same sites
        let mut sites: Vec<([i32; 3], bool)> = state.into_iter().collect();
        sites.sort();
        let tris = sites
            .iter()
            .map(|(c, _)| {
                let p = cells[c].1.expect("a reached cell is on the surface").0;
                [p, p, p]
            })
            .collect();
        let inside = sites.iter().map(|(_, s)| *s).collect();
        let normals = sites.iter().map(|(c, _)| cells[c].1.expect("a reached cell is on the surface").1).collect();
        let mut patch = Patch::new(tris, inside, Some((field, h)), seed, max_angle);
        patch.normals = normals;
        Some(patch)
    }

    /// The flooded sites' box.
    pub fn bounds(&self) -> (P3, P3) {
        self.bounds
    }

    /// How many sites were flooded, of how many the walk saw.
    pub fn flooded(&self) -> (usize, usize) {
        (self.summary.3, self.summary.2)
    }

    /// The flooded sites: the triangles (a field's surface points as degenerate ones).
    pub fn flooded_sites(&self) -> impl Iterator<Item = &[P3; 3]> {
        self.tris.iter().zip(&self.inside).filter(|(_, i)| **i).map(|(t, _)| t)
    }

    /// The site nearest `p` within the patch's reach, and its distance.
    fn nearest(&self, p: P3) -> Option<(u32, f64)> {
        let mut best = (None, REACH_MM);
        let mut roots = self.roots.map(|root| (root, root.map_or(f64::INFINITY, |i| box_distance(p, &self.nodes[i]))));
        if roots[1].1 < roots[0].1 {
            roots.swap(0, 1);
        }
        for (root, lower) in roots {
            if let Some(root) = root {
                SiteNode::nearest(&self.tris, &self.sites, &self.nodes, p, root, lower, &mut best);
            }
        }
        best.0.map(|i| (i, best.1))
    }

    /// Exact distances to the nearest flooded and other sites, each within [`REACH_MM`].
    fn nearest_both(&self, p: P3) -> (Option<f64>, Option<f64>) {
        let [a, b] = self.roots.map(|root| {
            let mut best = (None, REACH_MM);
            if let Some(root) = root {
                SiteNode::nearest(&self.tris, &self.sites, &self.nodes, p, root, box_distance(p, &self.nodes[root]), &mut best);
            }
            best.0.map(|_| best.1)
        });
        (a, b)
    }

    /// Signed distance to the flooded region (negative inside), from the bisector between the
    /// nearest flooded and the nearest other site; past [`REACH_MM`] from both, that far (a
    /// lower bound).
    pub fn distance(&self, p: P3) -> f64 {
        let far = REACH_MM;
        let (p, tol) = match &self.surface {
            // a field point: one value of the field and the nearest site's normal take it to its
            // surface point, unless that site is not on the surface the point is nearest
            // (another wall, which the walk did not reach): outside
            Some((field, h)) => {
                let tol = 1.5 * h;
                let d = field(p);
                // off the surface: look for a site at its surface point along the gradient
                let at = if d.abs() <= *h {
                    p
                } else {
                    let g = gradient(field, p, 0.25 * h);
                    std::array::from_fn(|k| p[k] - g[k] * d)
                };
                let Some((i, _)) = self.nearest(at) else { return far };
                let (site, n) = (self.tris[i as usize][0], self.normals[i as usize]);
                let off = (d - dot(sub(p, site), n)).abs();
                if off > tol {
                    return (off - tol).max(0.01);
                }
                (std::array::from_fn(|k| p[k] - n[k] * d), Some(tol))
            }
            None => (p, None),
        };
        let (a, b) = self.nearest_both(p);
        if a.is_none() && b.is_none() {
            return far;
        }
        // a field point whose surface point is on none the walk reached (another wall): outside
        if let Some(tol) = tol {
            let gap = a.unwrap_or(far).min(b.unwrap_or(far));
            if gap > tol {
                return (gap - tol).max(0.01);
            }
        }
        // a kind with no site within reach is at least that far: the bisector's bound
        0.5 * (a.unwrap_or(far) - b.unwrap_or(far))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_both_matches_a_brute_force_scan() {
        let mut state = 0x5eed_u64;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 32) as f64 / u32::MAX as f64
        };
        let mut tris = Vec::new();
        let mut inside = Vec::new();
        for i in 0..384 {
            let p: P3 = std::array::from_fn(|_| 20.0 * random() - 10.0);
            // Both field sites and triangles whose corners extend beyond their centroids.
            let t = if i % 3 == 0 {
                [p; 3]
            } else {
                std::array::from_fn(|_| std::array::from_fn(|k| p[k] + 3.0 * random() - 1.5))
            };
            tris.push(t);
            inside.push(random() < 0.5);
        }
        let patch = Patch::new(tris, inside, None, [0.0; 3], 0.0);
        for _ in 0..200 {
            let p: P3 = std::array::from_fn(|_| 32.0 * random() - 16.0);
            let mut expected: [Option<f64>; 2] = [None, None];
            for (i, t) in patch.tris.iter().enumerate() {
                let d = tri_distance(p, t);
                let best = &mut expected[if patch.inside[i] { 0 } else { 1 }];
                if d <= REACH_MM && best.is_none_or(|b| d < b) {
                    *best = Some(d);
                }
            }
            let (a, b) = patch.nearest_both(p);
            for (actual, expected) in [a, b].into_iter().zip(expected) {
                match (actual, expected) {
                    (Some(a), Some(b)) => assert!((a - b).abs() <= 1e-12, "at {p:?}: {a} != {b}"),
                    _ => assert_eq!(actual, expected, "at {p:?}"),
                }
            }
            let nearest = expected.into_iter().flatten().reduce(f64::min);
            match (patch.nearest(p), nearest) {
                (Some((i, a)), Some(b)) => {
                    assert!((a - b).abs() <= 1e-12, "nearest at {p:?}: {a} != {b}");
                    assert_eq!(a, tri_distance(p, &patch.tris[i as usize]));
                }
                (None, None) => {}
                _ => panic!("nearest disagrees with the scan at {p:?}"),
            }
        }
        // Empty trees and the inclusive reach boundary for either kind.
        let empty = Patch::new(Vec::new(), Vec::new(), None, [0.0; 3], 0.0);
        assert_eq!(empty.nearest_both([0.0; 3]), (None, None));
        assert_eq!(empty.nearest([0.0; 3]), None);
        for inside in [true, false] {
            let patch = Patch::new(vec![[[0.0; 3]; 3]], vec![inside], None, [0.0; 3], 0.0);
            let expected = if inside { (Some(REACH_MM), None) } else { (None, Some(REACH_MM)) };
            assert_eq!(patch.nearest_both([REACH_MM, 0.0, 0.0]), expected);
            assert_eq!(patch.nearest([REACH_MM, 0.0, 0.0]), Some((0, REACH_MM)));
            assert_eq!(patch.nearest_both([REACH_MM + 1e-6, 0.0, 0.0]), (None, None));
            assert_eq!(patch.nearest([REACH_MM + 1e-6, 0.0, 0.0]), None);
        }
    }

    /// A box's faces as triangles (4 x 4 quads a face) with their face normals.
    fn cube(s: f64) -> (Vec<[P3; 3]>, Vec<P3>) {
        let mut tris = Vec::new();
        let mut normals = Vec::new();
        for axis in 0..3 {
            for sign in [-1.0f64, 1.0] {
                let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                let corner = |a: f64, b: f64| -> P3 {
                    let mut p = [0.0; 3];
                    p[axis] = sign * s;
                    p[u] = -s + 2.0 * s * a / 4.0;
                    p[v] = -s + 2.0 * s * b / 4.0;
                    p
                };
                for i in 0..4 {
                    for j in 0..4 {
                        let (i, j) = (i as f64, j as f64);
                        tris.push([corner(i, j), corner(i + 1.0, j), corner(i + 1.0, j + 1.0)]);
                        tris.push([corner(i, j), corner(i + 1.0, j + 1.0), corner(i, j + 1.0)]);
                        let mut n = [0.0; 3];
                        n[axis] = sign;
                        normals.push(n);
                        normals.push(n);
                    }
                }
            }
        }
        (tris, normals)
    }

    #[test]
    fn a_mesh_flood_stops_at_the_edges_sharper_than_its_angle() {
        let (tris, normals) = cube(10.0);
        let p = Patch::flood(tris.clone(), &normals, [0.0, 0.0, 10.0], 30f64.to_radians(), 1.0).unwrap();
        assert_eq!(p.flooded(), (32, 192), "one face of six");
        assert!(p.distance([0.0, 0.0, 10.0]) < 0.0 && p.distance([2.0, -3.0, 9.0]) < 0.0, "the face and the wall under it");
        assert!(p.distance([10.0, 0.0, 0.0]) > 0.0 && p.distance([0.0, 0.0, -10.0]) > 0.0, "a side and the far face");
        // the edge between the top and a side: the bisector of the two faces
        assert!(p.distance([9.0, 0.0, 9.5]) < 0.0 && p.distance([9.5, 0.0, 9.0]) > 0.0);
        // past 90 degrees the whole box
        assert_eq!(Patch::flood(tris.clone(), &normals, [0.0, 0.0, 10.0], 91f64.to_radians(), 1.0).unwrap().flooded(), (192, 192));
        // a seed off the part floods nothing
        assert!(Patch::flood(tris, &normals, [0.0, 0.0, 30.0], 30f64.to_radians(), 1.0).is_none());
    }

    #[test]
    fn a_field_flood_follows_a_smooth_surface_and_stops_at_a_crease() {
        // a sphere of 10 cut flat at z = -5 (a sharp rim), hollowed to a 1.5 wall
        let solid = |p: P3| ((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 10.0).max(-5.0 - p[2]);
        let field: Field = Arc::new(move |p| solid(p).max(-(solid(p) + 1.5)));
        let p = Patch::flood_field(field.clone(), [0.0, 0.0, 10.2], 30f64.to_radians(), 1.0, 0.4).unwrap();
        let (n, of) = p.flooded();
        assert!(n > 1000 && n < of, "{n} of {of}");
        // the whole outer dome, not the flat bottom ring nor the inside
        assert!(p.distance([0.0, 0.0, 10.0]) < 0.0 && p.distance([10.0, 0.0, 0.0]) < 0.0 && p.distance([0.0, -8.5, -4.0]) < 0.0);
        assert!(p.distance([7.0, 0.0, -5.0]) > 0.0, "the cut face");
        assert!(p.distance([0.0, 0.0, 8.5]) > 0.0, "the inner wall");
        // a step into the wall from the dome is painted
        assert!(p.distance([0.0, 0.0, 9.5]) < 0.0);
        // too far from the surface: nothing
        assert!(Patch::flood_field(field, [0.0, 0.0, 14.0], 30f64.to_radians(), 1.0, 0.4).is_none());
    }
}
