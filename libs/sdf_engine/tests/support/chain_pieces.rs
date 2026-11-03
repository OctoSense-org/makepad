//! Synthetic wall pieces (GPU-EXEC §10.3) for the chain and rings tests. Loops of lines and arcs
//! are clipped to fine cells in f64, the way WALLBUILD emits them. Crossings get edge keys with
//! ordinals along the edge in +axis order, and joints get CELL_BIT keys. The pieces are grouped
//! per band fine cell and level slot by `piece_off`. `split` adds an in-cell junction inside
//! every piece and reverses the id order within cells: the same curves, split another way.
#![allow(dead_code)]
use makepad_sdf_engine::plane_abi::{cell_id, edge_id, key_lo, CELL_BIT, NONE};
use std::collections::BTreeMap;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

#[derive(Clone, Copy, Debug)]
pub enum Feature {
    Line { a: [f64; 2], b: [f64; 2] },
    /// From angle a0 through `sweep` (signed, radians).
    Arc { c: [f64; 2], r: f64, a0: f64, sweep: f64 },
}
impl Feature {
    pub fn at(&self, s: f64) -> [f64; 2] {
        match *self {
            Feature::Line { a, b } => [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s],
            Feature::Arc { c, r, a0, sweep } => {
                let t = a0 + sweep * s;
                [c[0] + r * t.cos(), c[1] + r * t.sin()]
            }
        }
    }
    fn reversed(&self) -> Feature {
        match *self {
            Feature::Line { a, b } => Feature::Line { a: b, b: a },
            Feature::Arc { c, r, a0, sweep } => Feature::Arc { c, r, a0: a0 + sweep, sweep: -sweep },
        }
    }
    /// Where the feature crosses grid lines inside (0, 1): (s, line), line = (axis, n) with
    /// axis 0 for x = n and 1 for y = n, ascending in s.
    fn crossings(&self) -> Vec<(f64, (u32, i64))> {
        let mut out = Vec::new();
        match *self {
            Feature::Line { a, b } => {
                for axis in 0..2 {
                    let (lo, hi) = (a[axis].min(b[axis]), a[axis].max(b[axis]));
                    for n in (lo.floor() as i64 + 1)..=(hi.ceil() as i64 - 1) {
                        let s = (n as f64 - a[axis]) / (b[axis] - a[axis]);
                        if s > 0.0 && s < 1.0 { out.push((s, (axis as u32, n))); }
                    }
                }
            }
            Feature::Arc { c, r, a0, sweep } => {
                for axis in 0..2 {
                    for n in ((c[axis] - r).floor() as i64 + 1)..=((c[axis] + r).ceil() as i64 - 1) {
                        let u = (n as f64 - c[axis]) / r;
                        if u.abs() >= 1.0 { continue; }
                        let bases = if axis == 0 { [u.acos(), -u.acos()] } else { [u.asin(), PI - u.asin()] };
                        for base in bases {
                            // every turn of this angle inside the sweep
                            let (from, to) = if sweep > 0.0 { (a0, a0 + sweep) } else { (a0 + sweep, a0) };
                            let mut t = base + ((from - base) / TAU).floor() * TAU;
                            while t < to {
                                if t > from {
                                    let s = (t - a0) / sweep;
                                    if s > 0.0 && s < 1.0 { out.push((s, (axis as u32, n))); }
                                }
                                t += TAU;
                            }
                        }
                    }
                }
            }
        }
        out.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        for w in out.windows(2) { assert!(w[1].0 - w[0].0 > 1e-9, "a feature crosses a grid node: {self:?}"); }
        out
    }
}

/// A loop or open chain, with the containment the test expects (indices into the scene's specs).
#[derive(Clone, Debug)]
pub struct LoopSpec {
    pub plane: u32,
    pub level: u32,
    pub features: Vec<Feature>,
    pub closed: bool,
    pub hole: bool,
    pub parent: Option<usize>,
    pub island: Option<usize>,
    pub base: Option<usize>,
}

/// A rounded rectangle, counter-clockwise (or clockwise), from its bottom edge.
pub fn rounded_rect(x0: f64, y0: f64, x1: f64, y1: f64, r: f64, ccw: bool) -> Vec<Feature> {
    let mut f = Vec::new();
    let corner = |f: &mut Vec<Feature>, c: [f64; 2], a0: f64| if r > 0.0 { f.push(Feature::Arc { c, r, a0, sweep: FRAC_PI_2 }) };
    f.push(Feature::Line { a: [x0 + r, y0], b: [x1 - r, y0] });
    corner(&mut f, [x1 - r, y0 + r], -FRAC_PI_2);
    f.push(Feature::Line { a: [x1, y0 + r], b: [x1, y1 - r] });
    corner(&mut f, [x1 - r, y1 - r], 0.0);
    f.push(Feature::Line { a: [x1 - r, y1], b: [x0 + r, y1] });
    corner(&mut f, [x0 + r, y1 - r], FRAC_PI_2);
    f.push(Feature::Line { a: [x0, y1 - r], b: [x0, y0 + r] });
    corner(&mut f, [x0 + r, y0 + r], PI);
    if ccw { f } else { f.iter().rev().map(Feature::reversed).collect() }
}

/// A whole circle from angle a0, counter-clockwise (or clockwise).
pub fn circle(c: [f64; 2], r: f64, a0: f64, ccw: bool) -> Vec<Feature> {
    vec![Feature::Arc { c, r, a0, sweep: if ccw { TAU } else { -TAU } }]
}

/// A closed polygon through `pts`.
pub fn poly(pts: &[[f64; 2]]) -> Vec<Feature> {
    (0..pts.len()).map(|i| Feature::Line { a: pts[i], b: pts[(i + 1) % pts.len()] }).collect()
}

#[derive(Clone, Copy, Debug)]
enum Kind { Edge(u32, f64), Cell(u32) }

/// A generated piece, with where it sits in its loop.
#[derive(Clone, Debug)]
pub struct GenPiece {
    pub p: [f32; 4],
    pub geo: [f32; 4],
    pub key: [u32; 4],
    pub info: [u32; 4],
    pub spec: usize,
    pub seq: usize,
}

pub struct Scene {
    pub nx: u32,
    pub ny: u32,
    pub planes: u32,
    /// The level of each level slot.
    pub levels: Vec<u32>,
    /// In id order.
    pub pieces: Vec<GenPiece>,
    pub piece_off: Vec<u32>,
    pub band_slot: Vec<u32>,
    pub specs: Vec<LoopSpec>,
    /// Per spec: its pieces' ids in chain order.
    pub chains: Vec<Vec<u32>>,
}
impl Scene {
    pub fn words<const N: usize>(rows: impl Iterator<Item = [u32; N]>) -> Vec<u32> { rows.flatten().collect() }
    pub fn piece_words(&self) -> Vec<u32> { Self::words(self.pieces.iter().map(|p| p.p.map(f32::to_bits))) }
    pub fn geo_words(&self) -> Vec<u32> { Self::words(self.pieces.iter().map(|p| p.geo.map(f32::to_bits))) }
    pub fn key_words(&self) -> Vec<u32> { Self::words(self.pieces.iter().map(|p| p.key)) }
    pub fn info_words(&self) -> Vec<u32> { Self::words(self.pieces.iter().map(|p| p.info)) }
    /// The spec of each GPU loop id: the chain numbers loops by their roots (a cycle's least
    /// piece id, an open chain's head) in id order.
    pub fn loop_ids(&self) -> Vec<usize> {
        let mut by_root: Vec<(u32, usize)> = self.chains.iter().enumerate().map(|(s, chain)| {
            (if self.specs[s].closed { *chain.iter().min().unwrap() } else { chain[0] }, s)
        }).collect();
        by_root.sort();
        by_root.into_iter().map(|(_, s)| s).collect()
    }
    /// The successor of each piece (NONE at an open tail).
    pub fn succ(&self) -> Vec<u32> {
        let mut succ = vec![NONE; self.pieces.len()];
        for (spec, chain) in self.specs.iter().zip(&self.chains) {
            for (i, &k) in chain.iter().enumerate() {
                succ[k as usize] = if i + 1 < chain.len() { chain[i + 1] } else if spec.closed { chain[0] } else { NONE };
            }
        }
        succ
    }
}

/// The scene of `specs` on an nx x ny coarse grid of `planes` planes, every coarse cell in the band.
pub fn scene(nx: u32, ny: u32, planes: u32, specs: Vec<LoopSpec>, split: bool) -> Scene {
    let mut levels: Vec<u32> = specs.iter().map(|s| s.level).collect();
    levels.sort();
    levels.dedup();
    let cells_x = nx as i64 * 10;
    struct Raw { p: [f32; 4], geo: [f32; 4], from: Kind, to: Kind, plane: u32, level: u32, feature: u32, cell: u32, spec: usize, seq: usize }
    let mut raw: Vec<Raw> = Vec::new();
    let mut feature_base = 0u32;
    for (si, spec) in specs.iter().enumerate() {
        // spans (feature, s0, s1) and the kind of each span's start
        let mut spans: Vec<(usize, f64, f64, Option<(u32, i64)>)> = Vec::new();
        if spec.closed && spec.features.len() == 1 {
            let c = spec.features[0].crossings();
            assert!(c.len() >= 2, "a whole circle inside one cell is not emitted");
            for j in 0..c.len() {
                let s1 = if j + 1 < c.len() { c[j + 1].0 } else { c[0].0 + 1.0 };
                spans.push((0, c[j].0, s1, Some(c[j].1)));
            }
        } else {
            for (fi, f) in spec.features.iter().enumerate() {
                let mut s = vec![(0.0, None)];
                s.extend(f.crossings().into_iter().map(|(s, l)| (s, Some(l))));
                s.push((1.0, None));
                for w in s.windows(2) { spans.push((fi, w[0].0, w[1].0, w[0].1)); }
            }
        }
        if split {
            spans = spans.into_iter().flat_map(|(f, s0, s1, l)| {
                let sm = s0 + 0.43 * (s1 - s0);
                [(f, s0, sm, l), (f, sm, s1, None)]
            }).collect();
        }
        let kind = |p: [f64; 2], line: Option<(u32, i64)>| -> Kind {
            match line {
                Some((0, n)) => Kind::Edge(edge_id(n as u32, p[1].floor() as u32, 1, nx), p[1]),
                Some((_, n)) => Kind::Edge(edge_id(p[0].floor() as u32, n as u32, 0, nx), p[0]),
                None => Kind::Cell(cell_id(p[0].floor() as u32, p[1].floor() as u32, nx)),
            }
        };
        let f32p = |p: [f64; 2]| [p[0] as f32, p[1] as f32];
        let starts: Vec<([f64; 2], Kind)> = spans.iter().map(|&(f, s0, _, l)| {
            let p = spec.features[f].at(s0);
            (p, kind(p, l))
        }).collect();
        let n = spans.len();
        for (j, &(f, s0, s1, _)) in spans.iter().enumerate() {
            let feature = &spec.features[f];
            let (a, from) = starts[j];
            let (b, to) = if j + 1 < n { starts[j + 1] } else if spec.closed { starts[0] } else {
                let p = feature.at(1.0);
                (p, kind(p, None))
            };
            let mid = feature.at(0.5 * (s0 + s1));
            let (fx, fy) = (mid[0].floor() as i64, mid[1].floor() as i64);
            assert!(fx >= 0 && fy >= 0 && fx < cells_x && fy < ny as i64 * 10, "a piece off the grid");
            let geo = match *feature {
                Feature::Line { .. } => [0.0; 4],
                Feature::Arc { c, r, sweep, .. } => [c[0] as f32, c[1] as f32, r as f32, ((s1 - s0) * sweep) as f32],
            };
            let (a, b) = (f32p(a), f32p(b));
            raw.push(Raw { p: [a[0], a[1], b[0], b[1]], geo, from, to, plane: spec.plane, level: spec.level,
                feature: feature_base + f as u32, cell: cell_id(fx as u32, fy as u32, nx), spec: si, seq: j });
        }
        feature_base += spec.features.len() as u32;
    }
    // ordinals: along each edge in +axis order, and in generation order inside a cell
    let mut on_edge: BTreeMap<(u32, u32, u32), Vec<f64>> = BTreeMap::new();
    let mut in_cell: BTreeMap<(u32, u32, u32), u32> = BTreeMap::new();
    let mut cell_ordinal: BTreeMap<(usize, usize), u32> = BTreeMap::new();
    for (i, r) in raw.iter().enumerate() {
        match r.from {
            Kind::Edge(e, pos) => on_edge.entry((r.plane, r.level, e)).or_default().push(pos),
            Kind::Cell(c) => {
                let next = in_cell.entry((r.plane, r.level, c)).or_default();
                cell_ordinal.insert((i, 0), *next);
                *next += 1;
            }
        }
    }
    for v in on_edge.values_mut() { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); }
    // an open chain's end junction
    for (i, r) in raw.iter().enumerate() {
        if let Kind::Cell(c) = r.to {
            let is_end = !specs[r.spec].closed && r.seq + 1 == raw.iter().filter(|q| q.spec == r.spec).count();
            if is_end {
                let next = in_cell.entry((r.plane, r.level, c)).or_default();
                cell_ordinal.insert((i, 1), *next);
                *next += 1;
            }
        }
    }
    let key = |i: usize, which: usize| -> [u32; 2] {
        let r = &raw[i];
        let k = if which == 0 { r.from } else { r.to };
        match k {
            Kind::Edge(e, pos) => {
                let at = on_edge[&(r.plane, r.level, e)].iter().position(|&q| q == pos).unwrap() as u32;
                [e, key_lo(r.level, at)]
            }
            Kind::Cell(c) => {
                // a junction's ordinal belongs to the piece that starts there
                let ord = if which == 0 { cell_ordinal[&(i, 0)] } else if let Some(&o) = cell_ordinal.get(&(i, 1)) { o } else {
                    let next = raw.iter().position(|q| q.spec == r.spec && q.seq == (r.seq + 1) % raw.iter().filter(|q| q.spec == r.spec).count()).unwrap();
                    cell_ordinal[&(next, 0)]
                };
                [CELL_BIT | c, key_lo(r.level, ord)]
            }
        }
    };
    let keys: Vec<[u32; 4]> = (0..raw.len()).map(|i| { let (f, t) = (key(i, 0), key(i, 1)); [f[0], f[1], t[0], t[1]] }).collect();
    // ids: by (band fine cell, level slot), then generation order (reversed when split)
    let stride = nx * 10 + 1;
    let level_slot = |l: u32| levels.iter().position(|&x| x == l).unwrap() as u32;
    let cell_index = |plane: u32, cell: u32| -> u32 {
        let (fx, fy) = (cell % stride, cell / stride);
        let slot = plane * nx * ny + (fy / 10) * nx + fx / 10;
        slot * 100 + (fy % 10) * 10 + fx % 10
    };
    let nlev = levels.len() as u32;
    let mut order: Vec<usize> = (0..raw.len()).collect();
    order.sort_by_key(|&i| {
        let r = &raw[i];
        let group = cell_index(r.plane, r.cell) * nlev + level_slot(r.level);
        (group, if split { usize::MAX - i } else { i })
    });
    let mut id_of = vec![0u32; raw.len()];
    for (id, &i) in order.iter().enumerate() { id_of[i] = id as u32; }
    let slots = planes * nx * ny;
    let mut count = vec![0u32; (slots * 100 * nlev) as usize + 1];
    let pieces: Vec<GenPiece> = order.iter().map(|&i| {
        let r = &raw[i];
        count[(cell_index(r.plane, r.cell) * nlev + level_slot(r.level)) as usize] += 1;
        GenPiece { p: r.p, geo: r.geo, key: keys[i], info: [r.plane, r.level, r.feature, r.cell], spec: r.spec, seq: r.seq }
    }).collect();
    let mut piece_off = Vec::with_capacity(count.len());
    let mut sum = 0;
    for c in count { piece_off.push(sum); sum += c; }
    let band_slot = (0..slots).collect();
    let mut chains: Vec<Vec<u32>> = vec![Vec::new(); specs.len()];
    for (i, r) in raw.iter().enumerate() { chains[r.spec].push(id_of[i]); }
    Scene { nx, ny, planes, levels, pieces, piece_off, band_slot, specs, chains }
}
