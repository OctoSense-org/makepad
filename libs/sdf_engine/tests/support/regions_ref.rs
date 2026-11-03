//! An f64 stand-in of the region passes over GPU-EXEC §10.3 records, for tests only. Grid units
//! (0.1 mm = 1.0), fine cells of one unit, ids per plane as plane_abi (node g = gy * NX + gx, cell
//! = its low corner's node, edge = 2 g + axis). A region is closed loops with its inside on the
//! left (contours counter-clockwise, holes clockwise).
//! - [`pieces`]: a loop set's level-0 pieces per fine cell (what extract_cells gives at level 0);
//! - [`Field`]: its exact signed distance (field_at) and the outward normal at the nearest seg;
//! - [`compose`]: the zero contour of a composition, per cell, from the terms' pieces;
//! - [`keys`]: positional endpoint keys (the edge by the half-open rule, the rank along it);
//! - [`chain`]: the composed pieces linked by keys into loops.
//! The compose kernel follows `compose` and `keys` step for step.
#![allow(dead_code)]
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub const CELL_BIT: u32 = 1 << 31;
/// Points closer than this (grid units) are one point; a boundary closer than this is coincident.
pub const TOL: f64 = 1e-5;
/// The ordinal of a node point on its +x edge (interior points of the edge rank from 0).
pub const NODE_ORDINAL: u32 = 0xff_ffff;

pub type P = [f64; 2];
pub type Loops = Vec<Vec<P>>;
pub type Cell = (i64, i64);

fn sub(a: P, b: P) -> P { [a[0] - b[0], a[1] - b[1]] }
fn dot(a: P, b: P) -> f64 { a[0] * b[0] + a[1] * b[1] }
fn cross(a: P, b: P) -> f64 { a[0] * b[1] - a[1] * b[0] }
fn lerp(a: P, b: P, s: f64) -> P { [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s] }
fn near(a: P, b: P) -> bool { (a[0] - b[0]).abs() <= TOL && (a[1] - b[1]).abs() <= TOL }
fn on_line(v: f64) -> bool { v == v.floor() }

/// A straight piece inside one fine cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Piece {
    pub a: P,
    pub b: P,
    pub cell: Cell,
}

/// The cell a piece belongs to: the one holding its midpoint, the cell above / right of a grid
/// line it lies along.
pub fn cell_of(a: P, b: P) -> Cell {
    let m = lerp(a, b, 0.5);
    (m[0].floor() as i64, m[1].floor() as i64)
}

/// The level-0 pieces of `loops` per fine cell: every seg cut where it crosses a grid line,
/// the cut points exact on the line.
pub fn pieces(loops: &Loops) -> BTreeMap<Cell, Vec<Piece>> {
    let mut out: BTreeMap<Cell, Vec<Piece>> = BTreeMap::new();
    for l in loops {
        for i in 0..l.len() {
            let (a, b) = (l[i], l[(i + 1) % l.len()]);
            let mut cuts: Vec<(f64, P)> = vec![(0.0, a), (1.0, b)];
            for axis in 0..2 {
                if a[axis] == b[axis] {
                    continue;
                }
                let (lo, hi) = (a[axis].min(b[axis]), a[axis].max(b[axis]));
                let mut v = lo.floor() + 1.0;
                while v < hi {
                    let s = (v - a[axis]) / (b[axis] - a[axis]);
                    let mut p = lerp(a, b, s);
                    p[axis] = v;
                    // a seg through a node meets both lines there: one cut, exact on both
                    let o = 1 - axis;
                    if (p[o] - p[o].round()).abs() < 1e-9 {
                        p[o] = p[o].round();
                    }
                    cuts.push((s, p));
                    v += 1.0;
                }
            }
            cuts.sort_by(|x, y| x.0.total_cmp(&y.0));
            cuts.dedup_by(|x, y| x.1 == y.1);
            for w in cuts.windows(2) {
                let (p, q) = (w[0].1, w[1].1);
                if p != q {
                    let cell = cell_of(p, q);
                    out.entry(cell).or_default().push(Piece { a: p, b: q, cell });
                }
            }
        }
    }
    out
}

/// A loop set's exact field.
pub struct Field<'a> {
    pub loops: &'a Loops,
}

impl Field<'_> {
    /// The signed distance at `p` (inside < 0, by the nonzero winding) and the outward normal of
    /// the nearest seg (its right side).
    pub fn at(&self, p: P) -> (f64, P) {
        let mut best = (f64::INFINITY, [0.0, 0.0]);
        let mut winding = 0i32;
        for l in self.loops {
            for i in 0..l.len() {
                let (a, b) = (l[i], l[(i + 1) % l.len()]);
                let d = sub(b, a);
                let len2 = dot(d, d);
                if len2 == 0.0 {
                    continue;
                }
                let s = (dot(sub(p, a), d) / len2).clamp(0.0, 1.0);
                let q = lerp(a, b, s);
                let dist = dot(sub(p, q), sub(p, q)).sqrt();
                if dist < best.0 {
                    let len = len2.sqrt();
                    best = (dist, [d[1] / len, -d[0] / len]);
                }
                if a[1] <= p[1] {
                    if b[1] > p[1] && cross(d, sub(p, a)) > 0.0 {
                        winding += 1;
                    }
                } else if b[1] <= p[1] && cross(d, sub(p, a)) < 0.0 {
                    winding -= 1;
                }
            }
        }
        (if winding != 0 { -best.0 } else { best.0 }, best.1)
    }

    pub fn inside(&self, p: P) -> bool {
        self.at(p).0 < 0.0
    }
}

/// A composition of terms: all = intersection, any = union.
#[derive(Clone, Debug)]
pub enum Ex {
    T(usize),
    Not(usize),
    All(Vec<Ex>),
    Any(Vec<Ex>),
}

pub fn eval(e: &Ex, inside: &dyn Fn(usize) -> bool) -> bool {
    match e {
        Ex::T(t) => inside(*t),
        Ex::Not(t) => !inside(*t),
        Ex::All(v) => v.iter().all(|e| eval(e, inside)),
        Ex::Any(v) => v.iter().any(|e| eval(e, inside)),
    }
}

/// A composed piece, its inside on the left, and the term it came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Out {
    pub a: P,
    pub b: P,
    pub cell: Cell,
    pub term: usize,
}

/// A term: its pieces per cell and its field.
pub struct Term<'a> {
    pub pieces: &'a BTreeMap<Cell, Vec<Piece>>,
    pub field: &'a Field<'a>,
}

/// The zero contour of `e` over `terms`, per cell. Each term piece is cut where another term's
/// piece of the cell crosses it, or where a collinear one starts or ends on it. A sub-piece is on
/// the composed boundary when the composition differs across it: on its left side the term is
/// inside, on its right outside; another term is inside or outside by the sign of its field at
/// the sub-piece's midpoint, or, where it coincides (|d| <= TOL), by its own orientation. A
/// stretch where terms coincide is emitted once, by the lowest of them.
pub fn compose(terms: &[Term], e: &Ex) -> BTreeMap<Cell, Vec<Out>> {
    let cells: BTreeSet<Cell> = terms.iter().flat_map(|t| t.pieces.keys().copied()).collect();
    let mut out: BTreeMap<Cell, Vec<Out>> = BTreeMap::new();
    for cell in cells {
        let list: Vec<(usize, Piece)> = terms.iter().enumerate().flat_map(|(t, term)| term.pieces.get(&cell).into_iter().flatten().map(move |p| (t, *p))).collect();
        for &(t, a) in &list {
            let d = sub(a.b, a.a);
            let len = dot(d, d).sqrt();
            let st = TOL / len;
            // cut parameters, with the exact point when the cut is another piece's endpoint
            let mut cuts: Vec<(f64, Option<P>)> = Vec::new();
            for &(t2, b) in &list {
                if t2 == t {
                    continue;
                }
                let e2 = sub(b.b, b.a);
                let len2 = dot(e2, e2).sqrt();
                let w = sub(b.a, a.a);
                let den = cross(d, e2);
                if den.abs() > 1e-12 * len * len2 {
                    let s = cross(w, e2) / den;
                    let u = cross(w, d) / den;
                    let ut = TOL / len2;
                    if s > st && s < 1.0 - st && u > -ut && u < 1.0 + ut {
                        let exact = if u.abs() <= ut { Some(b.a) } else if (u - 1.0).abs() <= ut { Some(b.b) } else { None };
                        cuts.push((s, exact));
                    }
                } else if cross(d, w).abs() / len <= TOL {
                    for q in [b.a, b.b] {
                        let s = dot(sub(q, a.a), d) / (len * len);
                        if s > st && s < 1.0 - st {
                            cuts.push((s, Some(q)));
                        }
                    }
                }
            }
            cuts.sort_by(|x, y| x.0.total_cmp(&y.0));
            cuts.dedup_by(|x, y| (x.0 - y.0) * len <= TOL);
            let mut points: Vec<P> = vec![a.a];
            points.extend(cuts.iter().map(|&(s, q)| q.unwrap_or_else(|| lerp(a.a, a.b, s))));
            points.push(a.b);
            let normal = [d[1] / len, -d[0] / len];
            for w in points.windows(2) {
                let (p0, p1) = (w[0], w[1]);
                let m = lerp(p0, p1, 0.5);
                let mut left = vec![false; terms.len()];
                let mut right = vec![false; terms.len()];
                let mut shadowed = false;
                for (t3, term) in terms.iter().enumerate() {
                    if t3 == t {
                        left[t3] = true;
                        continue;
                    }
                    let (v, n) = term.field.at(m);
                    if v.abs() > TOL {
                        left[t3] = v < 0.0;
                        right[t3] = v < 0.0;
                    } else {
                        let same = dot(n, normal) > 0.0;
                        left[t3] = same;
                        right[t3] = !same;
                        shadowed |= t3 < t;
                    }
                }
                let (l, r) = (eval(e, &|i| left[i]), eval(e, &|i| right[i]));
                if l == r || shadowed {
                    continue;
                }
                let (pa, pb) = if l { (p0, p1) } else { (p1, p0) };
                out.entry(cell).or_default().push(Out { a: pa, b: pb, cell, term: t });
            }
        }
    }
    out
}

/// An endpoint key: hi = an edge id, or CELL_BIT | cell; lo = level << 24 | ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Key {
    pub hi: u32,
    pub lo: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Place {
    Node(u32),
    /// An edge's interior: the edge id.
    Edge(u32),
    Cell(u32),
}

fn place(p: P, cell: Cell, nx: i64) -> (Place, f64) {
    let node = |x: i64, y: i64| (y * nx + x) as u32;
    match (on_line(p[0]), on_line(p[1])) {
        (true, true) => (Place::Node(2 * node(p[0] as i64, p[1] as i64)), 0.0),
        (true, false) => (Place::Edge(2 * node(p[0] as i64, p[1].floor() as i64) + 1), p[1]),
        (false, true) => (Place::Edge(2 * node(p[0].floor() as i64, p[1] as i64)), p[0]),
        (false, false) => (Place::Cell(node(cell.0, cell.1)), p[0]),
    }
}

/// Every composed piece's (from, to) keys at `level`: a point on a grid line takes its edge by the
/// half-open rule (a node its +x edge), anything else its cell; the ordinal is the point's rank
/// among the distinct composed endpoints of that edge (along it) or cell (by x, then y).
pub fn keys(out: &BTreeMap<Cell, Vec<Out>>, level: u32, nx: i64) -> Vec<(Out, Key, Key)> {
    let mut at: BTreeMap<Place, Vec<P>> = BTreeMap::new();
    for o in out.values().flatten() {
        for p in [o.a, o.b] {
            at.entry(place(p, o.cell, nx).0).or_default().push(p);
        }
    }
    let rank = |pl: Place, p: P| -> u32 {
        if let Place::Node(_) = pl {
            return NODE_ORDINAL;
        }
        let mut pts = at[&pl].clone();
        pts.sort_by(|x, y| x[0].total_cmp(&y[0]).then(x[1].total_cmp(&y[1])));
        pts.dedup_by(|x, y| near(*x, *y));
        pts.iter().position(|q| near(*q, p)).expect("its own point") as u32
    };
    let key = |p: P, cell: Cell| -> Key {
        let (pl, _) = place(p, cell, nx);
        let hi = match pl {
            Place::Node(e) | Place::Edge(e) => e,
            Place::Cell(c) => CELL_BIT | c,
        };
        Key { hi, lo: level << 24 | rank(pl, p) }
    };
    out.values().flatten().map(|o| (*o, key(o.a, o.cell), key(o.b, o.cell))).collect()
}

/// The composed pieces linked from key to key into closed loops (their points, the inside on the
/// left). Panics when a key does not link exactly one piece to exactly one other.
pub fn chain(keyed: &[(Out, Key, Key)]) -> Loops {
    let mut from: HashMap<Key, usize> = HashMap::new();
    for (i, (_, k, _)) in keyed.iter().enumerate() {
        assert!(from.insert(*k, i).is_none(), "two pieces start at {k:?}: {:?}", keyed[i].0);
    }
    let mut tos: BTreeSet<Key> = BTreeSet::new();
    for (o, _, k) in keyed {
        assert!(tos.insert(*k), "two pieces end at {k:?}: {o:?}");
        assert!(from.contains_key(k), "nothing continues from {k:?}: {o:?}");
    }
    let mut seen = vec![false; keyed.len()];
    let mut loops = Vec::new();
    for start in 0..keyed.len() {
        if seen[start] {
            continue;
        }
        let mut l = Vec::new();
        let mut i = start;
        while !seen[i] {
            seen[i] = true;
            l.push(keyed[i].0.a);
            i = from[&keyed[i].2];
        }
        assert_eq!(i, start, "a chain closes on its start");
        loops.push(l);
    }
    loops
}

/// The signed area of loops (positive for counter-clockwise contours).
pub fn area(loops: &Loops) -> f64 {
    loops.iter().map(|l| (0..l.len()).map(|i| cross(l[i], l[(i + 1) % l.len()])).sum::<f64>() / 2.0).sum()
}
