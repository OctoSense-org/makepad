//! The f64 mirror of feature_field's definitions (GPU-EXEC §10.3): the nearest-feature
//! value, side and biased signed distance d' of any FeatureSet in either join metric, for the
//! field's tests and the stages that read it (S3, S1b, chains). The including module brings
//! FeatureSet, FieldParams, FieldJoin, BIG, DOMAIN_SLACK, SEG_REFLEX and VERTEX_BIT into scope.
#![allow(dead_code)]
use super::{FeatureSet, FieldJoin, FieldParams, BIG, DOMAIN_SLACK, SEG_REFLEX, VERTEX_BIT};
use std::collections::HashMap;

pub struct Mirror<'a> {
    set: &'a FeatureSet,
    params: FieldParams,
    a: Vec<[f64; 2]>,
    e: Vec<[f64; 2]>,
    len: Vec<f64>,
    bias: Vec<f64>,
    cos: Vec<f64>,
    flags: Vec<u32>,
    plane: Vec<u32>,
    /// Features whose region at the field's depth reaches each coarse cell, and the segs
    /// of each fine row, per plane.
    bucket: HashMap<(u32, u32, u32), Vec<u32>>,
    rows: HashMap<(u32, u32), Vec<usize>>,
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 { a[0] * b[0] + a[1] * b[1] }
fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] { [a[0] - b[0], a[1] - b[1]] }
fn left(e: [f64; 2]) -> [f64; 2] { [-e[1], e[0]] }
fn neg(e: [f64; 2]) -> [f64; 2] { [-e[0], -e[1]] }
fn norm(v: [f64; 2]) -> [f64; 2] { let l = dot(v, v).sqrt(); if l > 0.0 { [v[0] / l, v[1] / l] } else { [1.0, 0.0] } }

impl<'a> Mirror<'a> {
    /// `loop_bias` per loop of the set, in loop_info order.
    pub fn new(set: &'a FeatureSet, loop_bias: &[f32], params: FieldParams) -> Self {
        let n = set.seg.len();
        let mut m = Mirror { set, params, a: vec![], e: vec![], len: vec![], bias: vec![], cos: vec![], flags: vec![], plane: vec![],
            bucket: HashMap::new(), rows: HashMap::new() };
        for s in 0..n {
            let q = set.seg[s];
            m.a.push([q[0] as f64, q[1] as f64]);
            let d = [q[2] as f64 - q[0] as f64, q[3] as f64 - q[1] as f64];
            m.len.push(dot(d, d).sqrt());
            m.e.push(norm(d));
            let root = set.seg_info[s][0];
            let l = set.loop_info.iter().position(|li| li[0] == root).unwrap();
            m.bias.push(loop_bias[l] as f64);
            m.plane.push(set.loop_info[l][2]);
        }
        for s in 0..n {
            let p = set.seg_info[s][2] as usize;
            let (e1, e) = (m.e[p], m.e[s]);
            let turn = e1[0] * e[1] - e1[1] * e[0];
            let cos = dot(e1, e);
            let mut f = if set.seg_info[s][3] & SEG_REFLEX != 0 { 1 } else if turn > 0.0 { 2 } else { 0 };
            if let FieldJoin::Miter(limit) = params.join {
                if f != 0 && 1.0 + cos <= 2.0 / (limit as f64 * limit as f64) { f |= 4; }
            }
            if m.len[s] > 0.0 { f |= 8; }
            m.cos.push(cos);
            m.flags.push(f);
        }
        for s in 0..n {
            let q = set.seg[s];
            m.rows.entry((m.plane[s], (q[1].min(q[3]) as f64).floor() as u32)).or_default().push(s);
        }
        let din = (params.depth_in + params.bias_max) as f64 + 0.01;
        let dout = (params.depth_out + params.bias_max) as f64 + 0.01;
        let features: Vec<u32> = m.features().collect();
        for f in features {
            let s = (f & !VERTEX_BIT) as usize;
            let (lo, hi) = if f & VERTEX_BIT != 0 {
                let d = if m.vertex_inside(s) { din } else { dout };
                let r = match params.join {
                    FieldJoin::Round => d,
                    FieldJoin::Miter(limit) => d / (0.5 * (1.0 + m.cos[s])).max(0.0).sqrt().max(1.0 / limit as f64),
                } + 0.01;
                ([m.a[s][0] - r, m.a[s][1] - r], [m.a[s][0] + r, m.a[s][1] + r])
            } else {
                let (e, n) = (m.e[s], left(m.e[s]));
                let mut lo = [f64::INFINITY; 2];
                let mut hi = [f64::NEG_INFINITY; 2];
                for t in [-0.01, m.len[s] + 0.01] {
                    for h in [-dout, din] {
                        for k in 0..2 {
                            let v = m.a[s][k] + t * e[k] + h * n[k];
                            lo[k] = lo[k].min(v);
                            hi[k] = hi[k].max(v);
                        }
                    }
                }
                (lo, hi)
            };
            let clamp = |v: f64, n: u32| (v / 10.0).floor().clamp(0.0, n as f64 - 1.0) as u32;
            for cy in clamp(lo[1], set.ny)..=clamp(hi[1], set.ny) {
                for cx in clamp(lo[0], set.nx)..=clamp(hi[0], set.nx) {
                    m.bucket.entry((m.plane[s], cx, cy)).or_default().push(f);
                }
            }
        }
        m
    }
    fn miter(&self) -> bool { matches!(self.params.join, FieldJoin::Miter(_)) }
    fn vertex_inside(&self, s: usize) -> bool { self.flags[s] & 1 != 0 }
    fn ms(&self, s: usize) -> ([f64; 2], [f64; 2]) {
        let p = self.set.seg_info[s][2] as usize;
        let (m1, m2) = (left(self.e[p]), left(self.e[s]));
        if self.vertex_inside(s) { (m1, m2) } else { (neg(m1), neg(m2)) }
    }
    fn vertex_metric(&self, s: usize, d: [f64; 2]) -> f64 {
        if !self.miter() { return dot(d, d).sqrt(); }
        let (m1, m2) = self.ms(s);
        let mut m = dot(d, m1).max(dot(d, m2));
        if self.flags[s] & 4 != 0 {
            let p = self.set.seg_info[s][2] as usize;
            m = m.max(dot(d, norm(sub(self.e[p], self.e[s]))));
        }
        m
    }
    /// A feature's value at p on a side, as the kernel's `value`.
    pub fn value(&self, f: u32, p: [f64; 2], inside: bool) -> f64 {
        let s = (f & !VERTEX_BIT) as usize;
        let d = sub(p, self.a[s]);
        let tau = DOMAIN_SLACK as f64;
        let b = if inside { self.bias[s] } else { -self.bias[s] };
        if f & VERTEX_BIT != 0 {
            if self.flags[s] & 3 == 0 || self.vertex_inside(s) != inside { return f64::INFINITY; }
            let pe = self.e[self.set.seg_info[s][2] as usize];
            if dot(d, pe) < -tau || dot(d, self.e[s]) > tau { return f64::INFINITY; }
            return self.vertex_metric(s, d).max(0.0) + b;
        }
        if self.flags[s] & 8 == 0 { return f64::INFINITY; }
        let t = dot(d, self.e[s]);
        if t < -tau || t > self.len[s] + tau { return f64::INFINITY; }
        let h = dot(d, left(self.e[s]));
        if inside { if h < -tau { f64::INFINITY } else { h.max(0.0) + b } }
        else if h > tau { f64::INFINITY } else { (-h).max(0.0) + b }
    }
    pub fn features(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.a.len() as u32).chain((0..self.a.len() as u32).filter(|&s| self.flags[s as usize] & 3 != 0).map(|s| s | VERTEX_BIT))
    }
    /// Inside by even-odd parity with vertices on the row counted below it and crossings at
    /// x <= p.x counted, i.e. at (x + ε, y + ε²) as the kernels perturb.
    pub fn inside(&self, plane: u32, p: [f64; 2]) -> bool {
        let mut parity = false;
        // only segs of the fine row holding p can cross its row (vertices on it count below)
        let Some(list) = self.rows.get(&(plane, p[1].floor() as u32)) else { return false };
        for &s in list {
            let q = self.set.seg[s];
            let (a, b) = ([q[0] as f64, q[1] as f64], [q[2] as f64, q[3] as f64]);
            if (a[1] > p[1]) != (b[1] > p[1]) {
                let x = if a[1] == p[1] { a[0] } else if b[1] == p[1] { b[0] } else { a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]) };
                if x <= p[0] { parity = !parity; }
            }
        }
        parity
    }
    /// The biased signed distance and its minimal features at p (BIG where none reaches).
    pub fn field(&self, plane: u32, p: [f64; 2]) -> (f64, bool) {
        let inside = self.inside(plane, p);
        let mut best = f64::INFINITY;
        let cell = ((p[0] / 10.0).floor().clamp(0.0, self.set.nx as f64 - 1.0) as u32, (p[1] / 10.0).floor().clamp(0.0, self.set.ny as f64 - 1.0) as u32);
        for &f in self.bucket.get(&(plane, cell.0, cell.1)).map(|v| v.as_slice()).unwrap_or(&[]) {
            best = best.min(self.value(f, p, inside));
        }
        if best == f64::INFINITY { best = BIG as f64; }
        (if inside { -best } else { best }, inside)
    }
}
