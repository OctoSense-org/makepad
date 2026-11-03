//! Closed loops for FeatureSet tests, in grid units: contours CCW, holes CW.
#![allow(dead_code)]

#[derive(Clone, Debug)]
pub struct Ring {
    pub points: Vec<[f64; 2]>,
    pub creases: Vec<bool>,
    pub hole: bool,
}

pub fn area(points: &[[f64; 2]]) -> f64 {
    (0..points.len()).map(|i| { let (a, b) = (points[i], points[(i + 1) % points.len()]); a[0] * b[1] - a[1] * b[0] }).sum::<f64>() * 0.5
}

/// Every corner a crease; a hole when clockwise.
pub fn polygon(points: &[[f64; 2]]) -> Ring {
    Ring { points: points.to_vec(), creases: vec![true; points.len()], hole: area(points) < 0.0 }
}

pub fn circle(c: [f64; 2], r: f64, n: usize, hole: bool) -> Ring {
    let mut points: Vec<[f64; 2]> = (0..n).map(|i| {
        let a = i as f64 / n as f64 * std::f64::consts::TAU + 0.1234;
        [c[0] + r * a.cos(), c[1] + r * a.sin()]
    }).collect();
    if hole { points.reverse(); }
    Ring { creases: vec![false; n], points, hole }
}

pub fn rotated_square(c: [f64; 2], h: f64, deg: f64) -> Ring {
    let (s, co) = deg.to_radians().sin_cos();
    polygon(&[[-h, -h], [h, -h], [h, h], [-h, h]].map(|[x, y]| [c[0] + co * x - s * y, c[1] + s * x + co * y]))
}
