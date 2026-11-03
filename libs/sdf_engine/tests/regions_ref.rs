//! The f64 region stand-in (support/regions_ref.rs) on its own: compositions chain into closed
//! loops whose inside is the composition's, also where terms share edges, coincide, lie on grid
//! lines or meet at nodes, and when a composition reads regions made by earlier ones.
#[path = "support/regions_ref.rs"]
mod regions_ref;
use regions_ref::*;

const NX: i64 = 64;

fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<P> {
    vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
}

fn hole(mut l: Vec<P>) -> Vec<P> {
    l.reverse();
    l
}

/// A star-shaped polygon: `n` vertices at angles 2πi/n around `c`, radii from a seeded sequence.
fn star(seed: &mut u64, c: P, r: (f64, f64), n: usize) -> Vec<P> {
    (0..n).map(|i| {
        let a = std::f64::consts::TAU * i as f64 / n as f64;
        let k = r.0 + (r.1 - r.0) * next(seed);
        [c[0] + k * a.cos(), c[1] + k * a.sin()]
    }).collect()
}

fn next(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64
}

/// Composes `e` over `sets`, chains it, and checks the loops' inside against the composition at
/// a grid of probes clear of every boundary. Returns the loops.
fn run(sets: &[Loops], e: &Ex) -> Loops {
    let fields: Vec<Field> = sets.iter().map(|l| Field { loops: l }).collect();
    let pieces: Vec<_> = sets.iter().map(pieces).collect();
    let terms: Vec<Term> = pieces.iter().zip(&fields).map(|(p, f)| Term { pieces: p, field: f }).collect();
    let out = compose(&terms, e);
    let loops = chain(&keys(&out, 64, NX));
    let result = Field { loops: &loops };
    let mut probes = 0;
    for i in 0..97 {
        for j in 0..97 {
            let p = [0.5 + i as f64 * 0.13 + 0.0011 * j as f64, 0.5 + j as f64 * 0.13 + 0.0007 * i as f64];
            if fields.iter().any(|f| f.at(p).0.abs() < 1e-3) {
                continue;
            }
            probes += 1;
            let want = eval(e, &|t| fields[t].inside(p));
            assert_eq!(result.inside(p), want, "inside at {p:?}");
        }
    }
    assert!(probes > 5000);
    loops
}

fn ops() -> Vec<(&'static str, Ex)> {
    vec![
        ("and", Ex::All(vec![Ex::T(0), Ex::T(1)])),
        ("or", Ex::Any(vec![Ex::T(0), Ex::T(1)])),
        ("minus", Ex::All(vec![Ex::T(0), Ex::Not(1)])),
        ("minus back", Ex::All(vec![Ex::T(1), Ex::Not(0)])),
    ]
}

#[test]
fn overlapping_squares_off_and_on_the_grid() {
    for (a, b) in [(rect(1.3, 1.3, 6.7, 6.7), rect(3.2, 2.4, 9.1, 5.5)), (rect(2.0, 2.0, 8.0, 8.0), rect(5.0, 4.0, 11.0, 6.0))] {
        let (aa, ab) = (area(&vec![a.clone()]), area(&vec![b.clone()]));
        let both = {
            let (x0, x1) = (a[0][0].max(b[0][0]), a[2][0].min(b[2][0]));
            let (y0, y1) = (a[0][1].max(b[0][1]), a[2][1].min(b[2][1]));
            (x1 - x0) * (y1 - y0)
        };
        let sets = vec![vec![a], vec![b]];
        for ((name, e), want) in ops().into_iter().zip([both, aa + ab - both, aa - both, ab - both]) {
            let loops = run(&sets, &e);
            assert!((area(&loops) - want).abs() < 1e-9, "{name}: area {} for {want}", area(&loops));
            assert_eq!(loops.len(), 1, "{name}: one loop");
        }
    }
}

#[test]
fn shared_edges_merge_and_identical_sets_compose_once() {
    for off in [0.0, 0.45] {
        // touching from both sides of x = 6 (+ off): the union has no inner edge
        let a = rect(2.0 + off, 2.0, 6.0 + off, 6.0);
        let b = rect(6.0 + off, 3.0, 9.0 + off, 5.0);
        let loops = run(&[vec![a.clone()], vec![b.clone()]], &Ex::Any(vec![Ex::T(0), Ex::T(1)]));
        assert_eq!(loops.len(), 1);
        assert!((area(&loops) - 22.0).abs() < 1e-9);
        // inside along one edge: the difference is A with a notch
        let c = rect(2.0 + off, 3.0, 5.0 + off, 5.0);
        let loops = run(&[vec![a.clone()], vec![c.clone()]], &Ex::All(vec![Ex::T(0), Ex::Not(1)]));
        assert_eq!(loops.len(), 1);
        assert!((area(&loops) - 10.0).abs() < 1e-9);
        let loops = run(&[vec![a.clone()], vec![c]], &Ex::All(vec![Ex::T(0), Ex::T(1)]));
        assert!((area(&loops) - 6.0).abs() < 1e-9);
        // the same set twice
        for (e, want) in [(Ex::All(vec![Ex::T(0), Ex::T(1)]), 16.0), (Ex::Any(vec![Ex::T(0), Ex::T(1)]), 16.0), (Ex::All(vec![Ex::T(0), Ex::Not(1)]), 0.0)] {
            let loops = run(&[vec![a.clone()], vec![a.clone()]], &e);
            assert!((area(&loops) - want).abs() < 1e-9, "{e:?}");
        }
    }
}

#[test]
fn holes_and_corners_on_nodes() {
    let ring = vec![rect(2.0, 2.0, 12.0, 12.0), hole(rect(5.0, 5.0, 9.0, 9.0))];
    let bar = vec![rect(7.0, 1.0, 9.0, 13.0)];
    let loops = run(&[ring.clone(), bar.clone()], &Ex::All(vec![Ex::T(0), Ex::Not(1)]));
    assert!((area(&loops) - 72.0).abs() < 1e-9);
    let loops = run(&[ring, bar], &Ex::Any(vec![Ex::T(0), Ex::T(1)]));
    assert!((area(&loops) - (84.0 + 4.0 + 8.0)).abs() < 1e-9);
}

#[test]
fn random_stars() {
    let mut seed = 7;
    for round in 0..24 {
        let n = 8 + (round % 5) * 5;
        let (ca, cb) = (6.0 + next(&mut seed) * 0.7, 6.5 + next(&mut seed) * 0.7);
        let a = star(&mut seed, [ca, 6.0], (2.0, 5.0), n);
        let b = star(&mut seed, [7.5, cb], (1.5, 4.5), n + 3);
        let sets = vec![vec![a], vec![b]];
        for (_, e) in ops() {
            run(&sets, &e);
        }
    }
}

/// Later compositions read regions made by earlier ones: their boundaries coincide piece for
/// piece. top = A & B, rest = A & !top, then rest | top is A again, with A's area.
#[test]
fn compositions_of_compositions() {
    let mut seed = 11;
    for _ in 0..8 {
        let a = vec![star(&mut seed, [6.0, 6.0], (2.5, 5.0), 17)];
        let b = vec![star(&mut seed, [8.0, 7.0], (2.0, 4.0), 13)];
        let top = run(&[a.clone(), b.clone()], &Ex::All(vec![Ex::T(0), Ex::T(1)]));
        let rest = run(&[a.clone(), top.clone()], &Ex::All(vec![Ex::T(0), Ex::Not(1)]));
        let whole = run(&[rest.clone(), top.clone()], &Ex::Any(vec![Ex::T(0), Ex::T(1)]));
        assert!((area(&whole) - area(&a)).abs() < 1e-6, "{} vs {}", area(&whole), area(&a));
        let none = run(&[rest, top], &Ex::All(vec![Ex::T(0), Ex::T(1)]));
        assert!(area(&none).abs() < 1e-6);
    }
}
