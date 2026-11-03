//! Rings on the GPU: synthetic outline, base and wall levels of three islands (one with walls
//! and a round hole, one whose first wall runs into a fin too thin for it, one too thin for any
//! wall), chained and turned into the layer plan.
#[path = "support/gpu_rig.rs"]
mod rig;
#[path = "support/chain_pieces.rs"]
mod chain_pieces;
use chain_pieces::{circle, poly, rounded_rect, scene, LoopSpec, Scene};
use makepad_sdf_engine::chain::{self, ChainGrid, ChainKernels, Pieces};
use makepad_sdf_engine::gpu_jobs::{BufRef, Counts, Finished, JobClass, Pool};
use makepad_sdf_engine::makepad_csg_sdf::plan::{Bias, FirstLayer, Join, LayerPlan, PlanSpec, WallSpec};
use makepad_sdf_engine::makepad_draw::Cx;
use makepad_sdf_engine::plane_abi::{wall_level, WallLevel, LEVEL_BASE, LEVEL_OUTLINE, LEVEL_REGION, NONE};
use makepad_sdf_engine::rings::{self, Cover, RingKernels, Rings, Row, FLAG_THIN, ROLE_FILL, ROLE_OUTLINE, ROLE_WALL};
use std::sync::Arc;

fn level(k: u32, which: WallLevel) -> u32 { wall_level(k, which) }

/// Two Miter walls of 0.45 mm: centres 2.25 and 6.75 grid units in, edges 4.5 and 9.
fn plan_spec() -> PlanSpec {
    let wall = |inset| WallSpec { inset, spacing: 0.45, width: 0.45 };
    PlanSpec {
        zs: Arc::from(vec![0.2, 0.4, 0.6]), wall_sets: vec![vec![wall(0.225), wall(0.675)]], layer_set: Arc::from(vec![0, 0, 0]),
        first_layer: FirstLayer { elephant_foot: 0.0 }, join: Join::Miter(3.0), comp: Bias::default(), texture: None,
        top_one_wall: false, thin_margin: 0.0, shells: None, arachne: None,
    }
}

/// What a loop should come out as: its role bits, wall, and the thin flag of its island.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Want { role: u32, wall: u32, thin: bool }
const NONE_ROLE: Want = Want { role: 0, wall: 0, thin: false };

/// The scene's loops and what each should come out as. Islands: a rect with a round hole and
/// both walls; a square with a fin too thin for wall 1's opening (thin); a strip too thin for
/// any wall (walls, but none and no fill).
fn islands() -> Vec<(LoopSpec, Want)> {
    let (c, r) = ([45.5, 35.5], 5.3);
    let at = |level, features, hole: bool, parent: Option<usize>, island: usize, base: Option<usize>| LoopSpec {
        plane: 0, level, features, closed: true, hole, parent, island: Some(island), base,
    };
    let fin = |d: f64| poly(&[[10.3 + d, 65.2 + d], [40.7 - d, 65.2 + d], [40.7 - d, 77.2 + d], [70.3 - d, 77.2 + d], [70.3 - d, 83.8 - d],
        [40.7 - d, 83.8 - d], [40.7 - d, 95.8 - d], [10.3 + d, 95.8 - d]]);
    let square = |d: f64| rounded_rect(10.3 + d, 65.2 + d, 40.7 - d, 95.8 - d, 0.0, true);
    let outline = Want { role: ROLE_OUTLINE, wall: 0, thin: false };
    let wall = |k| Want { role: ROLE_WALL, wall: k, thin: false };
    let fill = Want { role: ROLE_FILL, wall: 0, thin: false };
    vec![
        // the main island (loops 0..12)
        (at(LEVEL_OUTLINE, rounded_rect(10.3, 10.2, 80.7, 60.8, 0.0, true), false, None, 0, None), outline),
        (at(LEVEL_OUTLINE, circle(c, r, 0.0, false), true, Some(0), 0, None), outline),
        (at(LEVEL_BASE, rounded_rect(10.3, 10.2, 80.7, 60.8, 0.0, true), false, None, 0, Some(2)), NONE_ROLE),
        (at(LEVEL_BASE, circle(c, r, 0.0, false), true, Some(2), 0, Some(2)), NONE_ROLE),
        (at(level(1, WallLevel::Centre), rounded_rect(12.55, 12.45, 78.45, 58.55, 0.0, true), false, None, 0, Some(2)), wall(1)),
        (at(level(1, WallLevel::Centre), circle(c, r + 2.25, 0.0, false), true, Some(4), 0, Some(2)), wall(1)),
        (at(level(1, WallLevel::Edge), rounded_rect(14.8, 14.7, 76.2, 56.3, 0.0, true), false, None, 0, Some(2)), NONE_ROLE),
        (at(level(1, WallLevel::Edge), circle(c, r + 4.5, 0.0, false), true, Some(6), 0, Some(2)), NONE_ROLE),
        (at(level(2, WallLevel::Centre), rounded_rect(17.05, 16.95, 73.95, 54.05, 0.0, true), false, None, 0, Some(2)), wall(2)),
        (at(level(2, WallLevel::Centre), circle(c, r + 6.75, 0.0, false), true, Some(8), 0, Some(2)), wall(2)),
        (at(level(2, WallLevel::Edge), rounded_rect(19.3, 19.2, 71.7, 51.8, 0.0, true), false, None, 0, Some(2)), fill),
        (at(level(2, WallLevel::Edge), circle(c, r + 9.0, 0.0, false), true, Some(10), 0, Some(2)), fill),
        // the fin island (12..18): wall 1's centre runs into the fin, its edge does not
        (at(LEVEL_OUTLINE, fin(0.0), false, None, 12, None), Want { role: ROLE_OUTLINE, wall: 0, thin: true }),
        (at(LEVEL_BASE, fin(0.0), false, None, 12, Some(13)), Want { thin: true, ..NONE_ROLE }),
        (at(level(1, WallLevel::Centre), fin(2.25), false, None, 12, Some(13)), Want { thin: true, ..NONE_ROLE }),
        (at(level(1, WallLevel::Edge), square(4.5), false, None, 12, Some(13)), Want { thin: true, ..NONE_ROLE }),
        (at(level(2, WallLevel::Centre), square(6.75), false, None, 12, Some(13)), Want { thin: true, ..NONE_ROLE }),
        (at(level(2, WallLevel::Edge), square(9.0), false, None, 12, Some(13)), Want { thin: true, ..NONE_ROLE }),
        // the strip (18, 19): no wall fits, and its layer asks for walls: no fill
        (at(LEVEL_OUTLINE, rounded_rect(75.3, 65.2, 95.7, 68.8, 0.0, true), false, None, 18, None), outline),
        (at(LEVEL_BASE, rounded_rect(75.3, 65.2, 95.7, 68.8, 0.0, true), false, None, 18, Some(19)), NONE_ROLE),
    ]
}

/// Walls resolve against the base, which resolves against itself; the outline has no reference.
fn references() -> Vec<(u32, u32)> {
    let mut r = vec![(LEVEL_BASE, LEVEL_BASE)];
    for k in 1..=2 { for w in [WallLevel::Centre, WallLevel::Edge, WallLevel::Margin] { r.push((level(k, w), LEVEL_BASE)); } }
    r
}

struct Run { words: Vec<u32>, counts: Counts, rings: Rings, chain_words: Vec<u32> }
impl Run {
    fn word(&self, w: u32) -> u32 { self.counts.word(self.rings.word(w)) }
    fn rows(&self) -> Vec<Row> { rings::rows(&self.words, self.word(rings::WORD_POINTS), self.word(rings::WORD_TABLE)) }
    fn points(&self, r: &Row) -> Vec<[i32; 2]> {
        (r.first_point as usize..(r.first_point + r.n_points) as usize).map(|i| [self.words[2 * i] as i32, self.words[2 * i + 1] as i32]).collect()
    }
}

struct Kernels { chain: ChainKernels, rings: RingKernels }

fn upload(pool: &mut Pool, name: &'static str, words: Vec<u32>) -> BufRef {
    let buf = pool.buffer(name, words.len() as u64 * 4);
    pool.upload(buf, 0, rig::bytes(&words));
    buf
}

/// A scene's pieces on the GPU, their count in a claimed word.
fn pieces(pool: &mut Pool, scene: &Scene) -> Pieces {
    let np = scene.pieces.len() as u32;
    let count = pool.claim("pieces", 1).unwrap();
    pool.upload(BufRef::COUNTS, count.0 as u64 * 4, rig::bytes(&[np]));
    Pieces {
        piece: upload(pool, "piece", scene.piece_words()),
        piece_geo: upload(pool, "piece geo", scene.geo_words()),
        piece_key: upload(pool, "piece key", scene.key_words()),
        piece_info: upload(pool, "piece info", scene.info_words()),
        piece_off: upload(pool, "piece off", scene.piece_off.clone()),
        band_slot: upload(pool, "band slot", scene.band_slot.clone()),
        count, capacity: np, levels_per_cell: scene.levels.len() as u32,
    }
}

/// Chain and rings over `scene` at re-run factor k, the ring readback and the count words.
fn run(cx: &mut Cx, kernels: &Kernels, scene: &Scene, k: u32) -> Run { run_plan(cx, kernels, scene, None, &plan_spec(), k) }

/// The same with `spec`, and `cover`'s pieces chained as top one wall's cover.
fn run_plan(cx: &mut Cx, kernels: &Kernels, scene: &Scene, cover: Option<&Scene>, spec: &PlanSpec, k: u32) -> Run {
    let mut pool = Pool::new(32);
    let wall = pieces(&mut pool, scene);
    let grid = ChainGrid::new(scene.nx, scene.ny, scene.planes, LEVEL_OUTLINE, &references());
    let chain_words = chain::claim(&mut pool).unwrap();
    let (c, mut steps) = chain::steps(&kernels.chain, &mut pool, chain_words, &wall, &grid, k).unwrap();
    let cover = cover.map(|sc| {
        let cp = pieces(&mut pool, sc);
        let cover_words = chain::claim(&mut pool).unwrap();
        let (cc, cover_steps) = chain::steps(&kernels.chain, &mut pool, cover_words, &cp, &ChainGrid::new(sc.nx, sc.ny, sc.planes, NONE, &[]), k).unwrap();
        steps.extend(cover_steps);
        (cc, cp)
    });
    let spec = rings::ring_spec(spec, &[1]);
    let cover_ref = cover.as_ref().map(|(chain, pieces)| Cover { chain, pieces });
    let words = rings::claim(&mut pool).unwrap();
    let (r, ring_steps) = rings::steps(&kernels.rings, &mut pool, words, &c, &wall, &scene.levels, &grid, &spec, cover_ref, k).unwrap();
    steps.extend(ring_steps);
    steps.push(r.output());
    let (counts, output) = match rig::run_steps(cx, pool, steps, JobClass::Visible) {
        Finished::Done { counts, output, stats, .. } => {
            let mut by_ms = stats.kernels.clone();
            by_ms.sort_by(|a, b| b.2.total_cmp(&a.2));
            let top: Vec<String> = by_ms.iter().take(3).map(|(label, _, ms, jobs)| format!("{label} {ms:.2} ms/{jobs}")).collect();
            eprintln!("chain + rings: {} submission(s) ({}), {} jobs, {:.2} ms GPU; {}", stats.submissions, stats.ends, stats.jobs, stats.gpu_ms, top.join(", "));
            (counts, output.expect("the rings' output"))
        }
        Finished::Failed { error, .. } => panic!("rings: {error}"),
    };
    let chain_words = (0..chain::WORDS).map(|w| counts.word(c.word(w))).collect();
    Run { words: rig::words(&output), counts, rings: r, chain_words }
}

fn with_rings(test: impl FnOnce(&mut Cx, &Kernels)) {
    rig::with_gpu(|cx| {
        let kernels = Kernels { chain: ChainKernels::new(cx).unwrap(), rings: RingKernels::new(cx).unwrap() };
        test(cx, &kernels);
        rig::force_elements(cx, None);
    });
}

/// Grid units to nm as the GPU rounds them: f32, ties to even.
fn nm(p: [f64; 2]) -> [i32; 2] { p.map(|v| ((v as f32) * 100000.0).round_ties_even() as i32) }

/// A polygon's ring: its corners from the rightmost (then lowest), in its direction.
fn corners(spec: &LoopSpec) -> Vec<[i32; 2]> {
    let pts: Vec<[f64; 2]> = spec.features.iter().map(|f| f.at(0.0)).collect();
    let start = (0..pts.len()).max_by(|&a, &b| {
        let (pa, pb) = (pts[a].map(|v| v as f32), pts[b].map(|v| v as f32));
        pa[0].total_cmp(&pb[0]).then(pb[1].total_cmp(&pa[1]))
    }).unwrap();
    (0..pts.len()).map(|i| nm(pts[(start + i) % pts.len()])).collect()
}

/// The rows by spec, through the chain's loop numbering.
fn by_spec(scene: &Scene, run: &Run) -> Vec<Option<Row>> {
    let ids = scene.loop_ids();
    let rows = run.rows();
    (0..scene.specs.len()).map(|s| {
        let lid = ids.iter().position(|&x| x == s).unwrap() as u32;
        rows.iter().copied().find(|r| r.loop_id == lid)
    }).collect()
}

fn check(scene: &Scene, wants: &[Want], run: &Run) {
    assert_eq!(run.chain_words[chain::WORD_DEFECTS as usize], 0, "chain defects");
    let thin = wants.iter().filter(|w| w.thin && w.role & ROLE_OUTLINE != 0).count() as u32;
    assert_eq!([run.word(rings::WORD_OVERFLOW), run.word(rings::WORD_THIN)], [0, thin], "overflow, thin islands");
    let rows = by_spec(scene, run);
    assert_eq!(run.word(rings::WORD_TABLE) as usize, wants.iter().filter(|w| w.role != 0).count(), "table rows");
    for (s, want) in wants.iter().enumerate() {
        let spec = &scene.specs[s];
        let Some(row) = rows[s] else {
            assert_eq!(want.role, 0, "spec {s} has no row");
            continue;
        };
        assert_eq!((row.role, row.wall, row.flags & FLAG_THIN != 0), (want.role, want.wall, want.thin), "spec {s} role, wall, thin");
        assert_eq!(row.hole(), spec.hole, "spec {s} hole");
        let pts = run.points(&row);
        match spec.features[0] {
            chain_pieces::Feature::Arc { c, r, .. } => {
                // a whole circle: its lattice from index 0, (c.x + r, c.y)
                let m = ((r as f32) / (2.0 * chain::CHORD_TOL)).sqrt().ceil().max(1.0) as usize;
                assert_eq!(pts.len(), 4 * m, "spec {s} points");
                let first = nm([c[0] + r, c[1]]);
                assert!((pts[0][0] - first[0]).abs() <= 2 && (pts[0][1] - first[1]).abs() <= 2, "spec {s} starts at {:?}, not {first:?}", pts[0]);
                for p in &pts {
                    let d = ((p[0] as f64 / 1e5 - c[0]).hypot(p[1] as f64 / 1e5 - c[1]) - r).abs();
                    assert!(d < 1e-3, "spec {s} point {p:?} off its circle by {d}");
                }
                // clockwise: the second point is below the first
                assert!(pts[1][1] < pts[0][1], "spec {s} runs clockwise");
            }
            _ => assert_eq!(pts, corners(spec), "spec {s} ring"),
        }
        let bbox = [pts.iter().map(|p| p[0]).min().unwrap(), pts.iter().map(|p| p[1]).min().unwrap(), pts.iter().map(|p| p[0]).max().unwrap(), pts.iter().map(|p| p[1]).max().unwrap()];
        assert_eq!(row.bbox, bbox, "spec {s} bbox");
    }
}

#[test]
fn rings_roles_points_and_the_thick_test() {
    with_rings(|cx, kernels| {
        let (specs, wants): (Vec<LoopSpec>, Vec<Want>) = islands().into_iter().unzip();
        let sc = scene(10, 10, 1, specs, false);
        let r = run(cx, kernels, &sc, 2);
        check(&sc, &wants, &r);
    });
}

#[test]
fn ring_bytes_match_at_three_job_sizes_and_split_alike() {
    with_rings(|cx, kernels| {
        let (specs, wants): (Vec<LoopSpec>, Vec<Want>) = islands().into_iter().unzip();
        let sc = scene(10, 10, 1, specs.clone(), false);
        let mut first: Option<Vec<u32>> = None;
        for forced in [Some(1 << 20), Some(64), Some(1000)] {
            rig::force_elements(cx, forced);
            let r = run(cx, kernels, &sc, 2);
            let used = r.word(rings::WORD_USED) as usize;
            match &first {
                None => first = Some(r.words[..used].to_vec()),
                Some(f) => assert!(f[..] == r.words[..used], "jobs of {forced:?} change the rings' bytes"),
            }
        }
        rig::force_elements(cx, None);
        // the same curves split another way: the same rings, spec by spec
        let split = scene(10, 10, 1, specs, true);
        let (a, b) = (run(cx, kernels, &sc, 2), run(cx, kernels, &split, 2));
        check(&split, &wants, &b);
        let (ra, rb) = (by_spec(&sc, &a), by_spec(&split, &b));
        for s in 0..ra.len() {
            if let (Some(x), Some(y)) = (ra[s], rb[s]) { assert_eq!(a.points(&x), b.points(&y), "spec {s} rings differ when split"); }
        }
    });
}

#[test]
fn rings_decode_to_the_layer_plan() {
    with_rings(|cx, kernels| {
        let (specs, _): (Vec<LoopSpec>, Vec<Want>) = islands().into_iter().unzip();
        let sc = scene(10, 10, 1, specs, false);
        let r = run(cx, kernels, &sc, 2);
        let spec = plan_spec();
        let layers = rings::decode(&r.words, &r.counts, &r.rings, &spec, &[1], [0, 0]).unwrap();
        assert_eq!(layers.len(), 1);
        let (layer, plan): &(u32, LayerPlan) = &layers[0];
        assert_eq!(*layer, 1);
        let rows = by_spec(&sc, &r);
        let ring = |s: usize| r.points(&rows[s].unwrap());
        // islands by descending area: the main island, the fin, the strip
        assert_eq!(plan.islands.len(), 3);
        let [main, fin, strip] = [&plan.islands[0], &plan.islands[1], &plan.islands[2]];
        assert_eq!(main.outline.iter().map(|l| l.pts.clone()).collect::<Vec<_>>(), vec![ring(0), ring(1)]);
        assert!(!main.outline[0].hole && main.outline[1].hole);
        let walls = main.walls.as_ref().expect("the main island has walls");
        assert_eq!(walls.lines.len(), 2);
        for (k, (c, h)) in [(4, 5), (8, 9)].into_iter().enumerate() {
            assert_eq!(walls.lines[k].iter().map(|l| l.pts.clone()).collect::<Vec<_>>(), vec![ring(c), ring(h)], "wall {}", k + 1);
            assert!(walls.lines[k].iter().all(|l| l.closed && !l.odd && l.widths.len() == l.pts.len() && l.widths.iter().all(|&w| w == 0.45)));
        }
        assert_eq!(walls.fill.iter().map(|l| l.pts.clone()).collect::<Vec<_>>(), vec![ring(10), ring(11)]);
        let area = main.outline[0].area_mm2;
        assert!((area - 70.4 * 50.6 * 0.01).abs() < 1e-3, "outline area {area}");
        assert_eq!(fin.outline.iter().map(|l| l.pts.clone()).collect::<Vec<_>>(), vec![ring(12)]);
        assert!(fin.walls.is_none(), "the fin island is thin");
        assert_eq!(strip.outline.iter().map(|l| l.pts.clone()).collect::<Vec<_>>(), vec![ring(18)]);
        let none = strip.walls.as_ref().expect("the strip has walls, none fitting");
        assert!(none.lines.is_empty() && none.fill.is_empty());
    });
}

/// Top one wall. The main island is covered by a rect strictly inside it, left of its hole (its
/// ray ends on the hole), of 35.2 units²: over 1 % of its area (3474.5 net of the hole), under
/// 1 % of its outline's (3562.2): it keeps both walls. A square is covered by 2 units² along
/// its bottom edge (pieces cut from its base): one wall, its first edge level the fill.
#[test]
fn top_one_wall_counts_a_base_piece_by_its_cover() {
    with_rings(|cx, kernels| {
        let (mut specs, mut wants): (Vec<LoopSpec>, Vec<Want>) = islands().into_iter().take(12).unzip();
        let square = |d: f64| rounded_rect(10.3 + d, 65.2 + d, 40.7 - d, 95.8 - d, 0.0, true);
        let at = |level, d, base| LoopSpec { plane: 0, level, features: square(d), closed: true, hole: false, parent: None, island: Some(12), base };
        let none = Want { role: 0, wall: 0, thin: false };
        for (spec, want) in [
            (at(LEVEL_OUTLINE, 0.0, None), Want { role: ROLE_OUTLINE, ..none }),
            (at(LEVEL_BASE, 0.0, Some(13)), none),
            (at(level(1, WallLevel::Centre), 2.25, Some(13)), Want { role: ROLE_WALL, wall: 1, thin: false }),
            (at(level(1, WallLevel::Edge), 4.5, Some(13)), Want { role: ROLE_FILL, ..none }),
            (at(level(2, WallLevel::Centre), 6.75, Some(13)), none),
            (at(level(2, WallLevel::Edge), 9.0, Some(13)), none),
        ] {
            specs.push(spec);
            wants.push(want);
        }
        let sc = scene(10, 10, 1, specs, false);
        let cover_at = |features| LoopSpec { plane: 0, level: LEVEL_REGION, features, closed: true, hole: false, parent: None, island: None, base: None };
        let mut cover = scene(10, 10, 1, vec![
            cover_at(rounded_rect(20.3, 33.3, 25.8, 39.7, 0.0, true)),
            cover_at(rounded_rect(20.3, 65.2, 22.3, 66.2, 0.0, true)),
        ], false);
        // compose's info.z: term | seg << 3, a term 0 seg being a wall piece (here the square's base)
        let base_piece = sc.chains[13][0];
        for p in &mut cover.pieces {
            p.info[2] = if p.p[1] == 65.2 && p.p[3] == 65.2 { base_piece << 3 } else { 1 };
        }
        assert!(cover.pieces.iter().any(|p| p.info[2] & 7 == 0), "the square's cover has pieces of its base");
        let mut spec = plan_spec();
        spec.top_one_wall = true;
        let mut first: Option<Vec<u32>> = None;
        for forced in [Some(1 << 20), Some(64), Some(1000)] {
            rig::force_elements(cx, forced);
            let r = run_plan(cx, kernels, &sc, Some(&cover), &spec, 2);
            check(&sc, &wants, &r);
            let used = r.word(rings::WORD_USED) as usize;
            match &first {
                None => first = Some(r.words[..used].to_vec()),
                Some(f) => assert!(f[..] == r.words[..used], "jobs of {forced:?} change the rings' bytes"),
            }
        }
    });
}
