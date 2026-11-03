//! Packed class labels against CPU parity and a box field, including a composed class.
#[path = "support/gpu_rig.rs"]
mod gpu_rig;
#[path = "support/regions_ref.rs"]
mod regions_ref;
use makepad_sdf_engine::{
    gpu_jobs::{label, BufRef, Elements, Finished, JobClass, Output, Pass, Pool, Step},
    makepad_draw::{id, ComputeKernel, Cx, LiveId},
    plane_abi::{LEVEL_REGION, NONE},
    regions_compose::{self as rc, ComposeKernels, Round, RoundWords},
    regions_label::{self as rl, LabelKernels, NO_CLASS},
};
use regions_ref::{pieces, Loops, P};

const NXC: u32 = 2;
const PLANES: u32 = 3;

mod gather {
    use makepad_sdf_engine::makepad_draw::*;
    script_mod! {
        use mod.pod.*
        use mod.math.*
        use mod.shader
        mod.sdf.LabelTestGather = {
            workgroup_size: 64
            dispatch_range: shader.uniform(vec4u)
            src: shader.buffer(u32)
            dst: shader.buffer_rw(u32)
            main: fn() {
                if global_id().x >= self.dispatch_range.y { return }
                let i = self.dispatch_range.x + global_id().x
                self.dst[self.dispatch_range.z + i] = self.src[i]
            }
        }
        mod.sdf
    }
}

/// One term's FeatureSet over every plane, with its node sides (the band is the cells holding
/// segs) and its cover, built as plane tracing and feature_field lay them out.
struct Set { set_slot: Vec<u32>, seg_off: Vec<u32>, seg: Vec<[f32; 4]>, node_side: Vec<u32>, cover: Vec<u32>, slots: u32 }

/// The side of point (x + ε, y + ε²) by the parity of the plane's segs right of it.
fn side(segs: &[[f32; 4]], x: f64, y: f64) -> u32 {
    let mut s = 0;
    for g in segs {
        let (ax, ay, bx, by) = (g[0] as f64, g[1] as f64, g[2] as f64, g[3] as f64);
        if (ay > y) != (by > y) {
            let xc = if ay == y { ax } else if by == y { bx } else { ax + (y - ay) * (bx - ax) / (by - ay) };
            if xc > x { s ^= 1; }
        }
    }
    s
}

fn feature_set(planes: &[Loops]) -> Set {
    let cells = (NXC * NXC) as usize;
    let mut set = Set { set_slot: vec![NONE; cells * planes.len()], seg_off: Vec::new(), seg: Vec::new(), node_side: Vec::new(), cover: Vec::new(), slots: 0 };
    for (p, loops) in planes.iter().enumerate() {
        let mut fine: Vec<Vec<[f32; 4]>> = vec![Vec::new(); cells * 100];
        for (&(x, y), list) in &pieces(loops) {
            assert!(x >= 0 && y >= 0 && x < NXC as i64 * 10 && y < NXC as i64 * 10, "cell {x},{y} on the grid");
            let (c, local) = ((y / 10) as usize * NXC as usize + (x / 10) as usize, (y % 10 * 10 + x % 10) as usize);
            for q in list {
                let g = [q.a[0] as f32, q.a[1] as f32, q.b[0] as f32, q.b[1] as f32];
                if g[0] != g[2] || g[1] != g[3] { fine[c * 100 + local].push(g); }
            }
        }
        let plane_segs: Vec<[f32; 4]> = fine.iter().flatten().copied().collect();
        for c in 0..cells {
            let (cx, cy) = ((c % NXC as usize) as f64, (c / NXC as usize) as f64);
            set.cover.push(side(&plane_segs, cx * 10.0 + 5.0, cy * 10.0 + 5.0));
            if fine[c * 100..c * 100 + 100].iter().all(|v| v.is_empty()) { continue; }
            set.set_slot[p * cells + c] = set.slots;
            set.slots += 1;
            for local in 0..100 {
                set.seg_off.push(set.seg.len() as u32);
                set.seg.extend_from_slice(&fine[c * 100 + local]);
            }
            for n in 0..121 {
                set.node_side.push(side(&plane_segs, cx * 10.0 + (n % 11) as f64, cy * 10.0 + (n / 11) as f64));
            }
        }
    }
    set.seg_off.push(set.seg.len() as u32);
    set
}

fn bytes<T: Copy>(v: &[T]) -> Vec<u8> {
    let n = std::mem::size_of_val(v);
    let mut out = vec![0u8; n.max(4)];
    out[..n].copy_from_slice(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) });
    out
}

fn upload<T: Copy>(pool: &mut Pool, name: &'static str, v: &[T]) -> BufRef {
    let data = bytes(v);
    let b = pool.buffer(name, data.len() as u64);
    pool.upload(b, 0, data);
    b
}

fn term(pool: &mut Pool, set: &Set) -> rc::Term {
    let slot = upload(pool, "test set slot", &set.set_slot);
    let off = upload(pool, "test seg off", &set.seg_off);
    let seg = upload(pool, "test seg", &set.seg);
    let cover = upload(pool, "test cover", &set.cover);
    let nodes = upload(pool, "test node side", &set.node_side);
    rc::Term::set(slot, off, seg, slot, cover, nodes, 0)
}

fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<P> {
    vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
}

fn star(p: usize) -> Vec<P> {
    (0..10).map(|i| {
        let a = std::f64::consts::TAU * i as f64 / 10.0 + 0.17;
        let r = if i % 2 == 0 { 2.6 } else { 1.3 };
        [8.2 + r * a.cos(), 14.3 + p as f64 * 0.13 + r * a.sin()]
    }).collect()
}

struct Case {
    sets: Vec<Vec<Loops>>,
    a: Vec<Loops>,
    b: Vec<Loops>,
    boxes: [(f64, f64, f64, f64); 3],
    levels: Vec<f32>,
}

fn distance(b: (f64, f64, f64, f64), x: f64, y: f64) -> f64 {
    ((x - b.0).abs() - b.2).max((y - b.1).abs() - b.3)
}

/// A tiny field stub: every coarse cell is on the band; nf holds (feature, d' bits).
fn level_term(pool: &mut Pool, case: &Case) -> rc::Term {
    let mut nf = Vec::new();
    let mut cover = Vec::new();
    for &b in &case.boxes {
        for c in 0..NXC * NXC {
            let (cx, cy) = ((c % NXC * 10) as f64, (c / NXC * 10) as f64);
            cover.push(u32::from(distance(b, cx + 5.0, cy + 5.0) <= 0.0));
            for n in 0..121 {
                nf.extend([0, (distance(b, cx + (n % 11) as f64, cy + (n / 11) as f64) as f32).to_bits()]);
            }
        }
    }
    let band = upload(pool, "test band", &(0..PLANES * NXC * NXC).collect::<Vec<_>>());
    let cover = upload(pool, "test level cover", &cover);
    let nf = upload(pool, "test nf", &nf);
    // Labeling reads only field sides; no extracted pieces are needed.
    let off = upload(pool, "test level off", &[0u32]);
    let piece = upload(pool, "test level piece", &[[0f32; 4]]);
    rc::Term::level(band, cover, nf, off, piece, 1, 0, 0, case.levels.clone())
}

/// Runs normal, shifted and all-empty classes in one work item, gathering all label bytes.
fn run(cx: &mut Cx, compose: &ComposeKernels, labels: &LabelKernels, gather: &ComputeKernel, case: &Case) -> (Vec<u8>, Vec<u32>) {
    let mut pool = Pool::new(0);
    let a = feature_set(&case.a);
    let b = feature_set(&case.b);
    let words = RoundWords::claim(&mut pool, "test round", PLANES, 2).unwrap();
    let segs = (a.seg.len() + b.seg.len()) as u32;
    let round = Round {
        nx: NXC, ny: NXC, planes: PLANES, level: LEVEL_REGION,
        terms: vec![term(&mut pool, &a), term(&mut pool, &b)], flags: Vec::new(),
        tables: Round::uniform(&[rc::table(2, |bits| bits & 3 == 3), rc::table(2, |bits| bits & 3 == 1)], PLANES),
        cells: (a.slots + b.slots).min(PLANES * NXC * NXC), segs, outputs: segs * 4,
    };
    let (mut steps, composed) = rc::steps(compose, &mut pool, &round, words).unwrap();
    let mut classes: Vec<_> = case.sets.iter().enumerate().map(|(c, planes)| {
        if c == 2 { None } else { Some(term(&mut pool, &feature_set(planes))) }
    }).collect();
    classes.push(Some(composed.term(1, 0)));
    classes.push(Some(level_term(&mut pool, case)));
    let mut shifted = classes.clone();
    shifted[0] = None; // The first present class must initialize, even when its index is nonzero.
    shifted[1].as_mut().unwrap().plane_offset = 1;
    shifted[6].as_mut().unwrap().plane_offset = -1;
    let n = rl::label_words(NXC, NXC, PLANES);
    assert_eq!(n, 300);
    let readback = pool.readback("test labels", n as u64 * 4 * 3);
    let mut conflicts = Vec::new();
    for (i, classes) in [classes, shifted, vec![None; 7]].iter().enumerate() {
        let count = pool.claim("test conflicts", 1).unwrap();
        conflicts.push(count);
        let (src, s) = rl::steps(labels, &mut pool, NXC, NXC, PLANES, classes, count).unwrap();
        steps.extend(s);
        steps.push(Step::Pass(Pass::new(gather, label("test label gather"), Elements::Fixed(n), 1, 1 << 20,
            |b| if b == id!(src) { src } else { readback }).zw(i as u32 * n, 0)));
    }
    steps.push(Output::fixed(readback, n as u64 * 4 * 3));
    match gpu_rig::run_steps(cx, pool, steps, JobClass::Interactive) {
        Finished::Done { counts, output, .. } => {
            assert_eq!(counts.word(words.overflow), 0, "the compose round fits");
            (output.unwrap().to_vec(), conflicts.iter().map(|&w| counts.word(w)).collect())
        }
        Finished::Failed { error, .. } => panic!("labels failed: {error}"),
    }
}

fn plane_segs(loops: &Loops) -> Vec<[f32; 4]> {
    pieces(loops).values().flatten().map(|q| [q.a[0] as f32, q.a[1] as f32, q.b[0] as f32, q.b[1] as f32]).collect()
}

#[test]
fn labels_match_class_sides_and_count_conflicts_independent_of_job_size() {
    let case = Case {
        sets: vec![
            vec![vec![rect(1.0, 1.0, 5.0, 5.0)]; 3],
            vec![vec![rect(7.0, 1.0, 10.0, 5.0)], vec![rect(7.0, 1.0, 10.0, 5.0)], vec![rect(3.0, 3.0, 7.0, 7.0)]],
            vec![Vec::new(); 3],
            vec![vec![rect(1.0, 12.0, 5.0, 17.0)], Vec::new(), vec![rect(1.0, 12.0, 5.0, 17.0)]],
            (0..3).map(|p| vec![star(p)]).collect(),
        ],
        a: vec![vec![rect(12.0, 1.0, 18.0, 7.0)]; 3],
        b: vec![vec![rect(15.0, 0.5, 19.0, 8.0)]; 3],
        boxes: [(15.2, 14.4, 1.1, 1.5), (15.5, 14.2, 1.2, 1.4), (15.3, 14.6, 1.3, 1.2)],
        levels: vec![0.35, 0.65, -0.25],
    };
    let sets: Vec<Vec<_>> = case.sets.iter().map(|planes| planes.iter().map(plane_segs).collect()).collect();
    let a: Vec<_> = case.a.iter().map(plane_segs).collect();
    let b: Vec<_> = case.b.iter().map(plane_segs).collect();
    let mut expected = Vec::new();
    let mut conflicts = vec![0; 3];
    let mut normal_per_plane = [0; 3];
    for variant in 0..3 {
        for p in 0..3 {
            for gy in 0..20 {
                for gx in 0..20 {
                    let (x, y) = (gx as f64, gy as f64);
                    let mut inside = [false; 7];
                    if variant != 2 {
                        for c in 0..5 {
                            let q = p + usize::from(variant == 1 && c == 1);
                            inside[c] = q < 3 && side(&sets[c][q], x, y) != 0;
                        }
                        if variant == 1 { inside[0] = false; }
                        inside[5] = side(&a[p], x, y) != 0 && side(&b[p], x, y) == 0;
                        let q = if variant == 1 { p.checked_sub(1) } else { Some(p) };
                        inside[6] = q.is_some_and(|q| distance(case.boxes[q], x, y) <= case.levels[p] as f64);
                    }
                    expected.push(inside.iter().position(|&s| s).map_or(NO_CLASS, |c| c as u8));
                    if inside.iter().filter(|&&s| s).count() > 1 {
                        conflicts[variant] += 1;
                        if variant == 0 { normal_per_plane[p] += 1; }
                    }
                }
            }
        }
    }
    assert_eq!(normal_per_plane, [0, 0, 4], "only two classes on the final plane overlap");
    gpu_rig::with_gpu(|cx| {
        cx.with_vm(|vm| { gather::script_mod(vm); });
        let compose = ComposeKernels::new(cx).unwrap();
        let labels = LabelKernels::new(cx).unwrap();
        let gather = cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            let object = vm.value(module, id!(LabelTestGather)).as_object().unwrap();
            ComputeKernel::new(vm, object, "test label gather").unwrap()
        });
        let (actual, counts) = run(cx, &compose, &labels, &gather, &case);
        assert_eq!(actual.len(), expected.len());
        for (i, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(got, want, "variant {}, plane {}, node ({}, {})", i / 1200, i / 400 % 3, i % 20, i / 20 % 20);
        }
        assert_eq!(counts, conflicts, "one conflict per node inside multiple classes");
        gpu_rig::force_elements(cx, Some(64));
        let small = run(cx, &compose, &labels, &gather, &case);
        gpu_rig::force_elements(cx, None);
        assert_eq!(small, (actual, counts), "labels and counts are independent of job sizes");
    });
}
