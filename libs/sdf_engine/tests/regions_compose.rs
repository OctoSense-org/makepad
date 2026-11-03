//! The compose round on the GPU against the f64 stand-in (support/regions_ref.rs): the pieces
//! chain by their keys alone into closed loops; the area matches the f64 composition; probes clear
//! of every boundary see the composition's inside; the keys equal the positional keys of the same
//! pieces. Each case is its own plane of one round, so one work item answers many cases.
#[path = "support/regions_ref.rs"]
mod regions_ref;
#[path = "support/gpu_rig.rs"]
mod gpu_rig;
use makepad_sdf_engine::{chain::{self, ChainGrid, ChainKernels, Pieces}, gpu_jobs::{label, BufRef, Elements, Finished, JobClass, Output, Pass, Pool, Step},
    makepad_draw::{id, ComputeKernel, Cx, LiveId}, plane_abi::{CELL_BIT, LEVEL_REGION, NONE, SEG_REFLEX},
    regions_compose::{self as rc, ComposeKernels, Round, RoundWords}, regions_keep::{self, Keep, KeepKernels, KeepTest, TestKind}, regions_program::Mode, regions_set::{self, SetFrom, SetKernels}};
use std::collections::HashMap;
use regions_ref::*;

mod gather {
    use makepad_sdf_engine::makepad_draw::*;
    script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    // dst[z + i] = src[i]: the round's outputs gathered into one readback.
    mod.sdf.TestGather = {
        workgroup_size: 64
        dispatch_range: shader.uniform(vec4u)
        src: shader.buffer(vec4u)
        dst: shader.buffer_rw(vec4u)
        main: fn() {
            if global_id().x >= self.dispatch_range.y { return }
            let i = self.dispatch_range.x + global_id().x
            self.dst[self.dispatch_range.z + i] = self.src[i]
        }
    }
    // The same for u32 words.
    mod.sdf.TestGather32 = {
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

const NXC: u32 = 2;
const NODES: i64 = NXC as i64 * 10 + 1;
const LEVEL: u32 = LEVEL_REGION + 1;

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

/// The last round's answer: its pieces (all planes) with keys and info, the rounds' overflow,
/// and the output bytes.
struct Composed { pieces: Vec<([f32; 4], [u32; 4], [u32; 4])>, overflow: u32, raw: Vec<u8> }

/// A round's term at a plane offset: a set per plane, output o of an earlier round, or a slot of a
/// field's level pieces.
#[derive(Clone, Copy)]
enum Src<'a> { Sets(&'a [Loops], i32), Round(usize, u32, i32), Level(&'a Stub, u32) }

/// A field and its level pieces as extract lays them out, from boxes: d' = max(|x - cx| - hw,
/// |y - cy| - hh) (the box metric), so the level-L set is the box grown by L. Every coarse cell is
/// on the band; `piece_off` runs over (band fine cell × slots + slot).
struct Stub { band_slot: Vec<u32>, cover: Vec<u32>, nf: Vec<u32>, piece_off: Vec<u32>, piece: Vec<[f32; 4]>, slots: u32, levels: Vec<Vec<f32>> }

fn grown(b: (f64, f64, f64, f64), l: f64) -> Loops { vec![rect(b.0 - b.2 - l, b.1 - b.3 - l, b.0 + b.2 + l, b.1 + b.3 + l)] }

/// Per plane a box, and per slot its level on each plane.
fn stub(boxes: &[(f64, f64, f64, f64)], levels: &[Vec<f64>]) -> Stub {
    let cells = (NXC * NXC) as usize;
    let planes = boxes.len();
    let slots = levels.len();
    let d = |b: (f64, f64, f64, f64), x: f64, y: f64| ((x - b.0).abs() - b.2).max((y - b.1).abs() - b.3);
    let mut st = Stub { band_slot: (0..(planes * cells) as u32).collect(), cover: Vec::new(), nf: Vec::new(), piece_off: Vec::new(), piece: Vec::new(),
        slots: slots as u32, levels: levels.iter().map(|l| l.iter().map(|&v| v as f32).collect()).collect() };
    for (p, &b) in boxes.iter().enumerate() {
        let per_slot: Vec<_> = levels.iter().map(|l| pieces(&grown(b, l[p]))).collect();
        for c in 0..cells {
            let (cx, cy) = ((c % NXC as usize) as f64 * 10.0, (c / NXC as usize) as f64 * 10.0);
            st.cover.push(u32::from(d(b, cx + 5.0, cy + 5.0) <= 0.0));
            for n in 0..121 {
                st.nf.extend([0, (d(b, cx + (n % 11) as f64, cy + (n / 11) as f64) as f32).to_bits()]);
            }
            for local in 0..100i64 {
                let cell = ((c % NXC as usize) as i64 * 10 + local % 10, (c / NXC as usize) as i64 * 10 + local / 10);
                for per in &per_slot {
                    st.piece_off.push(st.piece.len() as u32);
                    for q in per.get(&cell).into_iter().flatten() {
                        let g = [q.a[0] as f32, q.a[1] as f32, q.b[0] as f32, q.b[1] as f32];
                        if g[0] != g[2] || g[1] != g[3] { st.piece.push(g); }
                    }
                }
            }
        }
    }
    st.piece_off.push(st.piece.len() as u32);
    st
}

/// A round: its terms, per output its composition with the flag off and on, and the flag
/// (round, output, plane offset).
#[derive(Clone)]
struct Spec<'a> { terms: Vec<Src<'a>>, outs: Vec<(Ex, Ex)>, flag: Option<(usize, u32, i32)> }

fn spec<'a>(terms: Vec<Src<'a>>, outs: &[Ex]) -> Spec<'a> { Spec { terms, outs: outs.iter().map(|e| (e.clone(), e.clone())).collect(), flag: None } }

/// Runs `rounds` in one work item, with the output capacity `outputs` per round (None: twice its
/// input segs per output), and reads the last round back.
fn compose_rounds(cx: &mut Cx, k: &ComposeKernels, planes: u32, rounds: &[Spec], outputs: Option<u32>) -> Composed {
    let mut pool = Pool::new(0);
    let mut steps = Vec::new();
    let mut done: Vec<(rc::RoundOut, Round, RoundWords)> = Vec::new();
    for Spec { terms, outs, flag } in rounds {
        let words = RoundWords::claim(&mut pool, "regions", planes, outs.len() as u32).unwrap();
        let (mut cells, mut segs, mut round_terms) = (0, 0, Vec::new());
        for src in terms {
            match *src {
                Src::Sets(sets, offset) => {
                    assert_eq!(sets.len() as u32, planes);
                    let set = feature_set(sets);
                    cells += set.slots;
                    segs += set.seg.len() as u32;
                    let set_slot = upload(&mut pool, "test set slot", &set.set_slot);
                    let (seg_off, seg) = (upload(&mut pool, "test seg off", &set.seg_off), upload(&mut pool, "test seg", &set.seg));
                    let (cover, node_side) = (upload(&mut pool, "test cover", &set.cover), upload(&mut pool, "test node side", &set.node_side));
                    round_terms.push(rc::Term::set(set_slot, seg_off, seg, set_slot, cover, node_side, offset));
                }
                Src::Round(i, o, offset) => {
                    cells += done[i].1.cells;
                    segs += done[i].1.outputs;
                    round_terms.push(done[i].0.term(o, offset));
                }
                Src::Level(st, slot) => {
                    cells += st.band_slot.len() as u32;
                    segs += st.piece.len() as u32;
                    let (band_slot, cover, nf) = (upload(&mut pool, "test band slot", &st.band_slot), upload(&mut pool, "test cover", &st.cover), upload(&mut pool, "test nf", &st.nf));
                    let (piece_off, piece) = (upload(&mut pool, "test piece off", &st.piece_off), upload(&mut pool, "test piece", &st.piece));
                    round_terms.push(rc::Term::level(band_slot, cover, nf, piece_off, piece, st.slots, slot, 0, st.levels[slot as usize].clone()));
                }
            }
        }
        let n = terms.len() as u32;
        let table = |e: &Ex| rc::table(n, |b| eval(e, &|t| b >> t & 1 != 0));
        let tables = outs.iter().map(|(off, on)| vec![[table(off), table(on), table(off), table(on)]; planes as usize]).collect();
        let flags = flag.iter().map(|&(i, o, dz)| done[i].0.flag(o, dz)).collect();
        let round = Round { nx: NXC, ny: NXC, planes, level: LEVEL, terms: round_terms, flags, tables, cells: cells.min(planes * NXC * NXC),
            segs, outputs: outputs.unwrap_or(segs * 2 * outs.len() as u32) };
        let (s, out) = rc::steps(k, &mut pool, &round, words).unwrap();
        steps.extend(s);
        done.push((out, round, words));
    }
    let (out, round, words) = done.last().unwrap();
    let cap = round.outputs;
    let gather = cx.with_vm(|vm| {
        let module = vm.heap().module(id!(sdf));
        let object = vm.value(module, id!(TestGather)).as_object().unwrap();
        ComputeKernel::new(vm, object, "test gather").unwrap()
    });
    let readback = pool.readback("test out", cap as u64 * 48);
    for (i, src) in [out.piece, out.key, out.info].into_iter().enumerate() {
        let p = Pass::new(&gather, label("test gather"), Elements::Fixed(cap), 1, 1 << 20, |b| if b == id!(src) { src } else { readback });
        steps.push(Step::Pass(p.zw(i as u32 * cap, 0)));
    }
    steps.push(Output::fixed(readback, cap as u64 * 48));
    let overflow_words: Vec<_> = done.iter().map(|d| d.2.overflow).collect();
    let words = *words;
    let (counts, output) = match gpu_rig::run_steps(cx, pool, steps, JobClass::Interactive) {
        Finished::Done { counts, output, .. } => (counts, output),
        Finished::Failed { error, .. } => panic!("the rounds failed: {error}"),
    };
    let raw = output.unwrap().to_vec();
    let w: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let count = counts.word(words.outputs).min(cap) as usize;
    let quad = |at: usize| [w[at], w[at + 1], w[at + 2], w[at + 3]];
    let o = cap as usize * 4;
    let pieces = (0..count).map(|i| (quad(i * 4).map(f32::from_bits), quad(o + i * 4), quad(2 * o + i * 4))).collect();
    let raw = raw[..count * 16].iter().chain(&raw[o * 4..o * 4 + count * 16]).chain(&raw[2 * o * 4..2 * o * 4 + count * 16]).copied().collect();
    Composed { pieces, overflow: overflow_words.iter().map(|&w| counts.word(w)).sum(), raw }
}

fn compose_gpu(cx: &mut Cx, k: &ComposeKernels, terms: &[(&[Loops], i32)], e: &[Ex], outputs: Option<u32>) -> Composed {
    let srcs = terms.iter().map(|&(s, o)| Src::Sets(s, o)).collect();
    compose_rounds(cx, k, terms[0].0.len() as u32, &[spec(srcs, e)], outputs)
}

/// Composes at the default capacity and again from capacity 1 (doubling while it overflows):
/// the same bytes.
fn compose_checked(cx: &mut Cx, k: &ComposeKernels, terms: &[(&[Loops], i32)], e: &[Ex]) -> Composed {
    let first = compose_gpu(cx, k, terms, e, None);
    assert_eq!(first.overflow, 0, "the default capacity holds the round");
    let mut cap = 1;
    let again = loop {
        let c = compose_gpu(cx, k, terms, e, Some(cap));
        if c.overflow == 0 { break c; }
        cap *= 2;
    };
    assert_eq!(first.raw, again.raw, "the output does not depend on the capacity");
    first
}

/// Checks output `o` on plane `p` of a composed round against the f64 composition of `sets` (one
/// per term, the sets the plane reads): the loops, their area, probes and keys.
fn check(c: &Composed, p: u32, o: u32, sets: &[&Loops], e: &Ex) {
    let level = LEVEL + o;
    let fields: Vec<Field> = sets.iter().map(|l| Field { loops: l }).collect();
    let ref_pieces: Vec<_> = sets.iter().map(|l| pieces(l)).collect();
    let terms: Vec<Term> = ref_pieces.iter().zip(&fields).map(|(p, f)| Term { pieces: p, field: f }).collect();
    let want = chain(&keys(&compose(&terms, e), level, NODES));
    let to = |v: f32| v as f64;
    let mine: Vec<(Out, Key, Key)> = c.pieces.iter().filter(|q| q.2[0] == p && q.2[1] == level).map(|(g, k, info)| {
        let cell = ((info[3] % NODES as u32) as i64, (info[3] / NODES as u32) as i64);
        (Out { a: [to(g[0]), to(g[1])], b: [to(g[2]), to(g[3])], cell, term: (info[2] & 7) as usize }, Key { hi: k[0], lo: k[1] }, Key { hi: k[2], lo: k[3] })
    }).collect();
    // the GPU's keys are the positional keys of its own pieces
    let mut by_cell: std::collections::BTreeMap<Cell, Vec<Out>> = Default::default();
    for (o, _, _) in &mine { by_cell.entry(o.cell).or_default().push(*o); }
    let positional: std::collections::HashMap<(u64, u64, u64, u64), (Key, Key)> = keys(&by_cell, level, NODES).into_iter()
        .map(|(o, a, b)| ((o.a[0].to_bits(), o.a[1].to_bits(), o.b[0].to_bits(), o.b[1].to_bits()), (a, b))).collect();
    for (o, a, b) in &mine {
        assert_eq!(positional[&(o.a[0].to_bits(), o.a[1].to_bits(), o.b[0].to_bits(), o.b[1].to_bits())], (*a, *b), "keys of {o:?}");
    }
    let loops = chain(&mine);
    assert_eq!(loops.len(), want.len(), "plane {p}: loops");
    assert!((area(&loops) - area(&want)).abs() < 1e-3, "plane {p}: area {} for {}", area(&loops), area(&want));
    let result = Field { loops: &loops };
    for i in 0..41 {
        for j in 0..41 {
            let q = [0.5 + i as f64 * 0.31 + 0.0011 * j as f64, 0.5 + j as f64 * 0.31 + 0.0007 * i as f64];
            if fields.iter().any(|f| f.at(q).0.abs() < 1e-3) { continue; }
            assert_eq!(result.inside(q), eval(e, &|t| fields[t].inside(q)), "plane {p}: inside at {q:?}");
        }
    }
}

fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<P> { vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]] }
fn hole(mut l: Vec<P>) -> Vec<P> { l.reverse(); l }
fn next(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64
}
fn star(seed: &mut u64, c: P, r: (f64, f64), n: usize) -> Vec<P> {
    (0..n).map(|i| {
        let a = std::f64::consts::TAU * i as f64 / n as f64;
        let k = r.0 + (r.1 - r.0) * next(seed);
        [c[0] + k * a.cos(), c[1] + k * a.sin()]
    }).collect()
}
fn ops() -> Vec<Ex> {
    vec![Ex::All(vec![Ex::T(0), Ex::T(1)]), Ex::Any(vec![Ex::T(0), Ex::T(1)]), Ex::All(vec![Ex::T(0), Ex::Not(1)]), Ex::All(vec![Ex::T(1), Ex::Not(0)])]
}

/// Two-term cases, one per plane: squares off and on the grid, shared and coincident edges,
/// holes with corners on nodes, the same set twice, and random stars.
fn cases() -> (Vec<Loops>, Vec<Loops>) {
    let mut a: Vec<Loops> = vec![
        vec![rect(1.3, 1.3, 6.7, 6.7)], vec![rect(2.0, 2.0, 8.0, 8.0)], vec![rect(2.0, 2.0, 6.0, 6.0)], vec![rect(2.45, 2.0, 6.45, 6.0)],
        vec![rect(2.0, 2.0, 6.0, 6.0)], vec![rect(2.0, 2.0, 12.0, 12.0), hole(rect(5.0, 5.0, 9.0, 9.0))], vec![rect(2.0, 2.0, 6.0, 6.0)],
    ];
    let mut b: Vec<Loops> = vec![
        vec![rect(3.2, 2.4, 9.1, 5.5)], vec![rect(5.0, 4.0, 11.0, 6.0)], vec![rect(6.0, 3.0, 9.0, 5.0)], vec![rect(6.45, 3.0, 9.45, 5.0)],
        vec![rect(2.0, 3.0, 5.0, 5.0)], vec![rect(7.0, 1.0, 9.0, 13.0)], vec![rect(2.0, 2.0, 6.0, 6.0)],
    ];
    let mut seed = 7;
    for round in 0..8 {
        let n = 8 + (round % 5) * 5;
        let (ca, cb) = (6.0 + next(&mut seed) * 0.7, 6.5 + next(&mut seed) * 0.7);
        a.push(vec![star(&mut seed, [ca, 6.0], (2.0, 5.0), n)]);
        b.push(vec![star(&mut seed, [7.5, cb], (1.5, 4.5), n + 3)]);
    }
    (a, b)
}

#[test]
fn compose_matches_the_f64_composition_and_chains_by_keys() {
    gpu_rig::with_gpu(|cx| {
        cx.with_vm(|vm| { gather::script_mod(vm); });
        let k = ComposeKernels::new(cx).unwrap();
        let (a, b) = cases();
        // the four operations as the outputs of one round
        let outs = ops();
        let c = compose_checked(cx, &k, &[(&a, 0), (&b, 0)], &outs);
        for (o, e) in outs.iter().enumerate() {
            for p in 0..a.len() {
                check(&c, p as u32, o as u32, &[&a[p], &b[p]], e);
            }
        }
        // the same at other job sizes
        gpu_rig::force_elements(cx, Some(64));
        let small = compose_gpu(cx, &k, &[(&a, 0), (&b, 0)], &outs, None);
        gpu_rig::force_elements(cx, None);
        assert_eq!(small.raw, c.raw, "the output does not depend on job sizes");
    });
}

/// A term read from the plane above (TFr = O_j & !O_{j+1}): the last plane reads nothing above,
/// so its composition is O_j itself. Three terms. Rounds reading earlier rounds' outputs in one
/// item, with no chain between them: top = A & B and rest = A & !B as one round's outputs, then
/// rest | top is A and rest & top is empty. A flag: B where the plane below has no A & B, else A.
#[test]
fn plane_offsets_three_terms_compositions_of_compositions_and_flags() {
    gpu_rig::with_gpu(|cx| {
        cx.with_vm(|vm| { gather::script_mod(vm); });
        let k = ComposeKernels::new(cx).unwrap();
        let o: Vec<Loops> = vec![vec![rect(2.0, 2.0, 12.0, 12.0)], vec![rect(4.3, 3.0, 10.0, 14.5)], vec![rect(3.0, 5.0, 8.0, 8.0)]];
        let e = Ex::All(vec![Ex::T(0), Ex::Not(1)]);
        let c = compose_checked(cx, &k, &[(&o, 0), (&o, 1)], std::slice::from_ref(&e));
        let none: Loops = Vec::new();
        for p in 0..o.len() {
            let above = if p + 1 < o.len() { &o[p + 1] } else { &none };
            check(&c, p as u32, 0, &[&o[p], above], &e);
        }
        let mut seed = 11;
        let (mut a, mut b, mut d): (Vec<Loops>, Vec<Loops>, Vec<Loops>) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..4 {
            a.push(vec![star(&mut seed, [6.0, 6.0], (2.5, 5.0), 17)]);
            b.push(vec![star(&mut seed, [8.0, 7.0], (2.0, 4.0), 13)]);
            d.push(vec![star(&mut seed, [7.0, 8.0], (1.5, 4.5), 11)]);
        }
        let three = Ex::Any(vec![Ex::All(vec![Ex::T(0), Ex::Not(1)]), Ex::All(vec![Ex::T(2), Ex::T(1)])]);
        let c = compose_checked(cx, &k, &[(&a, 0), (&b, 0), (&d, 0)], std::slice::from_ref(&three));
        for p in 0..a.len() { check(&c, p as u32, 0, &[&a[p], &b[p], &d[p]], &three); }
        let first = spec(vec![Src::Sets(&a, 0), Src::Sets(&b, 0)], &[Ex::All(vec![Ex::T(0), Ex::T(1)]), e.clone()]);
        for (last, want) in [(Ex::Any(vec![Ex::T(0), Ex::T(1)]), true), (Ex::All(vec![Ex::T(0), Ex::T(1)]), false)] {
            let rounds = [first.clone(), spec(vec![Src::Round(0, 1, 0), Src::Round(0, 0, 0)], &[last])];
            let got = compose_rounds(cx, &k, a.len() as u32, &rounds, None);
            assert_eq!(got.overflow, 0);
            let mut cap = 1;
            let again = loop {
                let c = compose_rounds(cx, &k, a.len() as u32, &rounds, Some(cap));
                if c.overflow == 0 { break c; }
                cap *= 2;
            };
            assert_eq!(got.raw, again.raw, "the output does not depend on the capacity");
            if want {
                for p in 0..a.len() { check(&got, p as u32, 0, &[&a[p]], &Ex::T(0)); }
            } else {
                assert!(got.pieces.is_empty(), "rest & top is empty: {:?}", &got.pieces[..got.pieces.len().min(4)]);
            }
        }
        // planes 0 and 2 overlap, plane 1 does not: the flag reads the plane below
        let fa: Vec<Loops> = vec![vec![rect(2.0, 2.0, 8.0, 8.0)], vec![rect(2.0, 2.0, 5.0, 5.0)], vec![rect(3.3, 2.0, 9.0, 7.5)]];
        let fb: Vec<Loops> = vec![vec![rect(5.0, 4.0, 11.0, 12.0)], vec![rect(7.0, 7.0, 12.0, 12.0)], vec![rect(1.0, 4.4, 6.0, 13.0)]];
        let top = spec(vec![Src::Sets(&fa, 0), Src::Sets(&fb, 0)], &[Ex::All(vec![Ex::T(0), Ex::T(1)])]);
        let pick = Spec { terms: vec![Src::Sets(&fa, 0), Src::Sets(&fb, 0)], outs: vec![(Ex::T(1), Ex::T(0))], flag: Some((0, 0, -1)) };
        let got = compose_rounds(cx, &k, 3, &[top, pick], None);
        assert_eq!(got.overflow, 0);
        for (p, below_overlaps) in [false, true, false].into_iter().enumerate() {
            let want = if below_overlaps { &fa[p] } else { &fb[p] };
            check(&got, p as u32, 0, &[want], &Ex::T(0));
        }
    });
}

/// Slots of a field's level pieces as terms: on each plane the composition of the box grown by
/// that plane's level (both signs, and 0) with a set, as the f64 composition of the grown box.
#[test]
fn level_sets_of_a_field_compose_as_terms() {
    gpu_rig::with_gpu(|cx| {
        cx.with_vm(|vm| { gather::script_mod(vm); });
        let k = ComposeKernels::new(cx).unwrap();
        let boxes = [(6.15, 7.35, 3.2, 2.15), (8.45, 6.55, 2.25, 4.1), (9.05, 9.25, 4.4, 3.3)];
        let levels = vec![vec![1.35, -1.15, 0.0], vec![-0.85, 2.05, -2.3]];
        let st = stub(&boxes, &levels);
        let b: Vec<Loops> = vec![vec![rect(4.3, 3.1, 9.2, 8.4)], vec![star(&mut 5, [8.0, 7.0], (2.0, 4.5), 13)], vec![rect(8.1, 2.2, 15.3, 10.7)]];
        let outs = ops();
        for slot in 0..2u32 {
            let rounds = [spec(vec![Src::Level(&st, slot), Src::Sets(&b, 0)], &outs)];
            let c = compose_rounds(cx, &k, boxes.len() as u32, &rounds, None);
            assert_eq!(c.overflow, 0);
            for (o, e) in outs.iter().enumerate() {
                for p in 0..boxes.len() {
                    check(&c, p as u32, o as u32, &[&grown(boxes[p], levels[slot as usize][p]), &b[p]], e);
                }
            }
        }
    });
}

/// A round's output 1 as a region set, read back: the set's buffers, the round's and the chain's,
/// and the count words.
struct SetRun { part: HashMap<&'static str, Vec<u32>>, segs: u32, loops: u32, slots: u32, fine: u32, chain_loops: u32, outputs: u32, output: u32 }

impl SetRun {
    /// The set's words that its counts define.
    fn defined(&self) -> Vec<u32> {
        let p = |name: &str, n: u32| self.part[name][..n as usize].to_vec();
        [p("seg", self.segs * 4), p("seg_info", self.segs * 4), p("seg_key", self.segs), p("loop_info", self.loops * 4), self.part["loop_off"].clone(),
            p("set_kept", self.slots), self.part["set_kept_off"].clone(), p("seg_off", self.fine + 1)].concat()
    }
}

type Kernels = (ComposeKernels, ChainKernels, SetKernels, KeepKernels);

fn kernels(cx: &mut Cx) -> Kernels {
    (ComposeKernels::new(cx).unwrap(), ChainKernels::new(cx).unwrap(), SetKernels::new(cx).unwrap(), KeepKernels::new(cx).unwrap())
}

/// Which loops a set keeps: all, every other chain loop, or a Keep's decision.
enum Sel { All, Alternate, Decide(Keep) }

/// A two-term round with outputs `outs`, chained (each output past the first resolved against
/// output 0), and the set of output `output` of the loops `sel` keeps.
fn set_run(cx: &mut Cx, k: &Kernels, a: &[Loops], b: &[Loops], outs: &[Ex], output: u32, sel: Sel) -> SetRun {
    let planes = a.len() as u32;
    let mut pool = Pool::new(0);
    let words = RoundWords::claim(&mut pool, "regions", planes, outs.len() as u32).unwrap();
    let (chain_words, set_words) = (chain::claim(&mut pool).unwrap(), regions_set::claim(&mut pool).unwrap());
    let (mut cells, mut segs, mut terms) = (0, 0, Vec::new());
    for sets in [a, b] {
        let set = feature_set(sets);
        cells += set.slots;
        segs += set.seg.len() as u32;
        let set_slot = upload(&mut pool, "test set slot", &set.set_slot);
        let (seg_off, seg) = (upload(&mut pool, "test seg off", &set.seg_off), upload(&mut pool, "test seg", &set.seg));
        let (cover, node_side) = (upload(&mut pool, "test cover", &set.cover), upload(&mut pool, "test node side", &set.node_side));
        terms.push(rc::Term::set(set_slot, seg_off, seg, set_slot, cover, node_side, 0));
    }
    let table = |e: &Ex| rc::table(2, |bits| eval(e, &|t| bits >> t & 1 != 0));
    let tables = outs.iter().map(|e| vec![[table(e); 4]; planes as usize]).collect();
    let no = outs.len() as u32;
    let round = Round { nx: NXC, ny: NXC, planes, level: LEVEL, terms, flags: Vec::new(), tables, cells: cells.min(planes * NXC * NXC), segs, outputs: segs * 4 };
    let (mut steps, out) = rc::steps(&k.0, &mut pool, &round, words).unwrap();
    let pieces = Pieces { piece: out.piece, piece_geo: out.geo, piece_key: out.key, piece_info: out.info, piece_off: out.ooff, band_slot: out.rslot,
        count: words.outputs, capacity: round.outputs, levels_per_cell: no };
    let refs: Vec<(u32, u32)> = (1..no).map(|o| (LEVEL + o, LEVEL)).collect();
    let (ch, s) = chain::steps(&k.1, &mut pool, chain_words, &pieces, &ChainGrid::new(NXC, NXC, planes, NONE, &refs), 1).unwrap();
    steps.extend(s);
    let keep_buf = match sel {
        Sel::All => None,
        Sel::Alternate => Some(upload(&mut pool, "test keep", &(0..ch.loop_capacity).map(|l| l % 2).collect::<Vec<u32>>())),
        Sel::Decide(keep) => {
            let (kept, s) = regions_keep::steps(&k.3, &mut pool, &ch, &keep).unwrap();
            steps.extend(s);
            Some(kept)
        }
    };
    let from = SetFrom { out, words, output, level: LEVEL + output, capacity: round.outputs, nx: NXC, ny: NXC, planes };
    let (set, s) = regions_set::steps(&k.2, &mut pool, set_words, &from, &ch, keep_buf).unwrap();
    steps.extend(s);
    let (cap, fine) = (round.outputs, round.cells * 100);
    let parts: Vec<(&'static str, BufRef, u32)> = vec![
        ("seg", set.seg, cap * 4), ("seg_info", set.seg_info, cap * 4), ("seg_key", set.seg_key, cap), ("loop_info", set.loop_info, set.loop_capacity * 4),
        ("loop_off", set.loop_off, planes + 1), ("set_kept", set.set_kept, round.cells), ("set_kept_off", set.set_kept_off, planes + 1),
        ("seg_off", set.seg_off, fine + 1), ("rslot", out.rslot, planes * NXC * NXC), ("rinfo", out.rinfo, round.cells * 4),
        ("piece", out.piece, cap * 4), ("key", out.key, cap * 4), ("ooff", out.ooff, fine * no + 1), ("piece_loop", ch.piece_loop, cap),
        ("loop_meta", ch.loop_meta, ch.loop_capacity * 4), ("loop_metric", ch.loop_metric, ch.loop_capacity * 8),
    ];
    let mut parts = parts;
    if let Some(kept) = keep_buf { parts.push(("kept", kept, ch.loop_capacity)); }
    let gather = cx.with_vm(|vm| {
        let module = vm.heap().module(id!(sdf));
        ComputeKernel::new(vm, vm.value(module, id!(TestGather32)).as_object().unwrap(), "test gather32").unwrap()
    });
    let total: u32 = parts.iter().map(|p| p.2).sum();
    let readback = pool.readback("test set out", total as u64 * 4);
    let mut at = 0;
    let mut offsets = Vec::new();
    for &(name, buf, n) in &parts {
        steps.push(Step::Pass(Pass::new(&gather, label("test gather32"), Elements::Fixed(n), 1, 1 << 20, |b| if b == id!(src) { buf } else { readback }).zw(at, 0)));
        offsets.push((name, at, n));
        at += n;
    }
    steps.push(Output::fixed(readback, total as u64 * 4));
    let (counts, bytes) = match gpu_rig::run_steps(cx, pool, steps, JobClass::Interactive) {
        Finished::Done { counts, output, .. } => (counts, output.expect("the set readback")),
        Finished::Failed { error, .. } => panic!("the set failed: {error}"),
    };
    assert_eq!(counts.word(words.overflow), 0, "the round holds its pieces");
    for (w, what) in [(chain::WORD_OVERFLOW, "loop overflow"), (chain::WORD_OPEN_TAILS, "open tails"), (chain::WORD_OPEN_HEADS, "open heads"), (chain::WORD_DUPLICATES, "duplicate keys")] {
        assert_eq!(counts.word(ch.word(w)), 0, "chain: {what}");
    }
    let w = gpu_rig::words(&bytes);
    let part = offsets.iter().map(|&(name, at, n)| (name, w[at as usize..(at + n) as usize].to_vec())).collect();
    SetRun { part, segs: counts.word(set.segs), loops: counts.word(set.loops), slots: counts.word(words.cells), fine: counts.word(words.fine),
        chain_loops: counts.word(ch.loops_word()), outputs: no, output }
}

/// The set's invariants, and its segs and loops against the round's output (the pieces of the
/// loops `kept` keeps, by chain loop id).
fn check_set(r: &SetRun, planes: u32, kept: &dyn Fn(u32) -> bool) {
    let p = |name: &str| &r.part[name];
    let quad = |v: &[u32], i: u32| -> [u32; 4] { let i = i as usize * 4; [v[i], v[i + 1], v[i + 2], v[i + 3]] };
    let f = |q: [u32; 4]| q.map(f32::from_bits);
    let (seg, seg_info, seg_key, ooff, piece, key, piece_loop) = (p("seg"), p("seg_info"), p("seg_key"), p("ooff"), p("piece"), p("key"), p("piece_loop"));
    // the segs are output 1's kept pieces in the round's order, cell by cell
    let mut want = Vec::new();
    let mut want_off = Vec::new();
    for e in 0..r.fine {
        want_off.push(want.len() as u32);
        let at = (e * r.outputs + r.output) as usize;
        for i in ooff[at]..ooff[at + 1] {
            if kept(piece_loop[i as usize]) { want.push(i); }
        }
    }
    want_off.push(want.len() as u32);
    assert_eq!(r.segs, want.len() as u32, "segs");
    assert_eq!(&p("seg_off")[..want_off.len()], &want_off[..], "seg_off over the round's fine cells");
    let rinfo = p("rinfo");
    let cells = NXC * NXC;
    let mut seg_plane = vec![0; want.len()];
    for e in 0..r.fine {
        let info = quad(rinfo, e / 100);
        seg_plane[want_off[e as usize] as usize..want_off[e as usize + 1] as usize].fill(info[0]);
        let (gx, gy) = ((info[1] * 10 + e % 100 % 10) as f32, (info[2] * 10 + e % 100 / 10) as f32);
        for s in want_off[e as usize]..want_off[e as usize + 1] {
            let g = f(quad(seg, s));
            assert_eq!(quad(seg, s), quad(piece, want[s as usize]), "seg {s} is its piece");
            assert!(g[0] != g[2] || g[1] != g[3], "seg {s} has a length");
            assert!([g[0], g[2]].iter().all(|&x| x >= gx && x <= gx + 1.0) && [g[1], g[3]].iter().all(|&y| y >= gy && y <= gy + 1.0), "seg {s} lies in fine cell ({gx}, {gy})");
        }
    }
    // links, roots, flags and keys
    let mut loops = Vec::new();
    for s in 0..r.segs {
        let [root, next, prev, flags] = quad(seg_info, s);
        let (g, gn, gp) = (quad(seg, s), quad(seg, next), quad(seg, prev));
        assert_eq!([g[2], g[3]], [gn[0], gn[1]], "seg {s} closes bitwise into {next}");
        assert_eq!(quad(seg_info, next)[2], s, "next's prev is seg {s}");
        let dir = |q: [u32; 4]| { let q = f(q); let (dx, dy) = (q[2] - q[0], q[3] - q[1]); let l = (dx * dx + dy * dy).sqrt(); [dx / l, dy / l] };
        let (e, e1) = (dir(g), dir(gp));
        assert_eq!(flags & SEG_REFLEX != 0, e1[0] * e[1] - e1[1] * e[0] < 0.0, "seg {s} reflex flag");
        let k = quad(key, want[s as usize]);
        assert_eq!(seg_key[s as usize], k[0], "seg {s} key");
        let rank = k[1] & 0xffffff;
        let ordinal = if rank == 0xffffff { 0 } else if k[0] & CELL_BIT == 0 { rank + 1 } else { rank };
        assert_eq!(flags >> 8 & 0xff, ordinal, "seg {s} ordinal");
        // walk its loop: every seg on it has the same root, the smallest index
        let mut n = 1;
        let (mut t, mut least) = (next, s);
        while t != s {
            assert_eq!(quad(seg_info, t)[0], root, "one root per loop");
            least = least.min(t);
            t = quad(seg_info, t)[1];
            n += 1;
            assert!(n <= r.segs, "seg {s}'s loop closes");
        }
        assert_eq!(root, least, "the root is the loop's smallest seg");
        if root == s { loops.push([s, n, seg_plane[s as usize], 0]); }
    }
    assert_eq!(r.loops, loops.len() as u32, "loops");
    for (i, l) in loops.iter().enumerate() {
        assert_eq!(quad(p("loop_info"), i as u32), *l, "loop {i}");
    }
    for q in 0..=planes {
        assert_eq!(p("loop_off")[q as usize], loops.iter().filter(|l| l[2] < q).count() as u32, "loop_off[{q}]");
        assert_eq!(p("set_kept_off")[q as usize], (0..r.slots).filter(|&s| quad(rinfo, s)[0] < q).count() as u32, "set_kept_off[{q}]");
    }
    for slot in 0..r.slots {
        let info = quad(rinfo, slot);
        assert_eq!(p("set_kept")[slot as usize], info[2] * NXC + info[1], "slot {slot}'s coarse cell");
        assert_eq!(p("rslot")[(info[0] * cells + info[2] * NXC + info[1]) as usize], slot, "set_slot maps back");
    }
}

/// Output 1 (A & !B) of a two-output round as a region set, whole and with every other chain loop
/// kept: the set's segs, links, flags, keys, loops and cell index, at any job size.
#[test]
fn region_sets_of_a_round_output() {
    gpu_rig::with_gpu(|cx| {
        cx.with_vm(|vm| { gather::script_mod(vm); });
        let k = kernels(cx);
        let (a, b) = cases();
        let planes = a.len() as u32;
        let outs = [Ex::All(vec![Ex::T(0), Ex::T(1)]), Ex::All(vec![Ex::T(0), Ex::Not(1)])];
        let whole = set_run(cx, &k, &a, &b, &outs, 1, Sel::All);
        assert!(whole.loops > planes, "loops on most planes");
        check_set(&whole, planes, &|_| true);
        let half = set_run(cx, &k, &a, &b, &outs, 1, Sel::Alternate);
        check_set(&half, planes, &|l| l % 2 == 1);
        assert!(half.loops > 0 && half.loops < whole.loops, "keep drops loops");
        gpu_rig::force_elements(cx, Some(64));
        let small = set_run(cx, &k, &a, &b, &outs, 1, Sel::All);
        gpu_rig::force_elements(cx, None);
        assert_eq!(small.defined(), whole.defined(), "the set does not depend on job sizes");
    });
}

/// A Keep over A's components with tests AreaOver(0.1 mm²) and Meets(B), on four planes with
/// the modes Hold(true), Hold(false), All and Nothing: the kept loops (a hole with its contour),
/// and the set of the kept loops.
#[test]
fn keeps_decide_per_component() {
    gpu_rig::with_gpu(|cx| {
        cx.with_vm(|vm| { gather::script_mod(vm); });
        let k = kernels(cx);
        // components (mm²): 0.16 meeting B, 0.04, 0.48 with a hole meeting B, 0.12
        let a_plane: Loops = vec![rect(1.0, 1.0, 5.0, 5.0), rect(7.0, 1.0, 9.0, 3.0), rect(11.0, 11.0, 19.0, 19.0), hole(rect(13.0, 13.0, 17.0, 17.0)), rect(1.0, 12.0, 3.0, 18.0)];
        let b_plane: Loops = vec![rect(4.0, 4.0, 8.0, 8.0), rect(12.0, 15.0, 14.0, 18.0)];
        let planes = 4;
        let (a, b) = (vec![a_plane; planes], vec![b_plane; planes]);
        let modes = vec![[Mode::Hold(true); 4], [Mode::Hold(false); 4], [Mode::All; 4], [Mode::Nothing; 4]];
        let keep = Keep { planes: planes as u32, level: LEVEL, tests: vec![
            KeepTest { kind: TestKind::AreaOver, output: 0, value: vec![0.1; planes] }, KeepTest { kind: TestKind::Meets, output: 1, value: Vec::new() }],
            modes, flags: Vec::new() };
        let outs = [Ex::T(0), Ex::All(vec![Ex::T(0), Ex::T(1)])];
        let r = set_run(cx, &k, &a, &b, &outs, 0, Sel::Decide(keep));
        // per plane, the contour areas (grid units²) kept: a hole goes with its contour (64)
        let want = |p: u32, area: f32| -> bool {
            let contour = if area < 0.0 { 64.0 } else { area };
            let hold = contour == 16.0 || contour == 64.0;
            match p { 0 => hold, 1 => !hold, 2 => true, _ => false }
        };
        let (meta, metric, kept) = (&r.part["loop_meta"], &r.part["loop_metric"], &r.part["kept"]);
        let mut own = 0;
        for l in 0..r.chain_loops as usize {
            if meta[l * 4 + 1] != LEVEL { continue; }
            own += 1;
            let (p, area) = (meta[l * 4], f32::from_bits(metric[l * 8]));
            assert_eq!(kept[l] != 0, want(p, area), "plane {p} loop of area {area}");
        }
        assert_eq!(own, planes * 5, "five loops of A per plane");
        check_set(&r, planes as u32, &|l| kept[l as usize] != 0);
        assert_eq!(r.loops, 3 + 2 + 5, "kept loops: 3 on plane 0, 2 on plane 1, all 5 on plane 2");
    });
}
