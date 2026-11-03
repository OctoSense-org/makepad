//! list_rank on the GPU job runtime against a host reference: cycles (root = smallest index)
//! and open chains (root = head), singletons of both kinds, a count below the bound, and the
//! same bytes at three job sizes.
#[path = "support/gpu_rig.rs"]
mod gpu_rig;
#[path = "support/pack.rs"]
mod pack;
use makepad_sdf_engine::{gpu_jobs::{self, BufRef, Finished, JobClass, Pool}, list_rank::{self, RankKernels, Succ}};

const NONE: u32 = u32::MAX;

/// Root, rank and predecessor per element, and each root's list id.
fn reference(succ: &[u32]) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = succ.len();
    let mut pred = vec![NONE; n];
    for (k, &s) in succ.iter().enumerate() { if s != NONE { pred[s as usize] = k as u32; } }
    let mut root = vec![NONE; n];
    let mut rank = vec![0; n];
    for k in 0..n {
        // walk back to the head, or once around the cycle
        let (mut at, mut min, mut head) = (k as u32, k as u32, None);
        for _ in 0..n {
            let p = pred[at as usize];
            if p == NONE { head = Some(at); break; }
            if p == k as u32 { break; }
            at = p;
            min = min.min(at);
        }
        root[k] = head.unwrap_or(min);
    }
    for k in 0..n {
        let (mut at, mut steps) = (k as u32, 0);
        while at != root[k] { at = pred[at as usize]; steps += 1; }
        rank[k] = steps;
    }
    let mut lists = Vec::with_capacity(n);
    let mut seen = 0;
    for k in 0..n { lists.push(seen); if root[k] == k as u32 { seen += 1; } }
    (root, rank, pred, lists)
}

/// Cycles of lengths 1..=5, open chains of lengths 1..=4, shuffled indices.
fn lists(seed: u64) -> Vec<u32> {
    let lens: Vec<(usize, bool)> = (1..=5).map(|l| (l, true)).chain((1..=4).map(|l| (l, false))).cycle().take(40).collect();
    let n: usize = lens.iter().map(|l| l.0).sum();
    let mut ids: Vec<u32> = (0..n as u32).collect();
    let mut x = seed;
    for i in (1..n).rev() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; ids.swap(i, (x % (i as u64 + 1)) as usize); }
    let mut succ = vec![NONE; n];
    let mut at = 0;
    for (len, cycle) in lens {
        let l = &ids[at..at + len];
        for i in 0..len {
            succ[l[i] as usize] = if i + 1 < len { l[i + 1] } else if cycle { l[0] } else { NONE };
        }
        at += len;
    }
    succ
}

#[test]
fn ranks_cycles_and_open_chains() {
    gpu_rig::with_gpu(|cx| {
        let kernels = RankKernels::new(cx).unwrap();
        let scan = gpu_jobs::ScanKernels::new(cx).unwrap();
        let copy = pack::kernel(cx);
        let succ = lists(7);
        let n = succ.len() as u32;
        let (root, rank, pred, ids) = reference(&succ);
        let mut first: Option<Vec<u32>> = None;
        for force in [None, Some(64), Some(4096)] {
            gpu_rig::force_elements(cx, force);
            let mut pool = Pool::new(0);
            let count = pool.claim("rank.count", 1).unwrap();
            let total = pool.claim("rank.lists", 1).unwrap();
            // 3 words per element, the successor in word 1; the bound is past the count
            let input = pool.buffer("succ", n as u64 * 12);
            pool.upload(input, 0, gpu_rig::bytes(&succ.iter().flat_map(|&s| [7u32, s, 9]).collect::<Vec<_>>()));
            pool.upload(BufRef::COUNTS, count.0 as u64 * 4, n.to_le_bytes().to_vec());
            let (mut steps, ranked) = list_rank::steps(&kernels, &scan, &mut pool, count, n + 37, Succ { buffer: input, stride: 3, at: 1 }, total).unwrap();
            let (out, packed) = pack::steps(&copy, &mut pool, &[("root", ranked.root), ("rank", ranked.rank), ("pred", ranked.pred), ("ids", ranked.list_scan)]);
            steps.extend(out);
            let (counts, output) = match gpu_rig::run_steps(cx, pool, steps, JobClass::Visible) {
                Finished::Done { counts, output, .. } => (counts, output.expect("the packed output")),
                Finished::Failed { error, .. } => panic!("rank (jobs {force:?}): {error}"),
            };
            let words = gpu_rig::words(&output);
            let get = |name| packed.get(&words, name)[..n as usize].to_vec();
            assert_eq!(get("root"), root, "roots ({force:?})");
            assert_eq!(get("rank"), rank, "ranks ({force:?})");
            assert_eq!(get("pred"), pred, "predecessors ({force:?})");
            let gpu_ids = get("ids");
            for k in 0..n as usize { if root[k] == k as u32 { assert_eq!(gpu_ids[k], ids[k], "list id of root {k} ({force:?})"); } }
            assert_eq!(counts.word(total), ids.last().unwrap() + u32::from(root[n as usize - 1] == n - 1), "list total");
            let all: Vec<u32> = ["root", "rank", "pred", "ids"].iter().flat_map(|name| get(name)).collect();
            match &first { None => first = Some(all), Some(f) => assert_eq!(f, &all, "bytes differ at job size {force:?}") }
        }
        gpu_rig::force_elements(cx, None);
    });
}

#[test]
fn reference_matches_by_hand() {
    // 0 -> 2 -> 0 (cycle), 1 -> 3 -> NONE (chain), 4 alone (chain), 5 -> 5 (cycle)
    let (root, rank, pred, ids) = reference(&[2, 3, 0, NONE, NONE, 5]);
    assert_eq!(root, [0, 1, 0, 1, 4, 5]);
    assert_eq!(rank, [0, 0, 1, 1, 0, 0]);
    assert_eq!(pred, [2, NONE, 0, 1, NONE, 5]);
    assert_eq!(ids, [0, 1, 2, 2, 2, 3]);
}
