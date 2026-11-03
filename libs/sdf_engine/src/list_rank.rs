//! List ranking on the GPU: successor lists of disjoint cycles and open chains (the outline's
//! segs, the chain's pieces, thin lines) to each element's root, its rank (steps from the root
//! along the successors), its predecessor, and the lists' ids in root order. A cycle's root is
//! its smallest index; an open chain (its tail's successor NONE) is rooted at its head.
//!
//! Pointer jumping along the predecessors in fixed rounds, ceil(log2 bound) for the roots and
//! again for the ranks. Every round reads one buffer pair and writes the other, so the result
//! never depends on job sizes. Element indices stay below 2^31.
use makepad_draw::*;
use crate::gpu_jobs::{label, BufRef, CountWord, Elements, JobError, Pass, Pool, ScanKernels, Step};
use crate::plane_abi::NONE;

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    // dispatch_range = (first, count, round or parity, 0); words.x: the element count word.
    mod.sdf.Rank = {
        precise_math: true
        workgroup_size: 64
        dispatch_range: shader.uniform(vec4u)
        words: shader.uniform(vec4u)
        counts: shader.buffer(u32)
        element: fn() -> u32 {
            if global_id().x >= self.dispatch_range.y { return 4294967295u }
            let e = self.dispatch_range.x + global_id().x
            if e >= self.counts[self.words.x] { return 4294967295u }
            return e
        }
    }
    mod.sdf.RankClear = {
        ..mod.sdf.Rank
        rank_pred: shader.buffer_rw(u32)
        main: fn() {
            let k = self.element()
            if k == 4294967295u { return }
            self.rank_pred[k] = 4294967295u
        }
    }
    // words: y, z the successor's stride and word in `succ`.
    mod.sdf.RankInit = {
        ..mod.sdf.Rank
        succ: shader.buffer(u32)
        rank_pred: shader.buffer_rw(u32)
        main: fn() {
            let k = self.element()
            if k == 4294967295u { return }
            let s = self.succ[k * self.words.y + self.words.z]
            if s != 4294967295u { self.rank_pred[s] = k }
        }
    }
    // A head (no predecessor) carries its index, every other element its index with the top
    // bit set: the minimum over a chain is its head, over a cycle its smallest index.
    mod.sdf.RankLink = {
        ..mod.sdf.Rank
        rank_pred: shader.buffer(u32)
        rank_m0: shader.buffer_rw(u32)
        rank_p0: shader.buffer_rw(u32)
        main: fn() {
            let k = self.element()
            if k == 4294967295u { return }
            let p = self.rank_pred[k]
            if p == 4294967295u { self.rank_m0[k] = k } else { self.rank_m0[k] = k | 2147483648u }
            self.rank_p0[k] = p
        }
    }
    // Round z: each element's minimum and predecessor pointer over twice the span (pair z % 2
    // into the other); a pointer past a head stays NONE.
    mod.sdf.RankRoots = {
        ..mod.sdf.Rank
        rank_m0: shader.buffer_rw(u32)
        rank_m1: shader.buffer_rw(u32)
        rank_p0: shader.buffer_rw(u32)
        rank_p1: shader.buffer_rw(u32)
        main: fn() {
            let k = self.element()
            if k == 4294967295u { return }
            if self.dispatch_range.z % 2u == 0u {
                let p = self.rank_p0[k]
                if p == 4294967295u { self.rank_m1[k] = self.rank_m0[k]; self.rank_p1[k] = p }
                else { self.rank_m1[k] = min(self.rank_m0[k], self.rank_m0[p]); self.rank_p1[k] = self.rank_p0[p] }
            } else {
                let p = self.rank_p1[k]
                if p == 4294967295u { self.rank_m0[k] = self.rank_m1[k]; self.rank_p0[k] = p }
                else { self.rank_m0[k] = min(self.rank_m1[k], self.rank_m1[p]); self.rank_p0[k] = self.rank_p1[p] }
            }
        }
    }
    // The roots (in pair z) are final: start the ranks back along the predecessors in pair 0.
    // It runs over the bound, so the flags past the count are 0 for the scan.
    mod.sdf.RankStart = {
        ..mod.sdf.Rank
        rank_pred: shader.buffer(u32)
        rank_m0: shader.buffer_rw(u32)
        rank_m1: shader.buffer(u32)
        rank_p0: shader.buffer_rw(u32)
        rank_root: shader.buffer_rw(u32)
        rank_flag: shader.buffer_rw(u32)
        main: fn() {
            if global_id().x >= self.dispatch_range.y { return }
            let k = self.dispatch_range.x + global_id().x
            if k >= self.counts[self.words.x] { self.rank_flag[k] = 0u; return }
            var root = self.rank_m0[k]
            if self.dispatch_range.z == 1u { root = self.rank_m1[k] }
            root = root & 2147483647u
            self.rank_root[k] = root
            if root == k {
                self.rank_flag[k] = 1u
                self.rank_m0[k] = 0u
                self.rank_p0[k] = 4294967295u
            } else {
                self.rank_flag[k] = 0u
                self.rank_m0[k] = 1u
                self.rank_p0[k] = self.rank_pred[k]
            }
        }
    }
    // Round z: distances to the root add along the predecessor pointers until they reach it.
    mod.sdf.RankSteps = {
        ..mod.sdf.Rank
        rank_m0: shader.buffer_rw(u32)
        rank_m1: shader.buffer_rw(u32)
        rank_p0: shader.buffer_rw(u32)
        rank_p1: shader.buffer_rw(u32)
        main: fn() {
            let k = self.element()
            if k == 4294967295u { return }
            if self.dispatch_range.z % 2u == 0u {
                let p = self.rank_p0[k]
                if p == 4294967295u { self.rank_m1[k] = self.rank_m0[k]; self.rank_p1[k] = p }
                else { self.rank_m1[k] = self.rank_m0[k] + self.rank_m0[p]; self.rank_p1[k] = self.rank_p0[p] }
            } else {
                let p = self.rank_p1[k]
                if p == 4294967295u { self.rank_m0[k] = self.rank_m1[k]; self.rank_p0[k] = p }
                else { self.rank_m0[k] = self.rank_m1[k] + self.rank_m1[p]; self.rank_p0[k] = self.rank_p1[p] }
            }
        }
    }
    mod.sdf.RankFinish = {
        ..mod.sdf.Rank
        rank_m0: shader.buffer(u32)
        rank_m1: shader.buffer(u32)
        rank: shader.buffer_rw(u32)
        main: fn() {
            let k = self.element()
            if k == 4294967295u { return }
            if self.dispatch_range.z == 1u { self.rank[k] = self.rank_m1[k] } else { self.rank[k] = self.rank_m0[k] }
        }
    }
    mod.sdf
}

/// The rounds that cover lists of up to `bound` elements.
pub fn rounds(bound: u32) -> u32 { 32 - bound.saturating_sub(1).leading_zeros() }

/// Elements of one job: a rank kernel does a few loads and stores per element, far within the
/// 50 ms bound at this size.
pub const CEILING: u32 = 1 << 20;
/// Work units per element.
const COST: u32 = 8;

/// The ranking kernels, made once.
#[derive(Clone)]
pub struct RankKernels { clear: ComputeKernel, init: ComputeKernel, link: ComputeKernel, roots: ComputeKernel, start: ComputeKernel, steps: ComputeKernel, finish: ComputeKernel }
impl RankKernels {
    pub fn new(cx: &mut Cx) -> Result<Self, String> {
        cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            let mut kernel = |name: LiveId, label: &str| -> Result<ComputeKernel, String> {
                let object = vm.value(module, name).as_object().ok_or_else(|| format!("{label} kernel missing"))?;
                ComputeKernel::new(vm, object, label)
            };
            Ok(Self {
                clear: kernel(id!(RankClear), "rank clear")?, init: kernel(id!(RankInit), "rank init")?, link: kernel(id!(RankLink), "rank link")?,
                roots: kernel(id!(RankRoots), "rank roots")?, start: kernel(id!(RankStart), "rank start")?, steps: kernel(id!(RankSteps), "rank steps")?,
                finish: kernel(id!(RankFinish), "rank finish")?,
            })
        })
    }
}

/// Element k's successor: word `stride * k + at` of `buffer` (NONE ends an open chain).
#[derive(Clone, Copy)]
pub struct Succ { pub buffer: BufRef, pub stride: u32, pub at: u32 }

/// What [`steps`] writes, one u32 per element below the count: `root`, `rank`, `pred` (NONE only
/// at an open chain's head), and `list_scan`, the roots before the element in index order (at a
/// root, its list's id).
#[derive(Clone, Copy, Debug)]
pub struct Ranked { pub root: BufRef, pub rank: BufRef, pub pred: BufRef, pub list_scan: BufRef }

/// Rank the `succ` lists (their element count in `count`, at most `bound`): the list total
/// goes into `lists`.
pub fn steps(kernels: &RankKernels, scan: &ScanKernels, pool: &mut Pool, count: CountWord, bound: u32, succ: Succ, lists: CountWord) -> Result<(Vec<Step>, Ranked), JobError> {
    let n = bound.max(1);
    let bytes = n as u64 * 4;
    let mut b = |name: &'static str| pool.buffer(name, bytes);
    let (m0, m1, p0, p1, flag) = (b("rank m0"), b("rank m1"), b("rank p0"), b("rank p1"), b("rank flag"));
    let ranked = Ranked { root: b("rank root"), rank: b("rank"), pred: b("rank pred"), list_scan: b("rank list scan") };
    let bind = |name: LiveId| match name {
        id!(succ) => succ.buffer, id!(rank_m0) => m0, id!(rank_m1) => m1, id!(rank_p0) => p0, id!(rank_p1) => p1,
        id!(rank_flag) => flag, id!(rank_root) => ranked.root, id!(rank) => ranked.rank, id!(rank_pred) => ranked.pred,
        _ => BufRef::RUNTIME,
    };
    let words = [count.0, succ.stride, succ.at, NONE];
    let pass = |kernel: &ComputeKernel, name: &str, elements: Elements, z: u32| -> Result<Pass, JobError> {
        Ok(Pass::new(kernel, label(name), elements, COST, CEILING, bind).param_u32x4(id!(words), words).map_err(JobError::Failed)?.zw(z, 0))
    };
    let counted = Elements::Counted { word: count, bound: n };
    let l = rounds(n);
    let mut out = vec![
        Step::Pass(pass(&kernels.clear, "rank clear", counted, 0)?),
        Step::Pass(pass(&kernels.init, "rank init", counted, 0)?),
        Step::Pass(pass(&kernels.link, "rank link", counted, 0)?),
        Step::Repeat(pass(&kernels.roots, "rank roots", counted, 0)?, l),
        Step::Pass(pass(&kernels.start, "rank start", Elements::Fixed(n), l % 2)?),
        Step::Repeat(pass(&kernels.steps, "rank steps", counted, 0)?, l),
        Step::Pass(pass(&kernels.finish, "rank finish", counted, l % 2)?),
    ];
    out.extend(scan.exclusive(pool, flag, ranked.list_scan, n, lists));
    Ok((out, ranked))
}

pub const KERNELS: [LiveId; 7] = [id!(RankClear), id!(RankInit), id!(RankLink), id!(RankRoots), id!(RankStart), id!(RankSteps), id!(RankFinish)];

#[cfg(test)]
mod tests {
    #[test]
    fn rounds_cover_the_bound() {
        assert_eq!(super::rounds(1), 0);
        assert_eq!(super::rounds(2), 1);
        assert_eq!(super::rounds(5), 3);
        assert_eq!(super::rounds(1 << 20), 20);
    }
}
