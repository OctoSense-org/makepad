//! A region's FeatureSet on the GPU (S1+S2, GPU-EXEC §9, §10.3): one output of a compose round,
//! chained, laid out as feature_set lays out traced loops, so feature_field reads either.
//!
//! - Segs are the output's pieces (lines in one fine cell each, inside on the left) on closed
//!   chain loops, in the round's order: slot-major over the round's slots (plane, coarse cell),
//!   then fine cell. A `keep` word per chain loop drops whole loops (a Keep round's components);
//!   the kept pieces are compacted in order, so a loop's smallest piece is still its smallest seg.
//!   Pieces past the chain's loop capacity are dropped: run the chain with no overflow.
//! - `seg_info` = (root, next, prev, flags): the chain's list ranking (root = the loop's smallest
//!   index) through the compaction; `SEG_REFLEX` where the turn from the previous seg's direction
//!   is negative, as feature_field's seg meta measures it; in bits 8..15 the start vertex's
//!   ordinal under its key (0 on a node, a fine edge's ranks from 1, a cell's from 0). No creases.
//! - `seg_key` = the start vertex's key hi (its fine edge, a node's +x edge, or CELL_BIT | cell).
//! - `loop_info` = (root, n, plane, 0) of the kept loops at the set's level, in the chain's loop
//!   order (ascending roots); `loop_off` per plane over it; `loop_bias` 0.
//! - The cell index is the round's: `set_slot` = its slot map, `set_kept` = each slot's coarse
//!   cell, `set_kept_off` per plane, and `seg_off` over slot · 100 + local (every fine cell of the
//!   round, the total after the last).
//!
//! Passes: `SetCount` (kept pieces per fine cell, 0 past the round's), the scan into `seg_off`
//! (the segs' count word), `SetFill` (segs, each seg's piece, each piece's seg), `SetInfo`,
//! `SetLoops` (kept loops, 0 past the chain's), the scan into the loops' positions (the loops'
//! count word), `SetLoopFill`, `SetPlanes` (loop_off and set_kept_off) and `SetSlots`.
use makepad_draw::*;
use crate::chain::Chain;
use crate::gpu_jobs::{label, BufRef, CountWord, Elements, JobError, Pass, Pool, ScanKernels, Step};
use crate::regions_compose::{RoundOut, RoundWords};

const CEILING: u32 = 1 << 20;

/// The set's kernels and the scan, made once per Cx.
#[derive(Clone)]
pub struct SetKernels {
    count: ComputeKernel,
    fill: ComputeKernel,
    info: ComputeKernel,
    loops: ComputeKernel,
    loop_fill: ComputeKernel,
    planes: ComputeKernel,
    slots: ComputeKernel,
    scan: ScanKernels,
}
impl SetKernels {
    pub fn new(cx: &mut Cx) -> Result<Self, String> {
        let scan = ScanKernels::new(cx)?;
        cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            let mut kernel = |name: LiveId, label: &str| -> Result<ComputeKernel, String> {
                let object = vm.value(module, name).as_object().ok_or_else(|| format!("{label} kernel missing"))?;
                ComputeKernel::new(vm, object, label)
            };
            Ok(Self {
                count: kernel(id!(SetCount), "region set count")?,
                fill: kernel(id!(SetFill), "region set fill")?,
                info: kernel(id!(SetInfo), "region set info")?,
                loops: kernel(id!(SetLoops), "region set loops")?,
                loop_fill: kernel(id!(SetLoopFill), "region set loop fill")?,
                planes: kernel(id!(SetPlanes), "region set planes")?,
                slots: kernel(id!(SetSlots), "region set slots")?,
                scan,
            })
        })
    }
}

/// Where a set comes from: output `output` of a round (`out`, `words`) of `capacity` pieces at
/// pieces' level `level` (the round's level + the output), on an nx × ny grid of `planes`.
#[derive(Clone, Copy, Debug)]
pub struct SetFrom {
    pub out: RoundOut,
    pub words: RoundWords,
    pub output: u32,
    pub level: u32,
    pub capacity: u32,
    pub nx: u32,
    pub ny: u32,
    pub planes: u32,
}

/// A region's FeatureSet, GPU-resident: the feature_set buffers, and its segs' and loops' count
/// words with their capacities.
#[derive(Clone, Copy, Debug)]
pub struct RegionSet {
    pub seg: BufRef,
    pub seg_info: BufRef,
    pub seg_key: BufRef,
    pub loop_off: BufRef,
    pub loop_info: BufRef,
    pub loop_bias: BufRef,
    pub set_slot: BufRef,
    pub set_kept: BufRef,
    pub set_kept_off: BufRef,
    pub seg_off: BufRef,
    pub segs: CountWord,
    pub loops: CountWord,
    pub capacity: u32,
    pub loop_capacity: u32,
    /// The round's cell capacity (the slots `set_kept` can hold).
    pub cells: u32,
}

/// The set's counts words (its segs, its loops), claimed before its item starts.
pub fn claim(pool: &mut Pool) -> Result<CountWord, JobError> { pool.claim("region set", 2) }

/// The set of `from`'s output, chained by `chain` (over the round's pieces), counting in
/// `words` (from [`claim`]); with `keep`, only the loops whose keep word is not 0.
pub fn steps(k: &SetKernels, pool: &mut Pool, words: CountWord, from: &SetFrom, chain: &Chain, keep: Option<BufRef>) -> Result<(RegionSet, Vec<Step>), JobError> {
    let (segs, loops) = (CountWord(words.0), CountWord(words.0 + 1));
    let (out, cap, lcap) = (&from.out, from.capacity.max(1), chain.loop_capacity.max(1));
    let fine = out.cells * 100;
    let set = RegionSet {
        seg: pool.buffer("region set seg", cap as u64 * 16),
        seg_info: pool.buffer("region set seg info", cap as u64 * 16),
        seg_key: pool.buffer("region set seg key", cap as u64 * 4),
        loop_off: pool.buffer("region set loop off", (from.planes as u64 + 1) * 4),
        loop_info: pool.buffer("region set loop info", lcap as u64 * 16),
        loop_bias: pool.buffer("region set loop bias", lcap as u64 * 4),
        set_slot: out.rslot,
        set_kept: pool.buffer("region set kept", out.cells.max(1) as u64 * 4),
        set_kept_off: pool.buffer("region set kept off", (from.planes as u64 + 1) * 4),
        seg_off: pool.buffer("region set seg off", (fine as u64 + 1) * 4),
        segs, loops, capacity: cap, loop_capacity: lcap, cells: out.cells,
    };
    let scnt = pool.buffer("region set seg count", (fine as u64 + 1) * 4);
    let seg_old = pool.buffer("region set seg piece", cap as u64 * 4);
    let piece_seg = pool.buffer("region set piece seg", from.capacity.max(1) as u64 * 4);
    let lcnt = pool.buffer("region set loop count", (lcap as u64 + 1) * 4);
    let lpos = pool.buffer("region set loop pos", (lcap as u64 + 1) * 4);
    let keep_buf = match keep {
        Some(b) => b,
        None => {
            let b = pool.buffer("region set keep all", 4);
            pool.upload(b, 0, vec![0; 4]);
            b
        }
    };
    let bind = |b: LiveId| -> BufRef {
        match b {
            id!(counts) => BufRef::COUNTS,
            id!(piece) => out.piece, id!(piece_key) => out.key, id!(ooff) => out.ooff, id!(rinfo) => out.rinfo,
            id!(succ) => chain.succ, id!(rank_root) => chain.ranked.root, id!(rank_pred) => chain.ranked.pred,
            id!(piece_loop) => chain.piece_loop, id!(loops) => chain.loops, id!(loop_meta) => chain.loop_meta,
            id!(keep) => keep_buf, id!(scnt) => scnt, id!(seg_off) => set.seg_off, id!(seg) => set.seg, id!(seg_old) => seg_old,
            id!(piece_seg) => piece_seg, id!(seg_info) => set.seg_info, id!(seg_key) => set.seg_key, id!(lcnt) => lcnt, id!(lpos) => lpos,
            id!(loop_info) => set.loop_info, id!(loop_bias) => set.loop_bias, id!(loop_off) => set.loop_off,
            id!(set_kept) => set.set_kept, id!(set_kept_off) => set.set_kept_off,
            _ => BufRef::RUNTIME,
        }
    };
    let grid = [from.nx, from.ny, from.planes, out.outputs];
    let pass = |kernel: &ComputeKernel, name: &str, elements: Elements, cost: u32, w: [u32; 4]| -> Result<Step, JobError> {
        let mut p = Pass::new(kernel, label(name), elements, cost, CEILING, |b| bind(b));
        for (param, v) in [(id!(words), w), (id!(grid), grid)] {
            if kernel.layout().slots.iter().any(|s| s.name == param) { p = p.param_u32x4(param, v).map_err(JobError::Failed)?; }
        }
        Ok(Step::Pass(p))
    };
    let has_keep = keep.is_some() as u32;
    let rw = &from.words;
    let lw = chain.loops_word();
    let mut steps = vec![pass(&k.count, "region set count", Elements::Fixed(fine + 1), 8, [rw.fine.0, from.output, has_keep, 0])?];
    steps.extend(k.scan.exclusive(pool, scnt, set.seg_off, fine + 1, segs));
    steps.push(pass(&k.fill, "region set fill", Elements::Counted { word: rw.fine, bound: fine }, 8, [rw.fine.0, from.output, has_keep, 0])?);
    steps.push(pass(&k.info, "region set info", Elements::Counted { word: segs, bound: cap }, 4, [segs.0, 0, 0, 0])?);
    steps.push(pass(&k.loops, "region set loops", Elements::Fixed(lcap + 1), 1, [lw.0, from.level, has_keep, 0])?);
    steps.extend(k.scan.exclusive(pool, lcnt, lpos, lcap + 1, loops));
    steps.push(pass(&k.loop_fill, "region set loop fill", Elements::Counted { word: lw, bound: lcap }, 2, [lw.0, from.level, has_keep, 0])?);
    steps.push(pass(&k.planes, "region set planes", Elements::Fixed(from.planes + 1), 32, [loops.0, rw.cells.0, 0, 0])?);
    steps.push(pass(&k.slots, "region set slots", Elements::Counted { word: rw.cells, bound: out.cells.max(1) }, 1, [rw.cells.0, 0, 0, 0])?);
    Ok((set, steps))
}

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    // grid = (nx, ny, planes, the round's outputs). keep: a word per chain loop (0 drops it),
    // read where words.z (has keep) is 1.
    mod.sdf.RegionSet = {
        precise_math: true
        workgroup_size: 64
        dispatch_range: shader.uniform(vec4u)
        words: shader.uniform(vec4u)
        counts: shader.buffer_rw(u32)
        element: fn() -> u32 {
            if global_id().x >= self.dispatch_range.y { return 4294967295u }
            return self.dispatch_range.x + global_id().x
        }
    }

    // Output words.y's pieces in fine cell e. A piece is kept when its loop is closed and kept
    // (with keep).
    mod.sdf.SetPieces = {
        ..mod.sdf.RegionSet
        grid: shader.uniform(vec4u)
        ooff: shader.buffer(u32)
        piece_loop: shader.buffer(u32)
        loop_meta: shader.buffer(vec4u)
        keep: shader.buffer(u32)
        range: fn(e: u32) -> vec2u {
            let at = e * self.grid.w + self.words.y
            return vec2u(self.ooff[at], self.ooff[at + 1u])
        }
        kept: fn(i: u32) -> bool {
            let l = self.piece_loop[i]
            // no || guards: the shader's || and && do not short-circuit
            if l == 4294967295u { return false }
            if (self.loop_meta[l].w & 2u) != 0u { return false }
            if self.words.z == 0u { return true }
            return self.keep[l] != 0u
        }
    }

    // words: x the count word of the round's fine cells, y the output, z has keep.
    mod.sdf.SetCount = {
        ..mod.sdf.SetPieces
        scnt: shader.buffer_rw(u32)
        main: fn() {
            let e = self.element()
            if e == 4294967295u { return }
            var n = 0u
            if e < self.counts[self.words.x] {
                let r = self.range(e)
                for i in r.x..r.y { if self.kept(i) { n += 1u } }
            }
            self.scnt[e] = n
        }
    }

    mod.sdf.SetFill = {
        ..mod.sdf.SetPieces
        piece: shader.buffer(vec4f)
        seg_off: shader.buffer(u32)
        seg: shader.buffer_rw(vec4f)
        seg_old: shader.buffer_rw(u32)
        piece_seg: shader.buffer_rw(u32)
        main: fn() {
            let e = self.element()
            if e == 4294967295u || e >= self.counts[self.words.x] { return }
            let r = self.range(e)
            var s = self.seg_off[e]
            for i in r.x..r.y {
                if self.kept(i) {
                    self.seg[s] = self.piece[i]
                    self.seg_old[s] = i
                    self.piece_seg[i] = s
                    s += 1u
                }
            }
        }
    }

    // words: x the segs' count word.
    mod.sdf.SetInfo = {
        ..mod.sdf.RegionSet
        seg: shader.buffer(vec4f)
        seg_old: shader.buffer(u32)
        piece_seg: shader.buffer(u32)
        succ: shader.buffer(u32)
        rank_root: shader.buffer(u32)
        rank_pred: shader.buffer(u32)
        piece_key: shader.buffer(vec4u)
        seg_info: shader.buffer_rw(vec4u)
        seg_key: shader.buffer_rw(u32)
        direction: fn(a: vec2, b: vec2) -> vec2 {
            let l = length(b - a)
            if l > 0.0 { return (b - a) / l }
            return vec2(1.0, 0.0)
        }
        main: fn() {
            let s = self.element()
            if s == 4294967295u || s >= self.counts[self.words.x] { return }
            let i = self.seg_old[s]
            let next = self.piece_seg[self.succ[i]]
            let prev = self.piece_seg[self.rank_pred[i]]
            let root = self.piece_seg[self.rank_root[i]]
            let q = self.seg[s]
            let pq = self.seg[prev]
            let e = self.direction(q.xy, q.zw)
            let e1 = self.direction(pq.xy, pq.zw)
            var flags = 0u
            if e1.x * e.y - e1.y * e.x < 0.0 { flags = 1u }
            let key = self.piece_key[i]
            let rank = key.y & 16777215u
            var ordinal = rank
            if rank == 16777215u { ordinal = 0u }
            else if key.x < 2147483648u { ordinal = rank + 1u }
            self.seg_info[s] = vec4u(root, next, prev, flags | (min(ordinal, 255u) << 8u))
            self.seg_key[s] = key.x
        }
    }

    // A chain loop is kept when it is closed, at the set's level, and kept (with keep).
    // words: x the chain's loops count word, y the set's level, z has keep.
    mod.sdf.SetLoopBase = {
        ..mod.sdf.RegionSet
        loop_meta: shader.buffer(vec4u)
        keep: shader.buffer(u32)
        kept: fn(l: u32) -> bool {
            let meta = self.loop_meta[l]
            if meta.y != self.words.y || (meta.w & 2u) != 0u { return false }
            if self.words.z == 0u { return true }
            return self.keep[l] != 0u
        }
    }

    mod.sdf.SetLoops = {
        ..mod.sdf.SetLoopBase
        lcnt: shader.buffer_rw(u32)
        main: fn() {
            let l = self.element()
            if l == 4294967295u { return }
            var n = 0u
            if l < self.counts[self.words.x] {
                if self.kept(l) { n = 1u }
            }
            self.lcnt[l] = n
        }
    }

    mod.sdf.SetLoopFill = {
        ..mod.sdf.SetLoopBase
        loops: shader.buffer(vec4u)
        lpos: shader.buffer(u32)
        piece_seg: shader.buffer(u32)
        loop_info: shader.buffer_rw(vec4u)
        loop_bias: shader.buffer_rw(u32)
        main: fn() {
            let l = self.element()
            if l == 4294967295u || l >= self.counts[self.words.x] { return }
            if !self.kept(l) { return }
            let at = self.lpos[l]
            let rec = self.loops[l]
            self.loop_info[at] = vec4u(self.piece_seg[rec.x], rec.y, self.loop_meta[l].x, 0u)
            self.loop_bias[at] = 0u
        }
    }

    // words: x the kept loops' count word, y the round's slots' count word.
    mod.sdf.SetPlanes = {
        ..mod.sdf.RegionSet
        grid: shader.uniform(vec4u)
        loop_info: shader.buffer(vec4u)
        rinfo: shader.buffer(vec4u)
        loop_off: shader.buffer_rw(u32)
        set_kept_off: shader.buffer_rw(u32)
        // The first of n records whose plane is at least p (which 0: loops, 1: slots).
        first: fn(p: u32, n: u32, which: u32) -> u32 {
            var lo = 0u
            var hi = n
            for bit in 0..32u {
                if lo >= hi { break }
                let mid = lo + (hi - lo) / 2u
                var plane = 0u
                if which == 0u { plane = self.loop_info[mid].z } else { plane = self.rinfo[mid].x }
                if plane < p { lo = mid + 1u } else { hi = mid }
            }
            return lo
        }
        main: fn() {
            let p = self.element()
            if p == 4294967295u || p > self.grid.z { return }
            self.loop_off[p] = self.first(p, self.counts[self.words.x], 0u)
            self.set_kept_off[p] = self.first(p, self.counts[self.words.y], 1u)
        }
    }

    // words: x the round's slots' count word.
    mod.sdf.SetSlots = {
        ..mod.sdf.RegionSet
        grid: shader.uniform(vec4u)
        rinfo: shader.buffer(vec4u)
        set_kept: shader.buffer_rw(u32)
        main: fn() {
            let slot = self.element()
            if slot == 4294967295u || slot >= self.counts[self.words.x] { return }
            let info = self.rinfo[slot]
            self.set_kept[slot] = info.z * self.grid.x + info.y
        }
    }
    mod.sdf
}
