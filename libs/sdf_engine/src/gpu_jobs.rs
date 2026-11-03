//! Fab's GPU job runtime (GPU-EXEC §6.1, §10.2): one per Cx, shared by every engine that
//! computes on the GPU. A work item is a list of [`Step`]s over the buffers of its [`Pool`]; the
//! runtime turns each pass into jobs of about [`JOB_MS`] of GPU time, packs jobs into
//! compute-only submissions, and reads the item's counts back once per submission. Progress is
//! completion-driven: a readback's callback signals the UI thread, whose next [`Jobs::pump`]
//! drains it and submits the next jobs. No frame or paint event is involved.
//!
//! - Job sizes come from measured rates, in work units per ms per (device, kernel label), from a
//!   fixed set of timer names attributed at each submission's readback (a sample that is missing or
//!   doubled is skipped, never charged to another job). An unmeasured kernel starts with a small
//!   job and doubles job by job through its first submission. A job never exceeds its pass's
//!   `ceiling`, the author's proven 50 ms element bound, so measurements steer sizes, never safety.
//!   No output depends on where jobs split. A job is about [`JOB_MS`] while interactive or visible
//!   work is pending (or a drag runs); else about [`BACKGROUND_JOB_MS`] on a GPU that draws the UI
//!   (a frame waits for at most one job), and its pass's ceiling on one that draws none
//!   ([`Jobs::set_draws_ui`]).
//! - A submission holds about [`SHORT_MS`] of estimated GPU time while interactive or visible work is
//!   pending anywhere (or a drag runs), else about [`LONG_MS`] (a phase of an item between its host
//!   steps is one submission); at most [`JOBS_PER_SUBMISSION`] jobs,
//!   then a completion mark (counts words [`COMPLETION`]) and the readbacks. [`IN_FLIGHT`] submissions
//!   are on the GPU in total, the highest class first; a lower class never holds both while higher
//!   work is pending.
//! - Memory: an item's buffers live in its [`Pool`], sized from exact counts; long-lived state is an
//!   external buffer the engine owns. A finished item's buffers serve the next item of a similar
//!   size; any other spare is dropped before anything new is reserved. The runtime's ledger document
//!   is a working set against the device budget. A refused reservation leaves the item waiting
//!   ([`Jobs::waiting`]); its owner decides to wait, split or report a misfit.
//! - Every item is answered exactly once ([`Finished`]): done, or a [`JobError`].
use makepad_draw::*;
use makepad_draw::makepad_platform::{gpu_admission::{GpuDocument, GpuReservation}, gpu_frame_timer};
use std::{collections::{HashMap, VecDeque}, rc::Rc, sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}, mpsc}, time::Instant};

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    // Budgeted evaluation that resumes across rounds (SDF kernels only). A round runs one
    // thread per active slot; `begin` grants each thread its work budget for the round. The
    // runtime's own jobs select a control mode in dispatch_range.w: 2 seeds the slots with the
    // first elements, 3 compacts finished slots and refills them from the next element; the
    // last compacting workgroup publishes the next round's indirect group count.
    mod.sdf.Jobs = {
        ..mod.sdf.Eval
        counts: shader.buffer_rw(u32)
        // Each slot's 256-word state, then the two active lists of `slots` (counts[18]) words
        // each: one buffer, so a resume kernel binds one less (push constants are 128 bytes).
        continuations: shader.buffer_rw(u32)
        dispatch_range: shader.uniform(vec4u)
        listed: fn(k: u32) -> u32 { return self.continuations[self.counts[18u] * 256u + k] }
        index: fn() -> u32 {
            let i = global_id().x
            if i >= self.counts[10u + self.dispatch_range.y] { return 4294967295u }
            return self.continuations[self.listed(self.dispatch_range.x + i) * 256u + 62u]
        }
        control: fn() -> bool {
            let mode = self.dispatch_range.w
            if mode == 0u { return false }
            let i = global_id().x
            if mode == 2u {
                // seed range: (end, slots, first, 2); only elements below end are seeded, so a
                // slot never holds an element its kernel skips (it would never be done)
                let end = self.dispatch_range.x
                let first = self.dispatch_range.z
                let seeded = min(self.dispatch_range.y, end - first)
                if i < seeded {
                    self.continuations[self.dispatch_range.y * 256u + i] = i
                    self.continuations[i * 256u] = 0u
                    self.continuations[i * 256u + 61u] = 0u
                    self.continuations[i * 256u + 62u] = first + i
                }
                if i == 0u {
                    self.counts[8u] = seeded
                    self.counts[10u] = seeded; self.counts[11u] = 0u
                    self.counts[12u] = 0u
                    self.counts[13u] = (seeded + 63u) / 64u
                    self.counts[14u] = 1u; self.counts[15u] = 1u
                    self.counts[16u] = first + seeded
                    self.counts[17u] = end
                    self.counts[18u] = self.dispatch_range.y
                }
                return true
            }
            let from = self.dispatch_range.y
            let into = 1u - from
            let slots = self.counts[18u]
            if i < self.counts[10u + from] {
                let slot = self.continuations[slots * 256u + from * slots + i]
                let task = slot * 256u
                var keep = self.continuations[task] != 2u
                if !keep {
                    let next = atomic_add(self.counts[16u], 1u)
                    if next < self.counts[17u] {
                        self.continuations[task] = 0u
                        self.continuations[task + 61u] = 0u
                        self.continuations[task + 62u] = next
                        keep = true
                    }
                }
                if keep {
                    let at = atomic_add(self.counts[10u + into], 1u)
                    self.continuations[slots * 256u + into * slots + at] = slot
                }
            }
            barrier()
            if local_index() == 0u {
                let finished = atomic_add(self.counts[12u], 1u) + 1u
                if finished == group_count().x {
                    let count = atomic_add(self.counts[10u + into], 0u)
                    self.counts[8u] = count
                    self.counts[13u] = (count + 63u) / 64u
                    self.counts[10u + from] = 0u
                    self.counts[12u] = 0u
                }
            }
            return true
        }
        task: fn() -> u32 { return self.listed(self.dispatch_range.x + global_id().x) * 256u }
        pending: fn() { self.continuations[self.task() + 61u] = 1u }
        begin: fn() -> bool {
            let task = self.task()
            if self.continuations[task] == 0u { self.continuations[task] = 1u; self.continuations[task + 64u] = 0u }
            else if self.continuations[task] == 2u { return false }
            self.continuations[task + 63u] = 32768u
            return true
        }
        done: fn() { self.continuations[self.task()] = 2u; return }
    }

    // The runtime's own small job. spec = (word, job, jobs, first): `jobs` == 0 writes
    // counts[word] = first (the completion mark); otherwise job j of a counted pass gets the
    // indirect arguments of elements first + j * job .. min(counts[word], first + (j + 1) * job),
    // at args[(at + j) * 4] (groups x, 1, 1, elements), `lanes` threads an element.
    mod.sdf.JobArgs = {
        workgroup_size: 64
        counts: shader.buffer_rw(u32)
        args: shader.buffer_rw(u32)
        spec: shader.uniform(vec4u)
        at: shader.uniform(u32)
        lanes: shader.uniform(u32)
        main: fn() {
            let j = global_id().x
            if self.spec.z == 0u {
                if j == 0u { self.counts[self.spec.x] = self.spec.w }
                return
            }
            if j >= self.spec.z { return }
            let count = self.counts[self.spec.x]
            let first = self.spec.w + j * self.spec.y
            var n = 0u
            if count > first { n = min(count - first, self.spec.y) }
            let k = (self.at + j) * 4u
            self.args[k] = (n * self.lanes + 63u) / 64u
            self.args[k + 1u] = 1u
            self.args[k + 2u] = 1u
            self.args[k + 3u] = n
        }
    }

    // The exclusive scan's three kernels ([`ScanKernels`]): integer sums, so the result never
    // depends on how the runtime splits jobs. A workgroup owns an element ([`Pass::lanes`]): a
    // block of 256 values (SCAN_BLOCK), 4 a thread, its threads' sums scanned in workgroup
    // memory; dispatch_range = (first block, blocks, values, total word).
    mod.sdf.ScanSums = {
        workgroup_size: 64
        source: shader.buffer(u32)
        sums: shader.buffer_rw(u32)
        part: shader.workgroup(u32, 64)
        dispatch_range: shader.uniform(vec4u)
        main: fn() {
            if group_id().x >= self.dispatch_range.y { return }
            let b = self.dispatch_range.x + group_id().x
            let li = local_index()
            var t = 0u
            for j in 0..4u {
                let k = b * 256u + li * 4u + j
                if k < self.dispatch_range.z { t += self.source[k] }
            }
            self.part[li] = t
            barrier()
            var step = 32u
            for r in 0..6u {
                if li < step { self.part[li] = self.part[li] + self.part[li + step] }
                barrier()
                step = step / 2u
            }
            if li == 0u { self.sums[b] = self.part[0u] }
        }
    }
    // One workgroup scans at most SCAN_SERIAL values in place, 64 a thread, and writes their total.
    mod.sdf.ScanSerial = {
        workgroup_size: 64
        values: shader.buffer_rw(u32)
        counts: shader.buffer_rw(u32)
        part: shader.workgroup(u32, 64)
        dispatch_range: shader.uniform(vec4u)
        main: fn() {
            if group_id().x != 0u { return }
            let li = local_index()
            let n = self.dispatch_range.z
            var t = 0u
            for j in 0..64u {
                let k = li * 64u + j
                if k < n { t += self.values[k] }
            }
            self.part[li] = t
            barrier()
            var d = 1u
            for r in 0..6u {
                var add = 0u
                if li >= d { add = self.part[li - d] }
                barrier()
                self.part[li] = self.part[li] + add
                barrier()
                d = d * 2u
            }
            var s = self.part[li] - t
            for j in 0..64u {
                let k = li * 64u + j
                if k < n {
                    let v = self.values[k]
                    self.values[k] = s
                    s += v
                }
            }
            if li == 63u { self.counts[self.dispatch_range.w] = self.part[63u] }
        }
    }
    mod.sdf.ScanApply = {
        workgroup_size: 64
        source: shader.buffer(u32)
        sums: shader.buffer(u32)
        destination: shader.buffer_rw(u32)
        part: shader.workgroup(u32, 64)
        dispatch_range: shader.uniform(vec4u)
        main: fn() {
            if group_id().x >= self.dispatch_range.y { return }
            let b = self.dispatch_range.x + group_id().x
            let li = local_index()
            let first = b * 256u + li * 4u
            var t = 0u
            for j in 0..4u {
                let k = first + j
                if k < self.dispatch_range.z { t += self.source[k] }
            }
            self.part[li] = t
            barrier()
            var d = 1u
            for r in 0..6u {
                var add = 0u
                if li >= d { add = self.part[li - d] }
                barrier()
                self.part[li] = self.part[li] + add
                barrier()
                d = d * 2u
            }
            var s = self.sums[b] + self.part[li] - t
            for j in 0..4u {
                let k = first + j
                if k < self.dispatch_range.z {
                    let v = self.source[k]
                    self.destination[k] = s
                    s += v
                }
            }
        }
    }
}

/// Values of one scan block: a workgroup sums them, and later offsets them.
pub const SCAN_BLOCK: u32 = 256;
/// Values the top of the scan tree takes: one workgroup scans them.
const SCAN_SERIAL: u32 = 4096;
/// Blocks of one scan job.
const SCAN_BLOCKS_CEILING: u32 = 16384;

/// The deterministic exclusive scan's kernels, made once (a host step builds its steps from them).
#[derive(Clone)]
pub struct ScanKernels { sums: ComputeKernel, serial: ComputeKernel, apply: ComputeKernel }
impl ScanKernels {
    pub fn new(cx: &mut Cx) -> Result<Self, String> {
        cx.with_vm(|vm| {
            let module = vm.heap().module(id!(sdf));
            let mut kernel = |name: LiveId, label: &str| -> Result<ComputeKernel, String> {
                let object = vm.value(module, name).as_object().ok_or_else(|| format!("{label} kernel missing"))?;
                ComputeKernel::new(vm, object, label)
            };
            Ok(Self { sums: kernel(id!(ScanSums), "scan sums")?, serial: kernel(id!(ScanSerial), "scan serial")?, apply: kernel(id!(ScanApply), "scan apply")? })
        })
    }

    /// Steps writing the exclusive prefix sums of the first `n` u32s of `input` (counts or
    /// flags, in id order) into `output` (another buffer) and their total into counts word
    /// `total` (claimed: a later pass may take it as its `Elements::Counted` source). Block
    /// sums of [`SCAN_BLOCK`] values, a scan of those (recursively, one workgroup at the top),
    /// then each block's offset added back: integer adds, so the same values give the same
    /// bytes however the jobs split. With `n` = 0 the total is 0 and nothing is read
    /// or written.
    pub fn exclusive(&self, pool: &mut Pool, input: BufRef, output: BufRef, n: u32, total: CountWord) -> Vec<Step> {
        let pass = |kernel: &ComputeKernel, name: &str, blocks: u32, cost: u32, bind: &dyn Fn(LiveId) -> BufRef| {
            Pass::new(kernel, label(name), Elements::Fixed(blocks), cost, SCAN_BLOCKS_CEILING, |b| bind(b)).lanes()
        };
        if n == 0 {
            // nothing to scan: the top reads no value and writes the total, 0
            return vec![Step::Pass(pass(&self.serial, "scan serial", 1, SCAN_SERIAL, &|b| if b == id!(counts) { BufRef::RUNTIME } else { input }).zw(0, total.0))];
        }
        // raw block sums per level, until one thread can scan the top
        let mut levels = vec![(input, n)];
        let mut steps = Vec::new();
        loop {
            let (source, values) = *levels.last().unwrap();
            if levels.len() > 1 && values <= SCAN_SERIAL { break; }
            let blocks = values.div_ceil(SCAN_BLOCK);
            let sums = pool.buffer("scan sums", blocks as u64 * 4);
            steps.push(Step::Pass(pass(&self.sums, "scan sums", blocks, SCAN_BLOCK, &|b| if b == id!(source) { source } else { sums }).zw(values, 0)));
            levels.push((sums, blocks));
        }
        let (top, values) = *levels.last().unwrap();
        steps.push(Step::Pass(pass(&self.serial, "scan serial", 1, SCAN_SERIAL, &|b| if b == id!(counts) { BufRef::RUNTIME } else { top }).zw(values, total.0)));
        // then down: each level's values offset by its block's scanned sum
        let mut scanned = top;
        for k in (0..levels.len() - 1).rev() {
            let (source, values) = levels[k];
            let destination = if k == 0 { output } else { pool.buffer("scan offsets", values as u64 * 4) };
            let sums = scanned;
            steps.push(Step::Pass(pass(&self.apply, "scan apply", values.div_ceil(SCAN_BLOCK), SCAN_BLOCK,
                &|b| if b == id!(source) { source } else if b == id!(sums) { sums } else { destination }).zw(values, 0)));
            scanned = destination;
        }
        steps
    }
}

/// Target GPU time of one job while interactive or visible work is pending.
pub const JOB_MS: f32 = 1.0;
/// Target GPU time of one job otherwise, on a GPU that draws the UI.
pub const BACKGROUND_JOB_MS: f32 = 4.0;
/// Dropped submissions of one item in a row before it fails visibly.
pub const DROPS_IN_A_ROW: u32 = 3;
/// Kernel labels the calibration keeps rates for.
const CALIBRATED_LABELS: usize = 256;
/// Submissions whose readback latency [`Jobs::readback_latencies`] keeps.
pub const READBACKS_KEPT: usize = 256;

/// A pass label: interned, so logs print its name.
pub fn label(name: &str) -> LiveId { LiveId::from_str_with_intern(name, InternLiveId::Yes) }

/// The GPU time every job of this process took, by its timestamps, in ns: only grows, so the
/// difference over a span (a plate's slice) is the GPU busy in it.
static GPU_BUSY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn gpu_busy_ns() -> u64 { GPU_BUSY_NS.load(std::sync::atomic::Ordering::Relaxed) }

/// Fixed cost of a job beside its elements (encoder, launch).
const OVERHEAD_MS: f32 = 0.01;
/// Submission length while class 1-2 work is pending or a drag runs.
pub const SHORT_MS: f32 = 4.0;
/// Submission length otherwise: an item's phase between host steps, at the plate's budget.
pub const LONG_MS: f32 = 64.0;
/// Timed dispatches in one submission: the fixed timer names of one slot.
pub const JOBS_PER_SUBMISSION: usize = 64;
/// Submissions on the GPU at once, in total.
pub const IN_FLIGHT: usize = 2;
/// Timer name sets: a set is reused only once its submission's readback arrived.
const TIMER_SLOTS: usize = 4;
/// A dispatch measured over this is logged (and asserted in debug builds) with its kernel and size.
pub const DISPATCH_LIMIT_MS: f32 = 50.0;
/// Counts words the runtime owns in every item: 8 pending slots, 9 and 19 the completion
/// marks, 10..=18 the resume cursors and the indirect round arguments (13..=15).
pub const RUNTIME_WORDS: std::ops::Range<u32> = 8..20;
/// An item's submissions write their serial into these words in turn. A counts readback may
/// be copied after the item's next submission ran, so each of the (at most [`IN_FLIGHT`])
/// submissions in flight has its own word, and a drop is never hidden by its successor's mark.
pub const COMPLETION: [u32; IN_FLIGHT] = [9, 19];
const PENDING: u32 = 8;
const ROUND_ARGS: u64 = 13 * 4;
/// Words of resume state per slot (`mod.sdf.Jobs`).
const SLOT_WORDS: u64 = 256;
/// A resume round grows its slots while twice its measured time stays within this.
const ROUND_TARGET_MS: f32 = 8.0;
const FIRST_SLOTS: u32 = 1024;
const MAX_ROUNDS: u32 = 16;

/// Priority classes, highest first (GPU-EXEC §6.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum JobClass {
    Background,
    Fine,
    Settling,
    Visible,
    Interactive,
}

/// Why a work item ended without its output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobError {
    /// Stale or superseded; never shown.
    Cancelled,
    /// The device was lost; the owner runs the item again after `Event::GpuReset`.
    DeviceLost,
    /// A submission of the item never completed (its completion mark is missing): a transient.
    /// The owner runs the item again from its start; [`DROPS_IN_A_ROW`] in a row fail it.
    Dropped,
    /// One item at its smallest size does not fit what the device can give now.
    Misfit { wanted_mib: u64, available_mib: u64, reason: String },
    /// A counted [`Output`] held more records than its capacity (GPU-EXEC §11, the K rule): the
    /// attempt delivers nothing; the owner runs the item again with that stage's K doubled, and
    /// past its static worst-case bound answers Failed.
    Overflow { buffer: &'static str, count: u32, cap: u32 },
    /// A GPU or count invariant failed; shown, never redone elsewhere.
    Failed(String),
}
impl std::fmt::Display for JobError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Cancelled => write!(f, "cancelled"),
            Self::DeviceLost => write!(f, "the GPU device was lost"),
            Self::Dropped => write!(f, "a GPU submission was dropped before completion"),
            Self::Misfit { wanted_mib, available_mib, reason } => write!(f, "{wanted_mib} MiB does not fit the {available_mib} MiB the GPU has free: {reason}"),
            Self::Overflow { buffer, count, cap } => write!(f, "{buffer} overflowed: {count} records past its capacity of {cap}"),
            Self::Failed(e) => write!(f, "{e}"),
        }
    }
}

/// A word of the item's counts buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CountWord(pub u32);

/// A buffer of an item's [`Pool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufRef(u32);
impl BufRef {
    /// The item's counts buffer (bound to every kernel's `counts` binding).
    pub const COUNTS: BufRef = BufRef(0);
    /// A binding the runtime fills itself (`counts`; `continuations` of a resume pass).
    pub const RUNTIME: BufRef = BufRef(u32::MAX);
}

/// A pass's elements. Element i of a job is `dispatch_range.x + global_id().x`, below
/// `dispatch_range.x + dispatch_range.y`.
#[derive(Clone, Copy, Debug)]
pub enum Elements {
    /// 0..n
    Fixed(u32),
    /// first..end
    Range { first: u32, end: u32 },
    /// 0..count, the count written by an earlier pass of this item into `word`; jobs go up to
    /// `bound`. A job's indirect arguments cover min(count - first, job) elements, so a job past the
    /// count is empty. dispatch_range.y is the job's size: the kernel also returns at once past
    /// its count word.
    Counted { word: CountWord, bound: u32 },
}
impl Elements {
    fn span(&self) -> (u32, u32) {
        match *self { Self::Fixed(n) => (0, n), Self::Range { first, end } => (first, end.max(first)), Self::Counted { bound, .. } => (0, bound) }
    }
}

/// One kernel over its elements.
pub struct Pass {
    pub kernel: ComputeKernel,
    /// Calibration key and timer label (one per kernel and work shape), interned: its name
    /// prints in logs ([`label`]).
    pub label: LiveId,
    pub params: ComputeParams,
    /// One per kernel binding, in its layout's order.
    pub buffers: Vec<BufRef>,
    pub elements: Elements,
    /// Work units per element (a resume pass: per slot and round).
    pub cost: u32,
    /// The most elements of one job: the author's proven 50 ms bound on the slowest supported GPU.
    pub ceiling: u32,
    /// dispatch_range.zw (a [`Step::Repeat`] writes its round into z).
    pub zw: [u32; 2],
    /// Threads per element: 1, or a whole workgroup (64) cooperating on each element
    /// ([`Pass::lanes`]). Not for a resume pass.
    pub lanes: u32,
}
impl Pass {
    /// Binds every buffer of `kernel` by its name.
    pub fn new(kernel: &ComputeKernel, label: LiveId, elements: Elements, cost: u32, ceiling: u32, mut bind: impl FnMut(LiveId) -> BufRef) -> Self {
        let buffers = kernel.layout().buffers.iter().map(|b| bind(b.name)).collect();
        Self { kernel: kernel.clone(), label, params: kernel.params(), buffers, elements, cost: cost.max(1), ceiling: ceiling.max(1), zw: [0; 2], lanes: 1 }
    }
    pub fn param(mut self, name: LiveId, bytes: &[u8]) -> Result<Self, String> { self.params.set(name, bytes)?; Ok(self) }
    pub fn param_u32(self, name: LiveId, value: u32) -> Result<Self, String> { self.param(name, &value.to_le_bytes()) }
    pub fn param_u32x4(self, name: LiveId, v: [u32; 4]) -> Result<Self, String> {
        self.param(name, &v.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>())
    }
    pub fn zw(mut self, z: u32, w: u32) -> Self { self.zw = [z, w]; self }
    /// One workgroup per element: element `dispatch_range.x + group_id().x`, its 64 threads
    /// told apart by `local_index()`. A job then splits only between elements, so a kernel may
    /// sync its workgroup with `barrier()` (its early return is uniform in the group).
    pub fn lanes(mut self) -> Self { self.lanes = 64; self }
}

pub type HostStep = Box<dyn FnOnce(&Counts, &mut Pool) -> Result<Vec<Step>, JobError>>;
pub enum Step {
    Pass(Pass),
    /// The pass `rounds` times, its round in dispatch_range.z.
    Repeat(Pass, u32),
    /// Budgeted evaluation of the pass's elements by a `mod.sdf.Jobs` kernel, over at most
    /// `slots_ceiling` slots, until every element is done.
    Resume { pass: Pass, slots_ceiling: u32 },
    /// Ends the submission here: the counts arrive as early as they can.
    Checkpoint,
    /// Runs on the UI thread once every earlier job's counts arrived; its steps run next, and
    /// the buffers it asks of the pool are reserved before they do.
    Host(HostStep),
    /// The item's output, read back with the final counts. Ends the submission.
    Output(Output),
}

/// What an [`Step::Output`] reads back.
#[derive(Clone, Copy, Debug)]
pub struct Output { pub buf: BufRef, pub len: OutputLen }
#[derive(Clone, Copy, Debug)]
pub enum OutputLen {
    /// These bytes from the buffer's start.
    Fixed(u64),
    /// `count` records of `stride` bytes, `count` being this word as the output is read: the
    /// buffer's first `cap` records are read back with the counts and cut to `count`. A count
    /// past `cap` answers [`JobError::Overflow`].
    Counted { word: CountWord, stride: u64, cap: u32 },
}
impl OutputLen {
    /// The bytes of a read of `read` bytes that are the output, given the counts as it was read;
    /// None past its capacity.
    fn of(&self, words: &[u32], read: usize) -> Option<usize> {
        match *self {
            OutputLen::Fixed(_) => Some(read),
            OutputLen::Counted { word, stride, cap } => {
                let count = words.get(word.0 as usize).copied().unwrap_or(0);
                (count <= cap).then_some(count as usize * stride as usize)
            }
        }
    }
}
impl Output {
    pub fn fixed(buf: BufRef, bytes: u64) -> Step { Step::Output(Self { buf, len: OutputLen::Fixed(bytes) }) }
    pub fn counted(buf: BufRef, word: CountWord, stride: u64, cap: u32) -> Step {
        Step::Output(Self { buf, len: OutputLen::Counted { word, stride, cap } })
    }
    /// The bytes read back.
    fn read_bytes(&self) -> Option<u64> {
        match self.len {
            OutputLen::Fixed(bytes) => Some(bytes),
            OutputLen::Counted { stride, cap, .. } => stride.checked_mul(cap as u64),
        }
    }
}

/// An item's output bytes: a fixed output's bytes, or a counted output's records.
#[derive(Clone)]
pub struct OutputBytes { bytes: Arc<[u8]>, len: usize }
impl std::ops::Deref for OutputBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] { &self.bytes[..self.len] }
}

/// The counts an item read back, with the names claimed in its pool.
pub struct Counts { words: Vec<u32>, claims: Vec<(&'static str, u32, u32)> }
impl Counts {
    pub fn at(&self, word: u32) -> u32 { self.words.get(word as usize).copied().unwrap_or(0) }
    pub fn word(&self, word: CountWord) -> u32 { self.at(word.0) }
    pub fn words(&self) -> &[u32] { &self.words }
    pub fn named(&self, name: &str) -> &[u32] {
        self.claims.iter().find(|c| c.0 == name).map_or(&[], |&(_, at, n)| &self.words[at as usize..(at + n) as usize])
    }
}

/// A ledger-admitted buffer.
pub struct Allocation {
    pub buffer: GpuBuffer,
    flags: GpuBufferFlags,
    _reservation: GpuReservation,
}
impl Allocation {
    #[track_caller]
    pub fn new(cx: &mut Cx, doc: &GpuDocument, size: u64, flags: GpuBufferFlags) -> Result<Self, String> {
        let reservation = doc.admit(size).map_err(|e| e.to_string())?;
        let buffer = GpuBuffer::new(cx, size, flags)?;
        Ok(Self { buffer, flags, _reservation: reservation })
    }
    pub fn size(&self) -> u64 { self.buffer.size() }
}

enum Held { External(GpuBuffer), Deferred, Pending, Owned(Allocation), Freed }
struct Entry { name: &'static str, bytes: u64, flags: GpuBufferFlags, held: Held }

/// An item's buffers. Sizes are exact; [`Pool::buffer`] reserves before the steps that follow run.
pub struct Pool {
    entries: Vec<Entry>,
    claims: Vec<(&'static str, u32, u32)>,
    counted_words: u32,
    started: bool,
    uploads: Vec<(BufRef, u64, Vec<u8>)>,
}
/// What a reservation was refused for.
#[derive(Clone, Debug)]
pub struct Refusal { pub wanted: u64, pub available: u64, pub reason: String }

impl Pool {
    /// `fixed_words` counts words the item's kernels address directly (the runtime's words
    /// 8..20 among them); [`Pool::claim`] adds named words after them.
    pub fn new(fixed_words: u32) -> Self {
        let fixed_words = fixed_words.max(RUNTIME_WORDS.end);
        let counts = Entry { name: "counts", bytes: 0, flags: GpuBufferFlags { readback: true, indirect: true }, held: Held::Pending };
        Self { entries: vec![counts], claims: Vec::new(), counted_words: fixed_words, started: false, uploads: Vec::new() }
    }
    /// Named counts words, claimed before the item starts.
    pub fn claim(&mut self, name: &'static str, words: u32) -> Result<CountWord, JobError> {
        if self.started { return Err(JobError::Failed(format!("counts word {name} claimed after its item started"))); }
        let at = self.counted_words;
        self.counted_words = at.checked_add(words).ok_or_else(|| JobError::Failed("counts address overflow".into()))?;
        self.claims.push((name, at, words));
        Ok(CountWord(at))
    }
    pub fn counts_words(&self) -> u32 { self.counted_words }
    fn add(&mut self, name: &'static str, bytes: u64, flags: GpuBufferFlags, held: Held) -> BufRef {
        let bytes = bytes.max(4).next_multiple_of(16);
        self.entries.push(Entry { name, bytes, flags, held });
        BufRef(self.entries.len() as u32 - 1)
    }
    /// A storage buffer of exactly `bytes` (rounded up to 16).
    pub fn buffer(&mut self, name: &'static str, bytes: u64) -> BufRef { self.add(name, bytes, GpuBufferFlags::default(), Held::Pending) }
    /// A buffer an [`Step::Output`] reads back.
    pub fn readback(&mut self, name: &'static str, bytes: u64) -> BufRef { self.add(name, bytes, GpuBufferFlags { readback: true, indirect: false }, Held::Pending) }
    /// A storage buffer whose size comes from counts: declared when the steps are built, sized
    /// by [`Pool::size`] in the host step of the checkpoint that counts it. Nothing is reserved
    /// for it until then, and a pass that binds it unsized fails.
    pub fn deferred(&mut self, name: &'static str) -> BufRef { self.add(name, 0, GpuBufferFlags::default(), Held::Deferred) }
    /// A deferred buffer that an [`Step::Output`] reads back.
    pub fn deferred_readback(&mut self, name: &'static str) -> BufRef { self.add(name, 0, GpuBufferFlags { readback: true, indirect: false }, Held::Deferred) }
    /// Sizes a deferred buffer to exactly `bytes` (rounded up to 16), reserved before the steps
    /// that follow run. Once only.
    pub fn size(&mut self, buf: BufRef, bytes: u64) -> Result<(), JobError> {
        let e = self.entries.get_mut(buf.0 as usize).ok_or_else(|| JobError::Failed("sizing a buffer of another pool".into()))?;
        if !matches!(e.held, Held::Deferred) { return Err(JobError::Failed(format!("{} is not a deferred buffer waiting for its size", e.name))); }
        e.bytes = bytes.max(4).next_multiple_of(16);
        e.held = Held::Pending;
        Ok(())
    }
    /// A buffer the engine owns (long-lived state, prepared inputs).
    pub fn external(&mut self, name: &'static str, buffer: GpuBuffer) -> BufRef {
        let bytes = buffer.size();
        let r = self.add(name, bytes, GpuBufferFlags::default(), Held::Pending);
        self.entries[r.0 as usize].held = Held::External(buffer);
        r
    }
    /// Written before the next job runs.
    pub fn upload(&mut self, buf: BufRef, offset: u64, bytes: Vec<u8>) { self.uploads.push((buf, offset, bytes)); }
    /// Dropped before the next reservation. Its jobs must all have run (a host step frees).
    pub fn free(&mut self, buf: BufRef) {
        if let Some(e) = self.entries.get_mut(buf.0 as usize) { e.held = Held::Freed; }
    }
    pub fn bytes(&self, buf: BufRef) -> u64 { self.entries.get(buf.0 as usize).map_or(0, |e| e.bytes) }
    pub fn name(&self, buf: BufRef) -> &'static str { self.entries.get(buf.0 as usize).map_or("", |e| e.name) }
    /// Bytes this pool holds.
    pub fn held(&self) -> u64 { self.entries.iter().map(|e| if let Held::Owned(a) = &e.held { a.size() } else { 0 }).sum() }
    fn gpu(&self, buf: BufRef) -> Option<GpuBuffer> {
        match &self.entries.get(buf.0 as usize)?.held {
            Held::External(b) => Some(b.clone()),
            Held::Owned(a) => Some(a.buffer.clone()),
            Held::Deferred | Held::Pending | Held::Freed => None,
        }
    }
    fn deferred_unsized(&self, buf: BufRef) -> bool { self.entries.get(buf.0 as usize).is_some_and(|e| matches!(e.held, Held::Deferred)) }
    fn pending_bytes(&self) -> u64 {
        self.entries.iter().filter(|e| matches!(e.held, Held::Pending)).map(|e| e.bytes).sum()
    }
    /// Freed buffers go, then every pending one is reserved (a matching spare first). On a
    /// refusal nothing more is reserved, and a later commit continues.
    fn commit(&mut self, cx: &mut Cx, doc: &GpuDocument, spare: &mut Vec<Allocation>) -> Result<(), Refusal> {
        if !self.started {
            self.started = true;
            self.entries[0].bytes = self.counted_words as u64 * 4;
            self.uploads.insert(0, (BufRef::COUNTS, 0, vec![0; self.counted_words as usize * 4]));
        }
        let mut dropped_spares = false;
        for e in self.entries.iter_mut().filter(|e| matches!(e.held, Held::Pending)) {
            // the smallest spare of the same kind that is at most twice the size
            let fit = spare.iter().enumerate()
                .filter(|(_, a)| a.size() >= e.bytes && a.size() <= e.bytes.saturating_mul(2) && a.flags.readback == e.flags.readback && a.flags.indirect == e.flags.indirect)
                .min_by_key(|(_, a)| a.size()).map(|(i, _)| i);
            if let Some(i) = fit { e.held = Held::Owned(spare.swap_remove(i)); continue; }
            // what no item asked for goes before anything new is reserved
            if !dropped_spares { spare.clear(); dropped_spares = true; }
            match Allocation::new(cx, doc, e.bytes, e.flags) {
                Ok(a) => e.held = Held::Owned(a),
                Err(reason) => {
                    let budget = cx.gpu_admission().budget();
                    let in_use = cx.gpu_admission().snapshot().in_use;
                    return Err(Refusal { wanted: e.bytes, available: budget.device_bytes.saturating_sub(in_use), reason: format!("{}: {reason}", e.name) });
                }
            }
        }
        for (buf, offset, bytes) in std::mem::take(&mut self.uploads) {
            let Some(gpu) = self.gpu(buf) else { return Err(Refusal { wanted: 0, available: 0, reason: "upload to a buffer that is not held".into() }) };
            if let Err(e) = gpu.update_owned(cx, offset, bytes) { return Err(Refusal { wanted: 0, available: 0, reason: e }); }
        }
        Ok(())
    }
    /// The owned buffers, for the next item.
    fn release(self, spare: &mut Vec<Allocation>) {
        for e in self.entries { if let Held::Owned(a) = e.held { spare.push(a); } }
    }
}

/// An item's identity in the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ItemId(pub u64);

/// What an owner says an item is ([`Jobs::tag`]): its part, revision and layers (first..end).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemTag { pub part: u64, pub revision: u64, pub layers: (u32, u32) }

/// One submission, for a span report ([`Jobs::record_submissions`]).
#[derive(Clone, Debug)]
pub struct SubmissionRecord {
    pub owner: u64,
    pub item: ItemId,
    pub tag: Option<ItemTag>,
    pub class: JobClass,
    /// Its submit returned.
    pub submitted: Instant,
    /// Its last readback arrived.
    pub arrived: Instant,
    /// The host took its readbacks.
    pub taken: Instant,
    /// Host time building and encoding it, before `submitted`.
    pub build_ms: f32,
    /// Its jobs' GPU ms by their timestamps.
    pub gpu_ms: f32,
    /// Per job: label, elements, ms (None: its timer gave no single sample).
    pub jobs: Vec<(LiveId, u32, Option<f32>)>,
}
static RECORDING: AtomicBool = AtomicBool::new(false);
static RECORDS: Mutex<Vec<SubmissionRecord>> = Mutex::new(Vec::new());

/// An item's output handed on where its readbacks land ([`Jobs::on_landed`]): the counts as the
/// output was read, the output, the item's stats through the submissions before its last, and
/// when the readbacks were checked. It runs on the readback's pool thread, never the UI thread,
/// so keep it short (spawn the work). A dropped submission or an output past its capacity never
/// lands: the item answers it ([`Finished::Failed`]).
pub type Landed = Box<dyn FnOnce(Counts, OutputBytes, ItemStats, Instant) + Send>;

/// The final submission's two readbacks meeting off the UI thread.
#[derive(Default)]
struct Meet { counts: Option<Arc<[u8]>>, output: Option<Arc<[u8]>>, landing: Option<Landing> }
struct Landing { hook: Landed, mark: u32, serial: u64, len: OutputLen, claims: Vec<(&'static str, u32, u32)>, stats: ItemStats }
impl Meet {
    fn arrive(meet: &Mutex<Meet>, is_output: bool, bytes: Arc<[u8]>) {
        let met = {
            let mut m = meet.lock().unwrap();
            if is_output { m.output = Some(bytes) } else { m.counts = Some(bytes) }
            if m.counts.is_some() && m.output.is_some() { m.landing.take().map(|l| (l, m.counts.take().unwrap(), m.output.take().unwrap())) } else { None }
        };
        let Some((l, counts, output)) = met else { return };
        let words: Vec<u32> = counts.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        if words.get(l.mark as usize).copied() != Some(l.serial as u32) { return }
        let Some(len) = l.len.of(&words, output.len()) else { return };
        (l.hook)(Counts { words, claims: l.claims }, OutputBytes { bytes: output, len }, l.stats, Instant::now());
    }
}

/// A new work item.
pub struct WorkItem {
    /// The engine it belongs to: [`Jobs::finished`] answers by owner.
    pub owner: u64,
    pub label: String,
    pub class: JobClass,
    /// Among items of one class: smaller goes first.
    pub order: u64,
    pub pool: Pool,
    pub steps: Vec<Step>,
}

/// Per item, for the owner's numbers.
#[derive(Clone, Debug, Default)]
pub struct ItemStats {
    pub gpu_ms: f32,
    pub submissions: u32,
    pub jobs: u32,
    pub max_job_ms: f32,
    pub max_job: String,
    pub peak_bytes: u64,
    /// Submission to readback, summed and at worst.
    pub readback_ms: f32,
    pub max_readback_ms: f32,
    /// Where a submission's time goes, summed: host time building and encoding it (`build`),
    /// submit to its last readback arriving less its GPU time (`wait`: queueing, fences, the
    /// readback copy and map), the readback arriving to the host taking it (`gap`), and the host
    /// taking the previous one to starting this one (`turn`: host steps, then a pump).
    pub build_ms: f32,
    pub wait_ms: f32,
    pub gap_ms: f32,
    pub turn_ms: f32,
    /// What ended each submission.
    pub ends: Ends,
    /// Per label: elements, ms, jobs.
    pub kernels: Vec<(LiveId, u64, f32, u32)>,
}

/// What ended an item's submissions, counted: its time budget, the jobs a submission holds, a
/// checkpoint, a host step (its counts first), a resume (its rounds continue from the counts), the
/// output, or the item's last steps.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ends { pub budget: u32, pub jobs: u32, pub checkpoint: u32, pub host: u32, pub resume: u32, pub output: u32, pub last: u32 }
impl std::fmt::Display for Ends {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "budget={} jobs={} checkpoint={} host={} resume={} output={} last={}", self.budget, self.jobs, self.checkpoint, self.host, self.resume, self.output, self.last)
    }
}
#[derive(Clone, Copy)]
enum End { Budget, Jobs, Checkpoint, Host, Resume, Output, Last }

/// The runtime's GPU time against the wall time it had work, since the app started: busy
/// fraction = the difference of two readings' `gpu_ms` over their `active_ms`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Busy {
    /// Timed jobs' GPU ms ([`gpu_busy_ns`]).
    pub gpu_ms: f64,
    /// Wall ms with an item unfinished.
    pub active_ms: f64,
    pub submissions: u64,
}

pub enum Finished {
    Done { id: ItemId, counts: Counts, output: Option<OutputBytes>, stats: ItemStats },
    Failed { id: ItemId, error: JobError },
}

#[derive(Default)]
struct Calibration {
    generation: u64,
    /// Units per ms, and when last measured (bounded: the least recently measured label goes).
    rates: HashMap<LiveId, (f32, u64)>,
    slots: HashMap<LiveId, u32>,
    tick: u64,
    forced: Option<u32>,
}
impl Calibration {
    fn est_ms(&self, label: LiveId, units: f64) -> f32 {
        self.rates.get(&label).map_or(JOB_MS, |r| OVERHEAD_MS + (units / r.0 as f64) as f32)
    }
    /// Elements of the next job: about `job_ms` (None: the ceiling), a multiple of 64 (but the
    /// remainder), within the ceiling.
    fn elements(&self, p: &Pass, ladder: u32, job_ms: Option<f32>) -> u32 {
        if let Some(n) = self.forced { return n.clamp(1, p.ceiling); }
        let n = match (self.rates.get(&p.label), job_ms) {
            (_, None) => p.ceiling as u64,
            (Some(rate), Some(ms)) => ((ms - OVERHEAD_MS) as f64 * rate.0 as f64 / p.cost as f64) as u64,
            (None, Some(_)) => (p.ceiling as u64 / 64).max(64) << ladder.min(6),
        };
        (n.min(p.ceiling as u64) as u32 / 64 * 64).max(64).min(p.ceiling)
    }
    fn observe(&mut self, label: LiveId, units: f64, ms: f32) {
        if units <= 0.0 || !(ms > 0.0) { return; }
        let measured = (units / (ms - OVERHEAD_MS).max(0.002) as f64) as f32;
        self.tick += 1;
        if !self.rates.contains_key(&label) && self.rates.len() >= CALIBRATED_LABELS {
            // kernels no longer loaded stop being measured: the stalest goes
            if let Some(stale) = self.rates.iter().min_by_key(|(_, r)| r.1).map(|(k, _)| *k) {
                self.rates.remove(&stale);
                self.slots.remove(&stale);
            }
        }
        let rate = self.rates.entry(label).or_insert((measured, 0));
        // slower at once, faster gradually (at most 4x a sample)
        rate.0 = if measured < rate.0 { measured } else { (rate.0 * 0.5 + measured * 0.5).min(rate.0 * 4.0) };
        rate.1 = self.tick;
    }
    fn slots(&self, label: LiveId, ceiling: u32) -> u32 {
        if let Some(n) = self.forced { return n.clamp(64, ceiling.max(64)); }
        self.slots.get(&label).copied().unwrap_or(FIRST_SLOTS).min(ceiling).max(64)
    }
    fn observe_round(&mut self, label: LiveId, slots: u32, ceiling: u32, ms: f32) {
        if self.forced.is_some() || !(ms > 0.0) { return; }
        let s = self.slots.entry(label).or_insert(FIRST_SLOTS.min(ceiling));
        if slots != *s { return; }
        if ms * 2.0 <= ROUND_TARGET_MS { *s = (*s * 2).min(ceiling); }
        else if ms > ROUND_TARGET_MS * 2.0 { *s = (*s / 2).max(64); }
    }
}

/// A test hook: every job (and every resume's slot count) of later submissions has `n`
/// elements, within its ceiling; None measures again.
pub fn force_elements(cx: &mut Cx, n: Option<u32>) {
    cx.global::<Jobs>().calibration.forced = n;
}

/// Rates per label (units per ms) and resume slots, for the numbers line.
pub fn calibration_line(cx: &mut Cx) -> String {
    let c = &cx.global::<Jobs>().calibration;
    let mut rates: Vec<_> = c.rates.iter().map(|(k, v)| format!("{k}={:.0}/ms", v.0)).collect();
    rates.sort();
    let mut slots: Vec<_> = c.slots.iter().map(|(k, v)| format!("{k}={v}")).collect();
    slots.sort();
    format!("rates [{}] slots [{}]", rates.join(" "), slots.join(" "))
}

struct ResumeState { slots: u32, ping: u32, seeded: bool, continuations: BufRef }

struct Item {
    id: ItemId,
    owner: u64,
    label: String,
    class: JobClass,
    order: u64,
    steps: VecDeque<Step>,
    pool: Pool,
    counts: Counts,
    in_flight: u32,
    /// A resume's rounds are on the GPU: its readback says whether it continues.
    resuming: bool,
    /// Submissions made: the next one's completion word.
    marks: u32,
    /// The front pass's next element and round, and the job ladder of an unmeasured kernel.
    cursor: u32,
    round: u32,
    ladder: u32,
    resume: Option<ResumeState>,
    resume_buffers: Option<(BufRef, u32)>,
    args: Option<BufRef>,
    output: Option<OutputBytes>,
    output_wanted: bool,
    waiting: Option<Refusal>,
    error: Option<JobError>,
    stats: ItemStats,
    /// When the host took its last submission's readbacks.
    taken: Option<Instant>,
    tag: Option<ItemTag>,
    landed: Option<Landed>,
}

struct JobRecord { label: LiveId, units: f64, elements: u32, calibrate: bool, round_slots: Option<(u32, u32)>, size: String }
struct Submission {
    serial: u64,
    owner: u64,
    tag: Option<ItemTag>,
    /// The counts word its completion mark is in.
    mark: u32,
    item: ItemId,
    class: JobClass,
    slot: usize,
    jobs: Vec<JobRecord>,
    resume: bool,
    output: Option<Output>,
    counts: Option<Result<Arc<[u8]>, String>>,
    output_bytes: Option<Result<Arc<[u8]>, String>>,
    submitted: Instant,
    /// Host time building and encoding it.
    build_ms: f32,
    /// Its last readback's arrival.
    arrived: Option<Instant>,
}
type Readback = (u64, bool, Result<Arc<[u8]>, String>, Instant);

/// The runtime, one per Cx ([`Cx::global`]).
pub struct Jobs {
    document: Option<Rc<GpuDocument>>,
    kernel: Option<ComputeKernel>,
    items: Vec<Item>,
    /// Ended items whose submissions are still on the GPU: their buffers stay until they finish.
    retiring: Vec<Item>,
    in_flight: Vec<Submission>,
    spare: Vec<Allocation>,
    finished: Vec<(u64, Finished)>,
    calibration: Calibration,
    slots: [bool; TIMER_SLOTS],
    serial: u64,
    next_id: u64,
    interactive: bool,
    pressure: bool,
    /// This GPU draws the UI (or may): background jobs stay about [`BACKGROUND_JOB_MS`].
    draws_ui: bool,
    /// The last [`READBACKS_KEPT`] submissions' submit-to-readback ms, oldest first.
    readbacks: VecDeque<f32>,
    busy: Busy,
    /// Since when an item is unfinished.
    active_since: Option<Instant>,
    tx: mpsc::Sender<Readback>,
    rx: mpsc::Receiver<Readback>,
}
impl Default for Jobs {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self { document: None, kernel: None, items: Vec::new(), retiring: Vec::new(), in_flight: Vec::new(), spare: Vec::new(),
            finished: Vec::new(), calibration: Calibration::default(), slots: [false; TIMER_SLOTS], serial: 0, next_id: 0,
            interactive: false, pressure: false, draws_ui: true, readbacks: VecDeque::new(), busy: Busy::default(), active_since: None, tx, rx }
    }
}

fn timer_name(slot: usize, i: usize) -> &'static str {
    static NAMES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    &NAMES.get_or_init(|| (0..TIMER_SLOTS * JOBS_PER_SUBMISSION).map(|k| format!("sdf job {}.{}", k / JOBS_PER_SUBMISSION, k % JOBS_PER_SUBMISSION)).collect())
        [slot * JOBS_PER_SUBMISSION + i]
}

fn check_dispatch_time(label: LiveId, size: &str, ms: f32) {
    if ms > DISPATCH_LIMIT_MS {
        log!("GPU dispatch over budget: kernel={} {} ms={:.6} limit_ms={}", label, size, ms, DISPATCH_LIMIT_MS);
        debug_assert!(ms <= DISPATCH_LIMIT_MS, "GPU dispatch over budget: kernel={label} {size} ms={ms:.6} limit_ms={DISPATCH_LIMIT_MS}");
    }
}

impl Jobs {
    /// The working-set document every item reserves from (prepared inputs too).
    pub fn document(cx: &mut Cx) -> Rc<GpuDocument> {
        if cx.global::<Jobs>().document.is_none() {
            let doc = Rc::new(cx.gpu_admission().register_working_set("gpu jobs"));
            cx.global::<Jobs>().document = Some(doc);
        }
        cx.global::<Jobs>().document.clone().unwrap()
    }

    /// A drag (or other pointer interaction) runs: every submission stays short.
    pub fn set_interactive(cx: &mut Cx, interactive: bool) { cx.global::<Jobs>().interactive = interactive; }
    /// Someone else needs this GPU now (an AI job on it, interaction the window manager
    /// reports): while set, every submission stays short, as during a drag.
    pub fn set_external_pressure(cx: &mut Cx, pressure: bool) { cx.global::<Jobs>().pressure = pressure; }
    /// Whether this GPU draws the UI (the default): one that draws none (a compute GPU) runs
    /// background jobs at their passes' ceilings.
    pub fn set_draws_ui(cx: &mut Cx, draws_ui: bool) { cx.global::<Jobs>().draws_ui = draws_ui; }

    /// Record every submission from now on (or stop): off by default, a check costs one relaxed load.
    pub fn record_submissions(on: bool) { RECORDING.store(on, Ordering::Relaxed); }
    /// The submissions recorded since the last call.
    pub fn take_submissions() -> Vec<SubmissionRecord> { std::mem::take(&mut *RECORDS.lock().unwrap()) }
    /// What item `id` is, for its submission records.
    pub fn tag(cx: &mut Cx, id: ItemId, tag: ItemTag) {
        if let Some(item) = cx.global::<Jobs>().items.iter_mut().find(|i| i.id == id) { item.tag = Some(tag); }
    }
    /// Item `id`'s output goes to `hook` where its readbacks land ([`Landed`]), before the UI
    /// thread takes them; its [`Finished::Done`] follows as ever. Set it before the item's
    /// output step is submitted (in the turn that started it).
    pub fn on_landed(cx: &mut Cx, id: ItemId, hook: Landed) {
        if let Some(item) = cx.global::<Jobs>().items.iter_mut().find(|i| i.id == id) { item.landed = Some(hook); }
    }
    /// Device bytes not in use: what new items may still reserve.
    pub fn headroom(cx: &mut Cx) -> u64 {
        let budget = cx.gpu_admission().budget();
        budget.device_bytes.saturating_sub(cx.gpu_admission().snapshot().in_use)
    }

    /// Starts an item. Its first buffers are reserved now; refused, nothing is kept and the
    /// owner decides (wait for its other items, split, or a misfit).
    pub fn start(cx: &mut Cx, mut item: WorkItem) -> Result<ItemId, Refusal> {
        if cx.global::<Jobs>().kernel.is_none() {
            let kernel = cx.with_vm(|vm| {
                let module = vm.heap().module(id!(sdf));
                let object = vm.value(module, id!(JobArgs)).as_object().ok_or("JobArgs kernel missing")?;
                ComputeKernel::new(vm, object, "gpu jobs")
            }).map_err(|reason| Refusal { wanted: 0, available: 0, reason })?;
            cx.global::<Jobs>().kernel = Some(kernel);
        }
        let doc = Self::document(cx);
        let mut spare = std::mem::take(&mut cx.global::<Jobs>().spare);
        let committed = item.pool.commit(cx, &doc, &mut spare);
        let jobs = cx.global::<Jobs>();
        jobs.spare = spare;
        committed?;
        jobs.next_id += 1;
        let id = ItemId(jobs.next_id);
        let peak = item.pool.held();
        let claims = item.pool.claims.clone();
        jobs.items.push(Item { id, owner: item.owner, label: item.label, class: item.class, order: item.order, steps: item.steps.into(),
            pool: item.pool, counts: Counts { words: Vec::new(), claims }, in_flight: 0, resuming: false, marks: 0, cursor: 0, round: 0, ladder: 0,
            resume: None, resume_buffers: None, args: None, output: None, output_wanted: false, waiting: None, error: None,
            stats: ItemStats { peak_bytes: peak, ..Default::default() }, taken: None, tag: None, landed: None });
        jobs.active_since.get_or_insert_with(Instant::now);
        SignalToUI::set_ui_signal();
        Ok(id)
    }

    /// Ends an item now (stale, superseded, evicted): no more of its jobs are submitted, and it
    /// is not answered. What is on the GPU finishes with its buffers held.
    pub fn abort(cx: &mut Cx, id: ItemId) {
        let jobs = cx.global::<Jobs>();
        let Some(at) = jobs.items.iter().position(|i| i.id == id) else { return };
        let item = jobs.items.remove(at);
        if item.in_flight > 0 { jobs.retiring.push(item); } else { item.pool.release(&mut jobs.spare); }
    }

    /// Answers for `owner`'s items since the last call.
    pub fn finished(cx: &mut Cx, owner: u64) -> Vec<Finished> {
        let jobs = cx.global::<Jobs>();
        let (mine, rest) = std::mem::take(&mut jobs.finished).into_iter().partition(|(o, _)| *o == owner);
        jobs.finished = rest;
        mine.into_iter().map(|(_, f)| f).collect()
    }

    /// `owner`'s items waiting for memory, with what was refused.
    pub fn waiting(cx: &mut Cx, owner: u64) -> Vec<(ItemId, Refusal)> {
        cx.global::<Jobs>().items.iter().filter(|i| i.owner == owner).filter_map(|i| i.waiting.clone().map(|r| (i.id, r))).collect()
    }
    /// Items of any owner that hold or wait for memory, and whether anything is on the GPU.
    pub fn occupancy(cx: &mut Cx) -> (usize, usize, usize) {
        let jobs = cx.global::<Jobs>();
        (jobs.items.len(), jobs.items.iter().filter(|i| i.waiting.is_some()).count(), jobs.in_flight.len())
    }
    /// Submit-to-readback ms of the last [`READBACKS_KEPT`] submissions of any owner, oldest
    /// first: bounded by job size (the in-flight submissions' GPU time and the wake).
    pub fn readback_latencies(cx: &mut Cx) -> Vec<f32> { cx.global::<Jobs>().readbacks.iter().copied().collect() }
    pub fn has_work(cx: &mut Cx) -> bool { let jobs = cx.global::<Jobs>(); !jobs.items.is_empty() || !jobs.in_flight.is_empty() }
    /// GPU and active ms so far ([`Busy`]).
    pub fn busy(cx: &mut Cx) -> Busy {
        let jobs = cx.global::<Jobs>();
        let mut busy = jobs.busy;
        busy.gpu_ms = gpu_busy_ns() as f64 / 1e6;
        if let Some(since) = jobs.active_since { busy.active_ms += since.elapsed().as_secs_f64() * 1000.0; }
        busy
    }

    /// The device was lost: every item is answered DeviceLost (its owner runs it again after
    /// the reset), and nothing on the old device is waited for.
    pub fn device_lost(cx: &mut Cx) {
        let jobs = cx.global::<Jobs>();
        jobs.in_flight.clear();
        jobs.retiring.clear();
        jobs.spare.clear();
        jobs.slots = [false; TIMER_SLOTS];
        for item in std::mem::take(&mut jobs.items) {
            jobs.finished.push((item.owner, Finished::Failed { id: item.id, error: JobError::DeviceLost }));
        }
        while jobs.rx.try_recv().is_ok() {}
        for i in 0..TIMER_SLOTS * JOBS_PER_SUBMISSION { gpu_frame_timer::take_watched(timer_name(i / JOBS_PER_SUBMISSION, i % JOBS_PER_SUBMISSION)); }
        SignalToUI::set_ui_signal();
    }

    /// Drains readbacks, runs host steps, reserves, and submits. Call on every signal and frame
    /// of every owner; it is cheap when there is nothing to do.
    pub fn pump(cx: &mut Cx) {
        // A lost device, or a new one since the last pump: everything on the old one is
        // answered DeviceLost once, and its owners run it again on the new device.
        let generation = cx.gpu_device_generation();
        let lost = cx.gpu_device_lost();
        let jobs = cx.global::<Jobs>();
        let replaced = jobs.calibration.generation != generation;
        if replaced { jobs.calibration.rates.clear(); jobs.calibration.slots.clear(); jobs.calibration.generation = generation; }
        if (lost || replaced) && (!jobs.items.is_empty() || !jobs.in_flight.is_empty() || !jobs.retiring.is_empty()) { Self::device_lost(cx); }
        if lost { return; }
        Self::drain(cx);
        Self::prepare(cx);
        // One submission per turn: each gets its own command buffer, so the first one's
        // readback does not wait for the second. The next turn submits another.
        if Self::submit_one(cx) && cx.global::<Jobs>().in_flight.len() < IN_FLIGHT { SignalToUI::set_ui_signal(); }
        let jobs = cx.global::<Jobs>();
        if jobs.items.is_empty() && jobs.in_flight.is_empty() {
            if let Some(since) = jobs.active_since.take() { jobs.busy.active_ms += since.elapsed().as_secs_f64() * 1000.0; }
            if jobs.retiring.is_empty() { jobs.spare.clear(); }
        }
    }

    fn drain(cx: &mut Cx) {
        let jobs = cx.global::<Jobs>();
        while let Ok((serial, output, result, at)) = jobs.rx.try_recv() {
            let Some(sub) = jobs.in_flight.iter_mut().find(|s| s.serial == serial) else { continue };
            if output { sub.output_bytes = Some(result) } else { sub.counts = Some(result) }
            sub.arrived = Some(sub.arrived.map_or(at, |a| a.max(at)));
        }
        while let Some(at) = jobs.in_flight.iter().position(|s| s.counts.is_some() && (s.output.is_none() || s.output_bytes.is_some())) {
            let sub = jobs.in_flight.remove(at);
            jobs.complete(sub);
        }
    }

    fn complete(&mut self, sub: Submission) {
        self.slots[sub.slot] = false;
        let arrived = sub.submitted.elapsed().as_secs_f32() * 1000.0;
        if self.readbacks.len() == READBACKS_KEPT { self.readbacks.pop_front(); }
        self.readbacks.push_back(arrived);
        // attributed here, per name: one sample, or none at all
        let mut samples = Vec::with_capacity(sub.jobs.len());
        for i in 0..sub.jobs.len() {
            let taken = gpu_frame_timer::take_watched(timer_name(sub.slot, i));
            samples.push(if taken.len() == 1 { Some(taken[0]) } else { None });
        }
        self.busy.submissions += 1;
        let now = Instant::now();
        if RECORDING.load(Ordering::Relaxed) {
            let jobs = sub.jobs.iter().zip(&samples).map(|(j, ms)| (j.label, j.elements, *ms)).collect();
            RECORDS.lock().unwrap().push(SubmissionRecord { owner: sub.owner, item: sub.item, tag: sub.tag, class: sub.class, submitted: sub.submitted,
                arrived: sub.arrived.unwrap_or(now), taken: now, build_ms: sub.build_ms, gpu_ms: samples.iter().flatten().sum(), jobs });
        }
        if let Some(at) = self.retiring.iter().position(|i| i.id == sub.item) {
            self.retiring[at].in_flight -= 1;
            if self.retiring[at].in_flight == 0 { self.retiring.remove(at).pool.release(&mut self.spare); }
            return;
        }
        let Some(item) = self.items.iter_mut().find(|i| i.id == sub.item) else { return };
        item.in_flight -= 1;
        item.stats.submissions += 1;
        item.stats.readback_ms += arrived;
        item.stats.max_readback_ms = item.stats.max_readback_ms.max(arrived);
        let at = sub.arrived.unwrap_or(now);
        let gpu: f32 = samples.iter().flatten().sum();
        item.stats.build_ms += sub.build_ms;
        item.stats.wait_ms += (at.duration_since(sub.submitted).as_secs_f32() * 1000.0 - gpu).max(0.0);
        item.stats.gap_ms += now.duration_since(at).as_secs_f32() * 1000.0;
        item.taken = Some(now);
        for (job, ms) in sub.jobs.iter().zip(&samples) {
            item.stats.jobs += 1;
            let Some(ms) = *ms else { continue };
            check_dispatch_time(job.label, &job.size, ms);
            item.stats.gpu_ms += ms;
            GPU_BUSY_NS.fetch_add((ms as f64 * 1e6) as u64, std::sync::atomic::Ordering::Relaxed);
            if ms > item.stats.max_job_ms { item.stats.max_job_ms = ms; item.stats.max_job = format!("{} {}", job.label, job.size); }
            match item.stats.kernels.iter_mut().find(|k| k.0 == job.label) {
                Some(k) => { k.1 += job.elements as u64; k.2 += ms; k.3 += 1; }
                None => item.stats.kernels.push((job.label, job.elements as u64, ms, 1)),
            }
            if job.calibrate { self.calibration.observe(job.label, job.units, ms); }
            if let Some((slots, ceiling)) = job.round_slots { self.calibration.observe_round(job.label, slots, ceiling, ms); }
        }
        let counts = match sub.counts.unwrap() {
            Ok(bytes) => bytes,
            Err(e) => { item.error.get_or_insert(JobError::Failed(format!("GPU counts readback failed: {e}"))); return; }
        };
        let words: Vec<u32> = counts.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        if words.get(sub.mark as usize).copied() != Some(sub.serial as u32) {
            log!("gpu jobs: submission {} of {} dropped before completion (mark {:?}); the item runs again", sub.serial, item.label, words.get(sub.mark as usize));
            item.error.get_or_insert(JobError::Dropped);
            return;
        }
        item.counts.words = words;
        if let (Some(o), Some(output)) = (sub.output, sub.output_bytes) {
            let bytes = match output {
                Ok(bytes) => bytes,
                Err(e) => { item.error.get_or_insert(JobError::Failed(format!("GPU output readback failed: {e}"))); return; }
            };
            let Some(len) = o.len.of(&item.counts.words, bytes.len()) else {
                let OutputLen::Counted { word, cap, .. } = o.len else { unreachable!("a fixed output always fits") };
                let buffer = item.pool.entries.get(o.buf.0 as usize).map_or("output", |e| e.name);
                item.error.get_or_insert(JobError::Overflow { buffer, count: item.counts.word(word), cap });
                return;
            };
            item.output = Some(OutputBytes { bytes, len });
        }
        if sub.resume {
            item.resuming = false;
            if item.counts.at(PENDING) == 0 {
                // every element of the front resume is done
                item.steps.pop_front();
                item.resume = None;
            }
        }
    }

    /// Host steps of items with nothing on the GPU, reservations, answers.
    fn prepare(cx: &mut Cx) {
        let doc = Self::document(cx);
        let mut at = 0;
        while at < cx.global::<Jobs>().items.len() {
            let mut item = cx.global::<Jobs>().items.swap_remove(at);
            let mut spare = std::mem::take(&mut cx.global::<Jobs>().spare);
            let result = Self::advance(cx, &doc, &mut item, &mut spare);
            let jobs = cx.global::<Jobs>();
            jobs.spare.append(&mut spare);
            match result {
                Ok(true) => {
                    let stats = std::mem::take(&mut item.stats);
                    let counts = std::mem::replace(&mut item.counts, Counts { words: Vec::new(), claims: Vec::new() });
                    jobs.finished.push((item.owner, Finished::Done { id: item.id, counts, output: item.output.take(), stats }));
                    item.pool.release(&mut jobs.spare);
                }
                Ok(false) => { jobs.items.push(item); let n = jobs.items.len() - 1; jobs.items.swap(at, n); at += 1; }
                Err(error) => {
                    jobs.finished.push((item.owner, Finished::Failed { id: item.id, error }));
                    if item.in_flight > 0 { jobs.retiring.push(item); } else { item.pool.release(&mut jobs.spare); }
                }
            }
        }
    }

    /// Ok(true): the item is complete.
    fn advance(cx: &mut Cx, doc: &GpuDocument, item: &mut Item, spare: &mut Vec<Allocation>) -> Result<bool, JobError> {
        if let Some(e) = item.error.take() { return Err(e); }
        if item.in_flight > 0 { return Ok(false); }
        loop {
            if item.waiting.is_some() || item.pool.pending_bytes() > 0 || !item.pool.uploads.is_empty() {
                match item.pool.commit(cx, doc, spare) {
                    Ok(()) => { item.waiting = None; item.stats.peak_bytes = item.stats.peak_bytes.max(item.pool.held()); }
                    Err(refusal) => {
                        if refusal.wanted == 0 { return Err(JobError::Failed(refusal.reason)); }
                        item.waiting = Some(refusal);
                        return Ok(false);
                    }
                }
            }
            // A counted pass anywhere in the queue: its job arguments are held before the first
            // submission, which may walk fixed passes into it.
            let counted = |s: &Step| matches!(s, Step::Pass(Pass { elements: Elements::Counted { .. }, .. }) | Step::Repeat(Pass { elements: Elements::Counted { .. }, .. }, _));
            if item.args.is_none() && item.steps.iter().any(counted) {
                item.args = Some(item.pool.add("job args", JOBS_PER_SUBMISSION as u64 * 16, GpuBufferFlags { readback: false, indirect: true }, Held::Pending));
                continue;
            }
            let rounds = match item.steps.front() { Some(Step::Repeat(_, r)) => *r, _ => 1 };
            match item.steps.front() {
                None => return Ok(!item.output_wanted || item.output.is_some()),
                // A pass with nothing left (no elements, or its last round done) goes here: what
                // follows it (a host step, a resume) runs in this prepare, never waiting on a
                // submission that has no job to carry.
                Some(Step::Pass(p)) | Some(Step::Repeat(p, _)) if item.round >= rounds || { let (first, end) = p.elements.span(); end <= first } => {
                    item.steps.pop_front(); item.cursor = 0; item.round = 0; item.ladder = 0;
                }
                Some(Step::Host(_)) => {
                    let Some(Step::Host(host)) = item.steps.pop_front() else { unreachable!() };
                    let steps = host(&item.counts, &mut item.pool)?;
                    for step in steps.into_iter().rev() { item.steps.push_front(step); }
                }
                Some(Step::Checkpoint) => { item.steps.pop_front(); }
                Some(Step::Resume { pass, slots_ceiling }) if item.resume.is_none() => {
                    let (first, end) = pass.elements.span();
                    if end <= first { item.steps.pop_front(); continue; }
                    let slots = cx.global::<Jobs>().calibration.slots(pass.label, *slots_ceiling).min((end - first).next_multiple_of(64));
                    let continuations = match item.resume_buffers {
                        Some((c, n)) if n >= slots => c,
                        _ => {
                            if let Some((c, _)) = item.resume_buffers.take() { item.pool.free(c); }
                            // slot states, then the two active lists
                            let c = item.pool.buffer("resume state", slots as u64 * (SLOT_WORDS + 2) * 4);
                            item.resume_buffers = Some((c, slots));
                            c
                        }
                    };
                    item.resume = Some(ResumeState { slots, ping: 0, seeded: false, continuations });
                }
                Some(_) => return Ok(false),
            }
        }
    }

    /// The highest-priority item that can submit now gets one submission.
    fn submit_one(cx: &mut Cx) -> bool {
        let jobs = cx.global::<Jobs>();
        if jobs.in_flight.len() >= IN_FLIGHT { return false; }
        let Some(slot) = jobs.slots.iter().position(|s| !*s) else { return false };
        let runnable = |i: &Item| i.error.is_none() && i.waiting.is_none() && !i.resuming && i.pool.pending_bytes() == 0
            && match i.steps.front() {
                Some(Step::Pass(p)) | Some(Step::Repeat(p, _)) => !matches!(p.elements, Elements::Counted { .. }) || i.args.is_some(),
                // a resume's buffers are prepared with nothing of the item on the GPU
                Some(Step::Resume { .. }) => i.resume.is_some(),
                Some(Step::Output(_)) => true,
                None => false,
                Some(Step::Host(_)) | Some(Step::Checkpoint) => false,
            };
        let top = jobs.items.iter().map(|i| i.class).max();
        let Some(at) = jobs.items.iter().enumerate().filter(|(_, i)| runnable(i))
            .max_by_key(|(_, i)| (i.class, std::cmp::Reverse(i.order), std::cmp::Reverse(i.id.0))).map(|(at, _)| at) else { return false };
        let class = jobs.items[at].class;
        // a lower class never holds both slots while higher work is pending
        if let Some(top) = top.filter(|&top| class < top) { if jobs.in_flight.iter().any(|s| s.class < top) { return false; } }
        let short = jobs.interactive || jobs.pressure || top.is_some_and(|c| c >= JobClass::Visible);
        let budget = if short { SHORT_MS } else { LONG_MS };
        let job_ms = if short { Some(JOB_MS) } else if jobs.draws_ui { Some(BACKGROUND_JOB_MS) } else { None };
        let mut item = jobs.items.swap_remove(at);
        let built = Self::build(cx, &mut item, slot, budget, job_ms);
        let jobs = cx.global::<Jobs>();
        let submitted = match built {
            Ok(Some(sub)) => { jobs.slots[slot] = true; item.in_flight += 1; jobs.in_flight.push(sub); true }
            Ok(None) => false,
            Err(e) => { item.error = Some(e); false }
        };
        jobs.items.push(item);
        let n = jobs.items.len() - 1;
        jobs.items.swap(at, n);
        submitted
    }

    fn resolve(item: &Item, p: &Pass, resume: Option<&ResumeState>) -> Result<Vec<GpuBuffer>, JobError> {
        p.kernel.layout().buffers.iter().zip(&p.buffers).map(|(binding, &buf)| {
            let buf = if binding.name == id!(counts) { BufRef::COUNTS }
                else if let (Some(r), true) = (resume, binding.name == id!(continuations)) { r.continuations }
                else { buf };
            item.pool.gpu(buf).ok_or_else(|| JobError::Failed(if item.pool.deferred_unsized(buf) {
                format!("{}: binding {} is deferred buffer {}, bound before its checkpoint sized it", p.label, binding.name, item.pool.name(buf))
            } else { format!("{}: binding {} has no buffer", p.label, binding.name) }))
        }).collect()
    }

    fn build(cx: &mut Cx, item: &mut Item, slot: usize, budget: f32, job_ms: Option<f32>) -> Result<Option<Submission>, JobError> {
        let started = Instant::now();
        let fail = |e: String| JobError::Failed(e);
        let kernel = cx.global::<Jobs>().kernel.clone().expect("lowered at start");
        let counts = item.pool.gpu(BufRef::COUNTS).ok_or_else(|| fail("item counts are not held".into()))?;
        // late samples of this name set belong to nobody
        for i in 0..JOBS_PER_SUBMISSION { gpu_frame_timer::take_watched(timer_name(slot, i)); }
        let mut pass = ComputePass::new("gpu jobs", None);
        let mut jobs: Vec<JobRecord> = Vec::new();
        let mut spent = 0.0f32;
        let mut resume = false;
        let mut output = None;
        cx.global::<Jobs>().serial += 1;
        let serial = cx.global::<Jobs>().serial;
        let room = JOBS_PER_SUBMISSION - 1;
        let end = 'steps: loop {
            if jobs.len() >= room { break End::Jobs; }
            if spent >= budget { break End::Budget; }
            let cal = &cx.global::<Jobs>().calibration;
            match item.steps.front() {
                None => break End::Last,
                Some(Step::Host(_)) => break End::Host,
                Some(Step::Checkpoint) => { item.steps.pop_front(); break End::Checkpoint; }
                Some(Step::Output(o)) => {
                    let o = *o;
                    item.steps.pop_front();
                    let gpu = item.pool.gpu(o.buf).ok_or_else(|| fail("output buffer is not held".into()))?;
                    let bytes = o.read_bytes().filter(|&b| b <= gpu.size())
                        .ok_or_else(|| fail(format!("output of {:?} past its buffer of {} bytes", o.len, gpu.size())))?;
                    output = Some((o, gpu, bytes));
                    item.output_wanted = true;
                    break End::Output;
                }
                Some(Step::Pass(p)) | Some(Step::Repeat(p, _)) => {
                    let rounds = if let Some(Step::Repeat(_, r)) = item.steps.front() { *r } else { 1 };
                    let (first, end) = p.elements.span();
                    if item.round >= rounds || end <= first {
                        item.steps.pop_front(); item.cursor = 0; item.round = 0; item.ladder = 0;
                        continue;
                    }
                    let cursor = item.cursor.max(first);
                    let elements = cal.elements(p, item.ladder, job_ms).min(end - cursor);
                    let units = elements as f64 * p.cost as f64;
                    let est = cal.est_ms(p.label, units);
                    if !jobs.is_empty() && spent + est > budget { break End::Budget; }
                    let buffers = Self::resolve(item, p, None)?;
                    let mut params = p.params.clone();
                    let z = if rounds > 1 || matches!(item.steps.front(), Some(Step::Repeat(..))) { item.round } else { p.zw[0] };
                    let range = [cursor, elements, z, p.zw[1]];
                    params.set(id!(dispatch_range), &range.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()).map_err(|e| fail(format!("{}: {e}", p.label)))?;
                    let name = timer_name(slot, jobs.len());
                    match p.elements {
                        Elements::Counted { word, .. } => {
                            // this job's arguments from the count, then the job
                            let args = item.pool.gpu(item.args.ok_or_else(|| fail("job args are not held".into()))?).ok_or_else(|| fail("job args are not held".into()))?;
                            let at = jobs.len() as u32;
                            let mut ap = kernel.params();
                            ap.set(id!(spec), &[word.0, elements, 1, cursor].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()).map_err(fail)?;
                            ap.set_u32(id!(at), at).map_err(fail)?;
                            ap.set_u32(id!(lanes), p.lanes).map_err(fail)?;
                            pass.dispatch(&kernel, [1, 1, 1], &ap, &Self::job_args_buffers(&kernel, &counts, &args), &[]).map_err(fail)?;
                            pass.dispatch_indirect_timed(name, &p.kernel, &args, at as u64 * 16, &params, &buffers, &[]).map_err(fail)?;
                        }
                        _ => pass.dispatch_timed(name, &p.kernel, [(elements as u64 * p.lanes as u64).div_ceil(64) as u32, 1, 1], &params, &buffers, &[]).map_err(fail)?,
                    }
                    jobs.push(JobRecord { label: p.label, units, elements, calibrate: true, round_slots: None,
                        size: format!("elements={cursor}..{} cost={} ceiling={}", cursor + elements, p.cost, p.ceiling) });
                    spent += est;
                    item.ladder += 1;
                    item.cursor = cursor + elements;
                    if item.cursor >= end { item.cursor = 0; item.round += 1; item.ladder = 0; }
                }
                // a resume's buffers are prepared with nothing of the item on the GPU: one reached
                // from earlier steps starts the next submission
                Some(Step::Resume { .. }) if item.resume.is_none() => break End::Resume,
                Some(Step::Resume { pass: p, slots_ceiling }) => {
                    let ceiling = *slots_ceiling;
                    let (first, end) = p.elements.span();
                    let r = item.resume.as_mut().expect("prepared before submission");
                    let units = r.slots as f64 * p.cost as f64;
                    let est = cal.est_ms(p.label, units);
                    if !jobs.is_empty() && spent + est > budget { break End::Budget; }
                    let rounds = if cal.rates.contains_key(&p.label) { (((budget - spent) / est) as u32).clamp(1, MAX_ROUNDS) } else { 1 }
                        .min(((room - jobs.len() - usize::from(!r.seeded)) / 2) as u32);
                    if rounds == 0 { break End::Jobs; }
                    let r = item.resume.as_ref().unwrap();
                    let (slots, seeded, ping0) = (r.slots, r.seeded, r.ping);
                    let rs = ResumeState { slots, ping: ping0, seeded, continuations: r.continuations };
                    let buffers = Self::resolve(item, p, Some(&rs))?;
                    let mut issue = |range: [u32; 4], indirect: bool, label: LiveId, calibrate: bool, jobs: &mut Vec<JobRecord>| -> Result<(), JobError> {
                        let mut params = p.params.clone();
                        params.set(id!(dispatch_range), &range.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()).map_err(|e| fail(format!("{}: {e}", p.label)))?;
                        let name = timer_name(slot, jobs.len());
                        if indirect { pass.dispatch_indirect_timed(name, &p.kernel, &counts, ROUND_ARGS, &params, &buffers, &[]).map_err(fail)?; }
                        else { pass.dispatch_timed(name, &p.kernel, [slots.div_ceil(64), 1, 1], &params, &buffers, &[]).map_err(fail)?; }
                        jobs.push(JobRecord { label, units: if calibrate { units } else { 0.0 }, elements: if calibrate { slots } else { 0 }, calibrate,
                            round_slots: calibrate.then_some((slots, ceiling)),
                            size: format!("slots={slots} elements={first}..{end} work_budget={} range={range:?}", p.cost) });
                        Ok(())
                    };
                    if !seeded { issue([end, slots, first, 2], false, label("resume seed"), false, &mut jobs)?; }
                    let mut ping = ping0;
                    for _ in 0..rounds {
                        issue([ping * slots, ping, 0, 0], true, p.label, true, &mut jobs)?;
                        issue([0, ping, 0, 3], true, label("resume compact"), false, &mut jobs)?;
                        ping = 1 - ping;
                    }
                    let r = item.resume.as_mut().unwrap();
                    r.seeded = true;
                    r.ping = ping;
                    resume = true;
                    break 'steps End::Resume;
                }
            }
        };
        if jobs.is_empty() && output.is_none() { return Ok(None); }
        let ends = &mut item.stats.ends;
        match end {
            End::Budget => ends.budget += 1, End::Jobs => ends.jobs += 1, End::Checkpoint => ends.checkpoint += 1, End::Host => ends.host += 1,
            End::Resume => ends.resume += 1, End::Output => ends.output += 1, End::Last => ends.last += 1,
        }
        // the completion mark, then the readbacks
        let args = item.args.and_then(|a| item.pool.gpu(a)).unwrap_or_else(|| counts.clone());
        let mut mark = kernel.params();
        let mark_word = COMPLETION[item.marks as usize % IN_FLIGHT];
        mark.set(id!(spec), &[mark_word, 0, 0, serial as u32].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()).map_err(fail)?;
        pass.dispatch(&kernel, [1, 1, 1], &mark, &Self::job_args_buffers(&kernel, &counts, &args), &[]).map_err(fail)?;
        pass.submit(cx);
        let tx = cx.global::<Jobs>().tx.clone();
        let words = item.pool.counted_words as u64 * 4;
        // the output goes on where it lands: its counts and bytes meet on the readback's thread
        let meet = match (&output, item.landed.take()) {
            (Some((o, _, bytes)), Some(hook)) => Some(Arc::new(Mutex::new(Meet {
                output: (*bytes == 0).then(|| Arc::from(&[][..])),
                landing: Some(Landing { hook, mark: mark_word, serial, len: o.len, claims: item.counts.claims.clone(), stats: item.stats.clone() }),
                ..Default::default()
            }))),
            (_, hook) => { item.landed = hook; None }
        };
        let read = |cx: &mut Cx, buffer: &GpuBuffer, bytes: u64, is_output: bool| {
            let (tx, meet) = (tx.clone(), meet.clone());
            cx.read_gpu_buffer(buffer, 0, bytes, move |result| {
                let result = result.map_err(|e| format!("{e:?}")).and_then(|b| if b.len() as u64 == bytes { Ok(b) } else { Err(format!("{} bytes, expected {bytes}", b.len())) });
                if let (Some(meet), Ok(b)) = (&meet, &result) { Meet::arrive(meet, is_output, b.clone()); }
                let _ = tx.send((serial, is_output, result, Instant::now()));
                SignalToUI::set_ui_signal();
            }).map_err(|e| fail(format!("GPU readback refused: {e:?}")))
        };
        read(cx, &counts, words, false)?;
        // an empty output reads nothing
        let mut output_bytes = None;
        match &output {
            Some((_, buffer, bytes)) if *bytes > 0 => read(cx, buffer, *bytes, true)?,
            Some(_) => output_bytes = Some(Ok(Arc::from(&[][..]))),
            None => {}
        }
        if resume { item.resuming = true; }
        item.marks = item.marks.wrapping_add(1);
        if let Some(taken) = item.taken { item.stats.turn_ms += started.duration_since(taken).as_secs_f32() * 1000.0; }
        let submitted = Instant::now();
        Ok(Some(Submission { serial, owner: item.owner, tag: item.tag, mark: mark_word, item: item.id, class: item.class, slot, jobs, resume, output: output.map(|o| o.0),
            counts: None, output_bytes, submitted, build_ms: submitted.duration_since(started).as_secs_f32() * 1000.0, arrived: None }))
    }

    fn job_args_buffers(kernel: &ComputeKernel, counts: &GpuBuffer, args: &GpuBuffer) -> Vec<GpuBuffer> {
        kernel.layout().buffers.iter().map(|b| if b.name == id!(counts) { counts.clone() } else { args.clone() }).collect()
    }
}
