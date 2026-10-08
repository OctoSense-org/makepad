//! Main-thread frame monitor feeding the PerfGraph widget (widgets crate).
//!
//! A fixed ring of per-frame samples: the paint-to-paint gap (frame pacing —
//! spikes here are the hiccups the user sees) plus per-channel CPU times.
//! Built-in channels cover the platform (event dispatch, script GC, pass
//! encode, drawable wait); apps register their own channels for anything
//! else they want plotted (physics, script tick, audio…):
//!
//! ```ignore
//! let ch = cx.perf_monitor_channel("physics", 0x6aa9ff);
//! // ...
//! cx.perf_monitor_add(ch, t0.elapsed().as_micros() as u64);
//! ```
//!
//! Collection is off until something (normally the PerfGraph widget) calls
//! `set_enabled(true)`; disabled adds are a single branch.

use std::{cell::Cell, collections::HashMap, rc::Rc};

pub const PERF_MONITOR_HISTORY: usize = 240;
pub const PERF_MONITOR_MAX_CHANNELS: usize = 12;
pub const PERF_MONITOR_MAX_WORK: usize = 256;

/// Aggregate by operation and static component type, never by user content or
/// instance ID. Scrolling through new widgets cannot grow this table unbounded.
#[derive(Clone)]
pub struct PerfWorkSample {
    pub operation: &'static str,
    pub component: &'static str,
    pub calls: u64,
    pub total_ns: u64,
    pub self_ns: u64,
    pub max_ns: u64,
}

/// A main-thread measurement. Finish descendants before their parents.
pub struct PerfWorkToken {
    slot: usize,
    started: f64,
    completed_self_ns: u64,
    generation: u64,
    owner: Rc<Cell<u32>>,
}

/// GPU completion handlers run off-thread; they park each presented frame's
/// GPU time here and the next `frame_boundary` folds it into the ring.
static GPU_ACCUM_US: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static GPU_COLLECT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Call from a GPU command-buffer completion handler (any thread) with one
/// presented frame's GPU interval in seconds.
pub fn perf_gpu_frame_completed(seconds: f64) {
    use std::sync::atomic::Ordering;
    if GPU_COLLECT.load(Ordering::Relaxed) && seconds.is_finite() && seconds > 0.0 {
        GPU_ACCUM_US.fetch_add((seconds * 1e6) as u32, Ordering::Relaxed);
    }
}

/// Index of a registered channel; hand out once, add to it every frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerfChannel(pub usize);

/// Built-in channels, registered in `PerfMonitor::default()`.
pub const PERF_CHANNEL_EVENT: PerfChannel = PerfChannel(0);
/// Splash/script VM execution — embedders add their eval/call time here.
pub const PERF_CHANNEL_SCRIPT: PerfChannel = PerfChannel(1);
pub const PERF_CHANNEL_GC: PerfChannel = PerfChannel(2);
pub const PERF_CHANNEL_DRAW: PerfChannel = PerfChannel(3);
pub const PERF_CHANNEL_DRAWABLE_WAIT: PerfChannel = PerfChannel(4);
/// GPU time of a presented frame (command-buffer start→end, completion
/// handler thread). Runs CONCURRENT with the CPU channels — read it as its
/// own series, not as part of the main-thread total.
pub const PERF_CHANNEL_GPU: PerfChannel = PerfChannel(5);

#[derive(Clone, Copy, Default)]
pub struct PerfMonitorFrame {
    /// Time between this paint and the previous one, milliseconds.
    pub gap_ms: f32,
    /// Per-channel CPU time this frame, microseconds.
    pub channel_us: [u32; PERF_MONITOR_MAX_CHANNELS],
}

#[derive(Clone)]
pub struct PerfChannelInfo {
    pub name: String,
    /// 0xRRGGBB plot color hint.
    pub color: u32,
}

pub struct PerfMonitor {
    enabled: bool,
    channels: Vec<PerfChannelInfo>,
    ring: Vec<PerfMonitorFrame>,
    at: usize,
    cur: PerfMonitorFrame,
    last_frame_time: Option<f64>,
    /// Active event-dispatch depth, used to attribute nested channel timing.
    pub(crate) event_depth: Rc<Cell<u32>>,
    /// Time app channels attributed while inside an event dispatch; deducted
    /// from the "event" channel so the stacked plot doesn't double-count.
    event_deduct: u32,
    /// Window repaints seen since enabling (see `frames_painted`).
    frames_painted: u64,
    work: Vec<PerfWorkSample>,
    work_index: HashMap<(&'static str, &'static str), usize>,
    completed_self_ns: u64,
    work_generation: u64,
    work_overflow: u64,
}

impl Default for PerfMonitor {
    fn default() -> Self {
        Self {
            enabled: false,
            channels: vec![
                PerfChannelInfo { name: "event".into(), color: 0x4fd06a },
                PerfChannelInfo { name: "script".into(), color: 0x58b6ff },
                PerfChannelInfo { name: "gc".into(), color: 0xd0c24f },
                PerfChannelInfo { name: "draw".into(), color: 0xff9a4f },
                PerfChannelInfo { name: "wait".into(), color: 0xe05555 },
                PerfChannelInfo { name: "gpu".into(), color: 0xb08cff },
            ],
            ring: Vec::new(),
            at: 0,
            cur: Default::default(),
            last_frame_time: None,
            event_depth: Rc::new(Cell::new(0)),
            event_deduct: 0,
            frames_painted: 0,
            work: Vec::new(),
            work_index: HashMap::new(),
            completed_self_ns: 0,
            work_generation: 0,
            work_overflow: 0,
        }
    }
}

impl PerfMonitor {
    pub fn set_enabled(&mut self, on: bool) {
        if self.enabled != on {
            self.work_generation = self.work_generation.wrapping_add(1);
        }
        self.enabled = on;
        GPU_COLLECT.store(on, std::sync::atomic::Ordering::Relaxed);
        if !on {
            self.last_frame_time = None;
            self.cur = Default::default();
            self.event_deduct = 0;
            GPU_ACCUM_US.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    fn work_slot(&mut self, operation: &'static str, component: &'static str) -> Option<usize> {
        if !self.enabled {
            return None;
        }
        let key = (operation, component);
        if let Some(slot) = self.work_index.get(&key) {
            return Some(*slot);
        }
        if self.work.len() == PERF_MONITOR_MAX_WORK {
            self.work_overflow = self.work_overflow.saturating_add(1);
            return None;
        }
        let slot = self.work.len();
        self.work.push(PerfWorkSample {
            operation,
            component,
            calls: 0,
            total_ns: 0,
            self_ns: 0,
            max_ns: 0,
        });
        self.work_index.insert(key, slot);
        Some(slot)
    }

    /// Disabled measurements neither read the clock nor allocate. Component
    /// names should be static type names, not per-message identifiers.
    pub fn begin_work(
        &mut self,
        operation: &'static str,
        component: &'static str,
    ) -> Option<PerfWorkToken> {
        let slot = self.work_slot(operation, component)?;
        Some(PerfWorkToken {
            slot,
            started: crate::cx::Cx::monotonic_now(),
            completed_self_ns: self.completed_self_ns,
            generation: self.work_generation,
            owner: self.event_depth.clone(),
        })
    }

    pub fn end_work(&mut self, token: Option<PerfWorkToken>) {
        let Some(token) = token else { return };
        if !self.enabled
            || token.generation != self.work_generation
            || !Rc::ptr_eq(&token.owner, &self.event_depth)
        {
            return;
        }
        let elapsed = ((crate::cx::Cx::monotonic_now() - token.started).max(0.0) * 1e9) as u64;
        // Descendant exclusive times sum to their whole subtree exactly once.
        let own = elapsed.saturating_sub(
            self.completed_self_ns
                .saturating_sub(token.completed_self_ns),
        );
        let sample = &mut self.work[token.slot];
        sample.calls = sample.calls.saturating_add(1);
        sample.total_ns = sample.total_ns.saturating_add(elapsed);
        sample.self_ns = sample.self_ns.saturating_add(own);
        sample.max_ns = sample.max_ns.max(elapsed);
        self.completed_self_ns = self.completed_self_ns.saturating_add(own);
    }

    /// Count work without timing it. Counter rows have zero timing fields.
    pub fn count_work(&mut self, operation: &'static str, component: &'static str, count: u64) {
        if let Some(slot) = self.work_slot(operation, component) {
            self.work[slot].calls = self.work[slot].calls.saturating_add(count);
        }
    }

    pub fn work(&self) -> &[PerfWorkSample] {
        &self.work
    }

    /// Measurements omitted because the bounded operation/type table is full.
    pub fn work_overflow(&self) -> u64 {
        self.work_overflow
    }

    /// Number of window repaints (frame boundaries) recorded while enabled.
    /// Lets a scripted driver pace itself to presented frames instead of
    /// queueing pass renders faster than the GPU retires them.
    pub fn frames_painted(&self) -> u64 {
        self.frames_painted
    }

    /// Register (or find by name) an app channel. Indexes are stable for the
    /// life of the process; past MAX_CHANNELS you share the last slot.
    pub fn channel(&mut self, name: &str, color: u32) -> PerfChannel {
        if let Some(index) = self.channels.iter().position(|c| c.name == name) {
            return PerfChannel(index);
        }
        if self.channels.len() >= PERF_MONITOR_MAX_CHANNELS {
            return PerfChannel(PERF_MONITOR_MAX_CHANNELS - 1);
        }
        self.channels.push(PerfChannelInfo { name: name.into(), color });
        PerfChannel(self.channels.len() - 1)
    }

    pub fn channels(&self) -> &[PerfChannelInfo] {
        &self.channels
    }

    pub fn add(&mut self, channel: PerfChannel, us: u64) {
        if !self.enabled {
            return;
        }
        let mut us = us as u32;
        if channel == PERF_CHANNEL_EVENT {
            // "event" is what's left of the dispatch after the app attributed
            // its own channels (script, physics, …) inside it.
            us = us.saturating_sub(self.event_deduct);
            self.event_deduct = 0;
        } else if self.event_depth.get() > 0 {
            self.event_deduct = self.event_deduct.saturating_add(us);
        }
        let slot = &mut self.cur.channel_us[channel.0.min(PERF_MONITOR_MAX_CHANNELS - 1)];
        *slot = slot.saturating_add(us);
    }

    /// Close the frame being accumulated and start the next. Called by the
    /// platform at the start of every window repaint.
    pub fn frame_boundary(&mut self, time: f64) {
        if !self.enabled {
            return;
        }
        self.frames_painted += 1;
        if self.ring.is_empty() {
            self.ring.resize(PERF_MONITOR_HISTORY, Default::default());
        }
        // Fold in GPU time completed since the last boundary (one frame late
        // by construction — fine for a monitor).
        let gpu_us = GPU_ACCUM_US.swap(0, std::sync::atomic::Ordering::Relaxed);
        if gpu_us > 0 {
            let slot = &mut self.cur.channel_us[PERF_CHANNEL_GPU.0];
            *slot = slot.saturating_add(gpu_us);
        }
        if let Some(last) = self.last_frame_time {
            self.cur.gap_ms = ((time - last) * 1000.0) as f32;
            self.ring[self.at] = self.cur;
            self.at = (self.at + 1) % PERF_MONITOR_HISTORY;
        }
        self.last_frame_time = Some(time);
        self.cur = Default::default();
    }

    /// Copy the history oldest→newest. Empty until enabled + first frames.
    pub fn read(&self, out: &mut Vec<PerfMonitorFrame>) {
        out.clear();
        if self.ring.is_empty() {
            return;
        }
        for i in 0..PERF_MONITOR_HISTORY {
            out.push(self.ring[(self.at + i) % PERF_MONITOR_HISTORY]);
        }
    }
}
