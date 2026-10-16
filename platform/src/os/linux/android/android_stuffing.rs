//! Buffer-stuffing recovery for the Choreographer-paced GLES window.
//!
//! The render thread paints on each vsync beat and queues its buffer most of
//! a refresh later (about 12 of 16.7 ms on a OnePlus 6), close to
//! SurfaceFlinger's latch. One slow frame (often the first after idle, or a
//! touch-down) misses the latch, and the next on-time frame queues behind it:
//! from then on every frame waits an extra refresh in the buffer queue, until
//! the app goes idle. While a finger drags content, that is a whole extra
//! frame of lag behind it for the rest of the gesture.
//!
//! Each frame's display time comes back through
//! EGL_ANDROID_get_frame_timestamps a few frames later. A frame is late when
//! it reached the screen half a refresh or more after its beat's expected
//! presentation time (Choreographer FrameData, API 33+; before that, the
//! shortest recent beat-to-screen time stands in). Two late frames from
//! consecutive beats mean the queue holds a frame too many: skipping one beat
//! drains it, and the frames after it are on time again. A drained queue has
//! no spare frame to absorb a slow one, so recovery runs only while a finger
//! is down, where the lag shows; a stuffed queue during a fling's coast is
//! invisible and is left alone.
//!
//! `setprop debug.makepad.stuffing off` turns it off (read once, at the first
//! beat); the `android.stuffing` trace topic logs each detection and skip.
use super::super::egl_sys::{EGLDisplay, EGLSurface, LibEgl};
use std::{collections::VecDeque, ffi::CStr};

const EGL_EXTENSIONS: i32 = 0x3055;
const EGL_TIMESTAMPS_ANDROID: i32 = 0x3430;
const EGL_DISPLAY_PRESENT_TIME_ANDROID: i32 = 0x343a;
const EGL_TIMESTAMP_PENDING_ANDROID: i64 = -2;
/// Recent beat-to-screen times that set the on-time baseline (ten seconds
/// at 60 Hz; idle frames count too, and they are never stuffed).
const BASELINE_FRAMES: usize = 600;

type NextId = unsafe extern "C" fn(EGLDisplay, EGLSurface, *mut u64) -> u32;
type Timestamps = unsafe extern "C" fn(EGLDisplay, EGLSurface, u64, i32, *const i32, *mut i64) -> u32;
type Supported = unsafe extern "C" fn(EGLDisplay, EGLSurface, i32) -> u32;

#[derive(Default)]
struct StuffingRecovery {
    initialized: bool,
    functions: Option<(NextId, Timestamps)>,
    /// Swapped frames waiting for their display time: (frame id, beat,
    /// the beat's expected presentation time or 0).
    pending: VecDeque<(u64, i64, i64)>,
    latencies: VecDeque<i64>,
    /// Frames from beats at or before this were rendered before the last
    /// skip and still carry the old queue; they never trigger another.
    armed_after: i64,
    skip_next_beat: bool,
    skips: u64,
    /// Consecutive late frames from consecutive beats, and the last observed beat.
    late_streak: u32,
    last_observed_beat: i64,
    /// A finger is on the screen. Only then does lag show (the content
    /// trails the finger); after lift-off a stuffed queue is invisible and
    /// its extra frame absorbs a slow one, so the queue is left alone.
    touch_active: bool,
    /// The beat being painted, the one before it, and recent beat intervals
    /// (their minimum is the refresh period).
    beat: i64,
    expected: i64,
    last_beat: i64,
    intervals: VecDeque<i64>,
    swapped_this_beat: bool,
}

impl StuffingRecovery {
    fn period(&self) -> i64 {
        self.intervals.iter().copied().min().unwrap_or(16_666_667)
    }

    unsafe fn initialize(&mut self, egl: &LibEgl, display: EGLDisplay, surface: EGLSurface) {
        if self.initialized {
            return;
        }
        self.initialized = true;
        let extensions = (egl.eglQueryString.unwrap())(display, EGL_EXTENSIONS);
        if extensions.is_null()
            || !CStr::from_ptr(extensions).to_bytes().split(|b| *b == b' ').any(|s| s == b"EGL_ANDROID_get_frame_timestamps")
        {
            return;
        }
        let next = (egl.eglGetProcAddress.unwrap())(c"eglGetNextFrameIdANDROID".as_ptr());
        let timestamps = (egl.eglGetProcAddress.unwrap())(c"eglGetFrameTimestampsANDROID".as_ptr());
        let supported = (egl.eglGetProcAddress.unwrap())(c"eglGetFrameTimestampSupportedANDROID".as_ptr());
        if next.is_null() || timestamps.is_null() || supported.is_null() {
            return;
        }
        let supported: Supported = std::mem::transmute(supported);
        if supported(display, surface, EGL_DISPLAY_PRESENT_TIME_ANDROID) == 0 {
            return;
        }
        if (egl.eglSurfaceAttrib.unwrap())(display, surface, EGL_TIMESTAMPS_ANDROID, 1) == 0 {
            return;
        }
        self.functions = Some((std::mem::transmute(next), std::mem::transmute(timestamps)));
    }

    /// Before swapping the frame painted for the vsync `beat` (CLOCK_MONOTONIC
    /// ns, 0 when unknown): read back the display times that have arrived and
    /// note this frame's id.
    unsafe fn before_swap(&mut self, egl: &LibEgl, display: EGLDisplay, surface: EGLSurface) {
        let (beat, expected, period) = (self.beat, self.expected, self.period());
        self.initialize(egl, display, surface);
        let Some((next, timestamps)) = self.functions else { return };
        while let Some(&(id, frame_beat, frame_expected)) = self.pending.front() {
            let mut present = 0i64;
            if timestamps(display, surface, id, 1, &EGL_DISPLAY_PRESENT_TIME_ANDROID, &mut present) == 0 {
                self.pending.pop_front();
                continue;
            }
            if present == EGL_TIMESTAMP_PENDING_ANDROID {
                break;
            }
            self.pending.pop_front();
            if present > 0 && frame_beat > 0 {
                self.observe(frame_beat, frame_expected, present, period);
            }
        }
        if self.pending.len() >= 16 {
            self.pending.pop_front();
        }
        let mut id = 0u64;
        if beat > 0 && next(display, surface, &mut id) != 0 {
            self.pending.push_back((id, beat, expected));
        }
    }

    /// A frame painted for `beat` reached the screen at `present`. Late:
    /// half a refresh or more after the beat's expected presentation time
    /// (Choreographer FrameData, API 33+), or before that API after the
    /// shortest recent beat-to-screen time.
    fn observe(&mut self, beat: i64, expected: i64, present: i64, period: i64) {
        let latency = present - beat;
        if latency <= 0 || latency > 8 * period {
            return;
        }
        let late = if expected > 0 {
            present - expected >= period / 2
        } else {
            self.latencies.push_back(latency);
            if self.latencies.len() > BASELINE_FRAMES {
                self.latencies.pop_front();
            }
            latency >= *self.latencies.iter().min().unwrap() + period / 2
        };
        // One late frame need not stuff the queue (an idle beat after it
        // drains it), and its display time arrives frames later, when the
        // queue may already be empty again. Stuffing makes EVERY following
        // frame late, so act only on two in a row from consecutive beats.
        let consecutive = beat - self.last_observed_beat <= period + period / 2;
        self.last_observed_beat = beat;
        self.late_streak = if late { if consecutive { self.late_streak + 1 } else { 1 } } else { 0 };
        if self.touch_active && beat > self.armed_after && self.late_streak >= 2 {
            self.skip_next_beat = true;
            self.late_streak = 0;
            if crate::makepad_error_log::trace_enabled("android.stuffing") {
                crate::log!("[android.stuffing] late beat={} present={} expected={} latency={}", beat, present, expected, latency);
            }
        }
    }

    /// Whether to leave the beat `beat` unpainted. Only a beat that follows a
    /// painted one is skipped: an idle app's next frame (a new gesture's
    /// first) always paints.
    fn take_skip(&mut self, beat: i64, painted_last_beat: bool) -> bool {
        let skip = std::mem::take(&mut self.skip_next_beat) && painted_last_beat && self.touch_active;
        if skip {
            self.armed_after = beat;
            self.skips += 1;
        }
        skip
    }
}

thread_local! {
    static RECOVERY: std::cell::RefCell<StuffingRecovery> = std::cell::RefCell::new(StuffingRecovery::default());
}

fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        extern "C" {
            fn __system_property_get(name: *const std::ffi::c_char, value: *mut std::ffi::c_char) -> std::ffi::c_int;
        }
        let mut value = [0u8; 92];
        let len = unsafe { __system_property_get(c"debug.makepad.stuffing".as_ptr(), value.as_mut_ptr() as *mut std::ffi::c_char) };
        let off = &value[..len.max(0) as usize] == b"off";
        crate::log!("[android.stuffing] recovery {}", if off { "off" } else { "on" });
        !off
    })
}

/// A vsync beat that will paint: whether to leave it unpainted instead.
pub(super) fn beat_begin(beat: i64, expected: i64) -> bool {
    RECOVERY.with(|r| {
        let mut r = r.borrow_mut();
        let painted_last_beat = std::mem::take(&mut r.swapped_this_beat);
        if r.last_beat > 0 && beat > r.last_beat && beat - r.last_beat < 100_000_000 {
            let interval = beat - r.last_beat;
            r.intervals.push_back(interval);
            if r.intervals.len() > 32 {
                r.intervals.pop_front();
            }
        }
        r.last_beat = beat;
        r.beat = beat;
        r.expected = expected;
        if !enabled() || beat <= 0 {
            return false;
        }
        let skip = r.take_skip(beat, painted_last_beat);
        if skip && crate::makepad_error_log::trace_enabled("android.stuffing") {
            crate::log!("[android.stuffing] skip beat={} total={}", beat, r.skips);
        }
        skip
    })
}

/// Right before eglSwapBuffers of the frame painted for the current beat.
pub(super) unsafe fn before_swap(egl: &LibEgl, display: EGLDisplay, surface: EGLSurface) {
    if enabled() {
        RECOVERY.with(|r| r.borrow_mut().before_swap(egl, display, surface));
    }
}

/// eglSwapBuffers queued the frame.
pub(super) fn swapped() {
    RECOVERY.with(|r| r.borrow_mut().swapped_this_beat = true);
}

/// The window surface went away: frame ids belong to it.
pub(super) fn surface_reset() {
    RECOVERY.with(|r| {
        let mut r = r.borrow_mut();
        r.initialized = false;
        r.functions = None;
        r.pending.clear();
        r.skip_next_beat = false;
    });
}

/// Whether a finger is down, from each touch update.
pub(super) fn set_touch_active(active: bool) {
    RECOVERY.with(|r| r.borrow_mut().touch_active = active);
}
