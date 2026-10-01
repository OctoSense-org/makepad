//! GPU time on GL backends (Android, OpenHarmony, Linux), from timer queries
//! (`GL_EXT_disjoint_timer_query`), for two trace topics:
//!
//! - `gpu.pass`: one line per pass, `[gpu.pass] <name> <ms>ms`, as the
//!   Metal backend prints;
//! - `gpu.draws`: GPU time per shader, summed over every draw call and
//!   printed every two seconds, slowest first. While it is on, `gpu.pass`
//!   is not measured: GL time queries cannot nest.
//!
//! Queries are asynchronous: their results are read a few frames later, when
//! the driver has them (`poll`, after each presented frame), so measuring
//! never stalls the pipeline. An interval the driver marks disjoint (a GPU
//! clock change, a context loss) is dropped rather than reported. Off, it
//! costs one trace-topic check per pass and per draw call.

use super::gl_sys::{self, GLint, GLuint, LibGl};
use crate::makepad_error_log::trace_enabled;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Queries waiting for results, at most: past this, new intervals are not
/// measured until older ones are read.
const MAX_PENDING: usize = 4096;
/// How often `gpu.draws` prints its per-shader totals.
const DRAWS_WINDOW_SECS: f64 = 2.0;

enum Measured {
    Pass(String),
    Draw(String),
}

#[derive(Default)]
struct GlTimer {
    supported: Option<bool>,
    free: Vec<GLuint>,
    pending: VecDeque<(GLuint, Measured)>,
    active: bool,
    /// Shader -> (draw calls, GPU ms) in the current `gpu.draws` window.
    draws: HashMap<String, (u32, f64)>,
    window_start: Option<Instant>,
    disjoint_drops: u32,
}

thread_local! {
    static TIMER: RefCell<GlTimer> = RefCell::new(GlTimer::default());
}

fn draws_topic() -> bool {
    trace_enabled("gpu.draws")
}

/// Starts timing a pass named `name` when `gpu.pass` is on; returns whether
/// it did (pass that to [`end`]).
pub fn begin_pass(gl: &LibGl, name: &str) -> bool {
    if !trace_enabled("gpu.pass") || draws_topic() {
        return false;
    }
    TIMER.with(|t| t.borrow_mut().begin(gl, || Measured::Pass(name.to_string())))
}

/// Starts timing one draw call when `gpu.draws` is on; `shader` names it
/// (only called then).
pub fn begin_draw(gl: &LibGl, shader: impl FnOnce() -> String) -> bool {
    if !draws_topic() {
        return false;
    }
    TIMER.with(|t| t.borrow_mut().begin(gl, || Measured::Draw(shader())))
}

/// Ends the interval `begin_*` started, if it started one.
pub fn end(gl: &LibGl, started: bool) {
    if !started {
        return;
    }
    TIMER.with(|t| {
        let mut t = t.borrow_mut();
        if t.active {
            if let Some(end_query) = gl.glEndQuery {
                unsafe { end_query(gl_sys::TIME_ELAPSED_EXT) };
            }
            t.active = false;
        }
    });
}

/// Reads the intervals the driver has finished and reports them.
pub fn poll(gl: &LibGl) {
    TIMER.with(|t| t.borrow_mut().poll(gl));
}

impl GlTimer {
    fn supported(&mut self, gl: &LibGl) -> bool {
        *self.supported.get_or_insert_with(|| {
            let functions = gl.glGenQueries.is_some()
                && gl.glBeginQuery.is_some()
                && gl.glEndQuery.is_some()
                && gl.glGetQueryObjectuiv.is_some()
                && gl.glGetQueryObjectui64v.is_some();
            let extension = unsafe {
                let s = (gl.glGetString)(gl_sys::EXTENSIONS);
                !s.is_null()
                    && std::ffi::CStr::from_ptr(s as *const std::ffi::c_char)
                        .to_string_lossy()
                        .contains("GL_EXT_disjoint_timer_query")
            };
            let ok = functions && extension;
            crate::log!(
                "gl timer: GPU timing {} (timer query functions {}, GL_EXT_disjoint_timer_query {})",
                if ok { "available" } else { "unavailable" },
                if functions { "found" } else { "missing" },
                if extension { "present" } else { "absent" }
            );
            ok
        })
    }

    fn begin(&mut self, gl: &LibGl, what: impl FnOnce() -> Measured) -> bool {
        // GL time queries cannot nest: an interval inside a measured one is
        // part of it.
        if self.active || self.pending.len() >= MAX_PENDING || !self.supported(gl) {
            return false;
        }
        let (Some(gen), Some(begin)) = (gl.glGenQueries, gl.glBeginQuery) else {
            return false;
        };
        let id = self.free.pop().unwrap_or_else(|| {
            let mut id = 0;
            unsafe { gen(1, &mut id) };
            id
        });
        if id == 0 {
            return false;
        }
        unsafe { begin(gl_sys::TIME_ELAPSED_EXT, id) };
        self.pending.push_back((id, what()));
        self.active = true;
        true
    }

    fn poll(&mut self, gl: &LibGl) {
        let (Some(available_of), Some(result_of)) = (gl.glGetQueryObjectuiv, gl.glGetQueryObjectui64v) else {
            return;
        };
        let mut disjoint: GLint = 0;
        if let Some(get) = gl.glGetIntegervQuery {
            unsafe { get(gl_sys::GPU_DISJOINT_EXT, &mut disjoint) };
        }
        // Never ask about the interval still being recorded.
        let readable = self.pending.len() - usize::from(self.active);
        for _ in 0..readable {
            let Some((id, _)) = self.pending.front() else { break };
            let id = *id;
            let mut available: GLuint = 0;
            unsafe { available_of(id, gl_sys::QUERY_RESULT_AVAILABLE, &mut available) };
            if available == 0 {
                break;
            }
            let mut nanos: u64 = 0;
            unsafe { result_of(id, gl_sys::QUERY_RESULT, &mut nanos) };
            let (_, measured) = self.pending.pop_front().unwrap();
            self.free.push(id);
            if disjoint != 0 {
                self.disjoint_drops += 1;
                continue;
            }
            let ms = nanos as f64 / 1_000_000.0;
            match measured {
                Measured::Pass(name) => crate::trace!("gpu.pass", "{} {:.3}ms", name, ms),
                Measured::Draw(shader) => {
                    let entry = self.draws.entry(shader).or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 += ms;
                }
            }
        }
        self.report_draws();
    }

    fn report_draws(&mut self) {
        let start = *self.window_start.get_or_insert_with(Instant::now);
        let secs = start.elapsed().as_secs_f64();
        if secs < DRAWS_WINDOW_SECS {
            return;
        }
        self.window_start = Some(Instant::now());
        if self.draws.is_empty() {
            return;
        }
        let mut rows: Vec<(String, (u32, f64))> = self.draws.drain().collect();
        rows.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
        let total: f64 = rows.iter().map(|r| r.1 .1).sum();
        let top: Vec<String> = rows
            .iter()
            .take(12)
            .map(|(shader, (n, ms))| format!("{shader} {ms:.2}ms/{n}"))
            .collect();
        crate::trace!(
            "gpu.draws",
            "{:.1}s: {:.2}ms GPU in {} shaders{} | {}",
            secs,
            total,
            rows.len(),
            if self.disjoint_drops > 0 { format!(" ({} disjoint dropped)", self.disjoint_drops) } else { String::new() },
            top.join(", ")
        );
        self.disjoint_drops = 0;
    }
}
