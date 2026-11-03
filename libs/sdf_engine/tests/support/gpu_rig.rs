//! The shared GPU rig for gpu_jobs Steps (GPU-EXEC §10.2): a HAL session with a 32 x 32
//! offscreen window and draw pass, the sdf script modules, and `run_steps`, which starts one
//! work item and turns the session until the runtime answers it. Kernel lanes test their Steps
//! with it: `#[path = "support/gpu_rig.rs"] mod gpu_rig;`
#![allow(dead_code)]
use makepad_sdf_engine::{self as engine, gpu_jobs::{self, Finished, JobClass, Jobs, Pool, Step, WorkItem}};
use engine::makepad_draw::{self as makepad_draw, *, makepad_platform::{makepad_hal::{self as hal, Hal, OpenReq, open_flags}, makepad_script::script_eval}};
use std::{cell::RefCell, rc::Rc, sync::atomic::{AtomicU64, Ordering}};

/// What the rig's event handler saw.
#[derive(Default)]
pub struct Rig {
    pub session: u32,
    /// Event::GpuLost and Event::GpuReset seen.
    pub losses: usize,
    pub resets: usize,
    /// Script errors registering the modules (the rig asserts there are none).
    pub setup_errors: Vec<String>,
}

struct Window { _window: WindowHandle, pass: DrawPass, root: DrawList2d }

/// Runs `run` with a Cx on a GPU session.
pub fn with_gpu<R>(run: impl FnOnce(&mut Cx) -> R) -> R {
    with_gpu_rig(|cx, _| run(cx))
}

/// [`with_gpu`], with the device events seen so far (a device loss test).
pub fn with_gpu_rig<R>(run: impl FnOnce(&mut Cx, &Rc<RefCell<Rig>>) -> R) -> R {
    // HAL sessions and the frame timer's watched names are process-wide: one rig at a time.
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _gpu = GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let rig = Rc::new(RefCell::new(Rig::default()));
    let window: Rc<RefCell<Option<Window>>> = Rc::new(RefCell::new(None));
    let (seen, drawn) = (rig.clone(), window.clone());
    let mut cx = Cx::new(Box::new(move |cx, event| {
        match event {
            Event::Startup => {
                cx.with_vm(|vm| {
                    vm.capture_errors();
                    makepad_draw::script_mod(vm);
                    // engine::script_mod registers the display modules too, whose materials read
                    // the widgets prelude; without widgets they get the minimal one the render
                    // tests use
                    vm.heap_mut().new_module(id!(prelude));
                    script_eval!(vm, { mod.prelude.widgets_internal = { ..mod.std, ..mod.pod, ..mod.math, ..mod.sdf, ..mod.shader, draw:mod.draw } });
                    vm.heap_mut().new_module(id!(widgets));
                    engine::script_mod(vm);
                    seen.borrow_mut().setup_errors = vm.take_errors();
                    vm.stop_capturing_errors();
                });
                let mut handle = WindowHandle::new(cx);
                handle.configure_window(cx, dvec2(32.0, 32.0), dvec2(0.0, 0.0), false, "GPU rig".into());
                let pass = DrawPass::new(cx);
                handle.set_pass(cx, &pass);
                *drawn.borrow_mut() = Some(Window { _window: handle, pass, root: DrawList2d::new(cx) });
                cx.redraw_all();
            }
            // the runtime answers everything on the old device DeviceLost; its next pump sees
            // the new device's generation
            Event::GpuLost => { seen.borrow_mut().losses += 1; Jobs::device_lost(cx); }
            Event::GpuReset => { seen.borrow_mut().resets += 1; Jobs::pump(cx); }
            Event::Draw(event) => {
                let mut w = drawn.borrow_mut();
                let Some(w) = w.as_mut() else { return };
                let mut draw = CxDraw::new(cx, event);
                let mut cx = Cx2d::new(&mut draw);
                cx.begin_pass(&w.pass, None);
                w.root.begin_always(&mut cx);
                cx.begin_root_turtle(dvec2(32.0, 32.0), Layout::flow_overlay());
                cx.end_pass_sized_turtle();
                w.root.end(&mut cx);
                cx.end_pass(&w.pass);
            }
            _ => {}
        }
    }));
    cx.init_cx_os();
    let hal = Hal::global();
    let mut req = OpenReq::from_env(0, open_flags::OFFSCREEN_WINDOWS, 1.0);
    req.flags |= open_flags::OFFSCREEN_WINDOWS;
    let info = hal.open(&req);
    assert_ne!(info.backend, hal::backend_id::HEADLESS, "the GPU rig requires a GPU");
    struct Session(&'static Hal, u32);
    impl Drop for Session { fn drop(&mut self) { self.0.close(self.1); } }
    let _session = Session(hal, info.session);
    cx.hal_attach_session(info);
    while window.borrow().is_none() { assert!(cx.hal_session_turn(1_000_000, true), "the HAL session closed before Startup"); }
    let setup = std::mem::take(&mut rig.borrow_mut().setup_errors);
    assert!(setup.is_empty(), "gpu rig: script errors registering the modules: {setup:#?}");
    rig.borrow_mut().session = info.session;
    run(&mut cx, &rig)
}

/// Every job of later submissions has `n` elements (within its pass's ceiling); None measures again.
pub fn force_elements(cx: &mut Cx, n: Option<u32>) {
    gpu_jobs::force_elements(cx, n);
}

/// Starts one work item of `steps` over `pool` and turns the session until the runtime answers
/// it. A hang fails at once: when a pump leaves nothing on the GPU and the next one submits
/// nothing either, no event can move the item.
pub fn run_steps(cx: &mut Cx, pool: Pool, steps: Vec<Step>, class: JobClass) -> Finished {
    static OWNER: AtomicU64 = AtomicU64::new(1 << 40);
    let owner = OWNER.fetch_add(1, Ordering::Relaxed);
    let item = WorkItem { owner, label: format!("gpu rig {owner}"), class, order: 0, pool, steps };
    if let Err(r) = Jobs::start(cx, item) {
        panic!("gpu rig: the item was refused at start: {} ({} bytes wanted, {} available)", r.reason, r.wanted, r.available);
    }
    let mut idle = 0;
    loop {
        Jobs::pump(cx);
        if let Some(finished) = Jobs::finished(cx, owner).pop() { return finished; }
        let (items, waiting, in_flight) = Jobs::occupancy(cx);
        if in_flight == 0 {
            // one more pump runs what the last answer unblocked (a failed build answers there)
            idle += 1;
            let refused: Vec<String> = Jobs::waiting(cx, owner).iter().map(|(_, r)| format!("{} ({} bytes wanted, {} available)", r.reason, r.wanted, r.available)).collect();
            assert!(idle < 2, "gpu rig: the item is stuck with nothing on the GPU: {items} items, {waiting} waiting for memory {refused:?}; {}", gpu_jobs::calibration_line(cx));
            continue;
        }
        idle = 0;
        let present = cx.frame_completion_serial() >= cx.frame_submission_serial();
        assert!(cx.hal_session_turn(1_000_000, present), "gpu rig: the HAL session closed");
    }
}

/// Little-endian bytes of `words` (an upload).
pub fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The u32 words of `bytes` (an output).
pub fn words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
}
