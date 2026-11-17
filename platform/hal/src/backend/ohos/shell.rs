use super::{ime, napi, surface::{OhosEglSurfaces, Registry}};
use crate::{backend::{exec::{Gpu, OsCmd}, gl::GlBackend, Backend, Pixels, Shell}, buffers::{CmdView, EventQueue}, session::{World, HOST}, *};
use std::{collections::HashSet, sync::{Arc, Mutex, MutexGuard, OnceLock}};

pub(super) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> { m.lock().unwrap_or_else(|e| e.into_inner()) }

#[derive(Clone, Debug)]
pub struct OhosParams {
    pub cache_path: String,
    pub data_path: String,
    /// physical pixels per vp
    pub density: f32,
    pub os_version: String,
    /// the bundle's version and last update time, which every new or reinstalled HAP changes
    pub app_build: String,
    /// the system's dark mode (at start, kept current by the ability's configuration updates)
    pub night_mode: bool,
}
impl Default for OhosParams {
    fn default() -> Self { Self { cache_path: String::new(), data_path: String::new(), density: 1., os_version: String::new(), app_build: String::new(), night_mode: false } }
}

pub(super) struct State {
    pub recipient: Option<(u32, Id)>,
    /// the session whose text field the soft keyboard may edit
    pub ime_session: Option<u32>,
    pub events: Vec<(Option<u32>, EventQueue)>,
    pub params: OhosParams,
    /// the display's frame rate for this app, which each `FrameReady` predicts the next vsync from
    pub refresh: f32,
    /// top, right, bottom, left in vp
    pub insets: [f32; 4],
    pub pending: HashSet<(u32, Id)>,
    pub active: bool,
    pub focused: bool,
    /// the soft keyboard's height in pixels and whether it is up
    pub ime: Option<(i32, bool)>,
    /// The session that asked for fullscreen: the system bars stay hidden until it restores its
    /// window or closes.
    pub fullscreen: Option<u32>,
    pub wake: Option<Arc<dyn Fn() + Send + Sync>>,
}
pub(super) struct Shared {
    pub registry: Mutex<Registry>,
    pub state: Mutex<State>,
    /// Lets go of the XComponent's window before `OnSurfaceDestroyed` returns (the GLES core's
    /// EGL surfaces).
    pub release_window: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
}
impl Shared {
    pub fn global() -> &'static Self {
        static VALUE: OnceLock<Shared> = OnceLock::new();
        VALUE.get_or_init(|| Self { registry: Mutex::new(Registry::default()), state: Mutex::new(State {
            recipient: None, ime_session: None, events: Vec::new(), params: OhosParams::default(), refresh: 60., insets: [0.; 4], pending: HashSet::new(), active: false, focused: false, ime: None, fullscreen: None, wake: None,
        }), release_window: Mutex::new(None) })
    }
    pub fn dpi(&self) -> f32 { lock(&self.state).params.density }
    pub fn emit(&self, time: u64, f: impl FnOnce(&mut EventQueue, Id)) {
        let mut q = EventQueue::new();
        q.set_time(time);
        let recipient = lock(&self.state).recipient;
        let id = recipient.map_or(Id::default(), |(_, id)| id);
        f(&mut q, id);
        let wake = {
            let mut s = lock(&self.state);
            s.events.push((recipient.map(|(sid, _)| sid), q));
            s.wake.clone()
        };
        if let Some(wake) = wake { wake(); }
    }
    pub fn display(&self) {
        let size = lock(&self.registry).state.size;
        let (dpi, refresh, insets, fullscreen) = { let s = lock(&self.state); (s.params.density, s.refresh, s.insets, s.fullscreen.is_some()) };
        let state = if fullscreen { window_state::FULLSCREEN } else { window_state::RESTORE };
        self.emit(clock_ns(), |q, id| {
            q.display(id, dpi, refresh);
            q.geom(id, 0., 0., size[0] as f64 / dpi as f64, size[1] as f64 / dpi as f64, dpi, state);
            q.safe_area(id, insets[0] as f64, insets[1] as f64, insets[2] as f64, insets[3] as f64);
        });
    }
    /// FrameReady for the windows that asked.
    pub fn frame(&self, time: u64) {
        let (pending, wake, period) = {
            let mut s = lock(&self.state);
            if !s.active { return; }
            (std::mem::take(&mut s.pending), s.wake.clone(), (1_000_000_000. / s.refresh as f64) as u64)
        };
        let mut events = Vec::with_capacity(pending.len());
        for (sid, id) in pending {
            let mut q = EventQueue::new(); q.set_time(time); q.frame_ready(id, time.saturating_add(period));
            events.push((Some(sid), q));
        }
        let any = !events.is_empty();
        lock(&self.state).events.extend(events);
        if any { if let Some(wake) = wake { wake(); } }
    }
}

struct OhosShell;
impl Shell for OhosShell {
    fn auto_adopt(&self) -> bool { true }
    fn lent(&mut self, sid: u32, alias: Id, host: Id, attach: bool) {
        let shared = Shared::global();
        if lock(&shared.registry).host != Some(host) { return; }
        lock(&shared.state).recipient = Some(if attach { (sid, alias) } else { (HOST, host) });
        shared.display();
        let (active, focused, ime, dpi) = { let s = lock(&shared.state); (s.active, s.focused, s.ime, s.params.density) };
        shared.emit(clock_ns(), |q, id| {
            q.focus(id, focused);
            if let Some((height, open)) = ime { q.ime_resize(id, height as f64 / dpi as f64, open); }
            q.lifecycle(if active { lifecycle::RESUME } else { lifecycle::PAUSE });
        });
    }
    fn command(&mut self, sid: u32, cmd: OsCmd) -> bool {
        match cmd {
            OsCmd::WindowGeom { id, .. } if lock(&Shared::global().state).recipient == Some((sid, id)) => {
                // The ability's window decides the geometry; an app cannot resize the XComponent.
                Shared::global().display(); true
            }
            OsCmd::WindowState { id, state: state @ (window_state::FULLSCREEN | window_state::RESTORE) } if lock(&Shared::global().state).recipient == Some((sid, id)) => {
                let fullscreen = state == window_state::FULLSCREEN;
                lock(&Shared::global().state).fullscreen = fullscreen.then_some(sid);
                napi::call("fullScreen", if fullscreen { "1" } else { "0" });
                Shared::global().display(); true
            }
            OsCmd::FrameRequest { id } => { lock(&Shared::global().state).pending.insert((sid, id)); napi::request_frame(); true }
            cmd @ (OsCmd::ImeShow { .. } | OsCmd::ImeHide { .. } | OsCmd::ImeSync { .. }) => {
                if lock(&Shared::global().state).ime_session != Some(sid) { return false; }
                ime::command(cmd)
            }
            OsCmd::ClipSet { text } => napi::call("copyToClipboard", &text),
            OsCmd::BackUnhandled => napi::call("backUnhandled", ""),
            OsCmd::MobileUi { op: mobile_ui::STATUS_BAR_DARK_ICONS, value } => napi::call("statusBarDarkIcons", if value != 0 { "1" } else { "0" }),
            OsCmd::MobileUi { op: mobile_ui::KEEP_SCREEN_ON, value } => napi::call("keepScreenOn", if value != 0 { "1" } else { "0" }),
            _ => false,
        }
    }
}

/// The backend `hal_open` makes on OpenHarmony: GLES 3 through EGL on the XComponent's window.
pub fn open(backend: u32, flags: u32, wake: Arc<dyn Fn() + Send + Sync>, notify: Arc<dyn Fn() + Send + Sync>) -> Result<Box<dyn Backend>, String> {
    if backend == backend_id::VULKAN { return Err("the OpenHarmony shell draws with GLES 3 only".into()); }
    Ok(Box::new(OhosBackend::with_gl(flags, wake, notify)?))
}

pub struct OhosBackend { core: Box<dyn Backend>, host: Option<Id>, offscreen: bool }
impl OhosBackend {
    /// `notify` makes every pump look (the ability's events and vsync callbacks).
    pub fn with_gl(flags: u32, wake: Arc<dyn Fn() + Send + Sync>, notify: Arc<dyn Fn() + Send + Sync>) -> Result<Self, String> {
        let offscreen = flags & open_flags::OFFSCREEN_WINDOWS != 0;
        let shell: Option<Box<dyn Shell>> = if offscreen { None } else { Some(Box::new(OhosShell)) };
        let core = GlBackend::with_source(wake, Box::new(OhosEglSurfaces { offscreen }), shell)?;
        if !offscreen { *lock(&Shared::global().release_window) = Some(core.window_release()); }
        lock(&Shared::global().state).wake = Some(notify);
        Ok(Self { core: Box::new(core), host: None, offscreen })
    }
    pub fn params() -> OhosParams { lock(&Shared::global().state).params.clone() }
    fn ensure_host(&mut self, world: &mut World) {
        if self.offscreen || self.host.is_some() { return; }
        let id = world.host_window([0.; 4], true);
        self.host = Some(id);
        let shared = Shared::global();
        lock(&shared.registry).host = Some(id);
        lock(&shared.state).recipient = Some((HOST, id));
        shared.display();
    }
    fn drain(&mut self, world: &mut World) {
        let events = std::mem::take(&mut lock(&Shared::global().state).events);
        for (sid, mut q) in events {
            let recipient = lock(&Shared::global().state).recipient;
            let remap = sid.is_none();
            let sid = sid.unwrap_or_else(|| recipient.map_or(HOST, |(sid, _)| sid));
            // Input is addressed to the window's borrower, including before its first frame.
            q.drain_with(|mut e, payload| {
                if remap { e.window = recipient.map_or(Id::default(), |(_, id)| id); }
                match e.decode(payload) {
                    Some(HalEvent::Geom { window, x, y, w, h, dpi, state }) => world.window_geom(sid, window, [x as f32, y as f32, w as f32, h as f32], dpi, state),
                    Some(HalEvent::Lifecycle { kind }) => world.note_lifecycle(kind),
                    _ => {}
                }
                world.to_session(sid, |q| q.push_with_payload(e, payload));
            });
        }
    }
}
impl Backend for OhosBackend {
    fn info(&self) -> HalInfo {
        let info = self.core.info();
        // the display's vsync answers frame requests
        let display_clock = if self.offscreen { 0 } else { info_flags::DISPLAY_CLOCK };
        HalInfo { dpi: Shared::global().dpi(), flags: info.flags | display_clock, ..info }
    }
    fn submit(&mut self, world: &mut World, sid: u32, view: &CmdView) {
        self.ensure_host(world);
        self.drain(world);
        let owner = lock(&Shared::global().state).recipient.map(|(sid, _)| sid);
        let allowed = owner == Some(sid) || world.session(sid).is_some_and(|s| s.focused && s.parent != 0 && Some(s.parent) == owner);
        lock(&Shared::global().state).ime_session = allowed.then_some(sid);
        self.core.submit(world, sid, view);
        self.drain(world);
    }
    fn step(&mut self, world: &mut World, sid: u32) { self.ensure_host(world); self.drain(world); self.core.step(world, sid); }
    fn next_due(&self, world: &World, sid: u32) -> Option<u64> { self.core.next_due(world, sid) }
    fn window_pixels(&mut self, world: &mut World, sid: u32, window: Option<Id>) -> Option<Pixels> { self.core.window_pixels(world, sid, window) }
    fn host_gpu(&mut self) -> Option<&mut dyn Gpu> { self.core.host_gpu() }
    fn session_opened(&mut self, world: &mut World, sid: u32) { self.ensure_host(world); self.core.session_opened(world, sid); }
    fn session_closed(&mut self, world: &mut World, sid: u32) {
        self.core.session_closed(world, sid);
        let shared = Shared::global();
        let mut state = lock(&shared.state);
        state.events.retain(|(s, _)| *s != Some(sid));
        state.pending.retain(|(s, _)| *s != sid);
        if state.recipient.is_some_and(|(s, _)| s == sid) { state.recipient = self.host.map(|id| (HOST, id)); }
        let restore = state.fullscreen == Some(sid);
        if restore { state.fullscreen = None; }
        drop(state);
        // its fullscreen ends with it, so the next app gets the system bars back
        if restore { napi::call("fullScreen", "0"); shared.display(); }
    }
    fn before_exit(&mut self) { self.core.before_exit(); }
}
