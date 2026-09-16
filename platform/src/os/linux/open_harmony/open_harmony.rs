use {
    crate::TextInputConfig,
    crate::cx::CxDependency,
    self::super::{
        super::gl_sys, super::gl_sys::LibGl, arkts_obj_ref::ArkTsObjRef, oh_callbacks::*,
        oh_media::CxOpenHarmonyMedia, raw_file::RawFileMgr,
    },
    crate::{
        cx::{Cx, OpenHarmonyParams, OsType},
        cx_api::{CxOsApi, CxOsOp, OpenUrlInPlace},
        draw_pass::{CxDrawPassParent, DrawPassClearColor, DrawPassClearDepth, DrawPassId},
        egl_sys::{self, LibEgl, EGL_NONE},
        event::{Event, KeyCode, KeyEvent, TouchUpdateEvent, VirtualKeyboardEvent, WindowGeom},
        gpu_info::GpuPerformance,
        makepad_math::*,
        os::cx_native::EventFlow,
        shared_framebuf::{PollTimer, PollTimers},
        thread::SignalToUI,
        window::CxWindowPool,
        WindowGeomChangeEvent,
    },
    napi_derive_ohos::napi,
    napi_ohos::{sys::*, Env, JsObject, NapiRaw},
    std::{ffi::CString, os::raw::c_void, ptr::null_mut, rc::Rc, sync::mpsc, time::Instant},
};

#[napi(js_name = "onCreate")]
pub fn ohos_ability_on_create(env: Env, ark_ts: JsObject) -> napi_ohos::Result<()> {
    let raw_env = env.raw();
    let raw_ark = unsafe { ark_ts.raw() };
    let mut arkts_ref = std::ptr::null_mut();

    let status = unsafe { napi_create_reference(raw_env, raw_ark, 1, &mut arkts_ref) };
    assert!(status == 0);

    let arkts_obj = ArkTsObjRef::new(raw_env, arkts_ref);
    let device_type = arkts_obj
        .get_string("deviceType")
        .unwrap_or("phone".to_string());
    let os_full_name = arkts_obj
        .get_string("osFullName")
        .unwrap_or("OpenHarmony".to_string());
    let display_density = arkts_obj.get_number("displayDensity").unwrap_or(3.25);
    let files_dir = arkts_obj.get_string("filesDir").unwrap();
    let cache_dir = arkts_obj.get_string("cacheDir").unwrap();
    let temp_dir = arkts_obj.get_string("tempDir").unwrap();
    let res_mgr = arkts_obj.get_property("resMgr").unwrap();

    // The entry ability collects the Want's `makepad.*` parameters into
    // `launchParameters` (a flat JSON object of strings). They become the
    // environment the rest of the platform already reads — STUDIO_HOST,
    // STUDIO_BUILD, STUDIO_CRATE and any app-level setting — before anything
    // resolves them, exactly like `cargo makepad android run` passes extras.
    if let Ok(parameters) = arkts_obj.get_string("launchParameters") {
        for (key, value) in flat_json_string_pairs(&parameters) {
            if let Some(name) = key.strip_prefix("makepad.") {
                crate::log!("launch parameter {name}={value}");
                // Both spellings Android's extras produce: `makepad.TRACE`
                // is MAKEPAD_TRACE to the platform, and the Studio settings
                // are read unprefixed (STUDIO_HOST, STUDIO_BUILD, STUDIO_CRATE).
                std::env::set_var(format!("MAKEPAD_{name}"), &value);
                std::env::set_var(name, value);
            }
        }
    }
    // The process starts without a usable HOME. The ability's files
    // directory is the app's home: where a bundled core keeps its state,
    // where anything reading `$HOME` on this platform should land.
    if std::env::var_os("HOME").map_or(true, |home| home.is_empty() || home == "/") {
        std::env::set_var("HOME", &files_dir);
    }
    // The trace topics were read at module load, before these parameters
    // existed; `--ps makepad.MAKEPAD_TRACE topic,topic` works from here on.
    crate::makepad_error_log::set_trace_topics(
        &std::env::var("MAKEPAD_TRACE").unwrap_or_default(),
    );

    let raw_file = RawFileMgr::new(raw_env, res_mgr);

    crate::log!("call onCreate, device_type = {}, os_full_name = {}, display_density = {}, files_dir = {}, cache_dir = {}, temp_dir = {}", device_type, os_full_name, display_density, files_dir,cache_dir,temp_dir);

    send_from_ohos_message(FromOhosMessage::Init {
        device_type,
        os_full_name,
        display_density,
        files_dir,
        cache_dir,
        temp_dir,
        raw_env,
        arkts_ref,
        raw_file,
    });
    Ok(())
}

impl Cx {
    fn main_loop(&mut self, from_ohos_rx: mpsc::Receiver<FromOhosMessage>) {
        crate::log!("entry main_loop");

        self.gpu_info.performance = GpuPerformance::Tier1;

        self.call_event_handler(&Event::Startup);
        self.redraw_all();

        while !self.os.quit {
            // Sleep until something arrives or the earliest timer is due; the
            // display is asked for a beat only when there is something to draw.
            let wait = self
                .os
                .timers
                .next_due_in()
                .map_or(std::time::Duration::from_secs(3600), std::time::Duration::from_secs_f64);
            match from_ohos_rx.recv_timeout(wait) {
                Ok(FromOhosMessage::VSync) => {
                    self.handle_all_pending_messages(&from_ohos_rx);
                    self.handle_other_events();
                    // Arm the next beat before this one's paint and swap, or
                    // a continuous animation lands every other beat.
                    if self.frame_wanted() {
                        super::oh_callbacks::request_vsync();
                    }
                    self.handle_drawing();
                }
                Ok(FromOhosMessage::Wake) => {
                    self.handle_all_pending_messages(&from_ohos_rx);
                    self.handle_other_events();
                }
                Ok(message) => self.handle_message(message),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    self.handle_other_events();
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    crate::error!("the ArkTS channel closed");
                    break;
                }
            }
            if self.frame_wanted() {
                super::oh_callbacks::request_vsync();
            }
        }
        self.call_event_handler(&Event::Shutdown);
    }

    /// Whether the next display beat has work: a dirty pass, a requested
    /// redraw or next frame, a time-driven shader, a platform op, released
    /// GPU storage still waiting to retire.
    fn frame_wanted(&self) -> bool {
        self.any_passes_dirty()
            || self.need_redrawing()
            || !self.new_next_frames.is_empty()
            || self.demo_time_repaint
            || self.os.first_after_resize
            || !self.platform_ops.is_empty()
            || self.opengl_retirement_pending()
    }

    fn handle_all_pending_messages(&mut self, from_ohos_rx: &mpsc::Receiver<FromOhosMessage>) {
        // Handle the messages that arrived during the last frame
        while let Ok(msg) = from_ohos_rx.try_recv() {
            self.handle_message(msg);
        }
    }

    fn handle_other_events(&mut self) {
        // Network runtime responses and Studio websocket messages. Every
        // other backend pumps these from its event loop; this one never did,
        // so a hub request (WidgetTreeDump, Screenshot, ...) sat unanswered.
        self.dispatch_network_runtime_events();

        // Timers
        let events = self.os.timers.get_dispatch();
        for event in events {
            self.handle_script_timer(&event);
            self.call_event_handler(&Event::Timer(event));
        }

        // Signals
        if SignalToUI::check_and_clear_ui_signal() {
            self.handle_media_signals();
            self.handle_script_signals();
            self.call_event_handler(&Event::Signal);
        }
        if SignalToUI::check_and_clear_action_signal() {
            self.handle_action_receiver();
        }

        // Video updates
        // let to_dispatch = self.get_video_updates();
        // for video_id in to_dispatch {
        //     let e = Event::VideoTextureUpdated(
        //         VideoTextureUpdatedEvent {
        //             video_id,
        //         }
        //     );
        //     self.call_event_handler(&e);
        // }

        // Live edits
        self.run_live_edit_if_needed("open-harmony");

        // Platform operations
        self.handle_platform_ops();
    }

    fn handle_drawing(&mut self) {
        if self.any_passes_dirty()
            || self.need_redrawing()
            || !self.new_next_frames.is_empty()
            || self.demo_time_repaint
        {
            let time_now = self.os.timers.time_now();
            if !self.new_next_frames.is_empty() {
                self.call_next_frame_event(time_now);
            }
            if self.need_redrawing() {
                self.call_draw_event(time_now);
                self.opengl_compile_shaders();
            }

            if self.os.first_after_resize {
                self.os.first_after_resize = false;
                self.redraw_all();
            }

            self.handle_repaint();

            // Script-VM garbage collection at a safe point after paint, as the
            // Android and macOS backends do: every eval allocates script
            // objects that only gc() reclaims. needs_gc() gates the sweep.
            if std::env::var_os("MAKEPAD_OHOS_GC").is_some() {
                self.with_vm(|vm| {
                    if vm.heap().needs_gc() {
                        vm.gc();
                    }
                });
            }
        } else {
            // Nothing to paint: released GPU storage (allocations waiting on
            // their completion fence, freed draw storage) is served here, on
            // the beat, without a present, as Android does. A hosted module
            // that frees storage every frame settles here instead of keeping
            // the repaint alive.
            let _ = self.opengl_maintain_instance_retirements();
        }
    }

    fn handle_message(&mut self, msg: FromOhosMessage) {
        match msg {
            FromOhosMessage::SurfaceCreated {
                window,
                width: _,
                height: _,
            } => unsafe {
                self.os.display.as_mut().unwrap().update_surface(window);
            },
            FromOhosMessage::SurfaceDestroyed => unsafe {
                self.os.display.as_mut().unwrap().destroy_surface();
            },
            FromOhosMessage::SurfaceChanged {
                window,
                width,
                height,
            } => {
                unsafe {
                    self.os.display.as_mut().unwrap().update_surface(window);
                }
                self.os.display_size = dvec2(width as f64, height as f64);
                let window_id = CxWindowPool::id_zero();
                let window = &mut self.windows[window_id];
                // Stash the OS-reported scale factor so a later
                // `set_window_dpi_override(None)` can recover the native scale.
                // OpenHarmony converts touch coords at the source so the
                // helper-based remap is unnecessary, but this field is also
                // consulted by `Cx::set_window_dpi_override`.
                window.os_dpi_factor = Some(self.os.dpi_factor);
                let old_geom = window.window_geom.clone();

                let dpi_factor = window.dpi_override.unwrap_or(self.os.dpi_factor);
                let size = self.os.display_size / dpi_factor;
                window.window_geom = WindowGeom {
                    dpi_factor,
                    can_fullscreen: false,
                    xr_is_presenting: false,
                    is_fullscreen: true,
                    is_topmost: true,
                    position: dvec2(0.0, 0.0),
                    inner_size: size,
                    outer_size: size,
                    ..Default::default()
                };
                let new_geom = window.window_geom.clone();
                self.call_event_handler(&Event::WindowGeomChange(WindowGeomChangeEvent {
                    window_id,
                    new_geom,
                    old_geom,
                }));
                if let Some(main_pass_id) = self.windows[window_id].main_pass_id {
                    self.redraw_pass_and_child_passes(main_pass_id);
                }
                self.redraw_all();
                self.os.first_after_resize = true;
                self.call_event_handler(&Event::ClearAtlasses);
            }
            FromOhosMessage::Touch(mut touches) => {
                let time = touches[0].time;
                let window = &mut self.windows[CxWindowPool::id_zero()];
                let dpi_factor = window.dpi_override.unwrap_or(self.os.dpi_factor);
                for touch in &mut touches {
                    // When the software keyboard shifted the UI in the vertical axis,
                    //we need to make the math here to keep touch events positions synchronized.
                    //if self.os.keyboard_visible {touch.abs.y += self.os.keyboard_panning_offset as f64};
                    //crate::log!("{} {:?} {} {}", time, touch.state, touch.uid, touch.abs);
                    touch.abs /= dpi_factor;
                }
                self.fingers.process_touch_update_start(time, &touches);
                let e = Event::TouchUpdate(TouchUpdateEvent {
                    time,
                    window_id: CxWindowPool::id_zero(),
                    touches,
                    modifiers: Default::default(),
                });
                self.call_event_handler(&e);
                let e = if let Event::TouchUpdate(e) = e {
                    e
                } else {
                    panic!()
                };
                self.fingers.process_touch_update_end(&e.touches);
            }
            FromOhosMessage::TextInput(e) => {
                self.call_event_handler(&Event::TextInput(e));
            }
            FromOhosMessage::DeleteLeft(length) => {
                for _ in 0..length {
                    let time = self.os.timers.time_now();
                    let e = KeyEvent {
                        key_code: KeyCode::Backspace,
                        is_repeat: false,
                        modifiers: Default::default(),
                        time,
                    };
                    self.keyboard.process_key_down(e.clone());
                    self.call_event_handler(&Event::KeyDown(e.clone()));
                    self.keyboard.process_key_up(e.clone());
                    self.call_event_handler(&Event::KeyUp(e));
                }
            }
            FromOhosMessage::ResizeTextIME(is_open, keyboard_height) => {
                let keyboard_height = (keyboard_height as f64) / self.os.dpi_factor;
                if is_open {
                    self.call_event_handler(&Event::VirtualKeyboard(
                        VirtualKeyboardEvent::DidShow {
                            height: keyboard_height,
                            time: self.os.timers.time_now(),
                        },
                    ))
                } else {
                    self.os.last_ime_config = None;
                    self.text_ime_was_dismissed();
                    self.call_event_handler(&Event::VirtualKeyboard(
                        VirtualKeyboardEvent::DidHide {
                            time: self.os.timers.time_now(),
                        },
                    ))
                }
            }
            _ => {}
        }
    }

    fn wait_init(&mut self, from_ohos_rx: &mpsc::Receiver<FromOhosMessage>) -> bool {
        // Anything may wake the channel before the ability has sent Init
        // (a signal posted from a background thread arrives as VSync), so
        // skip what is not Init instead of giving up on the first message.
        loop {
            match from_ohos_rx.recv() {
                Ok(FromOhosMessage::Init {
                    device_type,
                    os_full_name,
                    display_density,
                    files_dir,
                    cache_dir,
                    temp_dir,
                    raw_env,
                    arkts_ref,
                    raw_file,
                }) => {
                    self.os.dpi_factor = display_density;
                    self.os.raw_file = Some(raw_file);
                    self.os_type = OsType::OpenHarmony(OpenHarmonyParams {
                        files_dir,
                        cache_dir,
                        temp_dir,
                        device_type,
                        os_full_name,
                        display_density,
                    });
                    self.os.arkts_obj = Some(ArkTsObjRef::new(raw_env, arkts_ref));
                    return true;
                }
                Ok(other) => {
                    crate::log!("message before Init skipped: {}", ohos_message_name(&other));
                }
                Err(_) => {
                    crate::error!("Failed to receive init message from ArkTS layer");
                    return false;
                }
            }
        }
    }

    fn wait_surface_created(
        &mut self,
        from_ohos_rx: &mpsc::Receiver<FromOhosMessage>,
    ) -> *mut c_void {
        // The Studio websocket connects between Init and the XComponent's
        // surface and posts a wake-up through this channel; a single recv
        // took that wake-up for the surface, handed EGL a null window and
        // the app died in an assertion. Wait for the surface itself.
        loop {
            match from_ohos_rx.recv() {
                Ok(FromOhosMessage::SurfaceCreated {
                    window,
                    width,
                    height,
                }) => {
                    self.os.display_size = dvec2(width as f64, height as f64);
                    crate::log!(
                        "handle surface created, width={}, height={}, display_density={}",
                        width,
                        height,
                        self.os.dpi_factor
                    );
                    return window;
                }
                Ok(other) => {
                    crate::log!("message before SurfaceCreated skipped: {}", ohos_message_name(&other));
                }
                Err(_) => {
                    crate::error!("Can't recv SurfaceCreated from arkts");
                    return null_mut();
                }
            }
        }
    }


    pub fn ohos_init<F>(exports: JsObject, env: Env, startup: F)
    where
        F: FnOnce() -> Box<Cx> + Send + 'static,
    {
        // The async log sink only exists after init_log; every other entry
        // point starts it, this one never did, so every log record on
        // OpenHarmony was counted as dropped and nothing reached hilog.
        Cx::init_log();
        crate::log!("ohos init");
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(move || {
            std::panic::set_hook(Box::new(|info| {
                // Synchronously: with panic = "abort" the async sink never
                // gets to flush this line.
                super::oh_util::hilog_sync(&format!("[E] panic: {info}"));
                crate::log!("custom panic hook: {}", info);
            }));
            Cx::ohos_startup(startup);
        });

        if let Ok(xcomponent) = exports.get_named_property::<JsObject>("__NATIVE_XCOMPONENT_OBJ__")
        {
            register_xcomponent_callbacks(&env, &xcomponent);
        } else {
            crate::log!("Failed to get xcomponent in ohos_init");
        }
    }

    fn ohos_startup<F>(startup: F)
    where
        F: FnOnce() -> Box<Cx> + Send + 'static,
    {
        crate::log!("ohos startup");
        let (from_ohos_tx, from_ohos_rx) = mpsc::channel();
        let ohos_tx = from_ohos_tx.clone();
        init_globals(ohos_tx);

        std::thread::spawn(move || {
            let mut cx = startup();
            assert!(cx.wait_init(&from_ohos_rx));
            // `startup` resolved the Studio host at module load, before the
            // entry ability's onCreate turned the launch parameters into
            // environment (STUDIO_HOST and friends). Now that Init has
            // arrived they are set, so resolve again and dial the hub.
            let studio_http = crate::resolve_studio_http();
            if !studio_http.is_empty() {
                crate::log!("studio host from launch parameters: {studio_http}");
                cx.init_websockets(&studio_http);
            }
            cx.ohos_load_dependencies();

            let window = cx.wait_surface_created(&from_ohos_rx);

            let mut libegl = LibEgl::try_load().expect("can't load LibEGL");
            let (egl_context, egl_config, egl_display) = unsafe {
                egl_sys::create_egl_context(&mut libegl).expect("Can't create EGL context")
            };
            let libgl = LibGl::try_load(|s| {
                for s in s {
                    let s = CString::new(*s).unwrap();
                    let p = unsafe { libegl.eglGetProcAddress.unwrap()(s.as_ptr()) };
                    if !p.is_null() {
                        return p;
                    }
                }
                0 as _
            })
            .expect("Cant load openGL functions");

            let win_attr = vec![EGL_NONE];
            let surface = unsafe {
                (libegl.eglCreateWindowSurface.unwrap())(
                    egl_display,
                    egl_config,
                    window as _,
                    win_attr.as_ptr() as _,
                )
            };

            if surface.is_null() {
                let err_code = unsafe { (libegl.eglGetError.unwrap())() };
                crate::log!("eglCreateWindowSurface error code:{}", err_code);
            }
            assert!(!surface.is_null());

            crate::log!("eglCreateWindowSurface success");
            unsafe {
                (libegl.eglSwapBuffers.unwrap())(egl_display, surface);
            }

            if unsafe {
                (libegl.eglMakeCurrent.unwrap())(egl_display, surface, surface, egl_context)
            } == 0
            {
                panic!();
            }

            cx.os.display = Some(CxOhosDisplay {
                libegl,
                libgl,
                egl_display,
                egl_config,
                egl_context,
                surface,
                window,
            });

            register_vsync_callback(from_ohos_tx);
            cx.main_loop(from_ohos_rx);
            //TODO, destroy surface
        });
    }

    /// Ask the platform for a location fix. `sys.gps` in the widgets calls this
    /// on every read; on Android the JNI `LocationListener` feeds
    /// `makepad_platform::gps`. This line has no ArkTS location bridge yet (the
    /// build-tool lane's DevEco template carries one), so the call is a logged
    /// no-op and `gps::last_gps_fix()` stays `None` — cards see "no fix", never
    /// a stale or fake position.
    pub fn ohos_request_gps(&mut self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            crate::log!("ohos_request_gps: no location bridge on this line; sys.gps reports no fix");
        }
    }

    /// Every packaged raw file, read now, while the native resource manager
    /// is still usable. On HarmonyOS 6 the manager obtained at onCreate serves
    /// reads until the window stage is up and aborts (a CFI check inside the
    /// raw-file API) on any read after that, so nothing asks it later: script
    /// resources, fonts and images are all served from this table.
    pub fn ohos_load_dependencies(&mut self) {
        let Some(raw_file) = self.os.raw_file.clone() else {
            crate::error!("no resource manager: packaged resources unavailable");
            return;
        };
        let started = std::time::Instant::now();
        let root = self.package_root.clone().unwrap_or_else(|| "makepad".to_string());
        let prefix = format!("{root}/");
        let (mut files, mut bytes) = (0usize, 0usize);
        for path in raw_file.list_files(&root) {
            let mut data = Vec::new();
            let entry = match raw_file.read_to_end(&path, &mut data) {
                Ok(_) => {
                    files += 1;
                    bytes += data.len();
                    Some(Ok(Rc::new(data)))
                }
                Err(err) => Some(Err(format!("rawfile {path}: {err}"))),
            };
            let key = path.strip_prefix(&prefix).unwrap_or(&path).to_string();
            self.dependencies.insert(key, CxDependency { data: entry });
        }
        crate::log!(
            "packaged resources: {files} files, {} KiB, {:.0} ms",
            bytes / 1024,
            started.elapsed().as_secs_f64() * 1000.0
        );
    }

    pub fn draw_pass_to_fullscreen(&mut self, draw_pass_id: DrawPassId) {
        let draw_list_id = self.passes[draw_pass_id].main_draw_list_id.unwrap();

        self.setup_render_pass(draw_pass_id, false);

        // keep repainting in a loop
        //self.passes[draw_pass_id].paint_dirty = false;
        let gl = self.os.gl();
        unsafe {
            //direct_app.egl.make_current();
            (gl.glViewport)(
                0,
                0,
                self.os.display_size.x as i32,
                self.os.display_size.y as i32,
            );
        }

        let clear_color = if self.passes[draw_pass_id].color_textures.len() == 0 {
            self.passes[draw_pass_id].clear_color
        } else {
            match self.passes[draw_pass_id].color_textures[0].clear_color {
                DrawPassClearColor::InitWith(color) => color,
                DrawPassClearColor::ClearWith(color) => color,
            }
        };
        let clear_depth = match self.passes[draw_pass_id].clear_depth {
            DrawPassClearDepth::InitWith(depth) => depth,
            DrawPassClearDepth::ClearWith(depth) => depth,
        };

        if !self.passes[draw_pass_id].dont_clear {
            unsafe {
                (gl.glBindFramebuffer)(gl_sys::FRAMEBUFFER, 0);
                (gl.glClearDepthf)(clear_depth as f32);
                (gl.glClearColor)(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
                (gl.glClear)(gl_sys::COLOR_BUFFER_BIT | gl_sys::DEPTH_BUFFER_BIT);
            }
        }
        Self::set_default_depth_and_blend_mode(self.os.gl());

        let mut zbias = 0.0;
        let zbias_step = self.passes[draw_pass_id].zbias_step;

        self.render_view(draw_pass_id, draw_list_id, &mut zbias, zbias_step);

        // Studio screenshot: the shared GL context module that answers these
        // on desktop is not built for OpenHarmony, so read the framebuffer
        // here, before the swap, the same way it does.
        let request_ids = self.take_studio_screenshot_request_ids(0);
        if !request_ids.is_empty() {
            let w = self.os.display_size.x as u32;
            let h = self.os.display_size.y as u32;
            let mut pixels = vec![0u8; (w * h * 4) as usize];
            let gl = self.os.gl();
            unsafe {
                (gl.glReadPixels)(
                    0,
                    0,
                    w as i32,
                    h as i32,
                    gl_sys::RGBA,
                    gl_sys::UNSIGNED_BYTE,
                    pixels.as_mut_ptr() as *mut _,
                );
            }
            // GL reads bottom-up; the PNG wants rows top-down.
            let stride = (w * 4) as usize;
            for y in 0..(h as usize / 2) {
                let top = y * stride;
                let bot = (h as usize - 1 - y) * stride;
                for x in 0..stride {
                    pixels.swap(top + x, bot + x);
                }
            }
            match Self::encode_rgba_as_png(w, h, &pixels) {
                Ok(png) => Self::send_studio_screenshot_response(request_ids, w, h, png),
                Err(err) => crate::error!("studio screenshot png encode failed: {err}"),
            }
        }

        unsafe { self.os.display.as_mut().unwrap().swap_buffers() };

        //unsafe {
        //direct_app.drm.swap_buffers_and_wait(&direct_app.egl);
        //}
    }

    pub(crate) fn handle_repaint(&mut self) {
        let mut passes_todo = Vec::new();
        self.compute_pass_repaint_order(&mut passes_todo);
        self.repaint_id += 1;
        for draw_pass_id in &passes_todo {
            let uniforms_gen = self.next_uniform_gen();
            self.passes[*draw_pass_id]
                .set_time(self.os.timers.time_now() as f32, uniforms_gen);
            match self.passes[*draw_pass_id].parent.clone() {
                CxDrawPassParent::Xr => {}
                CxDrawPassParent::Window(_window_id) => {
                    self.draw_pass_to_fullscreen(*draw_pass_id);
                }
                CxDrawPassParent::DrawPass(_) => {
                    self.draw_pass_to_texture(*draw_pass_id, None);
                }
                CxDrawPassParent::None => {
                    self.draw_pass_to_texture(*draw_pass_id, None);
                }
            }
        }
    }

    fn handle_platform_ops(&mut self) -> EventFlow {
        while let Some(op) = self.platform_ops.pop_front() {
            //crate::log!("============ handle_platform_ops");
            match op {
                CxOsOp::CreateWindow(window_id) => {
                    let window = &mut self.windows[window_id];
                    window.os_dpi_factor = Some(self.os.dpi_factor);
                    let size = dvec2(
                        self.os.display_size.x / self.os.dpi_factor,
                        self.os.display_size.y / self.os.dpi_factor,
                    );
                    window.window_geom = WindowGeom {
                        dpi_factor: self.os.dpi_factor,
                        can_fullscreen: false,
                        xr_is_presenting: false,
                        is_fullscreen: true,
                        is_topmost: true,
                        position: dvec2(0.0, 0.0),
                        inner_size: size,
                        outer_size: size,
                        ..Default::default()
                    };
                    window.is_created = true;
                }
                CxOsOp::CreatePopupWindow {
                    window_id,
                    parent_window_id,
                    position,
                    size,
                    grab_keyboard,
                } => {
                    let window = &mut self.windows[window_id];
                    window.window_geom.position = position;
                    window.window_geom.inner_size = size;
                    window.window_geom.outer_size = size;
                    window.window_geom.dpi_factor = self.os.dpi_factor;
                    window.is_popup = true;
                    window.popup_parent = Some(parent_window_id);
                    window.popup_position = Some(position);
                    window.popup_size = Some(size);
                    window.popup_grab_keyboard = grab_keyboard;
                    window.is_created = true;
                }
                CxOsOp::StartTimer {
                    timer_id,
                    interval,
                    repeats,
                } => {
                    self.os
                        .timers
                        .timers
                        .insert(timer_id, PollTimer::new(interval, repeats));
                }
                CxOsOp::StopTimer(timer_id) => {
                    self.os.timers.timers.remove(&timer_id);
                }
                CxOsOp::Quit => {
                    self.os.quit = true;
                }
                CxOsOp::ShowTextIME(_area, _pos, config) => {
                    if self.os.last_ime_config.as_ref() != Some(&config) {
                        let _ = self.os.arkts_obj.as_mut().unwrap().call_js_function(
                            "showKeyBoard",
                            0,
                            std::ptr::null_mut(),
                        );
                        self.os.last_ime_config = Some(config);
                    }
                }
                CxOsOp::HideTextIME => {
                    self.os.last_ime_config = None;
                    let _ = self.os.arkts_obj.as_mut().unwrap().call_js_function(
                        "hideKeyBoard",
                        0,
                        std::ptr::null_mut(),
                    );
                    //self.os.keyboard_visible = false;
                    //unsafe {android_jni::to_java_show_keyboard(false);}
                }
                CxOsOp::StartExternalDragging { .. } => {
                    crate::error!("external file dragging is not implemented on OpenHarmony");
                    self.call_event_handler(&Event::DragEnd);
                }
                // Track selection is currently implemented on Linux GStreamer only.
                CxOsOp::SelectVideoTrack(_, _) | CxOsOp::SelectAudioTrack(_, _) => {}
                // The IME keeps no copy of the field's text on this platform
                // yet; there is nothing to bring in step.
                CxOsOp::SyncImeState { .. } => {}
                e => {
                    crate::error!("Not implemented on this platform: CxOsOp::{:?}", e);
                }
            }
        }
        EventFlow::Poll
    }
}

impl CxOsApi for Cx {
    fn init_cx_os(&mut self) {
        self.package_root = Some("makepad".to_string());
        self.native_load_dependencies();
    }

    fn open_url(&mut self, _url: &str, _in_place: OpenUrlInPlace) {
        crate::error!("open_url not implemented on this platform");
    }

    fn seconds_since_app_start(&self) -> f64 {
        Instant::now()
            .duration_since(self.os.start_time)
            .as_secs_f64()
    }
}

pub struct CxOhosDisplay {
    pub libegl: LibEgl,
    pub libgl: LibGl,
    pub egl_display: egl_sys::EGLDisplay,
    pub egl_config: egl_sys::EGLConfig,
    pub egl_context: egl_sys::EGLContext,
    pub surface: egl_sys::EGLSurface,
    pub window: *mut c_void, //event_handler: Box<dyn EventHandler>,
}

pub struct CxOs {
    pub first_after_resize: bool,
    pub display_size: Vec2d,
    pub dpi_factor: f64,
    pub media: CxOpenHarmonyMedia,
    pub quit: bool,
    pub timers: PollTimers,
    pub raw_file: Option<RawFileMgr>,
    pub arkts_obj: Option<ArkTsObjRef>,
    /// The keyboard config last shown, cleared when the keyboard goes down:
    /// a focused TextInput re-issues ShowTextIME every draw, and each call
    /// into ArkTS re-attaches the input method client, which drops what was
    /// typed in between.
    pub last_ime_config: Option<TextInputConfig>,
    pub(crate) start_time: Instant,
    pub(crate) display: Option<CxOhosDisplay>,
}

impl CxOs {
    pub(crate) fn gl(&self) -> &LibGl {
        &self.display.as_ref().unwrap().libgl
    }
}

impl Default for CxOs {
    fn default() -> Self {
        Self {
            first_after_resize: true,
            display_size: dvec2(1260 as f64, 2503 as f64),
            dpi_factor: 3.25,
            media: Default::default(),
            quit: false,
            timers: Default::default(),
            raw_file: None,
            arkts_obj: None,
            last_ime_config: None,
            start_time: Instant::now(),
            display: None,
        }
    }
}

impl CxOhosDisplay {
    unsafe fn destroy_surface(&mut self) {
        (self.libegl.eglMakeCurrent.unwrap())(
            self.egl_display,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        (self.libegl.eglDestroySurface.unwrap())(self.egl_display, self.surface);
        self.surface = std::ptr::null_mut();
    }

    unsafe fn update_surface(&mut self, window: *mut c_void) {
        if !self.window.is_null() {
            //todo release window
        }
        self.window = window;
        if self.surface.is_null() == false {
            self.destroy_surface();
        }

        let win_attr = vec![EGL_NONE];
        self.surface = (self.libegl.eglCreateWindowSurface.unwrap())(
            self.egl_display,
            self.egl_config,
            self.window as _,
            win_attr.as_ptr() as _,
        );

        if self.surface.is_null() {
            let err_code = unsafe { (self.libegl.eglGetError.unwrap())() };
            crate::log!("eglCreateWindowSurface error code:{}", err_code);
        }

        assert!(!self.surface.is_null());

        self.make_current();
    }

    unsafe fn swap_buffers(&mut self) {
        (self.libegl.eglSwapBuffers.unwrap())(self.egl_display, self.surface);
    }

    unsafe fn make_current(&mut self) {
        if (self.libegl.eglMakeCurrent.unwrap())(
            self.egl_display,
            self.surface,
            self.surface,
            self.egl_context,
        ) == 0
        {
            panic!();
        }
    }
}

/// The `"key":"value"` pairs of a flat JSON object of strings, in order.
/// Only `\"` and `\\` escapes are honoured — launch parameters are host
/// names, build ids and paths, never structured text.
fn flat_json_string_pairs(text: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut strings = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '"' {
            continue;
        }
        let mut s = String::new();
        loop {
            match chars.next() {
                Some('\\') => match chars.next() {
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    Some(other) => {
                        s.push('\\');
                        s.push(other);
                    }
                    None => break,
                },
                Some('"') | None => break,
                Some(other) => s.push(other),
            }
        }
        strings.push(s);
    }
    let mut iter = strings.into_iter();
    while let (Some(key), Some(value)) = (iter.next(), iter.next()) {
        pairs.push((key, value));
    }
    pairs
}

#[cfg(test)]
mod launch_parameter_tests {
    #[test]
    fn flat_pairs_come_out_in_order_with_escapes() {
        let pairs = super::flat_json_string_pairs(
            r#"{"makepad.STUDIO_HOST":"127.0.0.1:8002","makepad.PATH":"a\"b\\c"}"#,
        );
        assert_eq!(
            pairs,
            vec![
                ("makepad.STUDIO_HOST".into(), "127.0.0.1:8002".into()),
                ("makepad.PATH".into(), "a\"b\\c".into())
            ]
        );
        assert!(super::flat_json_string_pairs("{}").is_empty());
    }
}

/// The variant name of a channel message, for the startup log.
fn ohos_message_name(message: &FromOhosMessage) -> &'static str {
    match message {
        FromOhosMessage::Init { .. } => "Init",
        FromOhosMessage::SurfaceChanged { .. } => "SurfaceChanged",
        FromOhosMessage::SurfaceCreated { .. } => "SurfaceCreated",
        FromOhosMessage::SurfaceDestroyed => "SurfaceDestroyed",
        FromOhosMessage::VSync => "VSync",
        FromOhosMessage::Wake => "Wake",
        FromOhosMessage::Touch(_) => "Touch",
        FromOhosMessage::TextInput(_) => "TextInput",
        FromOhosMessage::DeleteLeft(_) => "DeleteLeft",
        FromOhosMessage::ResizeTextIME(..) => "ResizeTextIME",
    }
}

