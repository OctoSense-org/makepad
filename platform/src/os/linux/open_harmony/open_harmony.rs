use {
    self::super::{
        super::gl_sys, super::gl_sys::LibGl, arkts_obj_ref::ArkTsObjRef, oh_callbacks::*,
        oh_camera::OhCameraPlayer, oh_media::CxOpenHarmonyMedia, raw_file::RawFileMgr,
    },
    crate::{
        cx::{Cx, OpenHarmonyParams, OsType},
        cx_api::{CxOsApi, CxOsOp, OpenUrlInPlace},
        event::video_playback::{
            VideoDecodingErrorEvent, VideoPlaybackResourcesReleasedEvent, VideoSource,
            VideoYuvTexturesReady,
        },
        permission::{Permission, PermissionResult, PermissionStatus},
        texture::TextureFormat,
        draw_pass::{CxDrawPassParent, DrawPassClearColor, DrawPassClearDepth, DrawPassId},
        egl_sys::{self, LibEgl, EGL_NONE},
        event::{
            Event, KeyCode, KeyEvent, SafeAreaInsets, TouchUpdateEvent, VirtualKeyboardEvent,
            WindowGeom,
        },
        gpu_info::GpuPerformance,
        makepad_live_id::LiveId,
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
            match from_ohos_rx.recv() {
                Ok(FromOhosMessage::VSync) => {
                    self.handle_all_pending_messages(&from_ohos_rx);
                    self.handle_other_events();
                    self.handle_drawing();
                }
                Ok(message) => self.handle_message(message),
                Err(e) => {
                    crate::error!("Error receiving message: {:?}", e);
                }
            }
        }
        self.call_event_handler(&Event::Shutdown);
    }

    fn handle_all_pending_messages(&mut self, from_ohos_rx: &mpsc::Receiver<FromOhosMessage>) {
        // Handle the messages that arrived during the last frame
        while let Ok(msg) = from_ohos_rx.try_recv() {
            self.handle_message(msg);
        }
    }

    fn handle_other_events(&mut self) {
        // The remote instrument (hdc fport → 127.0.0.1) queues its commands off-thread.
        self.poll_control_channel();
        // Camera frames read back by the image receiver since the last vsync.
        self.poll_ohos_camera_players();
        // HTTP / WebSocket / storage completions (http_resource assets, script fetches).
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
        if self.any_passes_dirty() || self.need_redrawing() || !self.new_next_frames.is_empty() {
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
                let safe_area_insets = self.os.native_safe_area_insets.scale(1.0 / dpi_factor);
                window.window_geom = WindowGeom {
                    dpi_factor,
                    can_fullscreen: false,
                    xr_is_presenting: false,
                    is_fullscreen: true,
                    is_topmost: true,
                    position: dvec2(0.0, 0.0),
                    inner_size: size,
                    outer_size: size,
                    safe_area_insets,
                    ..Default::default()
                };
                let new_geom = window.window_geom.clone();
                self.display_context.safe_area_insets = safe_area_insets;
                self.update_safe_inset_script_values(safe_area_insets);
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
            FromOhosMessage::PermissionResult {
                permission,
                granted,
                can_retry,
            } => {
                let status = if granted {
                    PermissionStatus::Granted
                } else if can_retry {
                    PermissionStatus::DeniedCanRetry
                } else {
                    PermissionStatus::DeniedPermanent
                };
                let wanted = ohos_permission_from_name(&permission);
                let pending: Vec<(Permission, i32)> = self.os.pending_permissions.drain(..).collect();
                for (perm, request_id) in pending {
                    if Some(perm) == wanted {
                        crate::log!("ohos: permission {perm:?} → {status:?}");
                        self.call_event_handler(&Event::PermissionResult(PermissionResult {
                            permission: perm,
                            request_id,
                            status,
                        }));
                    } else {
                        self.os.pending_permissions.push((perm, request_id));
                    }
                }
            }
            FromOhosMessage::CaptureSaved { path, ok, uri } => {
                crate::log!("ohos: gallery {} {path}", if ok { "took" } else { "refused" });
                self.action(crate::video::CameraCaptureEvent {
                    input_id: Default::default(),
                    result: crate::video::CameraCaptureResult::SavedToLibrary { path, uri: if ok { Some(uri) } else { None } },
                });
            }
            FromOhosMessage::AvoidArea {
                top,
                right,
                bottom,
                left,
            } => {
                let insets = SafeAreaInsets {
                    top,
                    right,
                    bottom,
                    left,
                };
                if self.os.native_safe_area_insets != insets {
                    crate::log!("ohos: safe area px top {top} right {right} bottom {bottom} left {left}");
                    self.os.native_safe_area_insets = insets;
                    self.ohos_publish_safe_area();
                }
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
        if let Ok(FromOhosMessage::Init {
            device_type,
            os_full_name,
            display_density,
            files_dir,
            cache_dir,
            temp_dir,
            raw_env,
            arkts_ref,
            raw_file,
        }) = from_ohos_rx.recv()
        {
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
        } else {
            crate::error!("Failed to receive init message from ArkTS layer");
            return false;
        }
    }

    fn wait_surface_created(
        &mut self,
        from_ohos_rx: &mpsc::Receiver<FromOhosMessage>,
    ) -> *mut c_void {
        if let Ok(FromOhosMessage::SurfaceCreated {
            window,
            width,
            height,
        }) = from_ohos_rx.recv()
        {
            self.os.display_size = dvec2(width as f64, height as f64);
            crate::log!(
                "handle surface created, width={}, height={}, display_density={}",
                width,
                height,
                self.os.dpi_factor
            );
            return window;
        } else {
            crate::error!("Can't recv SurfaceCreated from arkts");
            return null_mut();
        }
    }

    pub fn ohos_init<F>(exports: JsObject, env: Env, startup: F)
    where
        F: FnOnce() -> Box<Cx> + Send + 'static,
    {
        crate::log!("ohos init");
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(move || {
            std::panic::set_hook(Box::new(|info| {
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

    /// A resource read straight from the HAP's rawfile, as the asset table
    /// lays it out: `path`, then `<package_root>/path` (`makepad/<crate>/…`).
    /// Dependencies register after startup (fonts, script resources), so
    /// `get_dependency` reads them on demand, as the Android asset path does.
    pub(crate) fn ohos_read_raw(&self, path: &str) -> Option<Vec<u8>> {
        let raw_file = self.os.raw_file.as_ref()?;
        let mut buffer = Vec::new();
        if raw_file.read_to_end(path, &mut buffer).is_ok() && !buffer.is_empty() {
            return Some(buffer);
        }
        if let Some(root) = self.package_root.as_deref() {
            let prefix = format!("{root}/");
            if !path.starts_with(&prefix) {
                buffer.clear();
                if raw_file.read_to_end(format!("{root}/{path}"), &mut buffer).is_ok() && !buffer.is_empty() {
                    return Some(buffer);
                }
            }
        }
        None
    }

    pub fn ohos_load_dependencies(&mut self) {
        let (mut ok, mut failed) = (0usize, 0usize);
        for (path, dep) in &mut self.dependencies {
            let mut buffer = Vec::<u8>::new();
            match self.os.raw_file.as_ref().unwrap().read_to_end(path, &mut buffer) {
                Ok(_) => { ok += 1; dep.data = Some(Ok(Rc::new(buffer))); }
                Err(e) => {
                    failed += 1;
                    crate::error!("ohos: cannot load dependency {path} from rawfile: {e}");
                    dep.data = Some(Err(format!("read_to_end failed: {e}")));
                }
            }
        }
        crate::log!("ohos: dependencies loaded: {ok} ok, {failed} failed");
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

        // Read only this app's framebuffer, before EGL hands it to the
        // compositor. The remote bridge's /g requests otherwise time out.
        let window_id = self.get_pass_window_id(draw_pass_id).map(|window| window.id());
        let requests = self.take_studio_screenshot_request_ids_for_window(0, window_id);
        if !requests.is_empty() {
            let width = self.os.display_size.x as u32;
            let height = self.os.display_size.y as u32;
            let stride = width as usize * 4;
            let mut pixels = vec![0u8; stride * height as usize];
            unsafe {
                let gl = self.os.gl();
                (gl.glReadPixels)(0, 0, width as i32, height as i32,
                    gl_sys::RGBA, gl_sys::UNSIGNED_BYTE, pixels.as_mut_ptr() as *mut _);
            }
            // OpenGL's origin is bottom-left; PNG rows start at the top.
            for row in 0..height as usize / 2 {
                let top = row * stride;
                let bottom = (height as usize - 1 - row) * stride;
                for column in 0..stride {
                    pixels.swap(top + column, bottom + column);
                }
            }
            if let Ok(png) = Self::encode_rgba_as_png(width, height, &pixels) {
                Self::send_studio_screenshot_response(requests, width, height, png);
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
                    let safe_area_insets =
                        self.os.native_safe_area_insets.scale(1.0 / self.os.dpi_factor);
                    window.window_geom = WindowGeom {
                        dpi_factor: self.os.dpi_factor,
                        can_fullscreen: false,
                        xr_is_presenting: false,
                        is_fullscreen: true,
                        is_topmost: true,
                        position: dvec2(0.0, 0.0),
                        inner_size: size,
                        outer_size: size,
                        safe_area_insets,
                        ..Default::default()
                    };
                    window.is_created = true;
                    self.display_context.safe_area_insets = safe_area_insets;
                    self.update_safe_inset_script_values(safe_area_insets);
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
                CxOsOp::ShowTextIME(_area, _pos, _config) => {
                    let _ = self.os.arkts_obj.as_mut().unwrap().call_js_function(
                        "showKeyBoard",
                        0,
                        std::ptr::null_mut(),
                    );
                }
                CxOsOp::HideTextIME => {
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
                CxOsOp::CheckPermission { permission, request_id } => {
                    // No NDK query for user_grant permissions without the
                    // token machinery; ArkTS answers through the request path.
                    self.ohos_request_permission(permission, request_id);
                }
                CxOsOp::RequestPermission { permission, request_id } => {
                    self.ohos_request_permission(permission, request_id);
                }
                CxOsOp::PrepareVideoPlayback(
                    video_id,
                    source,
                    _camera_preview_mode,
                    _external_texture_id,
                    _texture_id,
                    autoplay,
                    _should_loop,
                ) => {
                    if let Some(mut player) = self.os.media.camera_players.remove(&video_id) {
                        self.ohos_release_camera_player(&mut player);
                        self.call_event_handler(&Event::VideoPlaybackResourcesReleased(
                            VideoPlaybackResourcesReleasedEvent { video_id },
                        ));
                    }
                    match source {
                        VideoSource::Camera(input_id, format_id) => {
                            self.ohos_prepare_camera_playback(video_id, input_id, format_id, autoplay);
                        }
                        _ => {
                            self.call_event_handler(&Event::VideoDecodingError(VideoDecodingErrorEvent {
                                video_id,
                                error: "video playback is not implemented on OpenHarmony (camera sources only)".into(),
                            }));
                        }
                    }
                }
                CxOsOp::BeginVideoPlayback(video_id) | CxOsOp::ResumeVideoPlayback(video_id) => {
                    if let Some(player) = self.os.media.camera_players.get_mut(&video_id) {
                        player.playing = true;
                    }
                }
                CxOsOp::PauseVideoPlayback(video_id) => {
                    if let Some(player) = self.os.media.camera_players.get_mut(&video_id) {
                        player.playing = false;
                    }
                }
                CxOsOp::MuteVideoPlayback(_) | CxOsOp::UnmuteVideoPlayback(_) => {}
                // Touch screens have no pointer to shape.
                CxOsOp::SetCursor(_) => {}
                CxOsOp::CleanupVideoPlaybackResources(video_id) => {
                    if let Some(mut player) = self.os.media.camera_players.remove(&video_id) {
                        self.ohos_release_camera_player(&mut player);
                    }
                    self.call_event_handler(&Event::VideoPlaybackResourcesReleased(
                        VideoPlaybackResourcesReleasedEvent { video_id },
                    ));
                }
                CxOsOp::AttachCameraNativePreview { .. }
                | CxOsOp::UpdateCameraNativePreview { .. }
                | CxOsOp::DetachCameraNativePreview { .. } => {}
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
    /// Permission requests waiting for ArkTS to answer, by request id.
    pub(crate) pending_permissions: Vec<(Permission, i32)>,
    /// System bar avoid areas in physical pixels (the window is edge to edge).
    pub(crate) native_safe_area_insets: SafeAreaInsets,
    pub(crate) start_time: Instant,
    pub(crate) display: Option<CxOhosDisplay>,
}

impl Cx {
    /// `sys.gps` asks for a fix on every read; OpenHarmony has no location
    /// bridge yet, so this only notes the request once and the card sees
    /// "no fix" through `gps::last_gps_fix()`.
    pub fn ohos_request_gps(&mut self) {
        static ASKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !ASKED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            crate::log!("ohos: gps requested; no location service is wired up on OpenHarmony yet");
        }
    }
}

/// The ArkTS glue exposes one zero-argument request function per permission
/// (napi values cannot be built off the JS thread), and reports back with the
/// same short name through `handlePermissionResult`.
fn ohos_permission_js_function(permission: Permission) -> Option<(&'static str, &'static str)> {
    match permission {
        Permission::Camera => Some(("camera", "requestCameraPermission")),
        Permission::AudioInput => Some(("microphone", "requestMicrophonePermission")),
        Permission::Location => Some(("location", "requestLocationPermission")),
        Permission::HeadsetCamera | Permission::SceneAccess => None,
    }
}

fn ohos_permission_from_name(name: &str) -> Option<Permission> {
    match name {
        "camera" => Some(Permission::Camera),
        "microphone" => Some(Permission::AudioInput),
        "location" => Some(Permission::Location),
        _ => None,
    }
}

impl Cx {
    /// New avoid areas: republish every window's geometry so the app lays its
    /// chrome out inside the safe area.
    fn ohos_publish_safe_area(&mut self) {
        let window_ids: Vec<_> = self
            .windows
            .id_iter()
            .filter(|id| self.windows[*id].is_created)
            .collect();
        for window_id in window_ids {
            let window = &mut self.windows[window_id];
            let dpi_factor = window.dpi_override.unwrap_or(self.os.dpi_factor);
            let insets = self.os.native_safe_area_insets.scale(1.0 / dpi_factor);
            let old_geom = window.window_geom.clone();
            window.window_geom.safe_area_insets = insets;
            let new_geom = window.window_geom.clone();
            self.display_context.safe_area_insets = insets;
            self.update_safe_inset_script_values(insets);
            self.call_event_handler(&Event::WindowGeomChange(WindowGeomChangeEvent {
                window_id,
                new_geom,
                old_geom,
            }));
        }
        self.redraw_all();
    }

    fn ohos_request_permission(&mut self, permission: Permission, request_id: i32) {
        let Some((_, function)) = ohos_permission_js_function(permission) else {
            self.call_event_handler(&Event::PermissionResult(PermissionResult {
                permission,
                request_id,
                status: PermissionStatus::DeniedPermanent,
            }));
            return;
        };
        let Some(arkts) = self.os.arkts_obj.as_mut() else {
            crate::error!("ohos: no ArkTS bridge, cannot request {permission:?}");
            self.call_event_handler(&Event::PermissionResult(PermissionResult {
                permission,
                request_id,
                status: PermissionStatus::NotDetermined,
            }));
            return;
        };
        self.os.pending_permissions.push((permission, request_id));
        if let Err(e) = arkts.call_js_function(function, 0, std::ptr::null_mut()) {
            crate::error!("ohos: {function} failed: {e:?}");
            self.os.pending_permissions.retain(|(_, id)| *id != request_id);
            self.call_event_handler(&Event::PermissionResult(PermissionResult {
                permission,
                request_id,
                status: PermissionStatus::NotDetermined,
            }));
        }
    }

    /// Bind a Video widget to the camera: three R8 plane textures, the
    /// shared frame slot of the (started) stream, and the textures-ready
    /// event the widget needs before the first `VideoTextureUpdated`.
    fn ohos_prepare_camera_playback(
        &mut self,
        video_id: LiveId,
        input_id: crate::video::VideoInputId,
        format_id: crate::video::VideoFormatId,
        autoplay: bool,
    ) {
        let camera = self.os.media.camera();
        let (shared, rotation_steps, front) = {
            let mut cam = camera.lock().unwrap();
            cam.use_video_input(&[(input_id, format_id)]);
            (cam.frame_shared(), cam.rotation_steps(), cam.is_front(input_id))
        };
        let Some(shared) = shared else {
            self.call_event_handler(&Event::VideoDecodingError(VideoDecodingErrorEvent {
                video_id,
                error: "the camera could not be opened (see the ohos camera log lines)".into(),
            }));
            return;
        };
        let _ = front;
        let tex_y = self.textures.alloc(TextureFormat::VideoYuvPlane);
        let tex_u = self.textures.alloc(TextureFormat::VideoYuvPlane);
        let tex_v = self.textures.alloc(TextureFormat::VideoYuvPlane);
        let player = OhCameraPlayer {
            video_id,
            tex_y: tex_y.texture_id(),
            tex_u: tex_u.texture_id(),
            tex_v: tex_v.texture_id(),
            shared,
            rotation_steps,
            width: 0,
            height: 0,
            prepared: false,
            playing: autoplay,
        };
        self.os.media.camera_players.insert(video_id, player);
        self.call_event_handler(&Event::VideoYuvTexturesReady(VideoYuvTexturesReady::planes(
            video_id, tex_y, tex_u, tex_v,
        )));
    }

    fn ohos_release_camera_player(&mut self, player: &mut OhCameraPlayer) {
        player.playing = false;
        // The stream itself stops when no Video widget is bound to it any more.
        if self.os.media.camera_players.is_empty() {
            self.os.media.camera().lock().unwrap().use_video_input(&[]);
        }
    }
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
            pending_permissions: Vec::new(),
            native_safe_area_insets: SafeAreaInsets::default(),
            raw_file: None,
            arkts_obj: None,
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

    pub(crate) unsafe fn make_current(&mut self) {
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
