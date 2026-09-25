//! The device camera inside a script app: a live preview drawn natively, and
//! capture and control from script.
//!
//! ```text
//! cam := CameraPreview{width: Fill height: Fill facing: "back"
//!     on_capture: || show(ui.cam.last())       // app-relative path of the new file
//!     on_error: || say(ui.cam.error())}
//! ui.cam.start()   ui.cam.stop()   ui.cam.switch()   ui.cam.facing()
//! ui.cam.capture()                     // a JPEG into the app's storage
//! ui.cam.record_start()  ui.cam.record_stop()   // an MP4
//! ui.cam.set_zoom(2.0)  ui.cam.set_flash("off"|"on"|"auto"|"torch")
//! ui.cam.set_exposure(-1.0)  ui.cam.focus(x, y)   // x, y 0..1 in the preview
//! ui.cam.set_aspect(16 / 9)            // the viewfinder shape; reopens the camera
//! ```
//!
//! A tap on the preview focuses there and a pinch zooms, natively
//! (`tap_to_focus`, `pinch_zoom`, both on by default); `on_zoom` hears the
//! new ratio once a pinch ends.
//!
//! Frames never pass through script: the preview is a platform texture, and
//! script hears one call per finished capture. What an app under a policy
//! gets: `camera` to open it at all; captures land in its own storage jail;
//! `microphone` adds sound to a recording; `library` also offers each capture
//! to the system photo library.
//!
//! The camera is a device the person can see is on, so it is released as soon
//! as it is not needed: when the preview has not drawn for a second (its tile
//! left the screen), when the app closes ([`release_isolate_devices`], which
//! the host calls), and when its isolate is collected.
use crate::{
    makepad_derive_widget::*,
    makepad_draw::*,
    makepad_script::ScriptFnRef,
    video::*,
    view::View,
    widget::*,
    widget_async::{CxWidgetToScriptCallExt, ScriptAsyncResult},
};
use crate::makepad_draw::makepad_platform::permission::{Permission, PermissionStatus};
use crate::makepad_draw::makepad_platform::video::{
    CameraCaptureEvent, CameraCaptureRequest, CameraCaptureResult, CameraControl, CameraFlashMode, VideoFormat,
    VideoFormatId, VideoInputId, VideoInputsEvent, VideoPixelFormat,
};
use crate::gesture_view::PinchTracker;
use std::collections::HashMap;

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.*

    mod.widgets.CameraPreviewBase = #(CameraPreview::register_widget(vm))
    mod.widgets.CameraPreview = set_type_default() do mod.widgets.CameraPreviewBase{
        width: Fill
        height: Fill
        flow: Overlay
        preview := Video{width: Fill height: Fill autoplay: false show_controls: false}
    }
}

const WATCHDOG_SECONDS: f64 = 0.5;
/// How long a camera may be without a permission answer before it opens
/// anyway: a bare desktop binary never gets the prompt's completion, and the
/// capture backend asks for access itself.
const PERMISSION_GRACE_SECONDS: f64 = 2.0;

thread_local! {
    /// heap key -> the preview videos that heap's cameras opened.
    static OPEN: std::cell::RefCell<HashMap<usize, Vec<VideoRef>>> = Default::default();
}

/// Stop every camera an isolate opened. Called from the isolate GC, and by a
/// host when the app it runs closes, which can be long before the GC.
pub fn release_cameras(cx: &mut Cx, heap_key: usize) {
    let videos = OPEN.with(|o| o.borrow_mut().remove(&heap_key)).unwrap_or_default();
    for video in videos {
        video.stop_and_cleanup_resources(cx);
    }
}

/// Everything a closing app holds that the person can see: its cameras and
/// its web views. A host calls this when it closes an app.
pub fn release_isolate_devices(cx: &mut Cx, heap_key: usize) {
    release_cameras(cx, heap_key);
    crate::web_reader::gc_web_readers(cx, &[heap_key]);
}

#[derive(Clone, Debug)]
struct CameraChoice {
    input_id: VideoInputId,
    front: bool,
    formats: Vec<VideoFormat>,
}

impl CameraChoice {
    /// The best preview profile for a viewfinder aspect: nearest aspect, then
    /// the most pixels within 1080p, then the better pixel format.
    fn format_for(&self, aspect: f64) -> Option<(VideoFormatId, usize, usize)> {
        let usable = |f: &&VideoFormat| {
            matches!(f.pixel_format, VideoPixelFormat::NV12 | VideoPixelFormat::YUY2 | VideoPixelFormat::YUV420)
                && f.width <= 1920
                && f.height <= 1080
                && f.height > 0
        };
        let pixel = |f: &VideoFormat| match f.pixel_format {
            VideoPixelFormat::NV12 => 3,
            VideoPixelFormat::YUY2 => 2,
            VideoPixelFormat::YUV420 => 1,
            _ => 0,
        };
        let key = |f: &VideoFormat| {
            let d = ((f.width as f64 / f.height as f64) - aspect).abs();
            ((d * 50.0).round() as i64, -((f.width * f.height) as i64), -pixel(f))
        };
        let best = self.formats.iter().filter(usable).min_by(|a, b| key(a).cmp(&key(b)))?;
        Some((best.format_id, best.width, best.height))
    }
}

/// Which camera a platform name describes: front when it says so, or the
/// second of several that does not say it is the back one.
fn is_front(name: &str, index: usize, count: usize) -> bool {
    let name = name.to_lowercase();
    name.contains("front") || name.contains("facetime") || name.contains("user") || (count > 1 && index == 1 && !name.contains("back"))
}

fn flash_mode(name: &str) -> Option<CameraFlashMode> {
    match name {
        "off" => Some(CameraFlashMode::Off),
        "on" => Some(CameraFlashMode::On),
        "auto" => Some(CameraFlashMode::Auto),
        "torch" => Some(CameraFlashMode::Torch),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum State {
    #[default]
    Idle,
    Starting,
    Running,
    Stopping,
}

#[derive(Script, ScriptHook, Widget)]
pub struct CameraPreview {
    #[deref]
    view: View,
    /// `"back"` or `"front"`.
    #[live]
    facing: String,
    /// Width over height of the viewfinder: 4:3 for photos, 16:9 for video.
    #[live(1.3333)]
    aspect: f64,
    #[live]
    on_capture: ScriptFnRef,
    #[live]
    on_error: ScriptFnRef,
    #[live]
    on_zoom: ScriptFnRef,
    #[live(true)]
    tap_to_focus: bool,
    #[live(true)]
    pinch_zoom: bool,

    #[rust]
    pinch: PinchTracker,
    #[rust]
    pinch_base: f32,
    /// The app asked for a preview (`start()`), and has not stopped it.
    #[rust]
    wanted: bool,
    #[rust]
    state: State,
    #[rust]
    cameras: Vec<CameraChoice>,
    #[rust]
    input: Option<VideoInputId>,
    #[rust]
    open_front: bool,
    #[rust]
    open_aspect: f64,
    #[rust]
    permission: Option<PermissionStatus>,
    #[rust]
    asked_at: f64,
    #[rust]
    timer: Option<Timer>,
    #[rust]
    drawn: bool,
    #[rust]
    registered: bool,
    #[rust]
    recording: Option<String>,
    #[rust]
    last: String,
    #[rust]
    last_error: String,
    #[rust]
    zoom: f32,
}

impl CameraPreview {
    fn heap_key(&self) -> usize {
        self.view.source.heap_key()
    }

    fn granted(&self, grant: &str) -> bool {
        crate::splash_policy::service_allowed(self.heap_key(), grant).is_ok()
    }

    fn fail(&mut self, cx: &mut Cx, why: String) {
        log!("camera preview: {why}");
        self.last_error = why;
        if self.on_error.as_object() != ScriptObject::ZERO {
            cx.widget_to_script_call(self.widget_uid(), NIL, self.view.source.clone(), self.on_error.clone(), &[]);
        }
    }

    fn video(&self, cx: &Cx) -> VideoRef {
        self.view.video(cx, ids!(preview))
    }

    fn start(&mut self, cx: &mut Cx) -> bool {
        if !self.granted("camera.preview") {
            self.fail(cx, "this app was not granted the camera".into());
            return false;
        }
        self.wanted = true;
        if self.permission.is_none() {
            self.asked_at = cx.seconds_since_app_start();
            cx.request_permission(Permission::Camera);
        }
        if self.cameras.is_empty() {
            // The platform publishes its list once; a preview mounted later
            // asks again. The frame callback opens the capture session.
            cx.video_input(0, |_frame| {});
            cx.refresh_video_inputs();
        }
        if self.timer.is_none() {
            self.timer = Some(cx.start_interval(WATCHDOG_SECONDS));
        }
        self.drawn = true;
        self.drive(cx);
        true
    }

    fn stop(&mut self, cx: &mut Cx) {
        self.wanted = false;
        self.halt(cx);
        if let Some(timer) = self.timer.take() {
            cx.stop_timer(timer);
        }
    }

    /// Close the camera, keeping whether the app wants it: the watchdog
    /// reopens it when the preview draws again.
    fn halt(&mut self, cx: &mut Cx) {
        if matches!(self.state, State::Starting | State::Running) {
            if self.recording.take().is_some() {
                if let Some(input) = self.input {
                    cx.camera_capture(input, CameraCaptureRequest::StopVideo);
                }
            }
            self.state = State::Stopping;
            self.video(cx).stop_and_cleanup_resources(cx);
        }
    }

    /// One step towards what the app wants, driven by the platform's events.
    fn drive(&mut self, cx: &mut Cx) {
        let want_front = self.facing == "front";
        match self.state {
            State::Running if want_front != self.open_front || (self.aspect - self.open_aspect).abs() > 0.05 => {
                if self.cameras.iter().any(|c| c.front == want_front) {
                    self.halt(cx);
                }
            }
            State::Idle if self.wanted => {
                match self.permission {
                    Some(PermissionStatus::Granted) => {}
                    Some(status) => {
                        self.wanted = false;
                        self.fail(cx, format!("camera permission {status:?}"));
                        return;
                    }
                    None if cx.seconds_since_app_start() - self.asked_at < PERMISSION_GRACE_SECONDS => return,
                    None => self.permission = Some(PermissionStatus::Granted),
                }
                let Some(choice) = self.cameras.iter().find(|c| c.front == want_front).or(self.cameras.first()).cloned() else {
                    return;
                };
                let Some((format_id, _, _)) = choice.format_for(self.aspect) else { return };
                let video = self.video(cx);
                video.set_camera_preview_mode(cx, VideoCameraPreviewMode::Texture);
                video.set_source_camera(cx, choice.input_id, format_id);
                video.begin_playback(cx);
                if !self.registered {
                    self.registered = true;
                    OPEN.with(|o| o.borrow_mut().entry(self.heap_key()).or_default().push(video.clone()));
                }
                self.state = State::Starting;
                self.input = Some(choice.input_id);
                self.open_front = choice.front;
                self.open_aspect = self.aspect;
            }
            _ => {}
        }
    }

    fn capture_path(&self, cx: &Cx, kind: &str, ext: &str) -> Option<(String, String)> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let relative = format!("DCIM/{kind}_{now}.{ext}");
        // A policed app's captures are its own: they land in its jail. The
        // host's own surfaces write next to the host's data.
        let real = if crate::splash_policy::is_enforced(self.heap_key()) {
            if !crate::splash_storage::heap_has_room(self.heap_key()) {
                return None;
            }
            crate::splash_policy::local_path_for_heap(self.heap_key(), &relative)?
        } else {
            format!("{}/{relative}", cx.get_data_dir()?)
        };
        if let Some(parent) = std::path::Path::new(&real).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        Some((relative, real))
    }

    fn capture(&mut self, cx: &mut Cx) -> bool {
        let (Some(input), State::Running) = (self.input, self.state) else {
            self.fail(cx, "the camera is not running".into());
            return false;
        };
        let Some((relative, real)) = self.capture_path(cx, "IMG", "jpg") else {
            self.fail(cx, "this app's storage is full or missing".into());
            return false;
        };
        self.last = relative;
        let library = self.granted("library.write");
        cx.camera_capture(input, CameraCaptureRequest::Photo { path: real, library });
        true
    }

    fn record_start(&mut self, cx: &mut Cx) -> bool {
        let (Some(input), State::Running) = (self.input, self.state) else {
            self.fail(cx, "the camera is not running".into());
            return false;
        };
        if self.recording.is_some() {
            return true;
        }
        let Some((relative, real)) = self.capture_path(cx, "VID", "mp4") else {
            self.fail(cx, "this app's storage is full or missing".into());
            return false;
        };
        let audio = self.granted("microphone.record");
        if audio {
            cx.request_permission(Permission::AudioInput);
        }
        self.recording = Some(relative);
        let library = self.granted("library.write");
        cx.camera_capture(input, CameraCaptureRequest::StartVideo { path: real, audio, library });
        true
    }

    fn record_stop(&mut self, cx: &mut Cx) {
        if let (Some(input), Some(_)) = (self.input, self.recording.as_ref()) {
            cx.camera_capture(input, CameraCaptureRequest::StopVideo);
        }
    }

    fn control(&mut self, cx: &mut Cx, control: CameraControl) {
        if let (Some(input), State::Running) = (self.input, self.state) {
            cx.camera_control(input, control);
        }
    }

    fn on_capture(&mut self, cx: &mut Cx, result: &CameraCaptureResult) {
        match result {
            CameraCaptureResult::Photo { .. } => self.notify_capture(cx),
            CameraCaptureResult::VideoStopped { .. } => {
                if let Some(relative) = self.recording.take() {
                    self.last = relative;
                }
                self.notify_capture(cx);
            }
            CameraCaptureResult::Failed { what, error } => {
                if what == "video" {
                    self.recording = None;
                }
                self.fail(cx, format!("{what} failed: {error}"));
            }
            _ => {}
        }
    }

    fn notify_capture(&mut self, cx: &mut Cx) {
        if self.on_capture.as_object() != ScriptObject::ZERO {
            cx.widget_to_script_call(self.widget_uid(), NIL, self.view.source.clone(), self.on_capture.clone(), &[]);
        }
    }
}

impl CameraPreview {
    fn handle_gestures(&mut self, cx: &mut Cx, event: &Event) {
        let area = self.view.area();
        let rect = area.rect(cx);
        if let (true, Event::TouchUpdate(touches)) = (self.pinch_zoom, event) {
            use crate::gesture_view::PinchStep;
            match self.pinch.update(touches, area.clipped_rect(cx)) {
                PinchStep::Began(_) => self.pinch_base = if self.zoom > 0.0 { self.zoom } else { 1.0 },
                PinchStep::Moved(scale, _) => {
                    let zoom = (self.pinch_base * scale as f32).clamp(1.0, 10.0);
                    if (zoom - self.zoom).abs() >= 0.02 {
                        self.zoom = zoom;
                        self.control(cx, CameraControl::ZoomRatio(zoom));
                    }
                }
                PinchStep::Ended => {
                    if self.on_zoom.as_object() != ScriptObject::ZERO {
                        cx.widget_to_script_call(self.widget_uid(), NIL, self.view.source.clone(), self.on_zoom.clone(), &[(self.zoom as f64).into()]);
                    }
                }
                PinchStep::Nothing => {}
            }
        }
        if !self.tap_to_focus || rect.size.x <= 0.0 || rect.size.y <= 0.0 {
            return;
        }
        // A tap the controls drawn over the preview did not claim.
        if let Hit::FingerUp(e) = event.hits_with_capture_overload(cx, area, true) {
            let child_claimed = cx.fingers.digit_capture_areas(e.digit_id).into_iter().any(|a| a != area && a.rect(cx).is_inside_of(rect) && a.rect(cx).size != rect.size);
            if e.is_over && e.was_tap() && !self.pinch.pinching() && !child_claimed {
                let x = (e.abs.x - rect.pos.x) / rect.size.x;
                let y = (e.abs.y - rect.pos.y) / rect.size.y;
                self.control(cx, CameraControl::FocusPoint { x: x.clamp(0.0, 1.0), y: y.clamp(0.0, 1.0) });
            }
        }
    }
}

fn arg_number(vm: &mut ScriptVm, args: ScriptValue, index: usize) -> Option<f64> {
    let args_obj = args.as_object()?;
    let trap = vm.bx.threads.cur().trap.pass();
    vm.bx.heap.vec_value(args_obj, index, trap).as_number()
}

fn arg_string(vm: &mut ScriptVm, args: ScriptValue, index: usize) -> Option<String> {
    let args_obj = args.as_object()?;
    let trap = vm.bx.threads.cur().trap.pass();
    let value = vm.bx.heap.vec_value(args_obj, index, trap);
    vm.bx.heap.cast_to_owned_string(value, "copying a camera argument")
}

impl Widget for CameraPreview {
    fn script_call(&mut self, vm: &mut ScriptVm, method: LiveId, args: ScriptValue) -> ScriptAsyncResult {
        let ret = |v: ScriptValue| ScriptAsyncResult::Return(v);
        match method {
            m if m == live_id!(start) => ret(vm.with_cx_mut(|cx| self.start(cx)).into()),
            m if m == live_id!(stop) => {
                vm.with_cx_mut(|cx| self.stop(cx));
                ret(NIL)
            }
            m if m == live_id!(switch) => {
                self.facing = if self.facing == "front" { "back".into() } else { "front".into() };
                vm.with_cx_mut(|cx| self.drive(cx));
                ret(NIL)
            }
            m if m == live_id!(set_facing) => {
                if let Some(facing) = arg_string(vm, args, 0) {
                    self.facing = facing;
                    vm.with_cx_mut(|cx| self.drive(cx));
                }
                ret(NIL)
            }
            m if m == live_id!(set_aspect) => {
                if let Some(aspect) = arg_number(vm, args, 0).filter(|a| *a > 0.2 && *a < 5.0) {
                    self.aspect = aspect;
                    vm.with_cx_mut(|cx| self.drive(cx));
                }
                ret(NIL)
            }
            m if m == live_id!(facing) => ret(vm.bx.heap.new_string_from_str(if self.facing == "front" { "front" } else { "back" })),
            m if m == live_id!(capture) => ret(vm.with_cx_mut(|cx| self.capture(cx)).into()),
            m if m == live_id!(record_start) => ret(vm.with_cx_mut(|cx| self.record_start(cx)).into()),
            m if m == live_id!(record_stop) => {
                vm.with_cx_mut(|cx| self.record_stop(cx));
                ret(NIL)
            }
            m if m == live_id!(is_recording) => ret(self.recording.is_some().into()),
            m if m == live_id!(is_running) => ret((self.state == State::Running).into()),
            m if m == live_id!(set_zoom) => {
                if let Some(zoom) = arg_number(vm, args, 0) {
                    self.zoom = zoom.clamp(0.5, 20.0) as f32;
                    let zoom = self.zoom;
                    vm.with_cx_mut(|cx| self.control(cx, CameraControl::ZoomRatio(zoom)));
                }
                ret(NIL)
            }
            m if m == live_id!(zoom) => ret((if self.zoom > 0.0 { self.zoom as f64 } else { 1.0 }).into()),
            m if m == live_id!(set_flash) => {
                if let Some(mode) = arg_string(vm, args, 0).as_deref().and_then(flash_mode) {
                    vm.with_cx_mut(|cx| self.control(cx, CameraControl::Flash(mode)));
                }
                ret(NIL)
            }
            m if m == live_id!(set_exposure) => {
                if let Some(ev) = arg_number(vm, args, 0) {
                    vm.with_cx_mut(|cx| self.control(cx, CameraControl::ExposureBias(ev.clamp(-4.0, 4.0) as f32)));
                }
                ret(NIL)
            }
            m if m == live_id!(focus) => {
                if let (Some(x), Some(y)) = (arg_number(vm, args, 0), arg_number(vm, args, 1)) {
                    vm.with_cx_mut(|cx| {
                        self.control(cx, CameraControl::FocusPoint { x: x.clamp(0.0, 1.0), y: y.clamp(0.0, 1.0) })
                    });
                }
                ret(NIL)
            }
            m if m == live_id!(last) => ret(vm.bx.heap.new_string_from_str(&self.last)),
            m if m == live_id!(error) => ret(vm.bx.heap.new_string_from_str(&self.last_error)),
            _ => self.view.script_call(vm, method, args),
        }
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        self.view.handle_event(cx, event, scope);
        if self.timer.is_some_and(|t| t.is_event(event).is_some()) {
            if self.drawn {
                self.drawn = false;
                if self.state == State::Idle {
                    self.drive(cx);
                }
                self.view.redraw(cx);
            } else if matches!(self.state, State::Starting | State::Running) {
                // Not drawn since the last tick: the preview left the screen.
                self.halt(cx);
            }
        }
        if self.state == State::Running {
            self.handle_gestures(cx, event);
        }
        match event {
            Event::PermissionResult(result) if result.permission == Permission::Camera => {
                self.permission = Some(result.status);
                self.drive(cx);
            }
            Event::VideoInputs(VideoInputsEvent { descs }) => {
                let count = descs.len();
                self.cameras = descs
                    .iter()
                    .enumerate()
                    .map(|(i, d)| CameraChoice { input_id: d.input_id, front: is_front(&d.name, i, count), formats: d.formats.clone() })
                    .filter(|c| c.format_for(4.0 / 3.0).is_some())
                    .collect();
                self.drive(cx);
            }
            Event::VideoPlaybackPrepared(_) => {
                if self.state == State::Starting {
                    self.state = State::Running;
                }
            }
            Event::VideoPlaybackResourcesReleased(_) => {
                if self.state == State::Stopping {
                    self.state = State::Idle;
                }
            }
            Event::VideoDecodingError(e) => {
                if matches!(self.state, State::Starting | State::Running) {
                    self.state = State::Idle;
                    let why = format!("the camera stopped: {}", e.error);
                    self.fail(cx, why);
                }
            }
            Event::Actions(actions) => {
                for action in actions {
                    let Some(capture) = action.downcast_ref::<CameraCaptureEvent>() else { continue };
                    if Some(capture.input_id) == self.input {
                        let result = capture.result.clone();
                        self.on_capture(cx, &result);
                    }
                }
            }
            _ => {}
        }
    }

    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        self.drawn = true;
        self.view.draw_walk(cx, scope, walk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format(width: usize, height: usize, pixel_format: VideoPixelFormat, id: u64) -> VideoFormat {
        VideoFormat { format_id: VideoFormatId(LiveId(id)), width, height, frame_rate: Some(30.0), pixel_format }
    }

    #[test]
    fn the_preview_profile_matches_the_viewfinder_then_takes_the_most_pixels() {
        let choice = CameraChoice {
            input_id: VideoInputId(LiveId(1)),
            front: false,
            formats: vec![
                format(640, 480, VideoPixelFormat::NV12, 1),
                format(1440, 1080, VideoPixelFormat::NV12, 2),
                format(1920, 1080, VideoPixelFormat::NV12, 3),
                format(4032, 3024, VideoPixelFormat::NV12, 4),
            ],
        };
        assert_eq!(choice.format_for(4.0 / 3.0).map(|f| (f.1, f.2)), Some((1440, 1080)), "4:3, and within 1080p");
        assert_eq!(choice.format_for(16.0 / 9.0).map(|f| (f.1, f.2)), Some((1920, 1080)));
    }

    #[test]
    fn a_camera_is_front_when_it_says_so_or_is_the_unnamed_second() {
        assert!(is_front("Front Camera", 0, 2));
        assert!(is_front("FaceTime HD Camera", 0, 1));
        assert!(!is_front("Back Camera", 1, 2));
        assert!(is_front("camera1", 1, 2));
        assert!(!is_front("camera0", 0, 2));
    }

    #[test]
    fn flash_names_map_and_unknown_ones_are_ignored() {
        assert_eq!(flash_mode("torch"), Some(CameraFlashMode::Torch));
        assert_eq!(flash_mode("strobe"), None);
    }
}
