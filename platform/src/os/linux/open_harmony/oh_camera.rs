//! OpenHarmony camera as a Makepad video input and video-playback source.
//!
//! The camera NDK (`libohcamera.so`) streams preview frames into a surface.
//! There is no path from that surface into a GL texture that Makepad owns, so
//! the preview output is bound to an `ImageReceiver` (`libimage_receiver.so`)
//! and every frame is read back on the receiver's thread as a YUV 420
//! semi-planar buffer, converted to I420 and handed to the UI thread, which
//! uploads the three planes to the Video widget's textures. That is the same
//! shape as the desktop V4L2 camera player; the cost is one CPU copy per frame
//! at preview resolution, which the Mate 70 Air handles comfortably.
//!
//! Every NDK symbol is resolved with `dlopen`/`dlsym` at first use so the
//! platform crate links exactly as before; a device without the camera NDK
//! simply reports no video inputs.
use {
    super::super::module_loader::ModuleLoader,
    crate::{
        makepad_live_id::LiveId,
        thread::SignalToUI,
        video::*,
    },
    std::{
        collections::VecDeque,
        ffi::{c_char, c_void, CStr, CString},
        os::fd::AsRawFd,
        sync::{
            atomic::{AtomicBool, AtomicU64, Ordering},
            Arc, Mutex,
        },
    },
};

// ---------------------------------------------------------------------------
// NDK ABI
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct CameraSize {
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CameraProfile {
    format: i32,
    size: CameraSize,
}

#[repr(C)]
struct CameraDevice {
    camera_id: *mut c_char,
    position: i32,
    camera_type: i32,
    connection: i32,
}

#[repr(C)]
struct OutputCapability {
    preview_profiles: *mut *mut CameraProfile,
    preview_profiles_size: u32,
    photo_profiles: *mut *mut CameraProfile,
    photo_profiles_size: u32,
    video_profiles: *mut *mut c_void,
    video_profiles_size: u32,
    supported_metadata_object_types: *mut *mut c_void,
    metadata_types_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CameraFrameRateRange {
    min: u32,
    max: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CameraVideoProfile {
    format: i32,
    size: CameraSize,
    range: CameraFrameRateRange,
}

// OH_AVRecorder_Config and its parts (avrecorder_base.h)
#[repr(C)]
#[derive(Clone, Copy)]
struct AvProfile {
    audio_bitrate: i32,
    audio_channels: i32,
    audio_codec: i32,
    audio_sample_rate: i32,
    file_format: i32,
    video_bitrate: i32,
    video_codec: i32,
    video_frame_width: i32,
    video_frame_height: i32,
    video_frame_rate: i32,
    is_hdr: bool,
    enable_temporal_scale: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AvLocation {
    latitude: f32,
    longitude: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AvMetadataTemplate {
    key: *mut c_char,
    value: *mut c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AvMetadata {
    genre: *mut c_char,
    video_orientation: *mut c_char,
    location: AvLocation,
    custom_info: AvMetadataTemplate,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AvConfig {
    audio_source_type: i32,
    video_source_type: i32,
    profile: AvProfile,
    url: *mut c_char,
    file_generation_mode: i32,
    metadata: AvMetadata,
    max_duration: i32,
}

type RecorderStateCb = unsafe extern "C" fn(*mut c_void, i32, i32, *mut c_void);
type RecorderErrorCb = unsafe extern "C" fn(*mut c_void, i32, *const c_char, *mut c_void);

unsafe extern "C" fn on_recorder_state(_recorder: *mut c_void, state: i32, reason: i32, _user: *mut c_void) {
    crate::log!("ohos camera: recorder state {state} (reason {reason})");
}

unsafe extern "C" fn on_recorder_error(_recorder: *mut c_void, code: i32, msg: *const c_char, _user: *mut c_void) {
    let msg = if msg.is_null() { String::new() } else { CStr::from_ptr(msg).to_string_lossy().into_owned() };
    crate::error!("ohos camera: recorder error {code}: {msg}");
}

const AVRECORDER_MIC: i32 = 1;
const AVRECORDER_SURFACE_YUV: i32 = 0;
const AVRECORDER_VIDEO_AVC: i32 = 2;
const AVRECORDER_AUDIO_AAC: i32 = 3;
const AVRECORDER_CFT_MPEG_4: i32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct CameraPhotoCaptureSetting {
    quality: i32,
    rotation: i32,
    location: *mut c_void,
    mirror: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CameraPoint {
    x: f64,
    y: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ImageSize {
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeBufferPlane {
    offset: u64,
    row_stride: u32,
    column_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeBufferPlanes {
    plane_count: u32,
    planes: [NativeBufferPlane; 4],
}

const CAMERA_FORMAT_YUV_420_SP: i32 = 1003;
const CAMERA_FORMAT_JPEG: i32 = 2000;
const CAMERA_POSITION_FRONT: i32 = 2;
/// `OH_NativeBuffer_Format`: the semi-planar 4:2:0 layouts, in enum order.
const NATIVEBUFFER_PIXEL_FMT_YCBCR_420_SP: i32 = 24;
const NATIVEBUFFER_PIXEL_FMT_YCRCB_420_SP: i32 = 25;

type Rc32 = i32;
type ImageReceiverOnCallback = unsafe extern "C" fn(receiver: *mut c_void);

/// The NDK entry points, resolved once.
struct OhCameraApi {
    _libs: Vec<ModuleLoader>,
    // camera
    get_manager: unsafe extern "C" fn(*mut *mut c_void) -> Rc32,
    delete_manager: unsafe extern "C" fn(*mut c_void) -> Rc32,
    get_supported: unsafe extern "C" fn(*mut c_void, *mut *mut CameraDevice, *mut u32) -> Rc32,
    delete_supported: unsafe extern "C" fn(*mut c_void, *mut CameraDevice, u32) -> Rc32,
    get_capability:
        unsafe extern "C" fn(*mut c_void, *const CameraDevice, *mut *mut OutputCapability) -> Rc32,
    delete_capability: unsafe extern "C" fn(*mut c_void, *mut OutputCapability) -> Rc32,
    create_input: unsafe extern "C" fn(*mut c_void, *const CameraDevice, *mut *mut c_void) -> Rc32,
    input_open: unsafe extern "C" fn(*mut c_void) -> Rc32,
    input_close: unsafe extern "C" fn(*mut c_void) -> Rc32,
    input_release: unsafe extern "C" fn(*mut c_void) -> Rc32,
    create_session: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> Rc32,
    begin_config: unsafe extern "C" fn(*mut c_void) -> Rc32,
    add_input: unsafe extern "C" fn(*mut c_void, *mut c_void) -> Rc32,
    create_preview: unsafe extern "C" fn(
        *mut c_void,
        *const CameraProfile,
        *const c_char,
        *mut *mut c_void,
    ) -> Rc32,
    add_preview: unsafe extern "C" fn(*mut c_void, *mut c_void) -> Rc32,
    commit_config: unsafe extern "C" fn(*mut c_void) -> Rc32,
    session_start: unsafe extern "C" fn(*mut c_void) -> Rc32,
    session_stop: unsafe extern "C" fn(*mut c_void) -> Rc32,
    session_release: unsafe extern "C" fn(*mut c_void) -> Rc32,
    preview_release: unsafe extern "C" fn(*mut c_void) -> Rc32,
    preview_rotation: Option<unsafe extern "C" fn(*mut c_void, i32, *mut i32) -> Rc32>,
    // session controls (all optional: older NDKs lack some)
    set_focus_mode: Option<unsafe extern "C" fn(*mut c_void, i32) -> Rc32>,
    set_focus_point: Option<unsafe extern "C" fn(*mut c_void, CameraPoint) -> Rc32>,
    set_metering_point: Option<unsafe extern "C" fn(*mut c_void, CameraPoint) -> Rc32>,
    set_exposure_mode: Option<unsafe extern "C" fn(*mut c_void, i32) -> Rc32>,
    get_zoom_range: Option<unsafe extern "C" fn(*mut c_void, *mut f32, *mut f32) -> Rc32>,
    set_zoom: Option<unsafe extern "C" fn(*mut c_void, f32) -> Rc32>,
    get_exposure_bias_range:
        Option<unsafe extern "C" fn(*mut c_void, *mut f32, *mut f32, *mut f32) -> Rc32>,
    set_exposure_bias: Option<unsafe extern "C" fn(*mut c_void, f32) -> Rc32>,
    set_flash_mode: Option<unsafe extern "C" fn(*mut c_void, i32) -> Rc32>,
    // photo output
    create_photo_output: Option<
        unsafe extern "C" fn(*mut c_void, *const CameraProfile, *const c_char, *mut *mut c_void) -> Rc32,
    >,
    add_photo_output: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> Rc32>,
    photo_capture: Option<unsafe extern "C" fn(*mut c_void, CameraPhotoCaptureSetting) -> Rc32>,
    photo_rotation: Option<unsafe extern "C" fn(*mut c_void, i32, *mut i32) -> Rc32>,
    photo_release: Option<unsafe extern "C" fn(*mut c_void) -> Rc32>,
    image_buffer_size: Option<unsafe extern "C" fn(*mut c_void, u32, *mut usize) -> Rc32>,
    // video output + recorder (libavrecorder.so / libnative_window.so, API 18)
    create_video_output: Option<
        unsafe extern "C" fn(*mut c_void, *const CameraVideoProfile, *const c_char, *mut *mut c_void) -> Rc32,
    >,
    add_video_output: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> Rc32>,
    remove_video_output: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> Rc32>,
    video_output_start: Option<unsafe extern "C" fn(*mut c_void) -> Rc32>,
    video_output_stop: Option<unsafe extern "C" fn(*mut c_void) -> Rc32>,
    video_output_release: Option<unsafe extern "C" fn(*mut c_void) -> Rc32>,
    recorder_create: Option<unsafe extern "C" fn() -> *mut c_void>,
    recorder_prepare: Option<unsafe extern "C" fn(*mut c_void, *mut AvConfig) -> Rc32>,
    recorder_surface: Option<unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> Rc32>,
    recorder_rotation: Option<unsafe extern "C" fn(*mut c_void, i32) -> Rc32>,
    recorder_start: Option<unsafe extern "C" fn(*mut c_void) -> Rc32>,
    recorder_pause: Option<unsafe extern "C" fn(*mut c_void) -> Rc32>,
    recorder_resume: Option<unsafe extern "C" fn(*mut c_void) -> Rc32>,
    recorder_stop: Option<unsafe extern "C" fn(*mut c_void) -> Rc32>,
    recorder_release: Option<unsafe extern "C" fn(*mut c_void) -> Rc32>,
    recorder_set_state_cb: Option<unsafe extern "C" fn(*mut c_void, RecorderStateCb, *mut c_void) -> Rc32>,
    recorder_set_error_cb: Option<unsafe extern "C" fn(*mut c_void, RecorderErrorCb, *mut c_void) -> Rc32>,
    window_surface_id: Option<unsafe extern "C" fn(*mut c_void, *mut u64) -> Rc32>,
    // image receiver
    opts_create: unsafe extern "C" fn(*mut *mut c_void) -> Rc32,
    opts_set_size: unsafe extern "C" fn(*mut c_void, ImageSize) -> Rc32,
    opts_set_capacity: unsafe extern "C" fn(*mut c_void, i32) -> Rc32,
    opts_release: unsafe extern "C" fn(*mut c_void) -> Rc32,
    receiver_create: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> Rc32,
    receiver_surface_id: unsafe extern "C" fn(*mut c_void, *mut u64) -> Rc32,
    receiver_read_latest: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> Rc32,
    receiver_on: unsafe extern "C" fn(*mut c_void, ImageReceiverOnCallback) -> Rc32,
    receiver_off: unsafe extern "C" fn(*mut c_void) -> Rc32,
    receiver_release: unsafe extern "C" fn(*mut c_void) -> Rc32,
    // image
    image_size: unsafe extern "C" fn(*mut c_void, *mut ImageSize) -> Rc32,
    image_component_types: unsafe extern "C" fn(*mut c_void, *mut *mut u32, *mut usize) -> Rc32,
    image_byte_buffer: unsafe extern "C" fn(*mut c_void, u32, *mut *mut c_void) -> Rc32,
    image_row_stride: unsafe extern "C" fn(*mut c_void, u32, *mut i32) -> Rc32,
    image_format: Option<unsafe extern "C" fn(*mut c_void, *mut i32) -> Rc32>,
    image_release: unsafe extern "C" fn(*mut c_void) -> Rc32,
    // native buffer
    buffer_map: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> Rc32,
    buffer_map_planes:
        unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut NativeBufferPlanes) -> Rc32,
    buffer_unmap: unsafe extern "C" fn(*mut c_void) -> Rc32,
}

unsafe impl Send for OhCameraApi {}
unsafe impl Sync for OhCameraApi {}

static API: std::sync::OnceLock<Option<OhCameraApi>> = std::sync::OnceLock::new();

fn api() -> Option<&'static OhCameraApi> {
    API.get_or_init(|| {
        let load = |name: &str| ModuleLoader::load(name).ok();
        let camera = load("libohcamera.so")?;
        let receiver = load("libimage_receiver.so")?;
        let image = load("libohimage.so")?;
        let buffer = load("libnative_buffer.so")?;
        let recorder = load("libavrecorder.so");
        let window = load("libnative_window.so");
        macro_rules! rsym {
            ($name:literal) => {
                recorder.as_ref().and_then(|l| l.get_symbol($name).ok())
            };
        }
        macro_rules! sym {
            ($lib:expr, $name:literal) => {
                match $lib.get_symbol($name) {
                    Ok(f) => f,
                    Err(_) => {
                        crate::error!("ohos camera: missing NDK symbol {}", $name);
                        return None;
                    }
                }
            };
        }
        let api = OhCameraApi {
            get_manager: sym!(camera, "OH_Camera_GetCameraManager"),
            delete_manager: sym!(camera, "OH_Camera_DeleteCameraManager"),
            get_supported: sym!(camera, "OH_CameraManager_GetSupportedCameras"),
            delete_supported: sym!(camera, "OH_CameraManager_DeleteSupportedCameras"),
            get_capability: sym!(camera, "OH_CameraManager_GetSupportedCameraOutputCapability"),
            delete_capability: sym!(
                camera,
                "OH_CameraManager_DeleteSupportedCameraOutputCapability"
            ),
            create_input: sym!(camera, "OH_CameraManager_CreateCameraInput"),
            input_open: sym!(camera, "OH_CameraInput_Open"),
            input_close: sym!(camera, "OH_CameraInput_Close"),
            input_release: sym!(camera, "OH_CameraInput_Release"),
            create_session: sym!(camera, "OH_CameraManager_CreateCaptureSession"),
            begin_config: sym!(camera, "OH_CaptureSession_BeginConfig"),
            add_input: sym!(camera, "OH_CaptureSession_AddInput"),
            create_preview: sym!(camera, "OH_CameraManager_CreatePreviewOutput"),
            add_preview: sym!(camera, "OH_CaptureSession_AddPreviewOutput"),
            commit_config: sym!(camera, "OH_CaptureSession_CommitConfig"),
            session_start: sym!(camera, "OH_CaptureSession_Start"),
            session_stop: sym!(camera, "OH_CaptureSession_Stop"),
            session_release: sym!(camera, "OH_CaptureSession_Release"),
            preview_release: sym!(camera, "OH_PreviewOutput_Release"),
            preview_rotation: camera.get_symbol("OH_PreviewOutput_GetPreviewRotation").ok(),
            set_focus_mode: camera.get_symbol("OH_CaptureSession_SetFocusMode").ok(),
            set_focus_point: camera.get_symbol("OH_CaptureSession_SetFocusPoint").ok(),
            set_metering_point: camera.get_symbol("OH_CaptureSession_SetMeteringPoint").ok(),
            set_exposure_mode: camera.get_symbol("OH_CaptureSession_SetExposureMode").ok(),
            get_zoom_range: camera.get_symbol("OH_CaptureSession_GetZoomRatioRange").ok(),
            set_zoom: camera.get_symbol("OH_CaptureSession_SetZoomRatio").ok(),
            get_exposure_bias_range: camera.get_symbol("OH_CaptureSession_GetExposureBiasRange").ok(),
            set_exposure_bias: camera.get_symbol("OH_CaptureSession_SetExposureBias").ok(),
            set_flash_mode: camera.get_symbol("OH_CaptureSession_SetFlashMode").ok(),
            create_photo_output: camera.get_symbol("OH_CameraManager_CreatePhotoOutput").ok(),
            add_photo_output: camera.get_symbol("OH_CaptureSession_AddPhotoOutput").ok(),
            photo_capture: camera.get_symbol("OH_PhotoOutput_Capture_WithCaptureSetting").ok(),
            photo_rotation: camera.get_symbol("OH_PhotoOutput_GetPhotoRotation").ok(),
            photo_release: camera.get_symbol("OH_PhotoOutput_Release").ok(),
            image_buffer_size: image.get_symbol("OH_ImageNative_GetBufferSize").ok(),
            create_video_output: camera.get_symbol("OH_CameraManager_CreateVideoOutput").ok(),
            add_video_output: camera.get_symbol("OH_CaptureSession_AddVideoOutput").ok(),
            remove_video_output: camera.get_symbol("OH_CaptureSession_RemoveVideoOutput").ok(),
            video_output_start: camera.get_symbol("OH_VideoOutput_Start").ok(),
            video_output_stop: camera.get_symbol("OH_VideoOutput_Stop").ok(),
            video_output_release: camera.get_symbol("OH_VideoOutput_Release").ok(),
            recorder_create: rsym!("OH_AVRecorder_Create"),
            recorder_prepare: rsym!("OH_AVRecorder_Prepare"),
            recorder_surface: rsym!("OH_AVRecorder_GetInputSurface"),
            recorder_rotation: rsym!("OH_AVRecorder_UpdateRotation"),
            recorder_start: rsym!("OH_AVRecorder_Start"),
            recorder_pause: rsym!("OH_AVRecorder_Pause"),
            recorder_resume: rsym!("OH_AVRecorder_Resume"),
            recorder_stop: rsym!("OH_AVRecorder_Stop"),
            recorder_release: rsym!("OH_AVRecorder_Release"),
            recorder_set_state_cb: rsym!("OH_AVRecorder_SetStateCallback"),
            recorder_set_error_cb: rsym!("OH_AVRecorder_SetErrorCallback"),
            window_surface_id: window.as_ref().and_then(|l| l.get_symbol("OH_NativeWindow_GetSurfaceId").ok()),
            opts_create: sym!(receiver, "OH_ImageReceiverOptions_Create"),
            opts_set_size: sym!(receiver, "OH_ImageReceiverOptions_SetSize"),
            opts_set_capacity: sym!(receiver, "OH_ImageReceiverOptions_SetCapacity"),
            opts_release: sym!(receiver, "OH_ImageReceiverOptions_Release"),
            receiver_create: sym!(receiver, "OH_ImageReceiverNative_Create"),
            receiver_surface_id: sym!(receiver, "OH_ImageReceiverNative_GetReceivingSurfaceId"),
            receiver_read_latest: sym!(receiver, "OH_ImageReceiverNative_ReadLatestImage"),
            receiver_on: sym!(receiver, "OH_ImageReceiverNative_On"),
            receiver_off: sym!(receiver, "OH_ImageReceiverNative_Off"),
            receiver_release: sym!(receiver, "OH_ImageReceiverNative_Release"),
            image_size: sym!(image, "OH_ImageNative_GetImageSize"),
            image_component_types: sym!(image, "OH_ImageNative_GetComponentTypes"),
            image_byte_buffer: sym!(image, "OH_ImageNative_GetByteBuffer"),
            image_row_stride: sym!(image, "OH_ImageNative_GetRowStride"),
            image_format: image.get_symbol("OH_ImageNative_GetFormat").ok(),
            image_release: sym!(image, "OH_ImageNative_Release"),
            buffer_map: sym!(buffer, "OH_NativeBuffer_Map"),
            buffer_map_planes: sym!(buffer, "OH_NativeBuffer_MapPlanes"),
            buffer_unmap: sym!(buffer, "OH_NativeBuffer_Unmap"),
            _libs: {
                let mut libs = vec![camera, receiver, image, buffer];
                libs.extend(recorder);
                libs.extend(window);
                libs
            },
        };
        Some(api)
    })
    .as_ref()
}

fn cam_error(rc: i32) -> String {
    match rc {
        7_400_101 => "invalid argument".into(),
        7_400_102 => "operation not allowed in this session state".into(),
        7_400_103 => "session not configured".into(),
        // The camera service reports a missing CAMERA grant with this code too.
        7_400_201 => "camera unavailable (in use by another app, or ohos.permission.CAMERA not granted)".into(),
        7_400_202 => "camera disabled by device policy".into(),
        7_400_203 => "camera already in use here".into(),
        201 => "permission denied (ohos.permission.CAMERA)".into(),
        _ => format!("camera error {rc}"),
    }
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

/// One preview profile the camera offers, kept next to the Makepad format it
/// was published as.
#[derive(Clone)]
struct OhFormat {
    format: VideoFormat,
    profile: CameraProfile,
}

struct OhDevice {
    input_id: VideoInputId,
    name: String,
    index: isize,
    front: bool,
    formats: Vec<OhFormat>,
    /// JPEG still profiles the camera offers.
    photo_profiles: Vec<CameraProfile>,
    /// YUV video profiles (size + frame-rate range) for the recorder.
    video_profiles: Vec<CameraVideoProfile>,
}

/// A recording in progress on the session.
struct VideoRec {
    recorder: *mut c_void,
    output: *mut c_void,
    path: String,
    _file: std::fs::File,
    _url: CString,
    _strings: Vec<CString>,
}

/// Capture results waiting for the UI thread (`take_capture_results`).
static CAPTURE_RESULTS: Mutex<Vec<(VideoInputId, CameraCaptureResult)>> = Mutex::new(Vec::new());

pub fn take_capture_results() -> Vec<(VideoInputId, CameraCaptureResult)> {
    CAPTURE_RESULTS.lock().map(|mut r| std::mem::take(&mut *r)).unwrap_or_default()
}

fn push_capture_result(input_id: VideoInputId, result: CameraCaptureResult) {
    if let Ok(mut r) = CAPTURE_RESULTS.lock() {
        r.push((input_id, result));
    }
    SignalToUI::set_ui_signal();
}

/// State shared with the photo receiver's callback thread.
pub struct PhotoShared {
    receiver: AtomicU64,
    input_id: VideoInputId,
    /// Paths of the captures in flight, oldest first.
    pending: Mutex<VecDeque<(String, bool)>>,
}

static ACTIVE_PHOTO: Mutex<Option<Arc<PhotoShared>>> = Mutex::new(None);

/// A frame ready for the GL upload: tightly packed I420.
#[derive(Default)]
pub struct I420Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// State shared between the receiver callback thread and the UI thread.
pub struct FrameShared {
    receiver: AtomicU64,
    latest: Mutex<Option<I420Frame>>,
    spare: Mutex<Option<I420Frame>>,
    pub frames: AtomicU64,
    logged: AtomicBool,
    camera_frame_cb: Arc<Mutex<Option<CameraFrameInputFn>>>,
    video_frame_cb: Arc<Mutex<Option<VideoInputFn>>>,
    format: VideoFormat,
}

impl FrameShared {
    /// The newest converted frame, leaving nothing behind (latest wins).
    pub fn take_latest(&self) -> Option<I420Frame> {
        self.latest.lock().ok()?.take()
    }
    /// Return a frame's buffers for reuse.
    pub fn recycle(&self, frame: I420Frame) {
        if let Ok(mut spare) = self.spare.lock() {
            if spare.is_none() {
                *spare = Some(frame);
            }
        }
    }
}

/// The running preview session; handles are only used on the UI thread.
struct Session {
    input: *mut c_void,
    session: *mut c_void,
    output: *mut c_void,
    receiver: *mut c_void,
    shared: Arc<FrameShared>,
    rotation_steps: f32,
    front: bool,
    /// Still capture: the JPEG output and its receiver (null when the NDK lacks it).
    photo_output: *mut c_void,
    photo_receiver: *mut c_void,
    photo_shared: Option<Arc<PhotoShared>>,
    /// The preview size, which the recording matches.
    preview_size: CameraSize,
    input_id: VideoInputId,
    video: Option<VideoRec>,
}

/// The receiver callback gets only the receiver pointer, so the active
/// session's shared state is looked up here.
static ACTIVE: Mutex<Option<Arc<FrameShared>>> = Mutex::new(None);

pub struct OhCameraAccess {
    pub video_input_cb: [Arc<Mutex<Option<VideoInputFn>>>; MAX_VIDEO_DEVICE_INDEX],
    pub camera_frame_input_cb: [Arc<Mutex<Option<CameraFrameInputFn>>>; MAX_VIDEO_DEVICE_INDEX],
    manager: *mut c_void,
    device_list: *mut CameraDevice,
    device_count: u32,
    devices: Vec<OhDevice>,
    session: Option<Session>,
    active_inputs: Vec<(VideoInputId, VideoFormatId)>,
}

unsafe impl Send for OhCameraAccess {}

impl OhCameraAccess {
    pub fn new() -> Self {
        let mut access = Self {
            video_input_cb: std::array::from_fn(|_| Arc::new(Mutex::new(None))),
            camera_frame_input_cb: std::array::from_fn(|_| Arc::new(Mutex::new(None))),
            manager: std::ptr::null_mut(),
            device_list: std::ptr::null_mut(),
            device_count: 0,
            devices: Vec::new(),
            session: None,
            active_inputs: Vec::new(),
        };
        access.enumerate();
        access
    }

    fn enumerate(&mut self) {
        let Some(api) = api() else {
            crate::log!("ohos camera: NDK not available, no video inputs");
            return;
        };
        let mut manager = std::ptr::null_mut();
        let rc = unsafe { (api.get_manager)(&mut manager) };
        if rc != 0 || manager.is_null() {
            crate::error!("ohos camera: manager: {}", cam_error(rc));
            return;
        }
        let mut list: *mut CameraDevice = std::ptr::null_mut();
        let mut count = 0u32;
        let rc = unsafe { (api.get_supported)(manager, &mut list, &mut count) };
        if rc != 0 || list.is_null() {
            crate::error!("ohos camera: enumerate: {}", cam_error(rc));
            unsafe { (api.delete_manager)(manager) };
            return;
        }
        self.manager = manager;
        self.device_list = list;
        self.device_count = count;
        for i in 0..count as isize {
            let device = unsafe { &*list.offset(i) };
            let id = if device.camera_id.is_null() {
                format!("camera{i}")
            } else {
                unsafe { CStr::from_ptr(device.camera_id) }.to_string_lossy().into_owned()
            };
            let front = device.position == CAMERA_POSITION_FRONT;
            let name = format!("{} camera ({id})", if front { "Front" } else { "Back" });
            let mut cap: *mut OutputCapability = std::ptr::null_mut();
            let rc = unsafe { (api.get_capability)(manager, device, &mut cap) };
            let mut formats = Vec::new();
            let mut photo_profiles = Vec::new();
            let mut video_profiles = Vec::new();
            if rc == 0 && !cap.is_null() {
                let cap_ref = unsafe { &*cap };
                for p in 0..cap_ref.video_profiles_size as isize {
                    let profile = unsafe { *cap_ref.video_profiles.offset(p) } as *const CameraVideoProfile;
                    if !profile.is_null() && unsafe { (*profile).format } == CAMERA_FORMAT_YUV_420_SP {
                        video_profiles.push(unsafe { *profile });
                    }
                }
                for p in 0..cap_ref.photo_profiles_size as isize {
                    let profile = unsafe { *cap_ref.photo_profiles.offset(p) };
                    if !profile.is_null() && unsafe { (*profile).format } == CAMERA_FORMAT_JPEG {
                        photo_profiles.push(unsafe { *profile });
                    }
                }
                for p in 0..cap_ref.preview_profiles_size as isize {
                    let profile = unsafe { *cap_ref.preview_profiles.offset(p) };
                    if profile.is_null() {
                        continue;
                    }
                    let profile = unsafe { *profile };
                    if profile.format != CAMERA_FORMAT_YUV_420_SP {
                        continue;
                    }
                    let (w, h) = (profile.size.width as usize, profile.size.height as usize);
                    if formats
                        .iter()
                        .any(|f: &OhFormat| f.format.width == w && f.format.height == h)
                    {
                        continue;
                    }
                    let format_id =
                        VideoFormatId(LiveId::from_str(&format!("{id}:{w}x{h}:nv12")));
                    formats.push(OhFormat {
                        format: VideoFormat {
                            format_id,
                            width: w,
                            height: h,
                            frame_rate: Some(30.0),
                            pixel_format: VideoPixelFormat::NV12,
                        },
                        profile,
                    });
                }
                unsafe { (api.delete_capability)(manager, cap) };
            } else {
                crate::error!("ohos camera: {name}: capability: {}", cam_error(rc));
            }
            crate::log!("ohos camera: {name}: {} preview sizes, {} photo sizes, {} video profiles", formats.len(), photo_profiles.len(), video_profiles.len());
            self.devices.push(OhDevice {
                input_id: VideoInputId(LiveId::from_str(&id)),
                name,
                index: i,
                front,
                formats,
                photo_profiles,
                video_profiles,
            });
        }
    }

    pub fn get_updated_descs(&self) -> Vec<VideoInputDesc> {
        self.devices
            .iter()
            .map(|d| VideoInputDesc {
                input_id: d.input_id,
                name: d.name.clone(),
                formats: d.formats.iter().map(|f| f.format).collect(),
            })
            .collect()
    }

    pub fn format_size(&self, input_id: VideoInputId, format_id: VideoFormatId) -> Option<(u32, u32)> {
        let device = self.devices.iter().find(|d| d.input_id == input_id)?;
        let format = device.formats.iter().find(|f| f.format.format_id == format_id)?;
        Some((format.format.width as u32, format.format.height as u32))
    }

    pub fn is_front(&self, input_id: VideoInputId) -> bool {
        self.devices.iter().any(|d| d.input_id == input_id && d.front)
    }

    /// The shared frame slot of the running session, for a player to poll.
    pub fn frame_shared(&self) -> Option<Arc<FrameShared>> {
        self.session.as_ref().map(|s| s.shared.clone())
    }

    pub fn rotation_steps(&self) -> f32 {
        self.session.as_ref().map(|s| s.rotation_steps).unwrap_or(0.0)
    }

    /// Apply a runtime control to the running session.
    pub fn control(&mut self, input_id: VideoInputId, control: CameraControl) {
        let Some(api) = api() else { return };
        let Some(session) = self.session.as_ref() else {
            crate::log!("ohos camera: control {control:?} with no running session");
            return;
        };
        if !self.active_inputs.iter().any(|(id, _)| *id == input_id) {
            return;
        }
        let handle = session.session;
        let steps = session.rotation_steps as i32;
        let check = |what: &str, rc: Rc32| {
            if rc != 0 {
                crate::error!("ohos camera: {what}: {}", cam_error(rc));
            }
        };
        unsafe {
            match control {
                CameraControl::FocusPoint { x, y } => {
                    // The point is given on the rotated (displayed) preview;
                    // the camera wants it on the sensor image.
                    let (x, y) = (x.clamp(0.0, 1.0), y.clamp(0.0, 1.0));
                    let point = match steps.rem_euclid(4) {
                        1 => CameraPoint { x: y, y: 1.0 - x },
                        2 => CameraPoint { x: 1.0 - x, y: 1.0 - y },
                        3 => CameraPoint { x: 1.0 - y, y: x },
                        _ => CameraPoint { x, y },
                    };
                    if let Some(f) = api.set_focus_mode {
                        check("focus mode", f(handle, 2)); // FOCUS_MODE_AUTO
                    }
                    if let Some(f) = api.set_focus_point {
                        check("focus point", f(handle, point));
                    }
                    if let Some(f) = api.set_exposure_mode {
                        check("exposure mode", f(handle, 1)); // EXPOSURE_MODE_AUTO
                    }
                    if let Some(f) = api.set_metering_point {
                        check("metering point", f(handle, point));
                    }
                }
                CameraControl::ContinuousFocus => {
                    if let Some(f) = api.set_focus_mode {
                        check("focus mode", f(handle, 1)); // FOCUS_MODE_CONTINUOUS_AUTO
                    }
                    if let Some(f) = api.set_exposure_mode {
                        check("exposure mode", f(handle, 2)); // EXPOSURE_MODE_CONTINUOUS_AUTO
                    }
                }
                CameraControl::ZoomRatio(ratio) => {
                    let (mut lo, mut hi) = (1.0f32, 1.0f32);
                    if let Some(f) = api.get_zoom_range {
                        if f(handle, &mut lo, &mut hi) != 0 || hi < lo {
                            lo = 1.0;
                            hi = ratio.max(1.0);
                        }
                    }
                    if let Some(f) = api.set_zoom {
                        check("zoom", f(handle, ratio.clamp(lo, hi)));
                    }
                }
                CameraControl::ExposureBias(ev) => {
                    let (mut lo, mut hi, mut step) = (-4.0f32, 4.0f32, 0.0f32);
                    if let Some(f) = api.get_exposure_bias_range {
                        if f(handle, &mut lo, &mut hi, &mut step) != 0 || hi < lo {
                            lo = -4.0;
                            hi = 4.0;
                        }
                    }
                    if let Some(f) = api.set_exposure_bias {
                        check("exposure bias", f(handle, ev.clamp(lo, hi)));
                    }
                }
                CameraControl::Flash(mode) => {
                    let mode = match mode {
                        CameraFlashMode::Off => 0,
                        CameraFlashMode::On => 1,
                        CameraFlashMode::Auto => 2,
                        CameraFlashMode::Torch => 3,
                    };
                    if let Some(f) = api.set_flash_mode {
                        check("flash mode", f(handle, mode));
                    }
                }
            }
        }
    }

    /// Start (or stop, with an empty list) the preview stream. One stream at a
    /// time: the phone's camera service does not share a device between
    /// sessions anyway.
    pub fn use_video_input(&mut self, inputs: &[(VideoInputId, VideoFormatId)]) {
        if self.active_inputs == inputs && (inputs.is_empty() || self.session.is_some()) {
            return;
        }
        self.stop_session();
        self.active_inputs = inputs.to_vec();
        let Some(&(input_id, format_id)) = inputs.first() else {
            return;
        };
        match self.start_session(input_id, format_id) {
            Ok(session) => {
                if let Ok(mut active) = ACTIVE.lock() {
                    *active = Some(session.shared.clone());
                }
                self.session = Some(session);
            }
            Err(error) => crate::error!("ohos camera: {error}"),
        }
    }

    fn start_session(
        &mut self,
        input_id: VideoInputId,
        format_id: VideoFormatId,
    ) -> Result<Session, String> {
        let api = api().ok_or("camera NDK not available")?;
        if self.manager.is_null() {
            return Err("no camera manager".into());
        }
        let device = self
            .devices
            .iter()
            .find(|d| d.input_id == input_id)
            .ok_or("unknown video input")?;
        let format = device
            .formats
            .iter()
            .find(|f| f.format.format_id == format_id)
            .or(device.formats.first())
            .ok_or("camera offers no YUV preview profile")?
            .clone();
        let device_ptr = unsafe { self.device_list.offset(device.index) };
        let front = device.front;
        let name = device.name.clone();
        // The still profile with the preview's aspect, largest first.
        let aspect = format.profile.size.width as f64 / format.profile.size.height as f64;
        let photo_profile = device
            .photo_profiles
            .iter()
            .filter(|p| p.size.height > 0 && ((p.size.width as f64 / p.size.height as f64) - aspect).abs() < 0.03)
            .max_by_key(|p| p.size.width as u64 * p.size.height as u64)
            .copied();

        // Image receiver: the preview surface the camera streams into.
        let mut opts = std::ptr::null_mut();
        let rc = unsafe { (api.opts_create)(&mut opts) };
        if rc != 0 || opts.is_null() {
            return Err(format!("image receiver options: {rc}"));
        }
        unsafe {
            (api.opts_set_size)(
                opts,
                ImageSize {
                    width: format.profile.size.width,
                    height: format.profile.size.height,
                },
            );
            (api.opts_set_capacity)(opts, 4);
        }
        let mut receiver = std::ptr::null_mut();
        let rc = unsafe { (api.receiver_create)(opts, &mut receiver) };
        unsafe { (api.opts_release)(opts) };
        if rc != 0 || receiver.is_null() {
            return Err(format!("image receiver: {rc}"));
        }
        let mut surface_id = 0u64;
        let rc = unsafe { (api.receiver_surface_id)(receiver, &mut surface_id) };
        if rc != 0 {
            unsafe { (api.receiver_release)(receiver) };
            return Err(format!("receiver surface id: {rc}"));
        }

        let shared = Arc::new(FrameShared {
            receiver: AtomicU64::new(receiver as usize as u64),
            latest: Mutex::new(None),
            spare: Mutex::new(None),
            frames: AtomicU64::new(0),
            logged: AtomicBool::new(false),
            camera_frame_cb: self.camera_frame_input_cb[0].clone(),
            video_frame_cb: self.video_input_cb[0].clone(),
            format: format.format,
        });

        // The still output: a JPEG receiver of the photo size, added to the session before commit.
        let mut photo_output: *mut c_void = std::ptr::null_mut();
        let mut photo_receiver: *mut c_void = std::ptr::null_mut();
        let mut photo_surface = 0u64;
        if let (Some(profile), Some(_), Some(_)) = (photo_profile, api.create_photo_output, api.add_photo_output) {
            let mut opts = std::ptr::null_mut();
            if unsafe { (api.opts_create)(&mut opts) } == 0 && !opts.is_null() {
                unsafe {
                    (api.opts_set_size)(opts, ImageSize { width: profile.size.width, height: profile.size.height });
                    (api.opts_set_capacity)(opts, 2);
                }
                let rc = unsafe { (api.receiver_create)(opts, &mut photo_receiver) };
                unsafe { (api.opts_release)(opts) };
                if rc != 0 || photo_receiver.is_null() || unsafe { (api.receiver_surface_id)(photo_receiver, &mut photo_surface) } != 0 {
                    crate::error!("ohos camera: photo receiver: {rc}");
                    photo_receiver = std::ptr::null_mut();
                }
            }
        }

        // Camera pipeline: input → session → preview output on the receiver surface.
        let built = (|| -> Result<(*mut c_void, *mut c_void, *mut c_void), String> {
            let mut input = std::ptr::null_mut();
            let rc = unsafe { (api.create_input)(self.manager, device_ptr, &mut input) };
            if rc != 0 || input.is_null() {
                return Err(format!("camera input: {}", cam_error(rc)));
            }
            let rc = unsafe { (api.input_open)(input) };
            if rc != 0 {
                unsafe { (api.input_release)(input) };
                return Err(format!("open camera: {}", cam_error(rc)));
            }
            let sid = CString::new(surface_id.to_string()).unwrap();
            let mut output = std::ptr::null_mut();
            let rc = unsafe {
                (api.create_preview)(self.manager, &format.profile, sid.as_ptr(), &mut output)
            };
            if rc != 0 || output.is_null() {
                unsafe {
                    (api.input_close)(input);
                    (api.input_release)(input);
                }
                return Err(format!("preview output: {}", cam_error(rc)));
            }
            let mut session = std::ptr::null_mut();
            let rc = unsafe { (api.create_session)(self.manager, &mut session) };
            if rc != 0 || session.is_null() {
                unsafe {
                    (api.preview_release)(output);
                    (api.input_close)(input);
                    (api.input_release)(input);
                }
                return Err(format!("session: {}", cam_error(rc)));
            }
            if !photo_receiver.is_null() {
                if let (Some(profile), Some(create), Some(_)) = (photo_profile, api.create_photo_output, api.add_photo_output) {
                    let sid = CString::new(photo_surface.to_string()).unwrap();
                    let rc = unsafe { create(self.manager, &profile, sid.as_ptr(), &mut photo_output) };
                    if rc != 0 || photo_output.is_null() {
                        crate::error!("ohos camera: photo output: {}", cam_error(rc));
                        photo_output = std::ptr::null_mut();
                    }
                }
            }
            let mut steps: Vec<(&str, i32)> = vec![
                ("beginConfig", unsafe { (api.begin_config)(session) }),
                ("addInput", unsafe { (api.add_input)(session, input) }),
                ("addPreviewOutput", unsafe { (api.add_preview)(session, output) }),
            ];
            if !photo_output.is_null() {
                let rc = unsafe { (api.add_photo_output.unwrap())(session, photo_output) };
                if rc != 0 {
                    crate::error!("ohos camera: addPhotoOutput: {} (stills disabled)", cam_error(rc));
                    unsafe { (api.photo_release.unwrap_or(noop_release))(photo_output) };
                    photo_output = std::ptr::null_mut();
                }
            }
            steps.push(("commitConfig", unsafe { (api.commit_config)(session) }));
            steps.push(("start", unsafe { (api.session_start)(session) }));
            for (what, rc) in steps {
                if rc != 0 {
                    unsafe {
                        (api.session_release)(session);
                        (api.preview_release)(output);
                        if !photo_output.is_null() { (api.photo_release.unwrap_or(noop_release))(photo_output); }
                        (api.input_close)(input);
                        (api.input_release)(input);
                    }
                    return Err(format!("{what}: {}", cam_error(rc)));
                }
            }
            Ok((input, session, output))
        })();
        let (input, session, output) = match built {
            Ok(v) => v,
            Err(error) => {
                unsafe {
                    (api.receiver_release)(receiver);
                    if !photo_receiver.is_null() { (api.receiver_release)(photo_receiver); }
                }
                return Err(error);
            }
        };
        let photo_shared = if photo_output.is_null() {
            if !photo_receiver.is_null() { unsafe { (api.receiver_release)(photo_receiver) }; photo_receiver = std::ptr::null_mut(); }
            None
        } else {
            let shared = Arc::new(PhotoShared { receiver: AtomicU64::new(photo_receiver as usize as u64), input_id, pending: Mutex::new(VecDeque::new()) });
            if let Ok(mut active) = ACTIVE_PHOTO.lock() { *active = Some(shared.clone()); }
            let rc = unsafe { (api.receiver_on)(photo_receiver, on_photo_arrived) };
            if rc != 0 { crate::error!("ohos camera: photo receiver callback: {rc}"); }
            if let Some(p) = photo_profile { crate::log!("ohos camera: stills at {}x{}", p.size.width, p.size.height); }
            Some(shared)
        };

        // The camera reports how far the sensor image must turn to be upright
        // (90° for the Mate's back camera); the Video shader counts quarter
        // turns the other way, verified against the phone's own camera app.
        let mut rotation_steps = if front { 1.0 } else { 3.0 };
        if let Some(get_rotation) = api.preview_rotation {
            let mut rotation = 0i32;
            if unsafe { get_rotation(output, 0, &mut rotation) } == 0 {
                rotation_steps = ((4 - rotation.rem_euclid(360) / 90) % 4) as f32;
            }
        }
        let rc = unsafe { (api.receiver_on)(receiver, on_image_arrived) };
        if rc != 0 {
            crate::error!("ohos camera: receiver callback: {rc}");
        }
        crate::log!(
            "ohos camera: streaming {name} at {}x{} (rotation {} quarter turns)",
            format.format.width,
            format.format.height,
            rotation_steps
        );
        Ok(Session {
            input,
            session,
            output,
            receiver,
            shared,
            rotation_steps,
            front,
            photo_output,
            photo_receiver,
            photo_shared,
            preview_size: format.profile.size,
            input_id,
            video: None,
        })
    }

    /// Start recording the session into `path` through an AV recorder surface.
    fn start_video(&mut self, path: String, audio: bool) -> Result<(), String> {
        let api = api().ok_or("camera NDK not available")?;
        let session = self.session.as_mut().ok_or("no camera session is running")?;
        if session.video.is_some() {
            return Err("already recording".into());
        }
        let (Some(create), Some(prepare), Some(get_surface), Some(start), Some(surface_id), Some(create_output), Some(add_output), Some(out_start)) = (
            api.recorder_create, api.recorder_prepare, api.recorder_surface, api.recorder_start, api.window_surface_id,
            api.create_video_output, api.add_video_output, api.video_output_start,
        ) else {
            return Err("the AV recorder NDK is not available on this device".into());
        };
        let device = self.devices.iter().find(|d| d.input_id == session.input_id).ok_or("unknown input")?;
        let size = session.preview_size;
        let profile = device
            .video_profiles
            .iter()
            .filter(|p| p.size.width == size.width && p.size.height == size.height && p.range.min <= 30 && p.range.max >= 30)
            .max_by_key(|p| p.range.max)
            .or_else(|| device.video_profiles.iter().filter(|p| p.size.width == size.width && p.size.height == size.height).next())
            .copied()
            .ok_or_else(|| format!("no video profile at {}x{}", size.width, size.height))?;
        let fps = 30i32.clamp(profile.range.min as i32, profile.range.max as i32);

        std::path::Path::new(&path).parent().map(std::fs::create_dir_all);
        // The MP4 muxer seeks back to write the header: the descriptor must be read-write.
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path).map_err(|e| format!("create {path}: {e}"))?;
        let url = CString::new(format!("fd://{}", file.as_raw_fd())).unwrap();
        // The recorder reads every metadata string, so none may be null.
        let strings: Vec<CString> = ["90", "", "", ""].iter().map(|t| CString::new(*t).unwrap()).collect();
        let (orientation, genre, key, value) = (&strings[0], &strings[1], &strings[2], &strings[3]);
        let recorder = unsafe { create() };
        if recorder.is_null() {
            return Err("recorder create".into());
        }
        let mut config = AvConfig {
            audio_source_type: if audio { AVRECORDER_MIC } else { -1 },
            video_source_type: AVRECORDER_SURFACE_YUV,
            profile: AvProfile {
                audio_bitrate: if audio { 96_000 } else { 0 },
                audio_channels: if audio { 2 } else { 0 },
                audio_codec: if audio { AVRECORDER_AUDIO_AAC } else { 0 },
                audio_sample_rate: if audio { 48_000 } else { 0 },
                file_format: AVRECORDER_CFT_MPEG_4,
                video_bitrate: 12_000_000,
                video_codec: AVRECORDER_VIDEO_AVC,
                video_frame_width: size.width as i32,
                video_frame_height: size.height as i32,
                video_frame_rate: fps,
                is_hdr: false,
                enable_temporal_scale: false,
            },
            url: url.as_ptr() as *mut c_char,
            file_generation_mode: 0,
            metadata: AvMetadata {
                genre: genre.as_ptr() as *mut c_char,
                video_orientation: orientation.as_ptr() as *mut c_char,
                location: AvLocation { latitude: 0.0, longitude: 0.0 },
                custom_info: AvMetadataTemplate { key: key.as_ptr() as *mut c_char, value: value.as_ptr() as *mut c_char },
            },
            max_duration: 3600,
        };
        let release = api.recorder_release.unwrap_or(noop_release);
        // The NDK insists on both callbacks before Prepare ("callback_ is nullptr").
        if let Some(f) = api.recorder_set_state_cb { let _ = unsafe { f(recorder, on_recorder_state, std::ptr::null_mut()) }; }
        if let Some(f) = api.recorder_set_error_cb { let _ = unsafe { f(recorder, on_recorder_error, std::ptr::null_mut()) }; }
        let rc = unsafe { prepare(recorder, &mut config) };
        if rc != 0 {
            unsafe { release(recorder) };
            return Err(format!("recorder prepare: {rc}"));
        }
        if let Some(rot) = api.recorder_rotation {
            let _ = unsafe { rot(recorder, 90) };
        }
        let mut window = std::ptr::null_mut();
        let rc = unsafe { get_surface(recorder, &mut window) };
        let mut sid = 0u64;
        if rc != 0 || window.is_null() || unsafe { surface_id(window, &mut sid) } != 0 {
            unsafe { release(recorder) };
            return Err(format!("recorder surface: {rc}"));
        }
        let sid = CString::new(sid.to_string()).unwrap();
        let mut output = std::ptr::null_mut();
        let rc = unsafe { create_output(self.manager, &profile, sid.as_ptr(), &mut output) };
        if rc != 0 || output.is_null() {
            unsafe { release(recorder) };
            return Err(format!("video output: {}", cam_error(rc)));
        }
        let out_release = api.video_output_release.unwrap_or(noop_release);
        let steps: [(&str, i32); 4] = [
            ("beginConfig", unsafe { (api.begin_config)(session.session) }),
            ("addVideoOutput", unsafe { add_output(session.session, output) }),
            ("commitConfig", unsafe { (api.commit_config)(session.session) }),
            ("start", unsafe { (api.session_start)(session.session) }),
        ];
        for (what, rc) in steps {
            if rc != 0 {
                unsafe {
                    if let Some(remove) = api.remove_video_output { let _ = (api.begin_config)(session.session); let _ = remove(session.session, output); let _ = (api.commit_config)(session.session); }
                    out_release(output);
                    release(recorder);
                }
                return Err(format!("{what}: {}", cam_error(rc)));
            }
        }
        let rc = unsafe { out_start(output) };
        if rc != 0 {
            unsafe { out_release(output); release(recorder) };
            return Err(format!("video output start: {}", cam_error(rc)));
        }
        let rc = unsafe { start(recorder) };
        if rc != 0 {
            unsafe { let _ = api.video_output_stop.map(|f| f(output)); out_release(output); release(recorder) };
            return Err(format!("recorder start: {rc}"));
        }
        crate::log!("ohos camera: recording {path} at {}x{} {fps} fps{}", size.width, size.height, if audio { " with audio" } else { "" });
        session.video = Some(VideoRec { recorder, output, path, _file: file, _url: url, _strings: strings });
        Ok(())
    }

    /// Stop the recording, take the video output out of the session again.
    fn stop_video(&mut self) -> Result<String, String> {
        let api = api().ok_or("camera NDK not available")?;
        let session = self.session.as_mut().ok_or("no camera session is running")?;
        let rec = session.video.take().ok_or("not recording")?;
        unsafe {
            if let Some(stop) = api.recorder_stop { let _ = stop(rec.recorder); }
            if let Some(stop) = api.video_output_stop { let _ = stop(rec.output); }
            if let Some(remove) = api.remove_video_output {
                let _ = (api.begin_config)(session.session);
                let _ = remove(session.session, rec.output);
                let _ = (api.commit_config)(session.session);
                let _ = (api.session_start)(session.session);
            }
            if let Some(release) = api.video_output_release { let _ = release(rec.output); }
            if let Some(release) = api.recorder_release { let _ = release(rec.recorder); }
        }
        crate::log!("ohos camera: recording stopped: {}", rec.path);
        Ok(rec.path)
    }

    /// Take a photo or drive a recording on the running session.
    pub fn capture(&mut self, input_id: VideoInputId, request: CameraCaptureRequest) {
        let Some(api) = api() else { return };
        let fail = |what: &str, error: String| push_capture_result(input_id, CameraCaptureResult::Failed { what: what.into(), error });
        let Some(session) = self.session.as_ref() else {
            fail("capture", "no camera session is running".into());
            return;
        };
        match request {
            CameraCaptureRequest::Photo { path, library } => {
                let (Some(shared), Some(capture)) = (session.photo_shared.as_ref(), api.photo_capture) else {
                    fail("photo", "this camera session has no still output".into());
                    return;
                };
                if session.photo_output.is_null() {
                    fail("photo", "no photo output".into());
                    return;
                }
                let mut rotation = 90;
                if let Some(get) = api.photo_rotation {
                    let mut r = 0i32;
                    if unsafe { get(session.photo_output, 0, &mut r) } == 0 { rotation = r; }
                }
                if let Ok(mut pending) = shared.pending.lock() { pending.push_back((path.clone(), library)); }
                let setting = CameraPhotoCaptureSetting { quality: 0, rotation, location: std::ptr::null_mut(), mirror: session.front };
                let rc = unsafe { capture(session.photo_output, setting) };
                if rc != 0 {
                    if let Ok(mut pending) = shared.pending.lock() { pending.retain(|(p, _)| *p != path); }
                    fail("photo", cam_error(rc));
                } else {
                    crate::log!("ohos camera: capturing {path} (rotation {rotation})");
                }
            }
            CameraCaptureRequest::StartVideo { path, audio, .. } => match self.start_video(path.clone(), audio) {
                Ok(()) => push_capture_result(input_id, CameraCaptureResult::VideoStarted { path }),
                Err(error) => fail("video", error),
            },
            CameraCaptureRequest::StopVideo => match self.stop_video() {
                Ok(path) => push_capture_result(input_id, CameraCaptureResult::VideoStopped { path }),
                Err(error) => fail("video", error),
            },
            CameraCaptureRequest::PauseVideo | CameraCaptureRequest::ResumeVideo => {
                let resume = matches!(request, CameraCaptureRequest::ResumeVideo);
                let (Some(rec), Some(f)) = (session.video.as_ref(), if resume { api.recorder_resume } else { api.recorder_pause }) else {
                    fail("video", "not recording".into());
                    return;
                };
                let rc = unsafe { f(rec.recorder) };
                if rc != 0 {
                    fail("video", format!("recorder {}: {rc}", if resume { "resume" } else { "pause" }));
                } else {
                    push_capture_result(input_id, if resume { CameraCaptureResult::VideoResumed } else { CameraCaptureResult::VideoPaused });
                }
            }
        }
    }

    fn stop_session(&mut self) {
        if self.session.as_ref().map_or(false, |s| s.video.is_some()) {
            match self.stop_video() {
                Ok(path) => push_capture_result(self.session.as_ref().unwrap().input_id, CameraCaptureResult::VideoStopped { path }),
                Err(error) => crate::error!("ohos camera: {error}"),
            }
        }
        let Some(session) = self.session.take() else {
            return;
        };
        if let Ok(mut active) = ACTIVE.lock() {
            *active = None;
        }
        // Park the receiver pointers so a callback in flight reads nothing.
        session.shared.receiver.store(0, Ordering::Release);
        if let Some(photo) = &session.photo_shared { photo.receiver.store(0, Ordering::Release); }
        if let Ok(mut active) = ACTIVE_PHOTO.lock() { *active = None; }
        let Some(api) = api() else { return };
        unsafe {
            (api.receiver_off)(session.receiver);
            if !session.photo_receiver.is_null() { (api.receiver_off)(session.photo_receiver); }
            (api.session_stop)(session.session);
            (api.session_release)(session.session);
            (api.preview_release)(session.output);
            if !session.photo_output.is_null() { (api.photo_release.unwrap_or(noop_release))(session.photo_output); }
            (api.input_close)(session.input);
            (api.input_release)(session.input);
            (api.receiver_release)(session.receiver);
            if !session.photo_receiver.is_null() { (api.receiver_release)(session.photo_receiver); }
        }
        crate::log!("ohos camera: stopped");
    }
}

impl Drop for OhCameraAccess {
    fn drop(&mut self) {
        self.stop_session();
        if let Some(api) = api() {
            unsafe {
                if !self.device_list.is_null() {
                    (api.delete_supported)(self.manager, self.device_list, self.device_count);
                }
                if !self.manager.is_null() {
                    (api.delete_manager)(self.manager);
                }
            }
        }
    }
}

unsafe extern "C" fn noop_release(_: *mut c_void) -> Rc32 {
    0
}

// ---------------------------------------------------------------------------
// Still readback (photo receiver thread)
// ---------------------------------------------------------------------------

unsafe extern "C" fn on_photo_arrived(receiver: *mut c_void) {
    let shared = match ACTIVE_PHOTO.lock() {
        Ok(active) => active.clone(),
        Err(_) => None,
    };
    let Some(shared) = shared else { return };
    if shared.receiver.load(Ordering::Acquire) != receiver as usize as u64 {
        return;
    }
    let Some(api) = api() else { return };
    let Some((path, _library)) = shared.pending.lock().ok().and_then(|mut p| p.pop_front()) else {
        crate::log!("ohos camera: a still arrived that nobody asked for");
        return;
    };
    let mut image = std::ptr::null_mut();
    let rc = (api.receiver_read_latest)(receiver, &mut image);
    if rc != 0 || image.is_null() {
        push_capture_result(shared.input_id, CameraCaptureResult::Failed { what: "photo".into(), error: format!("read still: {rc}") });
        return;
    }
    let result = read_jpeg(api, image, &path);
    (api.image_release)(image);
    match result {
        Ok((width, height)) => {
            crate::log!("ohos camera: still written to {path} ({width}x{height})");
            push_capture_result(shared.input_id, CameraCaptureResult::Photo { path, width, height });
        }
        Err(error) => push_capture_result(shared.input_id, CameraCaptureResult::Failed { what: "photo".into(), error }),
    }
}

/// Copy the image's JPEG component into `path`.
unsafe fn read_jpeg(api: &OhCameraApi, image: *mut c_void, path: &str) -> Result<(u32, u32), String> {
    let mut size = ImageSize { width: 0, height: 0 };
    let _ = (api.image_size)(image, &mut size);
    let mut type_count = 0usize;
    if (api.image_component_types)(image, std::ptr::null_mut(), &mut type_count) != 0 || type_count == 0 {
        return Err("still has no components".into());
    }
    let mut types = vec![0u32; type_count];
    let mut types_out = types.as_mut_ptr();
    if (api.image_component_types)(image, &mut types_out, &mut type_count) != 0 {
        return Err("still component types".into());
    }
    let component = types[0];
    let mut buffer = std::ptr::null_mut();
    if (api.image_byte_buffer)(image, component, &mut buffer) != 0 || buffer.is_null() {
        return Err("still byte buffer".into());
    }
    let mut len = 0usize;
    if let Some(get_size) = api.image_buffer_size {
        let _ = get_size(image, component, &mut len);
    }
    let mut base: *mut c_void = std::ptr::null_mut();
    if (api.buffer_map)(buffer, &mut base) != 0 || base.is_null() {
        return Err("still buffer map".into());
    }
    if len == 0 {
        len = size.width as usize * size.height as usize;
    }
    let bytes = std::slice::from_raw_parts(base as *const u8, len);
    // Trim to the JPEG end-of-image marker: the buffer is a fixed allocation.
    let end = bytes.windows(2).rposition(|w| w == [0xFF, 0xD9]).map(|i| i + 2).unwrap_or(bytes.len());
    let written = std::path::Path::new(path).parent().map(std::fs::create_dir_all).transpose().and_then(|_| std::fs::write(path, &bytes[..end]));
    (api.buffer_unmap)(buffer);
    written.map_err(|e| format!("write {path}: {e}"))?;
    Ok((size.width, size.height))
}

// ---------------------------------------------------------------------------
// Frame readback (receiver thread)
// ---------------------------------------------------------------------------

unsafe extern "C" fn on_image_arrived(receiver: *mut c_void) {
    let shared = match ACTIVE.lock() {
        Ok(active) => active.clone(),
        Err(_) => None,
    };
    let Some(shared) = shared else { return };
    if shared.receiver.load(Ordering::Acquire) != receiver as usize as u64 {
        return;
    }
    let Some(api) = api() else { return };
    let mut image = std::ptr::null_mut();
    let rc = (api.receiver_read_latest)(receiver, &mut image);
    if rc != 0 || image.is_null() {
        return;
    }
    read_image(api, &shared, image);
    (api.image_release)(image);
}

/// Map the image's buffer, convert NV12/NV21 → I420 into the spare frame and
/// publish it as the latest one.
unsafe fn read_image(api: &OhCameraApi, shared: &FrameShared, image: *mut c_void) {
    let mut size = ImageSize { width: 0, height: 0 };
    if (api.image_size)(image, &mut size) != 0 || size.width == 0 || size.height == 0 {
        return;
    }
    // A null `types` argument is the count query; a non-null one must point
    // at an array of `typeSize` entries (the NDK writes through it blindly).
    let mut type_count = 0usize;
    if (api.image_component_types)(image, std::ptr::null_mut(), &mut type_count) != 0 || type_count == 0 {
        return;
    }
    let mut types = vec![0u32; type_count];
    let mut types_out = types.as_mut_ptr();
    if (api.image_component_types)(image, &mut types_out, &mut type_count) != 0 {
        return;
    }
    let component = types[0];
    let mut buffer = std::ptr::null_mut();
    if (api.image_byte_buffer)(image, component, &mut buffer) != 0 || buffer.is_null() {
        return;
    }
    let mut format = -1i32;
    if let Some(get_format) = api.image_format {
        let _ = get_format(image, &mut format);
    }
    let mut swap_uv = format == NATIVEBUFFER_PIXEL_FMT_YCRCB_420_SP;
    if format >= 0 && format != NATIVEBUFFER_PIXEL_FMT_YCBCR_420_SP && !swap_uv
        && !shared.logged.load(Ordering::Relaxed)
    {
        crate::error!("ohos camera: unexpected native buffer format {format}, treating as NV12");
    }

    // Plane geometry. The Mate's driver describes NV21 as three planes (Y, U,
    // V) whose `rowStride` holds the pixel stride (1, 2, 2) and whose U/V
    // offsets differ by one byte, so every field is sanity-checked against
    // the image size before it is trusted; MapPlanes failing falls back to a
    // packed NV12 guess from the component row stride.
    let width = size.width as u64;
    let pitch = |plane: &NativeBufferPlane, fallback: u32| -> u32 {
        let candidates = [plane.row_stride, plane.column_stride];
        candidates
            .into_iter()
            .filter(|&c| c as u64 >= width)
            .min()
            .unwrap_or(fallback)
    };
    let mut base: *mut c_void = std::ptr::null_mut();
    let mut planes = NativeBufferPlanes::default();
    let mut mapped = false;
    let (mut y_off, mut y_stride, mut uv_off, mut uv_stride) = (0u64, 0u32, 0u64, 0u32);
    if (api.buffer_map_planes)(buffer, &mut base, &mut planes) == 0
        && !base.is_null()
        && planes.plane_count >= 2
    {
        mapped = true;
        y_off = planes.planes[0].offset;
        y_stride = pitch(&planes.planes[0], size.width);
        let p1 = planes.planes[1];
        if planes.plane_count >= 3 {
            let p2 = planes.planes[2];
            uv_off = p1.offset.min(p2.offset);
            // Two separately described chroma planes one byte apart are the
            // interleaved pair; whichever comes first is the first byte.
            if p1.offset.abs_diff(p2.offset) == 1 {
                swap_uv = p1.offset > p2.offset;
            }
        } else {
            uv_off = p1.offset;
        }
        uv_stride = pitch(&p1, y_stride);
        if uv_off == 0 || uv_off < y_off + y_stride as u64 * size.height as u64 {
            uv_off = y_off + y_stride as u64 * size.height as u64;
        }
    } else {
        base = std::ptr::null_mut();
        if (api.buffer_map)(buffer, &mut base) == 0 && !base.is_null() {
            mapped = true;
            let mut stride = 0i32;
            if (api.image_row_stride)(image, component, &mut stride) != 0 || stride < size.width as i32 {
                stride = size.width as i32;
            }
            y_stride = stride as u32;
            uv_off = y_stride as u64 * size.height as u64;
            uv_stride = y_stride;
        }
    }
    if !mapped {
        return;
    }
    if !shared.logged.swap(true, Ordering::Relaxed) {
        crate::log!(
            "ohos camera: first frame {}x{} component {component} format {format} planes {} y(off {y_off} stride {y_stride}) uv(off {uv_off} stride {uv_stride}) swap_uv {swap_uv}",
            size.width,
            size.height,
            planes.plane_count
        );
    }
    let (w, h) = (size.width as usize, size.height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let base = base as *const u8;
    let y_src = base.add(y_off as usize);
    let uv_src = base.add(uv_off as usize);
    // The whole UV region must exist: the last chroma row needs cw*2 bytes.
    let mut frame = shared
        .spare
        .lock()
        .ok()
        .and_then(|mut s| s.take())
        .unwrap_or_default();
    frame.width = w as u32;
    frame.height = h as u32;
    frame.y.resize(w * h, 16);
    frame.u.resize(cw * ch, 128);
    frame.v.resize(cw * ch, 128);
    for row in 0..h {
        let src = std::slice::from_raw_parts(y_src.add(row * y_stride as usize), w);
        frame.y[row * w..row * w + w].copy_from_slice(src);
    }
    for row in 0..ch {
        let src = std::slice::from_raw_parts(uv_src.add(row * uv_stride as usize), cw * 2);
        let (u_dst, v_dst) = (&mut frame.u[row * cw..row * cw + cw], &mut frame.v[row * cw..row * cw + cw]);
        for col in 0..cw {
            let (a, b) = (src[col * 2], src[col * 2 + 1]);
            if swap_uv {
                u_dst[col] = b;
                v_dst[col] = a;
            } else {
                u_dst[col] = a;
                v_dst[col] = b;
            }
        }
    }
    // Raw-frame subscribers (the `camera_frame_input` API) see the mapped NV12 view.
    if let Ok(mut cb) = shared.camera_frame_cb.lock() {
        if let Some(cb) = cb.as_mut() {
            let y_bytes = std::slice::from_raw_parts(y_src, y_stride as usize * h);
            let uv_bytes = std::slice::from_raw_parts(uv_src, uv_stride as usize * ch);
            cb(CameraFrameRef {
                timestamp_ns: 0,
                width: w,
                height: h,
                layout: CameraFrameLayout::NV12,
                matrix: CameraColorMatrix::BT601,
                plane_count: 2,
                planes: [
                    CameraFramePlaneRef { bytes: y_bytes, row_stride: y_stride as usize, pixel_stride: 1 },
                    CameraFramePlaneRef { bytes: uv_bytes, row_stride: uv_stride as usize, pixel_stride: 2 },
                    CameraFramePlaneRef::empty(),
                ],
            });
        }
    }
    if let Ok(mut cb) = shared.video_frame_cb.lock() {
        if let Some(cb) = cb.as_mut() {
            let len = (uv_off.saturating_sub(y_off)) as usize + uv_stride as usize * ch;
            let bytes = std::slice::from_raw_parts(y_src, len);
            cb(VideoBufferRef { format: shared.format, data: VideoBufferRefData::U8(bytes) });
        }
    }
    (api.buffer_unmap)(buffer);

    if let Ok(mut latest) = shared.latest.lock() {
        if let Some(old) = latest.replace(frame) {
            shared.recycle(old);
        }
    }
    shared.frames.fetch_add(1, Ordering::Relaxed);
    SignalToUI::set_ui_signal();
}

// ---------------------------------------------------------------------------
// Video playback source (UI thread)
// ---------------------------------------------------------------------------

/// A Video widget bound to the camera stream: polls the shared frame slot on
/// every vsync and uploads it to the Y/U/V plane textures.
pub struct OhCameraPlayer {
    pub video_id: LiveId,
    pub tex_y: crate::texture::TextureId,
    pub tex_u: crate::texture::TextureId,
    pub tex_v: crate::texture::TextureId,
    pub shared: Arc<FrameShared>,
    pub rotation_steps: f32,
    pub width: u32,
    pub height: u32,
    pub prepared: bool,
    pub playing: bool,
}
