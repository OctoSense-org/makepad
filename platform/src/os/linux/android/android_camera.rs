use {
    self::super::acamera_sys::*,
    crate::{
        makepad_live_id::*,
        os::linux::{android::ndk_sys, gl_sys},
        texture::{CxTexturePool, TexturePixel},
        thread::SignalToUI,
        video::*,
        video_encode::camera_video_encoder::VideoEncoder,
    },
    std::collections::{HashMap, HashSet},
    std::ffi::{CStr, CString},
    std::os::raw::{c_int, c_void},
    std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

pub struct AndroidCameraDevice {
    camera_id_str: CString,
    desc: VideoInputDesc,
    sensor_orientation_degrees: i32,
    front_facing: bool,
    /// left, top, right, bottom of the sensor's active pixels: zoom crops and
    /// focus regions are expressed in this rectangle.
    active_array: [i32; 4],
    /// The largest JPEG the device offers, used for stills.
    still_size: Option<(i32, i32)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct CameraStreamKey {
    input_id: VideoInputId,
    format_id: VideoFormatId,
}

#[derive(Default)]
struct StreamDispatch {
    video_input_cbs: Vec<Arc<Mutex<Option<VideoInputFn>>>>,
    frame_input_cbs: Vec<Arc<Mutex<Option<CameraFrameInputFn>>>>,
    preview_frame_input_cbs: Vec<Arc<Mutex<Option<CameraFrameInputFn>>>>,
    preview_hardware_buffer_input_cbs: Vec<Arc<Mutex<Option<CameraHardwareBufferInputFn>>>>,
    encoders: Vec<Arc<Mutex<Option<VideoEncoder>>>>,
}

impl StreamDispatch {
    fn needs_image_reader(&self) -> bool {
        !self.video_input_cbs.is_empty()
            || !self.frame_input_cbs.is_empty()
            || !self.preview_frame_input_cbs.is_empty()
            || !self.preview_hardware_buffer_input_cbs.is_empty()
            || !self.encoders.is_empty()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AndroidImageReaderMode {
    CpuReadable,
    HardwareBufferYuv,
}

pub struct AndroidCameraHardwareBufferFrame {
    pub buffer: *mut ndk_sys::AHardwareBuffer,
    pub timestamp_ns: u64,
    pub width: u32,
    pub height: u32,
}

unsafe impl Send for AndroidCameraHardwareBufferFrame {}

impl Drop for AndroidCameraHardwareBufferFrame {
    fn drop(&mut self) {
        if !self.buffer.is_null() {
            unsafe {
                ndk_sys::AHardwareBuffer_release(self.buffer);
            }
        }
    }
}

pub type CameraHardwareBufferInputFn =
    Box<dyn FnMut(AndroidCameraHardwareBufferFrame) + Send + 'static>;

struct PreviewSubscription {
    stream: CameraStreamKey,
    frame_cb: Arc<Mutex<Option<CameraFrameInputFn>>>,
    hardware_buffer_cb: Arc<Mutex<Option<CameraHardwareBufferInputFn>>>,
    preview_window: *mut ANativeWindow,
}

struct CameraStreamNode {
    camera_id_str: CString,
    format: VideoFormat,
    controls: CameraControlState,
    dispatch: Arc<Mutex<StreamDispatch>>,
    session: Option<AndroidCaptureSession>,
    preview_window: *mut ANativeWindow,
    needs_image_reader: bool,
}

pub struct AndroidCaptureSession {
    capture_session: *mut ACameraCaptureSession,
    output_container: *mut ACaptureSessionOutputContainer,
    image_output: *mut ACaptureSessionOutput,
    preview_output: *mut ACaptureSessionOutput,
    camera_device: *mut ACameraDevice,
    image_target: *mut ACameraOutputTarget,
    preview_target: *mut ACameraOutputTarget,
    image_window: *mut ANativeWindow,
    preview_window: *mut ANativeWindow,
    image_reader: *mut AImageReader,
    capture_request: *mut ACaptureRequest,
    capture_context: *mut AndroidCaptureContext,
    still_reader: *mut AImageReader,
    still_window: *mut ANativeWindow,
    still_output: *mut ACaptureSessionOutput,
    still_target: *mut ACameraOutputTarget,
    still_context: *mut AndroidStillContext,
}

/// What a still capture needs once its JPEG arrives: where to write it and
/// which input to report it against.
pub struct AndroidStillContext {
    input_id: VideoInputId,
    pending: Mutex<Option<String>>,
    alive: Arc<AtomicBool>,
}

/// Controls applied to the repeating request of one camera.
#[derive(Clone, Copy, Debug)]
pub struct CameraControlState {
    pub zoom: f32,
    pub exposure_bias: f32,
    pub flash: CameraFlashMode,
    pub focus: Option<(f64, f64)>,
}

impl Default for CameraControlState {
    fn default() -> Self {
        Self { zoom: 1.0, exposure_bias: 0.0, flash: CameraFlashMode::Off, focus: None }
    }
}

/// Stills and recordings finished on a camera thread, waiting for the UI thread.
static CAPTURE_RESULTS: Mutex<Vec<(VideoInputId, CameraCaptureResult)>> = Mutex::new(Vec::new());

pub fn take_capture_results() -> Vec<(VideoInputId, CameraCaptureResult)> {
    std::mem::take(&mut *CAPTURE_RESULTS.lock().unwrap())
}

fn push_capture_result(input_id: VideoInputId, result: CameraCaptureResult) {
    CAPTURE_RESULTS.lock().unwrap().push((input_id, result));
    // The event loop blocks on Java messages; wake it so the action goes out.
    super::android_jni::send_from_java_message(super::android_jni::FromJavaMessage::Wake);
}

pub struct AndroidCaptureContext {
    dispatch: Arc<Mutex<StreamDispatch>>,
    format: VideoFormat,
    alive: Arc<AtomicBool>,
    reader_mode: AndroidImageReaderMode,
    logged_first_hardware_buffer_frame: AtomicBool,
}

impl AndroidCaptureSession {
    unsafe extern "C" fn device_on_disconnected(
        _context: *mut c_void,
        _device: *mut ACameraDevice,
    ) {
    }
    unsafe extern "C" fn device_on_error(
        _context: *mut c_void,
        _device: *mut ACameraDevice,
        error: c_int,
    ) {
        crate::warning!("Android camera: device error {}", error);
    }

    /// A still arrived: write it to the path the request carried and report it.
    unsafe extern "C" fn still_on_image_available(context: *mut c_void, reader: *mut AImageReader) {
        let context = &*(context as *mut AndroidStillContext);
        let mut image = std::ptr::null_mut();
        if AImageReader_acquireNextImage(reader, &mut image) != 0 || image.is_null() {
            return;
        }
        let path = context.pending.lock().unwrap().take();
        if !context.alive.load(Ordering::Relaxed) || path.is_none() {
            AImage_delete(image);
            return;
        }
        let path = path.unwrap();
        let mut data = std::ptr::null_mut();
        let mut len = 0i32;
        let ok = AImage_getPlaneData(image, 0, &mut data, &mut len) == 0 && !data.is_null() && len > 0;
        let (mut width, mut height) = (0i32, 0i32);
        AImage_getWidth(image, &mut width);
        AImage_getHeight(image, &mut height);
        let written = if ok {
            let bytes = std::slice::from_raw_parts(data, len as usize);
            // The caller names a file, not a folder: make the folder it asked for.
            if let Some(parent) = std::path::Path::new(&path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match std::fs::write(&path, bytes) {
                Ok(()) => true,
                Err(error) => {
                    crate::warning!("Android camera: writing {path} failed: {error}");
                    false
                }
            }
        } else {
            false
        };
        AImage_delete(image);
        if written {
            push_capture_result(
                context.input_id,
                CameraCaptureResult::Photo { path, width: width.max(0) as u32, height: height.max(0) as u32 },
            );
        } else {
            push_capture_result(
                context.input_id,
                CameraCaptureResult::Failed {
                    what: "photo".to_string(),
                    error: "the still could not be written".to_string(),
                },
            );
        }
    }

    unsafe extern "C" fn image_on_image_available(context: *mut c_void, reader: *mut AImageReader) {
        let context = &*(context as *mut AndroidCaptureContext);
        if !context.alive.load(Ordering::Relaxed) {
            return;
        }

        let mut image = std::ptr::null_mut();
        if AImageReader_acquireLatestImage(reader, &mut image) != 0 || image.is_null() {
            return;
        }

        let mut timestamp_ns = 0i64;
        let _ = AImage_getTimestamp(image, &mut timestamp_ns);

        let dispatch_snapshot = {
            let guard = context.dispatch.lock().unwrap();
            (
                guard.video_input_cbs.clone(),
                guard.frame_input_cbs.clone(),
                guard.preview_frame_input_cbs.clone(),
                guard.preview_hardware_buffer_input_cbs.clone(),
                guard.encoders.clone(),
            )
        };

        if context.reader_mode == AndroidImageReaderMode::HardwareBufferYuv {
            let mut hardware_buffer = std::ptr::null_mut();
            let mut image_format = -1i32;
            let _ = AImage_getFormat(image, &mut image_format);
            if AImage_getHardwareBuffer(image, &mut hardware_buffer) == 0
                && !hardware_buffer.is_null()
            {
                context
                    .logged_first_hardware_buffer_frame
                    .store(true, Ordering::Relaxed);
                for cb in &dispatch_snapshot.3 {
                    if let Ok(mut guard) = cb.try_lock() {
                        if let Some(cb) = &mut *guard {
                            ndk_sys::AHardwareBuffer_acquire(hardware_buffer);
                            cb(AndroidCameraHardwareBufferFrame {
                                buffer: hardware_buffer,
                                timestamp_ns: timestamp_ns.max(0) as u64,
                                width: context.format.width as u32,
                                height: context.format.height as u32,
                            });
                        }
                    }
                }
            } else {
                crate::warning!(
                    "Android headset camera: AImage_getHardwareBuffer failed for hardware-buffer reader format={} size={}x{}",
                    image_format,
                    context.format.width,
                    context.format.height,
                );
            }
            AImage_delete(image);
            return;
        }

        match context.format.pixel_format {
            VideoPixelFormat::MJPEG => {
                let mut data = std::ptr::null_mut();
                let mut len = 0;
                AImage_getPlaneData(image, 0, &mut data, &mut len);
                if !data.is_null() {
                    let data = std::slice::from_raw_parts(data as *const u8, len.max(0) as usize);
                    let frame_ref = CameraFrameRef {
                        timestamp_ns: timestamp_ns.max(0) as u64,
                        width: context.format.width,
                        height: context.format.height,
                        layout: CameraFrameLayout::Mjpeg,
                        matrix: CameraColorMatrix::Unknown,
                        plane_count: 1,
                        planes: [
                            CameraFramePlaneRef {
                                bytes: data,
                                row_stride: len.max(0) as usize,
                                pixel_stride: 1,
                            },
                            CameraFramePlaneRef::empty(),
                            CameraFramePlaneRef::empty(),
                        ],
                    };

                    for cb in &dispatch_snapshot.1 {
                        if let Ok(mut guard) = cb.try_lock() {
                            if let Some(cb) = &mut *guard {
                                cb(frame_ref);
                            }
                        }
                    }
                    for cb in &dispatch_snapshot.2 {
                        if let Ok(mut guard) = cb.try_lock() {
                            if let Some(cb) = &mut *guard {
                                cb(frame_ref);
                            }
                        }
                    }
                    for enc in &dispatch_snapshot.4 {
                        if let Ok(guard) = enc.try_lock() {
                            if let Some(enc) = &*guard {
                                enc.push_frame(frame_ref);
                            }
                        }
                    }
                    for cb in &dispatch_snapshot.0 {
                        if let Ok(mut guard) = cb.try_lock() {
                            if let Some(cb) = &mut *guard {
                                cb(VideoBufferRef {
                                    format: context.format,
                                    data: VideoBufferRefData::U8(data),
                                });
                            }
                        }
                    }
                }
            }
            VideoPixelFormat::YUV420 => {
                let w = context.format.width;
                let h = context.format.height;

                let mut y_data = std::ptr::null_mut();
                let mut y_len = 0i32;
                let mut y_row_stride = 0i32;
                AImage_getPlaneData(image, 0, &mut y_data, &mut y_len);
                AImage_getPlaneRowStride(image, 0, &mut y_row_stride);

                let mut u_data = std::ptr::null_mut();
                let mut u_len = 0i32;
                let mut u_row_stride = 0i32;
                let mut u_pixel_stride = 0i32;
                AImage_getPlaneData(image, 1, &mut u_data, &mut u_len);
                AImage_getPlaneRowStride(image, 1, &mut u_row_stride);
                AImage_getPlanePixelStride(image, 1, &mut u_pixel_stride);

                let mut v_data = std::ptr::null_mut();
                let mut v_len = 0i32;
                let mut v_row_stride = 0i32;
                let mut v_pixel_stride = 0i32;
                AImage_getPlaneData(image, 2, &mut v_data, &mut v_len);
                AImage_getPlaneRowStride(image, 2, &mut v_row_stride);
                AImage_getPlanePixelStride(image, 2, &mut v_pixel_stride);

                if !y_data.is_null() && !u_data.is_null() && !v_data.is_null() {
                    let y_slice = std::slice::from_raw_parts(y_data, y_len.max(0) as usize);
                    let u_slice = std::slice::from_raw_parts(u_data, u_len.max(0) as usize);
                    let v_slice = std::slice::from_raw_parts(v_data, v_len.max(0) as usize);

                    let frame_ref = CameraFrameRef {
                        timestamp_ns: timestamp_ns.max(0) as u64,
                        width: w,
                        height: h,
                        layout: CameraFrameLayout::I420,
                        matrix: CameraColorMatrix::BT601,
                        plane_count: 3,
                        planes: [
                            CameraFramePlaneRef {
                                bytes: y_slice,
                                row_stride: y_row_stride.max(0) as usize,
                                pixel_stride: 1,
                            },
                            CameraFramePlaneRef {
                                bytes: u_slice,
                                row_stride: u_row_stride.max(0) as usize,
                                pixel_stride: u_pixel_stride.max(1) as usize,
                            },
                            CameraFramePlaneRef {
                                bytes: v_slice,
                                row_stride: v_row_stride.max(0) as usize,
                                pixel_stride: v_pixel_stride.max(1) as usize,
                            },
                        ],
                    };

                    for cb in &dispatch_snapshot.1 {
                        if let Ok(mut guard) = cb.try_lock() {
                            if let Some(cb) = &mut *guard {
                                cb(frame_ref);
                            }
                        }
                    }
                    for cb in &dispatch_snapshot.2 {
                        if let Ok(mut guard) = cb.try_lock() {
                            if let Some(cb) = &mut *guard {
                                cb(frame_ref);
                            }
                        }
                    }
                    for enc in &dispatch_snapshot.4 {
                        if let Ok(guard) = enc.try_lock() {
                            if let Some(enc) = &*guard {
                                enc.push_frame(frame_ref);
                            }
                        }
                    }

                    for cb in &dispatch_snapshot.0 {
                        if let Ok(mut guard) = cb.try_lock() {
                            if let Some(cb) = &mut *guard {
                                let y_stride = y_row_stride.max(0) as usize;
                                let uv_row_stride = u_row_stride.max(0) as usize;
                                let v_uv_row_stride = v_row_stride.max(0) as usize;
                                let uv_pixel_stride = u_pixel_stride.max(1) as usize;
                                let v_pixel_stride = v_pixel_stride.max(1) as usize;
                                let cw = w.div_ceil(2);
                                let ch = h.div_ceil(2);
                                let y_size = w * h;
                                let uv_size = cw * ch;

                                let mut packed = Vec::with_capacity(y_size + uv_size * 2);
                                for row in 0..h {
                                    let src_start = row * y_stride;
                                    packed.extend_from_slice(&y_slice[src_start..src_start + w]);
                                }

                                for row in 0..ch {
                                    for col in 0..cw {
                                        let idx = row * uv_row_stride + col * uv_pixel_stride;
                                        packed.push(u_slice.get(idx).copied().unwrap_or(128));
                                    }
                                }

                                for row in 0..ch {
                                    for col in 0..cw {
                                        let idx = row * v_uv_row_stride + col * v_pixel_stride;
                                        packed.push(v_slice.get(idx).copied().unwrap_or(128));
                                    }
                                }

                                cb(VideoBufferRef {
                                    format: context.format,
                                    data: VideoBufferRefData::U8(&packed),
                                });
                            }
                        }
                    }
                }
            }
            _ => (),
        }

        AImage_delete(image);
    }

    unsafe extern "C" fn capture_on_started(
        _context: *mut c_void,
        _session: *mut ACameraCaptureSession,
        _request: *const ACaptureRequest,
        _timestamp: i64,
    ) {
    }
    unsafe extern "C" fn capture_on_progressed(
        _context: *mut c_void,
        _session: *mut ACameraCaptureSession,
        _request: *mut ACaptureRequest,
        _result: *const ACameraMetadata,
    ) {
    }
    unsafe extern "C" fn capture_on_completed(
        _context: *mut c_void,
        _session: *mut ACameraCaptureSession,
        _request: *mut ACaptureRequest,
        _result: *const ACameraMetadata,
    ) {
    }
    unsafe extern "C" fn capture_on_failed(
        _context: *mut c_void,
        _session: *mut ACameraCaptureSession,
        _request: *mut ACaptureRequest,
        _failure: *mut ACameraCaptureFailure,
    ) {
        crate::warning!("Android camera: capture failed");
    }
    unsafe extern "C" fn capture_on_sequence_completed(
        _context: *mut c_void,
        _session: *mut ACameraCaptureSession,
        sequence_id: ::std::os::raw::c_int,
        frame_number: i64,
    ) {
        crate::warning!(
            "Android camera: capture sequence completed sequence_id={} frame_number={}",
            sequence_id,
            frame_number,
        );
    }
    unsafe extern "C" fn capture_on_sequence_aborted(
        _context: *mut c_void,
        _session: *mut ACameraCaptureSession,
        sequence_id: ::std::os::raw::c_int,
    ) {
        crate::warning!(
            "Android camera: capture sequence aborted sequence_id={}",
            sequence_id,
        );
    }
    unsafe extern "C" fn capture_on_buffer_lost(
        _context: *mut c_void,
        _session: *mut ACameraCaptureSession,
        _request: *mut ACaptureRequest,
        _window: *mut ACameraWindowType,
        frame_number: i64,
    ) {
        crate::warning!(
            "Android camera: capture buffer lost frame_number={}",
            frame_number
        );
    }

    unsafe extern "C" fn session_on_closed(
        _context: *mut c_void,
        _session: *mut ACameraCaptureSession,
    ) {
    }
    unsafe extern "C" fn session_on_ready(
        _context: *mut c_void,
        _session: *mut ACameraCaptureSession,
    ) {
    }
    unsafe extern "C" fn session_on_active(
        _context: *mut c_void,
        _session: *mut ACameraCaptureSession,
    ) {
    }

    unsafe fn start(
        dispatch: Arc<Mutex<StreamDispatch>>,
        manager: *mut ACameraManager,
        camera_id: &CString,
        format: VideoFormat,
        preview_window: Option<*mut ANativeWindow>,
        needs_image_reader: bool,
        still: Option<(VideoInputId, i32, i32)>,
    ) -> Option<Self> {
        let reader_mode = {
            let guard = dispatch.lock().unwrap();
            if !guard.preview_hardware_buffer_input_cbs.is_empty()
                && guard.video_input_cbs.is_empty()
                && guard.frame_input_cbs.is_empty()
                && guard.preview_frame_input_cbs.is_empty()
                && guard.encoders.is_empty()
            {
                AndroidImageReaderMode::HardwareBufferYuv
            } else {
                AndroidImageReaderMode::CpuReadable
            }
        };
        let alive = Arc::new(AtomicBool::new(true));
        let capture_context = Box::into_raw(Box::new(AndroidCaptureContext {
            format,
            dispatch,
            alive,
            reader_mode,
            logged_first_hardware_buffer_frame: AtomicBool::new(false),
        }));

        let mut device_callbacks = ACameraDevice_StateCallbacks {
            onError: Some(Self::device_on_error),
            onDisconnected: Some(Self::device_on_disconnected),
            context: capture_context as *mut _,
        };
        let mut camera_device = std::ptr::null_mut();

        if ACameraManager_openCamera(
            manager,
            camera_id.as_ptr(),
            &mut device_callbacks,
            &mut camera_device,
        ) != 0
        {
            crate::log!("Error opening android camera");
            let _ = Box::from_raw(capture_context);
            return None;
        };

        let mut capture_request = std::ptr::null_mut();
        ACameraDevice_createCaptureRequest(camera_device, TEMPLATE_PREVIEW, &mut capture_request);

        let mut image_reader = std::ptr::null_mut();
        let mut image_window = std::ptr::null_mut();
        let mut image_target = std::ptr::null_mut();
        let mut image_output = std::ptr::null_mut();

        if needs_image_reader {
            let image_reader_result = match reader_mode {
                AndroidImageReaderMode::HardwareBufferYuv => AImageReader_newWithUsage(
                    format.width as _,
                    format.height as _,
                    AIMAGE_FORMAT_YUV_420_888,
                    ndk_sys::AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE,
                    3,
                    &mut image_reader,
                ),
                AndroidImageReaderMode::CpuReadable => {
                    let aimage_format = match format.pixel_format {
                        VideoPixelFormat::YUV420 => AIMAGE_FORMAT_YUV_420_888,
                        VideoPixelFormat::MJPEG => AIMAGE_FORMAT_JPEG,
                        _ => {
                            crate::log!(
                                "Android camera pixelformat not possible, should not happen"
                            );
                            ACameraDevice_close(camera_device);
                            let _ = Box::from_raw(capture_context);
                            return None;
                        }
                    };
                    AImageReader_new(
                        format.width as _,
                        format.height as _,
                        aimage_format,
                        2,
                        &mut image_reader,
                    )
                }
            };
            if image_reader_result != 0 || image_reader.is_null() {
                crate::warning!(
                    "Android camera: failed to create image reader for mode {:?}",
                    reader_mode as u32
                );
                ACameraDevice_close(camera_device);
                let _ = Box::from_raw(capture_context);
                return None;
            }

            let mut image_listener = AImageReader_ImageListener {
                context: capture_context as *mut _,
                onImageAvailable: Some(Self::image_on_image_available),
            };

            AImageReader_setImageListener(image_reader, &mut image_listener);

            AImageReader_getWindow(image_reader, &mut image_window);
            ANativeWindow_acquire(image_window);

            let image_target_result = ACameraOutputTarget_create(image_window, &mut image_target);
            if !image_target.is_null() {
                let _ = ACaptureRequest_addTarget(capture_request, image_target);
            } else {
                crate::warning!(
                    "Android camera: image target creation failed result={} mode={:?} size={}x{}",
                    image_target_result,
                    reader_mode as u32,
                    format.width,
                    format.height,
                );
            }

            if reader_mode == AndroidImageReaderMode::CpuReadable
                && format.pixel_format == VideoPixelFormat::MJPEG
            {
                let jpeg_quality = 60u8;
                ACaptureRequest_setEntry_u8(
                    capture_request,
                    ACAMERA_JPEG_QUALITY,
                    1,
                    &jpeg_quality,
                );
            }

            let _ = ACaptureSessionOutput_create(image_window, &mut image_output);
        }

        // A still needs its own JPEG stream, declared before the session is
        // configured: Android fixes the output set at that point.
        let mut still_reader = std::ptr::null_mut();
        let mut still_window = std::ptr::null_mut();
        let mut still_output = std::ptr::null_mut();
        let mut still_target = std::ptr::null_mut();
        let mut still_context = std::ptr::null_mut();
        if let Some((input_id, still_width, still_height)) = still {
            if AImageReader_new(
                still_width,
                still_height,
                AIMAGE_FORMAT_JPEG,
                2,
                &mut still_reader,
            ) == 0
                && !still_reader.is_null()
            {
                still_context = Box::into_raw(Box::new(AndroidStillContext {
                    input_id,
                    pending: Mutex::new(None),
                    alive: Arc::new(AtomicBool::new(true)),
                }));
                let mut still_listener = AImageReader_ImageListener {
                    context: still_context as *mut _,
                    onImageAvailable: Some(Self::still_on_image_available),
                };
                AImageReader_setImageListener(still_reader, &mut still_listener);
                AImageReader_getWindow(still_reader, &mut still_window);
                ANativeWindow_acquire(still_window);
                let _ = ACameraOutputTarget_create(still_window, &mut still_target);
                let _ = ACaptureSessionOutput_create(still_window, &mut still_output);
            } else {
                crate::warning!("Android camera: no still stream at {still_width}x{still_height}");
            }
        }

        let mut output_container = std::ptr::null_mut();
        ACaptureSessionOutputContainer_create(&mut output_container);
        if !still_output.is_null() {
            ACaptureSessionOutputContainer_add(output_container, still_output);
        }

        if !image_output.is_null() {
            ACaptureSessionOutputContainer_add(output_container, image_output);
        }

        let mut preview_target = std::ptr::null_mut();
        let mut preview_output = std::ptr::null_mut();
        let mut preview_window_ptr = std::ptr::null_mut();
        if let Some(preview_window) = preview_window {
            if !preview_window.is_null() {
                preview_window_ptr = preview_window;
                ANativeWindow_acquire(preview_window_ptr);
                let _ = ACameraOutputTarget_create(preview_window_ptr, &mut preview_target);
                if !preview_target.is_null() {
                    let _ = ACaptureRequest_addTarget(capture_request, preview_target);
                }
                let _ = ACaptureSessionOutput_create(preview_window_ptr, &mut preview_output);
                if !preview_output.is_null() {
                    ACaptureSessionOutputContainer_add(output_container, preview_output);
                }
            }
        }

        if image_output.is_null() && preview_output.is_null() {
            ACaptureSessionOutputContainer_free(output_container);
            ACaptureRequest_free(capture_request);
            ACameraDevice_close(camera_device);
            let _ = Box::from_raw(capture_context);
            return None;
        }

        let session_callbacks = ACameraCaptureSession_stateCallbacks {
            context: capture_context as *mut _,
            onClosed: Some(Self::session_on_closed),
            onReady: Some(Self::session_on_ready),
            onActive: Some(Self::session_on_active),
        };

        let mut capture_session = std::ptr::null_mut();

        let _ = ACameraDevice_createCaptureSession(
            camera_device,
            output_container,
            &session_callbacks,
            &mut capture_session,
        );
        let mut capture_callbacks = ACameraCaptureSession_captureCallbacks {
            context: capture_context as *mut _,
            onCaptureStarted: Some(Self::capture_on_started),
            onCaptureProgressed: Some(Self::capture_on_progressed),
            onCaptureCompleted: Some(Self::capture_on_completed),
            onCaptureFailed: Some(Self::capture_on_failed),
            onCaptureSequenceCompleted: Some(Self::capture_on_sequence_completed),
            onCaptureSequenceAborted: Some(Self::capture_on_sequence_aborted),
            onCaptureBufferLost: Some(Self::capture_on_buffer_lost),
        };

        let _ = ACameraCaptureSession_setRepeatingRequest(
            capture_session,
            &mut capture_callbacks,
            1,
            &mut capture_request,
            std::ptr::null_mut(),
        );
        Some(Self {
            image_reader,
            image_window,
            image_target,
            image_output,
            preview_output,
            preview_target,
            preview_window: preview_window_ptr,
            capture_request,
            capture_session,
            output_container,
            camera_device,
            capture_context,
            still_reader,
            still_window,
            still_output,
            still_target,
            still_context,
        })
    }

    /// Apply the person's controls to the repeating request. `rotation` is the
    /// quarter turns the preview is displayed with, so a tap maps to the sensor.
    unsafe fn apply_controls(&self, state: &CameraControlState, active_array: [i32; 4], rotation: i32) {
        if self.capture_request.is_null() || self.capture_session.is_null() {
            return;
        }
        let (left, top, right, bottom) = (active_array[0], active_array[1], active_array[2], active_array[3]);
        let (full_w, full_h) = ((right - left).max(1), (bottom - top).max(1));

        // Zoom: crop the active array around its centre. CONTROL_ZOOM_RATIO is
        // Android 11 and newer and many older HALs ignore it, the crop does not.
        let zoom = state.zoom.max(1.0);
        let crop_w = (full_w as f32 / zoom).round() as i32;
        let crop_h = (full_h as f32 / zoom).round() as i32;
        let crop = [
            left + (full_w - crop_w) / 2,
            top + (full_h - crop_h) / 2,
            crop_w.max(1),
            crop_h.max(1),
        ];
        ACaptureRequest_setEntry_i32(self.capture_request, ACAMERA_SCALER_CROP_REGION, 4, crop.as_ptr());
        let zoom_ratio = zoom;
        ACaptureRequest_setEntry_float(self.capture_request, ACAMERA_CONTROL_ZOOM_RATIO, 1, &zoom_ratio);

        // Exposure compensation, in the device's own steps (usually 1/6 EV).
        let steps = (state.exposure_bias / 0.1667).round() as i32;
        ACaptureRequest_setEntry_i32(self.capture_request, ACAMERA_CONTROL_AE_EXPOSURE_COMPENSATION, 1, &steps);

        // Flash. Torch stays on; auto and on are decided per capture by AE.
        let (ae_mode, flash_mode): (u8, u8) = match state.flash {
            CameraFlashMode::Off => (1, 0),
            CameraFlashMode::On => (3, 0),
            CameraFlashMode::Auto => (2, 0),
            CameraFlashMode::Torch => (1, 2),
        };
        ACaptureRequest_setEntry_u8(self.capture_request, ACAMERA_CONTROL_AE_MODE, 1, &ae_mode);
        ACaptureRequest_setEntry_u8(self.capture_request, ACAMERA_FLASH_MODE, 1, &flash_mode);

        match state.focus {
            Some((x, y)) => {
                // The point comes from the preview as displayed; undo its turn.
                let (x, y) = (x.clamp(0.0, 1.0), y.clamp(0.0, 1.0));
                let (sx, sy) = match rotation.rem_euclid(4) {
                    1 => (y, 1.0 - x),
                    2 => (1.0 - x, 1.0 - y),
                    3 => (1.0 - y, x),
                    _ => (x, y),
                };
                let cx = left + (sx * full_w as f64) as i32;
                let cy = top + (sy * full_h as f64) as i32;
                let half_w = (full_w / 10).max(1);
                let half_h = (full_h / 10).max(1);
                let region = [
                    (cx - half_w).max(left),
                    (cy - half_h).max(top),
                    (cx + half_w).min(right),
                    (cy + half_h).min(bottom),
                    1000,
                ];
                ACaptureRequest_setEntry_i32(self.capture_request, ACAMERA_CONTROL_AF_REGIONS, 5, region.as_ptr());
                ACaptureRequest_setEntry_i32(self.capture_request, ACAMERA_CONTROL_AE_REGIONS, 5, region.as_ptr());
                let af_mode: u8 = 1; // AUTO
                ACaptureRequest_setEntry_u8(self.capture_request, ACAMERA_CONTROL_AF_MODE, 1, &af_mode);
                let trigger: u8 = 1; // START
                ACaptureRequest_setEntry_u8(self.capture_request, ACAMERA_CONTROL_AF_TRIGGER, 1, &trigger);
            }
            None => {
                let af_mode: u8 = 4; // CONTINUOUS_PICTURE
                ACaptureRequest_setEntry_u8(self.capture_request, ACAMERA_CONTROL_AF_MODE, 1, &af_mode);
                let trigger: u8 = 0; // IDLE
                ACaptureRequest_setEntry_u8(self.capture_request, ACAMERA_CONTROL_AF_TRIGGER, 1, &trigger);
            }
        }

        let mut request = self.capture_request;
        let mut callbacks = Self::capture_callbacks(self.capture_context);
        let status = ACameraCaptureSession_setRepeatingRequest(
            self.capture_session,
            &mut callbacks,
            1,
            &mut request,
            std::ptr::null_mut(),
        );
        if status != 0 {
            crate::warning!("Android camera: applying controls failed with {status}");
        }
    }

    unsafe fn capture_callbacks(context: *mut AndroidCaptureContext) -> ACameraCaptureSession_captureCallbacks {
        ACameraCaptureSession_captureCallbacks {
            context: context as *mut _,
            onCaptureStarted: Some(Self::capture_on_started),
            onCaptureProgressed: Some(Self::capture_on_progressed),
            onCaptureCompleted: Some(Self::capture_on_completed),
            onCaptureFailed: Some(Self::capture_on_failed),
            onCaptureSequenceCompleted: Some(Self::capture_on_sequence_completed),
            onCaptureSequenceAborted: Some(Self::capture_on_sequence_aborted),
            onCaptureBufferLost: Some(Self::capture_on_buffer_lost),
        }
    }

    /// Take one still into `path`. `jpeg_orientation` is the sensor's own angle,
    /// which is what the JPEG's orientation tag expects.
    unsafe fn capture_still(
        &self,
        path: &str,
        jpeg_orientation: i32,
        state: &CameraControlState,
        active_array: [i32; 4],
    ) -> Result<(), String> {
        if self.still_target.is_null() || self.still_context.is_null() {
            return Err("this camera has no still stream".to_string());
        }
        *(*self.still_context).pending.lock().unwrap() = Some(path.to_string());

        let mut request = std::ptr::null_mut();
        let status = ACameraDevice_createCaptureRequest(self.camera_device, TEMPLATE_STILL_CAPTURE, &mut request);
        if status != 0 || request.is_null() {
            return Err(format!("the still request could not be built ({status})"));
        }
        ACaptureRequest_addTarget(request, self.still_target);

        let quality: u8 = 95;
        ACaptureRequest_setEntry_u8(request, ACAMERA_JPEG_QUALITY, 1, &quality);
        ACaptureRequest_setEntry_i32(request, ACAMERA_JPEG_ORIENTATION, 1, &jpeg_orientation);

        // Carry the live controls into the still: zoom crop and flash mainly.
        let (left, top, right, bottom) = (active_array[0], active_array[1], active_array[2], active_array[3]);
        let (full_w, full_h) = ((right - left).max(1), (bottom - top).max(1));
        let zoom = state.zoom.max(1.0);
        let crop_w = (full_w as f32 / zoom).round() as i32;
        let crop_h = (full_h as f32 / zoom).round() as i32;
        let crop = [left + (full_w - crop_w) / 2, top + (full_h - crop_h) / 2, crop_w.max(1), crop_h.max(1)];
        ACaptureRequest_setEntry_i32(request, ACAMERA_SCALER_CROP_REGION, 4, crop.as_ptr());
        let (ae_mode, flash_mode): (u8, u8) = match state.flash {
            CameraFlashMode::Off => (1, 0),
            CameraFlashMode::On => (3, 0),
            CameraFlashMode::Auto => (2, 0),
            CameraFlashMode::Torch => (1, 2),
        };
        ACaptureRequest_setEntry_u8(request, ACAMERA_CONTROL_AE_MODE, 1, &ae_mode);
        ACaptureRequest_setEntry_u8(request, ACAMERA_FLASH_MODE, 1, &flash_mode);

        let mut callbacks = Self::capture_callbacks(self.capture_context);
        let mut requests = request;
        let status = ACameraCaptureSession_capture(
            self.capture_session,
            &mut callbacks,
            1,
            &mut requests,
            std::ptr::null_mut(),
        );
        ACaptureRequest_free(request);
        if status != 0 {
            *(*self.still_context).pending.lock().unwrap() = None;
            return Err(format!("the camera refused the still ({status})"));
        }
        Ok(())
    }

    unsafe fn stop(self) {
        (*self.capture_context)
            .alive
            .store(false, Ordering::Relaxed);
        if !self.still_context.is_null() {
            (*self.still_context).alive.store(false, Ordering::Relaxed);
        }
        if !self.still_reader.is_null() {
            let mut none = AImageReader_ImageListener { context: std::ptr::null_mut(), onImageAvailable: None };
            AImageReader_setImageListener(self.still_reader, &mut none);
            AImageReader_delete(self.still_reader);
        }
        if !self.still_target.is_null() {
            ACameraOutputTarget_free(self.still_target);
        }
        if !self.still_output.is_null() {
            ACaptureSessionOutput_free(self.still_output);
        }
        if !self.still_window.is_null() {
            ANativeWindow_release(self.still_window);
        }
        if !self.still_context.is_null() {
            let _ = Box::from_raw(self.still_context);
        }

        if !self.image_reader.is_null() {
            let mut image_listener = AImageReader_ImageListener {
                context: std::ptr::null_mut(),
                onImageAvailable: None,
            };
            AImageReader_setImageListener(self.image_reader, &mut image_listener);
        }

        ACameraCaptureSession_stopRepeating(self.capture_session);
        ACameraCaptureSession_close(self.capture_session);
        ACaptureSessionOutputContainer_free(self.output_container);
        if !self.image_output.is_null() {
            ACaptureSessionOutput_free(self.image_output);
        }
        if !self.preview_output.is_null() {
            ACaptureSessionOutput_free(self.preview_output);
        }
        if !self.image_target.is_null() {
            ACaptureRequest_removeTarget(self.capture_request, self.image_target);
            ACameraOutputTarget_free(self.image_target);
        }
        if !self.preview_target.is_null() {
            ACaptureRequest_removeTarget(self.capture_request, self.preview_target);
            ACameraOutputTarget_free(self.preview_target);
        }
        ACaptureRequest_free(self.capture_request);
        if !self.image_window.is_null() {
            ANativeWindow_release(self.image_window);
        }
        if !self.preview_window.is_null() {
            ANativeWindow_release(self.preview_window);
        }
        if !self.image_reader.is_null() {
            AImageReader_delete(self.image_reader);
        }
        ACameraDevice_close(self.camera_device);
        let _ = Box::from_raw(self.capture_context);
    }
}

pub struct AndroidCameraAccess {
    pub video_input_cb: [Arc<Mutex<Option<VideoInputFn>>>; MAX_VIDEO_DEVICE_INDEX],
    pub camera_frame_input_cb: [Arc<Mutex<Option<CameraFrameInputFn>>>; MAX_VIDEO_DEVICE_INDEX],
    pub video_output_cb: [Arc<Mutex<Option<VideoOutputFn>>>; MAX_VIDEO_DEVICE_INDEX],
    pub video_encoder_config: [Arc<Mutex<Option<VideoEncoderConfig>>>; MAX_VIDEO_DEVICE_INDEX],
    video_encoder: [Arc<Mutex<Option<VideoEncoder>>>; MAX_VIDEO_DEVICE_INDEX],
    manager: *mut ACameraManager,
    devices: Vec<AndroidCameraDevice>,
    streams: HashMap<CameraStreamKey, CameraStreamNode>,
    slot_streams: [Option<CameraStreamKey>; MAX_VIDEO_DEVICE_INDEX],
    preview_subscriptions: HashMap<LiveId, PreviewSubscription>,
    active_inputs: Vec<(VideoInputId, VideoFormatId)>,
}

impl AndroidCameraAccess {
    pub fn new(change_signal: SignalToUI) -> Arc<Mutex<Self>> {
        unsafe {
            let manager = ACameraManager_create();

            change_signal.set();

            let camera_access = Arc::new(Mutex::new(Self {
                video_input_cb: Default::default(),
                camera_frame_input_cb: Default::default(),
                video_output_cb: Default::default(),
                video_encoder_config: Default::default(),
                video_encoder: Default::default(),
                devices: Default::default(),
                streams: Default::default(),
                slot_streams: [None; MAX_VIDEO_DEVICE_INDEX],
                preview_subscriptions: Default::default(),
                active_inputs: Vec::new(),
                manager,
            }));

            camera_access
        }
    }

    fn key_for(&self, input_id: VideoInputId, format_id: VideoFormatId) -> Option<CameraStreamKey> {
        let device = self.devices.iter().find(|d| d.desc.input_id == input_id)?;
        if device.desc.formats.iter().any(|f| f.format_id == format_id) {
            Some(CameraStreamKey {
                input_id,
                format_id,
            })
        } else {
            None
        }
    }

    fn format_for_key(&self, key: CameraStreamKey) -> Option<VideoFormat> {
        let device = self
            .devices
            .iter()
            .find(|d| d.desc.input_id == key.input_id)?;
        device
            .desc
            .formats
            .iter()
            .find(|f| f.format_id == key.format_id)
            .copied()
    }

    fn camera_id_for_key(&self, key: CameraStreamKey) -> Option<CString> {
        let device = self
            .devices
            .iter()
            .find(|d| d.desc.input_id == key.input_id)?;
        Some(device.camera_id_str.clone())
    }

    fn refresh_slot_encoder(&mut self, index: usize) {
        let key = self.slot_streams[index];
        let config_opt = *self.video_encoder_config[index].lock().unwrap();
        let output_present = self.video_output_cb[index].lock().unwrap().is_some();

        let Some(mut config) = config_opt else {
            if self.video_encoder[index].lock().unwrap().is_some() {
                *self.video_encoder[index].lock().unwrap() = None;
            }
            return;
        };

        if !matches!(config.source, VideoEncodeSource::Camera { .. }) {
            return;
        }

        let Some(key) = key else {
            *self.video_encoder[index].lock().unwrap() = None;
            return;
        };

        let Some(format) = self.format_for_key(key) else {
            *self.video_encoder[index].lock().unwrap() = None;
            return;
        };

        if !output_present {
            *self.video_encoder[index].lock().unwrap() = None;
            return;
        }

        config.width = format.width as u32;
        config.height = format.height as u32;
        if let Some(fps) = format.frame_rate {
            config.fps_num = fps.max(1.0).round() as u32;
            config.fps_den = 1;
        }
        config.source = VideoEncodeSource::Camera {
            input_id: key.input_id,
            format_id: key.format_id,
        };

        let output_cb = self.video_output_cb[index].clone();
        *self.video_encoder[index].lock().unwrap() = VideoEncoder::start(
            config,
            Box::new(move |packet| {
                if let Some(cb) = &mut *output_cb.lock().unwrap() {
                    cb(packet);
                }
            }),
        );
    }

    fn build_dispatch_for_key(
        &self,
        key: CameraStreamKey,
    ) -> (StreamDispatch, *mut ANativeWindow, bool) {
        let mut dispatch = StreamDispatch::default();
        let mut preview_window: *mut ANativeWindow = std::ptr::null_mut();

        for index in 0..MAX_VIDEO_DEVICE_INDEX {
            if self.slot_streams[index] != Some(key) {
                continue;
            }
            if self.video_input_cb[index].lock().unwrap().is_some() {
                dispatch
                    .video_input_cbs
                    .push(self.video_input_cb[index].clone());
            }
            if self.camera_frame_input_cb[index].lock().unwrap().is_some() {
                dispatch
                    .frame_input_cbs
                    .push(self.camera_frame_input_cb[index].clone());
            }
            if self.video_encoder[index].lock().unwrap().is_some() {
                dispatch.encoders.push(self.video_encoder[index].clone());
            }
        }

        for sub in self.preview_subscriptions.values() {
            if sub.stream != key {
                continue;
            }
            if sub.frame_cb.lock().unwrap().is_some() {
                dispatch.preview_frame_input_cbs.push(sub.frame_cb.clone());
            }
            if sub.hardware_buffer_cb.lock().unwrap().is_some() {
                dispatch
                    .preview_hardware_buffer_input_cbs
                    .push(sub.hardware_buffer_cb.clone());
            }
            if preview_window.is_null() && !sub.preview_window.is_null() {
                preview_window = sub.preview_window;
            }
        }

        let needs_image_reader = dispatch.needs_image_reader();
        (dispatch, preview_window, needs_image_reader)
    }

    fn restart_stream_if_needed(&mut self, key: CameraStreamKey) {
        let (dispatch, desired_preview_window, needs_image_reader) =
            self.build_dispatch_for_key(key);

        let Some(node) = self.streams.get_mut(&key) else {
            return;
        };

        *node.dispatch.lock().unwrap() = dispatch;

        let restart_needed = node.session.is_none()
            || node.preview_window != desired_preview_window
            || node.needs_image_reader != needs_image_reader;

        if !restart_needed {
            return;
        }

        if let Some(session) = node.session.take() {
            unsafe { session.stop() };
        }

        node.preview_window = desired_preview_window;
        node.needs_image_reader = needs_image_reader;

        if !needs_image_reader && desired_preview_window.is_null() {
            return;
        }

        let still = self
            .devices
            .iter()
            .find(|device| device.desc.input_id == key.input_id)
            .and_then(|device| device.still_size.map(|(w, h)| (key.input_id, w, h)));
        node.session = unsafe {
            AndroidCaptureSession::start(
                node.dispatch.clone(),
                self.manager,
                &node.camera_id_str,
                node.format,
                if desired_preview_window.is_null() {
                    None
                } else {
                    Some(desired_preview_window)
                },
                needs_image_reader,
                still,
            )
        };
        if let Some(session) = node.session.as_ref() {
            let (active_array, rotation) = self
                .devices
                .iter()
                .find(|device| device.desc.input_id == key.input_id)
                .map(|device| {
                    (
                        device.active_array,
                        ((360 - device.sensor_orientation_degrees).rem_euclid(360) / 90) % 4,
                    )
                })
                .unwrap_or(([0, 0, 0, 0], 0));
            unsafe { session.apply_controls(&node.controls, active_array, rotation) };
        }
    }

    /// Apply a control to the stream serving this input.
    pub fn control(&mut self, input_id: VideoInputId, control: CameraControl) {
        let Some(key) = self
            .streams
            .keys()
            .copied()
            .find(|key| key.input_id == input_id)
        else {
            return;
        };
        let Some(device) = self.devices.iter().find(|d| d.desc.input_id == input_id) else { return };
        let active_array = device.active_array;
        let rotation = ((360 - device.sensor_orientation_degrees).rem_euclid(360) / 90) % 4;
        let Some(node) = self.streams.get_mut(&key) else { return };
        match control {
            CameraControl::FocusPoint { x, y } => node.controls.focus = Some((x, y)),
            CameraControl::ContinuousFocus => node.controls.focus = None,
            CameraControl::ZoomRatio(zoom) => node.controls.zoom = zoom.max(1.0),
            CameraControl::ExposureBias(ev) => node.controls.exposure_bias = ev,
            CameraControl::Flash(mode) => node.controls.flash = mode,
        }
        let controls = node.controls;
        if let Some(session) = node.session.as_ref() {
            unsafe { session.apply_controls(&controls, active_array, rotation) };
        }
    }

    /// Take a still, or report that recording is not implemented here yet.
    pub fn capture(&mut self, input_id: VideoInputId, request: CameraCaptureRequest) {
        let Some(key) = self.streams.keys().copied().find(|key| key.input_id == input_id) else {
            push_capture_result(input_id, CameraCaptureResult::Failed {
                what: "capture".to_string(),
                error: "this camera is not open".to_string(),
            });
            return;
        };
        let Some(device) = self.devices.iter().find(|d| d.desc.input_id == input_id) else { return };
        let active_array = device.active_array;
        let jpeg_orientation = device.sensor_orientation_degrees.rem_euclid(360);
        let Some(node) = self.streams.get(&key) else { return };
        let controls = node.controls;
        let Some(session) = node.session.as_ref() else {
            push_capture_result(input_id, CameraCaptureResult::Failed {
                what: "capture".to_string(),
                error: "this camera has no running session".to_string(),
            });
            return;
        };
        match request {
            CameraCaptureRequest::Photo { path, .. } => {
                if let Err(error) = unsafe {
                    session.capture_still(&path, jpeg_orientation, &controls, active_array)
                } {
                    push_capture_result(input_id, CameraCaptureResult::Failed { what: "photo".to_string(), error });
                }
            }
            CameraCaptureRequest::StartVideo { .. }
            | CameraCaptureRequest::PauseVideo
            | CameraCaptureRequest::ResumeVideo
            | CameraCaptureRequest::StopVideo => {
                push_capture_result(input_id, CameraCaptureResult::Failed {
                    what: "video".to_string(),
                    error: "recording is not implemented on Android yet".to_string(),
                });
            }
        }
    }

    fn reconcile_streams(&mut self) {
        let mut required = HashSet::new();
        for key in self.slot_streams.iter().flatten() {
            required.insert(*key);
        }
        for sub in self.preview_subscriptions.values() {
            required.insert(sub.stream);
        }

        let existing_keys: Vec<_> = self.streams.keys().copied().collect();
        for key in existing_keys {
            if !required.contains(&key) {
                if let Some(mut node) = self.streams.remove(&key) {
                    if let Some(session) = node.session.take() {
                        unsafe { session.stop() };
                    }
                }
            }
        }

        for key in required.iter().copied() {
            if self.streams.contains_key(&key) {
                continue;
            }
            let Some(camera_id_str) = self.camera_id_for_key(key) else {
                continue;
            };
            let Some(format) = self.format_for_key(key) else {
                continue;
            };
            self.streams.insert(
                key,
                CameraStreamNode {
                    camera_id_str,
                    format,
                    controls: CameraControlState::default(),
                    dispatch: Arc::new(Mutex::new(StreamDispatch::default())),
                    session: None,
                    preview_window: std::ptr::null_mut(),
                    needs_image_reader: false,
                },
            );
        }

        let keys: Vec<_> = required.into_iter().collect();
        for key in keys {
            self.restart_stream_if_needed(key);
        }
    }

    pub fn use_video_input(&mut self, inputs: &[(VideoInputId, VideoFormatId)]) {
        self.active_inputs = inputs.to_vec();

        self.slot_streams = [None; MAX_VIDEO_DEVICE_INDEX];
        for (index, (input_id, format_id)) in inputs.iter().enumerate() {
            if index >= MAX_VIDEO_DEVICE_INDEX {
                break;
            }
            self.slot_streams[index] = self.key_for(*input_id, *format_id);
        }

        for index in 0..MAX_VIDEO_DEVICE_INDEX {
            self.refresh_slot_encoder(index);
        }

        self.reconcile_streams();
    }

    pub fn active_inputs(&self) -> Vec<(VideoInputId, VideoFormatId)> {
        self.active_inputs.clone()
    }

    pub fn register_preview(
        &mut self,
        video_id: LiveId,
        input_id: VideoInputId,
        format_id: VideoFormatId,
        frame_cb: Option<CameraFrameInputFn>,
        preview_window: Option<*mut ANativeWindow>,
    ) {
        let Some(stream) = self.key_for(input_id, format_id) else {
            crate::log!("camera: no stream for this input and format; the preview stays empty");
            return;
        };
        if let Some(old) = self.preview_subscriptions.remove(&video_id) {
            if !old.preview_window.is_null() {
                unsafe { ANativeWindow_release(old.preview_window) };
            }
        }

        self.preview_subscriptions.insert(
            video_id,
            PreviewSubscription {
                stream,
                frame_cb: Arc::new(Mutex::new(frame_cb)),
                hardware_buffer_cb: Arc::new(Mutex::new(None)),
                preview_window: preview_window.unwrap_or(std::ptr::null_mut()),
            },
        );
        self.reconcile_streams();
    }

    pub fn register_preview_hardware_buffer(
        &mut self,
        video_id: LiveId,
        input_id: VideoInputId,
        format_id: VideoFormatId,
        hardware_buffer_cb: CameraHardwareBufferInputFn,
        preview_window: Option<*mut ANativeWindow>,
    ) {
        let Some(stream) = self.key_for(input_id, format_id) else {
            return;
        };

        if let Some(old) = self.preview_subscriptions.remove(&video_id) {
            if !old.preview_window.is_null() {
                unsafe { ANativeWindow_release(old.preview_window) };
            }
        }

        self.preview_subscriptions.insert(
            video_id,
            PreviewSubscription {
                stream,
                frame_cb: Arc::new(Mutex::new(None)),
                hardware_buffer_cb: Arc::new(Mutex::new(Some(hardware_buffer_cb))),
                preview_window: preview_window.unwrap_or(std::ptr::null_mut()),
            },
        );
        self.reconcile_streams();
    }

    pub fn update_preview_window(
        &mut self,
        video_id: LiveId,
        preview_window: Option<*mut ANativeWindow>,
    ) {
        let Some(sub) = self.preview_subscriptions.get_mut(&video_id) else {
            return;
        };

        let next = preview_window.unwrap_or(std::ptr::null_mut());
        if sub.preview_window == next {
            return;
        }

        if !sub.preview_window.is_null() {
            unsafe { ANativeWindow_release(sub.preview_window) };
        }

        sub.preview_window = next;
        self.reconcile_streams();
    }

    pub fn unregister_preview(&mut self, video_id: LiveId) {
        if let Some(sub) = self.preview_subscriptions.remove(&video_id) {
            if !sub.preview_window.is_null() {
                unsafe { ANativeWindow_release(sub.preview_window) };
            }
        }
        self.reconcile_streams();
    }

    pub fn configure_video_encoder(
        &mut self,
        index: usize,
        config: VideoEncoderConfig,
        output: VideoOutputFn,
    ) -> Result<(), VideoEncodeError> {
        *self.video_output_cb[index].lock().unwrap() = Some(output);
        *self.video_encoder_config[index].lock().unwrap() = Some(config);

        if matches!(config.source, VideoEncodeSource::Camera { .. }) {
            self.refresh_slot_encoder(index);
            self.reconcile_streams();
            return Ok(());
        }

        let output_cb = self.video_output_cb[index].clone();
        *self.video_encoder[index].lock().unwrap() = VideoEncoder::start(
            config,
            Box::new(move |packet| {
                if let Some(cb) = &mut *output_cb.lock().unwrap() {
                    cb(packet);
                }
            }),
        );
        if self.video_encoder[index].lock().unwrap().is_none() {
            crate::error!("android video encoder unavailable for slot {}", index);
            return Err(VideoEncodeError::CodecUnavailable);
        }

        Ok(())
    }

    pub fn video_encoder_push_frame(&mut self, index: usize, frame: CameraFrameRef<'_>) {
        if let Some(encoder) = &*self.video_encoder[index].lock().unwrap() {
            encoder.push_frame(frame);
        }
    }

    pub fn video_encoder_request_keyframe(&mut self, index: usize) -> Result<(), VideoEncodeError> {
        let guard = self.video_encoder[index].lock().unwrap();
        let encoder = guard.as_ref().ok_or(VideoEncodeError::EncoderNotStarted)?;
        encoder.request_keyframe()
    }

    pub fn video_encoder_capture_texture_frame(
        &mut self,
        index: usize,
        timestamp_ns: u64,
        gl: &gl_sys::LibGl,
        textures: &mut CxTexturePool,
    ) -> Result<(), VideoEncodeError> {
        let config = self.video_encoder_config[index]
            .lock()
            .unwrap()
            .ok_or(VideoEncodeError::EncoderNotStarted)?;
        let texture_id = match config.source {
            VideoEncodeSource::Texture { texture_id } => texture_id,
            _ => return Err(VideoEncodeError::UnsupportedSource),
        };

        let encoder_guard = self.video_encoder[index].lock().unwrap();
        let encoder = encoder_guard
            .as_ref()
            .ok_or(VideoEncodeError::EncoderNotStarted)?;

        let cx_texture = &mut textures[texture_id];
        let alloc = cx_texture
            .alloc
            .as_ref()
            .ok_or(VideoEncodeError::InvalidTexture)?;
        if alloc.width == 0 || alloc.height == 0 {
            return Err(VideoEncodeError::InvalidTextureSize);
        }
        if alloc.pixel != TexturePixel::BGRAu8 {
            return Err(VideoEncodeError::UnsupportedTextureFormat);
        }

        let texture = cx_texture
            .os
            .gl_texture
            .ok_or(VideoEncodeError::InvalidTexture)?;

        let mut framebuffer = 0;
        unsafe {
            (gl.glGenFramebuffers)(1, &mut framebuffer);
            (gl.glBindFramebuffer)(gl_sys::FRAMEBUFFER, framebuffer);
            (gl.glFramebufferTexture2D)(
                gl_sys::FRAMEBUFFER,
                gl_sys::COLOR_ATTACHMENT0,
                gl_sys::TEXTURE_2D,
                texture,
                0,
            );
        }

        let mut rgba = vec![0u8; alloc.width * alloc.height * 4];
        unsafe {
            (gl.glReadPixels)(
                0,
                0,
                alloc.width as i32,
                alloc.height as i32,
                gl_sys::RGBA,
                gl_sys::UNSIGNED_BYTE,
                rgba.as_mut_ptr() as *mut _,
            );
            let read_err = (gl.glGetError)();
            (gl.glBindFramebuffer)(gl_sys::FRAMEBUFFER, 0);
            (gl.glDeleteFramebuffers)(1, &framebuffer);
            if read_err != gl_sys::NO_ERROR {
                return Err(VideoEncodeError::InvalidTexture);
            }
        }

        let mut frame = CameraFrameOwned::default();
        if !convert_rgba_8888_to_i420(
            &rgba,
            alloc.width,
            alloc.height,
            timestamp_ns,
            CameraColorMatrix::BT709,
            &mut frame,
        ) {
            return Err(VideoEncodeError::UnsupportedTextureFormat);
        }

        encoder.push_frame(CameraFrameRef {
            timestamp_ns: frame.timestamp_ns,
            width: frame.width,
            height: frame.height,
            layout: frame.layout,
            matrix: frame.matrix,
            plane_count: frame.plane_count,
            planes: [
                CameraFramePlaneRef {
                    bytes: &frame.planes[0].bytes,
                    row_stride: frame.planes[0].row_stride,
                    pixel_stride: frame.planes[0].pixel_stride,
                },
                CameraFramePlaneRef {
                    bytes: &frame.planes[1].bytes,
                    row_stride: frame.planes[1].row_stride,
                    pixel_stride: frame.planes[1].pixel_stride,
                },
                CameraFramePlaneRef {
                    bytes: &frame.planes[2].bytes,
                    row_stride: frame.planes[2].row_stride,
                    pixel_stride: frame.planes[2].pixel_stride,
                },
            ],
        });

        Ok(())
    }

    /// Degrees the renderer must turn a frame by so it is upright in a portrait
    /// window. ACAMERA_SENSOR_ORIENTATION is the CLOCKWISE angle the frame needs,
    /// while the YUV shader turns counter-clockwise, so it is the complement.
    /// Measured on a OnePlus 6: back 90, front 270 -> 270 and 90 quarter turns.
    pub fn sensor_orientation_for_input(&self, input_id: VideoInputId) -> i32 {
        self.devices
            .iter()
            .find(|device| device.desc.input_id == input_id)
            .map(|device| (360 - device.sensor_orientation_degrees).rem_euclid(360))
            .unwrap_or(0)
    }

    /// True for a camera whose frames read as a mirror (the selfie camera).
    pub fn is_front_facing(&self, input_id: VideoInputId) -> bool {
        self.devices
            .iter()
            .find(|device| device.desc.input_id == input_id)
            .map(|device| device.front_facing)
            .unwrap_or(false)
    }

    pub fn format_size(
        &self,
        input_id: VideoInputId,
        format_id: VideoFormatId,
    ) -> Option<(u32, u32)> {
        let device = self
            .devices
            .iter()
            .find(|device| device.desc.input_id == input_id)?;
        let format = device
            .desc
            .formats
            .iter()
            .find(|format| format.format_id == format_id)?;
        Some((format.width as u32, format.height as u32))
    }

    pub fn get_updated_descs(&mut self) -> Vec<VideoInputDesc> {
        self.devices.clear();
        unsafe {
            let mut camera_ids_ptr = std::ptr::null_mut();
            ACameraManager_getCameraIdList(self.manager, &mut camera_ids_ptr);
            let camera_ids = std::slice::from_raw_parts(
                (*camera_ids_ptr).cameraIds,
                (*camera_ids_ptr).numCameras as usize,
            );
            for i in 0..camera_ids.len() {
                let camera_id = camera_ids[i];
                let mut meta_data = std::ptr::null_mut();
                ACameraManager_getCameraCharacteristics(self.manager, camera_id, &mut meta_data);
                let camera_id_str = CStr::from_ptr(camera_id);

                let mut entry = std::mem::zeroed();
                if ACameraMetadata_getConstEntry(meta_data, ACAMERA_LENS_FACING, &mut entry) != 0 {
                    continue;
                };

                let front_facing = (*entry.data.u8_) == ACAMERA_LENS_FACING_FRONT;
                let name = if front_facing {
                    "Front Camera"
                } else if (*entry.data.u8_) == ACAMERA_LENS_FACING_BACK {
                    "Back Camera"
                } else if (*entry.data.u8_) == ACAMERA_LENS_FACING_EXTERNAL {
                    "External Camera"
                } else {
                    continue;
                };

                let mut sensor_orientation_degrees = 0i32;
                let mut orientation_entry = std::mem::zeroed();
                if ACameraMetadata_getConstEntry(
                    meta_data,
                    ACAMERA_SENSOR_ORIENTATION,
                    &mut orientation_entry,
                ) == 0
                    && orientation_entry.count > 0
                    && !orientation_entry.data.i32_.is_null()
                {
                    sensor_orientation_degrees = *orientation_entry.data.i32_;
                }

                let mut active_array = [0i32, 0, 0, 0];
                let mut array_entry = std::mem::zeroed();
                if ACameraMetadata_getConstEntry(
                    meta_data,
                    ACAMERA_SENSOR_INFO_ACTIVE_ARRAY_SIZE,
                    &mut array_entry,
                ) == 0
                    && array_entry.count >= 4
                    && !array_entry.data.i32_.is_null()
                {
                    for k in 0..4 {
                        active_array[k] = *array_entry.data.i32_.offset(k as isize);
                    }
                }

                let mut entry = std::mem::zeroed();
                ACameraMetadata_getConstEntry(
                    meta_data,
                    ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS,
                    &mut entry,
                );
                let mut still_size: Option<(i32, i32)> = None;
                let mut formats = Vec::new();
                for j in (0..entry.count as isize).step_by(4) {
                    if (*entry.data.i32_.offset(j + 3)) != 0 {
                        continue;
                    }
                    let format = *entry.data.i32_.offset(j) as u32;
                    let width = *entry.data.i32_.offset(j + 1);
                    let height = *entry.data.i32_.offset(j + 2);

                    if format == AIMAGE_FORMAT_JPEG
                        && still_size.map_or(true, |(w, h)| (width as i64 * height as i64) > (w as i64 * h as i64))
                    {
                        still_size = Some((width, height));
                    }
                    if format == AIMAGE_FORMAT_YUV_420_888 || format == AIMAGE_FORMAT_JPEG {
                        let format_id =
                            LiveId::from_str(&format!("{} {} {:?}", width, height, format)).into();

                        formats.push(VideoFormat {
                            format_id,
                            width: width as usize,
                            height: height as usize,
                            frame_rate: None,
                            pixel_format: if format == AIMAGE_FORMAT_YUV_420_888 {
                                VideoPixelFormat::YUV420
                            } else {
                                VideoPixelFormat::MJPEG
                            },
                        });
                    }
                }
                if !formats.is_empty() {
                    let input_id = LiveId::from_str(&format!("{:?}", camera_id_str)).into();
                    let desc = VideoInputDesc {
                        input_id,
                        name: name.to_string(),
                        formats,
                    };
                    self.devices.push(AndroidCameraDevice {
                        camera_id_str: camera_id_str.into(),
                        desc,
                        sensor_orientation_degrees,
                        front_facing,
                        active_array,
                        still_size,
                    });
                }
                ACameraMetadata_free(meta_data);
            }

            ACameraManager_deleteCameraIdList(camera_ids_ptr);
        }

        self.reconcile_streams();

        let mut descs = Vec::new();
        for device in &self.devices {
            descs.push(device.desc.clone());
        }
        descs
    }
}
