//! The OpenHarmony NDK calls the shell makes: Node-API, the XComponent's surface and touch
//! callbacks, the display's vsync, the native window's reference count, raw files, the input
//! method and the log.
#![allow(non_camel_case_types, non_snake_case)]
use std::ffi::{c_char, c_int, c_uint, c_void};

pub type napi_env = *mut c_void;
pub type napi_value = *mut c_void;
pub type napi_callback_info = *mut c_void;
pub type napi_threadsafe_function = *mut c_void;
pub type napi_status = c_int;
pub type napi_callback = unsafe extern "C" fn(env: napi_env, info: napi_callback_info) -> napi_value;
pub type napi_threadsafe_function_call_js = unsafe extern "C" fn(env: napi_env, js_callback: napi_value, context: *mut c_void, data: *mut c_void);

pub const NAPI_OK: napi_status = 0;
pub const NAPI_AUTO_LENGTH: usize = usize::MAX;
/// `napi_tsfn_nonblocking`
pub const NAPI_TSFN_NONBLOCKING: c_int = 0;

#[repr(C)]
pub struct napi_property_descriptor {
    pub utf8name: *const c_char,
    pub name: napi_value,
    pub method: Option<napi_callback>,
    pub getter: Option<napi_callback>,
    pub setter: Option<napi_callback>,
    pub value: napi_value,
    pub attributes: c_int,
    pub data: *mut c_void,
}

#[repr(C)]
pub struct napi_module {
    pub nm_version: c_int,
    pub nm_flags: c_uint,
    pub nm_filename: *const c_char,
    pub nm_register_func: Option<unsafe extern "C" fn(env: napi_env, exports: napi_value) -> napi_value>,
    pub nm_modname: *const c_char,
    pub nm_priv: *mut c_void,
    pub reserved: [*mut c_void; 4],
}

#[link(name = "ace_napi.z")]
extern "C" {
    pub fn napi_module_register(module: *mut napi_module);
    pub fn napi_define_properties(env: napi_env, object: napi_value, count: usize, properties: *const napi_property_descriptor) -> napi_status;
    pub fn napi_get_cb_info(env: napi_env, info: napi_callback_info, argc: *mut usize, argv: *mut napi_value, this: *mut napi_value, data: *mut *mut c_void) -> napi_status;
    pub fn napi_get_value_string_utf8(env: napi_env, value: napi_value, buf: *mut c_char, size: usize, len: *mut usize) -> napi_status;
    pub fn napi_get_value_double(env: napi_env, value: napi_value, out: *mut f64) -> napi_status;
    pub fn napi_get_value_bool(env: napi_env, value: napi_value, out: *mut bool) -> napi_status;
    pub fn napi_has_named_property(env: napi_env, object: napi_value, name: *const c_char, out: *mut bool) -> napi_status;
    pub fn napi_get_named_property(env: napi_env, object: napi_value, name: *const c_char, out: *mut napi_value) -> napi_status;
    pub fn napi_unwrap(env: napi_env, object: napi_value, out: *mut *mut c_void) -> napi_status;
    pub fn napi_get_undefined(env: napi_env, out: *mut napi_value) -> napi_status;
    pub fn napi_get_boolean(env: napi_env, value: bool, out: *mut napi_value) -> napi_status;
    pub fn napi_create_string_utf8(env: napi_env, text: *const c_char, len: usize, out: *mut napi_value) -> napi_status;
    pub fn napi_call_function(env: napi_env, this: napi_value, function: napi_value, argc: usize, argv: *const napi_value, out: *mut napi_value) -> napi_status;
    pub fn napi_create_threadsafe_function(env: napi_env, function: napi_value, resource: napi_value, resource_name: napi_value, max_queue: usize, threads: usize,
        finalize_data: *mut c_void, finalize: *mut c_void, context: *mut c_void, call_js: Option<napi_threadsafe_function_call_js>, out: *mut napi_threadsafe_function) -> napi_status;
    pub fn napi_call_threadsafe_function(function: napi_threadsafe_function, data: *mut c_void, mode: c_int) -> napi_status;
    pub fn napi_create_arraybuffer(env: napi_env, len: usize, data: *mut *mut c_void, out: *mut napi_value) -> napi_status;
    pub fn napi_is_arraybuffer(env: napi_env, value: napi_value, out: *mut bool) -> napi_status;
    pub fn napi_get_arraybuffer_info(env: napi_env, value: napi_value, data: *mut *mut c_void, len: *mut usize) -> napi_status;
    pub fn napi_is_typedarray(env: napi_env, value: napi_value, out: *mut bool) -> napi_status;
    pub fn napi_get_typedarray_info(env: napi_env, value: napi_value, kind: *mut c_int, len: *mut usize, data: *mut *mut c_void, buffer: *mut napi_value, offset: *mut usize) -> napi_status;
}

/// The property of a module's exports that holds the XComponent which loaded the library.
pub const OH_NATIVE_XCOMPONENT_OBJ: &std::ffi::CStr = c"__NATIVE_XCOMPONENT_OBJ__";
pub const OH_NATIVE_XCOMPONENT_MAX_TOUCH_POINTS_NUMBER: usize = 10;

pub mod touch_type {
    pub const DOWN: i32 = 0;
    pub const UP: i32 = 1;
    pub const MOVE: i32 = 2;
    pub const CANCEL: i32 = 3;
}

#[repr(C)]
pub struct OH_NativeXComponent { _opaque: [u8; 0] }

#[repr(C)]
pub struct OH_NativeXComponent_Callback {
    pub OnSurfaceCreated: Option<unsafe extern "C" fn(component: *mut OH_NativeXComponent, window: *mut c_void)>,
    pub OnSurfaceChanged: Option<unsafe extern "C" fn(component: *mut OH_NativeXComponent, window: *mut c_void)>,
    pub OnSurfaceDestroyed: Option<unsafe extern "C" fn(component: *mut OH_NativeXComponent, window: *mut c_void)>,
    pub DispatchTouchEvent: Option<unsafe extern "C" fn(component: *mut OH_NativeXComponent, window: *mut c_void)>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OH_NativeXComponent_TouchPoint {
    pub id: i32,
    pub screenX: f32,
    pub screenY: f32,
    pub x: f32,
    pub y: f32,
    pub type_: i32,
    pub size: f64,
    pub force: f32,
    pub timeStamp: i64,
    pub isPressed: bool,
}

#[repr(C)]
pub struct OH_NativeXComponent_TouchEvent {
    pub id: i32,
    pub screenX: f32,
    pub screenY: f32,
    pub x: f32,
    pub y: f32,
    pub type_: i32,
    pub size: f64,
    pub force: f32,
    pub deviceId: i64,
    pub timeStamp: i64,
    pub touchPoints: [OH_NativeXComponent_TouchPoint; OH_NATIVE_XCOMPONENT_MAX_TOUCH_POINTS_NUMBER],
    pub numPoints: u32,
}

#[link(name = "ace_ndk.z")]
extern "C" {
    pub fn OH_NativeXComponent_RegisterCallback(component: *mut OH_NativeXComponent, callback: *mut OH_NativeXComponent_Callback) -> i32;
    pub fn OH_NativeXComponent_GetXComponentSize(component: *mut OH_NativeXComponent, window: *const c_void, width: *mut u64, height: *mut u64) -> i32;
    pub fn OH_NativeXComponent_GetTouchEvent(component: *mut OH_NativeXComponent, window: *const c_void, event: *mut OH_NativeXComponent_TouchEvent) -> i32;
}

#[link(name = "native_window")]
extern "C" {
    pub fn OH_NativeWindow_NativeObjectReference(object: *mut c_void) -> i32;
    pub fn OH_NativeWindow_NativeObjectUnreference(object: *mut c_void) -> i32;
}

#[link(name = "native_vsync")]
extern "C" {
    pub fn OH_NativeVSync_Create(name: *const c_char, length: c_uint) -> *mut c_void;
    pub fn OH_NativeVSync_RequestFrame(vsync: *mut c_void, callback: unsafe extern "C" fn(timestamp: i64, data: *mut c_void), data: *mut c_void) -> c_int;
    pub fn OH_NativeVSync_GetPeriod(vsync: *mut c_void, period: *mut i64) -> c_int;
}

/// A raw file's place in the HAP: an open descriptor and the file's byte range in it.
#[repr(C)]
pub struct RawFileDescriptor64 {
    pub fd: c_int,
    pub start: i64,
    pub length: i64,
}

#[link(name = "rawfile.z")]
extern "C" {
    pub fn OH_ResourceManager_InitNativeResourceManager(env: napi_env, manager: napi_value) -> *mut c_void;
    pub fn OH_ResourceManager_OpenRawFile64(manager: *const c_void, name: *const c_char) -> *mut c_void;
    pub fn OH_ResourceManager_GetRawFileSize64(file: *mut c_void) -> i64;
    pub fn OH_ResourceManager_ReadRawFile64(file: *const c_void, out: *mut c_void, length: i64) -> i64;
    pub fn OH_ResourceManager_GetRawFileDescriptor64(file: *const c_void, descriptor: *mut RawFileDescriptor64) -> bool;
    pub fn OH_ResourceManager_CloseRawFile64(file: *mut c_void);
}

/// The editor the input method edits: its callbacks are the shell's (`ime.rs`).
#[repr(C)]
pub struct InputMethod_TextEditorProxy { _opaque: [u8; 0] }
/// The input method, while the editor is attached to it.
#[repr(C)]
pub struct InputMethod_InputMethodProxy { _opaque: [u8; 0] }
#[repr(C)]
pub struct InputMethod_TextConfig { _opaque: [u8; 0] }
#[repr(C)]
pub struct InputMethod_AttachOptions { _opaque: [u8; 0] }

/// `IME_ERR_OK`
pub const IME_ERR_OK: c_int = 0;
/// `InputMethod_KeyboardStatus`
pub const IME_KEYBOARD_STATUS_HIDE: c_int = 1;
/// `InputMethod_EnterKeyType`
pub mod ime_enter_key {
    pub const UNSPECIFIED: i32 = 0;
    pub const NONE: i32 = 1;
    pub const GO: i32 = 2;
    pub const SEARCH: i32 = 3;
    pub const SEND: i32 = 4;
    pub const NEXT: i32 = 5;
    pub const DONE: i32 = 6;
    pub const PREVIOUS: i32 = 7;
    pub const NEWLINE: i32 = 8;
}
/// `InputMethod_TextInputType`
pub mod ime_text_input {
    pub const TEXT: i32 = 0;
    pub const MULTILINE: i32 = 1;
    pub const NUMBER: i32 = 2;
    pub const PHONE: i32 = 3;
    pub const EMAIL_ADDRESS: i32 = 5;
    pub const URL: i32 = 6;
    pub const VISIBLE_PASSWORD: i32 = 7;
    pub const NUMBER_PASSWORD: i32 = 8;
    pub const NUMBER_DECIMAL: i32 = 12;
}
/// `InputMethod_Direction`
pub mod ime_direction {
    pub const UP: i32 = 1;
    pub const DOWN: i32 = 2;
    pub const LEFT: i32 = 3;
    pub const RIGHT: i32 = 4;
}
/// `InputMethod_ExtendAction`
pub mod ime_extend_action {
    pub const SELECT_ALL: i32 = 0;
    pub const CUT: i32 = 3;
    pub const COPY: i32 = 4;
}

pub type ime_editor = *mut InputMethod_TextEditorProxy;
#[link(name = "ohinputmethod")]
extern "C" {
    pub fn OH_TextEditorProxy_Create() -> ime_editor;
    pub fn OH_TextEditorProxy_SetGetTextConfigFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor, *mut InputMethod_TextConfig)) -> c_int;
    pub fn OH_TextEditorProxy_SetInsertTextFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor, *const u16, usize)) -> c_int;
    pub fn OH_TextEditorProxy_SetDeleteForwardFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor, i32)) -> c_int;
    pub fn OH_TextEditorProxy_SetDeleteBackwardFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor, i32)) -> c_int;
    pub fn OH_TextEditorProxy_SetSendKeyboardStatusFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor, c_int)) -> c_int;
    pub fn OH_TextEditorProxy_SetSendEnterKeyFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor, c_int)) -> c_int;
    pub fn OH_TextEditorProxy_SetMoveCursorFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor, c_int)) -> c_int;
    pub fn OH_TextEditorProxy_SetHandleSetSelectionFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor, i32, i32)) -> c_int;
    pub fn OH_TextEditorProxy_SetHandleExtendActionFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor, c_int)) -> c_int;
    pub fn OH_TextEditorProxy_SetGetLeftTextOfCursorFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor, i32, *mut u16, *mut usize)) -> c_int;
    pub fn OH_TextEditorProxy_SetGetRightTextOfCursorFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor, i32, *mut u16, *mut usize)) -> c_int;
    pub fn OH_TextEditorProxy_SetGetTextIndexAtCursorFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor) -> i32) -> c_int;
    pub fn OH_TextEditorProxy_SetReceivePrivateCommandFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor, *mut *mut c_void, usize) -> i32) -> c_int;
    pub fn OH_TextEditorProxy_SetSetPreviewTextFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor, *const u16, usize, i32, i32) -> i32) -> c_int;
    pub fn OH_TextEditorProxy_SetFinishTextPreviewFunc(editor: ime_editor, f: unsafe extern "C" fn(ime_editor)) -> c_int;
    pub fn OH_TextConfig_SetInputType(config: *mut InputMethod_TextConfig, kind: c_int) -> c_int;
    pub fn OH_TextConfig_SetEnterKeyType(config: *mut InputMethod_TextConfig, kind: c_int) -> c_int;
    pub fn OH_TextConfig_SetPreviewTextSupport(config: *mut InputMethod_TextConfig, supported: bool) -> c_int;
    pub fn OH_TextConfig_SetSelection(config: *mut InputMethod_TextConfig, start: i32, end: i32) -> c_int;
    pub fn OH_TextConfig_SetWindowId(config: *mut InputMethod_TextConfig, window: i32) -> c_int;
    pub fn OH_AttachOptions_Create(show_keyboard: bool) -> *mut InputMethod_AttachOptions;
    pub fn OH_AttachOptions_Destroy(options: *mut InputMethod_AttachOptions);
    pub fn OH_InputMethodController_Attach(editor: ime_editor, options: *mut InputMethod_AttachOptions, out: *mut *mut InputMethod_InputMethodProxy) -> c_int;
    pub fn OH_InputMethodController_Detach(input_method: *mut InputMethod_InputMethodProxy) -> c_int;
    pub fn OH_InputMethodProxy_ShowKeyboard(input_method: *mut InputMethod_InputMethodProxy) -> c_int;
    pub fn OH_InputMethodProxy_HideKeyboard(input_method: *mut InputMethod_InputMethodProxy) -> c_int;
    pub fn OH_InputMethodProxy_NotifySelectionChange(input_method: *mut InputMethod_InputMethodProxy, text: *mut u16, len: usize, start: c_int, end: c_int) -> c_int;
    pub fn OH_InputMethodProxy_NotifyConfigurationChange(input_method: *mut InputMethod_InputMethodProxy, enter_key: c_int, kind: c_int) -> c_int;
}

/// `LOG_APP`
pub const LOG_APP: c_int = 0;
/// `LOG_INFO`
pub const LOG_INFO: c_int = 4;

#[link(name = "hilog_ndk.z")]
extern "C" {
    pub fn OH_LOG_Print(kind: c_int, level: c_int, domain: c_uint, tag: *const c_char, format: *const c_char, ...) -> c_int;
}
