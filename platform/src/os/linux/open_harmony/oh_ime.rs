//! The input method through the NDK (`libohinputmethod`, API 12): no ArkTS.
//!
//! One text-editor proxy for the app; the input method's editing messages
//! arrive on its callbacks and become the same messages touch, keys and text
//! already use. Attach, show and hide run on the ArkTS main thread, in the
//! order makepad asks for them, so a hide can never be overtaken by an older
//! show (what `phone/ohos/keyboard.patch` used to serialize in ArkTS).
#![allow(non_camel_case_types)]

use super::oh_callbacks::{send_from_ohos_message, FromOhosMessage};
use super::oh_sys::{uv_loop_t, uv_queue_work, uv_work_t};
use crate::event::TextInputEvent;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::Mutex;

#[repr(C)]
pub struct InputMethod_TextEditorProxy {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct InputMethod_InputMethodProxy {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct InputMethod_TextConfig {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct InputMethod_AttachOptions {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct InputMethod_PrivateCommand {
    _unused: [u8; 0],
}

type Proxy = *mut InputMethod_TextEditorProxy;
const IME_ERR_OK: c_int = 0;
const IME_KEYBOARD_STATUS_HIDE: c_int = 1;
const IME_TEXT_INPUT_TYPE_TEXT: c_int = 0;
const IME_ENTER_KEY_NONE: c_int = 1;

#[link(name = "ohinputmethod")]
extern "C" {
    fn OH_TextEditorProxy_Create() -> Proxy;
    fn OH_TextEditorProxy_SetGetTextConfigFunc(p: Proxy, f: unsafe extern "C" fn(Proxy, *mut InputMethod_TextConfig)) -> c_int;
    fn OH_TextEditorProxy_SetInsertTextFunc(p: Proxy, f: unsafe extern "C" fn(Proxy, *const u16, usize)) -> c_int;
    fn OH_TextEditorProxy_SetDeleteForwardFunc(p: Proxy, f: unsafe extern "C" fn(Proxy, i32)) -> c_int;
    fn OH_TextEditorProxy_SetDeleteBackwardFunc(p: Proxy, f: unsafe extern "C" fn(Proxy, i32)) -> c_int;
    fn OH_TextEditorProxy_SetSendKeyboardStatusFunc(p: Proxy, f: unsafe extern "C" fn(Proxy, c_int)) -> c_int;
    fn OH_TextEditorProxy_SetSendEnterKeyFunc(p: Proxy, f: unsafe extern "C" fn(Proxy, c_int)) -> c_int;
    fn OH_TextEditorProxy_SetMoveCursorFunc(p: Proxy, f: unsafe extern "C" fn(Proxy, c_int)) -> c_int;
    fn OH_TextEditorProxy_SetHandleSetSelectionFunc(p: Proxy, f: unsafe extern "C" fn(Proxy, i32, i32)) -> c_int;
    fn OH_TextEditorProxy_SetHandleExtendActionFunc(p: Proxy, f: unsafe extern "C" fn(Proxy, c_int)) -> c_int;
    fn OH_TextEditorProxy_SetGetLeftTextOfCursorFunc(p: Proxy, f: unsafe extern "C" fn(Proxy, i32, *mut u16, *mut usize)) -> c_int;
    fn OH_TextEditorProxy_SetGetRightTextOfCursorFunc(p: Proxy, f: unsafe extern "C" fn(Proxy, i32, *mut u16, *mut usize)) -> c_int;
    fn OH_TextEditorProxy_SetGetTextIndexAtCursorFunc(p: Proxy, f: unsafe extern "C" fn(Proxy) -> i32) -> c_int;
    fn OH_TextEditorProxy_SetReceivePrivateCommandFunc(
        p: Proxy,
        f: unsafe extern "C" fn(Proxy, *mut *mut InputMethod_PrivateCommand, usize) -> i32,
    ) -> c_int;
    fn OH_TextEditorProxy_SetSetPreviewTextFunc(p: Proxy, f: unsafe extern "C" fn(Proxy, *const u16, usize, i32, i32) -> i32) -> c_int;
    fn OH_TextEditorProxy_SetFinishTextPreviewFunc(p: Proxy, f: unsafe extern "C" fn(Proxy)) -> c_int;
    fn OH_TextConfig_SetInputType(config: *mut InputMethod_TextConfig, input_type: c_int) -> c_int;
    fn OH_TextConfig_SetEnterKeyType(config: *mut InputMethod_TextConfig, enter_key_type: c_int) -> c_int;
    fn OH_AttachOptions_Create(show_keyboard: bool) -> *mut InputMethod_AttachOptions;
    fn OH_AttachOptions_Destroy(options: *mut InputMethod_AttachOptions);
    fn OH_InputMethodController_Attach(
        editor: Proxy,
        options: *mut InputMethod_AttachOptions,
        ime: *mut *mut InputMethod_InputMethodProxy,
    ) -> c_int;
    /// API 23.
    fn OH_InputMethodController_AttachWithUIContext(
        context: *mut c_void,
        editor: Proxy,
        options: *mut InputMethod_AttachOptions,
        ime: *mut *mut InputMethod_InputMethodProxy,
    ) -> c_int;
    fn OH_InputMethodController_Detach(ime: *mut InputMethod_InputMethodProxy) -> c_int;
    fn OH_InputMethodProxy_ShowKeyboard(ime: *mut InputMethod_InputMethodProxy) -> c_int;
    fn OH_InputMethodProxy_HideKeyboard(ime: *mut InputMethod_InputMethodProxy) -> c_int;
}

struct Ime {
    editor: Proxy,
    ime: *mut InputMethod_InputMethodProxy,
    uv_loop: *mut uv_loop_t,
    /// The XComponent node's UI context (`ArkUI_ContextHandle`): attaching
    /// needs it to know which window is editing.
    context: *mut c_void,
}
// Touched only on the ArkTS main thread (via `on_main`), behind the mutex.
unsafe impl Send for Ime {}

static IME: Mutex<Option<Ime>> = Mutex::new(None);

/// Called once at startup with the main thread's event loop.
pub fn init(uv_loop: *mut uv_loop_t) {
    let editor = unsafe { OH_TextEditorProxy_Create() };
    if editor.is_null() {
        crate::error!("ime: OH_TextEditorProxy_Create failed");
        return;
    }
    unsafe {
        OH_TextEditorProxy_SetGetTextConfigFunc(editor, get_text_config);
        OH_TextEditorProxy_SetInsertTextFunc(editor, insert_text);
        OH_TextEditorProxy_SetDeleteForwardFunc(editor, delete_forward);
        OH_TextEditorProxy_SetDeleteBackwardFunc(editor, delete_backward);
        OH_TextEditorProxy_SetSendKeyboardStatusFunc(editor, send_keyboard_status);
        OH_TextEditorProxy_SetSendEnterKeyFunc(editor, send_enter_key);
        OH_TextEditorProxy_SetMoveCursorFunc(editor, move_cursor);
        OH_TextEditorProxy_SetHandleSetSelectionFunc(editor, set_selection);
        OH_TextEditorProxy_SetHandleExtendActionFunc(editor, extend_action);
        OH_TextEditorProxy_SetGetLeftTextOfCursorFunc(editor, no_text);
        OH_TextEditorProxy_SetGetRightTextOfCursorFunc(editor, no_text);
        OH_TextEditorProxy_SetGetTextIndexAtCursorFunc(editor, text_index);
        OH_TextEditorProxy_SetReceivePrivateCommandFunc(editor, private_command);
        OH_TextEditorProxy_SetSetPreviewTextFunc(editor, preview_text);
        OH_TextEditorProxy_SetFinishTextPreviewFunc(editor, finish_preview);
    }
    *IME.lock().unwrap() = Some(Ime { editor, ime: std::ptr::null_mut(), uv_loop, context: std::ptr::null_mut() });
}

/// The UI context of makepad's XComponent node, once it is mounted.
pub fn set_context(context: *mut c_void) {
    if let Some(ime) = IME.lock().unwrap().as_mut() {
        ime.context = context;
    }
}

/// Show the keyboard: attach the editor the first time, then show.
pub fn show() {
    on_main(|| {
        let mut guard = IME.lock().unwrap();
        let Some(ime) = guard.as_mut() else { return };
        if ime.ime.is_null() {
            let options = unsafe { OH_AttachOptions_Create(true) };
            let res = unsafe {
                if ime.context.is_null() {
                    OH_InputMethodController_Attach(ime.editor, options, &mut ime.ime)
                } else {
                    OH_InputMethodController_AttachWithUIContext(ime.context, ime.editor, options, &mut ime.ime)
                }
            };
            unsafe { OH_AttachOptions_Destroy(options) };
            if res != IME_ERR_OK {
                crate::error!("ime: attach failed: {res}");
                ime.ime = std::ptr::null_mut();
            }
        } else {
            let res = unsafe { OH_InputMethodProxy_ShowKeyboard(ime.ime) };
            if res != IME_ERR_OK {
                crate::error!("ime: show failed: {res}");
            }
        }
    });
}

/// Hide the keyboard and detach, so hardware keys stop going to the input
/// method while nothing is being edited.
pub fn hide() {
    on_main(|| {
        let mut guard = IME.lock().unwrap();
        let Some(ime) = guard.as_mut() else { return };
        if !ime.ime.is_null() {
            unsafe {
                OH_InputMethodProxy_HideKeyboard(ime.ime);
                OH_InputMethodController_Detach(ime.ime);
            }
            ime.ime = std::ptr::null_mut();
        }
    });
}

/// Run `f` on the ArkTS main thread, after everything queued before it.
/// Run `f` on the ArkTS main thread, after everything queued before it
/// (ArkWeb and the input method both need it).
pub fn run_on_main(f: impl FnOnce() + Send + 'static) {
    on_main(f)
}

fn on_main(f: impl FnOnce() + Send + 'static) {
    let uv_loop = match IME.lock().unwrap().as_ref() {
        Some(ime) => ime.uv_loop,
        None => return,
    };
    extern "C" fn work(_req: *mut uv_work_t) {}
    extern "C" fn after(req: *mut uv_work_t, _status: c_int) {
        let req = unsafe { Box::from_raw(req) };
        let f = unsafe { Box::from_raw(req.data as *mut Box<dyn FnOnce()>) };
        f();
    }
    let f: Box<Box<dyn FnOnce()>> = Box::new(Box::new(f));
    let mut req: Box<uv_work_t> = Box::new(unsafe { std::mem::zeroed() });
    req.data = Box::into_raw(f) as *mut c_void;
    let req = Box::into_raw(req);
    if unsafe { uv_queue_work(uv_loop, req, Some(work), Some(after)) } != 0 {
        crate::error!("ime: uv_queue_work failed");
        unsafe {
            drop(Box::from_raw((*req).data as *mut Box<dyn FnOnce()>));
            drop(Box::from_raw(req));
        }
    }
}

// ---- editor callbacks: OpenHarmony key codes as in `oh_key_code` ----

unsafe extern "C" fn get_text_config(_p: Proxy, config: *mut InputMethod_TextConfig) {
    OH_TextConfig_SetInputType(config, IME_TEXT_INPUT_TYPE_TEXT);
    OH_TextConfig_SetEnterKeyType(config, IME_ENTER_KEY_NONE);
}

unsafe extern "C" fn insert_text(_p: Proxy, text: *const u16, length: usize) {
    if text.is_null() || length == 0 {
        return;
    }
    let input = String::from_utf16_lossy(std::slice::from_raw_parts(text, length));
    send_from_ohos_message(FromOhosMessage::TextInput(TextInputEvent {
        input,
        replace_last: false,
        was_paste: false,
        ..Default::default()
    }));
}

unsafe extern "C" fn delete_backward(_p: Proxy, length: i32) {
    send_from_ohos_message(FromOhosMessage::DeleteLeft(length));
}

unsafe extern "C" fn delete_forward(_p: Proxy, length: i32) {
    for _ in 0..length.max(0) {
        send_from_ohos_message(FromOhosMessage::ImeKey { code: 2071, modifiers: 0 });
    }
}

unsafe extern "C" fn send_keyboard_status(_p: Proxy, status: c_int) {
    if status == IME_KEYBOARD_STATUS_HIDE {
        send_from_ohos_message(FromOhosMessage::ResizeTextIME(false, 0));
    }
}

unsafe extern "C" fn send_enter_key(_p: Proxy, _enter_key_type: c_int) {
    send_from_ohos_message(FromOhosMessage::ImeKey { code: 2054, modifiers: 0 });
}

unsafe extern "C" fn move_cursor(_p: Proxy, direction: c_int) {
    // IME_DIRECTION_UP 1, DOWN 2, LEFT 3, RIGHT 4
    let code = match direction {
        1 => 2012,
        2 => 2013,
        3 => 2014,
        4 => 2015,
        _ => return,
    };
    send_from_ohos_message(FromOhosMessage::ImeKey { code, modifiers: 0 });
}

unsafe extern "C" fn extend_action(_p: Proxy, action: c_int) {
    // SELECT_ALL 0, CUT 3, COPY 4, PASTE 5: Ctrl+A, X, C, V
    let code = match action {
        0 => 2017,
        3 => 2040,
        4 => 2019,
        5 => 2038,
        _ => return,
    };
    send_from_ohos_message(FromOhosMessage::ImeKey { code, modifiers: 1 });
}

unsafe extern "C" fn set_selection(_p: Proxy, _start: i32, _end: i32) {}

/// makepad's editors keep their own text; the input method sees none of it.
unsafe extern "C" fn no_text(_p: Proxy, _number: i32, _text: *mut u16, length: *mut usize) {
    if !length.is_null() {
        *length = 0;
    }
}

unsafe extern "C" fn text_index(_p: Proxy) -> i32 {
    0
}

unsafe extern "C" fn private_command(_p: Proxy, _commands: *mut *mut InputMethod_PrivateCommand, _size: usize) -> i32 {
    0
}

unsafe extern "C" fn preview_text(_p: Proxy, _text: *const u16, _length: usize, _start: i32, _end: i32) -> i32 {
    0
}

unsafe extern "C" fn finish_preview(_p: Proxy) {}
