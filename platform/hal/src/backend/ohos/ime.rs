//! The soft keyboard. The system's input method edits an editor through callbacks; the editor
//! here is a copy of the focused field (its text, selection and composition) which the app
//! sends with `ImeSync`. Each edit changes the copy and goes to the app as the field's full
//! state, the way Android's editor reports. The input method is spoken to on the UI thread
//! (`napi::on_ui`), as ArkUI's own text fields do; its callbacks come on a thread of its own.
use super::{napi, shell::{lock, Shared}, sys::*};
use crate::{backend::exec::OsCmd, clock_ns, input::{self, ime_config as cfg}};
use makepad_key_code::KeyCode;
use std::sync::{atomic::{AtomicI32, AtomicUsize, Ordering}, Mutex};

/// Offsets are UTF-16 code units, the input method's own.
struct Editor {
    text: Vec<u16>,
    /// start <= end
    selection: [usize; 2],
    /// the preview text's range: what the input method is still composing
    composition: Option<[usize; 2]>,
    /// the field's `ImeShow::config`
    config: u32,
}
static EDITOR: Mutex<Editor> = Mutex::new(Editor { text: Vec::new(), selection: [0; 2], composition: None, config: 0 });
/// The config of the keyboard last asked for, while it is up: a focused field asks again at
/// each draw.
static SHOWN: Mutex<Option<u32>> = Mutex::new(None);
/// The editor proxy with the callbacks below (made once), and the input method's proxy while
/// the editor is attached. Both are the UI thread's.
static PROXY: AtomicUsize = AtomicUsize::new(0);
static ATTACHED: AtomicUsize = AtomicUsize::new(0);
static WINDOW: AtomicI32 = AtomicI32::new(0);

pub(super) fn set_window(id: i32) { WINDOW.store(id, Ordering::Release); }
/// The keyboard went down (the window's keyboard height), so the next show is a new one.
pub(super) fn closed() { *lock(&SHOWN) = None; }
/// Back while the keyboard is up puts the keyboard away and goes no further, as it does for
/// the system's own text fields; the field keeps its focus and its editor. On the UI thread.
pub(super) unsafe fn back() -> bool {
    let attached = ATTACHED.load(Ordering::Acquire) as *mut InputMethod_InputMethodProxy;
    let up = lock(&Shared::global().state).ime.is_some_and(|(_, open)| open);
    up && !attached.is_null() && OH_InputMethodProxy_HideKeyboard(attached) == IME_ERR_OK
}

fn high(unit: u16) -> bool { (0xD800..0xDC00).contains(&unit) }
fn low(unit: u16) -> bool { (0xDC00..0xE000).contains(&unit) }
impl Editor {
    /// `at`, or the unit before it when `at` is between a surrogate pair's halves.
    fn floor(&self, at: usize) -> usize {
        if at > 0 && at < self.text.len() && low(self.text[at]) && high(self.text[at - 1]) { at - 1 } else { at }
    }
    fn ceil(&self, at: usize) -> usize { if self.floor(at) != at { at + 1 } else { at } }
    /// `range` becomes `text`, with the caret after it.
    fn replace(&mut self, range: [usize; 2], text: &[u16]) -> usize {
        let end = range[1].min(self.text.len());
        let start = range[0].min(end);
        self.text.splice(start..end, text.iter().copied());
        self.selection = [start + text.len(); 2];
        start + text.len()
    }
    fn insert(&mut self, text: &[u16]) {
        let range = self.composition.take().unwrap_or(self.selection);
        self.replace(range, text);
    }
    /// The selection, or else `count` units before (`back`) or after the caret.
    fn delete(&mut self, count: usize, back: bool) {
        self.composition = None;
        let [start, end] = self.selection;
        let range = if start != end { [start, end] }
            else if back { [self.floor(start.saturating_sub(count)), start] }
            else { [start, self.ceil((start + count).min(self.text.len()))] };
        self.replace(range, &[]);
    }
    /// The text the input method is composing: it replaces `[start, end)` when that is a
    /// range of the text, else the composition so far, else the selection.
    fn preview(&mut self, text: &[u16], start: i32, end: i32) {
        let range = if start >= 0 && start <= end && end as usize <= self.text.len() { [start as usize, end as usize] }
            else { self.composition.unwrap_or(self.selection) };
        let caret = self.replace(range, text);
        self.composition = (!text.is_empty()).then_some([caret - text.len(), caret]);
    }
}

/// The input method's keyboard for a field's `ImeShow::config` (none for a field that takes no
/// typing), and its enter key.
fn keyboard(config: u32) -> (Option<i32>, i32) {
    let multiline = config & cfg::MULTILINE != 0;
    let secure = config & cfg::SECURE != 0;
    // None Text Ascii Url Numeric Tel Email Decimal Search
    let kind = match cfg::field(config, cfg::INPUT_MODE_SHIFT, 4) {
        _ if config & cfg::READ_ONLY != 0 => None,
        4 | 7 if secure => Some(ime_text_input::NUMBER_PASSWORD),
        1..=8 if secure => Some(ime_text_input::VISIBLE_PASSWORD),
        3 => Some(ime_text_input::URL),
        4 => Some(ime_text_input::NUMBER),
        5 => Some(ime_text_input::PHONE),
        6 => Some(ime_text_input::EMAIL_ADDRESS),
        7 => Some(ime_text_input::NUMBER_DECIMAL),
        1 | 2 | 8 => Some(if multiline { ime_text_input::MULTILINE } else { ime_text_input::TEXT }),
        _ => None,
    };
    // Default None Go Google Join Next Route Search Send Yahoo Done EmergencyCall Continue Previous,
    // each as the closest enter key the input method has
    use ime_enter_key::*;
    let enter = match cfg::field(config, cfg::RETURN_KEY_SHIFT, 4) {
        0 if multiline && config & cfg::SUBMIT_ON_ENTER == 0 => NEWLINE,
        1 => NONE,
        2 | 4 | 6 | 11 => GO,
        3 | 7 | 9 => SEARCH,
        8 => SEND,
        5 | 12 => NEXT,
        10 => DONE,
        13 => PREVIOUS,
        _ => UNSPECIFIED,
    };
    (kind, enter)
}

/// The shell's `ImeShow`, `ImeHide` and `ImeSync`.
pub(super) fn command(cmd: OsCmd) -> bool {
    match cmd {
        OsCmd::ImeShow { config, .. } => {
            {
                let mut shown = lock(&SHOWN);
                if *shown == Some(config) { return true; }
                *shown = Some(config);
            }
            lock(&EDITOR).config = config;
            napi::on_ui(move || unsafe { show(config) })
        }
        OsCmd::ImeHide { .. } => {
            *lock(&SHOWN) = None;
            napi::on_ui(|| unsafe { hide() })
        }
        OsCmd::ImeSync { selection, composition, text, .. } => {
            // Cx commands use character offsets; the input method's editor uses UTF-16 units.
            let units = |n: u32| text.chars().take(n as usize).map(char::len_utf16).sum::<usize>();
            let range = |r: [u32; 2]| { let (a, b) = (units(r[0]), units(r[1])); [a.min(b), a.max(b)] };
            {
                let mut e = lock(&EDITOR);
                e.selection = range(selection);
                e.composition = (composition[0] != composition[1]).then(|| range(composition));
                e.text = text.encode_utf16().collect();
            }
            napi::on_ui(|| unsafe { notify() })
        }
        _ => false,
    }
}

/// The editor proxy, made at the first show.
unsafe fn proxy() -> ime_editor {
    let made = PROXY.load(Ordering::Acquire) as ime_editor;
    if !made.is_null() { return made; }
    let p = OH_TextEditorProxy_Create();
    if p.is_null() { return p; }
    // the input method takes an editor only with every callback set
    OH_TextEditorProxy_SetGetTextConfigFunc(p, text_config);
    OH_TextEditorProxy_SetInsertTextFunc(p, insert_text);
    OH_TextEditorProxy_SetDeleteForwardFunc(p, delete_forward);
    OH_TextEditorProxy_SetDeleteBackwardFunc(p, delete_backward);
    OH_TextEditorProxy_SetSendKeyboardStatusFunc(p, keyboard_status);
    OH_TextEditorProxy_SetSendEnterKeyFunc(p, enter_key);
    OH_TextEditorProxy_SetMoveCursorFunc(p, move_cursor);
    OH_TextEditorProxy_SetHandleSetSelectionFunc(p, set_selection);
    OH_TextEditorProxy_SetHandleExtendActionFunc(p, extend_action);
    OH_TextEditorProxy_SetGetLeftTextOfCursorFunc(p, left_text);
    OH_TextEditorProxy_SetGetRightTextOfCursorFunc(p, right_text);
    OH_TextEditorProxy_SetGetTextIndexAtCursorFunc(p, index_at_cursor);
    OH_TextEditorProxy_SetReceivePrivateCommandFunc(p, private_command);
    OH_TextEditorProxy_SetSetPreviewTextFunc(p, preview_text);
    OH_TextEditorProxy_SetFinishTextPreviewFunc(p, finish_preview);
    PROXY.store(p as usize, Ordering::Release);
    p
}
unsafe fn show(config: u32) {
    let (Some(kind), enter) = keyboard(config) else { return hide() };
    let attached = ATTACHED.load(Ordering::Acquire) as *mut InputMethod_InputMethodProxy;
    if !attached.is_null() {
        // the same editor, for another field or after the keyboard was put away
        OH_InputMethodProxy_NotifyConfigurationChange(attached, enter, kind);
        if OH_InputMethodProxy_ShowKeyboard(attached) == IME_ERR_OK { return notify(); }
        // the system let go of the editor meanwhile (the app was in the background)
        ATTACHED.store(0, Ordering::Release);
    }
    let editor = proxy();
    let options = OH_AttachOptions_Create(true);
    let mut out = std::ptr::null_mut();
    let code = if editor.is_null() || options.is_null() { -1 } else { OH_InputMethodController_Attach(editor, options, &mut out) };
    if !options.is_null() { OH_AttachOptions_Destroy(options); }
    if crate::backend::log_enabled() { eprintln!("[hal-ohos] keyboard: attach {code}, input type {kind}, enter key {enter}"); }
    if code != IME_ERR_OK || out.is_null() {
        eprintln!("[hal-ohos] the input method refused the editor: {code}");
        *lock(&SHOWN) = None;
        return;
    }
    ATTACHED.store(out as usize, Ordering::Release);
    notify();
}
unsafe fn hide() {
    let attached = ATTACHED.swap(0, Ordering::AcqRel) as *mut InputMethod_InputMethodProxy;
    if !attached.is_null() {
        let code = OH_InputMethodController_Detach(attached);
        if crate::backend::log_enabled() { eprintln!("[hal-ohos] keyboard: detach {code}"); }
    }
}
/// Tells the input method the field's text and selection as they are now.
unsafe fn notify() {
    let attached = ATTACHED.load(Ordering::Acquire) as *mut InputMethod_InputMethodProxy;
    if attached.is_null() { return; }
    let (mut text, selection) = { let e = lock(&EDITOR); (e.text.clone(), e.selection) };
    OH_InputMethodProxy_NotifySelectionChange(attached, text.as_mut_ptr(), text.len(), selection[0] as i32, selection[1] as i32);
}

/// One edit by the input method: the field's new state for the app, and for the input method.
fn edit(what: &str, f: impl FnOnce(&mut Editor)) {
    let (text, selection, composition) = {
        let mut e = lock(&EDITOR);
        f(&mut e);
        let len = e.text.len();
        e.selection = [e.selection[0].min(len), e.selection[1].min(len)];
        (String::from_utf16_lossy(&e.text), e.selection, e.composition)
    };
    if crate::backend::log_enabled() { eprintln!("[hal-ohos] keyboard: {what}, {} units, selection {selection:?}, composing {composition:?}", text.encode_utf16().count()); }
    let [cs, ce] = composition.map_or([u32::MAX; 2], |c| c.map(|at| at as u32));
    Shared::global().emit(clock_ns(), |q, id| q.ime_state(id, selection[0] as u32, selection[1] as u32, cs, ce, text.as_bytes()));
    napi::on_ui(|| unsafe { notify() });
}
/// A key the field answers itself, as from a hardware keyboard.
fn key(code: KeyCode, mods: u32) {
    let code = code.to_index();
    Shared::global().emit(clock_ns(), |q, id| { q.key_down(id, code, false, mods); q.key_up(id, code, mods); });
}
unsafe fn units<'a>(text: *const u16, len: usize) -> &'a [u16] {
    if text.is_null() { &[] } else { std::slice::from_raw_parts(text, len) }
}

unsafe extern "C" fn text_config(_: ime_editor, config: *mut InputMethod_TextConfig) {
    let (kind, enter, selection) = { let e = lock(&EDITOR); let (kind, enter) = keyboard(e.config); (kind, enter, e.selection) };
    OH_TextConfig_SetInputType(config, kind.unwrap_or(ime_text_input::TEXT));
    OH_TextConfig_SetEnterKeyType(config, enter);
    OH_TextConfig_SetPreviewTextSupport(config, true);
    OH_TextConfig_SetSelection(config, selection[0] as i32, selection[1] as i32);
    let window = WINDOW.load(Ordering::Acquire);
    if window != 0 { OH_TextConfig_SetWindowId(config, window); }
}
unsafe extern "C" fn insert_text(_: ime_editor, text: *const u16, len: usize) {
    let text = units(text, len).to_vec();
    edit("insert", |e| e.insert(&text));
}
unsafe extern "C" fn delete_forward(_: ime_editor, count: i32) { edit("delete forward", |e| e.delete(count.max(0) as usize, false)); }
unsafe extern "C" fn delete_backward(_: ime_editor, count: i32) { edit("delete backward", |e| e.delete(count.max(0) as usize, true)); }
unsafe extern "C" fn keyboard_status(_: ime_editor, status: i32) {
    if crate::backend::log_enabled() { eprintln!("[hal-ohos] keyboard: status {status}"); }
    if status != IME_KEYBOARD_STATUS_HIDE { return; }
    closed();
    // The app hears it here too: a keyboard the system put away before it rose (a show as the
    // app comes back to the front) never had a height to report, and the app would go on
    // taking it for shown.
    lock(&Shared::global().state).ime = Some((0, false));
    Shared::global().emit(clock_ns(), |q, id| q.ime_resize(id, 0., false));
}
unsafe extern "C" fn enter_key(_: ime_editor, kind: i32) {
    if crate::backend::log_enabled() { eprintln!("[hal-ohos] keyboard: enter key {kind}"); }
    match kind {
        // the numbers of Android's editor actions, which the app reads them as
        ime_enter_key::GO..=ime_enter_key::PREVIOUS => Shared::global().emit(clock_ns(), |q, id| q.ime_action(id, kind as u32)),
        // a new line, or what the field does with Return
        _ => key(KeyCode::ReturnKey, 0),
    }
}
unsafe extern "C" fn move_cursor(_: ime_editor, direction: i32) {
    match direction {
        ime_direction::UP => key(KeyCode::ArrowUp, 0),
        ime_direction::DOWN => key(KeyCode::ArrowDown, 0),
        ime_direction::LEFT => key(KeyCode::ArrowLeft, 0),
        ime_direction::RIGHT => key(KeyCode::ArrowRight, 0),
        _ => {}
    }
}
unsafe extern "C" fn set_selection(_: ime_editor, start: i32, end: i32) {
    edit("selection", |e| {
        let (a, b) = (start.max(0) as usize, end.max(0) as usize);
        e.selection = [a.min(b), a.max(b)];
    });
}
/// The chords the app already answers. (Paste needs the pasteboard, which the shell does not read.)
unsafe extern "C" fn extend_action(_: ime_editor, action: i32) {
    match action {
        ime_extend_action::SELECT_ALL => key(KeyCode::KeyA, input::MOD_CONTROL),
        ime_extend_action::CUT => key(KeyCode::KeyX, input::MOD_CONTROL),
        ime_extend_action::COPY => key(KeyCode::KeyC, input::MOD_CONTROL),
        _ => {}
    }
}
/// Up to `count` units before the caret; `text` has room for them.
unsafe extern "C" fn left_text(_: ime_editor, count: i32, text: *mut u16, len: *mut usize) {
    if text.is_null() || len.is_null() { return; }
    let e = lock(&EDITOR);
    let caret = e.selection[0].min(e.text.len());
    let from = e.ceil(caret.saturating_sub(count.max(0) as usize)).min(caret);
    std::ptr::copy_nonoverlapping(e.text[from..caret].as_ptr(), text, caret - from);
    *len = caret - from;
}
unsafe extern "C" fn right_text(_: ime_editor, count: i32, text: *mut u16, len: *mut usize) {
    if text.is_null() || len.is_null() { return; }
    let e = lock(&EDITOR);
    let caret = e.selection[1].min(e.text.len());
    let to = e.floor((caret + count.max(0) as usize).min(e.text.len())).max(caret);
    std::ptr::copy_nonoverlapping(e.text[caret..to].as_ptr(), text, to - caret);
    *len = to - caret;
}
unsafe extern "C" fn index_at_cursor(_: ime_editor) -> i32 { lock(&EDITOR).selection[1] as i32 }
unsafe extern "C" fn private_command(_: ime_editor, _: *mut *mut std::ffi::c_void, _: usize) -> i32 { 0 }
unsafe extern "C" fn preview_text(_: ime_editor, text: *const u16, len: usize, start: i32, end: i32) -> i32 {
    let text = units(text, len).to_vec();
    edit("preview", |e| e.preview(&text, start, end));
    0
}
/// (The input method also says this as it lets go of the editor, with nothing composed.)
unsafe extern "C" fn finish_preview(_: ime_editor) {
    if lock(&EDITOR).composition.is_some() { edit("preview done", |e| e.composition = None); }
}
