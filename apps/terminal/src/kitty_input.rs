//! Frontend half of the kitty keyboard protocol: which Makepad key and text
//! events become which encoder events. The encoder (`term::key_encode`)
//! turns one normalized key event into bytes; this module decides what those
//! events are, which matters once a program asks for more than legacy keys.
//!
//! Written against the protocol spec
//! (https://sw.kovidgoyal.net/kitty/keyboard-protocol/):
//!
//!   * Report event types (2): key releases, and repeats, are reported. The
//!     widget sends a release on `KeyUp` and a repeat on an autorepeat
//!     `KeyDown`. Text keys still type text on press and repeat (only their
//!     release is an escape code) unless flag 8 is set.
//!   * Report all keys as escape codes (8): "text will not be sent, instead
//!     only key events are sent". A text key goes through the encoder as a
//!     key event; its text is only embedded when flag 16 asks for it.
//!     Modifier keys report their own presses and releases.
//!   * Text no key produced (an IME commit, a dead-key composition) is not a
//!     key event. With 8 and 16 it is reported against key number 0, as the
//!     spec asks; otherwise it goes out as UTF-8 so input methods keep
//!     working.
//!
//! Makepad delivers a typed character as two events whose order depends on
//! the platform: macOS sends the text (from the input context) before the
//! key-down, X11 and Windows send the key-down first. Under flag 8 they must
//! become ONE key event, so [`TextKeyPairing`] holds a text briefly for its
//! key-down (macOS order) or drops the text that follows a key-down it
//! already encoded (the other order). Text that finds no key-down is IME
//! text. None of this runs with flag 8 clear: legacy and lower-flag typing
//! write the text straight through, byte-identical to before.

use makepad_widgets::{KeyCode, KeyModifiers};

use crate::term::key_encode::{Key, KeyAction, KeyEvent, KeyMods, KittyFlags};

/// How long, in seconds, a text waits for the key-down that produced it,
/// and a key-down for the text it made. Both halves arrive in the same
/// native event on every platform; this only bounds a stray.
pub const PAIRING_WINDOW: f64 = 0.05;

/// The encoder's key for a Makepad key code, for the keys the widget does
/// not already send as special keys: text keys, the numpad, modifiers and
/// lock keys. `None` for keys the kitty protocol has no code for.
pub fn kitty_key_of(kc: KeyCode) -> Option<Key> {
    Some(match kc {
        KeyCode::KeyA => Key::KeyA,
        KeyCode::KeyB => Key::KeyB,
        KeyCode::KeyC => Key::KeyC,
        KeyCode::KeyD => Key::KeyD,
        KeyCode::KeyE => Key::KeyE,
        KeyCode::KeyF => Key::KeyF,
        KeyCode::KeyG => Key::KeyG,
        KeyCode::KeyH => Key::KeyH,
        KeyCode::KeyI => Key::KeyI,
        KeyCode::KeyJ => Key::KeyJ,
        KeyCode::KeyK => Key::KeyK,
        KeyCode::KeyL => Key::KeyL,
        KeyCode::KeyM => Key::KeyM,
        KeyCode::KeyN => Key::KeyN,
        KeyCode::KeyO => Key::KeyO,
        KeyCode::KeyP => Key::KeyP,
        KeyCode::KeyQ => Key::KeyQ,
        KeyCode::KeyR => Key::KeyR,
        KeyCode::KeyS => Key::KeyS,
        KeyCode::KeyT => Key::KeyT,
        KeyCode::KeyU => Key::KeyU,
        KeyCode::KeyV => Key::KeyV,
        KeyCode::KeyW => Key::KeyW,
        KeyCode::KeyX => Key::KeyX,
        KeyCode::KeyY => Key::KeyY,
        KeyCode::KeyZ => Key::KeyZ,
        KeyCode::Key0 => Key::Digit0,
        KeyCode::Key1 => Key::Digit1,
        KeyCode::Key2 => Key::Digit2,
        KeyCode::Key3 => Key::Digit3,
        KeyCode::Key4 => Key::Digit4,
        KeyCode::Key5 => Key::Digit5,
        KeyCode::Key6 => Key::Digit6,
        KeyCode::Key7 => Key::Digit7,
        KeyCode::Key8 => Key::Digit8,
        KeyCode::Key9 => Key::Digit9,
        KeyCode::Backtick => Key::Backquote,
        KeyCode::Minus => Key::Minus,
        KeyCode::Equals => Key::Equal,
        KeyCode::LBracket => Key::BracketLeft,
        KeyCode::RBracket => Key::BracketRight,
        KeyCode::Backslash => Key::Backslash,
        KeyCode::Semicolon => Key::Semicolon,
        KeyCode::Quote => Key::Quote,
        KeyCode::Comma => Key::Comma,
        KeyCode::Period => Key::Period,
        KeyCode::Slash => Key::Slash,
        KeyCode::Space => Key::Space,
        KeyCode::Numpad0 => Key::Numpad0,
        KeyCode::Numpad1 => Key::Numpad1,
        KeyCode::Numpad2 => Key::Numpad2,
        KeyCode::Numpad3 => Key::Numpad3,
        KeyCode::Numpad4 => Key::Numpad4,
        KeyCode::Numpad5 => Key::Numpad5,
        KeyCode::Numpad6 => Key::Numpad6,
        KeyCode::Numpad7 => Key::Numpad7,
        KeyCode::Numpad8 => Key::Numpad8,
        KeyCode::Numpad9 => Key::Numpad9,
        KeyCode::NumpadDecimal => Key::NumpadDecimal,
        KeyCode::NumpadDivide => Key::NumpadDivide,
        KeyCode::NumpadMultiply => Key::NumpadMultiply,
        KeyCode::NumpadSubtract => Key::NumpadSubtract,
        KeyCode::NumpadAdd => Key::NumpadAdd,
        KeyCode::NumpadEquals => Key::NumpadEqual,
        KeyCode::Shift => Key::ShiftLeft,
        KeyCode::Control => Key::ControlLeft,
        KeyCode::Alt => Key::AltLeft,
        KeyCode::Logo => Key::MetaLeft,
        KeyCode::Capslock => Key::CapsLock,
        KeyCode::Numlock => Key::NumLock,
        KeyCode::ScrollLock => Key::ScrollLock,
        KeyCode::PrintScreen => Key::PrintScreen,
        KeyCode::Pause => Key::Pause,
        _ => return None,
    })
}

pub fn mods_of(m: &KeyModifiers) -> KeyMods {
    KeyMods {
        shift: m.shift,
        ctrl: m.control,
        alt: m.alt,
        super_: m.logo,
        caps_lock: false,
        num_lock: false,
    }
}

/// The key event for a text, numpad or modifier key under report-all.
///
/// `text` is the text the platform produced for this press, when the widget
/// has it (macOS order); otherwise the key's own character stands in. The
/// base codepoint is the key's unshifted character, so shift+a is key 97
/// with shift, as the spec requires.
///
/// Text the key cannot make on its own (Option+a giving "å", a layout that
/// moves letters) was composed by the OS: the spec reports such text
/// against key 0, so it is returned separately and the caller encodes it
/// with [`crate::term::key_encode::encode_text`] when flag 16 asks for text.
pub fn text_key_event(
    kc: KeyCode,
    mods: &KeyModifiers,
    action: KeyAction,
    text: Option<&str>,
) -> Option<(KeyEvent, Option<String>)> {
    let key = kitty_key_of(kc)?;
    let own = kc.to_char(mods.shift);
    let base = kc.to_char(false);
    let mut mods = mods_of(mods);
    let mut consumed = KeyMods::default();
    let mut composed = None;
    let utf8 = match (text, own) {
        (Some(text), Some(own)) if text.chars().eq(std::iter::once(own)) => text.to_string(),
        (Some(text), _) if !text.is_empty() => {
            composed = Some(text.to_string());
            String::new()
        }
        (_, Some(own)) if !mods.ctrl && !mods.super_ => own.to_string(),
        _ => String::new(),
    };
    // The shift that made 'A' or '!' was used up producing the text.
    if !utf8.is_empty() && mods.shift && own != base {
        consumed.shift = true;
    }
    // A modifier key's own bit: set while it is held, per the spec.
    match key {
        Key::ShiftLeft => mods.shift = action != KeyAction::Release,
        Key::ControlLeft => mods.ctrl = action != KeyAction::Release,
        Key::AltLeft => mods.alt = action != KeyAction::Release,
        Key::MetaLeft => mods.super_ = action != KeyAction::Release,
        _ => {}
    }
    let event = KeyEvent {
        action,
        key,
        mods,
        consumed_mods: consumed,
        utf8,
        unshifted_codepoint: base.map(|c| c as u32).unwrap_or(0),
    };
    Some((event, composed))
}

/// What the widget does with a `TextInput` under report-all.
#[derive(Debug, PartialEq, Eq)]
pub enum TextOutcome {
    /// Hold it for the key-down that follows; flush it at the deadline.
    Hold,
    /// The key-down that made it was already encoded.
    Drop,
}

/// Pairs Makepad's key-down and text events into one kitty key event.
#[derive(Default)]
pub struct TextKeyPairing {
    pending: Option<String>,
    /// When a key-down was encoded without its text (seconds, a monotonic
    /// clock such as `Cx::monotonic_now`).
    encoded_key_at: Option<f64>,
}

impl TextKeyPairing {
    /// A text arrived (not a paste, not a composition preview).
    pub fn on_text(&mut self, text: &str, now: f64) -> TextOutcome {
        if let Some(at) = self.encoded_key_at.take() {
            if now - at <= PAIRING_WINDOW {
                return TextOutcome::Drop;
            }
        }
        match &mut self.pending {
            // Two texts with no key between them: both are unpaired, keep
            // them in order.
            Some(held) => held.push_str(text),
            None => self.pending = Some(text.to_string()),
        }
        TextOutcome::Hold
    }

    /// A key-down arrived. Returns the text held for it (macOS order). When
    /// there is none and `encoded` is true, the text that follows is this
    /// key's (key-down-first order) and is dropped.
    pub fn on_key_down(&mut self, encoded: bool, now: f64) -> Option<String> {
        let held = self.pending.take();
        self.encoded_key_at = (held.is_none() && encoded).then_some(now);
        held
    }

    /// A key-up ends any wait for text.
    pub fn on_key_up(&mut self) {
        self.encoded_key_at = None;
    }

    /// The held text no key-down claimed: IME or composed text.
    pub fn take_unpaired(&mut self) -> Option<String> {
        self.pending.take()
    }

    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }
}

/// True when typed text must go through the pairing (and the encoder)
/// instead of straight to the program.
pub fn text_as_key_events(flags: KittyFlags) -> bool {
    flags.has(KittyFlags::REPORT_ALL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::key_encode::{encode_key, encode_text, KeyEncodeOptions};

    fn opts(flags: u8) -> KeyEncodeOptions {
        KeyEncodeOptions {
            kitty_flags: KittyFlags(flags),
            alt_esc_prefix: true,
            ..Default::default()
        }
    }

    fn mods(shift: bool, control: bool, alt: bool, logo: bool) -> KeyModifiers {
        KeyModifiers {
            shift,
            control,
            alt,
            logo,
        }
    }

    fn enc(
        kc: KeyCode,
        m: KeyModifiers,
        action: KeyAction,
        text: Option<&str>,
        flags: u8,
    ) -> String {
        let (event, composed) = text_key_event(kc, &m, action, text).expect("a kitty key");
        let o = opts(flags);
        let mut out = encode_key(&event, &o);
        if let Some(text) = composed {
            if o.kitty_flags.has(KittyFlags::REPORT_ASSOCIATED) {
                out.extend(encode_text(&text, &o));
            }
        }
        String::from_utf8(out).unwrap()
    }

    const NONE: KeyModifiers = KeyModifiers {
        shift: false,
        control: false,
        alt: false,
        logo: false,
    };

    #[test]
    fn report_all_letters_digits_and_shifted() {
        assert_eq!(
            enc(KeyCode::KeyA, NONE, KeyAction::Press, Some("a"), 8),
            "\x1b[97u"
        );
        assert_eq!(
            enc(KeyCode::KeyA, NONE, KeyAction::Press, None, 8),
            "\x1b[97u"
        );
        assert_eq!(
            enc(KeyCode::Key1, NONE, KeyAction::Press, Some("1"), 8),
            "\x1b[49u"
        );
        // Shifted: base key, shift in the mods (spec: shift+a -> CSI 97;2u).
        let shift = mods(true, false, false, false);
        assert_eq!(
            enc(KeyCode::KeyA, shift, KeyAction::Press, Some("A"), 8),
            "\x1b[97;2u"
        );
        assert_eq!(
            enc(KeyCode::Key1, shift, KeyAction::Press, Some("!"), 8),
            "\x1b[49;2u"
        );
        assert_eq!(
            enc(KeyCode::Space, NONE, KeyAction::Press, Some(" "), 8),
            "\x1b[32u"
        );
    }

    #[test]
    fn report_all_with_associated_text() {
        let shift = mods(true, false, false, false);
        // Spec example: shift+a -> CSI 97;2;65u.
        assert_eq!(
            enc(KeyCode::KeyA, shift, KeyAction::Press, Some("A"), 8 | 16),
            "\x1b[97;2;65u"
        );
        assert_eq!(
            enc(KeyCode::KeyA, NONE, KeyAction::Press, Some("a"), 8 | 16),
            "\x1b[97;;97u"
        );
        assert_eq!(
            enc(KeyCode::Key1, shift, KeyAction::Press, Some("!"), 8 | 16),
            "\x1b[49;2;33u"
        );
        // No text with ctrl (ctrl+a makes no text).
        let ctrl = mods(false, true, false, false);
        assert_eq!(
            enc(KeyCode::KeyA, ctrl, KeyAction::Press, None, 8 | 16),
            "\x1b[97;5u"
        );
        // Releases carry no text.
        assert_eq!(
            enc(KeyCode::KeyA, NONE, KeyAction::Release, None, 2 | 8 | 16),
            "\x1b[97;1:3u"
        );
    }

    #[test]
    fn composed_text_goes_to_key_zero() {
        // Option+a gives "å" on macOS: the key event, then the text the OS
        // composed against key 0 (spec: alt+a -> CSI 0;;229u).
        let alt = mods(false, false, true, false);
        assert_eq!(
            enc(KeyCode::KeyA, alt, KeyAction::Press, Some("å"), 8 | 16),
            "\x1b[97;3u\x1b[0;;229u"
        );
        // Without flag 16 the key event alone.
        assert_eq!(
            enc(KeyCode::KeyA, alt, KeyAction::Press, Some("å"), 8),
            "\x1b[97;3u"
        );
    }

    #[test]
    fn ctrl_alt_super_combos() {
        let ctrl = mods(false, true, false, false);
        let alt = mods(false, false, true, false);
        let sup = mods(false, false, false, true);
        let ctrl_shift = mods(true, true, false, false);
        for flags in [1, 8, 1 | 8] {
            assert_eq!(
                enc(KeyCode::KeyC, ctrl, KeyAction::Press, None, flags),
                "\x1b[99;5u",
                "flags {flags}"
            );
            assert_eq!(
                enc(KeyCode::KeyC, sup, KeyAction::Press, None, flags),
                "\x1b[99;9u",
                "flags {flags}"
            );
            assert_eq!(
                enc(KeyCode::KeyC, ctrl_shift, KeyAction::Press, None, flags),
                "\x1b[99;6u",
                "flags {flags}"
            );
        }
        assert_eq!(
            enc(KeyCode::KeyA, alt, KeyAction::Press, Some("a"), 1),
            "\x1b[97;3u"
        );
    }

    #[test]
    fn press_repeat_release_across_flags() {
        let cases: &[(u8, KeyAction, &str)] = &[
            // Disambiguate only: text is text, no repeats or releases.
            (1, KeyAction::Press, "a"),
            (1, KeyAction::Repeat, "a"),
            (1, KeyAction::Release, ""),
            // Report event types: text on press/repeat, release as a code.
            (2, KeyAction::Press, "a"),
            (2, KeyAction::Repeat, "a"),
            (2, KeyAction::Release, "\x1b[97;1:3u"),
            (1 | 2, KeyAction::Release, "\x1b[97;1:3u"),
            // Report all: codes, no event types.
            (8, KeyAction::Press, "\x1b[97u"),
            (8, KeyAction::Repeat, "\x1b[97u"),
            (8, KeyAction::Release, ""),
            // Both.
            (2 | 8, KeyAction::Press, "\x1b[97u"),
            (2 | 8, KeyAction::Repeat, "\x1b[97;1:2u"),
            (2 | 8, KeyAction::Release, "\x1b[97;1:3u"),
        ];
        for &(flags, action, want) in cases {
            let text = (action != KeyAction::Release).then_some("a");
            assert_eq!(
                enc(KeyCode::KeyA, NONE, action, text, flags),
                want,
                "flags {flags} {action:?}"
            );
        }
    }

    #[test]
    fn modifier_keys_under_report_all() {
        let shift = mods(true, false, false, false);
        // Press sets the key's own bit, release clears it.
        assert_eq!(
            enc(KeyCode::Shift, shift, KeyAction::Press, None, 8),
            "\x1b[57441;2u"
        );
        assert_eq!(
            enc(KeyCode::Shift, NONE, KeyAction::Release, None, 2 | 8),
            "\x1b[57441;1:3u"
        );
        let ctrl = mods(false, true, false, false);
        assert_eq!(
            enc(KeyCode::Control, ctrl, KeyAction::Press, None, 2 | 8),
            "\x1b[57442;5u"
        );
        // Not reported without report-all.
        assert_eq!(
            enc(KeyCode::Shift, shift, KeyAction::Press, None, 1 | 2),
            ""
        );
    }

    #[test]
    fn ime_text_is_not_a_key_event() {
        for flags in [0u8, 1, 2, 8, 1 | 2 | 8] {
            assert_eq!(
                encode_text("你好", &opts(flags)),
                "你好".as_bytes(),
                "flags {flags}"
            );
        }
        for flags in [8 | 16, 31] {
            assert_eq!(
                encode_text("你好", &opts(flags)),
                b"\x1b[0;;20320:22909u",
                "flags {flags}"
            );
        }
        // Control characters never ride in the text field.
        assert_eq!(encode_text("é", &opts(8 | 16)), b"\x1b[0;;233u");
    }

    #[test]
    fn pairing_macos_order_text_then_key() {
        let mut p = TextKeyPairing::default();
        let t = 100.0;
        assert_eq!(p.on_text("a", t), TextOutcome::Hold);
        assert_eq!(p.on_key_down(true, t), Some("a".to_string()));
        // The following text (none on macOS) would not be dropped: nothing
        // was encoded without its text.
        assert!(!p.has_pending());
        assert_eq!(p.on_text("b", t), TextOutcome::Hold);
    }

    #[test]
    fn pairing_key_first_order_drops_the_duplicate_text() {
        let mut p = TextKeyPairing::default();
        let t = 100.0;
        assert_eq!(p.on_key_down(true, t), None);
        assert_eq!(p.on_text("a", t), TextOutcome::Drop);
        // Only the one text belongs to that key.
        assert_eq!(p.on_text("b", t), TextOutcome::Hold);
        assert_eq!(p.take_unpaired(), Some("b".to_string()));
        // A key-up ends the wait: a later IME commit is kept.
        assert_eq!(p.on_key_down(true, t), None);
        p.on_key_up();
        assert_eq!(p.on_text("你", t), TextOutcome::Hold);
    }

    #[test]
    fn pairing_ime_commit_has_no_key() {
        let mut p = TextKeyPairing::default();
        let t = 100.0;
        assert_eq!(p.on_text("你", t), TextOutcome::Hold);
        assert_eq!(p.on_text("好", t), TextOutcome::Hold);
        assert_eq!(p.take_unpaired(), Some("你好".to_string()));
        assert_eq!(p.take_unpaired(), None);
        // A stale key-down does not eat a later text.
        assert_eq!(p.on_key_down(true, t), None);
        assert_eq!(p.on_text("x", t + PAIRING_WINDOW * 2.0), TextOutcome::Hold);
    }

    #[test]
    fn only_report_all_routes_text_through_the_encoder() {
        for flags in [0u8, 1, 2, 4, 16, 1 | 2 | 4 | 16] {
            assert!(!text_as_key_events(KittyFlags(flags)), "flags {flags}");
        }
        assert!(text_as_key_events(KittyFlags(8)));
        assert!(text_as_key_events(KittyFlags(31)));
    }
}
