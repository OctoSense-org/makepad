//! Keyboard shortcuts: one table of key chords and the actions they run,
//! configured with Ghostty's `keybind` syntax in `settings.conf`.
//!
//! ```text
//! keybind = ctrl+shift+t=new_tab        # bind (replaces that chord's action)
//! keybind = ctrl+shift+d=unbind         # the key goes to the program again
//! keybind = ctrl+shift+z=ignore         # the key does nothing at all
//! keybind = alt+left=text:\x1bb         # type text (\x1b, \n, \u{...} escapes)
//! keybind = ctrl+shift+k=clear_screen
//! keybind = clear                       # drop every binding, defaults included
//! ```
//!
//! Lines apply in file order on top of the platform's defaults
//! ([`Keybinds::defaults`], exactly the keys the terminal has always
//! taken). A line that does not parse is skipped with a warning; the rest of
//! the file still applies.
//!
//! Who runs what: the tab widget (`crate::tabs`) runs the tab, split and
//! settings actions ([`Scope::Tabs`]); the focused terminal
//! (`crate::widget`) runs the rest ([`Scope::Pane`]). Each asks
//! [`Keybinds::decide`] on a key press: a bound chord is consumed; an
//! unbound one goes on to the program through the Kitty/legacy encoder.
//! While a pane's search bar is open, its own editing keys
//! ([`crate::search::bar_key`]) win over any binding.
//!
//! Keys are Makepad key codes: the physical key (a US layout's names), and
//! a chord matches only with exactly its modifiers.

use makepad_widgets::{KeyCode, KeyEvent, KeyModifiers};

use crate::panes::Dir;

/// The modifiers of a chord. `sup` is Cmd on macOS, the Windows/Super key
/// elsewhere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mods {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub sup: bool,
}

impl Mods {
    pub fn of(m: &KeyModifiers) -> Mods {
        Mods {
            shift: m.shift,
            ctrl: m.control,
            alt: m.alt,
            sup: m.logo,
        }
    }
}

/// A key and exactly the modifiers held with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trigger {
    pub key: KeyCode,
    pub mods: Mods,
}

impl Trigger {
    pub fn of(key: &KeyEvent) -> Trigger {
        Trigger {
            key: key.key_code,
            mods: Mods::of(&key.modifiers),
        }
    }
}

/// What a shortcut does. The names in the file are Ghostty's
/// ([`Action::config_name`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    NewTab,
    /// The focused split, or the tab when it has one pane (asks first
    /// while a job runs there).
    CloseSurface,
    /// The whole tab, every split in it.
    CloseTab,
    NextTab,
    PreviousTab,
    /// 1-based; past the last tab selects the last one.
    GotoTab(usize),
    LastTab,
    /// Split the focused pane: `Right` (side by side) or `Down` (stacked).
    NewSplit(Dir),
    GotoSplit(Dir),
    ToggleSplitZoom,
    /// Open the settings panel (again: close it).
    OpenConfig,
    /// Type a name for the selected tab.
    PromptTabTitle,
    StartSearch,
    /// `next` steps towards older output (search starts at the newest,
    /// as Enter does); `previous` towards newer. Only while the bar is
    /// open.
    NavigateSearch {
        next: bool,
    },
    /// Close the search bar. Only while it is open.
    EndSearch,
    CopyToClipboard,
    SelectAll,
    /// Drop the scrollback and the rows above the cursor (Ghostty's
    /// `clear_screen` away from a prompt). Not on the alternate screen.
    ClearScreen,
    /// Points, this pane only (until the settings change).
    IncreaseFontSize(f64),
    DecreaseFontSize(f64),
    ResetFontSize,
    ScrollPageUp,
    ScrollPageDown,
    ScrollToTop,
    ScrollToBottom,
    /// Lines; negative scrolls up (into history).
    ScrollPageLines(i32),
    /// Read `settings.conf` again now.
    ReloadConfig,
    /// Bytes to write to the program, as if typed (`text:`, `csi:`,
    /// `esc:`). `spelled` is the action as written, for the panel.
    Text {
        bytes: Vec<u8>,
        spelled: String,
    },
    /// Swallow the key: neither an action nor the program gets it.
    Ignore,
}

/// Which widget runs an action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// The tab widget: tabs, splits, the settings panel.
    Tabs,
    /// The focused terminal: search, clipboard, scrolling, text.
    Pane,
}

impl Action {
    pub fn scope(&self) -> Scope {
        match self {
            Action::NewTab
            | Action::CloseSurface
            | Action::CloseTab
            | Action::NextTab
            | Action::PreviousTab
            | Action::GotoTab(_)
            | Action::LastTab
            | Action::NewSplit(_)
            | Action::GotoSplit(_)
            | Action::ToggleSplitZoom
            | Action::OpenConfig
            | Action::PromptTabTitle => Scope::Tabs,
            _ => Scope::Pane,
        }
    }

    /// The action as `settings.conf` spells it.
    pub fn config_name(&self) -> String {
        let dir = |d: &Dir| match d {
            Dir::Left => "left",
            Dir::Right => "right",
            Dir::Up => "up",
            Dir::Down => "down",
        };
        match self {
            Action::NewTab => "new_tab".into(),
            Action::CloseSurface => "close_surface".into(),
            Action::CloseTab => "close_tab".into(),
            Action::NextTab => "next_tab".into(),
            Action::PreviousTab => "previous_tab".into(),
            Action::GotoTab(n) => format!("goto_tab:{n}"),
            Action::LastTab => "last_tab".into(),
            Action::NewSplit(d) => format!("new_split:{}", dir(d)),
            Action::GotoSplit(d) => format!("goto_split:{}", dir(d)),
            Action::ToggleSplitZoom => "toggle_split_zoom".into(),
            Action::OpenConfig => "open_config".into(),
            Action::PromptTabTitle => "prompt_tab_title".into(),
            Action::StartSearch => "start_search".into(),
            Action::NavigateSearch { next: true } => "navigate_search:next".into(),
            Action::NavigateSearch { next: false } => "navigate_search:previous".into(),
            Action::EndSearch => "end_search".into(),
            Action::CopyToClipboard => "copy_to_clipboard".into(),
            Action::SelectAll => "select_all".into(),
            Action::ClearScreen => "clear_screen".into(),
            Action::IncreaseFontSize(p) => format!("increase_font_size:{p}"),
            Action::DecreaseFontSize(p) => format!("decrease_font_size:{p}"),
            Action::ResetFontSize => "reset_font_size".into(),
            Action::ScrollPageUp => "scroll_page_up".into(),
            Action::ScrollPageDown => "scroll_page_down".into(),
            Action::ScrollToTop => "scroll_to_top".into(),
            Action::ScrollToBottom => "scroll_to_bottom".into(),
            Action::ScrollPageLines(n) => format!("scroll_page_lines:{n}"),
            Action::ReloadConfig => "reload_config".into(),
            Action::Text { spelled, .. } => spelled.clone(),
            Action::Ignore => "ignore".into(),
        }
    }

    /// What the settings panel calls it.
    pub fn label(&self) -> String {
        let dir = |d: &Dir| match d {
            Dir::Left => "left",
            Dir::Right => "right",
            Dir::Up => "up",
            Dir::Down => "down",
        };
        match self {
            Action::NewTab => "New tab".into(),
            Action::CloseSurface => "Close split or tab".into(),
            Action::CloseTab => "Close tab".into(),
            Action::NextTab => "Next tab".into(),
            Action::PreviousTab => "Previous tab".into(),
            Action::GotoTab(n) => format!("Go to tab {n}"),
            Action::LastTab => "Go to the last tab".into(),
            Action::NewSplit(Dir::Down) => "Split down".into(),
            Action::NewSplit(_) => "Split right".into(),
            Action::GotoSplit(d) => format!("Focus the split {}", dir(d)),
            Action::ToggleSplitZoom => "Zoom the split".into(),
            Action::OpenConfig => "Settings".into(),
            Action::PromptTabTitle => "Rename tab".into(),
            Action::StartSearch => "Search".into(),
            Action::NavigateSearch { next: true } => "Search: next (older) match".into(),
            Action::NavigateSearch { next: false } => "Search: previous (newer) match".into(),
            Action::EndSearch => "Search: close".into(),
            Action::CopyToClipboard => "Copy the selection".into(),
            Action::SelectAll => "Select all".into(),
            Action::ClearScreen => "Clear screen and scrollback".into(),
            Action::IncreaseFontSize(_) => "Bigger text".into(),
            Action::DecreaseFontSize(_) => "Smaller text".into(),
            Action::ResetFontSize => "Text size from settings".into(),
            Action::ScrollPageUp => "Scroll a page up".into(),
            Action::ScrollPageDown => "Scroll a page down".into(),
            Action::ScrollToTop => "Scroll to the top".into(),
            Action::ScrollToBottom => "Scroll to the bottom".into(),
            Action::ScrollPageLines(n) => format!("Scroll {n} lines"),
            Action::ReloadConfig => "Reload settings.conf".into(),
            Action::Text { .. } => "Type text".into(),
            Action::Ignore => "Nothing (key ignored)".into(),
        }
    }
}

/// One entry of the table.
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    pub trigger: Trigger,
    pub action: Action,
    /// `performable:`: take the key only when the action can run now;
    /// otherwise it goes on to the program.
    pub performable: bool,
    /// `unconsumed:`: run the action and still send the key to the
    /// program.
    pub unconsumed: bool,
}

/// What a widget should do with a key press.
#[derive(Clone, Debug, PartialEq)]
pub enum Decision {
    /// Run the action; `consume`: the key goes no further.
    Run { action: Action, consume: bool },
    /// Not this widget's: pass the key on (to the pane, or the program).
    Pass,
}

/// What a widget knows when it decides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Context {
    /// Tabs open (0: the tab widget is off, e.g. a preview pager).
    pub tabs: usize,
    /// The focused pane's search bar is open.
    pub search_open: bool,
}

impl Context {
    /// Whether `action` can do something now: `performable:` bindings and
    /// search stepping pass the key on when it cannot.
    fn can_perform(&self, action: &Action) -> bool {
        match action {
            Action::GotoTab(_) | Action::LastTab | Action::NextTab | Action::PreviousTab => {
                self.tabs > 1
            }
            Action::NavigateSearch { .. } | Action::EndSearch => self.search_open,
            _ => true,
        }
    }
}

/// A `keybind` line that did not apply, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeybindError {
    pub line: String,
    pub message: String,
}

/// The effective key-binding table.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Keybinds {
    pub bindings: Vec<Binding>,
    /// The lines that were skipped.
    pub errors: Vec<KeybindError>,
}

/// One parsed `keybind` value.
#[derive(Clone, Debug, PartialEq)]
pub enum Line {
    /// `clear`: drop every binding so far, defaults included.
    Clear,
    Unbind(Trigger),
    Bind(Binding),
}

/// The defaults: exactly the keys the terminal took before shortcuts were
/// configurable (the same on every platform, but search: Cmd+F and
/// Cmd+G on macOS, Ctrl+Shift+F elsewhere).
const DEFAULTS: &[&str] = &[
    "ctrl+shift+t=new_tab",
    "ctrl+shift+w=close_surface",
    "ctrl+tab=next_tab",
    "ctrl+page_down=next_tab",
    "ctrl+shift+tab=previous_tab",
    "ctrl+page_up=previous_tab",
    // A single tab leaves Alt+digit to the shell (readline's Meta+digit).
    "performable:alt+1=goto_tab:1",
    "performable:alt+2=goto_tab:2",
    "performable:alt+3=goto_tab:3",
    "performable:alt+4=goto_tab:4",
    "performable:alt+5=goto_tab:5",
    "performable:alt+6=goto_tab:6",
    "performable:alt+7=goto_tab:7",
    "performable:alt+8=goto_tab:8",
    "performable:alt+9=last_tab",
    "ctrl+comma=open_config",
    "ctrl+shift+d=new_split:right",
    "ctrl+shift+e=new_split:down",
    "ctrl+shift+z=toggle_split_zoom",
    "ctrl+shift+left=goto_split:left",
    "ctrl+shift+right=goto_split:right",
    "ctrl+shift+up=goto_split:up",
    "ctrl+shift+down=goto_split:down",
];

const DEFAULTS_MACOS: &[&str] = &[
    "cmd+f=start_search",
    "cmd+g=navigate_search:next",
    "cmd+shift+g=navigate_search:previous",
];

const DEFAULTS_OTHER: &[&str] = &["ctrl+shift+f=start_search"];

impl Keybinds {
    /// The platform's defaults (`mac`: macOS).
    pub fn defaults(mac: bool) -> Keybinds {
        let mut out = Keybinds::default();
        let platform = if mac { DEFAULTS_MACOS } else { DEFAULTS_OTHER };
        for line in DEFAULTS.iter().chain(platform) {
            let applied = out.apply(line);
            debug_assert!(applied.is_ok(), "default {line}: {applied:?}");
        }
        out
    }

    /// The defaults with `lines` (the `keybind` values, in file order)
    /// applied on top.
    pub fn build(lines: &[String], mac: bool) -> Keybinds {
        let mut out = Keybinds::defaults(mac);
        for line in lines {
            if let Err(message) = out.apply(line) {
                out.errors.push(KeybindError {
                    line: line.trim().to_owned(),
                    message,
                });
            }
        }
        out
    }

    /// For this platform.
    pub fn for_lines(lines: &[String]) -> Keybinds {
        Keybinds::build(lines, cfg!(target_os = "macos"))
    }

    /// Apply one `keybind` value.
    pub fn apply(&mut self, line: &str) -> Result<(), String> {
        match parse_line(line)? {
            Line::Clear => self.bindings.clear(),
            Line::Unbind(trigger) => self.bindings.retain(|b| b.trigger != trigger),
            Line::Bind(binding) => match self
                .bindings
                .iter_mut()
                .find(|b| b.trigger == binding.trigger)
            {
                Some(existing) => *existing = binding,
                None => self.bindings.push(binding),
            },
        }
        Ok(())
    }

    /// The binding of a key press, if its chord is bound.
    pub fn lookup(&self, key: &KeyEvent) -> Option<&Binding> {
        let trigger = Trigger::of(key);
        self.bindings.iter().find(|b| b.trigger == trigger)
    }

    /// The first chord bound to `action`.
    pub fn trigger_for(&self, action: &Action) -> Option<Trigger> {
        self.bindings
            .iter()
            .find(|b| &b.action == action)
            .map(|b| b.trigger)
    }

    /// What the widget running `scope` does with `key`. The caller has
    /// already given an open search bar its own keys.
    pub fn decide(&self, key: &KeyEvent, scope: Scope, ctx: &Context) -> Decision {
        let Some(binding) = self.lookup(key) else {
            return Decision::Pass;
        };
        if binding.action.scope() != scope {
            return Decision::Pass;
        }
        let performable = ctx.can_perform(&binding.action);
        // Search stepping is meaningful only in the bar: elsewhere the key
        // stays the program's (Ctrl+N stays readline's next line).
        let conditional = binding.performable
            || matches!(
                binding.action,
                Action::NavigateSearch { .. } | Action::EndSearch
            );
        if conditional && !performable {
            return Decision::Pass;
        }
        // The bar has the keyboard: text goes nowhere near the program.
        if ctx.search_open && matches!(binding.action, Action::Text { .. }) {
            return Decision::Pass;
        }
        Decision::Run {
            action: binding.action.clone(),
            consume: !binding.unconsumed,
        }
    }

    /// The table for people: each action with every chord bound to it, in
    /// table order, as (action label, config name, chords).
    pub fn describe(&self, mac: bool) -> Vec<(String, String, String)> {
        let mut out: Vec<(Action, Vec<String>)> = Vec::new();
        for b in &self.bindings {
            let chord = chord_label(&b.trigger, mac);
            match out.iter_mut().find(|(a, _)| *a == b.action) {
                Some((_, chords)) => chords.push(chord),
                None => out.push((b.action.clone(), vec![chord])),
            }
        }
        out.into_iter()
            .map(|(a, chords)| (a.label(), a.config_name(), chords.join(", ")))
            .collect()
    }
}

/// Every action there is (with its default parameter), for the settings
/// panel's list of what can be bound. `text:` and `ignore` are left out.
pub fn available_actions() -> Vec<Action> {
    vec![
        Action::NewTab,
        Action::CloseSurface,
        Action::CloseTab,
        Action::NextTab,
        Action::PreviousTab,
        Action::GotoTab(1),
        Action::LastTab,
        Action::NewSplit(Dir::Right),
        Action::NewSplit(Dir::Down),
        Action::GotoSplit(Dir::Left),
        Action::GotoSplit(Dir::Right),
        Action::GotoSplit(Dir::Up),
        Action::GotoSplit(Dir::Down),
        Action::ToggleSplitZoom,
        Action::OpenConfig,
        Action::PromptTabTitle,
        Action::StartSearch,
        Action::NavigateSearch { next: true },
        Action::NavigateSearch { next: false },
        Action::EndSearch,
        Action::CopyToClipboard,
        Action::SelectAll,
        Action::ClearScreen,
        Action::IncreaseFontSize(1.0),
        Action::DecreaseFontSize(1.0),
        Action::ResetFontSize,
        Action::ScrollPageUp,
        Action::ScrollPageDown,
        Action::ScrollToTop,
        Action::ScrollToBottom,
        Action::ScrollPageLines(-3),
        Action::ReloadConfig,
    ]
}

// ----------------------------------------------------------------------
// Parsing
// ----------------------------------------------------------------------

/// Parse one `keybind` value: `clear`, `TRIGGER=unbind`, or
/// `[performable:][unconsumed:]TRIGGER=ACTION`.
pub fn parse_line(line: &str) -> Result<Line, String> {
    let line = line.trim();
    if line.is_empty() {
        return Err("empty keybind".into());
    }
    if line == "clear" {
        return Ok(Line::Clear);
    }
    let Some(split) = split_at_equals(line) else {
        return Err("expected TRIGGER=ACTION (e.g. ctrl+shift+t=new_tab)".into());
    };
    let (trigger_text, action_text) = (line[..split].trim(), line[split + 1..].trim());
    let mut rest = trigger_text;
    let (mut performable, mut unconsumed) = (false, false);
    loop {
        if let Some(r) = rest.strip_prefix("performable:") {
            performable = true;
            rest = r;
        } else if let Some(r) = rest.strip_prefix("unconsumed:") {
            unconsumed = true;
            rest = r;
        } else if rest.starts_with("global:") || rest.starts_with("all:") {
            return Err(
                "the global: and all: prefixes are not supported (one window's keys only)".into(),
            );
        } else {
            break;
        }
    }
    let trigger = parse_trigger(rest)?;
    if action_text.is_empty() {
        return Err("missing action after =".into());
    }
    if action_text == "unbind" {
        return Ok(Line::Unbind(trigger));
    }
    let action = parse_action(action_text)?;
    Ok(Line::Bind(Binding {
        trigger,
        action,
        performable,
        unconsumed,
    }))
}

/// Where TRIGGER ends: the first `=` that is not itself the key
/// (`ctrl+==reset_font_size` binds Ctrl+=).
fn split_at_equals(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    (0..bytes.len()).find(|&i| bytes[i] == b'=' && i > 0 && bytes[i - 1] != b'+')
}

/// `ctrl+shift+t`, `cmd+,`, `alt+page_up`, `ctrl++`.
pub fn parse_trigger(text: &str) -> Result<Trigger, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("missing key before =".into());
    }
    let (mods_text, key_text) = if text == "+" {
        ("", "+")
    } else if let Some(mods) = text.strip_suffix("++") {
        (mods, "+")
    } else {
        match text.rsplit_once('+') {
            Some((mods, key)) => (mods, key),
            None => ("", text),
        }
    };
    if key_text.len() > 1 && key_text.contains('>') || mods_text.contains('>') {
        return Err("key sequences (a>b) are not supported yet".into());
    }
    let mut mods = Mods::default();
    if !mods_text.is_empty() {
        for m in mods_text.split('+') {
            let flag = match m.trim().to_ascii_lowercase().as_str() {
                "shift" => &mut mods.shift,
                "ctrl" | "control" => &mut mods.ctrl,
                "alt" | "opt" | "option" => &mut mods.alt,
                "super" | "cmd" | "command" => &mut mods.sup,
                "" => return Err(format!("empty modifier in `{text}`")),
                other => {
                    return Err(format!(
                        "unknown modifier `{other}` (shift, ctrl, alt/opt, super/cmd)"
                    ))
                }
            };
            if *flag {
                return Err(format!("modifier `{m}` twice"));
            }
            *flag = true;
        }
    }
    let (key, implied_shift) = parse_key(key_text)?;
    if implied_shift {
        mods.shift = true;
    }
    Ok(Trigger { key, mods })
}

/// A key name: a letter or digit, a punctuation character or its name, or
/// a named key. A shifted symbol (`+`, `?`, ...) implies Shift on its US
/// key.
fn parse_key(text: &str) -> Result<(KeyCode, bool), String> {
    let lower = text.trim().to_ascii_lowercase();
    let name = lower.strip_prefix("physical:").unwrap_or(&lower);
    let name = name
        .strip_prefix("key_")
        .filter(|n| n.len() == 1)
        .unwrap_or(name);
    let name = name
        .strip_prefix("digit_")
        .filter(|n| n.len() == 1)
        .unwrap_or(name);
    let mut chars = name.chars();
    if let (Some(c), None) = (chars.next(), chars.clone().next()) {
        if let Some(found) = char_key(c) {
            return Ok(found);
        }
    }
    let key = match name {
        "enter" | "return" => KeyCode::ReturnKey,
        "escape" | "esc" => KeyCode::Escape,
        "tab" => KeyCode::Tab,
        "backspace" => KeyCode::Backspace,
        "delete" | "del" => KeyCode::Delete,
        "insert" => KeyCode::Insert,
        "home" => KeyCode::Home,
        "end" => KeyCode::End,
        "page_up" | "pageup" => KeyCode::PageUp,
        "page_down" | "pagedown" => KeyCode::PageDown,
        "up" | "arrow_up" => KeyCode::ArrowUp,
        "down" | "arrow_down" => KeyCode::ArrowDown,
        "left" | "arrow_left" => KeyCode::ArrowLeft,
        "right" | "arrow_right" => KeyCode::ArrowRight,
        "space" => KeyCode::Space,
        "grave" | "backquote" | "grave_accent" => KeyCode::Backtick,
        "minus" => KeyCode::Minus,
        "equal" | "equals" => KeyCode::Equals,
        "bracket_left" | "left_bracket" => KeyCode::LBracket,
        "bracket_right" | "right_bracket" => KeyCode::RBracket,
        "backslash" => KeyCode::Backslash,
        "semicolon" => KeyCode::Semicolon,
        "quote" | "apostrophe" => KeyCode::Quote,
        "comma" => KeyCode::Comma,
        "period" => KeyCode::Period,
        "slash" => KeyCode::Slash,
        "plus" => return Ok((KeyCode::Equals, true)),
        "f1" => KeyCode::F1,
        "f2" => KeyCode::F2,
        "f3" => KeyCode::F3,
        "f4" => KeyCode::F4,
        "f5" => KeyCode::F5,
        "f6" => KeyCode::F6,
        "f7" => KeyCode::F7,
        "f8" => KeyCode::F8,
        "f9" => KeyCode::F9,
        "f10" => KeyCode::F10,
        "f11" => KeyCode::F11,
        "f12" => KeyCode::F12,
        "kp_0" | "numpad_0" => KeyCode::Numpad0,
        "kp_1" | "numpad_1" => KeyCode::Numpad1,
        "kp_2" | "numpad_2" => KeyCode::Numpad2,
        "kp_3" | "numpad_3" => KeyCode::Numpad3,
        "kp_4" | "numpad_4" => KeyCode::Numpad4,
        "kp_5" | "numpad_5" => KeyCode::Numpad5,
        "kp_6" | "numpad_6" => KeyCode::Numpad6,
        "kp_7" | "numpad_7" => KeyCode::Numpad7,
        "kp_8" | "numpad_8" => KeyCode::Numpad8,
        "kp_9" | "numpad_9" => KeyCode::Numpad9,
        "kp_enter" | "numpad_enter" => KeyCode::NumpadEnter,
        "kp_add" | "numpad_add" => KeyCode::NumpadAdd,
        "kp_subtract" | "numpad_subtract" => KeyCode::NumpadSubtract,
        "kp_multiply" | "numpad_multiply" => KeyCode::NumpadMultiply,
        "kp_divide" | "numpad_divide" => KeyCode::NumpadDivide,
        "kp_decimal" | "numpad_decimal" => KeyCode::NumpadDecimal,
        "kp_equal" | "numpad_equal" => KeyCode::NumpadEquals,
        "shift" | "ctrl" | "control" | "alt" | "opt" | "option" | "super" | "cmd" | "command" => {
            return Err(format!("`{name}` is a modifier, not a key"))
        }
        _ => return Err(format!("unknown key `{text}`")),
    };
    Ok((key, false))
}

/// A one-character key name: (key, whether Shift makes the character on a
/// US layout).
fn char_key(c: char) -> Option<(KeyCode, bool)> {
    use KeyCode::*;
    let plain = |k| Some((k, false));
    let shifted = |k| Some((k, true));
    match c {
        'a' => plain(KeyA),
        'b' => plain(KeyB),
        'c' => plain(KeyC),
        'd' => plain(KeyD),
        'e' => plain(KeyE),
        'f' => plain(KeyF),
        'g' => plain(KeyG),
        'h' => plain(KeyH),
        'i' => plain(KeyI),
        'j' => plain(KeyJ),
        'k' => plain(KeyK),
        'l' => plain(KeyL),
        'm' => plain(KeyM),
        'n' => plain(KeyN),
        'o' => plain(KeyO),
        'p' => plain(KeyP),
        'q' => plain(KeyQ),
        'r' => plain(KeyR),
        's' => plain(KeyS),
        't' => plain(KeyT),
        'u' => plain(KeyU),
        'v' => plain(KeyV),
        'w' => plain(KeyW),
        'x' => plain(KeyX),
        'y' => plain(KeyY),
        'z' => plain(KeyZ),
        '0' => plain(Key0),
        '1' => plain(Key1),
        '2' => plain(Key2),
        '3' => plain(Key3),
        '4' => plain(Key4),
        '5' => plain(Key5),
        '6' => plain(Key6),
        '7' => plain(Key7),
        '8' => plain(Key8),
        '9' => plain(Key9),
        '`' => plain(Backtick),
        '-' => plain(Minus),
        '=' => plain(Equals),
        '[' => plain(LBracket),
        ']' => plain(RBracket),
        '\\' => plain(Backslash),
        ';' => plain(Semicolon),
        '\'' => plain(Quote),
        ',' => plain(Comma),
        '.' => plain(Period),
        '/' => plain(Slash),
        '~' => shifted(Backtick),
        '!' => shifted(Key1),
        '@' => shifted(Key2),
        '#' => shifted(Key3),
        '$' => shifted(Key4),
        '%' => shifted(Key5),
        '^' => shifted(Key6),
        '&' => shifted(Key7),
        '*' => shifted(Key8),
        '(' => shifted(Key9),
        ')' => shifted(Key0),
        '_' => shifted(Minus),
        '+' => shifted(Equals),
        '{' => shifted(LBracket),
        '}' => shifted(RBracket),
        '|' => shifted(Backslash),
        ':' => shifted(Semicolon),
        '"' => shifted(Quote),
        '<' => shifted(Comma),
        '>' => shifted(Period),
        '?' => shifted(Slash),
        _ => None,
    }
}

fn parse_dir(value: &str, action: &str) -> Result<Dir, String> {
    Ok(match value {
        "left" => Dir::Left,
        "right" => Dir::Right,
        "up" | "top" => Dir::Up,
        "down" | "bottom" => Dir::Down,
        _ => {
            return Err(format!(
                "{action} takes left, right, up or down, not `{value}`"
            ))
        }
    })
}

fn parse_points(value: Option<&str>, action: &str) -> Result<f64, String> {
    match value {
        None => Ok(1.0),
        Some(v) => v
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|p| p.is_finite() && *p > 0.0 && *p <= 100.0)
            .ok_or_else(|| format!("{action} takes a size in points, not `{v}`")),
    }
}

/// `new_tab`, `goto_tab:3`, `text:\x1bb`, ...
pub fn parse_action(text: &str) -> Result<Action, String> {
    let text = text.trim();
    // The literal-text actions keep everything after the first colon.
    if let Some(body) = text.strip_prefix("text:") {
        let bytes = unescape(body)?;
        if bytes.is_empty() {
            return Err("text: needs something to type".into());
        }
        return Ok(Action::Text {
            bytes,
            spelled: text.to_owned(),
        });
    }
    if let Some(body) = text.strip_prefix("csi:") {
        if body.is_empty() || body.chars().any(char::is_control) {
            return Err("csi: needs the sequence after ESC [ (e.g. csi:2J)".into());
        }
        let mut bytes = b"\x1b[".to_vec();
        bytes.extend_from_slice(body.as_bytes());
        return Ok(Action::Text {
            bytes,
            spelled: text.to_owned(),
        });
    }
    if let Some(body) = text.strip_prefix("esc:") {
        if body.is_empty() || body.chars().any(char::is_control) {
            return Err("esc: needs the text after ESC (e.g. esc:b)".into());
        }
        let mut bytes = b"\x1b".to_vec();
        bytes.extend_from_slice(body.as_bytes());
        return Ok(Action::Text {
            bytes,
            spelled: text.to_owned(),
        });
    }
    let (name, param) = match text.split_once(':') {
        Some((n, p)) => (n.trim(), Some(p.trim())),
        None => (text, None),
    };
    let no_param = |action: Action| match param {
        None => Ok(action),
        Some(p) => Err(format!("{name} takes no parameter (got `{p}`)")),
    };
    match name {
        "new_tab" => no_param(Action::NewTab),
        "close_surface" => no_param(Action::CloseSurface),
        "close_tab" => no_param(Action::CloseTab),
        "next_tab" => no_param(Action::NextTab),
        "previous_tab" | "prev_tab" => no_param(Action::PreviousTab),
        "last_tab" => no_param(Action::LastTab),
        "goto_tab" => match param
            .and_then(|p| p.parse::<usize>().ok())
            .filter(|n| (1..=99).contains(n))
        {
            Some(n) => Ok(Action::GotoTab(n)),
            None => Err("goto_tab takes a tab number from 1 (e.g. goto_tab:1)".into()),
        },
        "new_split" => match parse_dir(param.unwrap_or(""), name)? {
            Dir::Right => Ok(Action::NewSplit(Dir::Right)),
            Dir::Down => Ok(Action::NewSplit(Dir::Down)),
            _ => Err("new_split takes right or down (a split opens after the focused pane)".into()),
        },
        "goto_split" => Ok(Action::GotoSplit(parse_dir(param.unwrap_or(""), name)?)),
        "toggle_split_zoom" => no_param(Action::ToggleSplitZoom),
        "open_config" | "open_settings" => no_param(Action::OpenConfig),
        "prompt_tab_title" | "prompt_surface_title" | "rename_tab" => {
            no_param(Action::PromptTabTitle)
        }
        "start_search" | "search" => no_param(Action::StartSearch),
        "end_search" => no_param(Action::EndSearch),
        "navigate_search" => match param {
            Some("next") => Ok(Action::NavigateSearch { next: true }),
            Some("previous") | Some("prev") => Ok(Action::NavigateSearch { next: false }),
            _ => Err("navigate_search takes next or previous".into()),
        },
        "copy_to_clipboard" | "copy" => no_param(Action::CopyToClipboard),
        "paste_from_clipboard" | "paste" | "paste_from_selection" => Err(
            "pasting is the system's Cmd+V / Ctrl+V and cannot be bound to another key yet".into(),
        ),
        "select_all" => no_param(Action::SelectAll),
        "clear_screen" => no_param(Action::ClearScreen),
        "increase_font_size" => Ok(Action::IncreaseFontSize(parse_points(param, name)?)),
        "decrease_font_size" => Ok(Action::DecreaseFontSize(parse_points(param, name)?)),
        "reset_font_size" => no_param(Action::ResetFontSize),
        "scroll_page_up" => no_param(Action::ScrollPageUp),
        "scroll_page_down" => no_param(Action::ScrollPageDown),
        "scroll_to_top" => no_param(Action::ScrollToTop),
        "scroll_to_bottom" => no_param(Action::ScrollToBottom),
        "scroll_page_lines" => match param
            .and_then(|p| p.parse::<i32>().ok())
            .filter(|n| *n != 0)
        {
            Some(n) => Ok(Action::ScrollPageLines(n)),
            None => Err("scroll_page_lines takes a non-zero line count (negative: up)".into()),
        },
        "reload_config" => no_param(Action::ReloadConfig),
        "ignore" => no_param(Action::Ignore),
        "unbind" => Err("unbind takes no parameter".into()),
        _ => Err(format!("unknown action `{name}`")),
    }
}

/// `text:` escapes, as Ghostty (Zig string literals) takes them: `\n`,
/// `\r`, `\t`, `\\`, `\"`, `\'`, `\xHH` (a byte) and `\u{HHHH}` (a
/// character, as UTF-8). Anything else is typed as written.
pub fn unescape(text: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            continue;
        }
        match chars.next() {
            Some('n') => out.push(b'\n'),
            Some('r') => out.push(b'\r'),
            Some('t') => out.push(b'\t'),
            Some('\\') => out.push(b'\\'),
            Some('"') => out.push(b'"'),
            Some('\'') => out.push(b'\''),
            Some('x') => {
                let hex: String = (0..2).filter_map(|_| chars.next()).collect();
                match u8::from_str_radix(&hex, 16) {
                    Ok(b) if hex.len() == 2 => out.push(b),
                    _ => return Err(format!("\\x needs two hex digits (got `\\x{hex}`)")),
                }
            }
            Some('u') => {
                if chars.next() != Some('{') {
                    return Err("\\u needs braces: \\u{1F600}".into());
                }
                let hex: String = chars.by_ref().take_while(|c| *c != '}').collect();
                let ch = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32);
                match ch {
                    Some(ch) if !hex.is_empty() && hex.len() <= 6 => {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                    _ => return Err(format!("\\u{{{hex}}} is not a character")),
                }
            }
            Some(other) => return Err(format!("unknown escape `\\{other}`")),
            None => return Err("a lone \\ at the end".into()),
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------
// Showing chords
// ----------------------------------------------------------------------

fn key_name(key: KeyCode) -> String {
    use KeyCode::*;
    let s = match key {
        ReturnKey => "Enter",
        NumpadEnter => "Keypad Enter",
        Escape => "Esc",
        Tab => "Tab",
        Backspace => "Backspace",
        Delete => "Delete",
        Insert => "Insert",
        Home => "Home",
        End => "End",
        PageUp => "PageUp",
        PageDown => "PageDown",
        ArrowUp => "\u{2191}",
        ArrowDown => "\u{2193}",
        ArrowLeft => "\u{2190}",
        ArrowRight => "\u{2192}",
        Space => "Space",
        Backtick => "`",
        Minus => "-",
        Equals => "=",
        LBracket => "[",
        RBracket => "]",
        Backslash => "\\",
        Semicolon => ";",
        Quote => "'",
        Comma => ",",
        Period => ".",
        Slash => "/",
        F1 => "F1",
        F2 => "F2",
        F3 => "F3",
        F4 => "F4",
        F5 => "F5",
        F6 => "F6",
        F7 => "F7",
        F8 => "F8",
        F9 => "F9",
        F10 => "F10",
        F11 => "F11",
        F12 => "F12",
        other => {
            let name = format!("{other:?}");
            let short = name
                .strip_prefix("Key")
                .filter(|n| !n.is_empty())
                .unwrap_or(&name);
            return short.to_owned();
        }
    };
    s.to_owned()
}

/// `Ctrl+Shift+T`, `Cmd+F` (macOS names Super Cmd and Alt Option).
pub fn chord_label(t: &Trigger, mac: bool) -> String {
    let mut out = String::new();
    if t.mods.ctrl {
        out.push_str("Ctrl+");
    }
    if t.mods.alt {
        out.push_str(if mac { "Opt+" } else { "Alt+" });
    }
    if t.mods.shift {
        out.push_str("Shift+");
    }
    if t.mods.sup {
        out.push_str(if mac { "Cmd+" } else { "Super+" });
    }
    out.push_str(&key_name(t.key));
    out
}

/// Every key code, for tests that sweep the whole keyboard.
#[cfg(test)]
pub(crate) const ALL_KEY_CODES: &[KeyCode] = {
    use KeyCode::*;
    &[
        Escape,
        Back,
        Backtick,
        Key0,
        Key1,
        Key2,
        Key3,
        Key4,
        Key5,
        Key6,
        Key7,
        Key8,
        Key9,
        Minus,
        Equals,
        Backspace,
        Tab,
        KeyQ,
        KeyW,
        KeyE,
        KeyR,
        KeyT,
        KeyY,
        KeyU,
        KeyI,
        KeyO,
        KeyP,
        LBracket,
        RBracket,
        ReturnKey,
        KeyA,
        KeyS,
        KeyD,
        KeyF,
        KeyG,
        KeyH,
        KeyJ,
        KeyK,
        KeyL,
        Semicolon,
        Quote,
        Backslash,
        KeyZ,
        KeyX,
        KeyC,
        KeyV,
        KeyB,
        KeyN,
        KeyM,
        Comma,
        Period,
        Slash,
        Control,
        Alt,
        Shift,
        Logo,
        Space,
        Capslock,
        F1,
        F2,
        F3,
        F4,
        F5,
        F6,
        F7,
        F8,
        F9,
        F10,
        F11,
        F12,
        PrintScreen,
        ScrollLock,
        Pause,
        Insert,
        Delete,
        Home,
        End,
        PageUp,
        PageDown,
        Numpad0,
        Numpad1,
        Numpad2,
        Numpad3,
        Numpad4,
        Numpad5,
        Numpad6,
        Numpad7,
        Numpad8,
        Numpad9,
        NumpadEquals,
        NumpadSubtract,
        NumpadAdd,
        NumpadDecimal,
        NumpadMultiply,
        NumpadDivide,
        Numlock,
        NumpadEnter,
        ArrowUp,
        ArrowDown,
        ArrowLeft,
        ArrowRight,
        Unknown,
    ]
};

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode, ctrl: bool, shift: bool, alt: bool, sup: bool) -> KeyEvent {
        KeyEvent {
            key_code: code,
            modifiers: KeyModifiers {
                shift,
                control: ctrl,
                alt,
                logo: sup,
            },
            ..Default::default()
        }
    }

    fn trig(text: &str) -> Trigger {
        parse_trigger(text).unwrap()
    }

    fn m(shift: bool, ctrl: bool, alt: bool, sup: bool) -> Mods {
        Mods {
            shift,
            ctrl,
            alt,
            sup,
        }
    }

    #[test]
    fn chords_and_modifier_aliases_parse() {
        let t = |k, mods| Trigger { key: k, mods };
        assert_eq!(
            trig("ctrl+shift+t"),
            t(KeyCode::KeyT, m(true, true, false, false))
        );
        assert_eq!(
            trig("control+Shift+T"),
            t(KeyCode::KeyT, m(true, true, false, false))
        );
        for sup in ["cmd", "super", "command"] {
            assert_eq!(
                trig(&format!("{sup}+k")),
                t(KeyCode::KeyK, m(false, false, false, true)),
                "{sup}"
            );
        }
        for alt in ["alt", "opt", "option"] {
            assert_eq!(
                trig(&format!("{alt}+b")),
                t(KeyCode::KeyB, m(false, false, true, false)),
                "{alt}"
            );
        }
        // Order does not matter; the key comes last.
        assert_eq!(trig("shift+ctrl+t"), trig("ctrl+shift+t"));
        assert_eq!(trig("f5"), t(KeyCode::F5, Mods::default()));
    }

    #[test]
    fn key_names_parse() {
        let k = |text: &str| trig(text).key;
        assert_eq!(k("ctrl+comma"), KeyCode::Comma);
        assert_eq!(k("ctrl+,"), KeyCode::Comma);
        assert_eq!(k("ctrl+page_up"), KeyCode::PageUp);
        assert_eq!(k("ctrl+pagedown"), KeyCode::PageDown);
        assert_eq!(k("ctrl+shift+arrow_left"), KeyCode::ArrowLeft);
        assert_eq!(k("ctrl+shift+left"), KeyCode::ArrowLeft);
        assert_eq!(k("alt+enter"), KeyCode::ReturnKey);
        assert_eq!(k("alt+return"), KeyCode::ReturnKey);
        assert_eq!(k("esc"), KeyCode::Escape);
        assert_eq!(k("cmd+equal"), KeyCode::Equals);
        assert_eq!(k("ctrl+="), KeyCode::Equals);
        assert_eq!(k("cmd+minus"), KeyCode::Minus);
        assert_eq!(k("cmd+0"), KeyCode::Key0);
        assert_eq!(k("cmd+digit_1"), KeyCode::Key1);
        assert_eq!(k("cmd+key_a"), KeyCode::KeyA);
        assert_eq!(k("ctrl+physical:a"), KeyCode::KeyA);
        assert_eq!(k("ctrl+bracket_left"), KeyCode::LBracket);
        assert_eq!(k("ctrl+grave"), KeyCode::Backtick);
        assert_eq!(k("kp_enter"), KeyCode::NumpadEnter);
        assert_eq!(k("f12"), KeyCode::F12);
        // A shifted symbol implies Shift on its US key.
        assert_eq!(
            trig("cmd+plus"),
            Trigger {
                key: KeyCode::Equals,
                mods: m(true, false, false, true)
            }
        );
        assert_eq!(trig("cmd++"), trig("cmd+plus"));
        assert_eq!(
            trig("ctrl+?"),
            Trigger {
                key: KeyCode::Slash,
                mods: m(true, true, false, false)
            }
        );
    }

    #[test]
    fn actions_parse() {
        let a = |text: &str| parse_action(text).unwrap();
        assert_eq!(a("new_tab"), Action::NewTab);
        assert_eq!(a("goto_tab:3"), Action::GotoTab(3));
        assert_eq!(a("new_split:right"), Action::NewSplit(Dir::Right));
        assert_eq!(a("new_split:down"), Action::NewSplit(Dir::Down));
        assert_eq!(a("goto_split:top"), Action::GotoSplit(Dir::Up));
        assert_eq!(
            a("navigate_search:previous"),
            Action::NavigateSearch { next: false }
        );
        assert_eq!(a("copy"), Action::CopyToClipboard);
        assert_eq!(a("open_settings"), Action::OpenConfig);
        assert_eq!(a("increase_font_size:1.5"), Action::IncreaseFontSize(1.5));
        assert_eq!(a("decrease_font_size"), Action::DecreaseFontSize(1.0));
        assert_eq!(a("scroll_page_lines:-3"), Action::ScrollPageLines(-3));
        assert_eq!(a("ignore"), Action::Ignore);
        // Every action's own spelling reads back as itself.
        for action in [
            Action::NewTab,
            Action::CloseSurface,
            Action::CloseTab,
            Action::NextTab,
            Action::PreviousTab,
            Action::GotoTab(7),
            Action::LastTab,
            Action::NewSplit(Dir::Down),
            Action::GotoSplit(Dir::Left),
            Action::ToggleSplitZoom,
            Action::OpenConfig,
            Action::PromptTabTitle,
            Action::StartSearch,
            Action::NavigateSearch { next: true },
            Action::EndSearch,
            Action::CopyToClipboard,
            Action::SelectAll,
            Action::ClearScreen,
            Action::IncreaseFontSize(2.0),
            Action::DecreaseFontSize(0.5),
            Action::ResetFontSize,
            Action::ScrollPageUp,
            Action::ScrollPageDown,
            Action::ScrollToTop,
            Action::ScrollToBottom,
            Action::ScrollPageLines(5),
            Action::ReloadConfig,
            Action::Ignore,
        ] {
            assert_eq!(a(&action.config_name()), action);
        }
        for action in available_actions() {
            assert_eq!(a(&action.config_name()), action);
            assert!(!action.label().is_empty());
        }
    }

    #[test]
    fn text_actions_unescape() {
        let bytes = |text: &str| match parse_action(text).unwrap() {
            Action::Text { bytes, .. } => bytes,
            other => panic!("{other:?}"),
        };
        assert_eq!(bytes(r"text:\x1bb"), b"\x1bb");
        assert_eq!(bytes(r"text:ls -la\n"), b"ls -la\n");
        assert_eq!(bytes(r"text:a\tb\\c\r"), b"a\tb\\c\r");
        assert_eq!(
            bytes(r"text:\u{4f60}\u{597d}"),
            "\u{4f60}\u{597d}".as_bytes()
        );
        assert_eq!(bytes("text:échec"), "échec".as_bytes());
        assert_eq!(bytes("text:a=b"), b"a=b", "an = in the text stays");
        assert_eq!(bytes("csi:2J"), b"\x1b[2J");
        assert_eq!(bytes("esc:b"), b"\x1bb");
        for bad in [
            r"text:\x1",
            r"text:\xzz",
            r"text:\u1234",
            r"text:\u{110000}",
            r"text:\q",
            r"text:ab\",
            "text:",
            "csi:",
            "esc:",
        ] {
            assert!(parse_action(bad).is_err(), "{bad}");
        }
        // The whole line keeps an = inside the text.
        match parse_line(r"ctrl+alt+e=text:echo a=b\n").unwrap() {
            Line::Bind(b) => assert_eq!(
                b.action,
                Action::Text {
                    bytes: b"echo a=b\n".to_vec(),
                    spelled: r"text:echo a=b\n".into()
                }
            ),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn lines_parse_with_prefixes_unbind_and_clear() {
        assert_eq!(parse_line("clear"), Ok(Line::Clear));
        assert_eq!(
            parse_line(" ctrl+shift+d = unbind "),
            Ok(Line::Unbind(trig("ctrl+shift+d")))
        );
        match parse_line("performable:unconsumed:alt+1=goto_tab:1").unwrap() {
            Line::Bind(b) => {
                assert!(b.performable && b.unconsumed);
                assert_eq!(b.trigger, trig("alt+1"));
            }
            other => panic!("{other:?}"),
        }
        match parse_line("ctrl+==reset_font_size").unwrap() {
            Line::Bind(b) => assert_eq!(
                (b.trigger.key, b.action),
                (KeyCode::Equals, Action::ResetFontSize)
            ),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn invalid_lines_say_why() {
        let err = |line: &str| parse_line(line).unwrap_err();
        assert!(err("ctrl+shift+t").contains("TRIGGER=ACTION"));
        assert!(err("ctrl+shift+t=").contains("missing action"));
        assert!(err("=new_tab").contains("TRIGGER=ACTION"));
        assert!(err("hyper+t=new_tab").contains("unknown modifier"));
        assert!(err("ctrl+ctrl+t=new_tab").contains("twice"));
        assert!(err("ctrl+nokey=new_tab").contains("unknown key"));
        assert!(err("ctrl+shift=new_tab").contains("modifier, not a key"));
        assert!(err("ctrl+a>n=new_tab").contains("sequences"));
        assert!(err("global:ctrl+t=new_tab").contains("not supported"));
        assert!(err("ctrl+t=launch_rockets").contains("unknown action `launch_rockets`"));
        assert!(err("ctrl+t=goto_tab:0").contains("goto_tab"));
        assert!(err("ctrl+t=new_split:left").contains("right or down"));
        assert!(err("ctrl+t=new_tab:2").contains("no parameter"));
        assert!(err("ctrl+t=paste_from_clipboard").contains("Cmd+V"));
        assert!(err("ctrl+t=increase_font_size:big").contains("points"));
        assert!(err("").contains("empty"));
    }

    /// Every key the terminal took before the table, on each platform:
    /// the defaults must reproduce exactly this.
    fn inventory(mac: bool) -> Vec<(KeyEvent, Action, bool)> {
        use KeyCode::*;
        let c = |k| key(k, true, false, false, false);
        let cs = |k| key(k, true, true, false, false);
        let a = |k| key(k, false, false, true, false);
        let mut out = vec![
            // tabs.rs: tab_command.
            (cs(KeyT), Action::NewTab, false),
            (cs(KeyW), Action::CloseSurface, false),
            (c(Tab), Action::NextTab, false),
            (c(PageDown), Action::NextTab, false),
            (cs(Tab), Action::PreviousTab, false),
            (c(PageUp), Action::PreviousTab, false),
            (c(Comma), Action::OpenConfig, false),
            (cs(KeyD), Action::NewSplit(Dir::Right), false),
            (cs(KeyE), Action::NewSplit(Dir::Down), false),
            (cs(KeyZ), Action::ToggleSplitZoom, false),
            (cs(ArrowLeft), Action::GotoSplit(Dir::Left), false),
            (cs(ArrowRight), Action::GotoSplit(Dir::Right), false),
            (cs(ArrowUp), Action::GotoSplit(Dir::Up), false),
            (cs(ArrowDown), Action::GotoSplit(Dir::Down), false),
            // Alt+digit only while two or more tabs are open.
            (a(Key9), Action::LastTab, true),
        ];
        for (n, k) in [Key1, Key2, Key3, Key4, Key5, Key6, Key7, Key8]
            .into_iter()
            .enumerate()
        {
            out.push((a(k), Action::GotoTab(n + 1), true));
        }
        // search.rs: search_key.
        if mac {
            out.push((
                key(KeyF, false, false, false, true),
                Action::StartSearch,
                false,
            ));
            out.push((
                key(KeyG, false, false, false, true),
                Action::NavigateSearch { next: true },
                false,
            ));
            out.push((
                key(KeyG, false, true, false, true),
                Action::NavigateSearch { next: false },
                false,
            ));
        } else {
            out.push((cs(KeyF), Action::StartSearch, false));
        }
        out
    }

    #[test]
    fn the_defaults_are_exactly_the_keys_taken_before() {
        for mac in [true, false] {
            let table = Keybinds::defaults(mac);
            assert!(table.errors.is_empty());
            let inventory = inventory(mac);
            assert_eq!(
                table.bindings.len(),
                inventory.len(),
                "mac {mac}: nothing more, nothing less"
            );
            for (k, action, performable) in inventory {
                let b = table
                    .lookup(&k)
                    .unwrap_or_else(|| panic!("mac {mac}: {k:?} unbound"));
                assert_eq!(
                    (&b.action, b.performable, b.unconsumed),
                    (&action, performable, false),
                    "mac {mac}: {k:?}"
                );
            }
        }
    }

    #[test]
    fn the_shell_keeps_its_own_keys_by_default() {
        use KeyCode::*;
        for mac in [true, false] {
            let table = Keybinds::defaults(mac);
            for k in [
                key(KeyT, true, false, false, false),      // transpose
                key(KeyW, true, false, false, false),      // kill word
                key(KeyF, true, false, false, false),      // forward char
                key(KeyR, true, false, false, false),      // history search
                key(KeyC, true, false, false, false),      // interrupt
                key(ArrowLeft, true, false, false, false), // word left
                key(Tab, false, false, false, false),
                key(KeyT, true, true, false, true), // Ctrl+Shift+Cmd+T is not Ctrl+Shift+T
                key(KeyB, false, false, true, false), // Meta+B
                key(KeyA, false, false, false, false),
            ] {
                assert_eq!(table.lookup(&k), None, "mac {mac}: {k:?}");
            }
            // Cmd+F is macOS's only; Ctrl+Shift+F everyone else's.
            assert_eq!(
                table
                    .lookup(&key(KeyF, false, false, false, true))
                    .is_some(),
                mac
            );
            assert_eq!(
                table.lookup(&key(KeyF, true, true, false, false)).is_some(),
                !mac
            );
        }
    }

    #[test]
    fn config_lines_remap_unbind_ignore_and_clear() {
        let lines = |ls: &[&str]| ls.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let t = Keybinds::build(
            &lines(&[
                "cmd+t=new_tab",
                "ctrl+shift+d=unbind",
                "ctrl+shift+z=ignore",
                "ctrl+shift+t=next_tab",
                r"alt+left=text:\x1bb",
                "bogus",
            ]),
            false,
        );
        let find = |k: KeyEvent| t.lookup(&k).map(|b| b.action.clone());
        assert_eq!(
            find(key(KeyCode::KeyT, false, false, false, true)),
            Some(Action::NewTab)
        );
        assert_eq!(
            find(key(KeyCode::KeyD, true, true, false, false)),
            None,
            "unbound: the program's again"
        );
        assert_eq!(
            find(key(KeyCode::KeyZ, true, true, false, false)),
            Some(Action::Ignore)
        );
        assert_eq!(
            find(key(KeyCode::KeyT, true, true, false, false)),
            Some(Action::NextTab),
            "a rebind replaces"
        );
        assert_eq!(
            find(key(KeyCode::ArrowLeft, false, false, true, false)),
            Some(Action::Text {
                bytes: b"\x1bb".to_vec(),
                spelled: r"text:\x1bb".into()
            })
        );
        assert_eq!(t.errors.len(), 1);
        assert_eq!(t.errors[0].line, "bogus");
        // Everything else is still the default.
        assert_eq!(
            find(key(KeyCode::KeyE, true, true, false, false)),
            Some(Action::NewSplit(Dir::Down))
        );
        // clear drops the defaults; later lines still apply.
        let t = Keybinds::build(&lines(&["clear", "ctrl+shift+n=new_tab"]), true);
        assert_eq!(t.bindings.len(), 1);
        assert_eq!(
            t.lookup(&key(KeyCode::KeyN, true, true, false, false))
                .map(|b| &b.action),
            Some(&Action::NewTab)
        );
        assert_eq!(
            t.lookup(&key(KeyCode::KeyT, true, true, false, false)),
            None
        );
    }

    #[test]
    fn each_widget_runs_only_its_own_actions() {
        let t = Keybinds::build(&[r"ctrl+alt+l=text:ls\r".to_string()], false);
        let ctx = Context {
            tabs: 3,
            search_open: false,
        };
        let new_tab = key(KeyCode::KeyT, true, true, false, false);
        assert_eq!(
            t.decide(&new_tab, Scope::Tabs, &ctx),
            Decision::Run {
                action: Action::NewTab,
                consume: true
            }
        );
        assert_eq!(t.decide(&new_tab, Scope::Pane, &ctx), Decision::Pass);
        let search = key(KeyCode::KeyF, true, true, false, false);
        assert_eq!(t.decide(&search, Scope::Tabs, &ctx), Decision::Pass);
        assert_eq!(
            t.decide(&search, Scope::Pane, &ctx),
            Decision::Run {
                action: Action::StartSearch,
                consume: true
            }
        );
        let text = key(KeyCode::KeyL, true, false, true, false);
        assert!(matches!(
            t.decide(&text, Scope::Pane, &ctx),
            Decision::Run {
                action: Action::Text { .. },
                consume: true
            }
        ));
        let plain = key(KeyCode::KeyL, false, false, false, false);
        assert_eq!(t.decide(&plain, Scope::Pane, &ctx), Decision::Pass);
    }

    #[test]
    fn performable_bindings_pass_the_key_on_when_they_cannot_run() {
        let t = Keybinds::defaults(false);
        let alt3 = key(KeyCode::Key3, false, false, true, false);
        let one = Context {
            tabs: 1,
            search_open: false,
        };
        let two = Context {
            tabs: 2,
            search_open: false,
        };
        assert_eq!(
            t.decide(&alt3, Scope::Tabs, &one),
            Decision::Pass,
            "Meta+3 for readline"
        );
        assert_eq!(
            t.decide(&alt3, Scope::Tabs, &two),
            Decision::Run {
                action: Action::GotoTab(3),
                consume: true
            }
        );
        // Without performable:, the key is taken even with one tab.
        let t = Keybinds::build(
            &[
                "ctrl+shift+1=goto_tab:1".into(),
                "unconsumed:ctrl+shift+n=new_tab".into(),
            ],
            false,
        );
        let k = key(KeyCode::Key1, true, true, false, false);
        assert_eq!(
            t.decide(&k, Scope::Tabs, &one),
            Decision::Run {
                action: Action::GotoTab(1),
                consume: true
            }
        );
        // unconsumed: runs and lets the program have the key too.
        assert_eq!(
            t.decide(
                &key(KeyCode::KeyN, true, true, false, false),
                Scope::Tabs,
                &one
            ),
            Decision::Run {
                action: Action::NewTab,
                consume: false
            }
        );
    }

    #[test]
    fn search_stepping_is_taken_only_while_the_bar_is_open() {
        let t = Keybinds::build(
            &[
                "ctrl+n=navigate_search:next".into(),
                r"ctrl+alt+x=text:x".into(),
            ],
            true,
        );
        let ctrl_n = key(KeyCode::KeyN, true, false, false, false);
        let closed = Context {
            tabs: 1,
            search_open: false,
        };
        let open = Context {
            tabs: 1,
            search_open: true,
        };
        assert_eq!(
            t.decide(&ctrl_n, Scope::Pane, &closed),
            Decision::Pass,
            "readline's next line"
        );
        assert_eq!(
            t.decide(&ctrl_n, Scope::Pane, &open),
            Decision::Run {
                action: Action::NavigateSearch { next: true },
                consume: true
            }
        );
        let cmd_g = key(KeyCode::KeyG, false, false, false, true);
        assert_eq!(t.decide(&cmd_g, Scope::Pane, &closed), Decision::Pass);
        // The open bar has the keyboard: no typing into the program.
        let text = key(KeyCode::KeyX, true, false, true, false);
        assert!(matches!(
            t.decide(&text, Scope::Pane, &closed),
            Decision::Run { .. }
        ));
        assert_eq!(t.decide(&text, Scope::Pane, &open), Decision::Pass);
    }

    #[test]
    fn a_reload_replaces_the_table() {
        // What a settings generation bump does: the widgets rebuild from
        // the new lines.
        let before = Keybinds::for_lines(&[]);
        let after = Keybinds::for_lines(&["ctrl+shift+t=unbind".into()]);
        let k = key(KeyCode::KeyT, true, true, false, false);
        assert!(before.lookup(&k).is_some());
        assert!(after.lookup(&k).is_none());
        assert_ne!(before, after);
    }

    #[test]
    fn the_table_describes_itself() {
        let t = Keybinds::defaults(false);
        let rows = t.describe(false);
        let next = rows.iter().find(|(_, name, _)| name == "next_tab").unwrap();
        assert_eq!(next.0, "Next tab");
        assert_eq!(next.2, "Ctrl+Tab, Ctrl+PageDown");
        let goto = rows
            .iter()
            .find(|(_, name, _)| name == "goto_tab:1")
            .unwrap();
        assert_eq!(goto.2, "Alt+1");
        let mac = Keybinds::defaults(true).describe(true);
        let search = mac
            .iter()
            .find(|(_, name, _)| name == "start_search")
            .unwrap();
        assert_eq!(search.2, "Cmd+F");
        assert_eq!(
            chord_label(&trig("ctrl+alt+shift+cmd+left"), true),
            "Ctrl+Opt+Shift+Cmd+\u{2190}"
        );
        assert_eq!(chord_label(&trig("super+k"), false), "Super+K");
    }
}
