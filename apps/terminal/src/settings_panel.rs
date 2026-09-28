//! The settings panel's model: which rows it shows, how each shows its
//! value and how Left/Right (or the row's arrows) change it. Drawing and
//! input live with the tabs (`crate::tabs`); every change goes through
//! `settings::update`, so it is written to the file and reaches every tab.
//!
//! Nothing here needs typing: free-form values (`TERM`, the shell) are
//! chosen from lists — the usual terminfo names, `/etc/shells` — and a
//! value set by hand in the file stays selectable.

use crate::settings::{BellStyle, CursorShape, NewTabCwd, Settings, TabBar, TabTitle, THEME_DESKTOP};
use crate::settings::{CJK_AUTO, CJK_NONE, DEFAULT_FONT};
use crate::settings::{FONT_SIZE_RANGE, LINE_HEIGHT_RANGE};
use crate::themes;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    Theme,
    Font,
    CjkFont,
    Opacity,
    FontSize,
    LineHeight,
    CursorShape,
    CursorBlink,
    Term,
    Scrollback,
    OptionAsMeta,
    Bell,
    CopyOnSelect,
    Shell,
    LoginShell,
    TabBar,
    NewTabDir,
    TabTitle,
    ConfirmClose,
    Profile,
    SaveProfile,
    DeleteProfile,
    ExternalControl,
}

/// What a row is: an on/off switch, a value stepped with the arrows (and,
/// for long lists, chosen from a filtered list), or an action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowKind {
    Toggle,
    Value,
    Action,
}

/// One entry of a row's list: the value it sets, what it shows, a hint.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub value: String,
    pub label: String,
    pub note: &'static str,
}

/// The panel, top to bottom: section titles and their rows.
pub const SECTIONS: &[(&str, &[Row])] = &[
    (
        "Appearance",
        &[
            Row::Theme,
            Row::Font,
            Row::CjkFont,
            Row::FontSize,
            Row::LineHeight,
            Row::Opacity,
            Row::CursorShape,
            Row::CursorBlink,
        ],
    ),
    (
        "Emulation & shell",
        &[
            Row::Term,
            Row::Scrollback,
            Row::OptionAsMeta,
            Row::Bell,
            Row::CopyOnSelect,
            Row::Shell,
            Row::LoginShell,
        ],
    ),
    ("Tabs", &[Row::TabBar, Row::NewTabDir, Row::TabTitle, Row::ConfirmClose]),
    ("Profiles", &[Row::Profile, Row::SaveProfile, Row::DeleteProfile]),
    ("Automation", &[Row::ExternalControl]),
];

/// Every row in panel order (keyboard navigation walks this).
pub fn rows() -> Vec<Row> {
    SECTIONS.iter().flat_map(|(_, rows)| rows.iter().copied()).collect()
}

const TERMS: &[&str] = &["xterm-256color", "xterm", "screen-256color", "tmux-256color", "xterm-kitty", "xterm-ghostty"];
const SCROLLBACK: &[usize] = &[1_000, 5_000, 10_000, 50_000, 100_000, 1_000_000];
const OPACITY: &[Option<f32>] =
    &[None, Some(1.0), Some(0.95), Some(0.9), Some(0.85), Some(0.8), Some(0.75), Some(0.7), Some(0.6), Some(0.5)];

/// The shells a person can pick: the default (`$SHELL`), each login shell
/// the system lists, and whatever the file names now.
pub fn shell_choices(current: &str) -> Vec<String> {
    let mut out = vec![String::new()];
    if let Ok(text) = std::fs::read_to_string("/etc/shells") {
        for line in text.lines().map(str::trim) {
            if line.starts_with('/') && !out.iter().any(|s| s == line) {
                out.push(line.to_owned());
            }
        }
    }
    if !out.iter().any(|s| s == current) {
        out.push(current.to_owned());
    }
    out
}

/// The option after (or before, `dir < 0`) `current` in `options`,
/// wrapping. A current value not in the list steps to the first/last.
fn cycle<T: PartialEq + Clone>(options: &[T], current: &T, dir: i32) -> T {
    let n = options.len() as i32;
    let next = match options.iter().position(|o| o == current) {
        Some(i) => (i as i32 + dir).rem_euclid(n),
        None if dir < 0 => n - 1,
        None => 0,
    };
    options[next as usize].clone()
}

/// "Auto (PingFang SC)": what `auto` resolves to here.
fn auto_cjk_label() -> String {
    match crate::fonts::auto_cjk() {
        Some(family) => format!("Auto ({})", family.name),
        None if !crate::fonts::ready() => "Auto".into(),
        None => "Auto (none found)".into(),
    }
}

fn yes(b: bool) -> String {
    if b { "On" } else { "Off" }.into()
}

/// A list's entries matching `filter` (case-insensitive, anywhere).
pub fn filter_choices(choices: &[Choice], filter: &str) -> Vec<Choice> {
    let filter = filter.trim().to_lowercase();
    choices.iter().filter(|c| filter.is_empty() || c.label.to_lowercase().contains(&filter)).cloned().collect()
}

impl Row {
    pub fn kind(self) -> RowKind {
        match self {
            Row::CursorBlink
            | Row::OptionAsMeta
            | Row::CopyOnSelect
            | Row::LoginShell
            | Row::ConfirmClose
            | Row::ExternalControl => {
                RowKind::Toggle
            }
            Row::SaveProfile | Row::DeleteProfile => RowKind::Action,
            _ => RowKind::Value,
        }
    }

    /// The row's full list, for rows too long to step through: Enter opens
    /// it as a filterable list.
    pub fn choices(self, s: &Settings) -> Option<Vec<Choice>> {
        let fonts = || {
            crate::fonts::families()
                .iter()
                .map(|f| Choice { value: f.name.clone(), label: f.name.clone(), note: if f.monospace { "mono" } else { "" } })
        };
        Some(match self {
            Row::Theme => std::iter::once(Choice { value: THEME_DESKTOP.into(), label: "Desktop".into(), note: "host" })
                .chain(themes::SCHEMES.iter().map(|scheme| Choice {
                    value: scheme.id.into(),
                    label: scheme.name.into(),
                    note: if scheme.light { "light" } else { "dark" },
                }))
                .collect(),
            Row::Font => std::iter::once(Choice { value: String::new(), label: DEFAULT_FONT.into(), note: "bundled" })
                .chain(fonts().filter(|c| c.value != DEFAULT_FONT))
                .collect(),
            Row::CjkFont => [
                Choice { value: CJK_AUTO.into(), label: auto_cjk_label(), note: "" },
                Choice { value: CJK_NONE.into(), label: "None".into(), note: "" },
            ]
            .into_iter()
            .chain(fonts())
            .collect(),
            Row::Shell => shell_choices(&s.shell)
                .into_iter()
                .map(|sh| Choice { label: if sh.is_empty() { "Default ($SHELL)".into() } else { sh.clone() }, value: sh, note: "" })
                .collect(),
            Row::Profile => crate::settings::list_profiles()
                .into_iter()
                .map(|name| Choice { value: name.clone(), label: name, note: "" })
                .collect(),
            _ => return None,
        })
    }

    /// `s` with this row set to `value` (a `Choice::value`).
    pub fn with_value(self, s: &Settings, value: &str) -> Settings {
        let mut s = s.clone();
        match self {
            Row::Theme => s.theme = value.to_owned(),
            Row::Font => s.font_family = value.to_owned(),
            Row::CjkFont => s.cjk_font = value.to_owned(),
            Row::Shell => s.shell = value.to_owned(),
            Row::Profile => {
                if let Some(profile) = crate::settings::load_profile(value) {
                    s = profile;
                }
            }
            _ => {}
        }
        s
    }

    /// The row's current value, as a `Choice::value`.
    pub fn current(self, s: &Settings) -> String {
        match self {
            Row::Theme => s.theme.clone(),
            Row::Font => s.font_family.clone(),
            Row::CjkFont => s.cjk_font.clone(),
            Row::Shell => s.shell.clone(),
            Row::Profile => s.profile.clone(),
            _ => String::new(),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Row::Theme => "Theme",
            Row::Font => "Font",
            Row::CjkFont => "CJK font",
            Row::Profile => "Profile",
            Row::SaveProfile => "Save as profile\u{2026}",
            Row::DeleteProfile => "Delete profile",
            Row::ExternalControl => "Allow terminal-ctl",
            Row::Opacity => "Background opacity",
            Row::FontSize => "Font size",
            Row::LineHeight => "Line height",
            Row::CursorShape => "Cursor",
            Row::CursorBlink => "Cursor blink",
            Row::Term => "TERM",
            Row::Scrollback => "Scrollback",
            Row::OptionAsMeta => "Option as Meta",
            Row::Bell => "Bell",
            Row::CopyOnSelect => "Copy on select",
            Row::Shell => "Shell",
            Row::LoginShell => "Login shell",
            Row::TabBar => "Tab bar",
            Row::NewTabDir => "New tab opens in",
            Row::TabTitle => "Tab title",
            Row::ConfirmClose => "Confirm closing a busy tab",
        }
    }

    /// The row applies to shells started after the change only.
    pub fn new_shells_only(self) -> bool {
        matches!(self, Row::Term | Row::Shell | Row::LoginShell)
    }

    /// An on/off row: Enter and a click flip it.
    pub fn is_toggle(self) -> bool {
        self.kind() == RowKind::Toggle
    }

    pub fn value(self, s: &Settings) -> String {
        match self {
            Row::Font => if s.font_family.is_empty() { DEFAULT_FONT.into() } else { s.font_family.clone() },
            Row::CjkFont => match s.cjk_font.as_str() {
                CJK_AUTO => auto_cjk_label(),
                CJK_NONE => "None".into(),
                name => name.to_owned(),
            },
            Row::Profile => {
                if s.profile.is_empty() {
                    "None".into()
                } else if crate::settings::load_profile(&s.profile)
                    .is_some_and(|saved| saved == Settings { profile: s.profile.clone(), ..s.clone() })
                {
                    s.profile.clone()
                } else {
                    format!("{} (changed)", s.profile)
                }
            }
            Row::SaveProfile | Row::DeleteProfile => String::new(),
            Row::ExternalControl => yes(s.external_control),
            Row::Theme => {
                if s.theme == THEME_DESKTOP {
                    "Desktop".into()
                } else {
                    themes::find(&s.theme).map_or_else(|| s.theme.clone(), |scheme| scheme.name.to_owned())
                }
            }
            Row::Opacity => s.background_opacity.map_or("Host default".into(), |a| format!("{:.0}%", a * 100.0)),
            Row::FontSize => format!("{}", s.font_size),
            Row::LineHeight => format!("{:.1}", s.line_height),
            Row::CursorShape => match s.cursor_shape {
                CursorShape::Block => "Block",
                CursorShape::Bar => "Bar",
                CursorShape::Underline => "Underline",
            }
            .into(),
            Row::CursorBlink => yes(s.cursor_blink),
            Row::Term => s.term.clone(),
            Row::Scrollback => format!("{} lines", s.scrollback_lines),
            Row::OptionAsMeta => yes(s.option_as_meta),
            Row::Bell => match s.bell {
                BellStyle::Visual => "Flash",
                BellStyle::None => "Off",
            }
            .into(),
            Row::CopyOnSelect => yes(s.copy_on_select),
            Row::Shell => {
                if s.shell.is_empty() {
                    "Default ($SHELL)".into()
                } else {
                    s.shell.clone()
                }
            }
            Row::LoginShell => yes(s.login_shell),
            Row::TabBar => match s.tab_bar {
                TabBar::Auto => "With 2+ tabs",
                TabBar::Always => "Always",
            }
            .into(),
            Row::NewTabDir => match s.new_tab_cwd {
                NewTabCwd::Inherit => "Current directory",
                NewTabCwd::Home => "Home",
            }
            .into(),
            Row::TabTitle => match s.tab_title {
                TabTitle::Program => "Program",
                TabTitle::Directory => "Directory",
            }
            .into(),
            Row::ConfirmClose => yes(s.confirm_close_running),
        }
    }

    /// `s` with this row moved one step (`dir` is -1 or +1).
    pub fn step(self, s: &Settings, dir: i32) -> Settings {
        if matches!(self, Row::Font | Row::CjkFont | Row::Profile) {
            let values: Vec<String> = self.choices(s).unwrap_or_default().into_iter().map(|c| c.value).collect();
            if values.is_empty() {
                return s.clone();
            }
            return self.with_value(s, &cycle(&values, &self.current(s), dir));
        }
        let mut s = s.clone();
        let up = dir > 0;
        match self {
            Row::Font | Row::CjkFont | Row::Profile | Row::SaveProfile | Row::DeleteProfile => {}
            Row::ExternalControl => s.external_control = !s.external_control,
            Row::Theme => {
                let mut ids = vec![THEME_DESKTOP];
                ids.extend(themes::SCHEMES.iter().map(|scheme| scheme.id));
                s.theme = cycle(&ids, &s.theme.as_str(), dir).to_owned();
            }
            Row::Opacity => s.background_opacity = cycle(OPACITY, &s.background_opacity, dir),
            Row::FontSize => {
                let v = s.font_size + if up { 0.5 } else { -0.5 };
                s.font_size = v.clamp(FONT_SIZE_RANGE.0, FONT_SIZE_RANGE.1);
            }
            Row::LineHeight => {
                let v = ((s.line_height * 10.0).round() + if up { 1.0 } else { -1.0 }) / 10.0;
                s.line_height = v.clamp(LINE_HEIGHT_RANGE.0, LINE_HEIGHT_RANGE.1);
            }
            Row::CursorShape => {
                s.cursor_shape = cycle(&[CursorShape::Block, CursorShape::Bar, CursorShape::Underline], &s.cursor_shape, dir)
            }
            Row::CursorBlink => s.cursor_blink = !s.cursor_blink,
            Row::Term => {
                let mut terms: Vec<String> = TERMS.iter().map(|t| (*t).to_owned()).collect();
                if !terms.contains(&s.term) {
                    terms.insert(0, s.term.clone());
                }
                s.term = cycle(&terms, &s.term, dir);
            }
            Row::Scrollback => {
                let near = SCROLLBACK.iter().copied().min_by_key(|v| v.abs_diff(s.scrollback_lines)).unwrap_or(10_000);
                s.scrollback_lines = if near == s.scrollback_lines { cycle(SCROLLBACK, &near, dir) } else { near };
            }
            Row::OptionAsMeta => s.option_as_meta = !s.option_as_meta,
            Row::Bell => s.bell = cycle(&[BellStyle::Visual, BellStyle::None], &s.bell, dir),
            Row::CopyOnSelect => s.copy_on_select = !s.copy_on_select,
            Row::Shell => s.shell = cycle(&shell_choices(&s.shell), &s.shell, dir),
            Row::LoginShell => s.login_shell = !s.login_shell,
            Row::TabBar => s.tab_bar = cycle(&[TabBar::Auto, TabBar::Always], &s.tab_bar, dir),
            Row::NewTabDir => s.new_tab_cwd = cycle(&[NewTabCwd::Inherit, NewTabCwd::Home], &s.new_tab_cwd, dir),
            Row::TabTitle => s.tab_title = cycle(&[TabTitle::Program, TabTitle::Directory], &s.tab_title, dir),
            Row::ConfirmClose => s.confirm_close_running = !s.confirm_close_running,
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_is_listed_once() {
        let rows = rows();
        assert_eq!(rows.len(), 23);
        for (i, row) in rows.iter().enumerate() {
            assert!(!rows[i + 1..].contains(row), "{row:?} twice");
        }
    }

    #[test]
    fn stepping_changes_every_row_and_comes_back() {
        let base = Settings::default();
        for row in rows() {
            if matches!(row.kind(), RowKind::Action) || matches!(row, Row::Font | Row::CjkFont | Row::Profile) {
                continue; // depend on the machine's fonts and saved profiles
            }
            let next = row.step(&base, 1);
            assert_ne!(next, base, "{row:?} did not change");
            assert_eq!(row.step(&next, -1), base, "{row:?} did not step back");
            assert!(!row.value(&next).is_empty());
        }
    }

    #[test]
    fn the_theme_walks_desktop_then_every_scheme() {
        let mut s = Settings::default();
        let mut seen = vec![s.theme.clone()];
        for _ in 0..themes::SCHEMES.len() {
            s = Row::Theme.step(&s, 1);
            seen.push(s.theme.clone());
        }
        assert_eq!(Row::Theme.step(&s, 1).theme, THEME_DESKTOP, "wraps");
        assert_eq!(seen.len(), themes::SCHEMES.len() + 1);
        assert!(seen.iter().skip(1).all(|id| themes::find(id).is_some()));
        assert_eq!(Row::Theme.value(&Settings { theme: "tokyo-night".into(), ..Settings::default() }), "Tokyo Night");
    }

    #[test]
    fn numbers_stay_in_range() {
        let mut s = Settings { font_size: FONT_SIZE_RANGE.1, line_height: LINE_HEIGHT_RANGE.0, ..Settings::default() };
        s = Row::FontSize.step(&s, 1);
        s = Row::LineHeight.step(&s, -1);
        assert_eq!(s.font_size, FONT_SIZE_RANGE.1);
        assert_eq!(s.line_height, LINE_HEIGHT_RANGE.0);
        assert_eq!(Row::LineHeight.step(&s, 1).line_height, 1.1);
    }

    #[test]
    fn a_hand_set_value_stays_selectable() {
        let s = Settings { term: "wezterm".into(), scrollback_lines: 7_000, ..Settings::default() };
        assert_eq!(Row::Term.step(&s, 1).term, "xterm-256color", "a hand-set TERM leads the list");
        assert_eq!(Row::Term.value(&s), "wezterm");
        assert_eq!(Row::Scrollback.step(&s, 1).scrollback_lines, 5_000, "snaps to the nearest step first");
        assert!(shell_choices("/opt/fish").contains(&"/opt/fish".to_owned()));
        assert_eq!(shell_choices("")[0], "", "the default comes first");
    }
}
