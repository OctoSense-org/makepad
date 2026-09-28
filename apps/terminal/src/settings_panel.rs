//! The settings panel's model: which rows it shows, how each shows its
//! value and how Left/Right (or the row's arrows) change it. Drawing and
//! input live with the tabs (`crate::tabs`); every change goes through
//! `settings::update`, so it is written to the file and reaches every tab.
//!
//! Nothing here needs typing: free-form values (`TERM`, the shell) are
//! chosen from lists — the usual terminfo names, `/etc/shells` — and a
//! value set by hand in the file stays selectable.

use crate::settings::{BellStyle, CursorShape, NewTabCwd, Settings, TabBar, TabTitle, THEME_DESKTOP};
use crate::settings::{FONT_SIZE_RANGE, LINE_HEIGHT_RANGE};
use crate::themes;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    Theme,
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
}

/// The panel, top to bottom: section titles and their rows.
pub const SECTIONS: &[(&str, &[Row])] = &[
    (
        "Appearance",
        &[Row::Theme, Row::Opacity, Row::FontSize, Row::LineHeight, Row::CursorShape, Row::CursorBlink],
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

fn yes(b: bool) -> String {
    if b { "On" } else { "Off" }.into()
}

impl Row {
    pub fn label(self) -> &'static str {
        match self {
            Row::Theme => "Theme",
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
        matches!(
            self,
            Row::CursorBlink | Row::OptionAsMeta | Row::CopyOnSelect | Row::LoginShell | Row::ConfirmClose
        )
    }

    pub fn value(self, s: &Settings) -> String {
        match self {
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
        let mut s = s.clone();
        let up = dir > 0;
        match self {
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
        assert_eq!(rows.len(), 17);
        for (i, row) in rows.iter().enumerate() {
            assert!(!rows[i + 1..].contains(row), "{row:?} twice");
        }
    }

    #[test]
    fn stepping_changes_every_row_and_comes_back() {
        let base = Settings::default();
        for row in rows() {
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
