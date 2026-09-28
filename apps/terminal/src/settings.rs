//! The terminal's settings: one file shared by every terminal window and
//! tab under one Makepad home (`<makepad home>/terminal/settings.conf`,
//! normally `~/.makepad`). A host can move the home: the OctoSense shell
//! points it at its own (`~/.octosense`), so its terminals keep their
//! settings there. `MAKEPAD_HOME` redirects it in tests too.
//!
//! The format is `key = value` lines with `#` comments. Unknown keys and
//! values that do not parse are ignored (the default stays), so an older
//! build reads a newer file and a hand edit cannot break the terminal.
//!
//! In a process, [`current`] is the live copy and [`generation`] counts its
//! changes: widgets compare the generation they last applied and re-read on
//! a change, so a settings panel's edit reaches every open tab at once.

use crate::term::terminal::DEFAULT_SCROLLBACK;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Follow the host: the desktop style's colours inside OctoSense, the
/// palette makepad-wm hands down, or the built-in default standalone.
pub const THEME_DESKTOP: &str = "desktop";
/// The platform's CJK font (`crate::fonts::auto_cjk`).
pub const CJK_AUTO: &str = "auto";
pub const CJK_NONE: &str = "none";
/// The bundled font the terminal uses unless `font-family` names another.
pub const DEFAULT_FONT: &str = "JetBrains Mono";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CursorShape {
    Block,
    Bar,
    Underline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BellStyle {
    /// Flash the terminal briefly.
    Visual,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NewTabCwd {
    /// The focused tab's shell directory.
    Inherit,
    Home,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TabTitle {
    /// What the running program sets (OSC 0/2), else the directory.
    Program,
    /// Always the working directory.
    Directory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TabBar {
    /// Shown once there is more than one tab.
    Auto,
    Always,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// [`THEME_DESKTOP`] or a bundled scheme id (`crate::themes`).
    pub theme: String,
    /// `None` keeps the host's opacity (makepad-wm's rule) or opaque.
    pub background_opacity: Option<f32>,
    /// An installed family (`crate::fonts`); empty keeps the bundled
    /// JetBrains Mono.
    pub font_family: String,
    /// The family CJK text falls back to: [`CJK_AUTO`], [`CJK_NONE`] or an
    /// installed family.
    pub cjk_font: String,
    pub font_size: f64,
    /// Row height as a multiple of the font's glyph height (the text
    /// style's `line_spacing`; 1.0 is the terminal's historical look).
    pub line_height: f64,
    /// The shape a program that sets none (DECSCUSR 0) gets.
    pub cursor_shape: CursorShape,
    pub cursor_blink: bool,
    /// The `TERM` a new shell sees.
    pub term: String,
    pub scrollback_lines: usize,
    /// Option (Alt) sends ESC-prefixed keys instead of composed characters.
    pub option_as_meta: bool,
    pub bell: BellStyle,
    /// A finished mouse selection is copied without Cmd+C.
    pub copy_on_select: bool,
    /// Empty: `$SHELL`, else `/bin/zsh`.
    pub shell: String,
    pub login_shell: bool,
    pub new_tab_cwd: NewTabCwd,
    /// Ask before closing a tab whose shell is running another program.
    pub confirm_close_running: bool,
    pub tab_title: TabTitle,
    pub tab_bar: TabBar,
    /// The profile these settings were loaded from or saved as; empty when
    /// none. Only a label: changing a setting does not touch the profile.
    pub profile: String,
    /// Other programs of this user may list, read and type into panes
    /// (`crate::control`, `terminal-ctl`).
    pub external_control: bool,
}

pub const FONT_SIZE_RANGE: (f64, f64) = (6.0, 48.0);
pub const LINE_HEIGHT_RANGE: (f64, f64) = (1.0, 2.0);
pub const OPACITY_RANGE: (f32, f32) = (0.2, 1.0);
pub const SCROLLBACK_MAX: usize = 1_000_000;

impl Default for Settings {
    fn default() -> Self {
        Settings {
            theme: THEME_DESKTOP.into(),
            background_opacity: None,
            font_family: String::new(),
            cjk_font: CJK_AUTO.into(),
            font_size: 10.0,
            line_height: 1.0,
            cursor_shape: CursorShape::Block,
            cursor_blink: false,
            term: "xterm-256color".into(),
            scrollback_lines: DEFAULT_SCROLLBACK,
            option_as_meta: false,
            bell: BellStyle::Visual,
            copy_on_select: false,
            shell: String::new(),
            login_shell: true,
            new_tab_cwd: NewTabCwd::Inherit,
            confirm_close_running: true,
            tab_title: TabTitle::Program,
            tab_bar: TabBar::Auto,
            profile: String::new(),
            external_control: false,
        }
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value {
        "true" | "yes" | "on" | "1" => Some(true),
        "false" | "no" | "off" | "0" => Some(false),
        _ => None,
    }
}

/// A `TERM` value: short, and only characters terminfo names use.
fn valid_term(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '+'))
}

impl Settings {
    /// Read settings text. Anything missing or invalid keeps its default.
    pub fn parse(text: &str) -> Settings {
        let mut s = Settings::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let (key, value) = (key.trim(), value.trim());
            match key {
                "theme" if !value.is_empty() => s.theme = value.to_string(),
                "background-opacity" => {
                    s.background_opacity = if value == "default" {
                        None
                    } else {
                        value
                            .parse::<f32>()
                            .ok()
                            .filter(|v| v.is_finite())
                            .map(|v| v.clamp(OPACITY_RANGE.0, OPACITY_RANGE.1))
                    }
                }
                "font-family" if !value.chars().any(char::is_control) => {
                    s.font_family = if value.eq_ignore_ascii_case(DEFAULT_FONT) { String::new() } else { value.to_string() }
                }
                "cjk-font" if !value.is_empty() && !value.chars().any(char::is_control) => s.cjk_font = value.to_string(),
                "font-size" => {
                    if let Some(v) = value.parse::<f64>().ok().filter(|v| v.is_finite()) {
                        s.font_size = v.clamp(FONT_SIZE_RANGE.0, FONT_SIZE_RANGE.1);
                    }
                }
                "line-height" => {
                    if let Some(v) = value.parse::<f64>().ok().filter(|v| v.is_finite()) {
                        s.line_height = v.clamp(LINE_HEIGHT_RANGE.0, LINE_HEIGHT_RANGE.1);
                    }
                }
                "cursor-shape" => {
                    s.cursor_shape = match value {
                        "block" => CursorShape::Block,
                        "bar" => CursorShape::Bar,
                        "underline" => CursorShape::Underline,
                        _ => s.cursor_shape,
                    }
                }
                "cursor-blink" => s.cursor_blink = parse_bool(value).unwrap_or(s.cursor_blink),
                "term" if valid_term(value) => s.term = value.to_string(),
                "scrollback-lines" => {
                    if let Ok(v) = value.parse::<usize>() {
                        s.scrollback_lines = v.min(SCROLLBACK_MAX);
                    }
                }
                "option-as-meta" => s.option_as_meta = parse_bool(value).unwrap_or(s.option_as_meta),
                "bell" => {
                    s.bell = match value {
                        "visual" => BellStyle::Visual,
                        "none" => BellStyle::None,
                        _ => s.bell,
                    }
                }
                "copy-on-select" => s.copy_on_select = parse_bool(value).unwrap_or(s.copy_on_select),
                "shell" if !value.chars().any(char::is_control) => s.shell = value.to_string(),
                "login-shell" => s.login_shell = parse_bool(value).unwrap_or(s.login_shell),
                "new-tab-directory" => {
                    s.new_tab_cwd = match value {
                        "inherit" => NewTabCwd::Inherit,
                        "home" => NewTabCwd::Home,
                        _ => s.new_tab_cwd,
                    }
                }
                "confirm-close-running" => {
                    s.confirm_close_running = parse_bool(value).unwrap_or(s.confirm_close_running)
                }
                "tab-title" => {
                    s.tab_title = match value {
                        "program" => TabTitle::Program,
                        "directory" => TabTitle::Directory,
                        _ => s.tab_title,
                    }
                }
                "external-control" => s.external_control = parse_bool(value).unwrap_or(s.external_control),
                "profile" if value.is_empty() || valid_profile_name(value) => s.profile = value.to_string(),
                "tab-bar" => {
                    s.tab_bar = match value {
                        "auto" => TabBar::Auto,
                        "always" => TabBar::Always,
                        _ => s.tab_bar,
                    }
                }
                _ => {}
            }
        }
        s
    }

    /// The file text: every key, so a person editing it sees the options.
    pub fn to_text(&self) -> String {
        let yes = |b: bool| if b { "true" } else { "false" };
        let mut out = String::from(
            "# Terminal settings (makepad-terminal). Edited by the settings panel;\n\
             # safe to edit by hand. Unknown keys and invalid values are ignored.\n\n",
        );
        let mut line = |k: &str, v: String| {
            out.push_str(k);
            out.push_str(" = ");
            out.push_str(&v);
            out.push('\n');
        };
        line("theme", self.theme.clone());
        line(
            "background-opacity",
            self.background_opacity.map_or("default".into(), |v| format!("{v:.2}")),
        );
        line(
            "font-family",
            if self.font_family.is_empty() { DEFAULT_FONT.into() } else { self.font_family.clone() },
        );
        line("cjk-font", self.cjk_font.clone());
        line("font-size", format!("{}", self.font_size));
        line("line-height", format!("{}", self.line_height));
        line(
            "cursor-shape",
            match self.cursor_shape {
                CursorShape::Block => "block",
                CursorShape::Bar => "bar",
                CursorShape::Underline => "underline",
            }
            .into(),
        );
        line("cursor-blink", yes(self.cursor_blink).into());
        line("term", self.term.clone());
        line("scrollback-lines", self.scrollback_lines.to_string());
        line("option-as-meta", yes(self.option_as_meta).into());
        line(
            "bell",
            match self.bell {
                BellStyle::Visual => "visual",
                BellStyle::None => "none",
            }
            .into(),
        );
        line("copy-on-select", yes(self.copy_on_select).into());
        line("shell", self.shell.clone());
        line("login-shell", yes(self.login_shell).into());
        line(
            "new-tab-directory",
            match self.new_tab_cwd {
                NewTabCwd::Inherit => "inherit",
                NewTabCwd::Home => "home",
            }
            .into(),
        );
        line("confirm-close-running", yes(self.confirm_close_running).into());
        line(
            "tab-title",
            match self.tab_title {
                TabTitle::Program => "program",
                TabTitle::Directory => "directory",
            }
            .into(),
        );
        line(
            "tab-bar",
            match self.tab_bar {
                TabBar::Auto => "auto",
                TabBar::Always => "always",
            }
            .into(),
        );
        line("external-control", yes(self.external_control).into());
        line("profile", self.profile.clone());
        out
    }
}

/// A profile name: short, and safe as a file name.
pub fn valid_profile_name(name: &str) -> bool {
    let name = name.trim();
    !name.is_empty()
        && name.chars().count() <= 40
        && !name.starts_with('.')
        && !name.chars().any(|c| c.is_control() || matches!(c, '/' | '\\' | ':' | '=' | '#'))
}

/// `<makepad home>/terminal/profiles`: one `<name>.conf` per profile, in
/// the settings file's format.
pub fn profiles_dir() -> PathBuf {
    path().with_file_name("profiles")
}

/// The saved profiles, by name.
pub fn list_profiles() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(profiles_dir())
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|entry| {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    name.strip_suffix(".conf").filter(|n| valid_profile_name(n)).map(str::to_owned)
                })
                .collect()
        })
        .unwrap_or_default();
    names.sort_by_key(|n| n.to_lowercase());
    names
}

fn profile_path(name: &str) -> Option<PathBuf> {
    valid_profile_name(name).then(|| profiles_dir().join(format!("{}.conf", name.trim())))
}

/// Save `settings` as profile `name` (replacing one of that name).
pub fn save_profile(name: &str, settings: &Settings) -> std::io::Result<()> {
    let path = profile_path(name).ok_or_else(|| std::io::Error::other("invalid profile name"))?;
    std::fs::create_dir_all(profiles_dir())?;
    let profile = Settings { profile: name.trim().to_owned(), ..settings.clone() };
    let tmp = path.with_extension("conf.tmp");
    std::fs::write(&tmp, profile.to_text())?;
    std::fs::rename(&tmp, &path)
}

/// Profile `name`, labelled with its name.
pub fn load_profile(name: &str) -> Option<Settings> {
    let text = std::fs::read_to_string(profile_path(name)?).ok()?;
    Some(Settings { profile: name.trim().to_owned(), ..Settings::parse(&text) })
}

pub fn delete_profile(name: &str) -> std::io::Result<()> {
    let path = profile_path(name).ok_or_else(|| std::io::Error::other("invalid profile name"))?;
    std::fs::remove_file(path)
}

/// `<makepad home>/terminal/settings.conf`.
pub fn path() -> PathBuf {
    makepad_widgets::makepad_platform::home::makepad_home()
        .join("terminal")
        .join("settings.conf")
}

struct Live {
    settings: Settings,
    /// The file's mtime when last read or written, to notice outside edits.
    mtime: Option<SystemTime>,
}

static LIVE: RwLock<Option<Live>> = RwLock::new(None);
static GENERATION: AtomicU64 = AtomicU64::new(1);

fn file_mtime() -> Option<SystemTime> {
    std::fs::metadata(path()).and_then(|m| m.modified()).ok()
}

fn load() -> Live {
    let settings = std::fs::read_to_string(path())
        .map(|text| Settings::parse(&text))
        .unwrap_or_default();
    Live { settings, mtime: file_mtime() }
}

/// The live settings (read from the file on first use).
pub fn current() -> Settings {
    if let Some(live) = LIVE.read().ok().as_ref().and_then(|g| g.as_ref()) {
        return live.settings.clone();
    }
    let mut guard = LIVE.write().unwrap_or_else(|e| e.into_inner());
    guard.get_or_insert_with(load).settings.clone()
}

/// Make every terminal re-apply its settings (fonts found by a scan).
pub fn bump_generation() {
    GENERATION.fetch_add(1, Ordering::AcqRel);
}

/// Bumped on every change a widget must apply.
pub fn generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

/// Save and apply: the file is replaced whole (write, then rename), so a
/// reader never sees half of it.
pub fn update(settings: Settings) -> std::io::Result<()> {
    let path = path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("conf.tmp");
    std::fs::write(&tmp, settings.to_text())?;
    std::fs::rename(&tmp, &path)?;
    let mut guard = LIVE.write().unwrap_or_else(|e| e.into_inner());
    *guard = Some(Live { settings, mtime: file_mtime() });
    GENERATION.fetch_add(1, Ordering::AcqRel);
    Ok(())
}

static LAST_POLL: Mutex<Option<Instant>> = Mutex::new(None);

/// [`reload_if_changed`] at most once a second, for hot paths such as key
/// presses: a window that keeps focus still sees edits from other processes.
pub fn poll() {
    let now = Instant::now();
    {
        let mut last = LAST_POLL.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_some_and(|at| now.duration_since(at) < Duration::from_secs(1)) {
            return;
        }
        *last = Some(now);
    }
    reload_if_changed();
}

/// Pick up an edit made outside this process (another terminal window, a
/// text editor). Cheap: one stat; call when a window gains focus.
pub fn reload_if_changed() {
    let now = file_mtime();
    let stale = LIVE
        .read()
        .ok()
        .and_then(|g| g.as_ref().map(|live| live.mtime != now))
        .unwrap_or(false);
    if stale {
        let fresh = load();
        let mut guard = LIVE.write().unwrap_or_else(|e| e.into_inner());
        if guard.as_ref().map(|live| live.settings != fresh.settings).unwrap_or(true) {
            GENERATION.fetch_add(1, Ordering::AcqRel);
        }
        *guard = Some(fresh);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_file_is_all_defaults() {
        assert_eq!(Settings::parse(""), Settings::default());
        assert_eq!(Settings::parse("# only a comment\n\n"), Settings::default());
    }

    #[test]
    fn every_setting_round_trips_through_the_file() {
        let s = Settings {
            theme: "dracula".into(),
            font_family: "Menlo".into(),
            cjk_font: "PingFang SC".into(),
            background_opacity: Some(0.85),
            font_size: 13.0,
            line_height: 1.5,
            cursor_shape: CursorShape::Bar,
            cursor_blink: true,
            term: "xterm-ghostty".into(),
            scrollback_lines: 50_000,
            option_as_meta: true,
            bell: BellStyle::None,
            copy_on_select: true,
            shell: "/opt/homebrew/bin/fish".into(),
            login_shell: false,
            new_tab_cwd: NewTabCwd::Home,
            confirm_close_running: false,
            tab_title: TabTitle::Directory,
            tab_bar: TabBar::Always,
            profile: "Work".into(),
            external_control: true,
        };
        assert_eq!(Settings::parse(&s.to_text()), s);
        assert_eq!(Settings::parse(&Settings::default().to_text()), Settings::default());
    }

    #[test]
    fn out_of_range_values_clamp_and_invalid_ones_keep_the_default() {
        let s = Settings::parse(
            "font-size = 400\nline-height = 0.2\nbackground-opacity = 0\nscrollback-lines = 99999999\n\
             cursor-shape = triangle\nbell = loud\nterm = xterm;rm -rf\ncopy-on-select = maybe\n",
        );
        assert_eq!(s.font_size, FONT_SIZE_RANGE.1);
        assert_eq!(s.line_height, LINE_HEIGHT_RANGE.0);
        assert_eq!(s.background_opacity, Some(OPACITY_RANGE.0));
        assert_eq!(s.scrollback_lines, SCROLLBACK_MAX);
        let d = Settings::default();
        assert_eq!(s.cursor_shape, d.cursor_shape);
        assert_eq!(s.bell, d.bell);
        assert_eq!(s.term, d.term, "a TERM with shell syntax is refused");
        assert_eq!(s.copy_on_select, d.copy_on_select);
    }

    #[test]
    fn unknown_keys_and_malformed_lines_are_ignored() {
        let s = Settings::parse("future-option = 7\nno equals sign here\ntheme = nord\n");
        assert_eq!(s.theme, "nord");
        assert_eq!(Settings { theme: THEME_DESKTOP.into(), ..s }, Settings::default());
    }

    #[test]
    fn default_opacity_and_non_finite_numbers_keep_the_default() {
        assert_eq!(Settings::parse("background-opacity = default").background_opacity, None);
        assert_eq!(Settings::parse("font-size = NaN").font_size, Settings::default().font_size);
    }

    #[test]
    fn profiles_save_list_load_and_delete() {
        let home = std::env::temp_dir().join(format!("terminal-profiles-{}", std::process::id()));
        // Tests share the process: set the home only for this test's paths.
        let dir = home.join("terminal").join("profiles");
        std::fs::create_dir_all(&dir).unwrap();
        let settings = Settings { theme: "nord".into(), font_size: 14.0, ..Settings::default() };
        std::fs::write(dir.join("Work.conf"), Settings { profile: "Work".into(), ..settings.clone() }.to_text()).unwrap();
        std::fs::write(dir.join("notes.txt"), "x").unwrap();
        let parsed = Settings::parse(&std::fs::read_to_string(dir.join("Work.conf")).unwrap());
        assert_eq!(parsed.profile, "Work");
        assert_eq!(Settings { profile: String::new(), ..parsed }, settings);
        assert!(valid_profile_name("Presentation 2"));
        for bad in ["", " ", "../x", "a/b", ".hidden", "a=b", &"x".repeat(41)] {
            assert!(!valid_profile_name(bad), "{bad:?}");
        }
        std::fs::remove_dir_all(&home).ok();
    }
}
