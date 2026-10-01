//! terminal: terminal emulator for Makepad. See Cargo.toml for provenance.

pub mod agent;
pub mod ai;
pub mod cell_glyph;
pub mod contrast;
pub mod control;
pub mod fonts;
pub mod gesture;
pub mod kitty_input;
pub mod links;
pub mod module;
pub mod panes;
pub mod procinfo;
#[cfg(test)]
mod robustness;
pub mod pty;
pub mod search;
// The platform's own module, not a second copy by source path: `Cx::pre_start`
// records there that this executable is its own PTY helper.
#[cfg(target_os = "macos")]
pub use makepad_widgets::makepad_platform::os::apple::pty_spawn;

/// The shell's home: the app's own data directory where the platform gives
/// one (Android, iOS, OpenHarmony: `HOME` there is not the app's to write),
/// else `HOME`.
pub fn home_dir() -> Option<std::path::PathBuf> {
    makepad_widgets::makepad_platform::home::platform_data_dir()
        .or_else(|| std::env::var_os("HOME").filter(|h| !h.is_empty()).map(std::path::PathBuf::from))
}
pub mod session;
pub mod settings;
pub mod settings_panel;
pub mod sprites;
pub mod sync_output;
pub mod tabs;
pub mod term;
pub mod text_run;
pub mod themes;
pub mod widget;

pub use module::TERMINAL_MODULE;
