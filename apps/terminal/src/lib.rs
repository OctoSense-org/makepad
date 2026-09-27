//! terminal: terminal emulator for Makepad. See Cargo.toml for provenance.

pub mod ai;
pub mod module;
pub mod pty;
// The platform's own module, not a second copy by source path: `Cx::pre_start`
// records there that this executable is its own PTY helper.
#[cfg(target_os = "macos")]
pub use makepad_widgets::makepad_platform::os::apple::pty_spawn;
pub mod session;
pub mod term;
pub mod widget;

pub use module::TERMINAL_MODULE;
