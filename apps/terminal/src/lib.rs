//! terminal: terminal emulator for Makepad. See Cargo.toml for provenance.

pub mod ai;
pub mod module;
pub mod pty;
pub mod session;
pub mod term;
pub mod widget;

pub use module::TERMINAL_MODULE;
