//! The host side's hooks for hosted child processes, as the Vulkan
//! backend (`vulkan_android.rs`) calls them.
//!
//! This fork does not host app processes on Android yet: the runtime that
//! does (makepad/makepad `work`'s `android_hosted.rs`, with its launcher,
//! relays and shared frames) replaces this file when that lands. Until then
//! no process ever takes the host sync role, so there are no frames to
//! fence and nothing is paced.
use std::os::fd::OwnedFd;

/// Acquire fences of child frames the host is about to sample: none.
pub(crate) fn take_acquire_fds() -> Vec<OwnedFd> {
    Vec::new()
}

/// A release fence for the children's next frames: there are no children.
pub(crate) fn publish_release_fd(fd: OwnedFd) {
    drop(fd);
}

/// Milliseconds since the last paced present (a trace field): none.
pub(crate) fn pace_ms() -> f64 {
    0.0
}
