//! The same semantic kit widgets used by Octoscript-Makepad's source templates.
//! This adapter compiles them against this app's Makepad lineage.
use crate as makepad_widgets;
use crate::{Cx, WidgetRef};

// Octos compositions contain native Buttons, Radios and Views; there is no
// source-artboard DesignButton wrapper in this backend.
fn set_design_selection(_root: &WidgetRef, _cx: &mut Cx, _selected: bool) {}

// `kit_shared.rs` is vendored from Octoscript-Makepad
// (crates/octoscript-widgets/src/kit_shared.rs @ 78f90f6e), the file the
// port branch reached through a sibling-checkout `#[path]` include.
#[path = "kit_shared.rs"]
mod shared;
pub use shared::*;
