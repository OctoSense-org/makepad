//! The same semantic kit widgets used by splash-makepad's source templates.
//! This adapter compiles them against this app's Makepad lineage.
use crate as makepad_widgets;
use crate::{Cx, WidgetRef};

// Octos compositions contain native Buttons, Radios and Views; there is no
// source-artboard DesignButton wrapper in this backend.
fn set_design_selection(_root: &WidgetRef, _cx: &mut Cx, _selected: bool) {}

#[path = "kit_shared.rs"]
mod kit_shared;
pub use kit_shared::*;
