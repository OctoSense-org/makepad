//! The same semantic kit widgets used by Octoscript-Makepad's source templates.
//! This adapter compiles them against this app's Makepad lineage.
use crate as makepad_widgets;
use crate::{Cx, WidgetRef};

// Octos compositions contain native Buttons, Radios and Views; there is no
// source-artboard DesignButton wrapper in this backend.
fn set_design_selection(_root: &WidgetRef, _cx: &mut Cx, _selected: bool) {}

#[path = "../../../Octoscript-Makepad/crates/octoscript-widgets/src/kit_shared.rs"]
mod shared;
pub use shared::*;
