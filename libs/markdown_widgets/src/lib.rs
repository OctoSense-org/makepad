//! Reusable native Markdown presentation for Makepad.
//!
//! The host retains ownership of source, storage, navigation and resource loading.
//! Supply decoded images through [`content::Images`]; renderers never fetch URLs.
//! Optional Cargo features enable syntax highlighting, math, color emoji, SVG
//! decoding and diagrams. Disabled extensions retain readable source fallbacks.
//! Math implements Makepad's TeX subset, not a LaTeX document engine. Mermaid
//! support follows the pinned renderer; legacy flow/sequence syntax has bounded
//! adapters. Unsupported diagrams retain their source with an explanation.
pub use makepad_markdown;
pub use makepad_widgets;

pub mod content;
pub mod content_view;
#[cfg(feature = "diagrams")]
pub mod diagram;
pub mod diagram_view;
pub mod emoji;
#[cfg(feature = "math")]
pub mod math;
pub mod math_view;
#[cfg(feature = "diagrams")]
pub mod mermaid;
pub mod selection_slot;
#[cfg(feature = "diagrams")]
pub mod sequence;
pub mod view;

pub use view::{MarkdownView, MarkdownViewRef};

/// Register after `makepad_widgets::script_mod` and before the application's UI.
pub fn script_mod(vm: &mut makepad_widgets::ScriptVm) {
    math_view::script_mod(vm);
    diagram_view::script_mod(vm);
    content_view::script_mod(vm);
    view::script_mod(vm);
}

fn color(rgb: u32) -> makepad_widgets::Vec4f {
    makepad_widgets::vec4(
        ((rgb >> 16) & 255) as f32 / 255.0,
        ((rgb >> 8) & 255) as f32 / 255.0,
        (rgb & 255) as f32 / 255.0,
        1.0,
    )
}

/// How much to shrink content `width` wide to fit `available`. Inside a Fit
/// container (such as a table cell) the available width is unknown (NaN) while
/// laying out, so content keeps its natural size rather than collapsing to a pixel.
pub(crate) fn fit_ratio(available: f64, width: f64) -> f64 {
    if available.is_finite() && available >= 1.0 && width > 0.0 {
        (available / width).min(1.0)
    } else {
        1.0
    }
}

#[cfg(test)]
mod fit_tests {
    #[test]
    fn unknown_width_keeps_natural_size() {
        assert_eq!(super::fit_ratio(f64::NAN, 80.0), 1.0);
        assert_eq!(super::fit_ratio(40.0, 80.0), 0.5);
        assert_eq!(super::fit_ratio(200.0, 80.0), 1.0);
    }
}
