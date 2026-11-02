//! Material paint as its own spatial program, separate from any shape's distance field.
//!
//! - [`program`]: what a script paints, as layers in document order (a fill of a region, a
//!   brushed stroke, a strokes file ([`strokes`]), a smooth-region flood ([`flood`]), an erase), per channel (colour, support block, support enforce). The last
//!   layer that holds a point decides it. A point is asked for directly (the slicer, exact) or
//!   through the baked bricks (the views).
//! - [`bake`]: the program sampled on a lattice where a region's boundary passes, deep insides
//!   one value per brick, nothing elsewhere; packed with the other parts' volumes into one
//!   float texture.
//! - `mod.draw.PaintLookup` (`draw/src/shader/paint_lookup.rs`), the one Splash lookup every renderer spreads (meshes
//!   and distance fields alike): one call at the shaded point, an edge antialiased over the
//!   pixel's footprint at any zoom.
//!
//! Paint never enters a distance program: a shape's field and its bricks are the same painted
//! or not, and a paint edit rebuilds nothing of the shape.

pub mod bake;
pub mod facets;
pub mod flood;
pub mod program;
/// The strokes file format (shared with the shape's sculpt strokes).
pub use makepad_csg_sdf::strokes;

pub use bake::{Atlas, Volume};
pub use facets::Facets;
pub use flood::Patch;
pub use program::{Channel, Layer, Op, Program, Region};
pub use strokes::{StrokeOp, Strokes, Swatch as StrokeSwatch};
/// A region solid's box (`None`: unbounded).
pub use makepad_csg_sdf::node_bounds as bounds_of;
