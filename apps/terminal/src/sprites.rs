//! Glyphs the terminal draws itself instead of taking from a font:
//! Powerline separators (U+E0B0–E0BF) and the shade blocks ░▒▓.
//!
//! Fonts draw these with side bearings and a line gap of their own, so a
//! prompt's arrows stop short of the row and leave a hairline between a
//! segment and its separator. Here each shape fills its cell exactly: the
//! straight sides are the cell's edges, and only a diagonal or a curve is
//! anti-aliased. The shapes follow the published Powerline and Powerline
//! Extra Symbols glyphs (each drawn edge to edge, full cell height).
//!
//! The shader in `widget.rs` (`DrawTermSprite`) evaluates the same distance
//! functions as [`Sprite::coverage`], which exists so the shapes can be
//! tested without a GPU.

/// The outline a sprite fills or strokes, in a cell `w` × `h` with y down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Shape {
    /// The whole cell (a shade block, at partial alpha).
    Fill = 0,
    /// A solid triangle from the left edge to a point at the right middle.
    Arrow = 1,
    /// The two sloped edges of `Arrow`, stroked (a thin chevron).
    ArrowLine = 2,
    /// Half an ellipse from the left edge, reaching the right edge.
    HalfDisc = 3,
    /// The curve of `HalfDisc`, stroked.
    HalfCircleLine = 4,
    /// The triangle under the top-left to bottom-right diagonal.
    Wedge = 5,
    /// The top-left to bottom-right diagonal, stroked.
    Diagonal = 6,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sprite {
    pub shape: Shape,
    /// Mirror left-right / top-bottom.
    pub flip_x: bool,
    pub flip_y: bool,
    /// Multiplies the foreground's alpha (shades: 25, 50, 75%).
    pub alpha: f32,
}

const fn sprite(shape: Shape, flip_x: bool, flip_y: bool) -> Sprite {
    Sprite {
        shape,
        flip_x,
        flip_y,
        alpha: 1.0,
    }
}

/// The sprite `ch` draws as, if it is one.
pub fn sprite_for(ch: char) -> Option<Sprite> {
    use Shape::*;
    Some(match ch {
        '\u{2591}' => Sprite {
            alpha: 0.25,
            ..sprite(Fill, false, false)
        },
        '\u{2592}' => Sprite {
            alpha: 0.5,
            ..sprite(Fill, false, false)
        },
        '\u{2593}' => Sprite {
            alpha: 0.75,
            ..sprite(Fill, false, false)
        },
        // Powerline: right and left hard and soft dividers.
        '\u{E0B0}' => sprite(Arrow, false, false),
        '\u{E0B1}' => sprite(ArrowLine, false, false),
        '\u{E0B2}' => sprite(Arrow, true, false),
        '\u{E0B3}' => sprite(ArrowLine, true, false),
        // Powerline Extra: right and left half circles, solid and thin.
        '\u{E0B4}' => sprite(HalfDisc, false, false),
        '\u{E0B5}' => sprite(HalfCircleLine, false, false),
        '\u{E0B6}' => sprite(HalfDisc, true, false),
        '\u{E0B7}' => sprite(HalfCircleLine, true, false),
        // Lower-left, lower-right, upper-left and upper-right triangles, and
        // the backslash and forward-slash separators between them.
        '\u{E0B8}' => sprite(Wedge, false, false),
        '\u{E0B9}' | '\u{E0BF}' => sprite(Diagonal, false, false),
        '\u{E0BA}' => sprite(Wedge, true, false),
        '\u{E0BB}' | '\u{E0BD}' => sprite(Diagonal, true, false),
        '\u{E0BC}' => sprite(Wedge, false, true),
        '\u{E0BE}' => sprite(Wedge, true, true),
        _ => return None,
    })
}

/// A character that is drawn as a picture rather than as text: box drawing,
/// block elements, braille and Powerline separators. Minimum contrast leaves
/// these alone, since their colour is a fill a TUI chose to match its
/// neighbours (a Powerline arrow is the colour of the segment it ends).
pub fn is_graphic(ch: char) -> bool {
    matches!(ch as u32, 0x2500..=0x259F | 0x2800..=0x28FF | 0xE0B0..=0xE0D4)
}

impl Sprite {
    /// Signed distance from the shape's edge at (x, y) in a `w` × `h` cell,
    /// negative inside; strokes are `thickness` wide.
    pub fn distance(&self, w: f32, h: f32, x: f32, y: f32, thickness: f32) -> f32 {
        let x = if self.flip_x { w - x } else { x };
        let y = if self.flip_y { h - y } else { y };
        let half = thickness * 0.5;
        match self.shape {
            Shape::Fill => -1.0,
            Shape::Arrow | Shape::ArrowLine => {
                // The edge from (0, 0) to (w, h/2), folded about the middle.
                let hh = h * 0.5;
                let fy = (y - hh).abs();
                let d = (x * hh + fy * w - w * hh) / (hh * hh + w * w).sqrt();
                if self.shape == Shape::Arrow {
                    d
                } else {
                    d.abs() - half
                }
            }
            Shape::HalfDisc | Shape::HalfCircleLine => {
                // The ellipse centred on the left edge's middle, radii w and
                // h/2; its distance to first order (value over gradient).
                let hh = h * 0.5;
                let (qx, qy) = (x / w, (y - hh) / hh);
                let k = (qx * qx + qy * qy).sqrt();
                let g = ((qx / w).powi(2) + (qy / hh).powi(2)).sqrt().max(1e-6);
                let d = (k - 1.0) * k / g;
                if self.shape == Shape::HalfDisc {
                    d
                } else {
                    d.abs() - half
                }
            }
            Shape::Wedge | Shape::Diagonal => {
                let d = (x * h - y * w) / (w * w + h * h).sqrt();
                if self.shape == Shape::Wedge {
                    d
                } else {
                    d.abs() - half
                }
            }
        }
    }

    /// The alpha a pixel of `pixel` size centred at (x, y) gets, before the
    /// sprite's own alpha: anti-aliased across one pixel of the edge.
    pub fn coverage(&self, w: f32, h: f32, x: f32, y: f32, thickness: f32, pixel: f32) -> f32 {
        (0.5 - self.distance(w, h, x, y, thickness) / pixel).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f32 = 9.0;
    const H: f32 = 20.0;

    /// Coverage sampled at pixel centres over a 9 × 20 cell.
    fn raster(s: Sprite) -> Vec<f32> {
        let mut out = Vec::new();
        for py in 0..H as usize {
            for px in 0..W as usize {
                out.push(s.coverage(W, H, px as f32 + 0.5, py as f32 + 0.5, 1.0, 1.0));
            }
        }
        out
    }

    fn at(r: &[f32], x: usize, y: usize) -> f32 {
        r[y * W as usize + x]
    }

    #[test]
    fn powerline_ranges_map_to_sprites() {
        for c in 0xE0B0..=0xE0BF {
            assert!(sprite_for(char::from_u32(c).unwrap()).is_some(), "{c:X}");
        }
        assert!(sprite_for('\u{E0AF}').is_none());
        assert!(
            sprite_for('\u{E0C0}').is_none(),
            "flames come from the Nerd Font"
        );
        assert!(sprite_for('a').is_none());
        assert!(
            sprite_for('█').is_none(),
            "full blocks stay with the font path"
        );
    }

    #[test]
    fn shades_are_flat_fills_at_quarter_steps() {
        for (c, a) in [('░', 0.25), ('▒', 0.5), ('▓', 0.75)] {
            let s = sprite_for(c).unwrap();
            assert_eq!(s.shape, Shape::Fill);
            assert_eq!(s.alpha, a);
            assert!(raster(s).iter().all(|&v| v == 1.0), "no dot pattern");
        }
    }

    #[test]
    fn the_hard_arrow_fills_its_cell_edge_to_edge() {
        let r = raster(sprite_for('\u{E0B0}').unwrap());
        // The whole left column and the middle row out to the tip.
        for y in 1..H as usize - 1 {
            assert_eq!(at(&r, 0, y), 1.0, "left edge at row {y}");
        }
        assert!(
            at(&r, W as usize - 1, H as usize / 2) > 0.3,
            "the tip reaches the right edge"
        );
        // Nothing in the right corners.
        assert_eq!(at(&r, W as usize - 1, 0), 0.0);
        assert_eq!(at(&r, W as usize - 1, H as usize - 1), 0.0);
    }

    #[test]
    fn left_and_right_arrows_are_mirror_images() {
        let right = raster(sprite_for('\u{E0B0}').unwrap());
        let left = raster(sprite_for('\u{E0B2}').unwrap());
        for y in 0..H as usize {
            for x in 0..W as usize {
                assert!((at(&right, x, y) - at(&left, W as usize - 1 - x, y)).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn opposite_wedges_tile_the_cell() {
        // A lower-left and an upper-right triangle make a full cell, the
        // anti-aliased diagonal summing to one (no seam, no overlap).
        let lower_left = raster(sprite_for('\u{E0B8}').unwrap());
        let upper_right = raster(sprite_for('\u{E0BE}').unwrap());
        for (a, b) in lower_left.iter().zip(&upper_right) {
            assert!((a + b - 1.0).abs() < 1e-4, "{a} + {b}");
        }
        let lower_right = raster(sprite_for('\u{E0BA}').unwrap());
        let upper_left = raster(sprite_for('\u{E0BC}').unwrap());
        for (a, b) in lower_right.iter().zip(&upper_left) {
            assert!((a + b - 1.0).abs() < 1e-4, "{a} + {b}");
        }
        // Lower-left is solid at the bottom-left corner and empty top-right.
        assert_eq!(at(&lower_left, 0, H as usize - 1), 1.0);
        assert_eq!(at(&lower_left, W as usize - 1, 0), 0.0);
    }

    #[test]
    fn the_half_disc_is_round_and_edge_to_edge() {
        let r = raster(sprite_for('\u{E0B4}').unwrap());
        for y in 2..H as usize - 2 {
            assert_eq!(at(&r, 0, y), 1.0, "flat left side at row {y}");
        }
        assert!(
            at(&r, W as usize - 1, H as usize / 2) > 0.5,
            "reaches the right edge"
        );
        assert_eq!(at(&r, W as usize - 1, 0), 0.0);
        // Anti-aliased: some pixels are partly covered.
        assert!(r.iter().any(|&v| v > 0.05 && v < 0.95));
    }

    #[test]
    fn thin_separators_are_strokes() {
        for c in ['\u{E0B1}', '\u{E0B5}', '\u{E0B9}', '\u{E0BB}'] {
            let r = raster(sprite_for(c).unwrap());
            let ink: f32 = r.iter().sum();
            // Roughly one pixel wide along a path about a cell high.
            assert!(ink > H * 0.6 && ink < H * 2.5, "{c:?}: {ink}");
        }
    }

    #[test]
    fn graphics_are_not_contrast_adjusted() {
        for c in ['─', '┼', '█', '░', '⣿', '\u{E0B0}', '\u{E0BC}'] {
            assert!(is_graphic(c), "{c:?}");
        }
        for c in ['a', '■', '\u{E0A0}', '\u{F113}'] {
            assert!(!is_graphic(c), "{c:?}");
        }
    }
}
