//! Minimum contrast between text and its background (the
//! `minimum-contrast` setting), measured as the WCAG 2 contrast ratio:
//! (L1 + 0.05) / (L2 + 0.05) over relative luminances, from 1 (the same
//! colour) to 21 (black on white).
//!
//! Text below the ratio is moved towards white when it is lighter than its
//! background, towards black when darker, just far enough to reach the
//! ratio. Mixing with white or black keeps the colour's hue, and moving
//! away from the background keeps light-on-dark text light. When that side
//! cannot reach the ratio (mid-grey text on a mid-grey background), the
//! other side is tried and the better of the two kept.
//!
//! Colours are sRGB components in 0..=1.

/// The ratio of `settings::Settings::minimum_contrast` that turns it off.
pub const OFF: f32 = 1.0;
pub const MAX: f32 = 21.0;

fn linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// WCAG relative luminance of an sRGB colour.
pub fn luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * linear(rgb[0]) + 0.7152 * linear(rgb[1]) + 0.0722 * linear(rgb[2])
}

/// WCAG contrast ratio of two colours, 1..=21, in either order.
pub fn ratio(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (la, lb) = (luminance(a), luminance(b));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// `fg` mixed towards `target` just far enough to reach `min` against
/// `bg`, or all the way if even `target` falls short.
fn towards(fg: [f32; 3], bg: [f32; 3], target: [f32; 3], min: f32) -> [f32; 3] {
    if ratio(target, bg) < min {
        return target;
    }
    // Contrast grows monotonically along the mix while moving away from
    // the background's luminance; 16 halvings are finer than 8-bit colour.
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..16 {
        let mid = (lo + hi) * 0.5;
        if ratio(mix(fg, target, mid), bg) >= min {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    mix(fg, target, hi)
}

/// `fg`, raised to at least `min` contrast against `bg` (unchanged when it
/// already has it, or when `min` is [`OFF`] or less).
pub fn ensure(fg: [f32; 3], bg: [f32; 3], min: f32) -> [f32; 3] {
    if min.is_nan() || min <= OFF {
        return fg;
    }
    let min = min.min(MAX);
    if ratio(fg, bg) >= min {
        return fg;
    }
    let (white, black) = ([1.0; 3], [0.0; 3]);
    let (first, second) = if luminance(fg) >= luminance(bg) {
        (white, black)
    } else {
        (black, white)
    };
    let a = towards(fg, bg, first, min);
    if ratio(a, bg) >= min {
        return a;
    }
    let b = towards(fg, bg, second, min);
    if ratio(b, bg) > ratio(a, bg) {
        b
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(c: u32) -> [f32; 3] {
        [
            ((c >> 16) & 0xff) as f32 / 255.0,
            ((c >> 8) & 0xff) as f32 / 255.0,
            (c & 0xff) as f32 / 255.0,
        ]
    }

    #[test]
    fn ratios_match_wcag() {
        assert!((ratio(hex(0x000000), hex(0xffffff)) - 21.0).abs() < 1e-3);
        assert!((ratio(hex(0xffffff), hex(0x000000)) - 21.0).abs() < 1e-3);
        assert!((ratio(hex(0x777777), hex(0x777777)) - 1.0).abs() < 1e-6);
        // Published reference pairs: #777 on white is 4.48:1, #767676 4.54:1.
        assert!((ratio(hex(0x777777), hex(0xffffff)) - 4.48).abs() < 0.01);
        assert!((ratio(hex(0x767676), hex(0xffffff)) - 4.54).abs() < 0.01);
        assert!((luminance(hex(0xffffff)) - 1.0).abs() < 1e-6);
        assert!((luminance(hex(0xff0000)) - 0.2126).abs() < 1e-6);
    }

    #[test]
    fn off_and_already_readable_colours_are_unchanged() {
        let (fg, bg) = (hex(0x808080), hex(0x303030));
        assert_eq!(ensure(fg, bg, OFF), fg);
        assert_eq!(ensure(fg, bg, 0.5), fg);
        assert_eq!(ensure(fg, bg, f32::NAN), fg);
        let white = hex(0xffffff);
        assert_eq!(ensure(white, bg, 4.5), white);
    }

    #[test]
    fn palette_244_on_236_becomes_readable_at_4_5() {
        // xterm-256 244 = #808080 and 236 = #303030: about 3.3:1.
        let (fg, bg) = (hex(0x808080), hex(0x303030));
        assert!(ratio(fg, bg) < 3.5);
        let out = ensure(fg, bg, 4.5);
        let r = ratio(out, bg);
        assert!((4.5..4.6).contains(&r), "just enough: {r}");
        // Lighter text on a dark background stays lighter, and grey stays grey.
        assert!(luminance(out) > luminance(fg));
        assert!((out[0] - out[1]).abs() < 1e-6 && (out[1] - out[2]).abs() < 1e-6);
    }

    #[test]
    fn hue_direction_is_kept() {
        // Dark blue on a darker blue: lightened towards white, still blue.
        let (fg, bg) = (hex(0x2040a0), hex(0x101840));
        let out = ensure(fg, bg, 4.5);
        assert!(ratio(out, bg) >= 4.5 - 1e-3);
        assert!(out[2] > out[1] && out[1] > out[0], "{out:?}");
        // Dark text on a light background is darkened.
        let (fg, bg) = (hex(0xb0b0c0), hex(0xf0f0f0));
        let out = ensure(fg, bg, 4.5);
        assert!(luminance(out) < luminance(fg));
        assert!(ratio(out, bg) >= 4.5 - 1e-3);
    }

    #[test]
    fn a_ratio_one_side_cannot_reach_uses_the_other() {
        // Grey just lighter than a light grey background: white is not
        // far enough away for 7:1, black is.
        let (fg, bg) = (hex(0xb8b8b8), hex(0xb0b0b0));
        let out = ensure(fg, bg, 7.0);
        assert!(ratio(out, bg) >= 7.0 - 1e-3, "{}", ratio(out, bg));
        assert!(luminance(out) < luminance(bg));
        // Nothing reaches 21:1 but black on white; get as close as possible.
        let out = ensure(hex(0x808080), hex(0x404040), 21.0);
        assert!(ratio(out, hex(0x404040)) > 7.0);
    }
}
