//! The terminal terminal widget: renders a `Session` and feeds it input.
//!
//! Rendering follows the cell-grid discipline (ghostty's renderer notes):
//! integer cell advance from the monospace font, background rects merged
//! into per-row runs, glyphs batched into one instance buffer, underline
//! styles drawn by a dedicated shader, block/bar/underline cursor with a
//! hollow variant when unfocused.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use makepad_widgets::text::geom::Point;
use makepad_widgets::text::rasterizer::RasterizedGlyph;
use makepad_widgets::makepad_draw::shader::draw_text::{ShapedTextGlyph, ShapedTextRun};
use makepad_widgets::*;

use crate::gesture::{boundary_col, past_threshold, MoveAction, Press, PressInfo};
use crate::links::{self, LinkHit};
use crate::pty::InputRejected;
use crate::session::{Session, SpawnOptions};
use crate::settings::{self as term_settings, BellStyle, CursorShape, Settings};
use crate::themes;
use std::time::{Duration, Instant};
use crate::term::color::Rgb;
use crate::kitty_input::{self, TextKeyPairing, TextOutcome};
use crate::term::key_encode::{
    encode_key, encode_text, Key, KeyAction, KeyEncodeOptions, KeyEvent as TermKeyEvent, KeyMods,
    KittyFlags,
};
use crate::term::modes::Mode;
use crate::term::mouse_encode::{
    encode_mouse, MouseButton as TermMouseButton, MouseEventKind, MouseFormat, MouseReport,
    MouseTracking,
};
use crate::cell_glyph::{
    cell_glyph, fit_glyphs, icon_columns, is_private_use, CellGlyph, Fit, GlyphCache,
};
use crate::text_run::{self, GlyphIn, Placed, RunCache, RunStyle, SegCell};
use crate::{contrast, sprites};
use crate::term::screen::CursorStyle;
use crate::term::style::{StyleColor, StyleFlags};
use crate::term::terminal::TermEvent;

// The scrollback search bar: a child module, so it reaches the widget's
// fields (the matching itself is `crate::search`).
#[path = "search_bar.rs"]
mod search_bar;
use crate::search::Search;
use search_bar::SearchUi;

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.*

    /** The terminal cell background: one flat premultiplied quad per run
     * of same-colored cells, batched into its own draw-call group. */
    set_type_default() do #(DrawTermBg::script_shader(vm)) {
        ..mod.draw.DrawQuad
        draw_call_group: @term_cell_bg
        /** default cell background */
        color: #x1a1b26
        pixel: fn() {
            return vec4(self.color.rgb * self.color.a, self.color.a)
        }
    }

    // Each cell shades a continuing stroke at its outer edges. Independent
    // antialiased vector paths fade at every cell boundary and make TUI rules
    // look dotted, especially when the terminal sits on a zoomed canvas.
    set_type_default() do #(DrawTermBox::script_shader(vm)) {
        ..mod.draw.DrawQuad
        draw_call_group: @term_box
        color: #fff
        arms: vec4(0.0, 0.0, 0.0, 0.0)
        center: vec2(0.0, 0.0)
        corner: vec2(0.0, 0.0)
        thickness: 1.0
        radius: 0.0
        pixel_size: 1.0
        pixel: fn() {
            let d = self.pos * self.rect_size - self.center
            let half = self.thickness * 0.5
            var distance = 10000.0
            if self.radius > 0.0 {
                let q = d * self.corner
                if q.x >= self.radius {
                    distance = abs(q.y) - half
                } else if q.y >= self.radius {
                    distance = abs(q.x) - half
                } else {
                    distance = abs(length(q - vec2(self.radius)) - self.radius) - half
                }
            } else {
                if self.arms.x + self.arms.y > 0.0 {
                    let left = if self.arms.x > 0.0 {-10000.0} else {-half}
                    let right = if self.arms.y > 0.0 {10000.0} else {half}
                    distance = max(abs(d.y) - half, max(left - d.x, d.x - right))
                }
                if self.arms.z + self.arms.w > 0.0 {
                    let top = if self.arms.z > 0.0 {-10000.0} else {-half}
                    let bottom = if self.arms.w > 0.0 {10000.0} else {half}
                    distance = min(distance, max(abs(d.x) - half, max(top - d.y, d.y - bottom)))
                }
            }
            let coverage = clamp(0.5 - distance / self.pixel_size, 0.0, 1.0)
            let alpha = self.color.a * coverage
            return vec4(self.color.rgb * alpha, alpha)
        }
    }

    // Powerline separators and shade blocks (`crate::sprites`): the shape's
    // straight sides are the quad's, so it meets its neighbours with no seam;
    // diagonals and curves are anti-aliased over one device pixel. The
    // distance functions match `Sprite::distance`.
    set_type_default() do #(DrawTermSprite::script_shader(vm)) {
        ..mod.draw.DrawQuad
        draw_call_group: @term_sprite
        color: #fff
        shape: 0.0
        flip: vec2(0.0, 0.0)
        thickness: 1.0
        pixel_size: 1.0
        pixel: fn() {
            let size = self.rect_size
            var p = self.pos * size
            if self.flip.x > 0.5 {
                p.x = size.x - p.x
            }
            if self.flip.y > 0.5 {
                p.y = size.y - p.y
            }
            let half = self.thickness * 0.5
            var distance = -1.0
            if self.shape > 0.5 && self.shape < 2.5 {
                // Arrow: the edge from the top-left corner to the right
                // middle, folded about the middle row.
                let hh = size.y * 0.5
                let fy = abs(p.y - hh)
                let d = (p.x * hh + fy * size.x - size.x * hh) / length(vec2(hh, size.x))
                distance = if self.shape < 1.5 {d} else {abs(d) - half}
            } else if self.shape > 2.5 && self.shape < 4.5 {
                // Half ellipse on the left edge's middle, radii w and h/2.
                let hh = size.y * 0.5
                let q = vec2(p.x / size.x, (p.y - hh) / hh)
                let k = length(q)
                let g = max(length(vec2(q.x / size.x, q.y / hh)), 0.000001)
                let d = (k - 1.0) * k / g
                distance = if self.shape < 3.5 {d} else {abs(d) - half}
            } else if self.shape > 4.5 {
                // Wedge under, or stroke along, the top-left to bottom-right
                // diagonal.
                let d = (p.x * size.y - p.y * size.x) / length(size)
                distance = if self.shape < 5.5 {d} else {abs(d) - half}
            }
            let coverage = clamp(0.5 - distance / self.pixel_size, 0.0, 1.0)
            let alpha = self.color.a * coverage
            return vec4(self.color.rgb * alpha, alpha)
        }
    }

    set_type_default() do #(DrawTermUnderline::script_shader(vm)) {
        ..mod.draw.DrawQuad
        draw_call_group: @term_underline
        color: #xc0caf5
        kind: 1.0
        on: fn(a: float) {
            return vec4(self.color.rgb * self.color.a * a, self.color.a * a)
        }
        pixel: fn() {
            if self.kind < 1.5 {
                return self.on(1.0)
            }
            if self.kind < 2.5 {
                // Double: two bands with a gap.
                if self.pos.y < 0.33 || self.pos.y > 0.66 {
                    return self.on(1.0)
                }
                return self.on(0.0)
            }
            if self.kind < 3.5 {
                // Curly: sine band.
                let center = 0.5 + sin(self.pos.x * self.rect_size.x * 1.2) * 0.3
                if abs(self.pos.y - center) < 0.22 {
                    return self.on(1.0)
                }
                return self.on(0.0)
            }
            if self.kind < 4.5 {
                // Dotted.
                if modf(self.pos.x * self.rect_size.x, 3.0) < 1.5 {
                    return self.on(1.0)
                }
                return self.on(0.0)
            }
            // Dashed.
            if modf(self.pos.x * self.rect_size.x, 8.0) < 5.0 {
                return self.on(1.0)
            }
            return self.on(0.0)
        }
    }

    set_type_default() do #(DrawTermCursor::script_shader(vm)) {
        ..mod.draw.DrawQuad
        draw_call_group: @term_cursor
        color: #xc0caf5
        hollow: 0.0
        pixel: fn() {
            if self.hollow < 0.5 {
                return vec4(self.color.rgb * self.color.a, self.color.a)
            }
            let border = 1.0
            let bx = border / self.rect_size.x
            let by = border / self.rect_size.y
            if self.pos.x < bx || self.pos.x > 1.0 - bx || self.pos.y < by || self.pos.y > 1.0 - by {
                return vec4(self.color.rgb, 1.0)
            }
            return vec4(0.0, 0.0, 0.0, 0.0)
        }
    }

    mod.widgets.MpTermBase = #(MpTerm::register_widget(vm))

    /** The terminal widget: cell-background, glyph, underline and cursor
     * draw layers over a monospace grid. */
    mod.widgets.MpTerm = set_type_default() do mod.widgets.MpTermBase {
        width: Fill
        height: Fill
        /** terminal font size in points 6..24 step 0.5 */
        font_size: 10.0
        /** horizontal inner padding in pixels 0..32 step 1 */
        pad_x: 6.0
        /** vertical inner padding in pixels 0..32 step 1 */
        pad_y: 4.0
        draw_bg +: {
            color: uniform(#x1a1b26)
            inset: uniform(0.0)
            corner_radius: uniform(0.0)
            frame_width: uniform(0.0)
            frame_color: uniform(#0000)
            pixel: fn() {
                let p = self.pos * self.rect_size
                if self.corner_radius > 0.0 {
                    let sdf = Sdf2d.viewport(p)
                    let edge = self.frame_width
                    sdf.box(edge, edge, self.rect_size.x - 2.0 * edge,
                        self.rect_size.y - 2.0 * edge, self.corner_radius)
                    sdf.fill_keep(self.color)
                    sdf.stroke(self.frame_color, edge)
                    return sdf.result
                }
                let tl = min(p.x, p.y)
                let br = min(self.rect_size.x - p.x, self.rect_size.y - p.y)
                if min(tl, br) < self.inset {
                    // Classic sunken client edge: shadow then black at the
                    // top/left, highlight then button face at bottom/right.
                    let edge = if tl < br {
                        if tl < 1.0 {#808080} else {#000000}
                    } else {
                        if br < 1.0 {#ffffff} else {theme.color_bg_app}
                    }
                    return edge
                }
                return vec4(self.color.rgb * self.color.a, self.color.a)
            }
        }
        draw_text +: {
            draw_call_group: @term_text
            // The shell and Omarchy prompts are designed around JetBrains
            // Mono. Liberation Mono lacks even common prompt symbols such as
            // U+276F. Symbols Nerd Font Mono (bundled, MIT; its icon sets
            // are MIT, OFL, CC BY and Apache, see resources/) draws the Nerd
            // Font icons prompts and file listers print, ahead of Font
            // Awesome and Inter, which map other glyphs to some of the same
            // private-use codepoints. Powerline separators are drawn as
            // sprites (`crate::sprites`), not from a font.
            text_style: TextStyle{
                font_family: FontFamily{
                    latin := FontMember{
                        res: crate_resource("makepad_widgets:resources/jetbrains_mono_variable.ttf")
                        asc: 0.0 desc: 0.0 weight: 400.0
                    }
                    nerd := FontMember{
                        res: crate_resource("self:resources/SymbolsNerdFontMono-Regular.ttf")
                        asc: 0.0 desc: 0.0
                    }
                    icons := FontMember{
                        res: crate_resource("makepad_widgets:resources/fa-solid-900.ttf")
                        asc: 0.0 desc: 0.0
                    }
                    emoji := FontMember{
                        res: crate_resource("makepad_widgets:resources/NotoColorEmoji.ttf")
                        asc: 0.0 desc: 0.0
                    }
                    symbols := FontMember{
                        res: crate_resource("makepad_widgets:resources/Inter.ttf")
                        asc: 0.0 desc: 0.0
                    }
                    // Chinese, Japanese kanji and other CJK ideographs: the
                    // same font makepad's theme falls back to. The grid
                    // gives each of them two cells.
                    chinese := FontMember{
                        res: crate_resource("makepad_widgets:resources/LXGWWenKaiRegular.ttf")
                        asc: 0.0 desc: 0.0
                    }
                }
                line_spacing: 1.0
            }
        }
        bold_text_style: TextStyle{
            font_family: FontFamily{
                latin := FontMember{
                    res: crate_resource("makepad_widgets:resources/jetbrains_mono_variable.ttf")
                    asc: 0.0 desc: 0.0 weight: 800.0
                }
                nerd := FontMember{
                    res: crate_resource("self:resources/SymbolsNerdFontMono-Regular.ttf")
                    asc: 0.0 desc: 0.0
                }
                icons := FontMember{
                    res: crate_resource("makepad_widgets:resources/fa-solid-900.ttf")
                    asc: 0.0 desc: 0.0
                }
                emoji := FontMember{
                    res: crate_resource("makepad_widgets:resources/NotoColorEmoji.ttf")
                    asc: 0.0 desc: 0.0
                }
                symbols := FontMember{
                    res: crate_resource("makepad_widgets:resources/Inter.ttf")
                    asc: 0.0 desc: 0.0
                }
                chinese := FontMember{
                    res: crate_resource("makepad_widgets:resources/LXGWWenKaiBold.ttf")
                    asc: 0.0 desc: 0.0
                }
            }
            line_spacing: 1.0
        }
        draw_cell_bg +: {}
        draw_underline +: {}
        draw_cursor +: {}
    }
}

#[derive(Script, ScriptHook)]
#[repr(C)]
struct DrawTermBg {
    #[deref]
    draw_super: DrawQuad,
    #[live]
    color: Vec4f,
}

#[derive(Script, ScriptHook)]
#[repr(C)]
struct DrawTermUnderline {
    #[deref]
    draw_super: DrawQuad,
    #[live]
    color: Vec4f,
    #[live]
    kind: f32,
}

#[derive(Script, ScriptHook)]
#[repr(C)]
struct DrawTermBox {
    #[deref]
    draw_super: DrawQuad,
    #[live]
    color: Vec4f,
    #[live]
    arms: Vec4f,
    #[live]
    center: Vec2f,
    #[live]
    corner: Vec2f,
    #[live]
    thickness: f32,
    #[live]
    radius: f32,
    #[live]
    pixel_size: f32,
}

#[derive(Script, ScriptHook)]
#[repr(C)]
struct DrawTermSprite {
    #[deref]
    draw_super: DrawQuad,
    #[live]
    color: Vec4f,
    #[live]
    shape: f32,
    #[live]
    flip: Vec2f,
    #[live]
    thickness: f32,
    #[live]
    pixel_size: f32,
}

#[derive(Script, ScriptHook)]
#[repr(C)]
struct DrawTermCursor {
    #[deref]
    draw_super: DrawQuad,
    #[live]
    color: Vec4f,
    #[live]
    hollow: f32,
}

#[derive(Clone, Debug, Default)]
pub enum MpTermAction {
    TitleChanged(String),
    PwdChanged(String),
    Bell,
    Exited,
    FileDropped {
        path: PathBuf,
    },
    PromptSubmitted,
    #[default]
    None,
}

/// Paths of the fonts a terminal draws with, beyond the bundled ones:
/// (regular, bold, CJK fallback).
type FontKey = (Option<String>, Option<String>, Option<String>);

/// A text style whose family is `primary` (if any), then the bundled
/// JetBrains Mono at `weight`, then `cjk` (if any), then the icon, emoji
/// and symbol fonts. Script members cannot be conditional, hence the arms.
fn terminal_text_style(vm: &mut ScriptVm, primary: Option<&str>, cjk: Option<&str>, weight: f64) -> ScriptValue {
    match (primary.map(str::to_owned), cjk.map(str::to_owned)) {
        (Some(primary), Some(cjk)) => script_eval!(vm, {
            use mod.prelude.widgets_internal.*
            TextStyle{
                font_family: FontFamily{
                    primary := FontMember{ res: file_resource(#(primary)) asc: 0.0 desc: 0.0 }
                    latin := FontMember{ res: crate_resource("makepad_widgets:resources/jetbrains_mono_variable.ttf") asc: 0.0 desc: 0.0 weight: #(weight) }
                    cjk := FontMember{ res: file_resource(#(cjk)) asc: 0.0 desc: 0.0 }
                    nerd := FontMember{ res: crate_resource("self:resources/SymbolsNerdFontMono-Regular.ttf") asc: 0.0 desc: 0.0 }
                    icons := FontMember{ res: crate_resource("makepad_widgets:resources/fa-solid-900.ttf") asc: 0.0 desc: 0.0 }
                    emoji := FontMember{ res: crate_resource("makepad_widgets:resources/NotoColorEmoji.ttf") asc: 0.0 desc: 0.0 }
                    symbols := FontMember{ res: crate_resource("makepad_widgets:resources/Inter.ttf") asc: 0.0 desc: 0.0 }
                }
                line_spacing: 1.0
            }
        }),
        (Some(primary), None) => script_eval!(vm, {
            use mod.prelude.widgets_internal.*
            TextStyle{
                font_family: FontFamily{
                    primary := FontMember{ res: file_resource(#(primary)) asc: 0.0 desc: 0.0 }
                    latin := FontMember{ res: crate_resource("makepad_widgets:resources/jetbrains_mono_variable.ttf") asc: 0.0 desc: 0.0 weight: #(weight) }
                    nerd := FontMember{ res: crate_resource("self:resources/SymbolsNerdFontMono-Regular.ttf") asc: 0.0 desc: 0.0 }
                    icons := FontMember{ res: crate_resource("makepad_widgets:resources/fa-solid-900.ttf") asc: 0.0 desc: 0.0 }
                    emoji := FontMember{ res: crate_resource("makepad_widgets:resources/NotoColorEmoji.ttf") asc: 0.0 desc: 0.0 }
                    symbols := FontMember{ res: crate_resource("makepad_widgets:resources/Inter.ttf") asc: 0.0 desc: 0.0 }
                }
                line_spacing: 1.0
            }
        }),
        (None, Some(cjk)) => script_eval!(vm, {
            use mod.prelude.widgets_internal.*
            TextStyle{
                font_family: FontFamily{
                    latin := FontMember{ res: crate_resource("makepad_widgets:resources/jetbrains_mono_variable.ttf") asc: 0.0 desc: 0.0 weight: #(weight) }
                    cjk := FontMember{ res: file_resource(#(cjk)) asc: 0.0 desc: 0.0 }
                    nerd := FontMember{ res: crate_resource("self:resources/SymbolsNerdFontMono-Regular.ttf") asc: 0.0 desc: 0.0 }
                    icons := FontMember{ res: crate_resource("makepad_widgets:resources/fa-solid-900.ttf") asc: 0.0 desc: 0.0 }
                    emoji := FontMember{ res: crate_resource("makepad_widgets:resources/NotoColorEmoji.ttf") asc: 0.0 desc: 0.0 }
                    symbols := FontMember{ res: crate_resource("makepad_widgets:resources/Inter.ttf") asc: 0.0 desc: 0.0 }
                }
                line_spacing: 1.0
            }
        }),
        (None, None) => script_eval!(vm, {
            use mod.prelude.widgets_internal.*
            TextStyle{
                font_family: FontFamily{
                    latin := FontMember{ res: crate_resource("makepad_widgets:resources/jetbrains_mono_variable.ttf") asc: 0.0 desc: 0.0 weight: #(weight) }
                    nerd := FontMember{ res: crate_resource("self:resources/SymbolsNerdFontMono-Regular.ttf") asc: 0.0 desc: 0.0 }
                    icons := FontMember{ res: crate_resource("makepad_widgets:resources/fa-solid-900.ttf") asc: 0.0 desc: 0.0 }
                    emoji := FontMember{ res: crate_resource("makepad_widgets:resources/NotoColorEmoji.ttf") asc: 0.0 desc: 0.0 }
                    symbols := FontMember{ res: crate_resource("makepad_widgets:resources/Inter.ttf") asc: 0.0 desc: 0.0 }
                }
                line_spacing: 1.0
            }
        }),
    }
}

/// A glyph of a shaped run, placed on the grid: relative to the run's
/// left edge and the baseline, and the cell (from the run's start) whose
/// colours it takes.
#[derive(Clone, Copy)]
struct RunGlyph {
    cell: u16,
    x: f32,
    y: f32,
    font_size_in_lpxs: f32,
    rasterized: RasterizedGlyph,
}

/// Neighbouring text cells shaped as one (`crate::text_run`): its text,
/// cell starts, colours and cluster flags are ranges of [`RunScratch`].
struct RunDraw {
    x: f64,
    y: f64,
    col: usize,
    row: usize,
    text: std::ops::Range<usize>,
    cells: std::ops::Range<usize>,
    style: RunStyle,
}

/// A frame's text runs, kept from frame to frame so drawing allocates
/// nothing for them once warm.
#[derive(Default)]
struct RunScratch {
    runs: Vec<RunDraw>,
    text: String,
    starts: Vec<usize>,
    colors: Vec<Vec4f>,
    centered: Vec<bool>,
    /// A row's cells as run segmentation sees them, and its runs.
    seg: Vec<SegCell>,
    seg_runs: Vec<std::ops::Range<usize>>,
}

impl RunScratch {
    fn clear(&mut self) {
        self.runs.clear();
        self.text.clear();
        self.starts.clear();
        self.colors.clear();
        self.centered.clear();
    }
}

#[derive(Clone, Copy)]
struct CachedGlyph {
    rasterized: RasterizedGlyph,
    font_size_in_lpxs: f32,
    x_offset_in_lpxs: f32,
    y_offset_in_lpxs: f32,
}

/// The tokyo-night terminal palette (terminal's default look; makepad-wm
/// re-themes at spawn time via --theme args later).
pub fn default_theme() -> ([Rgb; 16], Rgb, Rgb) {
    let base16 = [
        Rgb::new(0x1a, 0x1b, 0x26), // black = background
        Rgb::new(0xf7, 0x76, 0x8e),
        Rgb::new(0x9e, 0xce, 0x6a),
        Rgb::new(0xe0, 0xaf, 0x68),
        Rgb::new(0x7a, 0xa2, 0xf7),
        Rgb::new(0xad, 0x8e, 0xe6),
        Rgb::new(0x44, 0x9d, 0xab),
        Rgb::new(0xa9, 0xb1, 0xd6), // white = foreground
        Rgb::new(0x41, 0x48, 0x68), // bright black = muted
        Rgb::new(0xff, 0x7a, 0x93),
        Rgb::new(0xb9, 0xf2, 0x7c),
        Rgb::new(0xff, 0x9e, 0x64),
        Rgb::new(0x7d, 0xa6, 0xff),
        Rgb::new(0xbb, 0x9a, 0xf7),
        Rgb::new(0x0d, 0xb9, 0xd7),
        Rgb::new(0xc0, 0xca, 0xf5), // bright white = bright fg
    ];
    (
        base16,
        Rgb::new(0xa9, 0xb1, 0xd6),
        Rgb::new(0x1a, 0x1b, 0x26),
    )
}

/// Optional host frame. Padding belongs to the terminal grid, so drawing,
/// pointer input and IME use the same full-sized rounded surface.
#[derive(Clone, Copy, PartialEq)]
pub struct TerminalPresentationFrame {
    pub padding: f64,
    pub radius: f32,
    pub border_width: f32,
    pub border_color: Vec4f,
}

#[derive(Script, Widget)]
pub struct MpTerm {
    #[uid]
    uid: WidgetUid,
    #[source]
    source: ScriptObjectRef,
    #[walk]
    walk: Walk,
    #[layout]
    layout: Layout,
    #[redraw]
    #[live]
    draw_bg: DrawQuad,
    #[live]
    draw_text: DrawText,
    #[live]
    bold_text_style: TextStyle,
    #[live]
    draw_boxes: DrawTermBox,
    #[live]
    draw_sprites: DrawTermSprite,
    #[live]
    draw_dots: DrawVector,
    #[live]
    draw_cell_bg: DrawTermBg,
    #[live]
    draw_underline: DrawTermUnderline,
    #[live]
    draw_cursor: DrawTermCursor,
    #[live(10.0)]
    font_size: f64,
    #[live(6.0)]
    pad_x: f64,
    #[live(4.0)]
    pad_y: f64,

    #[rust]
    session: Option<Session>,
    #[rust]
    pub cwd: Option<PathBuf>,
    /// A transformed host supplies its screen area and local-to-screen mapping.
    #[rust]
    pub canvas_ime_anchor: Option<(Area, PopupAnchorTransform)>,
    /// A one-shot job instead of the interactive shell (`--preview` runs
    /// the pager on a file); the session ends when it exits.
    #[rust]
    pub command: Option<String>,
    /// An unloaded Quick Look panel: no session, and none respawned until
    /// the next retarget (`restart_with`).
    #[rust]
    dormant: bool,
    #[rust]
    area: Area,
    /// Key focus grabbed on the first draw (a terminal owns the keyboard
    /// without needing a click).
    #[rust]
    took_focus: bool,
    #[rust]
    rect: Rect,
    #[rust]
    cell_w: f64,
    #[rust]
    cell_h: f64,
    #[rust]
    cell_baseline: f64,
    #[rust]
    glyph_cache: GlyphCache<CachedGlyph>,
    /// Shaped runs of neighbouring text cells (`crate::text_run`).
    #[rust]
    run_cache: RunCache<RunGlyph>,
    /// The OpenType features text is shaped with (`font-features`).
    #[rust]
    font_features: Rc<Vec<text_run::FontFeature>>,
    #[rust]
    run_scratch: RunScratch,
    /// This frame's sprite draw call has been opened.
    #[rust]
    sprites_open: bool,
    #[rust]
    glyph_cache_key: (u64, u64, u64),
    /// Lines scrolled back from the bottom (0 = live).
    #[rust]
    view_offset: usize,
    // Selection in absolute (eviction-stable) rows.
    #[rust]
    sel_anchor: Option<(u64, usize)>,
    #[rust]
    sel_cursor: Option<(u64, usize)>,
    #[rust]
    selecting: bool,
    /// The word or line a double or triple click grabbed, as (row, start,
    /// end) with `end` exclusive: a drag extends the selection from it.
    #[rust]
    sel_unit: Option<(u64, usize, usize)>,
    /// What the current press is doing: a click, a selection drag, or a
    /// press that belongs to a mouse-reporting program.
    #[rust]
    press: Press,
    /// Fingers on a touchscreen, by digit, at their last y.
    #[rust]
    touches: Vec<(makepad_widgets::makepad_platform::event::DigitId, f64)>,
    #[rust]
    last_finger: Option<Vec2d>,
    #[rust]
    select_scroll_frame: NextFrame,
    #[rust]
    bell_frames: u8,
    #[rust]
    last_mouse_cell: Option<(u32, u32, u8)>,
    /// Precise (trackpad, Magic Mouse) vertical scroll not yet worth a line.
    #[rust]
    scroll_accum: f64,
    /// Background alpha (focused, unfocused): Omarchy's window opacity rule
    /// "0.78 0.70", handed down by makepad-wm via MAKEPAD_TERMINAL_OPACITY. Standalone
    /// runs are opaque. The shared swapchain is BGRA and the compositor
    /// blends premultiplied, so the wallpaper shows through for free.
    #[rust((1.0, 1.0))]
    bg_opacity: (f32, f32),
    #[rust]
    opaque_style: bool,
    #[rust]
    background_dimming: f32,
    #[rust(1.0)]
    presentation_font_scale: f64,
    #[rust]
    presentation_frame: Option<TerminalPresentationFrame>,
    #[rust]
    style_colors: Option<(Rgb, Rgb)>,
    #[rust]
    original_colors: Option<([Rgb; 16], Rgb, Rgb)>,
    /// The cursor colour the host handed down (MAKEPAD_TERMINAL_COLORS).
    #[rust]
    original_cursor: Option<Rgb>,
    /// The person's terminal settings as last applied (`crate::settings`)
    /// and the generation they came from.
    #[rust]
    settings: Settings,
    #[rust]
    settings_gen: u64,
    /// A blinking cursor's phase starts here; typing restarts it on.
    #[rust]
    blink_epoch: Option<Instant>,
    #[rust]
    blink_timer: Timer,
    #[rust]
    blink_armed: bool,
    /// Shown over the terminal for a few seconds when input was turned
    /// away: a paste refused because the program has stopped reading.
    #[rust]
    input_notice: Option<String>,
    #[rust]
    input_notice_timer: Timer,
    #[rust]
    input_notice_armed: bool,
    /// The composed character macOS delivers right after an Option+key that
    /// option-as-meta already sent as a Meta key.
    #[rust]
    swallow_text_until: Option<Instant>,
    /// Kitty report-all: pairs a key-down with the text it typed, so the
    /// program gets one key event (see `crate::kitty_input`).
    #[rust]
    kitty_pairing: TextKeyPairing,
    /// Flushes a text no key-down claimed (an IME commit).
    #[rust]
    kitty_text_timer: Timer,
    /// Keys whose press reached the program: only these report a release.
    #[rust]
    kitty_pressed: Vec<KeyCode>,
    /// Wakes the widget when a held synchronized frame (mode 2026) times
    /// out, so a program that died mid-frame can't freeze the screen.
    #[rust]
    sync_timer: Timer,
    #[rust]
    sync_timer_for: Option<f64>,
    /// The fonts applied from the settings, and the script objects that
    /// keep their resources alive.
    /// Set by a host for the one event it routes here (`crate::tabs`).
    #[rust]
    pub route_keys_here: bool,
    /// Set by a host that reports the title itself (`crate::tabs`: a tab's
    /// given name wins over the program's); the program's title then only
    /// goes to that host, not straight to the window manager.
    #[rust]
    pub titled_by_host: bool,
    #[rust]
    font_key: Option<FontKey>,
    /// The link under the pointer while the link modifier is held
    /// (`crate::links`): underlined, its target in the status line.
    #[rust]
    link_hover: Option<LinkHit>,
    /// Where the pointer hovers over this terminal, and whether the link
    /// modifier was held there.
    #[rust]
    link_pointer: Option<(Vec2d, bool)>,
    /// A modifier press on a link: it opens on release unless it drags.
    #[rust]
    link_press: Option<(LinkHit, Vec2d)>,
    #[rust]
    font_roots: Vec<ScriptObjectRef>,
    /// Scrollback search (Cmd+F / Ctrl+Shift+F): the matches and the bar.
    #[rust]
    search: Search,
    #[rust]
    search_ui: SearchUi,
}

impl ScriptHook for MpTerm {
    fn on_after_apply(
        &mut self,
        vm: &mut ScriptVm,
        _apply: &Apply,
        _scope: &mut Scope,
        _value: ScriptValue,
    ) {
        self.clear_glyph_caches();
        let style = desktop_style::current_style(vm);
        let retro = matches!(
            style,
            desktop_style::DesktopStyle::Windows2000 | desktop_style::DesktopStyle::NextStep
        );
        self.opaque_style = retro || style == desktop_style::DesktopStyle::Android;
        self.draw_bg.draw_vars.set_uniform(
            vm.cx_mut(),
            id!(inset),
            &[if retro { 2.0 } else { 0.0 }],
        );
        self.style_colors = if desktop_style::current_name(vm).is_some_and(|s| s != "omarchy") {
            makepad_wm_theme::current_for_vm(vm).and_then(|p| {
                Some((
                    parse_hex_rgb(p.get("term.foreground").or_else(|| p.get("foreground"))?)?,
                    parse_hex_rgb(p.get("term.background").or_else(|| p.get("background"))?)?,
                ))
            })
        } else {
            None
        };
        self.apply_colors();
    }
}

impl MpTerm {
    pub fn set_presentation_frame(&mut self, cx: &mut Cx, frame: Option<TerminalPresentationFrame>) {
        if self.presentation_frame != frame {
            self.presentation_frame = frame;
            self.draw_bg.redraw(cx);
        }
    }

    fn inner_padding(&self) -> DVec2 {
        self.presentation_frame.map_or(dvec2(self.pad_x, self.pad_y), |frame| {
            dvec2(frame.padding, frame.padding)
        })
    }

    /// Scale a hosted terminal's typography without replacing its session or
    /// changing the user's configured base font size.
    pub fn set_presentation_font_scale(&mut self, cx: &mut Cx, scale: f64) {
        let scale = scale.clamp(0.5, 2.0);
        if self.presentation_font_scale != scale {
            self.presentation_font_scale = scale;
            self.draw_bg.redraw(cx);
        }
    }
    /// A hosted presentation can deepen the surface without changing the PTY
    /// palette or restarting its session. Zero restores the provider theme.
    pub fn set_background_dimming(&mut self, cx: &mut Cx, amount: f32) {
        let amount = amount.clamp(0.0, 0.8);
        if self.background_dimming != amount {
            self.background_dimming = amount;
            self.draw_bg.redraw(cx);
        }
    }
    /// Start the session now rather than on the first draw: a restored
    /// resident that a hidden presentation has not drawn yet still runs.
    pub fn ensure_started(&mut self, cx: &mut Cx) {
        self.ensure_session(cx);
    }
    /// A hosting presentation is hiding this terminal: end any selection
    /// drag and its edge auto-scroll so nothing keeps requesting frames.
    pub fn cancel_gestures(&mut self, cx: &mut Cx) {
        self.press = Press::Idle;
        if self.selecting || self.last_finger.is_some() {
            self.selecting = false;
            self.last_finger = None;
            self.area.redraw(cx);
        }
    }
    pub fn child_pid(&self) -> Option<i32> {
        self.session.as_ref().map(Session::child_pid)
    }

    /// Whether keyboard input currently goes to this terminal's PTY area.
    pub fn has_input_focus(&self, cx: &Cx) -> bool {
        cx.has_key_focus(self.area)
    }

    /// Send the keyboard here (a tab was selected). Before its first frame
    /// the terminal takes focus as it draws.
    pub fn focus(&mut self, cx: &mut Cx) {
        if !self.area.is_empty() {
            cx.set_key_focus(self.area);
        }
        self.draw_bg.redraw(cx);
    }

    /// The job running in the foreground, `None` at the shell prompt.
    pub fn foreground_job(&self) -> Option<String> {
        self.session.as_ref().and_then(Session::foreground_job)
    }

    /// The name of the job, else the shell.
    pub fn foreground_name(&self) -> Option<String> {
        self.session.as_ref().and_then(Session::foreground_name)
    }

    /// Where the shell is: its OSC 7 report, else the process table.
    pub fn current_dir(&self) -> Option<PathBuf> {
        self.session
            .as_ref()
            .and_then(Session::shell_cwd)
            .or_else(|| self.cwd.clone())
    }

    /// Paste `text` the way a person's paste arrives (bracketed when the
    /// program asked for that).
    pub fn ai_paste(&mut self, text: &str) -> bool {
        self.paste_bytes(text)
    }

    /// Whether the session ran and has ended.
    pub fn has_exited(&self) -> bool {
        self.session.as_ref().is_some_and(|session| session.exited)
    }

    /// (background, foreground) of the colours in use, for chrome drawn
    /// around the terminal (the tab bar).
    pub fn chrome_colors(&self) -> Option<(Vec4f, Vec4f)> {
        let (_, fg, bg, _) = self.resolved_colors()?;
        Some((Self::rgb_to_vec4(bg, 1.0), Self::rgb_to_vec4(fg, 1.0)))
    }

    /// Rows currently painted in the widget, or the last `lines` rows of
    /// scrollback plus the active grid. Used by the terminal's AI service;
    /// row text comes from the same cell content the renderer uses.
    pub fn ai_screen_rows(
        &self,
        recent_lines: Option<usize>,
    ) -> Option<(Vec<String>, usize, usize)> {
        let session = self.session.as_ref()?;
        let screen = session.terminal.screen();
        let total = screen.total_rows();
        let (start, end) = match recent_lines {
            Some(lines) => (total.saturating_sub(lines), total),
            None => {
                let start = screen.scrollback.len().saturating_sub(self.view_offset);
                (start, (start + screen.rows).min(total))
            }
        };
        let rows = (start..end)
            .filter_map(|index| screen.row_virtual(index))
            .map(|row| row.text())
            .collect();
        Some((rows, screen.cursor.y, screen.cursor.x))
    }

    /// Type bytes through the live PTY's ordinary input path.
    pub fn ai_type_bytes(&mut self, bytes: &[u8]) -> bool {
        let Some(session) = self.session.as_mut() else {
            return false;
        };
        session.write(bytes);
        true
    }

    /// Insert one file exactly as a native terminal drop: shell-quoted path,
    /// trailing space, normal bracketed-paste protocol, and never Enter.
    /// The caller owns any file inspection/copying; this performs no file I/O.
    pub fn ai_drop_file(&mut self, path: &Path) -> bool {
        let Some(path) = path.to_str().filter(|text| Self::valid_drop_path(text)) else {
            return false;
        };
        let quoted = format!("'{}' ", path.replace('\'', "'\\''"));
        self.paste_bytes(&quoted)
    }

    fn valid_drop_path(path: &str) -> bool {
        Path::new(path).is_absolute() && path.len() <= 4096 && !path.chars().any(char::is_control)
    }

    /// Quick Look retarget (`WmEvent::PreviewFile`): tear down the current
    /// job — dropping the session kills the PTY child — and run a new one
    /// in place, same process, same window.
    pub fn restart_with(&mut self, cx: &mut Cx, cwd: Option<PathBuf>, command: Option<String>) {
        self.session = None;
        self.cwd = cwd;
        self.command = command;
        self.dormant = false;
        self.view_offset = 0;
        self.sel_anchor = None;
        self.sel_cursor = None;
        self.selecting = false;
        self.area.redraw(cx);
    }

    /// Quick Look unload (`WmEvent::PreviewUnload`): drop the job and idle
    /// blank until the next retarget. Not a close — the process stays warm.
    pub fn unload(&mut self, cx: &mut Cx) {
        self.session = None;
        self.dormant = true;
        self.view_offset = 0;
        self.sel_anchor = None;
        self.sel_cursor = None;
        self.selecting = false;
        self.area.redraw(cx);
    }

    fn ensure_session(&mut self, _cx: &mut Cx) {
        if self.session.is_some() || self.dormant {
            return;
        }
        let cols = 80;
        let rows = 24;
        if let Ok(spec) = std::env::var("MAKEPAD_TERMINAL_OPACITY") {
            let mut it = spec
                .split_whitespace()
                .filter_map(|s| s.parse::<f32>().ok());
            if let Some(active) = it.next() {
                let inactive = it.next().unwrap_or(active);
                self.bg_opacity = (active.clamp(0.0, 1.0), inactive.clamp(0.0, 1.0));
            }
        }
        self.load_settings();
        match Session::spawn_with(
            cols,
            rows,
            self.cwd.as_deref(),
            self.command.as_deref(),
            &SpawnOptions::from_settings(&self.settings),
        ) {
            Ok(session) => {
                // makepad-wm hands the splash theme's terminal palette down
                // via MAKEPAD_TERMINAL_COLORS; standalone runs use the bundled default.
                let (mut base16, mut fg, mut bg) = default_theme();
                if let Ok(env) = std::env::var("MAKEPAD_TERMINAL_COLORS") {
                    for pair in env.split(';') {
                        let Some((key, value)) = pair.split_once('=') else {
                            continue;
                        };
                        let Some(rgb) = parse_hex_rgb(value) else {
                            continue;
                        };
                        if let Some(idx) = key.strip_prefix("color") {
                            if let Ok(i) = idx.parse::<usize>() {
                                if i < 16 {
                                    base16[i] = rgb;
                                }
                            }
                        } else if key == "foreground" {
                            fg = rgb;
                        } else if key == "background" {
                            bg = rgb;
                        } else if key == "cursor" {
                            self.original_cursor = Some(rgb);
                        }
                    }
                }
                self.original_colors = Some((base16, fg, bg));
                self.session = Some(session);
                self.apply_colors();
            }
            Err(err) => {
                error!("terminal: failed to spawn shell: {}", err);
            }
        }
    }

    /// Read the live settings once (first draw or spawn).
    fn load_settings(&mut self) {
        if self.settings_gen == 0 {
            self.settings_gen = term_settings::generation();
            self.settings = term_settings::current();
            self.font_size = self.settings.font_size;
            self.draw_text.text_style.line_spacing = self.settings.line_height as f32;
            self.font_features =
                Rc::new(text_run::parse_font_features(&self.settings.font_features));
        }
    }

    fn clear_glyph_caches(&mut self) {
        self.glyph_cache.clear();
        self.run_cache.clear();
    }

    /// Apply a settings change: the live copy's generation moved past the
    /// one this terminal applied (a panel edit, or the file changed).
    fn sync_settings(&mut self, cx: &mut Cx) {
        let generation = term_settings::generation();
        if generation == self.settings_gen {
            return;
        }
        self.settings_gen = generation;
        self.settings = term_settings::current();
        self.font_size = self.settings.font_size;
        self.draw_text.text_style.line_spacing = self.settings.line_height as f32;
        if let Some(session) = self.session.as_mut() {
            session.terminal.set_scrollback(self.settings.scrollback_lines);
        }
        self.apply_colors();
        self.font_features = Rc::new(text_run::parse_font_features(&self.settings.font_features));
        self.apply_fonts(cx);
        self.clear_glyph_caches();
        self.draw_bg.redraw(cx);
    }

    /// Use the font families the settings name. Before the font scan has
    /// finished the bundled font stays; the scan's end re-syncs.
    fn apply_fonts(&mut self, cx: &mut Cx) {
        use term_settings::{CJK_AUTO, CJK_NONE};
        let s = &self.settings;
        if s.font_family.is_empty() && s.cjk_font == CJK_NONE && self.font_key.is_none() {
            return;
        }
        let cache = makepad_widgets::makepad_platform::home::makepad_home().join("terminal").join("fonts");
        // Fonts are prepared off the UI thread; until they all are, the
        // terminal keeps drawing with the fonts it has (the preparation
        // re-syncs every terminal when it is done).
        let mut pending = false;
        let mut path = |face: &crate::fonts::Face| match crate::fonts::prepared_path(face, &cache) {
            crate::fonts::Prepared::Ready(path) => path.map(|p| p.to_string_lossy().into_owned()),
            crate::fonts::Prepared::Pending => {
                pending = true;
                None
            }
        };
        let primary = (!s.font_family.is_empty()).then(|| crate::fonts::find(&s.font_family)).flatten();
        let cjk = match s.cjk_font.as_str() {
            CJK_NONE => None,
            CJK_AUTO => crate::fonts::auto_cjk(),
            name => crate::fonts::find(name),
        };
        let regular = primary.and_then(|family| path(&family.regular));
        let bold = primary
            .and_then(|family| family.bold.as_ref().and_then(|face| path(face)))
            .or_else(|| regular.clone());
        let cjk = cjk.and_then(|family| path(&family.regular)).filter(|p| Some(p) != regular.as_ref());
        if pending {
            return;
        }
        let key = (regular, bold, cjk);
        if self.font_key.as_ref() == Some(&key) {
            return;
        }
        let (regular_style, bold_style, roots) = cx.with_vm(|vm| {
            let regular = terminal_text_style(vm, key.0.as_deref(), key.2.as_deref(), 400.0);
            let bold = terminal_text_style(vm, key.1.as_deref(), key.2.as_deref(), 800.0);
            let roots = [regular, bold]
                .iter()
                .filter_map(|v| v.as_object())
                .map(|obj| vm.bx.heap.new_object_ref(obj))
                .collect::<Vec<_>>();
            (TextStyle::script_from_value(vm, regular), TextStyle::script_from_value(vm, bold), roots)
        });
        // A family that did not build (a host's script context without the
        // resource functions) would draw nothing: keep the fonts in use.
        if regular_style.font_family.member_ids().len() == 0 || bold_style.font_family.member_ids().len() == 0 {
            error!("terminal: the font family could not be built; keeping the current fonts");
            self.font_key = Some(key);
            return;
        }
        self.draw_text.text_style.font_family = regular_style.font_family;
        self.bold_text_style.font_family = bold_style.font_family;
        self.font_roots = roots;
        self.font_key = Some(key);
        self.clear_glyph_caches();
        self.glyph_cache_key = (0, 0, 0);
    }

    /// The palette, default colours and cursor colour to use: a bundled
    /// scheme the person chose, else the host's (desktop style, the palette
    /// makepad-wm hands down, or the built-in default).
    fn resolved_colors(&self) -> Option<([Rgb; 16], Rgb, Rgb, Option<Rgb>)> {
        if self.settings.theme != term_settings::THEME_DESKTOP {
            if let Some(scheme) = themes::find(&self.settings.theme) {
                let rgb = |c: u32| Rgb::new((c >> 16) as u8, (c >> 8) as u8, c as u8);
                let mut base16 = [Rgb::default(); 16];
                for (slot, c) in base16.iter_mut().zip(scheme.base16) {
                    *slot = rgb(c);
                }
                return Some((base16, rgb(scheme.foreground), rgb(scheme.background), Some(rgb(scheme.cursor))));
            }
        }
        let (mut palette, fg, bg) = self.original_colors?;
        let (fg, bg) = match self.style_colors {
            Some((style_fg, style_bg)) => {
                palette[0] = style_bg;
                palette[7] = style_fg;
                (style_fg, style_bg)
            }
            None => (fg, bg),
        };
        Some((palette, fg, bg, self.original_cursor))
    }

    /// Selected cells' (background, text) colours (`settings::selection_colors`).
    fn selection_colors(&self, default_fg: Rgb, default_bg: Rgb) -> (Vec4f, Vec4f) {
        let scheme = (self.settings.theme != term_settings::THEME_DESKTOP)
            .then(|| themes::find(&self.settings.theme))
            .flatten();
        let (bg, fg) =
            term_settings::selection_colors(&self.settings, scheme, default_fg, default_bg);
        (Self::rgb_to_vec4(bg, 1.0), Self::rgb_to_vec4(fg, 1.0))
    }

    fn apply_colors(&mut self) {
        let Some((palette, fg, bg, cursor)) = self.resolved_colors() else {
            return;
        };
        if let Some(session) = self.session.as_mut() {
            session.terminal.cursor_color = cursor;
            session.terminal.set_theme(&palette, fg, bg);
        }
    }

    /// (focused, unfocused) background alpha: the person's setting, else
    /// the host's rule (makepad-wm) or opaque.
    fn effective_opacity(&self) -> (f32, f32) {
        match self.settings.background_opacity {
            Some(a) => (a, (a - 0.08).max(term_settings::OPACITY_RANGE.0)),
            None => self.bg_opacity,
        }
    }

    fn blink_phase_on(&mut self) -> bool {
        let t = self.blink_epoch.get_or_insert_with(Instant::now).elapsed().as_secs_f64();
        ((t / BLINK_HALF_PERIOD) as u64) % 2 == 0
    }

    fn refresh_metrics(&mut self, cx: &mut Cx2d) {
        let font_size = self.font_size * self.presentation_font_scale;
        self.draw_text.text_style.font_size = font_size as f32;
        let key = (
            font_size.to_bits(),
            cx.current_dpi_factor().to_bits(),
            self.raster_scale().to_bits(),
        );
        if key != self.glyph_cache_key {
            self.clear_glyph_caches();
            self.glyph_cache_key = key;
        }
        if let Some(run) = self.draw_text.prepare_single_line_run(cx, "M") {
            let g = &run.glyphs[0];
            self.cell_w = g.advance_in_lpxs as f64;
            let glyph_h = (run.ascender_in_lpxs - run.descender_in_lpxs) as f64;
            self.cell_h = glyph_h * self.draw_text.text_style.line_spacing as f64;
            self.cell_baseline = (self.cell_h - glyph_h) * 0.5 + run.ascender_in_lpxs as f64;
        }
        if self.cell_w <= 0.0 {
            self.cell_w = font_size * 0.6;
        }
        if self.cell_h <= 0.0 {
            self.cell_h = font_size * 1.35;
        }
    }

    fn grid_size(&self) -> (usize, usize) {
        let padding = self.inner_padding();
        let cols = ((self.rect.size.x - padding.x * 2.0) / self.cell_w)
            .floor()
            .max(2.0) as usize;
        let rows = ((self.rect.size.y - padding.y * 2.0) / self.cell_h)
            .floor()
            .max(2.0) as usize;
        (cols, rows)
    }

    fn raster_scale(&self) -> f64 {
        self.canvas_ime_anchor
            .map(|(_, transform)| transform.scale)
            .filter(|scale| scale.is_finite())
            .unwrap_or(1.0)
            .clamp(0.1, 8.0)
    }

    /// Fast path: one codepoint, one glyph, cached by the char.
    fn cached_glyph(
        &mut self,
        cx: &mut Cx2d,
        ch: char,
        bold: bool,
        columns: u8,
    ) -> Option<CachedGlyph> {
        if let Some(hit) = self.glyph_cache.char(ch, bold, columns) {
            return *hit;
        }
        let mut buf = [0u8; 4];
        let text: &str = ch.encode_utf8(&mut buf);
        let fit = if is_private_use(ch) {
            Fit::Icon
        } else {
            Fit::Text
        };
        let cached = self
            .prepare_cell_run(cx, text, bold, columns, fit)
            .and_then(|run| run.first().copied());
        self.glyph_cache.insert_char(ch, bold, columns, cached);
        cached
    }

    /// Slow path: a grapheme cluster (combining marks, a ZWJ emoji sequence,
    /// a flag, a skin tone) shaped as one run and cached by the cluster, so
    /// the font joins and places its parts instead of stacking them.
    #[allow(clippy::too_many_arguments)]
    fn draw_cluster_glyphs(
        &mut self,
        cx: &mut Cx2d,
        cps: &[char],
        bold: bool,
        columns: u8,
        x: f64,
        y: f64,
        color: Vec4f,
    ) {
        if self.glyph_cache.cluster(cps, bold, columns).is_none() {
            let text: String = cps.iter().collect();
            let run = self.prepare_cell_run(cx, &text, bold, columns, Fit::Centered);
            self.glyph_cache.insert_cluster(cps, bold, columns, run);
        }
        let baseline = y + self.cell_baseline;
        if let Some(Some(run)) = self.glyph_cache.cluster(cps, bold, columns) {
            for glyph in run {
                let point = Point::new(
                    (x + glyph.x_offset_in_lpxs as f64) as f32,
                    baseline as f32 + glyph.y_offset_in_lpxs,
                );
                self.draw_text.draw_rasterized_glyph_abs(
                    cx,
                    point,
                    glyph.font_size_in_lpxs,
                    glyph.rasterized,
                    color,
                );
            }
        }
    }

    /// Shape `text` with the text style (bold or not) at the actual screen
    /// size and the settings' font features. Rasterized at the screen size
    /// with local-grid metrics kept (canvas zoom would otherwise scale a
    /// low-resolution glyph cached at 1x): returns the raster scale that
    /// divides the run's lengths back to grid points.
    fn shape_text(
        &mut self,
        cx: &mut Cx2d,
        text: &str,
        bold: bool,
    ) -> Option<(ShapedTextRun, f32)> {
        let scale = self.raster_scale() as f32;
        // Swapped in and out rather than cloned: a family is a list of
        // members, and runs are shaped by the hundred when a screen of new
        // text arrives.
        let font_size = self.draw_text.text_style.font_size;
        if bold {
            std::mem::swap(
                &mut self.draw_text.text_style.font_family,
                &mut self.bold_text_style.font_family,
            );
        }
        self.draw_text.text_style.font_size = font_size * scale;
        let shaped = self
            .draw_text
            .prepare_shaped_run(cx, text, &self.font_features);
        self.draw_text.text_style.font_size = font_size;
        if bold {
            std::mem::swap(
                &mut self.draw_text.text_style.font_family,
                &mut self.bold_text_style.font_family,
            );
        }
        Some((shaped?, scale))
    }

    /// A glyph's ink as (top, bottom) from the baseline, y down, in grid
    /// points.
    fn glyph_ink(glyph: &ShapedTextGlyph, scale: f32) -> Option<(f32, f32)> {
        let raster = glyph.rasterized.as_ref()?;
        let per_dpx = glyph.font_size_in_lpxs / scale / raster.dpxs_per_em;
        let bottom = -raster.origin_in_dpxs.y * per_dpx;
        let top = bottom - raster.atlas_image_bounds.size.height as f32 * per_dpx;
        Some((top, bottom))
    }

    /// Shape a run of neighbouring text cells as one and put its glyphs
    /// back on the grid (`crate::text_run::place_run`). `cell_starts` maps
    /// the cells to bytes of `text`; `centered` marks cluster cells.
    fn prepare_text_run(
        &mut self,
        cx: &mut Cx2d,
        text: &str,
        cell_starts: &[usize],
        centered: &[bool],
        bold: bool,
    ) -> Vec<RunGlyph> {
        let Some((run, scale)) = self.shape_text(cx, text, bold) else {
            return Vec::new();
        };
        let glyphs: Vec<GlyphIn> = run
            .glyphs
            .iter()
            .map(|g| GlyphIn {
                cluster: g.cluster,
                pen_x: g.pen_x_in_lpxs / scale,
                offset_x: g.offset_x_in_lpxs / scale,
                offset_y: g.offset_y_in_lpxs / scale,
                advance: g.advance_in_lpxs / scale,
                ink: Self::glyph_ink(g, scale),
            })
            .collect();
        let mut placed: Vec<Placed> = Vec::with_capacity(glyphs.len());
        text_run::place_run(
            cell_starts,
            centered,
            &glyphs,
            self.cell_w as f32,
            text_run::flows(text),
            &mut placed,
        );
        placed
            .iter()
            .filter_map(|p| {
                let g = &run.glyphs[p.glyph];
                Some(RunGlyph {
                    cell: p.cell as u16,
                    x: p.x,
                    y: p.y,
                    font_size_in_lpxs: g.font_size_in_lpxs / scale * p.scale,
                    rasterized: g.rasterized?,
                })
            })
            .collect()
    }

    /// Shape `text` as one run and fit it into `columns` cells: each glyph
    /// relative to the cell's left edge and baseline. `fit` says whether a
    /// run is kept left, centred, or fitted as an icon.
    fn prepare_cell_run(
        &mut self,
        cx: &mut Cx2d,
        text: &str,
        bold: bool,
        columns: u8,
        fit: Fit,
    ) -> Option<Vec<CachedGlyph>> {
        let (run, scale) = self.shape_text(cx, text, bold)?;
        // Proportional fallback symbols must stay inside the cells
        // allocated by the terminal, without shrinking normal mono glyphs
        // for small rounding differences in the grid advance.
        let available = self.cell_w as f32 * columns.max(1) as f32;
        let fit = fit_glyphs(fit, run.width_in_lpxs / scale, self.cell_w as f32, columns);
        // The vertical middle of the run's ink, so a shrunk run stays
        // centred on the line instead of sinking to the baseline.
        let (mut top, mut bottom) = (f32::MAX, f32::MIN);
        for g in &run.glyphs {
            if let Some((upper, lower)) = Self::glyph_ink(g, scale) {
                top = top.min(upper + g.offset_y_in_lpxs / scale);
                bottom = bottom.max(lower + g.offset_y_in_lpxs / scale);
            }
        }
        let center_y = if top <= bottom {
            (top + bottom) * 0.5
        } else {
            0.0
        };
        // A run starts inside its own cells whatever the font says (some
        // proportional CJK fonts report pen offsets far past one character).
        let first_x = run
            .glyphs
            .first()
            .map(|g| (g.pen_x_in_lpxs + g.offset_x_in_lpxs) / scale * fit.scale)?;
        let x_origin = if first_x.abs() > available {
            first_x
        } else {
            0.0
        };
        let glyphs: Vec<CachedGlyph> = run
            .glyphs
            .iter()
            .filter_map(|g| {
                Some(CachedGlyph {
                    rasterized: g.rasterized?,
                    font_size_in_lpxs: g.font_size_in_lpxs / scale * fit.scale,
                    x_offset_in_lpxs: (g.pen_x_in_lpxs + g.offset_x_in_lpxs) / scale * fit.scale
                        - x_origin
                        + fit.x_shift,
                    y_offset_in_lpxs: g.offset_y_in_lpxs / scale * fit.scale
                        + center_y * (1.0 - fit.scale),
                })
            })
            .collect();
        (!glyphs.is_empty()).then_some(glyphs)
    }

    /// A rule's thickness in whole device pixels, and the device pixels per
    /// layout point (canvas zoom included).
    fn rule_pixels(&self, cx: &Cx2d, heavy: bool) -> (f64, f64) {
        let physical = cx.current_dpi_factor().max(0.1) * self.raster_scale();
        let pixels = (self.font_size * self.presentation_font_scale * 0.07 * physical)
            .round()
            .max(1.0)
            * if heavy { 2.0 } else { 1.0 };
        (pixels, physical)
    }

    /// Powerline separators and shade blocks, drawn to fill their cells
    /// exactly (`crate::sprites`); thin separators use the light box rule.
    fn draw_sprite_glyph(
        &mut self,
        cx: &mut Cx2d,
        ch: char,
        x: f64,
        y: f64,
        columns: u8,
        color: Vec4f,
    ) -> bool {
        let Some(sprite) = sprites::sprite_for(ch) else {
            return false;
        };
        // Most screens have none: open the draw call on the first one.
        if !self.sprites_open {
            self.sprites_open = true;
            self.draw_sprites.new_draw_call(cx);
        }
        let (pixels, physical) = self.rule_pixels(cx, false);
        self.draw_sprites.color = vec4(color.x, color.y, color.z, color.w * sprite.alpha);
        self.draw_sprites.shape = sprite.shape as u8 as f32;
        self.draw_sprites.flip = vec2(sprite.flip_x as u8 as f32, sprite.flip_y as u8 as f32);
        self.draw_sprites.thickness = (pixels / physical) as f32;
        self.draw_sprites.pixel_size = (1.0 / physical) as f32;
        self.draw_sprites.draw_abs(
            cx,
            Rect {
                pos: dvec2(x, y),
                size: dvec2(self.cell_w * columns.max(1) as f64, self.cell_h),
            },
        );
        true
    }

    /// Cell geometry, rather than a font's side bearings/line gap, joins
    /// common TUI borders. Thin strokes occupy whole device pixels; rounded
    /// corners remain curves, and ANSI faint is applied once to the result.
    fn draw_box_glyph(&mut self, cx: &mut Cx2d, ch: char, x: f64, y: f64, color: Vec4f) -> bool {
        let (arms, heavy, rounded) = match ch {
            '─' => (3, false, false),
            '━' => (3, true, false),
            '│' => (12, false, false),
            '┃' => (12, true, false),
            '┌' => (10, false, false),
            '┏' => (10, true, false),
            '╭' => (10, false, true),
            '┐' => (9, false, false),
            '┓' => (9, true, false),
            '╮' => (9, false, true),
            '└' => (6, false, false),
            '┗' => (6, true, false),
            '╰' => (6, false, true),
            '┘' => (5, false, false),
            '┛' => (5, true, false),
            '╯' => (5, false, true),
            '├' => (14, false, false),
            '┣' => (14, true, false),
            '┤' => (13, false, false),
            '┫' => (13, true, false),
            '┬' => (11, false, false),
            '┳' => (11, true, false),
            '┴' => (7, false, false),
            '┻' => (7, true, false),
            '┼' => (15, false, false),
            '╋' => (15, true, false),
            _ => return false,
        };
        let scale = self.raster_scale();
        let dpi = cx.current_dpi_factor().max(0.1);
        let (pixels, physical) = self.rule_pixels(cx, heavy);
        let thickness = pixels / physical;
        let (mx, my) = (x + self.cell_w * 0.5, y + self.cell_h * 0.5);
        let translation = self
            .canvas_ime_anchor
            .map(|(_, transform)| transform.translation)
            .unwrap_or_default();
        let snap = |value: f64, translation: f64| {
            (((value * scale + translation) * dpi - pixels * 0.5).round() + pixels * 0.5) / physical
                - translation / scale
        };
        let (mx, my) = (snap(mx, translation.x), snap(my, translation.y));
        self.draw_boxes.color = color;
        self.draw_boxes.arms = vec4(
            (arms & 1 != 0) as u8 as f32,
            (arms & 2 != 0) as u8 as f32,
            (arms & 4 != 0) as u8 as f32,
            (arms & 8 != 0) as u8 as f32,
        );
        self.draw_boxes.center = vec2((mx - x) as f32, (my - y) as f32);
        self.draw_boxes.corner = vec2(
            if arms & 2 != 0 { 1.0 } else { -1.0 },
            if arms & 8 != 0 { 1.0 } else { -1.0 },
        );
        self.draw_boxes.thickness = thickness as f32;
        self.draw_boxes.radius = if rounded {
            (self.cell_w.min(self.cell_h) * 0.35).max(thickness) as f32
        } else {
            0.0
        };
        self.draw_boxes.pixel_size = (1.0 / physical) as f32;
        self.draw_boxes.draw_abs(
            cx,
            Rect {
                pos: dvec2(x, y),
                size: dvec2(self.cell_w, self.cell_h),
            },
        );
        true
    }

    /// Braille is a fixed two-by-four dot grid, also used by terminal plots
    /// and animated logos. Render all 256 patterns without font fallback.
    fn draw_braille_glyph(&mut self, ch: char, x: f64, y: f64, color: Vec4f) -> bool {
        let Some(pattern) = (ch as u32).checked_sub(0x2800).filter(|v| *v <= 0xff) else {
            return false;
        };
        if pattern == 0 {
            return true;
        }
        self.draw_dots.clear();
        self.draw_dots.set_color(color.x, color.y, color.z, color.w);
        let radius = (self.cell_w * 0.13).min(self.cell_h * 0.065) as f32;
        // Unicode dot numbering: 1,2,3,7 down the left; 4,5,6,8 right.
        for (bit, col, row) in [
            (0, 0, 0),
            (1, 0, 1),
            (2, 0, 2),
            (6, 0, 3),
            (3, 1, 0),
            (4, 1, 1),
            (5, 1, 2),
            (7, 1, 3),
        ] {
            if pattern & (1 << bit) != 0 {
                self.draw_dots.circle(
                    (x + self.cell_w * (0.25 + col as f64 * 0.5)) as f32,
                    (y + self.cell_h * (0.125 + row as f64 * 0.25)) as f32,
                    radius,
                );
            }
        }
        self.draw_dots.fill();
        true
    }

    fn rgb_to_vec4(rgb: Rgb, alpha: f32) -> Vec4f {
        vec4(
            rgb.r as f32 / 255.0,
            rgb.g as f32 / 255.0,
            rgb.b as f32 / 255.0,
            alpha,
        )
    }

    /// Resolve a cell's fg/bg honoring inverse (cell + global DECSCNM),
    /// bold-brightening for the 8 base palette colors, faint and invisible.
    fn resolve_colors(
        session: &Session,
        style: &crate::term::style::Style,
        global_inverse: bool,
    ) -> (Option<Vec4f>, Option<Vec4f>) {
        let term = &session.terminal;
        let mut fg_color = style.fg_color;
        // Classic bold-brightens: palette 0-7 + bold renders 8-15.
        if style.flags.has(StyleFlags::BOLD) {
            if let StyleColor::Palette(i) = fg_color {
                if i < 8 {
                    fg_color = StyleColor::Palette(i + 8);
                }
            }
        }
        let mut fg = fg_color.resolve(&term.palette, term.default_fg);
        let mut bg = style.bg_color.resolve_opt(&term.palette);

        let inverse = style.flags.has(StyleFlags::INVERSE) != global_inverse;
        if inverse {
            let old_fg = fg;
            fg = bg.unwrap_or(term.default_bg);
            bg = Some(old_fg);
        }

        if style.flags.has(StyleFlags::INVISIBLE) {
            return (None, bg.map(|b| Self::rgb_to_vec4(b, 1.0)));
        }
        let alpha = if style.flags.has(StyleFlags::FAINT) {
            0.55
        } else {
            1.0
        };
        (
            Some(Self::rgb_to_vec4(fg, alpha)),
            bg.map(|b| Self::rgb_to_vec4(b, 1.0)),
        )
    }

    // --------------------------------------------------------------
    // Selection
    // --------------------------------------------------------------

    fn sel_ordered(&self) -> Option<((u64, usize), (u64, usize))> {
        let a = self.sel_anchor?;
        let c = self.sel_cursor?;
        if a == c {
            return None;
        }
        Some(if a <= c { (a, c) } else { (c, a) })
    }

    fn selected_text(&self) -> Option<String> {
        let session = self.session.as_ref()?;
        let (start, end) = self.sel_ordered()?;
        let out = session.terminal.screen().selection_text(start, end);
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// Abs row + nearest column boundary (0..=cols) at a window position:
    /// where a character selection starts or ends.
    fn pick_boundary(&self, abs_pos: Vec2d) -> Option<(u64, usize)> {
        let (row, _) = self.pick(abs_pos)?;
        let cols = self.session.as_ref()?.terminal.screen().cols;
        let local_x = abs_pos.x - self.rect.pos.x - self.inner_padding().x;
        Some((row, boundary_col(local_x, self.cell_w, cols)))
    }

    /// Abs row + col at a window position.
    fn pick(&self, abs_pos: Vec2d) -> Option<(u64, usize)> {
        let session = self.session.as_ref()?;
        let screen = session.terminal.screen();
        let padding = self.inner_padding();
        let local_x = abs_pos.x - self.rect.pos.x - padding.x;
        let local_y = abs_pos.y - self.rect.pos.y - padding.y;
        let col = (local_x / self.cell_w).floor().max(0.0) as usize;
        let col = col.min(screen.cols.saturating_sub(1));
        let visual_row = (local_y / self.cell_h).floor().max(0.0) as usize;
        let top_virtual = self.top_virtual_row();
        let virt = (top_virtual + visual_row).min(screen.total_rows().saturating_sub(1));
        Some((screen.absolute_of_virtual(virt), col))
    }

    /// First visible virtual row for the current view offset.
    fn top_virtual_row(&self) -> usize {
        let Some(session) = self.session.as_ref() else {
            return 0;
        };
        let screen = session.terminal.screen();
        screen.scrollback.len().saturating_sub(self.view_offset)
    }

    // --------------------------------------------------------------
    // Input
    // --------------------------------------------------------------

    fn key_opts(&self) -> KeyEncodeOptions {
        let Some(session) = self.session.as_ref() else {
            return KeyEncodeOptions::default();
        };
        let term = &session.terminal;
        KeyEncodeOptions {
            cursor_key_application: term.modes.get(Mode::CursorKeys),
            keypad_key_application: term.modes.get(Mode::KeypadKeys),
            ignore_keypad_with_numlock: term.modes.get(Mode::IgnoreKeypadWithNumlock),
            alt_esc_prefix: term.modes.get(Mode::AltEscPrefix),
            modify_other_keys_state_2: term.modify_other_keys == 2,
            kitty_flags: KittyFlags(term.kitty_flags()),
            backarrow_key_mode: term.modes.get(Mode::BackarrowKeyMode),
        }
    }

    fn mods_of(m: &KeyModifiers) -> KeyMods {
        KeyMods {
            shift: m.shift,
            ctrl: m.control,
            alt: m.alt,
            super_: m.logo,
            caps_lock: false,
            num_lock: false,
        }
    }

    fn send_key(
        &mut self,
        cx: &mut Cx,
        key: Key,
        mods: &KeyModifiers,
        action: KeyAction,
        utf8: &str,
        unshifted: u32,
    ) {
        let opts = self.key_opts();
        let event = TermKeyEvent {
            action,
            key,
            mods: Self::mods_of(mods),
            consumed_mods: KeyMods::default(),
            utf8: utf8.to_string(),
            unshifted_codepoint: unshifted,
        };
        let bytes = encode_key(&event, &opts);
        if !bytes.is_empty() {
            self.scroll_to_bottom();
            if let Some(session) = self.session.as_mut() {
                session.write(&bytes);
            }
            self.redraw(cx);
        }
    }

    /// Write already-encoded key bytes, as a typed key does.
    fn write_key_bytes(&mut self, cx: &mut Cx, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        self.sel_anchor = None;
        self.sel_cursor = None;
        self.scroll_to_bottom();
        if let Some(session) = self.session.as_mut() {
            session.write(bytes);
        }
        self.redraw(cx);
    }

    /// Kitty report-all: a text, numpad or modifier key as one key event,
    /// with the text the platform typed for it when there is one.
    fn send_kitty_text_key(
        &mut self,
        cx: &mut Cx,
        e: &KeyEvent,
        action: KeyAction,
        text: Option<String>,
    ) {
        let opts = self.key_opts();
        let Some((event, composed)) =
            kitty_input::text_key_event(e.key_code, &e.modifiers, action, text.as_deref())
        else {
            if let Some(text) = text {
                self.write_key_bytes(cx, &encode_text(&text, &opts));
            }
            return;
        };
        let mut bytes = encode_key(&event, &opts);
        if let Some(composed) = composed {
            if opts.kitty_flags.has(KittyFlags::REPORT_ASSOCIATED) {
                bytes.extend(encode_text(&composed, &opts));
            }
        }
        self.write_key_bytes(cx, &bytes);
    }

    /// A release, under kitty report-event-types, for a key whose press the
    /// program saw.
    fn send_key_release(&mut self, cx: &mut Cx, e: &KeyEvent) {
        let Some(at) = self.kitty_pressed.iter().position(|k| *k == e.key_code) else {
            return;
        };
        self.kitty_pressed.swap_remove(at);
        if !self.key_opts().kitty_flags.has(KittyFlags::REPORT_EVENTS) {
            return;
        }
        if let Some(key) = Self::map_keycode(e.key_code) {
            self.send_key(cx, key, &e.modifiers, KeyAction::Release, "", 0);
        } else {
            self.send_kitty_text_key(cx, e, KeyAction::Release, None);
        }
    }

    /// Kitty report-all: the held text no key-down claimed goes out as text.
    fn flush_unpaired_text(&mut self, cx: &mut Cx) {
        if let Some(text) = self.kitty_pairing.take_unpaired() {
            let bytes = encode_text(&text, &self.key_opts());
            self.write_key_bytes(cx, &bytes);
        }
    }

    fn paste(&mut self, cx: &mut Cx, text: &str) {
        if self.paste_bytes(text) || self.input_notice.is_some() {
            self.redraw(cx);
        }
    }

    fn paste_bytes(&mut self, text: &str) -> bool {
        let Some(bracketed) = self
            .session
            .as_ref()
            .filter(|session| !session.exited)
            .map(|s| s.terminal.modes.get(Mode::BracketedPaste))
        else {
            return false;
        };
        let mut bytes = Vec::with_capacity(text.len() + 16);
        if bracketed {
            bytes.extend_from_slice(b"\x1b[200~");
            // Sanitize: no escape bytes inside a bracketed paste body.
            bytes.extend(text.bytes().filter(|b| *b != 0x1b));
            bytes.extend_from_slice(b"\x1b[201~");
        } else {
            // Legacy paste: newlines become CR like every terminal.
            for b in text.bytes() {
                bytes.push(if b == b'\n' { b'\r' } else { b });
            }
        }
        self.scroll_to_bottom();
        let Some(session) = self.session.as_mut() else {
            return false;
        };
        match session.write_paste(&bytes) {
            Ok(()) => true,
            // Never silently: say that (and why) the paste went nowhere.
            Err(InputRejected::QueueFull { pending }) => {
                self.input_notice = Some(format!(
                    "Paste dropped: the program is not reading its input ({} KiB still queued)",
                    pending.div_ceil(1024)
                ));
                self.input_notice_armed = false;
                false
            }
            Err(InputRejected::Closed) => false,
        }
    }

    fn scroll_to_bottom(&mut self) {
        self.view_offset = 0;
    }

    fn mouse_tracking(&self) -> (MouseTracking, MouseFormat) {
        let Some(session) = self.session.as_ref() else {
            return (MouseTracking::None, MouseFormat::X10);
        };
        let m = &session.terminal.modes;
        let tracking = if m.get(Mode::MouseEventAny) {
            MouseTracking::Any
        } else if m.get(Mode::MouseEventButton) {
            MouseTracking::Button
        } else if m.get(Mode::MouseEventNormal) {
            MouseTracking::Normal
        } else if m.get(Mode::MouseEventX10) {
            MouseTracking::X10
        } else {
            return (MouseTracking::None, MouseFormat::X10);
        };
        let format = if m.get(Mode::MouseFormatSgrPixels) {
            MouseFormat::SgrPixels
        } else if m.get(Mode::MouseFormatSgr) {
            MouseFormat::Sgr
        } else if m.get(Mode::MouseFormatUrxvt) {
            MouseFormat::Urxvt
        } else if m.get(Mode::MouseFormatUtf8) {
            MouseFormat::Utf8
        } else {
            MouseFormat::X10
        };
        (tracking, format)
    }

    fn mouse_cell(&self, abs: Vec2d) -> (u32, u32, u32, u32) {
        let padding = self.inner_padding();
        let (cols, rows) = self.session.as_ref().map(|session| {
            let screen = session.terminal.screen();
            (screen.cols, screen.rows)
        }).unwrap_or_else(|| self.grid_size());
        let x = (abs.x - self.rect.pos.x - padding.x)
            .clamp(0.0, (cols as f64 * self.cell_w - 1.0).max(0.0));
        let y = (abs.y - self.rect.pos.y - padding.y)
            .clamp(0.0, (rows as f64 * self.cell_h - 1.0).max(0.0));
        let col = ((x / self.cell_w).floor() as u32).min(cols.saturating_sub(1) as u32);
        let row = ((y / self.cell_h).floor() as u32).min(rows.saturating_sub(1) as u32);
        (col, row, x as u32, y as u32)
    }

    /// Send a mouse report if the app asked for it; true when consumed.
    fn report_mouse(
        &mut self,
        cx: &mut Cx,
        abs: Vec2d,
        kind: MouseEventKind,
        button: TermMouseButton,
        mods: &KeyModifiers,
    ) -> bool {
        let (tracking, format) = self.mouse_tracking();
        if tracking == MouseTracking::None || mods.shift {
            return false;
        }
        let (col, row, x_px, y_px) = self.mouse_cell(abs);
        // Motion dedup: only report when the cell (or button) changed.
        if kind == MouseEventKind::Motion {
            let sig = (col, row, button as u8);
            if self.last_mouse_cell == Some(sig) {
                return true;
            }
            self.last_mouse_cell = Some(sig);
        } else {
            self.last_mouse_cell = None;
        }
        let report = MouseReport {
            kind,
            button,
            mods: Self::mods_of(mods),
            col,
            row,
            x_px,
            y_px,
        };
        if let Some(bytes) = encode_mouse(&report, tracking, format) {
            if let Some(session) = self.session.as_mut() {
                session.write(&bytes);
                self.redraw(cx);
            }
        }
        true
    }

    /// Whole lines a scroll event is worth, and whether they go down. None
    /// when it moves nothing: a trackpad or Magic Mouse sends zero-delta
    /// contact events (and sub-pixel drift) while a finger merely rests on
    /// it, and a sideways swipe has no vertical part. Precise deltas add up
    /// to a line of cell height before one is sent, so resting on the device
    /// never scrolls a mouse-reporting app like Claude Code; a notched wheel
    /// stays at least one line per notch.
    /// Select the word (or, for a line, the whole row) at `abs` and start a
    /// drag that extends by that unit.
    fn begin_unit_selection(&mut self, cx: &mut Cx, abs: Vec2d, whole_line: bool) {
        let Some(pos) = self.pick(abs) else {
            return;
        };
        let (start, end) = if whole_line {
            (0, self.session.as_ref().map(|s| s.terminal.cols()).unwrap_or(80))
        } else {
            self.word_range(pos)
        };
        self.sel_unit = Some((pos.0, start, end));
        self.sel_anchor = Some((pos.0, start));
        self.sel_cursor = Some((pos.0, end));
        self.selecting = true;
        self.last_finger = Some(abs);
        self.select_scroll_frame = cx.new_next_frame();
        self.draw_bg.redraw(cx);
    }

    /// Extend the selection from the grabbed unit to the cell at `abs`,
    /// across as many lines as the pointer has moved.
    fn extend_selection(&mut self, abs: Vec2d) {
        let Some(pos) = self.pick(abs) else {
            return;
        };
        let Some((row, start, end)) = self.sel_unit else {
            // A character selection runs between column boundaries.
            if let Some(pos) = self.pick_boundary(abs) {
                self.sel_cursor = Some(pos);
            }
            return;
        };
        let cols = self.session.as_ref().map(|s| s.terminal.cols()).unwrap_or(80);
        let whole_line = start == 0 && end >= cols;
        if pos < (row, start) {
            self.sel_anchor = Some((row, end));
            self.sel_cursor = Some(if whole_line { (pos.0, 0) } else { pos });
        } else {
            self.sel_anchor = Some((row, start));
            self.sel_cursor = Some(if whole_line || pos.0 == row && pos.1 < end {
                (pos.0, if whole_line { cols } else { end })
            } else {
                (pos.0, (pos.1 + 1).min(cols))
            });
        }
    }

    /// A finger of a two-finger drag on a touchscreen moved by `dy`: the
    /// screen follows the fingers, by whole lines of the drag.
    fn touch_scroll(&mut self, cx: &mut Cx, abs: Vec2d, dy: f64, modifiers: &KeyModifiers) {
        let Some((alt_scroll, max)) = self.session.as_ref().map(|session| {
            let term = &session.terminal;
            (
                matches!(term.active, crate::term::terminal::ActiveScreen::Alternate)
                    && term.modes.get(Mode::MouseAlternateScroll),
                term.screen().scrollback.len(),
            )
        }) else {
            return;
        };
        // Both fingers report the drag; each carries half of it. Fingers
        // moving up show newer output, as on any touch list.
        let share = -dy / self.touches.len().max(1) as f64;
        if let Some((lines, down)) = scroll_step(&mut self.scroll_accum, share, false, false, self.cell_h) {
            self.scroll_by(cx, abs, lines, down, modifiers, alt_scroll, max);
        }
    }

    fn scroll_lines(&mut self, e: &FingerScrollEvent) -> Option<(usize, bool)> {
        scroll_step(
            &mut self.scroll_accum,
            e.scroll.y,
            e.is_mouse,
            matches!(e.phase, makepad_widgets::makepad_platform::event::ScrollPhase::Began),
            self.cell_h,
        )
    }

    fn handle_scroll(&mut self, cx: &mut Cx, e: &FingerScrollEvent) {
        let Some((alt_scroll, max)) = self.session.as_ref().map(|session| {
            let term = &session.terminal;
            (
                matches!(term.active, crate::term::terminal::ActiveScreen::Alternate)
                    && term.modes.get(Mode::MouseAlternateScroll),
                term.screen().scrollback.len(),
            )
        }) else {
            return;
        };
        let Some((lines, down)) = self.scroll_lines(e) else {
            return;
        };
        self.scroll_by(cx, e.abs, lines, down, &e.modifiers, alt_scroll, max);
    }

    /// Move `lines` towards newer output (`down`) or older: wheel reports to
    /// an app that asked for the mouse, arrow keys under alternate scroll,
    /// else the scrollback view.
    fn scroll_by(
        &mut self,
        cx: &mut Cx,
        abs: Vec2d,
        lines: usize,
        down: bool,
        modifiers: &KeyModifiers,
        alt_scroll: bool,
        max: usize,
    ) {
        let (tracking, _) = self.mouse_tracking();
        if tracking != MouseTracking::None && !modifiers.shift {
            let button = if down {
                TermMouseButton::WheelDown
            } else {
                TermMouseButton::WheelUp
            };
            for _ in 0..lines.min(8) {
                self.report_mouse(cx, abs, MouseEventKind::Press, button, modifiers);
            }
            return;
        }

        if alt_scroll {
            // Alternate scroll: wheel becomes arrow keys.
            let key = if down { Key::ArrowDown } else { Key::ArrowUp };
            for _ in 0..lines.min(8) {
                self.send_key(cx, key, &KeyModifiers::default(), KeyAction::Press, "", 0);
            }
            return;
        }

        // Scrollback.
        if down {
            self.view_offset = self.view_offset.saturating_sub(lines);
        } else {
            self.view_offset = (self.view_offset + lines).min(max);
        }
        self.redraw(cx);
    }

    fn map_keycode(kc: KeyCode) -> Option<Key> {
        Some(match kc {
            KeyCode::ReturnKey => Key::Enter,
            KeyCode::NumpadEnter => Key::NumpadEnter,
            KeyCode::Tab => Key::Tab,
            KeyCode::Backspace => Key::Backspace,
            KeyCode::Escape => Key::Escape,
            KeyCode::Delete => Key::Delete,
            KeyCode::Insert => Key::Insert,
            KeyCode::Home => Key::Home,
            KeyCode::End => Key::End,
            KeyCode::PageUp => Key::PageUp,
            KeyCode::PageDown => Key::PageDown,
            KeyCode::ArrowUp => Key::ArrowUp,
            KeyCode::ArrowDown => Key::ArrowDown,
            KeyCode::ArrowLeft => Key::ArrowLeft,
            KeyCode::ArrowRight => Key::ArrowRight,
            KeyCode::F1 => Key::F1,
            KeyCode::F2 => Key::F2,
            KeyCode::F3 => Key::F3,
            KeyCode::F4 => Key::F4,
            KeyCode::F5 => Key::F5,
            KeyCode::F6 => Key::F6,
            KeyCode::F7 => Key::F7,
            KeyCode::F8 => Key::F8,
            KeyCode::F9 => Key::F9,
            KeyCode::F10 => Key::F10,
            KeyCode::F11 => Key::F11,
            KeyCode::F12 => Key::F12,
            _ => return None,
        })
    }

    fn is_special(kc: KeyCode) -> bool {
        Self::map_keycode(kc).is_some()
    }

    // --------------------------------------------------------------
    // Drawing
    // --------------------------------------------------------------

    fn draw_terminal_background(&mut self, cx: &mut Cx2d, bg: Rgb, inverse: bool) {
        let alpha = if self.opaque_style { 1.0 }
            else if cx.has_key_focus(self.area) { self.effective_opacity().0 }
            else { self.effective_opacity().1 };
        let mut v = Self::rgb_to_vec4(bg, alpha);
        if !inverse {
            let scale = 1.0 - self.background_dimming;
            v.x *= scale;
            v.y *= scale;
            v.z *= scale;
        }
        self.draw_bg.draw_vars.set_uniform(cx, id!(color), &[v.x, v.y, v.z, v.w]);
        let (radius, width, color) = self.presentation_frame.map(|frame| {
            // The stroke extends both ways from its inset. Keep the inside
            // edge clear of the grid even at the smallest canvas zoom.
            (frame.radius, frame.border_width.min(frame.padding as f32 * 0.5), frame.border_color)
        }).unwrap_or((0.0, 0.0, vec4(0.0, 0.0, 0.0, 0.0)));
        self.draw_bg.draw_vars.set_uniform(cx, id!(corner_radius), &[radius]);
        self.draw_bg.draw_vars.set_uniform(cx, id!(frame_width), &[width]);
        self.draw_bg.draw_vars.set_uniform(cx, id!(frame_color), &[color.x, color.y, color.z, color.w]);
        self.draw_bg.draw_abs(cx, self.rect);
    }

    fn draw_terminal(&mut self, cx: &mut Cx2d) {
        // The session moves out of self for the draw so cached-glyph and
        // selection helpers can borrow self freely.
        let Some(mut session) = self.session.take() else {
            let bg = self.style_colors.map(|(_, bg)| bg)
                .or_else(|| self.original_colors.map(|(_, _, bg)| bg))
                .unwrap_or_else(|| default_theme().2);
            self.draw_terminal_background(cx, bg, false);
            return;
        };
        self.draw_terminal_inner(cx, &mut session);
        self.session = Some(session);
        self.draw_link_status(cx);
        self.draw_input_notice(cx);
    }

    fn draw_input_notice(&mut self, cx: &mut Cx2d) {
        let Some(notice) = self.input_notice.clone() else {
            return;
        };
        if !self.input_notice_armed {
            self.input_notice_armed = true;
            self.input_notice_timer = cx.start_timeout(INPUT_NOTICE_SECONDS);
        }
        let pad = 6.0;
        let width = (notice.chars().count() as f64 * self.cell_w + 2.0 * pad).min(self.rect.size.x);
        let height = self.cell_h + 2.0 * pad;
        let pos = dvec2(
            self.rect.pos.x + (self.rect.size.x - width).max(0.0) * 0.5,
            self.rect.pos.y + (self.rect.size.y - height - 8.0).max(0.0),
        );
        self.draw_cell_bg.new_draw_call(cx);
        self.draw_cell_bg.color = vec4(0.55, 0.12, 0.10, 0.92);
        self.draw_cell_bg.draw_abs(
            cx,
            Rect {
                pos,
                size: dvec2(width, height),
            },
        );
        self.draw_text.new_draw_call(cx);
        let color = self.draw_text.color;
        self.draw_text.color = vec4(1.0, 1.0, 1.0, 1.0);
        self.draw_text.draw_abs(cx, pos + dvec2(pad, pad), &notice);
        self.draw_text.color = color;
    }

    fn draw_terminal_inner(&mut self, cx: &mut Cx2d, session: &mut Session) {
        let output_changed = std::mem::take(&mut session.terminal.dirty);
        // Search: matches follow the output; history is read a step a frame.
        self.search_frame(cx, &session.terminal, output_changed);

        let has_focus = cx.has_key_focus(self.area);
        let padding = self.inner_padding();
        let origin_x = self.rect.pos.x + padding.x;
        let origin_y = self.rect.pos.y + padding.y;
        let (cell_w, cell_h) = (self.cell_w, self.cell_h);
        let global_inverse = session.terminal.modes.get(Mode::ReverseColors);
        let cursor_visible = session.terminal.modes.get(Mode::CursorVisible);
        let (default_bg, default_fg, cursor_color, cursor_style) = {
            let t = &session.terminal;
            (
                t.default_bg,
                t.default_fg,
                t.cursor_color.unwrap_or(t.default_fg),
                t.cursor_style,
            )
        };
        let cursor_style = effective_cursor_style(cursor_style, &self.settings);
        let blinking = has_focus
            && matches!(
                cursor_style,
                CursorStyle::BlinkingBlock | CursorStyle::BlinkingBar | CursorStyle::BlinkingUnderline
            );
        let blink_on = !blinking || self.blink_phase_on();
        if blinking && !self.blink_armed {
            self.blink_armed = true;
            self.blink_timer = cx.start_timeout(BLINK_HALF_PERIOD);
        }

        // Background fill honoring DECSCNM.
        let bg_fill = if global_inverse {
            default_fg
        } else {
            default_bg
        };
        self.draw_terminal_background(cx, bg_fill, global_inverse);

        let screen = session.terminal.screen();
        let rows = screen.rows;
        let cols = screen.cols;
        let top_virtual = screen.scrollback.len().saturating_sub(self.view_offset);
        let total = screen.total_rows();

        // Collect draw data first (bg runs, glyphs, decorations), then issue
        // batched draws per layer.
        struct BgRun {
            x: f64,
            y: f64,
            w: f64,
            color: Vec4f,
        }
        struct GlyphDraw<'a> {
            x: f64,
            y: f64,
            glyph: CellGlyph<'a>,
            color: Vec4f,
            bold: bool,
            columns: u8,
        }
        struct DecoDraw {
            x: f64,
            y: f64,
            w: f64,
            color: Vec4f,
            kind: f32,
            strike: bool,
        }
        let (sel_bg, sel_fg) = self.selection_colors(default_fg, default_bg);
        // Widened to whole wide chars and clusters, as copy sees it.
        let selection = self
            .sel_ordered()
            .map(|(start, end)| session.terminal.screen().snap_selection(start, end));
        let search_colors = self.search_colors();
        let mut search_marks: Vec<u8> = Vec::new();
        let min_contrast = self.settings.minimum_contrast;
        let link_hover = self.link_hover.clone();
        let bg_fill_color = Self::rgb_to_vec4(bg_fill, 1.0);
        // Neighbouring cells mostly share colours: adjust each pair once.
        let mut contrast_memo: Option<(Vec4f, Vec4f, Vec4f)> = None;
        let mut bg_runs: Vec<BgRun> = Vec::new();
        let mut glyphs: Vec<GlyphDraw> = Vec::with_capacity(rows * cols / 2);
        let mut decos: Vec<DecoDraw> = Vec::new();
        // The frame's text runs, in buffers kept from frame to frame.
        let mut scratch = std::mem::take(&mut self.run_scratch);
        scratch.clear();
        let RunScratch {
            runs,
            text: run_text,
            starts: run_starts,
            colors: run_colors,
            centered: run_centered,
            seg,
            seg_runs,
        } = &mut scratch;
        // Each text cell's colour and content in this row.
        let mut row_text: Vec<Option<(Vec4f, CellGlyph)>> = Vec::with_capacity(cols);

        // The cursor's cell, blinking or not: a ligature under it is broken
        // (shaped cell by cell) whichever phase the blink is in, so the
        // text does not change shape as the cursor blinks.
        // On a wide char or a multi-cell cluster it covers the whole unit,
        // from its head: (column, row, width).
        let cursor_cell = if self.view_offset == 0 && cursor_visible && !session.exited {
            let s = session.terminal.screen();
            let (col, width) = s.row(s.cursor.y).span_at(s.cursor.x.min(cols - 1));
            Some((col, s.cursor.y, width.min(cols - col)))
        } else {
            None
        };

        for vis_row in 0..rows {
            let virt = top_virtual + vis_row;
            if virt >= total {
                break;
            }
            let abs = screen.absolute_of_virtual(virt);
            let row = match screen.row_virtual(virt) {
                Some(r) => r,
                None => continue,
            };
            let y = origin_y + vis_row as f64 * cell_h;
            let mut run: Option<(usize, usize, Vec4f)> = None;
            seg.clear();
            row_text.clear();
            self.search_row_marks(abs, cols, &mut search_marks);
            for col in 0..cols {
                seg.push(SegCell::Blank);
                row_text.push(None);
                let cell = row.cell(col);
                let (fg, bg) = match cell {
                    Some(c) => Self::resolve_colors(session, &c.style, global_inverse),
                    None => (None, None),
                };
                // Selected cells take the selection colours: by default
                // inverse video (the default text color behind the default
                // background color), readable in light and dark themes alike.
                let selected = in_selection(selection, abs, col);
                // Search matches: the selection wins over them.
                let mark = search_marks.get(col).copied().unwrap_or(0);
                let found = Self::search_cell_colors(search_colors, mark);
                let (fg, bg) = match found {
                    _ if selected => (fg.map(|_| sel_fg), Some(sel_bg)),
                    Some((found_bg, found_fg)) => (fg.map(|_| found_fg), Some(found_bg)),
                    None => (fg, bg),
                };

                // Merge bg runs.
                match (&mut run, bg) {
                    (Some((_, end, color)), Some(c)) if *color == c && *end == col => {
                        *end = col + 1;
                    }
                    (prev, next) => {
                        if let Some((start, end, color)) = prev.take() {
                            bg_runs.push(BgRun {
                                x: origin_x + start as f64 * cell_w,
                                y,
                                w: (end - start) as f64 * cell_w,
                                color,
                            });
                        }
                        if let Some(c) = next {
                            *prev = Some((col, col + 1, c));
                        }
                    }
                }

                let Some(cell) = cell else { continue };
                let Some(fg) = fg else { continue };
                let glyph = cell_glyph(&cell.content);

                // Minimum contrast: text only (not a selection's chosen
                // colours, nor box, block and Powerline shapes, whose colour
                // is a fill matched to their neighbours).
                let fg = if min_contrast > contrast::OFF
                    && !selected
                    && found.is_none()
                    && !matches!(glyph, Some(CellGlyph::Char(ch)) if sprites::is_graphic(ch))
                {
                    let back = bg.unwrap_or(bg_fill_color);
                    match contrast_memo {
                        Some((f, b, out)) if f == fg && b == back => out,
                        _ => {
                            let out = contrast_fg(fg, back, min_contrast);
                            contrast_memo = Some((fg, back, out));
                            out
                        }
                    }
                } else {
                    fg
                };

                // Decorations.
                let underline = cell.style.flags.underline();
                let strike = cell.style.flags.has(StyleFlags::STRIKETHROUGH);
                if underline != crate::term::style::Underline::None || strike {
                    let ul_color = cell
                        .style
                        .underline_color
                        .resolve_opt(&session.terminal.palette)
                        .map(|c| Self::rgb_to_vec4(c, 1.0))
                        .unwrap_or(fg);
                    let width = cell.content.width().max(1) as f64 * cell_w;
                    if underline != crate::term::style::Underline::None {
                        decos.push(DecoDraw {
                            x: origin_x + col as f64 * cell_w,
                            y,
                            w: width,
                            color: ul_color,
                            kind: underline as u8 as f32,
                            strike: false,
                        });
                    }
                    if strike {
                        decos.push(DecoDraw {
                            x: origin_x + col as f64 * cell_w,
                            y,
                            w: width,
                            color: fg,
                            kind: 1.0,
                            strike: true,
                        });
                    }
                }
                // The hovered link underlines (unless already underlined).
                if underline == crate::term::style::Underline::None
                    && cell.content.width() > 0
                    && link_hover
                        .as_ref()
                        .is_some_and(|h| h.covers(abs, col, cell.hyperlink))
                {
                    decos.push(DecoDraw {
                        x: origin_x + col as f64 * cell_w,
                        y,
                        w: cell.content.width() as f64 * cell_w,
                        color: fg,
                        kind: 1.0,
                        strike: false,
                    });
                }

                // Text: narrow text cells are shaped with their neighbours
                // below; anything else is drawn on its own, one glyph for a
                // codepoint, one shaped run for a grapheme cluster.
                if let Some(glyph) =
                    glyph.filter(|&g| text_run::shapes_with_neighbours(g, cell.content.width()))
                {
                    seg[col] = SegCell::Text {
                        style: RunStyle {
                            bold: cell.style.flags.has(StyleFlags::BOLD),
                            italic: cell.style.flags.has(StyleFlags::ITALIC),
                        },
                        selected: selected || found.is_some(),
                    };
                    row_text[col] = Some((fg, glyph));
                } else if let Some(glyph) = glyph {
                    seg[col] = SegCell::Own;
                    let mut columns = cell.content.width();
                    let icon = matches!(glyph, CellGlyph::Char(ch) if is_private_use(ch) && !sprites::is_graphic(ch));
                    if icon {
                        let next_is_blank = col + 1 < cols
                            && row
                                .cell(col + 1)
                                .is_none_or(|next| cell_glyph(&next.content).is_none());
                        columns = icon_columns(columns, next_is_blank);
                    }
                    glyphs.push(GlyphDraw {
                        x: origin_x + col as f64 * cell_w,
                        y,
                        glyph,
                        color: fg,
                        bold: cell.style.flags.has(StyleFlags::BOLD),
                        columns,
                    });
                }
            }
            if let Some((start, end, color)) = run.take() {
                bg_runs.push(BgRun {
                    x: origin_x + start as f64 * cell_w,
                    y,
                    w: (end - start) as f64 * cell_w,
                    color,
                });
            }

            // Text runs: shaped across cells, split at blanks, styles, the
            // selection's edges and the cursor.
            let cursor_col = cursor_cell
                .filter(|&(_, cy, _)| cy == vis_row)
                .map(|(cx, _, width)| cx..cx + width);
            text_run::segment_row(seg, cursor_col, seg_runs);
            for range in seg_runs.iter() {
                let text_start = run_text.len();
                let cells_start = run_starts.len();
                for (color, glyph) in row_text[range.clone()].iter().flatten() {
                    run_starts.push(run_text.len() - text_start);
                    run_colors.push(*color);
                    match glyph {
                        CellGlyph::Char(ch) => run_text.push(*ch),
                        CellGlyph::Cluster(cps) => run_text.extend(cps.iter()),
                    }
                    run_centered.push(matches!(glyph, CellGlyph::Cluster(_)));
                }
                let SegCell::Text { style, .. } = seg[range.start] else {
                    continue;
                };
                runs.push(RunDraw {
                    x: origin_x + range.start as f64 * cell_w,
                    y,
                    col: range.start,
                    row: vis_row,
                    text: text_start..run_text.len(),
                    cells: cells_start..run_starts.len(),
                    style,
                });
            }
        }

        // Cursor (only when the live bottom is in view).
        let cursor = cursor_cell.filter(|_| blink_on);
        let block_cursor = has_focus
            && matches!(
                cursor_style,
                CursorStyle::Default | CursorStyle::BlinkingBlock | CursorStyle::SteadyBlock
            );

        // Layer 1: backgrounds.
        self.draw_cell_bg.new_draw_call(cx);
        for r in &bg_runs {
            self.draw_cell_bg.color = r.color;
            self.draw_cell_bg.draw_abs(
                cx,
                Rect {
                    pos: dvec2(r.x, r.y),
                    size: dvec2(r.w, cell_h),
                },
            );
        }

        // Layer 2: cursor under text (block) — text stays readable on top.
        if let Some((cx_col, cx_row, cx_width)) = cursor {
            let x = origin_x + cx_col as f64 * cell_w;
            // A block or underline covers every cell of the unit under it.
            let unit_w = cx_width.max(1) as f64 * cell_w;
            let y = origin_y + cx_row as f64 * cell_h;
            let color = Self::rgb_to_vec4(cursor_color, 1.0);
            self.draw_cursor.new_draw_call(cx);
            self.draw_cursor.color = color;
            self.draw_cursor.hollow = if has_focus { 0.0 } else { 1.0 };
            let (rect, hollow_override) = match cursor_style {
                CursorStyle::BlinkingBar | CursorStyle::SteadyBar => (
                    Rect {
                        pos: dvec2(x, y),
                        size: dvec2((cell_w * 0.15).max(1.5), cell_h),
                    },
                    Some(0.0),
                ),
                CursorStyle::BlinkingUnderline | CursorStyle::SteadyUnderline => (
                    Rect {
                        pos: dvec2(x, y + cell_h - (cell_h * 0.12).max(2.0)),
                        size: dvec2(unit_w, (cell_h * 0.12).max(2.0)),
                    },
                    Some(0.0),
                ),
                _ => (
                    Rect {
                        pos: dvec2(x, y),
                        size: dvec2(unit_w, cell_h),
                    },
                    None,
                ),
            };
            if let Some(h) = hollow_override {
                if !has_focus {
                    // Non-block cursors just dim when unfocused.
                    self.draw_cursor.color = vec4(color.x, color.y, color.z, 0.5);
                }
                self.draw_cursor.hollow = h;
            }
            self.draw_cursor.draw_abs(cx, rect);
        }

        // Layer 3: glyphs, one batch.
        self.draw_boxes.new_draw_call(cx);
        self.sprites_open = false;
        self.draw_dots.begin();
        self.draw_text.new_draw_call(cx);
        self.draw_text.begin_many_instances(cx);
        let baseline = self.cell_baseline;
        for g in &glyphs {
            // A block cursor inverts the glyph on top of it for contrast.
            let mut color = g.color;
            if let Some((ccol, crow, _)) = cursor {
                let gx = ((g.x - origin_x) / cell_w).round() as usize;
                let gy = ((g.y - origin_y) / cell_h).round() as usize;
                if gx == ccol
                    && gy == crow
                    && has_focus
                    && matches!(
                        cursor_style,
                        CursorStyle::Default
                            | CursorStyle::BlinkingBlock
                            | CursorStyle::SteadyBlock
                    )
                {
                    color = Self::rgb_to_vec4(default_bg, 1.0);
                }
            }
            let ch = match g.glyph {
                CellGlyph::Char(ch) => ch,
                CellGlyph::Cluster(cps) => {
                    self.draw_cluster_glyphs(cx, cps, g.bold, g.columns, g.x, g.y, color);
                    continue;
                }
            };
            // Everything drawn without a font sits at U+2500 and above.
            if ch >= '\u{2500}' {
                if self.draw_sprite_glyph(cx, ch, g.x, g.y, g.columns, color) {
                    continue;
                }
                if self.draw_box_glyph(cx, ch, g.x, g.y, color) {
                    continue;
                }
                if self.draw_braille_glyph(ch, g.x, g.y, color) {
                    continue;
                }
            }
            if let Some(glyph) = self.cached_glyph(cx, ch, g.bold, g.columns) {
                let point = Point::new(
                    (g.x + glyph.x_offset_in_lpxs as f64) as f32,
                    (g.y + baseline) as f32 + glyph.y_offset_in_lpxs,
                );
                self.draw_text.draw_rasterized_glyph_abs(
                    cx,
                    point,
                    glyph.font_size_in_lpxs,
                    glyph.rasterized,
                    color,
                );
            }
        }
        for r in runs.iter() {
            let text = &run_text[r.text.clone()];
            let starts = &run_starts[r.cells.clone()];
            if self.run_cache.get(text, starts, r.style).is_none() {
                let placed = self.prepare_text_run(
                    cx,
                    text,
                    starts,
                    &run_centered[r.cells.clone()],
                    r.style.bold,
                );
                self.run_cache.insert(text, starts, r.style, placed);
            }
            let Some(glyphs) = self.run_cache.get(text, starts, r.style) else {
                continue;
            };
            let colors = &run_colors[r.cells.clone()];
            for glyph in glyphs {
                let cell = glyph.cell as usize;
                // A block cursor inverts the glyph on top of it for contrast.
                let color = if block_cursor
                    && cursor.is_some_and(|(ccol, crow, _)| (ccol, crow) == (r.col + cell, r.row))
                {
                    Self::rgb_to_vec4(default_bg, 1.0)
                } else {
                    colors[cell]
                };
                let point = Point::new(
                    (r.x + glyph.x as f64) as f32,
                    (r.y + baseline) as f32 + glyph.y,
                );
                self.draw_text.draw_rasterized_glyph_abs(
                    cx,
                    point,
                    glyph.font_size_in_lpxs,
                    glyph.rasterized,
                    color,
                );
            }
        }
        self.run_scratch = scratch;
        self.draw_text.end_many_instances(cx);
        self.draw_dots.end(cx);

        // Layer 4: decorations.
        if !decos.is_empty() {
            self.draw_underline.new_draw_call(cx);
            for d in &decos {
                self.draw_underline.color = d.color;
                self.draw_underline.kind = d.kind;
                let (dy, dh) = if d.strike {
                    (d.y + cell_h * 0.5 - 1.0, 1.5)
                } else if d.kind >= 2.5 && d.kind < 3.5 {
                    // Curly gets a taller band.
                    (d.y + cell_h - 4.0, 4.0)
                } else if d.kind >= 1.5 && d.kind < 2.5 {
                    (d.y + cell_h - 4.0, 4.0)
                } else {
                    (d.y + cell_h - 2.0, 1.5)
                };
                self.draw_underline.draw_abs(
                    cx,
                    Rect {
                        pos: dvec2(d.x, dy),
                        size: dvec2(d.w, dh),
                    },
                );
            }
        }

        // Bell flash.
        if self.bell_frames > 0 {
            self.bell_frames -= 1;
            self.draw_cell_bg.new_draw_call(cx);
            self.draw_cell_bg.color = vec4(1.0, 1.0, 1.0, 0.06 * self.bell_frames as f32);
            self.draw_cell_bg.draw_abs(cx, self.rect);
            self.draw_bg.redraw(cx);
        }

        // Scrollback position indicator.
        if self.view_offset > 0 {
            let sb = session.terminal.screen().scrollback.len().max(1);
            let frac = self.view_offset as f64 / sb as f64;
            let h = (self.rect.size.y * 0.2).max(24.0);
            let track = self.rect.size.y - h;
            let y = self.rect.pos.y + track * (1.0 - frac);
            self.draw_cell_bg.new_draw_call(cx);
            self.draw_cell_bg.color = vec4(1.0, 1.0, 1.0, 0.25);
            self.draw_cell_bg.draw_abs(
                cx,
                Rect {
                    pos: dvec2(self.rect.pos.x + self.rect.size.x - 4.0, y),
                    size: dvec2(3.0, h),
                },
            );
        }

        self.draw_search_bar(cx, session.terminal.screen(), default_fg, default_bg);

        // Exited banner.
        if session.exited {
            self.draw_cell_bg.new_draw_call(cx);
            self.draw_cell_bg.color = vec4(0.0, 0.0, 0.0, 0.5);
            self.draw_cell_bg.draw_abs(cx, self.rect);
        }
    }

    fn pump_session(&mut self, cx: &mut Cx) {
        let mut actions: Vec<MpTermAction> = Vec::new();
        let mut needs_redraw = false;
        if let Some(session) = self.session.as_mut() {
            if session.drain() {
                needs_redraw = true;
            }
            // Inside wm, shell facts go to the compositor through the WM
            // API so it can title the tile and open new terminals in our
            // cwd (the Omarchy behavior).
            for event in session.take_events() {
                match event {
                    TermEvent::TitleChanged(title) => {
                        // Hosted in wm: the bar shows it (no-op standalone).
                        if !self.titled_by_host {
                            makepad_wm_api::set_title(cx, &title);
                        }
                        actions.push(MpTermAction::TitleChanged(title))
                    }
                    TermEvent::Bell => {
                        if self.settings.bell == BellStyle::Visual {
                            self.bell_frames = 6;
                        }
                        actions.push(MpTermAction::Bell);
                        needs_redraw = true;
                    }
                    TermEvent::ClipboardSet { text, .. } => {
                        cx.copy_to_clipboard(&text);
                    }
                    TermEvent::PwdChanged(url) => {
                        // file://host/path -> path
                        let path = url
                            .strip_prefix("file://")
                            .map(|rest| match rest.find('/') {
                                Some(idx) => rest[idx..].to_string(),
                                None => rest.to_string(),
                            })
                            .unwrap_or(url);
                        self.cwd = Some(PathBuf::from(&path));
                        // Hosted: new terminals open here (Omarchy's
                        // terminal-in-cwd); no-op standalone.
                        makepad_wm_api::set_cwd(cx, Path::new(&path));
                        actions.push(MpTermAction::PwdChanged(path));
                    }
                    TermEvent::Notification { .. } => {}
                }
            }
            if session.exited {
                actions.push(MpTermAction::Exited);
            }
        }
        for action in actions {
            cx.widget_action(self.uid, action);
        }
        self.arm_sync_timer(cx);
        if needs_redraw {
            self.draw_bg.redraw(cx);
            if self.link_hover.is_some() {
                self.update_link_hover(cx);
            }
        }
    }

    /// Keep a timer on the held synchronized frame's deadline. The drain
    /// that timer triggers shows the frame if its end never came.
    fn arm_sync_timer(&mut self, cx: &mut Cx) {
        let deadline = self.session.as_ref().and_then(|s| s.sync_deadline());
        if deadline == self.sync_timer_for {
            return;
        }
        cx.stop_timer(self.sync_timer);
        self.sync_timer = Timer::default();
        if let Some(deadline) = deadline {
            let wait = (deadline - Cx::monotonic_now()).max(0.0);
            // A hair past the deadline, so the drain it wakes finds it due.
            self.sync_timer = cx.start_timeout(wait + 0.005);
        }
        self.sync_timer_for = deadline;
    }
}

impl Widget for MpTerm {
    fn draw_walk(&mut self, cx: &mut Cx2d, _scope: &mut Scope, walk: Walk) -> DrawStep {
        cx.begin_turtle(walk, self.layout);
        self.rect = cx.turtle().rect();
        self.sync_settings(cx);
        self.apply_fonts(cx);
        self.refresh_metrics(cx);
        self.ensure_session(cx);

        let (cols, rows) = self.grid_size();
        if let Some(session) = self.session.as_mut() {
            session.resize(cols, rows);
            session.drain();
        }
        self.arm_sync_timer(cx);

        self.draw_terminal(cx);

        cx.end_turtle_with_area(&mut self.area);
        // A terminal OWNS the keyboard from its first frame — standalone
        // and hosted alike. Without this a hosted tile that was never
        // clicked silently dropped every forwarded key (the WM decides
        // which tile receives keys; inside the child this widget must
        // hold its own key focus for them to land).
        if !self.took_focus {
            self.took_focus = true;
            // Canvas hosts decide focus from actual input. Redrawing their
            // areas must still let the framework migrate an existing focus.
            if self.canvas_ime_anchor.is_none() {
                cx.set_key_focus(self.area);
            }
        }
        if self.session.is_some() && cx.has_key_focus(self.area) {
            let s = self
                .session
                .as_ref()
                .map(|s| {
                    let sc = s.terminal.screen();
                    (sc.cursor.x, sc.cursor.y)
                })
                .unwrap_or((0, 0));
            let ime = self.search_ui.caret.unwrap_or_else(|| {
                self.inner_padding()
                    + dvec2(s.0 as f64 * self.cell_w, (s.1 + 1) as f64 * self.cell_h)
            });
            if let Some((anchor, transform)) = self.canvas_ime_anchor {
                let screen = (self.rect.pos + ime) * transform.scale
                    + transform.translation;
                let cursor = screen - anchor.rect(cx).pos;
                cx.show_text_ime(anchor, cursor);
            } else {
                cx.show_text_ime(self.area, ime);
            }
        }
        DrawStep::done()
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event, _scope: &mut Scope) {
        self.search_next_frame(cx, event);
        if self.blink_timer.is_event(event).is_some() {
            self.blink_armed = false;
            self.draw_bg.redraw(cx);
        }
        if self.sync_timer.is_event(event).is_some() {
            // The held frame's deadline: the drain releases it.
            self.sync_timer_for = None;
            self.pump_session(cx);
        }
        if self.kitty_text_timer.is_event(event).is_some() {
            self.flush_unpaired_text(cx);
        }
        if self.input_notice_timer.is_event(event).is_some() {
            self.input_notice = None;
            self.input_notice_armed = false;
            self.draw_bg.redraw(cx);
        }
        if matches!(event, Event::KeyDown(_) | Event::TextInput(_)) {
            self.blink_epoch = None;
        }
        if matches!(event, Event::KeyDown(_) | Event::KeyUp(_)) {
            // The composed text an Option key produces arrives before its
            // key-up, so any later key event ends the swallow window.
            self.swallow_text_until = None;
        }
        if let Event::KeyDown(e) | Event::KeyUp(e) = event {
            self.link_modifier_changed(cx, e);
        }
        if matches!(event, Event::KeyDown(_)) {
            term_settings::poll();
            self.sync_settings(cx);
        }
        if matches!(event, Event::Drag(_) | Event::Drop(_)) {
            match event.drag_hits(cx, self.area) {
                DragHit::Drag(drag) => {
                    let accepts = self.session.as_ref().is_some_and(|session| !session.exited)
                        && drag.items.iter().any(|item| matches!(item, DragItem::FilePath {path,..} if Self::valid_drop_path(path)));
                    if let Ok(mut response) = drag.response.try_lock() {
                        *response = if accepts {
                            DragResponse::Copy
                        } else {
                            DragResponse::None
                        };
                    }
                }
                DragHit::Drop(drop) => {
                    if let Some(path) = drop.items.iter().find_map(|item| match item {
                        DragItem::FilePath { path, .. } if Self::valid_drop_path(path) => {
                            Some(path)
                        }
                        _ => None,
                    }) {
                        if self.ai_drop_file(Path::new(path)) {
                            cx.widget_action(
                                self.uid,
                                MpTermAction::FileDropped {
                                    path: PathBuf::from(path),
                                },
                            );
                            self.redraw(cx);
                        }
                    }
                }
                _ => {}
            }
        }
        if let Event::Signal = event {
            // No check_and_clear here: the event loop already consumed the
            // global flag to dispatch this event, so a second check would
            // only steal a signal raised since — including the one our own
            // budget-capped drain re-arms to continue a flood next tick.
            self.pump_session(cx);
        }

        // Edge auto-scroll while selecting.
        if self.selecting && self.select_scroll_frame.is_event(event).is_some() {
            self.select_scroll_frame = cx.new_next_frame();
            if let Some(abs) = self.last_finger {
                let band = self.cell_h.max(6.0);
                let top = self.rect.pos.y + band;
                let bottom = self.rect.pos.y + self.rect.size.y - band;
                let max = self
                    .session
                    .as_ref()
                    .map(|s| s.terminal.screen().scrollback.len())
                    .unwrap_or(0);
                if abs.y < top {
                    self.view_offset = (self.view_offset + 1).min(max);
                } else if abs.y > bottom {
                    self.view_offset = self.view_offset.saturating_sub(1);
                }
                self.extend_selection(abs);
                self.draw_bg.redraw(cx);
            }
        }

        // An in-process host (a module tile in the OctoSense shell) forwards
        // keys only to the tile it focused, and the click that opened this
        // terminal can leave no widget holding the keyboard. Claim it on the
        // first key then, and handle that key in this pass: a key-focus
        // change only takes effect after the current event.
        //
        // A host that routes keys itself (the tab widget, which knows which
        // pane is focused) sets `route_keys_here`: this terminal takes the
        // key whoever holds the keyboard, since a focus change the host made
        // is still pending until this event is over.
        let orphan_key = self.session.is_some() && (cx.key_focus() == Area::Empty || self.route_keys_here);
        let hit = match event {
            Event::KeyDown(e) if orphan_key => {
                cx.set_key_focus(self.area);
                Hit::KeyDown(e.clone())
            }
            Event::TextInput(e) if orphan_key => {
                cx.set_key_focus(self.area);
                Hit::TextInput(e.clone())
            }
            Event::KeyUp(e) if orphan_key => Hit::KeyUp(*e),
            _ => event.hits(cx, self.area),
        };
        // The search bar, while open, takes the keyboard: the program gets
        // no keys until it closes.
        if self.search_event(cx, &hit) {
            return;
        }
        match hit {
            Hit::FingerDown(e) => {
                cx.set_key_focus(self.area);
                if self.link_press_down(&e) {
                    return;
                }
                if e.device.is_touch() {
                    self.touches.retain(|(id, _)| *id != e.digit_id);
                    self.touches.push((e.digit_id, e.abs.y));
                    if self.touches.len() >= 2 {
                        // A second finger: the gesture is a scroll, whatever
                        // the first finger started.
                        self.scroll_accum = 0.0;
                        self.selecting = false;
                        self.press = Press::Touch;
                        return;
                    }
                }
                let info = PressInfo {
                    tap_count: e.tap_count,
                    touch: e.device.is_touch(),
                    mouse_reporting: self.mouse_tracking().0 != MouseTracking::None,
                    shift: e.modifiers.shift,
                };
                self.press = Press::down(info, (e.abs.x, e.abs.y));
                // Double click selects a word, triple click a line; holding
                // and dragging extends it. Local: the app never sees it.
                if self.press == Press::Unit {
                    self.begin_unit_selection(cx, e.abs, e.tap_count >= 3);
                    return;
                }
                // Any other press clears what was selected: a click only
                // focuses; a drag selects anew once it passes the threshold.
                if self.sel_anchor.is_some() {
                    self.sel_anchor = None;
                    self.sel_cursor = None;
                    self.draw_bg.redraw(cx);
                }
                self.sel_unit = None;
                if self.press.reports() {
                    self.report_mouse(
                        cx,
                        e.abs,
                        MouseEventKind::Press,
                        TermMouseButton::Left,
                        &e.modifiers,
                    );
                }
            }
            Hit::FingerMove(e) => {
                if let Some((_, origin)) = &self.link_press {
                    if past_threshold((origin.x, origin.y), (e.abs.x, e.abs.y)) {
                        self.link_press = None;
                    }
                    return;
                }
                if e.device.is_touch() {
                    let mut dy = None;
                    if let Some(t) = self.touches.iter_mut().find(|(id, _)| *id == e.digit_id) {
                        dy = Some(e.abs.y - t.1);
                        t.1 = e.abs.y;
                    }
                    if self.touches.len() >= 2 {
                        if let Some(dy) = dy {
                            self.touch_scroll(cx, e.abs, dy, &e.modifiers);
                        }
                        return;
                    }
                }
                match self.press.moved((e.abs.x, e.abs.y)) {
                    MoveAction::None => {}
                    MoveAction::Report => {
                        self.report_mouse(
                            cx,
                            e.abs,
                            MouseEventKind::Motion,
                            TermMouseButton::Left,
                            &e.modifiers,
                        );
                    }
                    MoveAction::StartChars { origin } => {
                        // The press became a drag: select characters from
                        // where it started, with edge auto-scroll.
                        self.sel_anchor = self.pick_boundary(dvec2(origin.0, origin.1));
                        self.sel_cursor = self.sel_anchor;
                        self.selecting = true;
                        self.select_scroll_frame = cx.new_next_frame();
                        self.extend_selection(e.abs);
                        self.last_finger = Some(e.abs);
                        self.draw_bg.redraw(cx);
                    }
                    MoveAction::Extend => {
                        self.extend_selection(e.abs);
                        self.last_finger = Some(e.abs);
                        self.draw_bg.redraw(cx);
                    }
                }
            }
            Hit::FingerUp(e) => {
                if e.device.is_touch() {
                    self.touches.retain(|(id, _)| *id != e.digit_id);
                }
                if let Some((hit, _)) = self.link_press.take() {
                    self.open_link(cx, hit);
                    return;
                }
                if self.press.reports() {
                    self.report_mouse(
                        cx,
                        e.abs,
                        MouseEventKind::Release,
                        TermMouseButton::Left,
                        &e.modifiers,
                    );
                }
                if self.selecting && self.settings.copy_on_select {
                    if let Some(text) = self.selected_text().filter(|t| !t.is_empty()) {
                        cx.copy_to_clipboard(&text);
                    }
                }
                if self.touches.is_empty() {
                    self.press = Press::Idle;
                }
                self.selecting = false;
                self.last_finger = None;
            }
            Hit::FingerHoverIn(e) | Hit::FingerHoverOver(e) => {
                let held = links::link_modifier(e.modifiers.logo, e.modifiers.control);
                self.link_pointer = Some((e.abs, held));
                self.update_link_hover(cx);
            }
            Hit::FingerHoverOut(_) => {
                self.link_pointer = None;
                self.update_link_hover(cx);
            }
            Hit::FingerScroll(e) => {
                self.handle_scroll(cx, &e);
            }
            Hit::KeyFocus(_) => {
                term_settings::reload_if_changed();
                self.sync_settings(cx);
                if let Some(session) = self.session.as_mut() {
                    if session.terminal.modes.get(Mode::FocusEvent) {
                        session.write(b"\x1b[I");
                    }
                }
                self.draw_bg.redraw(cx);
            }
            Hit::KeyFocusLost(_) => {
                self.kitty_pressed.clear();
                if let Some(session) = self.session.as_mut() {
                    if session.terminal.modes.get(Mode::FocusEvent) {
                        session.write(b"\x1b[O");
                    }
                }
                cx.hide_text_ime();
                self.draw_bg.redraw(cx);
            }
            Hit::KeyDown(e) => {
                if self.session.is_some()
                    && e.key_code == KeyCode::ReturnKey
                    && !e.is_repeat
                    && !e.modifiers.shift
                    && !e.modifiers.control
                    && !e.modifiers.logo
                    && !e.modifiers.alt
                {
                    cx.widget_action(self.uid, MpTermAction::PromptSubmitted);
                }
                let action = if e.is_repeat {
                    KeyAction::Repeat
                } else {
                    KeyAction::Press
                };
                // Kitty: only a press the program saw reports its release.
                if !e.modifiers.logo && !self.kitty_pressed.contains(&e.key_code) {
                    self.kitty_pressed.push(e.key_code);
                }
                // Kitty report-all: typed text becomes part of the key event.
                let report_all = kitty_input::text_as_key_events(self.key_opts().kitty_flags);
                let text_key = report_all
                    && !Self::is_special(e.key_code)
                    && !e.modifiers.logo
                    && !e.modifiers.control
                    && !(self.settings.option_as_meta && e.modifiers.alt)
                    && kitty_input::kitty_key_of(e.key_code).is_some();
                let held = if report_all {
                    self.kitty_pairing
                        .on_key_down(text_key, Cx::monotonic_now())
                } else {
                    None
                };
                if text_key {
                    self.send_kitty_text_key(cx, &e, action, held);
                } else if Self::is_special(e.key_code) {
                    // Clear selection on typing.
                    self.sel_anchor = None;
                    self.sel_cursor = None;
                    let key = Self::map_keycode(e.key_code).unwrap();
                    let action = if e.is_repeat {
                        KeyAction::Repeat
                    } else {
                        KeyAction::Press
                    };
                    self.send_key(cx, key, &e.modifiers, action, "", 0);
                } else if self.settings.option_as_meta
                    && e.modifiers.alt
                    && !e.modifiers.control
                    && !e.modifiers.logo
                {
                    // Option as Meta: the key goes out ESC-prefixed (or as a
                    // kitty Alt chord), not as the character macOS composes.
                    if let Some(ch) = e.key_code.to_char(e.modifiers.shift) {
                        let key = letter_key(ch).unwrap_or(Key::Unidentified);
                        let action = if e.is_repeat {
                            KeyAction::Repeat
                        } else {
                            KeyAction::Press
                        };
                        let text = ch.to_string();
                        self.send_key(cx, key, &e.modifiers, action, &text, ch.to_ascii_lowercase() as u32);
                        self.swallow_text_until = Some(Instant::now() + Duration::from_millis(100));
                    }
                } else if e.modifiers.control && !e.modifiers.logo {
                    if let Some(ch) = e.key_code.to_char(e.modifiers.shift) {
                        let key = letter_key(ch).unwrap_or(Key::Unidentified);
                        let action = if e.is_repeat {
                            KeyAction::Repeat
                        } else {
                            KeyAction::Press
                        };
                        self.send_key(
                            cx,
                            key,
                            &e.modifiers,
                            action,
                            "",
                            ch.to_ascii_lowercase() as u32,
                        );
                    }
                }
            }
            Hit::KeyUp(e) => {
                self.kitty_pairing.on_key_up();
                self.send_key_release(cx, &e);
            }
            Hit::TextInput(e) => {
                if e.replace_last {
                    return;
                }
                if let Some(until) = self.swallow_text_until.take() {
                    if !e.was_paste && Instant::now() <= until {
                        return;
                    }
                }
                if e.was_paste {
                    self.paste(cx, &e.input);
                } else {
                    let filtered: String = e
                        .input
                        .chars()
                        .filter(|c| *c != '\n' && *c != '\r')
                        .collect();
                    if !filtered.is_empty()
                        && kitty_input::text_as_key_events(self.key_opts().kitty_flags)
                    {
                        // Kitty report-all: wait for the key-down this text
                        // belongs to; text none claims is IME text.
                        if self.kitty_pairing.on_text(&filtered, Cx::monotonic_now())
                            == TextOutcome::Hold
                        {
                            self.kitty_text_timer = cx.start_timeout(kitty_input::PAIRING_WINDOW);
                        }
                    } else if !filtered.is_empty() {
                        self.sel_anchor = None;
                        self.sel_cursor = None;
                        self.scroll_to_bottom();
                        if let Some(session) = self.session.as_mut() {
                            session.write(filtered.as_bytes());
                        }
                        self.redraw(cx);
                    }
                }
            }
            Hit::TextCopy(e) => {
                if let Some(text) = self.selected_text() {
                    *e.response.borrow_mut() = Some(text);
                }
            }
            _ => {}
        }
    }
}

impl MpTerm {
    fn word_range(&self, pos: (u64, usize)) -> (usize, usize) {
        let Some(session) = self.session.as_ref() else {
            return (pos.1, pos.1 + 1);
        };
        let screen = session.terminal.screen();
        let Some(virt) = screen.virtual_of_absolute(pos.0) else {
            return (pos.1, pos.1 + 1);
        };
        let Some(row) = screen.row_virtual(virt) else {
            return (pos.1, pos.1 + 1);
        };
        // A wide char's or cluster's tails belong to its word.
        let kind_of = |col: usize| -> Option<bool> {
            let c = row
                .cell(row.head_of(col))
                .and_then(|c| c.content.primary())?;
            if c.is_whitespace() {
                None
            } else {
                Some(c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == '/')
            }
        };
        let Some(kind) = kind_of(pos.1) else {
            return (pos.1, pos.1 + 1);
        };
        let mut start = pos.1;
        while start > 0 && kind_of(start - 1) == Some(kind) {
            start -= 1;
        }
        let mut end = pos.1 + 1;
        while end < screen.cols && kind_of(end) == Some(kind) {
            end += 1;
        }
        (start, end)
    }

    fn redraw(&mut self, cx: &mut Cx) {
        self.draw_bg.redraw(cx);
    }
}

// ------------------------------------------------------------------
// Links (`crate::links`): modifier hover underlines, modifier click opens
// ------------------------------------------------------------------

impl MpTerm {
    /// The link at a window position, if the pointer is over one.
    fn link_under(&self, abs: Vec2d) -> Option<LinkHit> {
        let (row, col) = self.pick(abs)?;
        links::link_at(&self.session.as_ref()?.terminal, row, col)
    }

    /// Recompute the hovered link from the pointer and the modifier, and
    /// show a pointing hand over a link.
    fn update_link_hover(&mut self, cx: &mut Cx) {
        let hover = match self.link_pointer {
            Some((abs, true)) => self.link_under(abs),
            _ => None,
        };
        if self.link_pointer.is_some() {
            cx.set_cursor(if hover.is_some() {
                MouseCursor::Hand
            } else {
                MouseCursor::Text
            });
        }
        if hover != self.link_hover {
            self.link_hover = hover;
            self.draw_bg.redraw(cx);
        }
    }

    /// The link modifier went down or up while the pointer rests here.
    fn link_modifier_changed(&mut self, cx: &mut Cx, e: &KeyEvent) {
        if !matches!(e.key_code, KeyCode::Logo | KeyCode::Control) {
            return;
        }
        if let Some((abs, _)) = self.link_pointer {
            let held = links::link_modifier(e.modifiers.logo, e.modifiers.control);
            self.link_pointer = Some((abs, held));
            self.update_link_hover(cx);
        }
    }

    /// A press with the link modifier held on a link is ours, even when the
    /// program has mouse reporting on; it opens on release.
    fn link_press_down(&mut self, e: &FingerDownEvent) -> bool {
        self.link_press = None;
        if e.device.is_touch() || !links::link_modifier(e.modifiers.logo, e.modifiers.control) {
            return false;
        }
        let Some(hit) = self.link_under(e.abs) else {
            return false;
        };
        self.link_press = Some((hit, e.abs));
        self.press = Press::Idle;
        true
    }

    fn open_link(&mut self, cx: &mut Cx, hit: LinkHit) {
        let result = match &hit.target {
            Ok(target) => links::open(target)
                .map_err(|e| format!("Could not open {}: {e}", links::describe(target))),
            Err(refused) => Err(refused.to_string()),
        };
        if let Err(notice) = result {
            self.input_notice = Some(notice);
            self.input_notice_armed = false;
            self.draw_bg.redraw(cx);
        }
    }

    /// The hovered link's target along the bottom left, as browsers do.
    fn draw_link_status(&mut self, cx: &mut Cx2d) {
        // A notice (a refused link, say) takes the bottom of the view.
        let Some(label) = self
            .link_hover
            .as_ref()
            .map(LinkHit::label)
            .filter(|_| self.input_notice.is_none())
        else {
            return;
        };
        let pad = 4.0;
        let fit = ((self.rect.size.x - 4.0 * pad) / self.cell_w.max(1.0)).max(4.0) as usize;
        let label = if label.chars().count() > fit {
            let mut short: String = label.chars().take(fit - 1).collect();
            short.push('…');
            short
        } else {
            label
        };
        let width = label.chars().count() as f64 * self.cell_w + 2.0 * pad;
        let height = self.cell_h + 2.0 * pad;
        let pos = dvec2(
            self.rect.pos.x + pad,
            self.rect.pos.y + (self.rect.size.y - height - pad).max(0.0),
        );
        self.draw_cell_bg.new_draw_call(cx);
        self.draw_cell_bg.color = vec4(0.12, 0.13, 0.17, 0.94);
        self.draw_cell_bg.draw_abs(
            cx,
            Rect {
                pos,
                size: dvec2(width, height),
            },
        );
        self.draw_text.new_draw_call(cx);
        let color = self.draw_text.color;
        self.draw_text.color = vec4(0.86, 0.88, 0.95, 1.0);
        self.draw_text.draw_abs(cx, pos + dvec2(pad, pad), &label);
        self.draw_text.color = color;
    }
}

/// `fg` raised to `min` contrast against `bg`, its alpha (faint) kept.
fn contrast_fg(fg: Vec4f, bg: Vec4f, min: f32) -> Vec4f {
    let [r, g, b] = contrast::ensure([fg.x, fg.y, fg.z], [bg.x, bg.y, bg.z], min);
    vec4(r, g, b, fg.w)
}

fn parse_hex_rgb(s: &str) -> Option<Rgb> {
    let s = s.trim().trim_start_matches('#');
    if s.len() != 6 {
        return None;
    }
    Some(Rgb::new(
        u8::from_str_radix(&s[0..2], 16).ok()?,
        u8::from_str_radix(&s[2..4], 16).ok()?,
        u8::from_str_radix(&s[4..6], 16).ok()?,
    ))
}

fn letter_key(ch: char) -> Option<Key> {
    Some(match ch.to_ascii_lowercase() {
        'a' => Key::KeyA,
        'b' => Key::KeyB,
        'c' => Key::KeyC,
        'd' => Key::KeyD,
        'e' => Key::KeyE,
        'f' => Key::KeyF,
        'g' => Key::KeyG,
        'h' => Key::KeyH,
        'i' => Key::KeyI,
        'j' => Key::KeyJ,
        'k' => Key::KeyK,
        'l' => Key::KeyL,
        'm' => Key::KeyM,
        'n' => Key::KeyN,
        'o' => Key::KeyO,
        'p' => Key::KeyP,
        'q' => Key::KeyQ,
        'r' => Key::KeyR,
        's' => Key::KeyS,
        't' => Key::KeyT,
        'u' => Key::KeyU,
        'v' => Key::KeyV,
        'w' => Key::KeyW,
        'x' => Key::KeyX,
        'y' => Key::KeyY,
        'z' => Key::KeyZ,
        ' ' => Key::Space,
        '[' => Key::BracketLeft,
        ']' => Key::BracketRight,
        '\\' => Key::Backslash,
        '/' => Key::Slash,
        '-' => Key::Minus,
        '=' => Key::Equal,
        ';' => Key::Semicolon,
        '\'' => Key::Quote,
        ',' => Key::Comma,
        '.' => Key::Period,
        '`' => Key::Backquote,
        '0'..='9' => match ch {
            '0' => Key::Digit0,
            '1' => Key::Digit1,
            '2' => Key::Digit2,
            '3' => Key::Digit3,
            '4' => Key::Digit4,
            '5' => Key::Digit5,
            '6' => Key::Digit6,
            '7' => Key::Digit7,
            '8' => Key::Digit8,
            _ => Key::Digit9,
        },
        _ => return None,
    })
}

/// One scroll event's worth of terminal lines, and whether they go down.
/// `notched` is a classic wheel: every notch is at least one line. Precise
/// deltas (trackpad, Magic Mouse) add up in `accum` until they reach a line
/// of height `line`; a new gesture or a change of direction starts over.
/// None when nothing moves: zero-delta contact events, sideways swipes and
/// sub-line drift while a finger rests on the device.
fn scroll_step(
    accum: &mut f64,
    dy: f64,
    notched: bool,
    gesture_start: bool,
    line: f64,
) -> Option<(usize, bool)> {
    if notched {
        *accum = 0.0;
        if dy == 0.0 {
            return None;
        }
        return Some(((dy / 40.0).abs().ceil().max(1.0) as usize, dy > 0.0));
    }
    if gesture_start || dy * *accum < 0.0 {
        *accum = 0.0;
    }
    *accum += dy;
    let line = line.max(1.0);
    let lines = (accum.abs() / line).floor();
    if lines < 1.0 {
        return None;
    }
    let down = *accum > 0.0;
    *accum -= lines * line * accum.signum();
    Some((lines as usize, down))
}

/// Half of a cursor blink cycle, in seconds (on, then off).
const BLINK_HALF_PERIOD: f64 = 0.53;
/// How long a refused-input notice stays up.
const INPUT_NOTICE_SECONDS: f64 = 4.0;

/// The shape a cursor draws with: a program's DECSCUSR choice wins; with
/// none (`Default`) the person's settings decide shape and blink.
/// Whether cell (`abs_row`, `col`) lies in the ordered selection `sel`
/// (`end` exclusive).
fn in_selection(sel: Option<((u64, usize), (u64, usize))>, abs_row: u64, col: usize) -> bool {
    let Some(((sr, sc), (er, ec))) = sel else {
        return false;
    };
    if abs_row < sr || abs_row > er {
        return false;
    }
    if sr == er {
        return col >= sc && col < ec;
    }
    if abs_row == sr {
        return col >= sc;
    }
    if abs_row == er {
        return col < ec;
    }
    true
}

fn effective_cursor_style(style: CursorStyle, settings: &Settings) -> CursorStyle {
    if style != CursorStyle::Default {
        return style;
    }
    match (settings.cursor_shape, settings.cursor_blink) {
        (CursorShape::Block, true) => CursorStyle::BlinkingBlock,
        (CursorShape::Block, false) => CursorStyle::SteadyBlock,
        (CursorShape::Bar, true) => CursorStyle::BlinkingBar,
        (CursorShape::Bar, false) => CursorStyle::SteadyBar,
        (CursorShape::Underline, true) => CursorStyle::BlinkingUnderline,
        (CursorShape::Underline, false) => CursorStyle::SteadyUnderline,
    }
}
