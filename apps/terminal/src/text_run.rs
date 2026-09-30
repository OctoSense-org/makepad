//! Text runs: shaping across cells.
//!
//! Neighbouring narrow text cells that share a style are shaped as ONE run,
//! so the font's GSUB/GPOS sees their context: programming ligatures
//! (`!=`, `=>`, `->`), Arabic joining forms, and Indic conjuncts that span
//! cells. The shaped glyphs are then put back on the grid: each glyph
//! belongs to the cell its cluster starts in and is drawn relative to that
//! cell's left edge, so the grid stays authoritative for the cursor, the
//! selection and every width. A glyph whose cluster covers several cells (a
//! ligature, a conjunct) spans them.
//!
//! Glyphs are placed in logical order, one cell after another, left to
//! right: there is no bidi reordering (as in Ghostty and Kitty), so a
//! right-to-left word shows its letters, correctly joined, in the order
//! they were typed.
//!
//! Only the arithmetic lives here (segmenting a row, mapping clusters to
//! cells, fitting glyphs, the run cache, parsing `font-features`), so it is
//! tested without a GPU; `widget.rs` shapes and draws.

use crate::cell_glyph::CellGlyph;
use crate::term::unicode::is_syllable_script;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::ops::Range;

/// What decides which font a run is shaped with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RunStyle {
    pub bold: bool,
    pub italic: bool,
}

/// How a cell takes part in run shaping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegCell {
    /// Nothing to draw (empty, a space, the tail of a wide character):
    /// ends a run. Ligatures, joining and conjuncts never cross a space.
    Blank,
    /// Drawn on its own and never shaped with its neighbours: a wide glyph
    /// (CJK, emoji), a box, block, Powerline or braille sprite, an icon.
    Own,
    /// Narrow text: joins the neighbours it shares a style and selection
    /// state with.
    Text { style: RunStyle, selected: bool },
}

/// Whether a cell's glyph is shaped with its neighbours: narrow text below
/// U+2500. Wide cells (CJK, emoji), box drawing, blocks, braille, Powerline
/// and other private-use icons (all at U+2500 and above) are drawn on
/// their own, as before: they have no neighbours to join, and their fitting
/// (sprites, icons spilling into a blank cell) is per cell.
pub fn shapes_with_neighbours(glyph: CellGlyph, columns: u8) -> bool {
    let base = match glyph {
        CellGlyph::Char(ch) => ch,
        CellGlyph::Cluster(cps) => match cps.first() {
            Some(&ch) => ch,
            None => return false,
        },
    };
    match glyph {
        // A syllable cluster (Devanagari..Sinhala, one cell per base
        // letter) is text however many cells it has: its run keeps its
        // tail cells, so its glyphs are drawn across them at full size.
        CellGlyph::Cluster(_) if columns > 1 => is_syllable_script(base as u32),
        _ => columns == 1 && base < '\u{2500}',
    }
}

/// Split a row into runs (ranges of cells). A run is a maximal stretch of
/// `Text` cells with one style and one selection state, so a selection edge
/// never moves a glyph from one cell to another: a ligature half selected
/// is shaped as two halves, each drawn in its own cell's colours. The
/// cursor's cells (`cursor`: one, or every cell of the wide cluster it is
/// on) are always a run of their own, which breaks a ligature under the
/// cursor so the cell it is on shows its own character.
pub fn segment_row(cells: &[SegCell], cursor: Option<Range<usize>>, out: &mut Vec<Range<usize>>) {
    out.clear();
    let mut start: Option<(usize, RunStyle, bool)> = None;
    for (col, cell) in cells.iter().enumerate() {
        let key = match *cell {
            SegCell::Text { style, selected } => Some((style, selected)),
            _ => None,
        };
        let continues = match (start, key) {
            (Some((_, style, selected)), Some(key)) => {
                key == (style, selected)
                    && !cursor
                        .as_ref()
                        .is_some_and(|c| col == c.start || col == c.end)
            }
            _ => false,
        };
        if !continues {
            if let Some((s, _, _)) = start.take() {
                out.push(s..col);
            }
            if let Some((style, selected)) = key {
                start = Some((col, style, selected));
            }
        }
    }
    if let Some((s, _, _)) = start {
        out.push(s..cells.len());
    }
}

/// The cell (index into `cell_starts`) holding byte `cluster` of the run's
/// text, where `cell_starts[i]` is the byte at which cell `i`'s text begins.
pub fn cell_of_cluster(cell_starts: &[usize], cluster: usize) -> usize {
    cell_starts
        .partition_point(|&start| start <= cluster)
        .saturating_sub(1)
}

/// One shaped glyph, as the shaper placed it (layout points, y down).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GlyphIn {
    /// Byte in the run's text where the glyph's cluster starts.
    pub cluster: usize,
    pub pen_x: f32,
    pub offset_x: f32,
    /// The shaper's vertical offset (a raised mark is negative).
    pub offset_y: f32,
    pub advance: f32,
    /// The glyph's ink, (top, bottom) relative to the baseline, y down;
    /// `None` when it has none.
    pub ink: Option<(f32, f32)>,
}

/// Where a glyph is drawn: relative to the run's left edge and the
/// baseline, at `scale` times its shaped size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    /// Index into the glyphs given to [`place_run`].
    pub glyph: usize,
    /// The cell (from the run's start) it belongs to: its colours.
    pub cell: usize,
    pub x: f32,
    pub y: f32,
    pub scale: f32,
}

/// Put a shaped run back on the grid. `cell_starts` maps cells to bytes of
/// the shaped text; `centered[i]` says cell `i` holds a grapheme cluster
/// (centred when its glyphs are narrower than the cell, as a lone cluster
/// always was).
///
/// The glyphs of each cell are drawn from that cell's left edge, keeping
/// their shaped positions relative to one another (so marks stay on their
/// base, and a conjunct keeps its parts together). A cell's glyphs may use
/// the cells after it that have no glyphs of their own: that is a ligature
/// or conjunct spanning cells. When they are wider than those cells by more
/// than 5% (a proportional fallback font) they are shrunk to fit, and kept
/// vertically centred on the line, exactly as a single character was.
pub fn place_run(
    cell_starts: &[usize],
    centered: &[bool],
    glyphs: &[GlyphIn],
    cell_w: f32,
    out: &mut Vec<Placed>,
) {
    out.clear();
    let cells = cell_starts.len();
    if cells == 0 || glyphs.is_empty() {
        return;
    }
    // Each glyph's cell, and each cell's first glyph in shaped order.
    let owner: Vec<usize> = glyphs
        .iter()
        .map(|g| cell_of_cluster(cell_starts, g.cluster))
        .collect();
    let mut has_glyphs = vec![false; cells];
    for &cell in &owner {
        has_glyphs[cell] = true;
    }
    let mut next_owner = cells;
    let mut span_end = vec![cells; cells];
    for cell in (0..cells).rev() {
        span_end[cell] = next_owner;
        if has_glyphs[cell] {
            next_owner = cell;
        }
    }
    let owner = &owner[..];
    for cell in 0..cells {
        if !has_glyphs[cell] {
            continue;
        }
        // This cell's glyphs, in shaped order (runs are words: small).
        let mine = move || (0..glyphs.len()).filter(move |&i| owner[i] == cell);
        let anchor = mine().map(|i| glyphs[i].pen_x).fold(f32::MAX, f32::min);
        let advance: f32 = mine().map(|i| glyphs[i].advance).sum();
        let columns = (span_end[cell] - cell) as f32;
        let available = cell_w * columns;
        let fit = crate::cell_glyph::fit_run(
            advance,
            available,
            centered.get(cell).copied().unwrap_or(false),
        );
        // The middle of the cell's ink, so shrunk glyphs stay centred on
        // the line instead of sinking to the baseline.
        let (mut top, mut bottom) = (f32::MAX, f32::MIN);
        for i in mine() {
            if let Some((t, b)) = glyphs[i].ink {
                top = top.min(t + glyphs[i].offset_y);
                bottom = bottom.max(b + glyphs[i].offset_y);
            }
        }
        let center_y = if top <= bottom {
            (top + bottom) * 0.5
        } else {
            0.0
        };
        // A cell's glyphs start inside it whatever the font says (some
        // proportional fonts report pen offsets far past one character).
        let first = mine()
            .map(|i| (glyphs[i].pen_x + glyphs[i].offset_x - anchor) * fit.scale)
            .fold(f32::MAX, f32::min);
        let x_origin = if first.abs() > available { first } else { 0.0 };
        let cell_x = cell as f32 * cell_w;
        for i in mine() {
            let g = &glyphs[i];
            out.push(Placed {
                glyph: i,
                cell,
                x: cell_x + (g.pen_x + g.offset_x - anchor) * fit.scale - x_origin + fit.x_shift,
                y: g.offset_y * fit.scale + center_y * (1.0 - fit.scale),
                scale: fit.scale,
            });
        }
    }
}

/// A run's cache key: a hash of its text, plus its style. The text and
/// cell boundaries are stored with the value and compared on lookup, so a
/// hash collision (or the same text split into other cells) reshapes
/// instead of drawing another run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RunKey {
    hash: u64,
    style: RunStyle,
}

impl Hash for RunKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash ^ ((self.style.bold as u64) << 1 | self.style.italic as u64));
    }
}

/// The map's hasher: a [`RunKey`] already is a (keyed SipHash) hash of the
/// run, so it is used as it is instead of being hashed a second time.
#[derive(Default)]
struct PassThrough(u64);

impl Hasher for PassThrough {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ b as u64;
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

/// At most this many runs stay cached; past it the cache starts over (a
/// program printing endless distinct words must not grow it without bound).
pub const MAX_CACHED_RUNS: usize = 8192;

struct RunEntry<G> {
    text: Box<str>,
    cell_starts: Box<[usize]>,
    glyphs: Vec<G>,
}

/// Placed glyphs by run. The font features are not part of the key: the
/// widget clears the cache when they change, as it does for a font change.
pub struct RunCache<G> {
    runs: HashMap<RunKey, RunEntry<G>, BuildHasherDefault<PassThrough>>,
    /// Keys the text hash, per process (text comes from any program).
    seed: RandomState,
}

impl<G> Default for RunCache<G> {
    fn default() -> Self {
        Self {
            runs: HashMap::default(),
            seed: RandomState::new(),
        }
    }
}

impl<G> RunCache<G> {
    pub fn clear(&mut self) {
        self.runs.clear();
    }

    pub fn len(&self) -> usize {
        self.runs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    fn key(&self, text: &str, style: RunStyle) -> RunKey {
        RunKey {
            hash: self.seed.hash_one(text),
            style,
        }
    }

    pub fn get(&self, text: &str, cell_starts: &[usize], style: RunStyle) -> Option<&[G]> {
        let key = self.key(text, style);
        match self.runs.get(&key) {
            Some(entry) if &*entry.text == text && &*entry.cell_starts == cell_starts => {
                Some(&entry.glyphs)
            }
            _ => None,
        }
    }

    pub fn insert(&mut self, text: &str, cell_starts: &[usize], style: RunStyle, glyphs: Vec<G>) {
        if self.runs.len() >= MAX_CACHED_RUNS {
            self.runs.clear();
        }
        self.runs.insert(
            self.key(text, style),
            RunEntry {
                text: text.into(),
                cell_starts: cell_starts.into(),
                glyphs,
            },
        );
    }
}

/// An OpenType feature setting: a tag (four bytes, big-endian) and a value.
pub type FontFeature = (u32, u32);

/// A feature tag from 1 to 4 printable ASCII characters, padded with
/// spaces as OpenType does (`kern`, `ss01`, `cv05`).
fn feature_tag(name: &str) -> Option<u32> {
    if name.is_empty() || name.len() > 4 || !name.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return None;
    }
    let mut bytes = [b' '; 4];
    bytes[..name.len()].copy_from_slice(name.as_bytes());
    Some(u32::from_be_bytes(bytes))
}

/// Parse `font-features`: a comma- or space-separated list whose entries
/// are `tag` or `+tag` (on), `-tag` (off), or `tag=N` (`on`/`off`/`true`/
/// `false` too), as in Ghostty's `font-feature` and CSS. An entry that does
/// not parse is skipped; a later entry for a tag replaces an earlier one.
pub fn parse_font_features(text: &str) -> Vec<FontFeature> {
    let mut out: Vec<FontFeature> = Vec::new();
    for entry in text.split(|c: char| c == ',' || c.is_whitespace()) {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let parsed = if let Some(name) = entry.strip_prefix('-') {
            feature_tag(name).map(|tag| (tag, 0))
        } else if let Some(name) = entry.strip_prefix('+') {
            feature_tag(name).map(|tag| (tag, 1))
        } else if let Some((name, value)) = entry.split_once('=') {
            let value = match value.trim() {
                "on" | "true" => Some(1),
                "off" | "false" => Some(0),
                v => v.parse::<u32>().ok(),
            };
            feature_tag(name.trim()).zip(value)
        } else {
            feature_tag(entry).map(|tag| (tag, 1))
        };
        if let Some((tag, value)) = parsed {
            out.retain(|&(t, _)| t != tag);
            out.push((tag, value));
        }
    }
    out
}

/// The features that make programming ligatures: contextual alternates
/// (JetBrains Mono, Fira Code, Cascadia Code), standard and discretionary
/// ligatures.
const LIGATURE_FEATURES: [&str; 3] = ["calt", "liga", "dlig"];

/// Ligatures are on unless `calt` or `liga` is turned off.
pub fn ligatures_on(features: &str) -> bool {
    let off = |name| feature_tag(name).map(|tag| (tag, 0));
    let parsed = parse_font_features(features);
    !parsed
        .iter()
        .any(|f| Some(*f) == off("calt") || Some(*f) == off("liga"))
}

/// `features` with ligatures turned on or off: the ligature features'
/// entries are dropped, and `-calt, -liga, -dlig` appended to turn them
/// off. Every other entry is kept as written.
pub fn with_ligatures(features: &str, on: bool) -> String {
    let lig_tags: Vec<u32> = LIGATURE_FEATURES
        .iter()
        .filter_map(|n| feature_tag(n))
        .collect();
    let mut kept: Vec<&str> = features
        .split(|c: char| c == ',' || c.is_whitespace())
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .filter(|entry| {
            let tags = parse_font_features(entry);
            !tags.iter().any(|(tag, _)| lig_tags.contains(tag))
        })
        .collect();
    let off = ["-calt", "-liga", "-dlig"];
    if !on {
        kept.extend(off);
    }
    kept.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: SegCell = SegCell::Text {
        style: RunStyle {
            bold: false,
            italic: false,
        },
        selected: false,
    };
    const BOLD: SegCell = SegCell::Text {
        style: RunStyle {
            bold: true,
            italic: false,
        },
        selected: false,
    };
    const ITALIC: SegCell = SegCell::Text {
        style: RunStyle {
            bold: false,
            italic: true,
        },
        selected: false,
    };
    const SELECTED: SegCell = SegCell::Text {
        style: RunStyle {
            bold: false,
            italic: false,
        },
        selected: true,
    };

    fn runs(cells: &[SegCell], cursor: Option<usize>) -> Vec<Range<usize>> {
        let mut out = Vec::new();
        segment_row(cells, cursor.map(|c| c..c + 1), &mut out);
        out
    }

    #[test]
    fn runs_split_at_blanks_styles_and_own_cells() {
        use SegCell::{Blank, Own};
        // `a != b` : blank cells end runs.
        assert_eq!(
            runs(&[TEXT, Blank, TEXT, TEXT, Blank, TEXT], None),
            vec![0..1, 2..4, 5..6]
        );
        // Bold, italic and a sprite or wide glyph each break the run.
        assert_eq!(
            runs(&[TEXT, TEXT, BOLD, BOLD, ITALIC, Own, TEXT], None),
            vec![0..2, 2..4, 4..5, 6..7]
        );
        assert_eq!(runs(&[], None), Vec::<Range<usize>>::new());
        assert_eq!(runs(&[Blank, Own, Blank], None), Vec::<Range<usize>>::new());
        // A run reaching the end of the row.
        assert_eq!(runs(&[Blank, TEXT, TEXT], None), vec![1..3]);
    }

    #[test]
    fn narrow_text_joins_runs_and_the_rest_draws_alone() {
        use CellGlyph::{Char, Cluster};
        assert!(shapes_with_neighbours(Char('='), 1));
        assert!(shapes_with_neighbours(Char('\u{0645}'), 1), "Arabic meem");
        assert!(shapes_with_neighbours(
            Cluster(&['\u{0915}', '\u{094D}', '\u{0937}']),
            1
        ));
        assert!(shapes_with_neighbours(Cluster(&['e', '\u{0301}']), 1));
        assert!(!shapes_with_neighbours(Char('漢'), 2), "wide");
        assert!(!shapes_with_neighbours(
            Cluster(&['\u{1F44D}', '\u{1F3FD}']),
            2
        ));
        assert!(!shapes_with_neighbours(Char('\u{2500}'), 1), "box drawing");
        assert!(!shapes_with_neighbours(Char('\u{2588}'), 1), "block");
        assert!(!shapes_with_neighbours(Char('\u{E0B0}'), 1), "Powerline");
        assert!(
            !shapes_with_neighbours(Char('\u{F113}'), 1),
            "Nerd Font icon"
        );
        assert!(!shapes_with_neighbours(Char('\u{2800}'), 1), "braille");
        assert!(!shapes_with_neighbours(Cluster(&[]), 1));
    }

    #[test]
    fn a_selection_edge_splits_the_run() {
        // `->=` with the middle selected: three runs, none moves a glyph.
        assert_eq!(runs(&[TEXT, SELECTED, TEXT], None), vec![0..1, 1..2, 2..3]);
        assert_eq!(runs(&[SELECTED, SELECTED, TEXT], None), vec![0..2, 2..3]);
    }

    #[test]
    fn the_cursor_breaks_a_ligature() {
        // `a!=b` with the cursor on `=`: the cursor cell stands alone.
        let row = [TEXT, TEXT, TEXT, TEXT];
        assert_eq!(runs(&row, Some(2)), vec![0..2, 2..3, 3..4]);
        assert_eq!(runs(&row, Some(0)), vec![0..1, 1..4]);
        assert_eq!(runs(&row, Some(3)), vec![0..3, 3..4]);
        // Off the text (past the end, on a blank) it changes nothing.
        assert_eq!(runs(&row, Some(9)), vec![0..4]);
        assert_eq!(
            runs(&[TEXT, SegCell::Blank, TEXT], Some(1)),
            vec![0..1, 2..3]
        );
    }

    #[test]
    fn clusters_map_to_the_cell_they_start_in() {
        // Cells `a`, `é` (e + U+0301, 3 bytes), `b`.
        let starts = [0, 1, 4];
        assert_eq!(cell_of_cluster(&starts, 0), 0);
        assert_eq!(cell_of_cluster(&starts, 1), 1);
        assert_eq!(cell_of_cluster(&starts, 2), 1, "inside a cell");
        assert_eq!(cell_of_cluster(&starts, 4), 1 + 1);
        assert_eq!(cell_of_cluster(&starts, 99), 2);
    }

    fn glyph(cluster: usize, pen_x: f32, advance: f32) -> GlyphIn {
        GlyphIn {
            cluster,
            pen_x,
            advance,
            ..GlyphIn::default()
        }
    }

    fn place(starts: &[usize], centered: &[bool], glyphs: &[GlyphIn], cell_w: f32) -> Vec<Placed> {
        let mut out = Vec::new();
        place_run(starts, centered, glyphs, cell_w, &mut out);
        out
    }

    #[test]
    fn a_monospace_ligature_keeps_one_glyph_per_cell() {
        // JetBrains Mono's `!=`: calt swaps in two cell-wide halves.
        let placed = place(
            &[0, 1],
            &[false; 2],
            &[glyph(0, 0.0, 6.0), glyph(1, 6.0, 6.0)],
            6.0,
        );
        assert_eq!(placed.len(), 2);
        assert_eq!(
            (placed[0].cell, placed[0].x, placed[0].scale),
            (0, 0.0, 1.0)
        );
        assert_eq!(
            (placed[1].cell, placed[1].x, placed[1].scale),
            (1, 6.0, 1.0)
        );
    }

    #[test]
    fn a_ligature_glyph_spans_the_cells_it_covers() {
        // `fi` (or lam-alef) as one glyph two cells wide, then `x`.
        let placed = place(
            &[0, 1, 2],
            &[false; 3],
            &[glyph(0, 0.0, 12.0), glyph(2, 12.0, 6.0)],
            6.0,
        );
        assert_eq!(placed.len(), 2);
        assert_eq!(placed[0].cell, 0);
        assert_eq!(placed[0].scale, 1.0, "two cells are its room");
        assert_eq!((placed[1].cell, placed[1].x), (2, 12.0));
    }

    #[test]
    fn glyphs_snap_to_their_cells_whatever_the_font_advance() {
        // A proportional fallback (Arabic) shaped right to left: the shaper
        // returns visual order, so cell 1's glyph comes first at pen 0.
        let placed = place(
            &[0, 2],
            &[false; 2],
            &[glyph(2, 0.0, 4.0), glyph(0, 4.0, 5.0)],
            6.0,
        );
        let by_cell = |c| placed.iter().find(|p| p.cell == c).unwrap().x;
        assert_eq!(by_cell(0), 0.0, "logical first at the left");
        assert_eq!(by_cell(1), 6.0);
        // A wide glyph shrinks into its one cell.
        let placed = place(&[0], &[false], &[glyph(0, 0.0, 9.0)], 6.0);
        assert!((placed[0].scale - 6.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn marks_keep_their_offsets_on_their_base() {
        // `ệ` shaped from e + U+0323 + U+0302: marks with zero advance,
        // offsets up and down, pen at the end of the base.
        let glyphs = [
            glyph(0, 0.0, 6.0),
            GlyphIn {
                cluster: 0,
                pen_x: 6.0,
                offset_x: -4.0,
                offset_y: 2.0,
                ..GlyphIn::default()
            },
            GlyphIn {
                cluster: 0,
                pen_x: 6.0,
                offset_x: -4.5,
                offset_y: -3.0,
                ..GlyphIn::default()
            },
        ];
        let placed = place(&[0], &[true], &glyphs, 6.0);
        assert_eq!(placed.len(), 3);
        assert_eq!((placed[1].x, placed[1].y), (2.0, 2.0));
        assert_eq!((placed[2].x, placed[2].y), (1.5, -3.0));
    }

    #[test]
    fn a_narrow_cluster_is_centred_and_text_is_not() {
        let placed = place(&[0], &[true], &[glyph(0, 0.0, 4.0)], 6.0);
        assert_eq!(placed[0].x, 1.0);
        let placed = place(&[0], &[false], &[glyph(0, 0.0, 4.0)], 6.0);
        assert_eq!(placed[0].x, 0.0);
    }

    #[test]
    fn a_shrunk_cell_stays_centred_on_the_line() {
        let g = GlyphIn {
            ink: Some((-10.0, 0.0)),
            ..glyph(0, 0.0, 12.0)
        };
        let placed = place(&[0], &[false], &[g], 6.0);
        assert_eq!(placed[0].scale, 0.5);
        // The ink's middle, -5, moves up by half: drawn at -2.5 + -2.5.
        assert_eq!(placed[0].y, -2.5);
    }

    #[test]
    fn the_run_cache_hits_by_text_cells_and_style() {
        let mut cache: RunCache<u32> = RunCache::default();
        let plain = RunStyle::default();
        let bold = RunStyle {
            bold: true,
            italic: false,
        };
        assert!(cache.get("!=", &[0, 1], plain).is_none());
        cache.insert("!=", &[0, 1], plain, vec![1, 2]);
        assert_eq!(cache.get("!=", &[0, 1], plain), Some(&[1, 2][..]));
        assert!(cache.get("!=", &[0, 1], bold).is_none());
        // The same text split into other cells is another run.
        assert!(cache.get("e\u{301}", &[0, 1], plain).is_none());
        cache.insert("e\u{301}", &[0], plain, vec![3]);
        assert!(cache.get("e\u{301}", &[0, 1], plain).is_none());
        assert_eq!(cache.get("e\u{301}", &[0], plain), Some(&[3][..]));
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn the_run_cache_is_bounded() {
        let mut cache: RunCache<u32> = RunCache::default();
        for i in 0..MAX_CACHED_RUNS + 10 {
            cache.insert(&i.to_string(), &[0], RunStyle::default(), vec![]);
        }
        assert!(cache.len() <= MAX_CACHED_RUNS);
        let last = (MAX_CACHED_RUNS + 9).to_string();
        assert!(cache.get(&last, &[0], RunStyle::default()).is_some());
    }

    /// Shape `text` with rustybuzz (the text engine's shaper) as the widget
    /// would, and place it on cells `cell_w` wide: glyph ids and placements.
    fn shape_and_place(
        face: &rustybuzz::Face,
        text: &str,
        cell_starts: &[usize],
        centered: &[bool],
        features: &[FontFeature],
        cell_w: f32,
    ) -> (Vec<u16>, Vec<GlyphIn>, Vec<Placed>) {
        let mut buffer = rustybuzz::UnicodeBuffer::new();
        buffer.push_str(text);
        let features: Vec<_> = features
            .iter()
            .map(|&(tag, value)| {
                rustybuzz::Feature::new(
                    rustybuzz::ttf_parser::Tag::from_bytes(&tag.to_be_bytes()),
                    value,
                    ..,
                )
            })
            .collect();
        let shaped = rustybuzz::shape(face, &features, buffer);
        let upem = face.units_per_em() as f32;
        let mut pen = 0.0;
        let mut ids = Vec::new();
        let mut glyphs = Vec::new();
        for (info, pos) in shaped.glyph_infos().iter().zip(shaped.glyph_positions()) {
            ids.push(info.glyph_id as u16);
            glyphs.push(GlyphIn {
                cluster: info.cluster as usize,
                pen_x: pen,
                offset_x: pos.x_offset as f32 / upem,
                offset_y: -pos.y_offset as f32 / upem,
                advance: pos.x_advance as f32 / upem,
                ink: None,
            });
            pen += pos.x_advance as f32 / upem;
        }
        let mut placed = Vec::new();
        place_run(cell_starts, centered, &glyphs, cell_w, &mut placed);
        (ids, glyphs, placed)
    }

    fn char_starts(text: &str) -> Vec<usize> {
        text.char_indices().map(|(i, _)| i).collect()
    }

    /// JetBrains Mono (the bundled font) draws `!=`, `->` and `===` through
    /// `calt`: one cell-wide glyph per character, each mapped back to its
    /// own cell. `-calt` turns them back into the plain characters.
    #[test]
    fn jetbrains_mono_ligatures_map_back_to_their_cells() {
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../widgets/resources/jetbrains_mono_variable.ttf"
        ))
        .unwrap();
        let face = rustybuzz::Face::from_slice(&data, 0).unwrap();
        let cell_w = 0.6;
        for text in ["!=", "->", "===", "<=", ">=", "=>"] {
            let starts = char_starts(text);
            let centered = vec![false; starts.len()];
            let (lig, _, placed) = shape_and_place(&face, text, &starts, &centered, &[], cell_w);
            let (plain, _, _) = shape_and_place(
                &face,
                text,
                &starts,
                &centered,
                &parse_font_features("-calt"),
                cell_w,
            );
            let alone: Vec<u16> = text
                .chars()
                .map(|c| {
                    let one = c.to_string();
                    shape_and_place(&face, &one, &[0], &[false], &[], cell_w).0[0]
                })
                .collect();
            assert_ne!(lig, alone, "{text}: calt made no ligature");
            assert_eq!(plain, alone, "{text}: -calt keeps the plain glyphs");
            // Every cell gets its own glyph, at its own left edge.
            let mut cells: Vec<usize> = placed.iter().map(|p| p.cell).collect();
            cells.sort();
            assert_eq!(cells, (0..starts.len()).collect::<Vec<_>>(), "{text}");
            for p in &placed {
                assert!((p.x - p.cell as f32 * cell_w).abs() < 0.05, "{text}: {p:?}");
                assert_eq!(p.scale, 1.0);
            }
        }
    }

    /// The cells a line printed into a terminal makes: what segmentation
    /// sees, each cell's text, and where the cursor ended.
    fn printed(text: &str, mode_2027: bool) -> (Vec<SegCell>, Vec<String>, Vec<u8>, usize) {
        use crate::cell_glyph::cell_glyph;
        use crate::term::page::CellContent;
        use crate::term::stream::Stream;
        use crate::term::terminal::Terminal;
        let (mut stream, mut term) = (Stream::new(), Terminal::new(60, 2));
        let mode: &[u8] = if mode_2027 {
            b"\x1b[?2027h"
        } else {
            b"\x1b[?2027l"
        };
        stream.process(mode, &mut term);
        stream.process(text.as_bytes(), &mut term);
        let row = term.screen().row(0);
        let (mut seg, mut texts, mut widths) = (Vec::new(), Vec::new(), Vec::new());
        let empty = CellContent::Empty;
        // The tails of a syllable cluster stay in its run, as in widget.rs.
        let mut tails_until = 0;
        for col in 0..term.cols() {
            let content = row.cell(col).map_or(&empty, |cell| &cell.content);
            let cell_text = match content {
                CellContent::Char(c) | CellContent::WideChar(c) => c.to_string(),
                CellContent::Cluster(c) => c.cps.iter().collect(),
                _ => String::new(),
            };
            seg.push(match cell_glyph(content) {
                None if col < tails_until && *content == CellContent::WideTail => SegCell::Text {
                    style: RunStyle::default(),
                    selected: false,
                },
                None => SegCell::Blank,
                Some(g) if shapes_with_neighbours(g, content.width()) => {
                    tails_until = col + content.width() as usize;
                    SegCell::Text {
                        style: RunStyle::default(),
                        selected: false,
                    }
                }
                Some(_) => SegCell::Own,
            });
            texts.push(cell_text);
            widths.push(content.width());
        }
        (seg, texts, widths, term.screen().cursor.x)
    }

    /// Arabic and Devanagari keep the widths the grid gives them: the runs
    /// cover exactly the printed cells, in logical order, and the cursor
    /// ends where the cells do. With mode 2027 a Devanagari syllable takes
    /// one cell per base letter and its run keeps its tail cells.
    #[test]
    fn arabic_and_devanagari_runs_match_the_grid() {
        for (text, mode_2027, cells) in [
            ("مرحبا بالعالم", true, 13),
            ("مرحبا بالعالم", false, 13),
            // Mode 2027: a conjunct and its vowel signs are one cluster,
            // one cell per base letter: क्ष 2, त्रि 1, य 1, स्त्री 2,
            // न 1, म 1, स्ते 2, and the two spaces.
            ("क्षत्रिय स्त्री नमस्ते", true, 12),
            // Without it: per-codepoint wcwidth, spacing vowel signs take
            // a column, a virama none, so conjuncts span cells.
            ("क्षत्रिय स्त्री नमस्ते", false, 16),
        ] {
            let (seg, texts, widths, cursor) = printed(text, mode_2027);
            assert_eq!(cursor, cells, "{text} 2027={mode_2027}");
            assert_eq!(
                widths[..cells].iter().map(|&w| w as usize).sum::<usize>(),
                cells
            );
            let mut out = Vec::new();
            segment_row(&seg, None, &mut out);
            let words: Vec<String> = out.iter().map(|run| texts[run.clone()].concat()).collect();
            let expected: Vec<&str> = text.split(' ').collect();
            assert_eq!(words, expected, "{text} 2027={mode_2027}");
            // Every column of a run is a narrow cell, a cluster's head or
            // one of its tails: the widths add up to the run's length.
            for run in &out {
                let total: usize = widths[run.clone()].iter().map(|&w| w as usize).sum();
                assert_eq!(total, run.len(), "{text} 2027={mode_2027}");
            }
        }
    }

    /// A proportional fallback (the platform's Arabic or Devanagari font)
    /// shaped across cells draws inside the cells the grid gave the run:
    /// no glyph starts left of its run or ends past it. Skipped where the
    /// font is not installed.
    #[test]
    fn complex_scripts_draw_inside_their_cells() {
        let fonts = [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        ];
        let Some(data) = fonts.iter().find_map(|path| std::fs::read(path).ok()) else {
            eprintln!("no Arabic/Devanagari font installed: skipped");
            return;
        };
        let Some(face) = rustybuzz::Face::from_slice(&data, 0) else {
            return;
        };
        let cell_w = 0.6;
        for (text, mode_2027) in [
            ("مرحبا بالعالم", false),
            ("क्षत्रिय स्त्री नमस्ते धर्म", false),
            ("क्षत्रिय स्त्री नमस्ते धर्म", true),
        ] {
            let (seg, texts, _, _) = printed(text, mode_2027);
            let mut runs = Vec::new();
            segment_row(&seg, None, &mut runs);
            for run in runs {
                let cell_texts = &texts[run.clone()];
                let mut starts = Vec::new();
                let mut run_text = String::new();
                for t in cell_texts {
                    starts.push(run_text.len());
                    run_text.push_str(t);
                }
                let centered: Vec<bool> =
                    cell_texts.iter().map(|t| t.chars().count() > 1).collect();
                let (ids, glyphs, placed) =
                    shape_and_place(&face, &run_text, &starts, &centered, &[], cell_w);
                assert!(ids.iter().all(|&id| id != 0), "{run_text}: missing glyphs");
                let width = starts.len() as f32 * cell_w;
                for p in &placed {
                    let g = &glyphs[p.glyph];
                    // Marks have no advance and sit over their base.
                    if g.advance == 0.0 {
                        continue;
                    }
                    let left = p.x;
                    let right = p.x + g.advance * p.scale;
                    assert!(
                        left >= -0.01 && right <= width + 0.01,
                        "{run_text}: glyph {} at {left}..{right} of {width}",
                        ids[p.glyph]
                    );
                }
                // Each glyph is in the cell its cluster starts in.
                for p in &placed {
                    assert_eq!(p.cell, cell_of_cluster(&starts, glyphs[p.glyph].cluster));
                }
                // A syllable given a cell per base letter is drawn across
                // its cells at full size, not squeezed.
                for p in &placed {
                    let spans = cell_texts.get(p.cell + 1).is_some_and(|t| t.is_empty());
                    if spans {
                        assert_eq!(p.scale, 1.0, "{run_text}: cell {} shrunk", p.cell);
                    }
                }
            }
        }
    }

    fn tag(s: &str) -> u32 {
        feature_tag(s).unwrap()
    }

    #[test]
    fn font_features_parse() {
        assert_eq!(parse_font_features(""), vec![]);
        assert_eq!(
            parse_font_features("-calt, -liga"),
            vec![(tag("calt"), 0), (tag("liga"), 0)]
        );
        assert_eq!(
            parse_font_features("ss01 +zero cv05=3  kern=off,dlig=on"),
            vec![
                (tag("ss01"), 1),
                (tag("zero"), 1),
                (tag("cv05"), 3),
                (tag("kern"), 0),
                (tag("dlig"), 1)
            ]
        );
        // Short tags are padded; a later entry wins.
        assert_eq!(tag("cv"), u32::from_be_bytes(*b"cv  "));
        assert_eq!(parse_font_features("-calt, calt"), vec![(tag("calt"), 1)]);
        // Junk is skipped, the rest kept.
        assert_eq!(
            parse_font_features("-toolong, -, =1, ca$t, liga=x, -liga"),
            vec![(tag("liga"), 0)]
        );
    }

    #[test]
    fn the_ligature_toggle_edits_only_ligature_features() {
        assert!(ligatures_on(""));
        assert!(ligatures_on("ss01"));
        assert!(!ligatures_on("-calt"));
        assert!(!ligatures_on("liga=0"));
        assert!(ligatures_on("-dlig"));
        let off = with_ligatures("ss01, cv05=2", false);
        assert_eq!(off, "ss01, cv05=2, -calt, -liga, -dlig");
        assert!(!ligatures_on(&off));
        let on = with_ligatures(&off, true);
        assert_eq!(on, "ss01, cv05=2");
        assert!(ligatures_on(&on));
        assert_eq!(with_ligatures("", false), "-calt, -liga, -dlig");
        assert_eq!(with_ligatures("-calt -liga", true), "");
    }
}
