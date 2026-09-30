//! What a cell draws, and the cache of prepared glyphs behind it.
//!
//! A cell holds either one codepoint (`Char`, `WideChar`) or a grapheme
//! cluster: a base plus combining marks, a ZWJ emoji sequence, a flag's
//! two regional indicators, an emoji plus a skin tone or a variation
//! selector. One codepoint takes the fast path: one glyph, cached by the
//! char. A cluster is shaped as ONE run by the text engine (so the font's
//! GSUB/GPOS can join and place its parts) and cached by the cluster.
//!
//! Only the layout arithmetic and the cache live here, so they can be
//! tested without a GPU; `widget.rs` does the shaping and drawing.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::term::page::CellContent;

/// The glyph source of one cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellGlyph<'a> {
    /// One codepoint: the fast path.
    Char(char),
    /// Several codepoints that shape as one run.
    Cluster(&'a [char]),
}

/// What `content` draws, or `None` for a cell with nothing to draw (empty,
/// a blank, the tail of a wide character).
pub fn cell_glyph(content: &CellContent) -> Option<CellGlyph<'_>> {
    match content {
        CellContent::Char(c) | CellContent::WideChar(c) => {
            (*c != ' ').then_some(CellGlyph::Char(*c))
        }
        CellContent::Cluster(cluster) => match cluster.cps.as_slice() {
            [] => None,
            [c] => (*c != ' ').then_some(CellGlyph::Char(*c)),
            cps => Some(CellGlyph::Cluster(cps)),
        },
        CellContent::Empty | CellContent::WideTail | CellContent::WideSpacerHead => None,
    }
}

/// A cluster's cache key: a hash of its codepoints plus the style and cell
/// count the glyph was fitted to. The codepoints are kept next to the
/// cached value and compared on lookup, so a hash collision reshapes
/// instead of drawing the wrong cluster, and a lookup allocates nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClusterKey {
    pub hash: u64,
    pub bold: bool,
    pub columns: u8,
}

impl ClusterKey {
    pub fn new(cps: &[char], bold: bool, columns: u8) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        cps.hash(&mut hasher);
        Self {
            hash: hasher.finish(),
            bold,
            columns,
        }
    }
}

/// At most this many clusters stay cached; past it the cluster half of the
/// cache starts over (a program printing endless distinct combining
/// sequences must not grow it without bound).
pub const MAX_CACHED_CLUSTERS: usize = 4096;

/// A cached cluster: its codepoints (to rule out a hash collision) and its
/// prepared run.
type ClusterEntry<G> = (Box<[char]>, Option<Vec<G>>);

/// Prepared glyphs by cell content. `G` is one positioned glyph; `None`
/// records that nothing drawable came back, so it is not reshaped per frame.
#[derive(Debug)]
pub struct GlyphCache<G> {
    chars: HashMap<(char, bool, u8), Option<G>>,
    clusters: HashMap<ClusterKey, ClusterEntry<G>>,
}

impl<G> Default for GlyphCache<G> {
    fn default() -> Self {
        Self {
            chars: HashMap::new(),
            clusters: HashMap::new(),
        }
    }
}

impl<G> GlyphCache<G> {
    pub fn clear(&mut self) {
        self.chars.clear();
        self.clusters.clear();
    }

    pub fn char(&self, ch: char, bold: bool, columns: u8) -> Option<&Option<G>> {
        self.chars.get(&(ch, bold, columns))
    }

    pub fn insert_char(&mut self, ch: char, bold: bool, columns: u8, glyph: Option<G>) {
        self.chars.insert((ch, bold, columns), glyph);
    }

    pub fn cluster(&self, cps: &[char], bold: bool, columns: u8) -> Option<&Option<Vec<G>>> {
        match self.clusters.get(&ClusterKey::new(cps, bold, columns)) {
            Some((stored, run)) if **stored == *cps => Some(run),
            _ => None,
        }
    }

    pub fn insert_cluster(&mut self, cps: &[char], bold: bool, columns: u8, run: Option<Vec<G>>) {
        if self.clusters.len() >= MAX_CACHED_CLUSTERS {
            self.clusters.clear();
        }
        self.clusters
            .insert(ClusterKey::new(cps, bold, columns), (cps.into(), run));
    }

    #[cfg(test)]
    fn cluster_count(&self) -> usize {
        self.clusters.len()
    }
}

/// How a shaped run is fitted into the cells the terminal gave it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunFit {
    /// Uniform scale for glyph size and pen positions (1 = as shaped).
    pub scale: f32,
    /// Extra x offset that centres a narrower run in its cells.
    pub x_shift: f32,
}

/// Fit a run `advance` wide into `available` (cells times cell width). A
/// run only shrinks when it overflows by more than 5% (normal monospace
/// glyphs are not shrunk for rounding differences in the grid advance);
/// with `center`, a run narrower than its cells is centred in them (an
/// emoji a fallback font draws narrow, a lone mark).
pub fn fit_run(advance: f32, available: f32, center: bool) -> RunFit {
    let scale = if advance > available * 1.05 && advance > 0.0 {
        available / advance
    } else {
        1.0
    };
    let x_shift = if center {
        ((available - advance * scale) * 0.5).max(0.0)
    } else {
        0.0
    };
    RunFit { scale, x_shift }
}

/// A private-use codepoint: Nerd Font and other icon-font glyphs.
pub fn is_private_use(ch: char) -> bool {
    matches!(ch as u32, 0xE000..=0xF8FF | 0xF0000..=0xFFFFD | 0x100000..=0x10FFFD)
}

/// How a glyph is fitted into its cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    /// Text: kept at the left, shrunk only when it overflows.
    Text,
    /// A cluster: shrunk to fit, and centred when narrower.
    Centered,
    /// An icon (a private-use codepoint): a glyph one cell wide stays text;
    /// a wider one (a Nerd Font icon) is shrunk into, and centred in, the
    /// cells it may use.
    Icon,
}

/// Fit a run `advance` wide into `columns` cells of `cell` width.
pub fn fit_glyphs(fit: Fit, advance: f32, cell: f32, columns: u8) -> RunFit {
    let available = cell * columns.max(1) as f32;
    match fit {
        Fit::Text => fit_run(advance, available, false),
        Fit::Centered => fit_run(advance, available, true),
        Fit::Icon if advance <= cell * 1.05 => fit_run(advance, cell, false),
        Fit::Icon => fit_run(advance, available, true),
    }
}

/// The cells an icon may draw across: two when it is one cell wide and the
/// cell after it is blank (as Nerd Fonts' wide icons are drawn in most
/// terminals), else its own. The grid is not changed: the icon is still one
/// cell for the cursor and for what the program printed.
pub fn icon_columns(columns: u8, next_is_blank: bool) -> u8 {
    if columns == 1 && next_is_blank {
        2
    } else {
        columns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::page::Cluster;

    fn cluster(cps: &[char], width: u8) -> CellContent {
        CellContent::Cluster(Box::new(Cluster {
            cps: cps.to_vec(),
            width,
        }))
    }

    #[test]
    fn single_codepoints_take_the_fast_path() {
        assert_eq!(
            cell_glyph(&CellContent::Char('a')),
            Some(CellGlyph::Char('a'))
        );
        assert_eq!(
            cell_glyph(&CellContent::WideChar('漢')),
            Some(CellGlyph::Char('漢'))
        );
        // A one-codepoint cluster (never built by the printer today) is
        // still a plain char.
        assert_eq!(cell_glyph(&cluster(&['x'], 1)), Some(CellGlyph::Char('x')));
    }

    #[test]
    fn nothing_to_draw() {
        assert_eq!(cell_glyph(&CellContent::Empty), None);
        assert_eq!(cell_glyph(&CellContent::Char(' ')), None);
        assert_eq!(cell_glyph(&CellContent::WideTail), None);
        assert_eq!(cell_glyph(&CellContent::WideSpacerHead), None);
        assert_eq!(cell_glyph(&cluster(&[], 1)), None);
    }

    #[test]
    fn clusters_take_the_shaped_path() {
        let family = ['👨', '\u{200D}', '👩', '\u{200D}', '👧'];
        assert_eq!(
            cell_glyph(&cluster(&family, 2)),
            Some(CellGlyph::Cluster(&family[..]))
        );
        let flag = ['\u{1F1EF}', '\u{1F1F5}'];
        assert_eq!(
            cell_glyph(&cluster(&flag, 2)),
            Some(CellGlyph::Cluster(&flag[..]))
        );
        // A mark on a space still draws: the mark is the content.
        let mark = [' ', '\u{0301}'];
        assert_eq!(
            cell_glyph(&cluster(&mark, 1)),
            Some(CellGlyph::Cluster(&mark[..]))
        );
    }

    #[test]
    fn cluster_key_covers_codepoints_style_and_width() {
        let a = ['e', '\u{0301}'];
        let b = ['e', '\u{0300}'];
        assert_eq!(ClusterKey::new(&a, false, 1), ClusterKey::new(&a, false, 1));
        assert_ne!(ClusterKey::new(&a, false, 1), ClusterKey::new(&b, false, 1));
        assert_ne!(ClusterKey::new(&a, false, 1), ClusterKey::new(&a, true, 1));
        assert_ne!(ClusterKey::new(&a, false, 1), ClusterKey::new(&a, false, 2));
        // Order matters: e + mark is not mark + e.
        assert_ne!(
            ClusterKey::new(&['e', '\u{0301}'], false, 1),
            ClusterKey::new(&['\u{0301}', 'e'], false, 1)
        );
    }

    #[test]
    fn cache_hits_and_misses() {
        let mut cache: GlyphCache<u32> = GlyphCache::default();
        assert!(cache.char('a', false, 1).is_none());
        cache.insert_char('a', false, 1, Some(7));
        assert_eq!(cache.char('a', false, 1), Some(&Some(7)));
        assert!(cache.char('a', true, 1).is_none());

        let flag = ['\u{1F1EF}', '\u{1F1F5}'];
        assert!(cache.cluster(&flag, false, 2).is_none());
        cache.insert_cluster(&flag, false, 2, Some(vec![1]));
        assert_eq!(cache.cluster(&flag, false, 2), Some(&Some(vec![1])));
        assert!(cache.cluster(&flag, false, 1).is_none());
        // A cluster the fonts cannot draw is remembered as such.
        cache.insert_cluster(&['\u{0301}', '\u{0302}'], false, 1, None);
        assert_eq!(
            cache.cluster(&['\u{0301}', '\u{0302}'], false, 1),
            Some(&None)
        );

        cache.clear();
        assert!(cache.char('a', false, 1).is_none());
        assert!(cache.cluster(&flag, false, 2).is_none());
    }

    #[test]
    fn a_hash_collision_is_a_miss() {
        let mut cache: GlyphCache<u32> = GlyphCache::default();
        let real = ['e', '\u{0301}'];
        let other = ['a', '\u{0301}'];
        // Plant `other` under `real`'s key, as a colliding hash would.
        cache.clusters.insert(
            ClusterKey::new(&real, false, 1),
            (other.into(), Some(vec![9])),
        );
        assert!(cache.cluster(&real, false, 1).is_none());
    }

    #[test]
    fn cluster_cache_is_bounded() {
        let mut cache: GlyphCache<u32> = GlyphCache::default();
        for i in 0..MAX_CACHED_CLUSTERS as u32 + 10 {
            let base = char::from_u32(0x4E00 + i).unwrap();
            cache.insert_cluster(&[base, '\u{0301}'], false, 2, Some(vec![i]));
        }
        assert!(cache.cluster_count() <= MAX_CACHED_CLUSTERS);
        // The newest one survives the reset.
        let last = char::from_u32(0x4E00 + MAX_CACHED_CLUSTERS as u32 + 9).unwrap();
        assert!(cache.cluster(&[last, '\u{0301}'], false, 2).is_some());
    }

    #[test]
    fn icons_fit_one_or_two_cells() {
        assert!(is_private_use('\u{F113}'));
        assert!(is_private_use('\u{F0A5F}'));
        assert!(!is_private_use('a'));
        assert!(!is_private_use('漢'));
        // A Nerd Font Mono icon is an em wide: 10 against a 6 cell.
        let one = fit_glyphs(Fit::Icon, 10.0, 6.0, 1);
        assert!((one.scale - 0.6).abs() < 1e-6 && one.x_shift.abs() < 1e-6);
        let two = fit_glyphs(Fit::Icon, 10.0, 6.0, 2);
        assert_eq!(
            two,
            RunFit {
                scale: 1.0,
                x_shift: 1.0
            },
            "whole, centred in two cells"
        );
        // A private-use glyph a text font draws one cell wide (Powerline's
        // branch symbol in JetBrains Mono) stays where text would be.
        assert_eq!(
            fit_glyphs(Fit::Icon, 6.0, 6.0, 2),
            RunFit {
                scale: 1.0,
                x_shift: 0.0
            }
        );
        // Text never moves right; clusters centre.
        assert_eq!(fit_glyphs(Fit::Text, 3.0, 6.0, 2).x_shift, 0.0);
        assert_eq!(fit_glyphs(Fit::Centered, 6.0, 6.0, 2).x_shift, 3.0);
        assert_eq!(icon_columns(1, true), 2);
        assert_eq!(icon_columns(1, false), 1);
        assert_eq!(icon_columns(2, true), 2);
    }

    #[test]
    fn runs_fit_their_cells() {
        // A monospace glyph a hair wider than the grid is not shrunk.
        assert_eq!(
            fit_run(10.3, 10.0, false),
            RunFit {
                scale: 1.0,
                x_shift: 0.0
            }
        );
        // An emoji 30% too wide for two cells shrinks to fit exactly.
        let fit = fit_run(26.0, 20.0, true);
        assert!((fit.scale - 20.0 / 26.0).abs() < 1e-6);
        assert!(fit.x_shift.abs() < 1e-6);
        // A narrow run in two cells is centred.
        assert_eq!(
            fit_run(10.0, 20.0, true),
            RunFit {
                scale: 1.0,
                x_shift: 5.0
            }
        );
        assert_eq!(fit_run(10.0, 20.0, false).x_shift, 0.0);
        // Nothing to fit.
        assert_eq!(fit_run(0.0, 20.0, false).scale, 1.0);
    }
}
