//! The GPU records handed between the slicer's stages (plane tracing, fields,
//! wall pieces, chains, rings), with host mirrors for tests and decode.
//!
//! Every record is four u32 words (Nearest and PieceWidth: two). Floats are stored as f32 bits, in grid
//! units (0.1 mm = 1.0) relative to the part revision's origin. Ids are per
//! plane: node g = gy * NX + gx, cell = the node id of its low corner, edge
//! = 2 g + axis (0 = +x, 1 = +y). Kernels and these mirrors change together.

/// Marks an endpoint key inside a cell (hi = CELL_BIT | cell) rather than on an edge.
pub const CELL_BIT: u32 = 1 << 31;
/// Marks a feature id as the start vertex of a seg rather than the seg itself.
pub const VERTEX_BIT: u32 = 1 << 31;
/// A vertex feature in piece_info.feature: VERTEX_BIT | sub << FEATURE_SUB_SHIFT | seg,
/// where sub names which part of the vertex's join made the piece.
pub const FEATURE_SUB_SHIFT: u32 = 29;
pub const FEATURE_SEG_MASK: u32 = (1 << FEATURE_SUB_SHIFT) - 1;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexSub { RoundArc = 0, Line1 = 1, Line2 = 2, SquareCut = 3 }
pub fn vertex_feature(seg: u32, sub: VertexSub) -> u32 {
    debug_assert!(seg <= FEATURE_SEG_MASK);
    VERTEX_BIT | (sub as u32) << FEATURE_SUB_SHIFT | seg
}
/// An absent slot, feature or link.
pub const NONE: u32 = u32::MAX;
/// The layer index of a batch's shared halo plane.
pub const HALO: u32 = u32::MAX - 1;

/// SegInfo.flags: the start vertex turns right (reflex for inside distances).
pub const SEG_REFLEX: u32 = 1;
/// SegInfo.flags: the start vertex is a crease.
pub const SEG_CREASE: u32 = 2;
/// SegInfo.flags bits 8..15: the start vertex's ordinal within its cell (saddle pair index).
pub const SEG_ORDINAL_SHIFT: u32 = 8;
pub const SEG_ORDINAL_MASK: u32 = 0xff;

/// Levels: 0 is the outline, 1 the base (outline minus elephant foot, only
/// when the foot is nonzero), then three per classic wall (below 32). Thin-wall
/// beads take 32..63 and regions start at 64, so one chain never mixes them.
pub const LEVEL_OUTLINE: u32 = 0;
pub const LEVEL_BASE: u32 = 1;
pub const LEVEL_THIN: u32 = 32;
pub const LEVEL_REGION: u32 = 64;
/// S1b concentric levels take 128..191 (the class regions keep 64..127); 192..255 are unassigned. S1b's own
/// layout (regions_concentric.rs), for rounds r = 1..=R with R < 20: loops L_r = 128 + r, component boundaries
/// B_r = 148 + r - 1, wide tests T_r = 168 + r, and the narrow test N = 188; one extract takes 3R + 2 slots.
pub const LEVEL_CONCENTRIC: u32 = 128;
pub const LEVEL_UNASSIGNED: u32 = 192;
/// Concentric loop `k` (0-based) of a narrow solid region: the level (k + 1/2) * s0 of its own field.
pub fn concentric_level(k: u32) -> u32 {
    debug_assert!(k < 20);
    LEVEL_CONCENTRIC + k
}
/// `LoopMeta.flags`: a hole, and an open chain (a thin line that ends; its root is the head).
pub const LOOP_HOLE: u32 = 1;
pub const LOOP_OPEN: u32 = 2;
/// `S3Label.flags` bit0: the span is supported (its midpoint lies in the supported region, as walls::Label.supported);
/// a fully supported span has degree 0.
pub const LABEL_SUPPORTED: u32 = 1;
/// Classic walls a plan may ask for: their levels stay below LEVEL_THIN.
pub const MAX_WALLS: u32 = 10;
/// Wall `k` (1-based) at its centre (-d_k), its edge (-e_k), or its thick-test margin (-(e_k + m)).
pub fn wall_level(k: u32, which: WallLevel) -> u32 {
    debug_assert!((1..=MAX_WALLS).contains(&k));
    2 + 3 * (k - 1) + which as u32
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WallLevel { Centre = 0, Edge = 1, Margin = 2 }

/// Thin-wall bead `j` (0-based); the fill boundary is `thin_level(cap + 1)`.
pub fn thin_level(j: u32) -> u32 {
    debug_assert!(LEVEL_THIN + j < LEVEL_REGION);
    LEVEL_THIN + j
}

/// The low word of an endpoint key: level and the ordinal along its edge or cell.
pub fn key_lo(level: u32, ordinal: u32) -> u32 {
    debug_assert!(level < 256 && ordinal < 1 << 24);
    level << 24 | ordinal
}

/// Fine-grid ids of one plane (§10.1). `nx` is the part's coarse width, so NX = nx * 10 + 1 nodes across.
/// A node is gy * NX + gx; a cell is the node id of its low corner (never a compact cell index); an edge is 2 * node + axis.
pub fn node_id(gx: u32, gy: u32, nx: u32) -> u32 { gy * (nx * 10 + 1) + gx }
pub fn cell_id(fx: u32, fy: u32, nx: u32) -> u32 { node_id(fx, fy, nx) }
pub fn edge_id(gx: u32, gy: u32, axis: u32, nx: u32) -> u32 { debug_assert!(axis < 2); 2 * node_id(gx, gy, nx) + axis }

/// Thin-level endpoint ordinals at a skeleton node: rib sectors take `k < 8`, edge ranks `8..16`.
pub const THIN_NODE_STRIDE: u32 = 16;
pub const THIN_RIB_SECTORS: u32 = 8;
/// Thin-level endpoint ordinals on a vertex normal ray beyond reach.
pub const THIN_RAY_BIT: u32 = 1 << 23;

pub fn thin_rib_ordinal(node: u32, sector: u32) -> u32 {
    debug_assert!(sector < THIN_RIB_SECTORS && node < THIN_RAY_BIT / THIN_NODE_STRIDE);
    node * THIN_NODE_STRIDE + sector
}

pub fn thin_edge_ordinal(node: u32, rank: u32) -> u32 {
    debug_assert!(rank < THIN_NODE_STRIDE - THIN_RIB_SECTORS && node < THIN_RAY_BIT / THIN_NODE_STRIDE);
    node * THIN_NODE_STRIDE + THIN_RIB_SECTORS + rank
}

/// `ray` is 0 or 1: the two normal rays at a vertex, in loop order.
pub fn thin_ray_ordinal(vertex: u32, ray: u32) -> u32 {
    debug_assert!(ray < 2 && vertex < THIN_RAY_BIT / 2);
    THIN_RAY_BIT | vertex * 2 + ray
}

fn f(w: u32) -> f32 { f32::from_bits(w) }

macro_rules! record {
    ($(#[$m:meta])* $name:ident { $a:ident: $ta:tt, $b:ident: $tb:tt, $c:ident: $tc:tt, $d:ident: $td:tt }) => {
        $(#[$m])*
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name { pub $a: $ta, pub $b: $tb, pub $c: $tc, pub $d: $td }
        impl $name {
            /// Decodes every record in `words` (a stage dump), four words each.
            pub fn from_words(words: &[u32]) -> Vec<Self> {
                words.chunks_exact(4).map(|w| Self { $a: record!(@get $ta, w[0]), $b: record!(@get $tb, w[1]), $c: record!(@get $tc, w[2]), $d: record!(@get $td, w[3]) }).collect()
            }
        }
    };
    (@get f32, $w:expr) => { f($w) };
    (@get u32, $w:expr) => { $w };
}

record!(/// Per plane: the sample z in printer mm (BELOW_PLANE already subtracted) and its layer index or HALO.
    PlaneRec { z_mm: f32, layer: u32, _0: u32, _1: u32 });
record!(/// A crossing on a fine edge: its position and unit outward gradient.
    Cross { x: f32, y: f32, nx: f32, ny: f32 });
record!(/// An outline (or region) segment, directed with the inside on the left.
    Seg { ax: f32, ay: f32, bx: f32, by: f32 });
record!(/// A seg's loop root (min seg index), its neighbours and SEG_* flags.
    SegInfo { root: u32, next: u32, prev: u32, flags: u32 });
record!(/// A FeatureSet loop: its root seg, seg count and plane.
    LoopInfo { root: u32, n: u32, plane: u32, _0: u32 });
record!(/// A piece of a level curve inside one fine cell: its endpoints.
    Piece { x0: f32, y0: f32, x1: f32, y1: f32 });
record!(/// The arc a piece follows (r = 0 for a line).
    PieceGeo { cx: f32, cy: f32, r: f32, sweep: f32 });
record!(/// A piece's endpoint keys: hi = edge key or CELL_BIT | cell; lo = key_lo(level, ordinal).
    PieceKey { from_hi: u32, from_lo: u32, to_hi: u32, to_lo: u32 });
record!(/// Where a piece comes from.
    PieceInfo { plane: u32, level: u32, feature: u32, cell: u32 });
record!(/// A chained loop: its root piece, piece and point counts, first output point.
    LoopRec { root: u32, n_pieces: u32, n_points: u32, first_point: u32 });
record!(/// A chained loop's plane, level, island and flags (bit0 = hole).
    LoopMeta { plane: u32, level: u32, island: u32, flags: u32 });
record!(/// A chained loop's signed area (grid units squared); the bbox follows in a second record.
    LoopArea { area: f32, _0: f32, _1: f32, _2: f32 });
record!(/// A chained loop's bbox in grid units.
    LoopBox { x0: f32, y0: f32, x1: f32, y1: f32 });
record!(/// S3: where one overhang span of a wall loop starts: ring point `vertex` (the loop's own point index, from 0) plus `t` in [0, 1) along its segment to the
    /// next point (the closing segment included). A span runs to the next label's start, the last to the loop's end. The decode
    /// turns (vertex, t) into mm with f64 lengths of the read-back points, so no f32 arc length is ever summed on the GPU.
    S3Label { vertex: u32, t: f32, degree: f32, flags: u32 });

/// One nearest-feature node: the feature id and the biased signed distance in the join metric.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nearest { pub feature: u32, pub dist: f32 }
impl Nearest {
    pub fn from_words(words: &[u32]) -> Vec<Self> {
        words.chunks_exact(2).map(|w| Self { feature: w[0], dist: f(w[1]) }).collect()
    }
}

/// A thin-wall piece's bead width at each end (two words per piece).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PieceWidth { pub w0: f32, pub w1: f32 }
impl PieceWidth {
    pub fn from_words(words: &[u32]) -> Vec<Self> {
        words.chunks_exact(2).map(|w| Self { w0: f(w[0]), w1: f(w[1]) }).collect()
    }
}

/// S3: a full-support cut (a speed-ramp seed) at ring point `vertex` (the loop's own point index, from 0) plus `t` along its segment.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct S3Seed { pub vertex: u32, pub t: f32 }
impl S3Seed {
    pub fn from_words(words: &[u32]) -> Vec<Self> {
        words.chunks_exact(2).map(|w| Self { vertex: w[0], t: f(w[1]) }).collect()
    }
}

/// A GPU-resident ring point in grid units (before the nm readback), read by S3.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RingPt { pub x: f32, pub y: f32 }
impl RingPt {
    pub fn from_words(words: &[u32]) -> Vec<Self> {
        words.chunks_exact(2).map(|w| Self { x: f(w[0]), y: f(w[1]) }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn records_are_four_words_and_decode() {
        assert_eq!(std::mem::size_of::<Seg>(), 16);
        assert_eq!(std::mem::size_of::<PieceKey>(), 16);
        assert_eq!(std::mem::size_of::<Nearest>(), 8);
        let seg = Seg::from_words(&[1.5f32.to_bits(), 2.0f32.to_bits(), 0, 3.0f32.to_bits()]);
        assert_eq!(seg, vec![Seg { ax: 1.5, ay: 2.0, bx: 0.0, by: 3.0 }]);
        assert_eq!(wall_level(1, WallLevel::Centre), 2);
        assert_eq!(wall_level(2, WallLevel::Margin), 7);
        assert!(wall_level(MAX_WALLS, WallLevel::Margin) < LEVEL_THIN);
        assert_eq!(thin_level(0), LEVEL_THIN);
        let v = vertex_feature(7, VertexSub::SquareCut);
        assert_eq!((v & VERTEX_BIT != 0, v >> FEATURE_SUB_SHIFT & 3, v & FEATURE_SEG_MASK), (true, 3, 7));
        assert_eq!(key_lo(3, 1), 3 << 24 | 1);
        assert_eq!((cell_id(2, 1, 3), edge_id(2, 1, 1, 3)), (33, 67));
        assert_eq!((concentric_level(0), std::mem::size_of::<S3Label>(), std::mem::size_of::<RingPt>()), (128, 16, 8));
        assert_eq!((thin_rib_ordinal(2, 7), thin_edge_ordinal(2, 0), thin_ray_ordinal(5, 1)), (39, 40, THIN_RAY_BIT | 11));
    }
}
