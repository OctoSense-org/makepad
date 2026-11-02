//! The paint program: layers over a part's frame (millimetres), indexed by a uniform grid so a
//! point reads only the layers that can reach it.

use makepad_csg_math::Vec3d;
use makepad_csg_sdf::{distance, node_bounds, quat_from_degrees, transform_into, SdfNode, SdfPlace};
use crate::facets::Facets;
use crate::flood::Patch;
use crate::strokes::Strokes;
use std::collections::HashMap;
use std::sync::Arc;

pub type P3 = [f64; 3];

/// What a layer writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channel {
    /// The colour a surface shows and prints in (slot 0: the part's own).
    Color,
    /// Supports never go here.
    SupportBlock,
    /// Supports always go here.
    SupportEnforce,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    /// Slot `slot` (1.. into the program's palette; for support channels 1 = on).
    Paint { channel: Channel, slot: u8 },
    /// Back to slot 0 on the channel.
    Erase { channel: Channel },
}

impl Op {
    pub fn channel(&self) -> Channel {
        match self {
            Op::Paint { channel, .. } | Op::Erase { channel } => *channel,
        }
    }
    fn slot(&self) -> u8 {
        match self {
            Op::Paint { slot, .. } => *slot,
            Op::Erase { .. } => 0,
        }
    }
}

/// Where a layer applies: inside (distance <= 0).
#[derive(Clone, Debug)]
pub enum Region {
    /// The whole part.
    All,
    /// A solid's own distance field, grown by `grow` (a feature fill takes its surface band).
    Field { node: SdfNode, grow: f64 },
    /// A brushed stroke: a chain of capsules of one radius.
    Stroke { points: Vec<P3>, radius: f64 },
    /// A smooth-region fill as written: from `seed` while neighbouring normals stay within
    /// `angle` (radians). It holds nothing until the part's surface resolves it into a
    /// [`Region::Patch`] ([`Program::resolve_floods`]).
    Flood { seed: P3, angle: f64 },
    /// A flood resolved on a part's surface.
    Patch(Arc<Patch>),
    /// Painted leaves of an imported mesh. Placements are outermost first, each scaling
    /// uniformly, rotating in degrees, then moving from its inner frame to its outer frame.
    Facets { facets: Arc<Facets>, state: u8, frames: Vec<(P3, P3, f64)> },
    /// A strokes file (`m.strokes("vase.strokes")`): its paint strokes, the last one holding a
    /// point deciding it. `slots` maps the file's palette to the program's (a script binds the
    /// file's log after it compiles: [`Program::with_strokes`]).
    Strokes { file: String, strokes: Arc<Strokes>, slots: Arc<Vec<u8>> },
}

impl Region {
    /// The signed distance to the region at `p` (negative inside; the whole part: inside).
    pub fn distance(&self, p: P3) -> f64 {
        match self {
            Region::All => -1e9,
            Region::Field { node, grow } => distance(node, Vec3d::new(p[0], p[1], p[2])) - grow,
            Region::Stroke { points, radius } => (0..points.len().max(1)).map(|s| capsule(p, points[s.min(points.len() - 1)], points[(s + 1).min(points.len() - 1)])).fold(f64::MAX, f64::min) - radius,
            Region::Flood { .. } => 1e9,
            Region::Patch(patch) => patch.distance(p),
            Region::Facets { facets, state, frames } => {
                let mut p = Vec3d::new(p[0], p[1], p[2]);
                let mut size = 1.0;
                for &(at, rotate, scale) in frames {
                    p = transform_into(at, rotate, scale, p);
                    size *= scale;
                }
                facets.distance(*state, [p.x, p.y, p.z]) * size
            }
            Region::Strokes { strokes, .. } => {
                let h = strokes.sample(p);
                if h.paint.is_some() { -h.dist } else { h.dist }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub region: Region,
    pub op: Op,
}

/// One grid entry: a layer, and for a stroke the capsule (segment start) in it.
#[derive(Clone, Copy, Debug)]
struct Entry {
    layer: u32,
    seg: u32,
}

/// Grid cell edge (mm). A region is listed in every cell its box, grown by a cell, touches, so an
/// empty cell has no boundary within a cell of any of its points. The strokes index uses the
/// same grid, and the paint bake's bricks nest in it (eight samples at 0.4, 0.2 or 0.1 mm).
pub const CELL: f64 = crate::strokes::BRICK;

#[derive(Clone, Debug, Default)]
pub struct Program {
    pub layers: Vec<Layer>,
    /// Layers listed everywhere (the whole part, regions without a box).
    global: Vec<u32>,
    /// Per layer, what its distance is divided by: a field region with a Splash kernel may be
    /// any function (noise), steeper than a distance; its measured Lipschitz bound keeps the
    /// bake's whole-brick test and the edge ramp true. 1 for every other region.
    steep: Vec<f64>,
    grid: HashMap<[i32; 3], Vec<Entry>>,
    bounds: Option<(P3, P3)>,
}

fn cell_of(p: P3) -> [i32; 3] {
    std::array::from_fn(|k| (p[k] / CELL).floor() as i32)
}

use crate::strokes::capsule;

/// What a point reads on one channel: its slot, and a lower bound of the distance to the nearest
/// boundary that changes it (mm; the views' edge).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub slot: u8,
    pub dist: f64,
}

impl Program {
    pub fn new(layers: Vec<Layer>) -> Program {
        let mut p = Program { steep: vec![1.0; layers.len()], layers, ..Default::default() };
        let grow = |lo: P3, hi: P3, b: &mut Option<(P3, P3)>| {
            *b = Some(match *b {
                None => (lo, hi),
                Some((a, c)) => (std::array::from_fn(|k| a[k].min(lo[k])), std::array::from_fn(|k| c[k].max(hi[k]))),
            });
        };
        let mut bounds = None;
        for (li, l) in p.layers.iter().enumerate() {
            let li = li as u32;
            match &l.region {
                // a strokes file: its own brick index answers every point
                Region::All | Region::Strokes { .. } => p.global.push(li),
                Region::Field { node, grow: g } => match node_bounds(node) {
                    Some((lo, hi)) => {
                        p.steep[li as usize] = makepad_csg_sdf::SdfShape { root: node.clone(), min: lo, max: hi, materials: Vec::new() }.lipschitz();
                        let r = g + CELL;
                        let (lo, hi): (P3, P3) = (std::array::from_fn(|k| lo[k] - r), std::array::from_fn(|k| hi[k] + r));
                        grow(lo, hi, &mut bounds);
                        insert(&mut p.grid, lo, hi, Entry { layer: li, seg: 0 });
                    }
                    None => p.global.push(li),
                },
                Region::Stroke { points, radius } => {
                    let r = radius + CELL;
                    for s in 0..points.len().max(1) {
                        let a = points[s.min(points.len() - 1)];
                        let b = points[(s + 1).min(points.len() - 1)];
                        let lo: P3 = std::array::from_fn(|k| a[k].min(b[k]) - r);
                        let hi: P3 = std::array::from_fn(|k| a[k].max(b[k]) + r);
                        grow(lo, hi, &mut bounds);
                        insert(&mut p.grid, lo, hi, Entry { layer: li, seg: s as u32 });
                        if points.len() < 2 {
                            break;
                        }
                    }
                }
                // unresolved: holds nowhere
                Region::Flood { .. } => {}
                Region::Patch(patch) => {
                    let (lo, hi) = patch.bounds();
                    let r = crate::flood::REACH_MM + CELL;
                    let (lo, hi): (P3, P3) = (std::array::from_fn(|k| lo[k] - r), std::array::from_fn(|k| hi[k] + r));
                    grow(lo, hi, &mut bounds);
                    insert(&mut p.grid, lo, hi, Entry { layer: li, seg: 0 });
                }
                Region::Facets { facets, state, frames } => {
                    let Some((lo, hi)) = facets.bounds(*state) else { continue };
                    let places: Vec<_> = frames.iter().rev().map(|&(at, rotate, scale)| SdfPlace { at, rotate: quat_from_degrees(rotate), scale }).collect();
                    let (mut min, mut max) = ([f64::MAX; 3], [f64::MIN; 3]);
                    for corner in 0..8 {
                        let mut q = std::array::from_fn(|k| if corner & (1 << k) == 0 { lo[k] } else { hi[k] });
                        for place in &places {
                            q = place.out_of(q);
                        }
                        for k in 0..3 {
                            min[k] = min[k].min(q[k]);
                            max[k] = max[k].max(q[k]);
                        }
                    }
                    let size: f64 = frames.iter().map(|f| f.2).product();
                    // The site's search reach is in asset millimetres; the grid is in model millimetres.
                    let r = crate::flood::REACH_MM * size + CELL;
                    let (lo, hi): (P3, P3) = (min.map(|v| v - r), max.map(|v| v + r));
                    grow(lo, hi, &mut bounds);
                    insert(&mut p.grid, lo, hi, Entry { layer: li, seg: 0 });
                }
            }
        }
        p.bounds = bounds;
        p
    }

    /// The box every boxed layer lies in (grown by a cell); `None` when only whole-part layers.
    pub fn bounds(&self) -> Option<(P3, P3)> {
        self.bounds
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Whether any layer writes `channel`.
    pub fn has(&self, channel: Channel) -> bool {
        self.layers.iter().any(|l| l.op.channel() == channel)
    }

    /// The distinct layers (and capsules) that can reach a point, last layer first.
    fn layer_distance(&self, li: u32, entries: &[Entry], p: P3) -> f64 {
        match &self.layers[li as usize].region {
            Region::All => -1e9,
            Region::Field { node, grow } => distance(node, Vec3d::new(p[0], p[1], p[2])) / self.steep[li as usize] - grow,
            Region::Stroke { points, radius } => {
                let mut d = f64::MAX;
                for e in entries {
                    let s = e.seg as usize;
                    d = d.min(capsule(p, points[s], points[(s + 1).min(points.len() - 1)]));
                }
                d - radius
            }
            Region::Flood { .. } => 1e9,
            Region::Patch(patch) => patch.distance(p),
            r @ Region::Facets { .. } => r.distance(p),
            Region::Strokes { .. } => 1e9,
        }
    }

    /// A steep field layer's distance to its boundary at `p` by its local slope (central
    /// differences a fraction of `near` apart), at least its Lipschitz bound's; `None` for any
    /// other layer (a distance already).
    fn field_slope_distance(&self, li: u32, p: P3, near: f64) -> Option<f64> {
        let steep = self.steep[li as usize];
        let Region::Field { node, grow } = &self.layers[li as usize].region else { return None };
        if steep <= 1.0 + 1e-9 {
            return None;
        }
        let f = |q: P3| distance(node, Vec3d::new(q[0], q[1], q[2]));
        let h = near * 0.1;
        let g: [f64; 3] = std::array::from_fn(|k| {
            let (mut a, mut b) = (p, p);
            a[k] += h;
            b[k] -= h;
            (f(a) - f(b)) / (2.0 * h)
        });
        let slope = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().clamp(1e-6, steep);
        Some((f(p) - grow * steep).abs() / slope)
    }

    /// Whether a layer is a flood still to resolve on the part's surface.
    pub fn has_floods(&self) -> bool {
        self.layers.iter().any(|l| matches!(l.region, Region::Flood { .. }))
    }

    /// The program with each flood resolved by `resolve` (seed, angle) into its patch on the
    /// part (`None`: the seed is not on this part, the layer holds nothing).
    pub fn resolve_floods(&self, resolve: &mut dyn FnMut(P3, f64) -> Option<Patch>) -> Program {
        Program::new(
            self.layers
                .iter()
                .map(|l| match l.region {
                    Region::Flood { seed, angle } => Layer { region: resolve(seed, angle).map_or(l.region.clone(), |p| Region::Patch(Arc::new(p))), op: l.op },
                    _ => l.clone(),
                })
                .collect(),
        )
    }

    /// Whether a layer reads a strokes file.
    pub fn has_strokes(&self) -> bool {
        self.layers.iter().any(|l| matches!(l.region, Region::Strokes { .. }))
    }

    /// The program with each strokes layer's log (and its palette map) from `bind` (file ->
    /// log and slots; `None`: kept as it is). The grid is shared: strokes are not in it.
    pub fn with_strokes(&self, bind: &mut dyn FnMut(&str) -> Option<(Arc<Strokes>, Arc<Vec<u8>>)>) -> Program {
        let mut p = self.clone();
        for l in &mut p.layers {
            if let Region::Strokes { file, strokes, slots } = &mut l.region {
                if let Some((s, m)) = bind(file) {
                    *strokes = s;
                    *slots = m;
                }
            }
        }
        p
    }

    /// The log a strokes layer reads, to add to in place (a stroke being drawn).
    pub fn strokes_mut(&mut self, file: &str) -> Option<&mut Arc<Strokes>> {
        self.layers.iter_mut().find_map(|l| match &mut l.region {
            Region::Strokes { file: f, strokes, .. } if f == file => Some(strokes),
            _ => None,
        })
    }

    /// The palette map of a strokes layer, to add to in place.
    pub fn strokes_slots_mut(&mut self, file: &str) -> Option<&mut Arc<Vec<u8>>> {
        self.layers.iter_mut().find_map(|l| match &mut l.region {
            Region::Strokes { file: f, slots, .. } if f == file => Some(slots),
            _ => None,
        })
    }

    /// The strokes files the layers read, with their logs.
    pub fn strokes_files(&self) -> impl Iterator<Item = (&str, &Arc<Strokes>)> {
        self.layers.iter().filter_map(|l| match &l.region {
            Region::Strokes { file, strokes, .. } => Some((file.as_str(), strokes)),
            _ => None,
        })
    }

    /// When `next` is this program with only its strokes logs moved (the same layers
    /// otherwise): the capsule boxes that changed. `None`: anything else changed.
    pub fn strokes_change(&self, next: &Program) -> Option<Vec<(P3, P3)>> {
        if self.layers.len() != next.layers.len() {
            return None;
        }
        let mut boxes = Vec::new();
        for (a, b) in self.layers.iter().zip(&next.layers) {
            match (&a.region, &b.region) {
                (Region::Strokes { file: fa, strokes: sa, slots: ma }, Region::Strokes { file: fb, strokes: sb, slots: mb }) => {
                    if fa != fb || a.op != b.op || !mb.starts_with(ma) {
                        return None;
                    }
                    boxes.extend(sa.changed_boxes(sb, true));
                }
                (Region::Patch(pa), Region::Patch(pb)) if Arc::ptr_eq(pa, pb) => {}
                (Region::Patch(_), _) | (_, Region::Patch(_)) => return None,
                _ => {
                    if a.op != b.op || format!("{:?}", a.region) != format!("{:?}", b.region) {
                        return None;
                    }
                }
            }
        }
        Some(boxes)
    }

    /// The program with a strokes layer on top reading `strokes` (a part's first stroke, before
    /// its script reads the file).
    pub fn with_strokes_on_top(&self, file: &str, strokes: Arc<Strokes>, slots: Arc<Vec<u8>>) -> Program {
        let mut p = self.clone();
        p.global.push(p.layers.len() as u32);
        p.steep.push(1.0);
        p.layers.push(Layer { region: Region::Strokes { file: file.to_string(), strokes, slots }, op: Op::Paint { channel: Channel::Color, slot: 0 } });
        p
    }

    /// The channel at `p`: the last layer holding it decides; the distance is to the nearest
    /// boundary at or above that layer (a lower layer's edge is covered).
    pub fn sample(&self, channel: Channel, p: P3) -> Sample {
        self.sample_near(channel, p, 0.0)
    }

    /// [`Program::sample`] with the distance to a steep field region's boundary measured, where
    /// it is under `near` (mm), by the field's own slope there instead of its steepest anywhere:
    /// the lower bound puts a noise region's edge several times too close, and a view that
    /// ramps its edge over that distance smears it into a wide blend of the two colours. The
    /// estimate (the value over the local gradient: exact for a plane) is no lower bound, so
    /// a test that must not miss a boundary keeps [`Program::sample`].
    pub fn sample_near(&self, channel: Channel, p: P3, near: f64) -> Sample {
        let cell: &[Entry] = self.grid.get(&cell_of(p)).map_or(&[], |v| v.as_slice());
        // what the cell's list leaves out is a cell past its faces
        let mut dist = crate::strokes::floor_of(p);
        // Walk layers from the last: the cell's entries (sorted by layer) and the global list.
        let (mut ci, mut gi) = (cell.len(), self.global.len());
        while ci > 0 || gi > 0 {
            let cl = if ci > 0 { Some(cell[ci - 1].layer) } else { None };
            let gl = if gi > 0 { Some(self.global[gi - 1]) } else { None };
            let li = match (cl, gl) {
                (Some(c), Some(g)) => c.max(g),
                (Some(c), None) => c,
                (None, Some(g)) => g,
                (None, None) => break,
            };
            let mut from = ci;
            while from > 0 && cell[from - 1].layer == li {
                from -= 1;
            }
            let entries = &cell[from..ci];
            ci = from;
            if gl == Some(li) {
                gi -= 1;
            }
            let layer = &self.layers[li as usize];
            if layer.op.channel() != channel {
                continue;
            }
            if let Region::Strokes { strokes, slots, .. } = &layer.region {
                let h = strokes.sample(p);
                dist = dist.min(h.dist);
                match h.paint {
                    Some(slot) => return Sample { slot: slot.and_then(|s| slots.get(s as usize).copied()).unwrap_or(0), dist },
                    None => continue,
                }
            }
            let d = self.layer_distance(li, entries, p);
            let at = if d.abs() < near { self.field_slope_distance(li, p, near).unwrap_or(d.abs()) } else { d.abs() };
            dist = dist.min(at);
            if d <= 0.0 {
                return Sample { slot: layer.op.slot(), dist };
            }
        }
        Sample { slot: 0, dist }
    }
}

fn insert(grid: &mut HashMap<[i32; 3], Vec<Entry>>, lo: P3, hi: P3, e: Entry) {
    let (a, b) = (cell_of(lo), cell_of(hi));
    for z in a[2]..=b[2] {
        for y in a[1]..=b[1] {
            for x in a[0]..=b[0] {
                let v = grid.entry([x, y, z]).or_default();
                // a stroke's capsules come in order: one entry per capsule
                v.push(e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paint(slot: u8) -> Op {
        Op::Paint { channel: Channel::Color, slot }
    }

    #[test]
    fn the_last_layer_wins_and_erase_exposes_the_base() {
        let p = Program::new(vec![
            Layer { region: Region::All, op: paint(1) },
            Layer { region: Region::Stroke { points: vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], radius: 2.0 }, op: paint(2) },
            Layer { region: Region::Field { node: SdfNode::Transform { at: [5.0, 0.0, 0.0], rotate: [0.0; 3], scale: 1.0, of: Box::new(SdfNode::Sphere { radius: 1.0 }) }, grow: 0.0 }, op: Op::Erase { channel: Channel::Color } },
        ]);
        assert_eq!(p.sample(Channel::Color, [50.0, 50.0, 50.0]).slot, 1);
        let s = p.sample(Channel::Color, [2.0, 1.0, 0.0]);
        assert_eq!(s.slot, 2);
        assert!((s.dist - 1.0).abs() < 1e-9, "{s:?}");
        assert_eq!(p.sample(Channel::Color, [5.0, 0.0, 0.0]).slot, 0);
        assert_eq!(p.sample(Channel::SupportBlock, [2.0, 0.0, 0.0]).slot, 0);
    }
}
