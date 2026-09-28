//! The split layout of one tab: a binary tree whose leaves are panes (each
//! a terminal), laid out by splitting a rectangle side by side or stacked.
//! Pure geometry; `crate::tabs` owns the terminals.

use makepad_widgets::{dvec2, Rect};

/// A gap between panes, in pixels; the divider is drawn in it.
pub const GAP: f64 = 3.0;
/// No pane gets smaller than this share of its split.
const MIN_RATIO: f64 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Left,
    Right,
    Up,
    Down,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Leaf(u64),
    Split {
        /// Side by side (a divider running top to bottom), else stacked.
        side_by_side: bool,
        /// The first child's share.
        ratio: f64,
        first: Box<Node>,
        second: Box<Node>,
    },
}

/// A divider: where it is, and the split it resizes (by path from the root:
/// `false` = first child, `true` = second).
#[derive(Clone, Debug, PartialEq)]
pub struct Divider {
    pub rect: Rect,
    pub path: Vec<bool>,
    pub side_by_side: bool,
    /// The rectangle the split divides.
    pub area: Rect,
}

impl Node {
    pub fn leaves(&self) -> Vec<u64> {
        match self {
            Node::Leaf(id) => vec![*id],
            Node::Split { first, second, .. } => {
                let mut out = first.leaves();
                out.extend(second.leaves());
                out
            }
        }
    }

    pub fn contains(&self, id: u64) -> bool {
        self.leaves().contains(&id)
    }

    /// Split pane `at`: it keeps the first half, `new` takes the second.
    pub fn split(&mut self, at: u64, new: u64, side_by_side: bool) -> bool {
        match self {
            Node::Leaf(id) if *id == at => {
                *self = Node::Split {
                    side_by_side,
                    ratio: 0.5,
                    first: Box::new(Node::Leaf(at)),
                    second: Box::new(Node::Leaf(new)),
                };
                true
            }
            Node::Leaf(_) => false,
            Node::Split { first, second, .. } => first.split(at, new, side_by_side) || second.split(at, new, side_by_side),
        }
    }

    /// Remove pane `id`; its sibling takes the split's place. `None` when
    /// it was the only pane.
    pub fn remove(self, id: u64) -> Option<Node> {
        match self {
            Node::Leaf(leaf) if leaf == id => None,
            Node::Leaf(_) => Some(self),
            Node::Split { side_by_side, ratio, first, second } => {
                if matches!(*first, Node::Leaf(leaf) if leaf == id) {
                    return Some(*second);
                }
                if matches!(*second, Node::Leaf(leaf) if leaf == id) {
                    return Some(*first);
                }
                Some(Node::Split {
                    side_by_side,
                    ratio,
                    first: Box::new(first.remove(id)?),
                    second: Box::new(second.remove(id)?),
                })
            }
        }
    }

    /// Each pane's rectangle within `rect`.
    pub fn layout(&self, rect: Rect) -> Vec<(u64, Rect)> {
        let mut out = Vec::new();
        self.layout_into(rect, &mut out, &mut Vec::new(), &mut Vec::new());
        out
    }

    /// The dividers within `rect`.
    pub fn dividers(&self, rect: Rect) -> Vec<Divider> {
        let mut out = Vec::new();
        self.layout_into(rect, &mut Vec::new(), &mut out, &mut Vec::new());
        out
    }

    fn layout_into(&self, rect: Rect, panes: &mut Vec<(u64, Rect)>, dividers: &mut Vec<Divider>, path: &mut Vec<bool>) {
        match self {
            Node::Leaf(id) => panes.push((*id, rect)),
            Node::Split { side_by_side, ratio, first, second } => {
                let (a, d, b) = split_rect(rect, *side_by_side, *ratio);
                dividers.push(Divider { rect: d, path: path.clone(), side_by_side: *side_by_side, area: rect });
                path.push(false);
                first.layout_into(a, panes, dividers, path);
                path.pop();
                path.push(true);
                second.layout_into(b, panes, dividers, path);
                path.pop();
            }
        }
    }

    /// Set the ratio of the split at `path`.
    pub fn set_ratio(&mut self, path: &[bool], value: f64) {
        match (self, path.split_first()) {
            (Node::Split { ratio, .. }, None) => *ratio = value.clamp(MIN_RATIO, 1.0 - MIN_RATIO),
            (Node::Split { first, second, .. }, Some((&go_second, rest))) => {
                if go_second { second.set_ratio(rest, value) } else { first.set_ratio(rest, value) }
            }
            _ => {}
        }
    }

    /// Make every split even (after a pane opens or closes, or on request).
    pub fn equalize(&mut self) {
        if let Node::Split { ratio, first, second, .. } = self {
            *ratio = 0.5;
            first.equalize();
            second.equalize();
        }
    }
}

/// `rect` cut at `ratio` into (first, divider, second).
fn split_rect(rect: Rect, side_by_side: bool, ratio: f64) -> (Rect, Rect, Rect) {
    let Rect { pos, size } = rect;
    if side_by_side {
        let w = ((size.x - GAP) * ratio).round().max(0.0);
        (
            Rect { pos, size: dvec2(w, size.y) },
            Rect { pos: dvec2(pos.x + w, pos.y), size: dvec2(GAP, size.y) },
            Rect { pos: dvec2(pos.x + w + GAP, pos.y), size: dvec2((size.x - w - GAP).max(0.0), size.y) },
        )
    } else {
        let h = ((size.y - GAP) * ratio).round().max(0.0);
        (
            Rect { pos, size: dvec2(size.x, h) },
            Rect { pos: dvec2(pos.x, pos.y + h), size: dvec2(size.x, GAP) },
            Rect { pos: dvec2(pos.x, pos.y + h + GAP), size: dvec2(size.x, (size.y - h - GAP).max(0.0)) },
        )
    }
}

/// The pane next to `from` in direction `dir`: of the panes on that side
/// that overlap it across, the nearest (ties: the most overlap).
pub fn neighbour(panes: &[(u64, Rect)], from: u64, dir: Dir) -> Option<u64> {
    let (_, r) = panes.iter().find(|(id, _)| *id == from)?;
    let overlap = |a0: f64, a1: f64, b0: f64, b1: f64| (a1.min(b1) - a0.max(b0)).max(0.0);
    panes
        .iter()
        .filter(|(id, _)| *id != from)
        .filter_map(|(id, o)| {
            let (gap, across) = match dir {
                Dir::Left => (r.pos.x - (o.pos.x + o.size.x), overlap(r.pos.y, r.pos.y + r.size.y, o.pos.y, o.pos.y + o.size.y)),
                Dir::Right => (o.pos.x - (r.pos.x + r.size.x), overlap(r.pos.y, r.pos.y + r.size.y, o.pos.y, o.pos.y + o.size.y)),
                Dir::Up => (r.pos.y - (o.pos.y + o.size.y), overlap(r.pos.x, r.pos.x + r.size.x, o.pos.x, o.pos.x + o.size.x)),
                Dir::Down => (o.pos.y - (r.pos.y + r.size.y), overlap(r.pos.x, r.pos.x + r.size.x, o.pos.x, o.pos.x + o.size.x)),
            };
            (gap >= -0.5 && across > 0.0).then_some((*id, gap, across))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1).then(b.2.total_cmp(&a.2)))
        .map(|(id, _, _)| id)
}

/// The ratio a divider dragged to `at` gives its split.
pub fn ratio_at(divider: &Divider, at: f64) -> f64 {
    let area = divider.area;
    if divider.side_by_side {
        (at - area.pos.x) / (area.size.x - GAP).max(1.0)
    } else {
        (at - area.pos.y) / (area.size.y - GAP).max(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f64, y: f64, w: f64, h: f64) -> Rect {
        Rect { pos: dvec2(x, y), size: dvec2(w, h) }
    }

    #[test]
    fn splitting_and_closing_panes() {
        let mut tree = Node::Leaf(1);
        assert!(tree.split(1, 2, true));
        assert!(tree.split(2, 3, false));
        assert!(!tree.split(9, 4, true), "no such pane");
        assert_eq!(tree.leaves(), [1, 2, 3]);
        let tree = tree.remove(2).unwrap();
        assert_eq!(tree.leaves(), [1, 3]);
        let tree = tree.remove(1).unwrap();
        assert_eq!(tree, Node::Leaf(3));
        assert_eq!(tree.remove(3), None);
    }

    #[test]
    fn panes_tile_the_area_without_overlap() {
        let mut tree = Node::Leaf(1);
        tree.split(1, 2, true);
        tree.split(2, 3, false);
        let area = rect(0.0, 0.0, 803.0, 603.0);
        let panes = tree.layout(area);
        assert_eq!(panes[0], (1, rect(0.0, 0.0, 400.0, 603.0)));
        assert_eq!(panes[1], (2, rect(403.0, 0.0, 400.0, 300.0)));
        assert_eq!(panes[2], (3, rect(403.0, 303.0, 400.0, 300.0)));
        let dividers = tree.dividers(area);
        assert_eq!(dividers.len(), 2);
        assert_eq!(dividers[0].rect, rect(400.0, 0.0, GAP, 603.0));
        assert_eq!(dividers[1].path, [true]);
    }

    #[test]
    fn moving_between_panes_by_direction() {
        let mut tree = Node::Leaf(1);
        tree.split(1, 2, true);
        tree.split(2, 3, false);
        let panes = tree.layout(rect(0.0, 0.0, 803.0, 603.0));
        assert_eq!(neighbour(&panes, 1, Dir::Right), Some(2), "the nearest, most overlapping");
        assert_eq!(neighbour(&panes, 3, Dir::Left), Some(1));
        assert_eq!(neighbour(&panes, 3, Dir::Up), Some(2));
        assert_eq!(neighbour(&panes, 2, Dir::Down), Some(3));
        assert_eq!(neighbour(&panes, 1, Dir::Left), None);
    }

    #[test]
    fn dragging_a_divider_resizes_within_limits() {
        let mut tree = Node::Leaf(1);
        tree.split(1, 2, true);
        let area = rect(0.0, 0.0, 803.0, 600.0);
        let divider = tree.dividers(area).remove(0);
        tree.set_ratio(&divider.path, ratio_at(&divider, 200.0));
        assert_eq!(tree.layout(area)[0].1.size.x, 200.0);
        tree.set_ratio(&divider.path, ratio_at(&divider, 0.0));
        assert!(tree.layout(area)[0].1.size.x >= 80.0, "kept from collapsing");
        tree.equalize();
        assert_eq!(tree.layout(area)[0].1.size.x, 400.0);
    }
}
