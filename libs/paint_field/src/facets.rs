//! Imported facet paint in the mesh asset's frame (millimetres). Each state has a site tree;
//! all trees share the triangles and one permuted array of leaf indices.

use crate::flood::{box_distance, SiteNode, REACH_MM};
use crate::program::P3;
use std::hash::{Hash, Hasher};

pub struct Facets {
    tris: Vec<[[f32; 3]; 3]>,
    states: Vec<u8>,
    nodes: Vec<SiteNode>,
    sites: Vec<u32>,
    roots: Vec<(u8, usize)>,
    // The paint key reads Debug: hash the content once, when the asset loads.
    key: u64,
}

impl std::fmt::Debug for Facets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Facets {{ leaves: {}, key: {:016x} }}", self.tris.len(), self.key)
    }
}

impl Facets {
    /// Builds the trees once. Each leaf has one state: zero is unpainted.
    pub fn new(tris: Vec<[[f32; 3]; 3]>, states: Vec<u8>) -> Facets {
        assert_eq!(tris.len(), states.len(), "each facet needs one state");
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        for t in &tris {
            for p in t {
                for x in p {
                    x.to_bits().hash(&mut hash);
                }
            }
        }
        states.hash(&mut hash);
        let key = hash.finish();
        let mut sites: Vec<u32> = (0..tris.len()).map(|i| i as u32).collect();
        sites.sort_unstable_by_key(|&i| states[i as usize]);
        let mut nodes = Vec::new();
        let mut roots = Vec::new();
        let mut start = 0;
        while start < sites.len() {
            let state = states[sites[start] as usize];
            let end = start + sites[start..].partition_point(|&i| states[i as usize] == state);
            let root = SiteNode::build(&tris, &mut sites[start..end], start, &mut nodes);
            roots.push((state, root));
            start = end;
        }
        Facets { tris, states, nodes, sites, roots, key }
    }

    pub fn tris(&self) -> &[[[f32; 3]; 3]] {
        &self.tris
    }

    pub fn states(&self) -> &[u8] {
        &self.states
    }

    /// The states that occur, in ascending order, including zero when present.
    pub fn used(&self) -> impl Iterator<Item = u8> + '_ {
        self.roots.iter().map(|&(state, _)| state)
    }

    /// The box of this state's leaves in the asset frame.
    pub fn bounds(&self, state: u8) -> Option<(P3, P3)> {
        let &(_, root) = self.roots.iter().find(|&&(s, _)| s == state)?;
        let node = &self.nodes[root];
        Some((node.min, node.max))
    }

    /// The bisector of the nearest leaf of this state and the nearest other leaf, negative
    /// inside. Both searches stop at REACH_MM; past both, that distance is a lower bound.
    pub fn distance(&self, state: u8, p: P3) -> f64 {
        let mut best = [(None, REACH_MM); 2];
        for &(s, root) in &self.roots {
            SiteNode::nearest(&self.tris, &self.sites, &self.nodes, p, root, box_distance(p, &self.nodes[root]), &mut best[usize::from(s != state)]);
        }
        if best[0].0.is_none() && best[1].0.is_none() {
            REACH_MM
        } else {
            0.5 * (best[0].1 - best[1].1)
        }
    }
}
