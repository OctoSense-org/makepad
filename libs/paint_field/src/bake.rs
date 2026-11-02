//! The program baked for the views: a sparse brick volume per part, and every part's volume
//! packed into one float texture the shaders read (`shader.rs`).
//!
//! Lattice: sample `g` sits at `g * cell` in the part's frame; brick `b` holds samples
//! `8b .. 8b + 7` per axis. A sample is `slot * 256 + q`, `q` the distance to the nearest
//! boundary over the ramp (`RAMP` cells) in 0..255. A brick no boundary comes within a ramp of
//! is one slot in the index; only bricks a boundary passes hold samples.

use crate::program::{Channel, Program, Region, P3};

/// The ramp the distance is kept over, in cells.
pub const RAMP: f64 = 2.5;
/// Samples per brick edge.
pub const BS: i32 = 8;
/// Texels a brick takes (512 samples, four per texel).
pub const BRICK_TEXELS: usize = 128;
/// At most this many sample bricks per volume (2 KB each): a coarser lattice past it.
pub const MAX_BRICKS: usize = 32_768;

#[derive(Clone, Debug, Default)]
pub struct Volume {
    pub cell: f64,
    pub min: [i32; 3],
    pub dim: [i32; 3],
    /// Per brick: `-1 - slot` for a whole brick, else its sample brick.
    pub index: Vec<f32>,
    /// 512 samples per sample brick.
    pub samples: Vec<f32>,
}

impl Volume {
    pub fn bricks(&self) -> usize {
        self.samples.len() / 512
    }
    pub fn bytes(&self) -> usize {
        self.samples.len() * 4 + self.index.len() * 4
    }

    /// The lattice pitch for a program: a fifth of its finest stroke radius, snapped to 0.4,
    /// 0.2 or 0.1 mm so a brick nests in the program's grid cell (a strokes file's index brick);
    /// 0.4 mm without strokes.
    pub fn pitch(program: &Program) -> f64 {
        let finest = program
            .layers
            .iter()
            .filter_map(|l| match &l.region {
                Region::Stroke { radius, .. } => Some(*radius),
                Region::Strokes { strokes, .. } => strokes.finest_paint(),
                _ => None,
            })
            .fold(None, |a: Option<f64>, r| Some(a.map_or(r, |a| a.min(r))));
        let coarse = crate::program::CELL / BS as f64;
        finest.map_or(coarse, |r| {
            let k = (coarse / (r * 0.2).clamp(0.1, coarse)).log2().round().clamp(0.0, 2.0);
            coarse / 2f64.powi(k as i32)
        })
    }

    /// The program's `channel` over `part` (its box in the program's frame), at `cell` or
    /// coarser when the sample bricks would pass [`MAX_BRICKS`].
    pub fn bake(program: &Program, channel: Channel, part: (P3, P3), cell: f64) -> Volume {
        let mut cell = cell;
        loop {
            match Self::bake_at(program, channel, part, cell) {
                Some(v) => return v,
                None => cell *= 2.0,
            }
        }
    }

    fn bake_at(program: &Program, channel: Channel, part: (P3, P3), cell: f64) -> Option<Volume> {
        let ramp = RAMP * cell;
        let bs = BS as f64 * cell;
        let (lo, hi) = (part.0.map(|v| v - ramp), part.1.map(|v| v + ramp));
        let min: [i32; 3] = std::array::from_fn(|k| (lo[k] / bs).floor() as i32);
        let max: [i32; 3] = std::array::from_fn(|k| (hi[k] / bs).floor() as i32);
        let dim: [i32; 3] = std::array::from_fn(|k| max[k] - min[k] + 1);
        let n = (dim[0] * dim[1] * dim[2]) as usize;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
        let per = n.div_ceil(threads).max(1);
        let chunks: Vec<Vec<(f32, Option<Vec<f32>>)>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..n)
                .step_by(per)
                .map(|start| {
                    s.spawn(move || {
                        (start..(start + per).min(n))
                            .map(|i| {
                                let i = i as i32;
                                let b = [min[0] + i % dim[0], min[1] + (i / dim[0]) % dim[1], min[2] + i / (dim[0] * dim[1])];
                                brick(program, channel, cell, b)
                            })
                            .collect()
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("a paint bake thread")).collect()
        });
        let fine = chunks.iter().flatten().filter(|(_, s)| s.is_some()).count();
        if fine > MAX_BRICKS {
            return None;
        }
        let mut v = Volume { cell, min, dim, index: Vec::with_capacity(n), samples: Vec::with_capacity(fine * 512) };
        for (e, s) in chunks.into_iter().flatten() {
            match s {
                Some(s) => {
                    v.index.push(v.bricks() as f32);
                    v.samples.extend(s);
                }
                None => v.index.push(e),
            }
        }
        Some(v)
    }
}

/// One brick: whole (`-1 - slot`) or its 512 samples.
fn brick(program: &Program, channel: Channel, cell: f64, b: [i32; 3]) -> (f32, Option<Vec<f32>>) {
    let ramp = RAMP * cell;
    // A brick's samples and the gap to the next brick's first sample: half its diagonal.
    let reach = 0.5 * 8.0 * cell * 3f64.sqrt() + ramp;
    let c: P3 = std::array::from_fn(|k| (b[k] as f64 * 8.0 + 3.5) * cell);
    let s = program.sample(channel, c);
    if s.dist > reach {
        return (-1.0 - s.slot as f32, None);
    }
    let mut out = vec![0f32; 512];
    for (j, o) in out.iter_mut().enumerate() {
        let g = [b[0] * BS + (j & 7) as i32, b[1] * BS + ((j >> 3) & 7) as i32, b[2] * BS + (j >> 6) as i32];
        let s = program.sample_near(channel, g.map(|v| v as f64 * cell), ramp);
        let q = ((s.dist / ramp).min(1.0) * 255.0).round() as f32;
        *o = s.slot as f32 * 256.0 + q;
    }
    // a brick of one slot all deeper than the ramp is a whole brick
    let first = (out[0] / 256.0).floor();
    if out.iter().all(|v| (v / 256.0).floor() == first && v - first * 256.0 >= 255.0) {
        return (-1.0 - first, None);
    }
    (0.0, Some(out))
}

impl Volume {
    /// The bricks of `region` (the program's frame) baked again from `program`: what a stroke
    /// being drawn touches. Samples a brick no longer uses stay until [`Volume::bake`] runs; the
    /// result says false when they pass the live bricks (bake whole then).
    pub fn rebake(&mut self, program: &Program, channel: Channel, region: (P3, P3)) -> bool {
        self.rebake_boxes(program, channel, &[region])
    }

    /// [`Volume::rebake`] of every brick one of `boxes` reaches (a stroke's capsules, each its own
    /// box: a long diagonal stroke bakes its path, not its bounding box). A brick a sample
    /// brick no longer uses keeps its samples until the volume is packed again
    /// ([`Volume::compact`]); false when they pass the live ones (compact or bake whole then).
    pub fn rebake_boxes(&mut self, program: &Program, channel: Channel, boxes: &[(P3, P3)]) -> bool {
        let bs = BS as f64 * self.cell;
        let ramp = RAMP * self.cell;
        let mut seen = std::collections::HashSet::new();
        let mut todo: Vec<[i32; 3]> = Vec::new();
        for region in boxes {
            let lo: [i32; 3] = std::array::from_fn(|k| (((region.0[k] - ramp) / bs).floor() as i32).max(self.min[k]));
            let hi: [i32; 3] = std::array::from_fn(|k| (((region.1[k] + ramp) / bs).floor() as i32).min(self.min[k] + self.dim[k] - 1));
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    for x in lo[0]..=hi[0] {
                        if seen.insert([x, y, z]) {
                            todo.push([x, y, z]);
                        }
                    }
                }
            }
        }
        let cell = self.cell;
        let baked: Vec<(f32, Option<Vec<f32>>)> = if todo.len() < 16 {
            todo.iter().map(|b| brick(program, channel, cell, *b)).collect()
        } else {
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
            let per = todo.len().div_ceil(threads);
            std::thread::scope(|s| {
                let hs: Vec<_> = todo.chunks(per).map(|c| s.spawn(move || c.iter().map(|b| brick(program, channel, cell, *b)).collect::<Vec<_>>())).collect();
                hs.into_iter().flat_map(|h| h.join().expect("a paint rebake thread")).collect()
            })
        };
        for (b, r) in todo.into_iter().zip(baked) {
            let i = (((b[2] - self.min[2]) * self.dim[1] + (b[1] - self.min[1])) * self.dim[0] + (b[0] - self.min[0])) as usize;
            let old = self.index[i];
            match r {
                (e, None) => self.index[i] = e,
                // a brick that holds samples already takes the new ones in place
                (_, Some(samples)) if old >= 0.0 => {
                    let at = old as usize * 512;
                    self.samples[at..at + 512].copy_from_slice(&samples);
                }
                (_, Some(samples)) => {
                    self.index[i] = self.bricks() as f32;
                    self.samples.extend(samples);
                }
            }
        }
        let live = self.index.iter().filter(|e| **e >= 0.0).count();
        self.bricks() <= 2 * live.max(64) && self.bricks() <= MAX_BRICKS * 2
    }

    /// The samples no brick uses dropped (after rebakes).
    pub fn compact(&mut self) {
        let mut samples = Vec::with_capacity(self.samples.len());
        for e in &mut self.index {
            if *e >= 0.0 {
                let at = *e as usize * 512;
                let n = samples.len() / 512;
                samples.extend_from_slice(&self.samples[at..at + 512]);
                *e = n as f32;
            }
        }
        self.samples = samples;
    }
}

/// One part's entry in the atlas.
pub struct AtlasPart<'a> {
    pub volume: &'a Volume,
    /// Rows of the affine from the shader's position to the volume's frame (mm).
    pub from: [[f32; 4]; 3],
    /// Linear RGB per slot (slot 0 unused).
    pub palette: &'a [[f32; 3]],
}

/// Every part's volume as RGBA32F texels starting at texel `base` of a texture `width` wide
/// (the host's own table may sit before it). Per volume a header: `(cell, ramp, min.x, min.y)`,
/// `(min.z, dim.x, dim.y, dim.z)`, `(index, samples, palette, 0)` texel bases, three affine rows;
/// then the brick index (four per texel), the palette, the samples (128 texels a brick). All
/// texel numbers are absolute.
#[derive(Clone, Debug, Default)]
pub struct Atlas {
    /// The texels from `base` on (whole rows: the texture's rows from `base / width`).
    pub data: Vec<f32>,
    /// Each part's header texel, in the order given.
    pub headers: Vec<f32>,
}

impl Atlas {
    /// Parts that share a volume (one part seen through two frames: its mesh and its field)
    /// share its index and samples. `base` is a multiple of `width`.
    pub fn pack(parts: &[AtlasPart], base: usize, width: usize) -> Atlas {
        let mut t: Vec<[f32; 4]> = Vec::new();
        let mut headers = Vec::with_capacity(parts.len());
        let mut packed: Vec<(*const Volume, usize, usize, usize)> = Vec::new();
        for p in parts {
            let v = p.volume;
            let h = base + t.len();
            headers.push(h as f32);
            t.resize(t.len() + 6, [0.0; 4]);
            let (index, palette, samples) = match packed.iter().find(|e| std::ptr::eq(e.0, v)) {
                Some(e) => (e.1, e.2, e.3),
                None => {
                    let index = base + t.len();
                    for c in v.index.chunks(4) {
                        let mut x = [0.0; 4];
                        x[..c.len()].copy_from_slice(c);
                        t.push(x);
                    }
                    let palette = base + t.len();
                    t.push([0.0; 4]);
                    for c in p.palette.iter().skip(1) {
                        t.push([c[0], c[1], c[2], 1.0]);
                    }
                    let samples = base + t.len();
                    for c in v.samples.chunks(4) {
                        t.push([c[0], c[1], c[2], c[3]]);
                    }
                    packed.push((v as *const Volume, index, palette, samples));
                    (index, palette, samples)
                }
            };
            let k = h - base;
            t[k] = [v.cell as f32, (RAMP * v.cell) as f32, v.min[0] as f32, v.min[1] as f32];
            t[k + 1] = [v.min[2] as f32, v.dim[0] as f32, v.dim[1] as f32, v.dim[2] as f32];
            t[k + 2] = [index as f32, samples as f32, palette as f32, 0.0];
            t[k + 3] = p.from[0];
            t[k + 4] = p.from[1];
            t[k + 5] = p.from[2];
        }
        let rows = t.len().div_ceil(width.max(1));
        t.resize(rows * width, [0.0; 4]);
        Atlas { data: t.into_iter().flatten().collect(), headers }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{Layer, Op};

    #[test]
    fn a_stroke_bakes_sparse_and_reads_back() {
        let p = Program::new(vec![Layer { region: Region::Stroke { points: vec![[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], radius: 2.0 }, op: Op::Paint { channel: Channel::Color, slot: 1 } }]);
        let v = Volume::bake(&p, Channel::Color, ([-5.0, -5.0, -5.0], [25.0, 5.0, 5.0]), Volume::pitch(&p));
        assert!((v.cell - 0.4).abs() < 1e-12);
        assert!(v.bricks() > 0 && v.bricks() < v.index.len(), "{} of {}", v.bricks(), v.index.len());
        let part = AtlasPart { volume: &v, from: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], palette: &[[0.0; 3], [1.0, 0.0, 0.0]] };
        let a = Atlas::pack(&[part, AtlasPart { from: [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]], ..AtlasPart { volume: &v, from: [[0.0; 4]; 3], palette: &[[0.0; 3], [1.0, 0.0, 0.0]] } }], 256, 256);
        assert_eq!(a.data.len() % (256 * 4), 0);
        assert_eq!(a.headers[0], 256.0);
        // the second header shares the first's index and samples
        assert_eq!(a.data[(a.headers[1] as usize - 256 + 2) * 4], a.data[2 * 4]);
    }
}
