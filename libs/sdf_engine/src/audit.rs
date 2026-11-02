//! The storage atlas against the exact field, per frame, on the GPU's own texels.
//!
//! A host enables this for an object while debugging or running a regression.
//! The storage atlas is read after its pending compute writes, as native f16
//! bytes. A platform worker compares 27 samples per brick (including aprons)
//! with the twin and classifies stale values by generation:
//!
//! - `new`: the frame's own field, at the brick's place;
//! - `old`: the field of an earlier generation (the volume this one replaced), at that place;
//! - `slot`: what an earlier generation kept in this atlas slot, which belonged to ANOTHER
//!   brick of that generation (a slot handed on, its texels not baked yet);
//! - `next`: the field being baked, not published yet (at this place, or in this slot);
//! - `blank`: the atlas's clear value;
//! - `neither`: none of those.
//!
//! The answer is one log line per frame (`sdf audit:`), and with a directory one file per
//! frame that drew anything but `new`: per brick its class and its worst sample.

use crate::bake::{Baker, ATLAS_SLOTS_X};
use crate::kernel::KernelSlot;
use crate::program::lower_with;
use crate::twin;
use crate::volume::{Brick, Volume, BS};
use makepad_csg_math::thread_pool::parallel_map;
use makepad_csg_sdf::SdfShape;
use makepad_draw::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

/// Generations an answer is looked for in, the frame's own and the ones before.
const GENERATIONS: usize = 4;
/// The samples read per axis of a brick's nine.
const SAMPLES: [usize; 3] = [0, 4, 8];

/// One volume as a frame drew it, with its exact field.
struct Generation {
    number: u64,
    vol: Volume,
    shape: Arc<SdfShape>,
    /// The whole shape as one program.
    field: Vec<[f32; 4]>,
    kernels: Vec<KernelSlot>,
    /// Which brick each atlas slot holds.
    slots: HashMap<u32, [i32; 3]>,
}

impl Generation {
    fn new(number: u64, vol: &Volume, shape: &Arc<SdfShape>) -> Option<Generation> {
        let field = lower_with(&shape.root, &vol.kernels, vol.kernel_scale).ok()?.scaled(vol.field_scale).texels;
        let slots = (0..vol.bricks.len()).filter_map(|i| match vol.bricks[i] {
            Brick::Baked { slot, .. } => Some((slot, vol.brick_of(i))),
            _ => None,
        });
        Some(Generation { number, vol: vol.clone(), shape: shape.clone(), field, kernels: vol.kernels.clone(), slots: slots.collect() })
    }

    /// The copy's value at world points `ps` per this generation: the field within its band.
    fn exact(&self, ps: &[[f32; 3]]) -> Vec<f32> {
        twin::eval_many(&self.field, ps, &self.kernels).into_iter().map(|(d, _)| d.max(-self.vol.band).min(self.vol.band)).collect()
    }

    /// The sample points of brick `b`, in this generation's frame.
    fn points(&self, b: [i32; 3]) -> Vec<[f32; 3]> {
        let mut ps = Vec::with_capacity(SAMPLES.len().pow(3));
        for z in SAMPLES {
            for y in SAMPLES {
                for x in SAMPLES {
                    ps.push([(b[0] * BS + x as i32) as f32 * self.vol.cell, (b[1] * BS + y as i32) as f32 * self.vol.cell, (b[2] * BS + z as i32) as f32 * self.vol.cell]);
                }
            }
        }
        ps
    }
}

/// What a brick's samples turned out to be.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BrickClass {
    New,
    /// The field `n` generations back, at this brick's place.
    Old(u64),
    /// The slot's content `n` generations back: another brick's.
    Slot(u64),
    /// The field not published yet, at this brick's place (`false`) or in this slot (`true`).
    Next(bool),
    Blank,
    Neither,
}

/// A frame's bricks by class.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameTally {
    pub bricks: usize,
    pub new: usize,
    pub old: usize,
    pub slot: usize,
    pub next: usize,
    pub blank: usize,
    pub neither: usize,
    /// The largest distance from the exact field among the samples, in cells.
    pub err_cells: f32,
}

impl FrameTally {
    pub fn wrong(&self) -> usize {
        self.bricks - self.new
    }
}

/// The classes of one frame's bricks: `samples[slot]` are the 27 values read of each slot.
fn classify(gens: &[Arc<Generation>], next: Option<Arc<Generation>>, samples: &HashMap<u32, Vec<f32>>, mut per_brick: impl FnMut([i32; 3], u32, BrickClass, f32, String)) -> FrameTally {
    let g = &gens[0];
    let tol = |cell: f32| cell * 0.01;
    let bricks: Vec<([i32; 3], u32)> = g.slots.iter().map(|(s, b)| (*b, *s)).collect();
    let samples = Arc::new(samples.clone());
    let gens_arc: Arc<Vec<Arc<Generation>>> = Arc::new(gens.to_vec());
    let out = parallel_map(&bricks, move |chunk| {
        chunk
            .iter()
            .map(|&(b, slot)| {
                let gens = &*gens_arc;
                let g = &gens[0];
                let Some(got) = samples.get(&slot) else { return (b, slot, BrickClass::Neither, f32::MAX, String::new()) };
                let ps = g.points(b);
                let exact = g.exact(&ps);
                // past nine tenths of the band a value only sets a step (the march evaluates
                // nothing there exactly): two such values of one sign are alike
                let far = 0.9 * g.vol.band;
                let apart = |a: f32, x: f32| if a.abs() >= far && x.abs() >= far && (a > 0.0) == (x > 0.0) { 0.0 } else { (a - x).abs() };
                let diff = |e: &[f32]| got.iter().zip(e).map(|(a, x)| apart(*a, *x)).fold(0.0f32, f32::max);
                let err = diff(&exact);
                // the worst sample: where, what the copy holds, what the field says
                let worst = || {
                    let k = (0..got.len()).max_by(|a, b| apart(got[*a], exact[*a]).total_cmp(&apart(got[*b], exact[*b]))).unwrap_or(0);
                    format!("at {:?} got {} exact {}", ps[k], got[k], exact[k])
                };
                if err <= tol(g.vol.cell) {
                    return (b, slot, BrickClass::New, err / g.vol.cell, String::new());
                }
                if got.iter().all(|v| *v == g.vol.band) || got.iter().all(|v| *v == 0.0) {
                    return (b, slot, BrickClass::Blank, err / g.vol.cell, worst());
                }
                for old in gens.iter().skip(1) {
                    if diff(&old.exact(&ps)) <= tol(old.vol.cell) {
                        return (b, slot, BrickClass::Old(g.number - old.number), err / g.vol.cell, worst());
                    }
                }
                for old in gens.iter().skip(1) {
                    if let Some(held) = old.slots.get(&slot) {
                        if diff(&old.exact(&old.points(*held))) <= tol(old.vol.cell) {
                            return (b, slot, BrickClass::Slot(g.number - old.number), err / g.vol.cell, worst());
                        }
                    }
                }
                if let Some(n) = &next {
                    if diff(&n.exact(&ps)) <= tol(n.vol.cell) {
                        return (b, slot, BrickClass::Next(false), err / g.vol.cell, worst());
                    }
                    if let Some(held) = n.slots.get(&slot) {
                        if diff(&n.exact(&n.points(*held))) <= tol(n.vol.cell) {
                            return (b, slot, BrickClass::Next(true), err / g.vol.cell, worst());
                        }
                    }
                }
                (b, slot, BrickClass::Neither, err / g.vol.cell, worst())
            })
            .collect()
    });
    let mut t = FrameTally { bricks: out.len(), ..Default::default() };
    for (b, slot, class, err, worst) in out {
        match class {
            BrickClass::New => t.new += 1,
            BrickClass::Old(_) => t.old += 1,
            BrickClass::Slot(_) => t.slot += 1,
            BrickClass::Next(_) => t.next += 1,
            BrickClass::Blank => t.blank += 1,
            BrickClass::Neither => t.neither += 1,
        }
        if err != f32::MAX {
            t.err_cells = t.err_cells.max(err);
        }
        if class != BrickClass::New {
            per_brick(b, slot, class, err, worst);
        }
    }
    t
}

struct FrameData {
    object: String,
    frame: u64,
    age: u64,
    pending: (usize, usize, usize),
    gens: Vec<Arc<Generation>>,
    next: Option<Arc<Generation>>,
    samples: HashMap<u32, Vec<f32>>,
}

fn answer(f: FrameData, dir: Option<&str>) {
    let mut wrong: Vec<String> = Vec::new();
    let t = classify(&f.gens, f.next.clone(), &f.samples, |b, slot, class, err, worst| wrong.push(format!("{} {} {} slot {slot} {class:?} {err:.3} {worst}", b[0], b[1], b[2])));
    let g = &f.gens[0];
    log!(
        "sdf audit: {} frame {} gen {} age {} queued {}+{}+{} bricks {} new {} old {} slot {} next {} blank {} neither {} worst {:.2} cells; cell {} band {} min {:?} dim {:?}",
        f.object, f.frame, g.number, f.age, f.pending.0, f.pending.1, f.pending.2, t.bricks, t.new, t.old, t.slot, t.next, t.blank, t.neither, t.err_cells, g.vol.cell, g.vol.band, g.vol.min, g.vol.dim
    );
    if let Some(dir) = dir.filter(|_| !wrong.is_empty()) {
        let bounds = g.shape.root.clone();
        let text = format!("# gen {} age {} cell {} band {} min {:?} dim {:?} bounds {:?}\n{}\n", g.number, f.age, g.vol.cell, g.vol.band, g.vol.min, g.vol.dim, makepad_csg_sdf::node_bounds(&bounds), wrong.join("\n"));
        let _ = std::fs::write(format!("{dir}/frame-{:06}.txt", f.frame), text);
    }
}

/// One object's audit.
pub struct BakeAudit {
    object: String,
    dir: Option<String>,
    gens: VecDeque<Arc<Generation>>,
    published: u64,
    published_frame: u64,
    frame: u64,
    skipped: u64,
    pending: Arc<AtomicUsize>,
    /// The field being baked (by the volume's address), when it is not the one drawn.
    next: Option<(usize, Arc<Generation>)>,
}

impl BakeAudit {
    /// The host names the object and where a frame's bricks go.
    pub fn new(_cx: &mut Cx, object: &str, dir: Option<String>) -> Option<BakeAudit> {
        Some(BakeAudit {
            object: object.to_string(),
            dir,
            gens: VecDeque::new(),
            published: u64::MAX,
            published_frame: 0,
            frame: 0,
            skipped: 0,
            pending: Arc::new(AtomicUsize::new(0)),
            next: None,
        })
    }

    /// Whether answers are still to come (the host keeps its frames coming).
    pub fn busy(&self) -> bool {
        self.pending.load(Ordering::Acquire) != 0
    }

    pub fn directory(&self) -> Option<&str> {
        self.dir.as_deref()
    }

    /// Read this frame's atlas after the bake's pending compute writes.
    /// `vol` and `shape`: what the frame's march draws; `next`: the volume being baked.
    pub fn frame(&mut self, cx: &mut Cx2d, _parent: DrawPassId, baker: &Baker, vol: &Volume, shape: &Arc<SdfShape>, next: Option<(&Volume, &Arc<SdfShape>)>) {
        let next = next.filter(|(n, _)| !std::ptr::eq(*n, vol));
        match next {
            Some((n, s)) if self.next.as_ref().is_none_or(|(at, _)| *at != n as *const Volume as usize) => {
                self.next = Generation::new(u64::MAX, n, s).map(|g| (n as *const Volume as usize, Arc::new(g)));
            }
            None => self.next = None,
            _ => {}
        }
        self.frame += 1;
        if baker.published() != self.published {
            self.published = baker.published();
            self.published_frame = self.frame;
            let number = self.gens.front().map_or(0, |g| g.number + 1);
            if let Some(g) = Generation::new(number, vol, shape) {
                self.gens.push_front(Arc::new(g));
                self.gens.truncate(GENERATIONS);
            }
        }
        if self.gens.is_empty() {
            return;
        }
        let age = self.frame - self.published_frame;
        let gens: Vec<Arc<Generation>> = self.gens.iter().cloned().collect();
        let mut frame = FrameData { object: self.object.clone(), frame: self.frame, age,
            pending: baker.queued(), gens, next: self.next.as_ref().map(|n| n.1.clone()), samples: HashMap::new() };
        let pending = self.pending.clone();
        let dir = self.dir.clone();
        pending.fetch_add(1, Ordering::AcqRel);
        // Native R16F bytes, after this frame's compute producer. The platform
        // runs the callback on its worker pool, so decoding and the twin stay off UI.
        if let Err(error) = cx.read_render_texture(baker.atlas(), move |result| {
            match result {
                Ok((width, height, data)) => {
                    for slot in frame.gens[0].slots.keys() {
                        let s = *slot as usize;
                        let (x0, y0) = ((s % ATLAS_SLOTS_X) * 81, (s / ATLAS_SLOTS_X) * 9);
                        if y0 + 8 >= height { continue; }
                        let mut values = Vec::with_capacity(27);
                        for z in SAMPLES { for y in SAMPLES { for x in SAMPLES {
                            let at = ((y0 + y) * width + x0 + z * 9 + x) * 2;
                            let bits = u16::from_le_bytes([data[at], data[at + 1]]);
                            let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
                            let exponent = (bits >> 10) & 31;
                            let fraction = (bits & 1023) as f32;
                            let value = if exponent == 0 { fraction * 2.0f32.powi(-24) }
                                else { (1.0 + fraction / 1024.0) * 2.0f32.powi(exponent as i32 - 15) };
                            values.push(sign * value);
                        } } }
                        frame.samples.insert(*slot, values);
                    }
                    answer(frame, dir.as_deref());
                }
                Err(error) => log!("sdf audit: {} frame {}: {error}", frame.object, frame.frame),
            }
            pending.fetch_sub(1, Ordering::AcqRel);
            SignalToUI::set_ui_signal();
        }) {
            self.pending.fetch_sub(1, Ordering::AcqRel);
            self.skipped += 1;
            log!("sdf audit: {} frame {}: {error}", self.object, self.frame);
        }
    }
}
