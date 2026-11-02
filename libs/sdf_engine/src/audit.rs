//! The render copy against the exact field, per frame, on the GPU's own texels.
//!
//! A host turns this on for one object while it debugs a picture or runs a regression test
//! (nothing here runs otherwise). [`BakeAudit::frame`] records, after the frame's bake and
//! before the frame's march, a pass that writes every texel of the render copy's used rows as
//! the bits of its float into a BGRA8 target, and reads that back. A worker then takes each
//! baked brick the frame's index points at and compares 27 of its 729 samples (the apron's
//! among them) with the exact fields:
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

use crate::bake::{Baker, COPY_SLOTS_X, COPY_W};
use crate::kernel::KernelSlot;
use crate::program::lower_with;
use crate::twin;
use crate::volume::{Brick, Volume, BS};
use makepad_csg_math::thread_pool::parallel_map;
use makepad_csg_sdf::SdfShape;
use makepad_draw::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{mpsc, Arc};

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader.*
    use mod.draw

    // A texel of the render copy as its float's four bytes.
    mod.draw.SdfAudit = mod.std.set_type_default() do #(DrawSdfAudit::script_shader(vm)){
        ..mod.draw.DrawQuad
        alpha_blend: false
        copy_tex: texture_2d(float)
        copy_inv: uniform(vec2(1.0, 1.0))

        fragment: fn() {
            let uv = (floor(self.world.xy) + vec2(0.5, 0.5)) * self.copy_inv
            let b = asuint(self.copy_tex.sample_nearest(uv).x)
            self.fb0 = vec4(f32(b & 255u), f32((b >> 8u) & 255u), f32((b >> 16u) & 255u), f32((b >> 24u) & 255u)) * 0.003921568627450980
        }
    }
}

#[derive(Script, ScriptHook)]
#[repr(C)]
pub struct DrawSdfAudit {
    #[deref]
    pub draw_super: DrawQuad,
}

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
    pass: DrawPass,
    list: DrawList,
    draw: DrawSdfAudit,
    target: Option<(Texture, usize)>,
    gens: VecDeque<Arc<Generation>>,
    published: u64,
    published_frame: u64,
    frame: u64,
    skipped: u64,
    pending: Vec<(ReadbackTicket, u64, u64, (usize, usize, usize), Vec<Arc<Generation>>, Option<Arc<Generation>>)>,
    /// The field being baked (by the volume's address), when it is not the one drawn.
    next: Option<(usize, Arc<Generation>)>,
    to_worker: mpsc::Sender<FrameData>,
}

impl BakeAudit {
    /// The host names the object and where a frame's bricks go.
    pub fn new(cx: &mut Cx, object: &str, dir: Option<String>) -> Option<BakeAudit> {
        let draw = cx.with_vm(|vm| {
            let draw_mod = vm.heap().module(id!(draw));
            let base = vm.value(draw_mod, id!(SdfAudit)).as_object()?;
            Some(DrawSdfAudit::script_from_value(vm, base.into()))
        })?;
        let (to_worker, from_ui) = mpsc::channel::<FrameData>();
        let worker_dir = dir.clone();
        std::thread::Builder::new()
            .name("sdf-audit".into())
            .spawn(move || {
                while let Ok(f) = from_ui.recv() {
                    answer(f, worker_dir.as_deref());
                }
            })
            .ok()?;
        Some(BakeAudit {
            object: object.to_string(),
            dir,
            pass: DrawPass::new_with_name(cx, "sdf_audit"),
            list: DrawList::new(cx),
            draw,
            target: None,
            gens: VecDeque::new(),
            published: u64::MAX,
            published_frame: 0,
            frame: 0,
            skipped: 0,
            pending: Vec::new(),
            next: None,
            to_worker,
        })
    }

    /// The pass a baker hangs under while audited (it reads what the bake wrote).
    pub fn pass_id(&self, cx: &mut Cx, parent: DrawPassId) -> DrawPassId {
        self.pass.set_pass_parent(cx, parent);
        self.pass.draw_pass_id()
    }

    /// Whether answers are still to come (the host keeps its frames coming).
    pub fn busy(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn directory(&self) -> Option<&str> {
        self.dir.as_deref()
    }

    /// Takes the finished readbacks to the worker.
    pub fn poll(&mut self, cx: &mut Cx) {
        let tickets: Vec<ReadbackTicket> = self.pending.iter().map(|p| p.0).collect();
        for r in cx.try_take_texture_readbacks_for(&tickets) {
            let Some(at) = self.pending.iter().position(|p| p.0 == r.ticket) else { continue };
            let (_, frame, age, pending, gens, next) = self.pending.remove(at);
            let data = match r.data {
                Ok(d) => d,
                Err(e) => {
                    self.skipped += 1;
                    log!("sdf audit: {} frame {frame}: no readback: {e}", self.object);
                    continue;
                }
            };
            // the samples of each slot this frame's index points at (the payload goes now)
            let row = r.width * 4;
            let texel = |x: usize, y: usize| -> f32 {
                let o = y * row + x * 4;
                f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]])
            };
            let mut samples = HashMap::new();
            for slot in gens[0].slots.keys() {
                let s = *slot as usize;
                let (cx0, cy0) = ((s % COPY_SLOTS_X) * 81, (s / COPY_SLOTS_X) * 9);
                let mut v = Vec::with_capacity(SAMPLES.len().pow(3));
                for z in SAMPLES {
                    for y in SAMPLES {
                        for x in SAMPLES {
                            v.push(texel(cx0 + z * 9 + x, cy0 + y));
                        }
                    }
                }
                samples.insert(*slot, v);
            }
            let _ = self.to_worker.send(FrameData { object: self.object.clone(), frame, age, pending, gens, next, samples });
        }
    }

    /// This frame's copy of `baker`, under `parent` (records one pass; call after the bake).
    /// `vol` and `shape`: what the frame's march draws; `next`: the volume being baked.
    pub fn frame(&mut self, cx: &mut Cx2d, parent: DrawPassId, baker: &Baker, vol: &Volume, shape: &Arc<SdfShape>, next: Option<(&Volume, &Arc<SdfShape>)>) {
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
        self.poll(cx.cx);
        if self.gens.is_empty() {
            return;
        }
        let rows = (vol.slot_count as usize).div_ceil(COPY_SLOTS_X).max(1) * 9;
        if self.target.as_ref().is_none_or(|(_, r)| *r < rows) {
            let r = rows.next_power_of_two();
            let t = Texture::new_with_format(cx.cx, TextureFormat::RenderBGRAu8 { size: TextureSize::Fixed { width: COPY_W, height: r }, initial: true });
            self.target = Some((t, r));
        }
        let (target, r) = self.target.clone().unwrap();
        let size = dvec2(COPY_W as f64, r as f64);
        self.pass.set_size(cx, size);
        self.pass.clear_color_textures(cx.cx);
        self.pass.set_color_texture(cx, &target, DrawPassClearColor::InitWith(vec4(0.0, 0.0, 0.0, 0.0)));
        self.pass.clear_depth_texture(cx.cx);
        self.pass.set_pass_parent(cx.cx, parent);
        cx.begin_pass(&self.pass, Some(1.0));
        self.pass.set_size(cx, size);
        self.pass.set_dpi_factor(cx, 1.0);
        self.list.begin_always(cx);
        cx.begin_root_turtle(size, Layout::flow_overlay());
        {
            let c = &*cx.cx;
            let dv = &mut self.draw.draw_super.draw_vars;
            dv.set_texture(0, &baker.copy);
            dv.set_uniform(c, id!(copy_inv), &[1.0 / COPY_W as f32, 1.0 / baker.copy_rows as f32]);
        }
        self.draw.draw_super.draw_abs(cx, Rect { pos: dvec2(0.0, 0.0), size: dvec2(COPY_W as f64, rows as f64) });
        cx.end_pass_sized_turtle();
        self.list.end(cx);
        cx.end_pass(&self.pass);
        let age = self.frame - self.published_frame;
        let gens: Vec<Arc<Generation>> = self.gens.iter().cloned().collect();
        match target.read_back(cx.cx, ReadbackRequest { next_render: true }) {
            Ok(t) => self.pending.push((t, self.frame, age, baker.queued(), gens, self.next.as_ref().map(|n| n.1.clone()))),
            Err(e) => {
                self.skipped += 1;
                log!("sdf audit: {} frame {}: not read: {e}", self.object, self.frame);
            }
        }
    }
}
