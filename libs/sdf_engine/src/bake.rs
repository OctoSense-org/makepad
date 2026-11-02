//! Budgeted compute baking into a copy-on-write R16F/R8 brick atlas.
//!
//! A version publishes in the recording of its final slice. The compute pass
//! precedes its marching pass, including while native pipelines compile. An
//! asynchronous completion readback gates reuse of the bake list and retired
//! slots by the following version; the UI never waits for the device.

use crate::{kernel::KernelBake, volume::{Brick, Volume}};
use makepad_draw::*;
use makepad_draw::makepad_platform::gpu_admission::{GpuDocument, GpuReservation};
use makepad_draw::makepad_script::shader_ir_compute::StorageTextureFormat;
use std::{rc::Rc, sync::{Arc, atomic::{AtomicBool, Ordering}}};

pub const ATLAS_SLOTS_X: usize = 50;
pub const ATLAS_W: usize = ATLAS_SLOTS_X * 81;

#[derive(Clone)]
struct Buffer {
    gpu: GpuBuffer,
    data: Arc<[u8]>,
    _reservation: Rc<GpuReservation>,
}

impl Buffer {
    fn reserve(doc: &GpuDocument, old: Option<&Self>, bytes: usize) -> Result<Option<Rc<GpuReservation>>, String> {
        let size = bytes.max(16).next_power_of_two() as u64;
        if old.is_some_and(|b| b.gpu.size() >= size) { return Ok(None); }
        doc.admit(size).map(|r| Some(Rc::new(r))).map_err(|e| e.to_string())
    }

    fn prepare(cx: &mut Cx, old: Option<&Self>, bytes: usize, reservation: Option<Rc<GpuReservation>>) -> Result<Self, String> {
        let size = bytes.max(16).next_power_of_two() as u64;
        if let Some(old) = old.filter(|b| b.gpu.size() >= size) { return Ok(old.clone()); }
        let gpu = GpuBuffer::new(cx, size, GpuBufferFlags { readback: true, ..Default::default() })?;
        Ok(Self { gpu, data: Arc::from([]), _reservation: reservation.expect("reserved buffer growth") })
    }

    /// Coalesce changed vec4 ranges. Growing allocations upload their complete payload.
    fn upload(&mut self, cx: &mut Cx, data: Vec<u8>) -> Result<(), String> {
        if self.data.is_empty() {
            if !data.is_empty() { self.gpu.update(cx, 0, &data)?; }
        } else {
            let mut start = None;
            for offset in (0..data.len()).step_by(16) {
                let end = (offset + 16).min(data.len());
                let changed = self.data.get(offset..end) != Some(&data[offset..end]);
                if changed { start.get_or_insert(offset); }
                if !changed {
                    if let Some(first) = start.take() { self.gpu.update(cx, first as u64, &data[first..offset])?; }
                }
            }
            if let Some(first) = start { self.gpu.update(cx, first as u64, &data[first..])?; }
        }
        self.data = data.into();
        Ok(())
    }
}

#[derive(Clone)]
struct Atlas {
    distance: Texture,
    material: Texture,
    rows: usize,
    _reservation: Rc<GpuReservation>,
}

impl Atlas {
    fn new(cx: &mut Cx, rows: usize, reservation: GpuReservation) -> Self {
        let texture = |cx: &mut Cx, format| Texture::new_with_format(cx,
            TextureFormat::Storage { format, width: ATLAS_W, height: rows });
        Self {
            distance: texture(cx, StorageTextureFormat::R16f),
            material: texture(cx, StorageTextureFormat::R8unorm),
            rows, _reservation: Rc::new(reservation),
        }
    }
}

#[derive(Clone)]
struct Field {
    atlas: Atlas,
    index: Buffer,
    programs: Buffer,
    kernel: usize,
    meshes: Rc<[(LiveId, Rc<crate::mesh::MeshGpu>)]>,
}

struct Pending {
    field: Field,
    index: Vec<u8>,
    first: usize,
    count: usize,
}

pub struct Baker {
    document: Rc<GpuDocument>,
    field: Option<Field>,
    next: Option<Pending>,
    bricks: Option<Buffer>,
    kernels: Vec<KernelBake>,
    complete: Arc<AtomicBool>,
    fresh: bool,
    refused: bool,
    pub parent: Option<DrawPassId>,
    pub budget: usize,
    pub limits: crate::budget::Budget,
    pub baked: usize,
    kernel_cost: u64,
    recorded: usize,
    published: u64,
}

fn floats(data: Vec<[f32; 4]>) -> Vec<u8> {
    data.into_iter().flatten().flat_map(f32::to_le_bytes).collect()
}

impl Baker {
    /// All bakers belonging to an engine share its admission document.
    pub fn new(document: Rc<GpuDocument>) -> Self {
        let limits = crate::budget::Budget::for_platform();
        Self { document, field: None, next: None, bricks: None, kernels: Vec::new(),
            complete: Arc::new(AtomicBool::new(true)), fresh: true, refused: false,
            parent: None, budget: limits.start_bricks_per_frame, limits, baked: 0,
            kernel_cost: 0, recorded: 0, published: 0 }
    }

    /// False keeps the host's version waiting. A refused allocation never replaces
    /// the published field, and is retried when the host next services the engine.
    pub fn sync(&mut self, cx: &mut Cx, vol: &mut Volume) -> bool {
        if self.publishing() { return false; }
        match self.prepare(cx, vol) {
            Ok(()) => { self.refused = false; true }
            Err(error) => {
                if !self.refused { log!("sdf bake paused: {error}; retaining the published field"); }
                self.refused = true;
                false
            }
        }
    }

    fn prepare(&mut self, cx: &mut Cx, vol: &mut Volume) -> Result<(), String> {
        let old = self.field.as_ref().filter(|_| !self.fresh);
        let needed = (vol.slot_count as usize).div_ceil(ATLAS_SLOTS_X).max(1) * 9;
        let grow = old.is_none_or(|f| f.atlas.rows < needed);
        // Admit the entire growth before queuing any allocation. Check the
        // atlas first so a refused growth does not rebuild CPU upload data.
        let rows = needed.next_power_of_two();
        let atlas_reservation = if grow { Some(self.document.admit((ATLAS_W * rows * 3) as u64).map_err(|e| e.to_string())?) } else { None };
        // Reallocation has no carry/copy pass: every live slot is baked into the
        // new allocation while the old allocation remains published.
        if grow {
            vol.dirty = (0..vol.bricks.len()).filter(|i| matches!(vol.bricks[*i], Brick::Baked { .. })).collect();
        }
        let data = vol.buffers();
        let index_reservation = Buffer::reserve(&self.document, old.map(|f| &f.index), data.index.len() * 16)?;
        let program_reservation = Buffer::reserve(&self.document, old.map(|f| &f.programs), data.programs.len() * 16)?;
        let brick_reservation = Buffer::reserve(&self.document, self.bricks.as_ref(), data.bricks.len() * 16)?;
        let meshes = vol.kernels.iter().enumerate().filter_map(|(i, slot)| slot.mesh_asset().map(|asset| (i, asset)))
            .map(|(i, asset)| crate::mesh::upload(cx, &self.document, asset).map(|gpu| (crate::mesh::binding(i), gpu)))
            .collect::<Result<Vec<_>, _>>()?;
        let atlas = if let Some(reservation) = atlas_reservation { Atlas::new(cx, rows, reservation) }
            else { old.unwrap().atlas.clone() };
        let index = Buffer::prepare(cx, old.map(|f| &f.index), data.index.len() * 16, index_reservation)?;
        let mut programs = Buffer::prepare(cx, old.map(|f| &f.programs), data.programs.len() * 16, program_reservation)?;
        let mut bricks = Buffer::prepare(cx, self.bricks.as_ref(), data.bricks.len() * 16, brick_reservation)?;
        let kernel = match self.kernels.iter().position(|k| k.serves(&vol.kernels)) {
            Some(i) => i,
            None => {
                let k = KernelBake::new(cx, &vol.kernels).ok_or("cannot compile compute bake")?;
                self.kernels.push(k);
                self.kernels.len() - 1
            }
        };
        programs.upload(cx, floats(data.programs))?;
        let count = data.bricks.len() / 2;
        bricks.upload(cx, data.bricks.into_iter().flatten().flat_map(u32::to_le_bytes).collect())?;
        self.kernel_cost = vol.kernels.iter().map(|k| k.air().map_or_else(|| k.mesh_asset().unwrap().cost(), |g| g.cost)).sum();
        self.bricks = Some(bricks);
        self.next = Some(Pending { field: Field { atlas, index, programs, kernel, meshes: meshes.into() },
            index: floats(data.index), first: 0, count });
        self.fresh = false;
        vol.dirty.clear();
        vol.touched.clear();
        Ok(())
    }

    pub fn reset(&mut self) { debug_assert!(!self.publishing()); self.fresh = true; }
    pub fn atlas(&self) -> &Texture { &self.field.as_ref().expect("published field").atlas.distance }
    pub fn materials(&self) -> &Texture { &self.field.as_ref().expect("published field").atlas.material }
    pub fn rows(&self) -> usize { self.field.as_ref().map_or(0, |f| f.atlas.rows) }
    pub fn index_buffer(&self) -> &GpuBuffer { &self.field.as_ref().expect("published field").index.gpu }
    pub fn program_buffer(&self) -> &GpuBuffer { &self.field.as_ref().expect("published field").programs.gpu }
    pub fn mesh_buffers(&self) -> Rc<[(LiveId, Rc<crate::mesh::MeshGpu>)]> {
        self.field.as_ref().expect("published field").meshes.clone()
    }
    pub fn published(&self) -> u64 { self.published }
    pub fn queued(&self) -> (usize, usize, usize) {
        (self.next.as_ref().map_or(0, |n| n.count - n.first), 0, usize::from(!self.complete.load(Ordering::Acquire)))
    }
    pub fn publishing(&self) -> bool { self.next.is_some() || !self.complete.load(Ordering::Acquire) }
    pub fn idle(&self) -> bool { !self.publishing() }
    pub fn take_recorded(&mut self) -> usize { std::mem::take(&mut self.recorded) }

    /// Record one frame's slice. Publication and the final dispatch are in the
    /// same frame; GPU submission order makes the new slots visible to its march.
    pub fn draw(&mut self, cx: &mut Cx2d, vol: &Volume) -> bool {
        match self.record(cx.cx, vol) {
            Ok(published) => published,
            Err(error) => { log!("sdf bake: {error}"); false }
        }
    }

    fn record(&mut self, cx: &mut Cx, vol: &Volume) -> Result<bool, String> {
        let Some(next) = self.next.as_mut() else { return Ok(false) };
        let budget = self.budget.min(self.limits.bricks_for_kernels(self.kernel_cost)).max(1);
        let count = budget.min(next.count - next.first);
        let kernel = &self.kernels[next.field.kernel].compute;
        let mut pass = ComputePass::new("sdf bake", self.parent);
        let buffers: Vec<_> = kernel.layout().buffers.iter().map(|b| {
            if b.name == id!(program) { next.field.programs.gpu.clone() }
            else if b.name == id!(bricks) { self.bricks.as_ref().unwrap().gpu.clone() }
            else { next.field.meshes.iter().find(|(name, _)| *name == b.name).expect("mesh binding").1.buffer.clone() }
        }).collect();
        let textures: Vec<_> = kernel.layout().storage_textures.iter().map(|t| {
            if t.name == id!(atlas) { next.field.atlas.distance.clone() }
            else { next.field.atlas.material.clone() }
        }).collect();
        let mut first = next.first;
        while first < next.first + count {
            let n = (next.first + count - first).min(65535);
            let mut params = kernel.params();
            params.set_f32(id!(cell), vol.cell)?.set_f32(id!(band), vol.band)?
                .set_u32(id!(first), first as u32)?.set_u32(id!(count), n as u32)?;
            pass.dispatch(kernel, [n as u32, 1, 1], &params, &buffers, &textures)?;
            first += n;
        }
        if count != 0 { pass.submit(cx); }
        next.first += count;
        self.recorded += count;
        self.baked += count;
        if next.first != next.count { return Ok(false); }
        next.field.index.upload(cx, std::mem::take(&mut next.index))?;
        // The callback follows all pending compute commands, including ones held
        // for compilation. Do not overwrite this bake list or write recycled slots
        // in a subsequent version until those commands and earlier draws finish.
        self.complete.store(false, Ordering::Release);
        let complete = self.complete.clone();
        cx.read_gpu_buffer(&self.bricks.as_ref().unwrap().gpu, 0, 4, move |result| {
            if let Err(error) = result { log!("sdf bake completion: {error}; keeping subsequent versions queued"); return; }
            complete.store(true, Ordering::Release);
            SignalToUI::set_ui_signal();
        }).map_err(|e| e.to_string())?;
        self.field = Some(self.next.take().unwrap().field);
        self.published += 1;
        Ok(true)
    }

    pub fn march<'a>(&'a mut self, cx: &mut Cx, base: &'a mut crate::march::DrawSdfMarch) -> &'a mut crate::march::DrawSdfMarch {
        match self.field.as_ref().and_then(|f| self.kernels.get_mut(f.kernel)) {
            Some(k) => k.march(cx, base).unwrap_or(base),
            None => base,
        }
    }

    pub fn vram_bytes(&self) -> u64 {
        let bytes = |f: &Field| (ATLAS_W * f.atlas.rows * 3) as u64 + f.index.gpu.size() + f.programs.gpu.size()
            + f.meshes.iter().map(|(_, m)| m.buffer.size()).sum::<u64>();
        let mut total = self.field.as_ref().map_or(0, bytes);
        if let Some(n) = &self.next {
            let old = self.field.as_ref();
            if old.is_none_or(|f| f.atlas.distance.texture_id() != n.field.atlas.distance.texture_id()) { total += (ATLAS_W * n.field.atlas.rows * 3) as u64; }
            if old.is_none_or(|f| f.index.gpu != n.field.index.gpu) { total += n.field.index.gpu.size(); }
            if old.is_none_or(|f| f.programs.gpu != n.field.programs.gpu) { total += n.field.programs.gpu.size(); }
            for (_, mesh) in n.field.meshes.iter() {
                if old.is_none_or(|f| !f.meshes.iter().any(|(_, m)| Rc::ptr_eq(m, mesh))) { total += mesh.buffer.size(); }
            }
        }
        total + self.bricks.as_ref().map_or(0, |b| b.gpu.size())
    }

    pub fn release(&mut self, cx: &mut Cx) {
        for k in self.kernels.drain(..) { k.release(cx); }
    }
}
