//! Exact mesh leaves. The immutable query buffer is shared by geometry key.
use makepad_csg_sdf::mesh_field::{closest_on_tri, tri_winding, MeshField};
use makepad_draw::*;
use makepad_draw::makepad_platform::gpu_admission::{GpuDocument, GpuReservation};
use std::{collections::HashMap, rc::{Rc, Weak}, sync::Arc};

#[derive(Clone)]
pub struct MeshAsset {
    pub key: u64,
    pub words: Arc<[[u32; 4]]>,
}

impl MeshAsset {
    pub fn new(field: &MeshField) -> Self {
        Self { key: field.key(), words: field.query_words().into() }
    }

    /// Conservative query cost for the existing per-frame bake steering: both
    /// full traversals and every triangle's distance and solid angle. Pruning
    /// usually makes the measured cost much lower; this protects the first bake.
    pub fn cost(&self) -> u64 {
        let h = self.words[0];
        (h[3] - h[2]) as u64 * 512 + h[0] as u64 * 128
    }

    /// The same bounded traversal and arithmetic as mod.sdf.Eval.mesh_at.
    pub fn signed(&self, p: [f32; 3]) -> f32 {
        let words = &self.words;
        let header = words[0];
        if header[0] == 0 { return f32::INFINITY; }
        let xyz = |at: u32| { let w = words[at as usize]; [f32::from_bits(w[0]), f32::from_bits(w[1]), f32::from_bits(w[2])] };
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let dot = |a: [f32; 3], b: [f32; 3]| (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
        let len = |a| f32::sqrt(dot(a, a));
        let box_dist = |node: u32| {
            let lo = xyz(1 + node * 4);
            let hi = xyz(2 + node * 4);
            let mut d = 0.0;
            for k in 0..3 { let e = (lo[k] - p[k]).max(0.0).max(p[k] - hi[k]); d += e * e; }
            d
        };
        let tri = |i: u32| {
            let t = words[(header[2] + words[(header[3] + i) as usize][0]) as usize];
            [xyz(header[1] + t[0]), xyz(header[1] + t[1]), xyz(header[1] + t[2])]
        };
        let mut stack = [0u32; 32];
        let mut count = 1usize;
        let mut best = f32::INFINITY;
        for _ in 0..header[0] {
            if count == 0 { break; }
            count -= 1;
            let node = stack[count];
            if box_dist(node) >= best * best { continue; }
            let at = (1 + node * 4) as usize;
            let (left, right) = (words[at][3], words[at + 1][3]);
            if words[at + 2][3] != 0 {
                for i in left..right {
                    let [a, b, c] = tri(i);
                    let d = len(sub(p, closest_on_tri(p, a, b, c)));
                    if d < best { best = d; }
                }
            } else {
                let near_left = box_dist(left) < box_dist(right);
                stack[count] = if near_left { right } else { left };
                stack[count + 1] = if near_left { left } else { right };
                count += 2;
            }
        }
        stack[0] = 0;
        count = 1;
        let mut winding = 0.0;
        for _ in 0..header[0] {
            if count == 0 { break; }
            count -= 1;
            let at = 1 + stack[count] * 4;
            let d = sub(xyz(at + 3), p);
            let dl = len(d);
            let radius = f32::from_bits(words[(at + 3) as usize][3]);
            if dl > 2.0 * radius && dl > 0.0 {
                winding += dot(xyz(at + 2), d) / (4.0 * std::f32::consts::PI * dl * dl * dl);
                continue;
            }
            let (left, right) = (words[at as usize][3], words[(at + 1) as usize][3]);
            if words[(at + 2) as usize][3] != 0 {
                for i in left..right {
                    let [a, b, c] = tri(i);
                    winding += tri_winding(p, a, b, c);
                }
            } else {
                stack[count] = left;
                stack[count + 1] = right;
                count += 2;
            }
        }
        if winding > 0.5 { -best } else { best }
    }
}

pub struct MeshGpu {
    pub buffer: GpuBuffer,
    /// The device generation it was uploaded on (`Cx::gpu_device_generation`): after a
    /// device reset its contents are gone, and a holder uploads again.
    pub generation: u64,
    _reservation: GpuReservation,
}
impl MeshGpu {
    pub fn is_current(&self, cx: &Cx) -> bool { self.generation == cx.gpu_device_generation() }
}

/// Keyed by device generation and geometry: an upload made before a device reset is never
/// handed out after it, whoever saw the reset.
#[derive(Default)]
struct MeshCache(HashMap<(u64, u64), Weak<MeshGpu>>);

pub fn binding(slot: usize) -> LiveId { LiveId::from_str(&format!("mesh_{slot}")) }

pub fn upload(cx: &mut Cx, document: &GpuDocument, asset: &MeshAsset) -> Result<Rc<MeshGpu>, String> {
    let generation = cx.gpu_device_generation();
    if let Some(gpu) = cx.global::<MeshCache>().0.get(&(generation, asset.key)).and_then(Weak::upgrade) { return Ok(gpu); }
    cx.global::<MeshCache>().0.retain(|&(g, _), gpu| g == generation && gpu.strong_count() != 0);
    let size = (asset.words.len() * 16) as u64;
    let reservation = document.admit(size).map_err(|e| e.to_string())?;
    let buffer = GpuBuffer::new(cx, size, GpuBufferFlags::default())?;
    let bytes: Vec<_> = asset.words.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    buffer.update(cx, 0, &bytes)?;
    let gpu = Rc::new(MeshGpu { buffer, generation, _reservation: reservation });
    cx.global::<MeshCache>().0.insert((generation, asset.key), Rc::downgrade(&gpu));
    Ok(gpu)
}
