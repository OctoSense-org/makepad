//! Test-only: an item's buffers copied into one readback, so a test reads many outputs through
//! the runtime's one Output.
use makepad_sdf_engine::gpu_jobs::{label, BufRef, Elements, Output, Pass, Pool, Step};
use makepad_sdf_engine::makepad_draw::*;

script_mod! {
    use mod.pod.*
    use mod.math.*
    use mod.shader

    // dst[at + i] = src[i] for i in dispatch_range (words.x: at).
    mod.sdf.TestCopy = {
        workgroup_size: 64
        dispatch_range: shader.uniform(vec4u)
        words: shader.uniform(vec4u)
        src: shader.buffer(u32)
        dst: shader.buffer_rw(u32)
        main: fn() {
            if global_id().x >= self.dispatch_range.y { return }
            let i = self.dispatch_range.x + global_id().x
            self.dst[self.words.x + i] = self.src[i]
        }
    }
    mod.sdf
}

pub fn kernel(cx: &mut Cx) -> ComputeKernel {
    cx.with_vm(|vm| {
        script_mod(vm);
        let module = vm.heap().module(id!(sdf));
        let object = vm.value(module, id!(TestCopy)).as_object().unwrap();
        ComputeKernel::new(vm, object, "test copy").unwrap()
    })
}

/// Where each packed buffer starts, in words.
pub struct Packed { pub at: Vec<(&'static str, usize, usize)> }
impl Packed {
    pub fn get<'a>(&self, words: &'a [u32], name: &str) -> &'a [u32] {
        let &(_, at, n) = self.at.iter().find(|p| p.0 == name).unwrap_or_else(|| panic!("{name} not packed"));
        &words[at..at + n]
    }
}

/// Copies of `bufs` (each its whole pool size) into one readback buffer, then its Output.
pub fn steps(copy: &ComputeKernel, pool: &mut Pool, bufs: &[(&'static str, BufRef)]) -> (Vec<Step>, Packed) {
    let sizes: Vec<usize> = bufs.iter().map(|b| pool.bytes(b.1) as usize / 4).collect();
    let total: usize = sizes.iter().sum();
    let pack = pool.readback("test pack", total as u64 * 4);
    let (mut steps, mut at, mut packed) = (Vec::new(), 0usize, Vec::new());
    for (&(name, buf), &n) in bufs.iter().zip(&sizes) {
        let pass = Pass::new(copy, label("test copy"), Elements::Fixed(n as u32), 2, 1 << 20, |b| if b == id!(src) { buf } else { pack })
            .param_u32x4(id!(words), [at as u32, 0, 0, 0]).unwrap();
        steps.push(Step::Pass(pass));
        packed.push((name, at, n));
        at += n;
    }
    steps.push(Output::fixed(pack, total as u64 * 4));
    (steps, Packed { at: packed })
}
