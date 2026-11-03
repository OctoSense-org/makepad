//! Compile the production objects, including the AIR splice, with native validators.
use makepad_draw::*;
use makepad_draw::makepad_script::{shader_backend::ShaderBackend, shader_ir_compute::*, shader_ir_draw, shader_ir_hlsl, shader_ir_metal};
use makepad_sdf_engine::{kernel::{ComputeBake, KernelSlot}, makepad_csg_sdf::{SdfGrid, SdfKernel, SdfSplashExpr, SplashPoint}};

#[cfg(target_os = "macos")]
#[path = "../../../platform/script/tests/support/native_shader.rs"]
mod native_shader;
#[path = "support/meshes.rs"]
mod meshes;

// FXC's optimizer takes minutes on the bounded mesh traversal. Production HAL
// skips that pass too (d3d11/pipeline.rs::compile_stage). Keep strict validation
// and IEEE math; this test validates the native compiler, not its optimizer.
#[cfg(windows)]
fn hlsl(source: &str, entry: &str, target: &str, precise: bool) -> Result<(), String> {
    use std::ffi::{c_char, c_void, CString};
    #[repr(C)]
    struct Blob { vtable: *const Vtable }
    #[repr(C)]
    struct Vtable {
        query: usize, add_ref: usize,
        release: unsafe extern "system" fn(*mut Blob) -> u32,
        data: unsafe extern "system" fn(*mut Blob) -> *const u8,
        size: unsafe extern "system" fn(*mut Blob) -> usize,
    }
    #[link(name = "d3dcompiler")]
    extern "system" {
        fn D3DCompile(src: *const u8, size: usize, name: *const c_char, defines: *const c_void,
            include: *const c_void, entry: *const c_char, target: *const c_char, flags: u32,
            flags2: u32, code: *mut *mut Blob, errors: *mut *mut Blob) -> i32;
    }
    let entry = CString::new(entry).unwrap(); let target = CString::new(target).unwrap();
    unsafe {
        let mut code = std::ptr::null_mut(); let mut errors = std::ptr::null_mut();
        let flags = (1 << 2) | (1 << 11) | if precise { 1 << 13 } else { 0 };
        let hr = D3DCompile(source.as_ptr(), source.len(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
            entry.as_ptr(), target.as_ptr(), flags, 0, &mut code, &mut errors);
        let diagnostic = if errors.is_null() { format!("D3DCompile HRESULT {hr:#x}") } else {
            let vt = &*(*errors).vtable;
            String::from_utf8_lossy(std::slice::from_raw_parts((vt.data)(errors), (vt.size)(errors))).into_owned()
        };
        for blob in [code, errors] { if !blob.is_null() { ((*(*blob).vtable).release)(blob); } }
        if hr < 0 { Err(diagnostic) } else { Ok(()) }
    }
}

fn spirv(words: &[u32]) {
    #[cfg(target_os = "linux")]
    {
        static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let file = std::env::temp_dir().join(format!("sdf-shader-{}-{serial}.spv", std::process::id()));
        let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&file, bytes).unwrap();
        let out = std::process::Command::new("spirv-val").args(["--target-env", "vulkan1.2"]).arg(&file).output().expect("spirv-val required");
        std::fs::remove_file(file).unwrap();
        assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    }
    #[cfg(not(target_os = "linux"))]
    assert!(!words.is_empty());
}

fn validate_compute(k: &CompiledKernel, routed: bool, meshes: usize) {
    assert!(k.layout.precise_math);
    assert_eq!(k.layout.workgroup, [64, 1, 1]);
    // the bricks and the programs it reads, the slopes it writes, a buffer per mesh
    assert_eq!(k.layout.buffers.len(), 3 + meshes);
    assert_eq!(k.layout.storage_textures.len(), 2);
    assert!(k.layout.buffers.iter().all(|b| b.writable == (b.name == id!(slopes)) && b.stride == 16));
    for name in [id!(cell), id!(band), id!(first), id!(count)] {
        assert!(k.layout.slots.iter().any(|s| s.name == name));
    }
    match &k.code {
        KernelCode::Spirv(words) => spirv(words),
        KernelCode::Msl(source) => {
            if routed { assert!(source.contains("sdfk_"), "compute skipped AIR hook"); }
            #[cfg(target_os = "macos")]
            native_shader::metal(source, true).unwrap();
        }
        KernelCode::Hlsl(source) => {
            if routed { assert!(source.contains("sdfk_"), "compute skipped AIR hook"); }
            #[cfg(target_os = "windows")]
            hlsl(source, "kernel_main", "cs_5_0", true).unwrap();
        }
    }
}

fn validate_draw(vm: &mut ScriptVm, object: ScriptObject, name: &str, backend: ShaderBackend, vulkan: bool, routed: bool, meshes: usize) {
    let opts = shader_ir_draw::IrCompileOptions::new(backend, false, vulkan, false, false);
    let out = shader_ir_draw::compile_draw_shader_ir_with(vm, object, &opts);
    assert!(!out.has_errors, "{name}: {:?}", out.errors);
    assert!(vm.take_errors().is_empty(), "{name}: VM errors");
    assert!(out.io.iter().all(|io| io.name != id!(prog_tex) && io.name != id!(index_tex)), "{name}: obsolete texture binding");
    // the index, the programs, the slopes, a buffer per mesh
    if name == "March" { assert_eq!(out.storage_buffers.len(), 3 + meshes); }
    else if name != "Splat" { assert!(out.storage_buffers.iter().any(|b| b.name == id!(index))); }
    if vulkan {
        for module in shader_ir_draw::draw_spirv(&out, vm).unwrap() {
            spirv(&module.vertex);
            spirv(&module.fragment);
        }
    } else if backend == ShaderBackend::Metal {
        let (source, _) = shader_ir_metal::metal_draw(&out, vm, true);
        if routed && name == "March" { assert!(source.contains("sdfk_")); }
        #[cfg(target_os = "macos")]
        native_shader::metal(&source, name == "March").unwrap();
    } else {
        let (source, _) = shader_ir_hlsl::hlsl_draw(&out, vm, true);
        if routed && name == "March" { assert!(source.contains("sdfk_")); }
        #[cfg(target_os = "windows")]
        for (entry, profile) in [("vertex_main", "vs_5_0"), ("pixel_main", "ps_5_0")] {
            hlsl(&source, entry, profile, name == "March").unwrap();
        }
    }
}

#[test]
fn bake_and_draw_native_compilers() {
    let mut cx = Cx::new(Box::new(|_, _| {}));
    cx.with_vm(|vm| {
        vm.capture_errors();
        makepad_draw::script_mod(vm);
        // render-material imports the widget prelude's math/pod exports.
        // Supply those real modules without booting a widget application.
        vm.eval(ScriptMod::new("", "", "shader_prelude", 0, 0,
            "mod.prelude = { widgets_internal: { ..mod.pod, ..mod.math, ..mod.shader } }\nmod.prelude", &[]));
        makepad_sdf_engine::script_mod(vm);
        let errors = vm.take_errors();
        assert!(errors.is_empty(), "registration: {errors:?}");
        // The m.sdf knob field in fab/model/src/tests.rs (the handle document).
        // Compile the document's actual field through the same SdfSplashExpr path;
        // the model/document host is not needed to build a field's shader route.
        let field = vm.eval(ScriptMod::new("", "", "knob_field", 0, 0,
            "use mod.math.*\nlet field = fn(p) { return smooth_union(sdf_sphere(p - vec3(0, 12, 0), 8), sdf_cylinder(p - vec3(0, 6, 0), 4, 12), 3) }\n(field)", &[])).as_object().unwrap();
        let _keep = vm.new_object_ref(field);
        let field = SdfSplashExpr::compile(vm, field, SplashPoint::Vec3, &[]).unwrap();
        let slot = KernelSlot::new(&SdfKernel::splash(field, None)).unwrap();
        assert!(slot.air().is_some(), "fixture must have a GPU route");
        // a field reading a grid: its words come from the program buffer
        let grid_fn = vm.eval(ScriptMod::new("", "", "grid_field", 0, 0, "use mod.math.*\nlet field = fn(p) { return sdf_grid3(p) - 0.5 }\n(field)", &[])).as_object().unwrap();
        let _keep_grid = vm.new_object_ref(grid_fn);
        let grid = SdfGrid::new([0.0; 3], 1.0, [2, 2, 2], 0.0, vec![0.0f32; 8].into());
        let gridded = KernelSlot::new(&SdfKernel::splash(SdfSplashExpr::compile_gridded(vm, grid_fn, SplashPoint::Vec3, &[], grid).unwrap(), None)).unwrap();
        assert!(gridded.air().is_some(), "a gridded field has a GPU route");
        let module = vm.heap().module(id!(draw));
        let corpus = meshes::corpus();
        let mesh = KernelSlot::mesh(&std::sync::Arc::new(corpus[0].1.clone()));
        let cup = KernelSlot::mesh(&std::sync::Arc::new(corpus[1].1.clone()));
        for slots in [Vec::new(), vec![slot.clone()], vec![mesh.clone()], vec![gridded], vec![slot, mesh, cup]] {
            let routed = slots.iter().any(|s| s.air().is_some());
            let meshes = slots.iter().filter(|s| s.is_mesh()).count();
            let mut bake = ComputeBake::new(vm, &slots).unwrap();
            for (target, backend, vulkan) in [
                (ComputeTarget::Spirv, ShaderBackend::Glsl, true),
                (ComputeTarget::Msl, ShaderBackend::Metal, false),
                (ComputeTarget::Hlsl, ShaderBackend::Hlsl, false),
            ] {
                let k = bake.compile(vm, target).unwrap_or_else(|e| panic!("Bake {target:?} routed={routed}: {e}"));
                validate_compute(&k, routed, meshes);
                println!("Bake {target:?} routed={routed} validated");
                for name in ["TraceCoarse", "TraceCell"] {
                    let trace_module = vm.heap().module(id!(sdf));
                    let base = vm.value(trace_module, LiveId::from_str(name)).as_object().unwrap();
                    let object = bake.draw_variant(vm, base).unwrap();
                    let trace = compile_compute_kernel(vm, object, target)
                        .unwrap_or_else(|e| panic!("{name} {target:?} routed={routed}: {e}"));
                    assert!(trace.layout.precise_math);
                    assert!(trace.layout.push_size <= 128);
                    assert!(trace.layout.buffers.iter().filter(|b| b.writable).count() <= 8);
                    for (i, _) in slots.iter().enumerate().filter(|(_, s)| s.is_mesh()) {
                        assert!(trace.layout.buffers.iter().any(|b| b.name == makepad_sdf_engine::mesh::binding(i)), "trace mesh binding {i}");
                    }
                    match &trace.code {
                        KernelCode::Spirv(words) => spirv(words),
                        KernelCode::Msl(_source) => {
                            #[cfg(target_os = "macos")]
                            native_shader::metal(_source, true).unwrap();
                        }
                        KernelCode::Hlsl(_source) => {
                            #[cfg(target_os = "windows")]
                            hlsl(_source, "kernel_main", "cs_5_0", true).unwrap();
                        }
                    }
                    println!("{name} {target:?} routed={routed} validated");
                }
                for name in ["March", "Coarse", "Shade", "CastShadow", "Splat"] {
                    let base = vm.value(module, LiveId::from_str(&format!("Sdf{name}"))).as_object().unwrap();
                    let obj = if name == "March" { bake.draw_variant(vm, base).unwrap() } else { base };
                    validate_draw(vm, obj, name, backend, vulkan, routed, meshes);
                    println!("{name} {target:?} routed={routed} validated");
                }
            }
            bake.release(vm);
        }
    });
}
