use crate::heap::*;
use crate::makepad_live_id::live_id::*;
use crate::makepad_live_id_macros::*;
use crate::native::*;
use crate::shader_builtins::*;
use crate::value::*;
use crate::*;

pub fn define_math_module(heap: &mut ScriptHeap, native: &mut ScriptNative) {
    let math = heap.new_module(id!(math));
    define_shader_builtins(heap, math, native);
    // A colour's channels, 0..1 as written (sRGB for a `#hex`): `#ff8000.g`
    // is 0.50196…, as `.r .g .b .a` read a colour in shaders.
    native.set_type_getter(ScriptValueType::REDUX_COLOR, |vm, value, field| {
        let Some(c) = value.as_color() else { return NIL };
        let shift = match field {
            f if f == id!(r) => 24,
            f if f == id!(g) => 16,
            f if f == id!(b) => 8,
            f if f == id!(a) => 0,
            _ => return script_err_not_found!(vm.bx.threads.cur_ref().trap, "a colour has the fields r, g, b and a, not {:?}", field),
        };
        ScriptValue::from_f64(((c >> shift) & 0xff) as f64 / 255.0)
    });
}
