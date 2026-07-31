use crate::cx::OsType;
use crate::script::vm::*;
use crate::*;
use makepad_script::*;

/// The full `cx` module, including `quit`. Trusted source only.
pub fn script_mod(vm: &mut ScriptVm) {
    let cx = script_mod_sandboxed_inner(vm);

    // Terminating the process is not something a generated card may do, so this
    // is the one method the sandboxed surface withholds.
    vm.add_method(cx, id_lut!(quit), script_args_def!(), |vm, _args| {
        vm.cx_mut().request_quit(QuitReason::App);
        NIL
    });
}

/// The `cx` module a GENERATED card may hold: `os_type` for layout decisions,
/// without `quit`.
pub fn script_mod_sandboxed(vm: &mut ScriptVm) {
    script_mod_sandboxed_inner(vm);
}

fn script_mod_sandboxed_inner(vm: &mut ScriptVm) -> ScriptObject {
    let cx = vm.new_module(id_lut!(cx));

    set_script_value_to_api!(vm, cx.OsType);

    vm.add_method(cx, id_lut!(os_type), script_args_def!(), |vm, _args| {
        let os_type = vm.cx().os_type().clone();
        os_type.script_to_value(vm)
    });

    cx
}
