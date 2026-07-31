use crate::*;
use makepad_script::*;

pub mod cx;
pub mod draw;
pub mod event;
pub mod res;
pub mod script;
pub mod std;
pub mod timer;
pub mod vm;
pub use self::std::{fs, net, run};

/// The full platform surface. Trusted source only — see [`script_mod_sandboxed`].
pub fn script_mod(vm: &mut ScriptVm) {
    crate::script::cx::script_mod(vm);
    makepad_script_std::script_mod(vm);
    crate::script::timer::script_mod(vm);
    crate::script::res::script_mod(vm);
    crate::script::draw::script_mod(vm);
    crate::script::event::script_mod(vm);
}

/// The platform surface a GENERATED card may hold.
///
/// Identical to [`script_mod`] except that `cx.quit` and the `fs`, `run` and
/// `net` modules are never registered. `timer`, `res`, `draw` and `event` stay —
/// they are what rendering needs, and none of them reaches the filesystem,
/// a process or the network. In particular `res` supplies `crate_resource` and
/// `http_resource`, whose fetching happens on the Rust side rather than through
/// the script `net` module.
pub fn script_mod_sandboxed(vm: &mut ScriptVm) {
    crate::script::cx::script_mod_sandboxed(vm);
    makepad_script_std::script_mod_sandboxed(vm);
    crate::script::timer::script_mod(vm);
    crate::script::res::script_mod(vm);
    crate::script::draw::script_mod(vm);
    crate::script::event::script_mod(vm);
}
