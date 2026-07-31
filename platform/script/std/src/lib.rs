pub use makepad_network;
pub use makepad_script;

pub mod data;
pub mod fs;
pub mod net;
pub mod run;
pub mod task;
pub mod vm;

pub use data::*;
pub use net::*;
pub use run::*;
pub use task::*;
pub use vm::*;

use makepad_script::*;
use std::any::Any;

/// The full std surface: filesystem, subprocesses, async tasks and networking.
///
/// Only for VMs running TRUSTED source — the app's own scripts. A VM that
/// evaluates generated cards must use [`script_mod_sandboxed`] instead.
pub fn script_mod(vm: &mut ScriptVm) {
    crate::fs::script_mod(vm);
    crate::run::script_mod(vm);
    crate::task::script_mod(vm);
    crate::net::script_mod(vm);
}

/// The std surface a GENERATED card may hold: async plumbing, nothing else.
///
/// `fs`, `run` and `net` are not registered, so a card cannot read or write a
/// file, spawn a process, open a socket, or start a server — not because a
/// scanner rejects those calls, but because the names do not resolve. That
/// distinction matters: the previous guard was a five-string denylist over
/// generated source (`app/app/src/main.rs`), and its own comment conceded it
/// was "NOT a hard boundary … the real fix is VM-level capability gating".
///
/// `task` stays: it is async plumbing rather than a capability, and dropping it
/// would break `await` in framework code for no security gain.
///
/// Verified before removal: no `.splash` in the repository — including the
/// framework's own `makepad.splash` — references `fs.`, `run.` or `net.`.
pub fn script_mod_sandboxed(vm: &mut ScriptVm) {
    crate::task::script_mod(vm);
}

#[cfg(test)]
mod sandbox_tests {
    use super::*;

    /// Which of the capability-bearing modules a registration function installs.
    fn modules_installed(register: fn(&mut ScriptVm)) -> Vec<&'static str> {
        let mut host = ();
        let mut std = ();
        let mut vm = ScriptVm {
            host: &mut host,
            std: &mut std,
            bx: Box::new(ScriptVmBase::new()),
        };
        register(&mut vm);
        [("fs", id!(fs)), ("run", id!(run)), ("net", id!(net))]
            .into_iter()
            .filter(|(_, id)| vm.bx.heap.module(*id) != ScriptObject::ZERO)
            .map(|(name, _)| name)
            .collect()
    }

    /// The property the whole sandbox rests on: a generated card cannot name
    /// these, because they are never installed. Not a denylist — the lookup
    /// fails.
    #[test]
    fn sandboxed_std_installs_no_capability_modules() {
        assert_eq!(modules_installed(script_mod_sandboxed), Vec::<&str>::new());
    }

    /// The trusted surface is unchanged, so the app's own scripts keep working.
    #[test]
    fn full_std_still_installs_all_of_them() {
        assert_eq!(modules_installed(script_mod), vec!["fs", "run", "net"]);
    }
}

pub fn pump<H: Any>(host: &mut H, std: &mut ScriptStd, script_vm: &mut Option<Box<ScriptVmBase>>) {
    crate::run::handle_script_child_processes(host, std, script_vm);
    crate::net::handle_script_socket_streams(host, std, script_vm);
    crate::net::handle_script_http_servers(host, std, script_vm);
    crate::task::handle_script_tasks(host, std, script_vm);
}

pub fn pump_network_runtime<H: Any>(
    host: &mut H,
    std: &mut ScriptStd,
    script_vm: &mut Option<Box<ScriptVmBase>>,
) -> Vec<makepad_network::NetworkResponse> {
    let responses = crate::net::drain_network_runtime(std);
    if !responses.is_empty() {
        crate::net::handle_script_network_events(host, std, script_vm, &responses);
        crate::task::handle_script_tasks(host, std, script_vm);
    }
    responses
}
