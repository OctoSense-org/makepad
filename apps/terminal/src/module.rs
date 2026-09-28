//! The terminal as a module: the app a host seats in a tile in-process, in
//! an isolate of its own (the OctoSense shell links it as a system app).
//!
//! `register` adds the terminal's widget family to the isolate the host
//! prepared; `create` mints one `TermTabs{}` root there: the terminal with
//! its own tabs and settings panel (`crate::tabs`), so a tile holds several
//! shells without the window manager's help. Unlike a pure-UI
//! module, each instance starts a login shell in a PTY and a thread that
//! reads it (`session`), which is why it declares `process`. On macOS the
//! PTY helper is the host executable itself (`pty_spawn::screen_helper`): an
//! `app_main!` host answers `--exec-pty` in `Cx::pre_start`.
//!
//! The assistant gets the read tools only ([`crate::ai::read_only_manifest`]):
//! `run` types into a live, unsandboxed shell, and a host that ships the
//! terminal by default does not hand that to it.

use crate::tabs::TermTabs;
use crate::widget::MpTerm;
use makepad_ai_services::wire::{ServiceCall, ServiceManifest, ToolResult};
use makepad_app_module::*;
use makepad_widgets::*;

pub struct TerminalModule;

/// The one linked instance of the module description: immutable, no state.
pub static TERMINAL_MODULE: TerminalModule = TerminalModule;

impl AppModule for TerminalModule {
    fn id(&self) -> &'static str {
        "terminal"
    }

    fn label(&self) -> &'static str {
        "Terminal"
    }

    fn register(&self, vm: &mut ScriptVm) {
        crate::widget::script_mod(vm);
        crate::tabs::script_mod(vm);
    }

    fn open_schema(&self) -> OpenSchema {
        OpenSchema::new(1)
    }

    fn create(&self, vm: &mut ScriptVm, _open: ValidatedOpen, _handles: InstanceHandles) -> InstanceParts {
        let value = script_eval!(vm, {
            use mod.widgets.*
            TermTabs {}
        });
        let root = WidgetRef::script_from_value(vm, value);
        InstanceParts {
            root: root.clone(),
            executor: Box::new(TerminalExecutor { root }),
            // The session (PTY, shell, reader thread) is owned by the root
            // widget and ends when the host drops it.
            shutdown: Box::new(|_vm| {}),
        }
    }

    fn capabilities(&self) -> &'static [&'static str] {
        &["process", "clipboard"]
    }
}

/// The instance's read tools, answered from the live emulator at call time.
struct TerminalExecutor {
    root: WidgetRef,
}

impl ServiceExecutor for TerminalExecutor {
    fn manifest(&self) -> ServiceManifest {
        crate::ai::read_only_manifest()
    }

    fn execute(&mut self, cx: &mut Cx, call: &ServiceCall) -> ExecOutcome {
        // The assistant reads the selected tab.
        let term = self.root.borrow_mut::<TermTabs>().map(|mut tabs| tabs.active_term(cx));
        let result = term
            .as_ref()
            .and_then(|term| term.borrow_mut::<MpTerm>())
            .map(|mut term| crate::ai::answer_read_only(call, &mut *term))
            .unwrap_or_else(|| ToolResult::unavailable(&call.call_id, "the terminal is not open"));
        ExecOutcome::Done(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{ScreenState, TerminalTarget};
    use makepad_ai_services::wire::ToolOutcome;

    fn call(id: &str, tool: &str, args: &str) -> ServiceCall {
        ServiceCall { call_id: id.into(), tool: tool.into(), args: args.into() }
    }

    #[test]
    fn the_module_describes_itself_and_opens_empty() {
        let m = &TERMINAL_MODULE;
        assert_eq!(m.id(), "terminal");
        assert_eq!(m.label(), "Terminal");
        assert!(m.capabilities().contains(&"process"), "it starts a shell");
        assert!(
            !m.capabilities().iter().any(|c| c.starts_with("octos.")),
            "it asks for no assistant service of its own"
        );
        let schema = m.open_schema();
        assert_eq!(schema.version, 1);
        assert!(schema.empty_open().is_ok(), "no argument is required");
    }

    #[test]
    fn the_module_offers_reads_and_never_run() {
        let manifest = crate::ai::read_only_manifest();
        let names: Vec<&str> = manifest.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["read_screen", "read_scrollback"]);
        let full = crate::ai::manifest();
        assert!(full.tools.iter().any(|t| t.name == "run"), "the standalone app keeps run");
    }

    struct Fake {
        typed: Vec<u8>,
    }

    impl TerminalTarget for Fake {
        fn visible_screen(&self) -> Option<ScreenState> {
            Some(ScreenState { rows: vec!["$ ".into()], cursor_row: 0, cursor_col: 2, cwd: None })
        }
        fn recent_screen(&self, _lines: usize) -> Option<ScreenState> {
            self.visible_screen()
        }
        fn type_bytes(&mut self, bytes: &[u8]) -> bool {
            self.typed.extend_from_slice(bytes);
            true
        }
    }

    #[test]
    fn a_run_call_is_refused_and_types_nothing() {
        let mut target = Fake { typed: Vec::new() };
        let result = crate::ai::answer_read_only(&call("c1", "run", r#"{"command":"rm -rf ~"}"#), &mut target);
        assert!(target.typed.is_empty(), "nothing reached the shell");
        assert!(matches!(result.outcome, ToolOutcome::Refused), "run is refused");
        let read = crate::ai::answer_read_only(&call("c2", "read_screen", "{}"), &mut target);
        assert!(matches!(read.outcome, ToolOutcome::Ok), "reads still work");
    }
}
