//! Splash DSL → makepad widgets, translated **at runtime, on device**.
//!
//! `splash_catalog` bakes the translation in at build time (the makepad script
//! is generated on a host and compiled in). This app instead ships the raw
//! `catalog.splash` source and does the whole pipeline live on the phone:
//!
//!   catalog.splash  ──(splash-render, shared makepad-script VM)──▶  UiNode
//!   UiNode          ──(splash-makepad)────────────────────────────▶  makepad UI string
//!   makepad UI str  ──(Splash widget: eval + build + mount)────────▶  live widgets
//!
//! The `[patch]` in the workspace root points the splash crates' makepad-script
//! at this repo's local one, so `splash-render` evaluates the DSL on the *same*
//! VM the widgets run on. The generated makepad-UI string is mounted by the
//! built-in `Splash` widget (widgets/src/splash.rs), which evals it and builds
//! real makepad widgets — the runtime equivalent of the compiled `body +:`.

pub use makepad_widgets;

use makepad_widgets::*;

app_main!(App);

/// The Splash DSL source, shipped as-is and evaluated on device.
const CATALOG: &str = include_str!("catalog.splash");

script_mod! {
    use mod.prelude.widgets.*

    startup() do #(App::script_component(vm)){
        ui: Root{
            main_window := Window{
                window.inner_size: vec2(440, 900)
                body +: {
                    // A Splash slot: its children are produced at runtime from
                    // catalog.splash and mounted here by App::mount_runtime_ui.
                    // isolate: false → evaluate on the app's main VM, so the
                    // mounted card inherits the light theme (this VM's switch)
                    // and shares one heap (the isolate's smaller heap crashes
                    // the TextInput animator).
                    host := Splash{
                        isolate: false
                        width: Fill
                        height: Fill
                    }
                }
            }
        }
    }
}

#[derive(Script, ScriptHook)]
pub struct App {
    #[live]
    ui: WidgetRef,
    // makepad renders on demand; on this OnePlus an idle frame is not presented
    // to the panel, so we drive continuous frames to keep the display fed.
    #[rust]
    next_frame: NextFrame,
    #[rust]
    mounted: bool,
}

impl App {
    /// Evaluate the Splash DSL, translate it to makepad's dialect, and mount the
    /// result as live widgets under `host` — all at runtime, on device.
    fn mount_runtime_ui(&mut self, cx: &mut Cx) {
        // ① evaluate the DSL on the shared VM → backend-agnostic UiNode tree
        let node = splash_render::build(CATALOG, |_vm| {}).expect("splash evaluates");
        // ② translate UiNode → makepad component-script UI string
        let generated = splash_makepad::to_makepad_ui(&node);
        // ③ hand the string to the Splash widget: it evals + builds + mounts it.
        //    isolate:false (set in script) makes it eval on this app's main VM,
        //    so the card inherits the light theme and one heap.
        self.ui.widget(cx, ids!(host)).set_text(cx, &generated);
    }
}

impl MatchEvent for App {}

impl AppMain for App {
    fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
        // Light theme for the main VM (see splash_catalog). Note: the Splash
        // widget evaluates the mounted UI in its own isolate VM, so this switch
        // governs only main-VM widgets, not the mounted card (addressed
        // separately if the isolate needs the light theme).
        crate::makepad_widgets::theme_mod(vm);
        script_eval!(vm, {
            mod.theme = mod.themes.light
        });
        crate::makepad_widgets::widgets_mod(vm);
        self::script_mod(vm)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event) {
        if matches!(event, Event::Startup) && !self.mounted {
            self.mounted = true;
            self.next_frame = cx.new_next_frame();
            self.mount_runtime_ui(cx);
        }
        if self.next_frame.is_event(event).is_some() {
            cx.redraw_all();
            self.next_frame = cx.new_next_frame();
        }
        self.match_event(cx, event);
        self.ui.handle_event(cx, event, &mut Scope::empty());
    }
}
