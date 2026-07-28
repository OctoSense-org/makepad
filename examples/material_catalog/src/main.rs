//! A Material-Design component catalog authored in the Splash DSL and rendered
//! as makepad **native** widgets, translated on-device at runtime (see
//! splash_runtime for the pipeline). Adds two things for building out a large
//! catalog:
//!
//!  * a scrolling host (`ScrollYView`) so a tall catalog scrolls, and
//!  * **on-device hot reload** — if `/data/local/tmp/material_catalog.splash`
//!    exists it is used instead of the baked-in catalog, and the app re-reads +
//!    re-mounts it whenever it changes. So iterating the catalog is just
//!    `adb push new.splash /data/local/tmp/material_catalog.splash` — no rebuild.

pub use makepad_widgets;

use makepad_widgets::*;

app_main!(App);

/// Baked-in fallback, used when the device file is absent.
const BAKED: &str = include_str!("catalog.splash");
/// Push here to hot-reload: `adb push x.splash /data/local/tmp/material_catalog.splash`.
const DEVICE_PATH: &str = "/data/local/tmp/material_catalog.splash";

script_mod! {
    use mod.prelude.widgets.*

    startup() do #(App::script_component(vm)){
        ui: Root{
            main_window := Window{
                window.inner_size: vec2(440, 1400)
                body +: {
                    // The light "surface" page, scrollable for tall catalogs.
                    ScrollYView{
                        width: Fill
                        height: Fill
                        flow: Down
                        show_bg: true
                        draw_bg +: { color: #fef7ffff }
                        // The catalog is produced at runtime and mounted here.
                        host := Splash{
                            isolate: false
                            width: Fill
                            height: Fit
                        }
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
    // Drive continuous frames (this OnePlus doesn't present idle frames) and use
    // them to poll the hot-reload file.
    #[rust]
    next_frame: NextFrame,
    #[rust]
    last_src: String,
    #[rust]
    tick: u32,
}

impl App {
    /// The current catalog source: the pushed device file if present, else baked.
    fn current_source() -> String {
        std::fs::read_to_string(DEVICE_PATH).unwrap_or_else(|_| BAKED.to_string())
    }

    /// Re-translate + re-mount only when the source actually changed. A malformed
    /// edit (build returns None) is ignored so the previous UI stays on screen.
    fn reload_if_changed(&mut self, cx: &mut Cx) {
        let src = Self::current_source();
        if src == self.last_src {
            return;
        }
        self.last_src = src.clone();
        if let Some(node) = splash_render::build(&src, |_vm| {}) {
            let ui = splash_makepad::to_makepad_ui(&node);
            self.ui.widget(cx, ids!(host)).set_text(cx, &ui);
        }
    }
}

impl MatchEvent for App {}

impl AppMain for App {
    fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
        // Light theme for the native widgets (select themes.light between the two
        // widget-stdlib halves so widgets bake it — see splash_catalog).
        crate::makepad_widgets::theme_mod(vm);
        script_eval!(vm, {
            mod.theme = mod.themes.light
        });
        crate::makepad_widgets::widgets_mod(vm);
        self::script_mod(vm)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event) {
        if matches!(event, Event::Startup) {
            self.next_frame = cx.new_next_frame();
            self.reload_if_changed(cx);
        }
        if self.next_frame.is_event(event).is_some() {
            self.tick = self.tick.wrapping_add(1);
            // Poll the hot-reload file a few times a second.
            if self.tick % 20 == 0 {
                self.reload_if_changed(cx);
            }
            cx.redraw_all();
            self.next_frame = cx.new_next_frame();
        }
        self.match_event(cx, event);
        self.ui.handle_event(cx, event, &mut Scope::empty());
    }
}
