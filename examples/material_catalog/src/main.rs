//! A Material-Design component catalog authored in the Splash DSL and rendered
//! as makepad **native** widgets, translated on-device at runtime. Like MDC's
//! real catalog app it is **navigable**: a home list of component families, tap
//! one to open its demo screen, tap back to return.
//!
//!  * Scrolling host (`ScrollYView`) so a screen scrolls.
//!  * **Navigation** — nav Buttons carry a `tapto` route; the backend emits an
//!    `on_click` that writes the route into a `nav_signal` widget. Each frame the
//!    app reads that signal and, on change, re-mounts the target screen. The
//!    current route is injected into the DSL as `let screen = "…"`, so one
//!    `catalog.splash` renders every screen.
//!  * **On-device hot reload** — if `/data/local/tmp/material_catalog.splash`
//!    exists it is used instead of the baked catalog and re-mounted on change,
//!    so iterating is `adb push …` with no rebuild.

pub use makepad_widgets;

use makepad_widgets::*;

app_main!(App);

const BAKED: &str = include_str!("catalog.splash");
const DEVICE_PATH: &str = "/data/local/tmp/material_catalog.splash";

script_mod! {
    use mod.prelude.widgets.*

    startup() do #(App::script_component(vm)){
        ui: Root{
            main_window := Window{
                window.inner_size: vec2(440, 1400)
                body +: {
                    flow: Down
                    ScrollYView{
                        width: Fill
                        height: Fill
                        flow: Down
                        show_bg: true
                        draw_bg +: { color: #fef7ffff }
                        host := Splash{
                            isolate: false
                            width: Fill
                            height: Fit
                        }
                    }
                    // Compile-time routing signal: nav Buttons in the mounted
                    // catalog call ui.nav_signal.set_text(<route>); the app reads
                    // it each frame. It lives here (not in the Splash content) so
                    // ui.nav_signal resolves from the app Root on the main VM.
                    nav_signal := Label{ text: "" height: 0 draw_text.text_style.font_size: 1 }
                }
            }
        }
    }
}

#[derive(Script, ScriptHook)]
pub struct App {
    #[live]
    ui: WidgetRef,
    #[rust]
    next_frame: NextFrame,
    #[rust]
    last_src: String,
    #[rust]
    screen: String,
    #[rust]
    count: u32,
    #[rust]
    tick: u32,
    #[rust]
    started: bool,
}

impl App {
    /// The catalog source: the pushed device file if present, else baked.
    fn current_source() -> String {
        std::fs::read_to_string(DEVICE_PATH).unwrap_or_else(|_| BAKED.to_string())
    }

    /// Translate + mount the current screen. The active route is injected as a
    /// top-level `let screen`, so the single catalog renders the right screen.
    fn mount(&mut self, cx: &mut Cx) {
        let src = Self::current_source();
        self.last_src = src.clone();
        let route = if self.screen.is_empty() { "home" } else { &self.screen };
        // Inject the active route and live state into the DSL. `nav_route` (not
        // `screen` — that name is reserved/builtin in the VM and shadows the
        // injected value) carries the route; `tap_count` is live app state.
        let full = format!(
            "let nav_route = {route:?}\nlet tap_count = {}\n{src}",
            self.count
        );
        if let Some(node) = splash_render::build(&full, |_vm| {}) {
            let ui = splash_makepad::to_makepad_ui(&node);
            self.ui.widget(cx, ids!(host)).set_text(cx, &ui);
        }
    }
}

impl MatchEvent for App {}

impl AppMain for App {
    fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
        // Light theme for the native widgets (see splash_catalog).
        crate::makepad_widgets::theme_mod(vm);
        script_eval!(vm, {
            mod.theme = mod.themes.light
        });
        crate::makepad_widgets::widgets_mod(vm);
        self::script_mod(vm)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event) {
        if matches!(event, Event::Startup) && !self.started {
            self.started = true;
            self.screen = "home".to_string();
            self.next_frame = cx.new_next_frame();
            self.mount(cx);
        }
        if self.next_frame.is_event(event).is_some() {
            self.tick = self.tick.wrapping_add(1);
            // Navigation: a tapped nav Button wrote its route into `nav_signal`.
            let nav = self.ui.widget(cx, ids!(nav_signal)).text();
            if !nav.is_empty() {
                // Consume the signal so each tap fires exactly once.
                self.ui.widget(cx, ids!(nav_signal)).set_text(cx, "");
                if nav == "act:count" {
                    // A live-state action rather than a route change.
                    self.count = self.count.wrapping_add(1);
                } else {
                    self.screen = nav;
                }
                self.mount(cx);
            } else if self.tick % 20 == 0 && Self::current_source() != self.last_src {
                // Hot reload: the pushed catalog changed.
                self.mount(cx);
            }
            cx.redraw_all();
            self.next_frame = cx.new_next_frame();
        }
        self.match_event(cx, event);
        self.ui.handle_event(cx, event, &mut Scope::empty());
    }
}
