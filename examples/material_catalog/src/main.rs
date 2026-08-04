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
    // Per-component selection state, injected into the DSL so demos are stateful.
    #[rust]
    sel_tab: String,
    #[rust]
    sel_seg: String,
    #[rust]
    sel_date: String,
    #[rust]
    snack: bool,
    #[rust]
    dark: bool,
    // Which overlay/expander is currently open (dialog, menu, sheet, drawer,
    // sidesheet, overflow, tooltip, motion, "") — one at a time, so opening one
    // closes the rest. Drives real show/hide/dismiss, not a static mock.
    #[rust]
    open: String,
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
        let tab = if self.sel_tab.is_empty() { "overview" } else { &self.sel_tab };
        let seg = if self.sel_seg.is_empty() { "day" } else { &self.sel_seg };
        let date = if self.sel_date.is_empty() { "11" } else { &self.sel_date };
        let count = self.count;
        let snack = if self.snack { 1 } else { 0 };
        let dark = if self.dark { 1 } else { 0 };
        let open = self.open.as_str();
        // Inject the active route + live state as a single `let st = {…}` object
        // (one object binding is reliable where several top-level `let`s drop
        // bindings in this VM). The DSL reads st.route/st.count/st.tab/…/st.dark —
        // `st` avoids the reserved `screen`. Single-line, all-positional.
        let full = format!(
            "let st = {{ route: {:?}, count: {}, tab: {:?}, seg: {:?}, date: {:?}, snack: {}, dark: {}, open: {:?} }}\n{}",
            route, count, tab, seg, date, snack, dark, open, src
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
            // The signal Label reads back as whitespace (" ") when empty, so trim
            // it — otherwise a blank signal is treated as a real event and the
            // route is perpetually reset to " " (→ always home).
            let nav_raw = self.ui.widget(cx, ids!(nav_signal)).text();
            let nav = nav_raw.trim();
            if !nav.is_empty() {
                // Consume the signal so each tap fires exactly once.
                self.ui.widget(cx, ids!(nav_signal)).set_text(cx, "");
                if nav == "act:count" {
                    self.count = self.count.wrapping_add(1);
                } else if let Some(v) = nav.strip_prefix("tab:") {
                    self.sel_tab = v.to_string();
                } else if let Some(v) = nav.strip_prefix("seg:") {
                    self.sel_seg = v.to_string();
                } else if let Some(v) = nav.strip_prefix("date:") {
                    self.sel_date = v.to_string();
                } else if nav == "snack:show" {
                    self.snack = true;
                } else if nav == "snack:hide" {
                    self.snack = false;
                } else if nav == "theme:toggle" {
                    self.dark = !self.dark;
                } else if let Some(v) = nav.strip_prefix("open:") {
                    // Open/close an overlay or expander (one at a time).
                    self.open = v.to_string();
                } else {
                    // Anything else is a route change; close any open overlay.
                    self.screen = nav.to_string();
                    self.open = String::new();
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
