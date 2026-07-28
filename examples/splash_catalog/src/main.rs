//! A widget catalog authored in the Splash DSL, rendered by the makepad render
//! backend. The UI below is emitted by `splash-makepad::to_makepad_ui` from
//! `crates/splash-makepad/examples/catalog.splash` — the same shared VM +
//! UiNode Splash-OH renders to native ArkUI, here rendered by makepad widgets.
//!
//! Light theme: the catalog paints a light page + white cards, and the native
//! widgets are switched to makepad's `themes.light` (see `AppMain::script_mod`).

pub use makepad_widgets;

use makepad_widgets::*;

app_main!(App);

script_mod! {
    use mod.prelude.widgets.*

    startup() do #(App::script_component(vm)){
        ui: Root{
            main_window := Window{
                window.inner_size: vec2(440, 900)
                body +: {
                    RoundedView {
                        flow: Down
                        width: Fill
                        height: Fit
                        padding: 14
                        show_bg: true
                        draw_bg +: { color: #eef0f5ff }
                        Label {
                            height: 30
                            text: "Widget Catalog"
                            draw_text.text_style.font_size: 22
                            draw_text.color: #1c2028ff
                        }
                        Label {
                            height: 20
                            text: "Splash DSL → makepad native widgets"
                            draw_text.text_style.font_size: 13
                            draw_text.color: #707886ff
                        }
                        View {
                            flow: Down
                            width: Fill
                            height: 10
                        }
                        RoundedView {
                            flow: Down
                            width: Fill
                            height: Fit
                            padding: 16
                            show_bg: true
                            draw_bg +: { color: #ffffffff, radius: 16 }
                            Label {
                                height: 18
                                text: "BUTTONS"
                                draw_text.text_style.font_size: 11
                                draw_text.color: #707886ff
                            }
                            View {
                                flow: Down
                                width: Fill
                                height: 10
                            }
                            View {
                                flow: Right
                                align: Align{y: 0.5}
                                width: Fill
                                height: 42
                                Button {
                                    width: 120
                                    height: 38
                                    text: "Primary"
                                }
                                View {
                                    flow: Down
                                    width: 12
                                    height: 4
                                }
                                Button {
                                    width: 140
                                    height: 38
                                    text: "Secondary"
                                }
                            }
                        }
                        View {
                            flow: Down
                            width: Fill
                            height: 10
                        }
                        RoundedView {
                            flow: Down
                            width: Fill
                            height: Fit
                            padding: 16
                            show_bg: true
                            draw_bg +: { color: #ffffffff, radius: 16 }
                            Label {
                                height: 18
                                text: "SELECTION"
                                draw_text.text_style.font_size: 11
                                draw_text.color: #707886ff
                            }
                            View {
                                flow: Down
                                width: Fill
                                height: 10
                            }
                            View {
                                flow: Right
                                align: Align{y: 0.5}
                                width: Fill
                                height: 32
                                CheckBox {
                                    width: 160
                                    height: 30
                                    text: "Checkbox"
                                }
                                View {
                                    flow: Down
                                    width: 20
                                    height: 4
                                }
                                Toggle {
                                    width: 120
                                    height: 30
                                    text: "Toggle"
                                }
                            }
                            View {
                                flow: Down
                                width: Fill
                                height: 8
                            }
                            View {
                                flow: Right
                                align: Align{y: 0.5}
                                width: Fill
                                height: 32
                                RadioButton {
                                    width: 160
                                    height: 30
                                    text: "Option A"
                                }
                                View {
                                    flow: Down
                                    width: 20
                                    height: 4
                                }
                                RadioButton {
                                    width: 120
                                    height: 30
                                    text: "Option B"
                                }
                            }
                        }
                        View {
                            flow: Down
                            width: Fill
                            height: 10
                        }
                        RoundedView {
                            flow: Down
                            width: Fill
                            height: Fit
                            padding: 16
                            show_bg: true
                            draw_bg +: { color: #ffffffff, radius: 16 }
                            Label {
                                height: 18
                                text: "SLIDER"
                                draw_text.text_style.font_size: 11
                                draw_text.color: #707886ff
                            }
                            View {
                                flow: Down
                                width: Fill
                                height: 10
                            }
                            Slider {
                                height: 36
                                text: "Volume"
                            }
                        }
                        View {
                            flow: Down
                            width: Fill
                            height: 10
                        }
                        RoundedView {
                            flow: Down
                            width: Fill
                            height: Fit
                            padding: 16
                            show_bg: true
                            draw_bg +: { color: #ffffffff, radius: 16 }
                            Label {
                                height: 18
                                text: "TEXT INPUT"
                                draw_text.text_style.font_size: 11
                                draw_text.color: #707886ff
                            }
                            View {
                                flow: Down
                                width: Fill
                                height: 10
                            }
                            TextInput {
                                height: 40
                            }
                        }
                        View {
                            flow: Down
                            width: Fill
                            height: 10
                        }
                        RoundedView {
                            flow: Down
                            width: Fill
                            height: Fit
                            padding: 16
                            show_bg: true
                            draw_bg +: { color: #ffffffff, radius: 16 }
                            Label {
                                height: 18
                                text: "TYPOGRAPHY"
                                draw_text.text_style.font_size: 11
                                draw_text.color: #707886ff
                            }
                            View {
                                flow: Down
                                width: Fill
                                height: 10
                            }
                            Label {
                                height: 28
                                text: "Heading"
                                draw_text.text_style.font_size: 20
                                draw_text.color: #1c2028ff
                            }
                            Label {
                                height: 22
                                text: "Body copy rendered by makepad."
                                draw_text.text_style.font_size: 14
                                draw_text.color: #707886ff
                            }
                            View {
                                flow: Down
                                width: Fill
                                height: 2
                            }
                            Label {
                                height: 20
                                text: "one VM · one UiNode · two backends"
                                draw_text.text_style.font_size: 13
                                draw_text.color: #1e6cdcff
                            }
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
    // makepad renders on demand; on this OnePlus an idle frame is not presented
    // to the panel, so we drive continuous frames to keep the display fed.
    #[rust]
    next_frame: NextFrame,
}

impl MatchEvent for App {
    fn handle_startup(&mut self, cx: &mut Cx) {
        self.next_frame = cx.new_next_frame();
    }
}

impl AppMain for App {
    fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
        // Register makepad's widget stdlib in two halves so we can select the
        // LIGHT theme in between: `theme_mod` defines `mod.themes.{dark,light}`
        // and defaults `mod.theme = dark`; `widgets_mod` then bakes whatever
        // `mod.theme` points at into every widget template. Overriding to light
        // here makes the native Button/CheckBox/Slider/TextInput render light.
        crate::makepad_widgets::theme_mod(vm);
        script_eval!(vm, {
            mod.theme = mod.themes.light
        });
        crate::makepad_widgets::widgets_mod(vm);
        self::script_mod(vm)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event) {
        if self.next_frame.is_event(event).is_some() {
            cx.redraw_all();
            self.next_frame = cx.new_next_frame();
        }
        self.match_event(cx, event);
        self.ui.handle_event(cx, event, &mut Scope::empty());
    }
}
