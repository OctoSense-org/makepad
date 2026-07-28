//! A Splash card rendered by makepad — the makepad render backend, on device.
//!
//! The UI below is the makepad dialect `splash-makepad::to_makepad_ui` emits for
//! a Splash card: the same shared VM + `UiNode` that Splash-OH renders to native
//! ArkUI, rendered here by makepad's own widgets. Filled containers are
//! `RoundedView` (plain `View` does not paint `draw_bg` in this makepad), so the
//! card, its panels and the badge all show real backgrounds — not floating text.

pub use makepad_widgets;

use makepad_widgets::*;

app_main!(App);

script_mod! {
    use mod.prelude.widgets.*

    startup() do #(App::script_component(vm)){
        ui: Root{
            main_window := Window{
                window.inner_size: vec2(440, 420)
                body +: {
                    RoundedView{
                        width: Fill
                        height: Fill
                        flow: Down
                        padding: 20
                        spacing: 16
                        show_bg: true
                        draw_bg +: { color: #14161dff }

                        // title bar panel: red badge + wordmark
                        RoundedView{
                            width: Fill
                            height: Fit
                            flow: Right
                            spacing: 12
                            padding: 12
                            align: {y: 0.5}
                            show_bg: true
                            draw_bg +: { color: #232634ff, radius: 12.0 }
                            RoundedView{
                                width: 36
                                height: 26
                                show_bg: true
                                draw_bg +: { color: #ff3b3bff, radius: 6.0 }
                            }
                            Label{
                                text: "Splash \u{2192} makepad"
                                draw_text.text_style.font_size: 24
                                draw_text.color: #ffffffff
                            }
                        }

                        // a real panel with two lines
                        RoundedView{
                            width: Fill
                            height: Fit
                            flow: Down
                            spacing: 6
                            padding: 16
                            show_bg: true
                            draw_bg +: { color: #232634ff, radius: 12.0 }
                            Label{
                                text: "one VM \u{00b7} one UiNode \u{00b7} two backends"
                                draw_text.text_style.font_size: 14
                                draw_text.color: #9aa3b5ff
                            }
                            Label{
                                text: "ArkUI (Splash-OH) + makepad"
                                draw_text.text_style.font_size: 19
                                draw_text.color: #ffffffff
                            }
                        }

                        // two stat tiles in a row
                        View{
                            width: Fill
                            height: Fit
                            flow: Right
                            spacing: 12
                            RoundedView{
                                width: Fill
                                height: 76
                                flow: Down
                                padding: 12
                                spacing: 4
                                show_bg: true
                                draw_bg +: { color: #2a2f40ff, radius: 12.0 }
                                Label{ text: "BACKEND", draw_text.text_style.font_size: 11, draw_text.color: #8a93a5ff }
                                Label{ text: "makepad", draw_text.text_style.font_size: 20, draw_text.color: #ffffffff }
                            }
                            RoundedView{
                                width: Fill
                                height: 76
                                flow: Down
                                padding: 12
                                spacing: 4
                                show_bg: true
                                draw_bg +: { color: #2a2f40ff, radius: 12.0 }
                                Label{ text: "SIBLING", draw_text.text_style.font_size: 11, draw_text.color: #8a93a5ff }
                                Label{ text: "ArkUI", draw_text.text_style.font_size: 20, draw_text.color: #ffffffff }
                            }
                        }

                        // a real interactive widget
                        Button{
                            text: "A real makepad Button"
                        }

                        Label{
                            text: "OnePlus 6T \u{00b7} cargo-makepad"
                            draw_text.text_style.font_size: 13
                            draw_text.color: #7abeffff
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
        crate::makepad_widgets::script_mod(vm);
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
