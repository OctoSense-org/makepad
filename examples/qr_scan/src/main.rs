//! Exercises `Cx::show_qr_scanner`: tap "Scan QR", point the camera at a code,
//! and the full decoded text (and its length) shows up here and in the log
//! (`adb logcat | grep qr_scan`). Closing the scanner without a result bumps
//! the cancel counter and shows the reason. Off Android every scan request
//! is cancelled at once with reason "unsupported".
pub use makepad_widgets;

use makepad_widgets::makepad_platform::event::{NativeQrCancelled, NativeQrScanned};
use makepad_widgets::*;

app_main!(App);

script_mod! {
    use mod.prelude.widgets.*
    startup() do #(App::script_component(vm)){
        ui: Root{
            main_window := Window{
                window.inner_size: vec2(480, 640)
                body +: {
                    ScrollYView{
                        width: Fill
                        height: Fill
                        flow: Down
                        spacing: 14
                        padding: 24
                        scan_button := Button{
                            text: "Scan QR"
                        }
                        cancel_label := Label{
                            width: Fill
                            text: "Cancelled: 0"
                        }
                        length_label := Label{
                            width: Fill
                            text: "No code scanned yet"
                            draw_text.text_style.font_size: 14
                        }
                        text_label := Label{
                            width: Fill
                            text: ""
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
    #[rust]
    cancel_count: usize,
}

impl MatchEvent for App {
    fn handle_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        if self.ui.button(cx, ids!(scan_button)).clicked(actions) {
            log!("qr_scan: show_qr_scanner");
            cx.show_qr_scanner();
        }
        for action in actions {
            if let Some(scanned) = action.downcast_ref::<NativeQrScanned>() {
                let text = &scanned.json;
                log!("qr_scan: scanned {} chars: {}", text.chars().count(), text);
                self.ui.label(cx, ids!(length_label)).set_text(
                    cx,
                    &format!("Scanned {} chars ({} bytes)", text.chars().count(), text.len()),
                );
                self.ui.label(cx, ids!(text_label)).set_text(cx, text);
            } else if let Some(cancelled) = action.downcast_ref::<NativeQrCancelled>() {
                self.cancel_count += 1;
                log!("qr_scan: cancelled ({}) #{}", cancelled.reason, self.cancel_count);
                self.ui.label(cx, ids!(cancel_label)).set_text(
                    cx,
                    &format!("Cancelled: {} (last reason: {})", self.cancel_count, cancelled.reason),
                );
            }
        }
    }
}

impl AppMain for App {
    fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
        crate::makepad_widgets::script_mod(vm);
        self::script_mod(vm)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event) {
        self.match_event(cx, event);
        self.ui.handle_event(cx, event, &mut Scope::empty());
    }
}
