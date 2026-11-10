//! A widget that is a view (`#[deref]`) and also names a widget of its own (`#[find]`) has both
//! sets of children: the view's, and what the named widget holds. The derive listed only the
//! `#[find]` field's, so a lookup from the widget itself replaced the view's children in the
//! widget tree with those and found nothing inside the view.
use makepad_widgets::makepad_script::script;
use makepad_widgets::*;

script_mod! {
    use mod.prelude.widgets.*

    mod.widgets.ViewAndNamedBase = #(ViewAndNamed::register_widget(vm))
    mod.widgets.ViewAndNamed = set_type_default() do mod.widgets.ViewAndNamedBase{
        named: View{
            in_named := Label{ text: "in the named widget" }
        }
    }
}

#[derive(Script, ScriptHook, Widget)]
struct ViewAndNamed {
    #[deref]
    view: View,
    #[find]
    #[live]
    named: WidgetRef,
}

impl Widget for ViewAndNamed {
    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        self.view.draw_walk(cx, scope, walk)
    }
    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        self.view.handle_event(cx, event, scope)
    }
}

fn new_cx() -> Cx {
    let mut cx = Cx::new(Box::new(|_, _| {}));
    cx.with_vm(|vm| makepad_widgets::script_mod(vm));
    cx.with_vm(|vm| script_mod(vm));
    cx.with_vm(|vm| vm.capture_errors());
    cx
}

#[test]
fn the_view_children_and_the_named_widget_children_are_listed() {
    let mut cx = new_cx();
    let doc = cx.with_vm(|vm| {
        vm.eval(script! {
            use mod.prelude.widgets.*
            use mod.widgets.*
            ViewAndNamed{
                first := Label{ text: "first" }
                second := View{
                    inner := Label{ text: "inner" }
                }
            }
        })
    });
    let root = cx.with_vm(|vm| WidgetRef::script_from_value(vm, doc));
    let errs = cx.with_vm(|vm| vm.take_errors());
    assert!(errs.is_empty(), "{errs:?}");

    let mut ids = Vec::new();
    root.children(&mut |id, _| ids.push(id));
    ids.sort();
    let mut want = vec![live_id!(first), live_id!(second), live_id!(in_named)];
    want.sort();
    assert_eq!(ids, want);
}
