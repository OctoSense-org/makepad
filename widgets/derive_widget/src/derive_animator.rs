use proc_macro::TokenStream;

use makepad_micro_proc_macro::{error, TokenBuilder, TokenParser};

pub fn derive_animator_impl(input: TokenStream) -> TokenStream {
    let mut tb = TokenBuilder::new();
    let mut parser = TokenParser::new(input);
    let _main_attribs = parser.eat_attributes();

    parser.eat_ident("pub");
    if parser.eat_ident("struct") {
        let struct_name = parser.expect_any_ident().unwrap();
        let generic = parser.eat_generic();
        let types = parser.eat_all_types();
        let where_clause = parser.eat_where_clause(None); //Some("LiveUpdateHooks"));

        let fields = if let Some(_types) = types {
            return error("Unexpected type form");
        } else if let Some(fields) = parser.eat_all_struct_fields() {
            fields
        } else {
            return error("Unexpected field form");
        };

        // alright now. we have a field
        let animator_field = fields.iter().find(|field| field.name == "animator");

        if let Some(animator_field) = animator_field {
            // The #[source] field is the ScriptObjectRef this widget was built from,
            // and it is what names WHICH VM's heap the animator's objects live in.
            let source_field = fields
                .iter()
                .find(|field| field.attrs.iter().any(|attr| attr.name == "source"));

            if source_field.is_none() {
                tb.add("compile_error!(\"Animator derive requires a field with #[source] attribute to hold the ScriptObjectRef\");");
            }
            let source_name = source_field.map(|f| f.name.clone()).unwrap_or_default();
            // ROUTE EVERY ANIMATOR OPERATION TO THE VM THAT OWNS THE WIDGET.
            //
            // `Animator::play`/`cut`/`handle_event` and the `script_apply` that
            // follows all dereference script objects — the state's apply object,
            // the widget's defaults, the track snapshots — through `cx.with_vm`,
            // which is the MAIN VM. A widget instantiated inside a Splash card
            // lives in that card's ISOLATE heap (`alloc_splash_vm`), so the same
            // indices name unrelated objects there: on the first animator call the
            // generation check fired (`GenVec<ScriptObjectData> use-after-free`)
            // and the panic killed the makepad main loop thread. Measured on a
            // OnePlus 6: tapping a TextInput inside any generated card — this
            // profile's `Field` and the old L2 nav card's search box alike — died
            // in `animator_play(hover.down)`, leaving every makepad widget dead
            // while the app's NATIVE chrome (composer, FAB) kept working, which is
            // exactly the "card suddenly goes inert" bug. Buttons survived only
            // because their `cut(blink/hover)` calls resolve to no value before
            // any heap access.
            //
            // `script_ref_vm_id` reads the owning heap off the `#[source]` ref
            // itself, and `with_script_vm_id` + `with_cx_mut` runs the operation
            // with that isolate parked as the active VM, so every inner
            // `cx.with_vm` resolves the right heap. Main-heap widgets take the
            // untouched original path.
            let route_open = |tb: &mut TokenBuilder, source_name: &str| {
                tb.add("        let __anim_vm = crate::widget_async::CxSplashVmExt::script_ref_vm_id(cx, &self.")
                    .ident(source_name)
                    .add(");");
                tb.add("        if __anim_vm != crate::widget_async::MAIN_SPLASH_VM_ID {");
                tb.add("            return crate::widget_async::CxSplashVmExt::with_script_vm_id(cx, __anim_vm, |vm| {");
                tb.add("                crate::makepad_draw::makepad_platform::script::vm::ScriptVmCx::with_cx_mut(vm, |cx| {");
            };
            let route_close = |tb: &mut TokenBuilder| {
                tb.add("                })");
                tb.add("            });");
                tb.add("        }");
            };

            tb.add("impl").stream(generic.clone());
            tb.add("AnimatorImpl for")
                .ident(&struct_name)
                .stream(generic.clone())
                .stream(where_clause.clone())
                .add("{");

            tb.add("    fn animator_play_scoped(&mut self, cx: &mut Cx, state: &[LiveId;2], play: Option<Play>, scope:&mut Scope) {");
            // If the VM is already held (we're inside an apply walk's cx.with_vm),
            // calling play() -> cx.with_vm would re-enter and panic; defer instead.
            tb.add("        if cx.is_script_vm_held() { self.")
                .ident(&animator_field.name)
                .add(".defer_play(cx, state, play); return; }");
            route_open(&mut tb, &source_name);
            tb.add("                    if let Some(value) = self.")
                .ident(&animator_field.name)
                .add(".play(cx, state, play){");
            tb.add("                        cx.with_vm(|vm| self.script_apply(vm, &Apply::Animate, scope, value));");
            tb.add("                    }");
            route_close(&mut tb);
            tb.add("        if let Some(value) = self.")
                .ident(&animator_field.name)
                .add(".play(cx, state, play){");
            tb.add("            cx.with_vm(|vm| self.script_apply(vm, &Apply::Animate, scope, value));");
            tb.add("        }");
            tb.add("    }");

            tb.add(
                "    fn animator_in_state(&self, cx: &Cx, check_state_pair: &[LiveId; 2]) -> bool{",
            );
            tb.add("         self.")
                .ident(&animator_field.name)
                .add(".in_state(cx, check_state_pair)");
            tb.add("    }");

            tb.add("    fn animator_cut_scoped(&mut self, cx: &mut Cx, state: &[LiveId;2], scope:&mut Scope) {");
            // Same VM-held deferral as animator_play_scoped above.
            tb.add("         if cx.is_script_vm_held() { self.")
                .ident(&animator_field.name)
                .add(".defer_cut(cx, state); return; }");
            route_open(&mut tb, &source_name);
            tb.add("                    if let Some(value) = self.")
                .ident(&animator_field.name)
                .add(".cut(cx, state){");
            tb.add("                        cx.with_vm(|vm| self.script_apply(vm, &Apply::Animate, scope, value));");
            tb.add("                    }");
            route_close(&mut tb);
            tb.add("         if let Some(value) = self.")
                .ident(&animator_field.name)
                .add(".cut(cx, state){");
            tb.add("             cx.with_vm(|vm| self.script_apply(vm, &Apply::Animate, scope, value));");
            tb.add("         }");
            tb.add("    }");

            tb.add("    fn animator_handle_event_scoped(&mut self, cx: &mut Cx, event: &Event, scope:&mut Scope)->AnimatorAction{");
            route_open(&mut tb, &source_name);
            tb.add("                    let mut act = AnimatorAction::None;");
            tb.add("                    for value in self.")
                .ident(&animator_field.name)
                .add(".flush_deferred(cx){");
            tb.add("                        cx.with_vm(|vm| self.script_apply(vm, &Apply::Animate, scope, value));");
            tb.add("                    }");
            tb.add("                    if let Some(value) = self.")
                .ident(&animator_field.name)
                .add(".handle_event(cx, event, &mut act){");
            tb.add("                        cx.with_vm(|vm| self.script_apply(vm, &Apply::Animate, scope, value));");
            tb.add("                    }");
            tb.add("                    act");
            route_close(&mut tb);
            tb.add("         let mut act = AnimatorAction::None;");
            // Replay any cut/play that was deferred while the VM was held (the VM
            // is free during event handling). A no-op when nothing is queued.
            tb.add("         for value in self.")
                .ident(&animator_field.name)
                .add(".flush_deferred(cx){");
            tb.add("             cx.with_vm(|vm| self.script_apply(vm, &Apply::Animate, scope, value));");
            tb.add("         }");
            tb.add("         if let Some(value) = self.")
                .ident(&animator_field.name)
                .add(".handle_event(cx, event, &mut act){");
            tb.add("             cx.with_vm(|vm| self.script_apply(vm, &Apply::Animate, scope, value));");
            tb.add("         }");
            tb.add("         act");
            tb.add("    }");

            tb.add("}");
        }
        return tb.end();
    }
    parser.unexpected()
}
