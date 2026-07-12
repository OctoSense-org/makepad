use crate::{
    makepad_derive_widget::*,
    makepad_draw::*,
    view::View,
    widget::*,
    widget_async::{CxSplashVmExt, SplashVmId, MAIN_SPLASH_VM_ID},
    widget_tree::CxWidgetExt,
};

#[derive(Clone, Debug, Default)]
pub enum SplashAction {
    Notify {
        event_id: String,
        payload: String,
    },
    #[default]
    None,
}

/// Web-Mercator "slippy map" tile (x, y) covering `lat`/`lon` at zoom `z`
/// (the standard OSM/XYZ scheme used by Carto and WAQI tile servers).
fn slippy_tile(lat: f64, lon: f64, z: u32) -> (i64, i64) {
    let n = (1u64 << z) as f64;
    let x = ((lon + 180.0) / 360.0 * n).floor();
    let y = ((1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0 * n).floor();
    let max = (1i64 << z) - 1;
    ((x as i64).clamp(0, max), (y as i64).clamp(0, max))
}

pub fn register_agent_module(vm: &mut ScriptVm) {
    let agent = vm.new_module(id!(agent));
    vm.add_method(
        agent,
        id_lut!(notify),
        script_args_def!(event = NIL, payload = NIL),
        |vm, args| {
            let event_value = script_value!(vm, args.event);
            let payload_value = script_value!(vm, args.payload);

            let mut event_id = String::new();
            vm.bx.heap.cast_to_string(event_value, &mut event_id);

            let mut payload = String::new();
            vm.bx.heap.to_json_inner(payload_value, &mut payload);

            Cx::post_action(SplashAction::Notify { event_id, payload });
            NIL
        },
    );
    vm.set_injected_global(id!(agent), agent.into());

    // `sys` module — Rust helpers callable from generated splash code.
    // Extend this with more methods (data fetchers, formatters, card builders)
    // and teach the LLM to call them in the A2App prompt.
    let sys = vm.new_module(id!(sys));

    // sys.photo("tokyo skyline sunset") -> a full-screen 9:16 image URL for that
    // subject (pollinations.ai renders the prompt with an AI model, so the photo
    // always matches the subject). Centralises image sourcing in Rust so it can be
    // improved (curation, quality, providers) without touching the prompt.
    // Use as `Image{ src: http_resource(sys.photo("<q>")) }`.
    vm.add_method(
        sys,
        id_lut!(photo),
        script_args_def!(query = NIL),
        |vm, args| {
            let query_value = script_value!(vm, args.query);
            let mut query = String::new();
            vm.bx.heap.cast_to_string(query_value, &mut query);

            // AI-generated, always ON-TOPIC 9:16 portrait image. loremflickr
            // OR-matches comma tags, so a multi-word subject ("paris eiffel
            // tower sunny") returned unrelated photos (a cat statue). Pollinations
            // renders the full natural-language prompt, so the photo always
            // matches the subject and is high quality — the "nano banana"-style
            // AI source the app wants for beautiful full-screen backgrounds.
            let q = query.trim();
            let q = if q.is_empty() {
                "beautiful cinematic landscape scenery, golden hour"
            } else {
                q
            };
            // Percent-encode the prompt for a URL path segment (RFC 3986):
            // keep unreserved chars, encode everything else (incl. spaces) by
            // UTF-8 byte.
            let mut enc = String::with_capacity(q.len() * 3);
            let mut buf = [0u8; 4];
            for ch in q.chars() {
                if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '~') {
                    enc.push(ch);
                } else {
                    for b in ch.encode_utf8(&mut buf).as_bytes() {
                        enc.push('%');
                        enc.push(char::from_digit((b >> 4) as u32, 16).unwrap().to_ascii_uppercase());
                        enc.push(char::from_digit((b & 0xF) as u32, 16).unwrap().to_ascii_uppercase());
                    }
                }
            }
            // 1080x1920 = 9:16. nologo strips the watermark; model=flux is fast
            // and photoreal.
            let url = format!(
                "https://image.pollinations.ai/prompt/{enc}?width=1080&height=1920&nologo=true&model=flux"
            );
            vm.bx.heap.new_string_from_str(&url)
        },
    );

    // sys.satellite() -> a LIVE full-disk satellite cloud map (卫星云图) image URL:
    // Himawari-9 true-color over Asia-Pacific from NICT, refreshed every 10 min. The
    // frame timestamp is computed at CALL time (i.e. on each render), so a saved card
    // always shows recent clouds instead of a stale baked URL. The true-color full
    // disk is daylight only (dark over Asia at local night). The disk is square with
    // a black-space margin, so ImageFit.Smallest shows the whole Earth cleanly.
    // Use as `Image{ src: http_resource(sys.satellite()) fit: ImageFit.Smallest }`.
    vm.add_method(
        sys,
        id_lut!(satellite),
        script_args_def!(region = NIL),
        |vm, _args| {
            use std::time::{SystemTime, UNIX_EPOCH};
            // The newest published full-disk frame lags real time; back off 40 min
            // and floor to the 10-min cadence so the tile is reliably available.
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let t = now.saturating_sub(40 * 60);
            let t = t - (t % 600);
            let secs_of_day = t % 86_400;
            let hh = secs_of_day / 3_600;
            let mm = (secs_of_day % 3_600) / 60;
            // Civil date from days-since-epoch (Howard Hinnant's algorithm).
            let days = (t / 86_400) as i64;
            let z = days + 719_468;
            let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
            let doe = z - era * 146_097; // [0, 146096]
            let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
            let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
            let mp = (5 * doy + 2) / 153; // [0, 11]
            let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
            let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
            let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
            let url = format!(
                "https://himawari8.nict.go.jp/img/D531106/1d/550/{year:04}/{month:02}/{day:02}/{hh:02}{mm:02}00_0_0.png"
            );
            vm.bx.heap.new_string_from_str(&url)
        },
    );

    // sys.basemap(lat, lon) -> a dark base-map tile (Carto, no key) at the city, meant to
    // sit UNDER sys.airmap in an Overlay so the AQI colours have geographic context.
    // Zoom 7 frames the metro + surrounding region in one 256px tile.
    vm.add_method(
        sys,
        id_lut!(basemap),
        script_args_def!(lat = NIL, lon = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let (x, y) = slippy_tile(lat, lon, 7);
            let url = format!("https://a.basemaps.cartocdn.com/dark_all/7/{x}/{y}.png");
            vm.bx.heap.new_string_from_str(&url)
        },
    );

    // sys.airmap(lat, lon) -> a LIVE air-quality colour overlay tile (WAQI, US-EPA AQI
    // scale). Mostly transparent except where AQI data exists, so stack it OVER
    // sys.basemap(lat, lon) at the SAME lat/lon in an Overlay (both use zoom 7).
    // Use as `View{ flow: Overlay Image{basemap} Image{airmap} }`.
    vm.add_method(
        sys,
        id_lut!(airmap),
        script_args_def!(lat = NIL, lon = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let (x, y) = slippy_tile(lat, lon, 7);
            let url = format!("https://tiles.waqi.info/tiles/usepa-aqi/7/{x}/{y}.png?token=_");
            vm.bx.heap.new_string_from_str(&url)
        },
    );

    vm.set_injected_global(id!(sys), sys.into());
}

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.*

    mod.widgets.SplashBase = #(Splash::register_widget(vm))

    mod.widgets.Splash = set_type_default() do mod.widgets.SplashBase{
        width: Fill height: Fit
    }
}

#[derive(Script, ScriptHook, WidgetRef, WidgetRegister)]
pub struct Splash {
    #[uid]
    uid: WidgetUid,
    #[source]
    source: ScriptObjectRef,
    #[deref]
    pub view: View,
    #[live]
    body: ArcStringMut,
    #[rust]
    eval_generation: u64,
    #[rust]
    tick_timer: Timer,
    /// The unique_id used for the last full eval, so tick() runs in the same scope.
    #[rust]
    last_unique_id: usize,
    /// This Splash's own VM, allocated on first eval (upstream isolation model).
    #[rust]
    vm_id: SplashVmId,
    /// Body text of the previous eval. Used to detect streaming extensions
    /// (the new body forward-extends the old) so repeated set_text(full growing
    /// text) reuses ONE vm body instead of a fresh generation per frame.
    #[rust]
    last_eval_body: String,
}

/// Prefix for View-children mode: wraps code inside a View
const SPLASH_PREFIX_VIEW: &str = "use mod.prelude.widgets.*View{height:Fit, ";
/// Prefix for full-script mode: just imports, code must evaluate to a widget
const SPLASH_PREFIX_SCRIPT: &str = "use mod.prelude.widgets.*\n";
const SPLASH_EVAL_INSTRUCTION_LIMIT: usize = 200_000;

/// Detect whether Splash code is a full script (starts with `let`, `fn`,
/// or a widget constructor like `View{`, `SolidView{`) vs View children
/// (starts with properties like `flow:`, `width:`, or lowercase names).
fn is_full_script(body: &str) -> bool {
    let trimmed = body.trim_start();
    // Only treat as full script if it starts with scripting keywords
    // (let/fn/mod) — these can't appear inside a View{} property list.
    // Uppercase widget names (View{, SolidView{, Label{) stay in View-children mode.
    trimmed.starts_with("let ") || trimmed.starts_with("fn ") || trimmed.starts_with("mod.")
}

impl Splash {
    /// Stable identity for the streaming script body, based on pointer address.
    fn self_id(&self) -> usize {
        self as *const Self as usize
    }

    fn eval_body(&mut self, cx: &mut Cx) {
        let body = self.body.as_ref().to_string();
        if body.is_empty() {
            return;
        }

        // Stop any previous tick timer
        cx.stop_timer(self.tick_timer);

        // Allocate this Splash's own VM on first eval so streaming
        // (stream_append) evaluates in an isolated scope.
        if self.vm_id == MAIN_SPLASH_VM_ID {
            self.vm_id = cx.alloc_splash_vm();
        }

        // Only start a NEW vm body (bump the generation) on a genuine content
        // replacement — NOT a streaming extension of the previous body. aichat
        // streams runsplash by calling set_text() with the full, growing block
        // string every frame; without this each frame got its own generation,
        // accumulating dozens of stale bodies whose widgets/closures lingered
        // (clicking a button then hit a stale generation -> "widget not found in
        // tree" -> the app vanished). A forward-extension reuses the same
        // unique_id so eval_with_append_source does its incremental checkpoint
        // parse (the same path stream_append uses). Compare the raw body (not the
        // prefixed code) so an is_full_script flip can't cause a false miss.
        let is_extension =
            !self.last_eval_body.is_empty() && body.starts_with(self.last_eval_body.as_str());
        if !is_extension {
            self.eval_generation += 1;
        }
        self.last_eval_body = body.clone();
        let unique_id = self.self_id().wrapping_add(self.eval_generation as usize);
        self.last_unique_id = unique_id;

        // Choose prefix based on code style
        let prefix = if is_full_script(&body) {
            SPLASH_PREFIX_SCRIPT
        } else {
            SPLASH_PREFIX_VIEW
        };
        let code = format!("{}{}", prefix, body);

        let script_mod = ScriptMod {
            cargo_manifest_path: String::new(),
            module_path: String::new(),
            file: String::new(),
            line: unique_id,
            column: 0,
            code: String::new(),
            values: vec![],
        };

        log!(
            "[SPLASH] eval_body: {} bytes, prefix={}, uid={}, gen={}, ext={}",
            body.len(),
            if is_full_script(&body) {
                "script"
            } else {
                "view"
            },
            unique_id,
            self.eval_generation,
            is_extension
        );

        // Evaluate in THIS Splash's own isolated vm and inject a `ui` global
        // rooted at this Splash (self.uid). That scopes `ui.<id>` to this
        // Splash's subtree (find_flood), so ids like `display` don't collide
        // with other Splash apps in the same chat. Then register the widgets
        // under this vm and mark the tree dirty so lookups can resolve them.
        let vm_id = self.vm_id;
        let self_uid = self.uid;
        let new_view = cx.with_script_vm_id(vm_id, |vm| {
            crate::widget_async::inject_scoped_ui_global(vm, self_uid);
            let value = vm.with_instruction_limit(SPLASH_EVAL_INSTRUCTION_LIMIT, |vm| {
                vm.eval_with_append_source(script_mod, &code, NIL.into())
            });
            if !value.is_err() && !value.is_nil() {
                Some(View::script_from_value(vm, value))
            } else {
                None
            }
        });

        if let Some(view) = new_view {
            self.view = view;
            self.view.set_visible(cx, true);
            crate::widget_async::inject_splash_ui_handle(cx, self.vm_id, self.view.widget_uid());
            cx.widget_tree_mark_dirty(self.uid);
        }

        // If the Splash code defines fn tick(), auto-start a 1s interval
        if body.contains("fn tick(") || body.contains("fn tick (") {
            self.tick_timer = cx.start_interval(1.0);
        }
    }

    /// Call a named function defined in the Splash code's scope.
    pub fn call_fn(&mut self, cx: &mut Cx, name: LiveId) {
        let unique_id = self.last_unique_id;
        if unique_id == 0 {
            return;
        }

        cx.with_script_vm_id(self.vm_id, |vm| {
            // Find the body by matching the unique_id we used during eval
            // (body lives in this Splash's isolated vm, same as eval_body).
            let scope_obj = {
                let bodies = vm.bx.code.bodies.borrow();
                let mut found = None;
                for body in bodies.iter() {
                    if let ScriptSource::Mod(m) = &body.source {
                        if m.line == unique_id {
                            found = Some(body.scope.as_object());
                            break;
                        }
                    }
                }
                found
            };

            if let Some(scope) = scope_obj {
                let tick_fn = vm.bx.heap.scope_value(scope, name, vm.trap());
                if !tick_fn.is_nil() && !tick_fn.is_err() {
                    vm.call(tick_fn, &[]);
                }
            }
        });

        cx.redraw_all();
    }

    /// Start a new streaming session. Resets the accumulated code and
    /// increments the generation so the VM creates a fresh body.
    pub fn stream_begin(&mut self, cx: &mut Cx) {
        self.eval_generation += 1;
        self.body.set("");
        // Eval a minimal empty view to clear previous content
        self.body.set("View{}");
        self.eval_body(cx);
        self.body.set("");
        cx.redraw_all();
    }

    /// Append a chunk of Splash code and incrementally re-evaluate.
    /// The VM reuses the same body (fixed line ID) so only new tokens
    /// are tokenized and parsed via checkpoint-based streaming.
    pub fn stream_append(&mut self, cx: &mut Cx, chunk: &str) {
        // Append to body
        let mut current = self.body.as_ref().to_string();
        current.push_str(chunk);
        self.body.set(&current);

        let prefix = if is_full_script(&current) {
            SPLASH_PREFIX_SCRIPT
        } else {
            SPLASH_PREFIX_VIEW
        };
        let code = format!("{}{}", prefix, current);

        // Use a fixed line ID (based on self_id + current generation)
        // so eval_with_append_source finds the existing body and
        // only tokenizes/parses the new delta.
        let unique_id = self.self_id().wrapping_add(self.eval_generation as usize);

        let script_mod = ScriptMod {
            cargo_manifest_path: String::new(),
            module_path: String::new(),
            file: String::new(),
            line: unique_id,
            column: 0,
            code: String::new(),
            values: vec![],
        };

        let vm_id = self.vm_id;
        let self_uid = self.uid;
        let new_view = cx.with_script_vm_id(vm_id, |vm| {
            crate::widget_async::inject_scoped_ui_global(vm, self_uid);
            let value = vm.with_instruction_limit(SPLASH_EVAL_INSTRUCTION_LIMIT, |vm| {
                vm.eval_with_append_source(script_mod, &code, NIL.into())
            });
            if !value.is_err() && !value.is_nil() {
                Some(View::script_from_value(vm, value))
            } else {
                None
            }
        });

        if let Some(view) = new_view {
            self.view = view;
            // Make `ui` a global in this splash's VM (pointing at the freshly-built view root) so
            // helper `fn`s inside the block can use `ui.<id>.set_text(...)`, not just inline
            // handlers. Without this, calculators/forms that route through a helper silently fail.
            crate::widget_async::inject_splash_ui_handle(cx, self.vm_id, self.view.widget_uid());
            cx.widget_tree_mark_dirty(self.uid);
        }
    }
}

impl WidgetNode for Splash {
    fn widget_uid(&self) -> WidgetUid {
        self.uid
    }

    fn walk(&mut self, cx: &mut Cx) -> Walk {
        self.view.walk(cx)
    }

    fn area(&self) -> Area {
        self.view.area()
    }

    fn redraw(&mut self, cx: &mut Cx) {
        self.view.redraw(cx);
    }

    fn children(&self, visit: &mut dyn FnMut(LiveId, WidgetRef)) {
        self.view.children(visit);
    }
}

impl Drop for Splash {
    fn drop(&mut self) {
        // A Splash owns an isolate script VM. `Drop` has no `Cx`, so it can't free
        // the VM here; it just marks the id for reclamation. The isolate is torn
        // down later by `gc_dead_splash_isolates` (on the next isolate alloc, async
        // pump, or Splash event) while a `Cx` is available and nothing runs in it.
        crate::widget_async::mark_splash_isolate_dead(self.vm_id);
    }
}

impl Widget for Splash {
    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        // Handle tick timer — call tick() in the Splash code's scope
        if self.tick_timer.is_event(event).is_some() {
            self.call_fn(cx, id!(tick));
        }

        self.view.handle_event(cx, event, scope);
    }

    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        self.view.draw_walk(cx, scope, walk)
    }

    fn text(&self) -> String {
        self.body.as_ref().to_string()
    }

    fn set_text(&mut self, cx: &mut Cx, v: &str) {
        if self.body.as_ref() != v {
            self.body.set(v);
            self.eval_body(cx);
            // eval_body replaces self.view with a new View whose area is not
            // yet registered in the draw system, so self.redraw(cx) would be
            // a no-op.  Force a full redraw so the parent re-layouts.
            cx.redraw_all();
        }
    }
}

impl SplashRef {
    pub fn set_text(&self, cx: &mut Cx, v: &str) {
        if let Some(mut inner) = self.borrow_mut() {
            inner.set_text(cx, v);
        }
    }

    pub fn stream_begin(&self, cx: &mut Cx) {
        if let Some(mut inner) = self.borrow_mut() {
            inner.stream_begin(cx);
        }
    }

    pub fn stream_append(&self, cx: &mut Cx, chunk: &str) {
        if let Some(mut inner) = self.borrow_mut() {
            inner.stream_append(cx, chunk);
        }
    }
}
