use crate::{makepad_derive_widget::*, makepad_draw::*, view::View, widget::*};

// 空气质量图 — a filled contour map of US-AQI over the area around a place.
//
// A tile overlay (what `sys.airmap` returns) only marks discrete monitoring
// STATIONS and over most cities is very nearly empty; air quality is a field, and
// a contour reads as one. The shader interpolates a 4x4 grid of readings and
// colours the result by the EPA categories, with isolines at each band boundary.
//
// ## Why this is a Rust widget and not a `script_mod!` shader
//
// It used to be shader-only, which meant the CARD had to supply the field — as
// sixteen scalar uniforms:
//
//     AqiContour{ draw_bg.a0: sys.aqigrid(LAT, LON, 1.6, 0)
//                 draw_bg.a1: sys.aqigrid(LAT, LON, 1.6, 1)   ... through a15 }
//
// That put a GPU uniform layout into the authoring language. It is not a widget
// contract, it is an ABI, and it leaked for a mechanical reason: a `script_mod!`
// widget is shader-only, so there was no Rust `draw_walk` in which to fetch
// anything. A backend without shaders then had to pattern-match sixteen magic
// attribute names back into an array.
//
// So the widget now takes the SEMANTIC arguments and does the fetch itself:
//
//     AqiContour{ width: Fill height: 190 lat: … lon: … span: 1.6 }
//
// `draw_walk` builds the same multi-location request, reads the sixteen values
// out of one cached response, and writes the uniforms via `set_uniform`. The
// shader below is unchanged — same pixels, same cost. What changed is that a card
// can no longer express the field wrongly, and a non-shader backend gets one
// array attribute to map instead of sixteen names to recognise.
#[derive(Script, Widget)]
pub struct AqiContour {
    #[source]
    source: ScriptObjectRef,

    #[deref]
    view: View,

    /// Centre of the sampled area. Pass `sys.geocodenum(place, "lat"/"lon")` —
    /// never typed digits. Both are -9999 while that lookup is in flight, which
    /// this treats as "not ready" rather than sampling the Null Island ocean.
    #[live]
    lat: f64,
    #[live]
    lon: f64,

    /// Width of the sampled square in degrees. 1.6 suits a city.
    #[live(1.6)]
    span: f64,
}

/// Grid edge: 4x4 = the sixteen `a0..a15` uniforms the shader interpolates.
const N: usize = 4;

impl ScriptHook for AqiContour {}

impl Widget for AqiContour {
    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        self.view.handle_event(cx, event, scope);
    }

    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        self.resolve_field(cx);
        self.view.draw_walk(cx, scope, walk)
    }
}

impl AqiContour {
    /// Fetch the AQI field and push it into the shader's uniforms.
    ///
    /// All sixteen cells come from ONE multi-location open-meteo request — that
    /// API accepts comma-separated coordinates and answers with an array — so the
    /// whole field costs a single cached fetch. Returns quietly while the fetch
    /// (or the geocode feeding it) is still in flight; `script_data_fetch`
    /// redraws when the response lands, so the next draw fills it in.
    fn resolve_field(&mut self, cx: &mut Cx2d) {
        // -9999 is sys.geocodenum's loading sentinel. Sampling it would fetch a
        // point in the Atlantic and cache the answer under a bogus URL.
        if self.lat <= -900.0 || self.lon <= -900.0 {
            return;
        }
        let step = self.span / (N as f64 - 1.0);
        let mut lats = Vec::with_capacity(N * N);
        let mut lons = Vec::with_capacity(N * N);
        for r in 0..N {
            for c in 0..N {
                // Row 0 is the NORTH edge, so latitude DECREASES with the row
                // index — matching how the shader walks self.pos.y downward.
                lats.push(format!("{:.4}", self.lat + self.span / 2.0 - r as f64 * step));
                lons.push(format!("{:.4}", self.lon - self.span / 2.0 + c as f64 * step));
            }
        }
        let url = format!(
            "https://air-quality-api.open-meteo.com/v1/air-quality?latitude={}&longitude={}\
&current=us_aqi&timezone=auto",
            lats.join(","),
            lons.join(",")
        );
        let Some(bytes) = cx.script_data_fetch(&url) else {
            return;
        };
        let Ok(root) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            return;
        };
        for i in 0..(N * N) {
            let v = root
                .get(i)
                .and_then(|o| o.get("current"))
                .and_then(|c| c.get("us_aqi"))
                .and_then(|n| n.as_f64())
                .unwrap_or(0.0) as f32;
            // Uniform ids are a0..a15, matching the shader below.
            let id = LiveId::from_str(&format!("a{i}"));
            self.view.draw_bg.draw_vars.set_uniform(cx, id, &[v]);
        }
    }
}

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.View

    mod.widgets.AqiContourBase = #(AqiContour::register_widget(vm))

    mod.widgets.AqiContour = mod.widgets.AqiContourBase{
        width: Fill
        height: 210
        show_bg: true
        draw_bg +: {
            a0: uniform(0.0)   a1: uniform(0.0)   a2: uniform(0.0)   a3: uniform(0.0)
            a4: uniform(0.0)   a5: uniform(0.0)   a6: uniform(0.0)   a7: uniform(0.0)
            a8: uniform(0.0)   a9: uniform(0.0)   a10: uniform(0.0)  a11: uniform(0.0)
            a12: uniform(0.0)  a13: uniform(0.0)  a14: uniform(0.0)  a15: uniform(0.0)

            pixel: fn() {
                let sdf = Sdf2d.viewport(self.pos * self.rect_size)
                let w = self.rect_size.x
                let h = self.rect_size.y

                // Grid coordinates, 0..3 across and down.
                let u = clamp(self.pos.x, 0.0, 1.0) * 3.0
                let v = clamp(self.pos.y, 0.0, 1.0) * 3.0

                // Bilinear interpolation via the triangular-basis trick used in
                // `temp_bar`: each grid line contributes max(0, 1-|x-k|), which
                // sums to exactly linear interpolation between neighbours with no
                // branches and no arrays — neither of which MPSL offers here.
                let bu = vec4(
                    max(0.0, 1.0 - abs(u - 0.0)),
                    max(0.0, 1.0 - abs(u - 1.0)),
                    max(0.0, 1.0 - abs(u - 2.0)),
                    max(0.0, 1.0 - abs(u - 3.0))
                )
                let bv = vec4(
                    max(0.0, 1.0 - abs(v - 0.0)),
                    max(0.0, 1.0 - abs(v - 1.0)),
                    max(0.0, 1.0 - abs(v - 2.0)),
                    max(0.0, 1.0 - abs(v - 3.0))
                )

                let r0 = dot(vec4(self.a0, self.a1, self.a2, self.a3), bu)
                let r1 = dot(vec4(self.a4, self.a5, self.a6, self.a7), bu)
                let r2 = dot(vec4(self.a8, self.a9, self.a10, self.a11), bu)
                let r3 = dot(vec4(self.a12, self.a13, self.a14, self.a15), bu)
                let aqi = dot(vec4(r0, r1, r2, r3), bv)

                // EPA band index 0..5 by counting crossed breakpoints. Bands are
                // DISCRETE on purpose: a continuous ramp reads as a heatmap blob
                // and hides exactly the boundaries the map exists to show.
                let bi = step(50.0, aqi) + step(100.0, aqi) + step(150.0, aqi)
                       + step(200.0, aqi) + step(300.0, aqi)

                let col = vec3(0.000, 0.894, 0.000) * max(0.0, 1.0 - abs(bi - 0.0))
                        + vec3(1.000, 1.000, 0.000) * max(0.0, 1.0 - abs(bi - 1.0))
                        + vec3(1.000, 0.494, 0.000) * max(0.0, 1.0 - abs(bi - 2.0))
                        + vec3(1.000, 0.000, 0.000) * max(0.0, 1.0 - abs(bi - 3.0))
                        + vec3(0.561, 0.247, 0.592) * max(0.0, 1.0 - abs(bi - 4.0))
                        + vec3(0.494, 0.000, 0.137) * max(0.0, 1.0 - abs(bi - 5.0))

                // Isolines at each breakpoint. The window is in AQI units, so a
                // steep gradient draws a thin line and a flat field draws none —
                // which is the honest result when the air is uniform.
                let iso = clamp(
                      (1.0 - smoothstep(0.0, 2.0, abs(aqi - 50.0)))
                    + (1.0 - smoothstep(0.0, 2.5, abs(aqi - 100.0)))
                    + (1.0 - smoothstep(0.0, 3.0, abs(aqi - 150.0)))
                    + (1.0 - smoothstep(0.0, 3.5, abs(aqi - 200.0)))
                    + (1.0 - smoothstep(0.0, 5.0, abs(aqi - 300.0))), 0.0, 1.0)

                // Darken slightly within a band so the surface has some relief
                // instead of reading as flat paint.
                let shade = mix(0.86, 1.0, clamp(fract(aqi / 50.0), 0.0, 1.0))
                let body = mix(col * shade, vec3(1.0, 1.0, 1.0), iso * 0.55)

                sdf.box(0.0, 0.0, w, h, 18.0)
                sdf.fill(vec4(body, 0.82))
                return sdf.result
            }
        }
    }
}
