use crate::makepad_draw::*;

// 空气质量图 — a filled contour map of US-AQI over the area around a city.
//
// The WAQI/open-meteo tile overlay marks discrete monitoring STATIONS; air
// quality is a field, and a contour reads as one. The card samples a 4x4 grid of
// AQI readings (`sys.aqigrid`) and passes them as the 16 uniforms a0..a15,
// row-major with the NORTH row first. The shader interpolates them and colours
// the result by the EPA categories.
//
// Sixteen scalar uniforms rather than a texture because a `script_mod!` widget
// is shader-only — it has no Rust draw_walk in which to build and upload one —
// and because a card can already set float uniforms (`draw_bg.a0: …`) with no
// new DSL support. A 4x4 field is coarse, but AQI varies smoothly at city scale,
// so the interpolated surface is faithful; the grid is the sampling limit, not
// the render.
//
// Bilinear interpolation is done with the triangular-basis trick used in
// `temp_bar`: each grid line contributes max(0, 1-|x-k|), which sums to exactly
// linear interpolation between neighbours with no branches and no arrays —
// neither of which MPSL offers here.
script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.View

    mod.widgets.AqiContour = View{
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

                // Interpolate along each row, then between rows.
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
