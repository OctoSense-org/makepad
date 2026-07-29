use crate::makepad_draw::*;

// Temperature range bar for a forecast row, drawn by an SDF pixel shader.
//
// The bar spans the FULL width between the row's low and high labels: the cool
// end sits against the low reading, the warm end against the high one. Colour
// is a spectrum keyed to POSITION IN THE WEEK, not to absolute degrees —
// `lo`/`hi` are 0..1, already normalised by the card against the week's range.
// Keying it to absolute temperature instead makes a week that spans 28–39 °C
// sit entirely in the warm half, so every row draws the same orange.
//
//   lo: normalised position of this day's low   (0 = week's lowest)
//   hi: normalised position of this day's high  (1 = week's highest)
//
// The Android port of this widget is TempBarView in Splash-Android; the two
// must stay in step, because a2app's widget spec is the contract for both.
script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.View

    mod.widgets.TempBar = View{
        width: Fill
        height: 8
        show_bg: true
        draw_bg +: {
            // Raw degrees, not pre-normalised positions: a generated card can
            // pass sys.weathernum straight through, with the week's range as
            // wmin/wmax. Asking the model to normalise in the DSL invites
            // arithmetic mistakes for no benefit.
            tlo: uniform(0.0)
            thi: uniform(1.0)
            wmin: uniform(0.0)
            wmax: uniform(1.0)

            // Nine-stop cold→hot ramp, matching the Android view's palette.
            //
            // Inlined rather than a helper `fn`: `draw_bg +:` extends a FROZEN
            // prototype, and adding a new named function field to it fails at
            // runtime with "cannot push to frozen vec". Extra `uniform`s are
            // fine; extra functions are not.
            pixel: fn() {
                let sdf = Sdf2d.viewport(self.pos * self.rect_size)
                let h = self.rect_size.y
                // A rounded capsule spanning the whole widget; the row supplies
                // the width, so the bar always meets both labels.
                sdf.box(0.0, 0.0, self.rect_size.x, h, h * 0.5)

                // Normalise this day's low/high against the week, then pick the
                // spectrum colour at this pixel's position between them.
                let span = max(self.wmax - self.wmin, 0.001)
                let a = clamp((self.tlo - self.wmin) / span, 0.0, 1.0)
                let b = clamp((self.thi - self.wmin) / span, 0.0, 1.0)
                let x = clamp(mix(a, b, self.pos.x), 0.0, 1.0) * 8.0

                // Triangular-basis blend: each stop contributes
                // max(0, 1 - |x - k|), which is exactly linear interpolation
                // between neighbouring stops with no branches — MPSL `let`
                // bindings are immutable, so an if/else chain reassigning a
                // running colour will not compile.
                let col = vec3(0.118, 0.361, 1.000) * max(0.0, 1.0 - abs(x - 0.0))
                        + vec3(0.000, 0.639, 1.000) * max(0.0, 1.0 - abs(x - 1.0))
                        + vec3(0.000, 0.851, 0.753) * max(0.0, 1.0 - abs(x - 2.0))
                        + vec3(0.247, 0.749, 0.322) * max(0.0, 1.0 - abs(x - 3.0))
                        + vec3(0.776, 0.878, 0.086) * max(0.0, 1.0 - abs(x - 4.0))
                        + vec3(1.000, 0.769, 0.000) * max(0.0, 1.0 - abs(x - 5.0))
                        + vec3(1.000, 0.541, 0.000) * max(0.0, 1.0 - abs(x - 6.0))
                        + vec3(1.000, 0.294, 0.063) * max(0.0, 1.0 - abs(x - 7.0))
                        + vec3(0.878, 0.106, 0.106) * max(0.0, 1.0 - abs(x - 8.0))
                sdf.fill(vec4(col, 1.0))
                return sdf.result
            }
        }
    }
}
