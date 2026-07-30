use crate::makepad_draw::*;

// The sun's daily path — a hairline arc from sunrise to sunset with the sun
// riding it at the current time, replacing a pair of blunt SUNRISE / SUNSET
// number tiles.
//
// `progress` is the fraction of daylight elapsed: 0 at sunrise, 1 at sunset.
// Outside 0..1 it is night, and the sun is parked at the nearer horizon and
// dimmed rather than hidden — an empty box reads as a failed fetch.
//
// The arc is the top of a large circle whose chord IS the horizon line, so the
// curve meets the horizon exactly at the sunrise and sunset ends instead of
// floating above them. Radius comes from the chord half-width and the rise:
// R = (half^2 + rise^2) / (2*rise).
script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.View

    mod.widgets.SunArc = View{
        width: Fill
        height: 96
        show_bg: true
        draw_bg +: {
            progress: uniform(0.5)

            pixel: fn() {
                let sdf = Sdf2d.viewport(self.pos * self.rect_size)
                let w = self.rect_size.x
                let h = self.rect_size.y

                let pad = w * 0.07
                let half = (w - pad * 2.0) * 0.5
                let rise = h * 0.52
                let hy = h * 0.74
                let rr = (half * half + rise * rise) / (2.0 * rise)
                let ccx = w * 0.5
                let ccy = hy - rise + rr

                // Hairline ring, clipped to the sky side of the horizon.
                sdf.circle(ccx, ccy, rr)
                sdf.circle(ccx, ccy, rr - 1.5)
                sdf.subtract()
                sdf.rect(0.0, 0.0, w, hy)
                sdf.intersect()
                sdf.fill(vec4(1.0, 1.0, 1.0, 0.30))

                // Horizon hairline, dimmer than the arc so it recedes.
                sdf.rect(pad, hy - 0.5, w - pad * 2.0, 1.0)
                sdf.fill(vec4(1.0, 1.0, 1.0, 0.16))

                // Sun position. The arc spans +/- theta about vertical, where
                // sin(theta) = half / rr.
                let t = clamp(self.progress, 0.0, 1.0)
                let th = asin(clamp(half / rr, 0.0, 1.0))
                let ph = 0.0 - th + 2.0 * th * t
                let sx = ccx + rr * sin(ph)
                let sy = ccy - rr * cos(ph)

                // Dim once the sun is below the horizon, and warm it near the
                // ends the way real low-angle light goes orange.
                let up = step(0.0, self.progress) * step(self.progress, 1.0)
                let low = 1.0 - smoothstep(0.0, 0.35, min(t, 1.0 - t))
                let disc = mix(vec3(1.0, 0.843, 0.365), vec3(1.0, 0.569, 0.216), low)

                sdf.circle(sx, sy, 6.0)
                sdf.fill(vec4(disc, mix(0.30, 1.0, up)))
                sdf.circle(sx, sy, 11.0)
                sdf.fill(vec4(disc, mix(0.04, 0.16, up)))
                return sdf.result
            }
        }
    }
}
