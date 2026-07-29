use crate::makepad_draw::*;

// 月相 — the moon's lit fraction, drawn analytically rather than from artwork.
//
// `phase` is the position in the synodic cycle, 0..1: 0 new, 0.25 first quarter,
// 0.5 full, 0.75 last quarter. `sys.moonphase("phase")` supplies it.
//
// The terminator is the projection of the great circle dividing the lit half of
// the sphere from the dark half. Seen face-on it is a HALF-ELLIPSE whose width
// tracks cos(2*pi*phase), which is why a crescent's inner edge is curved while
// its outer edge is a true circular limb. Drawing it as two overlapping circles
// — the tempting shortcut — gives a lens-shaped crescent that is wrong at every
// phase except the quarters.
//
// A dark limb is rendered as faint earthshine rather than as a hole, so the disc
// still reads as a sphere at new moon instead of vanishing against the card.
script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.View

    mod.widgets.MoonPhase = View{
        width: 86
        height: 86
        show_bg: true
        draw_bg +: {
            phase: uniform(0.5)

            pixel: fn() {
                let sdf = Sdf2d.viewport(self.pos * self.rect_size)
                let w = self.rect_size.x
                let h = self.rect_size.y
                let r = min(w, h) * 0.44
                let cx = w * 0.5
                let cy = h * 0.5

                // Disc-local coordinates, normalised so the limb sits at 1.0.
                let dx = (self.pos.x * w - cx) / r
                let dy = (self.pos.y * h - cy) / r

                // k = cos(2*pi*phase): +1 at new, 0 at the quarters, -1 at full.
                // At height dy the terminator crosses x = k * sqrt(1 - dy^2).
                let k = cos(6.2831853 * self.phase)
                let s = sqrt(max(0.0, 1.0 - dy * dy))
                let tx = k * s

                // Waxing lights the RIGHT limb, waning the LEFT. Both branches
                // are evaluated and selected with mix rather than an if, because
                // MPSL `let` bindings are immutable — a running value cannot be
                // reassigned inside a branch.
                let aa = 1.6 / r
                let lit_wax = smoothstep(tx - aa, tx + aa, dx)
                let lit_wan = 1.0 - smoothstep(0.0 - tx - aa, 0.0 - tx + aa, dx)
                let lit = mix(lit_wan, lit_wax, step(self.phase, 0.5))

                let dark = vec3(0.145, 0.165, 0.216)
                let bright = vec3(0.965, 0.957, 0.910)

                // Maria — three broad, very low-contrast darkenings. Without
                // them a large disc reads as a flat token; with more contrast it
                // reads as a cartoon. Distance is measured in disc units so the
                // markings scale with the widget.
                let m1 = 1.0 - smoothstep(0.0, 0.44, length(vec2(dx + 0.22, dy + 0.20)))
                let m2 = 1.0 - smoothstep(0.0, 0.32, length(vec2(dx - 0.26, dy + 0.08)))
                let m3 = 1.0 - smoothstep(0.0, 0.28, length(vec2(dx + 0.06, dy - 0.34)))
                let maria = clamp(m1 * 0.55 + m2 * 0.42 + m3 * 0.38, 0.0, 1.0)
                let surface = mix(bright, bright * 0.88, maria)

                sdf.circle(cx, cy, r)
                sdf.fill(vec4(mix(dark, surface, lit), 1.0))
                return sdf.result
            }
        }
    }
}
