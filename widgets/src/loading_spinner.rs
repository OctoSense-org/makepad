use crate::makepad_draw::*;

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.View

    mod.widgets.LoadingSpinner = View{
        width: Fill
        height: Fill
        show_bg: true
        draw_bg +: {
            color: uniform(theme.color_makepad)

            rotation_speed: uniform(1.2)
            border_size: uniform(20.0)
            max_gap_ratio: uniform(0.92)
            min_gap_ratio: uniform(0.12)
            stroke_width: uniform(3.0)

            pixel: fn() {
                let sdf = Sdf2d.viewport(self.pos * self.rect_size)

                let radius = min(self.rect_size.x * 0.5, self.rect_size.y * 0.5) - self.stroke_width * 0.5
                let center = self.rect_size * 0.5

                let rotation = self.draw_pass.time * self.rotation_speed * 2.0 * PI

                let rotation_cycles = rotation / (2.0 * PI)
                let arc_phase = modf(rotation_cycles * 0.5, 1.0)

                let expand_phase = clamp(arc_phase / 0.55, 0.0, 1.0)
                let contract_phase = clamp((arc_phase - 0.55) / 0.45, 0.0, 1.0)

                let cycle = expand_phase * (1.0 - contract_phase)

                let gap_ratio = mix(self.min_gap_ratio, self.max_gap_ratio, cycle)
                let gap_radians = gap_ratio * 2.0 * PI

                let start_angle = rotation

                sdf.arc_round_caps(
                    center.x
                    center.y
                    radius
                    start_angle
                    start_angle + 2.0 * PI - gap_radians
                    self.stroke_width
                )

                return sdf.fill(self.color)
            }
        }
    }

    // The Material 3 "loading indicator": a solid shape that continuously morphs
    // (circle <-> scalloped clover) while rotating, off draw_pass.time — the
    // shape-morph animation, not a spinner arc.
    mod.widgets.LoadingMorph = View{
        width: Fill
        height: Fill
        show_bg: true
        draw_bg +: {
            color: uniform(theme.color_makepad)

            pixel: fn() {
                let center = self.rect_size * 0.5
                let t = self.draw_pass.time

                // Rotate the sample point so the shape spins.
                let a = t * 1.4
                let cs = cos(a)
                let sn = sin(a)
                let rel = self.pos * self.rect_size - center
                let rot = vec2(rel.x * cs - rel.y * sn, rel.x * sn + rel.y * cs) + center
                let sdf = Sdf2d.viewport(rot)

                let base = min(center.x, center.y) * 0.62

                // Cycle a sequence of M3 shapes, morphing each into the next. Each
                // preset is (half-width, half-height, corner) as a fraction of base:
                // 0 circle, 1 square, 2 pill, 3 rounded-rect. All from sdf.box, so no
                // polar/atan (this MPSL has no 2-arg atan) and sdf.fill is the return.
                let ph = t * 0.5
                let ph_mod = ph - floor(ph / 4.0) * 4.0
                let k = floor(ph_mod)
                let e = smoothstep(0.0, 1.0, ph_mod - k)

                let mut wa = 1.0
                let mut ha = 1.0
                let mut ca = 1.0
                if k > 0.5 {
                    wa = 1.0
                    ha = 1.0
                    ca = 0.30
                }
                if k > 1.5 {
                    wa = 1.42
                    ha = 0.60
                    ca = 0.60
                }
                if k > 2.5 {
                    wa = 1.24
                    ha = 0.80
                    ca = 0.22
                }

                let mut k2 = k + 1.0
                if k2 > 3.5 {
                    k2 = 0.0
                }
                let mut wb = 1.0
                let mut hb = 1.0
                let mut cb = 1.0
                if k2 > 0.5 {
                    wb = 1.0
                    hb = 1.0
                    cb = 0.30
                }
                if k2 > 1.5 {
                    wb = 1.42
                    hb = 0.60
                    cb = 0.60
                }
                if k2 > 2.5 {
                    wb = 1.24
                    hb = 0.80
                    cb = 0.22
                }

                let w = mix(wa, wb, e) * base
                let h = mix(ha, hb, e) * base
                let corner = mix(ca, cb, e) * base

                sdf.box(center.x - w, center.y - h, w * 2.0, h * 2.0, corner)
                return sdf.fill(self.color)
            }
        }
    }
}
