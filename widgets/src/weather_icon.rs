use crate::makepad_draw::*;

// Animated weather-condition icon, drawn entirely by an SDF pixel shader driven
// by `self.draw_pass.time` (continuous ~60fps animation; the host `Splash`
// widget's redraw pump keeps it ticking — it triggers on the `WeatherIcon`
// name in a card body). One `cond` uniform selects the condition so a generated
// card only sets a number:
//   0 sunny/clear   1 partly cloudy   2 cloudy/overcast   3 rain
//   4 thunderstorm  5 snow            6 wind              7 fog/haze
// Transparent background (no `sdf.clear`) so it floats over the card's photo.
script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.View

    mod.widgets.WeatherIcon = View{
        width: 96
        height: 96
        show_bg: true
        draw_bg +: {
            cond: uniform(0.0)
            // Mood-owned MONO ink: when it carries alpha, the finished glyph is
            // recoloured to a single-ink silhouette — the line-art restraint a
            // photo-mood card asks for instead of the filled multi-colour icon.
            // Alpha 0 (default) keeps the legacy colours.
            mono_ink: uniform(vec4(0.0))

            pixel: fn() {
                let sdf = Sdf2d.viewport(self.pos * self.rect_size)
                let t = self.draw_pass.time
                let w = self.rect_size.x
                let h = self.rect_size.y
                let c = self.cond

                if c < 0.5 {
                    // sunny — rotating rays + two-tone disc
                    let cx = w * 0.5
                    let cy = h * 0.5
                    let r = w * 0.12
                    sdf.rotate(t * 0.5, cx, cy)
                    sdf.box(cx - w*0.012, cy - r*3.0, w*0.024, r*1.1, w*0.012)  sdf.fill(#xffd36b)
                    sdf.rotate(0.7854, cx, cy)  sdf.box(cx - w*0.012, cy - r*3.0, w*0.024, r*1.1, w*0.012)  sdf.fill(#xffd36b)
                    sdf.rotate(0.7854, cx, cy)  sdf.box(cx - w*0.012, cy - r*3.0, w*0.024, r*1.1, w*0.012)  sdf.fill(#xffd36b)
                    sdf.rotate(0.7854, cx, cy)  sdf.box(cx - w*0.012, cy - r*3.0, w*0.024, r*1.1, w*0.012)  sdf.fill(#xffd36b)
                    sdf.rotate(0.7854, cx, cy)  sdf.box(cx - w*0.012, cy - r*3.0, w*0.024, r*1.1, w*0.012)  sdf.fill(#xffd36b)
                    sdf.rotate(0.7854, cx, cy)  sdf.box(cx - w*0.012, cy - r*3.0, w*0.024, r*1.1, w*0.012)  sdf.fill(#xffd36b)
                    sdf.rotate(0.7854, cx, cy)  sdf.box(cx - w*0.012, cy - r*3.0, w*0.024, r*1.1, w*0.012)  sdf.fill(#xffd36b)
                    sdf.rotate(0.7854, cx, cy)  sdf.box(cx - w*0.012, cy - r*3.0, w*0.024, r*1.1, w*0.012)  sdf.fill(#xffd36b)
                    sdf.circle(cx, cy, r*1.5)  sdf.fill(#xffb63c)
                    sdf.circle(cx, cy, r*1.1)  sdf.fill(#xffd06a)
                } else if c < 1.5 {
                    // partly cloudy — small sun + cloud
                    let sx = w*0.36  let sy = h*0.34  let sr = w*0.10
                    sdf.rotate(t*0.4, sx, sy)
                    sdf.box(sx-w*0.01, sy-sr*2.6, w*0.02, sr*1.0, w*0.01) sdf.fill(#xffd36b)
                    sdf.rotate(1.256, sx, sy) sdf.box(sx-w*0.01, sy-sr*2.6, w*0.02, sr*1.0, w*0.01) sdf.fill(#xffd36b)
                    sdf.rotate(1.256, sx, sy) sdf.box(sx-w*0.01, sy-sr*2.6, w*0.02, sr*1.0, w*0.01) sdf.fill(#xffd36b)
                    sdf.rotate(1.256, sx, sy) sdf.box(sx-w*0.01, sy-sr*2.6, w*0.02, sr*1.0, w*0.01) sdf.fill(#xffd36b)
                    sdf.rotate(1.256, sx, sy) sdf.box(sx-w*0.01, sy-sr*2.6, w*0.02, sr*1.0, w*0.01) sdf.fill(#xffd36b)
                    sdf.circle(sx, sy, sr*1.4) sdf.fill(#xffc247)
                    // undo the accumulated sun-ray rotation (t*0.4 + 4*1.256) about
                    // the sun pivot so the cloud below is drawn in the unrotated frame
                    // and stays put instead of orbiting the sun.
                    sdf.rotate(0.0 - t*0.4 - 5.024, sx, sy)
                    let cx = w*0.56  let cyy = h*0.56  let r = w*0.13
                    sdf.circle(cx - r*1.1, cyy + r*0.2, r*0.9) sdf.fill(#xc4cede)
                    sdf.circle(cx + r*0.2, cyy - r*0.5, r*1.15) sdf.fill(#xdbe6f5)
                    sdf.circle(cx + r*1.2, cyy + r*0.15, r*0.85) sdf.fill(#xc4cede)
                    sdf.box(cx - r*1.9, cyy + r*0.1, r*3.8, r*1.0, r*0.5) sdf.fill(#xd4e0f0)
                } else if c < 2.5 {
                    // cloudy — two drifting clouds
                    let d = sin(t*0.9)*w*0.04
                    let ax = w*0.40+d  let ay = h*0.42  let r = w*0.12
                    sdf.circle(ax - r*1.1, ay + r*0.2, r*0.9) sdf.fill(#xaab6c8)
                    sdf.circle(ax + r*0.2, ay - r*0.5, r*1.15) sdf.fill(#xbcc8da)
                    sdf.box(ax - r*1.9, ay + r*0.1, r*3.8, r*1.0, r*0.5) sdf.fill(#xb2becf)
                    let bx = w*0.58-d  let by = h*0.56  let r2 = w*0.14
                    sdf.circle(bx - r2*1.1, by + r2*0.2, r2*0.9) sdf.fill(#xc4cede)
                    sdf.circle(bx + r2*0.2, by - r2*0.5, r2*1.15) sdf.fill(#xdbe6f5)
                    sdf.circle(bx + r2*1.2, by + r2*0.15, r2*0.85) sdf.fill(#xc4cede)
                    sdf.box(bx - r2*1.9, by + r2*0.1, r2*3.8, r2*1.0, r2*0.5) sdf.fill(#xd4e0f0)
                } else if c < 3.5 {
                    // rain — cloud + falling drops
                    let cx = w*0.5  let cyy = h*0.34  let r = w*0.16
                    sdf.circle(cx - r*1.1, cyy + r*0.2, r*0.9) sdf.fill(#xc4cede)
                    sdf.circle(cx + r*0.2, cyy - r*0.5, r*1.15) sdf.fill(#xdbe6f5)
                    sdf.circle(cx + r*1.2, cyy + r*0.15, r*0.85) sdf.fill(#xc4cede)
                    sdf.box(cx - r*1.9, cyy + r*0.1, r*3.8, r*1.0, r*0.5) sdf.fill(#xd4e0f0)
                    // The travel must END inside the box: a drop is 15 tall and is drawn
                    // DOWNWARD from y0, so the last frame's bottom edge is
                    // base + span + 15. This read `+ 24.0`, which put that edge 39px
                    // BELOW the widget — the drops fell out of their own row and
                    // landed above the next row's cloud, where they read as its rain.
                    let base = cyy + r*1.2  let span = max(0.0, h - base - 15.0)
                    let y0 = base + fract(t*0.95)*span       sdf.box(cx - w*0.22, y0, 3.0, 15.0, 1.5) sdf.fill(#x6db6ff)
                    let y1 = base + fract(t*1.15+0.35)*span   sdf.box(cx - w*0.06, y1, 3.0, 15.0, 1.5) sdf.fill(#x6db6ff)
                    let y2 = base + fract(t*0.85+0.62)*span   sdf.box(cx + w*0.10, y2, 3.0, 15.0, 1.5) sdf.fill(#x6db6ff)
                    let y3 = base + fract(t*1.05+0.20)*span   sdf.box(cx + w*0.22, y3, 3.0, 15.0, 1.5) sdf.fill(#x6db6ff)
                } else if c < 4.5 {
                    // thunderstorm — dark cloud + lightning flash + drops
                    let cx = w*0.5  let cyy = h*0.3  let r = w*0.15
                    sdf.circle(cx - r*1.1, cyy + r*0.2, r*0.9) sdf.fill(#x9aa6b8)
                    sdf.circle(cx + r*0.2, cyy - r*0.5, r*1.15) sdf.fill(#xb0bccd)
                    sdf.circle(cx + r*1.2, cyy + r*0.15, r*0.85) sdf.fill(#x9aa6b8)
                    sdf.box(cx - r*1.9, cyy + r*0.1, r*3.8, r*1.0, r*0.5) sdf.fill(#xa6b2c3)
                    let base = cyy + r*1.2
                    let fl = fract(t*0.7)
                    if fl < 0.14 {
                        sdf.move_to(cx-w*0.02, base) sdf.line_to(cx-w*0.09, base+h*0.22) sdf.line_to(cx-w*0.01, base+h*0.22)
                        sdf.line_to(cx-w*0.07, base+h*0.46) sdf.line_to(cx+w*0.08, base+h*0.16) sdf.line_to(cx+w*0.0, base+h*0.16)
                        sdf.line_to(cx+w*0.06, base) sdf.close_path() sdf.fill(#xffd23c)
                    }
                    let span = max(0.0, h - base - 13.0)   // drop is 13 tall; keep its bottom inside h
                    let y0 = base + fract(t*1.1)*span       sdf.box(cx-w*0.20, y0, 3.0, 13.0, 1.5) sdf.fill(#x6db6ff)
                    let y1 = base + fract(t*1.3+0.5)*span   sdf.box(cx+w*0.16, y1, 3.0, 13.0, 1.5) sdf.fill(#x6db6ff)
                } else if c < 5.5 {
                    // snow — cloud + drifting flakes
                    let cx = w*0.5  let cyy = h*0.3  let r = w*0.15
                    sdf.circle(cx - r*1.1, cyy + r*0.2, r*0.9) sdf.fill(#xc4cede)
                    sdf.circle(cx + r*0.2, cyy - r*0.5, r*1.15) sdf.fill(#xdbe6f5)
                    sdf.circle(cx + r*1.2, cyy + r*0.15, r*0.85) sdf.fill(#xc4cede)
                    sdf.box(cx - r*1.9, cyy + r*0.1, r*3.8, r*1.0, r*0.5) sdf.fill(#xd4e0f0)
                    let base = cyy + r*1.3  let span = max(0.0, h - base - 4.5)   // flake radius 4.5
                    let f0 = fract(t*0.45)       sdf.circle(cx-w*0.18 + sin(f0*6.28)*w*0.03, base+f0*span, 4.5) sdf.fill(#xffffff)
                    let f1 = fract(t*0.38+0.4)   sdf.circle(cx+w*0.02 + sin(f1*6.28)*w*0.03, base+f1*span, 4.5) sdf.fill(#xffffff)
                    let f2 = fract(t*0.5+0.7)    sdf.circle(cx+w*0.17 + sin(f2*6.28)*w*0.03, base+f2*span, 4.5) sdf.fill(#xffffff)
                } else if c < 6.5 {
                    // wind — drifting gust lines
                    let sp = w + 80.0
                    let x0 = fract(t*0.35) * sp - 40.0
                    sdf.box(x0 - w*0.22, h*0.34, w*0.4, 6.0, 3.0)  sdf.fill(#xdfe9f5)
                    sdf.circle(x0 + w*0.18, h*0.34 + 3.0, 9.0)  sdf.stroke(#xdfe9f5, 5.0)
                    let x1 = fract(t*0.5 + 0.4) * sp - 40.0
                    sdf.box(x1 - w*0.26, h*0.52, w*0.46, 6.0, 3.0)  sdf.fill(#xc6d6ea)
                    sdf.circle(x1 + w*0.20, h*0.52 + 3.0, 9.0)  sdf.stroke(#xc6d6ea, 5.0)
                    let x2 = fract(t*0.42 + 0.7) * sp - 40.0
                    sdf.box(x2 - w*0.18, h*0.68, w*0.34, 6.0, 3.0)  sdf.fill(#xd2e0f0)
                } else {
                    // fog/haze — cloud + drifting bars
                    let cx = w*0.5  let cyy = h*0.28  let r = w*0.14
                    sdf.circle(cx - r*1.1, cyy + r*0.2, r*0.9) sdf.fill(#xb6bfcc)
                    sdf.circle(cx + r*0.2, cyy - r*0.5, r*1.15) sdf.fill(#xc8d2df)
                    sdf.box(cx - r*1.9, cyy + r*0.1, r*3.8, r*1.0, r*0.5) sdf.fill(#xbfc9d6)
                    let b = cyy + r*1.5
                    sdf.box(cx - w*0.28 + sin(t*0.9)*w*0.05, b, w*0.5, 6.0, 3.0) sdf.fill(#xdbe6f5)
                    sdf.box(cx - w*0.22 + sin(t*0.9+1.2)*w*0.05, b+h*0.13, w*0.44, 6.0, 3.0) sdf.fill(#xc6d2e2)
                    sdf.box(cx - w*0.26 + sin(t*0.9+2.3)*w*0.05, b+h*0.26, w*0.48, 6.0, 3.0) sdf.fill(#xd0dcec)
                }
                let mono = step(0.001, self.mono_ink.w)
                // Premultiplied, like everything Sdf2d returns: an
                // unpremultiplied ink here painted the whole quad solid.
                let ma = sdf.result.w * self.mono_ink.w
                return mix(sdf.result, vec4(self.mono_ink.xyz * ma, ma), mono)
            }
        }
    }
}
