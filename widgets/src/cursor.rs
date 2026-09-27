use makepad_draw::*;

// Vector cursors stay sharp at the display scale and composite on the GPU.
script_mod! {
    use mod.prelude.widgets_internal.*
    let CursorShape = #(CursorShape::script_api(vm))

    mod.widgets.DrawMouseCursor = mod.draw.DrawQuad {
        shape: instance(CursorShape.Default)
        color: uniform(theme.color_black)
        border_color: uniform(theme.color_white)

        // Closed cursor silhouettes are concave. Sdf2d's line-path clipping
        // intersects edge half-planes, which cannot fill those silhouettes.
        // Carry nearest-edge distance and signed winding for the complete path.
        path_edge: fn(path: vec2, a: vec2, b: vec2) -> vec2 {
            let p = self.pos * 24.0
            let edge = b - a
            let rel = p - a
            let t = clamp(dot(rel, edge) / max(dot(edge, edge), 0.00001), 0.0, 1.0)
            let distance = min(path.x, length(rel - edge * t))
            let side = edge.x * rel.y - edge.y * rel.x
            var winding = path.y
            if a.y <= p.y && b.y > p.y && side > 0.0 {
                winding += 1.0
            }
            if a.y > p.y && b.y <= p.y && side < 0.0 {
                winding -= 1.0
            }
            return vec2(distance, winding)
        }

        path_pixel: fn(path: vec2) -> vec4 {
            let sdf = Sdf2d.viewport(self.pos * 24.0)
            sdf.shape = path.x
            if abs(path.y) > 0.5 {
                sdf.shape = -path.x
            }
            sdf.fill_keep(self.color)
            sdf.stroke(self.border_color, 1.25)
            return sdf.result
        }

        arrow: fn() -> vec4 {
            var path = vec2(1e20, 0.0)
            path = self.path_edge(path, vec2(3.0, 2.0), vec2(3.0, 19.0))
            path = self.path_edge(path, vec2(3.0, 19.0), vec2(7.5, 15.0))
            path = self.path_edge(path, vec2(7.5, 15.0), vec2(11.0, 22.0))
            path = self.path_edge(path, vec2(11.0, 22.0), vec2(14.0, 20.5))
            path = self.path_edge(path, vec2(14.0, 20.5), vec2(10.5, 13.5))
            path = self.path_edge(path, vec2(10.5, 13.5), vec2(17.0, 13.5))
            path = self.path_edge(path, vec2(17.0, 13.5), vec2(3.0, 2.0))
            return self.path_pixel(path)
        }

        hand: fn(kind: float) -> vec4 {
            var path = vec2(1e20, 0.0)
            if kind < 0.5 {
                // Pointing finger: hotspot at its tip.
                path = self.path_edge(path, vec2(8.0, 12.0), vec2(8.0, 4.0))
                path = self.path_edge(path, vec2(8.0, 4.0), vec2(9.0, 2.0))
                path = self.path_edge(path, vec2(9.0, 2.0), vec2(11.0, 2.0))
                path = self.path_edge(path, vec2(11.0, 2.0), vec2(12.0, 4.0))
                path = self.path_edge(path, vec2(12.0, 4.0), vec2(12.0, 10.0))
                path = self.path_edge(path, vec2(12.0, 10.0), vec2(15.0, 9.0))
                path = self.path_edge(path, vec2(15.0, 9.0), vec2(20.0, 12.0))
                path = self.path_edge(path, vec2(20.0, 12.0), vec2(20.0, 16.0))
                path = self.path_edge(path, vec2(20.0, 16.0), vec2(17.0, 22.0))
                path = self.path_edge(path, vec2(17.0, 22.0), vec2(9.0, 22.0))
                path = self.path_edge(path, vec2(9.0, 22.0), vec2(3.0, 14.0))
                path = self.path_edge(path, vec2(3.0, 14.0), vec2(3.0, 12.0))
                path = self.path_edge(path, vec2(3.0, 12.0), vec2(5.0, 11.0))
                path = self.path_edge(path, vec2(5.0, 11.0), vec2(8.0, 12.0))
            } else {
                let top = 4.0 + (kind - 1.0) * 5.0
                path = self.path_edge(path, vec2(6.0, 13.0), vec2(5.0, top + 1.0))
                path = self.path_edge(path, vec2(5.0, top + 1.0), vec2(8.0, top))
                path = self.path_edge(path, vec2(8.0, top), vec2(9.0, 11.0))
                path = self.path_edge(path, vec2(9.0, 11.0), vec2(9.0, top - 1.0))
                path = self.path_edge(path, vec2(9.0, top - 1.0), vec2(12.0, top - 1.0))
                path = self.path_edge(path, vec2(12.0, top - 1.0), vec2(12.0, 11.0))
                path = self.path_edge(path, vec2(12.0, 11.0), vec2(13.0, top))
                path = self.path_edge(path, vec2(13.0, top), vec2(16.0, top))
                path = self.path_edge(path, vec2(16.0, top), vec2(16.0, 12.0))
                path = self.path_edge(path, vec2(16.0, 12.0), vec2(17.0, top + 2.0))
                path = self.path_edge(path, vec2(17.0, top + 2.0), vec2(20.0, top + 2.0))
                path = self.path_edge(path, vec2(20.0, top + 2.0), vec2(20.0, 16.0))
                path = self.path_edge(path, vec2(20.0, 16.0), vec2(17.0, 22.0))
                path = self.path_edge(path, vec2(17.0, 22.0), vec2(8.0, 22.0))
                path = self.path_edge(path, vec2(8.0, 22.0), vec2(2.0, 15.0))
                path = self.path_edge(path, vec2(2.0, 15.0), vec2(3.0, 12.0))
                path = self.path_edge(path, vec2(3.0, 12.0), vec2(6.0, 13.0))
            }
            return self.path_pixel(path)
        }

        resize: fn(axis: vec2, double_head: float, divider: float) -> vec4 {
            let p = self.pos * 24.0 - vec2(12.0)
            let q = vec2(dot(p, axis), dot(p, vec2(-axis.y, axis.x)))
            let sdf = Sdf2d.viewport(q)
            sdf.move_to(-8.0, 0.0)
            sdf.line_to(8.0, 0.0)
            sdf.move_to(4.0, -4.0)
            sdf.line_to(8.0, 0.0)
            sdf.line_to(4.0, 4.0)
            if double_head > 0.5 {
                sdf.move_to(-4.0, -4.0)
                sdf.line_to(-8.0, 0.0)
                sdf.line_to(-4.0, 4.0)
            }
            if divider > 0.5 {
                sdf.move_to(-1.5, -8.0)
                sdf.line_to(-1.5, 8.0)
                sdf.move_to(1.5, -8.0)
                sdf.line_to(1.5, 8.0)
            }
            sdf.stroke_keep(self.border_color, 3.5)
            sdf.stroke(self.color, 1.75)
            return sdf.result
        }

        // The match only picks a figure and its parameters; each figure is
        // drawn from one call site below. Every inlined call is compiled
        // again, and the match used to hold arrow() 4x, hand() 3x (31 path
        // edges each) and resize() 19x: 0.86 s to compile on Adreno 630.
        pixel: fn() -> vec4 {
            let sdf = Sdf2d.viewport(self.pos * 24.0)
            // 0: strokes built in sdf; 1: arrow; 2: hand; 3: resize;
            // 4: wait; 5: move (two resize arrows); 6: help (strokes over the arrow)
            var figure = 0.0
            var hand_kind = 0.0
            var axis = vec2(1.0, 0.0)
            var double_head = 0.0
            var divider = 0.0
            match self.shape {
                CursorShape.Hidden => { return vec4(0.0) }
                CursorShape.Default => { figure = 1.0 }
                CursorShape.Arrow => { figure = 1.0 }
                CursorShape.Hand => { figure = 2.0 }
                CursorShape.Grab => {
                    figure = 2.0
                    hand_kind = 1.0
                }
                CursorShape.Grabbing => {
                    figure = 2.0
                    hand_kind = 2.0
                }
                CursorShape.Text => {
                    sdf.move_to(8.0, 3.0)
                    sdf.line_to(16.0, 3.0)
                    sdf.move_to(12.0, 3.0)
                    sdf.line_to(12.0, 21.0)
                    sdf.move_to(8.0, 21.0)
                    sdf.line_to(16.0, 21.0)
                }
                CursorShape.Crosshair => {
                    sdf.move_to(12.0, 2.0)
                    sdf.line_to(12.0, 22.0)
                    sdf.move_to(2.0, 12.0)
                    sdf.line_to(22.0, 12.0)
                }
                CursorShape.Move => {
                    figure = 5.0
                    double_head = 1.0
                }
                CursorShape.Wait => { figure = 4.0 }
                CursorShape.NotAllowed => {
                    sdf.circle(12.0, 12.0, 8.0)
                    sdf.move_to(6.5, 6.5)
                    sdf.line_to(17.5, 17.5)
                }
                CursorShape.Help => {
                    sdf.move_to(15.0, 13.0)
                    sdf.line_to(16.0, 11.0)
                    sdf.line_to(20.0, 11.0)
                    sdf.line_to(22.0, 13.0)
                    sdf.line_to(21.0, 15.0)
                    sdf.line_to(18.0, 17.0)
                    sdf.line_to(18.0, 18.0)
                    sdf.move_to(18.0, 20.5)
                    sdf.line_to(18.0, 21.0)
                    figure = 6.0
                }
                CursorShape.NResize => {
                    figure = 3.0
                    axis = vec2(0.0, -1.0)
                }
                CursorShape.NeResize => {
                    figure = 3.0
                    axis = vec2(0.707107, -0.707107)
                }
                CursorShape.EResize => {
                    figure = 3.0
                    axis = vec2(1.0, 0.0)
                }
                CursorShape.SeResize => {
                    figure = 3.0
                    axis = vec2(0.707107, 0.707107)
                }
                CursorShape.SResize => {
                    figure = 3.0
                    axis = vec2(0.0, 1.0)
                }
                CursorShape.SwResize => {
                    figure = 3.0
                    axis = vec2(-0.707107, 0.707107)
                }
                CursorShape.WResize => {
                    figure = 3.0
                    axis = vec2(-1.0, 0.0)
                }
                CursorShape.NwResize => {
                    figure = 3.0
                    axis = vec2(-0.707107, -0.707107)
                }
                CursorShape.NsResize => {
                    figure = 3.0
                    axis = vec2(0.0, 1.0)
                    double_head = 1.0
                }
                CursorShape.NeswResize => {
                    figure = 3.0
                    axis = vec2(0.707107, -0.707107)
                    double_head = 1.0
                }
                CursorShape.EwResize => {
                    figure = 3.0
                    axis = vec2(1.0, 0.0)
                    double_head = 1.0
                }
                CursorShape.NwseResize => {
                    figure = 3.0
                    axis = vec2(0.707107, 0.707107)
                    double_head = 1.0
                }
                CursorShape.ColResize => {
                    figure = 3.0
                    axis = vec2(1.0, 0.0)
                    double_head = 1.0
                    divider = 1.0
                }
                CursorShape.RowResize => {
                    figure = 3.0
                    axis = vec2(0.0, 1.0)
                    double_head = 1.0
                    divider = 1.0
                }
                _ => { figure = 1.0 }
            }

            // Text, crosshair, not-allowed and the help mark: stroked paths.
            var over = vec4(0.0)
            if figure < 0.5 || figure > 5.5 {
                sdf.stroke_keep(self.border_color, 3.5)
                sdf.stroke(self.color, 1.75)
                if figure < 0.5 {
                    return sdf.result
                }
                over = sdf.result
            }
            // Resize arrows; move draws the horizontal one over the vertical.
            if figure > 2.5 && figure < 3.5 || figure > 4.5 && figure < 5.5 {
                let passes = if figure > 4.5 {2.0} else {1.0}
                var result = vec4(0.0)
                var pass = 0.0
                loop {
                    if pass >= passes { break }
                    let pass_axis = if pass > 0.5 {vec2(0.0, 1.0)} else {axis}
                    let r = self.resize(pass_axis, double_head, divider)
                    result = result + r * (1.0 - result.w)
                    pass = pass + 1.0
                }
                return result
            }
            if figure > 1.5 && figure < 2.5 {
                return self.hand(hand_kind)
            }
            if figure > 3.5 && figure < 4.5 {
                var path = vec2(1e20, 0.0)
                path = self.path_edge(path, vec2(6.0, 3.0), vec2(18.0, 3.0))
                path = self.path_edge(path, vec2(18.0, 3.0), vec2(18.0, 6.0))
                path = self.path_edge(path, vec2(18.0, 6.0), vec2(13.0, 12.0))
                path = self.path_edge(path, vec2(13.0, 12.0), vec2(18.0, 18.0))
                path = self.path_edge(path, vec2(18.0, 18.0), vec2(18.0, 21.0))
                path = self.path_edge(path, vec2(18.0, 21.0), vec2(6.0, 21.0))
                path = self.path_edge(path, vec2(6.0, 21.0), vec2(6.0, 18.0))
                path = self.path_edge(path, vec2(6.0, 18.0), vec2(11.0, 12.0))
                path = self.path_edge(path, vec2(11.0, 12.0), vec2(6.0, 6.0))
                path = self.path_edge(path, vec2(6.0, 6.0), vec2(6.0, 3.0))
                return self.path_pixel(path)
            }
            // Arrow, and help: its mark over the arrow.
            return over + self.arrow() * (1.0 - over.w)
        }
    }
}

pub(crate) fn hotspot(cursor: MouseCursor, size: DVec2) -> DVec2 {
    let point = match cursor {
        MouseCursor::Default | MouseCursor::Arrow | MouseCursor::Help => dvec2(3.0, 2.0),
        MouseCursor::Hand => dvec2(10.0, 2.0),
        _ => dvec2(12.0, 12.0),
    };
    point * size / 24.0
}

#[derive(Clone, Copy, Script, ScriptHook)]
#[repr(u32)]
enum CursorShape {
    Hidden = 0,
    #[pick]
    Default = 1,
    Crosshair = 2,
    Hand = 3,
    Arrow = 4,
    Move = 5,
    Text = 6,
    Wait = 7,
    Help = 8,
    NotAllowed = 9,
    Grab = 10,
    Grabbing = 11,
    NResize = 12,
    NeResize = 13,
    EResize = 14,
    SeResize = 15,
    SResize = 16,
    SwResize = 17,
    WResize = 18,
    NwResize = 19,
    NsResize = 20,
    NeswResize = 21,
    EwResize = 22,
    NwseResize = 23,
    ColResize = 24,
    RowResize = 25,
}

pub(crate) fn shape_value(cursor: MouseCursor) -> f32 {
    let shape = match cursor {
        MouseCursor::Hidden => CursorShape::Hidden,
        MouseCursor::Default => CursorShape::Default,
        MouseCursor::Crosshair => CursorShape::Crosshair,
        MouseCursor::Hand => CursorShape::Hand,
        MouseCursor::Arrow => CursorShape::Arrow,
        MouseCursor::Move => CursorShape::Move,
        MouseCursor::Text => CursorShape::Text,
        MouseCursor::Wait => CursorShape::Wait,
        MouseCursor::Help => CursorShape::Help,
        MouseCursor::NotAllowed => CursorShape::NotAllowed,
        MouseCursor::Grab => CursorShape::Grab,
        MouseCursor::Grabbing => CursorShape::Grabbing,
        MouseCursor::NResize => CursorShape::NResize,
        MouseCursor::NeResize => CursorShape::NeResize,
        MouseCursor::EResize => CursorShape::EResize,
        MouseCursor::SeResize => CursorShape::SeResize,
        MouseCursor::SResize => CursorShape::SResize,
        MouseCursor::SwResize => CursorShape::SwResize,
        MouseCursor::WResize => CursorShape::WResize,
        MouseCursor::NwResize => CursorShape::NwResize,
        MouseCursor::NsResize => CursorShape::NsResize,
        MouseCursor::NeswResize => CursorShape::NeswResize,
        MouseCursor::EwResize => CursorShape::EwResize,
        MouseCursor::NwseResize => CursorShape::NwseResize,
        MouseCursor::ColResize => CursorShape::ColResize,
        MouseCursor::RowResize => CursorShape::RowResize,
    };
    // Shader enum attributes are integer lanes in the f32 instance storage.
    f32::from_bits(shape as u32)
}
