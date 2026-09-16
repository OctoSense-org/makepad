use crate::shader::draw_vector::DrawVector;
/// SVG renderer: walks an SvgDocument and emits DrawVector calls.
use makepad_svg::animate::{
    eval_color_animation, eval_float_animation, eval_path_animation, eval_transform_animation,
};
use makepad_svg::document::*;
use makepad_svg::units::viewbox_transform;
use makepad_svg::{VectorPaint, VectorPath};
use std::collections::HashMap;

// Re-import path commands for shadow offset manipulation
use makepad_svg::path::PathCmd;

/// Pre-built gradient texture row mapping: gradient ID -> row index (as f32).
struct GradientMap {
    rows: HashMap<String, f32>,
}

pub fn render_svg(
    dv: &mut DrawVector,
    doc: &SvgDocument,
    offset_x: f32,
    offset_y: f32,
    target_w: f32,
    target_h: f32,
    time: f32,
) {
    let offset = Transform2d::translate(offset_x, offset_y);
    let base_transform = if let Some(ref vb) = doc.viewbox {
        let (sx, sy, tx, ty) = viewbox_transform(vb, target_w, target_h);
        offset.then(&Transform2d {
            a: sx,
            c: 0.0,
            e: tx,
            b: 0.0,
            d: sy,
            f: ty,
        })
    } else {
        offset
    };

    // Pre-build gradient texture rows for all gradients in defs.
    // Sort by ID so row assignment is deterministic across runs
    // (HashMap iteration order is randomized in Rust).
    let mut grad_map = GradientMap {
        rows: HashMap::new(),
    };
    let mut sorted_grad_ids: Vec<&String> = doc.defs.gradients.keys().collect();
    sorted_grad_ids.sort();
    for id in sorted_grad_ids {
        let grad = &doc.defs.gradients[id];
        if !grad.stops.is_empty() {
            let row_idx = dv.add_gradient_row(&grad.stops);
            grad_map.rows.insert(id.clone(), row_idx);
        }
    }

    render_nodes(dv, &doc.root, &doc.defs, &base_transform, time, &grad_map);
}

fn render_nodes(
    dv: &mut DrawVector,
    nodes: &[SvgNode],
    defs: &SvgDefs,
    parent_xf: &Transform2d,
    time: f32,
    grad_map: &GradientMap,
) {
    for node in nodes {
        match node {
            SvgNode::Group(group) => render_group(dv, group, defs, parent_xf, time, grad_map),
            SvgNode::Path(path) => render_path(dv, path, defs, parent_xf, time, grad_map),
            SvgNode::Rect(rect) => render_rect(dv, rect, defs, parent_xf, time, grad_map),
            SvgNode::Circle(circ) => render_circle(dv, circ, defs, parent_xf, time, grad_map),
            SvgNode::Ellipse(ell) => render_ellipse(dv, ell, defs, parent_xf, time, grad_map),
            SvgNode::Line(line) => render_line(dv, line, defs, parent_xf, time, grad_map),
            SvgNode::Polyline(poly) => render_polyline(dv, poly, defs, parent_xf, time, grad_map),
            SvgNode::Polygon(poly) => render_polygon(dv, poly, defs, parent_xf, time, grad_map),
            SvgNode::Use(use_node) => render_use(dv, use_node, defs, parent_xf, time, grad_map),
            // Text is laid out by the host from collect_text_cmds, not rasterised here.
            SvgNode::Text(_) => {}
        }
    }
}

fn render_group(
    dv: &mut DrawVector,
    group: &SvgGroup,
    defs: &SvgDefs,
    parent_xf: &Transform2d,
    time: f32,
    grad_map: &GradientMap,
) {
    // SVG transform nesting: child local coords -> element transform -> parent
    let mut local_xf = group.transform.clone();

    // animateTransform composes in the element's local space
    for at in &group.animate_transforms {
        if let Some(anim_xf) = eval_transform_animation(at, time) {
            local_xf = anim_xf.then(&local_xf);
        }
    }

    let xf = local_xf.then(parent_xf);
    render_nodes(dv, &group.children, defs, &xf, time, grad_map);
}

fn render_use(
    dv: &mut DrawVector,
    use_node: &SvgUse,
    defs: &SvgDefs,
    parent_xf: &Transform2d,
    time: f32,
    grad_map: &GradientMap,
) {
    let symbol = match defs.symbols.get(&use_node.href) {
        Some(s) => s,
        None => return,
    };

    // Build transform: parent * use.transform * translate(x,y) * viewbox_fit
    let mut local_xf = use_node.transform.clone();
    for at in &use_node.animate_transforms {
        if let Some(anim_xf) = eval_transform_animation(at, time) {
            local_xf = anim_xf.then(&local_xf);
        }
    }

    // Apply x/y offset
    let offset = Transform2d::translate(use_node.x, use_node.y);
    local_xf = offset.then(&local_xf);

    // If the symbol has a viewBox and the <use> specifies width/height,
    // apply a viewbox-to-viewport transform
    if let Some(ref vb) = symbol.viewbox {
        let w = use_node.width.unwrap_or(vb.width);
        let h = use_node.height.unwrap_or(vb.height);
        let (sx, sy, tx, ty) = viewbox_transform(vb, w, h);
        let vb_xf = Transform2d {
            a: sx,
            c: 0.0,
            e: tx,
            b: 0.0,
            d: sy,
            f: ty,
        };
        local_xf = vb_xf.then(&local_xf);
    }

    let xf = local_xf.then(parent_xf);

    // Propagate the <use> element's `color` property into the symbol children
    // so that `currentColor` paint values resolve to it.
    let prev_use_color = dv.cur_use_color;
    let use_color = use_node.style.color;
    // Only override if the <use> element explicitly set a color (non-default-black)
    if use_color != (0.0, 0.0, 0.0, 1.0) || prev_use_color.is_some() {
        dv.cur_use_color = Some(use_color);
    }

    render_nodes(dv, &symbol.children, defs, &xf, time, grad_map);

    dv.cur_use_color = prev_use_color;
}

fn apply_animated_style(style: &SvgStyle, animations: &[SvgAnimate], time: f32) -> SvgStyle {
    let mut s = style.clone();
    for anim in animations {
        match anim.attribute {
            AnimateAttribute::Fill => {
                if let Some(color) = eval_color_animation(anim, time) {
                    s.fill = Some(SvgPaint::Color(color.0, color.1, color.2, color.3));
                }
            }
            AnimateAttribute::Stroke => {
                if let Some(color) = eval_color_animation(anim, time) {
                    s.stroke = Some(SvgPaint::Color(color.0, color.1, color.2, color.3));
                }
            }
            AnimateAttribute::StrokeWidth => {
                if let Some(w) = eval_float_animation(anim, time) {
                    s.stroke_width = w;
                }
            }
            AnimateAttribute::Opacity => {
                if let Some(o) = eval_float_animation(anim, time) {
                    s.opacity = o.clamp(0.0, 1.0);
                }
            }
            AnimateAttribute::FillOpacity => {
                if let Some(o) = eval_float_animation(anim, time) {
                    s.fill_opacity = o.clamp(0.0, 1.0);
                }
            }
            AnimateAttribute::StrokeOpacity => {
                if let Some(o) = eval_float_animation(anim, time) {
                    s.stroke_opacity = o.clamp(0.0, 1.0);
                }
            }
            _ => {}
        }
    }
    s
}

fn render_path(
    dv: &mut DrawVector,
    svg_path: &SvgPath,
    defs: &SvgDefs,
    parent_xf: &Transform2d,
    time: f32,
    grad_map: &GradientMap,
) {
    let mut local_xf = svg_path.transform.clone();
    for at in &svg_path.animate_transforms {
        if let Some(anim_xf) = eval_transform_animation(at, time) {
            local_xf = anim_xf.then(&local_xf);
        }
    }
    let xf = local_xf.then(parent_xf);

    let style = apply_animated_style(&svg_path.style, &svg_path.animations, time);

    // Check for path morphing (animate d)
    let animated_path;
    let use_path = if let Some(anim) = svg_path
        .animations
        .iter()
        .find(|a| matches!(a.attribute, AnimateAttribute::D))
    {
        if let Some(p) = eval_path_animation(anim, time) {
            animated_path = p;
            &animated_path
        } else {
            &svg_path.path
        }
    } else {
        &svg_path.path
    };

    // Compute local-space bounding box of the path for gradient mapping
    let bbox = path_bbox(use_path);
    emit_shape(
        dv,
        |dv| emit_path(dv, use_path, &xf),
        &style,
        defs,
        &xf,
        &bbox,
        grad_map,
    );
}

fn render_rect(
    dv: &mut DrawVector,
    rect: &SvgRect,
    defs: &SvgDefs,
    parent_xf: &Transform2d,
    time: f32,
    grad_map: &GradientMap,
) {
    let mut local_xf = rect.transform.clone();
    for at in &rect.animate_transforms {
        if let Some(anim_xf) = eval_transform_animation(at, time) {
            local_xf = anim_xf.then(&local_xf);
        }
    }
    let xf = local_xf.then(parent_xf);
    let mut style = apply_animated_style(&rect.style, &rect.animations, time);

    // Wide tessellated fringes overlap at rectangle corners and can fold into
    // hard diagonal seams. Use DrawVector's Gaussian rounded-rect integral for
    // axis-aligned rectangles with a uniform transform.
    if let Some((x,y,w,h,radius,scale)) = rect_shadow_geometry(rect,&xf) {
        if let Some(filter)=style.filter.as_ref().and_then(|id|defs.filters.get(id)) {
            let saved_paint=dv.cur_paint.clone();
            for effect in &filter.effects {
                match effect {
                    SvgFilterEffect::DropShadow {dx,dy,std_dev,color} => {
                        dv.set_color(color.0,color.1,color.2,color.3*style.opacity);
                        dv.shadow(x,y,w,h,radius,(std_dev*scale).max(0.001),dx*xf.a,dy*xf.d);
                    }
                }
            }
            dv.cur_paint=saved_paint;
            style.filter=None;
        }
    }

    let r = rect.rx.max(rect.ry);
    let bbox = LocalBbox::new(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
    emit_shape(
        dv,
        |dv| {
            if r > 0.0 {
                emit_rounded_rect(dv, rect.x, rect.y, rect.width, rect.height, r, &xf);
            } else {
                emit_rect(dv, rect.x, rect.y, rect.width, rect.height, &xf);
            }
        },
        &style,
        defs,
        &xf,
        &bbox,
        grad_map,
    );
}

fn rect_shadow_geometry(rect:&SvgRect,xf:&Transform2d)->Option<(f32,f32,f32,f32,f32,f32)> {
    let scale=xf.a.abs();
    if xf.b.abs()>1e-6 || xf.c.abs()>1e-6 || (scale-xf.d.abs()).abs()>1e-6 || scale<1e-6 {
        return None;
    }
    let (x0,y0)=xf.apply(rect.x,rect.y);
    let (x1,y1)=xf.apply(rect.x+rect.width,rect.y+rect.height);
    Some((x0.min(x1),y0.min(y1),(x1-x0).abs(),(y1-y0).abs(),rect.rx.max(rect.ry)*scale,scale))
}

fn render_circle(
    dv: &mut DrawVector,
    circ: &SvgCircle,
    defs: &SvgDefs,
    parent_xf: &Transform2d,
    time: f32,
    grad_map: &GradientMap,
) {
    let mut local_xf = circ.transform.clone();
    for at in &circ.animate_transforms {
        if let Some(anim_xf) = eval_transform_animation(at, time) {
            local_xf = anim_xf.then(&local_xf);
        }
    }
    let xf = local_xf.then(parent_xf);
    let style = apply_animated_style(&circ.style, &circ.animations, time);

    // Animate geometry attributes
    let mut r = circ.r;
    let mut cx = circ.cx;
    let mut cy = circ.cy;
    for anim in &circ.animations {
        match anim.attribute {
            AnimateAttribute::R => {
                if let Some(v) = eval_float_animation(anim, time) {
                    r = v;
                }
            }
            AnimateAttribute::Cx => {
                if let Some(v) = eval_float_animation(anim, time) {
                    cx = v;
                }
            }
            AnimateAttribute::Cy => {
                if let Some(v) = eval_float_animation(anim, time) {
                    cy = v;
                }
            }
            _ => {}
        }
    }

    let bbox = LocalBbox::new(cx - r, cy - r, cx + r, cy + r);
    emit_shape(
        dv,
        |dv| emit_ellipse(dv, cx, cy, r, r, &xf),
        &style,
        defs,
        &xf,
        &bbox,
        grad_map,
    );
}

fn render_ellipse(
    dv: &mut DrawVector,
    ell: &SvgEllipse,
    defs: &SvgDefs,
    parent_xf: &Transform2d,
    time: f32,
    grad_map: &GradientMap,
) {
    let mut local_xf = ell.transform.clone();
    for at in &ell.animate_transforms {
        if let Some(anim_xf) = eval_transform_animation(at, time) {
            local_xf = anim_xf.then(&local_xf);
        }
    }
    let xf = local_xf.then(parent_xf);
    let style = apply_animated_style(&ell.style, &ell.animations, time);

    let bbox = LocalBbox::new(
        ell.cx - ell.rx,
        ell.cy - ell.ry,
        ell.cx + ell.rx,
        ell.cy + ell.ry,
    );
    emit_shape(
        dv,
        |dv| emit_ellipse(dv, ell.cx, ell.cy, ell.rx, ell.ry, &xf),
        &style,
        defs,
        &xf,
        &bbox,
        grad_map,
    );
}

fn render_line(
    dv: &mut DrawVector,
    line: &SvgLine,
    defs: &SvgDefs,
    parent_xf: &Transform2d,
    time: f32,
    grad_map: &GradientMap,
) {
    let mut local_xf = line.transform.clone();
    for at in &line.animate_transforms {
        if let Some(anim_xf) = eval_transform_animation(at, time) {
            local_xf = anim_xf.then(&local_xf);
        }
    }
    let xf = local_xf.then(parent_xf);
    let style = apply_animated_style(&line.style, &line.animations, time);

    let bbox = LocalBbox::new(
        line.x1.min(line.x2),
        line.y1.min(line.y2),
        line.x1.max(line.x2),
        line.y1.max(line.y2),
    );
    emit_shape(
        dv,
        |dv| {
            let (x1, y1) = xf.apply(line.x1, line.y1);
            let (x2, y2) = xf.apply(line.x2, line.y2);
            dv.move_to(x1, y1);
            dv.line_to(x2, y2);
        },
        &style,
        defs,
        &xf,
        &bbox,
        grad_map,
    );
}

fn render_polyline(
    dv: &mut DrawVector,
    poly: &SvgPolyline,
    defs: &SvgDefs,
    parent_xf: &Transform2d,
    time: f32,
    grad_map: &GradientMap,
) {
    let mut local_xf = poly.transform.clone();
    for at in &poly.animate_transforms {
        if let Some(anim_xf) = eval_transform_animation(at, time) {
            local_xf = anim_xf.then(&local_xf);
        }
    }
    let xf = local_xf.then(parent_xf);
    let style = apply_animated_style(&poly.style, &poly.animations, time);

    let bbox = points_bbox(&poly.points);
    emit_shape(
        dv,
        |dv| emit_points(dv, &poly.points, false, &xf),
        &style,
        defs,
        &xf,
        &bbox,
        grad_map,
    );
}

fn render_polygon(
    dv: &mut DrawVector,
    poly: &SvgPolygon,
    defs: &SvgDefs,
    parent_xf: &Transform2d,
    time: f32,
    grad_map: &GradientMap,
) {
    let mut local_xf = poly.transform.clone();
    for at in &poly.animate_transforms {
        if let Some(anim_xf) = eval_transform_animation(at, time) {
            local_xf = anim_xf.then(&local_xf);
        }
    }
    let xf = local_xf.then(parent_xf);
    let style = apply_animated_style(&poly.style, &poly.animations, time);

    let bbox = points_bbox(&poly.points);
    emit_shape(
        dv,
        |dv| emit_points(dv, &poly.points, true, &xf),
        &style,
        defs,
        &xf,
        &bbox,
        grad_map,
    );
}

// ---- Emit helpers: build path in DrawVector ----

/// Local-space bounding box for objectBoundingBox gradient mapping.
struct LocalBbox {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

impl LocalBbox {
    fn new(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    fn width(&self) -> f32 {
        self.max_x - self.min_x
    }
    fn height(&self) -> f32 {
        self.max_y - self.min_y
    }
}

fn emit_shape(
    dv: &mut DrawVector,
    build_path: impl Fn(&mut DrawVector),
    style: &SvgStyle,
    defs: &SvgDefs,
    xf: &Transform2d,
    bbox: &LocalBbox,
    grad_map: &GradientMap,
) {
    // Emit drop shadow geometry before the main shape
    if let Some(ref filter_id) = style.filter {
        if let Some(filter) = defs.filters.get(filter_id) {
            for effect in &filter.effects {
                match effect {
                    SvgFilterEffect::DropShadow {
                        dx,
                        dy,
                        std_dev,
                        color,
                    } => {
                        // Scale offset and blur by the transform
                        let scale = xf.scale_factor();
                        let sdx = dx * scale;
                        let sdy = dy * scale;
                        let sblur = std_dev * scale;

                        let saved_paint = dv.cur_paint.clone();
                        let a = color.3 * style.opacity;
                        dv.set_color(color.0, color.1, color.2, a); // VectorPaint::solid() will premultiply

                        // Build the shape path, then offset all vertices for shadow
                        build_path(dv);
                        // Offset all path points by scaled dx, dy
                        for cmd in &mut dv.path.cmds {
                            match cmd {
                                PathCmd::MoveTo(x, y) => {
                                    *x += sdx;
                                    *y += sdy;
                                }
                                PathCmd::LineTo(x, y) => {
                                    *x += sdx;
                                    *y += sdy;
                                }
                                PathCmd::BezierTo(cx1, cy1, cx2, cy2, x, y) => {
                                    *cx1 += sdx;
                                    *cy1 += sdy;
                                    *cx2 += sdx;
                                    *cy2 += sdy;
                                    *x += sdx;
                                    *y += sdy;
                                }
                                _ => {}
                            }
                        }
                        dv.shape_shadow(sblur.max(0.5));
                        dv.cur_paint = saved_paint;
                    }
                }
            }
        }
    }

    let opacity = style.opacity;
    dv.set_shape_id(style.shader_id);

    // For shapes with shader effects (shader_id > 0), compute the world-space
    // bounding box and store it so the pixel shader can derive proper UVs.
    if style.shader_id > 0.0 {
        let (wx0, wy0) = xf.apply(bbox.min_x, bbox.min_y);
        let (wx1, wy1) = xf.apply(bbox.max_x, bbox.min_y);
        let (wx2, wy2) = xf.apply(bbox.min_x, bbox.max_y);
        let (wx3, wy3) = xf.apply(bbox.max_x, bbox.max_y);
        let wmin_x = wx0.min(wx1).min(wx2).min(wx3);
        let wmin_y = wy0.min(wy1).min(wy2).min(wy3);
        let wmax_x = wx0.max(wx1).max(wx2).max(wx3);
        let wmax_y = wy0.max(wy1).max(wy2).max(wy3);
        dv.cur_effect_bbox = Some([wmin_x, wmin_y, wmax_x, wmax_y]);
    }

    // Resolve currentColor: prefer the <use> element's color override, then the style's own color
    let current_color = dv.cur_use_color.unwrap_or(style.color);
    let resolved_cc = SvgPaint::Color(
        current_color.0,
        current_color.1,
        current_color.2,
        current_color.3,
    );

    // Fill (SVG default fill is black when not specified)
    let default_fill = SvgPaint::Color(0.0, 0.0, 0.0, 1.0);
    let fill_paint = style.fill.as_ref().unwrap_or(&default_fill);
    {
        let paint = if matches!(fill_paint, SvgPaint::CurrentColor) {
            &resolved_cc
        } else {
            fill_paint
        };
        if !matches!(paint, SvgPaint::None) {
            build_path(dv);
            dv.path.fill_rule = Some(style.fill_rule);
            let fill_alpha = style.fill_opacity * opacity;
            set_paint(dv, paint, defs, fill_alpha, xf, bbox, grad_map);
            // Use pre-computed fringe (not GPU-expand) to avoid coincident-vertex
            // triangles that cause Metal GPU rasterization artifacts.
            dv.fill();
            dv.cur_gradient_row_v = -1.0; // reset after fill
            dv.path.clear();
        }
    }

    // Stroke
    if let Some(ref paint) = style.stroke {
        let paint = if matches!(paint, SvgPaint::CurrentColor) {
            &resolved_cc
        } else {
            paint
        };
        if !matches!(paint, SvgPaint::None) && style.stroke_width > 0.0 {
            build_path(dv);
            let stroke_alpha = style.stroke_opacity * opacity;
            set_paint(dv, paint, defs, stroke_alpha, xf, bbox, grad_map);
            let w = style.stroke_width * xf.scale_factor();
            // Prefer the device-aware fringe (cur_stroke_aa) so stroke AA lands at ~1 device px; else the old heuristic.
            let aa = if dv.cur_stroke_aa > 0.0 {
                dv.cur_stroke_aa
            } else {
                w.min(1.0)
            };
            if let Some(pattern)=&style.stroke_dasharray {
                let scale=xf.scale_factor();
                let pattern:Vec<f32>=pattern.iter().map(|v|v*scale).collect();
                dv.tess.flatten(&dv.path,0.25);
                if let Some(path)=dv.tess.dashed_path(&pattern,style.stroke_dashoffset*scale) {
                    dv.path=path;
                }
            }
            dv.stroke_opts(
                w,
                style.stroke_linecap,
                style.stroke_linejoin,
                style.stroke_miterlimit,
                aa,
            );
            dv.cur_gradient_row_v = -1.0; // reset after stroke
            dv.path.clear();
        }
    }

    dv.cur_effect_bbox = None;
}

fn set_paint(
    dv: &mut DrawVector,
    paint: &SvgPaint,
    defs: &SvgDefs,
    alpha: f32,
    xf: &Transform2d,
    bbox: &LocalBbox,
    grad_map: &GradientMap,
) {
    match paint {
        SvgPaint::None => {}
        SvgPaint::Color(r, g, b, a) => {
            let a = a * alpha;
            dv.set_color(*r, *g, *b, a); // VectorPaint::solid() will premultiply
            dv.cur_gradient_row_v = -1.0;
        }
        SvgPaint::GradientRef(id) => {
            if let Some(grad) = defs.gradients.get(id) {
                let mut adjusted;
                let grad = if alpha != 1.0 {
                    adjusted = grad.clone();
                    for stop in &mut adjusted.stops {
                        for channel in &mut stop.color { *channel *= alpha; }
                    }
                    &adjusted
                } else { grad };
                let vp = gradient_to_vector_paint(grad, xf, bbox);
                // Set gradient texture row if we have a pre-built row for this gradient
                if alpha != 1.0 && !grad.stops.is_empty() {
                    // Gradient textures are shared by ID. A translucent use
                    // needs its own premultiplied row, without changing peers.
                    dv.cur_gradient_row_v = dv.add_gradient_row(&grad.stops);
                } else if let Some(&row_idx) = grad_map.rows.get(id) {
                    dv.cur_gradient_row_v = row_idx;
                } else {
                    dv.cur_gradient_row_v = -1.0;
                }
                dv.set_paint(vp);
            } else {
                dv.set_color(0.0, 0.0, 0.0, alpha); // fallback black
                dv.cur_gradient_row_v = -1.0;
            }
        }
        SvgPaint::CurrentColor => {
            // Should already be resolved by emit_shape; fallback to black
            dv.set_color(0.0, 0.0, 0.0, alpha);
            dv.cur_gradient_row_v = -1.0;
        }
    }
}

fn gradient_to_vector_paint(grad: &SvgGradient, xf: &Transform2d, bbox: &LocalBbox) -> VectorPaint {
    if grad.stops.is_empty() {
        return VectorPaint::solid(0.0, 0.0, 0.0, 1.0);
    }

    match grad.kind {
        GradientKind::Linear => {
            let units = match grad.units {
                GradientUnits::ObjectBoundingBox => Transform2d::scale(bbox.width(), bbox.height())
                    .then(&Transform2d::translate(bbox.min_x, bbox.min_y)),
                GradientUnits::UserSpaceOnUse => Transform2d::identity(),
            };
            let gxf = grad.transform.then(&units).then(xf);
            let (x0w, y0w) = gxf.apply(grad.x1, grad.y1);
            // Gradient normals transform by the inverse transpose. Transforming
            // endpoints directly distorts color progression on non-square bounds.
            let dx = grad.x2 - grad.x1;
            let dy = grad.y2 - grad.y1;
            let det = gxf.a * gxf.d - gxf.b * gxf.c;
            let len2 = dx * dx + dy * dy;
            let (ex, ey) = if det.abs() > 1e-12 && len2 > 1e-12 {
                let nx = (gxf.d * dx - gxf.b * dy) / (det * len2);
                let ny = (-gxf.c * dx + gxf.a * dy) / (det * len2);
                let norm2 = nx * nx + ny * ny;
                if norm2 > 1e-20 {(nx / norm2, ny / norm2)} else {(0.0, 0.0)}
            } else {(0.0, 0.0)};
            let (x1w, y1w) = (x0w + ex, y0w + ey);
            VectorPaint::LinearGradient {
                x0: x0w,
                y0: y0w,
                x1: x1w,
                y1: y1w,
                stops: grad.stops.clone(),
            }
        }
        GradientKind::Radial => {
            let units = match grad.units {
                GradientUnits::ObjectBoundingBox => Transform2d::scale(bbox.width(), bbox.height())
                    .then(&Transform2d::translate(bbox.min_x, bbox.min_y)),
                GradientUnits::UserSpaceOnUse => Transform2d::identity(),
            };
            let gxf = grad.transform.then(&units).then(xf);
            let (cxw, cyw) = gxf.apply(grad.cx, grad.cy);
            // A unit circle is transformed in gradient space before the object
            // bounds. Row norms retain its x/y extents under an inner rotation.
            let sx = (gxf.a * gxf.a + gxf.c * gxf.c).sqrt();
            let sy = (gxf.b * gxf.b + gxf.d * gxf.d).sqrt();
            VectorPaint::RadialGradient {
                cx: cxw,
                cy: cyw,
                rx: grad.r * sx,
                ry: grad.r * sy,
                stops: grad.stops.clone(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangle_shadow_bounds_follow_reflected_source_geometry() {
        let doc=makepad_svg::parse::parse_svg(r#"<svg><rect x="10" y="20" width="100" height="40" rx="8"/></svg>"#);
        let SvgNode::Rect(rect)=&doc.root[0] else {panic!("expected rectangle")};
        let xf=Transform2d {a:-2.0,b:0.0,c:0.0,d:2.0,e:300.0,f:10.0};
        assert_eq!(rect_shadow_geometry(rect,&xf),Some((80.0,50.0,200.0,80.0,16.0,2.0)));
        assert!(rect_shadow_geometry(rect,&Transform2d {b:0.5,..xf}).is_none());
    }
    #[test]
    fn radial_gradient_uses_normalized_object_bounds() {
        let mut gradient=SvgGradient::new_radial();
        gradient.stops=vec![makepad_svg::GradientStop::new(0.0,0.0,0.0,0.0,1.0),
                            makepad_svg::GradientStop::new(1.0,1.0,1.0,1.0,1.0)];
        let paint=gradient_to_vector_paint(&gradient,&Transform2d::identity(),&LocalBbox::new(20.0,40.0,520.0,140.0));
        assert!((paint.color_at(395.0,90.0)[0]-0.5).abs()<0.0001);
        assert!((paint.color_at(270.0,115.0)[0]-0.5).abs()<0.0001);
    }
    #[test]
    fn bounding_box_gradient_preserves_normal_under_nonuniform_scaling() {
        let mut gradient=SvgGradient::new_linear();
        gradient.x2=1.0;gradient.y2=1.0;
        gradient.stops=vec![makepad_svg::GradientStop::new(0.0,0.0,0.0,0.0,1.0),
                            makepad_svg::GradientStop::new(1.0,1.0,1.0,1.0,1.0)];
        let paint=gradient_to_vector_paint(&gradient,&Transform2d::identity(),&LocalBbox::new(0.0,0.0,500.0,100.0));
        assert!((paint.color_at(500.0,0.0)[0]-0.5).abs()<0.0001);
        assert!((paint.color_at(0.0,100.0)[0]-0.5).abs()<0.0001);
    }
}

fn path_bbox(path: &VectorPath) -> LocalBbox {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for cmd in &path.cmds {
        match cmd {
            PathCmd::MoveTo(x, y) | PathCmd::LineTo(x, y) => {
                min_x = min_x.min(*x);
                min_y = min_y.min(*y);
                max_x = max_x.max(*x);
                max_y = max_y.max(*y);
            }
            PathCmd::BezierTo(cx1, cy1, cx2, cy2, x, y) => {
                for &(px, py) in &[(*cx1, *cy1), (*cx2, *cy2), (*x, *y)] {
                    min_x = min_x.min(px);
                    min_y = min_y.min(py);
                    max_x = max_x.max(px);
                    max_y = max_y.max(py);
                }
            }
            _ => {}
        }
    }
    if min_x > max_x {
        LocalBbox::new(0.0, 0.0, 0.0, 0.0)
    } else {
        LocalBbox::new(min_x, min_y, max_x, max_y)
    }
}

fn points_bbox(points: &[(f32, f32)]) -> LocalBbox {
    if points.is_empty() {
        return LocalBbox::new(0.0, 0.0, 0.0, 0.0);
    }
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for &(x, y) in points {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    LocalBbox::new(min_x, min_y, max_x, max_y)
}

fn emit_path(dv: &mut DrawVector, path: &VectorPath, xf: &Transform2d) {
    for cmd in &path.cmds {
        match cmd {
            PathCmd::MoveTo(x, y) => {
                let (tx, ty) = xf.apply(*x, *y);
                dv.move_to(tx, ty);
            }
            PathCmd::LineTo(x, y) => {
                let (tx, ty) = xf.apply(*x, *y);
                dv.line_to(tx, ty);
            }
            PathCmd::BezierTo(cx1, cy1, cx2, cy2, x, y) => {
                let (tcx1, tcy1) = xf.apply(*cx1, *cy1);
                let (tcx2, tcy2) = xf.apply(*cx2, *cy2);
                let (tx, ty) = xf.apply(*x, *y);
                dv.bezier_to(tcx1, tcy1, tcx2, tcy2, tx, ty);
            }
            PathCmd::Close => {
                dv.close();
            }
            PathCmd::Winding(_) => {} // handled by fill rule
        }
    }
}

fn emit_rect(dv: &mut DrawVector, x: f32, y: f32, w: f32, h: f32, xf: &Transform2d) {
    let (x0, y0) = xf.apply(x, y);
    let (x1, y1) = xf.apply(x + w, y);
    let (x2, y2) = xf.apply(x + w, y + h);
    let (x3, y3) = xf.apply(x, y + h);
    dv.move_to(x0, y0);
    dv.line_to(x1, y1);
    dv.line_to(x2, y2);
    dv.line_to(x3, y3);
    dv.close();
}

fn emit_rounded_rect(
    dv: &mut DrawVector,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    r: f32,
    xf: &Transform2d,
) {
    use std::f32::consts::PI;
    let r = r.min(w * 0.5).min(h * 0.5);
    if r < 0.1 {
        emit_rect(dv, x, y, w, h, xf);
        return;
    }
    // Emit rounded rect directly, transforming each point
    let (tx, ty) = xf.apply(x + r, y);
    dv.move_to(tx, ty);
    let (tx, ty) = xf.apply(x + w - r, y);
    dv.line_to(tx, ty);
    emit_arc(dv, x + w - r, y + r, r, r, -PI * 0.5, PI * 0.5, xf);
    let (tx, ty) = xf.apply(x + w, y + h - r);
    dv.line_to(tx, ty);
    emit_arc(dv, x + w - r, y + h - r, r, r, 0.0, PI * 0.5, xf);
    let (tx, ty) = xf.apply(x + r, y + h);
    dv.line_to(tx, ty);
    emit_arc(dv, x + r, y + h - r, r, r, PI * 0.5, PI * 0.5, xf);
    let (tx, ty) = xf.apply(x, y + r);
    dv.line_to(tx, ty);
    emit_arc(dv, x + r, y + r, r, r, PI, PI * 0.5, xf);
    dv.close();
}

fn emit_ellipse(dv: &mut DrawVector, cx: f32, cy: f32, rx: f32, ry: f32, xf: &Transform2d) {
    use std::f32::consts::PI;
    let (tx, ty) = xf.apply(cx + rx, cy);
    dv.move_to(tx, ty);
    emit_arc(dv, cx, cy, rx, ry, 0.0, PI * 2.0, xf);
    dv.close();
}

/// Emit arc bezier segments directly to DrawVector, transforming each control point.
#[allow(clippy::too_many_arguments)]
fn emit_arc(
    dv: &mut DrawVector,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    start: f32,
    sweep: f32,
    xf: &Transform2d,
) {
    use std::f32::consts::PI;
    let n = ((sweep.abs() / (PI * 0.5)).ceil() as usize).max(1);
    let sweep_per = sweep / n as f32;
    let k = (4.0 / 3.0) * (sweep_per / 4.0).tan();
    for i in 0..n {
        let a0 = start + sweep_per * i as f32;
        let a1 = a0 + sweep_per;
        let (s0, c0) = a0.sin_cos();
        let (s1, c1) = a1.sin_cos();
        let x0 = cx + c0 * rx;
        let y0 = cy + s0 * ry;
        let x1 = cx + c1 * rx;
        let y1 = cy + s1 * ry;
        let dx0 = -s0 * rx;
        let dy0 = c0 * ry;
        let dx1 = -s1 * rx;
        let dy1 = c1 * ry;
        if i == 0 {
            // first arc point is already emitted by caller (move_to or line_to)
        } else {
            let (tx, ty) = xf.apply(x0, y0);
            dv.line_to(tx, ty);
        }
        let (tcx1, tcy1) = xf.apply(x0 + dx0 * k, y0 + dy0 * k);
        let (tcx2, tcy2) = xf.apply(x1 - dx1 * k, y1 - dy1 * k);
        let (tx1, ty1) = xf.apply(x1, y1);
        dv.bezier_to(tcx1, tcy1, tcx2, tcy2, tx1, ty1);
    }
}

fn emit_points(dv: &mut DrawVector, points: &[(f32, f32)], close: bool, xf: &Transform2d) {
    if points.is_empty() {
        return;
    }
    let (x, y) = xf.apply(points[0].0, points[0].1);
    dv.move_to(x, y);
    for pt in &points[1..] {
        let (x, y) = xf.apply(pt.0, pt.1);
        dv.line_to(x, y);
    }
    if close {
        dv.close();
    }
}
