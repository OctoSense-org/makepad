use super::geometry::*;
use super::label::*;
use super::style::*;
use super::tile::*;
use crate::makepad_draw::vector::{LineJoin, Tessellator, VVertex, VectorPath};
use crate::{
    makepad_derive_widget::*, makepad_draw::*, widget::*, widget_async::ScriptAsyncResult,
    DrawRotatedText, DrawVector, PathGlyphInstance, PathTextPlacement, WidgetMatchEvent,
};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::fs;
use std::path::Path;

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.*
    use mod.draw
    use mod.geom
    use mod.math
    use mod.shader

    mod.draw.DrawMapVector = mod.std.set_type_default() do #(DrawMapVector::script_shader(vm)){
        ..mod.draw.DrawVector
        map_scale: uniform(vec2(1.0, 1.0))
        map_offset: uniform(vec2(0.0, 0.0))
        // --- navigation (heading-up) projection ---
        // nav_mode: 0 = off (flat map), 1 = 3D first-person chase view (pinhole
        // ground-plane projection, straight horizon), 2 = 2D heading-up.
        // nav_anchor: widget-px position of the vehicle on the FLAT map (the map
        // is centered on it). nav_rot: (sin, cos) of the bearing. All the pinhole
        // params are in the same px units as the flat map at the current zoom.
        nav_mode: uniform(0.0)
        nav_anchor: uniform(vec2(0.0, 0.0))
        nav_rot: uniform(vec2(0.0, 1.0))
        // nav_cam: (cam_height_px, sin(pitch), cos(pitch), tan(hfov/2)) — trig
        // is precomputed on the CPU so the vertex shader needs no sin/cos/tan.
        nav_cam: uniform(vec4(40.0, 0.310, 0.951, 0.76))
        // nav_screen: widget rect (x, y, w, h)
        nav_screen: uniform(vec4(0.0, 0.0, 0.0, 0.0))
        // nav_misc: (chase_px: camera sits this far behind the car,
        //            maxg_px: far haze clip, car2d_row_px: car row in 2D, tan(vfov/2))
        nav_misc: uniform(vec4(90.0, 500.0, 600.0, 0.533))
        // haze tint the far ground fades into (matches the sky/背景)
        nav_haze: uniform(vec4(0.847, 0.890, 0.929, 0.0))

        vertex: fn() {
            let pos = vec2(self.geom.x, self.geom.y);
            let flat = pos * self.map_scale + self.map_offset;
            // heading-up frame around the vehicle: ahead(+) along bearing, cross(+) right
            let rel = flat - self.nav_anchor;
            let ahead = rel.x * self.nav_rot.x - rel.y * self.nav_rot.y;
            let cross = rel.x * self.nav_rot.y + rel.y * self.nav_rot.x;
            let a = ahead + self.nav_misc.x;
            // TRUE pinhole ground-plane camera (Codex 2c). Camera at height h,
            // pitched down; a = forward ground distance from the camera. x/y are
            // rational functions of the SHARED depth z_cam, so a ground-plane
            // triangle maps EXACTLY to a screen triangle (straight edges) — no
            // densification, no atan/lat_a warp that only approximated it. Trig
            // is precomputed: nav_cam = (h, sinP, cosP, tan(hfov/2)),
            // nav_misc.w = tan(vfov/2).
            let z_cam = a * self.nav_cam.z + self.nav_cam.x * self.nav_cam.y;
            let y_cam = a * self.nav_cam.y - self.nav_cam.x * self.nav_cam.z;
            // Near plane = a z_cam floor set BELOW the nearest visible ground, so
            // all on-screen geometry is exact and only genuinely behind-camera
            // vertices clamp — those drop to a clean below-screen curtain
            // (connected, never flipping). The GPU does perspective via w=z_cam.
            let near3 = self.nav_cam.x * 0.6;
            let behind = 1.0 - step(near3, z_cam);
            let zc = max(z_cam, near3);
            let ndc_x = cross / (zc * self.nav_cam.w);
            let ndc_y = y_cam / (zc * self.nav_misc.w);
            let p3d = vec2(
                mix(
                    self.nav_screen.x + self.nav_screen.z * 0.5 * (1.0 + ndc_x),
                    self.nav_screen.x + self.nav_screen.z * 0.5 + cross * 0.30,
                    behind
                ),
                mix(
                    self.nav_screen.y + self.nav_screen.w * 0.5 * (1.0 - ndc_y),
                    self.nav_screen.y + self.nav_screen.w * 1.6,
                    behind
                )
            );
            let p2d = vec2(
                self.nav_screen.x + self.nav_screen.z * 0.5 + cross,
                self.nav_screen.y + self.nav_misc.z - ahead
            );
            let in_nav = step(0.5, self.nav_mode);
            let in_2d = step(1.5, self.nav_mode);
            let transformed = mix(flat, mix(p3d, p2d, in_2d), in_nav);
            let haze_t = in_nav * (1.0 - in_2d)
                * pow(clamp(a / self.nav_misc.y, 0.0, 1.0), 2.6) * 0.9;

            self.v_tcoord = vec2(self.geom.u, self.geom.v);
            self.v_color = vec4(self.geom.color_r, self.geom.color_g, self.geom.color_b, self.geom.color_a);
            // atmospheric haze: far ground dissolves into the horizon (premul colors)
            self.v_color = mix(
                self.v_color,
                vec4(
                    self.nav_haze.x * self.v_color.w,
                    self.nav_haze.y * self.v_color.w,
                    self.nav_haze.z * self.v_color.w,
                    self.v_color.w
                ),
                haze_t
            );
            self.v_stroke_mult = self.geom.stroke_mult;
            self.v_stroke_dist = self.geom.stroke_dist;
            self.v_shape_id = self.geom.shape_id;
            self.v_param0 = self.geom.param0;
            self.v_param5 = self.geom.param5;

            let grad_type = self.geom.param0;
            if grad_type > 0.5 && grad_type < 1.5 {
                let p0 = vec2(self.geom.param1, self.geom.param2) * self.map_scale + self.map_offset;
                let p1 = vec2(self.geom.param3, self.geom.param4) * self.map_scale + self.map_offset;
                self.v_param1 = p0.x;
                self.v_param2 = p0.y;
                self.v_param3 = p1.x;
                self.v_param4 = p1.y;
            } else if grad_type > 1.5 {
                let center = vec2(self.geom.param1, self.geom.param2) * self.map_scale + self.map_offset;
                self.v_param1 = center.x;
                self.v_param2 = center.y;
                self.v_param3 = self.geom.param3 * self.map_scale.x;
                self.v_param4 = self.geom.param4 * self.map_scale.y;
            } else if self.geom.shape_id > 0.5 {
                let bbox_min = vec2(self.geom.param1, self.geom.param2) * self.map_scale + self.map_offset;
                let bbox_max = vec2(self.geom.param3, self.geom.param4) * self.map_scale + self.map_offset;
                self.v_param1 = bbox_min.x;
                self.v_param2 = bbox_min.y;
                self.v_param3 = bbox_max.x;
                self.v_param4 = bbox_max.y;
            } else {
                self.v_param1 = self.geom.param1;
                self.v_param2 = self.geom.param2;
                self.v_param3 = self.geom.param3;
                self.v_param4 = self.geom.param4;
            }

            let shifted = transformed + self.draw_list.view_shift;
            self.v_world = shifted;

            // Flat-space clip reject: honor it for flat/2D only, and use the
            // stock offscreen sentinel (2,2,2,1) — NOT (0,0,0,0), which is
            // degenerate. In 3D SKIP it: the flat radius is non-conservative
            // once the near field magnifies an edge, so zeroing one vertex of a
            // triangle that still crosses the view deforms/vanishes it (Codex);
            // the pinhole instead maps the frustum into the rect and the GPU
            // clips behind-camera geometry via w.
            let in3d = in_nav * (1.0 - in_2d);
            let cr = self.geom.clip_radius * max(self.map_scale.x, self.map_scale.y);
            let clip = vec4(
                max(self.draw_clip.x, self.draw_list.view_clip.x - self.draw_list.view_shift.x),
                max(self.draw_clip.y, self.draw_list.view_clip.y - self.draw_list.view_shift.y),
                min(self.draw_clip.z, self.draw_list.view_clip.z - self.draw_list.view_shift.x),
                min(self.draw_clip.w, self.draw_list.view_clip.w - self.draw_list.view_shift.y)
            )

            if in3d < 0.5 && (transformed.x + cr < clip.x || transformed.y + cr < clip.y
                || transformed.x - cr > clip.z || transformed.y - cr > clip.w) {
                self.vertex_pos = vec4(2.0, 2.0, 2.0, 1.0);
                return
            }

            // Keep w = 1 (LINEAR screen-space varying interpolation). The pinhole
            // already gives each vertex its exact screen position, and a
            // ground-plane triangle maps to a straight-edged screen triangle
            // (perspective preserves lines) — so positions are exact WITHOUT a
            // w-divide. A perspective w-divide would instead make v_stroke_dist /
            // v_world interpolate perspective-correct, distorting the screen-space
            // stroke AA and SERRATING the route ribbon at steep view angles.
            let world = self.draw_list.view_transform * vec4(
                shifted.x,
                shifted.y,
                self.draw_depth + self.draw_call.zbias + self.geom.zbias,
                1.
            );
            self.v_world_clip = world;
            self.vertex_pos = self.draw_pass.camera_projection * (self.draw_pass.camera_view * world)
        }

        get_stroke_mask: fn() {
            if self.v_shape_id > 9.5 && self.v_shape_id < 10.5 {
                return self.dash(3.2, 2.4)
            }
            if self.v_shape_id > 10.5 && self.v_shape_id < 11.5 {
                return self.dash(2.0, 3.0)
            }
            return 1.0
        }
    }

    mod.widgets.MapViewBase = #(MapView::register_widget(vm))

    mod.widgets.MapView = set_type_default() do mod.widgets.MapViewBase{
        width: Fill
        height: Fill
        center_lon: 4.9041
        center_lat: 52.3676
        zoom: 14.0
        min_zoom: 11.0
        max_zoom: 17.0
        dark_theme: false
        use_network: false
        use_local_mbtiles: true
        style_light: MapThemeStyle{
            background: #xddd7cc
            status_text: #xdee9f4
            label: #x000000

            MapFillRule{group: "building" color: #xc6c0b5}
            MapFillRule{group: "water" color: #x9ecff2}
            MapFillRule{group: "landuse" value: "residential" color: #xe9e4dc}
            MapFillRule{group: "landuse" value: "commercial" color: #xe1dbd2}
            MapFillRule{group: "landuse" value: "retail" color: #xe1dbd2}
            MapFillRule{group: "landuse" value: "industrial" color: #xd6d1cb}
            MapFillRule{group: "landuse" value: "forest" color: #xc4deb0}
            MapFillRule{group: "landuse" value: "grass" color: #xd4e5bf}
            MapFillRule{group: "landuse" value: "meadow" color: #xd4e5bf}
            MapFillRule{group: "landuse" value: "farmland" color: #xd4e5bf}
            MapFillRule{group: "landuse" value: "*" color: #xe5dfd6}
            MapFillRule{group: "leisure" value: "park" color: #xc5e2b6}
            MapFillRule{group: "leisure" value: "garden" color: #xc5e2b6}
            MapFillRule{group: "leisure" value: "golf_course" color: #xc5e2b6}
            MapFillRule{group: "leisure" value: "pitch" color: #xb8db9f}
            MapFillRule{group: "leisure" value: "*" color: #xd1e8bf}

            MapRoadRule{kind: "motorway" sort_rank: 700 casing_color: #xc38d49 casing_width: 3.9 center_color: #xe2ad65 center_width: 3.0}
            MapRoadRule{kind: "trunk" sort_rank: 640 casing_color: #xc59f5f casing_width: 3.5 center_color: #xe8c17e center_width: 2.7}
            MapRoadRule{kind: "primary" sort_rank: 560 casing_color: #xc6b181 casing_width: 3.1 center_color: #xf0d39c center_width: 2.35}
            MapRoadRule{kind: "secondary" sort_rank: 470 casing_color: #xd0c8b6 casing_width: 2.75 center_color: #xf4e4c4 center_width: 2.0}
            MapRoadRule{kind: "busway" sort_rank: 470 casing_color: #xd0c8b6 casing_width: 2.75 center_color: #xf4e4c4 center_width: 2.0}
            MapRoadRule{kind: "tertiary" sort_rank: 390 casing_color: #xc6c0b3 casing_width: 2.4 center_color: #xf5ebd8 center_width: 1.7}
            MapRoadRule{kind: "residential" sort_rank: 310 casing_color: #xc2bcae casing_width: 2.0 center_color: #xfefefd center_width: 1.35}
            MapRoadRule{kind: "unclassified" sort_rank: 310 casing_color: #xc2bcae casing_width: 2.0 center_color: #xfefefd center_width: 1.35}
            MapRoadRule{kind: "living_street" sort_rank: 310 casing_color: #xc2bcae casing_width: 2.0 center_color: #xfefefd center_width: 1.35}
            MapRoadRule{kind: "service" sort_rank: 240 casing_color: #xc5bfb2 casing_width: 1.75 center_color: #xf6f2ea center_width: 1.1}
            MapRoadRule{kind: "pedestrian" sort_rank: 240 casing_color: #xc5bfb2 casing_width: 1.75 center_color: #xf6f2ea center_width: 1.1}
            MapRoadRule{kind: "cycleway" sort_rank: 160 center_color: #xb6afa1 center_width: 0.82}
            MapRoadRule{kind: "footway" sort_rank: 160 center_color: #xb6afa1 center_width: 0.82}
            MapRoadRule{kind: "path" sort_rank: 160 center_color: #xb6afa1 center_width: 0.82}
            MapRoadRule{kind: "steps" sort_rank: 160 center_color: #xb6afa1 center_width: 0.82}
            MapRoadRule{kind: "track" sort_rank: 160 center_color: #xb6afa1 center_width: 0.82}
            MapRoadRule{kind: "*" sort_rank: 280 casing_color: #xc3bcaf casing_width: 1.9 center_color: #xf5f1e9 center_width: 1.2}

            MapWaterwayRule{kind: "river" sort_rank: 140 casing_color: #x4a8fc3 casing_width: 1.83 center_color: #x73b5e4 center_width: 1.55}
            MapWaterwayRule{kind: "canal" sort_rank: 140 casing_color: #x4a8fc3 casing_width: 1.5 center_color: #x73b5e4 center_width: 1.22}
            MapWaterwayRule{kind: "stream" sort_rank: 140 casing_color: #x4a8fc3 casing_width: 1.18 center_color: #x73b5e4 center_width: 0.9}
            MapWaterwayRule{kind: "*" sort_rank: 140 casing_color: #x4a8fc3 casing_width: 1.1 center_color: #x73b5e4 center_width: 0.82}
            MapRailRule{sort_rank: 180 casing_color: #xb7b2a9 casing_width: 0.96 center_color: #x8f8a81 center_width: 0.62 center_shape_id: 10.0}
        }
        style_dark: MapThemeStyle{
            background: #x161b22
            status_text: #xb2c7d8
            label: #xe5eaf1

            MapFillRule{group: "building" color: #x383d46}
            MapFillRule{group: "water" color: #x204f74}
            MapFillRule{group: "landuse" value: "residential" color: #x2a2f36}
            MapFillRule{group: "landuse" value: "commercial" color: #x30343b}
            MapFillRule{group: "landuse" value: "retail" color: #x30343b}
            MapFillRule{group: "landuse" value: "industrial" color: #x282c32}
            MapFillRule{group: "landuse" value: "forest" color: #x243629}
            MapFillRule{group: "landuse" value: "grass" color: #x2a3c2d}
            MapFillRule{group: "landuse" value: "meadow" color: #x2a3c2d}
            MapFillRule{group: "landuse" value: "farmland" color: #x2a3c2d}
            MapFillRule{group: "landuse" value: "*" color: #x2d3239}
            MapFillRule{group: "leisure" value: "park" color: #x2f4a34}
            MapFillRule{group: "leisure" value: "garden" color: #x2f4a34}
            MapFillRule{group: "leisure" value: "golf_course" color: #x2f4a34}
            MapFillRule{group: "leisure" value: "pitch" color: #x32553a}
            MapFillRule{group: "leisure" value: "*" color: #x2b4230}

            MapRoadRule{kind: "motorway" sort_rank: 700 casing_color: #x8f6937 casing_width: 3.9 center_color: #xd29b54 center_width: 3.0}
            MapRoadRule{kind: "trunk" sort_rank: 640 casing_color: #x8c7141 casing_width: 3.5 center_color: #xc8a561 center_width: 2.7}
            MapRoadRule{kind: "primary" sort_rank: 560 casing_color: #x706857 casing_width: 3.1 center_color: #xb9aa86 center_width: 2.35}
            MapRoadRule{kind: "secondary" sort_rank: 470 casing_color: #x556170 casing_width: 2.75 center_color: #x95a1b1 center_width: 2.0}
            MapRoadRule{kind: "busway" sort_rank: 470 casing_color: #x556170 casing_width: 2.75 center_color: #x95a1b1 center_width: 2.0}
            MapRoadRule{kind: "tertiary" sort_rank: 390 casing_color: #x4b5765 casing_width: 2.4 center_color: #x7d899a center_width: 1.7}
            MapRoadRule{kind: "residential" sort_rank: 310 casing_color: #x404a57 casing_width: 2.0 center_color: #x677383 center_width: 1.35}
            MapRoadRule{kind: "unclassified" sort_rank: 310 casing_color: #x404a57 casing_width: 2.0 center_color: #x677383 center_width: 1.35}
            MapRoadRule{kind: "living_street" sort_rank: 310 casing_color: #x404a57 casing_width: 2.0 center_color: #x677383 center_width: 1.35}
            MapRoadRule{kind: "service" sort_rank: 240 casing_color: #x3e4753 casing_width: 1.75 center_color: #x5e6a79 center_width: 1.1}
            MapRoadRule{kind: "pedestrian" sort_rank: 240 casing_color: #x3e4753 casing_width: 1.75 center_color: #x5e6a79 center_width: 1.1}
            MapRoadRule{kind: "cycleway" sort_rank: 160 center_color: #x4f5966 center_width: 0.82}
            MapRoadRule{kind: "footway" sort_rank: 160 center_color: #x4f5966 center_width: 0.82}
            MapRoadRule{kind: "path" sort_rank: 160 center_color: #x4f5966 center_width: 0.82}
            MapRoadRule{kind: "steps" sort_rank: 160 center_color: #x4f5966 center_width: 0.82}
            MapRoadRule{kind: "track" sort_rank: 160 center_color: #x4f5966 center_width: 0.82}
            MapRoadRule{kind: "*" sort_rank: 280 casing_color: #x404a57 casing_width: 1.9 center_color: #x606c7b center_width: 1.2}

            MapWaterwayRule{kind: "river" sort_rank: 140 casing_color: #x2f6188 casing_width: 1.83 center_color: #x4f93c8 center_width: 1.55}
            MapWaterwayRule{kind: "canal" sort_rank: 140 casing_color: #x2f6188 casing_width: 1.5 center_color: #x4f93c8 center_width: 1.22}
            MapWaterwayRule{kind: "stream" sort_rank: 140 casing_color: #x2f6188 casing_width: 1.18 center_color: #x4f93c8 center_width: 0.9}
            MapWaterwayRule{kind: "*" sort_rank: 140 casing_color: #x2f6188 casing_width: 1.1 center_color: #x4f93c8 center_width: 0.82}
            MapRailRule{sort_rank: 180 casing_color: #x3f4650 casing_width: 0.96 center_color: #x707783 center_width: 0.62 center_shape_id: 10.0}
        }

        draw_bg +: {
            color: #xddd7cc
        }
        draw_label +: {
            color: #x000000
            text_style: theme.font_regular{font_size: 7}
        }
        draw_text +: {
            color: #xdee9f4
            text_style: theme.font_regular{font_size: 10}
        }
    }
}

// --- Draw shaders ---

/// Reference zoom the navigation route ribbon geometry is tessellated at.
/// Drawn with scale 2^(view_zoom - NAV_REF_Z), exactly like tile geometry.
const NAV_REF_Z: u32 = 16;

/// Seconds of no touch after a manual pan/zoom before the nav camera eases back
/// to the follow-cam (matches typical map apps' auto-recenter behavior).
const NAV_RECENTER_IDLE_SECS: f64 = 4.0;

/// Extra frames the map keeps rendering AFTER motion stops so a just-revealed
/// label finishes its ~0.22s fade-in (and any last tile paints) before going
/// idle. ~16 frames ≈ 0.27s at 60fps, covering the fade.
const NAV_SETTLE_FRAMES: u32 = 16;

/// Ready tile geometry shared across MapView instances on the UI thread.
/// Splash cards that animate re-evaluate ~1 Hz and REBUILD their widget tree,
/// so a per-instance tile cache would refetch + retessellate every second
/// (visible flicker). The owning `Geometry` lives here; instances hold
/// non-owning `Geometry::new_borrowed` handles.
struct SharedReadyTile {
    fill: Option<Geometry>,
    stroke: Option<Geometry>,
    feature_count: usize,
    labels: Vec<TileLabel>,
    last_used: u64,
}

thread_local! {
    static NAV_TILE_STORE: std::cell::RefCell<HashMap<TileKey, SharedReadyTile>> =
        std::cell::RefCell::new(HashMap::new());
    static NAV_TILE_STORE_TICK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    // In-flight network requests, GLOBAL: Splash cards rebuild their widget
    // tree ~1 Hz, and Overpass takes seconds — a per-instance pending map
    // would drop every response (the requester is gone by arrival) and then
    // re-request each second: a self-inflicted request storm. Any live
    // instance can claim a response via this map.
    static NAV_PENDING: std::cell::RefCell<HashMap<LiveId, PendingTileRequest>> =
        std::cell::RefCell::new(HashMap::new());
    // Per-tile request throttle (sim-clock seconds of the last attempt).
    static NAV_REQ_RECENT: std::cell::RefCell<HashMap<TileKey, f64>> =
        std::cell::RefCell::new(HashMap::new());
    // Global request-id counter so ids never collide across instances.
    static NAV_REQ_ID: std::cell::Cell<u64> = const { std::cell::Cell::new(1) };
}

thread_local! {
    // Car position in world px at the request zoom — set every nav draw; used
    // for distance-based store eviction (keep what's near the drive).
    static NAV_DRAW_CENTER: std::cell::Cell<(u32, f64, f64)> =
        const { std::cell::Cell::new((0, 0.0, 0.0)) };
}

/// Collect ALL loaded tiles at `zoom` within `radius` world px of the center,
/// straight from the shared store. This is the nav draw list: once a tile is
/// loaded it is DRAWN every frame it is anywhere near the viewport —
/// fetch/instance bookkeeping can never hide loaded content.
fn nav_store_draw_ids(
    zoom: u32,
    center_x: f64,
    center_y: f64,
    radius: f64,
) -> (
    Vec<(TileKey, GeometryId)>,
    Vec<(TileKey, GeometryId)>,
) {
    let mut keys: Vec<(TileKey, Option<GeometryId>, Option<GeometryId>)> = Vec::new();
    NAV_TILE_STORE.with(|store| {
        let store = store.borrow();
        for (k, t) in store.iter() {
            if k.z != zoom {
                continue;
            }
            let cx_ = (k.x as f64 + 0.5) * TILE_SIZE;
            let cy_ = (k.y as f64 + 0.5) * TILE_SIZE;
            if (cx_ - center_x).abs() > radius || (cy_ - center_y).abs() > radius {
                continue;
            }
            keys.push((
                *k,
                t.fill.as_ref().map(|g| g.geometry_id()),
                t.stroke.as_ref().map(|g| g.geometry_id()),
            ));
        }
    });
    keys.sort_unstable_by_key(|(k, _, _)| (k.y, k.x));
    let fills = keys
        .iter()
        .filter_map(|(k, f, _)| f.map(|g| (*k, g)))
        .collect();
    let strokes = keys
        .iter()
        .filter_map(|(k, _, s)| s.map(|g| (*k, g)))
        .collect();
    (fills, strokes)
}

/// Collect street/place labels from stored tiles near the car, as
/// (world_px_x, world_px_y, text, priority) at `zoom`. One representative
/// point per label (path midpoint), deduped by text (nearest wins). For the
/// nav view's upright projected labels.
fn nav_store_labels(
    zoom: u32,
    center_x: f64,
    center_y: f64,
    radius: f64,
) -> Vec<(f64, f64, String, u8)> {
    let mut best: HashMap<String, (f64, f64, u8, f64)> = HashMap::new();
    NAV_TILE_STORE.with(|store| {
        let store = store.borrow();
        for (k, t) in store.iter() {
            if k.z != zoom || t.labels.is_empty() {
                continue;
            }
            let ox = k.x as f64 * TILE_SIZE;
            let oy = k.y as f64 * TILE_SIZE;
            for lab in &t.labels {
                if lab.path_points.is_empty() {
                    continue;
                }
                let mid = &lab.path_points[lab.path_points.len() / 2];
                let wx = ox + mid.0 as f64;
                let wy = oy + mid.1 as f64;
                let dx = wx - center_x;
                let dy = wy - center_y;
                if dx.abs() > radius || dy.abs() > radius {
                    continue;
                }
                let d2 = dx * dx + dy * dy;
                match best.get(&lab.text) {
                    Some((_, _, _, pd)) if *pd <= d2 => {}
                    _ => {
                        best.insert(lab.text.clone(), (wx, wy, lab.priority, d2));
                    }
                }
            }
        }
    });
    let mut out: Vec<(f64, f64, String, u8)> = best
        .into_iter()
        .map(|(text, (x, y, pri, _))| (x, y, text, pri))
        .collect();
    // Priority first so majors win the per-frame cap, then by NAME as a stable
    // tie-break — the candidate set comes from a HashMap (non-deterministic
    // order), so without this the picked subset flickered between frames.
    out.sort_by(|a, b| a.3.cmp(&b.3).then_with(|| a.2.cmp(&b.2)));
    out
}

// --- OpenFreeMap MVT tile-URL template (resolved from the TileJSON) ---
// The tile path carries a dated version segment that rotates when the planet is
// re-cut, so the live template is read from the TileJSON once at runtime. Until
// it lands, the hardcoded default keeps tiles flowing.
thread_local! {
    static NAV_MVT_TEMPLATE: std::cell::RefCell<String> =
        std::cell::RefCell::new(OPENFREEMAP_DEFAULT_TEMPLATE.to_string());
    // bootstrap state: 0 = not started, 1 = TileJSON fetch in flight, 2 = resolved.
    static NAV_MVT_BOOTSTRAP: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    // the request id of the in-flight TileJSON fetch (to route its response).
    static NAV_MVT_TILEJSON_REQ: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn nav_mvt_template() -> String {
    NAV_MVT_TEMPLATE.with(|t| t.borrow().clone())
}

/// Build a concrete OpenFreeMap tile URL from the resolved template + tile key.
fn nav_mvt_tile_url(tile: TileKey) -> String {
    nav_mvt_template()
        .replace("{z}", &tile.z.to_string())
        .replace("{x}", &tile.x.to_string())
        .replace("{y}", &tile.y.to_string())
}

/// Parse the `tiles[0]` URL template out of the OpenFreeMap TileJSON and adopt
/// it (keeps the dated version segment current as the planet is re-cut).
fn nav_mvt_adopt_tilejson(body: &str) -> bool {
    let Some(v) = serde_json::from_str::<serde_json::Value>(body).ok() else {
        return false;
    };
    if let Some(tmpl) = v
        .get("tiles")
        .and_then(|t| t.get(0))
        .and_then(|u| u.as_str())
    {
        if tmpl.contains("{z}") && tmpl.contains("{x}") && tmpl.contains("{y}") {
            NAV_MVT_TEMPLATE.with(|t| *t.borrow_mut() = tmpl.to_string());
            return true;
        }
    }
    false
}

/// A fetched tile payload moved into the worker thread: raw MVT/PBF bytes (decoded
/// to Overpass-JSON there) or an Overpass-JSON string already.
enum TilePayload {
    Json(String),
    Mvt(Vec<u8>),
}

fn nav_pending_insert(id: LiveId, pending: PendingTileRequest) {
    NAV_PENDING.with(|p| p.borrow_mut().insert(id, pending));
}

fn nav_pending_take(id: &LiveId) -> Option<PendingTileRequest> {
    NAV_PENDING.with(|p| p.borrow_mut().remove(id))
}

fn nav_pending_len() -> usize {
    NAV_PENDING.with(|p| p.borrow().len())
}

/// Request ids for tile fetches.
///
/// MUST come from the same global source as every other `cx.http_request` id.
/// This used to be a private counter starting at 1, which collided head-on with
/// `LiveId::unique()` (also a counter from 1) used by `script_data_fetch` — so
/// the nav card, which fires geocoder searches and tile fetches concurrently,
/// delivered Photon's GeoJSON to the Overpass tile parser and the map stayed
/// blank ("Key not found elements").
fn nav_next_request_id() -> LiveId {
    LiveId::unique()
}

/// True if this tile was requested within the last `window` seconds
/// (and records now as the latest attempt otherwise).
fn nav_req_throttled(key: TileKey, window: f64) -> bool {
    // a request for this tile is already in flight — never double-request
    let in_flight = NAV_PENDING.with(|p| p.borrow().values().any(|q| q.tile_key == key));
    if in_flight {
        return true;
    }
    let now = crate::splash::sim_clock_secs();
    NAV_REQ_RECENT.with(|r| {
        let mut r = r.borrow_mut();
        if let Some(last) = r.get(&key) {
            if now - last < window {
                return true;
            }
        }
        r.insert(key, now);
        if r.len() > 4096 {
            r.retain(|_, t| now - *t < 300.0);
        }
        false
    })
}

thread_local! {
    // Tile parse workers, GLOBAL: a per-instance pool + channel meant every
    // 1 Hz Splash card rebuild spawned (and dropped) a whole thread pool each
    // second, and any parse finishing after its dispatching instance died was
    // silently lost — tiles appeared, vanished on rebuild, refilled late.
    static NAV_TILE_WORKERS: std::cell::RefCell<
        Option<(TagThreadPool<TileKey>, ToUIReceiver<TileWorkerMessage>)>,
    > = const { std::cell::RefCell::new(None) };
}

fn nav_workers_ensure(cx: &mut Cx) {
    NAV_TILE_WORKERS.with(|w| {
        let mut w = w.borrow_mut();
        if w.is_none() {
            let num_threads = cx.cpu_cores().max(3) - 2;
            *w = Some((TagThreadPool::new(cx, num_threads), ToUIReceiver::default()));
        }
    });
}

fn nav_workers_sender() -> ToUISender<TileWorkerMessage> {
    NAV_TILE_WORKERS.with(|w| w.borrow().as_ref().expect("nav workers").1.sender())
}

fn nav_workers_execute(tag: TileKey, job: impl FnOnce(TileKey) + Send + 'static) {
    NAV_TILE_WORKERS.with(|w| {
        w.borrow()
            .as_ref()
            .expect("nav workers")
            .0
            .execute_rev(tag, job)
    });
}

fn nav_workers_try_recv() -> Option<TileWorkerMessage> {
    NAV_TILE_WORKERS.with(|w| {
        w.borrow()
            .as_ref()
            .and_then(|(_, rx)| rx.try_recv().ok())
    })
}

/// Decoded + tessellated route shared across the 1 Hz card rebuilds — the
/// owning ribbon Geometry lives here; instances borrow it. Keyed by the
/// polyline content hash.
struct SharedNavRoute {
    geom: Option<Geometry>,
    origin: (f64, f64), // world px at NAV_REF_Z the geometry is rebased to
    pts: Rc<Vec<Vec2d>>,
    cum: Rc<Vec<f64>>,
    seq: u64, // last-used tick for LRU eviction (see ensure_nav_route: never bulk-clear)
}

thread_local! {
    static NAV_ROUTE_STORE: std::cell::RefCell<HashMap<u64, SharedNavRoute>> =
        std::cell::RefCell::new(HashMap::new());
    // Monotonic "last used" clock for NAV_ROUTE_STORE LRU eviction.
    static NAV_ROUTE_SEQ: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Next value of the NAV_ROUTE_STORE LRU clock (bumped on every adopt + insert).
fn next_nav_route_seq() -> u64 {
    NAV_ROUTE_SEQ.with(|c| {
        let v = c.get().wrapping_add(1);
        c.set(v);
        v
    })
}

/// Adopt a shared ready tile as a per-instance TileEntry (borrowed handles).
fn nav_store_adopt(key: &TileKey, frame_counter: u64) -> Option<TileEntry> {
    NAV_TILE_STORE.with(|store| {
        let mut store = store.borrow_mut();
        let shared = store.get_mut(key)?;
        let tick = NAV_TILE_STORE_TICK.with(|t| {
            let v = t.get() + 1;
            t.set(v);
            v
        });
        shared.last_used = tick;
        Some(TileEntry {
            state: TileLoadState::Ready {
                fill_geometry: shared
                    .fill
                    .as_ref()
                    .map(|g| Geometry::new_borrowed(g.geometry_id())),
                stroke_geometry: shared
                    .stroke
                    .as_ref()
                    .map(|g| Geometry::new_borrowed(g.geometry_id())),
                feature_count: shared.feature_count,
                labels: shared.labels.clone(),
            },
            last_used: frame_counter,
            attempts: 0,
        })
    })
}

/// Store owning geometry in the shared store (evicting LRU beyond a cap) and
/// return a borrowed-handle TileEntry for the calling instance.
fn nav_store_insert(
    key: TileKey,
    fill: Option<Geometry>,
    stroke: Option<Geometry>,
    feature_count: usize,
    labels: Vec<TileLabel>,
    frame_counter: u64,
) -> TileEntry {
    let entry = TileEntry {
        state: TileLoadState::Ready {
            fill_geometry: fill
                .as_ref()
                .map(|g| Geometry::new_borrowed(g.geometry_id())),
            stroke_geometry: stroke
                .as_ref()
                .map(|g| Geometry::new_borrowed(g.geometry_id())),
            feature_count,
            labels: labels.clone(),
        },
        last_used: frame_counter,
        attempts: 0,
    };
    NAV_TILE_STORE.with(|store| {
        let mut store = store.borrow_mut();
        let tick = NAV_TILE_STORE_TICK.with(|t| {
            let v = t.get() + 1;
            t.set(v);
            v
        });
        store.insert(
            key,
            SharedReadyTile {
                fill,
                stroke,
                feature_count,
                labels,
                last_used: tick,
            },
        );
        if store.len() > 900 {
            // Cap the shared owner store to BOUND MEMORY. The 3D chase view pulls a
            // wide tile radius (turn/horizon prefetch), so an unbounded store fills
            // to ~2.5 GB on entering drive and parks the process at the Android OOM
            // ceiling — then any later allocation (even a bottom-sheet redraw) aborts
            // with SIGABRT ("allocation failed"). 900 sits comfortably above the
            // active working set (per-instance cap 640), so only never-drawn FAR
            // history is trimmed — nothing visible changes.
            // Evict FARTHEST from the drive first (never what's near the car —
            // loaded content close to the viewport must not vanish; drawing only
            // ever touches the nearest ~86 tiles, so a drawn tile is never evicted).
            // Falls back to LRU when no nav draw has run yet.
            let (cz, cx_, cy_) = NAV_DRAW_CENTER.with(|c| c.get());
            let mut ranked: Vec<(u64, TileKey)> = store
                .iter()
                .map(|(k, v)| {
                    let rank = if cz != 0 {
                        if k.z == cz {
                            let dx = (k.x as f64 + 0.5) * TILE_SIZE - cx_;
                            let dy = (k.y as f64 + 0.5) * TILE_SIZE - cy_;
                            (dx * dx + dy * dy) as u64
                        } else {
                            u64::MAX / 2
                        }
                    } else {
                        u64::MAX - v.last_used
                    };
                    (rank, *k)
                })
                .collect();
            ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            for (_, k) in ranked.into_iter().take(96) {
                store.remove(&k);
            }
        }
    });
    entry
}

/// Decode a Google/OSRM polyline5 string into (lat, lon) pairs.
fn decode_polyline5(encoded: &str) -> Vec<(f64, f64)> {
    let bytes = encoded.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4);
    let (mut lat, mut lon): (i64, i64) = (0, 0);
    let mut i = 0usize;
    while i < bytes.len() {
        let decode_one = |i: &mut usize| -> Option<i64> {
            let (mut shift, mut result): (u32, i64) = (0, 0);
            loop {
                if *i >= bytes.len() {
                    return None;
                }
                let b = bytes[*i] as i64 - 63;
                *i += 1;
                if b < 0 {
                    return None;
                }
                result |= (b & 0x1f) << shift;
                shift += 5;
                if b < 0x20 {
                    break;
                }
            }
            Some(if result & 1 != 0 {
                !(result >> 1)
            } else {
                result >> 1
            })
        };
        let Some(dlat) = decode_one(&mut i) else { break };
        let Some(dlon) = decode_one(&mut i) else { break };
        lat += dlat;
        lon += dlon;
        out.push((lat as f64 * 1e-5, lon as f64 * 1e-5));
    }
    out
}

fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (la1, lo1, la2, lo2) = (
        lat1.to_radians(),
        lon1.to_radians(),
        lat2.to_radians(),
        lon2.to_radians(),
    );
    let a = ((la2 - la1) / 2.0).sin().powi(2)
        + la1.cos() * la2.cos() * ((lo2 - lo1) / 2.0).sin().powi(2);
    6371000.0 * 2.0 * a.sqrt().asin()
}

/// Meters per world pixel at `zoom` and latitude (256px web-mercator tiles).
fn meters_per_world_px(lat_deg: f64, zoom: f64) -> f64 {
    40075016.686 * lat_deg.to_radians().cos() / tile_world_size_zoom(zoom)
}

/// Latitude (degrees) of a normalized web-mercator y.
fn normalized_y_to_lat(y: f64) -> f64 {
    let n = std::f64::consts::PI * (1.0 - 2.0 * y);
    n.sinh().atan().to_degrees()
}

/// Per-frame navigation-projection parameters pushed into the DrawMapVector
/// shader. `mode` 0 = flat map, 1 = 3D chase FPV, 2 = 2D heading-up.
#[derive(Clone, Copy, Debug)]
pub struct NavShaderParams {
    pub mode: f32,
    pub anchor: Vec2f,
    pub rot: Vec2f,   // (sin bearing, cos bearing)
    pub cam: [f32; 4],  // cam_h_px, pitch, vfov/2, tan(hfov/2)
    pub screen: [f32; 4], // widget rect x,y,w,h
    pub misc: [f32; 4],  // chase_px, maxg_px, car2d_row_px, 0
    pub haze: [f32; 4],  // rgb + pad
}

impl Default for NavShaderParams {
    fn default() -> Self {
        Self {
            mode: 0.0,
            anchor: vec2(0.0, 0.0),
            rot: vec2(0.0, 1.0),
            cam: [40.0, 0.315, 0.49, 0.76],
            screen: [0.0; 4],
            misc: [90.0, 500.0, 600.0, 0.0],
            haze: [0.847, 0.890, 0.929, 0.0],
        }
    }
}

#[derive(Script, ScriptHook, Debug)]
#[repr(C)]
pub struct DrawMapVector {
    #[deref]
    pub draw_super: DrawVector,
    #[rust(vec2(1.0, 1.0))]
    pub map_scale: Vec2f,
    #[rust(vec2(0.0, 0.0))]
    pub map_offset: Vec2f,
    #[rust(NavShaderParams::default())]
    pub nav: NavShaderParams,
}

impl DrawMapVector {
    fn draw_geometry(
        &mut self,
        cx: &mut Cx2d,
        geometry_id: GeometryId,
        map_scale: Vec2f,
        map_offset: Vec2f,
    ) {
        self.map_scale = map_scale;
        self.map_offset = map_offset;
        self.draw_super.draw_vars.set_uniform(
            cx.cx,
            live_id!(map_scale),
            &[map_scale.x, map_scale.y],
        );
        self.draw_super.draw_vars.set_uniform(
            cx.cx,
            live_id!(map_offset),
            &[map_offset.x, map_offset.y],
        );
        let nav = self.nav;
        self.draw_super
            .draw_vars
            .set_uniform(cx.cx, live_id!(nav_mode), &[nav.mode]);
        self.draw_super.draw_vars.set_uniform(
            cx.cx,
            live_id!(nav_anchor),
            &[nav.anchor.x, nav.anchor.y],
        );
        self.draw_super
            .draw_vars
            .set_uniform(cx.cx, live_id!(nav_rot), &[nav.rot.x, nav.rot.y]);
        self.draw_super
            .draw_vars
            .set_uniform(cx.cx, live_id!(nav_cam), &nav.cam);
        self.draw_super
            .draw_vars
            .set_uniform(cx.cx, live_id!(nav_screen), &nav.screen);
        self.draw_super
            .draw_vars
            .set_uniform(cx.cx, live_id!(nav_misc), &nav.misc);
        self.draw_super
            .draw_vars
            .set_uniform(cx.cx, live_id!(nav_haze), &nav.haze);
        self.draw_super.draw_vars.geometry_id = Some(geometry_id);
        cx.new_draw_call(&self.draw_super.draw_vars);
        if self.draw_super.draw_vars.can_instance() {
            let new_area = cx.add_aligned_instance(&self.draw_super.draw_vars);
            self.draw_super.draw_vars.area =
                cx.update_area_refs(self.draw_super.draw_vars.area, new_area);
        }
    }
}

// --- MapView widget ---

#[derive(Script, Widget)]
pub struct MapView {
    #[uid]
    uid: WidgetUid,
    #[source]
    source: ScriptObjectRef,
    #[walk]
    walk: Walk,
    #[layout]
    layout: Layout,

    #[redraw]
    #[live]
    draw_bg: DrawColor,
    #[redraw]
    #[live]
    draw_map: DrawMapVector,
    #[redraw]
    #[live]
    draw_label: DrawRotatedText,
    #[redraw]
    #[live]
    draw_text: DrawText,
    // Small solid quads for nav label ground-markers (leader line) and the
    // vehicle puck — a separate DrawColor so it never clobbers `draw_bg`'s area
    // (which nav hit-testing reads).
    #[redraw]
    #[live]
    draw_dot: DrawColor,
    // Screen-space vector drawer for the vehicle puck + standing pins — precise
    // circles/triangles (no glyph-metric guessing). Not fed the nav-projection
    // uniforms, so it draws flat in pixel space.
    #[redraw]
    #[live]
    draw_ui: DrawVector,

    #[live(4.9041)]
    center_lon: f64,
    #[live(52.3676)]
    center_lat: f64,
    #[live(14.0)]
    zoom: f64,
    #[live(11.0)]
    min_zoom: f64,
    #[live(17.0)]
    max_zoom: f64,
    #[live(false)]
    dark_theme: bool,
    #[live]
    style_light: MapThemeStyle,
    #[live]
    style_dark: MapThemeStyle,
    #[live(true)]
    use_network: bool,
    #[live(true)]
    use_local_mbtiles: bool,
    // Online tile source for the NAV map: true = OpenFreeMap MVT (CDN, reliable);
    // false = Overpass raw-OSM (rate-limited public mirrors, the blank-map risk).
    // Only consulted when use_network && !use_local_mbtiles.
    #[live(true)]
    use_mvt: bool,

    // --- turn-by-turn navigation mode ---
    // "" = normal map; "3d" = first-person chase view (heading-up, pinhole
    // ground projection); "2d" = top-down heading-up. The vehicle follows
    // `nav_polyline` (OSRM polyline5) at `nav_speed_mph`, looping every
    // `nav_period` seconds on the same clock as `sys.simsecs` so DSL overlays
    // (banner countdown etc.) stay in lockstep with the map.
    #[live]
    nav_mode: ArcStringMut,
    #[live]
    nav_polyline: ArcStringMut,
    #[live(92.0)]
    nav_period: f64,
    #[live(34.0)]
    nav_speed_mph: f64,
    #[live(56.0)]
    nav_cam_h: f64, // camera height above ground, meters — higher vantage so
    // near-field features stream past more gently at speed (less "too rapid")
    #[live(0.37)]
    nav_pitch: f64, // camera pitch below horizon, radians (steeper to hold the
    // horizon in place after raising the camera)
    #[live(0.98)]
    nav_vfov: f64, // vertical field of view, radians
    #[live(1.30)]
    nav_hfov: f64, // horizontal field of view, radians
    #[live(800.0)]
    nav_maxg: f64, // far haze clip, meters
    #[live(0.62)]
    nav_carv: f64, // vehicle screen row in 3D (fraction of height)
    #[live(0.60)]
    nav_carv2d: f64, // vehicle screen row in 2D heading-up
    #[live(15.0)]
    nav_route_width: f64, // route ribbon core width, ground meters

    // Route annotation pins: (lat, lon, kind) where kind 0 = origin (green),
    // 1 = intermediate/途经点 (blue), 2 = destination (red). Set by the card
    // via `ui.<id>.set_route_markers("lat,lon,kind;lat,lon,kind;…")`. Drawn as
    // world-space dots appended to the ribbon geometry so they project through
    // the same shader in both plan and 3D modes (no CPU projection needed).
    #[rust]
    nav_markers: Vec<(f64, f64, u8)>,

    #[rust]
    nav_pts: Rc<Vec<Vec2d>>, // decoded route, normalized world coords
    #[rust]
    nav_cum: Rc<Vec<f64>>, // cumulative route meters
    #[rust]
    nav_poly_hash: u64,
    #[rust]
    nav_route_geom: Option<Geometry>,
    #[rust]
    nav_route_origin: (f64, f64),
    // labels drawn last frame — kept STICKY so the on-screen set doesn't
    // flicker frame-to-frame (the recompute/dedup/overlap order was flipping
    // which names showed each frame, which read as text "vanishing").
    #[rust]
    nav_labels_shown: HashSet<String>,
    // candidate labels (world_x, world_y, text, priority), refreshed a few Hz
    // (the store scan + string clones are too costly to redo every 60fps frame)
    #[rust]
    nav_labels_cache: Vec<(f64, f64, String, u8)>,
    // plan-overview label -> sim-clock time it FIRST appeared, so a newly-shown
    // name FADES IN (alpha ramp) instead of popping. Pruned to the visible set
    // each frame so a name that leaves and returns fades in again.
    #[rust]
    nav_label_seen: HashMap<String, f64>,
    // low-pass smoothed camera heading (radians) so turns rotate the map/ribbon
    // gently instead of whipping around (the abrupt swing read as the ribbon
    // vanishing from the bottom mid-turn)
    #[rust]
    nav_bearing: f64,
    #[rust]
    nav_bearing_init: bool,
    #[rust]
    nav_next_frame: NextFrame,
    // Frames still owed AFTER motion stops, so a just-revealed label finishes its
    // ~0.22s fade-in (and any in-flight tile paints) before the map goes fully
    // idle. Re-armed to NAV_SETTLE_FRAMES on any motion / tile arrival, counted
    // down each static frame. Gates the `nav_next_frame` re-arm so a STATIC
    // plan/preview map stops requesting frames (was pinning the GPU at ~100%).
    #[rust]
    nav_settle_frames: u32,
    // Gesture disambiguation state. `nav_gesture_on_map`: this gesture was claimed by
    // the map (a finger landed on it) — once claimed, pan/pinch is decided from ALL
    // active fingers, not just those still inside the (short) map rect, so a finger
    // drifting off the map doesn't flip pinch<->pan. `nav_gest`: committed gesture
    // (0 idle · 1 pan · 2 pinch). `nav_gest_hold`: frames a NEW finger-count has
    // persisted — a switch commits only after a few frames, rejecting momentary
    // flickers (a palm graze, or pinch jitter dropping a finger).
    #[rust]
    nav_gesture_on_map: bool,
    #[rust]
    nav_gest: u8,
    #[rust]
    nav_gest_hold: u8,

    #[rust]
    center_norm: Vec2d,
    #[rust]
    view_rect: Rect,
    #[rust]
    drag_start_abs: Option<Vec2d>,
    #[rust]
    drag_start_center_norm: Vec2d,
    // --- nav-mode touch gestures (pinch-zoom / pan) + recenter ---
    // The card's configured nav zoom, captured once, so the recenter button can
    // restore it after the user pinches/pans.
    #[rust]
    nav_home_zoom: f64,
    // Normalized center offset from the car (0,0 = follow the car). Single-finger
    // drag accumulates it; recenter clears it.
    #[rust]
    nav_pan: Vec2d,
    // Latest position of every ACTIVE touch (raw uid -> abs-pos), accumulated
    // across TouchUpdate events. A real 2-finger pinch is delivered as separate
    // per-finger events, so no single event carries both fingers — we must track
    // them to tell a genuine pinch (2 tracked) from a one-finger pan (1 tracked).
    #[rust]
    nav_touches: Vec<(u64, Vec2d)>,
    // (finger distance, zoom) captured when the 2nd finger lands — the pinch base.
    #[rust]
    nav_pinch_base: Option<(f64, f64, Vec2d)>,
    // (abs-pos, nav_pan) captured when a 1-finger drag begins — the pan anchor.
    // Pan is computed ABSOLUTELY from this anchor (not incrementally per move) so
    // a partial/coalesced FingerMove stream still yields a 1:1 drag; re-anchored
    // when a pinch releases back to one finger to avoid a jump.
    #[rust]
    nav_pan_drag: Option<(Vec2d, Vec2d)>,
    // Eased-animation TARGETS for the plan overview. The +/- buttons and the
    // my-location button set a target and the camera GLIDES to it (Google-style)
    // each frame; a direct-manipulation gesture (pinch/drag) clears the target and
    // takes over instantly. `Some` == animating that axis.
    #[rust]
    nav_zoom_anim: Option<f64>,
    #[rust]
    nav_pan_anim: Option<Vec2d>,
    // Auto-restore (like Google/Apple Maps): after the user pans/zooms, wait
    // IDLE seconds of no touch, then ease the camera back to the follow-cam
    // (nav_pan -> 0, zoom -> nav_home_zoom). `nav_last_touch` is the sim-clock
    // stamp of the last gesture; `nav_user_adjusted` gates the restore so it
    // only runs after an actual manual pan/zoom.
    #[rust]
    nav_last_touch: f64,
    #[rust]
    nav_user_adjusted: bool,
    // The car's world position (normalized) this frame — projected in the draw
    // path to place the moving vehicle puck.
    #[rust]
    nav_car_norm: Vec2d,
    #[rust]
    tiles: HashMap<TileKey, TileEntry>,
    #[rust]
    #[rust]
    next_request_id: u64,
    #[rust]
    visible_tiles: Vec<TileKey>,
    #[rust]
    frame_counter: u64,
    #[rust]
    status: String,
    #[rust]
    label_perf: LabelPerfStats,
    #[rust]
    local_source_missing_logged: bool,
    #[rust]
    #[rust]
    local_requested_tiles: HashSet<TileKey>,
    #[rust]
    local_missing_tiles: HashSet<TileKey>,
    #[rust]
    applied_dark_theme: Option<bool>,
    #[rust]
    style_epoch: u64,
    #[rust]
    compiled_style_light: CompiledMapTheme,
    #[rust]
    compiled_style_dark: CompiledMapTheme,
    #[rust]
    path_glyphs: Vec<PathGlyphInstance>,
    // Scratch buffers reused across frames to avoid per-frame allocations
    #[rust]
    scratch_draw_tiles: Vec<TileKey>,
    #[rust]
    scratch_draw_seen: HashSet<TileKey>,
    #[rust]
    scratch_descendant_tiles: Vec<TileKey>,
    #[rust]
    scratch_candidates: Vec<LabelCandidate>,
    #[rust]
    scratch_accepted_centers: HashMap<String, Vec<Vec2d>>,
    #[rust]
    scratch_accepted_bounds: Vec<Rect>,
    #[rust]
    scratch_accepted_plans: Vec<(f64, usize, usize)>,
    #[rust]
    scratch_screen_path: Vec<Vec2d>,
    #[rust]
    scratch_cumulative: Vec<f64>,
    #[rust]
    scratch_smooth_a: Vec<Vec2d>,
    #[rust]
    scratch_smooth_b: Vec<Vec2d>,
    #[rust]
    prev_status_label_perf: LabelPerfStats,
    #[rust]
    prev_status_counters: (usize, usize, usize, usize, usize, usize),
}

impl ScriptHook for MapView {
    fn on_after_apply(
        &mut self,
        _vm: &mut ScriptVm,
        apply: &Apply,
        _scope: &mut Scope,
        _value: ScriptValue,
    ) {
        if apply.is_eval() {
            return;
        }

        let min_zoom = self.min_zoom.max(0.0);
        let max_zoom = self.max_zoom.max(min_zoom);
        self.zoom = self.zoom.clamp(min_zoom, max_zoom);
        self.center_norm = lon_lat_to_normalized(self.center_lon, self.center_lat);
        self.wrap_and_clamp_center();
        self.normalize_source_mode();

        let previous_light = self.compiled_style_light.clone();
        let previous_dark = self.compiled_style_dark.clone();
        self.rebuild_compiled_styles();
        let styles_changed = previous_light != self.compiled_style_light
            || previous_dark != self.compiled_style_dark;
        if self.style_epoch == 0 {
            self.style_epoch = 1;
        }

        let theme_changed = self.applied_dark_theme != Some(self.dark_theme);
        if theme_changed || styles_changed {
            self.apply_theme_change();
            self.applied_dark_theme = Some(self.dark_theme);
        } else {
            self.apply_theme_palette();
        }

        if self.next_request_id == 0 {
            self.next_request_id = 1;
        }
        ensure_cache_dir();
        if self.status.is_empty() {
            self.status = "Loading Amsterdam tiles from local cache/mbtiles...".to_string();
        }
    }
}

impl Widget for MapView {
    // `ui.<id>.set_nav_polyline("<polyline5>")` — push the OSRM route into the
    // MapView from a card's `fn tick()` without re-evaluating the card. Lets a
    // zero-rebuild nav card feed the route once it loads (empty/unchanged is a
    // no-op; ensure_nav_route's content-hash guard tessellates only on change).
    fn script_call(
        &mut self,
        vm: &mut ScriptVm,
        method: LiveId,
        args: ScriptValue,
    ) -> ScriptAsyncResult {
        if method == live_id!(set_nav_polyline) {
            if let Some(args_obj) = args.as_object() {
                let trap = vm.bx.threads.cur().trap.pass();
                let value = vm.bx.heap.vec_value(args_obj, 0, trap);
                if !value.is_err() {
                    let s = vm.bx.heap.temp_string_with(|heap, out| {
                        heap.cast_to_string(value, out);
                        out.to_string()
                    });
                    if !s.trim().is_empty() && s.as_str() != self.nav_polyline.as_ref() {
                        self.nav_polyline.as_mut_empty().push_str(&s);
                        vm.with_cx_mut(|cx| self.redraw(cx));
                        crate::splash::splash_mark_tick_changed();
                    }
                }
            }
            return ScriptAsyncResult::Return(NIL);
        }
        // `ui.<id>.set_route_markers("lat,lon,kind;lat,lon,kind;…")` — the
        // trip-planner pins: origin (kind 0), each 途经点/intermediate stop
        // (kind 1), and destination (kind 2). Parsed into `nav_markers` and
        // re-tessellated into the route geometry so the dots project through
        // the same shader as the ribbon. Empty string clears the pins.
        if method == live_id!(set_route_markers) {
            if let Some(args_obj) = args.as_object() {
                let trap = vm.bx.threads.cur().trap.pass();
                let value = vm.bx.heap.vec_value(args_obj, 0, trap);
                if !value.is_err() {
                    let s = vm.bx.heap.temp_string_with(|heap, out| {
                        heap.cast_to_string(value, out);
                        out.to_string()
                    });
                    let mut next: Vec<(f64, f64, u8)> = Vec::new();
                    for part in s.split(';') {
                        let f: Vec<&str> = part.split(',').collect();
                        if f.len() >= 3 {
                            if let (Ok(lat), Ok(lon), Ok(kind)) = (
                                f[0].trim().parse::<f64>(),
                                f[1].trim().parse::<f64>(),
                                f[2].trim().parse::<u8>(),
                            ) {
                                if lat != 0.0 || lon != 0.0 {
                                    next.push((lat, lon, kind));
                                }
                            }
                        }
                    }
                    if next != self.nav_markers {
                        self.nav_markers = next;
                        self.nav_poly_hash = 0; // force ensure_nav_route re-tessellate
                        vm.with_cx_mut(|cx| self.redraw(cx));
                        crate::splash::splash_mark_tick_changed();
                    }
                }
            }
            return ScriptAsyncResult::Return(NIL);
        }
        // `ui.<id>.set_nav_recenter(...)` — the recenter button: drop the user's
        // pan + pinch and snap back to following the car at the card's zoom.
        if method == live_id!(set_nav_recenter) {
            self.nav_pan = dvec2(0.0, 0.0);
            if self.nav_home_zoom > 0.0 {
                self.zoom = self.nav_home_zoom;
            }
            self.nav_pinch_base = None;
            self.nav_touches.clear();
            self.nav_zoom_anim = None;
            self.nav_pan_anim = None;
            self.nav_user_adjusted = false;
            vm.with_cx_mut(|cx| self.redraw(cx));
            return ScriptAsyncResult::Return(NIL);
        }
        // `ui.<id>.nav_zoom_by(delta)` — the +/- zoom controls on the plan/nav
        // map: step the zoom and mark the camera user-adjusted so the fit stops
        // fighting it (a precise complement to pinch-zoom, e.g. one-handed use).
        if method == live_id!(nav_zoom_by) {
            if let Some(args_obj) = args.as_object() {
                let trap = vm.bx.threads.cur().trap.pass();
                let value = vm.bx.heap.vec_value(args_obj, 0, trap);
                if !value.is_err() {
                    let s = vm.bx.heap.temp_string_with(|heap, out| {
                        heap.cast_to_string(value, out);
                        out.to_string()
                    });
                    let delta: f64 = s.trim().parse().unwrap_or(0.0);
                    let zmin = self.min_zoom.max(3.0);
                    let zmax = self.max_zoom.max(zmin);
                    // animate toward the new zoom (glide, not snap) — the plan
                    // camera eases self.zoom -> nav_zoom_anim each frame.
                    let base = self.nav_zoom_anim.unwrap_or(self.zoom);
                    self.nav_zoom_anim = Some((base + delta).clamp(zmin, zmax));
                    self.nav_user_adjusted = true;
                    self.nav_last_touch = crate::splash::sim_clock_secs();
                    vm.with_cx_mut(|cx| self.redraw(cx));
                }
            }
            return ScriptAsyncResult::Return(NIL);
        }
        // `ui.<id>.nav_center_origin(...)` — the "my location" button on the plan
        // overview. octos has no live GPS, so the trip ORIGIN is the current-
        // location proxy: recenter on it at a street-level zoom, like a maps app's
        // locate button. Marks the camera user-adjusted so the route-fit stops
        // fighting it; the plan center is `nav_car_norm` (route centre) + `nav_pan`
        // every frame, so we center on the origin by offsetting the pan from it.
        if method == live_id!(nav_center_origin) {
            if let Some(&(lat, lon, _)) = self.nav_markers.first() {
                let o = lon_lat_to_normalized(lon, lat);
                let zmin = self.min_zoom.max(3.0);
                let zmax = self.max_zoom.max(zmin);
                // GLIDE to the origin (animate zoom + pan targets) instead of
                // jumping. plan center == nav_car_norm (route centre) + nav_pan
                // each frame, so target the pan that lands center on the origin.
                self.nav_zoom_anim = Some(16.0_f64.clamp(zmin, zmax));
                self.nav_pan_anim =
                    Some(dvec2(o.x - self.nav_car_norm.x, o.y - self.nav_car_norm.y));
                self.nav_user_adjusted = true;
                self.nav_last_touch = crate::splash::sim_clock_secs();
                vm.with_cx_mut(|cx| self.redraw(cx));
            }
            return ScriptAsyncResult::Return(NIL);
        }
        ScriptAsyncResult::MethodNotFound
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        self.handle_tile_worker_messages(cx);
        self.widget_match_event(cx, event, scope);

        if let Event::KeyDown(ke) = event {
            if ke.key_code == KeyCode::KeyT {
                self.set_dark_theme(cx, !self.dark_theme);
            }
        }

        if self.nav_kind() > 0 {
            // Navigation drives the camera, but allow a touch PINCH to zoom and a
            // one-finger PAN to look around; the recenter button restores follow.
            // capture_overload = false so single taps on the card's overlay
            // buttons (2D/3D, recenter, Exit) still reach them.
            if self.nav_next_frame.is_event(event).is_some() {
                self.redraw(cx);
            }
            if self.nav_home_zoom <= 0.0 {
                self.nav_home_zoom = self.zoom;
            }
            let zmin = self.min_zoom.max(0.0);
            let zmax = self.max_zoom.max(zmin);

            // Safety: whenever no finger is on the map, make sure the scroll-block
            // (set below during a pan/pinch) is cleared, so it can never persist past
            // a gesture and wedge scrolling on other cards. The gesture re-blocks per
            // TouchUpdate. (This block is a Cx global with no per-frame auto-reset.)
            if self.nav_touches.is_empty() {
                cx.unblock_scrolling();
            }

            // Claim the gesture for the map ONLY when a finger lands on the EXPOSED
            // map surface. hits() honours z-order, so a finger on the overlaid bottom
            // sheet / control pills / buttons returns Miss here and the map does NOT
            // pan under them. (Before this, the full-card map rect swallowed every
            // touch on the sheet — tapping the sheet just panned the map instead of
            // reaching its buttons, which reads as "app not responding".)
            // capture_overload=false leaves those overlay taps for the widgets on top.
            if let Hit::FingerDown(_) =
                event.hits_with_capture_overload(cx, self.draw_bg.area(), false)
            {
                self.nav_gesture_on_map = true;
            }

            // Touch gestures (one-finger PAN, two-finger PINCH-zoom). Android reports
            // EVERY active pointer on each event (the JNI loops getPointerCount), so
            // `te.touches` is the authoritative current set — REBUILD from it (a missed
            // Stop otherwise leaves a phantom finger that turns a pan into a pinch).
            // Once a finger has landed ON the map the gesture is "claimed" (above); from
            // then on pan vs pinch is decided from ALL active fingers, not just those
            // still inside the map rect, so a finger drifting off doesn't flip
            // pinch<->pan. Switches between pan and pinch are debounced a few frames to
            // reject momentary flickers (a palm graze, or pinch jitter).
            if let Event::TouchUpdate(te) = event {
                let active: Vec<_> = te
                    .touches
                    .iter()
                    .filter(|t| {
                        !matches!(t.state, crate::makepad_platform::event::TouchState::Stop)
                    })
                    .map(|t| t.abs)
                    .collect();
                self.nav_touches = te
                    .touches
                    .iter()
                    .filter(|t| {
                        !matches!(t.state, crate::makepad_platform::event::TouchState::Stop)
                    })
                    .map(|t| (t.uid, t.abs))
                    .collect();
                // map rect — used by the pinch focal-point math below
                let rect = self.draw_bg.area().clipped_rect(cx);

                if active.is_empty() {
                    // all fingers up — end the gesture and release the scroll block
                    self.nav_gesture_on_map = false;
                    self.nav_gest = 0;
                    self.nav_gest_hold = 0;
                    self.nav_pinch_base = None;
                    self.nav_pan_drag = None;
                    cx.unblock_scrolling();
                } else {
                    // the gesture was already claimed above via hits() (a finger on the
                    // EXPOSED map only). A finger that landed on the sheet/buttons leaves
                    // nav_gesture_on_map false, so it falls through here without panning.
                    if self.nav_gesture_on_map {
                        // block the enclosing drag-scrolling PortalList so pan/pinch move
                        // the MAP, not the card (it checks is_scrolling_allowed_within()
                        // before entering its drag, during child event-forwarding).
                        cx.block_scrolling_except_within(self.draw_bg.area());
                        // debounce pan<->pinch switches; the FIRST gesture commits at once
                        let n = active.len().min(2) as u8;
                        if self.nav_gest == 0 {
                            self.nav_gest = n;
                        } else if n != self.nav_gest {
                            self.nav_gest_hold += 1;
                            if self.nav_gest_hold >= 3 {
                                self.nav_gest = n;
                                self.nav_gest_hold = 0;
                            }
                        } else {
                            self.nav_gest_hold = 0;
                        }
                        let pts = &active;
                        if self.nav_gest >= 2 && pts.len() >= 2 {
                            // ---- two-finger PINCH, anchored on the focal point ----
                            let d = ((pts[0].x - pts[1].x).powi(2)
                                + (pts[0].y - pts[1].y).powi(2))
                            .sqrt()
                            .max(1.0);
                            match self.nav_pinch_base {
                                Some((base_d, base_z, fp0)) => {
                                    // Zoom around the FIXED focal point captured at pinch
                                    // start (`fp0`), not the live midpoint — a fixed
                                    // anchor keeps the spot you pinched put and only pans
                                    // to compensate for the SCALE change. PINCH_SENS > 1
                                    // makes a modest spread zoom snappily.
                                    const PINCH_SENS: f64 = 1.8;
                                    let prev_zoom = self.zoom;
                                    self.zoom = (base_z + PINCH_SENS * (d / base_d).log2())
                                        .clamp(zmin, zmax);
                                    let sc = dvec2(
                                        rect.pos.x + rect.size.x * 0.5,
                                        rect.pos.y + rect.size.y * 0.5,
                                    );
                                    let wo = tile_world_size_zoom(prev_zoom).max(1e-6);
                                    let wn = tile_world_size_zoom(self.zoom).max(1e-6);
                                    self.nav_pan.x += (fp0.x - sc.x) * (1.0 / wo - 1.0 / wn);
                                    self.nav_pan.y += (fp0.y - sc.y) * (1.0 / wo - 1.0 / wn);
                                    self.nav_zoom_anim = None;
                                    self.nav_pan_anim = None;
                                    self.nav_user_adjusted = true;
                                    self.nav_last_touch = crate::splash::sim_clock_secs();
                                    self.redraw(cx);
                                }
                                None => {
                                    // pinch just began — capture base distance, zoom, and
                                    // the finger midpoint as the fixed focal anchor
                                    let fp0 =
                                        dvec2((pts[0].x + pts[1].x) * 0.5, (pts[0].y + pts[1].y) * 0.5);
                                    self.nav_pinch_base = Some((d, self.view_zoom(), fp0));
                                }
                            }
                            self.nav_pan_drag = None;
                        } else if self.nav_gest == 1 && !pts.is_empty() {
                            // ---- one-finger PAN (absolute from the drag anchor) ----
                            self.nav_pinch_base = None;
                            let p = pts[0];
                            if let Some((abs0, pan0)) = self.nav_pan_drag {
                                let world = tile_world_size_zoom(self.view_zoom());
                                self.nav_pan.x = pan0.x - (p.x - abs0.x) / world;
                                self.nav_pan.y = pan0.y - (p.y - abs0.y) / world;
                                self.nav_pan_anim = None;
                                self.nav_user_adjusted = true;
                                self.nav_last_touch = crate::splash::sim_clock_secs();
                                self.redraw(cx);
                            } else {
                                // anchor a new drag (re-anchors after a pinch releases to
                                // one finger, so the pan doesn't jump)
                                self.nav_pan_drag = Some((p, self.nav_pan));
                            }
                        }
                        // else: committed gesture vs live-finger count mismatch mid-switch
                        // — hold (no pan, no zoom) until the debounce settles
                    }
                }
            }
            return;
        }

        match event.hits_with_capture_overload(cx, self.draw_bg.area(), true) {
            Hit::FingerDown(fe) if fe.is_primary_hit() => {
                self.drag_start_abs = Some(fe.abs);
                self.drag_start_center_norm = self.center_norm;
                cx.set_cursor(MouseCursor::Grabbing);
            }
            Hit::FingerMove(fe) => {
                if let Some(start_abs) = self.drag_start_abs {
                    let delta = fe.abs - start_abs;
                    let world_size = tile_world_size_zoom(self.view_zoom());
                    self.center_norm = self.drag_start_center_norm
                        - dvec2(delta.x / world_size, delta.y / world_size);
                    self.wrap_and_clamp_center();
                    self.redraw(cx);
                }
            }
            Hit::FingerUp(_) => {
                self.drag_start_abs = None;
                cx.set_cursor(MouseCursor::Grab);
            }
            Hit::FingerHoverIn(_) => {
                cx.set_cursor(MouseCursor::Grab);
            }
            Hit::FingerScroll(fs) => {
                let scroll = if fs.scroll.y.abs() > f64::EPSILON {
                    fs.scroll.y
                } else {
                    fs.scroll.x
                };
                self.zoom_with_anchor(cx, scroll, fs.abs);
            }
            _ => {}
        }
    }

    fn draw_walk(&mut self, cx: &mut Cx2d, _scope: &mut Scope, walk: Walk) -> DrawStep {
        let rect = cx.walk_turtle(walk);
        self.view_rect = rect;

        // Point the tile disk cache at the app's writable data dir (once).
        // Without this the relative cache path lands under the read-only "/"
        // cwd and nothing persists — every tile re-fetches from the network.
        if let Some(dir) = cx.get_data_dir() {
            set_tile_cache_base(&dir);
        }

        let nav_kind = self.nav_kind();
        if nav_kind > 0 {
            self.ensure_nav_route(cx);
            self.update_nav_camera(rect, nav_kind);
            // prefetch tiles along the route AHEAD of the car — THROTTLED to a
            // few Hz. Walking the whole loop corridor every 60fps frame starved
            // the frame budget (judder); the car moves ~0.5m per frame so a
            // ~4Hz refresh is plenty of lead time.
            if self.frame_counter % 15 == 0 {
                self.prefetch_route_ahead(cx);
            }
            // Only keep pumping frames while something is actually MOVING. A drive
            // follow-cam (2d/3d) tracks the sim vehicle every frame so it always
            // animates; a plan/preview map is STATIC unless a zoom/pan glide, a
            // live touch gesture, or a just-revealed label fade is in flight.
            // Re-arming unconditionally here pinned the GPU at ~100% on a frozen
            // map (the reported sluggishness). Async tile arrivals repaint via
            // their own UI signal (ToUISender → handle_tile_worker_messages), so
            // they don't depend on this loop.
            let moving = !self.is_plan()
                || self.nav_zoom_anim.is_some()
                || self.nav_pan_anim.is_some()
                || !self.nav_touches.is_empty();
            if moving {
                self.nav_settle_frames = NAV_SETTLE_FRAMES;
            }
            if moving || self.nav_settle_frames > 0 {
                self.nav_next_frame = cx.new_next_frame();
                if !moving {
                    self.nav_settle_frames -= 1;
                }
            }
        } else if self.draw_map.nav.mode != 0.0 {
            self.draw_map.nav = NavShaderParams::default();
            self.apply_theme_palette();
        }

        // For the flat plan/preview map (nav_kind 2), fill the whole map area with the
        // land tone before tiles, so regions beyond the fetched route corridor read as
        // empty map instead of the dark clear colour ("2nd half of the map is black").
        if nav_kind == 2 {
            let bg = self.active_style().background;
            self.draw_bg.color = bg;
        }
        self.draw_bg.draw_abs(cx, rect);
        let tile_rect = self.nav_tile_rect(rect, nav_kind);
        self.ensure_visible_tiles(cx, tile_rect);

        let view_zoom = self.view_zoom();
        let world_size = tile_world_size_zoom(view_zoom);
        let center_world = self.center_norm * world_size;
        // f64 base offset; geometry is TILE-LOCAL, so the per-tile offset
        // (base + origin*scale) stays small — no f32 catastrophic cancellation
        let off_x = rect.pos.x + rect.size.x * 0.5 - center_world.x;
        let off_y = rect.pos.y + rect.size.y * 0.5 - center_world.y;
        let tile_offset = |key: &TileKey| -> Vec2f {
            let (ox, oy) = tile_world_origin(*key);
            let scale = 2.0_f64.powf(view_zoom - key.z as f64);
            Vec2f {
                x: (off_x + ox * scale) as f32,
                y: (off_y + oy * scale) as f32,
            }
        };

        self.fill_draw_tile_keys();
        self.scratch_draw_tiles
            .sort_unstable_by_key(|key| (key.z, key.y, key.x));
        // Take draw_tiles out so we can pass &[TileKey] while mutating self for labels
        let draw_tiles = std::mem::take(&mut self.scratch_draw_tiles);

        // Only the 3D chase view (nav_kind 1) uses the pinhole tile cull below —
        // `nav_project_flat` is the 3D-pinhole projection, so it ONLY matches
        // what the shader draws in 3D. Plan preview + 2D heading-up (nav_kind 2)
        // draw with the shader's 2D projection, so culling them with 3D-pinhole
        // math rejects on-screen tiles and the map renders blank. Route those
        // through the normal (uncull) fill/stroke passes below — the shader still
        // applies their nav projection via the `nav.mode` uniform.
        if nav_kind == 1 {
            // Navigation: draw store tiles near the car, but CULL to those that
            // actually project on-screen — drawing all ~120 tiles in the radius
            // (240+ draw calls) overloaded the GPU and strobed. Loaded content
            // still can't vanish (every visible tile is in the store; proven).
            let zoom_u = self.request_zoom_level();
            let world = tile_world_size(zoom_u);
            let cw = self.center_norm * world;
            let lat = normalized_y_to_lat(self.center_norm.y);
            let mpp = meters_per_world_px(lat, zoom_u as f64);
            let radius = (self.nav_maxg.max(100.0) / mpp) * 3.0
                + rect.size.x.max(rect.size.y);
            NAV_DRAW_CENTER.with(|c| c.set((zoom_u, cw.x, cw.y)));
            let (fills, strokes) = nav_store_draw_ids(zoom_u, cw.x, cw.y, radius);
            let scale = 2.0_f64.powf(view_zoom - zoom_u as f64) as f32;
            let dscale = scale as f64;
            // Cull to VISIBLE tiles (drawing the whole radius overloads this
            // phone -> 20fps stutter, itself perceived as vanishing). Robust
            // test so nothing on-screen is ever dropped: (a) always draw the
            // ring of tiles immediately around the car; (b) otherwise draw if
            // the tile's projected corner BOUNDING BOX intersects the view rect
            // expanded by a 30% margin (bbox-intersect catches tiles straddling
            // the screen during turns; the earlier per-corner test missed them).
            // Budget PRIORITY per tile: Some(0) = strictly on-screen, Some(1) =
            // over-render margin only, None = culled. Sorting the budget purely
            // by distance-to-car let near-but-OFF-SCREEN tiles (below/behind the
            // car, inside the bottom over-render margin) consume budget slots and
            // STARVE on-screen SIDE tiles (which sit farther from the car, toward
            // the horizon) — so side features vanished before leaving the screen.
            // On-screen-first fixes it: only ~8-15 tiles are ever strictly on
            // screen at z15, well under budget, so nothing visible is cut.
            let priority = |key: &TileKey, me: &Self| -> Option<u8> {
                let (ox, oy) = tile_world_origin(*key);
                let tcx = ox + 0.5 * TILE_SIZE;
                let tcy = oy + 0.5 * TILE_SIZE;
                let near_ring = (tcx - cw.x).abs() < TILE_SIZE * 2.6
                    && (tcy - cw.y).abs() < TILE_SIZE * 2.6;
                let (mut minx, mut maxx, mut miny, mut maxy) = (1e18, -1e18, 1e18, -1e18);
                let mut any = false;
                for (cxo, cyo) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)] {
                    let wx = (ox + cxo * TILE_SIZE) * dscale;
                    let wy = (oy + cyo * TILE_SIZE) * dscale;
                    if let Some((sx, sy, _)) = me.nav_project_flat(off_x + wx, off_y + wy) {
                        minx = minx.min(sx);
                        maxx = maxx.max(sx);
                        miny = miny.min(sy);
                        maxy = maxy.max(sy);
                        any = true;
                    }
                }
                if !any {
                    // no corner projects (fully behind camera): only the near
                    // ring survives (near/under the car), everything else culled.
                    return if near_ring { Some(0) } else { None };
                }
                // strictly on-screen = projected bbox intersects the ACTUAL rect
                // (no margin). These MUST always get a budget slot.
                let on_screen = maxx >= rect.pos.x
                    && minx <= rect.pos.x + rect.size.x
                    && maxy >= rect.pos.y
                    && miny <= rect.pos.y + rect.size.y;
                if on_screen || near_ring {
                    return Some(0);
                }
                // over-render margin, ASYMMETRIC by where content enters the view:
                //  - SIDES (mx, 1.8 screens): turns rotate content in from L/R, so
                //    pre-render wide so nothing pops at the edge mid-turn.
                //  - TOP (mtop, 1.0 screen): the horizon reveals as you drive fwd.
                //  - BOTTOM (mbot, 0.25 screen): behind the car — content only
                //    LEAVES here, nothing enters, so a big bottom margin just
                //    wasted budget on never-seen tiles. Reclaimed for the sides.
                // (y grows DOWN: top edge = rect.pos.y, bottom = pos.y+size.y.)
                let mx = rect.size.x * 1.8;
                let mtop = rect.size.y * 1.0;
                let mbot = rect.size.y * 0.25;
                if maxx >= rect.pos.x - mx
                    && minx <= rect.pos.x + rect.size.x + mx
                    && maxy >= rect.pos.y - mtop
                    && miny <= rect.pos.y + rect.size.y + mbot
                {
                    Some(1)
                } else {
                    None
                }
            };
            // distance of a tile centre to the car (world px), for the budget
            let tdist = |k: &TileKey| -> u64 {
                let (ox, oy) = tile_world_origin(*k);
                let dx = ox + 0.5 * TILE_SIZE - cw.x;
                let dy = oy + 0.5 * TILE_SIZE - cw.y;
                (dx * dx + dy * dy) as u64
            };
            // Filter to visible-or-margin, sort ON-SCREEN-FIRST then nearest, then
            // HARD-CAP to a frame budget (a dense downtown must never blow the
            // budget — that stutter itself reads as vanishing). Buildings (FILL)
            // are the heavy geometry, so cap them tighter; roads (STROKE) are
            // light, so draw them far + wide (road network to the horizon).
            const NAV_FILL_BUDGET: usize = 26; // buildings/landuse — near only
            const NAV_STROKE_BUDGET: usize = 60; // roads — far + wide
            let mut fills: Vec<_> = fills
                .into_iter()
                .filter_map(|(k, id)| priority(&k, self).map(|p| (p, k, id)))
                .collect();
            let mut strokes: Vec<_> = strokes
                .into_iter()
                .filter_map(|(k, id)| priority(&k, self).map(|p| (p, k, id)))
                .collect();
            fills.sort_by_key(|(p, k, _)| (*p, tdist(k)));
            strokes.sort_by_key(|(p, k, _)| (*p, tdist(k)));
            fills.truncate(NAV_FILL_BUDGET);
            strokes.truncate(NAV_STROKE_BUDGET);
            for (_, key, id) in fills {
                let off = tile_offset(&key);
                self.draw_map
                    .draw_geometry(cx, id, Vec2f { x: scale, y: scale }, off);
            }
            for (_, key, id) in strokes {
                let off = tile_offset(&key);
                self.draw_map
                    .draw_geometry(cx, id, Vec2f { x: scale, y: scale }, off);
            }
        } else {
            // Fill pass
            for key in &draw_tiles {
                let Some(entry) = self.tiles.get(key) else {
                    continue;
                };
                if let TileLoadState::Ready { fill_geometry, .. } = &entry.state {
                    let Some(fill_geometry) = fill_geometry else {
                        continue;
                    };
                    let scale = 2.0_f64.powf(view_zoom - key.z as f64) as f32;
                    let off = tile_offset(key);
                    self.draw_map.draw_geometry(
                        cx,
                        fill_geometry.geometry_id(),
                        Vec2f { x: scale, y: scale },
                        off,
                    );
                }
            }

            // Stroke pass
            for key in &draw_tiles {
                let Some(entry) = self.tiles.get(key) else {
                    continue;
                };
                if let TileLoadState::Ready {
                    stroke_geometry, ..
                } = &entry.state
                {
                    let Some(stroke_geometry) = stroke_geometry else {
                        continue;
                    };
                    let scale = 2.0_f64.powf(view_zoom - key.z as f64) as f32;
                    let off = tile_offset(key);
                    self.draw_map.draw_geometry(
                        cx,
                        stroke_geometry.geometry_id(),
                        Vec2f { x: scale, y: scale },
                        off,
                    );
                }
            }
        }

        // Navigation route ribbon (world-space: perspective-tapers automatically).
        // Plan overview draws a constant-width SCREEN-space line instead (below),
        // since this ribbon shrinks to a hairline when zoomed out to fit the
        // whole route.
        if nav_kind > 0 && !self.is_plan() {
            if let Some(geom) = &self.nav_route_geom {
                let scale = 2.0_f64.powf(view_zoom - NAV_REF_Z as f64);
                let (rox, roy) = self.nav_route_origin;
                let off = Vec2f {
                    x: (off_x + rox * scale) as f32,
                    y: (off_y + roy * scale) as f32,
                };
                self.draw_map.draw_geometry(
                    cx,
                    geom.geometry_id(),
                    Vec2f {
                        x: scale as f32,
                        y: scale as f32,
                    },
                    off,
                );
            }
        }

        // Labels
        if view_zoom >= 13.0 && nav_kind == 0 {
            self.place_and_draw_labels(cx, &draw_tiles, view_zoom, (off_x, off_y), rect);
        } else if nav_kind > 0 {
            // nav view: UPRIGHT street/place labels, positioned by projecting
            // each label's anchor through the nav camera (the tilted 3D map has
            // no baked labels, so adjacent street names render here instead).
            // ONE draw_ui vector session for ALL markers (labels' standing pins,
            // route origin/via/dest pins, and the vehicle puck) — multiple
            // begin/end sessions per frame only flush the last, which was
            // dropping every pin except the puck.
            self.draw_ui.begin();
            if self.is_plan() {
                // Plan overview: a prominent constant-width route line + A/B pins
                // (no moving puck, no POI label clutter).
                self.draw_nav_route_line(rect, off_x, off_y, view_zoom);
                self.draw_nav_route_pins(cx, rect, off_x, off_y, view_zoom);
            } else {
                self.draw_nav_labels(cx, rect, off_x, off_y);
                self.draw_nav_route_pins(cx, rect, off_x, off_y, view_zoom);
                self.draw_nav_puck(cx, rect, off_x, off_y, view_zoom);
            }
            self.draw_ui.end(cx);
            // Plan street/place labels are TEXT (draw_text), so they go after the
            // draw_ui vector session — on top of the tiles + route line.
            if self.is_plan() {
                self.draw_nav_labels_plan(cx, rect, off_x, off_y, view_zoom);
            }
            self.label_perf = LabelPerfStats::default();
        } else {
            self.label_perf = LabelPerfStats::default();
        }

        // Put draw_tiles back into scratch buffer (preserves allocation)
        self.scratch_draw_tiles = draw_tiles;

        self.update_status_text();
        // self.draw_text.draw_abs(cx, dvec2(rect.pos.x + 10.0, rect.pos.y + 16.0), &self.status);
        DrawStep::done()
    }
}

impl WidgetMatchEvent for MapView {
    fn handle_http_response(
        &mut self,
        cx: &mut Cx,
        request_id: LiveId,
        response: &HttpResponse,
        _scope: &mut Scope,
    ) {
        // OpenFreeMap TileJSON bootstrap response? (tracked separately from the
        // tile pending map) — adopt the current dated tile-URL template.
        if request_id.0 != 0 && NAV_MVT_TILEJSON_REQ.with(|c| c.get()) == request_id.0 {
            NAV_MVT_TILEJSON_REQ.with(|c| c.set(0));
            NAV_MVT_BOOTSTRAP.with(|b| b.set(2));
            if response.status_code == 200 {
                if let Some(tj) = response.get_string_body() {
                    if nav_mvt_adopt_tilejson(&tj) {
                        log!("MapView: OpenFreeMap tile template refreshed from TileJSON");
                    }
                }
            }
            return;
        }

        let Some(pending) = nav_pending_take(&request_id) else {
            return;
        };
        let tile_key = pending.tile_key;
        let endpoint = pending.endpoint;

        if response.status_code != 200 {
            let preview = response
                .get_string_body()
                .unwrap_or_default()
                .chars()
                .take(120)
                .collect::<String>();
            self.mark_tile_failed(
                tile_key,
                &format!(
                    "endpoint {} http status {} body: {}",
                    endpoint, response.status_code, preview
                ),
            );
            self.update_status_text();
            self.redraw(cx);
            return;
        }

        // Resolve the payload to move into the worker: MVT/PBF bytes (decoded to
        // Overpass-JSON there via the shared bridge) or an Overpass-JSON string
        // already. All decode/parse/tessellate stays off the UI thread.
        let payload = if pending.is_mvt {
            match response.get_body() {
                Some(bytes) => TilePayload::Mvt(bytes.clone()),
                None => {
                    self.mark_tile_failed(
                        tile_key,
                        &format!("endpoint {} missing tile body", endpoint),
                    );
                    self.update_status_text();
                    self.redraw(cx);
                    return;
                }
            }
        } else {
            match response.get_string_body() {
                Some(b) => TilePayload::Json(b),
                None => {
                    self.mark_tile_failed(
                        tile_key,
                        &format!("endpoint {} missing utf8 response body", endpoint),
                    );
                    self.update_status_text();
                    self.redraw(cx);
                    return;
                }
            }
        };

        // Offload heavy decode + JSON parse + tessellation to the thread pool.
        self.ensure_tile_thread_pool(cx);
        let sender = nav_workers_sender();
        let style_epoch = self.style_epoch;
        let theme_style = self.active_style().clone();

        nav_workers_execute(tile_key, move |_tag| {
            // Reduce to an Overpass-JSON string (decode MVT if needed), then build.
            let json = match payload {
                TilePayload::Json(j) => Ok(j),
                TilePayload::Mvt(bytes) => mbtiles_tile_to_overpass_json(tile_key, &bytes),
            };
            // The failure path wants the body's head for diagnostics, so the body
            // must outlive the build attempt. `and_then` cannot do that — it
            // consumes the body on the way to `Err`, leaving nothing to report.
            // Carry it in the error instead, as `None` when the DECODE itself
            // failed and there is genuinely no body to show.
            let outcome = match json {
                Err(err) => Err((err, None)),
                Ok(body) => match build_tile_buffers_from_body(tile_key, &body, &theme_style) {
                    Ok(buffers) => Ok((body, buffers)),
                    Err(err) => Err((err, Some(body))),
                },
            };
            match outcome {
                Ok((body, buffers)) => {
                    store_tile_data_cache_on_disk(tile_key, &body);
                    let _ = sender.send(TileWorkerMessage::NetworkTileParsed {
                        style_epoch,
                        tile_key,
                        buffers,
                    });
                }
                Err((err, body)) => {
                    let detail = body
                        .map(|b| {
                            let head: String = b.chars().take(160).collect();
                            format!(" | body len={} head={head:?}", b.len())
                        })
                        .unwrap_or_default();
                    let _ = sender.send(TileWorkerMessage::NetworkTileParseFailed {
                        style_epoch,
                        tile_key,
                        error: format!("{err}{detail}"),
                    });
                }
            }
        });
    }

    fn handle_http_request_error(
        &mut self,
        cx: &mut Cx,
        request_id: LiveId,
        err: &HttpError,
        _scope: &mut Scope,
    ) {
        let Some(pending) = nav_pending_take(&request_id) else {
            return;
        };
        self.mark_tile_failed(
            pending.tile_key,
            &format!(
                "endpoint {} http request error: {:?}",
                pending.endpoint, err
            ),
        );
        self.update_status_text();
        self.redraw(cx);
    }
}

// --- MapView impl ---

impl MapView {
    fn rebuild_compiled_styles(&mut self) {
        self.compiled_style_light = self.style_light.compile();
        self.compiled_style_dark = self.style_dark.compile();
    }

    fn active_style(&self) -> &CompiledMapTheme {
        if self.dark_theme {
            &self.compiled_style_dark
        } else {
            &self.compiled_style_light
        }
    }

    fn normalize_source_mode(&mut self) {
        if self.use_local_mbtiles && self.use_network {
            log!("MapView: both sources enabled; selecting OFFLINE mode (mbtiles only). Set use_local_mbtiles:false for ONLINE mode.");
            self.use_network = false;
        } else if !self.use_local_mbtiles && !self.use_network {
            log!("MapView: no source enabled; selecting OFFLINE mode (mbtiles only).");
            self.use_local_mbtiles = true;
        }
    }

    fn set_dark_theme(&mut self, cx: &mut Cx, dark_theme: bool) {
        if self.dark_theme == dark_theme {
            return;
        }
        self.dark_theme = dark_theme;
        self.apply_theme_change();
        self.applied_dark_theme = Some(self.dark_theme);
        self.update_status_text();
        self.redraw(cx);
    }

    fn apply_theme_change(&mut self) {
        self.style_epoch = self.style_epoch.wrapping_add(1);
        if self.style_epoch == 0 {
            self.style_epoch = 1;
        }
        self.apply_theme_palette();
        self.tiles.clear();
        // NAV_PENDING is global — responses can still be claimed by live instances
        self.local_requested_tiles.clear();
    }

    fn apply_theme_palette(&mut self) {
        let (background, label) = {
            let style = self.active_style();
            (style.background, style.label)
        };
        self.draw_bg.color = background;
        self.draw_label.draw_super.color = label;
        self.draw_text.color = vec4(0.0, 0.0, 0.0, 1.0);
    }

    fn redraw(&mut self, cx: &mut Cx) {
        self.draw_bg.redraw(cx);
    }

    fn insert_ready_tile(&mut self, cx: &mut Cx, tile_key: TileKey, buffers: TileBuffers) {
        let fill_geometry = if !buffers.fill_indices.is_empty() && !buffers.fill_vertices.is_empty()
        {
            let geometry = Geometry::new(cx);
            geometry.update(cx, buffers.fill_indices, buffers.fill_vertices);
            Some(geometry)
        } else {
            None
        };

        let stroke_geometry =
            if !buffers.stroke_indices.is_empty() && !buffers.stroke_vertices.is_empty() {
                let geometry = Geometry::new(cx);
                geometry.update(cx, buffers.stroke_indices, buffers.stroke_vertices);
                Some(geometry)
            } else {
                None
            };

        // Owning geometry goes to the UI-thread shared store (survives the
        // 1 Hz Splash card rebuilds); this instance keeps borrowed handles.
        let entry = nav_store_insert(
            tile_key,
            fill_geometry,
            stroke_geometry,
            buffers.feature_count,
            buffers.labels,
            self.frame_counter,
        );
        self.tiles.insert(tile_key, entry);
    }

    fn handle_tile_worker_messages(&mut self, cx: &mut Cx) {
        let mut redraw = false;
        while let Some(msg) = nav_workers_try_recv() {
            match msg {
                TileWorkerMessage::LocalBatchLoaded {
                    style_epoch,
                    requested,
                    loaded,
                } => {
                    if style_epoch != self.style_epoch {
                        for key in &requested {
                            self.local_requested_tiles.remove(key);
                        }
                        continue;
                    }
                    for key in &requested {
                        self.local_requested_tiles.remove(key);
                    }

                    let mut loaded_keys = HashSet::with_capacity(loaded.len());
                    let mut empty_feature_tiles = Vec::<TileKey>::new();
                    for tile in loaded {
                        loaded_keys.insert(tile.tile_key);
                        self.local_missing_tiles.remove(&tile.tile_key);
                        if tile.buffers.feature_count == 0 {
                            empty_feature_tiles.push(tile.tile_key);
                        }
                        self.insert_ready_tile(cx, tile.tile_key, tile.buffers);
                    }
                    if !empty_feature_tiles.is_empty() {
                        empty_feature_tiles.sort_unstable();
                        log!("MapView: local mbtiles loaded {} tile(s) with 0 rendered features sample:{}", empty_feature_tiles.len(), format_tile_key_sample(&empty_feature_tiles, 8));
                    }
                    for key in requested {
                        if loaded_keys.contains(&key) {
                            continue;
                        }
                        self.local_missing_tiles.insert(key);
                        self.tiles.remove(&key);
                    }
                    redraw = true;
                }
                TileWorkerMessage::LocalBatchFailed {
                    style_epoch,
                    requested,
                    error,
                } => {
                    if style_epoch != self.style_epoch {
                        for key in &requested {
                            self.local_requested_tiles.remove(key);
                        }
                        continue;
                    }
                    log!("MapView: local mbtiles load failed: {}", error);
                    for key in requested {
                        self.local_requested_tiles.remove(&key);
                        self.tiles.remove(&key);
                    }
                    redraw = true;
                }
                TileWorkerMessage::NetworkTileParsed {
                    style_epoch,
                    tile_key,
                    buffers,
                } => {
                    if style_epoch != self.style_epoch {
                        continue;
                    }
                    self.insert_ready_tile(cx, tile_key, buffers);
                    redraw = true;
                }
                TileWorkerMessage::NetworkTileParseFailed {
                    style_epoch,
                    tile_key,
                    error,
                } => {
                    if style_epoch != self.style_epoch {
                        continue;
                    }
                    self.mark_tile_failed(tile_key, &format!("parse: {}", error));
                    redraw = true;
                }
            }
        }
        if redraw {
            self.update_status_text();
            // A tile just landed on a possibly-static map: give the map a short
            // render window so the newly-drawn tile's labels can fade in (the
            // `moving` guard in draw_walk would otherwise idle after one frame).
            self.nav_settle_frames = self.nav_settle_frames.max(NAV_SETTLE_FRAMES);
            self.redraw(cx);
        }
    }

    fn request_visible_tiles_from_local_source(&mut self, _cx: &mut Cx) {
        if !self.use_local_mbtiles {
            return;
        }

        let mbtiles_path = Path::new(LOCAL_MBTILES_PATH);
        if !mbtiles_path.is_file() {
            if !self.local_source_missing_logged {
                log!("MapView: local mbtiles source missing at {} (set use_local_mbtiles: false to disable)", LOCAL_MBTILES_PATH);
                self.local_source_missing_logged = true;
            }
            return;
        }

        let mut missing = Vec::<TileKey>::new();
        for key in &self.visible_tiles {
            if self.tiles.contains_key(key)
                || self.local_requested_tiles.contains(key)
                || self.local_missing_tiles.contains(key)
            {
                continue;
            }
            missing.push(*key);
        }
        if missing.is_empty() {
            return;
        }
        if missing.len() > MAX_LOCAL_TILE_BATCH {
            missing.truncate(MAX_LOCAL_TILE_BATCH);
        }

        for key in &missing {
            self.local_requested_tiles.insert(*key);
            self.tiles.insert(
                *key,
                TileEntry {
                    state: TileLoadState::LoadingLocal,
                    last_used: self.frame_counter,
                    attempts: 0,
                },
            );
        }

        let sender = nav_workers_sender();
        let requested = missing.clone();
        let mbtiles_path = LOCAL_MBTILES_PATH.to_string();
        let cache_dir = TILE_CACHE_DIR.to_string();
        let style_epoch = self.style_epoch;
        let theme_style = self.active_style().clone();
        let batch_tag = missing[0];

        nav_workers_execute(batch_tag, move |_tag| {
            let result = load_local_tile_batch(
                Path::new(&mbtiles_path),
                Path::new(&cache_dir),
                &requested,
                &theme_style,
            );
            match result {
                Ok(loaded) => {
                    let _ = sender.send(TileWorkerMessage::LocalBatchLoaded {
                        style_epoch,
                        requested,
                        loaded,
                    });
                }
                Err(error) => {
                    let _ = sender.send(TileWorkerMessage::LocalBatchFailed {
                        style_epoch,
                        requested,
                        error,
                    });
                }
            }
        });
    }

    fn mark_tile_failed(&mut self, tile_key: TileKey, reason: &str) {
        let attempts = self
            .tiles
            .get(&tile_key)
            .map_or(1, |entry| entry.attempts.saturating_add(1));
        let retry_delay = retry_delay_frames(attempts);
        let retry_after = self.frame_counter.saturating_add(retry_delay);
        self.tiles.insert(
            tile_key,
            TileEntry {
                state: TileLoadState::Failed { retry_after },
                last_used: self.frame_counter,
                attempts,
            },
        );
        log!(
            "MapView: tile z{} x{} y{} failed (attempt {}): {}",
            tile_key.z,
            tile_key.x,
            tile_key.y,
            attempts,
            reason
        );
    }

    fn wrap_and_clamp_center(&mut self) {
        self.center_norm.x = self.center_norm.x.rem_euclid(1.0);
        self.center_norm.y = self.center_norm.y.clamp(0.0, 1.0);
    }

    fn zoom_with_anchor(&mut self, cx: &mut Cx, scroll: f64, anchor_abs: Vec2d) {
        if scroll.abs() <= f64::EPSILON {
            return;
        }
        let current_zoom = self.view_zoom();
        let zoom_delta = (-scroll / 240.0).clamp(-1.0, 1.0);
        let min_zoom = self.min_zoom.max(0.0);
        let max_zoom = self.max_zoom.max(min_zoom);
        let new_zoom = (current_zoom + zoom_delta).clamp(min_zoom, max_zoom);
        if (new_zoom - current_zoom).abs() < 1e-4 {
            return;
        }

        if self.view_rect.size.x <= 0.0 || self.view_rect.size.y <= 0.0 {
            self.zoom = new_zoom;
            self.redraw(cx);
            return;
        }

        let old_world_size = tile_world_size_zoom(current_zoom);
        let new_world_size = tile_world_size_zoom(new_zoom);
        let rect_center = self.view_rect.pos + self.view_rect.size * 0.5;
        let old_center_world = self.center_norm * old_world_size;
        let anchor_world = old_center_world + (anchor_abs - rect_center);
        let anchor_norm = anchor_world / old_world_size;
        let new_center_world = anchor_norm * new_world_size - (anchor_abs - rect_center);

        self.zoom = new_zoom;
        self.center_norm = new_center_world / new_world_size;
        self.wrap_and_clamp_center();
        self.redraw(cx);
    }

    fn ensure_tile_thread_pool(&mut self, cx: &mut Cx) {
        nav_workers_ensure(cx);
    }

    /// 0 = normal map, 1 = 3D chase FPV, 2 = 2D heading-up.
    fn nav_kind(&self) -> u8 {
        match self.nav_mode.as_ref().trim() {
            "3d" | "3D" => 1,
            // "plan" is a 2D variant: same projection (renders the ribbon), but
            // a STATIC north-up camera fit to the whole route (route preview).
            "2d" | "2D" | "plan" => 2,
            _ => 0,
        }
    }

    fn is_plan(&self) -> bool {
        self.nav_mode.as_ref().trim() == "plan"
    }

    /// PLAN route-preview camera: STATIC, north-up, fit to the WHOLE route,
    /// framed into the top band above the card's summary sheet. Renders through
    /// the 2D nav projection (mode 2) so the route ribbon shows; unlike the live
    /// 2D follow-cam it does not track the sim vehicle.
    fn update_plan_preview_camera(&mut self, rect: Rect) {
        let (mut minx, mut maxx, mut miny, mut maxy) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for p in self.nav_pts.iter() {
            minx = minx.min(p.x);
            maxx = maxx.max(p.x);
            miny = miny.min(p.y);
            maxy = maxy.max(p.y);
        }
        // Route polyline not decoded yet (or absent) — frame the annotation
        // markers (origin + destination) so the camera still centers on the trip
        // and tiles load, instead of an f64::MAX bbox → NaN center → blank map.
        if self.nav_pts.is_empty() {
            for (lat, lon, _) in self.nav_markers.iter() {
                let p = lon_lat_to_normalized(*lon, *lat);
                minx = minx.min(p.x);
                maxx = maxx.max(p.x);
                miny = miny.min(p.y);
                maxy = maxy.max(p.y);
            }
        }
        if minx > maxx {
            return; // nothing to frame — keep the current center (no NaN)
        }
        // GLIDE toward any button-set target (+/- zoom, my-location). The plan map
        // redraws every frame (nav_next_frame), so easing self.zoom / self.nav_pan
        // toward the target here animates smoothly; a gesture cleared the target
        // and set the value directly, so this is a no-op during direct manipulation.
        if let Some(tz) = self.nav_zoom_anim {
            let d = tz - self.zoom;
            if d.abs() > 0.004 {
                self.zoom += d * 0.22;
            } else {
                self.zoom = tz;
                self.nav_zoom_anim = None;
            }
        }
        if let Some(tp) = self.nav_pan_anim {
            let dx = tp.x - self.nav_pan.x;
            let dy = tp.y - self.nav_pan.y;
            if dx.abs() > 1e-7 || dy.abs() > 1e-7 {
                self.nav_pan.x += dx * 0.22;
                self.nav_pan.y += dy * 0.22;
            } else {
                self.nav_pan = tp;
                self.nav_pan_anim = None;
            }
        }
        let cx = (minx + maxx) * 0.5;
        let cy = (miny + maxy) * 0.5;
        // Fit the WHOLE route on first show; once the user pinches/pans
        // (nav_user_adjusted) keep THEIR view and only re-anchor the pan to the
        // route centre. A recenter (set_nav_recenter) clears the flag to re-fit.
        if !self.nav_user_adjusted {
            let dx = (maxx - minx).max(1e-9);
            let dy = (maxy - miny).max(1e-9);
            // fit the route bbox into ~85% width and the ~34% tall band above the
            // summary sheet. NO z14 floor now — sub-z14 tiles load a coarse
            // major-roads layer (see overpass_query), so the ENTIRE route shows.
            // The summary sheet overlays the bottom ~40%, so fit the route into
            // the top ~55% (with side/vertical margin) so BOTH endpoints show.
            let fitw = rect.size.x * 0.80;
            let fith = rect.size.y * 0.50;
            let zx = (fitw / (dx * TILE_SIZE)).log2();
            let zy = (fith / (dy * TILE_SIZE)).log2();
            let zmin = self.min_zoom.max(3.0);
            let zmax = self.max_zoom.max(zmin);
            // Floor at z10 (a z10 tile ~40 km keeps a very long route's Overpass
            // query bounded); cap so a tiny route doesn't over-zoom.
            self.zoom = zx.min(zy).clamp(zmin, zmax).clamp(10.0, 15.5);
            self.nav_home_zoom = self.zoom;
        }
        // Route centre is the pan anchor; the user's pan offsets from it.
        self.nav_car_norm = dvec2(cx, cy);
        self.center_norm = dvec2(cx + self.nav_pan.x, cy + self.nav_pan.y);
        self.wrap_and_clamp_center();
        self.draw_map.nav = NavShaderParams {
            mode: 2.0,
            anchor: vec2(
                (rect.pos.x + rect.size.x * 0.5) as f32,
                (rect.pos.y + rect.size.y * 0.5) as f32,
            ),
            rot: vec2(0.0, 1.0), // north-up (static)
            cam: [40.0, 0.315, 0.49, 0.76],
            screen: [
                rect.pos.x as f32,
                rect.pos.y as f32,
                rect.size.x as f32,
                rect.size.y as f32,
            ],
            // misc.z = the route-bbox centre's screen row (30% down — centres the
            // route in the visible map area above the summary sheet)
            misc: [90.0, 500.0, (rect.size.y * 0.30) as f32, 0.533],
            haze: [0.847, 0.890, 0.929, 0.0],
        };
    }

    /// Decode `nav_polyline` and tessellate the route ribbon geometry
    /// (casing + semi-transparent core) at NAV_REF_Z. Cached by content hash —
    /// cheap on the 1 Hz card rebuilds.
    fn ensure_nav_route(&mut self, cx: &mut Cx) {
        let poly = self.nav_polyline.as_ref().trim().to_string();
        if poly.is_empty() {
            return;
        }
        let hash = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            poly.hash(&mut h);
            // Plan mode bakes the flat teardrop pins into the geometry; nav mode
            // must NOT (it draws upright billboard pins instead). Discriminate
            // the two so they get SEPARATE cached geometries — otherwise nav
            // adopts the plan geometry and the flat teardrops leak into the 3D
            // view.
            self.is_plan().hash(&mut h);
            // Fold the annotation pins in so a marker change re-tessellates (and
            // gets its own NAV_ROUTE_STORE entry) instead of adopting the old
            // geometry. Bit-cast the f64s — lat/lon are exact enough as bits.
            for (lat, lon, kind) in &self.nav_markers {
                lat.to_bits().hash(&mut h);
                lon.to_bits().hash(&mut h);
                kind.hash(&mut h);
            }
            h.finish()
        };
        if hash == self.nav_poly_hash && self.nav_route_geom.is_some() {
            return;
        }
        // Shared across the 1 Hz card rebuilds: decode + tessellate ONCE.
        let adopted = NAV_ROUTE_STORE.with(|s| {
            let mut s = s.borrow_mut();
            // bump the LRU clock on adopt so an actively-reused route can never
            // be the eviction victim while an instance still borrows its handle
            if let Some(r) = s.get_mut(&hash) {
                r.seq = next_nav_route_seq();
                Some((
                    r.geom
                        .as_ref()
                        .map(|g| Geometry::new_borrowed(g.geometry_id())),
                    r.origin,
                    r.pts.clone(),
                    r.cum.clone(),
                ))
            } else {
                None
            }
        });
        if let Some((geom, origin, pts, cum)) = adopted {
            self.nav_route_geom = geom;
            self.nav_route_origin = origin;
            self.nav_pts = pts;
            self.nav_cum = cum;
            self.nav_poly_hash = hash;
            return;
        }
        let coords0 = decode_polyline5(&poly);
        if coords0.len() < 2 {
            return;
        }
        // DENSIFY: OSRM points are ~10-50 m apart. Near the camera a long
        // ribbon quad projects badly — the GPU interpolates the NON-LINEAR
        // pinhole across the quad linearly, so the near ribbon kinks and can
        // degenerate/vanish. Subdivide every segment to <= ~6 m so each quad
        // is short and the perspective error is negligible.
        let mut coords: Vec<(f64, f64)> = Vec::with_capacity(coords0.len() * 4);
        for i in 0..coords0.len() {
            let (lat, lon) = coords0[i];
            coords.push((lat, lon));
            if i + 1 < coords0.len() {
                let (lat2, lon2) = coords0[i + 1];
                let seg = haversine_m(lat, lon, lat2, lon2);
                // cap per-segment subdivision: a pathological (gappy/malformed)
                // polyline segment could otherwise explode into millions of
                // points and OOM-abort the process (profile.small: panic='abort')
                let steps = ((seg / 6.0).floor() as usize).min(2048);
                for s in 1..steps {
                    let t = s as f64 / steps as f64;
                    coords.push((lat + (lat2 - lat) * t, lon + (lon2 - lon) * t));
                }
            }
        }
        let mut pts = Vec::with_capacity(coords.len());
        let mut cums = Vec::with_capacity(coords.len());
        let mut cum = 0.0_f64;
        let mut prev: Option<(f64, f64)> = None;
        for &(lat, lon) in &coords {
            if let Some((plat, plon)) = prev {
                cum += haversine_m(plat, plon, lat, lon);
            }
            pts.push(lon_lat_to_normalized(lon, lat));
            cums.push(cum);
            prev = Some((lat, lon));
        }
        self.nav_pts = Rc::new(pts);
        self.nav_cum = Rc::new(cums);
        self.nav_poly_hash = hash;

        let world = tile_world_size(NAV_REF_Z);
        let mid_lat = coords[coords.len() / 2].0;
        let mpp = meters_per_world_px(mid_lat, NAV_REF_Z as f64);
        // rebase to the first point (f64) so ribbon vertices stay small-f32
        let origin = (
            self.nav_pts[0].x * world,
            self.nav_pts[0].y * world,
        );
        self.nav_route_origin = origin;
        let pts_px: Vec<(f32, f32)> = self
            .nav_pts
            .iter()
            .map(|p| {
                (
                    (p.x * world - origin.0) as f32,
                    (p.y * world - origin.1) as f32,
                )
            })
            .collect();
        let core_w = (self.nav_route_width.max(4.0) / mpp) as f32;
        let casing_w = core_w * 1.55;

        let mut path = VectorPath::new();
        let mut tess = Tessellator::default();
        let mut tess_verts = Vec::<VVertex>::new();
        let mut tess_indices = Vec::<u32>::new();
        let mut vertices = Vec::<f32>::new();
        let mut indices = Vec::<u32>::new();
        let mut zbias = 0.0_f32;
        // dark casing then bright semi-transparent core (Google-style)
        append_route_ribbon_pass(
            &mut path,
            &pts_px,
            &mut tess,
            &mut tess_verts,
            &mut tess_indices,
            &mut vertices,
            &mut indices,
            casing_w,
            0x24528F,
            1.0,
            &mut zbias,
        );
        append_route_ribbon_pass(
            &mut path,
            &pts_px,
            &mut tess,
            &mut tess_verts,
            &mut tess_indices,
            &mut vertices,
            &mut indices,
            core_w,
            0x5B9BF8,
            1.0,
            &mut zbias,
        );
        // Annotation pins (origin / 途经点 / destination): flat world-space
        // teardrops read correctly only from the top-down PLAN camera — in the
        // tilted nav view they'd lie flat (wrong), so nav mode draws upright
        // billboard pins in `draw_nav_route_pins` instead. Plan mode only here.
        let pin_r = (core_w * 2.6).max(7.0);
        for &(mlat, mlon, kind) in &self.nav_markers {
            if !self.is_plan() {
                break;
            }
            let wp = lon_lat_to_normalized(mlon, mlat) * world;
            let mx = (wp.x - origin.0) as f32;
            let my = (wp.y - origin.1) as f32;
            let color = match kind {
                0 => 0x1DB954, // origin — green
                2 => 0xEA4335, // destination — red
                _ => 0x1A73E8, // 途经点 / intermediate — blue
            };
            append_marker_pin(
                &mut path, &mut tess, &mut tess_verts, &mut tess_indices,
                &mut vertices, &mut indices, mx, my, pin_r, color, &mut zbias,
            );
        }
        if !indices.is_empty() {
            let geometry = Geometry::new(cx);
            geometry.update(cx, indices, vertices);
            self.nav_route_geom = Some(Geometry::new_borrowed(geometry.geometry_id()));
            NAV_ROUTE_STORE.with(|s| {
                let mut s = s.borrow_mut();
                // Evict the LEAST-recently-used entries down to a cap. NEVER
                // bulk-clear(): instances hold BORROWED handles into this store,
                // including a just-dropped plan MapView whose final draw is still
                // queued during a plan->drive transition. Freeing a live entry
                // frees its GPU geometry slot; the pool reuses that slot for the
                // next route/tile and the queued draw then reads a wrong-sized
                // buffer -> native SIGSEGV (the reported "3D nav crash"). LRU keeps
                // the routes currently on screen (the one just drawn + the incoming
                // drive route) alive; only long-stale entries are dropped.
                const NAV_ROUTE_CAP: usize = 12;
                while s.len() >= NAV_ROUTE_CAP {
                    match s.iter().min_by_key(|(_, r)| r.seq).map(|(&k, _)| k) {
                        Some(k) => {
                            s.remove(&k);
                        }
                        None => break,
                    }
                }
                s.insert(
                    hash,
                    SharedNavRoute {
                        geom: Some(geometry),
                        origin,
                        pts: self.nav_pts.clone(),
                        cum: self.nav_cum.clone(),
                        seq: next_nav_route_seq(),
                    },
                );
            });
        }
    }

    /// Follow the route on the sim clock (same epoch as `sys.simsecs`, so DSL
    /// banner windows stay in lockstep), center the map on the vehicle and
    /// push the projection uniforms.
    fn update_nav_camera(&mut self, rect: Rect, nav_kind: u8) {
        if self.nav_pts.len() < 2 {
            return;
        }
        // PLAN preview: static fit-to-route camera (framed above the sheet),
        // not the sim follow-cam.
        if self.is_plan() {
            self.update_plan_preview_camera(rect);
            return;
        }
        let total = self.nav_cum.last().copied().unwrap_or(0.0);
        // CONSTANT-SPEED sim: drive at `nav_speed_mph` and LOOP at the route end
        // (period-normalized sweeping made long routes absurdly fast and short
        // ones crawl). Looping avoids the old "parked at destination" problem.
        // The card's banner clock uses the same `(clock*mps) % total`.
        let mps = (self.nav_speed_mph.max(1.0)) * 0.44704;
        let d = (crate::splash::sim_clock_secs() * mps) % total.max(1.0);

        let Some(car) = sample_polyline_point_at_distance(&self.nav_pts, &self.nav_cum, d) else {
            return;
        };
        // bearing from a short look-ahead (normalized coords; north = -y)
        let look = sample_polyline_point_at_distance(
            &self.nav_pts,
            &self.nav_cum,
            (d + 45.0).min(total),
        )
        .unwrap_or(car);
        let fwd = dvec2(look.x - car.x, look.y - car.y);
        let target_bearing = if fwd.x.abs() < 1e-12 && fwd.y.abs() < 1e-12 {
            self.nav_bearing
        } else {
            fwd.x.atan2(-fwd.y)
        };
        // low-pass ease toward the target heading (shortest angular way) so a
        // turn rotates the camera gently instead of snapping — the abrupt
        // rotation whipped the ribbon/road around and read as vanishing.
        if !self.nav_bearing_init {
            self.nav_bearing = target_bearing;
            self.nav_bearing_init = true;
        } else {
            let pi = std::f64::consts::PI;
            let mut diff = target_bearing - self.nav_bearing;
            while diff > pi {
                diff -= 2.0 * pi;
            }
            while diff < -pi {
                diff += 2.0 * pi;
            }
            self.nav_bearing += diff * 0.10;
        }
        let bearing = self.nav_bearing;
        self.nav_car_norm = car;

        // Auto-restore: after IDLE seconds without a touch, ease the manual
        // pan/zoom back to the follow-cam (like typical map apps' recenter).
        // Runs every frame while the follow loop redraws, so the ease animates.
        if self.nav_user_adjusted {
            let idle = crate::splash::sim_clock_secs() - self.nav_last_touch;
            if idle > NAV_RECENTER_IDLE_SECS {
                self.nav_pan.x *= 0.84;
                self.nav_pan.y *= 0.84;
                if self.nav_home_zoom > 0.0 {
                    self.zoom += (self.nav_home_zoom - self.zoom) * 0.16;
                }
                let z_done = self.nav_home_zoom <= 0.0
                    || (self.zoom - self.nav_home_zoom).abs() < 0.01;
                if self.nav_pan.x.abs() < 1e-6 && self.nav_pan.y.abs() < 1e-6 && z_done {
                    self.nav_pan = dvec2(0.0, 0.0);
                    if self.nav_home_zoom > 0.0 {
                        self.zoom = self.nav_home_zoom;
                    }
                    self.nav_user_adjusted = false;
                }
            }
        }

        // follow the car, plus any user pan offset (cleared by recenter)
        self.center_norm = dvec2(car.x + self.nav_pan.x, car.y + self.nav_pan.y);
        self.wrap_and_clamp_center();

        let view_zoom = self.view_zoom();
        let lat = normalized_y_to_lat(car.y);
        let mpp = meters_per_world_px(lat, view_zoom);
        let cam_h_px = (self.nav_cam_h.max(2.0) / mpp) as f32;
        let pitch = self.nav_pitch as f32;
        let (sph, cph) = (pitch.sin(), pitch.cos());
        let tan_v = ((self.nav_vfov * 0.5) as f32).tan();
        let tanh2 = ((self.nav_hfov * 0.5).tan()) as f32;
        // Place the car (ahead = 0, forward dist = chase) at screen row nav_carv
        // under the SAME pinhole the shader uses:
        //   ndc_y = (a·sinP - h·cosP) / ((a·cosP + h·sinP)·tan_v) = 1 - 2·carv
        // solved for a = chase. (The old atan mapping used a different form.)
        let ndc_car = 1.0 - 2.0 * self.nav_carv as f32;
        let denom = sph - ndc_car * tan_v * cph;
        let chase = if denom > 1e-3 {
            (cam_h_px * (cph + ndc_car * tan_v * sph) / denom).max(cam_h_px * 0.5)
        } else {
            cam_h_px * 2.0
        };
        let maxg_px = (self.nav_maxg.max(100.0) / mpp) as f32;

        self.draw_map.nav = NavShaderParams {
            mode: nav_kind as f32,
            anchor: vec2(
                (rect.pos.x + rect.size.x * 0.5) as f32,
                (rect.pos.y + rect.size.y * 0.5) as f32,
            ),
            rot: vec2(bearing.sin() as f32, bearing.cos() as f32),
            cam: [cam_h_px, sph, cph, tanh2],
            screen: [
                rect.pos.x as f32,
                rect.pos.y as f32,
                rect.size.x as f32,
                rect.size.y as f32,
            ],
            misc: [chase, maxg_px, (rect.size.y * self.nav_carv2d) as f32, tan_v],
            haze: [0.847, 0.890, 0.929, 0.0],
        };
        if nav_kind == 1 {
            // far clip + above-horizon show the background: make it the haze
            self.draw_bg.color = vec4(0.847, 0.890, 0.929, 1.0);
        }
    }

    /// Tile-coverage rect for the current mode: the 3D frustum sees far ahead
    /// (and the map rotates), so request a square neighbourhood around the car.
    fn nav_tile_rect(&self, rect: Rect, nav_kind: u8) -> Rect {
        if nav_kind == 0 {
            return rect;
        }
        let half = if nav_kind == 1 {
            let lat = normalized_y_to_lat(self.center_norm.y);
            let mpp = meters_per_world_px(lat, self.view_zoom());
            // fetch out to ~2x maxg so distant tiles are loaded before the
            // (now horizon-reaching) shader wants to draw them
            (self.nav_maxg.max(100.0) / mpp) * 2.0 + rect.size.x.max(rect.size.y) * 0.3
        } else {
            rect.size.x.max(rect.size.y) * 0.75
        };
        Rect {
            pos: dvec2(
                rect.pos.x + rect.size.x * 0.5 - half,
                rect.pos.y + rect.size.y * 0.5 - half,
            ),
            size: dvec2(half * 2.0, half * 2.0),
        }
    }

    /// Project a flat-map screen point through the nav camera (CPU mirror of
    /// the DrawMapVector 3D vertex path). Returns the on-screen position, or
    /// None if the point is behind the camera or past the haze horizon.
    fn nav_project_flat(&self, fx: f64, fy: f64) -> Option<(f64, f64, f64)> {
        // Same pinhole math as the DrawMapVector 3D shader path so the tile
        // cull matches exactly what is drawn. cam = (h, sinP, cosP, tan(hfov/2));
        // misc[3] = tan(vfov/2).
        let nav = &self.draw_map.nav;
        let rel_x = fx - nav.anchor.x as f64;
        let rel_y = fy - nav.anchor.y as f64;
        let ahead = rel_x * nav.rot.x as f64 - rel_y * nav.rot.y as f64;
        let cross = rel_x * nav.rot.y as f64 + rel_y * nav.rot.x as f64;
        let cam_h = nav.cam[0] as f64;
        let sph = nav.cam[1] as f64;
        let cph = nav.cam[2] as f64;
        let tan_h = nav.cam[3] as f64;
        let tan_v = nav.misc[3] as f64;
        let a = ahead + nav.misc[0] as f64; // + chase
        let z_cam = a * cph + cam_h * sph;
        if z_cam < cam_h * 0.6 {
            return None; // behind / too close to the camera (near-plane floor)
        }
        let y_cam = a * sph - cam_h * cph;
        let ndc_x = cross / (z_cam * tan_h);
        let ndc_y = y_cam / (z_cam * tan_v);
        let sx = nav.screen[0] as f64 + nav.screen[2] as f64 * 0.5 * (1.0 + ndc_x);
        let sy = nav.screen[1] as f64 + nav.screen[3] as f64 * 0.5 * (1.0 - ndc_y);
        Some((sx, sy, a))
    }

    /// Project a world point through the flat 2D nav camera (plan / 2D
    /// heading-up) — the CPU mirror of the DrawMapVector shader's `p2d` path, so
    /// route line + pins line up EXACTLY with the tiles in plan mode (unlike
    /// nav_project_flat, which is the 3D pinhole and only matches the 3D view).
    fn nav_project_plan(&self, fx: f64, fy: f64) -> (f64, f64) {
        let nav = &self.draw_map.nav;
        let rel_x = fx - nav.anchor.x as f64;
        let rel_y = fy - nav.anchor.y as f64;
        let ahead = rel_x * nav.rot.x as f64 - rel_y * nav.rot.y as f64;
        let cross = rel_x * nav.rot.y as f64 + rel_y * nav.rot.x as f64;
        let sx = nav.screen[0] as f64 + nav.screen[2] as f64 * 0.5 + cross;
        let sy = nav.screen[1] as f64 + nav.misc[2] as f64 - ahead;
        (sx, sy)
    }

    /// Draw upright street/place labels in the nav view by projecting each
    /// label's map anchor through the nav camera. Google-style: text stays
    /// horizontal (never tilted), shrinks with distance, majors win the cap.
    fn draw_nav_labels(&mut self, cx: &mut Cx2d, rect: Rect, off_x: f64, off_y: f64) {
        let zoom = self.request_zoom_level();
        let world = tile_world_size(zoom);
        let lat = normalized_y_to_lat(self.center_norm.y);
        let mpp = meters_per_world_px(lat, zoom as f64);
        // ADJACENT only: labels within ~260 m of the car (not the whole 800 m
        // frustum — the far-horizon roads were a cramped, flat-looking cluster)
        let near_m = 260.0;
        let radius = near_m / mpp;
        let cw = self.center_norm * world;
        // Refresh the candidate set only a few Hz (store scan + clones are
        // costly); project + draw the cached candidates every frame.
        if self.frame_counter % 10 == 0 || self.nav_labels_cache.is_empty() {
            self.nav_labels_cache = nav_store_labels(zoom, cw.x, cw.y, radius);
        }
        let mut labels = self.nav_labels_cache.clone();
        if labels.is_empty() {
            self.nav_labels_shown.clear();
            return;
        }
        // STICKY ordering: labels shown last frame come first (rank 0), so the
        // visible set stays stable as the car moves instead of flickering.
        {
            let shown = &self.nav_labels_shown;
            labels.sort_by_key(|(_, _, text, pri)| (!shown.contains(text) as u8, *pri));
        }
        let scale = 2.0_f64.powf(self.view_zoom() - zoom as f64);
        let cam_h = self.draw_map.nav.cam[0] as f64;
        // only label the near half of the frustum (ground-distance cutoff)
        let far_cut = (near_m / mpp) + cam_h;
        self.draw_text.color = vec4(0.16, 0.22, 0.30, 1.0);
        let mut drawn = 0;
        let mut placed: Vec<Rect> = Vec::new();
        let mut now_shown: HashSet<String> = HashSet::new();
        for (wx, wy, text, _pri) in labels {
            if drawn >= 12 {
                break;
            }
            let fx = off_x + wx * scale;
            let fy = off_y + wy * scale;
            let Some((sx, sy, a)) = self.nav_project_flat(fx, fy) else {
                continue;
            };
            if a > far_cut {
                continue; // too far — keep it to adjacent roads
            }
            // OVER-RENDER labels a half-screen beyond every edge so a name
            // glides off the side/bottom smoothly instead of popping out at
            // the boundary (same principle as the tiles).
            let mgx = rect.size.x * 0.5;
            if sx < rect.pos.x - mgx
                || sx > rect.pos.x + rect.size.x + mgx
                || sy < rect.pos.y - rect.size.y * 0.05
                || sy > rect.pos.y + rect.size.y * 1.05
            {
                continue;
            }
            // strong perspective depth cue: near ~19 px, far ~8 px
            let fs = (19.0 * (cam_h * 1.6 / a)).clamp(8.0, 19.0) as f32;
            let w = text.chars().count() as f64 * fs as f64 * 0.5;
            // 2.5D standing pin: an upright pin STANDS at the exact ground point
            // (tip down, head up — consistent with the perspective) and the
            // readable label rides above the head. Pin height + head scale with
            // distance, so near sites stand tall and far ones shrink.
            let stem_h = (fs as f64 * 2.0).max(14.0);
            let head_r = (fs as f64 * 0.42).max(2.6);
            let label_y = sy - stem_h - head_r - fs as f64 * 1.1;
            let lr = Rect {
                pos: dvec2(sx - w * 0.5, label_y),
                size: dvec2(w, fs as f64 * 1.3),
            };
            if placed
                .iter()
                .any(|p| rects_overlap_with_padding(*p, lr, 4.0))
            {
                continue;
            }
            placed.push(lr);
            // pin color: POIs (priority 3) blue, streets a muted slate
            let color = if _pri >= 3 {
                vec4(0.10, 0.45, 0.92, 1.0)
            } else {
                vec4(0.36, 0.45, 0.58, 1.0)
            };
            self.draw_upright_pin(cx, sx, sy, stem_h, head_r, color);
            // the readable label (upright), above the pin head
            self.draw_text.color = vec4(0.16, 0.22, 0.30, 1.0);
            self.draw_text.text_style.font_size = fs;
            self.draw_text
                .draw_abs(cx, dvec2(lr.pos.x, lr.pos.y), &text);
            now_shown.insert(text);
            drawn += 1;
        }
        self.nav_labels_shown = now_shown;
    }

    /// Plan-overview labels: flat street/place names across the WHOLE visible
    /// route (not the 260 m radius of the 3D chase labels), projected via
    /// nav_project_plan (so they sit on the tiles), dark text with a white halo
    /// so they read over roads/water. Called OUTSIDE the draw_ui session so the
    /// text batches on top.
    fn draw_nav_labels_plan(&mut self, cx: &mut Cx2d, rect: Rect, off_x: f64, off_y: f64, view_zoom: f64) {
        let zoom = self.request_zoom_level();
        let world = tile_world_size(zoom);
        let scale = 2.0_f64.powf(view_zoom - zoom as f64);
        let cw = self.center_norm * world;
        // radius (world-px) covering the visible map, so labels span the route
        let radius = (rect.size.x.max(rect.size.y)) * 0.6 / scale.max(1e-6);
        if self.frame_counter % 15 == 0 || self.nav_labels_cache.is_empty() {
            self.nav_labels_cache = nav_store_labels(zoom, cw.x, cw.y, radius);
        }
        let mut labels = self.nav_labels_cache.clone();
        if labels.is_empty() {
            return;
        }
        // STICKY + DETERMINISTIC order so the picked set doesn't FLICKER: labels
        // shown last frame win the collision test (rank 0), then higher priority,
        // then by name — a STABLE tie-break. Without it, `nav_store_labels`
        // returns candidates in non-deterministic (HashMap) order, so each cache
        // refresh re-picked a different subset and a static overview flashed.
        {
            let shown = &self.nav_labels_shown;
            labels.sort_by(|a, b| {
                (!shown.contains(&a.2) as u8, a.3)
                    .cmp(&(!shown.contains(&b.2) as u8, b.3))
                    .then_with(|| a.2.cmp(&b.2))
            });
        }
        let fs = 13.0f32;
        let now = crate::splash::sim_clock_secs();
        let mut drawn = 0;
        let mut placed: Vec<Rect> = Vec::new();
        let mut now_shown: HashSet<String> = HashSet::new();
        for (wx, wy, text, _pri) in labels {
            if drawn >= 14 {
                break;
            }
            let (sx, sy) = self.nav_project_plan(off_x + wx * scale, off_y + wy * scale);
            // on the map + above the summary sheet (~60% down)
            if sx < rect.pos.x + 8.0
                || sx > rect.pos.x + rect.size.x - 8.0
                || sy < rect.pos.y + 12.0
                || sy > rect.pos.y + rect.size.y * 0.60
            {
                continue;
            }
            let w = text.chars().count() as f64 * fs as f64 * 0.52;
            let lr = Rect {
                pos: dvec2(sx - w * 0.5, sy - fs as f64 * 0.6),
                size: dvec2(w, fs as f64 * 1.2),
            };
            // keep names out from under the top-right controls (the +/- zoom pill
            // and the my-location button) so they never render clipped behind UI.
            let ctrl_left = rect.pos.x + rect.size.x - 74.0;
            let ctrl_bottom = rect.pos.y + 250.0;
            if lr.pos.x + lr.size.x > ctrl_left && lr.pos.y < ctrl_bottom {
                continue;
            }
            if placed
                .iter()
                .any(|p| rects_overlap_with_padding(*p, lr, 6.0))
            {
                continue;
            }
            placed.push(lr);
            drawn += 1;
            // FADE-IN: ramp a newly-appeared name from 0 -> 1 over ~0.22s
            // (smoothstep) so it doesn't pop when panning/zooming reveals it.
            let seen = *self.nav_label_seen.entry(text.clone()).or_insert(now);
            let f = (((now - seen) / 0.22).clamp(0.0, 1.0)) as f32;
            let a = f * f * (3.0 - 2.0 * f);
            self.draw_text.text_style.font_size = fs;
            let (tx, ty) = (lr.pos.x, lr.pos.y);
            // 8-way halo (N/S/E/W + diagonals) for a rounder, crisper outline that
            // keeps names legible over roads/water.
            self.draw_text.color = vec4(1.0, 1.0, 1.0, 0.92 * a);
            for (dx, dy) in [
                (-1.2, 0.0), (1.2, 0.0), (0.0, -1.2), (0.0, 1.2),
                (-0.9, -0.9), (0.9, -0.9), (-0.9, 0.9), (0.9, 0.9),
            ] {
                self.draw_text.draw_abs(cx, dvec2(tx + dx, ty + dy), &text);
            }
            self.draw_text.color = vec4(0.17, 0.21, 0.27, a);
            self.draw_text.draw_abs(cx, dvec2(tx, ty), &text);
            now_shown.insert(text);
        }
        // prune first-seen times to the visible set so a name that leaves and
        // later returns fades in fresh instead of snapping back.
        self.nav_label_seen.retain(|k, _| now_shown.contains(k));
        self.nav_labels_shown = now_shown;
    }

    /// Draw an upright standing pin (screen-space billboard) whose tip sits at
    /// the projected ground point `(sx, sy)` and whose head stands `h` px above
    /// it — so it stands UP in the 2.5D scene (consistent with the perspective)
    /// instead of lying flat. `r` = head radius; both scale with distance.
    fn draw_upright_pin(&mut self, cx: &mut Cx2d, sx: f64, sy: f64, h: f64, r: f64, color: Vec4) {
        // Precise vector shapes (no glyph metrics): a shadow ellipse at the tip,
        // a vertical stem centered on sx, then a white ring + colored head disc
        // both centered EXACTLY on (sx, hy) — so the stem always meets the head
        // dead center.
        // NOTE: caller wraps all pins+puck in ONE draw_ui.begin()/end() session
        // (multiple sessions per frame only flush the last), so this just adds
        // shapes.
        let _ = cx;
        let (sxf, syf, rf) = (sx as f32, sy as f32, r as f32);
        let hyf = (sy - h) as f32; // head center
        self.draw_ui.set_color(0.05, 0.08, 0.12, 0.26);
        self.draw_ui.ellipse(sxf, syf + 1.5, rf * 0.85, rf * 0.5);
        self.draw_ui.fill_opts(LineJoin::Round, 4.0, 1.6); // softer shadow
        self.draw_ui
            .set_color(color.x, color.y, color.z, color.w);
        self.draw_ui.rect(sxf - 1.3, hyf, 2.6, h as f32);
        self.draw_ui.fill();
        self.draw_ui.set_color(1.0, 1.0, 1.0, 1.0);
        self.draw_ui.circle(sxf, hyf, rf * 1.32);
        self.draw_ui.fill_opts(LineJoin::Round, 4.0, 1.5); // softer ring
        self.draw_ui
            .set_color(color.x, color.y, color.z, color.w);
        self.draw_ui.circle(sxf, hyf, rf);
        self.draw_ui.fill_opts(LineJoin::Round, 4.0, 1.5); // softer head
    }

    /// Draw the route annotation pins (origin/via/destination) as upright
    /// standing pins in nav mode (the world-space teardrops only look right
    /// top-down, so plan mode keeps those; nav mode uses these billboards).
    /// Draw the route as a CONSTANT-SCREEN-WIDTH polyline for the plan overview.
    /// The world-space ribbon shrinks to a hairline when the camera zooms out to
    /// fit the whole A->B route, so plan mode draws the route in screen space
    /// instead — Google-Maps style: a white casing under a bright-blue core, at
    /// a fixed pixel width regardless of zoom. Added to the caller's draw_ui
    /// session (single begin/end).
    fn draw_nav_route_line(&mut self, rect: Rect, off_x: f64, off_y: f64, view_zoom: f64) {
        if self.nav_pts.len() < 2 {
            return;
        }
        let world = tile_world_size_zoom(view_zoom);
        // Project + decimate to ~220 screen points (the overview doesn't need
        // the full densified polyline; keeps the fill count bounded).
        let n = self.nav_pts.len();
        let step = (n / 220).max(1);
        let mut scr: Vec<(f32, f32)> = Vec::with_capacity(n / step + 2);
        let push = |me: &Self, p: Vec2d| {
            let (sx, sy) = me.nav_project_plan(off_x + p.x * world, off_y + p.y * world);
            // keep points within a margin of the map rect (the route line must
            // not bleed into the summary sheet below)
            if sx > rect.pos.x - 40.0
                && sx < rect.pos.x + rect.size.x + 40.0
                && sy > rect.pos.y - 40.0
                && sy < rect.pos.y + rect.size.y + 40.0
            {
                Some((sx as f32, sy as f32))
            } else {
                None
            }
        };
        let mut i = 0;
        while i < n {
            if let Some(s) = push(self, self.nav_pts[i]) {
                scr.push(s);
            }
            i += step;
        }
        if let Some(s) = push(self, self.nav_pts[n - 1]) {
            scr.push(s);
        }
        if scr.len() < 2 {
            return;
        }
        // Two passes: white casing (wider) under a bright-blue core.
        for (w, r, g, b) in [
            (8.0f32, 1.0f32, 1.0f32, 1.0f32),
            (5.0f32, 0.13f32, 0.45f32, 0.94f32),
        ] {
            self.draw_ui.set_color(r, g, b, 1.0);
            let hw = w * 0.5;
            for k in 0..scr.len() - 1 {
                let (x0, y0) = scr[k];
                let (x1, y1) = scr[k + 1];
                let (dx, dy) = (x1 - x0, y1 - y0);
                let len = (dx * dx + dy * dy).sqrt().max(1e-3);
                let (px, py) = (-dy / len * hw, dx / len * hw);
                self.draw_ui.move_to(x0 + px, y0 + py);
                self.draw_ui.line_to(x1 + px, y1 + py);
                self.draw_ui.line_to(x1 - px, y1 - py);
                self.draw_ui.line_to(x0 - px, y0 - py);
                self.draw_ui.close();
                self.draw_ui.fill_opts(LineJoin::Miter, 4.0, 1.7); // softer edge
                // round join at each vertex
                self.draw_ui.circle(x1, y1, hw);
                self.draw_ui.fill_opts(LineJoin::Round, 4.0, 1.7);
            }
            self.draw_ui.circle(scr[0].0, scr[0].1, hw);
            self.draw_ui.fill_opts(LineJoin::Round, 4.0, 1.7);
        }
    }

    fn draw_nav_route_pins(&mut self, cx: &mut Cx2d, rect: Rect, off_x: f64, off_y: f64, view_zoom: f64) {
        if self.nav_markers.is_empty() {
            return;
        }
        let world = tile_world_size_zoom(view_zoom);
        let cam_h = self.draw_map.nav.cam[0] as f64;
        let is_plan = self.is_plan();
        let markers = self.nav_markers.clone();
        for (mlat, mlon, kind) in markers {
            let n = lon_lat_to_normalized(mlon, mlat);
            // Plan uses the flat p2d projection (matches the tiles) at a constant
            // pin size; the 3D/2D chase view uses the pinhole + distance scale.
            let (sx, sy, sc) = if is_plan {
                let (sx, sy) = self.nav_project_plan(off_x + n.x * world, off_y + n.y * world);
                (sx, sy, 1.0f64)
            } else {
                let Some((sx, sy, a)) =
                    self.nav_project_flat(off_x + n.x * world, off_y + n.y * world)
                else {
                    continue;
                };
                (sx, sy, (cam_h * 1.6 / a).clamp(0.35, 1.35))
            };
            if sx < rect.pos.x - 60.0
                || sx > rect.pos.x + rect.size.x + 60.0
                || sy < rect.pos.y - 60.0
                || sy > rect.pos.y + rect.size.y + 80.0
            {
                continue;
            }
            let color = match kind {
                0 => vec4(0.11, 0.72, 0.33, 1.0), // origin green
                2 => vec4(0.92, 0.26, 0.21, 1.0), // dest red
                _ => vec4(0.10, 0.45, 0.92, 1.0), // via blue
            };
            self.draw_upright_pin(cx, sx, sy, 30.0 * sc, 9.0 * sc, color);
        }
    }

    /// Draw the moving vehicle puck at the car's projected screen position: a
    /// white halo + blue disc + white up-chevron. The heading-up camera keeps
    /// forward toward the top, so the chevron always points up. When the user
    /// pans/zooms the car moves off the anchor (and glides back on auto-restore).
    fn draw_nav_puck(&mut self, cx: &mut Cx2d, rect: Rect, off_x: f64, off_y: f64, view_zoom: f64) {
        if self.nav_pts.len() < 2 {
            return;
        }
        let world = tile_world_size_zoom(view_zoom);
        let fx = off_x + self.nav_car_norm.x * world;
        let fy = off_y + self.nav_car_norm.y * world;
        let Some((sx, sy, _)) = self.nav_project_flat(fx, fy) else {
            return;
        };
        if sx < rect.pos.x - 40.0
            || sx > rect.pos.x + rect.size.x + 40.0
            || sy < rect.pos.y - 40.0
            || sy > rect.pos.y + rect.size.y + 40.0
        {
            return;
        }
        // Precise vector puck: white halo + blue disc, then a white up-chevron
        // (filled triangle) centered on (sx,sy). No glyph metrics. Added to the
        // caller's single draw_ui session (see draw_upright_pin note).
        let _ = cx;
        let (sxf, syf) = (sx as f32, sy as f32);
        self.draw_ui.set_color(1.0, 1.0, 1.0, 1.0);
        self.draw_ui.circle(sxf, syf, 21.0);
        self.draw_ui.fill();
        self.draw_ui.set_color(0.10, 0.45, 0.92, 1.0);
        self.draw_ui.circle(sxf, syf, 17.0);
        self.draw_ui.fill();
        // up chevron centered on (sx,sy): apex above, base below (tuned so the
        // filled triangle's optical center lands on the disc center)
        self.draw_ui.set_color(1.0, 1.0, 1.0, 1.0);
        self.draw_ui.move_to(sxf, syf - 6.5);
        self.draw_ui.line_to(sxf - 7.5, syf + 7.5);
        self.draw_ui.line_to(sxf + 7.5, syf + 7.5);
        self.draw_ui.close();
        self.draw_ui.fill();
    }

    /// Cache the ENTIRE driven-loop corridor so nothing is ever re-fetched or
    /// re-parsed while looping. Walks the whole loop segment starting AT the
    /// car (so about-to-be-visible tiles are requested first), covering each
    /// route tile plus its 8 neighbours (the frustum sees to the sides too).
    /// Disk-cache-first, throttled, budget-capped per frame — it progressively
    /// warms over the first ~loop and then goes quiet (everything resident).
    fn prefetch_route_ahead(&mut self, cx: &mut Cx) {
        if self.nav_pts.len() < 2 {
            return;
        }
        let total = self.nav_cum.last().copied().unwrap_or(0.0);
        if total < 1.0 {
            return;
        }
        // constant-speed (matches update_nav_camera): drive at nav_speed_mph
        let mps = (self.nav_speed_mph.max(1.0)) * 0.44704;
        let d = (crate::splash::sim_clock_secs() * mps) % total.max(1.0);
        // the loop drives the whole route
        let loop_len = total;
        let zoom = self.request_zoom_level();
        let tiles_n = 1i32 << zoom;

        let mut seen: HashSet<TileKey> = HashSet::new();
        let mut budget = 6; // per frame; fills the whole loop over ~1-2 loops
        let mut ahead = 0.0_f64;
        while ahead <= loop_len && budget > 0 {
            let da = {
                let s = d + ahead;
                if s <= total {
                    s
                } else {
                    s - total
                }
            };
            ahead += 200.0;
            let Some(p) = sample_polyline_point_at_distance(&self.nav_pts, &self.nav_cum, da)
            else {
                continue;
            };
            let tx = (p.x * tiles_n as f64).floor() as i32;
            let ty = (p.y * tiles_n as f64).floor() as i32;
            // the route tile + a 5x5 neighbourhood — a turn swings the heading,
            // so the map to the SIDES of the route must be preloaded too (a
            // narrow corridor left the turn-revealed sides blank).
            'nb: for dy in -2..=2 {
                for dx in -2..=2 {
                    if budget == 0 {
                        break 'nb;
                    }
                    let key = TileKey {
                        z: zoom,
                        x: tx + dx,
                        y: ty + dy,
                    };
                    if !seen.insert(key) {
                        continue;
                    }
                    if self.tiles.contains_key(&key) {
                        continue;
                    }
                    if let Some(entry) = nav_store_adopt(&key, self.frame_counter) {
                        self.tiles.insert(key, entry);
                        continue;
                    }
                    if self.request_tile(cx, key, 0, true) {
                        budget -= 1;
                    }
                }
            }
        }
    }

    fn ensure_visible_tiles(&mut self, cx: &mut Cx, rect: Rect) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        self.visible_tiles = self.visible_tile_keys(rect);
        let target_zoom = self.request_zoom_level();

        self.ensure_tile_thread_pool(cx);
        self.request_visible_tiles_from_local_source(cx);

        let mut visible_set = HashSet::with_capacity(self.visible_tiles.len());
        for key in &self.visible_tiles {
            visible_set.insert(*key);
            if let Some(entry) = self.tiles.get_mut(key) {
                entry.last_used = self.frame_counter;
            }
        }

        let mut pending = nav_pending_len();

        for key in self.visible_tiles.clone() {
            let retry_attempt = self.tiles.get(&key).and_then(|entry| {
                if let TileLoadState::Failed { retry_after } = entry.state {
                    if entry.attempts < MAX_TILE_RETRIES && self.frame_counter >= retry_after {
                        return Some(entry.attempts);
                    }
                    // exhausted retries: cool down ~15s (900 frames), then
                    // start a fresh retry cycle — permanent holes are worse
                    // than an occasional extra request
                    if entry.attempts >= MAX_TILE_RETRIES
                        && self.frame_counter >= retry_after + 900
                    {
                        return Some(1);
                    }
                }
                None
            });
            if let Some(attempts) = retry_attempt {
                if pending < MAX_PENDING_REQUESTS && self.request_tile(cx, key, attempts, true) {
                    pending += 1;
                }
                continue;
            }
            if self.tiles.contains_key(&key) {
                continue;
            }
            // Ready geometry may already exist in the UI-thread shared store
            // (another instance — or this card's previous 1 Hz rebuild — parsed
            // it): adopt borrowed handles instead of refetching.
            if let Some(entry) = nav_store_adopt(&key, self.frame_counter) {
                self.tiles.insert(key, entry);
                continue;
            }
            if self.local_missing_tiles.contains(&key) {
                if self.use_network
                    && pending < MAX_PENDING_REQUESTS
                    && self.request_tile(cx, key, 0, true)
                {
                    pending += 1;
                }
                continue;
            }
            if self.request_tile(cx, key, 0, pending < MAX_PENDING_REQUESTS) {
                pending += 1;
            }
        }

        if self.tiles.len() > 640 {
            let frame_counter = self.frame_counter;
            let min_keep_zoom = target_zoom.saturating_sub(2);
            let max_keep_zoom = target_zoom.saturating_add(1);
            self.tiles.retain(|key, entry| {
                if visible_set.contains(key)
                    || matches!(
                        entry.state,
                        TileLoadState::LoadingNetwork | TileLoadState::LoadingLocal
                    )
                {
                    return true;
                }
                if key.z < min_keep_zoom || key.z > max_keep_zoom {
                    return false;
                }
                frame_counter.saturating_sub(entry.last_used) <= 240
            });
        }
        self.update_status_text();
    }

    fn visible_tile_keys(&self, rect: Rect) -> Vec<TileKey> {
        if rect.size.x <= 0.0 || rect.size.y <= 0.0 {
            return Vec::new();
        }
        let zoom = self.request_zoom_level();
        let world_size = tile_world_size(zoom);
        let center_world = self.center_norm * world_size;
        let half_size = dvec2(rect.size.x * 0.5, rect.size.y * 0.5);
        let top_left = center_world - half_size;
        let bottom_right = center_world + half_size;
        let tile_count = 1_i32 << zoom;

        let min_tx = (top_left.x / TILE_SIZE).floor() as i32 - 1;
        let max_tx = (bottom_right.x / TILE_SIZE).ceil() as i32 + 1;
        let min_ty = (top_left.y / TILE_SIZE).floor() as i32 - 1;
        let max_ty = (bottom_right.y / TILE_SIZE).ceil() as i32 + 1;

        let mut out = Vec::new();
        for ty in min_ty..=max_ty {
            if ty < 0 || ty >= tile_count {
                continue;
            }
            for tx in min_tx..=max_tx {
                out.push(TileKey {
                    z: zoom,
                    x: tx.rem_euclid(tile_count),
                    y: ty,
                });
            }
        }
        out.sort_unstable();
        out.dedup();

        let center_tx = (center_world.x / TILE_SIZE).floor() as i32;
        let center_ty = (center_world.y / TILE_SIZE).floor() as i32;
        out.sort_unstable_by_key(|key| {
            let dx = (key.x - center_tx).abs();
            let dy = (key.y - center_ty).abs();
            (dx + dy, key.y, key.x)
        });
        out
    }

    fn fill_draw_tile_keys(&mut self) {
        self.scratch_draw_tiles.clear();
        self.scratch_draw_seen.clear();

        for i in 0..self.visible_tiles.len() {
            let key = self.visible_tiles[i];
            if self.tile_is_ready(key) {
                if self.scratch_draw_seen.insert(key) {
                    self.scratch_draw_tiles.push(key);
                }
                continue;
            }
            if let Some(draw_key) = self.find_ready_ancestor(key) {
                if self.scratch_draw_seen.insert(draw_key) {
                    self.scratch_draw_tiles.push(draw_key);
                }
                continue;
            }
            self.fill_ready_descendants(key);
            for j in 0..self.scratch_descendant_tiles.len() {
                let draw_key = self.scratch_descendant_tiles[j];
                if self.scratch_draw_seen.insert(draw_key) {
                    self.scratch_draw_tiles.push(draw_key);
                }
            }
        }
    }

    fn tile_is_ready(&self, key: TileKey) -> bool {
        self.tiles.get(&key).is_some_and(|entry| {
            if let TileLoadState::Ready {
                fill_geometry,
                stroke_geometry,
                feature_count,
                ..
            } = &entry.state
            {
                *feature_count > 0 || fill_geometry.is_some() || stroke_geometry.is_some()
            } else {
                false
            }
        })
    }

    fn find_ready_ancestor(&self, mut key: TileKey) -> Option<TileKey> {
        while key.z > 0 {
            key = TileKey {
                z: key.z - 1,
                x: key.x / 2,
                y: key.y / 2,
            };
            if self.tile_is_ready(key) {
                return Some(key);
            }
        }
        None
    }

    fn fill_ready_descendants(&mut self, key: TileKey) {
        self.scratch_descendant_tiles.clear();
        for (candidate, entry) in &self.tiles {
            if !matches!(entry.state, TileLoadState::Ready { .. }) {
                continue;
            }
            if is_descendant_tile(*candidate, key) {
                self.scratch_descendant_tiles.push(*candidate);
            }
        }
    }

    fn request_tile(
        &mut self,
        cx: &mut Cx,
        tile_key: TileKey,
        attempts: u8,
        allow_network: bool,
    ) -> bool {
        if attempts == 0 && !self.use_local_mbtiles {
            let cache_path = tile_data_cache_path_for(tile_key);
            if let Ok(cached_body) = fs::read_to_string(&cache_path) {
                // Offload heavy JSON parsing + tessellation to the thread pool
                self.ensure_tile_thread_pool(cx);
                let sender = nav_workers_sender();
                let style_epoch = self.style_epoch;
                let theme_style = self.active_style().clone();
                self.tiles.insert(
                    tile_key,
                    TileEntry {
                        state: TileLoadState::LoadingLocal,
                        last_used: self.frame_counter,
                        attempts: 0,
                    },
                );
                nav_workers_execute(tile_key, move |_tag| {
                    match build_tile_buffers_from_body(tile_key, &cached_body, &theme_style) {
                        Ok(buffers) => {
                            let _ = sender.send(TileWorkerMessage::NetworkTileParsed {
                                style_epoch,
                                tile_key,
                                buffers,
                            });
                        }
                        Err(_err) => {
                            let _ = fs::remove_file(&cache_path);
                            let _ = sender.send(TileWorkerMessage::NetworkTileParseFailed {
                                style_epoch,
                                tile_key,
                                error: String::new(),
                            });
                        }
                    }
                });
                return false;
            }
        }

        if !allow_network || !self.use_network {
            return false;
        }

        // ~1 Hz card rebuilds re-run this for every missing tile: throttle so a
        // slow Overpass response (seconds) isn't re-requested each second.
        // Retries (attempts > 0) bypass it — endpoint failover must not wait.
        if attempts == 0 && nav_req_throttled(tile_key, 8.0) {
            self.tiles.insert(
                tile_key,
                TileEntry {
                    state: TileLoadState::LoadingNetwork,
                    last_used: self.frame_counter,
                    attempts,
                },
            );
            return false;
        }
        // One-time TileJSON bootstrap so the dated OpenFreeMap version stays
        // current; tiles proceed on the hardcoded default until it lands.
        if self.use_mvt {
            self.maybe_bootstrap_openfreemap(cx);
        }

        let request_id = nav_next_request_id();

        let (request, endpoint, is_mvt) = if self.use_mvt {
            // OpenFreeMap MVT: GET the versioned z/x/y.pbf (z already capped ≤14
            // by request_zoom_level, so this is a real OpenFreeMap tile).
            let mut req = HttpRequest::new(nav_mvt_tile_url(tile_key), HttpMethod::GET);
            req.set_header("User-Agent".to_string(), "makepad-map-view".to_string());
            (req, OPENFREEMAP_LABEL, true)
        } else {
            let query = overpass_query(tile_key);
            let endpoint = overpass_endpoint(tile_key, attempts);
            let mut req = HttpRequest::new(endpoint.to_string(), HttpMethod::POST);
            req.set_header("Content-Type".to_string(), "text/plain".to_string());
            req.set_header("Accept".to_string(), "application/json".to_string());
            req.set_header("User-Agent".to_string(), "makepad-map-view".to_string());
            req.set_body_string(&query);
            (req, endpoint, false)
        };

        nav_pending_insert(
            request_id,
            PendingTileRequest {
                tile_key,
                endpoint,
                is_mvt,
            },
        );
        self.tiles.insert(
            tile_key,
            TileEntry {
                state: TileLoadState::LoadingNetwork,
                last_used: self.frame_counter,
                attempts,
            },
        );
        cx.http_request(request_id, request);
        true
    }

    /// Fire the one-time OpenFreeMap TileJSON fetch to refresh the dated tile-URL
    /// version. Non-blocking — tiles keep flowing on the hardcoded default until
    /// (and unless) this lands.
    fn maybe_bootstrap_openfreemap(&mut self, cx: &mut Cx) {
        if NAV_MVT_BOOTSTRAP.with(|b| b.get()) != 0 {
            return; // already in flight or resolved this session
        }
        NAV_MVT_BOOTSTRAP.with(|b| b.set(1));
        let request_id = nav_next_request_id();
        NAV_MVT_TILEJSON_REQ.with(|c| c.set(request_id.0));
        let mut req = HttpRequest::new(OPENFREEMAP_TILEJSON_URL.to_string(), HttpMethod::GET);
        req.set_header("User-Agent".to_string(), "makepad-map-view".to_string());
        cx.http_request(request_id, req);
    }

    fn place_and_draw_labels(
        &mut self,
        cx: &mut Cx2d,
        draw_tiles: &[TileKey],
        view_zoom: f64,
        base_offset: (f64, f64),
        rect: Rect,
    ) {
        let mut label_perf = LabelPerfStats::default();
        self.collect_label_candidates(draw_tiles, view_zoom, base_offset, rect, &mut label_perf);
        if self.scratch_candidates.is_empty() {
            self.label_perf = label_perf;
            return;
        }
        self.scratch_candidates
            .sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        let candidate_budget = label_candidate_budget(view_zoom);
        if self.scratch_candidates.len() > candidate_budget {
            self.scratch_candidates.truncate(candidate_budget);
        }
        label_perf.candidates_kept = self.scratch_candidates.len();
        label_perf.shape_budget = label_shape_attempt_budget(view_zoom);

        self.path_glyphs.clear();
        // Clear but retain allocations from previous frames
        for v in self.scratch_accepted_centers.values_mut() {
            v.clear();
        }
        self.scratch_accepted_bounds.clear();
        self.scratch_accepted_plans.clear();

        for candidate_index in 0..self.scratch_candidates.len() {
            let candidate = &self.scratch_candidates[candidate_index];
            let close_repeat = self
                .scratch_accepted_centers
                .get(&candidate.name_key)
                .is_some_and(|centers| {
                    let r2 = candidate.repeat_distance * candidate.repeat_distance;
                    centers.iter().any(|c| {
                        let dx = c.x - candidate.center.x;
                        let dy = c.y - candidate.center.y;
                        dx * dx + dy * dy < r2
                    })
                });
            if close_repeat {
                label_perf.rejected_repeat += 1;
                continue;
            }

            let estimated_width =
                estimate_label_width_pixels(&candidate.text, candidate.font_scale);
            if candidate.path_length < estimated_width + 4.0 {
                label_perf.rejected_pre_short += 1;
                continue;
            }

            if label_perf.shaped_attempts >= label_perf.shape_budget {
                label_perf.rejected_budget +=
                    label_perf.candidates_kept.saturating_sub(candidate_index);
                break;
            }
            label_perf.shaped_attempts += 1;
            // Build placement needs mutable self for draw_label + path_glyphs,
            // but only reads scratch_candidates[candidate_index] immutably.
            // Safe because build_label_placement doesn't touch scratch_candidates.
            let candidate_ptr = &self.scratch_candidates[candidate_index] as *const LabelCandidate;
            let candidate_ref = unsafe { &*candidate_ptr };
            let Some(placement) = self.build_label_placement(cx, candidate_ref) else {
                label_perf.rejected_plan_none += 1;
                continue;
            };
            label_perf.shaped_ok += 1;
            if rect_outside_rect(placement.bounds, rect, LABEL_VIEW_MARGIN) {
                self.path_glyphs.truncate(placement.glyph_start);
                label_perf.rejected_outside += 1;
                continue;
            }
            if self.scratch_accepted_bounds.iter().any(|placed| {
                rects_overlap_with_padding(*placed, placement.bounds, LABEL_COLLISION_PADDING)
            }) {
                self.path_glyphs.truncate(placement.glyph_start);
                label_perf.rejected_collision += 1;
                continue;
            }

            let candidate = &self.scratch_candidates[candidate_index];
            let name_key = &candidate.name_key;
            if let Some(centers) = self.scratch_accepted_centers.get_mut(name_key) {
                centers.push(placement.center);
            } else {
                let key = name_key.clone();
                self.scratch_accepted_centers
                    .entry(key)
                    .or_default()
                    .push(placement.center);
            }
            self.scratch_accepted_bounds.push(placement.bounds);
            let glyph_count = placement.glyph_end - placement.glyph_start;
            label_perf.drawn_labels += 1;
            label_perf.drawn_glyphs += glyph_count;
            let score = candidate.score + candidate.source_rank as f64 * 2.0;
            self.scratch_accepted_plans
                .push((score, placement.glyph_start, placement.glyph_end));
        }

        self.scratch_accepted_plans
            .sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        for i in 0..self.scratch_accepted_plans.len() {
            let (_, start, end) = self.scratch_accepted_plans[i];
            self.draw_label
                .draw_path_glyphs(cx, &self.path_glyphs[start..end]);
        }
        self.label_perf = label_perf;
    }

    fn collect_label_candidates(
        &mut self,
        draw_tiles: &[TileKey],
        view_zoom: f64,
        base_offset: (f64, f64),
        rect: Rect,
        label_perf: &mut LabelPerfStats,
    ) {
        // Reuse scratch_candidates: clear but retain per-element heap allocations
        // (String, Vec<Vec2d>) from previous frames so they don't re-allocate.
        for c in self.scratch_candidates.iter_mut() {
            c.text.clear();
            c.name_key.clear();
            c.road_kind.clear();
            c.screen_path.clear();
        }
        let mut write_idx = 0usize;

        for key in draw_tiles {
            label_perf.draw_tiles += 1;
            let Some(entry) = self.tiles.get(key) else {
                continue;
            };
            let TileLoadState::Ready { labels, .. } = &entry.state else {
                continue;
            };
            if labels.is_empty() {
                continue;
            }
            label_perf.tiles_with_labels += 1;
            label_perf.labels_in_tiles += labels.len();
            let scale = 2.0_f64.powf(view_zoom - key.z as f64) as f32;
            let zoom_delta = (view_zoom - key.z as f64).abs();
            let (kox, koy) = tile_world_origin(*key);
            let key_offset = Vec2f {
                x: (base_offset.0 + kox * scale as f64) as f32,
                y: (base_offset.1 + koy * scale as f64) as f32,
            };

            for label in labels {
                label_perf.labels_scanned += 1;
                let Some(source_rank) = label_source_rank(&label.source_layer) else {
                    continue;
                };
                let name_key = normalize_label_key(label.text.as_str());
                if name_key.len() < 2 {
                    continue;
                }

                // Build screen_path into scratch buffer, then move it into candidate
                self.scratch_screen_path.clear();
                build_screen_polyline_into(
                    &label.path_points,
                    scale,
                    key_offset,
                    &mut self.scratch_screen_path,
                );
                if self.scratch_screen_path.len() < 2
                    || polyline_outside_rect(&self.scratch_screen_path, rect, LABEL_VIEW_MARGIN)
                {
                    continue;
                }
                self.scratch_cumulative.clear();
                polyline_cumulative_lengths_into(
                    &self.scratch_screen_path,
                    &mut self.scratch_cumulative,
                );
                let path_length = *self.scratch_cumulative.last().unwrap_or(&0.0);
                if path_length < LABEL_MIN_PATH_PIXELS {
                    continue;
                }
                let Some(center) = sample_polyline_point_at_distance(
                    &self.scratch_screen_path,
                    &self.scratch_cumulative,
                    path_length * 0.5,
                ) else {
                    continue;
                };
                if point_outside_rect(center, rect, LABEL_VIEW_MARGIN) {
                    continue;
                }

                let repeat_distance = repeat_distance_for_label(label.priority, source_rank);
                // Use a fixed font_scale per tile zoom level so that labels
                // don't shift along the path during continuous zoom.
                let mut font_scale = 0.92_f32;
                font_scale *= match label.priority {
                    1 => 1.08,
                    2 => 1.0,
                    _ => 0.92,
                };

                let score = source_rank as f64 * 1000.0
                    + (4_u8.saturating_sub(label.priority) as f64) * 120.0
                    + (220.0 - zoom_delta * 65.0)
                    + path_length.min(640.0) * 0.35;

                // Reuse existing candidate slot or push a new one
                if write_idx < self.scratch_candidates.len() {
                    let c = &mut self.scratch_candidates[write_idx];
                    c.text.push_str(&label.text);
                    c.name_key.push_str(&name_key);
                    c.road_kind.push_str(&label.road_kind);
                    c.source_rank = source_rank;
                    c.score = score;
                    c.path_length = path_length;
                    c.center = center;
                    c.repeat_distance = repeat_distance;
                    c.font_scale = font_scale;
                    c.screen_path.extend_from_slice(&self.scratch_screen_path);
                } else {
                    self.scratch_candidates.push(LabelCandidate {
                        text: label.text.clone(),
                        name_key,
                        road_kind: label.road_kind.clone(),
                        source_rank,
                        score,
                        path_length,
                        center,
                        repeat_distance,
                        font_scale,
                        screen_path: self.scratch_screen_path.clone(),
                    });
                }
                write_idx += 1;
                label_perf.candidates += 1;
            }
        }
        self.scratch_candidates.truncate(write_idx);
    }

    fn build_label_placement(
        &mut self,
        cx: &mut Cx2d,
        candidate: &LabelCandidate,
    ) -> Option<PathTextPlacement> {
        if candidate.screen_path.len() < 2 {
            return None;
        }

        // Smooth the candidate's screen_path into scratch_smooth_a,
        // using scratch_smooth_b and scratch_cumulative as temp buffers.
        let mut smooth_a = std::mem::take(&mut self.scratch_smooth_a);
        let mut smooth_b = std::mem::take(&mut self.scratch_smooth_b);
        let mut cum = std::mem::take(&mut self.scratch_cumulative);

        smooth_label_curve_into(
            &candidate.screen_path,
            &mut smooth_a,
            &mut smooth_b,
            &mut cum,
        );

        if smooth_a.len() < 2 {
            self.scratch_smooth_a = smooth_a;
            self.scratch_smooth_b = smooth_b;
            self.scratch_cumulative = cum;
            return None;
        }

        self.draw_label.draw_super.font_scale = candidate.font_scale;
        let run = self
            .draw_label
            .draw_super
            .prepare_single_line_run(cx, candidate.text.as_str());
        let run = match run {
            Some(r) if !r.glyphs.is_empty() => r,
            _ => {
                self.scratch_smooth_a = smooth_a;
                self.scratch_smooth_b = smooth_b;
                self.scratch_cumulative = cum;
                return None;
            }
        };

        // Build cumulative lengths for the smoothed path
        cum.clear();
        polyline_cumulative_lengths_into(&smooth_a, &mut cum);

        let text_width = run.width_in_lpxs;
        let start_distance = choose_label_start_distance(&smooth_a, &cum, text_width as f64);
        let start_distance = match start_distance {
            Some(d) => d,
            None => {
                self.scratch_smooth_a = smooth_a;
                self.scratch_smooth_b = smooth_b;
                self.scratch_cumulative = cum;
                return None;
            }
        };

        let mid_distance = start_distance + text_width as f64 * 0.5;
        let probe_delta = (text_width as f64 * 0.25).clamp(12.0, 42.0);
        let mid_tangent_angle =
            sample_polyline_tangent_angle_raw(&smooth_a, &cum, mid_distance, probe_delta);
        let mid_tangent_angle = match mid_tangent_angle {
            Some(a) => a,
            None => {
                self.scratch_smooth_a = smooth_a;
                self.scratch_smooth_b = smooth_b;
                self.scratch_cumulative = cum;
                return None;
            }
        };
        let reverse = choose_label_reverse(mid_tangent_angle);
        let label_angle_bias = if reverse { std::f32::consts::PI } else { 0.0 };

        let baseline_shift = (run.ascender_in_lpxs + run.descender_in_lpxs)
            * 0.5
            * LABEL_BASELINE_SHIFT_FACTOR as f32;

        let result = self.draw_label.place_text_along_path(
            &run,
            &smooth_a,
            &cum,
            start_distance,
            reverse,
            baseline_shift,
            label_angle_bias,
            LABEL_MAX_GLYPH_TURN_RADIANS,
            LABEL_GLYPH_ANGLE_BLEND,
            candidate.center,
            &mut self.path_glyphs,
        );

        self.scratch_smooth_a = smooth_a;
        self.scratch_smooth_b = smooth_b;
        self.scratch_cumulative = cum;
        result
    }

    fn update_status_text(&mut self) {
        let mut ready = 0usize;
        let mut loading = 0usize;
        let mut failed = 0usize;
        let mut retrying = 0usize;
        let mut exhausted = 0usize;
        let mut features = 0usize;

        for key in &self.visible_tiles {
            let Some(entry) = self.tiles.get(key) else {
                continue;
            };
            match &entry.state {
                TileLoadState::LoadingNetwork | TileLoadState::LoadingLocal => loading += 1,
                TileLoadState::Ready { feature_count, .. } => {
                    ready += 1;
                    features += *feature_count;
                }
                TileLoadState::Failed { .. } => {
                    failed += 1;
                    if entry.attempts >= MAX_TILE_RETRIES {
                        exhausted += 1;
                    } else {
                        retrying += 1;
                    }
                }
            }
        }

        let counters = (ready, loading, failed, retrying, exhausted, features);
        let lp = self.label_perf;
        // Skip format! if nothing changed since the last call
        if counters == self.prev_status_counters
            && lp == self.prev_status_label_perf
            && !self.status.is_empty()
        {
            return;
        }
        self.prev_status_counters = counters;
        self.prev_status_label_perf = lp;

        self.status = format!(
            "Amsterdam [{}|{}] z{:.2} (req:{})  ready:{}  loading:{}  failed:{}(retry:{} stuck:{})  features:{}  labels(tile:{} scan:{} cand:{}/{} shape:{}/{}(b:{}) draw:{} glyphs:{} rej:r{} ps{} p{} o{} c{} b{})",
            self.source_mode_label(), self.theme_label(), self.view_zoom(), self.request_zoom_level(),
            ready, loading, failed, retrying, exhausted, features,
            lp.labels_in_tiles, lp.labels_scanned, lp.candidates_kept, lp.candidates,
            lp.shaped_ok, lp.shaped_attempts, lp.shape_budget, lp.drawn_labels, lp.drawn_glyphs,
            lp.rejected_repeat, lp.rejected_pre_short, lp.rejected_plan_none,
            lp.rejected_outside, lp.rejected_collision, lp.rejected_budget,
        );
    }

    fn view_zoom(&self) -> f64 {
        let min = self.min_zoom.max(0.0);
        let max = self.max_zoom.max(min);
        self.zoom.clamp(min, max)
    }

    fn request_zoom_level(&self) -> u32 {
        let mut zoom = self.view_zoom().round() as u32;
        if self.use_local_mbtiles {
            zoom = zoom.clamp(LOCAL_MBTILES_MIN_ZOOM, LOCAL_MBTILES_MAX_ZOOM);
        } else if self.use_mvt {
            // OpenFreeMap maxzoom is 14; a closer view fetches the z14 tile and
            // overzooms it (the draw path already scales by view_zoom − key.z).
            zoom = zoom.min(OPENFREEMAP_MAX_ZOOM);
        }
        zoom
    }

    fn source_mode_label(&self) -> &'static str {
        if self.use_local_mbtiles {
            "offline"
        } else if self.use_network {
            "online"
        } else {
            "disabled"
        }
    }

    fn theme_label(&self) -> &'static str {
        if self.dark_theme {
            "dark"
        } else {
            "light"
        }
    }
}
