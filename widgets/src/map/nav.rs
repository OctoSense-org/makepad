//! Navigation layer for [`MapView`](super::view::MapView) — the surface the
//! AppCard nav cards drive:
//!
//! * declarative props — `nav_mode` (`""` | `"2d"` | `"3d"` | `"plan"` |
//!   `"follow"` | `"follow3d"`), `nav_polyline` (Google polyline5),
//!   `route_markers` (`"lat,lon,kind;…"`, kind 0 origin / 1 stop / 2
//!   destination), `route_badge`, and the `nav_*` tuning numbers;
//! * script methods — `ui.<map>.set_nav_polyline(s)`, `set_route_markers(s)`,
//!   `set_nav_recenter(_)`, `nav_zoom_by(delta)`, `nav_center_origin()`.
//!
//! It renders through the map's own overlay (route ribbon, puck, pins) and
//! camera (center / rotation / tilt / zoom) rather than the dedicated pinhole
//! ground projection the AppCard fork carried, so tiles, labels, terrain and
//! gestures stay the map's. `"2d"`/`"3d"` drive a simulated vehicle along the
//! route on the sim clock (the same clock as `sys.simsecs`, so DSL banners
//! stay in lockstep); `"follow"`/`"follow3d"` take the vehicle from the device
//! fix (or `center_lat`/`center_lon`) and move only when the device does;
//! `"plan"` is a static north-up fit of the whole route into the band above a
//! card's summary sheet.
use crate::makepad_draw::*;

use super::geometry::{lon_lat_to_normalized, sample_polyline_point_at_distance};
use super::{decode_polyline5, haversine_m};

/// Seconds without a touch after which a user-adjusted follow camera glides
/// back onto the vehicle.
pub(super) const NAV_RECENTER_IDLE_SECS: f64 = 4.0;
/// Camera tilt (degrees) for the 3D chase modes; the map's zoom-coupled tilt
/// cap still applies per frame.
pub(super) const NAV_CHASE_TILT_DEG: f64 = 60.0;
/// How far ahead of the vehicle (route metres) the heading is read, so the
/// camera turns into a bend rather than on it.
const NAV_LOOK_AHEAD_M: f64 = 45.0;
/// Resample spacing (metres) for the decoded route.
const NAV_RESAMPLE_M: f64 = 6.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum NavKind {
    /// A normal map.
    Off,
    /// Tilted chase view behind the vehicle, heading-up.
    Chase3d,
    /// Top-down heading-up (`2d`/`follow`) or the static plan fit (`plan`).
    HeadingUp2d,
}

pub(super) fn nav_kind(mode: &str) -> NavKind {
    match mode.trim() {
        "3d" | "3D" | "follow3d" => NavKind::Chase3d,
        "2d" | "2D" | "plan" | "follow" => NavKind::HeadingUp2d,
        _ => NavKind::Off,
    }
}

pub(super) fn is_plan(mode: &str) -> bool {
    mode.trim() == "plan"
}

/// FOLLOW: the camera goes where the DEVICE is, and nowhere on its own —
/// unlike `2d`/`3d`, which drive a simulated vehicle (a demo camera).
pub(super) fn is_follow(mode: &str) -> bool {
    matches!(mode.trim(), "follow" | "follow3d")
}

/// A real coordinate: finite, in range, and not the 0,0 an unresolved
/// endpoint reports (that is the Gulf of Guinea, not a place).
pub(super) fn is_a_place(lat: f64, lon: f64) -> bool {
    lat.is_finite()
        && lon.is_finite()
        && lat.abs() <= 90.0
        && lon.abs() <= 180.0
        && !(lat.abs() < 1e-9 && lon.abs() < 1e-9)
}

/// `"lat,lon,kind;…"` -> the pins. Kind 0 origin, 1 a stop, 2 the destination
/// (missing kind = a stop). Shared by the `set_route_markers` method and the
/// `route_markers` property so the two paths cannot drift. A 0,0 pin is
/// dropped: that is an unresolved endpoint.
pub(super) fn parse_route_markers(s: &str) -> Vec<(f64, f64, u8)> {
    let mut out = Vec::new();
    for part in s.split(';') {
        let f: Vec<&str> = part.split(',').collect();
        if f.len() < 2 {
            continue;
        }
        let lat: f64 = f[0].trim().parse().unwrap_or(f64::NAN);
        let lon: f64 = f[1].trim().parse().unwrap_or(f64::NAN);
        let kind: u8 = f.get(2).and_then(|k| k.trim().parse().ok()).unwrap_or(1);
        if !is_a_place(lat, lon) {
            continue;
        }
        out.push((lat, lon, kind));
    }
    out
}

/// Pin colour by kind: origin green, stop amber, destination red.
pub(super) fn marker_color(kind: u8) -> Vec4f {
    match kind {
        0 => Vec4f { x: 0.16, y: 0.66, z: 0.36, w: 1.0 },
        2 => Vec4f { x: 0.88, y: 0.24, z: 0.20, w: 1.0 },
        _ => Vec4f { x: 0.95, y: 0.62, z: 0.12, w: 1.0 },
    }
}

/// Everything the navigation layer keeps between frames.
#[derive(Default)]
pub(super) struct NavState {
    /// Decoded, resampled route in normalized world coordinates.
    pub pts: Vec<Vec2d>,
    /// Cumulative route metres, parallel to `pts`.
    pub cum: Vec<f64>,
    pub markers: Vec<(f64, f64, u8)>,
    poly_seen: String,
    markers_seen: String,
    /// Smoothed vehicle heading, radians clockwise from north.
    pub bearing: f64,
    bearing_init: bool,
    /// Vehicle position (normalized).
    pub car: Vec2d,
    /// User pan offset from the vehicle, plus the button-driven glide targets.
    pub pan: Vec2d,
    pub pan_anim: Option<Vec2d>,
    pub zoom_anim: Option<f64>,
    /// The user moved the camera; the follow-cam waits `NAV_RECENTER_IDLE_SECS`
    /// after the last touch and glides back.
    pub user_adjusted: bool,
    pub last_touch: f64,
    /// Zoom to glide back to (the card's zoom, or the plan fit).
    pub home_zoom: f64,
    // Follow mode: the vehicle is interpolated between device fixes so it
    // glides rather than jumps once a second.
    follow_d: f64,
    seg_from: f64,
    seg_to: f64,
    seg_t0: f64,
    seg_t1: f64,
    /// Per-frame pump while a nav camera is live (or a glide is in flight).
    pub next_frame: NextFrame,
}

impl NavState {
    /// Adopt the declarative polyline + pins. Returns true when either
    /// changed, i.e. the overlay's route/markers need rebuilding.
    pub fn adopt(&mut self, polyline: &str, markers: &str) -> bool {
        let mut changed = false;
        if markers != self.markers_seen {
            self.markers_seen = markers.to_string();
            let next = parse_route_markers(markers);
            if next != self.markers {
                self.markers = next;
                changed = true;
            }
        }
        let poly = polyline.trim();
        if poly != self.poly_seen {
            self.poly_seen = poly.to_string();
            self.set_route_coords(decode_polyline5(poly));
            changed = true;
        }
        changed
    }

    /// `(lat, lon)` route -> resampled normalized points + cumulative metres.
    fn set_route_coords(&mut self, coords0: Vec<(f64, f64)>) {
        self.pts.clear();
        self.cum.clear();
        self.bearing_init = false;
        self.follow_d = 0.0;
        self.seg_from = 0.0;
        self.seg_to = 0.0;
        self.seg_t0 = 0.0;
        self.seg_t1 = 0.0;
        if coords0.len() < 2 {
            return;
        }
        // Resample long straights so the vehicle glides and the traveled
        // split lands close to the vehicle instead of at the next vertex.
        let mut coords: Vec<(f64, f64)> = Vec::with_capacity(coords0.len() * 4);
        for i in 0..coords0.len() {
            let (lat, lon) = coords0[i];
            coords.push((lat, lon));
            if i + 1 < coords0.len() {
                let (lat2, lon2) = coords0[i + 1];
                let seg = haversine_m(lat, lon, lat2, lon2);
                let steps = ((seg / NAV_RESAMPLE_M).floor() as usize).min(2048);
                for s in 1..steps {
                    let t = s as f64 / steps as f64;
                    coords.push((lat + (lat2 - lat) * t, lon + (lon2 - lon) * t));
                }
            }
        }
        let mut cum = 0.0_f64;
        let mut prev: Option<(f64, f64)> = None;
        for &(lat, lon) in &coords {
            if let Some((plat, plon)) = prev {
                cum += haversine_m(plat, plon, lat, lon);
            }
            self.pts.push(lon_lat_to_normalized(lon, lat));
            self.cum.push(cum);
            prev = Some((lat, lon));
        }
    }

    pub fn has_route(&self) -> bool {
        self.pts.len() >= 2
    }

    pub fn total_m(&self) -> f64 {
        self.cum.last().copied().unwrap_or(0.0)
    }

    pub fn point_at(&self, d: f64) -> Option<Vec2d> {
        sample_polyline_point_at_distance(&self.pts, &self.cum, d)
    }

    /// Route metres at the point on the route closest to `p` (normalized).
    ///
    /// The CLOSEST segment, not the first one near enough: a route that
    /// doubles back — a U-turn, a cloverleaf, a street driven twice — has two
    /// segments by the same point, and taking the earlier would rewind the
    /// camera to a turn already made.
    pub fn distance_at(&self, p: Vec2d) -> f64 {
        let mut best = (f64::MAX, 0.0);
        for i in 0..self.pts.len().saturating_sub(1) {
            let a = self.pts[i];
            let b = self.pts[i + 1];
            let ab = b - a;
            let len2 = ab.x * ab.x + ab.y * ab.y;
            let t = if len2 > 0.0 {
                (((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let q = a + ab * t;
            let dq = (p - q).length();
            if dq < best.0 {
                best = (dq, self.cum[i] + (self.cum[i + 1] - self.cum[i]) * t);
            }
        }
        best.1
    }

    /// Index of the first route point at or past `d` — the overlay dims
    /// everything before it as already driven.
    pub fn traveled_index(&self, d: f64) -> usize {
        self.cum.partition_point(|&c| c < d)
    }

    /// Bounding box (normalized) of the route, or of the pins when there is
    /// no route yet.
    pub fn bounds(&self) -> Option<(Vec2d, Vec2d)> {
        let (mut min, mut max) = (dvec2(f64::MAX, f64::MAX), dvec2(f64::MIN, f64::MIN));
        let mut any = false;
        for p in &self.pts {
            min = dvec2(min.x.min(p.x), min.y.min(p.y));
            max = dvec2(max.x.max(p.x), max.y.max(p.y));
            any = true;
        }
        if !any {
            for &(lat, lon, _) in &self.markers {
                let p = lon_lat_to_normalized(lon, lat);
                min = dvec2(min.x.min(p.x), min.y.min(p.y));
                max = dvec2(max.x.max(p.x), max.y.max(p.y));
                any = true;
            }
        }
        any.then_some((min, max))
    }

    /// Ease the button-set zoom / pan targets one frame (glide, not snap).
    /// Returns true while something is still moving.
    pub fn tick_glide(&mut self, zoom: &mut f64) -> bool {
        let mut moving = false;
        if let Some(tz) = self.zoom_anim {
            let d = tz - *zoom;
            if d.abs() > 0.004 {
                *zoom += d * 0.22;
                moving = true;
            } else {
                *zoom = tz;
                self.zoom_anim = None;
            }
        }
        if let Some(tp) = self.pan_anim {
            let dx = tp.x - self.pan.x;
            let dy = tp.y - self.pan.y;
            if dx.abs() > 1e-7 || dy.abs() > 1e-7 {
                self.pan.x += dx * 0.22;
                self.pan.y += dy * 0.22;
                moving = true;
            } else {
                self.pan = tp;
                self.pan_anim = None;
            }
        }
        moving
    }

    /// Idle recenter for a user-adjusted follow camera: after
    /// `NAV_RECENTER_IDLE_SECS` the pan decays and the zoom glides home.
    pub fn tick_recenter(&mut self, now: f64, zoom: &mut f64) {
        if !self.user_adjusted || now - self.last_touch <= NAV_RECENTER_IDLE_SECS {
            return;
        }
        self.pan.x *= 0.84;
        self.pan.y *= 0.84;
        if self.home_zoom > 0.0 {
            *zoom += (self.home_zoom - *zoom) * 0.16;
        }
        let z_done = self.home_zoom <= 0.0 || (*zoom - self.home_zoom).abs() < 0.01;
        if self.pan.x.abs() < 1e-6 && self.pan.y.abs() < 1e-6 && z_done {
            self.pan = dvec2(0.0, 0.0);
            if self.home_zoom > 0.0 {
                *zoom = self.home_zoom;
            }
            self.user_adjusted = false;
        }
    }

    /// Follow mode: route metres for the vehicle, interpolated toward the
    /// latest fix's `target` over the interval the fixes arrive at, so the
    /// puck glides between once-a-second positions. A jump of more than 400 m
    /// (a new route, a tunnel exit) snaps instead of sweeping the map.
    pub fn follow_distance(&mut self, target: f64, now: f64) -> f64 {
        if (target - self.seg_to).abs() > 0.01 {
            self.seg_from = self.follow_d;
            self.seg_to = target;
            self.seg_t0 = self.seg_t1.max(now - 2.0);
            self.seg_t1 = now;
        }
        if (target - self.follow_d).abs() > 400.0 {
            self.follow_d = target;
            self.seg_from = target;
            self.seg_to = target;
            self.seg_t0 = now;
            self.seg_t1 = now;
        } else {
            let span = (self.seg_t1 - self.seg_t0).max(0.05);
            let frac = ((now - span - self.seg_t0) / span).clamp(0.0, 1.0);
            self.follow_d = self.seg_from + (self.seg_to - self.seg_from) * frac;
        }
        self.follow_d
    }

    /// Vehicle heading at route metres `d`, read `NAV_LOOK_AHEAD_M` ahead and
    /// smoothed so the camera swings into bends rather than snapping.
    pub fn heading_at(&mut self, car: Vec2d, d: f64) -> f64 {
        let total = self.total_m();
        let look = self
            .point_at((d + NAV_LOOK_AHEAD_M).min(total))
            .unwrap_or(car);
        let fwd = dvec2(look.x - car.x, look.y - car.y);
        let target = if fwd.x.abs() < 1e-12 && fwd.y.abs() < 1e-12 {
            self.bearing
        } else {
            fwd.x.atan2(-fwd.y)
        };
        if !self.bearing_init {
            self.bearing = target;
            self.bearing_init = true;
        } else {
            let pi = std::f64::consts::PI;
            let mut diff = target - self.bearing;
            while diff > pi {
                diff -= 2.0 * pi;
            }
            while diff < -pi {
                diff += 2.0 * pi;
            }
            self.bearing += diff * 0.10;
        }
        self.bearing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_markers_parse_and_drop_unresolved() {
        let pins = parse_route_markers("37.37,-121.96,0;0,0,1;37.40,-121.90;37.33,-121.89,2");
        assert_eq!(pins, vec![(37.37, -121.96, 0), (37.40, -121.90, 1), (37.33, -121.89, 2)]);
        assert!(parse_route_markers("").is_empty());
    }

    #[test]
    fn nav_modes_classify() {
        assert_eq!(nav_kind(""), NavKind::Off);
        assert_eq!(nav_kind(" 3d "), NavKind::Chase3d);
        assert_eq!(nav_kind("follow3d"), NavKind::Chase3d);
        assert_eq!(nav_kind("2d"), NavKind::HeadingUp2d);
        assert_eq!(nav_kind("plan"), NavKind::HeadingUp2d);
        assert!(is_plan("plan") && !is_plan("2d"));
        assert!(is_follow("follow") && !is_follow("3d"));
    }

    #[test]
    fn adopted_route_has_monotone_metres_and_closest_segment_distance() {
        let mut nav = NavState::default();
        // A ~1.1 km L-shape near Santa Clara: (lat, lon) legs east then north.
        let poly = super::super::encode_polyline5_for_test(&[
            (37.3700, -121.9700),
            (37.3700, -121.9600),
            (37.3760, -121.9600),
        ]);
        assert!(nav.adopt(&poly, "37.37,-121.97,0;37.376,-121.96,2"));
        assert!(nav.has_route());
        assert!(nav.total_m() > 1400.0 && nav.total_m() < 1700.0, "{}", nav.total_m());
        assert!(nav.cum.windows(2).all(|w| w[1] >= w[0]));
        // The corner is ~885 m in; a point just past it along the north leg
        // resolves to the north leg, not the (nearer-in-index) east leg.
        let corner = nav.distance_at(lon_lat_to_normalized(-121.9600, 37.3700));
        let north = nav.distance_at(lon_lat_to_normalized(-121.9600, 37.3730));
        assert!(north > corner + 200.0, "north={north} corner={corner}");
        assert_eq!(nav.markers.len(), 2);
        // Same inputs again: nothing changed.
        assert!(!nav.adopt(&poly, "37.37,-121.97,0;37.376,-121.96,2"));
    }
}
