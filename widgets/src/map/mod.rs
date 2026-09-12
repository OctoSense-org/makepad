pub mod archive;
pub mod drape;
pub mod geometry;
pub(crate) mod icons;
pub(crate) mod label;
pub mod overlay;
pub mod style;
pub mod tile;
pub(crate) mod tile_draw;
pub mod view;

pub use overlay::{MapMarker, MapPuck, MapRouteOverlay};
pub use view::*;

fn warm_shared_registries() {
    icons::warm_icon_registries();
    tile::warm_tile_registries();
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod bake_report;

/// Decode a Google/OSRM polyline5 string into (lat, lon) pairs.
pub(crate) fn decode_polyline5(encoded: &str) -> Vec<(f64, f64)> {
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
            Some(if result & 1 != 0 { !(result >> 1) } else { result >> 1 })
        };
        let Some(dlat) = decode_one(&mut i) else { break };
        let Some(dlon) = decode_one(&mut i) else { break };
        lat += dlat;
        lon += dlon;
        out.push((lat as f64 * 1e-5, lon as f64 * 1e-5));
    }
    out
}

/// Great-circle distance in metres.
pub(crate) fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
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
