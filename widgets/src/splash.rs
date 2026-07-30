use crate::{
    makepad_derive_widget::*,
    makepad_draw::*,
    view::View,
    widget::*,
    widget_async::{CxSplashVmExt, SplashVmId, MAIN_SPLASH_VM_ID},
    widget_tree::CxWidgetExt,
};
// `vm.host.cx_mut()` — reach the host Cx from a script helper (sys.weather fetch).
use crate::makepad_draw::makepad_platform::script::vm::ScriptVmCx;

#[derive(Clone, Debug, Default)]
pub enum SplashAction {
    Notify {
        event_id: String,
        payload: String,
    },
    #[default]
    None,
}

/// Web-Mercator "slippy map" tile (x, y) covering `lat`/`lon` at zoom `z`
/// (the standard OSM/XYZ scheme used by Carto and WAQI tile servers).
fn slippy_tile(lat: f64, lon: f64, z: u32) -> (i64, i64) {
    let n = (1u64 << z) as f64;
    let x = ((lon + 180.0) / 360.0 * n).floor();
    let y = ((1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0 * n).floor();
    let max = (1i64 << z) - 1;
    ((x as i64).clamp(0, max), (y as i64).clamp(0, max))
}

/// Fractional Web-Mercator tile coords plus the top-left tile of the 2x2
/// block that best CENTERS (lat, lon): the point's own tile joined with its
/// nearest neighbor in each axis, so the anchor always lands in the middle
/// half of the mosaic (offset fraction in [0.25, 0.75) per axis). Returns
/// (left_tile_x, top_tile_y, x_fraction_of_mosaic, y_fraction_of_mosaic).
/// Backs `sys.maptile` (URLs) and `sys.mappin` (pin offset) — the two MUST
/// agree, which is why the math lives in one place.
fn slippy_mosaic(lat: f64, lon: f64, z: u32) -> (i64, i64, f64, f64) {
    let n = (1u64 << z) as f64;
    let xf = (lon + 180.0) / 360.0 * n;
    let yf = (1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0 * n;
    let left = if xf.fract() >= 0.5 { xf.floor() } else { xf.floor() - 1.0 };
    let top = if yf.fract() >= 0.5 { yf.floor() } else { yf.floor() - 1.0 };
    let max = (1i64 << z) - 1;
    let left = (left as i64).clamp(0, (max - 1).max(0));
    let top = (top as i64).clamp(0, (max - 1).max(0));
    ((left), (top), (xf - left as f64) / 2.0, (yf - top as f64) / 2.0)
}

/// Yesterday's civil date (UTC) as `YYYY-MM-DD` — the most recent day for
/// which NASA GIBS daily global mosaics are guaranteed complete. Days-to-date
/// via Howard Hinnant's `civil_from_days` (no chrono dep in this crate).
fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Howard Hinnant's `civil_from_days`: days-since-Unix-epoch → (year, month, day).
/// `pub(crate)` so the StockPlot widget can reuse it for date tick labels.
pub(crate) fn civil_from_days(days: i64) -> (i64, u64, u64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn gibs_latest_date() -> String {
    let (y, m, d) = civil_from_days((now_unix_secs() / 86_400) as i64 - 1); // yesterday
    format!("{y:04}-{m:02}-{d:02}")
}

/// A GIBS `TIME=` string for `minutes_ago` in the past, floored to a 10-minute
/// boundary (the geostationary AHI/ABI granule cadence). Format
/// `YYYY-MM-DDTHH:MM:00Z`.
fn gibs_datetime(minutes_ago: i64) -> String {
    let target = (now_unix_secs() as i64 - minutes_ago * 60).max(0);
    let target = (target / 600) * 600; // floor to 10 min
    let (y, m, d) = civil_from_days(target / 86_400);
    let sod = target % 86_400;
    let (hh, mm) = (sod / 3600, (sod % 3600) / 60);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:00Z")
}

pub fn register_agent_module(vm: &mut ScriptVm) {
    let agent = vm.new_module(id!(agent));
    vm.add_method(
        agent,
        id_lut!(notify),
        script_args_def!(event = NIL, payload = NIL),
        |vm, args| {
            let event_value = script_value!(vm, args.event);
            let payload_value = script_value!(vm, args.payload);

            let mut event_id = String::new();
            vm.bx.heap.cast_to_string(event_value, &mut event_id);

            let mut payload = String::new();
            vm.bx.heap.to_json_inner(payload_value, &mut payload);

            Cx::post_action(SplashAction::Notify { event_id, payload });
            NIL
        },
    );
    vm.set_injected_global(id!(agent), agent.into());

    // `sys` module — Rust helpers callable from generated splash code.
    // Extend this with more methods (data fetchers, formatters, card builders)
    // and teach the LLM to call them in the A2App prompt.
    let sys = vm.new_module(id!(sys));

    // sys.photo("tokyo skyline sunset") -> a full-screen 9:16 image URL for that
    // subject (pollinations.ai renders the prompt with an AI model, so the photo
    // always matches the subject). Centralises image sourcing in Rust so it can be
    // improved (curation, quality, providers) without touching the prompt.
    // Use as `Image{ src: http_resource(sys.photo("<q>")) }`.
    vm.add_method(
        sys,
        id_lut!(photo),
        script_args_def!(query = NIL),
        |vm, args| {
            let query_value = script_value!(vm, args.query);
            let mut query = String::new();
            vm.bx.heap.cast_to_string(query_value, &mut query);

            // AI-generated, always ON-TOPIC 9:16 portrait image. loremflickr
            // OR-matches comma tags, so a multi-word subject ("paris eiffel
            // tower sunny") returned unrelated photos (a cat statue). Pollinations
            // renders the full natural-language prompt, so the photo always
            // matches the subject and is high quality — the "nano banana"-style
            // AI source the app wants for beautiful full-screen backgrounds.
            let q = query.trim();
            let q = if q.is_empty() {
                "beautiful cinematic landscape scenery, golden hour"
            } else {
                q
            };
            // Percent-encode the prompt for a URL path segment (RFC 3986):
            // keep unreserved chars, encode everything else (incl. spaces) by
            // UTF-8 byte.
            let mut enc = String::with_capacity(q.len() * 3);
            let mut buf = [0u8; 4];
            for ch in q.chars() {
                if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '~') {
                    enc.push(ch);
                } else {
                    for b in ch.encode_utf8(&mut buf).as_bytes() {
                        enc.push('%');
                        enc.push(char::from_digit((b >> 4) as u32, 16).unwrap().to_ascii_uppercase());
                        enc.push(char::from_digit((b & 0xF) as u32, 16).unwrap().to_ascii_uppercase());
                    }
                }
            }
            // 1080x1920 = 9:16. nologo strips the watermark; model=flux is fast
            // and photoreal.
            let url = format!(
                "https://image.pollinations.ai/prompt/{enc}?width=1080&height=1920&nologo=true&model=flux"
            );
            vm.bx.heap.new_string_from_str(&url)
        },
    );

    // sys.satellite(lat, lon) -> REAL satellite cloud imagery (卫星云图) for the city's
    // region: NASA GIBS WMS, MODIS Terra true-color corrected reflectance for yesterday
    // (UTC) — the most recent complete daily global mosaic. Actual clouds over actual
    // terrain, daylit everywhere, keyless, and a single GetMap call returns an
    // arbitrary-size image so a full-width pane needs no tile stitching. 2:1 aspect
    // (880x440 over a ~14°x7° box) — pair with `fit: ImageFit.CropToFill` in a wide pane.
    // Use as `Image{ src: http_resource(sys.satellite(LAT, LON)) fit: ImageFit.CropToFill }`.
    vm.add_method(
        sys,
        id_lut!(satellite),
        script_args_def!(lat = NIL, lon = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(35.68);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(139.65);
            // Keep the 7°-tall box inside the poles; wrap longitude edges.
            let lat = lat.clamp(-78.0, 78.0);
            let (min_lon, max_lon) = ((lon - 7.0).max(-180.0), (lon + 7.0).min(180.0));
            let (min_lat, max_lat) = (lat - 3.5, lat + 3.5);
            let date = gibs_latest_date();
            let url = format!(
                "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&SRS=EPSG:4326&BBOX={min_lon},{min_lat},{max_lon},{max_lat}&WIDTH=880&HEIGHT=440&FORMAT=image/jpeg&TIME={date}"
            );
            vm.bx.heap.new_string_from_str(&url)
        },
    );

    // sys.satellite_ir(lat, lon, frames_ago) -> ONE frame of a geostationary
    // cloud-motion loop (卫星云图动画). Clean-IR brightness-temperature (clouds =
    // white on dark), which — unlike the once-daily MODIS still in sys.satellite —
    // updates every 10 min and is available day AND night, so cycling `frames_ago`
    // 0..N via a Splash `fn tick()` + `ui.<img>.set_src(...)` animates real cloud
    // movement. `frames_ago` 0 = newest (a fixed ~80 min latency floor so the
    // granule is published), each +1 steps 10 min further back. Satellite picked by
    // longitude: Himawari (Asia/Pacific) vs GOES-East (Americas/Atlantic); both are
    // GIBS "best", keyless, snap TIME to the nearest granule. Use as
    // `Image{ src: http_resource(sys.satellite_ir(LAT, LON, N)) fit: ImageFit.CropToFill }`.
    vm.add_method(
        sys,
        id_lut!(satellite_ir),
        script_args_def!(lat = NIL, lon = NIL, frames_ago = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(35.68);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(139.65);
            let frames_ago = script_value!(vm, args.frames_ago)
                .as_number()
                .unwrap_or(0.0)
                .clamp(0.0, 24.0) as i64;
            let lat = lat.clamp(-78.0, 78.0);
            let (min_lon, max_lon) = ((lon - 7.0).max(-180.0), (lon + 7.0).min(180.0));
            let (min_lat, max_lat) = (lat - 3.5, lat + 3.5);
            // ~80 min latency floor + 10 min per older frame.
            let dt = gibs_datetime(80 + frames_ago * 10);
            // Himawari sees Asia/Pacific; GOES-East the Americas/Atlantic.
            let layer = if lon >= 60.0 || lon < -140.0 {
                "Himawari_AHI_Band13_Clean_Infrared"
            } else {
                "GOES-East_ABI_Band13_Clean_Infrared"
            };
            let url = format!(
                "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={layer}&SRS=EPSG:4326&BBOX={min_lon},{min_lat},{max_lon},{max_lat}&WIDTH=880&HEIGHT=440&FORMAT=image/png&TIME={dt}"
            );
            vm.bx.heap.new_string_from_str(&url)
        },
    );

    // sys.basemap(lat, lon) -> a warm, LABELLED base-map tile (Carto "Voyager", no key) at
    // the city, meant to sit UNDER sys.airmap in an Overlay so the AQI colours have legible
    // geographic context. `voyager_labels_under` keeps place labels BENEATH the translucent
    // AQI markers so both read clearly. Zoom 8 frames the metro itself (not the whole
    // region — fewer, larger AQI badges on top), and the `@2x` retina tile (512px) stays
    // sharp in a full-width pane.
    vm.add_method(
        sys,
        id_lut!(basemap),
        script_args_def!(lat = NIL, lon = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let (x, y) = slippy_tile(lat, lon, 8);
            let url = format!(
                "https://a.basemaps.cartocdn.com/rastertiles/voyager_labels_under/8/{x}/{y}@2x.png"
            );
            vm.bx.heap.new_string_from_str(&url)
        },
    );

    // sys.airmap(lat, lon) -> a LIVE air-quality colour overlay tile (WAQI, US-EPA AQI
    // scale). Mostly transparent except where AQI data exists, so stack it OVER
    // sys.basemap(lat, lon) at the SAME lat/lon in an Overlay (both use zoom 8 — one
    // zoom step in from the old 7 quarters the station-marker density, so the badges
    // read as a handful of legible chips instead of an overlapping pile).
    // Use as `View{ flow: Overlay Image{basemap} Image{airmap} }`.
    vm.add_method(
        sys,
        id_lut!(airmap),
        script_args_def!(lat = NIL, lon = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let (x, y) = slippy_tile(lat, lon, 8);
            let url = format!("https://tiles.waqi.info/tiles/usepa-aqi/8/{x}/{y}.png?token=_");
            vm.bx.heap.new_string_from_str(&url)
        },
    );

    // sys.maptile(lat, lon, zoom, "tl"|"tr"|"bl"|"br") -> one quadrant URL of a
    // CENTERED 2x2 Carto "Voyager" mosaic around (lat, lon). Unlike sys.basemap
    // (single tile, the point lands wherever it falls), the four quadrants are
    // chosen so the anchor sits in the middle half of the mosaic — pair with
    // sys.mappin for the 📍. Layout: two `flow: Right` rows of two square
    // Images inside a square pane; ALL FOUR calls must share the same lat/lon/
    // zoom. Zoom by intent: country 5, city 12, district 14, landmark 16.
    // Keyless Carto raster (fair use) — cards MUST caption the pane
    // "© OpenStreetMap contributors © CARTO". Subdomain rotates per quadrant so
    // the 4 GETs parallelize. Pure URL builder (feed to http_resource).
    vm.add_method(
        sys,
        id_lut!(maptile),
        script_args_def!(lat = NIL, lon = NIL, zoom = NIL, quad = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat)
                .as_number()
                .unwrap_or(0.0)
                .clamp(-85.0, 85.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let z = script_value!(vm, args.zoom)
                .as_number()
                .unwrap_or(12.0)
                .clamp(3.0, 17.0) as u32;
            let quad_v = script_value!(vm, args.quad);
            let mut quad = String::new();
            vm.bx.heap.cast_to_string(quad_v, &mut quad);
            let (left, top, _, _) = slippy_mosaic(lat, lon, z);
            let (dx, dy, sub) = match quad.trim().to_ascii_lowercase().as_str() {
                "tr" => (1, 0, "b"),
                "bl" => (0, 1, "c"),
                "br" => (1, 1, "d"),
                _ => (0, 0, "a"), // "tl" and anything unrecognized
            };
            let url = format!(
                "https://{sub}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{}/{}@2x.png",
                left + dx,
                top + dy
            );
            vm.bx.heap.new_string_from_str(&url)
        },
    );

    // sys.mappin(lat, lon, zoom, "x"|"y", size) -> the pin's offset IN DP from
    // the mosaic pane's top-left, for a SQUARE sys.maptile 2x2 pane `size` dp
    // wide. Same math as sys.maptile, so the pin lands exactly on the anchor.
    // Bind into layout like sys.stockbar heights — e.g. an Overlay child:
    //   View{ padding: Inset{ left: sys.mappin(LAT,LON,Z,"x",372) - 11
    //                         top:  sys.mappin(LAT,LON,Z,"y",372) - 22 }
    //         Label{ text: "📍" } }
    // (subtract half the glyph width / full height so the pin TIP marks the spot).
    // Pure math — no fetch, no sentinel.
    vm.add_method(
        sys,
        id_lut!(mappin),
        script_args_def!(lat = NIL, lon = NIL, zoom = NIL, axis = NIL, size = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat)
                .as_number()
                .unwrap_or(0.0)
                .clamp(-85.0, 85.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let z = script_value!(vm, args.zoom)
                .as_number()
                .unwrap_or(12.0)
                .clamp(3.0, 17.0) as u32;
            let axis_v = script_value!(vm, args.axis);
            let mut axis = String::new();
            vm.bx.heap.cast_to_string(axis_v, &mut axis);
            let size = script_value!(vm, args.size).as_number().unwrap_or(372.0);
            let (_, _, fx, fy) = slippy_mosaic(lat, lon, z);
            let f = if axis.trim().eq_ignore_ascii_case("y") { fy } else { fx };
            ScriptValue::from_f64(f * size)
        },
    );

    // sys.geocode(name, "field") -> a LIVE geocoding lookup (open-meteo geocoding
    // API, keyless — same free tier as the forecast API). Resolves a city/town/
    // landmark NAME to facts, so cards never rely on invented coordinates:
    //   sys.geocode("kyoto", "name")    -> "Kyoto"
    //   sys.geocode("kyoto", "country") -> "Japan"
    //   sys.geocode("kyoto", "admin1")  -> "Kyoto"
    //   sys.geocode("kyoto", "lat")     -> "35.0211"   (string, for captions)
    //   sys.geocode("kyoto", "lon")     -> "135.7539"
    //   also: "timezone", "population". Returns "—" while loading. For the
    // NUMBERS that anchor sys.maptile/mappin/places, use sys.geocodenum.
    vm.add_method(
        sys,
        id_lut!(geocode),
        script_args_def!(name = NIL, field = NIL),
        |vm, args| {
            let name_v = script_value!(vm, args.name);
            let mut name = String::new();
            vm.bx.heap.cast_to_string(name_v, &mut name);
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let url = geocode_url(&name);
            let path = match field.trim() {
                "lat" => "results.0.latitude",
                "lon" => "results.0.longitude",
                "name" => "results.0.name",
                "country" => "results.0.country",
                "admin1" => "results.0.admin1",
                "timezone" => "results.0.timezone",
                "population" => "results.0.population",
                other => return {
                    // Unknown field: pluck it verbatim under results.0 so new
                    // API fields work without a rebuild.
                    let out = match vm.host.cx_mut().script_data_fetch(&url) {
                        Some(bytes) => {
                            json_pluck(&bytes, &format!("results.0.{other}"))
                                .unwrap_or_else(|| "—".to_string())
                        }
                        None => vm.host.cx_mut().script_data_placeholder(&url),
                    };
                    vm.bx.heap.new_string_from_str(&out)
                },
            };
            let out = match vm.host.cx_mut().script_data_fetch(&url) {
                Some(bytes) => json_pluck(&bytes, path).unwrap_or_else(|| "—".to_string()),
                None => vm.host.cx_mut().script_data_placeholder(&url),
            };
            vm.bx.heap.new_string_from_str(&out)
        },
    );

    // sys.geocodenum(name, "lat"|"lon") -> the coordinate as a NUMBER, -9999
    // while the lookup loads / when the place is unknown — the anchor for a map
    // card. Guard the whole card body on it:
    //   let lat = sys.geocodenum("kyoto", "lat")
    //   let lon = sys.geocodenum("kyoto", "lon")
    //   if lat >= -9998 { <the card, using lat/lon everywhere> }
    // Shares sys.geocode's fetch (identical URL -> one request serves both).
    vm.add_method(
        sys,
        id_lut!(geocodenum),
        script_args_def!(name = NIL, field = NIL),
        |vm, args| {
            let name_v = script_value!(vm, args.name);
            let mut name = String::new();
            vm.bx.heap.cast_to_string(name_v, &mut name);
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let url = geocode_url(&name);
            let path = if field.trim() == "lon" {
                "results.0.longitude"
            } else {
                "results.0.latitude"
            };
            let n = vm
                .host
                .cx_mut()
                .script_data_fetch(&url)
                .and_then(|bytes| json_pluck(&bytes, path))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(-9999.0);
            ScriptValue::from_f64(n)
        },
    );

    // sys.route(lat1, lon1, lat2, lon2, "field") -> LIVE driving route stats
    // between two points (OSRM public demo server, keyless):
    //   "km"  -> "5.5 km"     "min" -> "10 min"
    //   "distance" -> meters raw    "duration" -> seconds raw
    // Returns "—" while loading (the demo server has no SLA — cards must
    // tolerate the dash). Chain from sys.geocodenum for both endpoints.
    vm.add_method(
        sys,
        id_lut!(route),
        script_args_def!(
            lat1 = NIL,
            lon1 = NIL,
            lat2 = NIL,
            lon2 = NIL,
            field = NIL,
            vias = NIL
        ),
        |vm, args| {
            let lat1 = script_value!(vm, args.lat1).as_number().unwrap_or(0.0);
            let lon1 = script_value!(vm, args.lon1).as_number().unwrap_or(0.0);
            let lat2 = script_value!(vm, args.lat2).as_number().unwrap_or(0.0);
            let lon2 = script_value!(vm, args.lon2).as_number().unwrap_or(0.0);
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let vias_v = script_value!(vm, args.vias);
            let mut vias = String::new();
            vm.bx.heap.cast_to_string(vias_v, &mut vias);
            let url = format!(
                "https://router.project-osrm.org/route/v1/driving/{}?overview=false",
                osrm_coords(lat1, lon1, lat2, lon2, &vias)
            );
            let out = match vm.host.cx_mut().script_data_fetch(&url) {
                Some(bytes) => match field.trim() {
                    "km" => json_pluck(&bytes, "routes.0.distance")
                        .and_then(|s| s.parse::<f64>().ok())
                        .map(|m| format!("{:.1} km", m / 1000.0))
                        .unwrap_or_else(|| "—".to_string()),
                    "min" => json_pluck(&bytes, "routes.0.duration")
                        .and_then(|s| s.parse::<f64>().ok())
                        .map(|s| format!("{:.0} min", s / 60.0))
                        .unwrap_or_else(|| "—".to_string()),
                    "duration" => json_pluck(&bytes, "routes.0.duration")
                        .unwrap_or_else(|| "—".to_string()),
                    _ => json_pluck(&bytes, "routes.0.distance")
                        .unwrap_or_else(|| "—".to_string()),
                },
                None => vm.host.cx_mut().script_data_placeholder(&url),
            };
            vm.bx.heap.new_string_from_str(&out)
        },
    );

    // sys.navroute(lat1, lon1, lat2, lon2, "field") -> LIVE turn-by-turn route
    // data (OSRM, keyless, ONE cached fetch shared with sys.navstep):
    //   "polyline" -> full route geometry (polyline5) — feed MapView.nav_polyline
    //   "km" -> "32.4 km"   "min" -> "33 min"   "" while loading.
    // Pair with MapView{ nav_mode: "3d" nav_polyline: sys.navroute(...) } for a
    // native, on-device Google-style live navigation card.
    vm.add_method(
        sys,
        id_lut!(navroute),
        script_args_def!(
            lat1 = NIL,
            lon1 = NIL,
            lat2 = NIL,
            lon2 = NIL,
            field = NIL,
            vias = NIL
        ),
        |vm, args| {
            let lat1 = script_value!(vm, args.lat1).as_number().unwrap_or(0.0);
            let lon1 = script_value!(vm, args.lon1).as_number().unwrap_or(0.0);
            let lat2 = script_value!(vm, args.lat2).as_number().unwrap_or(0.0);
            let lon2 = script_value!(vm, args.lon2).as_number().unwrap_or(0.0);
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let vias_v = script_value!(vm, args.vias);
            let mut vias = String::new();
            vm.bx.heap.cast_to_string(vias_v, &mut vias);
            let url = navroute_url(lat1, lon1, lat2, lon2, &vias);
            let out = match nav_route_cached(vm, &url) {
                Some(route) => match field.trim() {
                    "polyline" => route.polyline.clone(),
                    "km" => format!("{:.1} km", route.total_m / 1000.0),
                    "min" => format!("{:.0} min", route.total_s / 60.0),
                    // walking (~5 km/h) / cycling (~15 km/h) estimates, formatted
                    "walk" => format!("{:.0} min", (route.total_m / 1000.0) * 12.0),
                    "bike" => format!("{:.0} min", (route.total_m / 1000.0) * 4.0),
                    _ => route.polyline.clone(),
                },
                None => String::new(),
            };
            vm.bx.heap.new_string_from_str(&out)
        },
    );

    // sys.navstep(lat1, lon1, lat2, lon2, progress_m, "field") -> the live
    // turn-by-turn banner data at `progress_m` meters into the route (drive
    // progress_m from sys.simsecs: e.g. `let d = sys.simsecs(92) * 15.2`):
    //   "instr" -> "Turn left onto South Market Street"  (the UPCOMING maneuver)
    //   "dist"  -> "850 m" / "1.2 km"  counting-down distance to it
    //   "arrow" -> its glyph ("⬅")   "next_arrow" -> the one after
    //   "lane0".."lane5" -> lane glyphs for the maneuver ("" past the end)
    //   "lane0hot".."lane5hot" -> "1" if that lane is the recommended one
    //   "rem" -> "31.4 km" remaining   "remmin" -> "34" minutes remaining
    // Returns "" while the route loads. Same fetch as sys.navroute.
    vm.add_method(
        sys,
        id_lut!(navstep),
        script_args_def!(
            lat1 = NIL,
            lon1 = NIL,
            lat2 = NIL,
            lon2 = NIL,
            progress = NIL,
            field = NIL,
            vias = NIL
        ),
        |vm, args| {
            let lat1 = script_value!(vm, args.lat1).as_number().unwrap_or(0.0);
            let lon1 = script_value!(vm, args.lon1).as_number().unwrap_or(0.0);
            let lat2 = script_value!(vm, args.lat2).as_number().unwrap_or(0.0);
            let lon2 = script_value!(vm, args.lon2).as_number().unwrap_or(0.0);
            let d = script_value!(vm, args.progress).as_number().unwrap_or(0.0);
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let vias_v = script_value!(vm, args.vias);
            let mut vias = String::new();
            vm.bx.heap.cast_to_string(vias_v, &mut vias);
            let url = navroute_url(lat1, lon1, lat2, lon2, &vias);
            let out = match nav_route_cached(vm, &url) {
                Some(route) => nav_step_field(&route, d, field.trim()),
                None => String::new(),
            };
            vm.bx.heap.new_string_from_str(&out)
        },
    );

    // sys.navstepnum(lat1, lon1, lat2, lon2, progress_m, "field") -> NUMBERS for
    // layout binding: "frac" -> trip fraction 0..1 (progress bars). -1 while
    // loading.
    vm.add_method(
        sys,
        id_lut!(navstepnum),
        script_args_def!(
            lat1 = NIL,
            lon1 = NIL,
            lat2 = NIL,
            lon2 = NIL,
            progress = NIL,
            field = NIL,
            vias = NIL
        ),
        |vm, args| {
            let lat1 = script_value!(vm, args.lat1).as_number().unwrap_or(0.0);
            let lon1 = script_value!(vm, args.lon1).as_number().unwrap_or(0.0);
            let lat2 = script_value!(vm, args.lat2).as_number().unwrap_or(0.0);
            let lon2 = script_value!(vm, args.lon2).as_number().unwrap_or(0.0);
            let d = script_value!(vm, args.progress).as_number().unwrap_or(0.0);
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let vias_v = script_value!(vm, args.vias);
            let mut vias = String::new();
            vm.bx.heap.cast_to_string(vias_v, &mut vias);
            let url = navroute_url(lat1, lon1, lat2, lon2, &vias);
            let n = match nav_route_cached(vm, &url) {
                Some(route) => nav_step_num(&route, d, field.trim()),
                None => -1.0,
            };
            ScriptValue::from_f64(n)
        },
    );

    // sys.navsecs(period) -> the SAME looping clock as sys.simsecs, but it does
    // NOT arm the 1 Hz re-eval pump. For `fn tick()` cards that update named
    // widgets in place (ui.<id>.set_text) and must NEVER rebuild — e.g. the
    // live-navigation card, where a rebuild would tear the map widget down.
    vm.add_method(
        sys,
        id_lut!(navsecs),
        script_args_def!(period = NIL),
        |vm, args| {
            let period = script_value!(vm, args.period).as_number().unwrap_or(0.0);
            let secs = crate::splash::sim_clock_secs();
            let v = if period > 0.0 { secs % period } else { secs };
            ScriptValue::from_f64(v)
        },
    );

    // sys.simsecs(period) -> seconds since app start as a NUMBER, looping back
    // to 0 every `period` seconds (period <= 0 -> unbounded). THE animation
    // clock for cards: bind time windows to auto-advance content — a card that
    // calls it re-evaluates once per second (see the Splash pump):
    //   if sys.simsecs(70) >= 12 && sys.simsecs(70) < 15 { <frame/banner> }
    // Pure math — no fetch, no state, loops forever (replay for free).
    vm.add_method(
        sys,
        id_lut!(simsecs),
        script_args_def!(period = NIL),
        |vm, args| {
            let period = script_value!(vm, args.period).as_number().unwrap_or(0.0);
            let secs = crate::splash::sim_clock_secs();
            let v = if period > 0.0 { secs % period } else { secs };
            ScriptValue::from_f64(v)
        },
    );

    // sys.weather(lat, lon, "path") -> a LIVE value from the open-meteo forecast
    // API (temperature, humidity, wind, pressure, UV, 7-day highs/lows, sunrise/
    // sunset). `path` is dot-separated into the JSON; a numeric segment indexes an
    // array, e.g.:
    //   sys.weather(LAT, LON, "current.temperature_2m")     -> "27.3"
    //   sys.weather(LAT, LON, "current.relative_humidity_2m")-> "54"
    //   sys.weather(LAT, LON, "daily.temperature_2m_max.0")  -> "29.1"  (today)
    //   sys.weather(LAT, LON, "daily.sunrise.0")             -> "05:52" (HH:MM)
    // All fields for a given lat/lon share ONE cached fetch. Returns "—" while the
    // (async) request loads; the card auto-redraws when data arrives, so the value
    // fills in. THE LLM MUST CALL THIS FOR EVERY WEATHER NUMBER — never hardcode.
    vm.add_method(
        sys,
        id_lut!(weather),
        script_args_def!(lat = NIL, lon = NIL, path = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let path_value = script_value!(vm, args.path);
            let mut path = String::new();
            vm.bx.heap.cast_to_string(path_value, &mut path);
            let url = format!(
                "https://api.open-meteo.com/v1/forecast?latitude={lat:.4}&longitude={lon:.4}\
&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,surface_pressure,is_day\
&daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,uv_index_max,precipitation_probability_max\
&timezone=auto&forecast_days=7"
            );
            let value = match vm.host.cx_mut().script_data_fetch(&url) {
                // Union of two changes: round the plucked value (whole-degree
                // display), and show the terminal-failure placeholder rather than
                // a bare "—" when the fetch itself gives up (#17). The em dash
                // stays for the case where the fetch SUCCEEDED but the path is
                // absent, which is a card bug, not a network one.
                Some(bytes) => json_pluck(&bytes, path.trim())
                    .map(|v| round_display(path.trim(), v))
                    .unwrap_or_else(|| "—".to_string()),
                None => vm.host.cx_mut().script_data_placeholder(&url),
            };
            vm.bx.heap.new_string_from_str(&value)
        },
    );

    // sys.airquality(lat, lon, "path") -> a LIVE value from the open-meteo air-
    // quality API. e.g. sys.airquality(LAT, LON, "current.us_aqi") -> "42",
    // "current.pm2_5", "current.pm10", "current.ozone". Same "—"/redraw semantics
    // as sys.weather. (The AQI *map tile* is sys.airmap; this is the number.)
    vm.add_method(
        sys,
        id_lut!(airquality),
        script_args_def!(lat = NIL, lon = NIL, path = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let path_value = script_value!(vm, args.path);
            let mut path = String::new();
            vm.bx.heap.cast_to_string(path_value, &mut path);
            let url = format!(
                "https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat:.4}&longitude={lon:.4}\
&current=us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide&timezone=auto"
            );
            let value = match vm.host.cx_mut().script_data_fetch(&url) {
                Some(bytes) => json_pluck(&bytes, path.trim()).unwrap_or_else(|| "—".to_string()),
                None => vm.host.cx_mut().script_data_placeholder(&url),
            };
            vm.bx.heap.new_string_from_str(&value)
        },
    );

    // sys.weathernum(lat, lon, "path") / sys.aqinum(lat, lon, "path") -> the SAME
    // live open-meteo values as sys.weather / sys.airquality, but as a NUMBER, so
    // script conditions can branch on LIVE data — the enabling primitive for
    // COMPOSED cards (pick activities by temperature/precipitation, gate
    // "go outside" on AQI, switch content on is_day). Returns -9999 while the
    // fetch loads or when the path is absent/non-numeric — guard with
    // `>= -9998`; the card re-evaluates when the fetch lands (same redraw
    // semantics as the string helpers). Shares the string helpers' fetch
    // (identical URL -> one request serves both).
    vm.add_method(
        sys,
        id_lut!(weathernum),
        script_args_def!(lat = NIL, lon = NIL, path = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let path_value = script_value!(vm, args.path);
            let mut path = String::new();
            vm.bx.heap.cast_to_string(path_value, &mut path);
            let url = format!(
                "https://api.open-meteo.com/v1/forecast?latitude={lat:.4}&longitude={lon:.4}\
&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,surface_pressure,is_day\
&daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,uv_index_max,precipitation_probability_max\
&timezone=auto&forecast_days=7"
            );
            let n = vm
                .host
                .cx_mut()
                .script_data_fetch(&url)
                .and_then(|bytes| json_pluck(&bytes, path.trim()))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(-9999.0);
            ScriptValue::from_f64(n)
        },
    );
    vm.add_method(
        sys,
        id_lut!(aqinum),
        script_args_def!(lat = NIL, lon = NIL, path = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let path_value = script_value!(vm, args.path);
            let mut path = String::new();
            vm.bx.heap.cast_to_string(path_value, &mut path);
            let url = format!(
                "https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat:.4}&longitude={lon:.4}\
&current=us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide&timezone=auto"
            );
            let n = vm
                .host
                .cx_mut()
                .script_data_fetch(&url)
                .and_then(|bytes| json_pluck(&bytes, path.trim()))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(-9999.0);
            ScriptValue::from_f64(n)
        },
    );

    // sys.dayname(lat, lon, n, locale) -> the weekday LABEL for forecast row n.
    //   sys.dayname(LAT, LON, 0, "en") -> "Today"
    //   sys.dayname(LAT, LON, 1, "en") -> "Thu"
    //   sys.dayname(LAT, LON, 1, "zh") -> "周四"
    //
    // A card must NEVER write weekday names as literal strings, for two reasons
    // that both bit us before this existed:
    //
    //   * The generating model does not reliably know the date. On Wed
    //     2026-07-29 it emitted "Today, Wed, Thu, …" — repeating today as
    //     tomorrow, so every row after the first was mislabelled. Nobody
    //     notices, because a wrong weekday looks exactly like a right one.
    //   * Cards are PERSISTED and re-served (a2app_cards/), so a literal is
    //     stale the next morning even when it was correct when generated.
    //
    // The date comes from `daily.time.n` in the SAME cached forecast the
    // temperatures come from, so the labels are local to the FORECAST'S place —
    // not to wherever the phone happens to be. Until that fetch lands we fall
    // back to the device clock, which is right unless the place is across a date
    // boundary, and self-corrects on the redraw when the real date arrives.
    vm.add_method(
        sys,
        id_lut!(dayname),
        script_args_def!(lat = NIL, lon = NIL, n = NIL, locale = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let n = script_value!(vm, args.n).as_number().unwrap_or(0.0).max(0.0) as usize;
            let loc_v = script_value!(vm, args.locale);
            let mut loc = String::new();
            vm.bx.heap.cast_to_string(loc_v, &mut loc);
            let zh = loc.trim().to_ascii_lowercase().starts_with("zh");

            if n == 0 {
                let s = if zh { "今天" } else { "Today" };
                return vm.bx.heap.new_string_from_str(s);
            }

            let url = format!(
                "https://api.open-meteo.com/v1/forecast?latitude={lat:.4}&longitude={lon:.4}\
&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,surface_pressure,is_day\
&daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,uv_index_max,precipitation_probability_max\
&timezone=auto&forecast_days=7"
            );
            // Authoritative: the forecast's own local date for that row.
            let from_api = vm.host.cx_mut().script_data_fetch(&url).and_then(|bytes| {
                let date = json_pluck(&bytes, &format!("daily.time.{n}"))?;
                let mut it = date.trim().split('-');
                let y = it.next()?.parse::<i64>().ok()?;
                let m = it.next()?.parse::<u64>().ok()?;
                let d = it.next()?.parse::<u64>().ok()?;
                Some(weekday_from_days(days_from_civil(y, m, d)))
            });
            // Graceful: the device's own day, offset by n.
            let wd = from_api.unwrap_or_else(|| {
                weekday_from_days((now_unix_secs() / 86_400) as i64 + n as i64)
            });
            let s = if zh { DAY_ZH[wd] } else { DAY_EN[wd] };
            vm.bx.heap.new_string_from_str(s)
        },
    );

    // sys.weekmin(lat, lon) / sys.weekmax(lat, lon) -> the LOWEST low and HIGHEST
    // high across the 7-day forecast, for a TempBar's draw_bg.wmin / draw_bg.wmax.
    //
    // These exist because the card CANNOT know them. Every temperature on the card
    // is a live sys.weather call, so a generated card asking the model to name the
    // week's range is asking it to guess at numbers it has never seen — and it
    // guesses badly ("10 to 35" for a 27-39C week), which clamps every high to the
    // red end and pushes the whole week into the top of the scale.
    //
    // Shares the cached forecast fetch, so neither costs an extra request.
    // Falls back to a plausible temperate range if the fetch is still in flight,
    // rather than to 0/0 — a zero span would collapse every bar to one colour.
    vm.add_method(
        sys,
        id_lut!(weekmin),
        script_args_def!(lat = NIL, lon = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let n = week_extreme(vm, lat, lon, "daily.temperature_2m_min", false).unwrap_or(0.0);
            ScriptValue::from_f64(n)
        },
    );
    vm.add_method(
        sys,
        id_lut!(weekmax),
        script_args_def!(lat = NIL, lon = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let n = week_extreme(vm, lat, lon, "daily.temperature_2m_max", true).unwrap_or(30.0);
            ScriptValue::from_f64(n)
        },
    );

    // sys.moonphase("field") -> the CURRENT 月相 (moon phase), computed from the
    // device clock — no network, so it never shows a "—" placeholder.
    //   "name"         -> "Waxing Gibbous"  (one of the eight principal phases)
    //   "name_zh"      -> "盈凸月"           (the same phase, 八相 names)
    //   "illumination" -> "87"   (percent of the disc lit, 0-100)
    //   "phase"        -> "0.62" (position in the cycle, 0 new .. 0.5 full .. 1)
    // For the MoonPhase WIDGET uniform use sys.moonnum("phase"), which returns a
    // number: draw_bg.phase needs a float, and a string silently reads as 0.
    vm.add_method(
        sys,
        id_lut!(moonphase),
        script_args_def!(field = NIL),
        |vm, args| {
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let f = moon_phase_fraction();
            let value = match field.trim().to_ascii_lowercase().as_str() {
                "name" => moon_phase_name(f).to_string(),
                "name_zh" | "name_cn" => moon_phase_name_zh(f).to_string(),
                // Illuminated fraction is (1 - cos(2*pi*phase)) / 2: 0 at new,
                // 1 at full, and correctly non-linear in between.
                "illumination" | "illum" => {
                    let lit = (1.0 - (std::f64::consts::TAU * f).cos()) * 50.0;
                    format!("{}", lit.round() as i64)
                }
                _ => format!("{f:.2}"),
            };
            vm.bx.heap.new_string_from_str(&value)
        },
    );
    vm.add_method(
        sys,
        id_lut!(moonnum),
        script_args_def!(field = NIL),
        |vm, args| {
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let f = moon_phase_fraction();
            let n = match field.trim().to_ascii_lowercase().as_str() {
                "illumination" | "illum" => (1.0 - (std::f64::consts::TAU * f).cos()) * 50.0,
                _ => f,
            };
            ScriptValue::from_f64(n)
        },
    );

    // sys.daylight(lat, lon) -> fraction of DAYLIGHT elapsed: 0 at sunrise, 1 at
    // sunset, for the SunArc widget's draw_bg.progress. Before sunrise it is
    // negative and after sunset greater than 1, which the widget reads as night
    // and dims the sun rather than hiding it.
    //
    // Shares the cached sys.weather forecast fetch, so it costs no extra request.
    // Returns 0.5 while that fetch is in flight — the arc then shows a midday sun
    // for one redraw, which reads better than a sun jammed at the horizon.
    vm.add_method(
        sys,
        id_lut!(daylight),
        script_args_def!(lat = NIL, lon = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let url = format!(
                "https://api.open-meteo.com/v1/forecast?latitude={lat:.4}&longitude={lon:.4}\
&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,surface_pressure,is_day\
&daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,uv_index_max,precipitation_probability_max\
&timezone=auto&forecast_days=7"
            );
            let progress = vm
                .host
                .cx_mut()
                .script_data_fetch(&url)
                .and_then(|bytes| {
                    let rise = hhmm_to_minutes(&json_pluck(&bytes, "daily.sunrise.0")?)?;
                    let set = hhmm_to_minutes(&json_pluck(&bytes, "daily.sunset.0")?)?;

                    // "Now" comes from the DEVICE CLOCK shifted by the city's UTC
                    // offset — NOT from `current.time` in the response. The
                    // forecast fetch is cached, so its timestamp is frozen at
                    // whenever the card last fetched; using it parks the sun
                    // wherever it stood then and drifts further all day. Sunrise,
                    // sunset and the offset are all stable for the day, so only
                    // the current instant must come from a live source.
                    let offset = json_pluck(&bytes, "utc_offset_seconds")?
                        .parse::<f64>()
                        .ok()?;
                    let local = (now_unix_secs() as f64 + offset).rem_euclid(86_400.0);
                    let now = local / 60.0;

                    let span = set - rise;
                    if span <= 0.0 {
                        // Polar day or night — no meaningful fraction.
                        return None;
                    }
                    Some((now - rise) / span)
                })
                .unwrap_or(0.5);
            ScriptValue::from_f64(progress)
        },
    );

    // sys.aqigrid(lat, lon, span, idx) -> the US-AQI at one cell of a 4x4 grid
    // covering `span` degrees centred on (lat, lon), for the AqiContour widget's
    // a0..a15 uniforms. `idx` is 0..15, row-major with the NORTH row first.
    //
    // All sixteen cells come from ONE multi-location open-meteo request, which
    // that API serves by accepting comma-separated coordinates and returning an
    // array of results. Because the URL is identical for every idx, the sixteen
    // card-side calls collapse to a single cached fetch.
    vm.add_method(
        sys,
        id_lut!(aqigrid),
        script_args_def!(lat = NIL, lon = NIL, span = NIL, idx = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let span = script_value!(vm, args.span).as_number().unwrap_or(1.5);
            let idx = script_value!(vm, args.idx).as_number().unwrap_or(0.0) as usize;
            let idx = idx.min(15);

            // Row 0 is the NORTH edge, so latitude DECREASES with the row index —
            // matching how the shader walks self.pos.y downward.
            let step = span / 3.0;
            let mut lats = Vec::with_capacity(16);
            let mut lons = Vec::with_capacity(16);
            for r in 0..4 {
                for c in 0..4 {
                    lats.push(format!("{:.4}", lat + span / 2.0 - r as f64 * step));
                    lons.push(format!("{:.4}", lon - span / 2.0 + c as f64 * step));
                }
            }
            let url = format!(
                "https://air-quality-api.open-meteo.com/v1/air-quality?latitude={}&longitude={}\
&current=us_aqi&timezone=auto",
                lats.join(","),
                lons.join(",")
            );
            let n = vm
                .host
                .cx_mut()
                .script_data_fetch(&url)
                .and_then(|bytes| json_pluck(&bytes, &format!("{idx}.current.us_aqi")))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);
            ScriptValue::from_f64(n)
        },
    );

    // sys.stock("AAPL", "key") -> a LIVE value from Yahoo Finance for that ticker.
    // Same "—"/redraw semantics as sys.weather. `key` (case-insensitive):
    //   price | prev | high | low | open | currency | name | symbol
    //   change    -> price − previous close, signed, e.g. "+1.99"
    //   changepct -> percent change, signed, e.g. "+0.63%"
    // e.g. sys.stock("AAPL", "price"), sys.stock("TSLA", "changepct").
    vm.add_method(
        sys,
        id_lut!(stock),
        script_args_def!(symbol = NIL, field = NIL),
        |vm, args| {
            let sym_v = script_value!(vm, args.symbol);
            let mut symbol = String::new();
            vm.bx.heap.cast_to_string(sym_v, &mut symbol);
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let sym = sanitize_ticker(&symbol);
            let url = format!(
                "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=1d"
            );
            let m = |k: &str| format!("chart.result.0.meta.{k}");
            let out = match vm.host.cx_mut().script_data_fetch(&url) {
                None => vm.host.cx_mut().script_data_placeholder(&url),
                Some(bytes) => {
                    let num = |k: &str| json_pluck(&bytes, &m(k)).and_then(|s| s.parse::<f64>().ok());
                    // Monetary fields formatted to a consistent 2 decimals (Yahoo
                    // returns e.g. 201.5, which otherwise breaks the visual rhythm).
                    let money = |k: &str| num(k).map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into());
                    match field.trim().to_ascii_lowercase().as_str() {
                        "change" => match (num("regularMarketPrice"), num("chartPreviousClose")) {
                            (Some(p), Some(c)) => format!("{:+.2}", p - c),
                            _ => "—".to_string(),
                        },
                        "changepct" | "changepercent" => {
                            match (num("regularMarketPrice"), num("chartPreviousClose")) {
                                (Some(p), Some(c)) if c != 0.0 => format!("{:+.2}%", (p - c) / c * 100.0),
                                _ => "—".to_string(),
                            }
                        }
                        "price" => money("regularMarketPrice"),
                        "prev" | "prevclose" => money("chartPreviousClose"),
                        "high" => money("regularMarketDayHigh"),
                        "low" => money("regularMarketDayLow"),
                        "open" => money("regularMarketOpen"),
                        "currency" => json_pluck(&bytes, &m("currency")).unwrap_or_else(|| "—".into()),
                        "name" => json_pluck(&bytes, &m("longName"))
                            .or_else(|| json_pluck(&bytes, &m("shortName")))
                            .unwrap_or_else(|| "—".into()),
                        "symbol" => json_pluck(&bytes, &m("symbol")).unwrap_or_else(|| "—".into()),
                        "exchange" => json_pluck(&bytes, &m("fullExchangeName")).unwrap_or_else(|| "—".into()),
                        "52wh" | "yearhigh" => money("fiftyTwoWeekHigh"),
                        "52wl" | "yearlow" => money("fiftyTwoWeekLow"),
                        "vol" | "volume" => match num("regularMarketVolume") {
                            Some(v) if v >= 1e9 => format!("{:.2}B", v / 1e9),
                            Some(v) if v >= 1e6 => format!("{:.1}M", v / 1e6),
                            Some(v) if v >= 1e3 => format!("{:.1}K", v / 1e3),
                            Some(v) => format!("{v:.0}"),
                            None => "—".to_string(),
                        },
                        // Day-range position 0..100 (where price sits low→high), for a range bar.
                        "rangepct" => match (
                            num("regularMarketPrice"),
                            num("regularMarketDayLow"),
                            num("regularMarketDayHigh"),
                        ) {
                            (Some(p), Some(lo), Some(hi)) if hi > lo => {
                                format!("{:.0}", (((p - lo) / (hi - lo)) * 100.0).clamp(0.0, 100.0))
                            }
                            _ => "50".to_string(),
                        },
                        other => json_pluck(&bytes, other).unwrap_or_else(|| "—".into()),
                    }
                }
            };
            vm.bx.heap.new_string_from_str(&out)
        },
    );

    // sys.stockbar("AAPL", index, count) -> the HEIGHT (dp, a NUMBER) of bar
    // `index` of `count` in a price sparkline (Yahoo closes, normalized so the
    // range's low→~8dp and high→~158dp). Draw the chart as a bottom-aligned
    // `flow: Right` row of thin `SolidView{ height: sys.stockbar("AAPL", N,
    // COUNT) }` bars (an area/line of the price path). Returns a small height
    // while the async fetch loads, then the card re-evaluates.
    // Optional 5th arg selects the series range: "1d" (default, 5-minute
    // intraday), "1w", "1m", "6m", "1y" — empty/unknown tokens fall back to
    // "1d" so an unset `{{state.range}}` still draws the intraday chart. The
    // close series is resampled to `count` bars, so the SAME bar row serves
    // every range. One fetch per symbol×range (URL-deduped).
    vm.add_method(
        sys,
        id_lut!(stockbar),
        script_args_def!(symbol = NIL, index = NIL, count = NIL, maxh = NIL, range = NIL),
        |vm, args| {
            let sym_v = script_value!(vm, args.symbol);
            let mut symbol = String::new();
            vm.bx.heap.cast_to_string(sym_v, &mut symbol);
            let index = script_value!(vm, args.index).as_number().unwrap_or(0.0).max(0.0) as usize;
            let count = (script_value!(vm, args.count).as_number().unwrap_or(40.0) as usize).max(2);
            // Optional 4th arg: the chart's pixel height, so the area fills it
            // exactly with no peak clipping. Defaults to 150 (legacy behavior).
            let maxh = script_value!(vm, args.maxh).as_number().filter(|v| v.is_finite() && *v > 8.0 && *v < 10_000.0).unwrap_or(150.0);
            let range_v = script_value!(vm, args.range);
            let mut range = String::new();
            vm.bx.heap.cast_to_string(range_v, &mut range);
            let url = yahoo_chart_url(&symbol, &range);
            let h = match vm.host.cx_mut().script_data_fetch(&url) {
                Some(bytes) => stock_bar_height(&bytes, index, count, maxh),
                None => 6.0,
            };
            ScriptValue::from_f64(h)
        },
    );

    // sys.stockrange("<TICKER>", "<RANGE>", "field") -> a range-aware scalar for
    // the SAME close series the chart bars draw (same URL → the one fetch is
    // shared). `sys.stock("high"/"low"/"change"/"changepct")` is DAY-only, so a
    // card whose chart switches range uses THIS for the Y-axis labels and the
    // change line. Range tokens as in sys.stockbar ("1d" default). Fields
    // (case-insensitive): "high" | "low" (range extremes, 2dp), "change" |
    // "changepct" (signed first→last close over the range), "up" ("1"/"0",
    // last >= first — for green/red styling). "—" while loading / on error.
    vm.add_method(
        sys,
        id_lut!(stockrange),
        script_args_def!(symbol = NIL, range = NIL, field = NIL),
        |vm, args| {
            let sym_v = script_value!(vm, args.symbol);
            let mut symbol = String::new();
            vm.bx.heap.cast_to_string(sym_v, &mut symbol);
            let range_v = script_value!(vm, args.range);
            let mut range = String::new();
            vm.bx.heap.cast_to_string(range_v, &mut range);
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let url = yahoo_chart_url(&symbol, &range);
            let out = match vm.host.cx_mut().script_data_fetch(&url) {
                None => vm.host.cx_mut().script_data_placeholder(&url),
                Some(bytes) => stock_range_field(&bytes, field.trim()),
            };
            vm.bx.heap.new_string_from_str(&out)
        },
    );

    // sys.movers(index, "field") -> the LIVE "top gainers" list (Yahoo day_gainers
    // screener, no auth). index 0 = the biggest % gainer today, up to 9. Fields
    // (case-insensitive): symbol, name, price, change (signed), changepct (signed %),
    // high, low, prev, open, 52wh, 52wl, vol, marketcap, currency, exchange.
    // ONE fetch (deduped by URL) serves all 10 rows × all fields. Use for a top-10
    // movers LIST card; tap a row to open the per-ticker detail (sys.stock/stockbar).
    vm.add_method(
        sys,
        id_lut!(movers),
        script_args_def!(index = NIL, field = NIL),
        |vm, args| {
            let index = script_value!(vm, args.index).as_number().unwrap_or(0.0).max(0.0) as i64;
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_gainers&count=10".to_string();
            let base = format!("finance.result.0.quotes.{index}");
            let out = match vm.host.cx_mut().script_data_fetch(&url) {
                None => vm.host.cx_mut().script_data_placeholder(&url),
                Some(bytes) => {
                    let raw = |k: &str| json_pluck(&bytes, &format!("{base}.{k}"));
                    let num = |k: &str| raw(k).and_then(|s| s.parse::<f64>().ok());
                    let money = |k: &str| num(k).map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into());
                    match field.trim().to_ascii_lowercase().as_str() {
                        "symbol" => raw("symbol").unwrap_or_else(|| "—".into()),
                        "name" => raw("displayName")
                            .or_else(|| raw("shortName"))
                            .or_else(|| raw("longName"))
                            .unwrap_or_else(|| "—".into()),
                        "price" => money("regularMarketPrice"),
                        "change" => num("regularMarketChange").map(|v| format!("{v:+.2}")).unwrap_or_else(|| "—".into()),
                        "changepct" | "changepercent" => num("regularMarketChangePercent").map(|v| format!("{v:+.2}%")).unwrap_or_else(|| "—".into()),
                        "high" => money("regularMarketDayHigh"),
                        "low" => money("regularMarketDayLow"),
                        "prev" | "prevclose" => money("regularMarketPreviousClose"),
                        "open" => money("regularMarketOpen"),
                        "52wh" | "yearhigh" => money("fiftyTwoWeekHigh"),
                        "52wl" | "yearlow" => money("fiftyTwoWeekLow"),
                        "currency" => raw("currency").unwrap_or_else(|| "—".into()),
                        "exchange" => raw("fullExchangeName").unwrap_or_else(|| "—".into()),
                        "vol" | "volume" => match num("regularMarketVolume") {
                            Some(v) if v >= 1e9 => format!("{:.2}B", v / 1e9),
                            Some(v) if v >= 1e6 => format!("{:.1}M", v / 1e6),
                            Some(v) if v >= 1e3 => format!("{:.1}K", v / 1e3),
                            Some(v) => format!("{v:.0}"),
                            None => "—".to_string(),
                        },
                        "marketcap" | "mktcap" | "cap" => match num("marketCap") {
                            Some(v) if v >= 1e12 => format!("{:.2}T", v / 1e12),
                            Some(v) if v >= 1e9 => format!("{:.1}B", v / 1e9),
                            Some(v) if v >= 1e6 => format!("{:.0}M", v / 1e6),
                            Some(v) => format!("{v:.0}"),
                            None => "—".to_string(),
                        },
                        other => raw(other).unwrap_or_else(|| "—".into()),
                    }
                }
            };
            vm.bx.heap.new_string_from_str(&out)
        },
    );

    // sys.news(index, "key") -> a LIVE Hacker News front-page story (index 0..).
    // Same "—"/redraw semantics. `key` (case-insensitive):
    //   title | url | author | points | comments
    // e.g. sys.news(0, "title"), sys.news(0, "points"), sys.news(1, "title").
    vm.add_method(
        sys,
        id_lut!(news),
        script_args_def!(index = NIL, field = NIL),
        |vm, args| {
            let idx = script_value!(vm, args.index).as_number().unwrap_or(0.0).max(0.0) as i64;
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let key = match field.trim().to_ascii_lowercase().as_str() {
                "title" => "title",
                "url" => "url",
                "author" | "by" => "author",
                "points" | "score" => "points",
                "comments" | "num_comments" => "num_comments",
                _ => "title",
            };
            let url =
                "https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage=12".to_string();
            let path = format!("hits.{idx}.{key}");
            let out = match vm.host.cx_mut().script_data_fetch(&url) {
                Some(bytes) => json_pluck(&bytes, &path).unwrap_or_else(|| "—".to_string()),
                None => vm.host.cx_mut().script_data_placeholder(&url),
            };
            vm.bx.heap.new_string_from_str(&out)
        },
    );

    // sys.places(lat, lon, "category", index, "field") -> a REAL nearby venue
    // from OpenStreetMap (Overpass API, keyless): row `index` (0 = nearest) of
    // the named places within 4 km, sorted by distance. THE LLM MUST CALL THIS
    // FOR EVERY VENUE NAME/DISTANCE — an invented "Riverside Park" that isn't
    // actually there destroys trust in the whole card; these are real mapped
    // venues. Category tokens (case-insensitive): park, garden, trail, museum,
    // cafe, cinema, gym, library, pool, viewpoint, playground — unknown tokens
    // fall back to park. Fields (case-insensitive):
    //   name     -> "Ryland Park"
    //   distance -> "0.7 km"   (from the request point, one decimal)
    //   lat|lon  -> "37.3423"  (4 decimals — chain into sys.basemap/sys.photo)
    //   count    -> total places found (ignores `index`)
    // Returns "—" while the (async) fetch loads or when index/field is out of
    // range; the card re-evaluates when data lands (same redraw semantics as
    // sys.weather). Unnamed OSM elements are skipped and duplicate names
    // deduped (one park is often mapped as several ways), so every index is a
    // distinct, nameable venue. ONE fetch (URL-deduped) serves all rows ×
    // fields of a list card.
    vm.add_method(
        sys,
        id_lut!(places),
        script_args_def!(lat = NIL, lon = NIL, category = NIL, index = NIL, field = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let cat_v = script_value!(vm, args.category);
            let mut category = String::new();
            vm.bx.heap.cast_to_string(cat_v, &mut category);
            let index = script_value!(vm, args.index).as_number().unwrap_or(0.0).max(0.0) as usize;
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let url = overpass_places_url(lat, lon, &category);
            let out = match vm.host.cx_mut().script_data_fetch(&url) {
                None => vm.host.cx_mut().script_data_placeholder(&url),
                Some(bytes) => places_field(&bytes, lat, lon, index, field.trim()),
            };
            vm.bx.heap.new_string_from_str(&out)
        },
    );

    // sys.placesnum(lat, lon, "category") -> how many places sys.places would
    // list, as a NUMBER, so script conditions can branch on LIVE availability
    // BEFORE laying out rows — skip the "parks" section when none are mapped,
    // cap a list at the real row count, fall back to another category.
    // Returns -9999 while the fetch loads / on a bad response — guard with
    // `>= 0` (the same sentinel convention as sys.weathernum); a genuine
    // "nothing nearby" is 0. Shares sys.places' fetch (identical URL -> ONE
    // request serves both helpers).
    vm.add_method(
        sys,
        id_lut!(placesnum),
        script_args_def!(lat = NIL, lon = NIL, category = NIL),
        |vm, args| {
            let lat = script_value!(vm, args.lat).as_number().unwrap_or(0.0);
            let lon = script_value!(vm, args.lon).as_number().unwrap_or(0.0);
            let cat_v = script_value!(vm, args.category);
            let mut category = String::new();
            vm.bx.heap.cast_to_string(cat_v, &mut category);
            let url = overpass_places_url(lat, lon, &category);
            let n = match vm.host.cx_mut().script_data_fetch(&url) {
                None => -9999.0,
                Some(bytes) => match places_parse(&bytes, lat, lon) {
                    Some(places) => places.len() as f64,
                    None => -9999.0,
                },
            };
            ScriptValue::from_f64(n)
        },
    );

    // sys.search("<free text>", index, "name"|"label"|"lat"|"lon") -> the i-th
    // RESULT of a free-text place/POI/address search (Photon, keyless). This is
    // the Google-Maps "search a location" step: bind rows to it to show tappable
    // search results, then a row's tap writes the picked lat/lon/name into state
    // and the card routes there. "" while loading / past the last hit.
    vm.add_method(
        sys,
        id_lut!(search),
        script_args_def!(query = NIL, index = NIL, field = NIL),
        |vm, args| {
            let q_v = script_value!(vm, args.query);
            let mut query = String::new();
            vm.bx.heap.cast_to_string(q_v, &mut query);
            let index = script_value!(vm, args.index).as_number().unwrap_or(0.0).max(0.0) as usize;
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let url = search_url(query.trim());
            let out = match vm.host.cx_mut().script_data_fetch(&url) {
                None => String::new(),
                Some(bytes) => search_field(&bytes, index, field.trim()),
            };
            vm.bx.heap.new_string_from_str(&out)
        },
    );

    // sys.searchnum("<free text>", index, "count"|"lat"|"lon") -> NUMBERS for
    // the search hits: "count" (index ignored) = how many hits, to guard/cap the
    // results list; "lat"/"lon" = hit `index`'s coordinate as a NUMBER, to feed
    // straight into sys.navroute/sys.route (sys.search returns strings, for
    // display). -9999 while loading / bad response / out of range (guard `>= 0`;
    // 0 hits = 0). Shares sys.search's fetch (identical URL -> ONE request).
    vm.add_method(
        sys,
        id_lut!(searchnum),
        script_args_def!(query = NIL, index = NIL, field = NIL),
        |vm, args| {
            let q_v = script_value!(vm, args.query);
            let mut query = String::new();
            vm.bx.heap.cast_to_string(q_v, &mut query);
            let index = script_value!(vm, args.index).as_number().unwrap_or(0.0).max(0.0) as usize;
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let url = search_url(query.trim());
            let n = match vm.host.cx_mut().script_data_fetch(&url) {
                None => -9999.0,
                Some(bytes) => match search_parse(&bytes) {
                    None => -9999.0,
                    Some(hits) => match field.trim().to_ascii_lowercase().as_str() {
                        "lat" => hits.get(index).map(|h| h.lat).unwrap_or(-9999.0),
                        "lon" => hits.get(index).map(|h| h.lon).unwrap_or(-9999.0),
                        _ => hits.len() as f64, // "count" / default
                    },
                },
            };
            ScriptValue::from_f64(n)
        },
    );

    // sys.navroutenum(lat1, lon1, lat2, lon2, "km"|"min", vias) -> the route's
    // distance (km) or driving duration (min) as a NUMBER, so a card can derive
    // other travel modes (walk ≈ km*12 min, bike ≈ km*4 min) and arrival math.
    // Shares sys.navroute's cached fetch. -1 while loading.
    vm.add_method(
        sys,
        id_lut!(navroutenum),
        script_args_def!(
            lat1 = NIL,
            lon1 = NIL,
            lat2 = NIL,
            lon2 = NIL,
            field = NIL,
            vias = NIL
        ),
        |vm, args| {
            let lat1 = script_value!(vm, args.lat1).as_number().unwrap_or(0.0);
            let lon1 = script_value!(vm, args.lon1).as_number().unwrap_or(0.0);
            let lat2 = script_value!(vm, args.lat2).as_number().unwrap_or(0.0);
            let lon2 = script_value!(vm, args.lon2).as_number().unwrap_or(0.0);
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let vias_v = script_value!(vm, args.vias);
            let mut vias = String::new();
            vm.bx.heap.cast_to_string(vias_v, &mut vias);
            let url = navroute_url(lat1, lon1, lat2, lon2, &vias);
            let n = match nav_route_cached(vm, &url) {
                Some(route) => match field.trim() {
                    "min" => route.total_s / 60.0,
                    _ => route.total_m / 1000.0,
                },
                None => -1.0,
            };
            ScriptValue::from_f64(n)
        },
    );

    // sys.coord("lat,lon|name", field) -> a field of a place a card stored in
    // one scalar state key (a picked search result / waypoint / origin), so it
    // survives across screens and feeds routing. Format: "lat,lon" or
    // "lat,lon|Display Name".
    //   "lat"/"lon" -> NUMBER (feeds sys.navroute/sys.route lat/lon args)
    //   "latlon"    -> clean "lat,lon" STRING (feeds the navroute `vias` arg)
    //   "name"      -> the display name STRING (after the '|'), or ""
    // lat/lon are -9999 when unparseable (guard `>= -900`).
    vm.add_method(
        sys,
        id_lut!(coord),
        script_args_def!(s = NIL, field = NIL),
        |vm, args| {
            let s_v = script_value!(vm, args.s);
            let mut s = String::new();
            vm.bx.heap.cast_to_string(s_v, &mut s);
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let (coords, name) = match s.trim().split_once('|') {
                Some((c, n)) => (c.trim(), n.trim()),
                None => (s.trim(), ""),
            };
            let mut it = coords.split(',');
            let lat = it.next().and_then(|p| p.trim().parse::<f64>().ok());
            let lon = it.next().and_then(|p| p.trim().parse::<f64>().ok());
            match field.trim().to_ascii_lowercase().as_str() {
                "name" => vm.bx.heap.new_string_from_str(name),
                "latlon" => {
                    let out = match (lat, lon) {
                        (Some(a), Some(o)) => format!("{a},{o}"),
                        _ => String::new(),
                    };
                    vm.bx.heap.new_string_from_str(&out)
                }
                "lon" => ScriptValue::from_f64(lon.unwrap_or(-9999.0)),
                _ => ScriptValue::from_f64(lat.unwrap_or(-9999.0)),
            }
        },
    );

    // sys.gps("lat"|"lon"|"acc"|"ok") -> the device's last-known GPS fix, read
    // SYNCHRONOUSLY from the platform global (NO network fetch — so, unlike
    // sys.search/navroute, it must NOT gate the card via body_binds_live_data).
    // lat/lon/acc are numbers, -9999 when there is no fix yet; "ok" is 1 when a
    // fix exists else 0. Cards guard with `sys.gps("ok") >= 1` before trusting
    // lat/lon (same sentinel idiom as sys.coord). Fed by the Android
    // LocationListener through JNI onLocation -> makepad_platform::gps.
    vm.add_method(
        sys,
        id_lut!(gps),
        script_args_def!(field = NIL),
        |vm, args| {
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let fix = crate::makepad_draw::makepad_platform::gps::last_gps_fix();
            match field.trim().to_ascii_lowercase().as_str() {
                "ok" => ScriptValue::from_f64(if fix.is_some() { 1.0 } else { 0.0 }),
                "lon" => ScriptValue::from_f64(fix.map(|f| f.lon).unwrap_or(-9999.0)),
                "acc" => ScriptValue::from_f64(fix.map(|f| f.acc as f64).unwrap_or(-9999.0)),
                _ => ScriptValue::from_f64(fix.map(|f| f.lat).unwrap_or(-9999.0)),
            }
        },
    );

    vm.set_injected_global(id!(sys), sys.into());
}

/// True if a Splash body calls any live-data helper (sys.weather/airquality/
/// stock/news/movers/places). Such cards must re-evaluate when their async
/// fetch lands (the value is baked into a Label at eval time), so we arm the
/// frame pump + watch the data-fetch epoch for them. Keep in sync with the
/// data `sys.*` helpers. (Substring matches also cover the -num variants:
/// "sys.weather" matches sys.weathernum, "sys.places" matches sys.placesnum.)
fn body_binds_live_data(body: &str) -> bool {
    body.contains("sys.weather")
        || body.contains("sys.airquality")
        // `sys.aqinum` shares no prefix with `sys.airquality` — without its own
        // check an aqinum-only card would never re-evaluate when its fetch lands.
        || body.contains("sys.aqinum")
        || body.contains("sys.stock")
        || body.contains("sys.news")
        || body.contains("sys.movers")
        || body.contains("sys.places")
        // covers sys.search + sys.searchnum — the search-results card must
        // re-evaluate once the free-text search fetch lands
        || body.contains("sys.search")
        // substring covers sys.geocodenum too (same trick as weather/weathernum)
        || body.contains("sys.geocode")
        || body.contains("sys.route")
        // covers sys.navroute/navstep/navstepnum — the nav card's body must
        // re-evaluate ONCE when the OSRM fetch lands (fills nav_polyline)
        || body.contains("sys.nav")
}

/// Height (dp) of bar `index` of `count` for an intraday sparkline, from Yahoo's
/// 5-minute close series, normalized so the day's min→~8dp and max→`maxh`+8dp
/// (so a chart of pixel height `maxh`+8 shows the full range without clipping).
/// Map a card-facing range token to Yahoo chart-API `(range, interval)` params.
/// Tokens (case-insensitive): "1d" (default), "1w", "1m", "6m", "1y" — the five
/// chips an iOS-Stocks-style card shows. Empty/unknown tokens (e.g. an unset
/// `{{state.range}}` rendering as "") fall back to the intraday default so a
/// card is never blank because of a bad token.
fn yahoo_range_params(token: &str) -> (&'static str, &'static str) {
    match token.trim().to_ascii_lowercase().as_str() {
        "1w" | "5d" => ("5d", "30m"),
        "1m" | "1mo" => ("1mo", "1d"),
        "6m" | "6mo" => ("6mo", "1d"),
        "1y" => ("1y", "1wk"),
        _ => ("1d", "5m"),
    }
}

/// Sanitize a card-supplied ticker into a safe URL path segment. Card bodies
/// are LLM-generated (semi-trusted): a symbol containing `?`/`#`/`/`/`%`
/// would rewrite the request target (and split the fetch-dedup key), and a
/// `\0` — which Splash string literals CAN carry — reaches the Android HTTP
/// layer's `CString::new(url).unwrap()` and aborts the process. Keep only the
/// characters real Yahoo tickers use (letters, digits, `.` `-` `^` `=`),
/// uppercased, capped at 16.
pub(crate) fn sanitize_ticker(symbol: &str) -> String {
    let out: String = symbol
        .trim()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '^' | '='))
        .map(|c| c.to_ascii_uppercase())
        .take(16)
        .collect();
    // A real ticker has at least one letter/digit. Reject a result that is
    // empty or all-punctuation (e.g. ".." — which a URL canonicalizer would
    // resolve as a path-traversal segment, rewriting the request target). The
    // callers treat "" as "no symbol" and skip the fetch.
    if out.chars().any(|c| c.is_ascii_alphanumeric()) {
        out
    } else {
        String::new()
    }
}

/// The ONE Yahoo chart-API URL for a symbol×range. `sys.stockbar`,
/// `sys.stockrange` and the `StockPlot` widget all build their URL here, so
/// they share a single `script_data_fetch` cache entry — one request per
/// symbol×range serves the plot, the bars and every scalar on the card.
pub(crate) fn yahoo_chart_url(symbol: &str, range: &str) -> String {
    let sym = sanitize_ticker(symbol);
    let (yr, yi) = yahoo_range_params(range);
    format!("https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval={yi}&range={yr}")
}

/// Range-aware scalar for `sys.stockrange`, computed from the SAME Yahoo close
/// series the chart bars draw: the range's high/low extremes and the range's
/// own first→last change. Returns "—" when the series is absent/short so the
/// card shows the standard loading placeholder.
fn stock_range_field(bytes: &[u8], field: &str) -> String {
    let root: serde_json::Value = match serde_json::from_slice(bytes) {
        Ok(v) => v,
        Err(_) => return "—".to_string(),
    };
    let vals: Vec<f64> = match root
        .pointer("/chart/result/0/indicators/quote/0/close")
        .and_then(|c| c.as_array())
    {
        Some(a) => a.iter().filter_map(|x| x.as_f64()).collect(),
        None => return "—".to_string(),
    };
    if vals.is_empty() {
        return "—".to_string();
    }
    let mn = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (first, last) = (vals[0], vals[vals.len() - 1]);
    match field.to_ascii_lowercase().as_str() {
        "high" => format!("{mx:.2}"),
        "low" => format!("{mn:.2}"),
        "change" => format!("{:+.2}", last - first),
        "changepct" | "changepercent" => {
            if first != 0.0 {
                format!("{:+.2}%", (last - first) / first * 100.0)
            } else {
                "—".to_string()
            }
        }
        "up" => (if last >= first { "1" } else { "0" }).to_string(),
        _ => "—".to_string(),
    }
}

fn stock_bar_height(bytes: &[u8], index: usize, count: usize, maxh: f64) -> f64 {
    let span = (maxh - 8.0).max(8.0);
    let root: serde_json::Value = match serde_json::from_slice(bytes) {
        Ok(v) => v,
        Err(_) => return 6.0,
    };
    let arr = match root
        .pointer("/chart/result/0/indicators/quote/0/close")
        .and_then(|c| c.as_array())
    {
        Some(a) => a,
        None => return 6.0,
    };
    let vals: Vec<f64> = arr.iter().filter_map(|x| x.as_f64()).collect();
    if vals.len() < 2 {
        return 6.0;
    }
    let mn = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if mx <= mn {
        return 24.0;
    }
    let denom = (count.max(2) - 1) as f64;
    let pos = ((index as f64 / denom) * (vals.len() - 1) as f64).round() as usize;
    let val = vals[pos.min(vals.len() - 1)];
    8.0 + (val - mn) / (mx - mn) * span
}

/// Extract a scalar from an open-meteo JSON body at a dot-path, formatted for
/// display. A numeric path segment indexes into an array; other segments are
/// object keys. Returns None if the path is absent or the leaf isn't a scalar.
/// ISO datetimes ("2026-07-13T05:52", as open-meteo returns for sunrise/sunset)
/// are shortened to "HH:MM".
fn json_pluck(bytes: &[u8], path: &str) -> Option<String> {
    let root: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let mut cur = &root;
    for seg in path.split('.') {
        cur = if let Ok(idx) = seg.parse::<usize>() {
            cur.get(idx)?
        } else {
            cur.get(seg)?
        };
    }
    let s = match cur {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        _ => return None,
    };
    // open-meteo ISO datetime -> HH:MM (sunrise/sunset).
    if s.len() >= 16 && s.as_bytes().get(10) == Some(&b'T') {
        return Some(s[11..16].to_string());
    }
    Some(s)
}

/// Howard Hinnant's `days_from_civil` — the inverse of `civil_from_days`.
/// (year, month, day) → days since the Unix epoch.
fn days_from_civil(y: i64, m: u64, d: u64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe as i64 - 719_468
}

/// Weekday for a days-since-epoch count, 0 = Sunday. 1970-01-01 was a Thursday,
/// hence the +4.
fn weekday_from_days(z: i64) -> usize {
    (((z + 4) % 7 + 7) % 7) as usize
}

/// Abbreviated weekday names, index 0 = Sunday.
const DAY_EN: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const DAY_ZH: [&str; 7] = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];

/// Reduce a 7-element `daily.*` temperature array to its min or max.
///
/// `round_display` is deliberately NOT applied: this feeds a shader uniform, where
/// the extra precision is free and rounding the span would visibly quantise the
/// bar colours.
fn week_extreme(
    vm: &mut ScriptVm,
    lat: f64,
    lon: f64,
    path: &str,
    want_max: bool,
) -> Option<f64> {
    let url = format!(
        "https://api.open-meteo.com/v1/forecast?latitude={lat:.4}&longitude={lon:.4}\
&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,surface_pressure,is_day\
&daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,uv_index_max,precipitation_probability_max\
&timezone=auto&forecast_days=7"
    );
    let bytes = vm.host.cx_mut().script_data_fetch(&url)?;
    let mut acc: Option<f64> = None;
    for i in 0..7 {
        let Some(v) = json_pluck(&bytes, &format!("{path}.{i}")) else {
            continue;
        };
        let Ok(n) = v.parse::<f64>() else { continue };
        acc = Some(match acc {
            None => n,
            Some(a) if want_max => a.max(n),
            Some(a) => a.min(n),
        });
    }
    acc
}

/// Mean synodic month — new moon to new moon — in seconds.
const SYNODIC_SECS: f64 = 29.530_588_853 * 86_400.0;

/// A known new moon: 2000-01-06 18:14 UTC, the epoch conventionally used for
/// simple phase arithmetic.
const NEW_MOON_EPOCH: f64 = 947_182_440.0;

/// Position in the synodic cycle, 0..1: 0 new, 0.25 first quarter, 0.5 full,
/// 0.75 last quarter.
///
/// This is the MEAN cycle, not a full lunar theory: the true phase wanders by up
/// to about half a day because the Moon's orbit is elliptical. That is invisible
/// on a rendered disc and in an illumination percentage rounded to units, which
/// is all this feeds, and it avoids pulling an ephemeris into the widget crate.
fn moon_phase_fraction() -> f64 {
    let elapsed = now_unix_secs() as f64 - NEW_MOON_EPOCH;
    let f = (elapsed % SYNODIC_SECS) / SYNODIC_SECS;
    if f < 0.0 {
        f + 1.0
    } else {
        f
    }
}

/// The principal-phase name for a point in the cycle. The four exact phases (new,
/// quarters, full) name a narrow window around the instant; the rest of the cycle
/// is crescent or gibbous, waxing before full and waning after.
fn moon_phase_name(f: f64) -> &'static str {
    if f < 0.0335 || f >= 0.9665 {
        "New Moon"
    } else if f < 0.2165 {
        "Waxing Crescent"
    } else if f < 0.2835 {
        "First Quarter"
    } else if f < 0.4665 {
        "Waxing Gibbous"
    } else if f < 0.5335 {
        "Full Moon"
    } else if f < 0.7165 {
        "Waning Gibbous"
    } else if f < 0.7835 {
        "Last Quarter"
    } else {
        "Waning Crescent"
    }
}

/// The principal-phase name in Chinese — the traditional 八相 names, so a Chinese
/// card is not forced to print "Full Moon" in the middle of otherwise Chinese
/// text. Same boundaries as `moon_phase_name`.
fn moon_phase_name_zh(f: f64) -> &'static str {
    if f < 0.0335 || f >= 0.9665 {
        "新月"
    } else if f < 0.2165 {
        "蛾眉月"
    } else if f < 0.2835 {
        "上弦月"
    } else if f < 0.4665 {
        "盈凸月"
    } else if f < 0.5335 {
        "满月"
    } else if f < 0.7165 {
        "亏凸月"
    } else if f < 0.7835 {
        "下弦月"
    } else {
        "残月"
    }
}

/// "HH:MM" → minutes since midnight. json_pluck has already reduced open-meteo's
/// ISO local datetimes to this form.
fn hhmm_to_minutes(s: &str) -> Option<f64> {
    let (h, m) = s.trim().split_once(':')?;
    Some(h.trim().parse::<f64>().ok()? * 60.0 + m.trim().parse::<f64>().ok()?)
}

/// Round a DISPLAY reading to a whole number.
///
/// open-meteo reports a decimal, so an untouched card reads "29.5°", "8.45" and
/// "5.0 km/h" where every real weather app shows "29°", "8" and "5 km/h". The
/// decimals are noise at this precision and they cost real layout: a hero
/// temperature is set 60-76pt, so ".5" is a wide, attention-grabbing appendage on
/// the largest element of the card; in a forecast row it pushes the lo/hi labels
/// wide enough to squeeze the TempBar between them; and in a detail tile ".45"
/// pads a value already close to overflowing its box.
///
/// a2app's weather spec has always PROMISED this for temperatures ("show
/// whole-degree temps — the helper rounds temperature paths automatically"); it
/// was documented but never implemented, so every generated card carried them.
///
/// Scoped BY PATH NAME to the three quantities a card renders as a headline
/// number. Everything else is left alone, deliberately:
///   - `sys.airquality` shares `json_pluck`, and an AQI or pm2_5 reading must not
///     be touched here;
///   - "daily.sunrise.0" has already become "05:52", which must not be parsed as
///     a number;
///   - humidity and precipitation probability are already integers.
/// Values that do not parse as a number pass through untouched — including the
/// "—" placeholder shown while the fetch is in flight.
fn round_display(path: &str, value: String) -> String {
    let rounds = path.contains("temperature")
        || path.contains("uv_index")
        || path.contains("wind_speed");
    if !rounds {
        return value;
    }
    match value.parse::<f64>() {
        Ok(n) => format!("{}", n.round() as i64),
        Err(_) => value,
    }
}

/// Overpass API endpoint (keyless, no auth). overpass-api.de is the primary
/// public instance; https://overpass.kumi.systems/api/interpreter is a
/// drop-in mirror should it ever rate-limit.
const OVERPASS_URL: &str = "https://overpass-api.de/api/interpreter";

/// Map a card-facing category token to the ONE OSM tag filter with the best
/// worldwide coverage for that idea. "trail" uses nature_reserve because real
/// trail routes are OSM *relations*, which need a far heavier query shape than
/// the node/way `around` filter used here. Unknown tokens fall back to park —
/// the most widely mapped leisure tag — so a novel word never yields an empty
/// card.
fn overpass_filter(category: &str) -> (&'static str, &'static str) {
    match category.trim().to_ascii_lowercase().as_str() {
        "garden" => ("leisure", "garden"),
        "trail" => ("leisure", "nature_reserve"),
        "museum" => ("tourism", "museum"),
        "cafe" => ("amenity", "cafe"),
        "cinema" => ("amenity", "cinema"),
        "gym" => ("leisure", "fitness_centre"),
        "library" => ("amenity", "library"),
        "pool" => ("leisure", "swimming_pool"),
        "viewpoint" => ("tourism", "viewpoint"),
        "playground" => ("leisure", "playground"),
        "attraction" => ("tourism", "attraction"),
        "restaurant" | "food" => ("amenity", "restaurant"),
        "hotel" => ("tourism", "hotel"),
        // Google-Maps "add a stop along the route" categories:
        "gas" | "fuel" => ("amenity", "fuel"),
        "coffee" => ("amenity", "cafe"),
        "store" | "supermarket" | "grocery" => ("shop", "supermarket"),
        "pharmacy" => ("amenity", "pharmacy"),
        "atm" | "bank" => ("amenity", "bank"),
        "ev" | "charging" => ("amenity", "charging_station"),
        "parking" => ("amenity", "parking"),
        _ => ("leisure", "park"), // "park" and any unknown token
    }
}

// --- turn-by-turn navigation support (sys.navroute / sys.navstep) ---

struct NavStepInfo {
    cum_start: f64, // meters from route start where this step begins
    distance: f64,  // step length, meters
    kind: String,   // maneuver.type
    modifier: String,
    name: String,
    lanes: Vec<(String, bool)>, // (indication, is-recommended) at the maneuver
}

struct ParsedNavRoute {
    polyline: String,
    total_m: f64,
    total_s: f64,
    steps: Vec<NavStepInfo>,
}

thread_local! {
    static NAV_ROUTE_CACHE: std::cell::RefCell<
        std::collections::HashMap<String, std::rc::Rc<ParsedNavRoute>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Build OSRM's `;`-joined `{lon},{lat}` coordinate path: origin, then any
/// intermediate waypoints, then destination. `vias` is a `"lat,lon;lat,lon"`
/// string (empty / non-coord segments are ignored) so old 2-point callers pass
/// "" and get the exact same 2-coord path.
fn osrm_coords(lat1: f64, lon1: f64, lat2: f64, lon2: f64, vias: &str) -> String {
    let mut coords = format!("{lon1:.5},{lat1:.5}");
    for seg in vias.split(';') {
        let mut it = seg.trim().split(',');
        if let (Some(la), Some(lo)) = (it.next(), it.next()) {
            if let (Ok(la), Ok(lo)) = (la.trim().parse::<f64>(), lo.trim().parse::<f64>()) {
                coords.push_str(&format!(";{lo:.5},{la:.5}"));
            }
        }
    }
    coords.push_str(&format!(";{lon2:.5},{lat2:.5}"));
    coords
}

/// One URL per (from, vias…, to) so sys.navroute, sys.navstep and the MapView
/// widget all share a single deduped OSRM fetch. The via list keys into the
/// cache for free (the full URL is the key).
fn navroute_url(lat1: f64, lon1: f64, lat2: f64, lon2: f64, vias: &str) -> String {
    format!(
        "https://router.project-osrm.org/route/v1/driving/{}?overview=full&steps=true",
        osrm_coords(lat1, lon1, lat2, lon2, vias)
    )
}

fn nav_route_cached(vm: &mut ScriptVm, url: &str) -> Option<std::rc::Rc<ParsedNavRoute>> {
    if let Some(hit) = NAV_ROUTE_CACHE.with(|c| c.borrow().get(url).cloned()) {
        return Some(hit);
    }
    let bytes = vm.host.cx_mut().script_data_fetch(url)?;
    let route = parse_nav_route(&bytes)?;
    let rc = std::rc::Rc::new(route);
    NAV_ROUTE_CACHE.with(|c| c.borrow_mut().insert(url.to_string(), rc.clone()));
    Some(rc)
}

/// One serde parse of the OSRM response -> compact step table.
fn parse_nav_route(bytes: &[u8]) -> Option<ParsedNavRoute> {
    let root: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let route = root.get("routes")?.get(0)?;
    let polyline = route.get("geometry")?.as_str()?.to_string();
    let total_m = route.get("distance")?.as_f64()?;
    let total_s = route.get("duration")?.as_f64()?;
    // Iterate ALL legs (a route with K waypoints has K+1 legs), concatenating
    // their steps with a CONTINUOUS `cum` accumulator so cum_start stays
    // "meters from the route start" across waypoints — otherwise the banner
    // would stop guiding after the first via-point.
    let legs = route.get("legs")?.as_array()?;
    let mut steps = Vec::new();
    let mut cum = 0.0_f64;
    for leg in legs {
      let Some(steps_v) = leg.get("steps").and_then(|v| v.as_array()) else {
          continue;
      };
      for s in steps_v {
        let distance = s.get("distance").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let man = s.get("maneuver");
        let kind = man
            .and_then(|m| m.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let modifier = man
            .and_then(|m| m.get("modifier"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let name = s
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let mut lanes = Vec::new();
        if let Some(lv) = s
            .get("intersections")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("lanes"))
            .and_then(|v| v.as_array())
        {
            for l in lv.iter().take(6) {
                let ind = l
                    .get("indications")
                    .and_then(|v| v.get(0))
                    .and_then(|v| v.as_str())
                    .unwrap_or("straight")
                    .to_string();
                let valid = l.get("valid").and_then(|v| v.as_bool()).unwrap_or(false);
                lanes.push((ind, valid));
            }
        }
        steps.push(NavStepInfo {
            cum_start: cum,
            distance,
            kind,
            modifier,
            name,
            lanes,
        });
        cum += distance;
      }
    }
    if steps.is_empty() {
        return None;
    }
    Some(ParsedNavRoute {
        polyline,
        total_m,
        total_s,
        steps,
    })
}

fn nav_fmt_dist(m: f64) -> String {
    if m >= 1000.0 {
        format!("{:.1} km", m / 1000.0)
    } else {
        format!("{:.0} m", m.max(0.0))
    }
}

fn nav_arrow(step: &NavStepInfo) -> String {
    let m = step.modifier.as_str();
    match step.kind.as_str() {
        "arrive" => "🏁",
        "depart" => "▶",
        "merge" => "⤴",
        "on ramp" => "↗",
        "off ramp" => "↘",
        "fork" => {
            if m.contains("left") {
                "↖"
            } else {
                "↗"
            }
        }
        "roundabout" | "rotary" => "➡",
        "exit roundabout" | "exit rotary" => "↗",
        _ => match m {
            "left" => "⬅",
            "right" => "➡",
            "slight left" => "↖",
            "slight right" => "↗",
            "sharp left" => "⬅",
            "sharp right" => "➡",
            "uturn" => "⟲",
            _ => "⬆",
        },
    }
    .to_string()
}

fn nav_lane_glyph(indication: &str) -> &'static str {
    match indication {
        "left" | "sharp left" => "⬅",
        "right" | "sharp right" => "➡",
        "slight left" => "↖",
        "slight right" => "↗",
        "uturn" => "⟲",
        _ => "⬆",
    }
}

fn nav_instr(step: &NavStepInfo) -> String {
    let road = if step.name.trim().is_empty() {
        "the road".to_string()
    } else {
        step.name.clone()
    };
    let m = step.modifier.as_str();
    match step.kind.as_str() {
        "depart" => format!("Head out on {road}"),
        "arrive" => "Arrived at destination".to_string(),
        "merge" => format!("Merge onto {road}"),
        "on ramp" => "Take the ramp".to_string(),
        "off ramp" => "Take the exit".to_string(),
        "fork" => format!(
            "Keep {} at the fork",
            if m.contains("left") { "left" } else { "right" }
        ),
        "roundabout" | "rotary" => "Enter the roundabout".to_string(),
        "exit roundabout" | "exit rotary" => format!("Exit onto {road}"),
        "new name" => format!("Continue onto {road}"),
        _ => {
            if m.is_empty() {
                "Continue".to_string()
            } else {
                format!("Turn {m} onto {road}")
            }
        }
    }
}

/// Lanes for a maneuver: OSRM's, or synthesized from the modifier when absent
/// (car-grade nav always shows a lane strip near a turn).
fn nav_lanes(step: &NavStepInfo) -> Vec<(String, bool)> {
    if !step.lanes.is_empty() {
        let m = step.modifier.as_str();
        let any_valid = step.lanes.iter().any(|(_, v)| *v);
        return step
            .lanes
            .iter()
            .enumerate()
            .map(|(i, (ind, valid))| {
                let hot = if any_valid {
                    *valid
                } else {
                    // no valid flags: highlight by side
                    if m.contains("right") {
                        i == step.lanes.len() - 1
                    } else {
                        i == 0
                    }
                };
                (ind.clone(), hot)
            })
            .collect();
    }
    let m = step.modifier.as_str();
    if m.contains("left") {
        vec![
            ("left".into(), true),
            ("straight".into(), false),
            ("straight".into(), false),
        ]
    } else if m.contains("right") {
        vec![
            ("straight".into(), false),
            ("straight".into(), false),
            ("right".into(), true),
        ]
    } else {
        vec![("straight".into(), true), ("straight".into(), false)]
    }
}

/// Numeric per-step fields for layout binding (see sys.navstepnum).
fn nav_step_num(route: &ParsedNavRoute, d: f64, field: &str) -> f64 {
    let mut si = 0usize;
    for (i, s) in route.steps.iter().enumerate() {
        if d >= s.cum_start {
            si = i;
        } else {
            break;
        }
    }
    match field {
        "frac" => {
            if route.total_m > 0.0 {
                (d / route.total_m).clamp(0.0, 1.0)
            } else {
                0.0
            }
        }
        // meters to the upcoming maneuver — gate lane strips on `< 320`
        "dist" => {
            let cur_end = route
                .steps
                .get(si)
                .map(|s| s.cum_start + s.distance)
                .unwrap_or(route.total_m);
            (cur_end - d).max(0.0)
        }
        _ => -1.0,
    }
}

fn nav_step_field(route: &ParsedNavRoute, d: f64, field: &str) -> String {
    // current step = the one whose span contains d
    let mut si = 0usize;
    for (i, s) in route.steps.iter().enumerate() {
        if d >= s.cum_start {
            si = i;
        } else {
            break;
        }
    }
    let cur_end = route
        .steps
        .get(si)
        .map(|s| s.cum_start + s.distance)
        .unwrap_or(route.total_m);
    let next = route.steps.get(si + 1);
    match field {
        "instr" => next
            .map(nav_instr)
            .unwrap_or_else(|| "Arrived at destination".to_string()),
        "dist" => nav_fmt_dist((cur_end - d).max(10.0)),
        "arrow" => next
            .map(nav_arrow)
            .unwrap_or_else(|| "🏁".to_string()),
        "next_arrow" => route
            .steps
            .get(si + 2)
            .map(nav_arrow)
            .unwrap_or_else(|| "🏁".to_string()),
        "road" => route
            .steps
            .get(si)
            .map(|s| s.name.clone())
            .unwrap_or_default(),
        "rem" => nav_fmt_dist((route.total_m - d).max(0.0)),
        "remmin" => {
            let frac = if route.total_m > 0.0 {
                (1.0 - d / route.total_m).clamp(0.0, 1.0)
            } else {
                0.0
            };
            format!("{:.0}", (route.total_s * frac / 60.0).ceil().max(1.0))
        }
        f if f.starts_with("lane") => {
            let hot_query = f.ends_with("hot");
            let idx: usize = f
                .trim_start_matches("lane")
                .trim_end_matches("hot")
                .parse()
                .unwrap_or(99);
            let lanes = next.map(nav_lanes).unwrap_or_default();
            match lanes.get(idx) {
                Some((ind, hot)) => {
                    if hot_query {
                        if *hot { "1".to_string() } else { "0".to_string() }
                    } else {
                        nav_lane_glyph(ind).to_string()
                    }
                }
                None => String::new(),
            }
        }
        _ => String::new(),
    }
}

/// The open-meteo geocoding lookup URL for a place name — one URL per name so
/// sys.geocode + sys.geocodenum share the same deduped fetch.
fn geocode_url(name: &str) -> String {
    format!(
        "https://geocoding-api.open-meteo.com/v1/search?name={}&count=1&language=en&format=json",
        percent_encode_query(name.trim())
    )
}

/// Percent-encode a string for a URL query VALUE (RFC 3986): unreserved chars
/// pass through, everything else (space and Overpass QL's `[]();:,=`) becomes
/// %XX per UTF-8 byte. Local so the crate needs no url-encoding dependency
/// (same approach as the sys.photo prompt encoder).
fn percent_encode_query(s: &str) -> String {
    let mut enc = String::with_capacity(s.len() * 3);
    let mut buf = [0u8; 4];
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '~') {
            enc.push(ch);
        } else {
            for b in ch.encode_utf8(&mut buf).as_bytes() {
                enc.push('%');
                enc.push(char::from_digit((b >> 4) as u32, 16).unwrap().to_ascii_uppercase());
                enc.push(char::from_digit((b & 0xF) as u32, 16).unwrap().to_ascii_uppercase());
            }
        }
    }
    enc
}

/// The Overpass query URL for named `category` places within 4 km of lat/lon:
/// nodes AND ways (most parks/pools are mapped as ways; `out center` gives
/// each way a single representative point), capped at 30 elements to keep the
/// response phone-sized. Lat/lon are fixed to 4 decimals (~11 m) so GPS jitter
/// doesn't defeat the URL-level fetch dedup — sys.places and sys.placesnum
/// build this SAME url, so one request serves both.
fn overpass_places_url(lat: f64, lon: f64, category: &str) -> String {
    let (key, value) = overpass_filter(category);
    let around = format!("around:4000,{lat:.4},{lon:.4}");
    let query = format!(
        "[out:json][timeout:25];(node[{key}={value}]({around});way[{key}={value}]({around}););out center 30;"
    );
    format!("{OVERPASS_URL}?data={}", percent_encode_query(&query))
}

/// A named OSM place from an Overpass response, with its distance from the
/// request point.
struct NearbyPlace {
    name: String,
    dist_km: f64,
    lat: f64,
    lon: f64,
}

/// Great-circle distance in km (haversine, mean Earth radius). Sub-1% error
/// is invisible at the one-decimal precision cards display.
fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (dlat, dlon) = ((lat2 - lat1).to_radians(), (lon2 - lon1).to_radians());
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * 6371.0 * a.sqrt().min(1.0).asin()
}

/// Parse an Overpass `out center` body into named places sorted nearest-first.
/// Position: `lat`/`lon` for nodes, `center.lat`/`center.lon` for ways (what
/// `out center` emits for areas). Unnamed elements are skipped (a card can't
/// show "way 24680") and duplicate names deduped keeping the nearest — one
/// park is often mapped as several ways sharing a name. None when the body
/// isn't an Overpass JSON response (vs Some(empty) for a valid "nothing
/// nearby"), so callers can tell error from zero.
fn places_parse(bytes: &[u8], lat: f64, lon: f64) -> Option<Vec<NearbyPlace>> {
    let root: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let elements = root.get("elements")?.as_array()?;
    let mut out = Vec::new();
    for el in elements {
        let name = match el.pointer("/tags/name").and_then(|n| n.as_str()) {
            Some(n) if !n.trim().is_empty() => n.trim().to_string(),
            _ => continue,
        };
        let coord = |key: &str, center_ptr: &str| {
            el.get(key)
                .and_then(|v| v.as_f64())
                .or_else(|| el.pointer(center_ptr).and_then(|v| v.as_f64()))
        };
        let (plat, plon) = match (coord("lat", "/center/lat"), coord("lon", "/center/lon")) {
            (Some(a), Some(o)) => (a, o),
            _ => continue,
        };
        out.push(NearbyPlace {
            name,
            dist_km: haversine_km(lat, lon, plat, plon),
            lat: plat,
            lon: plon,
        });
    }
    out.sort_by(|a, b| a.dist_km.total_cmp(&b.dist_km));
    let mut seen = std::collections::HashSet::new();
    out.retain(|p| seen.insert(p.name.clone()));
    Some(out)
}

/// Field lookup for `sys.places`: row `index` of the parsed nearest-first
/// list. "count" ignores `index` so a header can show it without a dummy row.
/// "—" for unknown fields, out-of-range indices, or a malformed body — the
/// same placeholder the other live helpers show while loading.
fn places_field(bytes: &[u8], lat: f64, lon: f64, index: usize, field: &str) -> String {
    let places = match places_parse(bytes, lat, lon) {
        Some(p) => p,
        None => return "—".to_string(),
    };
    let f = field.to_ascii_lowercase();
    if f == "count" {
        return places.len().to_string();
    }
    let p = match places.get(index) {
        Some(p) => p,
        None => return "—".to_string(),
    };
    match f.as_str() {
        "name" => p.name.clone(),
        "distance" | "dist" => format!("{:.1} km", p.dist_km),
        "lat" => format!("{:.4}", p.lat),
        "lon" => format!("{:.4}", p.lon),
        _ => "—".to_string(),
    }
}

/// Free-text place/POI/address search via Photon (komoot) — keyless, built for
/// search & autocomplete, returns a ranked list. Backs `sys.search`/`sys.searchnum`
/// so a card can show tappable search RESULTS (the Google-Maps "search a place"
/// step) — unlike `sys.geocode` (one place-name → facts) or `sys.places`
/// (category + radius). `lang=en`, capped to 8 hits.
fn search_url(query: &str) -> String {
    format!(
        "https://photon.komoot.io/api/?q={}&limit=8&lang=en",
        percent_encode_query(query)
    )
}

/// One search hit: a display name, a secondary label (city/region/country), a
/// human category ("Museum", "Cafe"…), and coordinates to route to.
struct SearchHit {
    name: String,
    label: String,
    cat: String,
    lat: f64,
    lon: f64,
}

/// Turn Photon's osm_key/osm_value into a friendly category label ("Art gallery",
/// "Cafe", "Park"…), Title-cased, so a result row reads like a Google category.
fn photon_category(key: &str, value: &str) -> String {
    let v = match value {
        "" => key,
        other => other,
    };
    let pretty: String = v
        .replace('_', " ")
        .split_whitespace()
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_ascii_uppercase().to_string() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    pretty
}

/// Parse a Photon GeoJSON body into ranked hits. `geometry.coordinates` is
/// `[lon, lat]`. The primary name is `properties.name`, else
/// `housenumber street`, else the locality; the label chains the admin parts
/// (street/city/state/country) so ambiguous names ("Springfield") are
/// distinguishable. None when the body isn't valid GeoJSON (vs Some(empty) for
/// "no matches"), so callers tell error from zero.
fn search_parse(bytes: &[u8]) -> Option<Vec<SearchHit>> {
    let root: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let feats = root.get("features")?.as_array()?;
    let mut out = Vec::new();
    for ft in feats {
        let p = ft.get("properties");
        let coords = ft.pointer("/geometry/coordinates").and_then(|c| c.as_array());
        let (lon, lat) = match coords.and_then(|c| {
            Some((c.first()?.as_f64()?, c.get(1)?.as_f64()?))
        }) {
            Some(v) => v,
            None => continue,
        };
        let field = |k: &str| {
            p.and_then(|p| p.get(k))
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        let name = field("name")
            .or_else(|| match (field("housenumber"), field("street")) {
                (Some(h), Some(s)) => Some(format!("{h} {s}")),
                (_, Some(s)) => Some(s),
                _ => None,
            })
            .or_else(|| field("city"))
            .unwrap_or_else(|| "Unnamed place".to_string());
        let mut parts: Vec<String> = Vec::new();
        for k in ["street", "city", "state", "country"] {
            if let Some(v) = field(k) {
                if v != name && !parts.contains(&v) {
                    parts.push(v);
                }
            }
        }
        let cat = photon_category(
            field("osm_key").as_deref().unwrap_or(""),
            field("osm_value").as_deref().unwrap_or(""),
        );
        out.push(SearchHit {
            name,
            label: parts.join(", "),
            cat,
            lat,
            lon,
        });
    }
    Some(out)
}

/// Field lookup for `sys.search`: hit `index` of the ranked list. "count"
/// ignores `index`. "—"/"" for out-of-range or a bad body.
fn search_field(bytes: &[u8], index: usize, field: &str) -> String {
    let hits = match search_parse(bytes) {
        Some(h) => h,
        None => return String::new(),
    };
    let f = field.to_ascii_lowercase();
    if f == "count" {
        return hits.len().to_string();
    }
    let h = match hits.get(index) {
        Some(h) => h,
        None => return String::new(),
    };
    match f.as_str() {
        "name" => h.name.clone(),
        "label" | "addr" | "address" => h.label.clone(),
        "cat" | "category" => h.cat.clone(),
        "lat" => format!("{:.5}", h.lat),
        "lon" => format!("{:.5}", h.lon),
        _ => String::new(),
    }
}

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.*

    mod.widgets.SplashBase = #(Splash::register_widget(vm))

    mod.widgets.Splash = set_type_default() do mod.widgets.SplashBase{
        width: Fill height: Fit
    }
}

#[derive(Script, ScriptHook, WidgetRef, WidgetRegister)]
pub struct Splash {
    #[uid]
    uid: WidgetUid,
    #[source]
    source: ScriptObjectRef,
    #[deref]
    pub view: View,
    #[live]
    body: ArcStringMut,
    #[rust]
    eval_generation: u64,
    #[rust]
    tick_timer: Timer,
    /// The unique_id used for the last full eval, so tick() runs in the same scope.
    #[rust]
    last_unique_id: usize,
    /// This Splash's own VM, allocated on first eval (upstream isolation model).
    #[rust]
    vm_id: SplashVmId,
    /// Body text of the previous eval. Used to detect streaming extensions
    /// (the new body forward-extends the old) so repeated set_text(full growing
    /// text) reuses ONE vm body instead of a fresh generation per frame.
    #[rust]
    last_eval_body: String,
    /// Per-frame redraw pump for time-based shaders. A card drawn once has a
    /// frozen `self.draw_pass.time`, so any `pixel: fn(){… self.draw_pass.time …}`
    /// animation (rain, drifting clouds, sun rays, wind) renders but never moves.
    /// When the evaluated body uses `draw_pass.time` we keep requesting the next
    /// frame and redrawing the view, giving continuous ~60fps animation without
    /// any timer/state/asset. Off (NextFrame::default()) for static cards so
    /// they cost nothing.
    #[rust]
    anim_next_frame: NextFrame,
    #[rust]
    animating: bool,
    /// Value of the global script-data-fetch epoch at the last eval. When a live
    /// `sys.weather`/`sys.airquality` fetch this card fired completes, the epoch
    /// bumps; the per-frame pump notices the change and re-evaluates the body so
    /// the "—" placeholders are replaced with the loaded values.
    #[rust]
    last_data_epoch: u64,
    /// Whole-second value of the sim clock at the last eval — cards that call
    /// `sys.simsecs` re-evaluate when it advances (1 Hz animation driver).
    #[rust]
    last_sim_tick: u64,
}

/// Monotonic seconds since process start — the time source behind
/// `sys.simsecs` and the Splash 1 Hz re-eval driver. Deliberately NOT wall
/// clock: it can never jump backwards.
pub(crate) fn sim_clock_secs() -> f64 {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
}

/// Prefix for View-children mode: wraps code inside a View
const SPLASH_PREFIX_VIEW: &str = "use mod.prelude.widgets.*View{height:Fit, ";
/// Prefix for full-script mode: just imports, code must evaluate to a widget
const SPLASH_PREFIX_SCRIPT: &str = "use mod.prelude.widgets.*\n";
// 200_000 silently truncated large multi-page cards (a 22-page turn-by-turn
// nav card parsed+evaled clean but attached nothing) — the streaming
// parse+execute loop counts compile-fed opcodes too, so budget scales with
// BODY SIZE, not just executed work. 1M covers ~100KB bodies.
const SPLASH_EVAL_INSTRUCTION_LIMIT: usize = 1_000_000;

/// Detect whether Splash code is a full script (starts with `let`, `fn`,
/// or a widget constructor like `View{`, `SolidView{`) vs View children
/// (starts with properties like `flow:`, `width:`, or lowercase names).
fn is_full_script(body: &str) -> bool {
    let trimmed = body.trim_start();
    // Only treat as full script if it starts with scripting keywords
    // (let/fn/mod) — these can't appear inside a View{} property list.
    // Uppercase widget names (View{, SolidView{, Label{) stay in View-children mode.
    trimmed.starts_with("let ") || trimmed.starts_with("fn ") || trimmed.starts_with("mod.")
}

/// Does this View-children body's FIRST widget ask to fill its parent?
///
/// `SPLASH_PREFIX_VIEW` wraps the body in `View{height:Fit, …}`, which is
/// correct for a short inline snippet but collapses a full-bleed card to zero
/// height: `Fill` inside `Fit` resolves to nothing, so the card evaluates with
/// no error and draws no pixels. Scan just the first widget's property list —
/// a nested `height: Fill` deeper in the tree is fine, it is only the ROOT
/// asking for its parent's height that the `Fit` wrapper cannot satisfy.
fn root_wants_fill(body: &str) -> bool {
    let trimmed = body.trim_start();
    // Property list of the first widget: up to its first nested `{`, or the
    // whole first line, whichever ends sooner.
    let head_end = trimmed
        .find('\n')
        .unwrap_or(trimmed.len())
        .min(trimmed.len());
    let head = &trimmed[..head_end];
    head.replace(' ', "").contains("height:Fill")
}

/// Does the body's `{`/`}` depth ever dip below zero? Braces inside string
/// literals and comments (line comments, and block comments — ended by the
/// first `*/`, matching the DSL tokenizer) are ignored.
///
/// Depth going negative is the precise signature of a corrupt card whose text
/// gained extra `}` (e.g. a streamed card damaged mid-persist): the surplus
/// brace closes the root container early, later children fall out of the
/// evaluated tree, and the card renders as a fragment. A *healthy* body —
/// including every mid-stream prefix of a well-formed card — never dips below
/// zero, so this gate cannot misfire on progressive rendering. (A merely
/// *truncated* body stays at depth ≥ 0; the parser auto-closes it and the
/// partial card renders, which is the intended streaming behavior.)
fn braces_go_negative(body: &str) -> bool {
    let mut depth: i64 = 0;
    let mut chars = body.chars().peekable();
    let mut quote: Option<char> = None;
    let mut line_comment = false;
    let mut block_comment = false;
    while let Some(c) = chars.next() {
        if line_comment {
            if c == '\n' {
                line_comment = false;
            }
            continue;
        }
        if block_comment {
            // The DSL tokenizer ends a block comment at the FIRST `*/` (no
            // nesting) — mirror that exactly so the scan never disagrees
            // with the parser about what is code.
            if c == '*' && chars.peek() == Some(&'/') {
                chars.next();
                block_comment = false;
            }
            continue;
        }
        if let Some(q) = quote {
            if c == '\\' {
                chars.next();
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '"' | '\'' => quote = Some(c),
            '/' if chars.peek() == Some(&'/') => {
                chars.next();
                line_comment = true;
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                block_comment = true;
            }
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

impl Splash {
    /// Stable identity for the streaming script body, based on pointer address.
    fn self_id(&self) -> usize {
        self as *const Self as usize
    }

    fn eval_body(&mut self, cx: &mut Cx) {
        let body = self.body.as_ref().to_string();
        if body.is_empty() {
            return;
        }

        // Stop any previous tick timer
        cx.stop_timer(self.tick_timer);

        // Allocate this Splash's own VM on first eval so streaming
        // (stream_append) evaluates in an isolated scope.
        if self.vm_id == MAIN_SPLASH_VM_ID {
            self.vm_id = cx.alloc_splash_vm();
        }

        // Only start a NEW vm body (bump the generation) on a genuine content
        // replacement — NOT a streaming extension of the previous body. aichat
        // streams runsplash by calling set_text() with the full, growing block
        // string every frame; without this each frame got its own generation,
        // accumulating dozens of stale bodies whose widgets/closures lingered
        // (clicking a button then hit a stale generation -> "widget not found in
        // tree" -> the app vanished). A forward-extension reuses the same
        // unique_id so eval_with_append_source does its incremental checkpoint
        // parse (the same path stream_append uses). Compare the raw body (not the
        // prefixed code) so an is_full_script flip can't cause a false miss.
        let is_extension =
            !self.last_eval_body.is_empty() && body.starts_with(self.last_eval_body.as_str());
        if !is_extension {
            self.eval_generation += 1;
        }
        self.last_eval_body = body.clone();
        let unique_id = self.self_id().wrapping_add(self.eval_generation as usize);
        self.last_unique_id = unique_id;

        // Choose prefix based on code style
        let prefix = if is_full_script(&body) {
            SPLASH_PREFIX_SCRIPT
        } else {
            SPLASH_PREFIX_VIEW
        };
        let code = format!("{}{}", prefix, body);

        let script_mod = ScriptMod {
            cargo_manifest_path: String::new(),
            module_path: String::new(),
            file: String::new(),
            line: unique_id,
            column: 0,
            code: String::new(),
            values: vec![],
        };

        log!(
            "[SPLASH] eval_body: {} bytes, prefix={}, uid={}, gen={}, ext={}",
            body.len(),
            if is_full_script(&body) {
                "script"
            } else {
                "view"
            },
            unique_id,
            self.eval_generation,
            is_extension
        );

        // Evaluate in THIS Splash's own isolated vm and inject a `ui` global
        // rooted at this Splash (self.uid). That scopes `ui.<id>` to this
        // Splash's subtree (find_flood), so ids like `display` don't collide
        // with other Splash apps in the same chat. Then register the widgets
        // under this vm and mark the tree dirty so lookups can resolve them.
        let vm_id = self.vm_id;
        let self_uid = self.uid;
        let new_view = cx.with_script_vm_id(vm_id, |vm| {
            crate::widget_async::inject_scoped_ui_global(vm, self_uid);
            let value = vm.with_instruction_limit(SPLASH_EVAL_INSTRUCTION_LIMIT, |vm| {
                vm.eval_with_append_source(script_mod, &code, NIL.into())
            });
            if !value.is_err() && !value.is_nil() {
                Some(View::script_from_value(vm, value))
            } else {
                None
            }
        });

        if let Some(view) = new_view {
            // Corrupt-card guard (upstream): a body whose brace depth goes
            // negative but still "evaluates" is the parser silently recovering
            // from a card that gained extra `}` (e.g. a stream damaged
            // mid-persist) — the surplus brace closes the root early and the
            // card renders as a fragment. Treat it as an eval failure so the
            // quiet-period failure card fires. A healthy mid-stream prefix never
            // dips below zero, so progressive rendering is unaffected.
            if !braces_go_negative(&body) {
                self.view = view;
                self.view.set_visible(cx, true);
                // register_view_subtree roots `ui` at the Splash node (nav's
                // zero-rebuild ui.<id> fix) — supersedes inject_splash_ui_handle
                // at the wrapper uid + the separate mark_dirty.
                self.register_view_subtree(cx);
            } else {
                log!(
                    "[SPLASH] eval succeeded but brace depth went negative (corrupt card) — treating as eval failure"
                );
            }
        }

        // If the Splash code defines fn tick(), auto-start a 1s interval
        if body.contains("fn tick(") || body.contains("fn tick (") {
            self.tick_timer = cx.start_interval(1.0);
        }

        // If the card animates via a time-based shader, start the per-frame
        // redraw pump so `self.draw_pass.time` advances (see `anim_next_frame`).
        // Trigger on inline `draw_pass.time` OR on `WeatherIcon` (whose animated
        // shader lives in the widget def, so the body-scan wouldn't otherwise
        // see it).
        self.arm_animation_pump(cx, &body);

        // Record the data-fetch epoch AT this eval, so the per-frame pump only
        // re-evaluates when a LATER fetch completes (see handle_event).
        self.last_data_epoch = cx.script_data_fetch_epoch();
    }

    /// Start (or stop) the per-frame redraw pump based on whether `body`
    /// uses a time-based shader. Shared by `eval_body` and `stream_append`
    /// so streamed cards animate too. Triggers on inline `draw_pass.time`
    /// OR on `WeatherIcon` (whose animated shader lives in the widget def,
    /// so a body-scan wouldn't otherwise see it). See `anim_next_frame`.
    fn arm_animation_pump(&mut self, cx: &mut Cx, body: &str) {
        // Also arm for live-data cards (sys.weather/sys.airquality) so the pump
        // runs and can re-evaluate them when their async data arrives, even if
        // the card has no time-based shader of its own.
        // EXCEPT `fn tick()` cards (e.g. nav): they manage their own updates in
        // place via tick() (1 Hz) and their widgets drive their own frames (the
        // MapView), and their epoch-driven re-eval is suppressed anyway (see the
        // pump handler). Arming here just made the pump repaint the whole card —
        // MapView + tiles — 60x/s forever, pinning the GPU. Let them idle.
        let is_tick = body.contains("fn tick(") || body.contains("fn tick (");
        self.animating = body.contains("draw_pass.time")
            || body.contains("WeatherIcon")
            || body.contains("sys.simsecs")
            || (body_binds_live_data(body) && !is_tick);
        if self.animating {
            self.anim_next_frame = cx.new_next_frame();
        } else {
            self.anim_next_frame = NextFrame::default();
        }
    }

    /// Register the freshly-evaluated card subtree in the widget tree and
    /// re-root this Splash VM's `ui` global. Shared by `eval_body` and
    /// `stream_append`.
    ///
    /// Two things must hold for `ui.<id>` (fn tick(), helper fns, handlers) to
    /// resolve the card's `name := Widget{}` children:
    ///
    /// 1. The `:=` names must actually be IN the widget tree. The lazy sync
    ///    (`children()` walk on the next query) can't guarantee that: eval runs
    ///    inside draw/event dispatch where this Splash and its ancestors are
    ///    RefCell-borrowed, so `try_children` fails there and the new subtree
    ///    would stay unregistered until some unrelated quiescent query happens
    ///    to sync the tree. Register the subtree NOW from the owned `self.view`
    ///    (its children are freshly built and unborrowed), exactly mirroring
    ///    what `Splash::children()` reports (the wrapper view is skipped; its
    ///    children attach directly under the Splash uid).
    ///
    /// 2. The `ui` handle must be rooted at a uid the tree indexes. That is
    ///    `self.uid` (the Splash node) — NOT `self.view.widget_uid()`: because
    ///    `Splash::children()` forwards the wrapper view's children, the
    ///    wrapper's own uid never enters the tree, so a handle rooted there
    ///    could never anchor its scoped subtree search and always fell through
    ///    to the global fallback (breaking per-card id scoping, and failing
    ///    outright when the names weren't globally findable).
    fn register_view_subtree(&mut self, cx: &mut Cx) {
        for (name, child) in self.view.children.iter() {
            cx.widget_tree_insert_child_deep(self.uid, *name, child.clone());
        }
        crate::widget_async::inject_splash_ui_handle(cx, self.vm_id, self.uid);
        cx.widget_tree_mark_dirty(self.uid);
    }

    /// Call a named function defined in the Splash code's scope.
    pub fn call_fn(&mut self, cx: &mut Cx, name: LiveId) {
        let unique_id = self.last_unique_id;
        if unique_id == 0 {
            return;
        }

        cx.with_script_vm_id(self.vm_id, |vm| {
            // Find the body by matching the unique_id we used during eval
            // (body lives in this Splash's isolated vm, same as eval_body).
            let scope_obj = {
                let bodies = vm.bx.code.bodies.borrow();
                let mut found = None;
                for body in bodies.iter() {
                    if let ScriptSource::Mod(m) = &body.source {
                        if m.line == unique_id {
                            found = Some(body.scope.as_object());
                            break;
                        }
                    }
                }
                found
            };

            if let Some(scope) = scope_obj {
                let tick_fn = vm.bx.heap.scope_value(scope, name, vm.trap());
                if !tick_fn.is_nil() && !tick_fn.is_err() {
                    vm.call(tick_fn, &[]);
                }
            }
        });

        cx.redraw_all();
    }

    /// Start a new streaming session. Resets the accumulated code and
    /// increments the generation so the VM creates a fresh body.
    pub fn stream_begin(&mut self, cx: &mut Cx) {
        self.eval_generation += 1;
        self.body.set("");
        // Eval a minimal empty view to clear previous content
        self.body.set("View{}");
        self.eval_body(cx);
        self.body.set("");
        cx.redraw_all();
    }

    /// Append a chunk of Splash code and incrementally re-evaluate.
    /// The VM reuses the same body (fixed line ID) so only new tokens
    /// are tokenized and parsed via checkpoint-based streaming.
    pub fn stream_append(&mut self, cx: &mut Cx, chunk: &str) {
        // Append to body
        let mut current = self.body.as_ref().to_string();
        current.push_str(chunk);
        self.body.set(&current);

        let prefix = if is_full_script(&current) {
            SPLASH_PREFIX_SCRIPT
        } else {
            SPLASH_PREFIX_VIEW
        };
        let code = format!("{}{}", prefix, current);

        // Use a fixed line ID (based on self_id + current generation)
        // so eval_with_append_source finds the existing body and
        // only tokenizes/parses the new delta.
        let unique_id = self.self_id().wrapping_add(self.eval_generation as usize);

        let script_mod = ScriptMod {
            cargo_manifest_path: String::new(),
            module_path: String::new(),
            file: String::new(),
            line: unique_id,
            column: 0,
            code: String::new(),
            values: vec![],
        };

        let vm_id = self.vm_id;
        let self_uid = self.uid;
        let new_view = cx.with_script_vm_id(vm_id, |vm| {
            crate::widget_async::inject_scoped_ui_global(vm, self_uid);
            let value = vm.with_instruction_limit(SPLASH_EVAL_INSTRUCTION_LIMIT, |vm| {
                vm.eval_with_append_source(script_mod, &code, NIL.into())
            });
            if !value.is_err() && !value.is_nil() {
                Some(View::script_from_value(vm, value))
            } else {
                None
            }
        });

        if let Some(view) = new_view {
            // Same corrupt-card guard as eval_body: never adopt a fragment
            // built from a body whose brace depth went negative.
            if !braces_go_negative(&current) {
                self.view = view;
                // register_view_subtree roots `ui` at the Splash node so helper
                // `fn`s can use `ui.<id>.set_text(...)` (nav's zero-rebuild fix).
                self.register_view_subtree(cx);
            } else {
                log!(
                    "[SPLASH] stream_append: eval succeeded but brace depth went negative (corrupt card) — view not adopted"
                );
            }
        }
        // Streamed cards must arm the animation pump too (eval_body isn't called
        // on this path), or time-based shaders (WeatherIcon / draw_pass.time)
        // render once and freeze.
        self.arm_animation_pump(cx, &current);
    }
}

impl WidgetNode for Splash {
    fn widget_uid(&self) -> WidgetUid {
        self.uid
    }

    fn walk(&mut self, cx: &mut Cx) -> Walk {
        self.view.walk(cx)
    }

    fn area(&self) -> Area {
        self.view.area()
    }

    fn redraw(&mut self, cx: &mut Cx) {
        self.view.redraw(cx);
    }

    fn children(&self, visit: &mut dyn FnMut(LiveId, WidgetRef)) {
        self.view.children(visit);
    }
}

impl Drop for Splash {
    fn drop(&mut self) {
        // A Splash owns an isolate script VM. `Drop` has no `Cx`, so it can't free
        // the VM here; it just marks the id for reclamation. The isolate is torn
        // down later by `gc_dead_splash_isolates` (on the next isolate alloc, async
        // pump, or Splash event) while a `Cx` is available and nothing runs in it.
        crate::widget_async::mark_splash_isolate_dead(self.vm_id);
    }
}

thread_local! {
    /// Set by in-place tick setters (Label::set_text, MapView route setters) when
    /// they ACTUALLY change a value. A `fn tick()` card's 1 Hz forced repaint is
    /// then skipped when a tick changed nothing (e.g. a static plan map whose
    /// route/labels are unchanged) — so the GL surface stops swapping every
    /// second, which was flickering the native overlays (composer/FAB) composited
    /// over it. The drive view changes the car/ETA each tick, so it still repaints.
    static SPLASH_TICK_CHANGED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Called by an in-place widget setter when it changes a value during `tick()`.
pub fn splash_mark_tick_changed() {
    SPLASH_TICK_CHANGED.with(|c| c.set(true));
}

impl Widget for Splash {
    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        // Handle tick timer — call tick() in the Splash code's scope
        if self.tick_timer.is_event(event).is_some() {
            SPLASH_TICK_CHANGED.with(|c| c.set(false));
            self.call_fn(cx, id!(tick));
            // tick() updates widgets in place (ui.<id>.set_text/…) — e.g. the nav
            // card's live per-mode ETA and drive car position. Those set_* calls
            // don't self-schedule a paint on their own, so repaint here — but ONLY
            // when a setter actually changed a value this tick. A static card (a
            // plan map with an unchanged route/labels) forced a surface swap every
            // second otherwise, and that swap flickered the native composer/FAB
            // overlays over the GL surface.
            if SPLASH_TICK_CHANGED.with(|c| c.get()) {
                self.view.redraw(cx);
            }
        }

        // Per-frame redraw pump for time-based shaders: redraw the view (so the
        // pixel shaders re-run with an advanced `self.draw_pass.time`) and queue
        // the next frame. Self-sustaining while `animating`.
        if self.animating && self.anim_next_frame.is_event(event).is_some() {
            // Live data (sys.weather/sys.airquality) loads asynchronously; when a
            // fetch completes the global epoch bumps. Re-evaluate the body ONCE per
            // change so the "—" placeholders baked in at eval time are replaced by
            // the loaded values (a plain repaint never re-runs the script). eval_body
            // only reads cached data / fires still-pending fetches — it never bumps
            // the epoch — so this settles and cannot loop.
            let epoch = cx.script_data_fetch_epoch();
            // Time-driven cards: a body that calls `sys.simsecs` re-evaluates
            // once per whole second, so bindings derived from the sim clock
            // (moving markers, progress bars, auto-advancing banners) animate
            // without any tick/state plumbing.
            let sim_tick = sim_clock_secs() as u64;
            let sim_due =
                sim_tick != self.last_sim_tick && self.body.as_ref().contains("sys.simsecs");
            // A `fn tick()` card manages its own updates in place (ui.<id>.set_*)
            // and must NEVER rebuild: re-evaluating it destroys and recreates its
            // widgets — for a nav card that means tearing down the MapView (whole
            // map blanks to a flat route line) on EVERY tile fetch, since tile
            // HTTP responses bump the global data epoch. So suppress epoch-driven
            // re-eval for tick cards; they push loaded data via tick().
            let is_tick_card = self.body.as_ref().contains("fn tick(")
                || self.body.as_ref().contains("fn tick (");
            // A genuine PER-FRAME animation is a time-driven shader (`draw_pass.time`
            // or the WeatherIcon's animated shader). A card that merely binds async
            // live data is STATIC once loaded — repainting it every frame (below)
            // just pins the GPU. So only repaint per-frame for real animations;
            // live-data cards still repaint on their epoch change (the `if` above).
            let needs_frame_anim = {
                let b = self.body.as_ref();
                b.contains("draw_pass.time") || b.contains("WeatherIcon")
            };
            if (epoch != self.last_data_epoch
                && body_binds_live_data(self.body.as_ref())
                && !is_tick_card)
                || sim_due
            {
                self.last_sim_tick = sim_tick;
                self.eval_body(cx);
                cx.redraw_all();
            } else if needs_frame_anim {
                self.view.redraw(cx);
            }
            self.anim_next_frame = cx.new_next_frame();
        }

        self.view.handle_event(cx, event, scope);
    }

    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        self.view.draw_walk(cx, scope, walk)
    }

    fn text(&self) -> String {
        self.body.as_ref().to_string()
    }

    fn set_text(&mut self, cx: &mut Cx, v: &str) {
        if self.body.as_ref() != v {
            self.body.set(v);
            self.eval_body(cx);
            // eval_body replaces self.view with a new View whose area is not
            // yet registered in the draw system, so self.redraw(cx) would be
            // a no-op.  Force a full redraw so the parent re-layouts.
            cx.redraw_all();
        }
    }
}

impl SplashRef {
    pub fn set_text(&self, cx: &mut Cx, v: &str) {
        if let Some(mut inner) = self.borrow_mut() {
            inner.set_text(cx, v);
        }
    }

    pub fn stream_begin(&self, cx: &mut Cx) {
        if let Some(mut inner) = self.borrow_mut() {
            inner.stream_begin(cx);
        }
    }

    pub fn stream_append(&self, cx: &mut Cx, chunk: &str) {
        if let Some(mut inner) = self.borrow_mut() {
            inner.stream_append(cx, chunk);
        }
    }
}
