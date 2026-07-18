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
                        None => "—".to_string(),
                    };
                    vm.bx.heap.new_string_from_str(&out)
                },
            };
            let out = match vm.host.cx_mut().script_data_fetch(&url) {
                Some(bytes) => json_pluck(&bytes, path).unwrap_or_else(|| "—".to_string()),
                None => "—".to_string(),
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
        script_args_def!(lat1 = NIL, lon1 = NIL, lat2 = NIL, lon2 = NIL, field = NIL),
        |vm, args| {
            let lat1 = script_value!(vm, args.lat1).as_number().unwrap_or(0.0);
            let lon1 = script_value!(vm, args.lon1).as_number().unwrap_or(0.0);
            let lat2 = script_value!(vm, args.lat2).as_number().unwrap_or(0.0);
            let lon2 = script_value!(vm, args.lon2).as_number().unwrap_or(0.0);
            let field_v = script_value!(vm, args.field);
            let mut field = String::new();
            vm.bx.heap.cast_to_string(field_v, &mut field);
            let url = format!(
                "https://router.project-osrm.org/route/v1/driving/{lon1:.4},{lat1:.4};{lon2:.4},{lat2:.4}?overview=false"
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
                None => "—".to_string(),
            };
            vm.bx.heap.new_string_from_str(&out)
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
            let mut value = match vm.host.cx_mut().script_data_fetch(&url) {
                Some(bytes) => json_pluck(&bytes, path.trim()).unwrap_or_else(|| "—".to_string()),
                None => "—".to_string(),
            };
            // Temperatures render as whole degrees (no decimal) — a weather card
            // shows "27°", not "27.3°". Only *temperature* paths round; wind / UV /
            // pressure keep their natural precision, and non-numeric values
            // (sunrise "05:52", the "—" placeholder) pass through untouched.
            if path.contains("temperature") {
                if let Ok(n) = value.parse::<f64>() {
                    value = n.round().to_string();
                }
            }
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
            let value = vm
                .host
                .cx_mut()
                .script_data_fetch(&url)
                .and_then(|bytes| json_pluck(&bytes, path.trim()))
                .unwrap_or_else(|| "—".to_string());
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
                None => "—".to_string(),
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
                None => "—".to_string(),
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
                None => "—".to_string(),
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
                None => "—".to_string(),
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
                None => "—".to_string(),
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
        // substring covers sys.geocodenum too (same trick as weather/weathernum)
        || body.contains("sys.geocode")
        || body.contains("sys.route")
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
        "restaurant" => ("amenity", "restaurant"),
        "hotel" => ("tourism", "hotel"),
        _ => ("leisure", "park"), // "park" and any unknown token
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
    /// Whether the CURRENT body has ever evaluated to a view. Reset on a
    /// genuine content replacement, kept across streaming extensions.
    #[rust]
    render_ok: bool,
    /// Quiet-period timer. A streamed body is syntactically incomplete for most
    /// of its life, so a failed eval mid-stream is normal and must not be
    /// surfaced. This fires only once the body has stopped growing, at which
    /// point a still-empty view means the card really did fail.
    #[rust]
    failure_timer: Timer,
}

/// Prefix for View-children mode: wraps code inside a View
const SPLASH_PREFIX_VIEW: &str = "use mod.prelude.widgets.*View{height:Fit, ";
/// Prefix for full-script mode: just imports, code must evaluate to a widget
const SPLASH_PREFIX_SCRIPT: &str = "use mod.prelude.widgets.*\n";
const SPLASH_EVAL_INSTRUCTION_LIMIT: usize = 200_000;
/// How long the body must stop growing before a still-unrendered card is
/// declared failed. Long enough to outlast a stalled network chunk, short
/// enough that a dead card doesn't look like a hung app.
const SPLASH_FAILURE_DELAY: f64 = 3.0;
/// Offset for the failure card's vm body id, so it never collides with a
/// generation of the real body (whose parser state is what just failed).
const SPLASH_FAILURE_ID_SALT: usize = 0x5f_a1_1e_d0;
/// Shown in place of the blank view when a body cannot be parsed. Deliberately
/// tiny and literal — it must not itself depend on anything that can fail.
const SPLASH_FAILURE_CARD: &str = r#"SolidView{ width: Fill height: Fit flow: Down new_batch: true draw_bg.color: #ffffff padding: Inset{left: 16 top: 14 right: 16 bottom: 14}
    Label{ text: "Card failed to render" draw_text.color: #000000 draw_text.text_style.font_size: 15 }
    Label{ text: "The generated card DSL did not parse. See the Makepad log for the parse errors." draw_text.color: #6a6a6a draw_text.text_style.font_size: 11 margin: Inset{top: 6} }
}"#;

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
            // New content, not a continuation — nothing has rendered for it yet.
            self.render_ok = false;
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
                // LOCAL DEBUG: this failure was SILENT — a card whose eval
                // errors (e.g. instruction-limit) left the old/empty view
                // with no trace.
                crate::log!(
                    "[SPLASH] eval FAILED: err={} nil={} value={:?}",
                    value.is_err(),
                    value.is_nil(),
                    value
                );
                None
            }
        });

        if let Some(view) = new_view {
            self.view = view;
            self.view.set_visible(cx, true);
            crate::widget_async::inject_splash_ui_handle(cx, self.vm_id, self.view.widget_uid());
            cx.widget_tree_mark_dirty(self.uid);
            self.render_ok = true;
        }
        // NOTE: on failure `self.view` is deliberately left alone — mid-stream
        // that keeps the last good frame on screen. `set_text` arms
        // `failure_timer` so a body that never recovers still surfaces instead
        // of sitting there as a blank, zero-height view.

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

    /// The body stopped growing and still hasn't produced a view. Try once more
    /// from a clean parser, then fall back to a visible failure card.
    ///
    /// Before this existed a card that failed to parse left `self.view` as the
    /// default empty View — `height: Fit` over no children draws zero pixels,
    /// so the app showed a blank screen with no signal anywhere in the UI. The
    /// parse errors went to the log and nothing else: `report_error` only sets
    /// `ScriptParser::had_error`, which nothing reads.
    fn show_eval_failure(&mut self, cx: &mut Cx) {
        if self.render_ok || self.body.as_ref().is_empty() {
            return;
        }

        // Retry from scratch first. A streamed body parses incrementally from a
        // checkpoint, so one bad token early in the stream poisons every later
        // continuation — even when the completed text is perfectly valid.
        // Clearing `last_eval_body` makes the next eval a non-extension, which
        // bumps the generation and therefore allocates a fresh vm body + parser.
        self.last_eval_body.clear();
        self.eval_body(cx);
        if self.render_ok {
            return;
        }

        // Genuinely broken. Evaluate the failure card under its own body id so
        // it cannot inherit the parser state that just failed.
        log!(
            "[SPLASH] body of {} bytes never rendered (gen={}) — showing failure card",
            self.body.as_ref().len(),
            self.eval_generation
        );
        let code = format!("{}{}", SPLASH_PREFIX_SCRIPT, SPLASH_FAILURE_CARD);
        let script_mod = ScriptMod {
            cargo_manifest_path: String::new(),
            module_path: String::new(),
            file: String::new(),
            line: self.self_id().wrapping_add(SPLASH_FAILURE_ID_SALT),
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
            self.view = view;
            self.view.set_visible(cx, true);
            crate::widget_async::inject_splash_ui_handle(cx, self.vm_id, self.view.widget_uid());
            cx.widget_tree_mark_dirty(self.uid);
            cx.redraw_all();
        } else {
            // The failure card itself failed — that is a bug in this file, not
            // in the generated DSL. Say so rather than going quiet again.
            log!("[SPLASH] failure card did not evaluate; view stays blank");
        }
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
        self.animating = body.contains("draw_pass.time")
            || body.contains("WeatherIcon")
            || body_binds_live_data(body);
        if self.animating {
            self.anim_next_frame = cx.new_next_frame();
        } else {
            self.anim_next_frame = NextFrame::default();
        }
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
                // LOCAL DEBUG: this failure was SILENT — a card whose eval
                // errors (e.g. instruction-limit) left the old/empty view
                // with no trace.
                crate::log!(
                    "[SPLASH] eval FAILED: err={} nil={} value={:?}",
                    value.is_err(),
                    value.is_nil(),
                    value
                );
                None
            }
        });

        if let Some(view) = new_view {
            self.view = view;
            // Make `ui` a global in this splash's VM (pointing at the freshly-built view root) so
            // helper `fn`s inside the block can use `ui.<id>.set_text(...)`, not just inline
            // handlers. Without this, calculators/forms that route through a helper silently fail.
            crate::widget_async::inject_splash_ui_handle(cx, self.vm_id, self.view.widget_uid());
            cx.widget_tree_mark_dirty(self.uid);
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

impl Widget for Splash {
    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        // Handle tick timer — call tick() in the Splash code's scope
        if self.tick_timer.is_event(event).is_some() {
            self.call_fn(cx, id!(tick));
        }

        // The body has stopped growing. If it never rendered, surface that.
        if self.failure_timer.is_event(event).is_some() {
            self.show_eval_failure(cx);
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
            if epoch != self.last_data_epoch && body_binds_live_data(self.body.as_ref()) {
                self.eval_body(cx);
                cx.redraw_all();
            } else {
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

            // aichat streams a card by calling set_text() with the full growing
            // block, so every chunk lands here. Re-arm the quiet-period check on
            // each one: while text keeps arriving the timer keeps being pushed
            // back, and it only fires once the body has gone still. That is the
            // point at which "no view yet" means failure rather than "not
            // finished streaming".
            cx.stop_timer(self.failure_timer);
            if !self.render_ok {
                self.failure_timer = cx.start_timeout(SPLASH_FAILURE_DELAY);
            }
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
