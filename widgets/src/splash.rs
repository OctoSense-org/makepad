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
fn civil_from_days(days: i64) -> (i64, u64, u64) {
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
                Some(bytes) => json_pluck(&bytes, path.trim()).unwrap_or_else(|| {
                    log!(
                        "[WXTRACE] sys.weather pluck MISS path={:?} ({} bytes) head={:?}",
                        path.trim(),
                        bytes.len(),
                        String::from_utf8_lossy(&bytes[..bytes.len().min(80)])
                    );
                    "—".to_string()
                }),
                None => "—".to_string(),
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
            let value = vm
                .host
                .cx_mut()
                .script_data_fetch(&url)
                .and_then(|bytes| json_pluck(&bytes, path.trim()))
                .unwrap_or_else(|| "—".to_string());
            vm.bx.heap.new_string_from_str(&value)
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
            let sym = symbol.trim().to_ascii_uppercase();
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
    // `index` of `count` in an intraday sparkline (Yahoo 5-minute closes for the
    // day, normalized so the day's low→~8dp and high→~158dp). Draw the chart as a
    // bottom-aligned `flow: Right` row of thin `SolidView{ height: sys.stockbar(
    // "AAPL", N, COUNT) }` bars (an area/line of the price path). Returns a small
    // height while the async fetch loads, then the card re-evaluates.
    vm.add_method(
        sys,
        id_lut!(stockbar),
        script_args_def!(symbol = NIL, index = NIL, count = NIL, maxh = NIL),
        |vm, args| {
            let sym_v = script_value!(vm, args.symbol);
            let mut symbol = String::new();
            vm.bx.heap.cast_to_string(sym_v, &mut symbol);
            let index = script_value!(vm, args.index).as_number().unwrap_or(0.0).max(0.0) as usize;
            let count = (script_value!(vm, args.count).as_number().unwrap_or(40.0) as usize).max(2);
            // Optional 4th arg: the chart's pixel height, so the area fills it
            // exactly with no peak clipping. Defaults to 150 (legacy behavior).
            let maxh = script_value!(vm, args.maxh).as_number().filter(|v| *v > 8.0).unwrap_or(150.0);
            let sym = symbol.trim().to_ascii_uppercase();
            let url = format!(
                "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=5m&range=1d"
            );
            let h = match vm.host.cx_mut().script_data_fetch(&url) {
                Some(bytes) => stock_bar_height(&bytes, index, count, maxh),
                None => 6.0,
            };
            ScriptValue::from_f64(h)
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

    vm.set_injected_global(id!(sys), sys.into());
}

/// True if a Splash body calls any live-data helper (sys.weather/airquality/
/// stock/news). Such cards must re-evaluate when their async fetch lands (the
/// value is baked into a Label at eval time), so we arm the frame pump + watch
/// the data-fetch epoch for them. Keep in sync with the data `sys.*` helpers.
fn body_binds_live_data(body: &str) -> bool {
    body.contains("sys.weather")
        || body.contains("sys.airquality")
        || body.contains("sys.stock")
        || body.contains("sys.news")
        || body.contains("sys.movers")
}

/// Height (dp) of bar `index` of `count` for an intraday sparkline, from Yahoo's
/// 5-minute close series, normalized so the day's min→~8dp and max→`maxh`+8dp
/// (so a chart of pixel height `maxh`+8 shows the full range without clipping).
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
}

/// Prefix for View-children mode: wraps code inside a View
const SPLASH_PREFIX_VIEW: &str = "use mod.prelude.widgets.*View{height:Fit, ";
/// Prefix for full-script mode: just imports, code must evaluate to a widget
const SPLASH_PREFIX_SCRIPT: &str = "use mod.prelude.widgets.*\n";
const SPLASH_EVAL_INSTRUCTION_LIMIT: usize = 200_000;

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
            self.view = view;
            self.view.set_visible(cx, true);
            crate::widget_async::inject_splash_ui_handle(cx, self.vm_id, self.view.widget_uid());
            cx.widget_tree_mark_dirty(self.uid);
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
