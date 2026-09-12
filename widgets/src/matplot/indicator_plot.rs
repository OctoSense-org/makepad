//! `IndicatorPlot` — several countries' World Bank series on one axis.
//!
//! The sibling of `StockPlot`, and deliberately the same shape: the card NAMES
//! what to plot (which countries, which indicator, how many years) and the
//! widget fetches it. A card never carries sixty numbers through the DSL —
//! §4's no-facts rule reaches the chart too, and a series baked into a card is
//! wrong the moment the World Bank revises it.
//!
//! Data: `api.worldbank.org/v2/country/{A;B}/indicator/{code}` — keyless, and
//! the same `Cx::script_data_fetch` cache every other live binding uses, so
//! ONE request per (countries × indicator × span) serves the chart and every
//! scalar the card shows beside it. While it is in flight the widget draws the
//! standard placeholder and watches the fetch epoch, exactly as StockPlot does.

use crate::{
    makepad_derive_widget::*,
    makepad_draw::*,
    matplot::plot_view::{nice_ticks, PlotView},
    matplot::types::LineStyle,
    widget::*,
};

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.*

    mod.widgets.IndicatorPlotBase = #(IndicatorPlot::register_widget(vm))

    mod.widgets.IndicatorPlot = set_type_default() do mod.widgets.IndicatorPlotBase{
        width: Fill
        height: 190

        // One colour per series, in order. Five is the practical ceiling for a
        // phone-width legend; a sixth country would need a different reading.
        color_1: #x0a84ff
        color_2: #xff9f0a
        color_3: #x30d158
        color_4: #xbf5af2
        color_5: #xff453a
        baseline_color: #xffffff30
        grid_color: #xffffff14
        text_color: #xffffff59
        border_color: #x00000000
        show_border: false
        show_grid: true
        show_ticks: true
        tick_font_size: 9.0
        plot_margin: Inset{left: 6.0, top: 8.0, right: 44.0, bottom: 30.0}

        draw_bg +: {
            draw_depth: 0.0
            color: #x00000000
        }
        draw_grid +: { draw_depth: 0.1 }
        draw_vector +: { draw_depth: 2.0 }
        draw_text +: {
            draw_depth: 3.0
            text_style: theme.font_regular{}
        }
    }
}

/// One country's answered series, newest-last.
#[derive(Default, Clone)]
struct CountrySeries {
    /// The display name the API answered ("China"), not the code the card sent.
    name: String,
    /// (year, value) pairs, ascending by year, nulls dropped.
    years: Vec<f64>,
    values: Vec<f64>,
}

#[derive(Script, ScriptHook, Widget)]
pub struct IndicatorPlot {
    #[source]
    source: ScriptObjectRef,
    #[deref]
    plot_view: PlotView,

    /// Semicolon- or comma-separated ISO3 codes, e.g. "CHN,IND". Empty → the
    /// placeholder, and nothing is fetched.
    #[live]
    countries: String,

    /// A World Bank indicator code, e.g. "NY.GDP.MKTP.KD.ZG" (GDP growth %).
    #[live]
    indicator: String,

    /// How many years back from the latest. 0 → the API's default span.
    #[live(30.0)]
    years: f64,

    // Five named slots rather than a list: a Vec<Vec4> is not a scriptable
    // field type, and five is the practical ceiling for a phone-width legend.
    #[live]
    pub color_1: Vec4,
    #[live]
    pub color_2: Vec4,
    #[live]
    pub color_3: Vec4,
    #[live]
    pub color_4: Vec4,
    #[live]
    pub color_5: Vec4,
    /// The zero rule — PlotView has no such field; StockPlot declares its own
    /// baseline the same way.
    #[live]
    pub baseline_color: Vec4,
    #[live(2.0)]
    line_width: f32,
    #[live(true)]
    show_legend: bool,
    #[live(true)]
    show_zero_line: bool,

    // ---- fetched series, cached per URL ----
    #[rust]
    url: String,
    #[rust]
    series: Vec<CountrySeries>,
    #[rust]
    loaded: bool,
    #[rust]
    failed: bool,

    // ---- redraw pump while the async fetch is pending ----
    #[rust]
    pump: NextFrame,
    #[rust]
    last_epoch: u64,
}

impl Widget for IndicatorPlot {
    fn handle_event(&mut self, cx: &mut Cx, event: &Event, _scope: &mut Scope) {
        // Same pump as StockPlot: while our URL is pending, redraw whenever any
        // fetch lands and let draw_walk retry ours. Gated on a non-empty url so
        // a card that clears its countries lets the pump lapse.
        if self.pump.is_event(event).is_some()
            && !self.url.is_empty()
            && !self.loaded
            && !self.failed
        {
            let epoch = cx.script_data_fetch_epoch();
            if epoch != self.last_epoch {
                self.last_epoch = epoch;
                self.redraw(cx);
            } else {
                self.pump = cx.new_next_frame();
            }
        }
        self.plot_view.handle_plot_event(cx, event);
    }

    fn draw_walk(&mut self, cx: &mut Cx2d, _scope: &mut Scope, walk: Walk) -> DrawStep {
        self.ensure_data(cx);

        let drawable = self.series.iter().any(|s| s.values.len() >= 2);
        if drawable {
            // ONE viewport across every series, or two countries would be drawn
            // to different scales on the same axis — the comparison the card
            // exists to make would be a lie.
            let (mut x0, mut x1) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY);
            for s in &self.series {
                for (&x, &y) in s.years.iter().zip(s.values.iter()) {
                    x0 = x0.min(x);
                    x1 = x1.max(x);
                    y0 = y0.min(y);
                    y1 = y1.max(y);
                }
            }
            if x1 <= x0 {
                x1 = x0 + 1.0;
            }
            if y1 <= y0 {
                y1 = y0 + 1.0;
            }
            let pad = (y1 - y0) * 0.10;
            self.plot_view.set_viewport(x0, x1, y0 - pad, y1 + pad);
        }

        self.plot_view.begin(cx, walk);
        if drawable {
            self.draw_grid_and_ticks(cx);
            self.draw_all_series(cx);
            if self.show_legend {
                self.draw_legend(cx);
            }
        } else {
            let pr = self.plot_view.plot_rect().clone();
            let color = self.plot_view.text_color;
            let label = if self.failed {
                "no data for that indicator"
            } else {
                "\u{2014}"
            };
            self.plot_view.draw_text_centered_px(
                cx,
                pr.pos.x + pr.size.x * 0.5,
                pr.pos.y + pr.size.y * 0.5,
                label,
                color,
                12.0,
            );
        }
        self.plot_view.end(cx);
        DrawStep::done()
    }
}

impl IndicatorPlot {
    fn ensure_data(&mut self, cx: &mut Cx) {
        let codes = sanitize_codes(&self.countries);
        let indicator = sanitize_indicator(&self.indicator);
        if codes.is_empty() || indicator.is_empty() {
            if !self.url.is_empty() {
                self.url.clear();
                self.loaded = false;
                self.failed = false;
                self.series.clear();
            }
            return;
        }
        let url = worldbank_url(&codes, &indicator, self.years);
        if url != self.url {
            self.url = url;
            self.loaded = false;
            self.failed = false;
            self.series.clear();
        }
        if self.loaded || self.failed {
            return;
        }
        match cx.script_data_fetch(&self.url) {
            Some(bytes) => {
                if self.parse(&bytes) {
                    self.loaded = true;
                } else {
                    // A 2xx body with no usable series is terminal: the cache
                    // serves the same bytes forever, so re-parsing cannot help.
                    self.failed = true;
                }
            }
            None if cx.script_data.resources.data_fetch_failed_terminally(&self.url) => {
                self.failed = true;
            }
            None => {
                self.last_epoch = cx.script_data_fetch_epoch();
                self.pump = cx.new_next_frame();
            }
        }
    }

    /// World Bank answers `[meta, rows]`, newest year first, one row per
    /// (country, year), with `value: null` for years it has no observation.
    /// Series come back in the ORDER THE CARD ASKED, not the order the API
    /// happened to answer — the legend colour must mean the same thing as the
    /// card's own list of countries.
    fn parse(&mut self, bytes: &[u8]) -> bool {
        self.series.clear();
        let root: serde_json::Value = match serde_json::from_slice(bytes) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let rows = match root.get(1).and_then(|r| r.as_array()) {
            Some(r) if !r.is_empty() => r,
            _ => return false,
        };
        let asked = sanitize_codes(&self.countries);
        let mut by_code: Vec<CountrySeries> = asked
            .iter()
            .map(|_| CountrySeries::default())
            .collect();
        for row in rows {
            let iso3 = row
                .get("countryiso3code")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_ascii_uppercase();
            let Some(slot) = asked.iter().position(|c| *c == iso3) else {
                continue;
            };
            let (Some(year), Some(value)) = (
                row.get("date")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok()),
                row.get("value").and_then(|v| v.as_f64()),
            ) else {
                continue;
            };
            if !year.is_finite() || !value.is_finite() {
                continue;
            }
            let s = &mut by_code[slot];
            if s.name.is_empty() {
                s.name = row
                    .pointer("/country/value")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&iso3)
                    .to_string();
            }
            s.years.push(year);
            s.values.push(value);
        }
        // Ascending by year: the API answers newest-first and a polyline drawn
        // in that order is the same shape mirrored, which reads as a different
        // history.
        for s in &mut by_code {
            let mut pairs: Vec<(f64, f64)> =
                s.years.iter().copied().zip(s.values.iter().copied()).collect();
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            s.years = pairs.iter().map(|p| p.0).collect();
            s.values = pairs.iter().map(|p| p.1).collect();
        }
        self.series = by_code;
        self.series.iter().any(|s| s.values.len() >= 2)
    }

    fn draw_grid_and_ticks(&mut self, cx: &mut Cx2d) {
        let vp = self.plot_view.viewport().clone();
        let pr = self.plot_view.plot_rect().clone();
        let text_color = self.plot_view.text_color;
        let size = self.plot_view.tick_font_size;

        if self.plot_view.show_grid || self.plot_view.show_ticks {
            for ty in nice_ticks(vp.y_min, vp.y_max, 3) {
                let (_, py) = self.plot_view.tdata_to_px(0.0, ty);
                if self.plot_view.show_grid {
                    self.plot_view.draw_grid_line_h_px(cx, py as f64);
                }
                if self.plot_view.show_ticks {
                    let label = format_value(ty);
                    self.plot_view.draw_text_px(
                        cx,
                        pr.pos.x + pr.size.x + 6.0,
                        py as f64 - size as f64 * 0.6,
                        &label,
                        text_color,
                        size,
                    );
                }
            }
        }

        // Zero is a real reading on a growth chart — below it is contraction —
        // so it gets its own brighter rule rather than blending into the grid.
        if self.show_zero_line && vp.y_min < 0.0 && vp.y_max > 0.0 {
            let (_, pz) = self.plot_view.tdata_to_px(0.0, 0.0);
            let c = self.baseline_color;
            let x0 = pr.pos.x as f32;
            let x1 = (pr.pos.x + pr.size.x) as f32;
            self.plot_view.set_color(c);
            self.plot_view.line_px(x0, pz, x1, pz, 1.0);
        }

        // First / middle / last YEAR under the plot.
        if self.plot_view.show_ticks {
            let ly = pr.pos.y + pr.size.y + 4.0;
            let y0 = vp.x_min.round() as i64;
            let y1 = vp.x_max.round() as i64;
            let ym = ((vp.x_min + vp.x_max) * 0.5).round() as i64;
            let est = |s: &str| s.len() as f64 * size as f64 * 0.55;
            let first = y0.to_string();
            let mid = ym.to_string();
            let last = y1.to_string();
            self.plot_view
                .draw_text_px(cx, pr.pos.x, ly, &first, text_color, size);
            let mid_w = est(&mid);
            self.plot_view.draw_text_px(
                cx,
                pr.pos.x + pr.size.x * 0.5 - mid_w * 0.5,
                ly,
                &mid,
                text_color,
                size,
            );
            let last_w = est(&last);
            self.plot_view.draw_text_px(
                cx,
                pr.pos.x + pr.size.x - last_w,
                ly,
                &last,
                text_color,
                size,
            );
        }

        if self.plot_view.show_border {
            self.plot_view.draw_plot_border(cx);
        }
    }

    fn draw_all_series(&mut self, cx: &mut Cx2d) {
        let _ = cx;
        let width = self.line_width;
        let series = self.series.clone();
        for (i, s) in series.iter().enumerate() {
            if s.values.len() < 2 {
                continue;
            }
            let color = self.color_at(i);
            self.plot_view
                .draw_polyline_data(&s.years, &s.values, color, width, LineStyle::Solid);
        }
    }

    /// Names beside their own colours, under the plot. Without it two lines are
    /// two anonymous shapes and the card would have to caption them in prose.
    fn draw_legend(&mut self, cx: &mut Cx2d) {
        let pr = self.plot_view.plot_rect().clone();
        let size = self.plot_view.tick_font_size;
        let y = pr.pos.y + pr.size.y + 16.0;
        let mut x = pr.pos.x;
        let series = self.series.clone();
        for (i, s) in series.iter().enumerate() {
            if s.values.len() < 2 || s.name.is_empty() {
                continue;
            }
            let color = self.color_at(i);
            // A short rule in the series' colour, then its name.
            self.plot_view.set_color(color);
            self.plot_view.line_px(
                x as f32,
                y as f32 + size * 0.4,
                x as f32 + 14.0,
                y as f32 + size * 0.4,
                2.0,
            );
            self.plot_view
                .draw_text_px(cx, x + 18.0, y, &s.name, color, size);
            x += 18.0 + s.name.len() as f64 * size as f64 * 0.58 + 14.0;
        }
    }

    fn color_at(&self, i: usize) -> Vec4 {
        match i % 5 {
            0 => self.color_1,
            1 => self.color_2,
            2 => self.color_3,
            3 => self.color_4,
            _ => self.color_5,
        }
    }
}

/// ISO3 codes, uppercased, in the order the card listed them. Anything that is
/// not three letters is dropped rather than sent: the URL is a path segment and
/// a card must not be able to write one.
fn sanitize_codes(raw: &str) -> Vec<String> {
    raw.split(|c| c == ',' || c == ';' || c == ' ')
        .map(|s| s.trim())
        .filter(|s| s.len() == 3 && s.chars().all(|c| c.is_ascii_alphabetic()))
        .map(|s| s.to_ascii_uppercase())
        .take(5)
        .collect()
}

/// A World Bank indicator code is dotted uppercase alphanumerics
/// (`NY.GDP.MKTP.KD.ZG`). Same reasoning as the codes: this lands in a path.
fn sanitize_indicator(raw: &str) -> String {
    let t = raw.trim();
    if !t.is_empty()
        && t.len() <= 32
        && t.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_')
    {
        t.to_ascii_uppercase()
    } else {
        String::new()
    }
}

/// One URL per (countries × indicator × span), so the fetch cache dedupes it
/// across the chart and every scalar helper on the same card.
pub fn worldbank_url(codes: &[String], indicator: &str, years: f64) -> String {
    let span = if years >= 1.0 {
        let n = years.min(120.0) as i64;
        // The API takes an absolute range; anchoring it to a fixed recent year
        // keeps the URL stable frame to frame (a moving "now" would defeat the
        // cache and refetch forever).
        let end = 2025;
        format!("&date={}:{}", end - n + 1, end)
    } else {
        String::new()
    };
    format!(
        "https://api.worldbank.org/v2/country/{}/indicator/{}?format=json&per_page=600{}",
        codes.join(";"),
        indicator,
        span
    )
}

/// Axis labels: a growth rate wants a decimal, a GDP total wants a magnitude.
fn format_value(v: f64) -> String {
    let a = v.abs();
    if a >= 1e12 {
        format!("{:.1}T", v / 1e12)
    } else if a >= 1e9 {
        format!("{:.1}B", v / 1e9)
    } else if a >= 1e6 {
        format!("{:.1}M", v / 1e6)
    } else if a >= 100.0 {
        format!("{:.0}", v)
    } else {
        format!("{:.1}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_iso3_and_ordered_as_asked() {
        assert_eq!(sanitize_codes("chn,ind"), vec!["CHN", "IND"]);
        assert_eq!(sanitize_codes("CHN; IND"), vec!["CHN", "IND"]);
        // Not three letters, or not letters at all: dropped, never sent.
        assert!(sanitize_codes("../etc/passwd").is_empty());
        assert_eq!(sanitize_codes("CHN,XX,IND"), vec!["CHN", "IND"]);
    }

    #[test]
    fn an_indicator_code_cannot_carry_a_path() {
        assert_eq!(sanitize_indicator("ny.gdp.mktp.kd.zg"), "NY.GDP.MKTP.KD.ZG");
        assert!(sanitize_indicator("../../secret").is_empty());
        assert!(sanitize_indicator("a b").is_empty());
    }

    #[test]
    fn the_url_is_stable_for_the_same_request() {
        let codes = sanitize_codes("CHN,IND");
        let a = worldbank_url(&codes, "NY.GDP.MKTP.KD.ZG", 30.0);
        let b = worldbank_url(&codes, "NY.GDP.MKTP.KD.ZG", 30.0);
        assert_eq!(a, b, "a moving span would defeat the fetch cache");
        assert!(a.contains("country/CHN;IND/"), "{a}");
        assert!(a.contains("date=1996:2025"), "{a}");
    }
}
