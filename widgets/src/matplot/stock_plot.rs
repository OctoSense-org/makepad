// StockPlot — a real line/area price chart for generated stock cards, built on
// the PlotView cartesian engine vendored from mofa-org/makepad-matplot (MIT).
//
// The card DSL only names a ticker and a range:
//
//     StockPlot{ symbol: "TSLA" range: "1d" }
//
// and the widget does the rest: it builds the SAME Yahoo chart-API URL as
// `sys.stockbar`/`sys.stockrange` (crate::splash::yahoo_chart_url) and loads it
// through `Cx::script_data_fetch`, so one URL-deduped request per symbol×range
// serves the plot AND every scalar helper on the card. While the async fetch is
// pending it renders a dim "—" placeholder and arms a NextFrame pump that
// watches the data-fetch epoch; when the response lands it redraws itself with
// the full series — no card re-evaluation required.
//
// Rendering (all on PlotView's DrawVector layer): translucent area fill under a
// 2dp close-price line, auto-colored green/red by the range's first→last
// direction (matching the card convention driven by sys.stockrange "up"), a
// dashed baseline at the range's first close, hairline y-grid with right-edge
// price labels, and three time labels under the plot (HH:MM exchange-local for
// intraday ranges, M/D otherwise — from the same response's timestamps).

use crate::matplot::plot_view::{nice_ticks, PlotView};
use crate::matplot::types::LineStyle;
use crate::splash::{civil_from_days, yahoo_chart_url};
use crate::{makepad_derive_widget::*, makepad_draw::*, widget::*};

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.*

    mod.widgets.StockPlotBase = #(StockPlot::register_widget(vm))

    mod.widgets.StockPlot = set_type_default() do mod.widgets.StockPlotBase{
        width: Fill
        height: 160

        up_color: #x30d158
        down_color: #xff453a
        baseline_color: #xffffff30
        grid_color: #xffffff14
        text_color: #xffffff59
        border_color: #x00000000
        show_border: false
        show_grid: true
        show_ticks: true
        tick_font_size: 9.0
        plot_margin: Inset{left: 6.0, top: 8.0, right: 46.0, bottom: 18.0}

        draw_bg +: {
            draw_depth: 0.0
            color: #x00000000
        }

        draw_grid +: {
            draw_depth: 0.1
        }

        draw_vector +: {
            draw_depth: 2.0
        }

        draw_text +: {
            draw_depth: 3.0
            text_style: theme.font_regular{}
        }
    }
}

#[derive(Script, ScriptHook, Widget)]
pub struct StockPlot {
    #[source]
    source: ScriptObjectRef,
    #[deref]
    plot_view: PlotView,

    /// Ticker symbol, e.g. "AAPL". Empty → placeholder (nothing is fetched).
    #[live]
    pub symbol: String,
    /// Range token, same vocabulary as sys.stockbar: "1d" (default when empty
    /// or unknown), "1w", "1m", "6m", "1y".
    #[live]
    pub range: String,
    /// Line/area color when the range closed up (last >= first close).
    #[live]
    pub up_color: Vec4,
    /// Line/area color when the range closed down.
    #[live]
    pub down_color: Vec4,
    /// Dashed reference line at the range's first close.
    #[live]
    pub baseline_color: Vec4,
    /// Alpha of the area fill under the price line (line color, faded).
    #[live(0.16)]
    pub fill_alpha: f32,
    #[live(2.0)]
    pub line_width: f32,
    #[live(true)]
    pub show_baseline: bool,

    // ---- fetched series, cached per URL (symbol×range) ----
    #[rust]
    url: String,
    #[rust]
    closes: Vec<f64>,
    #[rust]
    stamps: Vec<f64>,
    #[rust]
    gmtoff: f64,
    #[rust]
    has_time: bool,
    #[rust]
    loaded: bool,

    // ---- redraw pump while the async fetch is pending ----
    #[rust]
    pump: NextFrame,
    #[rust]
    last_epoch: u64,
}

impl Widget for StockPlot {
    fn handle_event(&mut self, cx: &mut Cx, event: &Event, _scope: &mut Scope) {
        // While the fetch is pending, watch the global data-fetch epoch each
        // frame; when ANY fetch lands, redraw — draw_walk retries our URL and
        // re-arms the pump if it is still the one pending. Mirrors the Splash
        // live-data pump, but scoped to this widget (a plot-only card needs no
        // body re-evaluation to fill in).
        if self.pump.is_event(event).is_some() && !self.loaded {
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

        let n = self.closes.len();
        if n >= 2 {
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for &v in &self.closes {
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
            }
            if mx <= mn {
                mx = mn + 1.0;
            }
            let pad = (mx - mn) * 0.06;
            self.plot_view
                .set_viewport(0.0, (n - 1) as f64, mn - pad, mx + pad);
        }

        self.plot_view.begin(cx, walk);
        if n >= 2 {
            self.draw_grid_and_ticks(cx);
            self.draw_series(cx);
        } else {
            // Loading (or empty symbol): the standard dim placeholder; the
            // pump redraws us into the real chart when the fetch lands.
            let pr = self.plot_view.plot_rect().clone();
            let color = self.plot_view.text_color;
            self.plot_view.draw_text_centered_px(
                cx,
                pr.pos.x + pr.size.x * 0.5,
                pr.pos.y + pr.size.y * 0.5,
                "—",
                color,
                12.0,
            );
        }
        self.plot_view.end(cx);
        DrawStep::done()
    }
}

impl StockPlot {
    /// Resolve the Yahoo chart URL for the current symbol×range and (re)load
    /// the close series through the shared script-data-fetch cache.
    fn ensure_data(&mut self, cx: &mut Cx) {
        if self.symbol.trim().is_empty() {
            return;
        }
        let url = yahoo_chart_url(&self.symbol, &self.range);
        if url != self.url {
            self.url = url;
            self.loaded = false;
            self.closes.clear();
            self.stamps.clear();
        }
        if self.loaded {
            return;
        }
        match cx.script_data_fetch(&self.url) {
            Some(bytes) => {
                self.parse(&bytes);
                self.loaded = true;
            }
            None => {
                // Pending (or retrying): arm the epoch watch.
                self.last_epoch = cx.script_data_fetch_epoch();
                self.pump = cx.new_next_frame();
            }
        }
    }

    /// Parse Yahoo chart JSON into aligned (timestamp, close) pairs, skipping
    /// null closes (Yahoo pads intraday series with nulls).
    fn parse(&mut self, bytes: &[u8]) {
        self.closes.clear();
        self.stamps.clear();
        self.gmtoff = 0.0;
        self.has_time = false;
        let root: serde_json::Value = match serde_json::from_slice(bytes) {
            Ok(v) => v,
            Err(_) => return,
        };
        let result = match root.pointer("/chart/result/0") {
            Some(r) => r,
            None => return,
        };
        self.gmtoff = result
            .pointer("/meta/gmtoffset")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let closes = result
            .pointer("/indicators/quote/0/close")
            .and_then(|c| c.as_array());
        let stamps = result.pointer("/timestamp").and_then(|t| t.as_array());
        match (closes, stamps) {
            (Some(cl), Some(ts)) => {
                self.has_time = true;
                for (c, t) in cl.iter().zip(ts.iter()) {
                    if let (Some(c), Some(t)) = (c.as_f64(), t.as_f64()) {
                        self.closes.push(c);
                        self.stamps.push(t);
                    }
                }
            }
            (Some(cl), None) => {
                for (i, c) in cl.iter().enumerate() {
                    if let Some(c) = c.as_f64() {
                        self.closes.push(c);
                        self.stamps.push(i as f64);
                    }
                }
            }
            _ => {}
        }
    }

    fn draw_grid_and_ticks(&mut self, cx: &mut Cx2d) {
        let vp = self.plot_view.viewport().clone();
        let pr = self.plot_view.plot_rect().clone();
        let text_color = self.plot_view.text_color;
        let size = self.plot_view.tick_font_size;

        // Horizontal hairlines with price labels in the right margin.
        if self.plot_view.show_grid || self.plot_view.show_ticks {
            for ty in nice_ticks(vp.y_min, vp.y_max, 3) {
                let (_, py) = self.plot_view.tdata_to_px(0.0, ty);
                if self.plot_view.show_grid {
                    self.plot_view.draw_grid_line_h_px(cx, py as f64);
                }
                if self.plot_view.show_ticks {
                    let label = format_price(ty);
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

        // Three interior vertical hairlines (quarter positions).
        if self.plot_view.show_grid {
            for f in [0.25, 0.5, 0.75] {
                let tx = vp.x_min + vp.x_range() * f;
                let (px, _) = self.plot_view.tdata_to_px(tx, 0.0);
                self.plot_view.draw_grid_line_v_px(cx, px as f64);
            }
        }

        // First / middle / last time labels under the plot, from the fetched
        // timestamps: HH:MM (exchange-local) for intraday spans, M/D otherwise.
        if self.plot_view.show_ticks && self.has_time && self.stamps.len() >= 2 {
            let n = self.stamps.len();
            let intraday = self.stamps[n - 1] - self.stamps[0] <= 2.0 * 86_400.0;
            let ly = pr.pos.y + pr.size.y + 4.0;
            let first = self.format_stamp(self.stamps[0], intraday);
            let mid = self.format_stamp(self.stamps[n / 2], intraday);
            let last = self.format_stamp(self.stamps[n - 1], intraday);
            let est = |s: &str| s.len() as f64 * size as f64 * 0.55;
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

    fn draw_series(&mut self, cx: &mut Cx2d) {
        let _ = cx;
        let n = self.closes.len();
        let vp = self.plot_view.viewport().clone();
        let up = self.closes[n - 1] >= self.closes[0];
        let color = if up { self.up_color } else { self.down_color };

        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();

        // Area fill: the close path closed down to the viewport floor.
        let mut pxs = xs.clone();
        pxs.push((n - 1) as f64);
        pxs.push(0.0);
        let mut pys = self.closes.clone();
        pys.push(vp.y_min);
        pys.push(vp.y_min);
        let fill = Vec4 {
            x: color.x,
            y: color.y,
            z: color.z,
            w: self.fill_alpha,
        };
        self.plot_view.fill_polygon_data(&pxs, &pys, fill);

        // Dashed baseline at the range's first close.
        if self.show_baseline {
            let baseline_color = self.baseline_color;
            self.plot_view
                .draw_hline(self.closes[0], baseline_color, 1.0, LineStyle::Dashed);
        }

        // The price line, with a marker on the latest close.
        let line_width = self.line_width;
        self.plot_view
            .draw_polyline_data(&xs, &self.closes, color, line_width, LineStyle::Solid);
        let (lpx, lpy) = self.plot_view.data_to_px((n - 1) as f64, self.closes[n - 1]);
        self.plot_view.fill_circle_px(lpx, lpy, 3.0, color);
    }

    /// Format a Unix timestamp for an x tick: exchange-local "HH:MM" for
    /// intraday spans, "M/D" for longer ranges.
    fn format_stamp(&self, ts: f64, intraday: bool) -> String {
        let t = (ts + self.gmtoff) as i64;
        if intraday {
            let sod = t.rem_euclid(86_400);
            format!("{:02}:{:02}", sod / 3600, (sod % 3600) / 60)
        } else {
            let (_, m, d) = civil_from_days(t.div_euclid(86_400));
            format!("{m}/{d}")
        }
    }
}

/// Price tick label with sensible precision for the narrow right margin.
fn format_price(v: f64) -> String {
    let a = v.abs();
    if a >= 1000.0 {
        format!("{v:.0}")
    } else if a >= 100.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}
