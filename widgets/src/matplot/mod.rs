// matplot — real plotting for generated cards.
//
// `types.rs` and `plot_view.rs` are vendored from mofa-org/makepad-matplot
// (https://github.com/mofa-org/makepad-matplot, commit b510f6b, MIT license):
// the PlotView cartesian engine (viewport/scales, nice ticks, DrawVector
// line/area/marker/dash helpers, queued text) from its Makepad 2.0 "Splash"
// port. `stock_plot.rs` is this fork's card-facing widget on top of it — a
// live Yahoo price line/area chart driven purely by DSL properties
// (`StockPlot{ symbol: "TSLA" range: "1d" }`), sharing the sys.stockbar /
// sys.stockrange fetch cache.

pub mod plot_view;
pub mod stock_plot;
pub mod types;

use crate::ScriptVm;

pub fn script_mod(vm: &mut ScriptVm) {
    crate::matplot::types::script_mod(vm);
    crate::matplot::plot_view::script_mod(vm);
    crate::matplot::stock_plot::script_mod(vm);
}
