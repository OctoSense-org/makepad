//! The scrollback search bar of a terminal pane: its keys, its query, the
//! highlight colours and its drawing. The matching is `crate::search`; the
//! key bindings are `crate::search::search_key`.
//!
//! While the bar is open it has the keyboard: every key and all typed or
//! pasted text go to the query, none to the program. Escape closes it and
//! leaves the view where it is.

use makepad_widgets::*;

use super::MpTerm;
use crate::search::{self, SearchKey, MARK_CURRENT};
use crate::settings as term_settings;
use crate::term::color::Rgb;
use crate::term::screen::Screen;
use crate::term::terminal::{ActiveScreen, Terminal};
use crate::term::unicode::char_width;

/// The bar's state beside the search itself.
#[derive(Default)]
pub(super) struct SearchUi {
    pub open: bool,
    /// The query is shown selected: typing replaces it (a reopened bar).
    replace: bool,
    bar: Rect,
    toggle: Rect,
    /// The caret's bottom-left, relative to the terminal's rect, for the
    /// IME.
    pub caret: Option<DVec2>,
    /// Asked for while history is still being read: a frame reads more.
    next_frame: NextFrame,
    /// The end of history (`evicted + scrollback.len()`) at the last sync:
    /// a view scrolled back stays on its rows as output arrives.
    history_end: Option<u64>,
}

/// (background, text) colours of a match and of the current match.
pub(super) type SearchColors = [(Vec4f, Vec4f); 2];

fn text_cells(text: &str) -> usize {
    text.chars().map(|c| char_width(c as u32) as usize).sum()
}

fn mix(a: Vec4f, b: Vec4f, t: f32) -> Vec4f {
    vec4(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        a.w + (b.w - a.w) * t,
    )
}

impl MpTerm {
    /// The bar's share of an input event: true when it took it.
    pub(super) fn search_event(&mut self, cx: &mut Cx, hit: &Hit) -> bool {
        let open = self.search_ui.open;
        match hit {
            Hit::KeyDown(e) => {
                if let Some(key) = search::search_key(e, open) {
                    self.search_command(cx, key);
                    return true;
                }
                open
            }
            Hit::KeyUp(_) => open,
            Hit::TextInput(e) if open => {
                if !e.replace_last {
                    let text: String = e.input.chars().filter(|c| !c.is_control()).collect();
                    if !text.is_empty() {
                        let mut query = if self.search_ui.replace {
                            String::new()
                        } else {
                            self.search.query().to_string()
                        };
                        query.push_str(&text);
                        self.search_set(cx, query);
                    }
                }
                true
            }
            Hit::FingerDown(e) if open && self.search_ui.bar.contains(e.abs) => {
                cx.set_key_focus(self.area);
                if self.search_ui.toggle.contains(e.abs) {
                    self.search_command(cx, SearchKey::ToggleRegex);
                }
                true
            }
            _ => false,
        }
    }

    fn search_command(&mut self, cx: &mut Cx, key: SearchKey) {
        let query = self.search.query().to_string();
        match key {
            SearchKey::Open => {
                if !self.search_ui.open {
                    self.search_ui.open = true;
                    self.search_ui.replace = !query.is_empty();
                    self.search_ui.history_end = None;
                    self.search.invalidate();
                }
            }
            SearchKey::Close => {
                self.search_ui.open = false;
                self.search_ui.replace = false;
                self.search_ui.caret = None;
            }
            SearchKey::Older | SearchKey::Newer => {
                self.search_ui.replace = false;
                let older = key == SearchKey::Older;
                self.with_search_terminal(|term, terminal| {
                    term.search_sync(terminal, false);
                    let bottom = term.view_bottom(terminal.screen());
                    term.search.step(older, bottom);
                    term.search_sync(terminal, false);
                });
            }
            SearchKey::ToggleRegex => {
                let regex = !self.search.regex();
                self.search.set_query(&query, regex);
                self.search_ui.replace = false;
            }
            SearchKey::DeleteChar if self.search_ui.replace => self.search_set(cx, String::new()),
            SearchKey::DeleteChar => {
                let mut query = query;
                query.pop();
                self.search_set(cx, query);
            }
            SearchKey::DeleteWord if self.search_ui.replace => self.search_set(cx, String::new()),
            SearchKey::DeleteWord => self.search_set(cx, search::delete_word(&query)),
            SearchKey::Clear => self.search_set(cx, String::new()),
        }
        self.redraw(cx);
    }

    fn search_set(&mut self, cx: &mut Cx, query: String) {
        let regex = self.search.regex();
        self.search.set_query(&query, regex);
        self.search_ui.replace = false;
        // Matched now, so the view moves to the first match while typing.
        self.with_search_terminal(|term, terminal| {
            term.search_sync(terminal, false);
        });
        self.redraw(cx);
    }

    fn with_search_terminal(&mut self, f: impl FnOnce(&mut MpTerm, &Terminal)) {
        if let Some(session) = self.session.take() {
            f(self, &session.terminal);
            self.session = Some(session);
        }
    }

    /// The last row in view, absolute.
    fn view_bottom(&self, screen: &Screen) -> u64 {
        let top = screen.scrollback.len().saturating_sub(self.view_offset);
        screen.absolute_of_virtual(top + screen.rows - 1)
    }

    /// A frame's search work, before the rows are drawn: the matches follow
    /// the output, and while history is still being read the next frame
    /// reads more.
    pub(super) fn search_frame(&mut self, cx: &mut Cx2d, terminal: &Terminal, changed: bool) {
        if self.search_sync(terminal, changed) {
            self.search_ui.next_frame = cx.new_next_frame();
        }
    }

    pub(super) fn search_next_frame(&mut self, cx: &mut Cx, event: &Event) {
        if self.search_ui.next_frame.is_event(event).is_some() {
            self.redraw(cx);
        }
    }

    /// Bring the search up to date with `terminal` and move the view to a
    /// newly current match. `changed`: output arrived since the last frame.
    /// True while history is still being read.
    pub(super) fn search_sync(&mut self, terminal: &Terminal, changed: bool) -> bool {
        if !self.search_ui.open {
            return false;
        }
        let screen = terminal.screen();
        // While searching, a view scrolled back keeps showing the same rows
        // (the match being looked at) as output pushes history up.
        let history_end = screen.evicted + screen.scrollback.len() as u64;
        if let Some(before) = self.search_ui.history_end.replace(history_end) {
            if self.view_offset > 0 && history_end > before {
                let grown = (history_end - before) as usize;
                self.view_offset = (self.view_offset + grown).min(screen.scrollback.len());
            }
        }
        let alternate = matches!(terminal.active, ActiveScreen::Alternate);
        let more = self
            .search
            .sync(screen, alternate, changed, self.view_bottom(screen));
        if self.search.take_reveal() {
            if let Some(current) = self.search.current() {
                self.view_offset = search::reveal_offset(screen, self.view_offset, &current);
            }
        }
        more
    }

    /// Highlight colours while the bar is open.
    pub(super) fn search_colors(&self) -> Option<SearchColors> {
        if !self.search_ui.open {
            return None;
        }
        let ((bg, fg), (cur_bg, cur_fg)) = term_settings::search_colors(&self.settings);
        let v = |c: Rgb| Self::rgb_to_vec4(c, 1.0);
        Some([(v(bg), v(fg)), (v(cur_bg), v(cur_fg))])
    }

    /// Row `abs`'s search marks (`crate::search::row_marks`), or none.
    pub(super) fn search_row_marks(&self, abs: u64, cols: usize, marks: &mut Vec<u8>) {
        if self.search_ui.open {
            self.search.row_marks(abs, cols, marks);
        } else {
            marks.clear();
        }
    }

    /// A marked cell's (background, text) colours.
    pub(super) fn search_cell_colors(
        colors: Option<SearchColors>,
        mark: u8,
    ) -> Option<(Vec4f, Vec4f)> {
        match (colors, mark) {
            (_, 0) | (None, _) => None,
            (Some(c), MARK_CURRENT) => Some(c[1]),
            (Some(c), _) => Some(c[0]),
        }
    }

    /// Text as the bar draws it (proportionally, not on the cell grid).
    fn bar_text_width(&self, cx: &mut Cx2d, text: &str) -> f64 {
        self.draw_text
            .prepare_single_line_run(cx, text)
            .map_or(text_cells(text) as f64 * self.cell_w, |run| {
                run.width_in_lpxs as f64
            })
    }

    /// The bar, top right over the terminal.
    pub(super) fn draw_search_bar(&mut self, cx: &mut Cx2d, screen: &Screen, fg: Rgb, bg: Rgb) {
        if !self.search_ui.open {
            self.search_ui.caret = None;
            return;
        }
        let fg = Self::rgb_to_vec4(fg, 1.0);
        let bg = Self::rgb_to_vec4(bg, 1.0);
        let panel = mix(bg, fg, 0.12);
        let border = mix(bg, fg, 0.4);
        let dim = mix(panel, fg, 0.6);
        let (cw, ch) = (self.cell_w, self.cell_h);
        let pad = (ch * 0.3).round().max(3.0);
        let margin = 8.0;
        let status = self.search.status(screen);
        let status_text = status.text();
        let status_w = 14.0 * cw;
        let toggle_w = 3.0 * cw;
        let width = (26.0 * cw + status_w + toggle_w + 2.0 * pad + 2.0 * cw)
            .min(self.rect.size.x - 2.0 * margin)
            .max(0.0);
        let height = ch + 2.0 * pad;
        let bar = Rect {
            pos: dvec2(
                self.rect.pos.x + self.rect.size.x - margin - width,
                self.rect.pos.y + margin,
            ),
            size: dvec2(width, height),
        };
        let toggle = Rect {
            pos: dvec2(bar.pos.x + width - pad - toggle_w, bar.pos.y + pad * 0.5),
            size: dvec2(toggle_w, ch + pad),
        };
        let query_x = bar.pos.x + pad;
        let query_cells = ((toggle.pos.x - cw - status_w - query_x) / cw)
            .floor()
            .max(1.0) as usize;
        // The end of a long query, where the caret is.
        let query = self.search.query();
        let mut shown: String = query.to_string();
        if text_cells(&shown) > query_cells {
            let mut tail: Vec<char> = Vec::new();
            let mut used = 1;
            for c in query.chars().rev() {
                used += char_width(c as u32) as usize;
                if used > query_cells {
                    break;
                }
                tail.push(c);
            }
            shown = std::iter::once('…').chain(tail.into_iter().rev()).collect();
        }
        let text_y = bar.pos.y + pad;
        let shown_w = self.bar_text_width(cx, &shown);
        let regex = self.search.regex();
        let [_, (accent_bg, accent_fg)] = self.search_colors().unwrap_or([(fg, bg); 2]);

        self.draw_cell_bg.new_draw_call(cx);
        self.draw_cell_bg.color = border;
        self.draw_cell_bg.draw_abs(cx, bar);
        self.draw_cell_bg.color = panel;
        self.draw_cell_bg.draw_abs(
            cx,
            Rect {
                pos: bar.pos + dvec2(1.0, 1.0),
                size: bar.size - dvec2(2.0, 2.0),
            },
        );
        if self.search_ui.replace && !shown.is_empty() {
            self.draw_cell_bg.color = mix(panel, fg, 0.3);
            self.draw_cell_bg.draw_abs(
                cx,
                Rect {
                    pos: dvec2(query_x, text_y),
                    size: dvec2(shown_w, ch),
                },
            );
        }
        let focused = cx.has_key_focus(self.area);
        if focused {
            self.draw_cell_bg.color = fg;
            self.draw_cell_bg.draw_abs(
                cx,
                Rect {
                    pos: dvec2(query_x + shown_w, text_y),
                    size: dvec2((cw * 0.12).max(1.5), ch),
                },
            );
        }
        self.draw_cell_bg.color = if regex {
            accent_bg
        } else {
            mix(panel, fg, 0.08)
        };
        self.draw_cell_bg.draw_abs(cx, toggle);

        self.draw_text.new_draw_call(cx);
        let color = self.draw_text.color;
        if shown.is_empty() {
            self.draw_text.color = dim;
            self.draw_text
                .draw_abs(cx, dvec2(query_x + cw * 0.5, text_y), "Search");
        } else {
            self.draw_text.color = fg;
            self.draw_text.draw_abs(cx, dvec2(query_x, text_y), &shown);
        }
        if !status_text.is_empty() {
            let w = self.bar_text_width(cx, &status_text);
            self.draw_text.color =
                if status.error.is_some() || status.total == 0 && !status.scanning {
                    mix(dim, vec4(0.9, 0.3, 0.3, 1.0), 0.6)
                } else {
                    dim
                };
            self.draw_text
                .draw_abs(cx, dvec2(toggle.pos.x - cw - w, text_y), &status_text);
        }
        self.draw_text.color = if regex { accent_fg } else { dim };
        self.draw_text
            .draw_abs(cx, dvec2(toggle.pos.x + cw * 0.5, text_y), ".*");
        self.draw_text.color = color;

        self.search_ui.bar = bar;
        self.search_ui.toggle = toggle;
        self.search_ui.caret = Some(dvec2(query_x + shown_w, text_y + ch) - self.rect.pos);
    }
}
