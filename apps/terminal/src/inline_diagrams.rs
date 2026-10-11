//! Mermaid is a presentation of transcript cells, never input to the PTY.
//! Keep the original rows (including soft wraps) so cursor addressing, search,
//! selection, copy and interactive CLI redraws retain their terminal semantics.
use crate::diagram_preview::{is_header, looks_like_statement, MAX_SOURCE};
use crate::term::{modes::Mode, screen::Screen, terminal::{ActiveScreen, Terminal}};
use crate::widget::DrawTermBg;
use makepad_markdown_widgets::{content::PreviewImage, diagram};
use makepad_widgets::*;
use std::time::{Duration, Instant};

const MAX_BLOCKS: usize = 4;
const MAX_ROWS: usize = 512;
const QUIET: Duration = Duration::from_millis(250);

struct SourceBlock {
    start: u64,
    end: u64, // exclusive, physical rows
    source: String,
    cells: String,
}

struct Entry {
    block: SourceBlock,
    image: Option<WidgetRef>,
    dimensions: Vec2d,
    failed: bool,
    source_visible: bool,
    zoom: f64,
    fit: bool,
    pan: Vec2d,
    rect: Rect,
    header: Rect,
    canvas: Rect,
    visible: bool,
}

impl Entry {
    fn new(block: SourceBlock) -> Self {
        let fit = block.end - block.start < 10;
        Self { block, image: None, dimensions: dvec2(1.0, 1.0), failed: false,
            source_visible: false, zoom: 1.0, fit, pan: dvec2(0.0, 0.0),
            rect: Rect::default(), header: Rect::default(), canvas: Rect::default(), visible: false }
    }

    fn image_size(&self) -> Vec2d {
        let width = (self.canvas.size.x - 16.0).max(1.0);
        let height = (self.canvas.size.y - 16.0).max(1.0);
        let mut scale = (width / self.dimensions.x).min(1.0);
        if self.fit { scale = scale.min(height / self.dimensions.y); }
        self.dimensions * (scale * self.zoom)
    }

    fn clamp_pan(&mut self) {
        let size = self.image_size();
        self.pan.x = self.pan.x.clamp(0.0, (size.x + 16.0 - self.canvas.size.x).max(0.0));
        self.pan.y = self.pan.y.clamp(0.0, (size.y + 16.0 - self.canvas.size.y).max(0.0));
    }
}

#[derive(Default)]
pub(crate) struct InlineDiagrams {
    entries: Vec<Entry>,
    job: Option<(u64, String, TaskHandle<Result<PreviewImage, String>>)>,
    timer: Timer,
    changed: Option<Instant>,
    scanned: Option<(u64, usize)>,
    enabled: bool,
}

pub(crate) enum InlineClick {
    None,
    Control,
    Select { start: u64, end: u64 },
}

impl InlineDiagrams {
    pub fn clear(&mut self, cx: &mut Cx) {
        cx.stop_timer(self.timer);
        *self = Self::default();
    }

    fn arm(&mut self, cx: &mut Cx) {
        if self.timer.is_empty() { self.timer = cx.start_interval(0.05); }
    }

    /// Run before collecting terminal glyphs: overlaying a quad alone cannot
    /// hide glyphs batched into the terminal's separate draw-call groups.
    pub fn prepare(&mut self, cx: &mut Cx, terminal: &Terminal, top: usize, changed: bool, searching: bool) {
        let modes = &terminal.modes;
        let was_enabled = self.enabled;
        self.enabled = !searching && terminal.active == ActiveScreen::Primary
            && ![Mode::MouseEventAny, Mode::MouseEventButton, Mode::MouseEventNormal, Mode::MouseEventX10]
                .into_iter().any(|m| modes.get(m));
        for entry in &mut self.entries { entry.visible = false; }
        if !self.enabled {
            self.changed = None;
            self.scanned = None;
            return;
        }
        let screen = terminal.screen();
        if changed || !was_enabled {
            self.changed = Some(Instant::now());
            self.scanned = None;
            // Never cover rewritten text with a stale result, even for one
            // debounce frame. Cursor-in-block also reveals editable source.
            self.entries.retain(|entry| unchanged(screen, &entry.block));
            self.arm(cx);
        }
        let viewport = (screen.absolute_of_virtual(top), screen.cols);
        if self.scanned != Some(viewport) && self.changed.is_none_or(|t| t.elapsed() >= QUIET) {
            let mut old = std::mem::take(&mut self.entries);
            self.entries = blocks(screen, top).into_iter().map(|block| {
                let index = old.iter().position(|e| e.block.start == block.start && e.block.source == block.source)
                    .or_else(|| old.iter().position(|e| e.block.source == block.source));
                if let Some(index) = index {
                    let mut entry = old.remove(index);
                    entry.block = block;
                    entry
                } else { Entry::new(block) }
            }).collect();
            // Keep a small working set across scrolling. Source/zoom choices
            // survive leaving the viewport without retaining unbounded textures.
            for entry in old {
                if self.entries.len() == MAX_BLOCKS { break; }
                if unchanged(screen, &entry.block) { self.entries.push(entry); }
            }
            self.scanned = Some(viewport);
            self.changed = None;
            self.poll(cx);
        }
    }

    pub fn covers(&self, row: u64) -> bool {
        self.enabled && self.entries.iter().any(|e| e.image.is_some() && !e.source_visible
            && row >= e.block.start && row < e.block.end)
    }

    pub fn handle_timer(&mut self, cx: &mut Cx, event: &Event) -> bool {
        if self.timer.is_event(event).is_none() { return false; }
        self.poll(cx);
        true
    }

    fn poll(&mut self, cx: &mut Cx) {
        if let Some(result) = self.job.as_mut().and_then(|(_, _, job)| job.try_take()) {
            let (start, source, _) = self.job.take().unwrap();
            if let Some(entry) = self.entries.iter_mut().find(|e| e.block.start == start && e.block.source == source) {
                match result.ok().and_then(Result::ok) {
                    Some(image) => {
                        let widget = cx.with_vm(|vm| {
                            let value = script_eval!(vm, {use mod.widgets.* Image {fit: ImageFit.Stretch}});
                            WidgetRef::script_from_value(vm, value)
                        });
                        if widget.as_image().load_png_from_data(cx, &image.png).is_ok() {
                            entry.dimensions = dvec2(image.width as f64, image.height as f64);
                            entry.image = Some(widget);
                        } else { entry.failed = true; }
                    }
                    None => entry.failed = true, // Unsupported syntax keeps the original text.
                }
            }
        }
        if self.enabled && self.job.is_none() && self.changed.is_none() {
            if let Some(entry) = self.entries.iter().find(|e| e.image.is_none() && !e.failed) {
                let source = entry.block.source.clone();
                let input = source.clone();
                if let Ok(job) = cx.task_pool().try_submit(Lane::Heavy, move || diagram::render("mermaid", &input)) {
                    self.job = Some((entry.block.start, source, job));
                }
            }
        }
        if self.job.is_some() || self.changed.is_some()
            || (self.enabled && self.entries.iter().any(|e| e.image.is_none() && !e.failed)) {
            self.arm(cx);
        } else {
            cx.stop_timer(self.timer);
            self.timer = Timer::default();
        }
    }

    pub fn draw(&mut self, cx: &mut Cx2d, bg: &mut DrawTermBg, text: &mut DrawText,
        viewport: Rect, top: u64, cell_h: f64, selection: Option<((u64, usize), (u64, usize))>) {
        if !self.enabled { return; }
        cx.begin_turtle(Walk::fixed(viewport.size.x, viewport.size.y).with_abs_pos(viewport.pos), Layout::default());
        for entry in &mut self.entries {
            if entry.image.is_none() { continue; }
            let y = viewport.pos.y + (entry.block.start as f64 - top as f64) * cell_h;
            let height = (entry.block.end - entry.block.start) as f64 * cell_h;
            let rect = Rect {pos: dvec2(viewport.pos.x, y), size: dvec2(viewport.size.x, height)};
            let bottom = (rect.pos.y + rect.size.y).min(viewport.pos.y + viewport.size.y);
            let header_y = y.max(viewport.pos.y);
            if bottom <= header_y { continue; }
            entry.visible = true;
            entry.rect = rect;
            let header_h = (cell_h + 12.0).min(height * 0.5);
            entry.header = Rect {pos: dvec2(rect.pos.x, header_y), size: dvec2(rect.size.x, header_h.min(bottom - header_y))};
            // Source mode exposes every original cell, with only a small
            // right-aligned return control instead of a second editor.
            if entry.source_visible {
                entry.header.pos.x += (rect.size.x - 88.0).max(0.0);
                entry.header.size.x = 88.0_f64.min(rect.size.x);
            } else {
                bg.new_draw_call(cx);
                bg.color = vec4(1.0, 1.0, 1.0, 1.0);
                bg.draw_abs(cx, rect);
                entry.canvas = Rect {pos: dvec2(rect.pos.x, y + header_h), size: dvec2(rect.size.x, height - header_h)};
                entry.clamp_pan();
                let size = entry.image_size();
                let pos = entry.canvas.pos + dvec2(((entry.canvas.size.x - size.x) * 0.5).max(8.0), 8.0) - entry.pan;
                let canvas_top = entry.canvas.pos.y.max(entry.header.pos.y + entry.header.size.y);
                cx.begin_turtle(Walk::fixed(entry.canvas.size.x, (bottom - canvas_top).max(0.0))
                    .with_abs_pos(dvec2(entry.canvas.pos.x, canvas_top)), Layout::default());
                let image = entry.image.as_ref().unwrap();
                image.draw_walk_all(cx, &mut Scope::empty(), Walk::fixed(size.x, size.y).with_abs_pos(pos));
                cx.end_turtle();
            }
            // Sticky controls stay available when the start of a large block
            // is above the viewport. They never cover neighbouring output.
            cx.begin_turtle(Walk::fixed(entry.header.size.x, entry.header.size.y).with_abs_pos(entry.header.pos), Layout::default());
            bg.new_draw_call(cx);
            bg.color = vec4(0.10, 0.15, 0.22, 1.0);
            bg.draw_abs(cx, entry.header);
            text.new_draw_call(cx);
            let color = text.color;
            text.color = vec4(0.82, 0.89, 0.98, 1.0);
            let pos = entry.header.pos + dvec2(8.0, 5.0);
            if entry.source_visible { text.draw_abs(cx, pos, "Diagram"); }
            else {
                let selected = selection.is_some_and(|(a, b)| a.0 <= entry.block.start && b.0 >= entry.block.end - 1);
                let title = if selected && entry.header.size.x >= 540.0 {"Mermaid · selected · copy returns source"}
                    else if entry.header.size.x >= 670.0 {"Mermaid · scroll to pan · Shift+scroll transcript"} else {"Mermaid"};
                text.draw_abs(cx, pos, title);
                let right = entry.header.pos.x + entry.header.size.x;
                for (offset, label) in [(224.0, "Source"), (136.0, "−"), (96.0, "+"), (56.0, if entry.fit {"Width"} else {"Fit"})] {
                    text.draw_abs(cx, dvec2(right - offset + 8.0, pos.y), label);
                }
            }
            text.color = color;
            cx.end_turtle();
        }
        cx.end_turtle();
    }

    pub fn click(&mut self, point: Vec2d) -> InlineClick {
        if !self.enabled { return InlineClick::None; }
        for entry in &mut self.entries {
            if !entry.visible || !entry.rect.contains(point) { continue; }
            if entry.header.contains(point) {
                let right = entry.header.pos.x + entry.header.size.x - point.x;
                if entry.source_visible || (136.0..224.0).contains(&right) {
                    entry.source_visible = !entry.source_visible;
                } else if right <= 56.0 { entry.fit = !entry.fit; entry.zoom = 1.0; entry.pan = dvec2(0.0, 0.0); }
                else if right <= 96.0 { entry.zoom = (entry.zoom * 1.25).min(8.0); }
                else if right <= 136.0 { entry.zoom = (entry.zoom / 1.25).max(0.125); }
                return InlineClick::Control;
            }
            if !entry.source_visible { return InlineClick::Select {start: entry.block.start, end: entry.block.end}; }
        }
        InlineClick::None
    }

    pub fn pan(&mut self, e: &FingerScrollEvent) -> bool {
        if !self.enabled || e.modifiers.shift { return false; }
        for entry in &mut self.entries {
            if !entry.visible || entry.source_visible || !entry.rect.contains(e.abs) { continue; }
            let old = entry.pan;
            entry.pan += e.scroll;
            entry.clamp_pan();
            return old != entry.pan; // At the image boundary, scroll the transcript normally.
        }
        false
    }

    pub fn control_at(&self, point: Vec2d) -> bool {
        self.enabled && self.entries.iter().any(|e| e.visible && e.header.contains(point)
            && (e.source_visible || point.x >= e.header.pos.x + e.header.size.x - 224.0))
    }
}

fn unchanged(screen: &Screen, block: &SourceBlock) -> bool {
    let cursor = screen.absolute_of_virtual(screen.scrollback.len() + screen.cursor.y);
    block.start >= screen.evicted && block.end <= screen.absolute_of_virtual(screen.total_rows())
        && !(block.start..block.end).contains(&cursor)
        && screen.selection_text((block.start, 0), (block.end - 1, screen.cols)) == block.cells
}

/// Reconstruct logical lines, preserving spaces across soft wraps. Limit both
/// rows and cells so pathological columns/history cannot monopolize a frame.
fn blocks(screen: &Screen, top: usize) -> Vec<SourceBlock> {
    struct Line { text: String, start: usize, end: usize }
    let budget = (256 * 1024 / screen.cols.max(1)).min(MAX_ROWS * 2);
    let start = top.saturating_sub(MAX_ROWS.min(budget / 2));
    let end = (start + budget).min(screen.total_rows());
    let mut lines = Vec::new();
    let mut line = Line {text: String::new(), start, end: start};
    for index in start..end {
        let Some(row) = screen.row_virtual(index) else { break; };
        for cell in &row.cells { cell.content.push_text(&mut line.text); }
        line.end = index + 1;
        if !row.wrapped {
            line.text.truncate(line.text.trim_end().len());
            lines.push(line);
            line = Line {text: String::new(), start: index + 1, end: index + 1};
        }
    }
    let mut found = Vec::new();
    let mut i = 0;
    while i < lines.len() && found.len() < MAX_BLOCKS {
        let first = lines[i].text.trim();
        let fence = match first { "```mermaid" => Some("```"), "~~~mermaid" => Some("~~~"), _ => None };
        // Raw flowcharts are emitted by CLI fallbacks. Other families need a
        // fence because their ordinary words are indistinguishable from prose.
        if fence.is_none() && !matches!(first.split_whitespace().next(), Some("flowchart" | "graph")) {
            i += 1; continue;
        }
        let content = i + usize::from(fence.is_some());
        let mut j = content;
        let mut source = String::new();
        let mut quoted = false;
        let mut last = content;
        let mut bounded = true;
        while j < lines.len() && j - i < MAX_ROWS {
            let line = &lines[j].text;
            let trimmed = line.trim();
            if fence.is_some_and(|f| trimmed == f) { break; }
            if fence.is_none() && j > content && !quoted && !trimmed.is_empty() {
                let bare_id = trimmed.trim_end_matches(';').chars().all(|c| c.is_alphanumeric() || c == '_');
                if is_header(trimmed) || (!looks_like_statement(trimmed) && !bare_id) { break; }
            }
            if source.len() + line.len() + 1 > MAX_SOURCE { bounded = false; break; }
            source.push_str(line);
            source.push('\n');
            let mut escaped = false;
            for ch in line.chars() {
                if ch == '"' && !escaped { quoted = !quoted; }
                escaped = ch == '\\' && !escaped;
            }
            if !trimmed.is_empty() { last = j + 1; }
            j += 1;
        }
        let complete = fence.map_or(j < lines.len(), |f| lines.get(j).is_some_and(|l| l.text.trim() == f));
        let finish = if fence.is_some() { j + 1 } else { last };
        if complete && bounded && j - i < MAX_ROWS && finish > content && lines[finish - 1].end > top && lines[i].start < top + screen.rows {
            let start_row = screen.absolute_of_virtual(lines[i].start);
            let end_row = screen.absolute_of_virtual(lines[finish - 1].end);
            let cells = screen.selection_text((start_row, 0), (end_row - 1, screen.cols));
            let block = SourceBlock {start: start_row, end: end_row, source: source.trim().to_owned(), cells};
            // A shorter source has too little vertical room for a readable
            // viewport and controls; leave it as text for the explicit preview.
            if end_row - start_row >= 4 && unchanged(screen, &block) { found.push(block); }
        }
        i = j.max(i + 1);
    }
    found
}
