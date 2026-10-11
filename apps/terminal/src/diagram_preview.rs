//! A native Mermaid preview of explicitly selected terminal output. PTY cells
//! remain unchanged: interactive CLIs still own their cursor and screen layout.
//! Layout/rasterization uses the same bounded renderer as MarkdownDiagram and
//! runs on Makepad's task pool, never while drawing the terminal.
use makepad_markdown_widgets::{content::PreviewImage, diagram};
use makepad_widgets::*;

pub const MAX_SOURCE: usize = 16 * 1024;

/// Find the latest fenced Mermaid block, or a plain Mermaid source block (as
/// emitted by CLI source fallbacks). Soft terminal wrapping is joined by the
/// terminal before calling this; application-inserted hard wraps stay editable.
pub fn source_from_output(text: &str) -> Option<String> {
    let lines: Vec<_> = text.lines().collect();
    let mut start = lines.iter().rposition(|line| {
        let line = line.trim();
        line == "```mermaid" || line == "~~~mermaid" || is_header(line)
    })?;
    if let Some(previous) = lines[..start].iter().rposition(|line| !line.trim().is_empty()) {
        if matches!(lines[previous].trim(), "```mermaid" | "~~~mermaid") {
            start = previous;
        }
    }
    let fenced = lines[start].trim().ends_with("mermaid");
    let start = start + usize::from(fenced);
    let flowchart = lines.get(start).is_some_and(|line| {
        matches!(line.split_whitespace().next(), Some("flowchart" | "graph"))
    });
    let mut source = String::new();
    let mut blank = false;
    let mut quoted = false;
    for line in &lines[start..] {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            break;
        }
        // A raw flowchart ends before the next shell prompt or prose, even
        // when there is no blank separator. Quoted labels can span lines.
        let bare_id = !trimmed.is_empty()
            && trimmed.trim_end_matches(';').chars().all(|c| c.is_alphanumeric() || c == '_');
        if !fenced && !quoted && !source.is_empty() && !trimmed.is_empty()
            && (flowchart || blank) && !looks_like_statement(trimmed) && !bare_id {
            break;
        }
        if source.len() + line.len() + 1 > MAX_SOURCE {
            return None;
        }
        source.push_str(line);
        source.push('\n');
        let mut escaped = false;
        for ch in line.chars() {
            if ch == '"' && !escaped { quoted = !quoted; }
            escaped = ch == '\\' && !escaped;
        }
        blank = trimmed.is_empty();
    }
    Some(source.trim().to_owned())
}

pub(crate) fn is_header(line: &str) -> bool {
    matches!(line.split_whitespace().next(), Some(
        "flowchart" | "graph" | "sequenceDiagram" | "classDiagram" |
        "stateDiagram" | "stateDiagram-v2" | "erDiagram" | "pie" |
        "gantt" | "mindmap" | "gitGraph" | "timeline"
    ))
}

pub(crate) fn looks_like_statement(line: &str) -> bool {
    line.starts_with("%%") || line.contains("--") || line.contains("->") ||
    line.contains('[') || line.contains('{') || line.contains('(') || matches!(line.split_whitespace().next(), Some(
        "classDef" | "class" | "style" | "linkStyle" | "subgraph" | "end" |
        "direction" | "participant" | "actor" | "Note" | "note" | "loop" |
        "alt" | "else" | "opt" | "par" | "and" | "rect" | "activate" | "deactivate"
    ))
}

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.*
    mod.widgets.TermDiagramPreview = #(TermDiagramPreview::register_widget(vm)) {
        width: Fill height: Fill
        ui: View {
            width: Fill height: Fill flow: Down spacing: 0
            show_bg: true draw_bg +: {color: #x171d2a pixel: fn() {return self.color}}
            header := View {
                width: Fill height: 52 flow: Right spacing: 8
                padding: Inset{left: 12 right: 12 top: 8 bottom: 8}
                align: Align{y: 0.5}
                back := Button {text: "Back to terminal"}
                Label {text: "Mermaid" draw_text +: {color: #xe5eaf5}}
                View {width: Fill height: 1}
                source_toggle := Button {text: "Source"}
                zoom_out := Button {text: "−"}
                zoom_in := Button {text: "+"}
                fit := Button {text: "Fit width"}
            }
            status := Label {
                width: Fill height: Fit
                margin: Inset{left: 16 bottom: 10 right: 16}
                draw_text +: {color: #xaebbd1 text_style: theme.font_regular{font_size: 10}}
                text: "Select Mermaid source in the terminal, or paste it in Source."
            }
            canvas := ScrollXYView {
                width: Fill height: Fill
                show_bg: true draw_bg +: {color: #xffffffff pixel: fn() {return self.color}}
                padding: Inset{left: 16 right: 16 top: 16 bottom: 16}
                image := Image {width: 1 height: 1 fit: ImageFit.Stretch}
            }
            source_panel := View {
                visible: false width: Fill height: Fill flow: Down spacing: 8
                padding: Inset{left: 16 right: 16 top: 8 bottom: 16}
                editor_scroll := ScrollXYView {
                    width: Fill height: Fill
                    editor := TextInput {
                        width: Fill height: Fit is_multiline: true
                        empty_text: "Paste Mermaid source here, then choose Render."
                    }
                }
                source_actions := View {
                    width: Fill height: Fit flow: Right spacing: 8
                    render := Button {text: "Render"}
                    copy := Button {text: "Copy source"}
                }
            }
        }
    }
}

#[derive(Script, Widget)]
pub struct TermDiagramPreview {
    #[uid]
    uid: WidgetUid,
    #[source]
    source: ScriptObjectRef,
    #[walk]
    walk: Walk,
    #[redraw]
    #[find]
    #[live]
    ui: WidgetRef,
    #[rust]
    job: Option<(u64, TaskHandle<Result<PreviewImage, String>>)>,
    #[rust]
    generation: u64,
    #[rust]
    timer: Timer,
    #[rust]
    dimensions: Option<(f64, f64)>,
    #[rust(1.0)]
    zoom: f64,
    #[rust(true)]
    fit_width: bool,
    #[rust]
    editing: bool,
    #[rust]
    pub closed: bool,
    #[rust]
    pending: Option<String>,
}

impl ScriptHook for TermDiagramPreview {
    fn on_after_apply(&mut self, vm: &mut ScriptVm, _: &Apply, _: &mut Scope, _: ScriptValue) {
        vm.cx_mut().widget_tree_insert_child_deep(self.uid, id!(ui), self.ui.clone());
    }
}

impl TermDiagramPreview {
    pub fn open(&mut self, cx: &mut Cx, source: Option<String>) {
        self.closed = false;
        self.generation = self.generation.wrapping_add(1);
        self.pending = None;
        self.ui.view(cx, ids!(canvas)).set_scroll_pos(cx, dvec2(0.0, 0.0));
        self.dimensions = None;
        self.fit_width = true;
        self.ui.text_input(cx, ids!(editor)).set_text(cx, source.as_deref().unwrap_or(""));
        self.set_editing(cx, source.is_none());
        if let Some(source) = source {
            self.request(cx, source);
        } else {
            self.pending = None;
            self.status(cx, "No Mermaid source found. Paste a diagram in Source, then choose Render.");
        }
        self.redraw(cx);
    }

    fn status(&self, cx: &mut Cx, text: &str) {
        self.ui.label(cx, ids!(status)).set_text(cx, text);
    }

    fn set_editing(&mut self, cx: &mut Cx, editing: bool) {
        self.editing = editing;
        self.ui.view(cx, ids!(canvas)).set_visible(cx, !editing);
        self.ui.view(cx, ids!(source_panel)).set_visible(cx, editing);
        self.ui.button(cx, ids!(source_toggle)).set_text(cx, if editing {"Diagram"} else {"Source"});
        self.redraw(cx);
    }

    fn request(&mut self, cx: &mut Cx, source: String) {
        self.generation = self.generation.wrapping_add(1);
        self.pending = None;
        self.dimensions = None;
        if source.len() > MAX_SOURCE {
            self.status(cx, "Diagram exceeds 16 KB. Split it into smaller diagrams.");
            return;
        }
        self.pending = Some(source);
        self.status(cx, "Rendering diagram…");
        if self.timer.is_empty() {
            self.timer = cx.start_interval(0.05);
        }
        self.poll(cx);
    }

    fn poll(&mut self, cx: &mut Cx) {
        if let Some((generation, result)) = self.job.as_mut().and_then(|(generation, job)| job.try_take().map(|r| (*generation, r))) {
            self.job = None;
            // A new source supersedes an in-flight result without piling up jobs.
            if generation == self.generation && self.pending.is_none() && !self.closed {
                match result.map_err(|e| format!("Diagram task failed: {e}")).and_then(|r| r) {
                    Ok(image) => {
                        match self.ui.image(cx, ids!(image)).load_png_from_data(cx, &image.png) {
                            Ok(()) => {
                                self.dimensions = Some((image.width as f64, image.height as f64));
                                self.status(cx, "Scroll to explore · + / − to zoom · Source to edit");
                            }
                            Err(error) => self.status(cx, &format!("Could not display diagram: {error:?}")),
                        }
                    }
                    Err(error) => {
                        self.set_editing(cx, true);
                        self.status(cx, &format!("{error}. Source is preserved below."));
                    }
                }
                self.redraw(cx);
            }
        }
        if self.job.is_none() {
            if let Some(source) = self.pending.clone() {
                match cx.task_pool().try_submit(Lane::Heavy, move || diagram::render("mermaid", &source)) {
                    Ok(job) => { self.job = Some((self.generation, job)); self.pending = None; },
                    Err(_) => self.status(cx, "Renderer is busy; waiting for a worker…"),
                }
            }
        }
        if self.job.is_none() && self.pending.is_none() {
            cx.stop_timer(self.timer);
            self.timer = Timer::default();
        }
    }

    pub fn close(&mut self, cx: &mut Cx) {
        self.closed = true;
        self.pending = None;
        self.redraw(cx);
    }
}

impl Widget for TermDiagramPreview {
    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        if self.timer.is_event(event).is_some() {
            self.poll(cx);
        }
        if self.closed { return; }
        let actions = cx.capture_actions(|cx| self.ui.handle_event(cx, event, scope));
        if self.ui.button(cx, ids!(back)).clicked(&actions) || matches!(event, Event::KeyDown(k) if k.key_code == KeyCode::Escape) {
            self.close(cx);
        }
        if self.ui.button(cx, ids!(source_toggle)).clicked(&actions) {
            self.set_editing(cx, !self.editing);
        }
        if self.ui.button(cx, ids!(render)).clicked(&actions) {
            let text = self.ui.text_input(cx, ids!(editor)).text();
            let source = source_from_output(&text).unwrap_or(text);
            self.set_editing(cx, false);
            self.request(cx, source);
        }
        if self.ui.button(cx, ids!(copy)).clicked(&actions) {
            cx.copy_to_clipboard(&self.ui.text_input(cx, ids!(editor)).text());
        }
        if self.ui.button(cx, ids!(zoom_in)).clicked(&actions) {
            self.zoom = (self.zoom * 1.25).min(4.0);
            self.fit_width = false;
            self.redraw(cx);
        }
        if self.ui.button(cx, ids!(zoom_out)).clicked(&actions) {
            self.zoom = (self.zoom / 1.25).max(0.1);
            self.fit_width = false;
            self.redraw(cx);
        }
        if self.ui.button(cx, ids!(fit)).clicked(&actions) {
            self.fit_width = true;
            self.redraw(cx);
        }
    }

    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        let (width, height) = self.dimensions.unwrap_or((1.0, 1.0));
        if self.fit_width {
            let available = match walk.width { Size::Fixed(w) => w, _ => cx.turtle().inner_rect().size.x };
            self.zoom = ((available - 48.0).max(1.0) / width).min(1.0);
        }
        self.ui.image(cx, ids!(image)).set_walk_and_fit(cx,
            Walk::new(Size::Fixed(width * self.zoom), Size::Fixed(height * self.zoom)), ImageFit::Stretch);
        self.ui.draw_walk(cx, scope, walk)
    }
}
