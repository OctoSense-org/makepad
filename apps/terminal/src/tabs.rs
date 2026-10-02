//! Native tabs: several shells in one terminal window, drawn and switched by
//! the terminal itself rather than by a window manager, so they work the
//! same standalone, in makepad-wm and as an OctoSense module.
//!
//! `TermTabs{ term := MpTerm{} }` keeps the `term` template and mints one
//! `MpTerm` per tab from it. Only the selected tab draws and receives input;
//! every tab keeps pumping its PTY (signals, timers), so background jobs keep
//! running and their output is there when the tab is selected again.
//!
//! Keys (while the terminal holds the keyboard), by default; every one is
//! a shortcut in `crate::keybinds` and can be changed in `settings.conf`:
//!
//! | Ctrl+Shift+T | new tab (in the current tab's directory by default) |
//! | Ctrl+Shift+W | close tab (asks first while a program runs in it)    |
//! | Ctrl+Tab / Ctrl+PageDown       | next tab                           |
//! | Ctrl+Shift+Tab / Ctrl+PageUp   | previous tab                       |
//! | Alt+1..8 / Alt+9               | that tab / the last tab            |
//! | Ctrl+,       | settings                                             |
//!
//! Alt+digit is taken only while more than one tab is open (`performable:`),
//! so a single shell keeps Meta+digit (readline's numeric argument).
//!
//! A long press (held 0.5 s) or a double click on a tab renames it in
//! place: Enter or a click elsewhere takes the name, Esc drops it, and an
//! empty name gives the tab its automatic label back. A name lives as long
//! as its tab, and a program's own title never replaces it.
//!
//! Closing the whole terminal (Cmd+Q, the window's close button, a window
//! manager's close, a host closing the module) goes through
//! [`TermTabs::request_close_all`]: while any tab or pane runs a job and
//! `confirm-close-running` is on, it answers [`CloseDecision::Veto`] and
//! asks in the bar, naming every job it would end; a yes emits
//! [`ModuleCloseAction::Confirmed`] and the caller closes then.

use std::path::{Path, PathBuf};

use makepad_app_module::{CloseDecision, ModuleCloseAction};

use makepad_widgets::makepad_platform::event::TouchState;
use makepad_widgets::widget_tree::CxWidgetExt;
use makepad_widgets::*;

use crate::keybinds::{self, Keybinds};
use crate::search;
use crate::settings::{self as term_settings, NewTabCwd, Settings, TabBar, TabTitle};
use crate::panes::{self, Dir, Divider, Node};
use crate::settings_panel::{self, Choice, Row, RowKind};
use std::sync::atomic::{AtomicU64, Ordering};
use crate::widget::{MpTerm, MpTermAction};

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.*

    set_type_default() do #(DrawTabShape::script_shader(vm)) {
        ..mod.draw.DrawQuad
        color: #x1a1b26
        radius: 6.0
        /** 1: round the top corners only (a tab), 0: all four (a button) */
        tab: 1.0
        pixel: fn() {
            let sdf = Sdf2d.viewport(self.pos * self.rect_size)
            let extra = self.tab * self.radius
            sdf.box(0.0, 0.0, self.rect_size.x, self.rect_size.y + extra, self.radius)
            sdf.fill(self.color)
            return sdf.result
        }
    }

    mod.widgets.TermTabsBase = #(TermTabs::register_widget(vm))

    /** A terminal with native tabs. */
    mod.widgets.TermTabs = set_type_default() do mod.widgets.TermTabsBase {
        width: Fill
        height: Fill
        flow: Down
        /** tab bar height in pixels 20..48 step 1 */
        bar_height: 30.0
        draw_bar +: { color: #x16161e }
        draw_tab +: {}
        draw_label +: {
            text_style: theme.font_regular{ font_size: 9.0 }
            color: #xa9b1d6
        }
        draw_icon +: {
            text_style: theme.font_icons{ font_size: 8.5 }
            color: #xa9b1d6
        }
        draw_panel +: { color: #x1f2335 }
        draw_divider +: { color: #x16161e }
        draw_heading +: {
            text_style: theme.font_bold{ font_size: 10.5 }
            color: #xc0caf5
        }
        term := MpTerm{}
    }
}

#[derive(Script, ScriptHook)]
#[repr(C)]
pub struct DrawTabShape {
    #[deref]
    draw_super: DrawQuad,
    #[live]
    color: Vec4f,
    #[live]
    radius: f32,
    #[live]
    tab: f32,
}

#[derive(Clone, Debug, Default)]
pub enum TermTabsAction {
    /// The last tab was closed (Ctrl+Shift+W). A standalone window quits; a
    /// hosted one gets a fresh shell on its next frame.
    LastTabClosed,
    #[default]
    None,
}

/// A tab bar command, from a key or a click.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TabCommand {    New,
    Close,
    Next,
    Previous,
    /// 0-based; `usize::MAX` is the last tab.
    Select(usize),
    Settings,
    /// Split the focused pane: side by side, or stacked.
    SplitRight,
    SplitDown,
    /// Move to the pane in that direction.
    Focus(Dir),
    /// Show only the focused pane (again: all of them).
    Zoom,
    /// The whole selected tab, every pane in it.
    CloseTab,
    /// Type a name for the selected tab.
    Rename,
}

/// The tab command a shortcut's action is, if it is one of the tab
/// widget's (`crate::keybinds::Scope::Tabs`).
pub fn tab_command_of(action: &keybinds::Action) -> Option<TabCommand> {
    use keybinds::Action as A;
    Some(match action {
        A::NewTab => TabCommand::New,
        A::CloseSurface => TabCommand::Close,
        A::CloseTab => TabCommand::CloseTab,
        A::NextTab => TabCommand::Next,
        A::PreviousTab => TabCommand::Previous,
        A::GotoTab(n) => TabCommand::Select(n.saturating_sub(1)),
        A::LastTab => TabCommand::Select(usize::MAX),
        A::NewSplit(Dir::Down) => TabCommand::SplitDown,
        A::NewSplit(_) => TabCommand::SplitRight,
        A::GotoSplit(dir) => TabCommand::Focus(*dir),
        A::ToggleSplitZoom => TabCommand::Zoom,
        A::OpenConfig => TabCommand::Settings,
        A::PromptTabTitle => TabCommand::Rename,
        _ => return None,
    })
}

/// The keys the tab widget took before shortcuts were configurable; the
/// default table must answer the same (`the_tab_keys_map_to_commands`).
#[cfg(test)]
fn legacy_tab_command(key: &KeyEvent, tabs: usize) -> Option<TabCommand> {
    let m = &key.modifiers;
    if m.logo {
        return None;
    }
    if m.control && !m.alt {
        match key.key_code {
            KeyCode::KeyT if m.shift => return Some(TabCommand::New),
            KeyCode::KeyW if m.shift => return Some(TabCommand::Close),
            KeyCode::Tab if m.shift => return Some(TabCommand::Previous),
            KeyCode::Tab => return Some(TabCommand::Next),
            KeyCode::PageDown if !m.shift => return Some(TabCommand::Next),
            KeyCode::PageUp if !m.shift => return Some(TabCommand::Previous),
            KeyCode::Comma if !m.shift => return Some(TabCommand::Settings),
            KeyCode::KeyD if m.shift => return Some(TabCommand::SplitRight),
            KeyCode::KeyE if m.shift => return Some(TabCommand::SplitDown),
            KeyCode::KeyZ if m.shift => return Some(TabCommand::Zoom),
            KeyCode::ArrowLeft if m.shift => return Some(TabCommand::Focus(Dir::Left)),
            KeyCode::ArrowRight if m.shift => return Some(TabCommand::Focus(Dir::Right)),
            KeyCode::ArrowUp if m.shift => return Some(TabCommand::Focus(Dir::Up)),
            KeyCode::ArrowDown if m.shift => return Some(TabCommand::Focus(Dir::Down)),
            _ => {}
        }
    }
    if m.alt && !m.control && !m.shift && tabs > 1 {
        let digit = match key.key_code {
            KeyCode::Key1 => 1,
            KeyCode::Key2 => 2,
            KeyCode::Key3 => 3,
            KeyCode::Key4 => 4,
            KeyCode::Key5 => 5,
            KeyCode::Key6 => 6,
            KeyCode::Key7 => 7,
            KeyCode::Key8 => 8,
            KeyCode::Key9 => return Some(TabCommand::Select(usize::MAX)),
            _ => return None,
        };
        return Some(TabCommand::Select(digit - 1));
    }
    None
}

/// `path` with the home directory shown as `~`.
fn tilde(path: &Path) -> String {
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        if path == home {
            return "~".into();
        }
        if let Ok(rest) = path.strip_prefix(&home) {
            return format!("~/{}", rest.display());
        }
    }
    path.display().to_string()
}

/// A directory as a tab title: its last component (`~` for home).
fn dir_title(path: &Path) -> String {
    let shown = tilde(path);
    if shown == "~" || shown == "/" {
        return shown;
    }
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or(shown)
}

/// What a tab is called under `mode`: the program's own title (OSC 0/2) or
/// the job's name, else the directory, else the shell's name.
pub fn tab_label(mode: TabTitle, osc: &str, job: Option<&str>, dir: Option<&Path>, shell: Option<&str>) -> String {
    let dir = dir.map(dir_title);
    let label = match mode {
        TabTitle::Program => {
            if !osc.trim().is_empty() {
                Some(osc.trim().to_owned())
            } else {
                job.map(str::to_owned).or(dir)
            }
        }
        TabTitle::Directory => dir,
    };
    label.or_else(|| shell.map(str::to_owned)).unwrap_or_else(|| "shell".into())
}

/// What a tab is called: the name the person gave it, else its
/// [`tab_label`]. A program's title never replaces a given name.
pub fn tab_title(
    custom: Option<&str>,
    mode: TabTitle,
    osc: &str,
    job: Option<&str>,
    dir: Option<&Path>,
    shell: Option<&str>,
) -> String {
    match custom.map(str::trim).filter(|c| !c.is_empty()) {
        Some(custom) => custom.to_owned(),
        None => tab_label(mode, osc, job, dir, shell),
    }
}

/// The longest name a tab takes, in characters.
const TAB_NAME_MAX: usize = 64;

/// A tab's name being typed in the tab bar: the text and the caret (in
/// characters).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct NameEdit {
    text: String,
    caret: usize,
}

/// What a key did to a name being typed.
#[derive(Clone, Debug, PartialEq, Eq)]
enum NameKey {
    Editing,
    /// Take the name; None (nothing typed) gives the tab its automatic
    /// label back.
    Commit(Option<String>),
    Cancel,
}

impl NameEdit {
    fn new(text: &str) -> NameEdit {
        let text: String = text.chars().take(TAB_NAME_MAX).collect();
        NameEdit { caret: text.chars().count(), text }
    }

    fn byte_at(&self, caret: usize) -> usize {
        self.text.char_indices().nth(caret).map_or(self.text.len(), |(i, _)| i)
    }

    /// Typed or pasted text, at the caret (control characters and line
    /// breaks dropped, the length capped).
    fn insert(&mut self, typed: &str) {
        let room = TAB_NAME_MAX.saturating_sub(self.text.chars().count());
        let typed: String = typed.chars().filter(|c| !c.is_control()).take(room).collect();
        let at = self.byte_at(self.caret);
        self.text.insert_str(at, &typed);
        self.caret += typed.chars().count();
    }

    fn key(&mut self, key: &KeyEvent) -> NameKey {
        let len = self.text.chars().count();
        match key.key_code {
            KeyCode::ReturnKey | KeyCode::NumpadEnter => return NameKey::Commit(self.committed()),
            KeyCode::Escape => return NameKey::Cancel,
            KeyCode::Backspace if self.caret > 0 => {
                self.caret -= 1;
                let at = self.byte_at(self.caret);
                self.text.remove(at);
            }
            KeyCode::Delete if self.caret < len => {
                let at = self.byte_at(self.caret);
                self.text.remove(at);
            }
            KeyCode::ArrowLeft => self.caret = self.caret.saturating_sub(1),
            KeyCode::ArrowRight => self.caret = (self.caret + 1).min(len),
            KeyCode::Home | KeyCode::ArrowUp => self.caret = 0,
            KeyCode::End | KeyCode::ArrowDown => self.caret = len,
            _ => {}
        }
        NameKey::Editing
    }

    fn committed(&self) -> Option<String> {
        Some(self.text.trim().to_owned()).filter(|t| !t.is_empty())
    }
}

/// Pane ids are unique in the process: the control socket names panes by
/// them (`crate::control`).
static NEXT_PANE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_TAB_ID: AtomicU64 = AtomicU64::new(1);

/// One terminal in a tab.
pub(crate) struct Pane {
    pub(crate) id: u64,
    pub(crate) term: WidgetRef,
    /// The program's own title (OSC 0/2); empty when it set none.
    pub(crate) osc_title: String,
    /// Polled: the foreground job (None at the prompt), the shell's name
    /// and its directory.
    pub(crate) job: Option<String>,
    pub(crate) shell: Option<String>,
    pub(crate) dir: Option<PathBuf>,
    /// Rang the bell while not in view.
    bell: bool,
    /// The screen at the last poll, and when it last changed (an agent
    /// printing is working).
    screen_hash: u64,
    changed_at: Option<std::time::Instant>,
}

impl Pane {
    pub(crate) fn label(&self, mode: TabTitle) -> String {
        tab_label(mode, &self.osc_title, self.job.as_deref(), self.dir.as_deref(), self.shell.as_deref())
    }

    pub(crate) fn with_term<R>(&self, f: impl FnOnce(&mut MpTerm) -> R) -> Option<R> {
        self.term.borrow_mut::<MpTerm>().map(|mut term| f(&mut term))
    }
}

/// A tab: one or more panes, split in a tree.
pub(crate) struct Tab {
    /// Unique in the process: a rename in progress follows its tab while
    /// others open and close.
    id: u64,
    /// The name the person gave the tab (in memory only); None: the
    /// automatic label.
    custom_title: Option<String>,
    pub(crate) panes: Vec<Pane>,
    tree: Node,
    pub(crate) focused: u64,
    /// Only the focused pane is shown.
    zoomed: bool,
}

impl Tab {
    fn new(pane: Pane) -> Tab {
        Tab {
            id: NEXT_TAB_ID.fetch_add(1, Ordering::Relaxed),
            custom_title: None,
            tree: Node::Leaf(pane.id),
            focused: pane.id,
            zoomed: false,
            panes: vec![pane],
        }
    }

    pub(crate) fn focused_pane(&self) -> &Pane {
        self.panes.iter().find(|p| p.id == self.focused).unwrap_or(&self.panes[0])
    }

    fn pane_mut(&mut self, id: u64) -> Option<&mut Pane> {
        self.panes.iter_mut().find(|p| p.id == id)
    }

    /// The tab's name (see [`tab_title`]); an automatic one counts the
    /// panes of a split.
    fn label(&self, mode: TabTitle) -> String {
        let pane = self.focused_pane();
        let custom = self.custom_title.as_deref();
        let label = tab_title(custom, mode, &pane.osc_title, pane.job.as_deref(), pane.dir.as_deref(), pane.shell.as_deref());
        if custom.is_none() && self.panes.len() > 1 {
            format!("{label} \u{00b7} {}", self.panes.len())
        } else {
            label
        }
    }

    fn bell(&self) -> bool {
        self.panes.iter().any(|p| p.bell)
    }

    fn with_term<R>(&self, f: impl FnOnce(&mut MpTerm) -> R) -> Option<R> {
        self.focused_pane().with_term(f)
    }

    /// The panes in view and where, within `body`.
    fn layout(&self, body: Rect) -> Vec<(u64, Rect)> {
        if self.zoomed || self.panes.len() == 1 {
            vec![(self.focused, body)]
        } else {
            self.tree.layout(body)
        }
    }
}

/// What the pointer is over in the tab bar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BarHit {
    Tab(usize),
    CloseTab(usize),
    NewTab,
    Settings,
    ConfirmClose,
    CancelClose,
}

/// A close waiting for the person to confirm.
enum PendingClose {
    /// Tab `tab` (or one pane of it) runs `job`.
    Tab {
        tab: usize,
        /// One pane of the tab, or (None) the whole tab.
        pane: Option<u64>,
        job: String,
    },
    /// The whole terminal (quit, window close, host close) would end
    /// `jobs`: (tab label, job) in tab order.
    All { jobs: Vec<(String, String)> },
}

/// Every running job, as (tab label, job), from each tab's label and the
/// foreground job of each of its panes (None: at the prompt). Tab order,
/// then pane order; a split tab with two jobs lists both.
pub fn collect_running_jobs<I, P>(tabs: I) -> Vec<(String, String)>
where
    I: IntoIterator<Item = (String, P)>,
    P: IntoIterator<Item = Option<String>>,
{
    tabs.into_iter()
        .flat_map(|(label, panes)| panes.into_iter().flatten().map(move |job| (label.clone(), job)))
        .collect()
}

/// Whether closing the whole terminal must ask first: only while a job
/// runs, `confirm-close-running` is on, and the person has not already said
/// yes to this close.
pub fn close_all_decision(confirm_running: bool, confirmed: bool, jobs: &[(String, String)]) -> CloseDecision {
    if confirm_running && !confirmed && !jobs.is_empty() {
        CloseDecision::Veto
    } else {
        CloseDecision::Allow
    }
}

/// The question the bar asks before closing the whole terminal.
pub fn close_all_message(jobs: &[(String, String)]) -> String {
    // A tab named after its job says nothing more: "“sleep”", not
    // "“sleep” in sleep".
    let named: Vec<String> = jobs
        .iter()
        .map(|(tab, job)| {
            if tab == job {
                format!("\u{201c}{job}\u{201d}")
            } else {
                format!("\u{201c}{job}\u{201d} in {tab}")
            }
        })
        .collect();
    match jobs.len() {
        1 => format!("Closing the terminal ends {}. Close anyway?", named[0]),
        n => format!("Closing the terminal ends {n} running jobs: {}. Close anyway?", named.join(", ")),
    }
}

#[derive(Script, Widget)]
pub struct TermTabs {
    #[uid]
    uid: WidgetUid,
    #[source]
    source: ScriptObjectRef,
    #[walk]
    walk: Walk,
    #[layout]
    layout: Layout,

    #[redraw]
    #[live]
    draw_bar: DrawColor,
    #[live]
    draw_tab: DrawTabShape,
    #[live]
    draw_label: DrawText,
    #[live]
    draw_icon: DrawText,
    /// The settings panel's card (its area takes the panel's input).
    #[live]
    draw_panel: DrawColor,
    #[live]
    draw_heading: DrawText,
    #[live]
    draw_divider: DrawColor,
    /// The selected tab's dividers and body, from the last draw.
    #[rust]
    dividers: Vec<Divider>,
    #[rust]
    body: Rect,
    /// A divider being dragged.
    #[rust]
    drag: Option<Divider>,
    #[live(30.0)]
    bar_height: f64,

    /// The `term := MpTerm{}` template every tab is minted from.
    #[rust]
    template: Option<ScriptObjectRef>,
    #[rust]
    tabs: Vec<Tab>,
    #[rust]
    active: usize,
    /// Tabs off: a `--preview` pager is one job in one window.
    #[rust(true)]
    tabs_enabled: bool,
    #[rust]
    settings: Settings,
    #[rust]
    settings_gen: u64,
    /// The shortcut table from the settings' `keybind` lines.
    #[rust]
    keys: Keybinds,
    #[rust]
    poll_timer: Timer,
    #[rust]
    hover: Option<BarHit>,
    #[rust]
    pending_close: Option<PendingClose>,
    /// The person said yes to closing the whole terminal: a second ask
    /// (a host asking again before it closes) is allowed.
    #[rust]
    close_confirmed: bool,
    /// Hit rectangles of the last drawn bar.
    #[rust]
    hits: Vec<(Rect, BarHit)>,
    /// The title last reported to the host for the selected tab.
    #[rust]
    reported_title: String,
    #[rust]
    panel: Option<Panel>,
    /// Tabs or panes changed: tell the control socket at the end of the
    /// event (its `list` must not lag a second behind a switch).
    #[rust]
    publish_soon: bool,
    /// A font change waiting for the selection to rest: rasterizing every
    /// font passed on the way costs the text engine's glyph atlas (it keeps
    /// them all), so fonts preview once browsing pauses.
    #[rust]
    pending_apply: Option<Settings>,
    #[rust]
    apply_timer: Timer,
    /// The panel draws in an overlay list: above every terminal layer.
    #[rust]
    panel_list: Option<DrawList2d>,
    /// A tab's name being typed; it owns the keyboard meanwhile.
    #[rust]
    rename: Option<Rename>,
    /// A press on a tab, and the timer that makes it a long one.
    #[rust]
    press: Option<TabPress>,
    #[rust]
    press_timer: Timer,
}

/// A tab being renamed in the tab bar.
struct Rename {
    /// `Tab::id`.
    tab: u64,
    edit: NameEdit,
    /// The field, from the last draw: a click outside it takes the name.
    rect: Rect,
}

/// A press on a tab that becomes a rename if held without moving.
struct TabPress {
    /// `Tab::id`.
    tab: u64,
    abs: DVec2,
}

/// How long a press on a tab is held to rename it, and how far it may
/// wander meanwhile (desktop platforms send no `LongPress` for a mouse).
const HOLD_TO_RENAME: f64 = 0.5;
const HOLD_SLOP: f64 = 6.0;

/// The open settings panel.
#[derive(Default)]
struct Panel {
    /// Index into `settings_panel::rows()`.
    selected: usize,
    /// Rows scrolled off the top.
    scroll: usize,
    /// The selection last scrolled into view: the draw keeps a NEW
    /// selection visible, but leaves a mouse-wheel scroll alone.
    shown_selected: Option<usize>,
    rect: Rect,
    hits: Vec<(Rect, PanelHit)>,
    /// Rows (or list entries) that fit, from the last draw.
    visible: usize,
    mode: PanelMode,
    /// Delete profile was pressed once; the next press deletes.
    confirm_delete: bool,
    /// The outcome of the last action, shown in the footer.
    message: Option<String>,
}

#[derive(Default)]
enum PanelMode {
    #[default]
    Rows,
    /// A row's list, filtered by what is typed.
    Choose(Chooser),
    /// Typing a name to save the settings as a profile.
    Name(String),
}

struct Chooser {
    row: Row,
    choices: Vec<Choice>,
    filter: String,
    /// Index into the filtered list.
    selected: usize,
    scroll: usize,
    /// The selection last scrolled into view (see `Panel::shown_selected`).
    shown_selected: Option<usize>,
    /// The settings when the list opened: Esc goes back to them.
    original: Settings,
    /// Opened before the font scan finished: refill when it has.
    awaiting_fonts: bool,
}

struct PanelColors {
    fg: Vec4f,
    white: Vec4f,
    card: Vec4f,
    line: Vec4f,
    dim: Vec4f,
    accent: Vec4f,
}

/// How many rows fit in `height` from row `scroll` on, with the header of
/// each section that has a row in view.
fn rows_fitting(scroll: usize, height: f64, row_h: f64) -> usize {
    let mut used = 0.0;
    let mut fit = 0;
    let mut index = 0;
    for (_, rows) in settings_panel::SECTIONS {
        let mut header = false;
        for _ in rows.iter() {
            let i = index;
            index += 1;
            if i < scroll {
                continue;
            }
            let need = if header { row_h } else { row_h * 2.0 };
            if used + need > height {
                return fit.max(1);
            }
            used += need;
            header = true;
            fit += 1;
        }
    }
    fit.max(1)
}

fn chooser_len(panel: &Option<Panel>) -> usize {
    match panel.as_ref().map(|p| &p.mode) {
        Some(PanelMode::Choose(chooser)) => chooser.choices.len(),
        _ => 0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PanelHit {
    Close,
    Select(usize),
    /// Open a row's list or run its action.
    Activate(usize),
    Step(usize, i32),
    /// An entry of the open list (index into the filtered list).
    Choice(usize),
    Back,
    Reset,
}

impl ScriptHook for TermTabs {
    fn on_after_new(&mut self, vm: &mut ScriptVm) {
        self.panel_list = Some(DrawList2d::script_new(vm));
    }

    fn on_after_apply(&mut self, vm: &mut ScriptVm, apply: &Apply, _scope: &mut Scope, value: ScriptValue) {
        if apply.is_eval() {
            return;
        }
        if let Some(obj) = value.as_object() {
            vm.vec_with(obj, |vm, vec| {
                for kv in vec {
                    if kv.key.as_id() == Some(id!(term)) {
                        if let Some(template) = kv.value.as_object() {
                            self.template = Some(vm.bx.heap.new_object_ref(template));
                        }
                    }
                }
            });
        }
    }
}

impl TermTabs {
    /// The selected tab's terminal, opening the first tab if none is open
    /// yet (so a host can set its `cwd`/`command` before it starts).
    pub fn active_term(&mut self, cx: &mut Cx) -> WidgetRef {
        if self.tabs.is_empty() {
            self.open_tab(cx, None);
        }
        self.tabs.get(self.active).map(|tab| tab.focused_pane().term.clone()).unwrap_or_default()
    }

    /// Turn the tab bar and its keys off (a preview window) or on.
    pub fn set_tabs_enabled(&mut self, cx: &mut Cx, enabled: bool) {
        self.tabs_enabled = enabled;
        self.redraw(cx);
    }

    pub fn tab_count(&self) -> usize {
        self.tabs.len()
    }

    /// Every job running in any tab or pane, as (tab label, job): what
    /// closing the whole terminal would end.
    pub fn running_jobs(&self) -> Vec<(String, String)> {
        let mode = self.settings.tab_title;
        collect_running_jobs(self.tabs.iter().map(|tab| {
            let jobs: Vec<Option<String>> = tab.panes.iter().map(|p| p.with_term(|term| term.foreground_job()).flatten()).collect();
            (tab.label(mode), jobs)
        }))
    }

    /// The whole terminal is asked to close: Cmd+Q, the window's close
    /// button, a window manager's close, a host closing the module. `Allow`:
    /// close now. `Veto`: a job runs and the bar now asks; a yes emits
    /// [`ModuleCloseAction::Confirmed`], a no leaves everything running.
    /// A preview pager never asks (it is one throwaway job).
    pub fn request_close_all(&mut self, cx: &mut Cx) -> CloseDecision {
        if !self.tabs_enabled {
            return CloseDecision::Allow;
        }
        let jobs = self.running_jobs();
        let decision = close_all_decision(self.settings.confirm_close_running, self.close_confirmed, &jobs);
        if decision == CloseDecision::Veto {
            self.commit_rename(cx);
            self.pending_close = Some(PendingClose::All { jobs });
            // The bar may be hidden (one tab): the whole terminal relays
            // out around it, not just the bar's last area.
            self.refresh(cx);
        }
        decision
    }

    /// The person answered the question [`Self::request_close_all`] asked.
    fn answer_close_all(&mut self, cx: &mut Cx, close: bool) {
        self.pending_close = None;
        if close {
            self.close_confirmed = true;
            cx.widget_action(self.uid, ModuleCloseAction::Confirmed);
        }
        self.refresh(cx);
    }

    /// Confirm the question in the bar (Enter, the red button).
    fn confirm_pending(&mut self, cx: &mut Cx) {
        match self.pending_close.take() {
            Some(PendingClose::Tab { tab, pane, .. }) => self.finish_close(cx, tab, pane),
            Some(PendingClose::All { .. }) => self.answer_close_all(cx, true),
            None => {}
        }
    }

    pub fn active_index(&self) -> usize {
        self.active
    }

    /// Carry out a tab command, as its key or click would.
    pub fn run_command(&mut self, cx: &mut Cx, command: TabCommand) {
        match command {
            TabCommand::New => {
                let cwd = self.new_cwd();
                self.open_tab(cx, cwd);
            }
            TabCommand::Close => {
                let pane = self.tabs.get(self.active).filter(|t| t.panes.len() > 1).map(|t| t.focused);
                self.request_close(cx, self.active, pane)
            }
            TabCommand::SplitRight => self.split(cx, true),
            TabCommand::SplitDown => self.split(cx, false),
            TabCommand::Focus(dir) => self.focus_towards(cx, dir),
            TabCommand::Zoom => {
                if let Some(tab) = self.tabs.get_mut(self.active) {
                    tab.zoomed = !tab.zoomed && tab.panes.len() > 1;
                }
                self.refresh(cx);
            }
            TabCommand::Next if self.tabs.len() > 1 => self.select(cx, (self.active + 1) % self.tabs.len()),
            TabCommand::Previous if self.tabs.len() > 1 => {
                self.select(cx, (self.active + self.tabs.len() - 1) % self.tabs.len())
            }
            TabCommand::Select(index) if !self.tabs.is_empty() => {
                self.select(cx, index.min(self.tabs.len() - 1))
            }
            TabCommand::Settings => self.toggle_panel(cx),
            TabCommand::CloseTab => self.request_close(cx, self.active, None),
            TabCommand::Rename => self.begin_rename(cx, self.active),
            _ => {}
        }
    }

    /// A key press bound to one of this widget's shortcuts (tabs, splits,
    /// the settings panel): run it. True when the key goes no further;
    /// the rest goes to the focused pane, which runs its own shortcuts and
    /// sends what is left to the program.
    fn key_shortcut(&mut self, cx: &mut Cx, key: &KeyEvent) -> bool {
        let search_open = self
            .tabs
            .get(self.active)
            .and_then(|tab| tab.focused_pane().with_term(|t| t.search_open()))
            == Some(true);
        // An open search bar's own keys win over every shortcut.
        if search_open && search::bar_key(key).is_some() {
            return false;
        }
        let ctx = keybinds::Context {
            tabs: self.tabs.len(),
            search_open,
        };
        let keybinds::Decision::Run { action, consume } =
            self.keys.decide(key, keybinds::Scope::Tabs, &ctx)
        else {
            return false;
        };
        if let Some(command) = tab_command_of(&action) {
            self.run_command(cx, command);
        }
        if consume {
            // Whatever text the platform types for this press is not the
            // program's either.
            if let Some(tab) = self.tabs.get(self.active) {
                tab.focused_pane().with_term(|t| t.swallow_key_text());
            }
        }
        consume
    }

    /// Where a new tab or pane starts: the focused pane's directory, or
    /// home, as the settings say.
    fn new_cwd(&self) -> Option<PathBuf> {
        match self.settings.new_tab_cwd {
            NewTabCwd::Inherit => self.tabs.get(self.active).and_then(|tab| {
                let pane = tab.focused_pane();
                pane.with_term(|term| term.current_dir()).flatten().or_else(|| pane.dir.clone())
            }),
            NewTabCwd::Home => std::env::var_os("HOME").map(PathBuf::from),
        }
    }

    fn new_pane(&mut self, cx: &mut Cx, cwd: Option<PathBuf>) -> Option<Pane> {
        let Some(template) = self.template.as_ref() else {
            error!("TermTabs has no `term` template");
            return None;
        };
        let value: ScriptValue = template.as_object().into();
        let term = cx.with_vm(|vm| WidgetRef::script_from_value(vm, value));
        if let Some(mut t) = term.borrow_mut::<MpTerm>() {
            t.cwd = cwd.clone();
            // The tab reports its title (a name given to it wins over the
            // program's).
            t.titled_by_host = true;
        }
        let id = NEXT_PANE_ID.fetch_add(1, Ordering::Relaxed);
        cx.widget_tree_insert_child(self.uid, LiveId(id), term.clone());
        Some(Pane {
            id,
            term,
            osc_title: String::new(),
            job: None,
            shell: None,
            dir: cwd,
            bell: false,
            screen_hash: 0,
            changed_at: None,
        })
    }

    fn open_tab(&mut self, cx: &mut Cx, cwd: Option<PathBuf>) {
        let Some(pane) = self.new_pane(cx, cwd) else {
            return;
        };
        let at = if self.tabs.is_empty() { 0 } else { self.active + 1 };
        self.tabs.insert(at, Tab::new(pane));
        self.select(cx, at);
    }

    /// Split the focused pane; the new pane takes the keyboard.
    fn split(&mut self, cx: &mut Cx, side_by_side: bool) {
        if self.tabs.is_empty() {
            return;
        }
        let cwd = self.new_cwd();
        let Some(pane) = self.new_pane(cx, cwd) else {
            return;
        };
        let tab = &mut self.tabs[self.active];
        let (at, id) = (tab.focused, pane.id);
        tab.tree.split(at, id, side_by_side);
        tab.panes.push(pane);
        tab.focused = id;
        tab.zoomed = false;
        self.report_title(cx);
        self.refresh(cx);
    }

    fn focus_pane(&mut self, cx: &mut Cx, id: u64) {
        let Some(tab) = self.tabs.get_mut(self.active) else {
            return;
        };
        if let Some(pane) = tab.pane_mut(id) {
            pane.bell = false;
            pane.with_term(|term| term.focus(cx));
            tab.focused = id;
        }
        self.report_title(cx);
        self.refresh(cx);
    }

    fn focus_towards(&mut self, cx: &mut Cx, dir: Dir) {
        let Some(tab) = self.tabs.get(self.active) else {
            return;
        };
        if tab.zoomed {
            return;
        }
        if let Some(id) = panes::neighbour(&tab.tree.layout(self.body), tab.focused, dir) {
            self.focus_pane(cx, id);
        }
    }

    /// Close one pane; the last pane of a tab closes the tab.
    fn close_pane(&mut self, cx: &mut Cx, index: usize, id: u64) {
        let Some(tab) = self.tabs.get_mut(index) else {
            return;
        };
        if tab.panes.len() <= 1 {
            self.close(cx, index);
            return;
        }
        self.pending_close = None;
        let old = tab.layout(self.body);
        tab.panes.retain(|p| p.id != id);
        crate::control::forget(id);
        if let Some(tree) = std::mem::replace(&mut tab.tree, Node::Leaf(0)).remove(id) {
            tab.tree = tree;
        }
        tab.zoomed = false;
        self.publish_soon = true;
        cx.widget_tree_mark_dirty(self.uid);
        if tab.focused == id {
            // The keyboard goes to a neighbour, else the first pane left.
            let next = [Dir::Left, Dir::Up, Dir::Right, Dir::Down]
                .into_iter()
                .find_map(|dir| panes::neighbour(&old, id, dir))
                .filter(|n| tab.panes.iter().any(|p| p.id == *n))
                .unwrap_or(tab.panes[0].id);
            if index == self.active {
                self.focus_pane(cx, next);
            } else {
                tab.focused = next;
            }
        }
        self.refresh(cx);
    }

    fn select(&mut self, cx: &mut Cx, index: usize) {
        if index >= self.tabs.len() {
            return;
        }
        if index != self.active {
            if let Some(old) = self.tabs.get(self.active) {
                for pane in &old.panes {
                    pane.with_term(|term| term.cancel_gestures(cx));
                }
            }
        }
        self.active = index;
        let tab = &mut self.tabs[index];
        for pane in &mut tab.panes {
            pane.bell = false;
        }
        tab.with_term(|term| term.focus(cx));
        self.report_title(cx);
        self.refresh(cx);
    }

    /// Close a tab, or one pane of it, asking first while a job runs there.
    fn request_close(&mut self, cx: &mut Cx, index: usize, pane: Option<u64>) {
        self.commit_rename(cx);
        let Some(tab) = self.tabs.get(index) else {
            return;
        };
        if self.settings.confirm_close_running {
            let job = tab
                .panes
                .iter()
                .filter(|p| pane.is_none_or(|id| p.id == id))
                .find_map(|p| p.with_term(|term| term.foreground_job()).flatten());
            if let Some(job) = job {
                self.pending_close = Some(PendingClose::Tab { tab: index, pane, job });
                if index != self.active {
                    self.select(cx, index);
                }
                self.redraw(cx);
                return;
            }
        }
        self.finish_close(cx, index, pane);
    }

    fn finish_close(&mut self, cx: &mut Cx, index: usize, pane: Option<u64>) {
        match pane {
            Some(id) => self.close_pane(cx, index, id),
            None => self.close(cx, index),
        }
    }

    fn close(&mut self, cx: &mut Cx, index: usize) {
        if index >= self.tabs.len() {
            return;
        }
        self.pending_close = None;
        // Dropping the terminal drops its session: the shell and its jobs
        // get SIGHUP'd with the PTY.
        for pane in &self.tabs[index].panes {
            crate::control::forget(pane.id);
        }
        self.tabs.remove(index);
        self.publish_soon = true;
        cx.widget_tree_mark_dirty(self.uid);
        if self.tabs.is_empty() {
            self.active = 0;
            cx.widget_action(self.uid, TermTabsAction::LastTabClosed);
            self.redraw(cx);
            return;
        }
        if self.active > index || self.active >= self.tabs.len() {
            self.active = self.active.saturating_sub(1);
        }
        let active = self.active;
        self.active = usize::MAX;
        self.select(cx, active);
    }

    /// Tell the host the selected tab's title (the window title).
    fn report_title(&mut self, cx: &mut Cx) {
        self.publish_soon = true;
        let Some(tab) = self.tabs.get(self.active) else {
            return;
        };
        let title = tab.label(self.settings.tab_title);
        if title != self.reported_title {
            self.reported_title = title.clone();
            // Hosted by a window manager (makepad-wm, the OctoSense
            // desktop), the tile's title bar is the WM's (a no-op
            // standalone). A preview keeps the title its host gave it.
            if self.tabs_enabled {
                makepad_wm_api::set_title(cx, &title);
            }
            cx.widget_action(self.uid, MpTermAction::TitleChanged(title));
        }
    }

    fn sync_settings(&mut self, cx: &mut Cx) {
        let generation = term_settings::generation();
        if generation != self.settings_gen {
            self.settings_gen = generation;
            self.settings = term_settings::current();
            self.keys = Keybinds::for_lines(&self.settings.keybinds);
            crate::control::set_enabled(self.settings.external_control);
            self.refresh(cx);
        }
    }

    /// Refresh what each tab runs and where; redraw the bar if a label moved.
    fn poll_tabs(&mut self, cx: &mut Cx) {
        let mut changed = false;
        for tab in &mut self.tabs {
            for pane in &mut tab.panes {
                let facts = pane.with_term(|term| (term.foreground_job(), term.foreground_name(), term.current_dir()));
                if let Some((job, name, dir)) = facts {
                    let shell = if job.is_none() { name } else { pane.shell.clone() };
                    if (&job, &shell, &dir) != (&pane.job, &pane.shell, &pane.dir) {
                        pane.job = job;
                        pane.shell = shell;
                        if dir.is_some() {
                            pane.dir = dir;
                        }
                        changed = true;
                    }
                }
            }
        }
        if changed {
            self.report_title(cx);
            self.redraw(cx);
        }
        if self.settings.external_control {
            self.publish_panes();
        }
    }

    /// A pane's agent and state, from its program, title and screen.
    fn pane_state(pane: &mut Pane) -> (Option<&'static str>, crate::agent::State) {
        use std::hash::{Hash, Hasher};
        let (rows, exited) = pane
            .with_term(|term| (term.ai_screen_rows(None).map(|(rows, _, _)| rows).unwrap_or_default(), term.has_exited()))
            .unwrap_or_default();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        rows.hash(&mut hasher);
        let hash = hasher.finish();
        if hash != pane.screen_hash {
            if pane.screen_hash != 0 {
                pane.changed_at = Some(std::time::Instant::now());
            }
            pane.screen_hash = hash;
        }
        let since_change = pane.changed_at.map(|at| at.elapsed());
        let agent = crate::agent::detect_agent(pane.job.as_deref(), &pane.osc_title);
        let state = crate::agent::detect_state(agent, pane.job.as_deref(), exited, &rows, since_change);
        (agent, state)
    }

    /// Tell the control socket about every pane.
    fn publish_panes(&mut self) {
        let mode = self.settings.tab_title;
        let active = self.active;
        let mut infos = Vec::new();
        for (index, tab) in self.tabs.iter_mut().enumerate() {
            let (focused, custom) = (tab.focused, tab.custom_title.clone());
            for pane in &mut tab.panes {
                let (agent, state) = Self::pane_state(pane);
                infos.push(crate::control::PaneInfo {
                    pane: pane.id,
                    tab: index,
                    title: custom.clone().unwrap_or_else(|| pane.label(mode)),
                    cwd: pane.dir.as_ref().map(|d| d.display().to_string()).unwrap_or_default(),
                    program: pane.job.clone().or_else(|| pane.shell.clone()).unwrap_or_default(),
                    agent,
                    state: state.as_str(),
                    focused: index == active && pane.id == focused,
                });
            }
        }
        crate::control::publish(infos);
    }

    /// Answer the control socket's requests for this widget's panes.
    fn answer_control(&mut self, cx: &mut Cx) {
        use crate::control::{err, ok, Request};
        use makepad_strict_json::{s, Value};
        let ids: Vec<u64> = self.tabs.iter().flat_map(|t| t.panes.iter().map(|p| p.id)).collect();
        for (request, reply) in crate::control::take_requests(|id| ids.contains(&id)) {
            let Some(pane) = self.tabs.iter_mut().flat_map(|t| t.panes.iter_mut()).find(|p| p.id == request.pane()) else {
                let _ = reply.send(err("no such pane"));
                continue;
            };
            let answer = match request {
                Request::Read { lines, .. } => {
                    // The screen (lines = 0), or its last `lines` lines of
                    // text and scrollback: blank rows below the output are
                    // not lines.
                    let recent = (lines > 0).then_some(lines + 1000);
                    match pane.with_term(|term| term.ai_screen_rows(recent)).flatten() {
                        Some((mut rows, cursor_row, cursor_col)) => {
                            while rows.last().is_some_and(|r| r.trim().is_empty()) {
                                rows.pop();
                            }
                            if lines > 0 {
                                rows.drain(..rows.len().saturating_sub(lines));
                            }
                            ok(vec![
                            ("text", s(rows.join("\n").trim_end())),
                            ("cursor", Value::Arr(vec![Value::Int(cursor_row as i64), Value::Int(cursor_col as i64)])),
                            ])
                        }
                        None => err("the pane has no session"),
                    }
                }
                Request::Prompt { text, submit, force, .. } => {
                    let (agent, state) = Self::pane_state(pane);
                    if let Some(why) = crate::agent::refuse_prompt(&text) {
                        err(why)
                    } else if state == crate::agent::State::Blocked && !force {
                        err("an approval or question is on screen (state: blocked)")
                    } else if state == crate::agent::State::Exited {
                        err("the pane's program has exited")
                    } else {
                        let typed = pane
                            .with_term(|term| {
                                let pasted = term.ai_paste(&text);
                                if pasted && submit {
                                    term.ai_type_bytes(b"\r");
                                }
                                pasted
                            })
                            .unwrap_or(false);
                        pane.term.redraw(cx);
                        if typed {
                            ok(vec![("agent", agent.map_or(Value::Null, s)), ("state", s(state.as_str()))])
                        } else {
                            err("the pane could not take input")
                        }
                    }
                }
            };
            let _ = reply.send(answer);
        }
    }

    /// Keys are this widget's when nobody holds the keyboard or one of its
    /// panes does (any tab: a focus change to the selected tab may still be
    /// pending).
    fn keyboard_is_ours(&self, cx: &Cx) -> bool {
        cx.key_focus() == Area::Empty
            || self
                .tabs
                .iter()
                .flat_map(|tab| tab.panes.iter())
                .any(|p| p.with_term(|term| term.has_input_focus(cx)) == Some(true))
    }

    fn show_bar(&self) -> bool {
        self.tabs_enabled && (self.settings.tab_bar == TabBar::Always || self.tabs.len() > 1)
    }

    fn bar_hit(&self, abs: DVec2) -> Option<BarHit> {
        self.hits.iter().rev().find(|(rect, _)| rect.contains(abs)).map(|(_, hit)| *hit)
    }

    fn on_bar_click(&mut self, cx: &mut Cx, hit: BarHit, middle: bool) {
        match hit {
            BarHit::Tab(i) if middle => self.request_close(cx, i, None),
            BarHit::Tab(i) => self.select(cx, i),
            BarHit::CloseTab(i) => self.request_close(cx, i, None),
            BarHit::NewTab => self.run_command(cx, TabCommand::New),
            BarHit::Settings => self.run_command(cx, TabCommand::Settings),
            BarHit::ConfirmClose => self.confirm_pending(cx),
            BarHit::CancelClose => {
                self.pending_close = None;
                self.redraw(cx);
            }
        }
        // A click on the bar must not leave the keyboard with nobody.
        if let Some(tab) = self.tabs.get(self.active) {
            tab.with_term(|term| term.focus(cx));
        }
    }

    /// Start typing a name for tab `index` (a long press or a double click
    /// on it): the tab's label becomes a field.
    fn begin_rename(&mut self, cx: &mut Cx, index: usize) {
        self.cancel_press(cx);
        let Some(tab) = self.tabs.get(index) else {
            return;
        };
        if !self.tabs_enabled || self.rename.as_ref().is_some_and(|r| r.tab == tab.id) {
            return;
        }
        let (id, name) = (tab.id, tab.custom_title.clone().unwrap_or_default());
        self.commit_rename(cx);
        if index != self.active {
            self.select(cx, index);
        }
        self.rename = Some(Rename { tab: id, edit: NameEdit::new(&name), rect: Rect::default() });
        self.redraw(cx);
    }

    /// Take the name being typed (a click elsewhere, a close).
    fn commit_rename(&mut self, cx: &mut Cx) {
        if let Some(rename) = self.rename.take() {
            self.name_tab(cx, rename.tab, rename.edit.committed());
        }
    }

    /// Give tab `id` a name; None gives it its automatic label back.
    fn name_tab(&mut self, cx: &mut Cx, id: u64, name: Option<String>) {
        if let Some(tab) = self.tabs.iter_mut().find(|t| t.id == id) {
            tab.custom_title = name;
        }
        self.report_title(cx);
        self.redraw(cx);
    }

    /// A key while a name is typed: it never reaches the shell.
    fn rename_key(&mut self, cx: &mut Cx, key: &KeyEvent) {
        let Some(rename) = self.rename.as_mut() else {
            return;
        };
        match rename.edit.key(key) {
            NameKey::Editing => {}
            NameKey::Commit(name) => {
                let id = rename.tab;
                self.rename = None;
                self.name_tab(cx, id, name);
            }
            NameKey::Cancel => self.rename = None,
        }
        self.redraw(cx);
    }

    fn rename_text(&mut self, cx: &mut Cx, input: &str) {
        if let Some(rename) = self.rename.as_mut() {
            rename.edit.insert(input);
            self.redraw(cx);
        }
    }

    fn cancel_press(&mut self, cx: &mut Cx) {
        if self.press.take().is_some() {
            cx.stop_timer(self.press_timer);
        }
    }

    /// Route a pane's actions: titles and bells update the pane; the
    /// focused pane of the selected tab also passes them on to the host.
    fn take_pane_actions(&mut self, cx: &mut Cx, index: usize, id: u64, actions: ActionsBuf, exited: &mut Vec<(usize, u64)>) {
        let selected = index == self.active && self.tabs[index].focused == id;
        let mut forward = ActionsBuf::new();
        for action in actions {
            let Some(wa) = action.as_widget_action() else {
                forward.push(action);
                continue;
            };
            match wa.cast::<MpTermAction>() {
                MpTermAction::TitleChanged(title) => {
                    let Some(pane) = self.tabs[index].pane_mut(id) else {
                        continue;
                    };
                    if pane.osc_title != title {
                        pane.osc_title = title;
                        self.redraw(cx);
                        if selected {
                            self.report_title(cx);
                        }
                    }
                    continue;
                }
                MpTermAction::Bell if !selected => {
                    if let Some(pane) = self.tabs[index].pane_mut(id) {
                        pane.bell = true;
                    }
                    self.redraw(cx);
                }
                // A shell that exits closes its pane, unless it is the last
                // of the last tab: that one keeps the terminal's own exited
                // state.
                MpTermAction::Exited if self.tabs.len() > 1 || self.tabs[index].panes.len() > 1 => {
                    exited.push((index, id));
                    continue;
                }
                _ => {}
            }
            if selected {
                forward.push(action);
            }
        }
        if !forward.is_empty() {
            cx.extend_actions(forward);
        }
    }
}

/// Keyboard events: the focused pane's alone.
fn is_key_input(event: &Event) -> bool {
    matches!(
        event,
        Event::KeyDown(_)
            | Event::KeyUp(_)
            | Event::TextInput(_)
            | Event::TextRangeReplace(_)
            | Event::TextCopy(_)
            | Event::TextCut(_)
    )
}

/// Events every tab needs, selected or not: only the selected tab gets
/// input (its area is the only one drawn).
fn is_input(event: &Event) -> bool {
    matches!(
        event,
        Event::MouseDown(_)
            | Event::MouseMove(_)
            | Event::MouseUp(_)
            | Event::MouseLeave(_)
            | Event::TouchUpdate(_)
            | Event::LongPress(_)
            | Event::Scroll(_)
            | Event::KeyDown(_)
            | Event::KeyUp(_)
            | Event::TextInput(_)
            | Event::TextRangeReplace(_)
            | Event::TextCopy(_)
            | Event::TextCut(_)
            | Event::Drag(_)
            | Event::Drop(_)
    )
}

fn mix(a: Vec4f, b: Vec4f, t: f32) -> Vec4f {
    vec4(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        a.w + (b.w - a.w) * t,
    )
}

fn luminance(c: Vec4f) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

const ICON_PLUS: &str = "\u{f067}";
const ICON_CLOSE: &str = "\u{f00d}";
const ICON_GEAR: &str = "\u{f013}";
const ICON_PREV: &str = "\u{f053}";
const ICON_NEXT: &str = "\u{f054}";

impl Widget for TermTabs {
    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        // Changes an earlier event made (shortcuts return early), published
        // before this one runs: the key-up follows within milliseconds.
        if std::mem::take(&mut self.publish_soon) && self.settings.external_control {
            self.publish_panes();
        }
        self.sync_settings(cx);
        if self.settings.external_control {
            self.answer_control(cx);
        }
        if self.apply_timer.is_event(event).is_some() {
            if let Some(next) = self.pending_apply.take() {
                self.apply(cx, next);
                self.refresh(cx);
            }
        }
        if self.poll_timer.is_event(event).is_some() {
            self.poll_tabs(cx);
        }
        // A press held on a tab: rename it.
        if self.press_timer.is_event(event).is_some() {
            if let Some(press) = self.press.take() {
                if let Some(index) = self.tabs.iter().position(|t| t.id == press.tab) {
                    self.begin_rename(cx, index);
                }
            }
        }

        if self.tabs_enabled && self.keyboard_is_ours(cx) {
            if self.pending_close.is_some() {
                // The confirmation owns the keyboard until it is answered.
                match event {
                    Event::KeyDown(key) => {
                        match key.key_code {
                            KeyCode::ReturnKey | KeyCode::KeyY => self.confirm_pending(cx),
                            KeyCode::Escape | KeyCode::KeyN => {
                                self.pending_close = None;
                                self.redraw(cx);
                            }
                            _ => {}
                        }
                        return;
                    }
                    Event::KeyUp(_) | Event::TextInput(_) => return,
                    _ => {}
                }
            } else if self.rename.is_some() {
                // The name being typed owns the keyboard: nothing reaches
                // the shell until it is taken or dropped.
                match event {
                    Event::KeyDown(key) => {
                        self.rename_key(cx, key);
                        return;
                    }
                    Event::TextInput(input) => {
                        self.rename_text(cx, &input.input);
                        return;
                    }
                    Event::KeyUp(_) | Event::TextRangeReplace(_) | Event::TextCopy(_) | Event::TextCut(_) => return,
                    _ => {}
                }
            } else if self.panel.is_some() {
                // The panel owns the keyboard while it is open.
                match event {
                    Event::KeyDown(key) => {
                        self.panel_key(cx, key);
                        return;
                    }
                    Event::TextInput(input) => {
                        if !input.was_paste || self.panel.as_ref().is_some_and(|p| !matches!(p.mode, PanelMode::Rows)) {
                            self.panel_text(cx, &input.input);
                        }
                        return;
                    }
                    Event::KeyUp(_) => return,
                    _ => {}
                }
            } else if let Event::KeyDown(key) = event {
                if self.key_shortcut(cx, key) {
                    return;
                }
            }
        }

        // A click or touch outside the name being typed takes it.
        if let Some(rect) = self.rename.as_ref().map(|r| r.rect) {
            let outside = match event {
                Event::MouseDown(e) => !rect.contains(e.abs),
                Event::TouchUpdate(e) => {
                    e.touches.iter().any(|t| t.state == TouchState::Start && !rect.contains(t.abs))
                }
                _ => false,
            };
            if outside {
                self.commit_rename(cx);
            }
        }

        match event.hits(cx, self.draw_bar.area()) {
            Hit::FingerHoverIn(e) | Hit::FingerHoverOver(e) => {
                let hover = self.bar_hit(e.abs);
                if hover != self.hover {
                    self.hover = hover;
                    self.redraw(cx);
                }
            }
            Hit::FingerHoverOut(_) => {
                if self.hover.take().is_some() {
                    self.redraw(cx);
                }
            }
            // A press in the name being typed stays with it.
            Hit::FingerDown(e) if self.rename.as_ref().is_some_and(|r| r.rect.contains(e.abs)) => {}
            Hit::FingerDown(e) => {
                let middle = e.device.mouse_button().is_some_and(|b| b.contains(MouseButton::MIDDLE));
                let primary = !middle && e.device.mouse_button().is_none_or(|b| b.contains(MouseButton::PRIMARY));
                self.cancel_press(cx);
                match self.bar_hit(e.abs) {
                    // A double click on a tab renames it (as does a long
                    // press: held, below).
                    Some(BarHit::Tab(i)) if primary && e.tap_count == 2 => {
                        self.on_bar_click(cx, BarHit::Tab(i), false);
                        self.begin_rename(cx, i);
                    }
                    Some(BarHit::Tab(i)) if primary => {
                        self.on_bar_click(cx, BarHit::Tab(i), false);
                        if let Some(tab) = self.tabs.get(i) {
                            self.press = Some(TabPress { tab: tab.id, abs: e.abs });
                            self.press_timer = cx.start_timeout(HOLD_TO_RENAME);
                        }
                    }
                    Some(hit) => self.on_bar_click(cx, hit, middle),
                    // A double click on the empty bar opens a tab.
                    None if e.tap_count == 2 => self.on_bar_click(cx, BarHit::NewTab, false),
                    None => {}
                }
            }
            Hit::FingerMove(e) => {
                if self.press.as_ref().is_some_and(|p| (e.abs - p.abs).length() > HOLD_SLOP) {
                    self.cancel_press(cx);
                }
            }
            Hit::FingerUp(_) => self.cancel_press(cx),
            // A touch platform's own long press (Android, iOS).
            Hit::FingerLongPress(e) => {
                if let Some(BarHit::Tab(i)) = self.bar_hit(e.abs) {
                    self.begin_rename(cx, i);
                }
            }
            _ => {}
        }

        if self.panel.is_some() {
            match event.hits(cx, self.draw_panel.area()) {
                Hit::FingerDown(e) => {
                    let hit = self.panel.as_ref().and_then(|panel| {
                        panel.hits.iter().rev().find(|(rect, _)| rect.contains(e.abs)).map(|(_, hit)| *hit)
                    });
                    if let Some(hit) = hit {
                        self.panel_click(cx, hit);
                    }
                }
                Hit::FingerScroll(e) => {
                    let step = if e.scroll.y > 0.0 { 1 } else if e.scroll.y < 0.0 { -1 } else { 0 };
                    self.panel_scroll(cx, step);
                }
                _ => {}
            }
            // The terminal under the panel is inert; a click on it closes
            // the panel.
            if let Event::MouseDown(e) = event {
                let inside = self.panel.as_ref().is_some_and(|panel| panel.rect.contains(e.abs))
                    || self.hits.iter().any(|(rect, _)| rect.contains(e.abs));
                if !inside {
                    self.toggle_panel(cx);
                }
            }
        }

        if self.panel.is_none() && self.drag_divider(cx, event) {
            return;
        }
        // A click in another pane focuses it: decided by where it landed,
        // not by the key focus (that change lands after this event).
        if let (Event::MouseDown(e), None) = (event, self.panel.as_ref()) {
            let clicked = self
                .tabs
                .get(self.active)
                .and_then(|tab| tab.layout(self.body).into_iter().find(|(_, r)| r.contains(e.abs)).map(|(id, _)| id))
                .filter(|id| self.tabs.get(self.active).is_some_and(|t| t.focused != *id));
            if let Some(id) = clicked {
                self.focus_pane(cx, id);
            }
        }

        let mut exited = Vec::new();
        for index in 0..self.tabs.len() {
            let in_view = index == self.active && self.panel.is_none();
            let (focused, zoomed) = (self.tabs[index].focused, self.tabs[index].zoomed);
            let ids: Vec<(u64, WidgetRef)> = self.tabs[index].panes.iter().map(|p| (p.id, p.term.clone())).collect();
            for (id, term) in ids {
                // Every pane pumps its PTY; the keyboard goes to the focused
                // pane, the pointer to the panes in view.
                let deliver = if !is_input(event) {
                    true
                } else if !in_view {
                    false
                } else if is_key_input(event) {
                    id == focused
                } else {
                    !zoomed || id == focused
                };
                if !deliver {
                    continue;
                }
                let route = in_view && id == focused && is_key_input(event);
                if route {
                    if let Some(mut t) = term.borrow_mut::<MpTerm>() {
                        t.route_keys_here = true;
                    }
                }
                let actions = cx.capture_actions(|cx| term.handle_event(cx, event, scope));
                if route {
                    if let Some(mut t) = term.borrow_mut::<MpTerm>() {
                        t.route_keys_here = false;
                    }
                }
                if !actions.is_empty() {
                    self.take_pane_actions(cx, index, id, actions, &mut exited);
                }
            }
        }
        exited.sort_unstable();
        exited.dedup();
        for (index, id) in exited.into_iter().rev() {
            self.close_pane(cx, index, id);
        }
        if std::mem::take(&mut self.publish_soon) && self.settings.external_control {
            self.publish_panes();
        }
    }

    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        self.sync_settings(cx);
        if self.tabs.is_empty() {
            self.open_tab(cx, None);
        }
        if self.poll_timer.is_empty() {
            self.poll_timer = cx.start_interval(1.0);
            // Font settings and the font list need the installed fonts.
            crate::fonts::warm();
        }
        cx.begin_turtle(walk, self.layout);
        self.hits.clear();
        if self.show_bar() || self.pending_close.is_some() {
            self.draw_tab_bar(cx);
        }
        let body = cx.walk_turtle(Walk::fill());
        self.body = body;
        self.draw_panes(cx, scope, body);
        cx.end_turtle();
        if let Some(mut list) = self.panel_list.take() {
            list.begin_overlay_reuse(cx);
            // Cover the body's far corner: a host that seats the terminal
            // in a tile draws it at an offset inside a larger pass.
            let pass = cx.current_pass_size();
            let size = dvec2(pass.x.max(body.pos.x + body.size.x), pass.y.max(body.pos.y + body.size.y));
            cx.begin_root_turtle(size, Layout::default());
            if self.panel.is_some() {
                self.draw_settings_panel(cx, body);
            }
            cx.end_pass_sized_turtle();
            list.end(cx);
            self.panel_list = Some(list);
        }
        DrawStep::done()
    }
}

impl TermTabs {
    /// The selected tab's panes, the dividers between them; unfocused
    /// panes dimmed a little.
    fn draw_panes(&mut self, cx: &mut Cx2d, scope: &mut Scope, body: Rect) {
        let Some(tab) = self.tabs.get(self.active) else {
            self.dividers.clear();
            return;
        };
        let layout = tab.layout(body);
        let split = layout.len() > 1;
        self.dividers = if split { tab.tree.dividers(body) } else { Vec::new() };
        if split {
            let (bg, _) = self.chrome();
            self.draw_divider.color = mix(bg, vec4(0.0, 0.0, 0.0, 1.0), if luminance(bg) < 0.5 { 0.45 } else { 0.15 });
            for divider in &self.dividers {
                self.draw_divider.draw_abs(cx, divider.rect);
            }
        }
        for (id, rect) in layout {
            let Some(pane) = tab.panes.iter().find(|p| p.id == id) else {
                continue;
            };
            let dim = if split && id != tab.focused { 0.18 } else { 0.0 };
            pane.with_term(|term| term.set_background_dimming(cx, dim));
            let walk = Walk::new(Size::Fixed(rect.size.x), Size::Fixed(rect.size.y)).with_abs_pos(rect.pos);
            pane.term.draw_walk_all(cx, scope, walk);
        }
    }

    /// Resize splits by dragging the gap between panes. True when the event
    /// was the drag's.
    fn drag_divider(&mut self, cx: &mut Cx, event: &Event) -> bool {
        match event {
            Event::MouseDown(e) => {
                let grab = self.dividers.iter().find(|d| {
                    let r = d.rect;
                    let pad = 2.0;
                    Rect { pos: dvec2(r.pos.x - pad, r.pos.y - pad), size: dvec2(r.size.x + pad * 2.0, r.size.y + pad * 2.0) }
                        .contains(e.abs)
                });
                if let Some(divider) = grab {
                    self.drag = Some(divider.clone());
                    return true;
                }
                false
            }
            Event::MouseMove(e) => {
                if let Some(divider) = &self.drag {
                    let at = if divider.side_by_side { e.abs.x } else { e.abs.y };
                    let ratio = panes::ratio_at(divider, at);
                    let path = divider.path.clone();
                    if let Some(tab) = self.tabs.get_mut(self.active) {
                        tab.tree.set_ratio(&path, ratio);
                    }
                    self.refresh(cx);
                    return true;
                }
                if let Some(divider) = self.dividers.iter().find(|d| d.rect.contains(e.abs)) {
                    cx.set_cursor(if divider.side_by_side { MouseCursor::ColResize } else { MouseCursor::RowResize });
                }
                false
            }
            Event::MouseUp(_) => self.drag.take().is_some(),
            _ => false,
        }
    }

    /// (background, foreground) of the selected tab's colours.
    fn chrome(&self) -> (Vec4f, Vec4f) {
        self.tabs
            .get(self.active)
            .and_then(|tab| tab.with_term(|term| term.chrome_colors()).flatten())
            .unwrap_or((vec4(0.102, 0.106, 0.149, 1.0), vec4(0.663, 0.694, 0.839, 1.0)))
    }

    fn draw_tab_bar(&mut self, cx: &mut Cx2d) {
        let (bg, fg) = self.chrome();
        let dark = luminance(bg) < 0.5;
        let bar_bg = mix(bg, vec4(0.0, 0.0, 0.0, 1.0), if dark { 0.35 } else { 0.08 });
        let hover_bg = mix(bar_bg, fg, 0.08);
        let dim_fg = mix(bar_bg, fg, 0.6);

        let h = self.bar_height;
        let bar = cx.walk_turtle(Walk::new(Size::fill(), Size::Fixed(h)));
        self.draw_bar.color = bar_bg;
        self.draw_bar.draw_abs(cx, bar);

        if let Some(pending) = self.pending_close.as_ref() {
            let message = match pending {
                PendingClose::Tab { pane, job, .. } => {
                    let what = if pane.is_some() { "pane" } else { "tab" };
                    format!("\u{201c}{job}\u{201d} is running in this {what}. Close it?")
                }
                PendingClose::All { jobs } => close_all_message(jobs),
            };
            let mut x = bar.pos.x + bar.size.x - 8.0;
            // The buttons' room first, so a long list of jobs is cut short
            // rather than drawn under them.
            let buttons: f64 = ["Cancel  (Esc)", "Close  (Enter)"].iter().map(|t| self.text_width(cx, t) + 28.0).sum();
            let message = self.fit(cx, &message, (bar.size.x - 8.0 - buttons - 24.0).max(0.0));
            self.draw_label.color = fg;
            self.draw_label.draw_abs(cx, dvec2(bar.pos.x + 12.0, bar.pos.y + (h - 12.0) * 0.5), &message);
            for (text, hit, fill) in [
                ("Cancel  (Esc)", BarHit::CancelClose, hover_bg),
                ("Close  (Enter)", BarHit::ConfirmClose, vec4(0.85, 0.30, 0.35, 1.0)),
            ] {
                let width = self.text_width(cx, text) + 20.0;
                x -= width;
                let rect = Rect { pos: dvec2(x, bar.pos.y + 4.0), size: dvec2(width, h - 8.0) };
                self.draw_tab.tab = 0.0;
                self.draw_tab.color = if self.hover == Some(hit) { mix(fill, fg, 0.15) } else { fill };
                self.draw_tab.draw_abs(cx, rect);
                self.draw_label.color = if hit == BarHit::ConfirmClose { vec4(1.0, 1.0, 1.0, 1.0) } else { fg };
                self.draw_label.draw_abs(cx, dvec2(x + 10.0, bar.pos.y + (h - 12.0) * 0.5), text);
                self.hits.push((rect, hit));
                x -= 8.0;
            }
            return;
        }

        let button = h - 6.0;
        let pad = 6.0;
        let gear = Rect { pos: dvec2(bar.pos.x + bar.size.x - pad - button, bar.pos.y + 3.0), size: dvec2(button, button) };
        let tabs_room = (gear.pos.x - bar.pos.x - pad - button - 8.0).max(0.0);
        let count = self.tabs.len().max(1) as f64;
        let tab_w = (tabs_room / count).clamp(48.0, 220.0);
        let mut x = bar.pos.x + pad;
        for index in 0..self.tabs.len() {
            let rect = Rect { pos: dvec2(x, bar.pos.y + 4.0), size: dvec2(tab_w - 2.0, h - 4.0) };
            let selected = index == self.active;
            let hovered = matches!(self.hover, Some(BarHit::Tab(i)) | Some(BarHit::CloseTab(i)) if i == index);
            if selected || hovered {
                self.draw_tab.tab = 1.0;
                self.draw_tab.color = if selected { bg } else { hover_bg };
                self.draw_tab.draw_abs(cx, rect);
            }
            self.hits.push((rect, BarHit::Tab(index)));

            let close = Rect {
                pos: dvec2(rect.pos.x + rect.size.x - 20.0, rect.pos.y + (rect.size.y - 16.0) * 0.5),
                size: dvec2(16.0, 16.0),
            };
            let tab = &self.tabs[index];
            if self.rename.as_ref().is_some_and(|r| r.tab == tab.id) {
                // Nothing typed: the automatic label, dimmed, is what the
                // tab is called once the field is taken empty.
                let placeholder = tab.focused_pane().label(self.settings.tab_title);
                self.draw_name_field(cx, rect, bg, fg, dim_fg, &placeholder);
                x += tab_w;
                continue;
            }
            let show_close = selected || hovered;
            let mut label = tab.label(self.settings.tab_title);
            if tab.bell() {
                label = format!("\u{2022} {label}");
            }
            let room = rect.size.x - 20.0 - if show_close { 20.0 } else { 0.0 };
            let label = self.fit(cx, &label, room);
            self.draw_label.color = if selected { fg } else { dim_fg };
            self.draw_label.draw_abs(cx, dvec2(rect.pos.x + 10.0, rect.pos.y + (rect.size.y - 12.0) * 0.5), &label);
            if show_close {
                if self.hover == Some(BarHit::CloseTab(index)) {
                    self.draw_tab.tab = 0.0;
                    self.draw_tab.color = mix(bg, fg, 0.15);
                    self.draw_tab.draw_abs(cx, close);
                }
                self.draw_icon.color = dim_fg;
                self.draw_icon.draw_abs(cx, dvec2(close.pos.x + 4.0, close.pos.y + 3.0), ICON_CLOSE);
                self.hits.push((close, BarHit::CloseTab(index)));
            }
            x += tab_w;
        }

        let plus = Rect { pos: dvec2(x + 2.0, bar.pos.y + 3.0), size: dvec2(button, button) };
        for (rect, hit, icon) in [(plus, BarHit::NewTab, ICON_PLUS), (gear, BarHit::Settings, ICON_GEAR)] {
            if self.hover == Some(hit) {
                self.draw_tab.tab = 0.0;
                self.draw_tab.color = hover_bg;
                self.draw_tab.draw_abs(cx, rect);
            }
            self.draw_icon.color = dim_fg;
            self.draw_icon.draw_abs(cx, dvec2(rect.pos.x + (button - 10.0) * 0.5, rect.pos.y + (button - 12.0) * 0.5), icon);
            self.hits.push((rect, hit));
        }
    }

    /// The name being typed, in place of tab `rect`'s label: a field with
    /// the text and a caret, kept in view as it grows past the tab.
    fn draw_name_field(&mut self, cx: &mut Cx2d, rect: Rect, bg: Vec4f, fg: Vec4f, dim_fg: Vec4f, placeholder: &str) {
        let Some(edit) = self.rename.as_ref().map(|r| r.edit.clone()) else {
            return;
        };
        let field = Rect { pos: dvec2(rect.pos.x + 3.0, rect.pos.y + 2.0), size: dvec2(rect.size.x - 6.0, rect.size.y - 5.0) };
        self.draw_tab.tab = 1.0;
        self.draw_tab.color = bg;
        self.draw_tab.draw_abs(cx, rect);
        self.draw_tab.tab = 0.0;
        self.draw_tab.color = mix(bg, fg, 0.14);
        self.draw_tab.draw_abs(cx, field);
        let text_x = field.pos.x + 7.0;
        let text_y = rect.pos.y + (rect.size.y - 12.0) * 0.5;
        let room = field.size.x - 14.0;
        let chars: Vec<char> = edit.text.chars().collect();
        let caret = edit.caret.min(chars.len());
        let caret_x = if chars.is_empty() {
            self.draw_label.color = mix(bg, dim_fg, 0.7);
            let shown = self.fit(cx, placeholder, room);
            self.draw_label.draw_abs(cx, dvec2(text_x, text_y), &shown);
            text_x
        } else {
            // Scroll the text so the caret stays in the field.
            let mut start = 0;
            while start < caret && self.text_width(cx, &chars[start..caret].iter().collect::<String>()) > room - 2.0 {
                start += 1;
            }
            let shown = self.fit(cx, &chars[start..].iter().collect::<String>(), room);
            self.draw_label.color = fg;
            self.draw_label.draw_abs(cx, dvec2(text_x, text_y), &shown);
            text_x + self.text_width(cx, &chars[start..caret].iter().collect::<String>())
        };
        self.draw_label.color = fg;
        self.draw_label.draw_abs(cx, dvec2(caret_x - 1.0, text_y), "\u{258f}");
        if let Some(rename) = self.rename.as_mut() {
            rename.rect = rect;
        }
    }

    /// Redraw the bar, the selected terminal and the panel's overlay (the
    /// bar's area is empty while it is hidden).
    fn refresh(&mut self, cx: &mut Cx) {
        self.redraw(cx);
        if let Some(tab) = self.tabs.get(self.active) {
            for pane in &tab.panes {
                pane.term.redraw(cx);
            }
        }
        if let Some(list) = &self.panel_list {
            list.redraw(cx);
        }
    }

    fn toggle_panel(&mut self, cx: &mut Cx) {
        self.panel = match self.panel.take() {
            Some(_) => None,
            None => {
                // The font rows list installed fonts: start looking now.
                crate::fonts::warm();
                Some(Panel::default())
            }
        };
        self.refresh(cx);
    }

    fn panel_key(&mut self, cx: &mut Cx, key: &KeyEvent) {
        let Some(panel) = self.panel.as_mut() else {
            return;
        };
        panel.message = None;
        match &mut panel.mode {
            PanelMode::Rows => self.rows_key(cx, key),
            PanelMode::Choose(_) => self.chooser_key(cx, key),
            PanelMode::Name(name) => match key.key_code {
                KeyCode::ReturnKey => {
                    let name = name.trim().to_owned();
                    self.save_profile(cx, &name);
                }
                KeyCode::Escape => panel.mode = PanelMode::Rows,
                KeyCode::Backspace => {
                    name.pop();
                }
                _ => {}
            },
        }
        self.refresh(cx);
    }

    /// Typed text goes to the list filter or the profile name.
    fn panel_text(&mut self, cx: &mut Cx, input: &str) {
        let Some(panel) = self.panel.as_mut() else {
            return;
        };
        let typed: String = input.chars().filter(|c| !c.is_control()).collect();
        match &mut panel.mode {
            PanelMode::Choose(chooser) => {
                let room = 64usize.saturating_sub(chooser.filter.chars().count());
                chooser.filter.extend(typed.chars().take(room));
                chooser.selected = 0;
                chooser.scroll = 0;
            }
            PanelMode::Name(name) => {
                if name.chars().count() + typed.chars().count() <= 40 {
                    name.push_str(&typed);
                }
            }
            PanelMode::Rows => return,
        }
        self.refresh(cx);
    }

    fn rows_key(&mut self, cx: &mut Cx, key: &KeyEvent) {
        let rows = settings_panel::rows();
        let Some(panel) = self.panel.as_mut() else {
            return;
        };
        let selected = panel.selected.min(rows.len() - 1);
        let row = rows[selected];
        if !matches!(key.key_code, KeyCode::ReturnKey | KeyCode::Space) {
            panel.confirm_delete = false;
        }
        match key.key_code {
            KeyCode::Escape => self.panel = None,
            // The settings shortcut (Ctrl+, by default) closes it again.
            _ if self.keys.lookup(key).is_some_and(|b| b.action == keybinds::Action::OpenConfig) => self.panel = None,
            KeyCode::ArrowUp => panel.selected = selected.saturating_sub(1),
            KeyCode::ArrowDown => panel.selected = (selected + 1).min(rows.len() - 1),
            KeyCode::ArrowLeft if row.kind() == RowKind::Value => self.apply_step(cx, row, -1),
            KeyCode::ArrowRight if row.kind() == RowKind::Value => self.apply_step(cx, row, 1),
            KeyCode::ReturnKey | KeyCode::Space => self.activate_row(cx, selected),
            _ => return,
        }
        if let Some(panel) = self.panel.as_mut() {
            let visible = panel.visible.max(1);
            if panel.selected < panel.scroll {
                panel.scroll = panel.selected;
            } else if panel.selected >= panel.scroll + visible {
                panel.scroll = panel.selected + 1 - visible;
            }
        }
    }

    /// Enter on a row: flip a switch, open a long list, run an action, or
    /// step a short value.
    fn activate_row(&mut self, cx: &mut Cx, index: usize) {
        let row = settings_panel::rows()[index];
        match row.kind() {
            RowKind::Toggle => self.apply_step(cx, row, 1),
            RowKind::Action => self.row_action(cx, row),
            RowKind::Value => match row.choices(&self.settings) {
                Some(choices) => self.open_chooser(row, choices),
                None => self.apply_step(cx, row, 1),
            },
        }
    }

    fn open_chooser(&mut self, row: Row, choices: Vec<Choice>) {
        let current = row.current(&self.settings);
        let selected = choices.iter().position(|c| c.value == current).unwrap_or(0);
        if let Some(panel) = self.panel.as_mut() {
            if choices.is_empty() && row == Row::Profile {
                panel.message = Some("No saved profiles yet".into());
                return;
            }
            panel.mode = PanelMode::Choose(Chooser {
                row,
                choices,
                filter: String::new(),
                selected,
                scroll: selected.saturating_sub(4),
                shown_selected: Some(selected),
                original: self.settings.clone(),
                awaiting_fonts: matches!(row, Row::Font | Row::CjkFont) && !crate::fonts::ready(),
            });
        }
    }

    fn chooser_key(&mut self, cx: &mut Cx, key: &KeyEvent) {
        let Some(PanelMode::Choose(chooser)) = self.panel.as_mut().map(|p| &mut p.mode) else {
            return;
        };
        let shown = settings_panel::filter_choices(&chooser.choices, &chooser.filter);
        let last = shown.len().saturating_sub(1);
        let before = chooser.selected;
        match key.key_code {
            KeyCode::ArrowUp => chooser.selected = chooser.selected.saturating_sub(1),
            KeyCode::ArrowDown => chooser.selected = (chooser.selected + 1).min(last),
            KeyCode::PageUp => chooser.selected = chooser.selected.saturating_sub(10),
            KeyCode::PageDown => chooser.selected = (chooser.selected + 10).min(last),
            KeyCode::Backspace => {
                chooser.filter.pop();
                chooser.selected = 0;
                chooser.scroll = 0;
            }
            KeyCode::ReturnKey => {
                if let Some(choice) = shown.get(chooser.selected) {
                    let (row, value) = (chooser.row, choice.value.clone());
                    self.pick(cx, row, &value);
                }
                return;
            }
            KeyCode::Escape => {
                // Back out: undo the live preview.
                let original = chooser.original.clone();
                if let Some(panel) = self.panel.as_mut() {
                    panel.mode = PanelMode::Rows;
                }
                if original != self.settings {
                    self.apply(cx, original);
                }
                return;
            }
            _ => return,
        }
        let (row, selected) = (chooser.row, chooser.selected);
        if selected != before && row != Row::Profile && !row.read_only() {
            // Preview as the selection moves (themes, fonts, the shell).
            if let Some(choice) = shown.get(selected) {
                let next = row.with_value(&self.current_settings(), &choice.value);
                if matches!(row, Row::Font | Row::CjkFont) {
                    self.apply_later(cx, next);
                } else {
                    self.apply(cx, next);
                }
            }
        }
    }

    /// Take `value` for `row` and close its list.
    fn pick(&mut self, cx: &mut Cx, row: Row, value: &str) {
        let next = row.with_value(&self.current_settings(), value);
        if let Some(panel) = self.panel.as_mut() {
            panel.mode = PanelMode::Rows;
            if row.read_only() {
                // A list to read: nothing to take, nothing to save.
                return;
            }
            if row == Row::Profile {
                panel.message = Some(format!("Loaded \u{201c}{value}\u{201d}"));
            }
        }
        self.apply(cx, next);
    }

    fn row_action(&mut self, cx: &mut Cx, row: Row) {
        let profile = self.settings.profile.clone();
        let Some(panel) = self.panel.as_mut() else {
            return;
        };
        match row {
            Row::Shortcuts => {
                let choices = row.choices(&self.settings).unwrap_or_default();
                self.open_chooser(row, choices);
            }
            Row::SaveProfile => panel.mode = PanelMode::Name(profile),
            Row::DeleteProfile if profile.is_empty() => panel.message = Some("No profile is loaded".into()),
            Row::DeleteProfile if !panel.confirm_delete => panel.confirm_delete = true,
            Row::DeleteProfile => {
                panel.confirm_delete = false;
                panel.message = Some(match term_settings::delete_profile(&profile) {
                    Ok(()) => format!("Deleted \u{201c}{profile}\u{201d}"),
                    Err(err) => format!("Could not delete: {err}"),
                });
                let next = Settings { profile: String::new(), ..self.settings.clone() };
                self.apply(cx, next);
            }
            _ => {}
        }
    }

    fn save_profile(&mut self, cx: &mut Cx, name: &str) {
        if !term_settings::valid_profile_name(name) {
            if let Some(panel) = self.panel.as_mut() {
                panel.message = Some("Use 1\u{2013}40 characters, without / \\ : = #".into());
            }
            return;
        }
        let result = term_settings::save_profile(name, &self.settings);
        if let Some(panel) = self.panel.as_mut() {
            panel.mode = PanelMode::Rows;
            panel.message = Some(match &result {
                Ok(()) => format!("Saved \u{201c}{name}\u{201d}"),
                Err(err) => format!("Could not save: {err}"),
            });
        }
        if result.is_ok() {
            let next = Settings { profile: name.to_owned(), ..self.settings.clone() };
            self.apply(cx, next);
        }
    }

    fn panel_click(&mut self, cx: &mut Cx, hit: PanelHit) {
        let rows = settings_panel::rows();
        if let Some(panel) = self.panel.as_mut() {
            panel.message = None;
            if !matches!(hit, PanelHit::Activate(i) if rows.get(i) == Some(&Row::DeleteProfile)) {
                panel.confirm_delete = false;
            }
        }
        match hit {
            PanelHit::Close => self.panel = None,
            PanelHit::Select(i) => {
                if let Some(panel) = self.panel.as_mut() {
                    panel.selected = i;
                }
            }
            PanelHit::Activate(i) => {
                if let Some(panel) = self.panel.as_mut() {
                    panel.selected = i;
                }
                self.activate_row(cx, i);
            }
            PanelHit::Step(i, dir) => {
                if let Some(panel) = self.panel.as_mut() {
                    panel.selected = i;
                }
                self.apply_step(cx, rows[i], dir);
            }
            PanelHit::Choice(i) => {
                let picked = match self.panel.as_ref().map(|p| &p.mode) {
                    Some(PanelMode::Choose(chooser)) => settings_panel::filter_choices(&chooser.choices, &chooser.filter)
                        .get(i)
                        .map(|choice| (chooser.row, choice.value.clone())),
                    _ => None,
                };
                if let Some((row, value)) = picked {
                    self.pick(cx, row, &value);
                }
            }
            PanelHit::Back => {
                let original = match self.panel.as_ref().map(|p| &p.mode) {
                    Some(PanelMode::Choose(chooser)) => Some(chooser.original.clone()),
                    _ => None,
                };
                if let Some(panel) = self.panel.as_mut() {
                    panel.mode = PanelMode::Rows;
                }
                if let Some(original) = original.filter(|o| *o != self.settings) {
                    self.apply(cx, original);
                }
            }
            PanelHit::Reset => {
                // Shortcuts are the file's to change, not the panel's: a
                // reset keeps the `keybind` lines it cannot show again.
                let next = Settings {
                    profile: self.settings.profile.clone(),
                    keybinds: self.settings.keybinds.clone(),
                    ..Settings::default()
                };
                self.apply(cx, next);
            }
        }
        self.refresh(cx);
    }

    fn panel_scroll(&mut self, cx: &mut Cx, step: i64) {
        let rows = settings_panel::rows().len();
        let Some(panel) = self.panel.as_mut() else {
            return;
        };
        match &mut panel.mode {
            PanelMode::Choose(chooser) => {
                let shown = settings_panel::filter_choices(&chooser.choices, &chooser.filter).len();
                let max = shown.saturating_sub(panel.visible.max(1));
                chooser.scroll = (chooser.scroll as i64 + step * 3).clamp(0, max as i64) as usize;
            }
            _ => {
                let max = rows.saturating_sub(panel.visible.max(1));
                panel.scroll = (panel.scroll as i64 + step).clamp(0, max as i64) as usize;
            }
        }
        self.refresh(cx);
    }

    fn apply_step(&mut self, cx: &mut Cx, row: Row, dir: i32) {
        let next = row.step(&self.current_settings(), dir);
        if matches!(row, Row::Font | Row::CjkFont) {
            self.apply_later(cx, next);
        } else {
            self.apply(cx, next);
        }
    }

    /// The settings as the panel shows them: a waiting font change included.
    fn current_settings(&self) -> Settings {
        self.pending_apply.clone().unwrap_or_else(|| self.settings.clone())
    }

    /// Apply `settings` once the selection has rested.
    fn apply_later(&mut self, cx: &mut Cx, settings: Settings) {
        self.pending_apply = Some(settings);
        cx.stop_timer(self.apply_timer);
        self.apply_timer = cx.start_timeout(0.18);
        self.refresh(cx);
    }

    /// Save `settings`: the file, and every tab through the generation.
    fn apply(&mut self, cx: &mut Cx, settings: Settings) {
        self.pending_apply = None;
        if let Err(err) = term_settings::update(settings) {
            error!("terminal: could not save settings to {}: {err}", term_settings::path().display());
        }
        self.sync_settings(cx);
    }

    fn draw_settings_panel(&mut self, cx: &mut Cx2d, body: Rect) {
        const ROW_H: f64 = 24.0;
        let (bg, fg) = self.chrome();
        let dark = luminance(bg) < 0.5;
        let white = vec4(1.0, 1.0, 1.0, 1.0);
        let black = vec4(0.0, 0.0, 0.0, 1.0);
        let colors = PanelColors {
            fg,
            white,
            card: mix(bg, if dark { white } else { black }, if dark { 0.06 } else { 0.04 }),
            line: mix(mix(bg, if dark { white } else { black }, if dark { 0.06 } else { 0.04 }), fg, 0.12),
            dim: mix(mix(bg, if dark { white } else { black }, if dark { 0.06 } else { 0.04 }), fg, 0.6),
            accent: vec4(0.478, 0.635, 0.969, 1.0),
        };

        let width = (body.size.x - 24.0).clamp(200.0, 400.0);
        let rect = Rect {
            pos: dvec2(body.pos.x + body.size.x - width - 12.0, body.pos.y + 10.0),
            size: dvec2(width, (body.size.y - 20.0).max(120.0)),
        };
        let mut hits = Vec::new();

        self.draw_panel.new_draw_call(cx);
        self.draw_panel.color = colors.card;
        self.draw_panel.draw_abs(cx, rect);
        self.draw_tab.new_draw_call(cx);
        self.draw_heading.new_draw_call(cx);
        self.draw_label.new_draw_call(cx);
        self.draw_icon.new_draw_call(cx);

        let left = rect.pos.x + 16.0;
        let right = rect.pos.x + rect.size.x - 16.0;
        let footer_y = rect.pos.y + rect.size.y - 46.0;
        let top = rect.pos.y + 40.0;
        let close = Rect { pos: dvec2(right - 18.0, rect.pos.y + 10.0), size: dvec2(20.0, 20.0) };
        self.draw_icon.color = colors.dim;
        self.draw_icon.draw_abs(cx, dvec2(close.pos.x + 5.0, close.pos.y + 4.0), ICON_CLOSE);
        hits.push((close, PanelHit::Close));

        let mode = self.panel.as_ref().map(|p| match &p.mode {
            PanelMode::Rows => 0,
            PanelMode::Choose(_) => 1,
            PanelMode::Name(_) => 2,
        });
        let (heading, keys, visible) = match mode {
            Some(1) => {
                let visible = self.draw_chooser(cx, &colors, left, right, top, footer_y, ROW_H, &mut hits);
                ("Settings", "\u{2191}\u{2193} choose   type to filter   Enter pick   Esc back", visible)
            }
            Some(2) => {
                self.draw_name_input(cx, &colors, left, right, top);
                ("Save profile", "Enter save   Esc cancel", 0)
            }
            _ => {
                let visible = self.draw_rows(cx, &colors, rect, left, right, top, footer_y, ROW_H, &mut hits);
                ("Settings", "\u{2191}\u{2193} choose   \u{2190}\u{2192} change   Enter open   Esc close", visible)
            }
        };
        self.draw_heading.color = colors.fg;
        self.draw_heading.draw_abs(cx, dvec2(left, rect.pos.y + 12.0), heading);

        self.draw_tab.tab = 0.0;
        self.draw_tab.color = colors.line;
        self.draw_tab.draw_abs(cx, Rect { pos: dvec2(left, footer_y), size: dvec2(right - left, 1.0) });
        let message = self.panel.as_ref().and_then(|p| p.message.clone());
        self.draw_label.color = if message.is_some() { colors.fg } else { colors.dim };
        let first_line = message.unwrap_or_else(|| keys.to_owned());
        let first_line = self.fit(cx, &first_line, right - left);
        self.draw_label.draw_abs(cx, dvec2(left, footer_y + 8.0), &first_line);
        self.draw_label.color = colors.dim;
        let file = tilde(&term_settings::path());
        let file = self.fit(cx, &file, right - left - 60.0);
        self.draw_label.draw_abs(cx, dvec2(left, footer_y + 26.0), &file);
        if mode == Some(0) {
            let reset_w = self.text_width(cx, "Reset") + 16.0;
            let reset = Rect { pos: dvec2(right - reset_w, footer_y + 20.0), size: dvec2(reset_w, 20.0) };
            self.draw_tab.color = colors.line;
            self.draw_tab.draw_abs(cx, reset);
            self.draw_label.color = colors.fg;
            self.draw_label.draw_abs(cx, dvec2(reset.pos.x + 8.0, reset.pos.y + 4.0), "Reset");
            hits.push((reset, PanelHit::Reset));
        }

        if let Some(panel) = self.panel.as_mut() {
            panel.rect = rect;
            panel.hits = hits;
            panel.visible = visible;
        }
    }

    /// The rows, grouped under their sections; returns how many fit.
    #[allow(clippy::too_many_arguments)]
    fn draw_rows(
        &mut self,
        cx: &mut Cx2d,
        c: &PanelColors,
        rect: Rect,
        left: f64,
        right: f64,
        top: f64,
        bottom: f64,
        row_h: f64,
        hits: &mut Vec<(Rect, PanelHit)>,
    ) -> usize {
        let shown_settings = self.current_settings();
        let (selected, mut scroll, confirm_delete, shown) =
            self.panel.as_ref().map_or((0, 0, false, None), |p| (p.selected, p.scroll, p.confirm_delete, p.shown_selected));
        // Keep a newly selected row in view, counting section headers; a
        // mouse-wheel scroll (same selection) stays where it was put.
        if shown != Some(selected) {
            scroll = scroll.min(selected);
            while selected >= scroll + rows_fitting(scroll, bottom - top, row_h) {
                scroll += 1;
            }
        }
        if let Some(panel) = self.panel.as_mut() {
            panel.scroll = scroll;
            panel.shown_selected = Some(selected);
        }
        let mut y = top;
        let mut index = 0;
        let mut visible = 0;
        'sections: for (title, rows) in settings_panel::SECTIONS {
            let mut header_drawn = false;
            for row in rows.iter().copied() {
                let i = index;
                index += 1;
                if i < scroll {
                    continue;
                }
                if !header_drawn {
                    if y + row_h * 2.0 > bottom {
                        break 'sections;
                    }
                    self.draw_label.color = c.dim;
                    self.draw_label.draw_abs(cx, dvec2(left, y + 8.0), &title.to_uppercase());
                    y += row_h;
                    header_drawn = true;
                }
                if y + row_h > bottom {
                    break 'sections;
                }
                visible += 1;
                let row_rect = Rect { pos: dvec2(rect.pos.x + 8.0, y), size: dvec2(rect.size.x - 16.0, row_h) };
                if i == selected {
                    self.draw_tab.tab = 0.0;
                    self.draw_tab.color = c.line;
                    self.draw_tab.draw_abs(cx, row_rect);
                }
                let text_y = y + (row_h - 12.0) * 0.5;
                let label = if row == Row::DeleteProfile && confirm_delete && i == selected {
                    format!("Press Enter again to delete \u{201c}{}\u{201d}", self.settings.profile)
                } else {
                    row.label().to_owned()
                };
                self.draw_label.color = if row == Row::DeleteProfile && confirm_delete { c.accent } else { c.fg };
                self.draw_label.draw_abs(cx, dvec2(left, text_y), &label);
                let mut label_w = self.text_width(cx, &label);
                if row.new_shells_only() {
                    self.draw_label.color = c.dim;
                    self.draw_label.draw_abs(cx, dvec2(left + label_w + 6.0, text_y), "new tabs");
                    label_w += 6.0 + self.text_width(cx, "new tabs");
                }
                match row.kind() {
                    RowKind::Toggle => {
                        hits.push((row_rect, PanelHit::Select(i)));
                        let on = row.value(&shown_settings) == "On";
                        let pill = Rect { pos: dvec2(right - 30.0, y + 5.0), size: dvec2(30.0, 14.0) };
                        self.draw_tab.tab = 0.0;
                        self.draw_tab.radius = 7.0;
                        self.draw_tab.color = if on { c.accent } else { c.line };
                        self.draw_tab.draw_abs(cx, pill);
                        let knob_x = if on { pill.pos.x + 17.0 } else { pill.pos.x + 1.0 };
                        self.draw_tab.radius = 6.0;
                        self.draw_tab.color = if on { c.white } else { c.dim };
                        self.draw_tab.draw_abs(cx, Rect { pos: dvec2(knob_x, pill.pos.y + 1.0), size: dvec2(12.0, 12.0) });
                        let toggle = Rect { pos: dvec2(pill.pos.x - 6.0, y), size: dvec2(42.0, row_h) };
                        hits.push((toggle, PanelHit::Step(i, 1)));
                    }
                    RowKind::Action => {
                        hits.push((row_rect, PanelHit::Activate(i)));
                        // A summary beside an action that opens a list.
                        let value = row.value(&shown_settings);
                        if !value.is_empty() {
                            let room = (right - (left + label_w + 16.0)).max(40.0);
                            let value = self.fit(cx, &value, room);
                            let vw = self.text_width(cx, &value);
                            self.draw_label.color = c.dim;
                            self.draw_label.draw_abs(cx, dvec2(right - vw, text_y), &value);
                        }
                    }
                    RowKind::Value => {
                        let has_list = matches!(row, Row::Theme | Row::Font | Row::CjkFont | Row::Shell | Row::Profile);
                        hits.push((row_rect, if has_list { PanelHit::Activate(i) } else { PanelHit::Select(i) }));
                        let room = (right - 40.0 - (left + label_w + 16.0)).max(40.0);
                        let value = self.fit(cx, &row.value(&shown_settings), room);
                        let vw = self.text_width(cx, &value);
                        let next = Rect { pos: dvec2(right - 16.0, y), size: dvec2(20.0, row_h) };
                        let prev = Rect { pos: dvec2(right - 16.0 - vw - 24.0, y), size: dvec2(20.0, row_h) };
                        self.draw_label.color = if i == selected { c.fg } else { c.dim };
                        self.draw_label.draw_abs(cx, dvec2(right - 20.0 - vw, text_y), &value);
                        self.draw_icon.color = c.dim;
                        self.draw_icon.draw_abs(cx, dvec2(prev.pos.x + 6.0, text_y + 1.0), ICON_PREV);
                        self.draw_icon.draw_abs(cx, dvec2(next.pos.x + 6.0, text_y + 1.0), ICON_NEXT);
                        hits.push((prev, PanelHit::Step(i, -1)));
                        hits.push((next, PanelHit::Step(i, 1)));
                    }
                }
                y += row_h;
            }
        }
        visible
    }

    /// A row's filterable list; returns how many entries fit.
    #[allow(clippy::too_many_arguments)]
    fn draw_chooser(
        &mut self,
        cx: &mut Cx2d,
        c: &PanelColors,
        left: f64,
        right: f64,
        top: f64,
        bottom: f64,
        row_h: f64,
        hits: &mut Vec<(Rect, PanelHit)>,
    ) -> usize {
        let settings = self.current_settings();
        let shown_settings = settings.clone();
        let Some(PanelMode::Choose(chooser)) = self.panel.as_mut().map(|p| &mut p.mode) else {
            return 0;
        };
        if chooser.awaiting_fonts && crate::fonts::ready() {
            chooser.awaiting_fonts = false;
            chooser.choices = chooser.row.choices(&settings).unwrap_or_default();
        }
        let shown = settings_panel::filter_choices(&chooser.choices, &chooser.filter);
        let visible = (((bottom - top - row_h * 2.0) / row_h).floor().max(1.0)) as usize;
        chooser.selected = chooser.selected.min(shown.len().saturating_sub(1));
        // Keep a newly selected entry in view; a mouse-wheel scroll (same
        // selection) stays where it was put.
        if chooser.shown_selected != Some(chooser.selected) {
            if chooser.selected < chooser.scroll {
                chooser.scroll = chooser.selected;
            } else if chooser.selected >= chooser.scroll + visible {
                chooser.scroll = chooser.selected + 1 - visible;
            }
            chooser.shown_selected = Some(chooser.selected);
        }
        chooser.scroll = chooser.scroll.min(shown.len().saturating_sub(visible));
        let (row, filter, selected, scroll) = (chooser.row, chooser.filter.clone(), chooser.selected, chooser.scroll);
        let current = row.current(&shown_settings);

        // Title with a back arrow, then the filter.
        let back = Rect { pos: dvec2(left - 6.0, top), size: dvec2(right - left + 12.0, row_h) };
        self.draw_icon.color = c.dim;
        self.draw_icon.draw_abs(cx, dvec2(left, top + 7.0), ICON_PREV);
        self.draw_label.color = c.fg;
        self.draw_label.draw_abs(cx, dvec2(left + 16.0, top + 6.0), row.label());
        let count = format!("{} of {}", shown.len(), chooser_len(&self.panel));
        let cw = self.text_width(cx, &count);
        self.draw_label.color = c.dim;
        self.draw_label.draw_abs(cx, dvec2(right - cw, top + 6.0), &count);
        hits.push((back, PanelHit::Back));
        let field = Rect { pos: dvec2(left - 6.0, top + row_h + 2.0), size: dvec2(right - left + 12.0, row_h - 4.0) };
        self.draw_tab.tab = 0.0;
        self.draw_tab.color = c.line;
        self.draw_tab.draw_abs(cx, field);
        let (text, color) = if filter.is_empty() {
            ("Type to filter".to_owned(), c.dim)
        } else {
            // The end of what was typed, where the caret is.
            let mut shown: String = filter.clone();
            while shown.chars().count() > 1 && self.text_width(cx, &shown) > right - left - 12.0 {
                shown.remove(0);
            }
            (format!("{shown}\u{258f}"), c.fg)
        };
        self.draw_label.color = color;
        self.draw_label.draw_abs(cx, dvec2(left, field.pos.y + 4.0), &text);

        let mut y = top + row_h * 2.0 + 4.0;
        for (i, choice) in shown.iter().enumerate().skip(scroll).take(visible) {
            let item = Rect { pos: dvec2(left - 8.0, y), size: dvec2(right - left + 16.0, row_h) };
            if i == selected {
                self.draw_tab.color = c.line;
                self.draw_tab.draw_abs(cx, item);
            }
            let text_y = y + (row_h - 12.0) * 0.5;
            if choice.value == current {
                self.draw_label.color = c.accent;
                self.draw_label.draw_abs(cx, dvec2(left, text_y), "\u{2713}");
            }
            let note_w = if choice.note.is_empty() { 0.0 } else { self.text_width(cx, &choice.note) + 8.0 };
            let label = self.fit(cx, &choice.label, right - left - 16.0 - note_w);
            self.draw_label.color = if i == selected { c.fg } else { mix(c.dim, c.fg, 0.5) };
            self.draw_label.draw_abs(cx, dvec2(left + 16.0, text_y), &label);
            if !choice.note.is_empty() {
                self.draw_label.color = c.dim;
                self.draw_label.draw_abs(cx, dvec2(right - note_w + 8.0, text_y), &choice.note);
            }
            hits.push((item, PanelHit::Choice(i)));
            y += row_h;
        }
        if shown.is_empty() {
            self.draw_label.color = c.dim;
            let empty = if row == Row::Font && !crate::fonts::ready() { "Looking for installed fonts\u{2026}" } else { "Nothing matches" };
            self.draw_label.draw_abs(cx, dvec2(left + 16.0, y + 6.0), empty);
        }
        visible
    }

    fn draw_name_input(&mut self, cx: &mut Cx2d, c: &PanelColors, left: f64, right: f64, top: f64) {
        let Some(PanelMode::Name(name)) = self.panel.as_ref().map(|p| &p.mode) else {
            return;
        };
        let name = name.clone();
        self.draw_label.color = c.dim;
        self.draw_label.draw_abs(cx, dvec2(left, top + 6.0), "Save the current settings as a profile named:");
        let field = Rect { pos: dvec2(left - 6.0, top + 28.0), size: dvec2(right - left + 12.0, 24.0) };
        self.draw_tab.tab = 0.0;
        self.draw_tab.color = c.line;
        self.draw_tab.draw_abs(cx, field);
        self.draw_label.color = c.fg;
        self.draw_label.draw_abs(cx, dvec2(left, field.pos.y + 6.0), &format!("{name}\u{258f}"));
        let exists = term_settings::list_profiles().iter().any(|p| p.eq_ignore_ascii_case(name.trim()));
        if exists {
            self.draw_label.color = c.accent;
            self.draw_label.draw_abs(cx, dvec2(left, field.pos.y + 34.0), "A profile with this name is replaced");
        }
        let saved = term_settings::list_profiles();
        if !saved.is_empty() {
            self.draw_label.color = c.dim;
            let list = self.fit(cx, &format!("Saved: {}", saved.join(", ")), right - left);
            self.draw_label.draw_abs(cx, dvec2(left, field.pos.y + 56.0), &list);
        }
    }

    fn text_width(&self, cx: &mut Cx2d, text: &str) -> f64 {
        self.draw_label
            .prepare_single_line_run(cx, text)
            .map_or(0.0, |run| run.width_in_lpxs as f64)
    }

    /// `text` cut with an ellipsis to fit `width`.
    fn fit(&self, cx: &mut Cx2d, text: &str, width: f64) -> String {
        if self.text_width(cx, text) <= width {
            return text.to_owned();
        }
        let chars: Vec<char> = text.chars().collect();
        let mut keep = chars.len();
        while keep > 0 {
            keep -= 1;
            let candidate: String = chars[..keep].iter().collect::<String>() + "\u{2026}";
            if self.text_width(cx, &candidate) <= width {
                return candidate;
            }
        }
        String::new()
    }
}

impl TermTabsRef {
    pub fn active_term(&self, cx: &mut Cx) -> WidgetRef {
        self.borrow_mut().map(|mut tabs| tabs.active_term(cx)).unwrap_or_default()
    }

    /// See [`TermTabs::request_close_all`]; no widget: nothing to lose.
    pub fn request_close_all(&self, cx: &mut Cx) -> CloseDecision {
        self.borrow_mut().map(|mut tabs| tabs.request_close_all(cx)).unwrap_or_default()
    }

    pub fn set_tabs_enabled(&self, cx: &mut Cx, enabled: bool) {
        if let Some(mut tabs) = self.borrow_mut() {
            tabs.set_tabs_enabled(cx, enabled);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode, control: bool, shift: bool, alt: bool) -> KeyEvent {
        KeyEvent {
            key_code: code,
            modifiers: KeyModifiers { control, shift, alt, logo: false },
            ..Default::default()
        }
    }

    #[test]
    fn running_jobs_are_gathered_from_every_tab_and_pane() {
        let none: Vec<(String, Vec<Option<String>>)> = vec![("~".into(), vec![None]), ("src".into(), vec![None, None])];
        assert!(collect_running_jobs(none).is_empty(), "every shell at its prompt");
        let tabs = vec![
            ("~".to_string(), vec![None]),
            ("build".to_string(), vec![Some("cargo".to_string()), None, Some("sleep".to_string())]),
            ("vim".to_string(), vec![Some("vim".to_string())]),
        ];
        assert_eq!(
            collect_running_jobs(tabs),
            vec![
                ("build".to_string(), "cargo".to_string()),
                ("build".to_string(), "sleep".to_string()),
                ("vim".to_string(), "vim".to_string()),
            ],
            "tab order, then pane order; both jobs of a split tab"
        );
    }

    #[test]
    fn closing_everything_asks_only_while_a_job_runs_and_confirmation_is_on() {
        let jobs = vec![("build".to_string(), "cargo".to_string())];
        assert_eq!(close_all_decision(true, false, &jobs), CloseDecision::Veto, "running + confirm on");
        assert_eq!(close_all_decision(true, false, &[]), CloseDecision::Allow, "nothing running");
        assert_eq!(close_all_decision(false, false, &jobs), CloseDecision::Allow, "confirmation off");
        assert_eq!(close_all_decision(true, true, &jobs), CloseDecision::Allow, "already confirmed");
    }

    #[test]
    fn the_close_question_names_every_job() {
        let one = vec![("sleep".to_string(), "sleep".to_string())];
        assert_eq!(
            close_all_message(&one),
            "Closing the terminal ends \u{201c}sleep\u{201d}. Close anyway?"
        );
        let two = vec![
            ("build".to_string(), "cargo".to_string()),
            ("notes".to_string(), "vim".to_string()),
        ];
        assert_eq!(
            close_all_message(&two),
            "Closing the terminal ends 2 running jobs: \u{201c}cargo\u{201d} in build, \u{201c}vim\u{201d} in notes. Close anyway?"
        );
    }

    /// What the default table has the tab widget do with `key`.
    fn tab_command(key: &KeyEvent, tabs: usize) -> Option<TabCommand> {
        table_tab_command(
            &Keybinds::defaults(cfg!(target_os = "macos")),
            key,
            tabs,
            false,
        )
    }

    fn table_tab_command(
        keys: &Keybinds,
        key: &KeyEvent,
        tabs: usize,
        search_open: bool,
    ) -> Option<TabCommand> {
        let ctx = keybinds::Context { tabs, search_open };
        match keys.decide(key, keybinds::Scope::Tabs, &ctx) {
            keybinds::Decision::Run { action, .. } => tab_command_of(&action),
            keybinds::Decision::Pass => None,
        }
    }

    #[test]
    fn the_default_table_takes_exactly_the_old_tab_keys() {
        // Every key, every modifier combination, one to four tabs, both
        // platforms: the table answers what the hard-coded match did.
        for mac in [false, true] {
            let keys = Keybinds::defaults(mac);
            for &code in keybinds::ALL_KEY_CODES {
                for bits in 0..16u8 {
                    let mut k = key(code, bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
                    k.modifiers.logo = bits & 8 != 0;
                    for tabs in 1..=4 {
                        assert_eq!(
                            table_tab_command(&keys, &k, tabs, false),
                            legacy_tab_command(&k, tabs),
                            "mac {mac}, {tabs} tabs: {:?} {:?}",
                            k.key_code,
                            k.modifiers
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_remap_moves_a_tab_command_and_an_open_search_bar_keeps_its_keys() {
        let keys = Keybinds::for_lines(&[
            "cmd+t=new_tab".into(),
            "ctrl+shift+t=unbind".into(),
            "ctrl+r=next_tab".into(),
        ]);
        let mut cmd_t = key(KeyCode::KeyT, false, false, false);
        cmd_t.modifiers.logo = true;
        assert_eq!(
            table_tab_command(&keys, &cmd_t, 1, false),
            Some(TabCommand::New)
        );
        assert_eq!(
            table_tab_command(&keys, &key(KeyCode::KeyT, true, true, false), 1, false),
            None
        );
        // Ctrl+R toggles regex in an open search bar, whatever is bound.
        let ctrl_r = key(KeyCode::KeyR, true, false, false);
        assert_eq!(
            table_tab_command(&keys, &ctrl_r, 2, false),
            Some(TabCommand::Next)
        );
        assert!(
            search::bar_key(&ctrl_r).is_some(),
            "key_shortcut passes it to the bar"
        );
    }

    #[test]
    fn the_tab_keys_map_to_commands() {
        let t = |code, c, s, a, n| tab_command(&key(code, c, s, a), n);
        assert_eq!(t(KeyCode::KeyT, true, true, false, 1), Some(TabCommand::New));
        assert_eq!(t(KeyCode::KeyW, true, true, false, 1), Some(TabCommand::Close));
        assert_eq!(t(KeyCode::Tab, true, false, false, 2), Some(TabCommand::Next));
        assert_eq!(t(KeyCode::Tab, true, true, false, 2), Some(TabCommand::Previous));
        assert_eq!(t(KeyCode::PageDown, true, false, false, 2), Some(TabCommand::Next));
        assert_eq!(t(KeyCode::PageUp, true, false, false, 2), Some(TabCommand::Previous));
        assert_eq!(t(KeyCode::Key3, false, false, true, 4), Some(TabCommand::Select(2)));
        assert_eq!(t(KeyCode::Key9, false, false, true, 4), Some(TabCommand::Select(usize::MAX)));
        assert_eq!(t(KeyCode::Comma, true, false, false, 1), Some(TabCommand::Settings));
        assert_eq!(t(KeyCode::KeyD, true, true, false, 1), Some(TabCommand::SplitRight));
        assert_eq!(t(KeyCode::KeyE, true, true, false, 1), Some(TabCommand::SplitDown));
        assert_eq!(t(KeyCode::ArrowLeft, true, true, false, 1), Some(TabCommand::Focus(Dir::Left)));
        assert_eq!(t(KeyCode::KeyZ, true, true, false, 1), Some(TabCommand::Zoom));
        assert_eq!(t(KeyCode::ArrowLeft, true, false, false, 1), None, "Ctrl+Left is the shell's (word left)");
    }

    #[test]
    fn the_shell_keeps_its_own_keys() {
        let t = |code, c, s, a, n| tab_command(&key(code, c, s, a), n);
        assert_eq!(t(KeyCode::KeyT, true, false, false, 1), None, "Ctrl+T is the shell's (transpose)");
        assert_eq!(t(KeyCode::KeyW, true, false, false, 1), None, "Ctrl+W is the shell's (kill word)");
        assert_eq!(t(KeyCode::Key3, false, false, true, 1), None, "Meta+digit with a single tab");
        assert_eq!(t(KeyCode::Tab, false, false, false, 2), None);
        let mut cmd = key(KeyCode::KeyT, true, true, false);
        cmd.modifiers.logo = true;
        assert_eq!(tab_command(&cmd, 1), None);
    }

    #[test]
    fn a_tab_is_named_by_its_program_or_its_directory() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/u".into());
        let src = PathBuf::from(&home).join("src");
        let p = TabTitle::Program;
        let d = TabTitle::Directory;
        assert_eq!(tab_label(p, "vim notes.md", Some("vim"), Some(&src), Some("zsh")), "vim notes.md");
        assert_eq!(tab_label(p, "", Some("htop"), Some(&src), Some("zsh")), "htop");
        assert_eq!(tab_label(p, "", None, Some(&src), Some("zsh")), "src");
        assert_eq!(tab_label(p, "", None, Some(Path::new(&home)), None), "~");
        assert_eq!(tab_label(d, "vim notes.md", Some("vim"), Some(&src), None), "src");
        assert_eq!(tab_label(d, "", None, None, Some("zsh")), "zsh");
        assert_eq!(tab_label(p, "  ", None, None, None), "shell");
    }

    #[test]
    fn a_given_name_wins_over_the_program_the_job_and_the_directory() {
        let src = PathBuf::from("/tmp/src");
        let p = TabTitle::Program;
        let t = |custom| tab_title(custom, p, "vim notes.md", Some("vim"), Some(&src), Some("zsh"));
        assert_eq!(t(Some("build box")), "build box", "a name beats the program's own title");
        assert_eq!(t(Some("  build box ")), "build box");
        assert_eq!(t(None), "vim notes.md", "no name: the program's title");
        assert_eq!(t(Some("  ")), "vim notes.md", "a blank name is no name");
        assert_eq!(tab_title(Some("ops"), TabTitle::Directory, "", None, Some(&src), None), "ops");
        assert_eq!(tab_title(None, p, "", Some("htop"), Some(&src), None), "htop");
        assert_eq!(tab_title(None, p, "", None, Some(&src), None), "src");
    }

    #[test]
    fn a_name_is_typed_edited_taken_or_dropped() {
        let k = |code| key(code, false, false, false);
        let mut edit = NameEdit::new("");
        edit.insert("build box");
        assert_eq!(edit.text, "build box");
        for _ in 0..3 {
            assert_eq!(edit.key(&k(KeyCode::ArrowLeft)), NameKey::Editing);
        }
        edit.key(&k(KeyCode::Backspace));
        edit.insert("-");
        assert_eq!(edit.text, "build-box");
        edit.key(&k(KeyCode::Home));
        edit.key(&k(KeyCode::Delete));
        edit.insert("\u{00e9}\n\t");
        assert_eq!(edit.text, "\u{00e9}uild-box", "line breaks and tabs are dropped");
        edit.key(&k(KeyCode::End));
        edit.key(&k(KeyCode::ArrowRight));
        edit.insert("!");
        assert_eq!(edit.text, "\u{00e9}uild-box!");
        assert_eq!(edit.key(&k(KeyCode::ReturnKey)), NameKey::Commit(Some("\u{00e9}uild-box!".into())));
        assert_eq!(edit.key(&k(KeyCode::Escape)), NameKey::Cancel);

        let mut cleared = NameEdit::new("ops");
        for _ in 0..5 {
            cleared.key(&k(KeyCode::Backspace));
        }
        assert_eq!(cleared.key(&k(KeyCode::ReturnKey)), NameKey::Commit(None), "empty: the automatic label again");
        assert_eq!(NameEdit::new("  ").committed(), None);

        let mut long = NameEdit::new("");
        long.insert(&"x".repeat(100));
        assert_eq!(long.text.chars().count(), TAB_NAME_MAX);
    }
}
