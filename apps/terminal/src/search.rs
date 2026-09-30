//! Search in scrollback: the matcher, the incremental scan over history
//! and the screen, and the search bar's key bindings.
//!
//! Text. Each logical line (rows joined by soft wraps) is read the way copy
//! reads a selection (`Screen::selection_text`): every cell's text once
//! (`CellContent::push_text`), nothing for a wide char's tails or a spacer
//! head, a space for a blank cell inside a wrapped row, trailing blanks
//! trimmed. Every piece of text remembers its cell, so a match maps back to
//! whole cells: a match that starts or ends inside a wide char, a grapheme
//! cluster or an Indic letter's cell covers that whole cell, as a selection
//! does. A match is a `(row, column)` range in absolute (eviction-stable)
//! rows, the form a selection takes.
//!
//! Case. Smart case: a query with no uppercase letter matches either case.
//! With regex on, an escape (`\W`, `\S`, `\p{Lu}`) does not count as an
//! uppercase letter.
//!
//! Bounded work. The scan reads history newest first, about
//! [`CELLS_PER_STEP`] cells per call to [`Search::sync`] (the widget calls
//! it once a frame; a few milliseconds in a release build), so the first
//! matches appear at once and a long history never holds up a frame: 100k
//! rows of 120 columns take about 75 frames. It keeps the newest [`MAX_MATCHES`] matches and stops there,
//! which the status reports ("10000+").
//!
//! New output. History rows never change once written, so their matches
//! stay put; rows that leave the front of history take their matches with
//! them. Rows of the live screen (and the logical line reaching into it)
//! can change at any time: after output they are read again. A resize, a
//! switch to or from the alternate screen, or a cleared history reads
//! everything again, as does output so fast that more than
//! [`LIVE_ROWS_MAX`] rows arrived since the last frame. The current match
//! is kept where it was, or moves to the nearest one if its text changed.

use std::ops::Range;
use std::panic::{catch_unwind, AssertUnwindSafe};

use makepad_regex::{ParseOptions, Regex};
use makepad_widgets::{KeyCode, KeyEvent};

use crate::keybinds::Action;
use crate::term::page::{CellContent, Row};
use crate::term::screen::Screen;

/// (absolute row, column).
pub type Pos = (u64, usize);

/// Cells `start..end` in row-major order, `end` exclusive: the form of a
/// selection (`Screen::snap_selection`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Match {
    pub start: Pos,
    pub end: Pos,
}

/// The matches a search keeps; older ones are not looked for.
pub const MAX_MATCHES: usize = 10_000;
/// History cells read per [`Search::sync`] (whole rows, at least 64).
pub const CELLS_PER_STEP: usize = 160_000;

/// History rows read per [`Search::sync`] at `cols` columns.
pub fn rows_per_step(cols: usize) -> usize {
    (CELLS_PER_STEP / cols.max(1)).max(64)
}
/// Changed live rows read at once; more starts the scan over.
pub const LIVE_ROWS_MAX: usize = 8_192;

/// Whether the query asks for case to matter (smart case).
pub fn case_sensitive(query: &str, regex: bool) -> bool {
    let mut chars = query.chars();
    while let Some(c) = chars.next() {
        if regex && c == '\\' {
            match chars.next() {
                // A Unicode class name: `\p{Lu}`, `\PL`.
                Some('p') | Some('P') => {
                    if chars.clone().next() == Some('{') {
                        for c in chars.by_ref() {
                            if c == '}' {
                                break;
                            }
                        }
                    } else {
                        chars.next();
                    }
                }
                _ => {}
            }
            continue;
        }
        if c.is_uppercase() {
            return true;
        }
    }
    false
}

enum Kind {
    /// Plain text; `fold`: compared in lowercase.
    Plain { needle: String, fold: bool },
    /// `anchored`: the pattern starts with `^`, so only a line's first
    /// match counts (each later match is looked for in the rest of the
    /// line, where `^` would match again).
    Regex { re: Box<Regex>, anchored: bool },
}

/// A compiled query.
pub struct Matcher {
    kind: Kind,
    /// Scratch: the lowercased haystack and each of its bytes' offset in
    /// the original.
    folded: String,
    map: Vec<usize>,
}

impl Matcher {
    /// `Ok(None)` for an empty query; `Err` for a regex that does not parse.
    pub fn new(query: &str, regex: bool) -> Result<Option<Matcher>, String> {
        if query.is_empty() {
            return Ok(None);
        }
        let fold = !case_sensitive(query, regex);
        let kind = if regex {
            let options = ParseOptions {
                ignore_case: fold,
                ..ParseOptions::default()
            };
            let re = catch_unwind(|| Regex::new_with_options(query, options))
                .map_err(|_| "invalid regex".to_string())?
                .map_err(|e| e.message)?;
            Kind::Regex {
                re: Box::new(re),
                anchored: query.starts_with('^'),
            }
        } else {
            let needle = if fold {
                fold_str(query)
            } else {
                query.to_string()
            };
            Kind::Plain { needle, fold }
        };
        Ok(Some(Matcher {
            kind,
            folded: String::new(),
            map: Vec::new(),
        }))
    }

    /// Every non-empty, non-overlapping match in `hay`, as byte ranges in
    /// order, appended to `out`.
    pub fn find(&mut self, hay: &str, out: &mut Vec<Range<usize>>) {
        match &self.kind {
            Kind::Plain {
                needle,
                fold: false,
            } => {
                for (at, _) in hay.match_indices(needle.as_str()) {
                    out.push(at..at + needle.len());
                }
            }
            Kind::Plain { needle, fold: true } => {
                if hay.is_ascii() {
                    self.folded.clear();
                    self.folded.push_str(hay);
                    self.folded.make_ascii_lowercase();
                    for (at, _) in self.folded.match_indices(needle.as_str()) {
                        out.push(at..at + needle.len());
                    }
                    return;
                }
                // Lowercasing can change a character's length: keep where
                // each folded byte came from.
                self.folded.clear();
                self.map.clear();
                for (at, c) in hay.char_indices() {
                    for lower in c.to_lowercase() {
                        self.folded.push(lower);
                    }
                    self.map.resize(self.folded.len(), at);
                }
                self.map.push(hay.len());
                for (at, _) in self.folded.match_indices(needle.as_str()) {
                    let (start, end) = (self.map[at], self.map[at + needle.len()]);
                    if end > start {
                        out.push(start..end);
                    }
                }
            }
            Kind::Regex { re, anchored } => {
                // A regex engine failure on odd input finds nothing on this
                // line rather than taking the terminal down.
                let _ = catch_unwind(AssertUnwindSafe(|| {
                    let mut slots = [None; 2];
                    let mut from = 0;
                    while from <= hay.len() {
                        slots[0] = None;
                        slots[1] = None;
                        if !re.run(&hay[from..], &mut slots) {
                            break;
                        }
                        let (Some(s), Some(e)) = (slots[0], slots[1]) else {
                            break;
                        };
                        let (s, e) = (from + s, from + e);
                        if e > s {
                            out.push(s..e);
                            if *anchored {
                                break;
                            }
                            from = e;
                        } else {
                            // An empty match: skip a character.
                            match hay[s..].chars().next() {
                                Some(c) => from = s + c.len_utf8(),
                                None => break,
                            }
                        }
                    }
                }));
            }
        }
    }
}

fn fold_str(s: &str) -> String {
    s.chars().flat_map(char::to_lowercase).collect()
}

/// One cell's text in a line being searched.
#[derive(Clone, Copy, Debug)]
struct CellText {
    byte: usize,
    row: u64,
    col: usize,
    width: usize,
}

/// A logical line's text and where each piece of it is on screen.
#[derive(Default)]
struct LineText {
    text: String,
    cells: Vec<CellText>,
}

fn row_at(screen: &Screen, abs: u64) -> Option<&Row> {
    screen.row_virtual(screen.virtual_of_absolute(abs)?)
}

/// The first row of the logical line holding `abs`, not before `lo`.
fn line_start(screen: &Screen, abs: u64, lo: u64) -> u64 {
    let mut r = abs;
    while r > lo && row_at(screen, r - 1).is_some_and(|row| row.wrapped) {
        r -= 1;
    }
    r
}

impl LineText {
    /// Rows `first..=last`, read as copy reads them.
    fn read(&mut self, screen: &Screen, first: u64, last: u64) {
        self.text.clear();
        self.cells.clear();
        let cols = screen.cols;
        for abs in first..=last {
            let Some(row) = row_at(screen, abs) else {
                continue;
            };
            let wrapped = abs < last && row.wrapped;
            let n = if wrapped {
                cols
            } else {
                row.cells.len().min(cols)
            };
            for col in 0..n {
                let content = row.cell(col).map(|c| &c.content);
                if matches!(
                    content,
                    Some(CellContent::WideTail) | Some(CellContent::WideSpacerHead)
                ) {
                    continue;
                }
                let width = content.map_or(1, |c| c.width().max(1) as usize);
                self.cells.push(CellText {
                    byte: self.text.len(),
                    row: abs,
                    col,
                    width: width.min(cols - col),
                });
                match content {
                    Some(content) => content.push_text(&mut self.text),
                    None => self.text.push(' '),
                }
            }
        }
        let trimmed = self.text.trim_end_matches(' ').len();
        self.text.truncate(trimmed);
        while self.cells.last().is_some_and(|c| c.byte >= trimmed) {
            self.cells.pop();
        }
    }

    /// The cells a byte range of the text covers.
    fn cells_of(&self, range: &Range<usize>) -> Option<Match> {
        if range.is_empty() || self.cells.is_empty() {
            return None;
        }
        let first = self
            .cells
            .partition_point(|c| c.byte <= range.start)
            .checked_sub(1)?;
        let last = self
            .cells
            .partition_point(|c| c.byte < range.end)
            .checked_sub(1)?;
        let (a, b) = (self.cells[first], self.cells[last.max(first)]);
        Some(Match {
            start: (a.row, a.col),
            end: (b.row, b.col + b.width),
        })
    }
}

/// What forces a search to start over when it changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    alternate: bool,
    cols: usize,
    rows: usize,
}

/// The status line of the search bar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Status {
    /// The current match counted from the newest (1 = the last one
    /// printed).
    pub current: Option<usize>,
    pub total: usize,
    /// More matches than [`MAX_MATCHES`]: older ones were not looked for.
    pub capped: bool,
    /// History is still being read.
    pub scanning: bool,
    pub error: Option<String>,
    pub empty_query: bool,
}

impl Status {
    pub fn text(&self) -> String {
        if self.empty_query {
            return String::new();
        }
        if self.error.is_some() {
            return "invalid regex".into();
        }
        let more = if self.capped { "+" } else { "" };
        let dots = if self.scanning { "…" } else { "" };
        match (self.total, self.current) {
            (0, _) if self.scanning => "searching…".into(),
            (0, _) => "no matches".into(),
            (total, Some(n)) => format!("{n} of {total}{more}{dots}"),
            (total, None) => format!("{total}{more} matches{dots}"),
        }
    }
}

/// A search over one screen: the query, its matches and the scan's state.
#[derive(Default)]
pub struct Search {
    query: String,
    regex: bool,
    matcher: Option<Matcher>,
    error: Option<String>,
    /// Ascending and non-overlapping.
    matches: Vec<Match>,
    current: Option<Match>,
    /// The current match moved: the view should show it.
    reveal: bool,
    /// History rows `evicted..scan_hi` are still to be read, newest first.
    scan_hi: u64,
    /// The first row output can still change: the line holding the top of
    /// the live screen when last read.
    live_from: u64,
    /// `evicted + scrollback.len()` when last read.
    history_end: u64,
    /// `None`: start over on the next sync.
    shape: Option<Shape>,
    capped: bool,
    line: LineText,
    found: Vec<Range<usize>>,
}

impl Search {
    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn regex(&self) -> bool {
        self.regex
    }

    pub fn matches(&self) -> &[Match] {
        &self.matches
    }

    pub fn current(&self) -> Option<Match> {
        self.current
    }

    /// A new query or mode: matched from scratch on the next sync, and the
    /// current match picked again near the view.
    pub fn set_query(&mut self, query: &str, regex: bool) {
        self.query = query.to_string();
        self.regex = regex;
        match Matcher::new(query, regex) {
            Ok(matcher) => {
                self.matcher = matcher;
                self.error = None;
            }
            Err(e) => {
                self.matcher = None;
                self.error = Some(e);
            }
        }
        self.matches.clear();
        self.current = None;
        self.shape = None;
    }

    /// Read everything again on the next sync (a reopened search bar).
    pub fn invalidate(&mut self) {
        self.shape = None;
        self.current = None;
    }

    /// Whether the view should move to the current match; clears it.
    pub fn take_reveal(&mut self) -> bool {
        std::mem::take(&mut self.reveal)
    }

    fn scanning(&self, screen: &Screen) -> bool {
        self.matcher.is_some() && !self.capped && self.scan_hi > screen.evicted
    }

    pub fn status(&self, screen: &Screen) -> Status {
        let current = self
            .current
            .and_then(|c| self.matches.binary_search(&c).ok())
            .map(|i| self.matches.len() - i);
        Status {
            current,
            total: self.matches.len(),
            capped: self.capped,
            scanning: self.scanning(screen),
            error: self.error.clone(),
            empty_query: self.query.is_empty(),
        }
    }

    /// Bring the matches up to date with `screen` and read the next part of
    /// history. `changed`: output arrived since the last sync. `view_bottom`:
    /// the last row in view, where the first current match is looked for.
    /// True while history is still being read (call again next frame).
    pub fn sync(
        &mut self,
        screen: &Screen,
        alternate: bool,
        changed: bool,
        view_bottom: u64,
    ) -> bool {
        if self.matcher.is_none() {
            self.matches.clear();
            self.current = None;
            return false;
        }
        let shape = Shape {
            alternate,
            cols: screen.cols,
            rows: screen.rows,
        };
        let history_end = screen.evicted + screen.scrollback.len() as u64;
        let end = screen.evicted + screen.total_rows() as u64;
        if self.shape != Some(shape) || history_end < self.history_end {
            self.restart(screen, shape);
        } else if changed || history_end != self.history_end {
            // Rows gone off the front of history.
            let gone = self.matches.partition_point(|m| m.start.0 < screen.evicted);
            self.matches.drain(..gone);
            let live = self.live_from.max(screen.evicted);
            if end - live > LIVE_ROWS_MAX as u64 {
                self.restart(screen, shape);
            } else {
                // The rows output may have changed, read again.
                let keep = self.matches.partition_point(|m| m.start.0 < live);
                self.matches.truncate(keep);
                self.scan_hi = self.scan_hi.min(live);
                let mut fresh = Vec::new();
                self.scan_back(screen, live, end, usize::MAX, &mut fresh);
                fresh.reverse();
                self.matches.extend(fresh);
                if self.matches.len() > MAX_MATCHES {
                    let over = self.matches.len() - MAX_MATCHES;
                    self.matches.drain(..over);
                    self.capped = true;
                }
                self.live_from = line_start(screen, history_end, screen.evicted);
                self.history_end = history_end;
            }
            self.keep_current();
        }
        if self.scanning(screen) {
            let lo = screen.evicted;
            let mut older = Vec::new();
            let budget = rows_per_step(screen.cols);
            self.scan_hi = self.scan_back(screen, lo, self.scan_hi, budget, &mut older);
            older.reverse();
            let room = MAX_MATCHES.saturating_sub(self.matches.len());
            if older.len() > room {
                older.drain(..older.len() - room);
                self.capped = true;
            }
            older.append(&mut self.matches);
            self.matches = older;
        }
        let more = self.scanning(screen);
        if self.current.is_none() && !self.matches.is_empty() {
            let above = self.matches.partition_point(|m| m.start.0 <= view_bottom);
            if above > 0 {
                self.current = Some(self.matches[above - 1]);
            } else if !more {
                self.current = Some(self.matches[0]);
            }
            self.reveal = self.current.is_some();
        }
        more
    }

    fn restart(&mut self, screen: &Screen, shape: Shape) {
        self.matches.clear();
        self.capped = false;
        self.shape = Some(shape);
        self.scan_hi = screen.evicted + screen.total_rows() as u64;
        self.history_end = screen.evicted + screen.scrollback.len() as u64;
        self.live_from = line_start(screen, self.history_end, screen.evicted);
    }

    /// After a partial re-read: the current match stays if it is still
    /// there, else the nearest one before it takes its place.
    fn keep_current(&mut self) {
        let Some(current) = self.current else {
            return;
        };
        if self.matches.binary_search(&current).is_ok() {
            return;
        }
        let before = self.matches.partition_point(|m| *m < current);
        self.current = match before {
            0 => self.matches.first().copied(),
            n => Some(self.matches[n - 1]),
        };
    }

    /// Read whole logical lines backwards from the row before `hi`, not
    /// before `lo`, for at least one line and up to `budget` rows. Their
    /// matches go to `out` newest first. Returns where the read stopped.
    fn scan_back(
        &mut self,
        screen: &Screen,
        lo: u64,
        hi: u64,
        budget: usize,
        out: &mut Vec<Match>,
    ) -> u64 {
        let Some(matcher) = self.matcher.as_mut() else {
            return lo;
        };
        let mut hi = hi;
        let mut read = 0usize;
        while hi > lo && read < budget {
            let last = hi - 1;
            let first = line_start(screen, last, lo);
            self.line.read(screen, first, last);
            read += (last - first + 1) as usize;
            hi = first;
            if self.line.text.is_empty() {
                continue;
            }
            self.found.clear();
            matcher.find(&self.line.text, &mut self.found);
            let at = out.len();
            let mut prev_end: Option<Pos> = None;
            for range in &self.found {
                let Some(m) = self.line.cells_of(range) else {
                    continue;
                };
                // Two matches inside one cell (a cluster) are one.
                if prev_end.is_some_and(|end| m.start < end) {
                    continue;
                }
                prev_end = Some(m.end);
                out.push(m);
            }
            out[at..].reverse();
        }
        hi
    }

    /// Step to the match above (`older`) or below the current one, wrapping
    /// around. With none current yet, the one nearest `view_bottom`.
    pub fn step(&mut self, older: bool, view_bottom: u64) -> Option<Match> {
        if self.matches.is_empty() {
            return None;
        }
        let len = self.matches.len();
        let next = match self
            .current
            .and_then(|c| self.matches.binary_search(&c).ok())
        {
            Some(i) if older => (i + len - 1) % len,
            Some(i) => (i + 1) % len,
            None => {
                let above = self.matches.partition_point(|m| m.start.0 <= view_bottom);
                above.saturating_sub(1)
            }
        };
        self.current = Some(self.matches[next]);
        self.reveal = true;
        self.current
    }

    /// Mark row `abs`'s matched cells in `marks` (one per column):
    /// [`MARK_MATCH`], or [`MARK_CURRENT`] for the current match.
    pub fn row_marks(&self, abs: u64, cols: usize, marks: &mut Vec<u8>) {
        marks.clear();
        marks.resize(cols, 0);
        let first = self.matches.partition_point(|m| m.end.0 < abs);
        for m in self.matches[first..]
            .iter()
            .take_while(|m| m.start.0 <= abs)
        {
            let from = if m.start.0 == abs { m.start.1 } else { 0 };
            let to = if m.end.0 == abs { m.end.1 } else { cols };
            let mark = if Some(*m) == self.current {
                MARK_CURRENT
            } else {
                MARK_MATCH
            };
            for slot in marks.iter_mut().take(to.min(cols)).skip(from) {
                *slot = mark;
            }
        }
    }
}

pub const MARK_MATCH: u8 = 1;
pub const MARK_CURRENT: u8 = 2;

/// The view offset (rows scrolled back from the bottom) that shows `m`:
/// unchanged when its first row is in view, else with that row a third of
/// the way down.
pub fn reveal_offset(screen: &Screen, view_offset: usize, m: &Match) -> usize {
    let history = screen.scrollback.len();
    let top = history.saturating_sub(view_offset);
    let Some(row) = screen.virtual_of_absolute(m.start.0) else {
        return view_offset;
    };
    if row >= top && row < top + screen.rows {
        return view_offset;
    }
    history - row.saturating_sub(screen.rows / 3).min(history)
}

// ------------------------------------------------------------------
// Key bindings
// ------------------------------------------------------------------

/// What a key does to the search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchKey {
    /// Open the bar (or keep it open and keep typing).
    Open,
    Close,
    /// The match above the current one (towards older output).
    Older,
    /// The match below it (towards newer output).
    Newer,
    ToggleRegex,
    DeleteChar,
    DeleteWord,
    Clear,
}

/// The search bar's own keys, while it is open: editing the query and
/// stepping with Enter. They win over every shortcut (`crate::keybinds`),
/// which holds the rest: opening the bar (Cmd+F on macOS, Ctrl+Shift+F
/// elsewhere) and Cmd+G / Cmd+Shift+G. Search starts at the newest
/// output, so Enter goes up.
pub fn bar_key(key: &KeyEvent) -> Option<SearchKey> {
    let m = &key.modifiers;
    let plain = !m.control && !m.alt && !m.logo;
    let step = |shift: bool| {
        if shift {
            SearchKey::Newer
        } else {
            SearchKey::Older
        }
    };
    match key.key_code {
        KeyCode::Escape => Some(SearchKey::Close),
        KeyCode::ReturnKey | KeyCode::NumpadEnter | KeyCode::F3 if plain => Some(step(m.shift)),
        KeyCode::ArrowUp if plain && !m.shift => Some(SearchKey::Older),
        KeyCode::ArrowDown if plain && !m.shift => Some(SearchKey::Newer),
        KeyCode::KeyR if m.control && !m.alt && !m.logo && !m.shift => Some(SearchKey::ToggleRegex),
        KeyCode::Backspace if m.logo => Some(SearchKey::Clear),
        KeyCode::Backspace if m.alt || m.control => Some(SearchKey::DeleteWord),
        KeyCode::Backspace => Some(SearchKey::DeleteChar),
        KeyCode::KeyU if m.control && !m.alt && !m.logo => Some(SearchKey::Clear),
        KeyCode::KeyW if m.control && !m.alt && !m.logo && !m.shift => Some(SearchKey::DeleteWord),
        _ => None,
    }
}

/// A shortcut's effect on the search, if it is a search action.
pub fn search_action(action: &Action) -> Option<SearchKey> {
    match action {
        Action::StartSearch => Some(SearchKey::Open),
        Action::NavigateSearch { next: true } => Some(SearchKey::Older),
        Action::NavigateSearch { next: false } => Some(SearchKey::Newer),
        Action::EndSearch => Some(SearchKey::Close),
        _ => None,
    }
}

/// What `key` does to the search under the platform's default shortcuts
/// (`mac`): the bar's keys while it is `open`, then the table's.
#[cfg(test)]
pub fn search_key_for(key: &KeyEvent, open: bool, mac: bool) -> Option<SearchKey> {
    use crate::keybinds::{Context, Decision, Keybinds, Scope};
    if open {
        if let Some(k) = bar_key(key) {
            return Some(k);
        }
    }
    let ctx = Context {
        tabs: 1,
        search_open: open,
    };
    match Keybinds::defaults(mac).decide(key, Scope::Pane, &ctx) {
        Decision::Run { action, .. } => search_action(&action),
        Decision::Pass => None,
    }
}

/// `query` without its last word (and the blanks after it).
pub fn delete_word(query: &str) -> String {
    let trimmed = query.trim_end();
    let cut = trimmed
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_whitespace())
        .map_or(0, |(i, c)| i + c.len_utf8());
    trimmed[..cut].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::stream::Stream;
    use crate::term::terminal::Terminal;
    use makepad_widgets::KeyModifiers;

    fn term(cols: usize, rows: usize, scrollback: usize, text: &str) -> Terminal {
        let mut t = Terminal::with_scrollback(cols, rows, scrollback);
        Stream::new().process(text.replace('\n', "\r\n").as_bytes(), &mut t);
        t
    }

    /// Every match of `query`, read to the end.
    fn search(t: &Terminal, query: &str, regex: bool) -> Search {
        let mut s = Search::default();
        s.set_query(query, regex);
        while s.sync(t.screen(), false, false, u64::MAX) {}
        s
    }

    /// Each match's text as copy would give it.
    fn texts(t: &Terminal, s: &Search) -> Vec<String> {
        s.matches()
            .iter()
            .map(|m| t.screen().selection_text(m.start, m.end))
            .collect()
    }

    #[test]
    fn plain_matches_map_to_cells() {
        let t = term(20, 3, 100, "foo bar foo\nxfoo");
        let s = search(&t, "foo", false);
        let m: Vec<_> = s.matches().iter().map(|m| (m.start, m.end)).collect();
        assert_eq!(m, [((0, 0), (0, 3)), ((0, 8), (0, 11)), ((1, 1), (1, 4))]);
    }

    #[test]
    fn empty_query_and_no_match() {
        let t = term(20, 3, 100, "hello");
        let s = search(&t, "", false);
        assert!(s.matches().is_empty());
        assert_eq!(s.status(t.screen()).text(), "");
        let s = search(&t, "absent", false);
        assert!(s.matches().is_empty());
        assert_eq!(s.status(t.screen()).text(), "no matches");
    }

    #[test]
    fn smart_case() {
        let t = term(30, 3, 100, "Error error ERROR");
        assert_eq!(search(&t, "error", false).matches().len(), 3);
        assert_eq!(search(&t, "Error", false).matches().len(), 1);
        assert_eq!(search(&t, "ERROR", false).matches().len(), 1);
        // Escapes are not uppercase letters; a literal one is.
        assert!(!case_sensitive(r"\W\S\p{Lu}\PL", true));
        assert!(case_sensitive(r"\wE", true));
        assert!(case_sensitive(r"\W", false));
        assert_eq!(search(&t, r"e\w+", true).matches().len(), 3);
        assert_eq!(search(&t, r"E\w+", true).matches().len(), 2);
    }

    #[test]
    fn case_folding_beyond_ascii() {
        let t = term(30, 3, 100, "ÄRGER ärger Straße");
        assert_eq!(texts(&t, &search(&t, "ärger", false)), ["ÄRGER", "ärger"]);
        assert_eq!(texts(&t, &search(&t, "straße", false)), ["Straße"]);
    }

    #[test]
    fn regex_matches() {
        let t = term(30, 4, 100, "id=12 id=345\nnone\nid=7");
        let s = search(&t, r"id=\d+", true);
        assert_eq!(texts(&t, &s), ["id=12", "id=345", "id=7"]);
        // Empty matches are skipped, not looped on.
        assert_eq!(search(&t, "x*", true).matches().len(), 0);
        assert_eq!(texts(&t, &search(&t, "^id", true)), ["id", "id"]);
        let t = term(30, 3, 100, "idid");
        assert_eq!(search(&t, "^id", true).matches().len(), 1);
        assert_eq!(search(&t, "id$", true).matches().len(), 1);
        // A bad pattern reports, finds nothing.
        let s = search(&t, "(", true);
        assert!(s.matches().is_empty());
        assert_eq!(s.status(t.screen()).text(), "invalid regex");
    }

    #[test]
    fn matches_cross_soft_wraps() {
        // "hello world" wrapped at 8 columns: "hello wo" / "rld".
        let t = term(8, 3, 100, "hello world\n");
        let s = search(&t, "world", false);
        assert_eq!(
            s.matches(),
            [Match {
                start: (0, 6),
                end: (1, 3)
            }]
        );
        assert_eq!(texts(&t, &s), ["world"]);
        // A hard newline is not a wrap.
        let t = term(8, 3, 100, "wor\nld");
        assert!(search(&t, "world", false).matches().is_empty());
    }

    #[test]
    fn wide_chars_cover_both_cells() {
        let t = term(20, 3, 100, "ab你好cd");
        let s = search(&t, "好c", false);
        assert_eq!(
            s.matches(),
            [Match {
                start: (0, 4),
                end: (0, 7)
            }]
        );
        // A wide char pushed to the next row by the wrap: the spacer head
        // adds no text, so the match still joins across it.
        let t = term(5, 3, 100, "abcd你好");
        let s = search(&t, "d你", false);
        assert_eq!(
            s.matches(),
            [Match {
                start: (0, 3),
                end: (1, 2)
            }]
        );
        assert_eq!(texts(&t, &s), ["d你"]);
    }

    #[test]
    fn clusters_are_matched_whole() {
        // नमस्ते: Indic letters take a cell each, a conjunct's parts join
        // its cell; a match inside a cell covers the cell.
        let t = term(30, 3, 100, "say नमस्ते now");
        let s = search(&t, "नमस्ते", false);
        assert_eq!(texts(&t, &s), ["नमस्ते"]);
        let m = s.matches()[0];
        assert_eq!(m.start, (0, 4));
        let part = search(&t, "स", false);
        assert_eq!(part.matches().len(), 1);
        let cell = part.matches()[0];
        let row = t.screen().row(0);
        let (head, width) = row.span_at(cell.start.1);
        assert_eq!((cell.start.1, cell.end.1), (head, head + width));
        // A ZWJ emoji and a combining mark are one cell each.
        let t = term(30, 3, 100, "x👨\u{200D}👩\u{200D}👧y e\u{0301}z");
        let s = search(&t, "\u{200D}👩", false);
        assert_eq!(
            s.matches(),
            [Match {
                start: (0, 1),
                end: (0, 3)
            }]
        );
        let s = search(&t, "e\u{0301}z", false);
        assert_eq!(texts(&t, &s), ["e\u{0301}z"]);
    }

    #[test]
    fn two_hits_in_one_cell_are_one_match() {
        // e + two combining acutes is one cell holding the mark twice.
        let t = term(30, 3, 100, "e\u{0301}\u{0301}");
        let s = search(&t, "\u{0301}", false);
        assert_eq!(
            s.matches(),
            [Match {
                start: (0, 0),
                end: (0, 1)
            }]
        );
    }

    #[test]
    fn history_is_read_in_steps_newest_first() {
        let mut text = String::new();
        let step = rows_per_step(40);
        for i in 0..(step * 2 + 100) {
            text.push_str(&format!("line {i}\n"));
        }
        let t = term(40, 5, 100_000, &text);
        let mut s = Search::default();
        s.set_query("line", false);
        assert!(s.sync(t.screen(), false, false, u64::MAX));
        let first = s.matches().len();
        assert!(first > 0 && first <= step);
        // The newest lines come first; the current one is the last printed.
        assert_eq!(s.status(t.screen()).current, Some(1));
        assert!(s.status(t.screen()).text().ends_with('…'));
        while s.sync(t.screen(), false, false, u64::MAX) {}
        assert_eq!(s.matches().len(), step * 2 + 100);
        assert!(s.matches().windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn matches_are_capped() {
        let text = "aa\n".repeat(MAX_MATCHES / 2 + 10);
        let t = term(10, 5, 100_000, &text);
        let s = search(&t, "a", false);
        assert_eq!(s.matches().len(), MAX_MATCHES);
        assert!(s.status(t.screen()).text().contains("+"));
        // The newest are kept: the last line printed is above the cursor's.
        let last_row = t.screen().evicted + t.screen().total_rows() as u64 - 2;
        assert_eq!(s.matches().last().unwrap().start.0, last_row);
    }

    #[test]
    fn stepping_wraps_and_counts_from_the_newest() {
        let t = term(20, 3, 100, "m1\nm2\nm3\n");
        let mut s = search(&t, "m", false);
        // Picked near the view's bottom: the newest.
        assert_eq!(s.status(t.screen()).text(), "1 of 3");
        s.step(true, 0);
        assert_eq!(s.status(t.screen()).text(), "2 of 3");
        s.step(true, 0);
        s.step(true, 0);
        assert_eq!(s.status(t.screen()).text(), "1 of 3", "wraps to the newest");
        s.step(false, 0);
        assert_eq!(s.status(t.screen()).text(), "3 of 3", "wraps to the oldest");
        assert!(s.take_reveal());
        assert!(!s.take_reveal());
    }

    #[test]
    fn new_output_keeps_positions() {
        let mut stream = Stream::new();
        let mut t = Terminal::with_scrollback(20, 3, 5);
        stream.process(b"needle 1\r\nx\r\nx\r\nx\r\n", &mut t);
        let mut s = Search::default();
        s.set_query("needle", false);
        while s.sync(t.screen(), false, false, u64::MAX) {}
        assert_eq!(s.matches().len(), 1);
        let first = s.matches()[0];
        // More output, including a new match, scrolls the first one up.
        stream.process(b"needle 2\r\ny\r\n", &mut t);
        while s.sync(t.screen(), false, true, u64::MAX) {}
        assert_eq!(s.matches().len(), 2);
        assert_eq!(s.matches()[0], first, "absolute rows do not move");
        assert_eq!(texts(&t, &s), ["needle", "needle"]);
        // A line the program rewrites is read again.
        stream.process(b"\x1b[2K\rneedle 3", &mut t);
        while s.sync(t.screen(), false, true, u64::MAX) {}
        assert_eq!(s.matches().len(), 3);
        // History evicts the oldest: their matches go with them.
        stream.process("z\r\n".repeat(6).as_bytes(), &mut t);
        while s.sync(t.screen(), false, true, u64::MAX) {}
        assert!(s.matches().iter().all(|m| m.start.0 >= t.screen().evicted));
        assert_eq!(texts(&t, &s), ["needle"]);
    }

    #[test]
    fn current_match_survives_output() {
        let mut stream = Stream::new();
        let mut t = Terminal::with_scrollback(20, 3, 100);
        stream.process(b"hit a\r\nhit b\r\nhit c\r\n", &mut t);
        let mut s = Search::default();
        s.set_query("hit", false);
        while s.sync(t.screen(), false, false, u64::MAX) {}
        s.step(true, 0);
        let current = s.current().unwrap();
        stream.process(b"more\r\nhit d\r\n", &mut t);
        s.sync(t.screen(), false, true, u64::MAX);
        assert_eq!(s.current(), Some(current));
        assert_eq!(s.status(t.screen()).text(), "3 of 4");
    }

    #[test]
    fn a_resize_or_screen_switch_starts_over() {
        let mut t = term(10, 3, 100, "abc abc abc abc");
        let mut s = search(&t, "abc", false);
        assert_eq!(s.matches().len(), 4);
        t.resize(20, 3);
        while s.sync(t.screen(), false, true, u64::MAX) {}
        assert_eq!(texts(&t, &s), ["abc"; 4]);
        assert!(!s.sync(t.screen(), true, true, u64::MAX));
        assert_eq!(s.matches().len(), 4, "same text on the other screen here");
    }

    #[test]
    fn row_marks_cover_wrapped_matches() {
        let t = term(8, 3, 100, "hello world\n");
        let s = search(&t, "world", false);
        let mut marks = Vec::new();
        s.row_marks(0, 8, &mut marks);
        assert_eq!(marks, [0, 0, 0, 0, 0, 0, 2, 2]);
        s.row_marks(1, 8, &mut marks);
        assert_eq!(marks, [2, 2, 2, 0, 0, 0, 0, 0]);
        s.row_marks(2, 8, &mut marks);
        assert!(marks.iter().all(|m| *m == 0));
    }

    #[test]
    fn reveal_scrolls_only_when_needed() {
        let mut text = String::new();
        for i in 0..100 {
            text.push_str(&format!("row {i}\n"));
        }
        let t = term(20, 10, 1000, &text);
        let screen = t.screen();
        let history = screen.scrollback.len();
        let s = search(&t, "row 20", false);
        let m = s.matches()[0];
        let offset = reveal_offset(screen, 0, &m);
        let top = history - offset;
        let virt = screen.virtual_of_absolute(m.start.0).unwrap();
        assert!(virt >= top && virt < top + screen.rows);
        assert_eq!(virt - top, screen.rows / 3);
        // Already in view: the view stays.
        assert_eq!(reveal_offset(screen, offset + 2, &m), offset + 2);
        // A match on the live screen goes back to the bottom.
        let s = search(&t, "row 99", false);
        assert_eq!(reveal_offset(screen, 50, &s.matches()[0]), 0);
    }

    fn key(code: KeyCode, control: bool, shift: bool, alt: bool, logo: bool) -> KeyEvent {
        KeyEvent {
            key_code: code,
            is_repeat: false,
            modifiers: KeyModifiers {
                shift,
                control,
                alt,
                logo,
            },
            time: 0.0,
        }
    }

    #[test]
    fn search_keys() {
        let k = |code, c, s, a, l, open, mac| search_key_for(&key(code, c, s, a, l), open, mac);
        assert_eq!(
            k(KeyCode::KeyF, false, false, false, true, false, true),
            Some(SearchKey::Open)
        );
        assert_eq!(
            k(KeyCode::KeyF, true, true, false, false, false, false),
            Some(SearchKey::Open)
        );
        // Ctrl+F stays the shell's (forward char); Cmd+F is not bound off macOS.
        assert_eq!(
            k(KeyCode::KeyF, true, false, false, false, false, false),
            None
        );
        assert_eq!(
            k(KeyCode::KeyF, false, false, false, true, false, false),
            None
        );
        // Closed, nothing else is taken.
        assert_eq!(
            k(KeyCode::Escape, false, false, false, false, false, true),
            None
        );
        assert_eq!(
            k(KeyCode::ReturnKey, false, false, false, false, false, true),
            None
        );
        let open = |code, c, s, a, l| k(code, c, s, a, l, true, true);
        assert_eq!(
            open(KeyCode::Escape, false, false, false, false),
            Some(SearchKey::Close)
        );
        assert_eq!(
            open(KeyCode::ReturnKey, false, false, false, false),
            Some(SearchKey::Older)
        );
        assert_eq!(
            open(KeyCode::ReturnKey, false, true, false, false),
            Some(SearchKey::Newer)
        );
        assert_eq!(
            open(KeyCode::KeyG, false, false, false, true),
            Some(SearchKey::Older)
        );
        assert_eq!(
            open(KeyCode::KeyG, false, true, false, true),
            Some(SearchKey::Newer)
        );
        assert_eq!(
            open(KeyCode::KeyR, true, false, false, false),
            Some(SearchKey::ToggleRegex)
        );
        assert_eq!(
            open(KeyCode::Backspace, false, false, false, false),
            Some(SearchKey::DeleteChar)
        );
        assert_eq!(
            open(KeyCode::Backspace, false, false, true, false),
            Some(SearchKey::DeleteWord)
        );
        assert_eq!(
            open(KeyCode::Backspace, false, false, false, true),
            Some(SearchKey::Clear)
        );
        assert_eq!(open(KeyCode::KeyA, false, false, false, false), None);
    }

    /// The search keys before shortcuts were configurable, verbatim.
    fn legacy_search_key_for(key: &KeyEvent, open: bool, mac: bool) -> Option<SearchKey> {
        let m = &key.modifiers;
        let code = key.key_code;
        // Open: Cmd+F on macOS, Ctrl+Shift+F elsewhere.
        let open_key = if mac {
            m.logo && !m.control && !m.alt && !m.shift
        } else {
            m.control && m.shift && !m.alt && !m.logo
        };
        if code == KeyCode::KeyF && open_key {
            return Some(SearchKey::Open);
        }
        if !open {
            return None;
        }
        let plain = !m.control && !m.alt && !m.logo;
        let step = |shift: bool| {
            if shift {
                SearchKey::Newer
            } else {
                SearchKey::Older
            }
        };
        match code {
            KeyCode::Escape => Some(SearchKey::Close),
            KeyCode::ReturnKey | KeyCode::NumpadEnter | KeyCode::F3 if plain => Some(step(m.shift)),
            // Cmd+G / Cmd+Shift+G on macOS; Ctrl+Shift+G / Ctrl+Alt+Shift+G is
            // left alone elsewhere (Enter and F3 step).
            KeyCode::KeyG if mac && m.logo && !m.control && !m.alt => Some(step(m.shift)),
            KeyCode::ArrowUp if plain && !m.shift => Some(SearchKey::Older),
            KeyCode::ArrowDown if plain && !m.shift => Some(SearchKey::Newer),
            KeyCode::KeyR if m.control && !m.alt && !m.logo && !m.shift => {
                Some(SearchKey::ToggleRegex)
            }
            KeyCode::Backspace if m.logo => Some(SearchKey::Clear),
            KeyCode::Backspace if m.alt || m.control => Some(SearchKey::DeleteWord),
            KeyCode::Backspace => Some(SearchKey::DeleteChar),
            KeyCode::KeyU if m.control && !m.alt && !m.logo => Some(SearchKey::Clear),
            KeyCode::KeyW if m.control && !m.alt && !m.logo && !m.shift => {
                Some(SearchKey::DeleteWord)
            }
            _ => None,
        }
    }

    #[test]
    fn the_default_table_and_the_bar_take_exactly_the_old_search_keys() {
        for mac in [false, true] {
            for &code in crate::keybinds::ALL_KEY_CODES {
                for bits in 0..16u8 {
                    let k = key(
                        code,
                        bits & 1 != 0,
                        bits & 2 != 0,
                        bits & 4 != 0,
                        bits & 8 != 0,
                    );
                    for open in [false, true] {
                        assert_eq!(
                            search_key_for(&k, open, mac),
                            legacy_search_key_for(&k, open, mac),
                            "mac {mac}, open {open}: {code:?} {:?}",
                            k.modifiers
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn words_are_deleted() {
        assert_eq!(delete_word("foo bar"), "foo ");
        assert_eq!(delete_word("foo bar  "), "foo ");
        assert_eq!(delete_word("foo"), "");
        assert_eq!(delete_word(""), "");
    }
}
