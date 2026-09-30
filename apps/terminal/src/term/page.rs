//! Cell and row storage.
//!
//! Ghostty stores cells in paged memory with interned styles; here rows own
//! their cells inline (styles are small) and scrollback rows are truncated
//! at the last meaningful cell to keep memory sane. The cell semantics —
//! wide heads/tails, spacer heads at wrapped wide chars, grapheme clusters —
//! are ported from ghostty `src/terminal/page.zig`.

use crate::term::style::Style;

/// What a cell displays.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum CellContent {
    /// Never written; renders as background.
    #[default]
    Empty,
    /// Single codepoint, width 1.
    Char(char),
    /// Single codepoint, width 2. The following cell must be `WideTail`.
    WideChar(char),
    /// A column after the first of a wide char or a wide cluster: a
    /// cluster `width` cells wide is followed by `width - 1` of these.
    WideTail,
    /// The columns left at the end of a row when a wide char or cluster
    /// had to wrap: renders as blank, marks that the wrap was forced by
    /// width (ghostty `spacer_head`). A cluster wider than 2 can leave more
    /// than one.
    WideSpacerHead,
    /// A multi-codepoint grapheme cluster (mode 2027 or combining marks).
    Cluster(Box<Cluster>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cluster {
    pub cps: Vec<char>,
    /// 1 or 2.
    pub width: u8,
}

impl CellContent {
    /// Display width this content occupies (0 for tails/spacers — they are
    /// covered by their head cell).
    pub fn width(&self) -> u8 {
        match self {
            CellContent::Empty | CellContent::Char(_) => 1,
            CellContent::WideChar(_) => 2,
            CellContent::WideTail | CellContent::WideSpacerHead => 0,
            CellContent::Cluster(c) => c.width,
        }
    }

    /// A cell whose content continues into `WideTail` cells after it.
    pub fn is_wide_head(&self) -> bool {
        match self {
            CellContent::WideChar(_) => true,
            CellContent::Cluster(c) => c.width >= 2,
            _ => false,
        }
    }

    /// The primary codepoint, if any.
    pub fn primary(&self) -> Option<char> {
        match self {
            CellContent::Char(c) | CellContent::WideChar(c) => Some(*c),
            CellContent::Cluster(c) => c.cps.first().copied(),
            _ => None,
        }
    }

    /// Append the textual content to `out` (for selection/copy).
    pub fn push_text(&self, out: &mut String) {
        match self {
            CellContent::Empty => out.push(' '),
            CellContent::Char(c) | CellContent::WideChar(c) => out.push(*c),
            CellContent::Cluster(c) => out.extend(c.cps.iter()),
            CellContent::WideTail | CellContent::WideSpacerHead => {}
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cell {
    pub content: CellContent,
    pub style: Style,
    /// Hyperlink id into the terminal's hyperlink table; 0 = none.
    pub hyperlink: u32,
}

impl Cell {
    pub fn is_default(&self) -> bool {
        self.content == CellContent::Empty && self.style.is_default() && self.hyperlink == 0
    }

    /// A blank cell carrying only a background style (erase semantics: bg
    /// color survives, everything else resets — ghostty erase behavior).
    pub fn blank_with_bg(style: &Style) -> Cell {
        Cell {
            content: CellContent::Empty,
            style: Style {
                bg_color: style.bg_color,
                ..Style::default()
            },
            hyperlink: 0,
        }
    }
}

/// OSC 133 semantic row marking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SemanticPrompt {
    #[default]
    Output,
    Prompt,
    PromptContinuation,
    Input,
}

#[derive(Clone, Debug, Default)]
pub struct Row {
    /// May be shorter than the grid width; cells beyond are default.
    pub cells: Vec<Cell>,
    /// This row soft-wraps into the next row.
    pub wrapped: bool,
    pub semantic: SemanticPrompt,
}

impl Row {
    pub fn new() -> Row {
        Row::default()
    }

    #[inline]
    pub fn cell(&self, col: usize) -> Option<&Cell> {
        self.cells.get(col)
    }

    /// Mutable access, growing storage (with default cells) as needed.
    #[inline]
    pub fn cell_mut(&mut self, col: usize) -> &mut Cell {
        if col >= self.cells.len() {
            self.cells.resize(col + 1, Cell::default());
        }
        &mut self.cells[col]
    }

    /// Trim trailing default cells (called when a row moves to scrollback).
    pub fn trim(&mut self) {
        while self.cells.last().map(|c| c.is_default()).unwrap_or(false) {
            self.cells.pop();
        }
        self.cells.shrink_to_fit();
    }

    /// Clear the whole row to blank-with-bg of `style`, at `cols` width.
    /// A fully default clear stores nothing (empty vec).
    pub fn clear(&mut self, style: &Style, cols: usize) {
        self.wrapped = false;
        self.semantic = SemanticPrompt::Output;
        let blank = Cell::blank_with_bg(style);
        if blank.is_default() {
            self.cells.clear();
        } else {
            self.cells.clear();
            self.cells.resize(cols, blank);
        }
    }

    /// If `col` lands on a wide tail, step back to its head column.
    pub fn head_of(&self, col: usize) -> usize {
        let mut col = col;
        while col > 0
            && self
                .cell(col)
                .is_some_and(|c| c.content == CellContent::WideTail)
        {
            col -= 1;
        }
        col
    }

    /// The columns the character or cluster covering `col` occupies, as
    /// (head column, width): a wide char or a multi-cell cluster is one
    /// unit for the cursor, the selection and copy. Width is at least 1.
    pub fn span_at(&self, col: usize) -> (usize, usize) {
        let head = self.head_of(col);
        let width = self
            .cell(head)
            .map_or(1, |c| c.content.width().max(1) as usize);
        // A stray tail (its head overwritten) stands for itself.
        if head + width <= col {
            return (col, 1);
        }
        (head, width)
    }

    /// Clearing/overwriting `col` must not leave pieces of a wide char or
    /// cluster: every other cell of the unit covering `col` is blanked (its
    /// head, its tails). The caller writes `col` itself afterwards.
    pub fn split_wide_at(&mut self, col: usize, style: &Style) {
        let len = self.cells.len();
        if col >= len {
            return;
        }
        let head = self.head_of(col);
        let width = self.cells[head].content.width().max(1) as usize;
        let tail = self.cells[col].content == CellContent::WideTail;
        if head == col && width < 2 {
            return;
        }
        // A tail's head (even a narrow one, if the tail was orphaned) and
        // the tails of the unit.
        if tail && head < col {
            self.cells[head] = Cell::blank_with_bg(style);
        }
        for c in head + 1..(head + width.max(col - head + 1)).min(len) {
            if c != col && self.cells[c].content == CellContent::WideTail {
                self.cells[c] = Cell::blank_with_bg(style);
            }
        }
    }

    /// Row text with trailing whitespace trimmed (for copy).
    pub fn text(&self) -> String {
        let mut s = String::new();
        for cell in &self.cells {
            cell.content.push_text(&mut s);
        }
        while s.ends_with(' ') {
            s.pop();
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_split() {
        let mut row = Row::new();
        *row.cell_mut(0) = Cell {
            content: CellContent::WideChar('漢'),
            ..Default::default()
        };
        *row.cell_mut(1) = Cell {
            content: CellContent::WideTail,
            ..Default::default()
        };
        // Overwriting the tail blanks the head.
        row.split_wide_at(1, &Style::default());
        assert_eq!(row.cells[0].content, CellContent::Empty);
        assert_eq!(row.cells[1].content, CellContent::WideTail);
    }

    fn cluster_row(width: u8) -> Row {
        let mut row = Row::new();
        *row.cell_mut(0) = Cell {
            content: CellContent::Char('a'),
            ..Default::default()
        };
        *row.cell_mut(1) = Cell {
            content: CellContent::Cluster(Box::new(Cluster {
                cps: "स्ते".chars().collect(),
                width,
            })),
            ..Default::default()
        };
        for col in 2..1 + width as usize {
            row.cell_mut(col).content = CellContent::WideTail;
        }
        *row.cell_mut(1 + width as usize) = Cell {
            content: CellContent::Char('b'),
            ..Default::default()
        };
        row
    }

    #[test]
    fn multi_cell_cluster_is_one_unit() {
        let row = cluster_row(3);
        assert!(row.cells[1].content.is_wide_head());
        for col in 1..4 {
            assert_eq!(row.head_of(col), 1);
            assert_eq!(row.span_at(col), (1, 3));
        }
        assert_eq!(row.span_at(0), (0, 1));
        assert_eq!(row.span_at(4), (4, 1));
        assert_eq!(row.text(), "aस्तेb");
    }

    #[test]
    fn multi_cell_split() {
        // Overwriting the middle tail blanks the head and the other tail.
        let mut row = cluster_row(3);
        row.split_wide_at(2, &Style::default());
        let contents: Vec<_> = row.cells.iter().map(|c| c.content.clone()).collect();
        assert_eq!(contents[1], CellContent::Empty);
        assert_eq!(contents[2], CellContent::WideTail);
        assert_eq!(contents[3], CellContent::Empty);
        assert_eq!(contents[4], CellContent::Char('b'));
        // Overwriting the head blanks every tail.
        let mut row = cluster_row(4);
        row.split_wide_at(1, &Style::default());
        assert!(row.cells[2..5]
            .iter()
            .all(|c| c.content == CellContent::Empty));
        assert_eq!(row.cells[5].content, CellContent::Char('b'));
        assert_eq!(row.cells[0].content, CellContent::Char('a'));
    }

    #[test]
    fn trim_drops_default_tail() {
        let mut row = Row::new();
        *row.cell_mut(0) = Cell {
            content: CellContent::Char('a'),
            ..Default::default()
        };
        row.cell_mut(9); // grow with defaults
        assert_eq!(row.cells.len(), 10);
        row.trim();
        assert_eq!(row.cells.len(), 1);
    }
}
