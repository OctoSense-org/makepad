//! The pointer gesture state machine behind selection in the terminal.
//!
//! Kept free of Makepad types so it can be tested on its own: the widget
//! feeds it presses and moves and acts on what it returns.
//!
//! - A mouse click that does not move selects nothing: it focuses the pane
//!   and clears any selection. Dragging further than [`DRAG_THRESHOLD`]
//!   starts a character selection from the press point, as in Ghostty,
//!   Terminal.app and iTerm2.
//! - A double click selects a word, a triple click a line; dragging after
//!   either extends by that unit.
//! - When the program asked for mouse reports (vim, tmux, htop with mouse
//!   mode on) a single click and its drag go to the program; Shift makes the
//!   press local, so Shift+drag selects.
//! - One finger on a touchscreen neither selects nor scrolls (two fingers
//!   scroll, handled by the widget); a double tap selects a word.

/// How far, in logical pixels, a press must travel before it is a drag.
pub const DRAG_THRESHOLD: f64 = 4.0;

/// What the current press is doing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Press {
    /// No press in progress.
    #[default]
    Idle,
    /// The press went to a mouse-reporting program; its moves and release
    /// follow it there.
    App,
    /// A local press that has not yet moved past the threshold: released
    /// here, it is a plain click.
    Click { origin: (f64, f64) },
    /// A character selection, anchored where the press started.
    Chars,
    /// A word or line selection from a double or triple click.
    Unit,
    /// One finger on a touchscreen: no selection, no scroll.
    Touch,
}

/// The facts about a press that decide what it becomes.
#[derive(Clone, Copy, Debug, Default)]
pub struct PressInfo {
    pub tap_count: u32,
    pub touch: bool,
    /// The program has mouse tracking on.
    pub mouse_reporting: bool,
    pub shift: bool,
}

/// What the widget does for a move.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MoveAction {
    /// Nothing yet (a click still under the threshold, or a finger).
    None,
    /// Report the motion to the program.
    Report,
    /// The press just became a drag: anchor a character selection at the
    /// press origin and extend it to the pointer.
    StartChars { origin: (f64, f64) },
    /// Extend the selection to the pointer.
    Extend,
}

impl Press {
    /// A press begins at `pos`.
    pub fn down(info: PressInfo, pos: (f64, f64)) -> Press {
        if info.tap_count >= 2 {
            // Word or line selection is local even under mouse reporting:
            // the program saw only the first click.
            return Press::Unit;
        }
        if info.touch {
            return Press::Touch;
        }
        if info.mouse_reporting && !info.shift {
            return Press::App;
        }
        Press::Click { origin: pos }
    }

    /// The pointer moved to `pos` while pressed.
    pub fn moved(&mut self, pos: (f64, f64)) -> MoveAction {
        match *self {
            Press::Idle | Press::Touch => MoveAction::None,
            Press::App => MoveAction::Report,
            Press::Click { origin } => {
                if past_threshold(origin, pos) {
                    *self = Press::Chars;
                    MoveAction::StartChars { origin }
                } else {
                    MoveAction::None
                }
            }
            Press::Chars | Press::Unit => MoveAction::Extend,
        }
    }

    /// The press is selecting (drives edge auto-scroll and copy-on-select).
    pub fn is_selecting(&self) -> bool {
        matches!(self, Press::Chars | Press::Unit)
    }

    /// The press belongs to the program, so its release is reported.
    pub fn reports(&self) -> bool {
        matches!(self, Press::App)
    }
}

/// Whether `pos` has moved far enough from `origin` to be a drag.
pub fn past_threshold(origin: (f64, f64), pos: (f64, f64)) -> bool {
    let (dx, dy) = (pos.0 - origin.0, pos.1 - origin.1);
    dx * dx + dy * dy >= DRAG_THRESHOLD * DRAG_THRESHOLD
}

/// The column boundary nearest `local_x` (0..=cols): a character
/// selection runs between boundaries, so a drag from the left half of a
/// cell includes it and one from its right half does not.
pub fn boundary_col(local_x: f64, cell_w: f64, cols: usize) -> usize {
    if cell_w <= 0.0 {
        return 0;
    }
    ((local_x / cell_w).round().max(0.0) as usize).min(cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mouse() -> PressInfo {
        PressInfo {
            tap_count: 1,
            ..Default::default()
        }
    }

    #[test]
    fn click_without_movement_selects_nothing() {
        let mut p = Press::down(mouse(), (10.0, 10.0));
        assert_eq!(
            p,
            Press::Click {
                origin: (10.0, 10.0)
            }
        );
        assert_eq!(p.moved((11.0, 11.0)), MoveAction::None);
        assert_eq!(p.moved((12.0, 10.0)), MoveAction::None);
        assert!(!p.is_selecting());
        assert!(!p.reports());
    }

    #[test]
    fn drag_past_threshold_starts_character_selection() {
        let mut p = Press::down(mouse(), (10.0, 10.0));
        assert_eq!(p.moved((13.0, 10.0)), MoveAction::None);
        assert_eq!(
            p.moved((14.0, 10.0)),
            MoveAction::StartChars {
                origin: (10.0, 10.0)
            }
        );
        assert_eq!(p, Press::Chars);
        assert!(p.is_selecting());
        assert_eq!(p.moved((40.0, 30.0)), MoveAction::Extend);
        // Back under the threshold it stays a selection.
        assert_eq!(p.moved((10.0, 10.0)), MoveAction::Extend);
    }

    #[test]
    fn threshold_is_a_distance_not_per_axis() {
        assert!(!past_threshold((0.0, 0.0), (2.0, 3.0)));
        assert!(past_threshold((0.0, 0.0), (3.0, 3.0)));
        assert!(past_threshold((0.0, 0.0), (0.0, -4.0)));
    }

    #[test]
    fn double_and_triple_click_select_by_unit() {
        for taps in [2, 3] {
            let mut p = Press::down(
                PressInfo {
                    tap_count: taps,
                    ..mouse()
                },
                (0.0, 0.0),
            );
            assert_eq!(p, Press::Unit);
            assert!(p.is_selecting());
            assert_eq!(p.moved((1.0, 0.0)), MoveAction::Extend);
        }
    }

    #[test]
    fn one_finger_touch_neither_selects_nor_reports() {
        let info = PressInfo {
            touch: true,
            mouse_reporting: true,
            ..mouse()
        };
        let mut p = Press::down(info, (0.0, 0.0));
        assert_eq!(p, Press::Touch);
        assert_eq!(p.moved((0.0, 200.0)), MoveAction::None);
        assert!(!p.is_selecting());
        assert!(!p.reports());
        // A double tap still selects a word.
        let p = Press::down(
            PressInfo {
                tap_count: 2,
                ..info
            },
            (0.0, 0.0),
        );
        assert_eq!(p, Press::Unit);
    }

    #[test]
    fn mouse_reporting_sends_click_and_drag_to_the_program() {
        let info = PressInfo {
            mouse_reporting: true,
            ..mouse()
        };
        let mut p = Press::down(info, (0.0, 0.0));
        assert_eq!(p, Press::App);
        assert!(p.reports());
        assert_eq!(p.moved((50.0, 50.0)), MoveAction::Report);
        assert!(!p.is_selecting());
    }

    #[test]
    fn shift_overrides_mouse_reporting() {
        let info = PressInfo {
            mouse_reporting: true,
            shift: true,
            ..mouse()
        };
        let mut p = Press::down(info, (0.0, 0.0));
        assert_eq!(p, Press::Click { origin: (0.0, 0.0) });
        assert!(!p.reports());
        assert_eq!(
            p.moved((20.0, 0.0)),
            MoveAction::StartChars { origin: (0.0, 0.0) }
        );
        assert!(p.is_selecting());
    }

    #[test]
    fn boundaries_round_to_the_nearest_column() {
        assert_eq!(boundary_col(0.0, 8.0, 80), 0);
        assert_eq!(boundary_col(3.9, 8.0, 80), 0);
        assert_eq!(boundary_col(4.1, 8.0, 80), 1);
        assert_eq!(boundary_col(-20.0, 8.0, 80), 0);
        assert_eq!(boundary_col(10_000.0, 8.0, 80), 80);
        assert_eq!(boundary_col(5.0, 0.0, 80), 0);
    }
}
