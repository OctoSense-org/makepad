//! Synchronized output (DEC private mode 2026).
//!
//! A program brackets a frame with `CSI ? 2026 h` (begin) and
//! `CSI ? 2026 l` (end) so the terminal never shows it half drawn. While a
//! frame is open the bytes after the begin are held, not parsed: the
//! emulator, and so everything the widget draws for any reason (a blink, a
//! host redraw), stays on the last complete frame. The end marker releases
//! the whole frame in one parse and one redraw.
//!
//! A program that dies mid-frame must not freeze the screen: after
//! [`SYNC_TIMEOUT`] (about a second, as other terminals use) or
//! [`SYNC_HOLD_LIMIT`] held bytes, the frame is released as it is. A resize
//! releases it too, since the program redraws for the new size anyway.
//!
//! The begin is detected by parsing (any `DECSET` that turns the mode on,
//! alone or with other modes); the end, in bytes that are not parsed yet,
//! by the exact `CSI ? 2026 l` programs send. Replies to queries inside a
//! frame wait with it, which is how other terminals behave as well.

use crate::term::modes::Mode;
use crate::term::stream::Stream;
use crate::term::terminal::Terminal;

/// The longest a frame is held for an end marker that doesn't come, in
/// seconds. Times here are seconds on a monotonic clock
/// (`Cx::monotonic_now`).
pub const SYNC_TIMEOUT: f64 = 1.0;

/// The most bytes a frame may hold before it is shown anyway.
pub const SYNC_HOLD_LIMIT: usize = 8 << 20;

/// End synchronized update.
const END: &[u8] = b"\x1b[?2026l";

#[derive(Default)]
pub struct SyncGate {
    /// Unparsed bytes of the open frame.
    held: Vec<u8>,
    /// When the open frame began; `None` while bytes flow straight through.
    since: Option<f64>,
    /// `held[..scanned]` holds no end marker (except one cut at the end).
    scanned: usize,
    /// The mode as last seen, to catch the moment it turns on.
    mode_on: bool,
}

impl SyncGate {
    /// A frame is open and being held.
    pub fn holding(&self) -> bool {
        self.since.is_some()
    }

    /// When the open frame will be released even without its end.
    pub fn deadline(&self) -> Option<f64> {
        self.since.map(|since| since + SYNC_TIMEOUT)
    }

    /// Feed program output. True when the emulator changed.
    pub fn feed(
        &mut self,
        bytes: &[u8],
        stream: &mut Stream,
        terminal: &mut Terminal,
        now: f64,
    ) -> bool {
        if self.since.is_some() {
            return self.hold(bytes, stream, terminal, now);
        }
        match self.parse(bytes, stream, terminal) {
            None => !bytes.is_empty(),
            Some(rest) => {
                self.since = Some(now);
                self.hold(&bytes[rest..], stream, terminal, now);
                true
            }
        }
    }

    /// Release the open frame if it has been held too long. True when that
    /// changed the emulator.
    pub fn expire(&mut self, stream: &mut Stream, terminal: &mut Terminal, now: f64) -> bool {
        match self.deadline() {
            Some(deadline) if now >= deadline => self.release(stream, terminal),
            _ => false,
        }
    }

    /// Show the open frame as it is (timeout, resize, program exit). True
    /// when anything was held.
    pub fn release(&mut self, stream: &mut Stream, terminal: &mut Terminal) -> bool {
        if self.since.take().is_none() {
            return false;
        }
        self.scanned = 0;
        let held = std::mem::take(&mut self.held);
        // The mode is still on, so no new frame can begin in here: the
        // program gets its bytes parsed live until it ends the mode.
        self.parse(&held, stream, terminal);
        true
    }

    /// Hold bytes of an open frame; release it when its end arrives.
    fn hold(
        &mut self,
        bytes: &[u8],
        stream: &mut Stream,
        terminal: &mut Terminal,
        now: f64,
    ) -> bool {
        self.held.extend_from_slice(bytes);
        let mut changed = false;
        loop {
            let Some(at) = find(&self.held[self.scanned..], END) else {
                self.scanned = self.held.len().saturating_sub(END.len() - 1);
                if self.held.len() > SYNC_HOLD_LIMIT {
                    changed |= self.release(stream, terminal);
                }
                return changed;
            };
            // The frame is complete: parse it, and what follows, at once.
            let work = std::mem::take(&mut self.held);
            self.since = None;
            self.scanned = 0;
            changed = true;
            match self.parse(&work, stream, terminal) {
                None => return changed,
                Some(rest) => {
                    // Another frame began after this one ended.
                    debug_assert!(rest >= at + END.len());
                    self.since = Some(now);
                    self.held.extend_from_slice(&work[rest..]);
                }
            }
        }
    }

    /// Parse bytes; stop right after a sequence that turns the mode on and
    /// return where the rest starts.
    fn parse(
        &mut self,
        bytes: &[u8],
        stream: &mut Stream,
        terminal: &mut Terminal,
    ) -> Option<usize> {
        for (i, &b) in bytes.iter().enumerate() {
            stream.next(b, terminal);
            // Only DECSET/DECRST (h, l) and RIS (ESC c) change the mode.
            if matches!(b, b'h' | b'l' | b'c') {
                let on = terminal.modes.get(Mode::SynchronizedOutput);
                let began = on && !self.mode_on;
                self.mode_on = on;
                if began {
                    return Some(i + 1);
                }
            }
        }
        None
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (SyncGate, Stream, Terminal) {
        (SyncGate::default(), Stream::new(), Terminal::new(20, 3))
    }

    fn row(t: &Terminal, y: usize) -> String {
        t.screen().row(y).text().trim_end().to_string()
    }

    #[test]
    fn a_frame_is_held_until_it_ends() {
        let (mut g, mut s, mut t) = setup();
        let now = 10.0;
        assert!(g.feed(b"old frame", &mut s, &mut t, now));
        // Begin, then half a frame: nothing of it reaches the screen.
        assert!(g.feed(b"\x1b[?2026h\x1b[H\x1b[2Jnew", &mut s, &mut t, now));
        assert!(g.holding());
        assert_eq!(row(&t, 0), "old frame");
        assert!(!g.feed(b" frame, half", &mut s, &mut t, now));
        assert_eq!(row(&t, 0), "old frame");
        assert!(t.modes.get(Mode::SynchronizedOutput), "DECRQM sees it on");
        // The end shows the whole frame at once.
        assert!(g.feed(b" done\x1b[?2026l", &mut s, &mut t, now));
        assert!(!g.holding());
        assert_eq!(row(&t, 0), "new frame, half done");
        assert!(!t.modes.get(Mode::SynchronizedOutput));
    }

    #[test]
    fn the_end_marker_may_be_split_across_reads() {
        let (mut g, mut s, mut t) = setup();
        let now = 10.0;
        g.feed(b"\x1b[?2026hA\x1b[?20", &mut s, &mut t, now);
        assert_eq!(row(&t, 0), "");
        g.feed(b"26", &mut s, &mut t, now);
        assert!(g.holding());
        g.feed(b"lB", &mut s, &mut t, now);
        assert!(!g.holding());
        assert_eq!(row(&t, 0), "AB");
    }

    #[test]
    fn frames_back_to_back_in_one_read() {
        let (mut g, mut s, mut t) = setup();
        let now = 10.0;
        g.feed(
            b"\x1b[?2026h1\x1b[?2026l\x1b[?2026h2\x1b[?2026l\x1b[?2026h3",
            &mut s,
            &mut t,
            now,
        );
        // The first two are complete; the third is held.
        assert!(g.holding());
        assert_eq!(row(&t, 0), "12");
        g.feed(b"\x1b[?2026l", &mut s, &mut t, now);
        assert_eq!(row(&t, 0), "123");
    }

    #[test]
    fn begin_combined_with_other_modes() {
        let (mut g, mut s, mut t) = setup();
        let now = 10.0;
        g.feed(b"\x1b[?25;2026hX", &mut s, &mut t, now);
        assert!(g.holding());
        assert_eq!(row(&t, 0), "");
        g.feed(b"\x1b[?2026l", &mut s, &mut t, now);
        assert_eq!(row(&t, 0), "X");
    }

    #[test]
    fn a_frame_that_never_ends_is_released_after_the_timeout() {
        let (mut g, mut s, mut t) = setup();
        let now = 10.0;
        g.feed(b"\x1b[?2026hstuck", &mut s, &mut t, now);
        assert!(!g.expire(&mut s, &mut t, now + SYNC_TIMEOUT / 2.0));
        assert_eq!(row(&t, 0), "");
        assert_eq!(g.deadline(), Some(now + SYNC_TIMEOUT));
        assert!(g.expire(&mut s, &mut t, now + SYNC_TIMEOUT));
        assert!(!g.holding());
        assert_eq!(row(&t, 0), "stuck");
        // Output keeps flowing live while the mode stays on...
        assert!(g.feed(b"!", &mut s, &mut t, now + SYNC_TIMEOUT));
        assert_eq!(row(&t, 0), "stuck!");
        // ...and the next frame is held again once the mode was reset.
        g.feed(b"\x1b[?2026l\x1b[?2026h?", &mut s, &mut t, now);
        assert!(g.holding());
        assert_eq!(row(&t, 0), "stuck!");
    }

    #[test]
    fn release_shows_the_frame_as_it_is() {
        let (mut g, mut s, mut t) = setup();
        let now = 10.0;
        g.feed(b"\x1b[?2026hpart", &mut s, &mut t, now);
        assert!(g.release(&mut s, &mut t));
        assert_eq!(row(&t, 0), "part");
        assert!(!g.release(&mut s, &mut t));
    }

    #[test]
    fn a_huge_frame_is_shown_rather_than_held_without_bound() {
        let (mut g, mut s, mut t) = setup();
        let now = 10.0;
        g.feed(b"\x1b[?2026h", &mut s, &mut t, now);
        let chunk = vec![b'x'; 1 << 20];
        let mut released = false;
        for _ in 0..9 {
            released |= g.feed(&chunk, &mut s, &mut t, now);
        }
        assert!(released);
        assert!(!g.holding());
    }

    #[test]
    fn plain_output_flows_through() {
        let (mut g, mut s, mut t) = setup();
        let now = 10.0;
        assert!(g.feed(b"hello \x1b[?25l\x1b[?25hworld", &mut s, &mut t, now));
        assert!(!g.holding());
        assert_eq!(row(&t, 0), "hello world");
        // A reset while a frame is open: RIS arrives inside the held bytes
        // and is parsed with them.
        g.feed(b"\x1b[?2026h\x1bc\x1b[?2026l", &mut s, &mut t, now);
        assert!(!g.holding());
        assert_eq!(row(&t, 0), "");
    }
}
