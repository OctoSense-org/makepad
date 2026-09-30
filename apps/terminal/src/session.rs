//! A terminal session: PTY + emulator + reader thread, glued to the Makepad
//! UI thread via SignalToUI. All emulation runs on the UI thread; the reader
//! and writer threads only move bytes. Nothing here blocks the UI thread on
//! the PTY: input is queued for the writer thread (see [`PtyWriter`]), which
//! feeds it to the program as fast as the program reads it.

use std::io;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use makepad_widgets::makepad_platform::thread::SignalToUI;
use makepad_widgets::Cx;

use crate::pty::{InputRejected, Pty, PtyWriter};
use crate::sync_output::SyncGate;
use crate::term::stream::Stream;
use crate::term::terminal::{TermEvent, Terminal};

/// Wall-clock budget for one [`Session::drain`] pass.
///
/// A flooding shell (`yes x`) writes far faster than the emulator can
/// consume, so an unbounded drain never reaches an empty channel: it
/// starves the event loop for as long as the flood lasts — no frame, no
/// timer, and in an wm-hosted child not even the host-is-gone check, so
/// the tile stays black and the orphan keeps burning a core after its host
/// dies. Bounded, every pass hands the loop back in time to paint, and the
/// rest of the backlog is picked up on the next one.
const DRAIN_BUDGET: Duration = Duration::from_millis(4);

/// Read-thread backlog, in chunks of up to 64 KiB. A bounded channel is
/// what makes the budget safe: when the emulator falls behind, the reader
/// blocks on `send`, the PTY buffer fills and the shell is throttled by
/// its own `write` — instead of the queue growing without limit.
const BACKLOG_CHUNKS: usize = 32;

pub struct Session {
    pub terminal: Terminal,
    stream: Stream,
    pty: Pty,
    writer: PtyWriter,
    rx: Receiver<Vec<u8>>,
    /// Holds a synchronized-output frame (mode 2026) until it is complete.
    sync: SyncGate,
    pub exited: bool,
}

/// How a session's shell starts. The default is the historical behaviour:
/// `$SHELL -l`, `TERM=xterm-256color`, the emulator's default scrollback.
#[derive(Clone, Debug, PartialEq)]
pub struct SpawnOptions {
    /// `None`: `$SHELL`, else `/bin/zsh`.
    pub shell: Option<String>,
    pub login: bool,
    pub term: String,
    pub scrollback: usize,
}

impl Default for SpawnOptions {
    fn default() -> Self {
        SpawnOptions {
            shell: None,
            login: true,
            term: "xterm-256color".into(),
            scrollback: crate::term::terminal::DEFAULT_SCROLLBACK,
        }
    }
}

impl SpawnOptions {
    /// The options the person's settings ask for.
    pub fn from_settings(settings: &crate::settings::Settings) -> Self {
        SpawnOptions {
            shell: (!settings.shell.trim().is_empty()).then(|| settings.shell.trim().to_owned()),
            login: settings.login_shell,
            term: settings.term.clone(),
            scrollback: settings.scrollback_lines,
        }
    }
}

impl Session {
    /// PID of this session’s shell, for exact host activity relationships.
    pub fn child_pid(&self) -> i32 { self.pty.child_pid() }

    /// The name of the job the shell is running in the foreground, `None`
    /// while the shell sits at its prompt (or when the platform can't tell).
    pub fn foreground_job(&self) -> Option<String> {
        let pgrp = self.pty.foreground_pgrp()?;
        if pgrp == self.child_pid() || self.exited {
            return None;
        }
        crate::procinfo::name(pgrp)
    }

    /// The name of whatever holds the foreground: the job, else the shell.
    pub fn foreground_name(&self) -> Option<String> {
        let pgrp = self.pty.foreground_pgrp().unwrap_or_else(|| self.child_pid());
        crate::procinfo::name(pgrp).or_else(|| crate::procinfo::name(self.child_pid()))
    }

    /// The shell's working directory, read from the process table: shells
    /// that never report it with OSC 7 still get a new tab opened there.
    pub fn shell_cwd(&self) -> Option<std::path::PathBuf> {
        crate::procinfo::cwd(self.child_pid())
    }

    pub fn spawn(
        cols: usize,
        rows: usize,
        cwd: Option<&Path>,
        shell: Option<&str>,
        command: Option<&str>,
    ) -> io::Result<Session> {
        let options = SpawnOptions { shell: shell.map(str::to_owned), ..SpawnOptions::default() };
        Self::spawn_with(cols, rows, cwd, command, &options)
    }

    /// Spawn with the person's terminal settings: shell, login flag, `TERM`
    /// and scrollback (see `crate::settings`).
    pub fn spawn_with(
        cols: usize,
        rows: usize,
        cwd: Option<&Path>,
        command: Option<&str>,
        options: &SpawnOptions,
    ) -> io::Result<Session> {
        let mut pty = Pty::spawn_opts(
            cols.max(2) as u16,
            rows.max(2) as u16,
            options.shell.as_deref(),
            options.login,
            command,
            &[("TERM", options.term.as_str())],
            cwd,
        )?;
        let writer = pty.start_writer()?;
        let mut reader = pty.take_reader();
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(BACKLOG_CHUNKS);
        std::thread::Builder::new()
            .name("terminal-pty-read".into())
            .spawn(move || {
                while let Some(bytes) = reader.read() {
                    if tx.send(bytes).is_err() {
                        return;
                    }
                    SignalToUI::set_ui_signal();
                }
                // EOF: closing the channel is the exit notification.
                drop(tx);
                SignalToUI::set_ui_signal();
            })
            .ok();
        Ok(Session {
            terminal: Terminal::with_scrollback(cols.max(2), rows.max(2), options.scrollback),
            stream: Stream::new(),
            pty,
            writer,
            rx,
            sync: SyncGate::default(),
            exited: false,
        })
    }

    /// Drain pending PTY output into the emulator. Call on Event::Signal
    /// (and once per frame while visible). Returns true when anything
    /// changed and a redraw is needed.
    ///
    /// At most [`DRAIN_BUDGET`] of parsing per call: a flood is rendered at
    /// the frame cadence rather than swallowing the event loop whole. When
    /// the budget cuts a pass short the UI signal is re-armed, so the next
    /// tick continues where this one stopped.
    pub fn drain(&mut self) -> bool {
        let deadline = Instant::now() + DRAIN_BUDGET;
        let now = Cx::monotonic_now();
        // A frame held past its timeout is shown as it is.
        let mut changed = self.sync.expire(&mut self.stream, &mut self.terminal, now);
        let mut backlog = false;
        loop {
            match self.rx.try_recv() {
                Ok(bytes) => {
                    // Bytes of an open synchronized frame are held, not
                    // parsed: nothing changes on screen until it ends.
                    changed |= self
                        .sync
                        .feed(&bytes, &mut self.stream, &mut self.terminal, now);
                    if Instant::now() >= deadline {
                        backlog = true;
                        break;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // The program is gone: show whatever it left.
                    changed |= self.sync.release(&mut self.stream, &mut self.terminal);
                    if !self.exited && self.pty.child_exited() {
                        self.exited = true;
                        changed = true;
                    }
                    break;
                }
            }
        }
        // Terminal-generated replies go back to the shell, queued behind
        // any input already on its way.
        let outbound = self.terminal.take_outbound();
        if !outbound.is_empty() {
            let _ = self.writer.send_reply(outbound);
        }
        // Unfinished work must wake the UI again by itself: the reader
        // thread only signals on a fresh read, and with a full backlog it
        // is blocked on `send`, so nothing else would.
        if backlog {
            SignalToUI::set_ui_signal();
        }
        changed
    }

    /// When a held synchronized frame will be shown even if it never ends;
    /// the UI must drain again then.
    pub fn sync_deadline(&self) -> Option<f64> {
        self.sync.deadline()
    }

    pub fn take_events(&mut self) -> Vec<TermEvent> {
        self.terminal.take_events()
    }

    /// Queue typed input (key encodings, mouse and focus reports) for the
    /// shell. Never blocks, and never refused while the PTY lives.
    pub fn write(&mut self, bytes: &[u8]) {
        let _ = self.writer.send(bytes.to_vec());
    }

    /// Queue a paste (or a file drop) for the shell, whole or not at all.
    /// Never blocks: `Ok` means the live PTY's writer accepted every byte,
    /// not that the program has read them yet. Refused while the program
    /// leaves [`crate::pty::PASTE_QUEUE_LIMIT`] bytes or more unread.
    pub fn write_paste(&mut self, bytes: &[u8]) -> Result<(), InputRejected> {
        if self.exited || bytes.is_empty() {
            return Err(InputRejected::Closed);
        }
        self.writer.send_paste(bytes.to_vec())
    }

    /// Input bytes queued but not yet read by the program.
    pub fn pending_input(&self) -> usize {
        self.writer.pending()
    }

    pub fn resize(&mut self, cols: usize, rows: usize) {
        let cols = cols.max(2);
        let rows = rows.max(2);
        if cols == self.terminal.cols() && rows == self.terminal.rows() {
            return;
        }
        // A held frame was drawn for the old size and the program redraws
        // for the new one: show it now rather than hold across the reflow.
        self.sync.release(&mut self.stream, &mut self.terminal);
        self.terminal.resize(cols, rows);
        let _ = self.pty.resize(cols as u16, rows as u16);
    }
}

#[cfg(test)]
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod tests {
    use super::*;

    /// Dropping a session whose job is sitting idle must RETURN.
    ///
    /// The reader thread parks in `read()` on the pty master, and `close()`
    /// on the last descriptor of a master with a reader parked on it blocks
    /// in the kernel until that reader leaves — which it only does once the
    /// slave side is gone. Closing before killing the job was therefore a
    /// deadlock between the UI thread and its own reader: a Quick-Look
    /// panel that retargeted at the next file (`MpTerm::restart_with` drops
    /// the session) froze the whole child, and the panel kept showing the
    /// previous file's last frame forever.
    #[test]
    fn dropping_an_idle_session_does_not_block() {
        let mut session =
            Session::spawn(80, 24, None, None, Some("sleep 30")).expect("a pty session");
        // Give the job time to start and the reader thread time to park.
        std::thread::sleep(Duration::from_millis(300));
        session.drain();

        // Drop on another thread so a regression FAILS instead of hanging
        // the test binary.
        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            drop(session);
            let _ = done_tx.send(());
        });
        assert!(
            done_rx.recv_timeout(Duration::from_secs(10)).is_ok(),
            "Session::drop blocked with the reader thread parked in read()"
        );
    }

    /// Wait (draining) until `want` holds, up to `secs`.
    #[allow(clippy::disallowed_methods, clippy::disallowed_types)]
    fn drain_until(session: &mut Session, secs: u64, want: impl Fn(&Session) -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(secs);
        while Instant::now() < deadline {
            session.drain();
            if want(session) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }

    fn screen_text(session: &Session) -> String {
        let screen = session.terminal.screen();
        (0..screen.rows)
            .map(|y| screen.row(y).text())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// A program in raw mode that is not reading leaves the tty's input
    /// queue full: a write to the master then waits until it reads. Input
    /// used to be written synchronously on the UI thread (and the reader
    /// had switched the shared master to blocking), so a large paste froze
    /// the whole window — every tab and pane — for as long as the program
    /// slept. Now every input call returns at once, the bytes wait in the
    /// session's writer queue, and they reach the program, all of them and
    /// in order (paste, keys, paste), once it reads.
    #[test]
    #[allow(clippy::disallowed_methods, clippy::disallowed_types)]
    fn input_to_a_program_that_is_not_reading_never_blocks() {
        let dir = std::env::temp_dir().join(format!("mpterm-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("received");
        let _ = std::fs::remove_file(&out);

        let paste: Vec<u8> = (0..(1usize << 20) / 8)
            .flat_map(|i| format!("{:07}\n", i % 10_000_000).into_bytes())
            .collect();
        let keys = b"typed-after-the-paste";
        let tail = b"and-a-second-paste";
        let total = paste.len() + keys.len() + tail.len();
        // Raw mode, say so, then don't read for a while; then read exactly
        // what we will send.
        let command = format!(
            "stty raw -echo; printf READY; sleep 3; head -c {total} > '{}'; printf DONE; sleep 30",
            out.display()
        );
        let mut session =
            Session::spawn(80, 24, None, None, Some(&command)).expect("a pty session");
        assert!(
            drain_until(&mut session, 10, |s| screen_text(s).contains("READY")),
            "raw mode set"
        );

        // Do the writes on another thread so a regression FAILS here
        // instead of hanging the test binary.
        let (tx, rx) = mpsc::channel();
        let (paste_c, keys_c, tail_c) = (paste.clone(), keys.to_vec(), tail.to_vec());
        std::thread::spawn(move || {
            let t0 = Instant::now();
            let first = session.write_paste(&paste_c);
            session.write(&keys_c);
            let second = session.write_paste(&tail_c);
            let _ = tx.send((t0.elapsed(), first, second, session));
        });
        let (took, first, second, mut session) = rx
            .recv_timeout(Duration::from_secs(2))
            .expect("writing a 1 MiB paste blocked the caller");
        assert!(
            took < Duration::from_millis(200),
            "input calls took {took:?}"
        );
        assert_eq!((first, second), (Ok(()), Ok(())));
        assert!(session.pending_input() > 0, "the program has not read yet");

        // Once the program reads, everything arrives, in order.
        assert!(
            drain_until(&mut session, 30, |s| screen_text(s).contains("DONE")),
            "the program got all its input: {} bytes still queued",
            session.pending_input()
        );
        assert_eq!(session.pending_input(), 0);
        let received = std::fs::read(&out).unwrap();
        let mut expected = paste;
        expected.extend_from_slice(keys);
        expected.extend_from_slice(tail);
        assert!(
            received == expected,
            "{} bytes received, {} sent, in order",
            received.len(),
            expected.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With the program not reading, pastes are bounded — a refused one is
    /// refused whole and reported — while keys are still accepted; and a
    /// session with input stuck in its queue closes at once.
    #[test]
    #[allow(clippy::disallowed_methods, clippy::disallowed_types)]
    fn a_full_input_queue_refuses_pastes_not_keys_and_closes_promptly() {
        let mut session = Session::spawn(
            80,
            24,
            None,
            None,
            Some("stty raw -echo; printf READY; sleep 30"),
        )
        .expect("a pty session");
        assert!(
            drain_until(&mut session, 10, |s| screen_text(s).contains("READY")),
            "raw mode set"
        );

        let big = vec![b'x'; crate::pty::PASTE_QUEUE_LIMIT];
        assert_eq!(
            session.write_paste(&big),
            Ok(()),
            "one paste of any size is accepted"
        );
        match session.write_paste(b"more") {
            Err(InputRejected::QueueFull { pending }) => assert!(pending > 0),
            other => panic!("a paste over the limit must be refused, got {other:?}"),
        }
        let before = session.pending_input();
        session.write(b"k");
        assert_eq!(
            session.pending_input(),
            before + 1,
            "a key is never refused"
        );

        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            drop(session);
            let _ = done_tx.send(());
        });
        assert!(
            done_rx.recv_timeout(Duration::from_secs(5)).is_ok(),
            "closing a session with unread input queued blocked"
        );
    }

    /// Mode 2026 through a real PTY: a frame drawn in pieces with pauses
    /// is never visible half done; the end shows it whole. A frame whose
    /// end never comes is shown after the timeout.
    #[test]
    #[allow(clippy::disallowed_methods, clippy::disallowed_types)]
    fn synchronized_frames_are_shown_whole() {
        let command = "printf 'OLD'; sleep 0.5; \
             printf '\\033[?2026h\\033[H\\033[2JNEW-top'; sleep 0.6; \
             printf ' NEW-bottom\\033[?2026l'; sleep 0.6; \
             printf '\\033[?2026h STUCK'; sleep 30";
        let options = SpawnOptions {
            shell: Some("/bin/sh".into()),
            login: false,
            ..SpawnOptions::default()
        };
        let mut session =
            Session::spawn_with(40, 4, None, Some(command), &options).expect("a pty session");
        assert!(
            drain_until(&mut session, 10, |s| screen_text(s).contains("OLD")),
            "first frame"
        );
        // While the frame is open the screen keeps the old one.
        assert!(
            drain_until(&mut session, 5, |s| s.sync_deadline().is_some()),
            "frame opened"
        );
        let t = Instant::now();
        while t.elapsed() < Duration::from_millis(300) {
            session.drain();
            let text = screen_text(&session);
            assert!(
                text.contains("OLD") && !text.contains("NEW"),
                "half frame shown: {text:?}"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            drain_until(&mut session, 5, |s| screen_text(s)
                .contains("NEW-top NEW-bottom")),
            "whole frame at its end"
        );
        // An unterminated frame: held, then shown once its timeout passes.
        assert!(
            drain_until(&mut session, 5, |s| s.sync_deadline().is_some()),
            "stuck frame opened"
        );
        let opened = Instant::now();
        assert!(!screen_text(&session).contains("STUCK"));
        assert!(
            drain_until(&mut session, 5, |s| screen_text(s).contains("STUCK")),
            "released"
        );
        assert!(
            opened.elapsed()
                >= Duration::from_secs_f64(crate::sync_output::SYNC_TIMEOUT)
                    - Duration::from_millis(100)
        );
        assert_eq!(session.sync_deadline(), None);
    }

    /// A tab asks its session what runs in it (confirm-before-close, the
    /// program title) and where the shell is (a new tab opens there).
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn a_session_reports_its_foreground_job_and_cwd() {
        let dir = std::env::temp_dir().canonicalize().unwrap();
        let options = SpawnOptions { shell: Some("/bin/sh".into()), login: false, ..SpawnOptions::default() };
        let mut session = Session::spawn_with(80, 24, Some(&dir), None, &options).expect("a pty session");
        let wait_for = |session: &mut Session, want: &dyn Fn(&Session) -> bool| {
            for _ in 0..100 {
                session.drain();
                if want(session) {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            false
        };
        assert!(wait_for(&mut session, &|s| s.foreground_name().is_some()), "the shell is named");
        assert_eq!(session.foreground_job(), None, "an idle shell runs no job");
        assert_eq!(session.shell_cwd().map(|p| p.canonicalize().unwrap()), Some(dir));

        session.write(b"sleep 30\n");
        assert!(
            wait_for(&mut session, &|s| s.foreground_job().as_deref() == Some("sleep")),
            "the running job is reported: {:?}",
            session.foreground_job()
        );
    }
}
