//! A terminal session: PTY + emulator + reader thread, glued to the Makepad
//! UI thread via SignalToUI. All emulation runs on the UI thread; the reader
//! thread only moves bytes.

use std::io;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use makepad_widgets::makepad_platform::thread::SignalToUI;

use crate::pty::{Pty, PtyWriter};
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
        let writer = pty.writer_clone();
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
        let mut changed = false;
        let mut backlog = false;
        loop {
            match self.rx.try_recv() {
                Ok(bytes) => {
                    self.stream.process(&bytes, &mut self.terminal);
                    changed = true;
                    if Instant::now() >= deadline {
                        backlog = true;
                        break;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if !self.exited && self.pty.child_exited() {
                        self.exited = true;
                        changed = true;
                    }
                    break;
                }
            }
        }
        // Terminal-generated replies go straight back to the shell.
        let outbound = self.terminal.take_outbound();
        if !outbound.is_empty() {
            let _ = self.writer.send(outbound);
        }
        // Unfinished work must wake the UI again by itself: the reader
        // thread only signals on a fresh read, and with a full backlog it
        // is blocked on `send`, so nothing else would.
        if backlog {
            SignalToUI::set_ui_signal();
        }
        changed
    }

    pub fn take_events(&mut self) -> Vec<TermEvent> {
        self.terminal.take_events()
    }

    /// Write input bytes (key encodings, paste) to the shell.
    pub fn write(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let _ = self.writer.send(bytes.to_vec());
    }

    /// Whether the full input was accepted by this live PTY, for acknowledged
    /// file drops. Does not infer that the application consumed the input.
    pub fn try_write(&mut self, bytes: &[u8]) -> bool {
        !self.exited && !bytes.is_empty() && self.writer.send(bytes.to_vec()).is_ok()
    }

    pub fn resize(&mut self, cols: usize, rows: usize) {
        let cols = cols.max(2);
        let rows = rows.max(2);
        if cols == self.terminal.cols() && rows == self.terminal.rows() {
            return;
        }
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
