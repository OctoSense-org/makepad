//! The control socket: lets a program of the same user list the terminal's
//! panes, read their screens and type prompts into them — what OctoLoop's
//! outer loop does with herdr (`herdr agent list` / `agent prompt`) — so
//! agents in split panes of one window can be driven from outside.
//! `terminal-ctl` is its client.
//!
//! Off unless the settings allow it (`external-control = true`). Each
//! terminal process listens on `<makepad home>/terminal/control/<pid>.sock`
//! (directory 0700). One JSON request per line, one JSON reply per line:
//!
//! ```text
//! {"cmd":"list"}
//! {"cmd":"read","pane":"4242.3","lines":40}
//! {"cmd":"prompt","pane":"4242.3","text":"run the tests","submit":true,"force":false}
//! ```
//!
//! `prompt` pastes the text (bracketed when the program asked for that) and
//! presses Enter. It refuses while an approval or question is on screen,
//! and a lone approval key (`y`, `n`, …) always: answering a dialog is the
//! person's call. `force` overrides the first, never the second.

use makepad_strict_json::{self as json, obj, s, Value};
use makepad_widgets::error;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A pane as `list` reports it.
#[derive(Clone, Debug, PartialEq)]
pub struct PaneInfo {
    pub pane: u64,
    pub tab: usize,
    pub title: String,
    pub cwd: String,
    pub program: String,
    pub agent: Option<&'static str>,
    pub state: &'static str,
    pub focused: bool,
}

impl PaneInfo {
    fn to_json(&self) -> Value {
        obj(vec![
            ("id", s(pane_id(self.pane))),
            ("tab", Value::Int(self.tab as i64 + 1)),
            ("title", s(&self.title)),
            ("cwd", s(&self.cwd)),
            ("program", s(&self.program)),
            ("agent", self.agent.map_or(Value::Null, s)),
            ("state", s(self.state)),
            ("focused", Value::Bool(self.focused)),
        ])
    }
}

/// A request only the UI thread can answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    Read { pane: u64, lines: usize },
    Prompt { pane: u64, text: String, submit: bool, force: bool },
}

impl Request {
    pub fn pane(&self) -> u64 {
        match self {
            Request::Read { pane, .. } | Request::Prompt { pane, .. } => *pane,
        }
    }
}

pub type Reply = mpsc::Sender<Value>;

/// A request waiting for the UI thread. Leaving the queue is its claim,
/// made exactly once under the registry lock: either the UI thread takes
/// it (and runs it) or its asker's timeout cancels it (and it never runs).
struct Queued {
    id: u64,
    request: Request,
    reply: Reply,
}

struct Registry {
    /// Pane id → (info, when it was last published).
    panes: BTreeMap<u64, (PaneInfo, Instant)>,
    queue: Vec<Queued>,
    next_id: u64,
    enabled: bool,
}

static REGISTRY: Mutex<Registry> =
    Mutex::new(Registry { panes: BTreeMap::new(), queue: Vec::new(), next_id: 0, enabled: false });
static STARTED: std::sync::Once = std::sync::Once::new();

/// A pane's id outside the process: `<pid>.<pane>`.
pub fn pane_id(pane: u64) -> String {
    format!("{}.{}", std::process::id(), pane)
}

/// `<pid>.<pane>` (or a bare pane number) → the pane, if it is this
/// process's.
pub fn parse_pane_id(id: &str) -> Option<u64> {
    match id.split_once('.') {
        Some((pid, pane)) => (pid.parse::<u32>().ok()? == std::process::id()).then(|| pane.parse().ok()).flatten(),
        None => id.parse().ok(),
    }
}

pub fn control_dir(home: &std::path::Path) -> PathBuf {
    home.join("terminal").join("control")
}

/// Turn the socket on or off (from the settings). The listener starts the
/// first time it is turned on and then answers only while it is on.
pub fn set_enabled(enabled: bool) {
    lock().enabled = enabled;
    if enabled {
        STARTED.call_once(start_listener);
    }
}

fn lock() -> std::sync::MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Replace what `panes` says about these panes (a terminal widget
/// publishes its own after each poll).
pub fn publish(panes: Vec<PaneInfo>) {
    let now = Instant::now();
    let mut registry = lock();
    for info in panes {
        registry.panes.insert(info.pane, (info, now));
    }
}

pub fn forget(pane: u64) {
    lock().panes.remove(&pane);
}

/// The requests for panes `owns` claims, taken off the queue. Taking one
/// commits to running it: its asker waits for the reply from then on. A
/// request whose asker timed out is no longer queued (`cancel`).
pub fn take_requests(owns: impl Fn(u64) -> bool) -> Vec<(Request, Reply)> {
    let mut registry = lock();
    if registry.queue.is_empty() {
        return Vec::new();
    }
    let (mine, rest): (Vec<_>, Vec<_>) = registry.queue.drain(..).partition(|queued| owns(queued.request.pane()));
    registry.queue = rest;
    mine.into_iter().map(|queued| (queued.request, queued.reply)).collect()
}

/// Take request `id` back off the queue so it never runs. False when the
/// UI thread already took it: then it runs, and replies.
fn cancel(id: u64) -> bool {
    let mut registry = lock();
    let before = registry.queue.len();
    registry.queue.retain(|queued| queued.id != id);
    registry.queue.len() != before
}

pub fn ok(fields: Vec<(&str, Value)>) -> Value {
    let mut all = vec![("ok", Value::Bool(true))];
    all.extend(fields);
    obj(all)
}

pub fn err(message: impl Into<String>) -> Value {
    obj(vec![("ok", Value::Bool(false)), ("error", s(message))])
}

/// Panes not published for this long belong to a closed window.
const STALE: Duration = Duration::from_secs(5);
/// How long a request waits for the UI thread.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(5);
const NOT_ANSWERED: &str = "the terminal did not answer";

/// Answer one request line. `list` is answered here; `read` and `prompt`
/// wait for the UI thread (`take_requests`).
pub fn answer(line: &str) -> Value {
    answer_within(line, ANSWER_TIMEOUT)
}

fn answer_within(line: &str, timeout: Duration) -> Value {
    let Ok(request) = json::parse(line.as_bytes()) else {
        return err("the request is not JSON");
    };
    if !lock().enabled {
        return err("external control is off (settings: external-control)");
    }
    let pane = || {
        request.get("pane").and_then(Value::as_str).and_then(parse_pane_id).ok_or("no such pane in this terminal")
    };
    let queued = match request.get("cmd").and_then(Value::as_str) {
        Some("list") => {
            let mut registry = lock();
            registry.panes.retain(|_, (_, at)| at.elapsed() < STALE);
            let panes = registry.panes.values().map(|(info, _)| info.to_json()).collect();
            return ok(vec![("pid", Value::Int(std::process::id() as i64)), ("panes", Value::Arr(panes))]);
        }
        Some("read") => match pane() {
            Ok(pane) => {
                let lines = request.get("lines").and_then(Value::as_u64).unwrap_or(0).min(10_000) as usize;
                Request::Read { pane, lines }
            }
            Err(e) => return err(e),
        },
        Some("prompt") => match pane() {
            Ok(pane) => Request::Prompt {
                pane,
                text: request.get("text").and_then(Value::as_str).unwrap_or("").to_owned(),
                submit: request.get("submit").and_then(Value::as_bool).unwrap_or(true),
                force: request.get("force").and_then(Value::as_bool).unwrap_or(false),
            },
            Err(e) => return err(e),
        },
        _ => return err("unknown cmd (list, read, prompt)"),
    };
    if !lock().panes.contains_key(&queued.pane()) {
        return err("no such pane in this terminal");
    }
    let (tx, rx) = mpsc::channel();
    let id = {
        let mut registry = lock();
        registry.next_id += 1;
        let id = registry.next_id;
        registry.queue.push(Queued { id, request: queued, reply: tx });
        id
    };
    makepad_widgets::makepad_platform::thread::SignalToUI::set_ui_signal();
    if let Ok(reply) = rx.recv_timeout(timeout) {
        return reply;
    }
    // Still queued: the UI thread never saw it, and now never will.
    if cancel(id) {
        return err(NOT_ANSWERED);
    }
    // The UI thread took it first: it runs, so report how it went. (The UI
    // thread replies or drops the sender as soon as it has handled it.)
    rx.recv().unwrap_or_else(|_| err("the terminal took the request but did not answer"))
}

#[cfg(unix)]
fn start_listener() {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixListener;

    let dir = control_dir(&makepad_widgets::makepad_platform::home::makepad_home());
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
    let pid = std::process::id();
    let mut path = dir.join(format!("{pid}.sock"));
    // A Unix socket path is limited to about 100 bytes: under a long home
    // the socket lives in a private directory in /tmp, and `<pid>.path` in
    // the control directory points there (terminal-ctl follows it).
    if path.as_os_str().len() >= 100 {
        let uid = std::fs::metadata(&dir).map(|m| std::os::unix::fs::MetadataExt::uid(&m)).unwrap_or(0);
        let short = PathBuf::from(format!("/tmp/makepad-terminal-{uid}"));
        if std::fs::create_dir_all(&short).is_err() {
            return;
        }
        let _ = std::fs::set_permissions(&short, std::fs::Permissions::from_mode(0o700));
        path = short.join(format!("{pid}.sock"));
        if std::fs::write(dir.join(format!("{pid}.path")), path.to_string_lossy().as_bytes()).is_err() {
            return;
        }
    }
    let _ = std::fs::remove_file(&path);
    let Ok(listener) = UnixListener::bind(&path) else {
        error!("terminal: the control socket could not listen at {}", path.display());
        return;
    };
    let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    let _ = std::thread::Builder::new().name("terminal-control".into()).spawn(move || {
        for stream in listener.incoming().flatten() {
            let _ = std::thread::Builder::new().name("terminal-control-conn".into()).spawn(move || {
                let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
                let mut writer = match stream.try_clone() {
                    Ok(w) => w,
                    Err(_) => return,
                };
                let reader = BufReader::new(stream);
                for line in reader.lines() {
                    let Ok(line) = line else {
                        break;
                    };
                    if line.len() > 1 << 20 {
                        break;
                    }
                    let reply = answer(&line).to_json();
                    if writer.write_all(format!("{reply}\n").as_bytes()).is_err() {
                        break;
                    }
                }
            });
        }
    });
}

#[cfg(not(unix))]
fn start_listener() {}

/// Tests that switch the socket on or off hold this: the switch is global.
#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

/// The switch without the listener (tests).
#[cfg(test)]
pub(crate) fn set_enabled_for_tests(on: bool) {
    lock().enabled = on;
}

// Host-only tests: std clocks are fine here (the lints guard wasm).
#[cfg(test)]
#[allow(clippy::disallowed_types, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn requests_are_checked_before_they_queue() {
        let _serial = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_enabled_for_test(false);
        assert_eq!(answer(r#"{"cmd":"list"}"#).get("ok"), Some(&Value::Bool(false)), "off by default");
        set_enabled_for_test(true);
        assert_eq!(answer("not json").get("ok"), Some(&Value::Bool(false)));
        assert_eq!(answer(r#"{"cmd":"dance"}"#).get("ok"), Some(&Value::Bool(false)));
        let unknown = answer(r#"{"cmd":"read","pane":"1.999999"}"#);
        assert_eq!(unknown.get("ok"), Some(&Value::Bool(false)), "another process's pane");

        publish(vec![PaneInfo {
            pane: 7,
            tab: 0,
            title: "claude".into(),
            cwd: "/tmp".into(),
            program: "claude".into(),
            agent: Some("claude"),
            state: "idle",
            focused: true,
        }]);
        let list = answer(r#"{"cmd":"list"}"#);
        let panes = list.get("panes").and_then(Value::as_arr).unwrap();
        let mine = panes.iter().find(|p| p.get("id").and_then(Value::as_str) == Some(&pane_id(7))).unwrap();
        assert_eq!(mine.get("agent").and_then(Value::as_str), Some("claude"));
        assert_eq!(mine.get("tab").and_then(Value::as_i64), Some(1));

        // A read waits for the UI thread; answer it from here.
        let asker = std::thread::spawn(|| answer(&format!(r#"{{"cmd":"read","pane":"{}","lines":5}}"#, pane_id(7))));
        let mut served = false;
        for _ in 0..200 {
            let taken = take_requests(|pane| pane == 7);
            if let Some((req, reply)) = taken.into_iter().next() {
                assert_eq!(req, Request::Read { pane: 7, lines: 5 });
                reply.send(ok(vec![("text", s("$ "))])).unwrap();
                served = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(served);
        assert_eq!(asker.join().unwrap().get("text").and_then(Value::as_str), Some("$ "));
        forget(7);
    }

    fn set_enabled_for_test(on: bool) {
        lock().enabled = on;
    }

    fn publish_pane(pane: u64) {
        publish(vec![PaneInfo {
            pane,
            tab: 0,
            title: "sh".into(),
            cwd: "/tmp".into(),
            program: "sh".into(),
            agent: None,
            state: "idle",
            focused: true,
        }]);
    }

    fn prompt_line(pane: u64, text: &str) -> String {
        format!(r#"{{"cmd":"prompt","pane":"{}","text":"{text}","submit":true}}"#, pane_id(pane))
    }

    /// The UI thread was busy past the timeout: the asker got an error, so
    /// the prompt must not be typed when the UI thread wakes up.
    #[test]
    fn a_timed_out_prompt_never_runs_later() {
        let _serial = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_enabled_for_test(true);
        publish_pane(9);
        let started = Instant::now();
        let reply = answer_within(&prompt_line(9, "rm -rf build"), Duration::from_millis(100));
        assert!(started.elapsed() >= Duration::from_millis(100));
        assert_eq!(reply.get("ok"), Some(&Value::Bool(false)));
        assert_eq!(reply.get("error").and_then(Value::as_str), Some(NOT_ANSWERED));
        let late = take_requests(|pane| pane == 9);
        assert!(late.is_empty(), "a timed-out prompt is still queued: {:?}", late.iter().map(|(r, _)| r).collect::<Vec<_>>());
        forget(9);
    }

    /// Normal path for a prompt: taken in time, the asker gets the reply.
    #[test]
    fn a_prompt_answered_in_time_gets_its_reply() {
        let _serial = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_enabled_for_test(true);
        publish_pane(10);
        let asker = std::thread::spawn(|| answer(&prompt_line(10, "make test")));
        let mut taken = Vec::new();
        for _ in 0..400 {
            taken = take_requests(|pane| pane == 10);
            if !taken.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(taken.len(), 1);
        let (request, reply) = taken.pop().unwrap();
        assert_eq!(request, Request::Prompt { pane: 10, text: "make test".into(), submit: true, force: false });
        reply.send(ok(vec![])).unwrap();
        assert_eq!(asker.join().unwrap().get("ok"), Some(&Value::Bool(true)));
        assert!(take_requests(|pane| pane == 10).is_empty(), "taken exactly once");
        forget(10);
    }

    /// Taken just before the timeout, answered after it: the request ran,
    /// so the asker gets the real reply, not a timeout.
    #[test]
    fn a_request_taken_before_the_timeout_reports_its_real_reply() {
        let _serial = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_enabled_for_test(true);
        publish_pane(11);
        let asker = std::thread::spawn(|| answer_within(&prompt_line(11, "ls"), Duration::from_millis(200)));
        let mut taken = Vec::new();
        while taken.is_empty() {
            taken = take_requests(|pane| pane == 11);
            std::thread::sleep(Duration::from_millis(1));
        }
        std::thread::sleep(Duration::from_millis(400));
        taken.pop().unwrap().1.send(ok(vec![("typed", Value::Bool(true))])).unwrap();
        let reply = asker.join().unwrap();
        assert_eq!(reply.get("typed"), Some(&Value::Bool(true)), "{}", reply.to_json());
        forget(11);
    }

    /// The timeout and the UI thread race for the same entry: exactly one
    /// wins. Either the request ran and the asker got its reply, or it
    /// never ran and the asker got the timeout.
    #[test]
    fn timeout_and_take_never_both_win() {
        let _serial = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_enabled_for_test(true);
        publish_pane(13);
        let (mut ran, mut timed_out) = (0, 0);
        for round in 0..300u64 {
            let asker = std::thread::spawn(|| answer_within(&prompt_line(13, "echo"), Duration::from_millis(2)));
            // One take, somewhere around the 2 ms deadline.
            std::thread::sleep(Duration::from_micros(1000 + (round * 37) % 2500));
            let mut taken = take_requests(|pane| pane == 13);
            // Past the asker's wait, nothing is left to take.
            let reply = if taken.is_empty() {
                let reply = asker.join().unwrap();
                assert!(take_requests(|pane| pane == 13).is_empty(), "round {round}: queued after its timeout");
                reply
            } else {
                assert_eq!(taken.len(), 1);
                taken.pop().unwrap().1.send(ok(vec![("ran", Value::Bool(true))])).unwrap();
                asker.join().unwrap()
            };
            if reply.get("ran") == Some(&Value::Bool(true)) {
                ran += 1;
            } else {
                assert_eq!(reply.get("error").and_then(Value::as_str), Some(NOT_ANSWERED), "round {round}");
                timed_out += 1;
            }
        }
        forget(13);
        assert_eq!(ran + timed_out, 300);
    }

    #[test]
    fn pane_ids_name_this_process() {
        assert_eq!(parse_pane_id(&pane_id(3)), Some(3));
        assert_eq!(parse_pane_id("3"), Some(3));
        assert_eq!(parse_pane_id("1.3"), None);
        assert_eq!(parse_pane_id("x"), None);
    }
}
