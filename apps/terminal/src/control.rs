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

struct Registry {
    /// Pane id → (info, when it was last published).
    panes: BTreeMap<u64, (PaneInfo, Instant)>,
    queue: Vec<(Request, Reply)>,
    enabled: bool,
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry { panes: BTreeMap::new(), queue: Vec::new(), enabled: false });
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

/// The requests for panes `owns` claims, taken off the queue.
pub fn take_requests(owns: impl Fn(u64) -> bool) -> Vec<(Request, Reply)> {
    let mut registry = lock();
    if registry.queue.is_empty() {
        return Vec::new();
    }
    let (mine, rest): (Vec<_>, Vec<_>) = registry.queue.drain(..).partition(|(req, _)| owns(req.pane()));
    registry.queue = rest;
    mine
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

/// Answer one request line. `list` is answered here; `read` and `prompt`
/// wait for the UI thread (`take_requests`).
pub fn answer(line: &str) -> Value {
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
    lock().queue.push((queued, tx));
    makepad_widgets::makepad_platform::thread::SignalToUI::set_ui_signal();
    rx.recv_timeout(ANSWER_TIMEOUT).unwrap_or_else(|_| err("the terminal did not answer"))
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

#[cfg(test)]
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

    #[test]
    fn pane_ids_name_this_process() {
        assert_eq!(parse_pane_id(&pane_id(3)), Some(3));
        assert_eq!(parse_pane_id("3"), Some(3));
        assert_eq!(parse_pane_id("1.3"), None);
        assert_eq!(parse_pane_id("x"), None);
    }
}
