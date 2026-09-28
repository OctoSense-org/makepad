//! terminal-ctl: drive the Makepad terminal's panes from outside — the
//! client of its control socket (`makepad_terminal::control`). Built for
//! OctoLoop's outer loop, which drives agents the same way with herdr
//! (`herdr agent list` / `herdr agent prompt`).
//!
//!   terminal-ctl list [--json]
//!   terminal-ctl read <pane> [--lines N]
//!   terminal-ctl prompt <pane> [--no-submit] [--force] <text…>   (`-` reads stdin)
//!
//! A pane is `<pid>.<pane>` as `list` prints it. Every running terminal
//! with "Allow terminal-ctl" on answers; sockets are looked for under
//! `$MAKEPAD_HOME`, `~/.makepad` and the OctoSense home (`$OCTOSENSE_HOME`,
//! `~/.octosense`), or `--home DIR`.
//!
//! Exit codes: 0 done, 1 refused or failed (the reason on stderr), 2 usage.

use makepad_strict_json::{self as json, obj, s, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage:
  terminal-ctl list [--json]
  terminal-ctl read <pane> [--lines N]
  terminal-ctl prompt <pane> [--no-submit] [--force] <text...>   ('-' reads stdin)
options: --home DIR  (where the terminal keeps its settings; repeatable)";

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut homes = Vec::new();
    while let Some(i) = args.iter().position(|a| a == "--home") {
        if i + 1 >= args.len() {
            return usage();
        }
        homes.push(PathBuf::from(args.remove(i + 1)));
        args.remove(i);
    }
    if homes.is_empty() {
        homes = default_homes();
    }
    let flag = |args: &mut Vec<String>, name: &str| {
        let found = args.iter().any(|a| a == name);
        args.retain(|a| a != name);
        found
    };
    let Some(cmd) = (!args.is_empty()).then(|| args.remove(0)) else {
        return usage();
    };
    match cmd.as_str() {
        "list" => {
            let as_json = flag(&mut args, "--json");
            list(&homes, as_json)
        }
        "read" => {
            let lines = match args.iter().position(|a| a == "--lines") {
                Some(i) if i + 1 < args.len() => {
                    let n = args.remove(i + 1);
                    args.remove(i);
                    match n.parse::<u64>() {
                        Ok(n) => n,
                        Err(_) => return usage(),
                    }
                }
                Some(_) => return usage(),
                None => 0,
            };
            let [pane] = args.as_slice() else {
                return usage();
            };
            let reply = send(&homes, pane, obj(vec![("cmd", s("read")), ("pane", s(pane.as_str())), ("lines", Value::Int(lines as i64))]));
            match reply {
                Ok(v) if ok(&v) => {
                    println!("{}", v.get("text").and_then(Value::as_str).unwrap_or(""));
                    ExitCode::SUCCESS
                }
                Ok(v) => fail(&v),
                Err(e) => fail_msg(&e),
            }
        }
        "prompt" => {
            let no_submit = flag(&mut args, "--no-submit");
            let force = flag(&mut args, "--force");
            if args.len() < 2 {
                return usage();
            }
            let pane = args.remove(0);
            let text = if args == ["-"] {
                let mut text = String::new();
                if std::io::stdin().read_to_string(&mut text).is_err() {
                    return fail_msg("could not read stdin");
                }
                text.trim_end_matches('\n').to_owned()
            } else {
                args.join(" ")
            };
            let request = obj(vec![
                ("cmd", s("prompt")),
                ("pane", s(pane.as_str())),
                ("text", s(text)),
                ("submit", Value::Bool(!no_submit)),
                ("force", Value::Bool(force)),
            ]);
            match send(&homes, &pane, request) {
                Ok(v) if ok(&v) => ExitCode::SUCCESS,
                Ok(v) => fail(&v),
                Err(e) => fail_msg(&e),
            }
        }
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        _ => usage(),
    }
}

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::from(2)
}

fn ok(v: &Value) -> bool {
    v.get("ok").and_then(Value::as_bool) == Some(true)
}

fn fail(v: &Value) -> ExitCode {
    fail_msg(v.get("error").and_then(Value::as_str).unwrap_or("failed"))
}

fn fail_msg(message: &str) -> ExitCode {
    eprintln!("terminal-ctl: {message}");
    ExitCode::from(1)
}

fn default_homes() -> Vec<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut out = Vec::new();
    for var in ["MAKEPAD_HOME", "OCTOSENSE_HOME"] {
        if let Some(dir) = std::env::var_os(var) {
            out.push(PathBuf::from(dir));
        }
    }
    if let Some(home) = home {
        out.push(home.join(".makepad"));
        out.push(home.join(".octosense"));
    }
    out.dedup();
    out
}

/// Every live control socket: (pid, path). A socket nobody answers is a
/// terminal that is gone; it is removed.
fn sockets(homes: &[PathBuf]) -> Vec<(u32, PathBuf)> {
    let mut out = Vec::new();
    for home in homes {
        let dir = home.join("terminal").join("control");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let mut path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_owned();
            // `<pid>.path` points at a socket kept under /tmp (long homes).
            let (stem, pointer) = match (name.strip_suffix(".sock"), name.strip_suffix(".path")) {
                (Some(stem), _) => (stem, false),
                (_, Some(stem)) => (stem, true),
                _ => continue,
            };
            let Ok(pid) = stem.parse::<u32>() else {
                continue;
            };
            if pointer {
                match std::fs::read_to_string(&path) {
                    Ok(target) if !target.trim().is_empty() => path = PathBuf::from(target.trim()),
                    _ => continue,
                }
            }
            if out.iter().any(|(p, _)| *p == pid) {
                continue;
            }
            out.push((pid, path));
        }
    }
    out
}

#[cfg(unix)]
fn request(path: &std::path::Path, body: &Value) -> Result<Value, String> {
    use std::os::unix::net::UnixStream;
    let mut stream = match UnixStream::connect(path) {
        Ok(stream) => stream,
        Err(_) => {
            let _ = std::fs::remove_file(path);
            return Err("the terminal is gone".into());
        }
    };
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(10)));
    stream.write_all(format!("{}\n", body.to_json()).as_bytes()).map_err(|e| e.to_string())?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line).map_err(|e| e.to_string())?;
    json::parse(line.trim().as_bytes()).map_err(|e| format!("bad reply: {e}"))
}

#[cfg(not(unix))]
fn request(_path: &std::path::Path, _body: &Value) -> Result<Value, String> {
    Err("terminal-ctl needs Unix sockets".into())
}

fn send(homes: &[PathBuf], pane: &str, body: Value) -> Result<Value, String> {
    let Some((pid, _)) = pane.split_once('.') else {
        return Err(format!("pane ids look like <pid>.<pane> (see `terminal-ctl list`), not {pane:?}"));
    };
    let pid: u32 = pid.parse().map_err(|_| format!("bad pane id {pane:?}"))?;
    let socket = sockets(homes).into_iter().find(|(p, _)| *p == pid).map(|(_, path)| path);
    let socket = socket.ok_or_else(|| format!("no terminal with pid {pid} allows control"))?;
    request(&socket, &body)
}

fn list(homes: &[PathBuf], as_json: bool) -> ExitCode {
    let mut panes = Vec::new();
    for (_, path) in sockets(homes) {
        if let Ok(reply) = request(&path, &obj(vec![("cmd", s("list"))])) {
            if let Some(found) = reply.get("panes").and_then(Value::as_arr) {
                panes.extend(found.iter().cloned());
            }
        }
    }
    if as_json {
        println!("{}", Value::Arr(panes).to_json());
        return ExitCode::SUCCESS;
    }
    let field = |p: &Value, k: &str| p.get(k).and_then(Value::as_str).unwrap_or("-").to_owned();
    println!("{:<12} {:<10} {:<9} {:<3} {:<24} CWD", "PANE", "AGENT", "STATE", "TAB", "TITLE");
    for p in &panes {
        let focused = if p.get("focused").and_then(Value::as_bool) == Some(true) { "*" } else { "" };
        let tab = p.get("tab").and_then(Value::as_i64).unwrap_or(0);
        let mut title = field(p, "title");
        if title.chars().count() > 24 {
            title = title.chars().take(23).collect::<String>() + "\u{2026}";
        }
        println!(
            "{:<12} {:<10} {:<9} {:<3} {:<24} {}",
            format!("{}{focused}", field(p, "id")),
            field(p, "agent"),
            field(p, "state"),
            tab,
            title,
            field(p, "cwd")
        );
    }
    if panes.is_empty() {
        eprintln!("terminal-ctl: no terminal allows control (Settings \u{2192} Automation \u{2192} Allow terminal-ctl)");
    }
    ExitCode::SUCCESS
}
