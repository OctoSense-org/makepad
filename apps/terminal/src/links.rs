//! Clickable links: OSC 8 hyperlinks and URLs and paths found in the text.
//!
//! Holding the link modifier (Cmd on macOS, Ctrl elsewhere) while the
//! pointer is over a link underlines it and shows its target; clicking with
//! the modifier held opens it. A plain click never opens anything. The
//! modifier click is handled here even when the program has mouse reporting
//! on, the way Ghostty, iTerm2 and WezTerm do it, so links stay usable in
//! tmux or vim with the mouse on.
//!
//! - OSC 8 links are stored as ids in the cells (`crate::term::hyperlink`);
//!   every visible cell with the hovered id underlines, so the pieces of a
//!   link printed apart under one `id=` hover together.
//! - Text links are found lazily, only in the logical line (soft-wrapped
//!   rows joined) under the pointer, when the pointer moves with the
//!   modifier held: http(s), file and mailto URLs, and absolute or `~/`
//!   paths with an optional `:line[:col]`.
//! - Opening goes through a small scheme allowlist (http, https, mailto,
//!   file; paths are files); anything else is refused. The target is handed
//!   to the OS opener (`open` on macOS, `xdg-open` on Linux and the BSDs,
//!   `explorer` on Windows) as one argument, never through a shell.
//! - A host can take over opening with [`set_opener`] (the OctoSense shell
//!   could route web links to its own reader); setting
//!   `MAKEPAD_TERMINAL_LINK_OPENER=log` prints what would open to stderr
//!   instead of launching anything, for tests and the remote instrument.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::term::page::CellContent;
use crate::term::screen::Screen;
use crate::term::terminal::Terminal;

/// The most rows of one logical line scanned for text links.
const MAX_LINE_ROWS: usize = 64;

/// Environment variable that replaces the OS opener; `log` prints instead.
pub const OPENER_ENV: &str = "MAKEPAD_TERMINAL_LINK_OPENER";

/// Whether the link modifier is held: Cmd on macOS, Ctrl elsewhere.
pub fn link_modifier(logo: bool, control: bool) -> bool {
    if cfg!(target_os = "macos") {
        logo
    } else {
        control
    }
}

// ----------------------------------------------------------------------
// Targets and the scheme allowlist
// ----------------------------------------------------------------------

/// What a link opens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenTarget {
    /// An http, https or mailto URL.
    Url(String),
    /// A local file or directory (from a `file://` URL or a path), with the
    /// line and column a `path:line:col` named. The OS opener ignores them;
    /// a host opener may not.
    File {
        path: PathBuf,
        line: Option<u32>,
        col: Option<u32>,
    },
}

/// Why a link is not opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// A scheme outside the allowlist (`javascript:`, `ssh:`, custom app
    /// schemes...).
    Scheme(String),
    /// A `file://` URL naming another host (e.g. `ls --hyperlink` over SSH).
    RemoteFile(String),
    /// Not a URL or path we can make sense of.
    Malformed,
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refused::Scheme(s) => write!(
                f,
                "Not opened: \"{s}:\" links are not allowed (only http, https, mailto and file)"
            ),
            Refused::RemoteFile(h) => write!(f, "Not opened: the file is on another host ({h})"),
            Refused::Malformed => write!(f, "Not opened: not a valid link"),
        }
    }
}

/// Map a URI (an OSC 8 target or a detected URL) through the allowlist.
pub fn classify_uri(uri: &str) -> Result<OpenTarget, Refused> {
    let colon = uri.find(':').ok_or(Refused::Malformed)?;
    let scheme = &uri[..colon];
    let valid_scheme = scheme
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic())
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    if !valid_scheme {
        return Err(Refused::Malformed);
    }
    let rest = &uri[colon + 1..];
    match scheme.to_ascii_lowercase().as_str() {
        "http" | "https" => {
            let host = rest.strip_prefix("//").ok_or(Refused::Malformed)?;
            if host.is_empty() || host.starts_with('/') || uri.chars().any(char::is_whitespace) {
                return Err(Refused::Malformed);
            }
            Ok(OpenTarget::Url(uri.to_string()))
        }
        "mailto" => {
            if rest.is_empty() || uri.chars().any(char::is_whitespace) {
                return Err(Refused::Malformed);
            }
            Ok(OpenTarget::Url(uri.to_string()))
        }
        "file" => classify_file_url(rest),
        _ => Err(Refused::Scheme(scheme.to_string())),
    }
}

/// `file:` URLs: `file:///p`, `file://localhost/p`, `file://<this host>/p`
/// or `file:/p`. The path is percent-decoded.
fn classify_file_url(rest: &str) -> Result<OpenTarget, Refused> {
    let rest = rest.split(['?', '#']).next().unwrap_or("");
    let path = match rest.strip_prefix("//") {
        Some(auth_path) => {
            let slash = auth_path.find('/').ok_or(Refused::Malformed)?;
            let host = &auth_path[..slash];
            if !host.is_empty() && !host.eq_ignore_ascii_case("localhost") && !is_local_host(host) {
                return Err(Refused::RemoteFile(host.to_string()));
            }
            &auth_path[slash..]
        }
        None => rest,
    };
    if !path.starts_with('/') {
        return Err(Refused::Malformed);
    }
    let path = percent_decode(path).ok_or(Refused::Malformed)?;
    Ok(OpenTarget::File {
        path: PathBuf::from(path),
        line: None,
        col: None,
    })
}

/// A detected path, with `~/` expanded against `home`.
pub fn classify_path(
    path: &str,
    line: Option<u32>,
    col: Option<u32>,
    home: Option<&Path>,
) -> Result<OpenTarget, Refused> {
    let path = if let Some(rest) = path.strip_prefix("~/") {
        home.ok_or(Refused::Malformed)?.join(rest)
    } else if path.starts_with('/') {
        PathBuf::from(path)
    } else {
        return Err(Refused::Malformed);
    };
    Ok(OpenTarget::File { path, line, col })
}

fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    let s = String::from_utf8(out).ok()?;
    (!s.contains('\0')).then_some(s)
}

#[cfg(unix)]
fn is_local_host(host: &str) -> bool {
    extern "C" {
        fn gethostname(name: *mut std::os::raw::c_char, len: usize) -> std::os::raw::c_int;
    }
    let mut buf = [0u8; 256];
    // SAFETY: the buffer is valid for its length; gethostname writes at
    // most that many bytes.
    if unsafe { gethostname(buf.as_mut_ptr().cast(), buf.len()) } != 0 {
        return false;
    }
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let name = String::from_utf8_lossy(&buf[..len]);
    // `ls --hyperlink` prints the full name; compare the short one too.
    let short = |h: &str| h.split('.').next().unwrap_or("").to_ascii_lowercase();
    name.eq_ignore_ascii_case(host) || (!host.is_empty() && short(&name) == short(host))
}

#[cfg(not(unix))]
fn is_local_host(host: &str) -> bool {
    std::env::var("COMPUTERNAME").is_ok_and(|name| name.eq_ignore_ascii_case(host))
}

// ----------------------------------------------------------------------
// Opening
// ----------------------------------------------------------------------

/// A replacement for the OS opener.
pub type Opener = dyn Fn(&OpenTarget) -> io::Result<()> + Send + Sync;

static OPENER: Mutex<Option<Box<Opener>>> = Mutex::new(None);

/// Install (or with `None`, remove) a process-wide opener used instead of
/// the OS one: the hook for an in-process host that wants to route links
/// itself, and for tests. It only ever sees allowlisted targets.
pub fn set_opener(opener: Option<Box<Opener>>) {
    *OPENER.lock().unwrap_or_else(|e| e.into_inner()) = opener;
}

/// Open an allowlisted target.
pub fn open(target: &OpenTarget) -> io::Result<()> {
    if let Some(opener) = OPENER.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        return opener(target);
    }
    if std::env::var(OPENER_ENV).is_ok_and(|v| v == "log") {
        eprintln!("[terminal-links] open {}", describe(target));
        return Ok(());
    }
    let arg = match target {
        OpenTarget::Url(url) => url.clone().into(),
        OpenTarget::File { path, .. } => {
            if !path.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("{} does not exist", path.display()),
                ));
            }
            path.clone().into_os_string()
        }
    };
    let program = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(windows) {
        "explorer"
    } else {
        "xdg-open"
    };
    let mut child = std::process::Command::new(program)
        .arg(arg)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    // Reap it without blocking the UI.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

/// The target as shown in the status line and the log.
pub fn describe(target: &OpenTarget) -> String {
    match target {
        OpenTarget::Url(url) => url.clone(),
        OpenTarget::File { path, line, col } => {
            let mut s = path.display().to_string();
            if let Some(line) = line {
                s.push_str(&format!(":{line}"));
                if let Some(col) = col {
                    s.push_str(&format!(":{col}"));
                }
            }
            s
        }
    }
}

// ----------------------------------------------------------------------
// Detection in plain text
// ----------------------------------------------------------------------

/// A link found in text, as a char range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    pub start: usize,
    pub end: usize,
    pub kind: FoundKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FoundKind {
    Url(String),
    Path {
        path: String,
        line: Option<u32>,
        col: Option<u32>,
    },
}

const SCHEMES: [&str; 4] = ["https://", "http://", "file://", "mailto:"];

/// Every URL and path in `text`, in order, not overlapping.
pub fn detect(text: &[char]) -> Vec<Found> {
    let mut found = Vec::new();
    let mut i = 0;
    while i < text.len() {
        if let Some(f) = url_at(text, i).or_else(|| path_at(text, i)) {
            i = f.end;
            found.push(f);
        } else {
            i += 1;
        }
    }
    found
}

/// The link covering char `at`, if any.
pub fn detect_at(text: &[char], at: usize) -> Option<Found> {
    detect(text)
        .into_iter()
        .find(|f| f.start <= at && at < f.end)
}

fn starts_with_ci(text: &[char], at: usize, pat: &str) -> bool {
    let mut i = at;
    for p in pat.chars() {
        match text.get(i) {
            Some(c) if c.to_ascii_lowercase() == p => i += 1,
            _ => return false,
        }
    }
    true
}

fn url_char(c: char) -> bool {
    !c.is_whitespace()
        && !c.is_control()
        && !matches!(c, '<' | '>' | '"' | '`' | '{' | '}' | '|' | '\\' | '^')
}

fn path_char(c: char) -> bool {
    url_char(c) && !matches!(c, '\'' | ';')
}

/// Drop trailing punctuation that ends a sentence or closes a bracket the
/// link did not open: `(see https://x.y/a_(b))` keeps `a_(b)`,
/// `https://x.y.` loses the period.
fn trim_end(text: &[char], start: usize, mut end: usize, extra: &[char]) -> usize {
    while end > start {
        let last = text[end - 1];
        let unbalanced = |open: char, close: char| {
            last == close && {
                let s = &text[start..end];
                s.iter().filter(|&&c| c == close).count() > s.iter().filter(|&&c| c == open).count()
            }
        };
        if matches!(last, '.' | ',' | ';' | '!' | '?' | '\'' | '"' | '*')
            || extra.contains(&last)
            || unbalanced('(', ')')
            || unbalanced('[', ']')
        {
            end -= 1;
        } else {
            break;
        }
    }
    end
}

fn url_at(text: &[char], i: usize) -> Option<Found> {
    if i > 0 && (text[i - 1].is_alphanumeric() || matches!(text[i - 1], '/' | '.' | '-' | '+')) {
        return None;
    }
    let scheme = SCHEMES.iter().find(|s| starts_with_ci(text, i, s))?;
    let body = i + scheme.chars().count();
    let mut end = body;
    while end < text.len() && url_char(text[end]) {
        end += 1;
    }
    let end = trim_end(text, body, end, &[':']);
    if end <= body {
        return None;
    }
    let url: String = text[i..end].iter().collect();
    let rest = &text[body..end];
    let ok = match *scheme {
        "mailto:" => rest.contains(&'@'),
        "file://" => rest.contains(&'/'),
        _ => rest[0] != '/',
    };
    ok.then_some(Found {
        start: i,
        end,
        kind: FoundKind::Url(url),
    })
}

fn path_at(text: &[char], i: usize) -> Option<Found> {
    let lead = match (text[i], text.get(i + 1)) {
        ('/', _) => 1,
        ('~', Some('/')) => 2,
        _ => return None,
    };
    if i > 0 {
        let prev = text[i - 1];
        if !(prev.is_whitespace()
            || matches!(prev, '(' | '[' | '<' | '\'' | '"' | '`' | '=' | ':' | ','))
        {
            return None;
        }
    }
    // `//` starts a URL's authority (`xhttps://x`), not a path.
    if lead == 1 && text.get(i + 1) == Some(&'/') {
        return None;
    }
    let mut end = i + lead;
    while end < text.len() && path_char(text[end]) {
        // A `:` before another path ends this one (`PATH=/a:/b`).
        if text[end] == ':' && matches!(text.get(end + 1), Some('/' | '~')) {
            break;
        }
        end += 1;
    }
    let end = trim_end(text, i + lead, end, &[':']);
    let raw: String = text[i..end].iter().collect();
    let (path, line, col) = split_line_col(&raw);
    // Something beyond the slashes, with at least one letter or digit.
    let body = &path[lead..];
    if !body.chars().any(|c| c.is_alphanumeric()) {
        return None;
    }
    Some(Found {
        start: i,
        end,
        kind: FoundKind::Path {
            path: path.to_string(),
            line,
            col,
        },
    })
}

/// `x.rs:12:5` -> (`x.rs`, 12, 5); `x.rs:12` -> (`x.rs`, 12, None).
fn split_line_col(s: &str) -> (&str, Option<u32>, Option<u32>) {
    fn number(s: &str) -> Option<u32> {
        if s.is_empty() || s.len() > 9 || !s.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        s.parse().ok()
    }
    let Some((head, last)) = s.rsplit_once(':') else {
        return (s, None, None);
    };
    let Some(n) = number(last) else {
        return (s, None, None);
    };
    if let Some((path, mid)) = head.rsplit_once(':') {
        if let Some(line) = number(mid) {
            return (path, Some(line), Some(n));
        }
    }
    (head, Some(n), None)
}

// ----------------------------------------------------------------------
// Links on the screen
// ----------------------------------------------------------------------

/// A logical line's text and, for each char, the cell it came from as
/// (absolute row, head column).
pub struct LineText {
    pub chars: Vec<char>,
    pub cells: Vec<(u64, usize)>,
}

/// The logical line (soft-wrapped rows joined, at most
/// [`MAX_LINE_ROWS`] each way) through absolute row `abs`.
pub fn logical_line(screen: &Screen, abs: u64) -> Option<LineText> {
    let virt = screen.virtual_of_absolute(abs)?;
    let total = screen.total_rows();
    if virt >= total {
        return None;
    }
    let wrapped = |v: usize| screen.row_virtual(v).is_some_and(|r| r.wrapped);
    let mut first = virt;
    while first > 0 && virt - first < MAX_LINE_ROWS && wrapped(first - 1) {
        first -= 1;
    }
    let mut last = virt;
    while last + 1 < total && last - virt < MAX_LINE_ROWS && wrapped(last) {
        last += 1;
    }
    let mut line = LineText {
        chars: Vec::new(),
        cells: Vec::new(),
    };
    for v in first..=last {
        let row = screen.row_virtual(v)?;
        let row_abs = screen.absolute_of_virtual(v);
        // A wrapped row runs to the edge (trailing blanks are text).
        let width = if row.wrapped {
            screen.cols
        } else {
            row.cells.len().min(screen.cols)
        };
        for col in 0..width {
            let content = row.cell(col).map(|c| &c.content);
            match content {
                None | Some(CellContent::Empty) => {
                    line.chars.push(' ');
                    line.cells.push((row_abs, col));
                }
                Some(CellContent::Char(c)) | Some(CellContent::WideChar(c)) => {
                    line.chars.push(*c);
                    line.cells.push((row_abs, col));
                }
                Some(CellContent::Cluster(cluster)) => {
                    for &c in &cluster.cps {
                        line.chars.push(c);
                        line.cells.push((row_abs, col));
                    }
                }
                Some(CellContent::WideTail) | Some(CellContent::WideSpacerHead) => {}
            }
        }
    }
    Some(line)
}

/// The link under the pointer.
#[derive(Clone, Debug, PartialEq)]
pub struct LinkHit {
    /// The OSC 8 id, or 0 for a link found in the text.
    pub osc8: u32,
    /// For a text link, the cells it covers as (absolute row, first column,
    /// end column) per row.
    pub spans: Vec<(u64, usize, usize)>,
    /// The URI or text as printed.
    pub text: String,
    pub target: Result<OpenTarget, Refused>,
}

impl LinkHit {
    /// Whether the cell at (`abs`, `col`), carrying OSC 8 id `link`, is part
    /// of this link.
    pub fn covers(&self, abs: u64, col: usize, link: u32) -> bool {
        if self.osc8 != 0 {
            return link == self.osc8;
        }
        self.spans
            .iter()
            .any(|&(row, start, end)| row == abs && (start..end).contains(&col))
    }

    /// The status line text.
    pub fn label(&self) -> String {
        match &self.target {
            Ok(target) => describe(target),
            Err(refused) => format!("{}  ({})", self.text, refused_short(refused)),
        }
    }
}

fn refused_short(r: &Refused) -> &'static str {
    match r {
        Refused::Scheme(_) => "scheme not allowed",
        Refused::RemoteFile(_) => "remote file",
        Refused::Malformed => "not a valid link",
    }
}

/// The link at cell (`abs`, `col`) of the active screen: an OSC 8 link if
/// the cell has one, else a URL or path found in its logical line.
pub fn link_at(term: &Terminal, abs: u64, col: usize) -> Option<LinkHit> {
    let screen = term.screen();
    let row = screen.row_virtual(screen.virtual_of_absolute(abs)?)?;
    let head = row.head_of(col);
    let id = row.cell(head).map_or(0, |c| c.hyperlink);
    if let Some(uri) = term.hyperlink_uri(id) {
        return Some(LinkHit {
            osc8: id,
            spans: Vec::new(),
            text: uri.to_string(),
            target: classify_uri(uri),
        });
    }
    let line = logical_line(screen, abs)?;
    let at = line.cells.iter().position(|&cell| cell == (abs, head))?;
    let found = detect_at(&line.chars, at)?;
    let mut spans: Vec<(u64, usize, usize)> = Vec::new();
    for &(row_abs, col) in &line.cells[found.start..found.end] {
        let width = screen
            .row_virtual(screen.virtual_of_absolute(row_abs)?)
            .and_then(|r| r.cell(col))
            .map_or(1, |c| c.content.width().max(1) as usize);
        match spans.last_mut() {
            Some((r, _, end)) if *r == row_abs && *end >= col => *end = (*end).max(col + width),
            _ => spans.push((row_abs, col, col + width)),
        }
    }
    let text: String = line.chars[found.start..found.end].iter().collect();
    let target = match &found.kind {
        FoundKind::Url(url) => classify_uri(url),
        FoundKind::Path { path, line, col } => {
            let home = std::env::var_os("HOME").map(PathBuf::from);
            classify_path(path, *line, *col, home.as_deref())
        }
    };
    Some(LinkHit {
        osc8: 0,
        spans,
        text,
        target,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::stream::Stream;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    fn found(s: &str) -> Vec<String> {
        let text = chars(s);
        detect(&text)
            .into_iter()
            .map(|f| text[f.start..f.end].iter().collect())
            .collect()
    }

    #[test]
    fn urls_and_trailing_punctuation() {
        assert_eq!(found("see https://x.y."), ["https://x.y"]);
        assert_eq!(found("(https://x.y/a_(b))"), ["https://x.y/a_(b)"]);
        assert_eq!(found("(https://x.y/a)"), ["https://x.y/a"]);
        assert_eq!(
            found("[doc](https://x.y/d?q=1&r=2#f)"),
            ["https://x.y/d?q=1&r=2#f"]
        );
        assert_eq!(found("<https://x.y/p>,"), ["https://x.y/p"]);
        assert_eq!(
            found("'http://x.y/p', \"https://a.b\""),
            ["http://x.y/p", "https://a.b"]
        );
        assert_eq!(
            found("go to https://x.y/wiki/Foo_[bar]!"),
            ["https://x.y/wiki/Foo_[bar]"]
        );
        assert_eq!(found("HTTPS://X.Y/Up"), ["HTTPS://X.Y/Up"]);
        assert_eq!(found("https://x.y:8080/p:"), ["https://x.y:8080/p"]);
        assert_eq!(found("mailto:me@x.y."), ["mailto:me@x.y"]);
        assert_eq!(found("file:///tmp/a%20b.txt"), ["file:///tmp/a%20b.txt"]);
        assert_eq!(
            found("https://中文.example/路径 next"),
            ["https://中文.example/路径"]
        );
    }

    #[test]
    fn not_urls() {
        assert!(found("https:// nothing").is_empty());
        assert!(found("xhttps://x.y").is_empty());
        assert!(found("mailto:nobody").is_empty());
        assert!(found("https:///p").is_empty());
        assert!(found("javascript:alert(1)").is_empty());
    }

    #[test]
    fn paths_with_line_and_column() {
        let text = chars("error at ~/src/x.rs:12:5: bad, see /etc/hosts.");
        let f = detect(&text);
        assert_eq!(f.len(), 2);
        assert_eq!(
            f[0].kind,
            FoundKind::Path {
                path: "~/src/x.rs".into(),
                line: Some(12),
                col: Some(5)
            }
        );
        assert_eq!(
            f[1].kind,
            FoundKind::Path {
                path: "/etc/hosts".into(),
                line: None,
                col: None
            }
        );
        assert_eq!(found("~/home/x.rs:12"), ["~/home/x.rs:12"]);
        assert_eq!(found("(/usr/bin/env)"), ["/usr/bin/env"]);
        assert_eq!(found("PATH=/usr/bin:/bin"), ["/usr/bin", "/bin"]);
        assert_eq!(found("--out=/tmp/o.txt"), ["/tmp/o.txt"]);
    }

    #[test]
    fn not_paths() {
        assert!(found("a / b").is_empty());
        assert!(found("and/or 1/2 x//y").is_empty());
        assert!(found("~ and ~user/x").is_empty());
        // The path inside a URL is part of the URL, not a path of its own.
        assert_eq!(found("https://x.y/a/b"), ["https://x.y/a/b"]);
    }

    #[test]
    fn split_line_col_forms() {
        assert_eq!(split_line_col("/a.rs:3:4"), ("/a.rs", Some(3), Some(4)));
        assert_eq!(split_line_col("/a.rs:3"), ("/a.rs", Some(3), None));
        assert_eq!(split_line_col("/a:b"), ("/a:b", None, None));
        assert_eq!(split_line_col("/a"), ("/a", None, None));
        assert_eq!(
            split_line_col("/a:1234567890"),
            ("/a:1234567890", None, None)
        );
    }

    #[test]
    fn scheme_allowlist() {
        assert_eq!(
            classify_uri("https://x.y/p"),
            Ok(OpenTarget::Url("https://x.y/p".into()))
        );
        assert!(classify_uri("HTTP://x.y").is_ok());
        assert!(classify_uri("mailto:a@b.c").is_ok());
        assert_eq!(
            classify_uri("javascript:alert(1)"),
            Err(Refused::Scheme("javascript".into()))
        );
        assert_eq!(
            classify_uri("ssh://host"),
            Err(Refused::Scheme("ssh".into()))
        );
        assert_eq!(
            classify_uri("vscode://open?x"),
            Err(Refused::Scheme("vscode".into()))
        );
        assert_eq!(
            classify_uri("data:text/html,x"),
            Err(Refused::Scheme("data".into()))
        );
        assert_eq!(classify_uri("no scheme"), Err(Refused::Malformed));
        assert_eq!(classify_uri("1http://x"), Err(Refused::Malformed));
        assert_eq!(classify_uri("https:x"), Err(Refused::Malformed));
        assert_eq!(classify_uri("https://"), Err(Refused::Malformed));
        assert_eq!(classify_uri("https://x y"), Err(Refused::Malformed));
    }

    #[test]
    fn file_urls() {
        let file = |p: &str| {
            Ok(OpenTarget::File {
                path: PathBuf::from(p),
                line: None,
                col: None,
            })
        };
        assert_eq!(classify_uri("file:///tmp/a%20b.txt"), file("/tmp/a b.txt"));
        assert_eq!(classify_uri("file://localhost/tmp/x"), file("/tmp/x"));
        assert_eq!(classify_uri("file:/tmp/x?q#f"), file("/tmp/x"));
        assert_eq!(
            classify_uri("file://far.example.org/tmp/x"),
            Err(Refused::RemoteFile("far.example.org".into()))
        );
        assert_eq!(classify_uri("file:///tmp/%00x"), Err(Refused::Malformed));
        assert_eq!(classify_uri("file:///tmp/%zz"), Err(Refused::Malformed));
        assert_eq!(classify_uri("file://relative"), Err(Refused::Malformed));
    }

    #[test]
    fn paths_expand_home() {
        let home = Path::new("/home/me");
        assert_eq!(
            classify_path("~/x.rs", Some(1), None, Some(home)),
            Ok(OpenTarget::File {
                path: PathBuf::from("/home/me/x.rs"),
                line: Some(1),
                col: None
            })
        );
        assert_eq!(
            classify_path("~/x", None, None, None),
            Err(Refused::Malformed)
        );
        assert_eq!(
            classify_path("x", None, None, Some(home)),
            Err(Refused::Malformed)
        );
    }

    // ---- On a terminal --------------------------------------------------

    fn term(cols: usize, rows: usize, bytes: &[u8]) -> Terminal {
        let mut t = Terminal::new(cols, rows);
        Stream::new().process(bytes, &mut t);
        t
    }

    fn abs(t: &Terminal, y: usize) -> u64 {
        let s = t.screen();
        s.absolute_of_virtual(s.scrollback.len() + y)
    }

    #[test]
    fn osc8_cells_carry_the_link_across_a_wrap() {
        let t = term(
            8,
            3,
            b"ab\x1b]8;;https://x.y/\x1b\\0123456789\x1b]8;;\x07 z",
        );
        let s = t.screen();
        let row0 = s.row(0);
        let row1 = s.row(1);
        assert_eq!(row0.cells[0].hyperlink, 0);
        let id = row0.cells[2].hyperlink;
        assert_ne!(id, 0);
        assert!(row0.cells[2..8].iter().all(|c| c.hyperlink == id));
        assert!(row0.wrapped);
        assert!(row1.cells[0..4].iter().all(|c| c.hyperlink == id));
        assert_eq!(row1.cells[4].hyperlink, 0);
        assert_eq!(t.hyperlink_uri(id), Some("https://x.y/"));
        let hit = link_at(&t, abs(&t, 1), 1).unwrap();
        assert_eq!(hit.osc8, id);
        assert!(hit.covers(abs(&t, 0), 5, id));
        assert!(!hit.covers(abs(&t, 1), 5, 0));
        // Copy gives the visible text, not the URI.
        assert_eq!(
            s.selection_text((abs(&t, 0), 0), (abs(&t, 1), 6)),
            "ab0123456789 z"
        );
    }

    #[test]
    fn explicit_id_pieces_hover_together() {
        let t = term(
            20,
            3,
            b"\x1b]8;id=L;https://a/\x1b\\one\x1b]8;;\x1b\\ - \x1b]8;id=L;https://a/\x1b\\two\x1b]8;;\x1b\\",
        );
        let row = t.screen().row(0);
        let (one, two) = (row.cells[0].hyperlink, row.cells[6].hyperlink);
        assert_eq!(one, two);
        assert_eq!(row.cells[4].hyperlink, 0);
        assert_eq!(t.hyperlinks.len(), 1);
        // Without an id the two opens are separate links.
        let t = term(
            20,
            3,
            b"\x1b]8;;https://a/\x1b\\one\x1b]8;;\x1b\\ \x1b]8;;https://a/\x1b\\two",
        );
        let row = t.screen().row(0);
        assert_ne!(row.cells[0].hyperlink, row.cells[4].hyperlink);
    }

    #[test]
    fn refused_osc8_closes_the_open_link() {
        // An overlong URI (the VT parser already drops C0 controls inside
        // an OSC string; the OSC parser tests cover the rest).
        let long = format!(
            "\x1b]8;;https://a/\x1b\\ab\x1b]8;;https://b/{}\x1b\\cd",
            "b".repeat(2100)
        );
        let t = term(20, 3, long.as_bytes());
        let row = t.screen().row(0);
        assert_ne!(row.cells[0].hyperlink, 0);
        assert_eq!(row.cells[2].hyperlink, 0);
    }

    #[test]
    fn links_survive_reflow() {
        let mut t = term(
            10,
            4,
            b"\x1b]8;id=r;https://r/\x1b\\abcdefghijkl\x1b]8;;\x1b\\ tail",
        );
        let id = t.screen().row(0).cells[0].hyperlink;
        t.resize(6, 4);
        let s = t.screen();
        let linked: String = (0..s.total_rows())
            .filter_map(|v| s.row_virtual(v))
            .flat_map(|r| r.cells.iter())
            .filter(|c| c.hyperlink == id)
            .map(|c| c.content.primary().unwrap_or(' '))
            .collect();
        assert_eq!(linked, "abcdefghijkl");
        t.resize(30, 4);
        let row = t.screen().row(0);
        assert_eq!(row.text(), "abcdefghijkl tail");
        assert!(row.cells[..12].iter().all(|c| c.hyperlink == id));
        assert_eq!(row.cells[12].hyperlink, 0);
        assert_eq!(t.hyperlink_uri(id), Some("https://r/"));
    }

    #[test]
    fn scrollback_eviction_frees_links() {
        let mut t = Terminal::with_scrollback(20, 2, 5);
        let mut s = Stream::new();
        for i in 0..40 {
            s.process(
                format!("\x1b]8;;https://x/{i}\x1b\\l{i}\x1b]8;;\x1b\\\r\n").as_bytes(),
                &mut t,
            );
        }
        assert_eq!(t.hyperlinks.len(), 40);
        t.gc_hyperlinks();
        // 5 history rows + 2 active rows (one of them the empty cursor row).
        assert_eq!(t.hyperlinks.len(), 6);
        let kept = t.screen().scrollback[0].cells[0].hyperlink;
        assert_eq!(t.hyperlink_uri(kept), Some("https://x/34"));
        // Erasing the screen and the history frees the rest.
        s.process(b"\x1b[3J\x1b[2J", &mut t);
        t.gc_hyperlinks();
        assert!(t.hyperlinks.is_empty());
    }

    #[test]
    fn the_table_collects_itself() {
        let mut t = Terminal::with_scrollback(20, 2, 10);
        let mut s = Stream::new();
        for i in 0..5000 {
            s.process(
                format!("\x1b]8;;https://x/{i}\x1b\\l\x1b]8;;\x1b\\\r\n").as_bytes(),
                &mut t,
            );
        }
        // Collections keep the table near the live links, never near 5000.
        assert!(t.hyperlinks.len() <= 1024 + 12, "{}", t.hyperlinks.len());
        assert!(t.hyperlinks.capacity() <= 1024 + 12);
        let last = t.screen().row(0).cells[0].hyperlink;
        assert_eq!(t.hyperlink_uri(last), Some("https://x/4999"));
    }

    #[test]
    fn text_links_on_screen() {
        let t = term(
            20,
            4,
            "go https://github.com/OctoSense-org now ~/home/x.rs:12".as_bytes(),
        );
        // Row 0: "go https://github.co", wrapped, row 1: "m/OctoSense-org now ", ...
        let hit = link_at(&t, abs(&t, 1), 2).unwrap();
        assert_eq!(hit.osc8, 0);
        assert_eq!(hit.text, "https://github.com/OctoSense-org");
        assert_eq!(hit.spans, vec![(abs(&t, 0), 3, 20), (abs(&t, 1), 0, 15)]);
        assert_eq!(
            hit.target,
            Ok(OpenTarget::Url("https://github.com/OctoSense-org".into()))
        );
        assert!(link_at(&t, abs(&t, 1), 17).is_none());
        let hit = link_at(&t, abs(&t, 2), 3).unwrap();
        assert_eq!(hit.text, "~/home/x.rs:12");
        match hit.target {
            Ok(OpenTarget::File { path, line, col }) => {
                assert!(path.ends_with("home/x.rs"));
                assert_eq!((line, col), (Some(12), None));
            }
            other => panic!("{other:?}"),
        }
        assert!(link_at(&t, abs(&t, 0), 0).is_none());
    }

    #[test]
    fn wide_chars_map_to_their_cells() {
        let t = term(20, 2, "漢字 https://x.y/漢 z".as_bytes());
        let hit = link_at(&t, abs(&t, 0), 6).unwrap();
        assert_eq!(hit.text, "https://x.y/漢");
        // Two wide chars (4 cells) and a space, 12 narrow cells and one wide.
        assert_eq!(hit.spans, vec![(abs(&t, 0), 5, 19)]);
        // The wide char's tail column finds the same link.
        assert_eq!(link_at(&t, abs(&t, 0), 18).map(|h| h.text), Some(hit.text));
    }

    #[test]
    fn hook_opener_sees_allowlisted_targets() {
        use std::sync::{Arc, Mutex as StdMutex};
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let sink = seen.clone();
        set_opener(Some(Box::new(move |t: &OpenTarget| {
            sink.lock().unwrap().push(describe(t));
            Ok(())
        })));
        open(&OpenTarget::Url("https://x.y".into())).unwrap();
        open(&OpenTarget::File {
            path: "/nope/x.rs".into(),
            line: Some(3),
            col: Some(4),
        })
        .unwrap();
        set_opener(None);
        assert_eq!(*seen.lock().unwrap(), ["https://x.y", "/nope/x.rs:3:4"]);
    }

    #[test]
    fn modifier_per_platform() {
        assert_eq!(link_modifier(true, false), cfg!(target_os = "macos"));
        assert_eq!(link_modifier(false, true), !cfg!(target_os = "macos"));
    }
}
