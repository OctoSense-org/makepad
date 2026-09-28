//! What runs in a pane, for the control socket (`crate::control`): which
//! coding agent (Claude Code, Codex, octoscode), and whether it is idle,
//! working, or blocked on an approval or a question — read from the
//! foreground program, the title it sets and the screen, the way herdr's
//! agent detection does.

/// A pane's state as the control socket reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// An agent (or the shell) waits for input.
    Idle,
    /// An agent is producing output or says it can be interrupted.
    Working,
    /// An approval or question is on screen: a prompt must not be typed.
    Blocked,
    /// A program that is not an agent runs in the foreground.
    Running,
    Exited,
}

impl State {
    pub fn as_str(self) -> &'static str {
        match self {
            State::Idle => "idle",
            State::Working => "working",
            State::Blocked => "blocked",
            State::Running => "running",
            State::Exited => "exited",
        }
    }
}

/// The agent a pane runs, from its foreground program's name and the title
/// the program set.
/// The program comes first: a title outlives the program that set it (the
/// shell does not reset it), so it only names an agent the program's name
/// does not (Claude Code can run as `node`).
pub fn detect_agent(job: Option<&str>, title: &str) -> Option<&'static str> {
    const AGENTS: &[&str] = &["octoscode", "claude", "codex"];
    let job = job?.to_lowercase();
    let by_job = AGENTS.iter().find(|name| job == **name || job.starts_with(&format!("{name}-")));
    if let Some(name) = by_job {
        return Some(name);
    }
    if job == "octos" {
        return Some("octoscode");
    }
    if matches!(job.as_str(), "node" | "bun" | "deno" | "python" | "python3") {
        let title = title.to_lowercase();
        return AGENTS.iter().find(|name| title.contains(**name)).copied();
    }
    None
}

/// Lines an approval or question dialog shows (lower case). Only the last
/// rows of the screen are looked at, where the agents draw these.
const BLOCKED: &[&str] = &[
    "do you want to",
    "would you like to",
    "allow once",
    "allow always",
    "don't ask again",
    "approve this",
    "requires approval",
    "approval required",
    "permission required",
    "(y/n)",
    "[y/n]",
    "press enter to confirm",
    "enter to confirm",
    "trust this folder",
    "allow command",
];

/// How many rows from the bottom hold the agent's input area and dialogs.
const TAIL_ROWS: usize = 16;
/// A working agent redraws its spinner or elapsed-time counter at least
/// once a second; a screen still for longer is waiting.
const WORKING_WINDOW: f64 = 2.5;
/// What Claude Code and Codex show in their status line while they work.
const WORKING: &[&str] = &["esc to interrupt", "ctrl+c to interrupt"];
/// Where that status line sits: just above the input box.
const STATUS_ROWS: usize = 8;

/// The state from the screen and how long ago it last changed. A working
/// agent animates its status line, so a still screen is not working even
/// when an old "esc to interrupt" line is left in view.
pub fn detect_state(
    agent: Option<&str>,
    job: Option<&str>,
    exited: bool,
    rows: &[String],
    since_change: Option<std::time::Duration>,
) -> State {
    if exited {
        return State::Exited;
    }
    let tail: Vec<String> = rows
        .iter()
        .rev()
        .filter(|r| !r.trim().is_empty())
        .take(TAIL_ROWS)
        .map(|r| r.to_lowercase())
        .collect();
    let shows = |needles: &[&str], rows: usize| tail.iter().take(rows).any(|row| needles.iter().any(|n| row.contains(n)));
    if agent.is_none() {
        return if job.is_some() { State::Running } else { State::Idle };
    }
    let secs = since_change.map_or(f64::INFINITY, |d| d.as_secs_f64());
    if shows(BLOCKED, TAIL_ROWS) {
        return State::Blocked;
    }
    let working = match agent {
        // These say so: their idle screens can animate (Codex's welcome),
        // and a status line left behind on a still screen is stale.
        Some("claude") | Some("codex") => shows(WORKING, STATUS_ROWS) && secs < 15.0,
        _ => secs < WORKING_WINDOW,
    };
    if working {
        State::Working
    } else {
        State::Idle
    }
}

/// Why `text` must not be typed as a prompt: a single approval key would
/// answer a dialog instead of asking something.
pub fn refuse_prompt(text: &str) -> Option<&'static str> {
    let t = text.trim();
    if t.is_empty() {
        return Some("empty prompt");
    }
    if matches!(t, "y" | "Y" | "n" | "N" | "s" | "S" | "yes" | "no" | "1" | "2" | "3" | "4") {
        return Some("a lone approval key is not a prompt");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(lines: &[&str]) -> Vec<String> {
        lines.iter().map(|l| l.to_string()).collect()
    }

    #[test]
    fn agents_are_known_by_program_or_title() {
        assert_eq!(detect_agent(Some("claude"), ""), Some("claude"));
        assert_eq!(detect_agent(Some("node"), "\u{2733} Claude Code"), Some("claude"));
        assert_eq!(detect_agent(Some("codex"), "\u{2733} Claude Code"), Some("codex"), "a stale title");
        assert_eq!(detect_agent(Some("zsh"), "Claude Code"), None, "the agent has exited");
        assert_eq!(detect_agent(Some("codex"), ""), Some("codex"));
        assert_eq!(detect_agent(Some("octoscode"), ""), Some("octoscode"));
        assert_eq!(detect_agent(Some("vim"), "notes.md"), None);
        assert_eq!(detect_agent(None, ""), None);
    }

    #[test]
    fn an_agent_is_idle_working_or_blocked() {
        let idle = rows(&["> ", "  ? for shortcuts"]);
        let secs = |s: u64| Some(std::time::Duration::from_secs(s));
        assert_eq!(detect_state(Some("claude"), Some("claude"), false, &idle, None), State::Idle);
        let working = rows(&["\u{2733} Thinking\u{2026} (esc to interrupt)", "> "]);
        assert_eq!(detect_state(Some("claude"), Some("claude"), false, &working, secs(1)), State::Working);
        assert_eq!(
            detect_state(Some("claude"), Some("claude"), false, &working, secs(60)),
            State::Idle,
            "a status line left behind on a still screen"
        );
        assert_eq!(detect_state(Some("codex"), Some("codex"), false, &idle, secs(1)), State::Idle, "an animated idle screen");
        assert_eq!(detect_state(Some("octoscode"), Some("octoscode"), false, &idle, secs(1)), State::Working, "output moving");
        let trust = rows(&["Accessing workspace:", "\u{276f} No, exit", "  Yes, I trust this folder", "Enter to confirm \u{00b7} Esc to cancel"]);
        assert_eq!(detect_state(Some("claude"), Some("claude"), false, &trust, None), State::Blocked);
        let asking = rows(&["Bash command", "  rm -rf build", "Do you want to proceed?", "\u{276f} 1. Yes", "  2. No"]);
        assert_eq!(detect_state(Some("claude"), Some("claude"), false, &asking, secs(0)), State::Blocked);
    }

    #[test]
    fn a_shell_is_idle_or_running_and_a_dead_one_exited() {
        let prompt = rows(&["user@host ~ % "]);
        assert_eq!(detect_state(None, None, false, &prompt, None), State::Idle);
        assert_eq!(detect_state(None, Some("htop"), false, &prompt, None), State::Running);
        assert_eq!(detect_state(Some("claude"), None, true, &prompt, None), State::Exited);
    }

    #[test]
    fn approval_keys_are_not_prompts() {
        assert!(refuse_prompt("y").is_some());
        assert!(refuse_prompt(" N ").is_some());
        assert!(refuse_prompt("1").is_some(), "Claude Code answers dialogs with digits");
        assert!(refuse_prompt("").is_some());
        assert_eq!(refuse_prompt("yes, run the tests"), None);
        assert_eq!(refuse_prompt("summarize the diff"), None);
    }
}
