//! The peer link: an app's own channel to ITS octos app agent (OctoSense
//! ADR 0004 §5).
//!
//! This is not the AI services bus. The bus ([`crate::port`]) exposes an
//! app's tools to one central conversation; the peer link connects an app
//! to the agent the host created for it, and to nobody else. Hosted by a
//! shell in its own process, the link rides the studio `Custom` frames
//! under its own envelope key, [`PEER_KEY`], which the bus never parses; an
//! in-process module gets the same API over a channel ([`OctosPeer::open`]).
//! Either way the app sees the same [`PeerEvent`]s and does not know how it
//! is hosted.
//!
//! Up (app → host):
//!
//! - [`PeerUp::Request`] `{req_id, method, args}`, `method` one of
//!   [`PEER_METHODS`] (`octos.session.open` opens a request context and
//!   answers `{"context": …}`; the others name that context in `args`);
//! - [`PeerUp::ToolResult`] `{call_id, ok, data | error |
//!   awaiting_confirmation}`.
//!
//! Down (host → app):
//!
//! - [`PeerDown::Reply`] `{req_id, ok, data | error}`, one per request;
//! - [`PeerDown::Event`] `{req_id, event}`, a request's streamed turn events
//!   before its reply;
//! - [`PeerDown::ToolCall`] `{call_id, name, args, risk, confirm_required,
//!   timeout_ms, account, context_id, client, caller}`: the agent calls one
//!   of the app's tools;
//! - [`PeerDown::ToolCancel`] `{call_id}`;
//! - [`PeerDown::ContextClosed`] `{context, reason}`: the host closed a
//!   context (sign-out, revoked grant).
//!
//! Identity. A frame never says who sent it: the host stamps the app from
//! the socket (or the channel) and the account, context and caller of a
//! tool call from its own records. The app may use only contexts it opened.
//!
//! Host obligations the app can rely on, and that the client also keeps on
//! its own side: one result per call, nothing after a cancel, and a
//! `confirm_required` call is acknowledged ([`OctosPeer::awaiting_confirmation`])
//! before the app shows its own confirmation sheet.

use makepad_platform::studio::AppToStudio;
use makepad_platform::thread::SignalToUI;
use makepad_platform::{Cx, Event};
use makepad_strict_json::{self as json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{channel, Receiver, Sender};

/// The envelope key of a peer-link frame inside a studio `Custom` frame.
pub const PEER_KEY: &str = "octos_peer";
/// The requests an app may make, by exact name.
pub const PEER_METHODS: [&str; 5] = [
    "octos.session.open",
    "octos.session.history",
    "octos.turn.start",
    "octos.turn.interrupt",
    "octos.context.close",
];
/// Bytes of one peer-link frame; larger frames are dropped unread.
pub const MAX_PEER_FRAME_BYTES: usize = 1024 * 1024;
/// Nesting a peer frame may have (turn events carry kernel payloads).
pub const MAX_PEER_DEPTH: u32 = 32;
/// Longest call id or context id.
pub const MAX_PEER_ID_BYTES: usize = 128;

/// Who is calling one of the app's tools, as the host stamped it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PeerCaller {
    /// The app's own agent (its own session, or a context of `client`).
    OwnAgent,
    /// Another app's agent (a cross-app call).
    App(String),
    /// The system agent.
    SystemAgent,
}

impl PeerCaller {
    pub fn as_wire(&self) -> String {
        match self {
            PeerCaller::OwnAgent => "own_agent".into(),
            PeerCaller::App(app) => format!("app:{app}"),
            PeerCaller::SystemAgent => "system_agent".into(),
        }
    }
    pub fn from_wire(s: &str) -> Option<PeerCaller> {
        match s {
            "own_agent" => Some(PeerCaller::OwnAgent),
            "system_agent" => Some(PeerCaller::SystemAgent),
            _ => s.strip_prefix("app:").filter(|a| !a.is_empty()).map(|a| PeerCaller::App(a.to_string())),
        }
    }
}

/// How much a tool can break, as the host tells the app.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerRisk {
    Read,
    Act,
    Destructive,
}

impl PeerRisk {
    pub fn as_wire(self) -> &'static str {
        match self {
            PeerRisk::Read => "read",
            PeerRisk::Act => "act",
            PeerRisk::Destructive => "destructive",
        }
    }
    pub fn from_wire(s: &str) -> Option<PeerRisk> {
        match s {
            "read" => Some(PeerRisk::Read),
            "act" => Some(PeerRisk::Act),
            "destructive" => Some(PeerRisk::Destructive),
            _ => None,
        }
    }
}

/// One call of the agent to one of the app's tools.
#[derive(Clone, Debug, PartialEq)]
pub struct PeerToolCall {
    pub call_id: String,
    pub name: String,
    pub args: Value,
    pub risk: PeerRisk,
    /// The app shows its own confirmation sheet (after acknowledging).
    pub confirm_required: bool,
    pub timeout_ms: u64,
    /// The account the call acts for (the peer's), stamped by the host.
    pub account: Option<String>,
    /// The request context the call came from; `None`: the agent's own session.
    pub context_id: Option<String>,
    /// The app's own label for the context's client (a mini app id).
    pub client: Option<String>,
    pub caller: PeerCaller,
}

/// How the app answers a tool call.
#[derive(Clone, Debug, PartialEq)]
pub enum PeerToolOutcome {
    Ok(Value),
    Error(String),
    /// The app is showing its confirmation sheet; the final answer follows.
    AwaitingConfirmation,
}

/// App → host.
#[derive(Clone, Debug, PartialEq)]
pub enum PeerUp {
    Request { req_id: u64, method: String, args: Value },
    ToolResult { call_id: String, outcome: PeerToolOutcome },
}

/// Host → app.
#[derive(Clone, Debug, PartialEq)]
pub enum PeerDown {
    Reply { req_id: u64, result: Result<Value, String> },
    Event { req_id: u64, event: Value },
    ToolCall(PeerToolCall),
    ToolCancel { call_id: String },
    ContextClosed { context: String, reason: String },
}

fn opt_str(v: &Option<String>) -> Value {
    v.as_ref().map(|s| json::s(s.clone())).unwrap_or(Value::Null)
}

fn get_str(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_string)
}

fn get_opt_str(v: &Value, key: &str) -> Result<Option<String>, ()> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Str(s)) => Ok(Some(s.clone())),
        Some(_) => Err(()),
    }
}

fn id_ok(id: &str) -> bool {
    !id.is_empty() && id.len() <= MAX_PEER_ID_BYTES && !id.chars().any(char::is_control)
}

fn envelope(inner: Value) -> String {
    json::obj(vec![(PEER_KEY, inner)]).to_json()
}

/// The inner object of a peer frame, or `None` for any other frame.
fn open_envelope(frame: &str) -> Option<Value> {
    if frame.len() > MAX_PEER_FRAME_BYTES || !frame.contains(PEER_KEY) {
        return None;
    }
    let value = json::parse_depth(frame.as_bytes(), MAX_PEER_DEPTH).ok()?;
    match value {
        Value::Obj(mut pairs) if pairs.len() == 1 && pairs[0].0 == PEER_KEY => Some(pairs.remove(0).1),
        _ => None,
    }
}

impl PeerUp {
    pub fn to_json(&self) -> String {
        envelope(match self {
            PeerUp::Request { req_id, method, args } => json::obj(vec![
                ("up", json::s("request")),
                ("req_id", Value::Int(*req_id as i64)),
                ("method", json::s(method.clone())),
                ("args", args.clone()),
            ]),
            PeerUp::ToolResult { call_id, outcome } => {
                let mut pairs = vec![("up", json::s("tool_result")), ("call_id", json::s(call_id.clone()))];
                match outcome {
                    PeerToolOutcome::Ok(data) => {
                        pairs.push(("ok", Value::Bool(true)));
                        pairs.push(("data", data.clone()));
                    }
                    PeerToolOutcome::Error(error) => {
                        pairs.push(("ok", Value::Bool(false)));
                        pairs.push(("error", json::s(error.clone())));
                    }
                    PeerToolOutcome::AwaitingConfirmation => {
                        pairs.push(("ok", Value::Bool(false)));
                        pairs.push(("awaiting_confirmation", Value::Bool(true)));
                    }
                }
                json::obj(pairs)
            }
        })
    }

    /// `None` for frames that are not the peer link's, over the cap, or
    /// malformed. Refuses unknown methods and bad ids.
    pub fn parse(frame: &str) -> Option<PeerUp> {
        let v = open_envelope(frame)?;
        match v.get("up")?.as_str()? {
            "request" => {
                let req_id = v.get("req_id")?.as_u64()?;
                let method = get_str(&v, "method")?;
                if !PEER_METHODS.contains(&method.as_str()) {
                    return None;
                }
                let args = v.get("args").cloned().unwrap_or(Value::Obj(Vec::new()));
                if !matches!(args, Value::Obj(_)) {
                    return None;
                }
                Some(PeerUp::Request { req_id, method, args })
            }
            "tool_result" => {
                let call_id = get_str(&v, "call_id").filter(|c| id_ok(c))?;
                let outcome = if v.get("awaiting_confirmation").and_then(Value::as_bool) == Some(true) {
                    PeerToolOutcome::AwaitingConfirmation
                } else if v.get("ok")?.as_bool()? {
                    PeerToolOutcome::Ok(v.get("data").cloned().unwrap_or(Value::Null))
                } else {
                    PeerToolOutcome::Error(get_str(&v, "error").unwrap_or_else(|| "failed".into()))
                };
                Some(PeerUp::ToolResult { call_id, outcome })
            }
            _ => None,
        }
    }
}

impl PeerDown {
    pub fn to_json(&self) -> String {
        envelope(match self {
            PeerDown::Reply { req_id, result } => {
                let mut pairs = vec![("down", json::s("reply")), ("req_id", Value::Int(*req_id as i64))];
                match result {
                    Ok(data) => {
                        pairs.push(("ok", Value::Bool(true)));
                        pairs.push(("data", data.clone()));
                    }
                    Err(error) => {
                        pairs.push(("ok", Value::Bool(false)));
                        pairs.push(("error", json::s(error.clone())));
                    }
                }
                json::obj(pairs)
            }
            PeerDown::Event { req_id, event } => json::obj(vec![
                ("down", json::s("event")),
                ("req_id", Value::Int(*req_id as i64)),
                ("event", event.clone()),
            ]),
            PeerDown::ToolCall(c) => json::obj(vec![
                ("down", json::s("tool_call")),
                ("call_id", json::s(c.call_id.clone())),
                ("name", json::s(c.name.clone())),
                ("args", c.args.clone()),
                ("risk", json::s(c.risk.as_wire())),
                ("confirm_required", Value::Bool(c.confirm_required)),
                ("timeout_ms", Value::Int(c.timeout_ms.min(i64::MAX as u64) as i64)),
                ("account", opt_str(&c.account)),
                ("context_id", opt_str(&c.context_id)),
                ("client", opt_str(&c.client)),
                ("caller", json::s(c.caller.as_wire())),
            ]),
            PeerDown::ToolCancel { call_id } => {
                json::obj(vec![("down", json::s("tool_cancel")), ("call_id", json::s(call_id.clone()))])
            }
            PeerDown::ContextClosed { context, reason } => json::obj(vec![
                ("down", json::s("context_closed")),
                ("context", json::s(context.clone())),
                ("reason", json::s(reason.clone())),
            ]),
        })
    }

    pub fn parse(frame: &str) -> Option<PeerDown> {
        let v = open_envelope(frame)?;
        match v.get("down")?.as_str()? {
            "reply" => {
                let req_id = v.get("req_id")?.as_u64()?;
                let result = if v.get("ok")?.as_bool()? {
                    Ok(v.get("data").cloned().unwrap_or(Value::Null))
                } else {
                    Err(get_str(&v, "error").unwrap_or_else(|| "failed".into()))
                };
                Some(PeerDown::Reply { req_id, result })
            }
            "event" => Some(PeerDown::Event { req_id: v.get("req_id")?.as_u64()?, event: v.get("event")?.clone() }),
            "tool_call" => {
                let call_id = get_str(&v, "call_id").filter(|c| id_ok(c))?;
                Some(PeerDown::ToolCall(PeerToolCall {
                    call_id,
                    name: get_str(&v, "name").filter(|n| id_ok(n))?,
                    args: v.get("args").cloned().unwrap_or(Value::Obj(Vec::new())),
                    risk: PeerRisk::from_wire(v.get("risk")?.as_str()?)?,
                    confirm_required: v.get("confirm_required")?.as_bool()?,
                    timeout_ms: v.get("timeout_ms")?.as_u64()?,
                    account: get_opt_str(&v, "account").ok()?,
                    context_id: get_opt_str(&v, "context_id").ok()?,
                    client: get_opt_str(&v, "client").ok()?,
                    caller: PeerCaller::from_wire(v.get("caller")?.as_str()?)?,
                }))
            }
            "tool_cancel" => Some(PeerDown::ToolCancel { call_id: get_str(&v, "call_id").filter(|c| id_ok(c))? }),
            "context_closed" => Some(PeerDown::ContextClosed {
                context: get_str(&v, "context")?,
                reason: get_str(&v, "reason").unwrap_or_default(),
            }),
            _ => None,
        }
    }
}

/// What the app reads from its peer link each frame.
#[derive(Clone, Debug, PartialEq)]
pub enum PeerEvent {
    /// The one answer to a request.
    Reply { req_id: u64, result: Result<Value, String> },
    /// A streamed event of a request's turn.
    TurnEvent { req_id: u64, event: Value },
    /// The agent calls one of the app's tools: answer with
    /// [`OctosPeer::tool_result`] (and [`OctosPeer::awaiting_confirmation`]
    /// first when `confirm_required`).
    ToolCall(PeerToolCall),
    /// Stop that call; no result is expected any more.
    ToolCancel { call_id: String },
    ContextClosed { context: String, reason: String },
}

/// The host's end of an in-process link.
pub struct PeerLink {
    /// App → host.
    pub up: Receiver<PeerUp>,
    /// Host → app.
    pub down: Sender<PeerDown>,
}

/// In-process links opened since the host last looked, parked on `Cx`.
/// The host takes them right after it created the module instance that
/// opened them: a link's identity is the instance it was opened in, never
/// anything it says.
#[derive(Default)]
pub struct PendingPeerLinks {
    pub links: Vec<PeerLink>,
}

impl PendingPeerLinks {
    pub fn take(&mut self) -> Vec<PeerLink> {
        std::mem::take(&mut self.links)
    }
}

enum PeerTransport {
    Hosted,
    InProcess { up: Sender<PeerUp>, down: Receiver<PeerDown> },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CallState {
    Open,
    Acknowledged,
}

/// An app's link to its agent.
pub struct OctosPeer {
    transport: PeerTransport,
    next_req: u64,
    /// Requests without their reply yet.
    pending: HashSet<u64>,
    /// Tool calls that still take a result.
    calls: HashMap<String, (CallState, bool)>,
}

impl OctosPeer {
    /// Open the link: over the hub socket when a shell hosts this process,
    /// else in-process, its host end parked in [`PendingPeerLinks`].
    pub fn open(cx: &mut Cx) -> OctosPeer {
        if cx.in_makepad_studio() {
            return OctosPeer::hosted();
        }
        let (peer, link) = OctosPeer::in_process();
        cx.global::<PendingPeerLinks>().links.push(link);
        peer
    }

    /// The hosted transport (the caller knows a shell hosts the process).
    pub fn hosted() -> OctosPeer {
        OctosPeer::with(PeerTransport::Hosted)
    }

    /// An in-process link and the host's end of it.
    pub fn in_process() -> (OctosPeer, PeerLink) {
        let (up_tx, up_rx) = channel();
        let (down_tx, down_rx) = channel();
        (OctosPeer::with(PeerTransport::InProcess { up: up_tx, down: down_rx }), PeerLink { up: up_rx, down: down_tx })
    }

    fn with(transport: PeerTransport) -> OctosPeer {
        OctosPeer { transport, next_req: 1, pending: HashSet::new(), calls: HashMap::new() }
    }

    /// Send a request; its id correlates the events and the reply.
    pub fn request(&mut self, method: &str, args: Value) -> u64 {
        let req_id = self.next_req;
        self.next_req += 1;
        self.pending.insert(req_id);
        self.send(PeerUp::Request { req_id, method: method.to_string(), args });
        req_id
    }

    /// `octos.session.open`: a request context for `client` (the app's own
    /// label, e.g. a mini app id) on the current account. Replies
    /// `{"context": id}`.
    pub fn open_session(&mut self, client: Option<&str>) -> u64 {
        let args = json::obj(vec![("client", client.map(json::s).unwrap_or(Value::Null))]);
        self.request("octos.session.open", args)
    }

    pub fn start_turn(&mut self, context: &str, text: &str) -> u64 {
        self.request("octos.turn.start", json::obj(vec![("context", json::s(context)), ("text", json::s(text))]))
    }

    pub fn history(&mut self, context: &str) -> u64 {
        self.request("octos.session.history", json::obj(vec![("context", json::s(context))]))
    }

    pub fn interrupt(&mut self, context: &str) -> u64 {
        self.request("octos.turn.interrupt", json::obj(vec![("context", json::s(context))]))
    }

    pub fn close_context(&mut self, context: &str) -> u64 {
        self.request("octos.context.close", json::obj(vec![("context", json::s(context))]))
    }

    /// Acknowledge a `confirm_required` call before showing the sheet.
    pub fn awaiting_confirmation(&mut self, call_id: &str) -> bool {
        match self.calls.get_mut(call_id) {
            Some((state @ CallState::Open, true)) => {
                *state = CallState::Acknowledged;
                self.send(PeerUp::ToolResult { call_id: call_id.to_string(), outcome: PeerToolOutcome::AwaitingConfirmation });
                true
            }
            _ => false,
        }
    }

    /// The one result of a call. `false` (and nothing sent) for a call that
    /// was already answered, cancelled, never made, or that needs a
    /// confirmation not yet acknowledged.
    pub fn tool_result(&mut self, call_id: &str, outcome: Result<Value, String>) -> bool {
        match self.calls.get(call_id) {
            Some((CallState::Open, true)) => return false,
            Some(_) => {}
            None => return false,
        }
        self.calls.remove(call_id);
        let outcome = match outcome {
            Ok(v) => PeerToolOutcome::Ok(v),
            Err(e) => PeerToolOutcome::Error(e),
        };
        self.send(PeerUp::ToolResult { call_id: call_id.to_string(), outcome });
        true
    }

    /// Requests still waiting for their reply.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// What arrived. Hosted: the peer envelope of `Custom` frames; in-process:
    /// the channel, drained on every event.
    pub fn handle_event(&mut self, _cx: &mut Cx, event: &Event) -> Vec<PeerEvent> {
        let frames: Vec<PeerDown> = match &self.transport {
            PeerTransport::Hosted => match event {
                Event::Custom(frame) => PeerDown::parse(frame).into_iter().collect(),
                _ => Vec::new(),
            },
            PeerTransport::InProcess { down, .. } => down.try_iter().collect(),
        };
        self.accept(frames)
    }

    fn accept(&mut self, frames: Vec<PeerDown>) -> Vec<PeerEvent> {
        let mut out = Vec::new();
        for frame in frames {
            match frame {
                PeerDown::Reply { req_id, result } => {
                    if self.pending.remove(&req_id) {
                        out.push(PeerEvent::Reply { req_id, result });
                    }
                }
                PeerDown::Event { req_id, event } => {
                    if self.pending.contains(&req_id) {
                        out.push(PeerEvent::TurnEvent { req_id, event });
                    }
                }
                PeerDown::ToolCall(call) => {
                    // A repeated call id is the same call: delivered once.
                    if !self.calls.contains_key(&call.call_id) {
                        self.calls.insert(call.call_id.clone(), (CallState::Open, call.confirm_required));
                        out.push(PeerEvent::ToolCall(call));
                    }
                }
                PeerDown::ToolCancel { call_id } => {
                    if self.calls.remove(&call_id).is_some() {
                        out.push(PeerEvent::ToolCancel { call_id });
                    }
                }
                PeerDown::ContextClosed { context, reason } => out.push(PeerEvent::ContextClosed { context, reason }),
            }
        }
        out
    }

    fn send(&self, up: PeerUp) {
        match &self.transport {
            PeerTransport::Hosted => Cx::send_studio_message(AppToStudio::Custom(up.to_json())),
            PeerTransport::InProcess { up: tx, .. } => {
                if tx.send(up).is_ok() {
                    SignalToUI::set_ui_signal();
                }
            }
        }
    }

    #[cfg(test)]
    fn test_drain(&mut self) -> Vec<PeerEvent> {
        let frames: Vec<PeerDown> = match &self.transport {
            PeerTransport::InProcess { down, .. } => down.try_iter().collect(),
            PeerTransport::Hosted => Vec::new(),
        };
        self.accept(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(id: &str, confirm: bool) -> PeerToolCall {
        PeerToolCall {
            call_id: id.into(),
            name: "send".into(),
            args: json::obj(vec![("to", json::s("#room"))]),
            risk: PeerRisk::Destructive,
            confirm_required: confirm,
            timeout_ms: 30_000,
            account: Some("acct".into()),
            context_id: Some("ctx-1".into()),
            client: Some("mini".into()),
            caller: PeerCaller::App("calendar".into()),
        }
    }

    #[test]
    fn frames_round_trip_under_their_own_envelope() {
        let ups = [
            PeerUp::Request { req_id: 7, method: "octos.turn.start".into(), args: json::obj(vec![("context", json::s("c")), ("text", json::s("hi"))]) },
            PeerUp::ToolResult { call_id: "k1".into(), outcome: PeerToolOutcome::Ok(json::obj(vec![("n", Value::Int(3))])) },
            PeerUp::ToolResult { call_id: "k1".into(), outcome: PeerToolOutcome::Error("no".into()) },
            PeerUp::ToolResult { call_id: "k1".into(), outcome: PeerToolOutcome::AwaitingConfirmation },
        ];
        for up in ups {
            let frame = up.to_json();
            assert!(frame.starts_with("{\"octos_peer\":"), "{frame}");
            assert_eq!(PeerUp::parse(&frame), Some(up));
            assert_eq!(PeerDown::parse(&frame), None, "an up frame is never a down frame");
        }
        let downs = [
            PeerDown::Reply { req_id: 1, result: Ok(json::obj(vec![("context", json::s("c"))])) },
            PeerDown::Reply { req_id: 1, result: Err("not granted".into()) },
            PeerDown::Event { req_id: 2, event: json::obj(vec![("method", json::s("message/delta"))]) },
            PeerDown::ToolCall(call("k2", true)),
            PeerDown::ToolCancel { call_id: "k2".into() },
            PeerDown::ContextClosed { context: "c".into(), reason: "signed_out".into() },
        ];
        for down in downs {
            let frame = down.to_json();
            assert_eq!(PeerDown::parse(&frame), Some(down));
            assert_eq!(PeerUp::parse(&frame), None);
        }
    }

    /// The exact frames the shell's side (OctoSense `peer_link/wire.rs`)
    /// pins as its fixtures: change both or neither.
    #[test]
    fn the_wire_is_the_shells_fixture() {
        let fixtures = [
            (PeerUp::Request { req_id: 7, method: "octos.turn.start".into(), args: json::obj(vec![("context", json::s("c")), ("text", json::s("hi"))]) },
             r#"{"octos_peer":{"up":"request","req_id":7,"method":"octos.turn.start","args":{"context":"c","text":"hi"}}}"#),
            (PeerUp::ToolResult { call_id: "k1".into(), outcome: PeerToolOutcome::Ok(json::obj(vec![("n", Value::Int(3))])) },
             r#"{"octos_peer":{"up":"tool_result","call_id":"k1","ok":true,"data":{"n":3}}}"#),
            (PeerUp::ToolResult { call_id: "k1".into(), outcome: PeerToolOutcome::Error("no".into()) },
             r#"{"octos_peer":{"up":"tool_result","call_id":"k1","ok":false,"error":"no"}}"#),
            (PeerUp::ToolResult { call_id: "k1".into(), outcome: PeerToolOutcome::AwaitingConfirmation },
             r#"{"octos_peer":{"up":"tool_result","call_id":"k1","ok":false,"awaiting_confirmation":true}}"#),
        ];
        for (up, frame) in fixtures {
            assert_eq!(up.to_json(), frame);
        }
        // What the shell writes (serde_json, keys in its own order) parses.
        let shell = r#"{"octos_peer":{"account":"device","args":{},"call_id":"c1","caller":"own_agent","client":"mini","confirm_required":false,"context_id":"n-pl1-1","down":"tool_call","name":"lookup","risk":"read","timeout_ms":30000}}"#;
        match PeerDown::parse(shell) {
            Some(PeerDown::ToolCall(c)) => {
                assert_eq!((c.account.as_deref(), c.client.as_deref(), c.caller), (Some("device"), Some("mini"), PeerCaller::OwnAgent));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn other_frames_and_bad_frames_are_not_the_links() {
        // The AI bus's own envelope, the WM's, and junk are ignored.
        assert_eq!(PeerUp::parse(r#"{"wm_ai":{"from":null,"msg":{"Unregister":[]}}}"#), None);
        assert_eq!(PeerUp::parse(r#"{"wm":{"Hosted":{}}}"#), None);
        assert_eq!(PeerUp::parse("octos_peer"), None);
        // An unknown method, a non-object args, a second top-level key.
        assert_eq!(PeerUp::parse(r#"{"octos_peer":{"up":"request","req_id":1,"method":"octos.admin","args":{}}}"#), None);
        assert_eq!(PeerUp::parse(r#"{"octos_peer":{"up":"request","req_id":1,"method":"octos.turn.start","args":[1]}}"#), None);
        assert_eq!(PeerUp::parse(r#"{"octos_peer":{"up":"request","req_id":1,"method":"octos.turn.start"},"x":1}"#), None);
        // An oversized frame is dropped unread.
        let big = format!(r#"{{"octos_peer":{{"up":"request","req_id":1,"method":"octos.turn.start","args":{{"text":"{}"}}}}}}"#, "a".repeat(MAX_PEER_FRAME_BYTES));
        assert_eq!(PeerUp::parse(&big), None);
        // A frame never carries identity the host would take: an `app` or
        // `from` field is simply not part of the wire.
        let claimed = r#"{"octos_peer":{"up":"request","req_id":1,"method":"octos.session.open","args":{},"app":"rinx"}}"#;
        assert!(matches!(PeerUp::parse(claimed), Some(PeerUp::Request { .. })));
    }

    #[test]
    fn the_client_keeps_once_per_call_nothing_after_cancel_and_ack_first() {
        let (mut peer, link) = OctosPeer::in_process();
        link.down.send(PeerDown::ToolCall(call("a", false))).unwrap();
        link.down.send(PeerDown::ToolCall(call("a", false))).unwrap();
        link.down.send(PeerDown::ToolCall(call("b", true))).unwrap();
        link.down.send(PeerDown::ToolCall(call("c", false))).unwrap();
        link.down.send(PeerDown::ToolCancel { call_id: "c".into() }).unwrap();
        let events = peer.test_drain();
        assert_eq!(events.len(), 4, "a repeated call id is delivered once: {events:?}");
        assert!(peer.tool_result("a", Ok(Value::Null)));
        assert!(!peer.tool_result("a", Ok(Value::Null)), "once per call");
        assert!(!peer.tool_result("c", Ok(Value::Null)), "nothing after cancel");
        assert!(!peer.tool_result("b", Ok(Value::Null)), "a confirmation is acknowledged first");
        assert!(peer.awaiting_confirmation("b"));
        assert!(!peer.awaiting_confirmation("b"));
        assert!(peer.tool_result("b", Err("declined".into())));
        let up: Vec<PeerUp> = link.up.try_iter().collect();
        assert_eq!(up.len(), 3, "{up:?}");
        assert!(matches!(&up[1], PeerUp::ToolResult { call_id, outcome: PeerToolOutcome::AwaitingConfirmation } if call_id == "b"));
    }

    #[test]
    fn replies_and_events_reach_only_their_open_request() {
        let (mut peer, link) = OctosPeer::in_process();
        let open = peer.open_session(Some("mini"));
        match link.up.try_recv().unwrap() {
            PeerUp::Request { req_id, method, args } => {
                assert_eq!((req_id, method.as_str()), (open, "octos.session.open"));
                assert_eq!(args.get("client").and_then(Value::as_str), Some("mini"));
            }
            other => panic!("{other:?}"),
        }
        link.down.send(PeerDown::Event { req_id: 99, event: Value::Null }).unwrap();
        link.down.send(PeerDown::Reply { req_id: open, result: Ok(json::obj(vec![("context", json::s("c1"))])) }).unwrap();
        link.down.send(PeerDown::Reply { req_id: open, result: Ok(Value::Null) }).unwrap();
        let events = peer.test_drain();
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(peer.pending(), 0);
    }
}
