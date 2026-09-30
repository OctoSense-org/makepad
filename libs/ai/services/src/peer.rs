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
//!   context (sign-out, revoked grant);
//! - [`PeerDown::Conversation`] `{context, event}`: an event of the app's
//!   conversation (a context opened without a `client`), in EITHER lane
//!   (ADR 0004 §6): the person's (this app's own turns) and the system
//!   agent's (its `peer_send_input` turns on the app's agent), streamed live
//!   whoever started the turn and whether or not a request of this app is
//!   still open. `event` is the host's turn event with its `lane`
//!   (`person` | `system_agent`), `speaker` (`{kind, label?}`), a user
//!   message's `display_text`, the text streamed so far (`text`) and, on
//!   `turn/started`, the turn's `request`. The client surfaces it as
//!   [`PeerEvent::Conversation`] ([`ConversationEvent`]).
//!
//! Turns queue per context. A context runs one turn of its own at a time
//! (the host refuses a second while one runs), so [`OctosPeer::start_turn`]
//! on a context whose turn has not answered yet queues the message here and
//! sends it when that turn answers: at most [`MAX_QUEUED_TURNS`] wait per
//! context ([`PeerTurnError::QueueFull`] beyond), and closing the context
//! ([`OctosPeer::close_context`], or the host's `context_closed`) or
//! stopping it ([`OctosPeer::interrupt`]) cancels what still waits, each
//! with an error reply.
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
use std::collections::{HashMap, HashSet, VecDeque};
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
/// Messages that may wait on one context behind its running turn.
pub const MAX_QUEUED_TURNS: usize = 8;
/// Conversation events held while a context is still opening (the host
/// follows the conversation before its open answers); more are dropped.
pub const MAX_EARLY_CONVERSATION_EVENTS: usize = 256;

/// Which lane of the app's conversation an event is in (ADR 0004 §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerLane {
    /// The person's lane: this app's own turns (and its other handles').
    Person,
    /// The system agent's lane: its `peer_send_input` turns on the app's agent.
    SystemAgent,
}

impl PeerLane {
    pub fn as_wire(self) -> &'static str {
        match self {
            PeerLane::Person => "person",
            PeerLane::SystemAgent => "system_agent",
        }
    }
    pub fn from_wire(s: &str) -> Option<PeerLane> {
        match s {
            "person" => Some(PeerLane::Person),
            "system_agent" => Some(PeerLane::SystemAgent),
            _ => None,
        }
    }
}

/// Who spoke in a turn, as the host says: `kind` is `person`,
/// `system_agent` or `app`; `label` the name the host gave (the app's).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeerSpeaker {
    pub kind: String,
    pub label: Option<String>,
}

/// One event of the app's conversation, in either lane.
#[derive(Clone, Debug, PartialEq)]
pub struct ConversationEvent {
    /// The context (handle) the app opened and follows.
    pub context: String,
    /// `None` only if the host sent no (or an unknown) lane.
    pub lane: Option<PeerLane>,
    /// The kernel event: `turn/started`, `projection/envelope`,
    /// `progress/updated`, `message/delta`, `turn/completed`, …
    pub method: String,
    /// A `projection/envelope`'s payload type: `user_message`,
    /// `assistant_delta`, `assistant_persisted`, `turn_terminal`, …
    pub envelope: Option<String>,
    /// A piece of the answer: an `assistant_delta` envelope's text, or a
    /// `message/delta`'s. Envelopes may arrive out of `seq` order.
    pub delta: Option<String>,
    pub turn_id: Option<String>,
    pub speaker: Option<PeerSpeaker>,
    /// A user message's text without the kernel's speaker marker.
    pub display_text: Option<String>,
    /// The turn's answer streamed so far.
    pub text: Option<String>,
    /// The whole event as the host sent it.
    pub event: Value,
}

impl ConversationEvent {
    /// `None` when `event` is not an object with a `method`.
    pub fn from_frame(context: &str, event: Value) -> Option<ConversationEvent> {
        let method = get_str(&event, "method")?;
        let params = event.get("params");
        let payload = params.and_then(|p| p.get("payload"));
        let envelope = (method == "projection/envelope").then(|| payload.and_then(|p| get_str(p, "type"))).flatten();
        let delta = match (method.as_str(), envelope.as_deref()) {
            ("message/delta", _) => params.and_then(|p| get_str(p, "text")),
            (_, Some("assistant_delta")) => payload.and_then(|p| p.get("data")).and_then(|d| get_str(d, "text")),
            _ => None,
        };
        let speaker = event.get("speaker").and_then(|s| {
            Some(PeerSpeaker { kind: get_str(s, "kind")?, label: get_str(s, "label") })
        });
        Some(ConversationEvent {
            context: context.to_string(),
            lane: event.get("lane").and_then(Value::as_str).and_then(PeerLane::from_wire),
            method,
            envelope,
            delta,
            turn_id: params.and_then(|p| get_str(p, "turn_id")),
            speaker,
            display_text: get_str(&event, "display_text"),
            text: get_str(&event, "text"),
            event,
        })
    }

    /// The system agent's lane: a turn the app did not start.
    pub fn is_system_agent(&self) -> bool {
        self.lane == Some(PeerLane::SystemAgent)
    }

    /// The turn ended: `turn/completed`, `turn/error`, or a
    /// `turn_terminal` envelope (its `outcome` says how).
    pub fn is_turn_end(&self) -> bool {
        matches!(self.method.as_str(), "turn/completed" | "turn/error") || self.envelope.as_deref() == Some("turn_terminal")
    }

    /// On `turn/started`, the words that started the turn (the kernel
    /// writes the user message only when the turn ends).
    pub fn request_text(&self) -> Option<&str> {
        self.event.get("request").and_then(|r| r.get("text")).and_then(Value::as_str)
    }
}

/// Why a message was not taken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PeerTurnError {
    /// [`MAX_QUEUED_TURNS`] messages already wait behind this context's
    /// running turn.
    QueueFull { context: String, limit: usize },
}

impl std::fmt::Display for PeerTurnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PeerTurnError::QueueFull { context, limit } => write!(
                f,
                "queue_full: {limit} messages already wait behind the running turn of {context}; send again once it answers"
            ),
        }
    }
}

impl std::error::Error for PeerTurnError {}

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
    /// An event of either lane of the app's conversation (see the module
    /// docs); the host sends it for a context opened without a `client`.
    Conversation { context: String, event: Value },
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
            PeerDown::Conversation { context, event } => json::obj(vec![
                ("down", json::s("conversation")),
                ("context", json::s(context.clone())),
                ("event", event.clone()),
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
            "conversation" => {
                let context = get_str(&v, "context").filter(|c| id_ok(c))?;
                let event = v.get("event")?.clone();
                if !matches!(event, Value::Obj(_)) {
                    return None;
                }
                Some(PeerDown::Conversation { context, event })
            }
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
    /// An event of the app's conversation, in either lane: the system
    /// agent's turns on the app's agent arrive here live, as do the
    /// person's (this app's own turns, also streamed to their request as
    /// [`PeerEvent::TurnEvent`]). Only for contexts this link opened and
    /// has not closed.
    Conversation(ConversationEvent),
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

/// One context's turns: the one sent (no reply yet) and those waiting.
#[derive(Default)]
struct TurnLane {
    running: Option<u64>,
    queued: VecDeque<(u64, Value)>,
}

/// An app's link to its agent.
pub struct OctosPeer {
    transport: PeerTransport,
    next_req: u64,
    /// Requests without their reply yet (queued turns included).
    pending: HashSet<u64>,
    /// Tool calls that still take a result.
    calls: HashMap<String, (CallState, bool)>,
    /// `octos.session.open` requests without their reply yet.
    opening: HashSet<u64>,
    /// Contexts this link opened and has not closed.
    contexts: HashSet<String>,
    /// Each context's running and queued turns.
    turns: HashMap<String, TurnLane>,
    /// Conversation events for a context not yet known, held while an open
    /// is answering (bounded by [`MAX_EARLY_CONVERSATION_EVENTS`]).
    early: Vec<(String, Value)>,
    /// What the client answers itself (a full queue, a cancelled message),
    /// delivered with the next frames.
    local: Vec<PeerEvent>,
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
        OctosPeer {
            transport,
            next_req: 1,
            pending: HashSet::new(),
            calls: HashMap::new(),
            opening: HashSet::new(),
            contexts: HashSet::new(),
            turns: HashMap::new(),
            early: Vec::new(),
            local: Vec::new(),
        }
    }

    /// Send a request; its id correlates the events and the reply.
    ///
    /// `octos.turn.start` goes through the context's queue, as with
    /// [`OctosPeer::start_turn`]; a full queue is answered at once with an
    /// error [`PeerEvent::Reply`] (`queue_full: …`) on the next
    /// [`OctosPeer::handle_event`]. `octos.context.close` and
    /// `octos.turn.interrupt` cancel the context's waiting messages.
    pub fn request(&mut self, method: &str, args: Value) -> u64 {
        let req_id = self.next_req;
        self.next_req += 1;
        let context = get_str(&args, "context");
        match (method, context) {
            ("octos.turn.start", Some(context)) => {
                if let Err(e) = self.submit_turn(req_id, &context, args) {
                    self.answer_locally(req_id, e.to_string());
                }
                return req_id;
            }
            ("octos.session.open", _) => {
                self.opening.insert(req_id);
            }
            ("octos.context.close", Some(context)) => {
                self.forget_context(&context, "cancelled: the conversation was closed");
            }
            ("octos.turn.interrupt", Some(context)) => {
                self.cancel_queued(&context, "cancelled: the conversation was stopped");
            }
            _ => {}
        }
        self.pending.insert(req_id);
        self.send(PeerUp::Request { req_id, method: method.to_string(), args });
        req_id
    }

    /// `octos.session.open`: a request context for `client` (the app's own
    /// label, e.g. a mini app id) on the current account. Replies
    /// `{"context": id}`. Without a `client` it is the app's conversation,
    /// whose events in both lanes arrive as [`PeerEvent::Conversation`].
    pub fn open_session(&mut self, client: Option<&str>) -> u64 {
        let args = json::obj(vec![("client", client.map(json::s).unwrap_or(Value::Null))]);
        self.request("octos.session.open", args)
    }

    /// A message in `context`: sent now, or queued behind the context's
    /// running turn and sent when that turn answers. What started it is
    /// left unsaid (the host counts it as unknown); see
    /// [`OctosPeer::start_turn_from`].
    pub fn start_turn(&mut self, context: &str, text: &str) -> Result<u64, PeerTurnError> {
        self.start_turn_args(context, json::obj(vec![("context", json::s(context)), ("text", json::s(text))]))
    }

    /// [`OctosPeer::start_turn`] saying what started the turn: `person`
    /// (the person in the app's UI), `app`, `schedule`, `background` or
    /// `incoming`.
    pub fn start_turn_from(&mut self, context: &str, text: &str, trigger: &str) -> Result<u64, PeerTurnError> {
        self.start_turn_args(
            context,
            json::obj(vec![("context", json::s(context)), ("text", json::s(text)), ("trigger", json::s(trigger))]),
        )
    }

    fn start_turn_args(&mut self, context: &str, args: Value) -> Result<u64, PeerTurnError> {
        let req_id = self.next_req;
        self.submit_turn(req_id, context, args)?;
        self.next_req += 1;
        Ok(req_id)
    }

    fn submit_turn(&mut self, req_id: u64, context: &str, args: Value) -> Result<(), PeerTurnError> {
        let lane = self.turns.entry(context.to_string()).or_default();
        if lane.running.is_none() {
            lane.running = Some(req_id);
        } else if lane.queued.len() >= MAX_QUEUED_TURNS {
            return Err(PeerTurnError::QueueFull { context: context.to_string(), limit: MAX_QUEUED_TURNS });
        } else {
            lane.queued.push_back((req_id, args));
            self.pending.insert(req_id);
            return Ok(());
        }
        self.pending.insert(req_id);
        self.send(PeerUp::Request { req_id, method: "octos.turn.start".into(), args });
        Ok(())
    }

    /// Messages waiting on `context` behind its running turn.
    pub fn queued(&self, context: &str) -> usize {
        self.turns.get(context).map_or(0, |l| l.queued.len())
    }

    pub fn history(&mut self, context: &str) -> u64 {
        self.request("octos.session.history", json::obj(vec![("context", json::s(context))]))
    }

    /// Stop: the context's running turn (on the app's conversation, both
    /// lanes'); the messages still waiting on it are cancelled.
    pub fn interrupt(&mut self, context: &str) -> u64 {
        self.request("octos.turn.interrupt", json::obj(vec![("context", json::s(context))]))
    }

    /// Close a context: its waiting messages and its running turn are
    /// answered here with an error (the host sends nothing more for them),
    /// and its conversation events stop.
    pub fn close_context(&mut self, context: &str) -> u64 {
        self.request("octos.context.close", json::obj(vec![("context", json::s(context))]))
    }

    /// Whether `context` was opened on this link and is not closed.
    pub fn is_open(&self, context: &str) -> bool {
        self.contexts.contains(context)
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

    /// Requests still waiting for their reply (queued messages included).
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

    /// One hosted frame read without a `Cx` (a host that routes studio
    /// frames itself, or a test): what it means for the app, empty for a
    /// frame that is not the peer link's. Answers the client made itself
    /// come first.
    pub fn receive_frame(&mut self, frame: &str) -> Vec<PeerEvent> {
        self.accept(PeerDown::parse(frame).into_iter().collect())
    }

    fn answer_locally(&mut self, req_id: u64, error: String) {
        self.local.push(PeerEvent::Reply { req_id, result: Err(error) });
        SignalToUI::set_ui_signal();
    }

    /// Cancel what waits on `context`; the running turn keeps going.
    fn cancel_queued(&mut self, context: &str, why: &str) {
        let queued = match self.turns.get_mut(context) {
            Some(lane) => std::mem::take(&mut lane.queued),
            None => return,
        };
        for (req_id, _) in queued {
            if self.pending.remove(&req_id) {
                self.answer_locally(req_id, why.to_string());
            }
        }
    }

    /// The context is gone (closed by the app or the host): what waits and
    /// what runs are answered here, and its events stop.
    fn forget_context(&mut self, context: &str, why: &str) {
        self.cancel_queued(context, why);
        if let Some(lane) = self.turns.remove(context) {
            if let Some(req_id) = lane.running {
                if self.pending.remove(&req_id) {
                    self.answer_locally(req_id, why.to_string());
                }
            }
        }
        self.contexts.remove(context);
        self.early.retain(|(c, _)| c != context);
    }

    /// A turn answered: the next waiting message of its context is sent.
    fn turn_answered(&mut self, req_id: u64) {
        let Some(context) = self.turns.iter().find(|(_, l)| l.running == Some(req_id)).map(|(c, _)| c.clone()) else {
            return;
        };
        let lane = self.turns.get_mut(&context).expect("present");
        lane.running = None;
        // A message whose request was cancelled meanwhile is skipped.
        while let Some((next, args)) = self.turns.get_mut(&context).and_then(|l| l.queued.pop_front()) {
            if self.pending.contains(&next) {
                self.turns.get_mut(&context).expect("present").running = Some(next);
                self.send(PeerUp::Request { req_id: next, method: "octos.turn.start".into(), args });
                return;
            }
        }
    }

    fn conversation(&mut self, context: String, event: Value, out: &mut Vec<PeerEvent>) {
        if self.contexts.contains(&context) {
            if let Some(e) = ConversationEvent::from_frame(&context, event) {
                out.push(PeerEvent::Conversation(e));
            }
        } else if !self.opening.is_empty() && self.early.len() < MAX_EARLY_CONVERSATION_EVENTS {
            // The host follows a conversation before its open answers.
            self.early.push((context, event));
        }
    }

    fn accept(&mut self, frames: Vec<PeerDown>) -> Vec<PeerEvent> {
        let mut out = std::mem::take(&mut self.local);
        for frame in frames {
            match frame {
                PeerDown::Reply { req_id, result } => {
                    if !self.pending.remove(&req_id) {
                        continue;
                    }
                    let opened = self.opening.remove(&req_id);
                    let context = match (&result, opened) {
                        (Ok(data), true) => get_str(data, "context").filter(|c| id_ok(c)),
                        _ => None,
                    };
                    out.push(PeerEvent::Reply { req_id, result });
                    if let Some(context) = context {
                        self.contexts.insert(context.clone());
                        let (mine, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut self.early).into_iter().partition(|(c, _)| *c == context);
                        self.early = rest;
                        for (c, e) in mine {
                            self.conversation(c, e, &mut out);
                        }
                    }
                    if self.opening.is_empty() {
                        self.early.clear();
                    }
                    self.turn_answered(req_id);
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
                PeerDown::ContextClosed { context, reason } => {
                    self.forget_context(&context, &format!("cancelled: the host closed the conversation ({reason})"));
                    out.append(&mut self.local);
                    out.push(PeerEvent::ContextClosed { context, reason });
                }
                PeerDown::Conversation { context, event } => self.conversation(context, event, &mut out),
            }
        }
        out.append(&mut self.local);
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

    /// Hand one frame the shell wrote to the client's in-process end.
    fn feed(link: &PeerLink, frame: &str) {
        link.down.send(PeerDown::parse(frame).unwrap_or_else(|| panic!("the client reads the shell's frame: {frame}"))).unwrap();
    }

    fn sent(link: &PeerLink) -> Vec<(u64, String)> {
        link.up
            .try_iter()
            .filter_map(|up| match up {
                PeerUp::Request { req_id, method, args } => {
                    let text = get_str(&args, "text").unwrap_or_default();
                    Some((req_id, format!("{method} {text}").trim().to_string()))
                }
                _ => None,
            })
            .collect()
    }

    fn open_conversation(peer: &mut OctosPeer, link: &PeerLink, context: &str) {
        let open = peer.open_session(None);
        sent(link);
        feed(link, &format!(r#"{{"octos_peer":{{"data":{{"context":"{context}","session":{{"conversation":true,"open":true}}}},"down":"reply","ok":true,"req_id":{open}}}}}"#));
        let events = peer.test_drain();
        assert!(matches!(events.as_slice(), [PeerEvent::Reply { result: Ok(_), .. }]), "{events:?}");
        assert!(peer.is_open(context));
    }

    /// Conversation frames RECORDED from the shell (OctoSense
    /// `peer_link/link.rs` on a real octos kernel, the test
    /// `real_kernel_a_process_app_hears_the_system_agents_lane_live`): the
    /// system agent's `peer_send_input` turn on the app's agent, as the
    /// app's process got them, in arrival order (`turn_terminal`, seq 4,
    /// came before `assistant_delta`, seq 3). Progress events left out.
    const SYSTEM_AGENT_TURN: [&str; 4] = [
        r#"{"octos_peer":{"context":"pl7-1","down":"conversation","event":{"lane":"system_agent","method":"turn/started","params":{"session_id":"_main:api:octosense#peer-news-22a12f90","timestamp":"2026-09-30T18:37:09.291003Z","topic":"peer-news-22a12f90","turn_id":"01a0f39b-311f-7253-b014-c8b253da2226"},"request":{"speaker":{"kind":"system_agent"},"text":"SHOW_SHARED"},"speaker":{"kind":"system_agent"}}}}"#,
        r#"{"octos_peer":{"context":"pl7-1","down":"conversation","event":{"display_text":"SHOW_SHARED","lane":"system_agent","method":"projection/envelope","params":{"cursor":{"seq":5,"stream":"_main:api:octosense#peer-news-22a12f90\u0000~cwd-6c36f355926b57dc"},"payload":{"data":{"text":"[from the system agent] SHOW_SHARED"},"type":"user_message"},"seq":1,"session_id":"_main:api:octosense","thread_id":"01a0f39b-311f-7253-b014-c8b253da2226","topic":"peer-news-22a12f90","turn_id":"01a0f39b-311f-7253-b014-c8b253da2226"},"speaker":{"kind":"system_agent"}}}}"#,
        r#"{"octos_peer":{"context":"pl7-1","down":"conversation","event":{"lane":"system_agent","method":"projection/envelope","params":{"cursor":{"seq":14,"stream":"_main:api:octosense#peer-news-22a12f90\u0000~cwd-6c36f355926b57dc"},"payload":{"data":{"outcome":"completed","token_usage":{"input_tokens":10,"output_tokens":5}},"type":"turn_terminal"},"seq":4,"session_id":"_main:api:octosense","thread_id":"01a0f39b-311f-7253-b014-c8b253da2226","topic":"peer-news-22a12f90","turn_id":"01a0f39b-311f-7253-b014-c8b253da2226"},"speaker":{"kind":"system_agent"}}}}"#,
        r#"{"octos_peer":{"context":"pl7-1","down":"conversation","event":{"lane":"system_agent","method":"projection/envelope","params":{"cursor":{"seq":10,"stream":"_main:api:octosense#peer-news-22a12f90\u0000~cwd-6c36f355926b57dc"},"payload":{"data":{"assistant_segment_id":"01a0f39b-311f-7253-b014-c8b253da2226:assistant:iteration:1","text":"SHARED NONE"},"type":"assistant_delta"},"seq":3,"session_id":"_main:api:octosense","thread_id":"01a0f39b-311f-7253-b014-c8b253da2226","topic":"peer-news-22a12f90","turn_id":"01a0f39b-311f-7253-b014-c8b253da2226"},"speaker":{"kind":"system_agent"}}}}"#,
    ];

    #[test]
    fn the_system_agents_lane_arrives_live_as_conversation_events() {
        let (mut peer, link) = OctosPeer::in_process();
        open_conversation(&mut peer, &link, "pl7-1");
        for frame in SYSTEM_AGENT_TURN {
            feed(&link, frame);
        }
        // The person's lane on the same conversation, with its speaker and
        // the user message without the kernel's marker.
        feed(&link, r#"{"octos_peer":{"context":"pl7-1","down":"conversation","event":{"display_text":"focus on tech","lane":"person","method":"projection/envelope","params":{"payload":{"data":{"text":"[from the person: Notes] focus on tech"},"type":"user_message"},"session_id":"_main:api:octosense#peerctx-notes.1","turn_id":"t-p"},"speaker":{"kind":"person","label":"Notes"}}}}"#);
        // Not a context of this link: dropped.
        feed(&link, r#"{"octos_peer":{"context":"pl9-9","down":"conversation","event":{"lane":"system_agent","method":"turn/started","params":{"turn_id":"x"}}}}"#);
        let events: Vec<ConversationEvent> = peer
            .test_drain()
            .into_iter()
            .map(|e| match e {
                PeerEvent::Conversation(c) => c,
                other => panic!("only conversation events: {other:?}"),
            })
            .collect();
        assert_eq!(events.len(), 5, "{events:?}");
        let turn = Some("01a0f39b-311f-7253-b014-c8b253da2226");
        assert!(events[..4].iter().all(|e| e.is_system_agent() && e.turn_id.as_deref() == turn && e.context == "pl7-1"));
        let started = &events[0];
        assert_eq!((started.method.as_str(), started.request_text()), ("turn/started", Some("SHOW_SHARED")));
        assert_eq!(started.speaker, Some(PeerSpeaker { kind: "system_agent".into(), label: None }));
        let asked = &events[1];
        assert_eq!((asked.envelope.as_deref(), asked.display_text.as_deref()), (Some("user_message"), Some("SHOW_SHARED")), "the words without the kernel's marker");
        assert!(events[2].is_turn_end() && events[2].envelope.as_deref() == Some("turn_terminal"));
        assert!(!events[..2].iter().any(ConversationEvent::is_turn_end));
        assert_eq!((events[3].envelope.as_deref(), events[3].delta.as_deref()), (Some("assistant_delta"), Some("SHARED NONE")));
        let person = &events[4];
        assert_eq!((person.lane, person.display_text.as_deref()), (Some(PeerLane::Person), Some("focus on tech")));
        assert_eq!(person.speaker.as_ref().and_then(|s| s.label.as_deref()), Some("Notes"));
        assert_eq!(peer.pending(), 0, "a conversation event needs no open request");
        // Closed: nothing more of it.
        peer.close_context("pl7-1");
        feed(&link, SYSTEM_AGENT_TURN[1]);
        assert!(peer.test_drain().iter().all(|e| !matches!(e, PeerEvent::Conversation(_))));
    }

    #[test]
    fn conversation_events_before_the_open_answers_are_held_for_it() {
        let (mut peer, link) = OctosPeer::in_process();
        let open = peer.open_session(None);
        // The shell follows the conversation before its open replies.
        feed(&link, SYSTEM_AGENT_TURN[0]);
        feed(&link, &format!(r#"{{"octos_peer":{{"data":{{"context":"pl7-1","session":{{}}}},"down":"reply","ok":true,"req_id":{open}}}}}"#));
        feed(&link, SYSTEM_AGENT_TURN[1]);
        let events = peer.test_drain();
        assert!(matches!(&events[0], PeerEvent::Reply { req_id, result: Ok(_) } if *req_id == open), "the reply names the context first: {events:?}");
        assert!(matches!(&events[1], PeerEvent::Conversation(c) if c.method == "turn/started"));
        assert!(matches!(&events[2], PeerEvent::Conversation(c) if c.envelope.as_deref() == Some("user_message")));
        assert_eq!(events.len(), 3);
        // With no open in flight, an unknown context's events are dropped.
        feed(&link, r#"{"octos_peer":{"context":"other","down":"conversation","event":{"method":"turn/started"}}}"#);
        assert!(peer.test_drain().is_empty());
    }

    #[test]
    fn a_second_message_on_a_handle_waits_for_the_running_turn() {
        let (mut peer, link) = OctosPeer::in_process();
        open_conversation(&mut peer, &link, "c1");
        let first = peer.start_turn_from("c1", "one", "person").unwrap();
        let second = peer.start_turn_from("c1", "two", "person").unwrap();
        let third = peer.start_turn("c1", "three").unwrap();
        assert_eq!(sent(&link), vec![(first, "octos.turn.start one".to_string())], "only the first is sent");
        assert_eq!((peer.queued("c1"), peer.pending()), (2, 3));
        // Another handle's turn is not held by this one's.
        let other = peer.start_turn("c2", "elsewhere").unwrap();
        assert_eq!(sent(&link), vec![(other, "octos.turn.start elsewhere".to_string())]);
        // The first answers: the second goes, with its trigger.
        link.down.send(PeerDown::Event { req_id: first, event: json::obj(vec![("method", json::s("message/delta"))]) }).unwrap();
        link.down.send(PeerDown::Reply { req_id: first, result: Ok(Value::Null) }).unwrap();
        let events = peer.test_drain();
        assert!(matches!(events.as_slice(), [PeerEvent::TurnEvent { req_id: a, .. }, PeerEvent::Reply { req_id: b, .. }] if *a == first && *b == first), "{events:?}");
        let up: Vec<PeerUp> = link.up.try_iter().collect();
        match up.as_slice() {
            [PeerUp::Request { req_id, args, .. }] => {
                assert_eq!(*req_id, second);
                assert_eq!((args.get("text").and_then(Value::as_str), args.get("trigger").and_then(Value::as_str)), (Some("two"), Some("person")));
            }
            other => panic!("{other:?}"),
        }
        // An error answer releases the queue too.
        link.down.send(PeerDown::Reply { req_id: second, result: Err("turn failed".into()) }).unwrap();
        peer.test_drain();
        assert_eq!(sent(&link), vec![(third, "octos.turn.start three".to_string())]);
        link.down.send(PeerDown::Reply { req_id: third, result: Ok(Value::Null) }).unwrap();
        peer.test_drain();
        assert!(sent(&link).is_empty());
        assert_eq!((peer.queued("c1"), peer.pending()), (0, 1), "only c2's turn is open");
    }

    #[test]
    fn a_full_queue_refuses_with_a_clear_error() {
        let (mut peer, link) = OctosPeer::in_process();
        peer.start_turn("c1", "running").unwrap();
        for i in 0..MAX_QUEUED_TURNS {
            peer.start_turn("c1", &format!("waiting {i}")).unwrap();
        }
        let err = peer.start_turn("c1", "one too many").unwrap_err();
        assert_eq!(err, PeerTurnError::QueueFull { context: "c1".into(), limit: MAX_QUEUED_TURNS });
        assert!(err.to_string().starts_with("queue_full: 8 messages already wait"), "{err}");
        assert_eq!(peer.pending(), 1 + MAX_QUEUED_TURNS, "a refused message is not pending");
        // The raw request path answers the same, as an error reply.
        let raw = peer.request("octos.turn.start", json::obj(vec![("context", json::s("c1")), ("text", json::s("raw"))]));
        let events = peer.test_drain();
        assert!(matches!(events.as_slice(), [PeerEvent::Reply { req_id, result: Err(e) }] if *req_id == raw && e.starts_with("queue_full")), "{events:?}");
        assert_eq!(sent(&link).len(), 1, "only the running turn went out");
    }

    #[test]
    fn closing_or_stopping_cancels_what_waits() {
        let (mut peer, link) = OctosPeer::in_process();
        open_conversation(&mut peer, &link, "c1");
        let running = peer.start_turn("c1", "a").unwrap();
        let waiting = peer.start_turn("c1", "b").unwrap();
        sent(&link);
        // Stop: what waits is cancelled; the running turn gets its own answer.
        let stop = peer.interrupt("c1");
        let events = peer.test_drain();
        assert!(matches!(events.as_slice(), [PeerEvent::Reply { req_id, result: Err(e) }] if *req_id == waiting && e.contains("stopped")), "{events:?}");
        assert_eq!(sent(&link), vec![(stop, "octos.turn.interrupt".to_string())]);
        link.down.send(PeerDown::Reply { req_id: running, result: Err("interrupted".into()) }).unwrap();
        link.down.send(PeerDown::Reply { req_id: stop, result: Ok(Value::Null) }).unwrap();
        assert_eq!(peer.test_drain().len(), 2);
        assert!(sent(&link).is_empty(), "nothing waited any more");
        // Close: the running turn and what waits are answered here (the host
        // drops a closed context's replies), and a late reply is ignored.
        let running = peer.start_turn("c1", "c").unwrap();
        let waiting = peer.start_turn("c1", "d").unwrap();
        sent(&link);
        let close = peer.close_context("c1");
        let events = peer.test_drain();
        let cancelled: Vec<u64> = events
            .iter()
            .filter_map(|e| match e {
                PeerEvent::Reply { req_id, result: Err(e) } if e.contains("closed") => Some(*req_id),
                _ => None,
            })
            .collect();
        assert_eq!(cancelled, vec![waiting, running]);
        assert!(!peer.is_open("c1"));
        link.down.send(PeerDown::Reply { req_id: running, result: Ok(Value::Null) }).unwrap();
        link.down.send(PeerDown::Reply { req_id: close, result: Ok(Value::Null) }).unwrap();
        assert!(matches!(peer.test_drain().as_slice(), [PeerEvent::Reply { req_id, .. }] if *req_id == close));
        assert_eq!(sent(&link), vec![(close, "octos.context.close".to_string())], "the waiting message never went out");
        assert_eq!(peer.pending(), 0);
    }

    #[test]
    fn the_hosts_close_cancels_what_waits_too() {
        let (mut peer, link) = OctosPeer::in_process();
        open_conversation(&mut peer, &link, "c1");
        let running = peer.start_turn("c1", "a").unwrap();
        let waiting = peer.start_turn("c1", "b").unwrap();
        feed(&link, r#"{"octos_peer":{"context":"c1","down":"context_closed","reason":"signed_out"}}"#);
        let events = peer.test_drain();
        assert_eq!(events.len(), 3, "{events:?}");
        assert!(matches!(&events[0], PeerEvent::Reply { req_id, result: Err(e) } if *req_id == waiting && e.contains("signed_out")));
        assert!(matches!(&events[1], PeerEvent::Reply { req_id, .. } if *req_id == running));
        assert!(matches!(&events[2], PeerEvent::ContextClosed { context, .. } if context == "c1"));
        assert_eq!(peer.pending(), 0);
    }

    #[test]
    fn a_conversation_frame_is_well_formed_or_dropped() {
        let frame = PeerDown::Conversation { context: "pl7-1".into(), event: json::obj(vec![("method", json::s("turn/started"))]) }.to_json();
        assert_eq!(frame, r#"{"octos_peer":{"down":"conversation","context":"pl7-1","event":{"method":"turn/started"}}}"#);
        assert_eq!(PeerDown::parse(&frame).map(|d| d.to_json()), Some(frame));
        assert_eq!(PeerDown::parse(r#"{"octos_peer":{"down":"conversation","context":"c","event":[1]}}"#), None);
        assert_eq!(PeerDown::parse(r#"{"octos_peer":{"down":"conversation","event":{}}}"#), None);
        assert_eq!(PeerDown::parse(r#"{"octos_peer":{"down":"conversation","context":"","event":{}}}"#), None);
        // The older kernel events read the same way.
        let delta = ConversationEvent::from_frame("c", json::obj(vec![("method", json::s("message/delta")), ("params", json::obj(vec![("text", json::s("hi"))]))])).unwrap();
        assert_eq!((delta.delta.as_deref(), delta.envelope.as_deref(), delta.is_turn_end()), (Some("hi"), None, false));
        assert!(ConversationEvent::from_frame("c", json::obj(vec![("method", json::s("turn/completed"))])).unwrap().is_turn_end());
        // Without a method it is not an event the app can use.
        assert_eq!(ConversationEvent::from_frame("c", json::obj(vec![("lane", json::s("person"))])), None);
    }

    #[test]
    fn a_hosted_client_reads_frames_without_a_cx() {
        let mut peer = OctosPeer::hosted();
        let open = peer.open_session(None);
        let events = peer.receive_frame(&format!(r#"{{"octos_peer":{{"data":{{"context":"pl7-1"}},"down":"reply","ok":true,"req_id":{open}}}}}"#));
        assert_eq!(events.len(), 1);
        let events = peer.receive_frame(SYSTEM_AGENT_TURN[0]);
        assert!(matches!(events.as_slice(), [PeerEvent::Conversation(c)] if c.is_system_agent()), "{events:?}");
        assert!(peer.receive_frame(r#"{"wm":{"Hosted":{}}}"#).is_empty());
    }
}
