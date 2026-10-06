//! An app's connection to its host's octos kernel, in the kernel's own UI
//! protocol (OUP, `octos-ui/v1alpha1`; OctoSense ADR 0003).
//!
//! Most apps reach octos only through their agent ([`crate::peer`]). An app
//! that is itself a full octos client (OctosCode) speaks OUP: JSON-RPC text
//! frames. Inside a shell it never dials the kernel and never holds or
//! stores a token. It opens a port with [`OctosUiPort::open`], and the host
//! takes the other end, parked in [`PendingUiPorts`], right after the module
//! code that opened it ran. As with a peer link, the port's identity is the
//! instance that opened it, never anything a frame says, and the host decides
//! what the port may do: it forwards the frames it allows to its kernel and
//! answers the others itself. A host that grants the app no port closes it
//! ([`UiPortEvent::Closed`]).
//!
//! The port outlives a kernel restart: the host connects again behind it and
//! says so with [`UiPortEvent::Reset`]. Requests in flight then get no reply,
//! and the app opens its sessions again, as a client does after a reconnect.
//!
//! In-process only: an app a shell hosts as its own process gets a port that
//! is closed at once.

use makepad_platform::thread::SignalToUI;
use makepad_platform::Cx;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};

/// What the host sends down a port.
#[derive(Clone, Debug, PartialEq)]
pub enum UiPortEvent {
    /// A kernel frame for this app: a reply, a notification or a request.
    Frame(String),
    /// The kernel behind the port restarted. The port stays open.
    Reset { reason: String },
    /// The host closed the port for good: it was not granted, or the
    /// instance is closing. Nothing more arrives, and sends fail.
    Closed { reason: String },
}

/// The host's end of a port.
pub struct UiPortLink {
    /// App → host: the app's JSON-RPC frames, one per message.
    pub up: Receiver<String>,
    /// Host → app.
    pub down: UiPortDown,
}

/// Sends events down a port, from any thread.
#[derive(Clone)]
pub struct UiPortDown(Sender<UiPortEvent>);

impl UiPortDown {
    /// Send `event` to the app and wake its UI thread, for an app that reads
    /// the port there. `false`: the app dropped its end.
    pub fn send(&self, event: UiPortEvent) -> bool {
        let sent = self.0.send(event).is_ok();
        SignalToUI::set_ui_signal();
        sent
    }
}

/// Ports opened since the host last looked, parked on `Cx`.
#[derive(Default)]
pub struct PendingUiPorts {
    pub ports: Vec<UiPortLink>,
}

impl PendingUiPorts {
    pub fn take(&mut self) -> Vec<UiPortLink> {
        std::mem::take(&mut self.ports)
    }
}

/// An app's end of its port. It may move to a worker thread that runs the
/// app's connection and waits there ([`OctosUiPort::recv`]), or stay on the
/// UI thread and poll ([`OctosUiPort::try_recv`]); [`OctosUiPort::sender`]
/// writes from any other thread.
pub struct OctosUiPort {
    up: Sender<String>,
    down: Receiver<UiPortEvent>,
    closed: Option<String>,
}

/// Writes frames to a port from any thread.
#[derive(Clone)]
pub struct UiPortSender(Sender<String>);

impl UiPortSender {
    /// Send one JSON-RPC frame. `Err` once the host closed the port.
    pub fn send(&self, frame: impl Into<String>) -> Result<(), String> {
        self.0.send(frame.into()).map_err(|_| HOST_CLOSED.to_string())
    }
}

const HOST_CLOSED: &str = "the host closed the port";

impl OctosUiPort {
    /// Open a port to the host's kernel.
    pub fn open(cx: &mut Cx) -> OctosUiPort {
        let (port, link) = OctosUiPort::in_process();
        if cx.in_makepad_studio() {
            link.down.send(UiPortEvent::Closed { reason: "an app hosted as its own process has no kernel port".into() });
            return port;
        }
        cx.global::<PendingUiPorts>().ports.push(link);
        port
    }

    /// A port and the host's end of it, for a host that hands it out itself.
    pub fn in_process() -> (OctosUiPort, UiPortLink) {
        let (up_tx, up_rx) = channel();
        let (down_tx, down_rx) = channel();
        (OctosUiPort { up: up_tx, down: down_rx, closed: None }, UiPortLink { up: up_rx, down: UiPortDown(down_tx) })
    }

    /// A sender for the port's frames, for a client that writes from another
    /// thread than the one reading.
    pub fn sender(&self) -> UiPortSender {
        UiPortSender(self.up.clone())
    }

    /// Send one JSON-RPC frame. `Err` once the port is closed.
    pub fn send(&mut self, frame: impl Into<String>) -> Result<(), String> {
        if let Some(reason) = &self.closed {
            return Err(reason.clone());
        }
        if self.up.send(frame.into()).is_err() {
            return Err(self.close(HOST_CLOSED));
        }
        Ok(())
    }

    /// The next event, waiting for it. Once the port is closed this answers
    /// [`UiPortEvent::Closed`] at once.
    pub fn recv(&mut self) -> UiPortEvent {
        if let Some(reason) = &self.closed {
            return UiPortEvent::Closed { reason: reason.clone() };
        }
        match self.down.recv() {
            Ok(event) => self.note(event),
            Err(_) => UiPortEvent::Closed { reason: self.close(HOST_CLOSED) },
        }
    }

    /// The next event if one is waiting, for a client that reads the port
    /// on its UI thread (the host wakes it with a signal).
    pub fn try_recv(&mut self) -> Option<UiPortEvent> {
        if let Some(reason) = &self.closed {
            return Some(UiPortEvent::Closed { reason: reason.clone() });
        }
        match self.down.try_recv() {
            Ok(event) => Some(self.note(event)),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(UiPortEvent::Closed { reason: self.close(HOST_CLOSED) }),
        }
    }

    /// Why the port is closed, once it is.
    pub fn closed(&self) -> Option<&str> {
        self.closed.as_deref()
    }

    fn note(&mut self, event: UiPortEvent) -> UiPortEvent {
        if let UiPortEvent::Closed { reason } = &event {
            self.closed = Some(reason.clone());
        }
        event
    }

    fn close(&mut self, reason: &str) -> String {
        self.closed.get_or_insert_with(|| reason.to_string()).clone()
    }
}
