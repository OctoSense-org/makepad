//! OpenHarmony network backend.
//!
//! No platform shim registers a backend here yet, so this covers what the
//! standard library can do on its own: clear-text websockets through the
//! plain-TCP implementation (the Studio hub's `PlainTcp` transport, `ws://`).
//! TLS websockets and HTTP requests still need a registered platform backend
//! (`register_platform_backend`), as on Android.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use makepad_live_id::LiveId;

use crate::backend::{EventSink, NetworkBackend};
use crate::plain_web_socket::PlainWebSocket;
use crate::types::{
    HttpRequest, NetworkError, NetworkResponse, WebSocketMessage, WebSocketTransport, WsMessage,
    WsSend,
};

pub(crate) struct OhosBackend {
    sockets: Mutex<HashMap<LiveId, PlainWebSocket>>,
}

impl OhosBackend {
    fn new() -> Self {
        Self {
            sockets: Mutex::new(HashMap::new()),
        }
    }
}

impl NetworkBackend for OhosBackend {
    fn http_start(
        &self,
        _request_id: LiveId,
        _request: HttpRequest,
        _sink: EventSink,
    ) -> Result<(), NetworkError> {
        Err(NetworkError::backend(
            "HTTP on OpenHarmony needs a registered platform backend",
        ))
    }

    fn http_cancel(&self, _request_id: LiveId) -> Result<(), NetworkError> {
        Err(NetworkError::backend(
            "HTTP on OpenHarmony needs a registered platform backend",
        ))
    }

    fn ws_open(
        &self,
        socket_id: LiveId,
        request: HttpRequest,
        sink: EventSink,
    ) -> Result<(), NetworkError> {
        let split = request.split_url();
        let use_plain = match request.websocket_transport {
            WebSocketTransport::PlainTcp => true,
            WebSocketTransport::Platform => false,
            WebSocketTransport::Auto => matches!(split.proto, "ws" | "http"),
        };
        if !use_plain {
            return Err(NetworkError::backend(
                "TLS websockets on OpenHarmony need a registered platform backend",
            ));
        }
        let (sender, receiver) = std::sync::mpsc::channel::<WebSocketMessage>();
        let socket = PlainWebSocket::open(socket_id, request, sender);
        {
            let mut sockets = self
                .sockets
                .lock()
                .map_err(|_| NetworkError::backend("ohos websocket lock poisoned"))?;
            sockets.insert(socket_id, socket);
        }
        let _ = sink.emit(NetworkResponse::WsOpened { socket_id });
        std::thread::spawn(move || {
            while let Ok(message) = receiver.recv() {
                if sink.emit(map_ws_event(socket_id, message)).is_err() {
                    break;
                }
            }
        });
        Ok(())
    }

    fn ws_send(&self, socket_id: LiveId, message: WsSend) -> Result<(), NetworkError> {
        let mut sockets = self
            .sockets
            .lock()
            .map_err(|_| NetworkError::backend("ohos websocket lock poisoned"))?;
        let socket = sockets
            .get_mut(&socket_id)
            .ok_or_else(|| NetworkError::backend(format!("ohos websocket {socket_id} not open")))?;
        let outbound = match message {
            WsSend::Binary(data) => WebSocketMessage::Binary(data),
            WsSend::Text(data) => WebSocketMessage::String(data),
        };
        socket
            .send_message(outbound)
            .map_err(|_| NetworkError::backend("ohos websocket send failed"))
    }

    fn ws_close(&self, socket_id: LiveId) -> Result<(), NetworkError> {
        let mut sockets = self
            .sockets
            .lock()
            .map_err(|_| NetworkError::backend("ohos websocket lock poisoned"))?;
        if let Some(mut socket) = sockets.remove(&socket_id) {
            socket.close();
        }
        Ok(())
    }
}

fn map_ws_event(socket_id: LiveId, message: WebSocketMessage) -> NetworkResponse {
    match message {
        WebSocketMessage::Error(message) => NetworkResponse::WsError { socket_id, message },
        WebSocketMessage::Binary(data) => NetworkResponse::WsMessage {
            socket_id,
            message: WsMessage::Binary(data),
        },
        WebSocketMessage::String(data) => NetworkResponse::WsMessage {
            socket_id,
            message: WsMessage::Text(data),
        },
        WebSocketMessage::Opened => NetworkResponse::WsOpened { socket_id },
        WebSocketMessage::Closed => NetworkResponse::WsClosed { socket_id },
    }
}

pub(crate) fn create_backend() -> Arc<dyn NetworkBackend> {
    Arc::new(OhosBackend::new())
}
