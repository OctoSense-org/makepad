use std::sync::{Arc, Mutex, OnceLock};

use makepad_live_id::LiveId;

use super::{EventSink, NetworkBackend};
use crate::types::{HttpRequest, NetworkError, WsSend};

// OpenHarmony has no stable C HTTP client API, so the real backend lives in
// makepad-platform where the ArkTS bridge (@ohos.net.http requestInStream)
// and the NAPI callback plumbing are available. makepad-platform registers it
// here at startup; until then every call reports a clear error instead of
// silently dropping traffic. This mirrors the Android platform-backend slot.
fn backend_slot() -> &'static Mutex<Option<Arc<dyn NetworkBackend>>> {
    static SLOT: OnceLock<Mutex<Option<Arc<dyn NetworkBackend>>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

pub fn register_platform_backend(backend: Arc<dyn NetworkBackend>) {
    if let Ok(mut slot) = backend_slot().lock() {
        *slot = Some(backend);
    }
}

pub fn clear_platform_backend() {
    if let Ok(mut slot) = backend_slot().lock() {
        *slot = None;
    }
}

fn platform_backend() -> Result<Arc<dyn NetworkBackend>, NetworkError> {
    if let Some(b) = backend_slot()
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().map(Arc::clone))
    {
        return Ok(b);
    }
    // Nothing ever registered an ArkTS-bridge backend, which meant every
    // cx.http_request on OpenHarmony failed silently — Splash data bindings
    // like sys.weather() rendered as "—" forever. reqwest links and works
    // fine on this target, so fall back to it rather than erroring.
    Ok(reqwest_backend())
}

fn reqwest_backend() -> Arc<dyn NetworkBackend> {
    static B: OnceLock<Arc<dyn NetworkBackend>> = OnceLock::new();
    B.get_or_init(|| Arc::new(ReqwestBackend) as Arc<dyn NetworkBackend>)
        .clone()
}

/// One pooled client for the whole process.
///
/// Building a `Client` per request re-initialises TLS every time and pools
/// nothing, which is ruinous for the map: `MapView` issues up to 6 concurrent
/// Overpass queries whose responses run past a megabyte, and against reqwest's
/// 30s DEFAULT timeout those reliably failed with "error sending request" (and,
/// when the body read was the part that timed out, a `None` body reported
/// upstream as "missing utf8 response body"). Pool the connections and allow a
/// timeout that suits a large response over mobile data.
fn shared_client(ignore_ssl_cert: bool) -> Result<reqwest::blocking::Client, reqwest::Error> {
    fn build(ignore_ssl_cert: bool) -> Result<reqwest::blocking::Client, reqwest::Error> {
        reqwest::blocking::Client::builder()
            .danger_accept_invalid_certs(ignore_ssl_cert)
            .timeout(std::time::Duration::from_secs(60))
            .connect_timeout(std::time::Duration::from_secs(15))
            .pool_max_idle_per_host(8)
            .build()
    }
    // Only the common (certificate-checking) client is cached; ignoring cert
    // errors is rare and must not leak into the shared pool.
    if ignore_ssl_cert {
        return build(true);
    }
    static CLIENT: OnceLock<Result<reqwest::blocking::Client, String>> = OnceLock::new();
    match CLIENT.get_or_init(|| build(false).map_err(|e| e.to_string())) {
        Ok(c) => Ok(c.clone()),
        // Rebuild to surface the real `reqwest::Error` (it is not Clone).
        Err(_) => build(false),
    }
}

/// Minimal HTTP-only backend for OpenHarmony, backed by reqwest on a worker
/// thread. WebSockets are not implemented here (the app uses its own client).
struct ReqwestBackend;

impl NetworkBackend for ReqwestBackend {
    fn http_start(
        &self,
        request_id: LiveId,
        request: HttpRequest,
        sink: EventSink,
    ) -> Result<(), NetworkError> {
        std::thread::spawn(move || {
            let method = match request.method {
                crate::types::HttpMethod::GET => reqwest::Method::GET,
                crate::types::HttpMethod::HEAD => reqwest::Method::HEAD,
                crate::types::HttpMethod::POST => reqwest::Method::POST,
                crate::types::HttpMethod::PUT => reqwest::Method::PUT,
                crate::types::HttpMethod::DELETE => reqwest::Method::DELETE,
                crate::types::HttpMethod::OPTIONS => reqwest::Method::OPTIONS,
                crate::types::HttpMethod::TRACE => reqwest::Method::TRACE,
                crate::types::HttpMethod::CONNECT => reqwest::Method::CONNECT,
                crate::types::HttpMethod::PATCH => reqwest::Method::PATCH,
            };
            let client = match shared_client(request.ignore_ssl_cert) {
                Ok(c) => c,
                Err(e) => {
                    let _ = sink.emit(crate::types::NetworkResponse::HttpError {
                        request_id,
                        error: crate::types::HttpError {
                            metadata_id: request.metadata_id,
                            message: format!("client build: {e}"),
                        },
                    });
                    return;
                }
            };
            let mut rb = client.request(method, &request.url);
            for (k, vals) in request.headers.iter() {
                for v in vals {
                    rb = rb.header(k.as_str(), v.as_str());
                }
            }
            if let Some(body) = request.body.clone() {
                rb = rb.body(body);
            }
            match rb.send() {
                Ok(res) => {
                    let status = res.status().as_u16();
                    let mut headers: std::collections::BTreeMap<String, Vec<String>> =
                        Default::default();
                    for (k, v) in res.headers().iter() {
                        headers
                            .entry(k.as_str().to_string())
                            .or_default()
                            .push(v.to_str().unwrap_or("").to_string());
                    }
                    // Report a failed body read as an ERROR. Mapping it to
                    // `None` makes a mid-download failure look like a served
                    // empty body, which surfaces far downstream as a confusing
                    // "missing utf8 response body" with no cause attached.
                    let body = match res.bytes() {
                        Ok(b) => Some(b.to_vec()),
                        Err(e) => {
                            let _ = sink.emit(crate::types::NetworkResponse::HttpError {
                                request_id,
                                error: crate::types::HttpError {
                                    metadata_id: request.metadata_id,
                                    message: format!("body read: {e}"),
                                },
                            });
                            return;
                        }
                    };
                    let _ = sink.emit(crate::types::NetworkResponse::HttpResponse {
                        request_id,
                        response: crate::types::HttpResponse {
                            metadata_id: request.metadata_id,
                            status_code: status,
                            headers,
                            body,
                        },
                    });
                }
                Err(e) => {
                    let _ = sink.emit(crate::types::NetworkResponse::HttpError {
                        request_id,
                        error: crate::types::HttpError {
                            metadata_id: request.metadata_id,
                            message: format!("{e}"),
                        },
                    });
                }
            }
        });
        Ok(())
    }

    fn http_cancel(&self, _request_id: LiveId) -> Result<(), NetworkError> {
        Ok(())
    }

    fn ws_open(
        &self,
        _socket_id: LiveId,
        _request: HttpRequest,
        _sink: EventSink,
    ) -> Result<(), NetworkError> {
        Err(NetworkError::Unsupported("websockets on OpenHarmony"))
    }

    fn ws_send(&self, _socket_id: LiveId, _message: WsSend) -> Result<(), NetworkError> {
        Err(NetworkError::Unsupported("websockets on OpenHarmony"))
    }

    fn ws_close(&self, _socket_id: LiveId) -> Result<(), NetworkError> {
        Err(NetworkError::Unsupported("websockets on OpenHarmony"))
    }
}

struct OhosDelegatingBackend;

impl NetworkBackend for OhosDelegatingBackend {
    fn http_start(
        &self,
        request_id: LiveId,
        request: HttpRequest,
        sink: EventSink,
    ) -> Result<(), NetworkError> {
        platform_backend()?.http_start(request_id, request, sink)
    }

    fn http_cancel(&self, request_id: LiveId) -> Result<(), NetworkError> {
        platform_backend()?.http_cancel(request_id)
    }

    fn ws_open(
        &self,
        socket_id: LiveId,
        request: HttpRequest,
        sink: EventSink,
    ) -> Result<(), NetworkError> {
        platform_backend()?.ws_open(socket_id, request, sink)
    }

    fn ws_send(&self, socket_id: LiveId, message: WsSend) -> Result<(), NetworkError> {
        platform_backend()?.ws_send(socket_id, message)
    }

    fn ws_close(&self, socket_id: LiveId) -> Result<(), NetworkError> {
        platform_backend()?.ws_close(socket_id)
    }
}

pub(crate) fn create_backend() -> Arc<dyn NetworkBackend> {
    Arc::new(OhosDelegatingBackend)
}
