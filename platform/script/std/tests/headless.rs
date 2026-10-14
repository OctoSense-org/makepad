use makepad_script_std::makepad_network::{
    EventSink, HttpRequest, HttpResponse, NetworkBackend, NetworkResponse, NetworkRuntime, WsSend,
};
use makepad_script_std::makepad_script::{makepad_live_id::LiveId, script, ScriptArrayStorage, *};
use makepad_script_std::{pump_network_runtime, script_mod, with_vm, ScriptStd};
use std::collections::BTreeMap;
use std::sync::Arc;

struct TestBackend;

impl NetworkBackend for TestBackend {
    fn http_start(
        &self,
        request_id: LiveId,
        _request: HttpRequest,
        sink: EventSink,
    ) -> Result<(), makepad_script_std::makepad_network::NetworkError> {
        let response = HttpResponse::new(LiveId(42), 200, BTreeMap::new(), Some(b"ok".to_vec()));
        sink.emit(NetworkResponse::HttpResponse {
            request_id,
            response,
        })
    }

    fn http_cancel(
        &self,
        _request_id: LiveId,
    ) -> Result<(), makepad_script_std::makepad_network::NetworkError> {
        Ok(())
    }

    fn ws_open(
        &self,
        _socket_id: LiveId,
        _request: HttpRequest,
        _sink: EventSink,
    ) -> Result<(), makepad_script_std::makepad_network::NetworkError> {
        Ok(())
    }

    fn ws_send(
        &self,
        _socket_id: LiveId,
        _message: WsSend,
    ) -> Result<(), makepad_script_std::makepad_network::NetworkError> {
        Ok(())
    }

    fn ws_close(
        &self,
        _socket_id: LiveId,
    ) -> Result<(), makepad_script_std::makepad_network::NetworkError> {
        Ok(())
    }
}

#[test]
fn headless_http_request_resolves_promise_through_script_std() {
    let runtime = Arc::new(NetworkRuntime::with_backend(Arc::new(TestBackend)));
    let mut host = ScriptVmHost::new((), ScriptStd::with_network_runtime(runtime));
    host.script_vm = Some(Box::new(ScriptVmBase::new()));

    let promise = with_vm(&mut host, |vm| {
        script_mod(vm);
        vm.eval(script! {
            use mod.std
            use mod.net
            let p = std.promise()
            let req = net.HttpRequest{
                url: "https://example.com"
                method: net.HttpMethod.GET
            }
            net.http_request(req) do net.HttpEvents{
                on_response: |res| p.resolve(res.status_code)
                on_error: |_err| p.resolve(-1)
            }
            p
        })
    });

    let promise = promise
        .as_handle()
        .expect("script should return a promise handle");
    assert_eq!(host.std.data.http_requests.len(), 1);

    let responses = pump_network_runtime(&mut host);
    assert_eq!(responses.len(), 1);
    assert!(host.std.data.http_requests.is_empty());

    let tasks = host.std.data.tasks.tasks.borrow();
    let task = tasks
        .iter()
        .find(|task| task.handle == promise)
        .expect("promise task should exist");
    match host
        .script_vm
        .as_ref()
        .unwrap()
        .heap
        .array_storage(task.queue.as_array())
    {
        ScriptArrayStorage::ScriptValue(values) => {
            assert_eq!(values.len(), 1);
            assert_eq!(values[0].as_f64(), Some(200.0));
        }
        _ => panic!("unexpected promise queue storage kind"),
    }
}

/// The script-side `net.HttpRequest{..}` constructor exposes `ignore_ssl_cert`
/// and the flag reaches the backend untouched — loopback/dev gateways with
/// self-signed certs depend on it. (Field presence was once mis-reported as
/// missing; this pins the round-trip so the question never needs re-asking.)
#[test]
fn headless_http_request_ignore_ssl_cert_roundtrip() {
    struct CaptureBackend {
        seen: std::sync::Mutex<Option<bool>>,
    }

    impl NetworkBackend for CaptureBackend {
        fn http_start(
            &self,
            request_id: LiveId,
            request: HttpRequest,
            sink: EventSink,
        ) -> Result<(), makepad_script_std::makepad_network::NetworkError> {
            *self.seen.lock().unwrap() = Some(request.ignore_ssl_cert);
            let response =
                HttpResponse::new(LiveId(42), 200, BTreeMap::new(), Some(b"ok".to_vec()));
            sink.emit(NetworkResponse::HttpResponse {
                request_id,
                response,
            })
        }

        fn http_cancel(
            &self,
            _request_id: LiveId,
        ) -> Result<(), makepad_script_std::makepad_network::NetworkError> {
            Ok(())
        }

        fn ws_open(
            &self,
            _socket_id: LiveId,
            _request: HttpRequest,
            _sink: EventSink,
        ) -> Result<(), makepad_script_std::makepad_network::NetworkError> {
            Ok(())
        }

        fn ws_send(
            &self,
            _socket_id: LiveId,
            _message: WsSend,
        ) -> Result<(), makepad_script_std::makepad_network::NetworkError> {
            Ok(())
        }

        fn ws_close(
            &self,
            _socket_id: LiveId,
        ) -> Result<(), makepad_script_std::makepad_network::NetworkError> {
            Ok(())
        }
    }

    let backend = Arc::new(CaptureBackend {
        seen: std::sync::Mutex::new(None),
    });
    let runtime = Arc::new(NetworkRuntime::with_backend(backend.clone()));
    let mut host = ScriptVmHost::new((), ScriptStd::with_network_runtime(runtime));
    host.script_vm = Some(Box::new(ScriptVmBase::new()));

    let promise = with_vm(&mut host, |vm| {
        script_mod(vm);
        vm.eval(script! {
            use mod.std
            use mod.net
            let p = std.promise()
            let req = net.HttpRequest{
                url: "https://127.0.0.1:8195/"
                method: net.HttpMethod.GET
                headers: {"User-Agent": "makepad-test/1.0"}
                ignore_ssl_cert: true
            }
            net.http_request(req) do net.HttpEvents{
                on_response: |res| p.resolve(res.status_code)
                on_error: |_err| p.resolve(-1)
            }
            p
        })
    });
    assert!(!promise.is_err(), "script eval failed: {promise:?}");

    let responses = pump_network_runtime(&mut host);
    assert_eq!(responses.len(), 1);
    assert_eq!(
        *backend.seen.lock().unwrap(),
        Some(true),
        "ignore_ssl_cert did not reach the backend"
    );
}
