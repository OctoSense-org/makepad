//! Socket streams for the process. OpenHarmony has no TLS library for native code, so a TLS
//! stream is one of the ability's `TLSSocket`s (ArkTS, the system's TLS stack and its CA
//! store): the calls here ask the ability through `napi::call` and wait for its answers, which
//! arrive on the UI thread. The system checks the certificate's chain; that the certificate is
//! for the host is checked here (`cert_host`). A plain stream is std's. Any thread but the UI
//! thread may call.
//!
//! A TLSSocket hands over what it receives as it comes and cannot be told to wait, so bytes
//! the app has not read yet are held here, without a bound.
use super::{napi, shell::lock};
use std::collections::{HashMap, VecDeque};
use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// How long a TLS stream may take to connect and shake hands.
const OPEN_TIMEOUT: Duration = Duration::from_secs(30);

enum Stream { Plain(Arc<TcpStream>), Tls(Arc<Tls>) }
struct Tls { state: Mutex<TlsState>, changed: Condvar }
#[derive(Default)]
struct TlsState {
    /// the host the server's certificate must name (none for a stream that skips validation)
    host: Option<String>,
    /// the handshake's result, once the ability has it
    open: Option<bool>,
    received: VecDeque<u8>,
    closed: bool,
    /// sends asked of the ability and sends it answered, in order
    asked: u64,
    answered: u64,
    send_failed: bool,
    /// the server's leaf certificate (DER)
    certificate: Option<Vec<u8>>,
    /// write and read, in milliseconds (0: none)
    timeouts: [i32; 2],
}
impl Tls {
    /// Waits for `done`, until `deadline` (false then).
    fn wait<'a>(&'a self, mut s: MutexGuard<'a, TlsState>, deadline: Option<Instant>, done: impl Fn(&TlsState) -> bool) -> (MutexGuard<'a, TlsState>, bool) {
        loop {
            if done(&s) { return (s, true); }
            s = match deadline {
                None => self.changed.wait(s).unwrap_or_else(|e| e.into_inner()),
                Some(deadline) => {
                    let Some(left) = deadline.checked_duration_since(Instant::now()).filter(|left| !left.is_zero()) else { return (s, false) };
                    self.changed.wait_timeout(s, left).unwrap_or_else(|e| e.into_inner()).0
                }
            };
        }
    }
}
fn deadline(ms: i32) -> Option<Instant> { (ms > 0).then(|| Instant::now() + Duration::from_millis(ms as u64)) }

fn streams() -> &'static Mutex<HashMap<i64, Stream>> {
    static STREAMS: OnceLock<Mutex<HashMap<i64, Stream>>> = OnceLock::new();
    STREAMS.get_or_init(Default::default)
}
fn tls(id: i64) -> Option<Arc<Tls>> {
    match lock(streams()).get(&id) { Some(Stream::Tls(tls)) => Some(tls.clone()), _ => None }
}
fn plain(id: i64) -> Option<Arc<TcpStream>> {
    match lock(streams()).get(&id) { Some(Stream::Plain(tcp)) => Some(tcp.clone()), _ => None }
}

/// Opens a stream (TLS when `tls`); `id` names it for the other calls. False when it failed.
pub fn open(id: i64, host: &str, port: i32, tls: bool, ignore_cert: bool) -> bool {
    let Ok(port) = u16::try_from(port) else { return false };
    if !tls {
        let Ok(tcp) = TcpStream::connect((host, port)) else { return false };
        let _ = tcp.set_nodelay(true);
        lock(streams()).insert(id, Stream::Plain(Arc::new(tcp)));
        return true;
    }
    if host.is_empty() || host.contains(['\n', '\r', '\0']) { return false; }
    let state = TlsState { host: (!ignore_cert).then(|| host.to_string()), ..Default::default() };
    let stream = Arc::new(Tls { state: Mutex::new(state), changed: Condvar::new() });
    lock(streams()).insert(id, Stream::Tls(stream.clone()));
    let ok = napi::call("socketOpen", &format!("{id}\n{host}\n{port}\n{}", ignore_cert as u8)) && {
        let (s, _) = stream.wait(lock(&stream.state), Some(Instant::now() + OPEN_TIMEOUT), |s| s.open.is_some());
        s.open == Some(true)
    };
    if !ok { close(id); }
    ok
}

/// Reads into `out` (blocking until some bytes arrive or the read timeout passes). `Some(n)`: `n`
/// bytes, 0 for a read timeout; `None`: the stream is closed or failed.
pub fn read_into(id: i64, out: &mut [u8]) -> Option<usize> {
    if let Some(tcp) = plain(id) {
        return match (&*tcp).read(out) {
            Ok(0) => None,
            Ok(n) => Some(n),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted) => Some(0),
            Err(_) => None,
        };
    }
    let stream = tls(id)?;
    let s = lock(&stream.state);
    let until = deadline(s.timeouts[1]);
    let (mut s, got) = stream.wait(s, until, |s| !s.received.is_empty() || s.closed);
    if !got { return Some(0); }
    if s.received.is_empty() { return None; }
    let n = s.received.len().min(out.len());
    for (to, from) in out.iter_mut().zip(s.received.drain(..n)) { *to = from; }
    Some(n)
}

/// Writes all of `data`. Returns its length, or -1.
pub fn write(id: i64, data: &[u8]) -> i32 {
    let len = data.len().min(i32::MAX as usize);
    if let Some(tcp) = plain(id) {
        return if (&*tcp).write_all(&data[..len]).is_ok() { len as i32 } else { -1 };
    }
    let Some(stream) = tls(id) else { return -1 };
    let (turn, until) = {
        let mut s = lock(&stream.state);
        if s.closed || s.send_failed { return -1; }
        s.asked += 1;
        (s.asked, deadline(s.timeouts[0]))
    };
    if !napi::call_with("socketSend", &id.to_string(), &data[..len]) { return -1; }
    // The ability answers each send once the system took it, so a sender keeps the stream's pace.
    let (s, got) = stream.wait(lock(&stream.state), until, |s| s.answered >= turn || s.closed);
    if got && s.answered >= turn && !s.send_failed { len as i32 } else { -1 }
}

/// Sets the read (`read`) or write timeout in milliseconds (0: none).
pub fn set_timeout(id: i64, read: bool, ms: i32) {
    if let Some(tcp) = plain(id) {
        let timeout = (ms > 0).then(|| Duration::from_millis(ms as u64));
        let _ = if read { tcp.set_read_timeout(timeout) } else { tcp.set_write_timeout(timeout) };
    } else if let Some(stream) = tls(id) {
        lock(&stream.state).timeouts[read as usize] = ms;
    }
}

/// The server's leaf certificate (DER) of a TLS stream.
pub fn peer_certificate(id: i64) -> Option<Vec<u8>> { lock(&tls(id)?.state).certificate.clone() }

pub fn close(id: i64) {
    match lock(streams()).remove(&id) {
        Some(Stream::Plain(tcp)) => { let _ = tcp.shutdown(Shutdown::Both); }
        Some(Stream::Tls(stream)) => {
            lock(&stream.state).closed = true;
            stream.changed.notify_all();
            napi::call("socketClose", &id.to_string());
        }
        None => {}
    }
}

/// What the ability reports (`napi.rs`), on the UI thread.
fn report(id: i64, f: impl FnOnce(&mut TlsState)) {
    if let Some(stream) = tls(id) {
        f(&mut lock(&stream.state));
        stream.changed.notify_all();
    }
}
pub(super) fn opened(id: i64, ok: bool, certificate: &[u8]) {
    report(id, |s| {
        let named = s.host.as_deref().is_none_or(|host| crate::backend::cert_host::certificate_names_host(certificate, host));
        if ok && !named { eprintln!("[hal-ohos] a TLS stream's certificate is not for its host: refused"); }
        s.open = Some(ok && named);
        s.certificate = (!certificate.is_empty()).then(|| certificate.to_vec());
    });
}
pub(super) fn received(id: i64, data: &[u8]) { report(id, |s| s.received.extend(data)); }
pub(super) fn sent(id: i64, ok: bool) { report(id, |s| { s.answered += 1; s.send_failed |= !ok; }); }
pub(super) fn closed(id: i64) { report(id, |s| { s.closed = true; s.open.get_or_insert(false); }); }
