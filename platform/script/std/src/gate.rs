//! The one question every way out of an isolate asks: may THIS heap reach
//! THIS URL?
//!
//! Four paths issue requests on a script's behalf — `net.http_request`,
//! `net.web_socket`, `res.http_resource` (artwork) and the data fetches
//! behind `sys.*` — and a network grant that covers only the first is not a
//! grant. All of them ask here. Listening servers (`net.http_server`) and raw
//! streams (`net.socket_stream`) name no URL, so they ask the socket gate. The answer comes from a gate the embedding host installs; with no
//! gate installed, everything is allowed, which is the behaviour every
//! existing host had. A host that installs a gate answers per heap, so an
//! app's allowlist follows its isolate and nothing else.
use std::cell::Cell;

/// `(heap_key, url) -> allowed`. A plain fn so it can live in a thread-local
/// without lifetimes; the host keeps its own per-heap table.
pub type ScriptUrlGate = fn(usize, &str) -> bool;

thread_local! {
    static URL_GATE: Cell<Option<ScriptUrlGate>> = const { Cell::new(None) };
    static MEDIA_GATE: Cell<Option<ScriptUrlGate>> = const { Cell::new(None) };
    static SOCKET_GATE: Cell<Option<ScriptSocketGate>> = const { Cell::new(None) };
}

/// `heap_key -> allowed`: may this heap open what a host list cannot
/// describe — a listening server (`net.http_server`) or a raw TCP/TLS stream
/// (`net.socket_stream`)? A server answers whoever connects, and a raw
/// stream speaks any protocol to any port, so neither fits a per-URL gate.
pub type ScriptSocketGate = fn(usize) -> bool;

/// Install (or clear) the gate for listening servers and raw sockets. With
/// no socket gate, both are allowed, as they were for every existing host.
pub fn set_script_socket_gate(gate: Option<ScriptSocketGate>) {
    SOCKET_GATE.with(|g| g.set(gate));
}

/// Whether `heap_key` may listen or open a raw socket. True when no gate is
/// installed.
pub fn script_sockets_allowed(heap_key: usize) -> bool {
    SOCKET_GATE.with(|g| match g.get() {
        Some(gate) => gate(heap_key),
        None => true,
    })
}

/// Install (or clear) the gate for media loads (`http_resource`: images and
/// other artwork). A host can grant an app artwork from anywhere without
/// granting it requests to anywhere. With no media gate, media answers to
/// the URL gate like every other path.
pub fn set_script_media_gate(gate: Option<ScriptUrlGate>) {
    MEDIA_GATE.with(|g| g.set(gate));
}

/// Whether `heap_key` may load media from `url`.
pub fn script_media_url_allowed(heap_key: usize, url: &str) -> bool {
    match MEDIA_GATE.with(|g| g.get()) {
        Some(gate) => gate(heap_key, url),
        None => script_url_allowed(heap_key, url),
    }
}

/// Install (or clear) the gate for this thread. Isolates are UI-thread-owned,
/// so this is the thread that matters.
pub fn set_script_url_gate(gate: Option<ScriptUrlGate>) {
    URL_GATE.with(|g| g.set(gate));
}

/// Whether `heap_key` may reach `url`. True when no gate is installed.
pub fn script_url_allowed(heap_key: usize, url: &str) -> bool {
    URL_GATE.with(|g| match g.get() {
        Some(gate) => gate(heap_key, url),
        None => true,
    })
}

/// `host:port` when the URL names a port, lowercased. A host that serves an
/// app from a loopback port of its own lists exactly that, so the grant does
/// not extend to every other loopback service on the machine.
pub fn url_host_port(url: &str) -> Option<String> {
    let rest = url.split_once("://").map(|(_, r)| r)?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let hostport = authority.rsplit('@').next()?;
    let (host, port) = if hostport.starts_with('[') {
        let end = hostport.find(']')?;
        (&hostport[..=end], hostport[end + 1..].strip_prefix(':')?)
    } else {
        hostport.split_once(':')?
    };
    if host.is_empty() || port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(format!("{}:{port}", host.to_ascii_lowercase()))
}

/// The host part of a URL, lowercased, without scheme, port, path or
/// credentials: what an allowlist entry is compared against.
pub fn url_host(url: &str) -> Option<String> {
    let rest = url.split_once("://").map(|(_, r)| r)?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let host = authority.rsplit('@').next()?;
    // An IPv6 literal keeps its brackets; anything else drops a port.
    let host = if host.starts_with('[') {
        host.split(']').next().map(|h| format!("{h}]"))?
    } else {
        host.split(':').next()?.to_string()
    };
    if host.is_empty() {
        return None;
    }
    Some(host.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_is_extracted_without_scheme_port_path_or_credentials() {
        assert_eq!(url_host("https://Api.Example.com:8443/v1?x=1").as_deref(), Some("api.example.com"));
        assert_eq!(url_host("http://user:pw@127.0.0.1:8170/ux").as_deref(), Some("127.0.0.1"));
        assert_eq!(url_host("https://[::1]:80/").as_deref(), Some("[::1]"));
        assert_eq!(url_host("not a url"), None);
        assert_eq!(url_host("https:///nohost"), None);
    }

    #[test]
    fn host_port_is_only_reported_when_a_port_is_named() {
        assert_eq!(url_host_port("http://127.0.0.1:8170/x").as_deref(), Some("127.0.0.1:8170"));
        assert_eq!(url_host_port("https://api.example/x"), None);
        assert_eq!(url_host_port("https://[::1]:8080/").as_deref(), Some("[::1]:8080"));
        assert_eq!(url_host_port("http://h:notaport/"), None);
    }

    #[test]
    fn no_gate_allows_everything_and_a_gate_decides() {
        set_script_url_gate(None);
        assert!(script_url_allowed(1, "https://anything.example"));
        fn only_heap_seven(heap: usize, _url: &str) -> bool {
            heap == 7
        }
        set_script_url_gate(Some(only_heap_seven));
        assert!(script_url_allowed(7, "https://anything.example"));
        assert!(!script_url_allowed(8, "https://anything.example"));
        set_script_url_gate(None);
    }

    #[test]
    fn media_falls_back_to_the_url_gate_until_it_has_its_own() {
        fn only_heap_seven(heap: usize, _url: &str) -> bool {
            heap == 7
        }
        fn every_heap(_heap: usize, _url: &str) -> bool {
            true
        }
        set_script_url_gate(Some(only_heap_seven));
        assert!(!script_media_url_allowed(8, "https://cdn.example/a.jpg"));
        set_script_media_gate(Some(every_heap));
        assert!(script_media_url_allowed(8, "https://cdn.example/a.jpg"));
        assert!(!script_url_allowed(8, "https://cdn.example/a.jpg"), "requests keep their own gate");
        set_script_media_gate(None);
        set_script_url_gate(None);
    }

    #[test]
    fn no_socket_gate_allows_sockets_and_a_gate_decides() {
        set_script_socket_gate(None);
        assert!(script_sockets_allowed(8));
        fn only_heap_seven(heap: usize) -> bool {
            heap == 7
        }
        set_script_socket_gate(Some(only_heap_seven));
        assert!(script_sockets_allowed(7));
        assert!(!script_sockets_allowed(8));
        set_script_socket_gate(None);
    }
}
