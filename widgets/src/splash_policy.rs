//! What an isolate is allowed, enforced where the runtime acts.
//!
//! The `host` bridge reports a capability list and the storage jail enforces
//! a quota, but until now nothing REFUSED a request on the strength of that
//! list, and nothing bounded how much script an isolate could run over its
//! life. This module holds the per-heap policy for both, and the rest of the
//! splash runtime asks it:
//!
//! - [`service_allowed`] — before a `host.request` is queued (ADR 0002 §2);
//! - [`url_allowed`] — from every network path, through the gate installed
//!   in `makepad-script-std` (ADR 0002 §3): `net.http_request`,
//!   `net.web_socket`, `sys.*` data fetches, artwork, map tiles, web cards
//!   and a `Video`'s network source;
//! - [`sockets_allowed`] — before `net.http_server` listens or
//!   `net.socket_stream` connects. Neither names a URL a host list could
//!   judge, and no capability covers them (`net` is requests to the listed
//!   hosts), so a policed isolate gets neither;
//! - [`charge`] — after every evaluation and callback, against a cumulative
//!   instruction budget (ADR 0002 §4).
//!
//! Like the jail and the bridge, state is host-side and keyed by heap, where
//! script can neither read nor raise it. An isolate with no policy set keeps
//! the behaviour every existing host relied on: services pass through to the
//! host, URLs are allowed, and only the per-evaluation cap applies.
use crate::makepad_draw::makepad_platform::makepad_script_std;
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct HeapPolicy {
    /// Granted capability names. A service `a.b.c` needs capability `a`, or
    /// the exact name `a.b.c`.
    pub capabilities: Vec<String>,
    /// Hosts this heap may reach, lowercased, exact. Empty means none.
    pub hosts: Vec<String>,
    /// Script instructions this heap may run over its life. None = unbounded
    /// (the per-evaluation cap still applies).
    pub instruction_budget: Option<u64>,
    pub instructions_used: u64,
    /// Set once the budget is spent; nothing runs in this heap afterwards.
    pub exhausted: bool,
}

thread_local! {
    static POLICIES: RefCell<HashMap<usize, HeapPolicy>> = RefCell::new(HashMap::new());
}

/// Set (or replace) the policy for a heap. Enforcement starts here: a heap
/// that never had this called is not enforced.
pub fn set_policy_for_heap(heap_key: usize, capabilities: Vec<String>, hosts: Vec<String>, instruction_budget: Option<u64>) {
    // A policy nothing consults is not a policy: whoever sets one, the gates
    // that read it are in place from here on.
    install_url_gate();
    POLICIES.with(|p| {
        let mut p = p.borrow_mut();
        let used = p.get(&heap_key).map(|old| old.instructions_used).unwrap_or(0);
        let exhausted = instruction_budget.map(|b| used >= b).unwrap_or(false);
        p.insert(
            heap_key,
            HeapPolicy {
                capabilities,
                hosts: hosts.into_iter().map(|h| h.to_ascii_lowercase()).collect(),
                instruction_budget,
                instructions_used: used,
                exhausted,
            },
        );
    });
}

/// Drop policies for reclaimed isolates (from the isolate GC).
pub(crate) fn gc_policies(dead_heaps: &[usize]) {
    POLICIES.with(|p| {
        let mut p = p.borrow_mut();
        for heap in dead_heaps {
            p.remove(heap);
        }
    });
}

/// Whether this heap is under an enforced policy at all.
pub fn is_enforced(heap_key: usize) -> bool {
    POLICIES.with(|p| p.borrow().contains_key(&heap_key))
}

/// May this heap ask the host for `service`? `Ok` when no policy is set (the
/// host decides, as before) or when the policy grants it; `Err` names what is
/// missing, in words meant for a log.
pub fn service_allowed(heap_key: usize, service: &str) -> Result<(), String> {
    POLICIES.with(|p| {
        let p = p.borrow();
        let Some(policy) = p.get(&heap_key) else { return Ok(()) };
        if policy.exhausted {
            return Err("this app's instruction budget is spent".into());
        }
        let family = service.split('.').next().unwrap_or(service);
        if policy.capabilities.iter().any(|c| c == service || c == family) {
            Ok(())
        } else {
            Err(format!("this app was not granted {family:?}, which {service:?} needs"))
        }
    })
}

/// May this heap reach `url`? True when no policy is set. With a policy, the
/// URL's host must be listed exactly; a URL with no host is refused.
pub fn url_allowed(heap_key: usize, url: &str) -> bool {
    POLICIES.with(|p| {
        let p = p.borrow();
        let Some(policy) = p.get(&heap_key) else { return true };
        if policy.exhausted {
            return false;
        }
        // A listed `host` matches any port; a listed `host:port` matches
        // only that port, which is how a host lists its own asset origin.
        let host = makepad_script_std::url_host(url);
        let host_port = makepad_script_std::url_host_port(url);
        policy.hosts.iter().any(|h| Some(h) == host.as_ref() || Some(h) == host_port.as_ref())
    })
}

/// May this heap open a listening server or a raw socket? Only when it is
/// not policed. A server answers whoever connects and a raw stream speaks any
/// protocol to any port; a contained app's way out is a request to a listed
/// host, and nothing wider.
pub fn sockets_allowed(heap_key: usize) -> bool {
    !is_enforced(heap_key)
}

/// May this heap know where the device is? Location is a service family
/// like any other, granted as `location`; an unpoliced heap keeps the old
/// behaviour. Every reader of the platform fix on a script's behalf asks
/// here: `sys.gps`, the blank-name geocode fallback, the map's follow camera.
pub fn location_allowed(heap_key: usize) -> bool {
    service_allowed(heap_key, "location.get").is_ok()
}

/// The device's last GPS fix as this heap may see it: `None` without the
/// `location` grant, exactly as if the device had no fix yet, so a card's
/// no-fix path is also its no-permission path.
pub fn gps_fix_for_heap(heap_key: usize) -> Option<crate::makepad_draw::makepad_platform::gps::GpsFix> {
    if location_allowed(heap_key) {
        crate::makepad_draw::makepad_platform::gps::last_gps_fix()
    } else {
        None
    }
}

/// May this heap read what the user keeps in the host: saved lists
/// (`sys.watchlist`, cities, reading, topics), stored preferences and the page
/// open in the reader. These are published process-wide for the host's own
/// cards; an app under a policy needs the `profile` grant to see them.
pub fn profile_allowed(heap_key: usize) -> bool {
    service_allowed(heap_key, "profile.read").is_ok()
}

/// A file a widget was told to read, as this heap may read it. An unpoliced
/// heap reads the path as given. A policed heap's paths are app-visible paths
/// inside its storage jail, so a card cannot point a widget (a map archive,
/// say) at a file of the host's; with no jail, it reads nothing.
pub fn local_path_for_heap(heap_key: usize, path: &str) -> Option<String> {
    if !is_enforced(heap_key) {
        return Some(path.to_string());
    }
    let root = crate::splash_storage::root_for_heap(heap_key)?;
    let real = crate::splash_storage::resolve_jailed(&root, path).ok()?;
    crate::splash_storage::verify_no_symlinks(&root, &real).ok()?;
    Some(real.to_string_lossy().into_owned())
}

/// The hosts a policed heap may reach, or `None` for an unpoliced heap. For
/// a surface the gate cannot stand in front of — a system WebView fetches on
/// its own — so it can be told the same list.
pub fn hosts_for_heap(heap_key: usize) -> Option<Vec<String>> {
    POLICIES.with(|p| p.borrow().get(&heap_key).map(|policy| policy.hosts.clone()))
}

/// The host of `url` when it is https and names a public host: not loopback,
/// private, link-local, shared or unspecified, not a single-label or
/// `.local`/`.internal`/`.localhost` name, and not an IP written in a form
/// only some resolvers read as one (`0x7f.1`). What a grant to reach "any
/// public page" may reach, and nothing on the device's own network.
pub fn public_https_host(url: &str) -> Result<String, String> {
    if !url.get(..8).is_some_and(|scheme| scheme.eq_ignore_ascii_case("https://")) {
        return Err("only https:// URLs are allowed".into());
    }
    let host = makepad_script_std::url_host(url).ok_or("missing host")?;
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = bare.parse::<std::net::IpAddr>() {
        return if is_public_ip(ip) { Ok(host) } else { Err(format!("host not permitted (private/internal): {host}")) };
    }
    let last = host.rsplit('.').next().unwrap_or("");
    let numeric = last.bytes().all(|b| b.is_ascii_digit()) || last.starts_with("0x");
    if numeric || !host.contains('.') || host.starts_with('[') {
        return Err(format!("host not permitted (not a public name): {host}"));
    }
    if host == "localhost" || [".localhost", ".internal", ".local", ".lan", ".home.arpa"].iter().any(|s| host.ends_with(s)) {
        return Err(format!("host not permitted (private/internal): {host}"));
    }
    Ok(host)
}

fn is_public_ip(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, ..] = v4.octets();
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_multicast()
                || a == 0
                || (a == 100 && (64..128).contains(&b)))
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_public_ip(IpAddr::V4(v4));
            }
            let first = v6.segments()[0];
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (first & 0xfe00) == 0xfc00
                || (first & 0xffc0) == 0xfe80)
        }
    }
}

/// May this heap load `url` as media (an image, artwork)? Its host list
/// answers first; the `images` grant adds any public https host, for an app
/// that shows pictures from wherever its content links (a feed reader's
/// thumbnails). A load is still a request, so the grant is its own consent.
pub fn media_allowed(heap_key: usize, url: &str) -> bool {
    url_allowed(heap_key, url)
        || (is_enforced(heap_key)
            && may_run(heap_key)
            && service_allowed(heap_key, "images.any").is_ok()
            && public_https_host(url).is_ok())
}

/// May this heap open `url` as a page in the system WebView? Its host list
/// answers first; the `web` grant adds any public https page, for a reader.
/// Such a page gets no bridge into the app.
pub fn page_allowed(heap_key: usize, url: &str) -> bool {
    url_allowed(heap_key, url)
        || (is_enforced(heap_key)
            && may_run(heap_key)
            && service_allowed(heap_key, "web.open").is_ok()
            && public_https_host(url).is_ok())
}

/// Record `instructions` run by this heap. Returns false once the budget is
/// spent, and stays false: the heap is exhausted from then on.
pub fn charge(heap_key: usize, instructions: u64) -> bool {
    POLICIES.with(|p| {
        let mut p = p.borrow_mut();
        let Some(policy) = p.get_mut(&heap_key) else { return true };
        policy.instructions_used = policy.instructions_used.saturating_add(instructions);
        if let Some(budget) = policy.instruction_budget {
            if policy.instructions_used >= budget {
                policy.exhausted = true;
            }
        }
        !policy.exhausted
    })
}

/// Whether this heap may still run anything.
pub fn may_run(heap_key: usize) -> bool {
    POLICIES.with(|p| p.borrow().get(&heap_key).map(|policy| !policy.exhausted).unwrap_or(true))
}

/// Instructions used so far, for a host that wants to show it.
pub fn instructions_used(heap_key: usize) -> u64 {
    POLICIES.with(|p| p.borrow().get(&heap_key).map(|policy| policy.instructions_used).unwrap_or(0))
}

/// The gate installed into `makepad-script-std`, once per thread, so every
/// request path consults the table above.
pub(crate) fn install_url_gate() {
    thread_local! { static INSTALLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
    INSTALLED.with(|installed| {
        if !installed.get() {
            makepad_script_std::set_script_url_gate(Some(url_allowed));
            makepad_script_std::set_script_media_gate(Some(media_allowed));
            makepad_script_std::set_script_socket_gate(Some(sockets_allowed));
            installed.set(true);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unpoliced_heap_behaves_as_before() {
        gc_policies(&[900]);
        assert!(service_allowed(900, "location.get").is_ok());
        assert!(url_allowed(900, "https://anything.example/x"));
        assert!(charge(900, 1_000_000));
        assert!(may_run(900));
    }

    #[test]
    fn a_service_needs_its_capability_family_or_its_exact_name() {
        set_policy_for_heap(901, vec!["location".into(), "ledger.read".into()], vec![], None);
        assert!(service_allowed(901, "location.get").is_ok());
        assert!(service_allowed(901, "location.watch").is_ok());
        assert!(service_allowed(901, "ledger.read").is_ok());
        assert!(service_allowed(901, "ledger.write").is_err(), "ledger.read does not cover ledger.write");
        assert!(service_allowed(901, "clipboard.read").is_err());
        gc_policies(&[901]);
    }

    #[test]
    fn a_url_must_name_a_listed_host_exactly() {
        set_policy_for_heap(902, vec!["net".into()], vec!["Api.Weather.Example".into()], None);
        assert!(url_allowed(902, "https://api.weather.example/v1"));
        assert!(url_allowed(902, "HTTPS://API.WEATHER.EXAMPLE:443/"));
        assert!(!url_allowed(902, "https://weather.example/"), "a parent domain is not the listed host");
        assert!(!url_allowed(902, "https://evil.api.weather.example/"), "nor is a subdomain");
        assert!(!url_allowed(902, "http://127.0.0.1:8170/ux-images/x.svg"), "the artwork server that leaked");
        assert!(!url_allowed(902, "garbage"));
        gc_policies(&[902]);
    }

    #[test]
    fn a_listed_host_port_matches_only_that_port() {
        set_policy_for_heap(907, vec![], vec!["127.0.0.1:5000".into()], None);
        assert!(url_allowed(907, "http://127.0.0.1:5000/assets/a.svg"));
        assert!(!url_allowed(907, "http://127.0.0.1:8170/ux-images/a.svg"), "another loopback service");
        assert!(!url_allowed(907, "http://127.0.0.1/a.svg"), "nor the bare host");
        gc_policies(&[907]);
    }

    #[test]
    fn an_empty_host_list_reaches_nothing_even_with_the_capability() {
        set_policy_for_heap(903, vec!["net".into()], vec![], None);
        assert!(!url_allowed(903, "https://api.weather.example/"));
        gc_policies(&[903]);
    }

    #[test]
    fn the_budget_is_cumulative_and_exhaustion_is_permanent() {
        set_policy_for_heap(904, vec!["location".into()], vec!["a.example".into()], Some(1000));
        assert!(charge(904, 400));
        assert!(charge(904, 400));
        assert!(may_run(904));
        assert!(!charge(904, 400), "the third charge crosses the budget");
        assert!(!may_run(904));
        assert!(service_allowed(904, "location.get").is_err(), "an exhausted heap gets no services");
        assert!(!url_allowed(904, "https://a.example/"), "nor network");
        assert!(!charge(904, 1), "and stays exhausted");
        assert_eq!(instructions_used(904), 1201);
        gc_policies(&[904]);
    }

    #[test]
    fn replacing_a_policy_keeps_what_was_already_spent() {
        set_policy_for_heap(905, vec![], vec![], Some(100));
        charge(905, 90);
        set_policy_for_heap(905, vec![], vec![], Some(100));
        assert_eq!(instructions_used(905), 90, "a re-push of the same policy is not a refill");
        set_policy_for_heap(905, vec![], vec![], Some(50));
        assert!(!may_run(905), "lowering the budget below what is spent exhausts the heap");
        gc_policies(&[905]);
    }

    #[test]
    fn location_needs_the_location_grant() {
        gc_policies(&[908]);
        assert!(location_allowed(908), "an unpoliced heap reads the fix as before");
        set_policy_for_heap(908, vec!["net".into()], vec![], None);
        assert!(!location_allowed(908));
        assert!(gps_fix_for_heap(908).is_none(), "no grant reads as no fix");
        set_policy_for_heap(908, vec!["location".into()], vec![], None);
        assert!(location_allowed(908));
        assert!(!profile_allowed(908), "location does not cover the user's saved lists");
        set_policy_for_heap(908, vec!["profile".into()], vec![], None);
        assert!(profile_allowed(908));
        gc_policies(&[908]);
    }

    #[test]
    fn a_policed_heap_reads_local_files_only_inside_its_jail() {
        gc_policies(&[909]);
        assert_eq!(local_path_for_heap(909, "/etc/hosts").as_deref(), Some("/etc/hosts"));
        set_policy_for_heap(909, vec![], vec![], None);
        assert!(local_path_for_heap(909, "maps/world.mkmap").is_none(), "no jail, no files");
        crate::splash_storage::set_root_for_heap(909, Some("/jail/app".into()));
        assert_eq!(
            local_path_for_heap(909, "maps/world.mkmap").as_deref(),
            Some("/jail/app/maps/world.mkmap")
        );
        assert!(local_path_for_heap(909, "../../etc/hosts").is_none(), "no climbing out");
        crate::splash_storage::set_root_for_heap(909, None);
        gc_policies(&[909]);
    }

    #[test]
    fn a_public_host_is_https_and_off_the_devices_network() {
        for ok in ["https://news.ycombinator.com/", "https://CDN.Example.org:8443/a.jpg", "https://8.8.8.8/", "https://[2606:4700::1111]/"] {
            assert!(public_https_host(ok).is_ok(), "{ok}");
        }
        for bad in [
            "http://example.com/",
            "https://localhost/",
            "https://127.0.0.1/",
            "https://10.0.0.8/",
            "https://172.20.1.1/",
            "https://192.168.1.1/",
            "https://169.254.169.254/latest/meta-data",
            "https://100.64.0.1/",
            "https://0.0.0.0/",
            "https://[::1]/",
            "https://[fd00::1]/",
            "https://[fe80::1]/",
            "https://[::ffff:127.0.0.1]/",
            "https://0x7f.1/",
            "https://2130706433/",
            "https://router/",
            "https://printer.local/",
            "https://nas.home.arpa/",
            "https://evil.com@127.0.0.1/",
        ] {
            assert!(public_https_host(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn images_and_web_grants_open_public_https_only() {
        set_policy_for_heap(910, vec!["net".into()], vec!["hn.algolia.com".into()], None);
        assert!(media_allowed(910, "https://hn.algolia.com/logo.png"), "its own hosts, as before");
        assert!(!media_allowed(910, "https://cdn.example.org/a.jpg"));
        assert!(!page_allowed(910, "https://example.org/story"));
        set_policy_for_heap(910, vec!["net".into(), "images".into(), "web".into()], vec!["hn.algolia.com".into()], None);
        assert!(media_allowed(910, "https://cdn.example.org/a.jpg"));
        assert!(page_allowed(910, "https://example.org/story"));
        assert!(!media_allowed(910, "https://192.168.1.1/a.jpg"), "never the device's network");
        assert!(!page_allowed(910, "http://example.org/story"), "never plain http");
        assert!(!url_allowed(910, "https://cdn.example.org/a.jpg"), "neither grant widens requests");
        gc_policies(&[910]);
        assert!(media_allowed(910, "https://anything.example/"), "an unpoliced heap, as before");
    }

    #[test]
    fn a_policed_heap_opens_no_server_and_no_raw_socket_whatever_it_holds() {
        gc_policies(&[911]);
        assert!(sockets_allowed(911), "an unpoliced heap, as before");
        set_policy_for_heap(911, vec!["net".into(), "storage".into(), "web".into()], vec!["127.0.0.1".into()], None);
        assert!(!sockets_allowed(911), "no grant and no listed host covers a socket");
        assert!(!makepad_script_std::script_sockets_allowed(911), "setting a policy installs the gate");
        assert!(makepad_script_std::script_url_allowed(911, "ws://127.0.0.1:9/"), "the url gate is in too");
        gc_policies(&[911]);
        assert!(makepad_script_std::script_sockets_allowed(911));
    }

    #[test]
    fn gc_forgets_a_heap_entirely() {
        set_policy_for_heap(906, vec![], vec![], Some(1));
        charge(906, 5);
        gc_policies(&[906]);
        assert!(!is_enforced(906));
        assert!(may_run(906));
    }
}
