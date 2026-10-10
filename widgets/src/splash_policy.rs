//! What an isolate is allowed, where the runtime acts.
//!
//! OctoSense's ruling of 8 October 2026 removed the per-app runtime gates on
//! reach and compute: the OS is the outer protection and the host's API
//! surface the boundary, as for a browser. An app's host list and its
//! instruction budget are declarations, shown when it is installed, not
//! limits while it runs:
//!
//! - no URL, media or socket gate is installed in `makepad-script-std`, so
//!   `net.http_request`, `net.web_socket`, `net.socket_stream`,
//!   `net.http_server`, `sys.*` fetches, artwork, map tiles and a `Video`'s
//!   network source reach whatever the device can. [`url_allowed`],
//!   [`media_allowed`], [`page_allowed`] and [`sockets_allowed`] allow
//!   everything, for the hosts that still ask;
//! - [`charge`] counts what an isolate runs, for a host that shows it
//!   ([`instructions_used`]), but never stops it. The per-evaluation cap
//!   still ends a runaway evaluation.
//!
//! What a policy still decides, for a policed heap:
//!
//! - [`service_allowed`] — before a `host.request` is queued (ADR 0002 §2).
//!   The ruling removes this check as well, once the host checks the family
//!   grant itself; App Hub's dispatcher does not yet, so dropping it here
//!   would open every host service to every app;
//! - what the runtime reads for a script with no host in between: the
//!   device's location ([`location_allowed`]), the person's saved lists
//!   ([`profile_allowed`]), the camera preview and `agent.notify`;
//! - files: a policed heap reads only inside its storage jail
//!   ([`local_path_for_heap`]).
//!
//! Like the jail and the bridge, state is host-side and keyed by heap, where
//! script can neither read nor raise it. An isolate with no policy set keeps
//! the behaviour every existing host relied on: services pass through to the
//! host.
use crate::makepad_draw::makepad_platform::makepad_script_std;
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct HeapPolicy {
    /// Granted capability names. A service `a.b.c` needs capability `a`, or
    /// the exact name `a.b.c`.
    pub capabilities: Vec<String>,
    /// The hosts the app declares, lowercased: shown when it is installed,
    /// not a limit on what it reaches.
    pub hosts: Vec<String>,
    /// The instruction budget the app declares: counted against, never
    /// enforced.
    pub instruction_budget: Option<u64>,
    pub instructions_used: u64,
}

thread_local! {
    static POLICIES: RefCell<HashMap<usize, HeapPolicy>> = RefCell::new(HashMap::new());
}

/// Set (or replace) the policy for a heap. Enforcement starts here: a heap
/// that never had this called is not enforced.
pub fn set_policy_for_heap(heap_key: usize, capabilities: Vec<String>, hosts: Vec<String>, instruction_budget: Option<u64>) {
    POLICIES.with(|p| {
        let mut p = p.borrow_mut();
        let used = p.get(&heap_key).map(|old| old.instructions_used).unwrap_or(0);
        p.insert(
            heap_key,
            HeapPolicy {
                capabilities,
                hosts: hosts.into_iter().map(|h| h.to_ascii_lowercase()).collect(),
                instruction_budget,
                instructions_used: used,
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
        let family = service.split('.').next().unwrap_or(service);
        if policy.capabilities.iter().any(|c| c == service || c == family) {
            Ok(())
        } else {
            Err(format!("this app was not granted {family:?}, which {service:?} needs"))
        }
    })
}

/// May this heap reach `url`? Yes, policed or not: an app's host list is a
/// declaration, not a gate (the ruling of 8 October 2026).
pub fn url_allowed(_heap_key: usize, _url: &str) -> bool {
    true
}

/// May this heap open a listening server or a raw socket? Yes, policed or
/// not, as for [`url_allowed`].
pub fn sockets_allowed(_heap_key: usize) -> bool {
    true
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

/// The host of `url` when it is https and names a public host: not loopback,
/// private, link-local, shared or unspecified, not a single-label or
/// `.local`/`.internal`/`.localhost` name, and not an IP written in a form
/// only some resolvers read as one (`0x7f.1`). A web card's `http.fetch`
/// bridge, which runs a card's own JavaScript requests natively, reaches only
/// such hosts.
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

/// May this heap load `url` as media (an image, artwork)? Yes, as for
/// [`url_allowed`]: the `images` capability is a declaration.
pub fn media_allowed(_heap_key: usize, _url: &str) -> bool {
    true
}

/// May this heap open `url` as a page in the system WebView? Yes, as for
/// [`url_allowed`]: the `web` capability is a declaration. Such a page gets
/// no bridge into the app.
pub fn page_allowed(_heap_key: usize, _url: &str) -> bool {
    true
}

/// Record `instructions` run by this heap. Always true: the budget an app
/// declares is counted against, never enforced.
pub fn charge(heap_key: usize, instructions: u64) -> bool {
    POLICIES.with(|p| {
        if let Some(policy) = p.borrow_mut().get_mut(&heap_key) {
            policy.instructions_used = policy.instructions_used.saturating_add(instructions);
        }
    });
    true
}

/// Whether this heap may still run anything. Always: nothing exhausts a heap.
pub fn may_run(_heap_key: usize) -> bool {
    true
}

/// Instructions used so far, for a host that wants to show it.
pub fn instructions_used(heap_key: usize) -> u64 {
    POLICIES.with(|p| p.borrow().get(&heap_key).map(|policy| policy.instructions_used).unwrap_or(0))
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

    /// The host list is a declaration (the ruling of 8 October 2026): a
    /// policed heap reaches any URL, listed or not, with or without `net`.
    #[test]
    fn a_policed_heap_reaches_any_url() {
        set_policy_for_heap(902, vec!["net".into()], vec!["Api.Weather.Example".into()], None);
        assert!(url_allowed(902, "https://api.weather.example/v1"));
        assert!(url_allowed(902, "https://weather.example/"), "a host it does not list");
        assert!(url_allowed(902, "http://127.0.0.1:8170/ux-images/x.svg"), "plain http to this device");
        set_policy_for_heap(902, vec![], vec![], None);
        assert!(url_allowed(902, "https://api.weather.example/"), "nor does it need net or a host");
        gc_policies(&[902]);
    }

    /// An app's instruction budget is counted, for a host that shows it, and
    /// never stops the app; a re-push of its policy keeps the count.
    #[test]
    fn the_budget_is_counted_and_stops_nothing() {
        set_policy_for_heap(904, vec!["location".into()], vec!["a.example".into()], Some(1000));
        assert!(charge(904, 400));
        assert!(charge(904, 400));
        assert!(charge(904, 400), "crossing the budget stops nothing");
        assert!(may_run(904));
        assert!(service_allowed(904, "location.get").is_ok(), "its grants still answer");
        assert_eq!(instructions_used(904), 1200);
        set_policy_for_heap(904, vec![], vec![], Some(100));
        assert_eq!(instructions_used(904), 1200, "a re-push of its policy is not a reset");
        assert!(may_run(904), "nor does a lower budget stop it");
        gc_policies(&[904]);
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

    /// Pictures and pages need neither `images` nor `web`: both are
    /// declarations now.
    #[test]
    fn media_and_pages_need_no_grant() {
        set_policy_for_heap(910, vec![], vec![], None);
        assert!(media_allowed(910, "https://cdn.example.org/a.jpg"));
        assert!(media_allowed(910, "http://192.168.1.1/a.jpg"));
        assert!(page_allowed(910, "https://example.org/story"));
        assert!(page_allowed(910, "http://example.org/story"));
        gc_policies(&[910]);
    }

    /// Setting a policy installs no gate in `makepad-script-std`: a policed
    /// heap opens sockets and servers, and reaches any URL, through it.
    #[test]
    fn a_policy_installs_no_gate() {
        set_policy_for_heap(911, vec![], vec![], Some(1));
        assert!(sockets_allowed(911));
        assert!(makepad_script_std::script_sockets_allowed(911));
        assert!(makepad_script_std::script_url_allowed(911, "ws://127.0.0.1:9/"));
        assert!(makepad_script_std::script_media_url_allowed(911, "http://10.0.0.8/a.jpg"));
        gc_policies(&[911]);
    }

    #[test]
    fn gc_forgets_a_heap_entirely() {
        set_policy_for_heap(906, vec![], vec![], Some(1));
        charge(906, 5);
        assert!(is_enforced(906));
        gc_policies(&[906]);
        assert!(!is_enforced(906));
        assert_eq!(instructions_used(906), 0);
    }
}
