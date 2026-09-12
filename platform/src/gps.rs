//! Last-known device GPS fix.
//!
//! Written by the Android and HarmonyOS location listeners (or an explicit
//! development track) and read by Splash `sys.gps(...)` and the map camera.
//! Platforms without a location provider keep `last_gps_fix()` as `None`.
use std::sync::Mutex;

#[derive(Clone, Copy, Debug)]
pub struct GpsFix {
    pub lat: f64,
    pub lon: f64,
    /// horizontal accuracy in metres
    pub acc: f32,
}

static LAST_GPS_FIX: Mutex<Option<GpsFix>> = Mutex::new(None);
static LAST_EPOCH_FIX: Mutex<Option<GpsFix>> = Mutex::new(None);

/// Whether an injected track owns the position — see [`claim_fake_gps`].
static FAKE_GPS_ACTIVE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// An injected track (the `FAKE_GPS_FILE` harness hook) claims the position:
/// from now on the REAL `LocationListener` is ignored.
///
/// Both feeds funnel into the same fix, and for two days that was moot — this
/// handset had no live fix at all. The moment its network location woke back
/// up, live nav started receiving interleaved positions: the walked track
/// point, then the device's actual (stationary, ~250 m accuracy) location,
/// 700 ms apart — and the follow camera "jumped randomly" between a simulated
/// drive and the user's sofa. A simulation that is running IS the position;
/// mixing in the real one is not honesty, it is two truths fighting.
pub fn claim_fake_gps() {
    FAKE_GPS_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// The `LocationListener`'s entry: a real fix, honoured only while no injected
/// track has claimed the position.
pub fn set_gps_fix_from_listener(lat: f64, lon: f64, acc: f32) {
    if FAKE_GPS_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    set_gps_fix(lat, lon, acc);
}

/// How far the device must move before a fix counts as news, in metres.
///
/// This is the ESCALATION RATE, and escalating is expensive. An epoch change
/// re-resolves a card — realize, lower, evaluate, rebuild its widget tree — on the
/// UI thread, so it lands inside a frame. Measured on a OnePlus 6 while driving:
/// hitches and re-resolves correlate exactly 1:1, and the hitches were 40, 43, 101
/// and **327 ms**. A third of a second of frozen map is the stutter, and no amount
/// of camera smoothing hides it because the camera is not what stops.
///
/// 40 m. It was raised to 250 to cut the stutter and that DID NOT WORK — measured,
/// 4 re-resolves in 50 s of driving at both values. The bumps were never mostly
/// GPS's: a successful `script_data_fetch` bumps the same epoch (see
/// `res.rs`'s `finish_data_fetch`), so every route, place and retry lands as a card
/// rebuild too. Rate-limiting this source alone buys nothing and costs text
/// freshness, so it is back where it was.
///
/// The camera wakes on a changed fix independently of this threshold, so
/// what is left is the banner TEXT, and 40 m keeps the distance remaining honest.
///
/// The stutter's real fix is to stop ESCALATING a value change into a structural
/// rebuild: update the changed text in place. That is what the L2 app did with
/// `ui.instr.set_text()`, and why its contract says never to force a rebuild while
/// driving. `widget_tree.rs` already has the patch machinery
/// (`test_property_patch_no_structural_rebuild`); wiring L0's re-resolve into it is
/// the outstanding work.
const MOVED_ENOUGH_M: f64 = 40.0;

/// Store a fresh fix from a platform listener or the development track.
///
/// A NEW POSITION IS NEW DATA, so it bumps the script data-fetch epoch exactly
/// as a landed HTTP fetch does. Without that this function was a dead end: the
/// fix was stored and nothing asked again.
///
/// An L0 card bakes its `sys.*` values in when the ledger is resolved and
/// re-resolves only on an epoch change, and a GPS fix is not a fetch — so a
/// navigation card's `sys.gps("lat")` was frozen at whatever the fix had been
/// when the card was built. Measured on a OnePlus 6: the follow camera, the turn
/// instruction and the distance remaining were all correct, all live, and none of
/// them ever moved. Every part of that card was right except that nothing told it
/// to look again.
pub fn set_gps_fix(lat: f64, lon: f64, acc: f32) {
    if !lat.is_finite() || !lon.is_finite() || lat.abs() > 90.0 || lon.abs() > 180.0 {
        return;
    }
    let Ok(mut g) = LAST_GPS_FIX.lock() else {
        return;
    };
    let position_changed = g.map(|p| p.lat != lat || p.lon != lon).unwrap_or(true);
    let Ok(mut epoch_fix) = LAST_EPOCH_FIX.lock() else { return };
    let moved = match *epoch_fix {
        // Degrees to metres: 111_320 per degree of latitude, and per degree of
        // longitude scaled by the cosine of it. Planar over a few metres, which
        // is all this comparison spans.
        Some(p) => {
            let dy = (lat - p.lat) * 111_320.0;
            let dx = (lon - p.lon) * 111_320.0 * lat.to_radians().cos();
            (dx * dx + dy * dy).sqrt() >= MOVED_ENOUGH_M
        }
        // The FIRST fix always counts. A card built before the device knew where
        // it was is showing a placeholder, and this is what replaces it.
        None => true,
    };
    *g = Some(GpsFix { lat, lon, acc });
    if moved { *epoch_fix = *g; }
    drop(epoch_fix);
    // Dropped before bumping: re-resolving a card reads `last_gps_fix()`, and
    // holding the lock across that is a deadlock waiting for a fast fix.
    drop(g);
    if moved {
        crate::script::res::bump_data_fetch_epoch();
    }
    // Wake sleeping follow cameras for every changed fix, independently of the
    // much coarser epoch that invalidates structural card data.
    if position_changed {
        crate::thread::SignalToUI::set_ui_signal();
    }
}

/// The most recent fix, or `None` if the device has not produced one yet.
pub fn last_gps_fix() -> Option<GpsFix> {
    LAST_GPS_FIX.lock().ok().and_then(|g| *g)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fix that MOVED bumps the epoch; jitter does not.
    ///
    /// The bump is what makes a navigation card look again — an L0 card bakes its
    /// `sys.*` values in at resolve time and re-resolves only on an epoch change,
    /// and a GPS fix is not a fetch. Without it the follow camera, the turn
    /// instruction and the distance remaining were all live, all correct, and all
    /// frozen at the fix the card was built with.
    ///
    /// The threshold is the other half. The epoch is global, so bumping on every
    /// fix re-resolves every card on screen — a weather card rebuilding once a
    /// second because the handset is sitting on a desk.
    #[test]
    fn a_fix_bumps_the_epoch_only_when_it_moved() {
        let epoch = || crate::script::res::data_fetch_epoch_for_test();

        // The first fix always counts: a card built before the device knew where
        // it was is showing a placeholder, and this is what replaces it.
        *LAST_GPS_FIX.lock().unwrap() = None;
        *LAST_EPOCH_FIX.lock().unwrap() = None;
        let before = epoch();
        set_gps_fix(37.2600, -122.0300, 8.0);
        assert!(epoch() > before, "the first fix must bump");

        // Ten metres of drift is not news: an epoch change re-resolves the whole
        // card, and a turn instruction does not change over ten metres.
        // 0.00009° of latitude is ~10 m.
        let settled = epoch();
        set_gps_fix(37.260090, -122.030000, 8.0);
        assert_eq!(epoch(), settled, "drift must not re-resolve every card");

        // Small successive fixes must accumulate against the last published
        // position; comparing only adjacent fixes never refreshes a slow walk.
        for i in 2..=4 { set_gps_fix(37.2600 + i as f64 * 0.00009, -122.0300, 8.0); }
        assert!(epoch() > settled, "cumulative movement must refresh data");
        let settled = epoch();
        set_gps_fix(f64::NAN, -122.03, 8.0);
        assert_eq!(epoch(), settled);
        assert!(last_gps_fix().unwrap().lat.is_finite());

        // A hundred metres is. 0.0009° is ~100 m.
        set_gps_fix(37.260900, -122.030000, 8.0);
        assert!(epoch() > settled, "real movement must refresh the banner");
    }
}
