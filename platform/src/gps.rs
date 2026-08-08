//! Last-known device GPS fix.
//!
//! Written by the Android `LocationListener` through the JNI `onLocation`
//! receiver (`os/linux/android/android_jni.rs`) and read SYNCHRONOUSLY by the
//! Splash `sys.gps(...)` helper (in the widgets crate). It lives here in
//! `makepad-platform` because that is the one crate both the JNI writer and the
//! `sys.*` reader (widgets -> draw -> platform) can reach. On non-Android
//! targets nothing writes it, so `last_gps_fix()` simply stays `None`.
use std::sync::Mutex;

#[derive(Clone, Copy, Debug)]
pub struct GpsFix {
    pub lat: f64,
    pub lon: f64,
    /// horizontal accuracy in metres
    pub acc: f32,
}

static LAST_GPS_FIX: Mutex<Option<GpsFix>> = Mutex::new(None);

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
/// The camera no longer depends on this at all — it reads the fix every frame — so
/// what is left is the banner TEXT, and 40 m keeps the distance remaining honest.
///
/// The stutter's real fix is to stop ESCALATING a value change into a structural
/// rebuild: update the changed text in place. That is what the L2 app did with
/// `ui.instr.set_text()`, and why its contract says never to force a rebuild while
/// driving. `widget_tree.rs` already has the patch machinery
/// (`test_property_patch_no_structural_rebuild`); wiring L0's re-resolve into it is
/// the outstanding work.
const MOVED_ENOUGH_M: f64 = 40.0;

/// Store a fresh fix (called from the Android JNI `onLocation` receiver).
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
    let Ok(mut g) = LAST_GPS_FIX.lock() else {
        return;
    };
    let moved = match *g {
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
    // Dropped before bumping: re-resolving a card reads `last_gps_fix()`, and
    // holding the lock across that is a deadlock waiting for a fast fix.
    drop(g);
    if moved {
        crate::script::res::bump_data_fetch_epoch();
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
        let before = epoch();
        set_gps_fix(37.2600, -122.0300, 8.0);
        assert!(epoch() > before, "the first fix must bump");

        // Ten metres of drift is not news: an epoch change re-resolves the whole
        // card, and a turn instruction does not change over ten metres.
        // 0.00009° of latitude is ~10 m.
        let settled = epoch();
        set_gps_fix(37.260090, -122.030000, 8.0);
        assert_eq!(epoch(), settled, "drift must not re-resolve every card");

        // A hundred metres is. 0.0009° is ~100 m.
        set_gps_fix(37.260900, -122.030000, 8.0);
        assert!(epoch() > settled, "real movement must refresh the banner");
    }
}
