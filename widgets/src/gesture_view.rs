//! Gestures for script apps: a View whose taps, long presses, swipes, drags
//! and pinches call the app's script, and a bottom sheet that rests at three
//! heights and follows the finger between them.
//!
//! Recognition stays native. A pinch or a drag is dozens of touch updates a
//! second; handing each to script would spend an installed app's instruction
//! budget on arithmetic and stutter on an old phone. `GestureView` calls the
//! script once per recognized gesture, and for the continuous ones (a drag, a
//! pinch) at most once a frame and only when the value moved. `SheetView`
//! needs no script at all while it moves; it tells the app where it came to
//! rest.
//!
//! Neither widget grants anything: they are vocabulary, like a Button, and
//! every isolate has them.
//!
//! ```text
//! GestureView{
//!     on_tap: |x, y| open_at(x, y)
//!     on_double_tap: |x, y| toggle_zoom()
//!     on_long_press: |x, y| show_menu(x, y)
//!     on_swipe: |dx, dy| if dx < 0 { next() } else { previous() }
//!     on_pan: |dx, dy, phase| drag(dx, dy, phase)      // phase 0 begin, 1 move, 2 end
//!     on_pinch: |scale, x, y, phase| zoom(scale, phase) // scale relative to the pinch start
//!     Image{...}
//! }
//! SheetView{ on_detent: |detent| sheet_moved(detent)  ...children }  // 0 peek, 1 half, 2 full
//! ```
//! Coordinates are relative to the view's top left.
use crate::{
    makepad_derive_widget::*,
    makepad_draw::*,
    makepad_script::ScriptFnRef,
    view::View,
    widget::*,
    widget_async::CxWidgetToScriptCallExt,
};
use crate::makepad_draw::makepad_platform::event::{DigitId, TouchState, TouchUpdateEvent};

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.*

    mod.widgets.GestureViewBase = #(GestureView::register_widget(vm))
    mod.widgets.GestureView = set_type_default() do mod.widgets.GestureViewBase{
        width: Fill
        height: Fit
    }

    mod.widgets.SheetViewBase = #(SheetView::register_widget(vm))
    mod.widgets.SheetView = set_type_default() do mod.widgets.SheetViewBase{
        width: Fill
        height: Fill
        flow: Down
    }
}

/// Movement under this is still a tap: a finger is never perfectly still.
const TAP_SLOP: f64 = 10.0;
/// A second tap this soon after the first, this close to it, is a double tap.
const DOUBLE_TAP_SECONDS: f64 = 0.3;
const DOUBLE_TAP_DISTANCE: f64 = 24.0;
/// A release this fast after this much travel along one axis is a swipe.
const SWIPE_DISTANCE: f64 = 60.0;
const SWIPE_SECONDS: f64 = 0.45;
/// A pinch reports again once its scale has moved this much.
const PINCH_STEP: f64 = 0.02;

/// A swipe's direction, when the travel was one: the dominant axis carries
/// at least 1.5 times the other, and far enough, fast enough.
fn swipe(dx: f64, dy: f64, seconds: f64) -> Option<(f64, f64)> {
    if seconds > SWIPE_SECONDS {
        return None;
    }
    if dx.abs() >= SWIPE_DISTANCE && dx.abs() >= dy.abs() * 1.5 {
        return Some((dx, 0.0));
    }
    if dy.abs() >= SWIPE_DISTANCE && dy.abs() >= dx.abs() * 1.5 {
        return Some((0.0, dy));
    }
    None
}

#[derive(Clone, Copy, Debug)]
struct Contact {
    uid: u64,
    pos: DVec2,
}

/// Two fingers down: the spread they started at.
#[derive(Clone, Copy, Debug)]
struct Spread {
    first: u64,
    second: u64,
    distance: f64,
}

/// What one touch update did to a pinch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum PinchStep {
    #[default]
    Nothing,
    Began(DVec2),
    Moved(f64, DVec2),
    Ended,
}

/// Follows the fingers that started inside the view and reports a pinch.
#[derive(Clone, Debug, Default)]
pub(crate) struct PinchTracker {
    contacts: Vec<Contact>,
    spread: Option<Spread>,
}

impl PinchTracker {
    pub(crate) fn update(&mut self, event: &TouchUpdateEvent, bounds: Rect) -> PinchStep {
        for touch in &event.touches {
            match touch.state {
                TouchState::Start => {
                    if bounds.contains(touch.abs) && !self.contacts.iter().any(|c| c.uid == touch.uid) {
                        self.contacts.push(Contact { uid: touch.uid, pos: touch.abs });
                    }
                }
                TouchState::Stop => self.contacts.retain(|c| c.uid != touch.uid),
                TouchState::Move | TouchState::Stable => {
                    if let Some(contact) = self.contacts.iter_mut().find(|c| c.uid == touch.uid) {
                        contact.pos = touch.abs;
                    }
                }
            }
        }
        let contact = |uid: u64| self.contacts.iter().find(|c| c.uid == uid).copied();
        match self.spread {
            None if self.contacts.len() >= 2 => {
                let (first, second) = (self.contacts[0], self.contacts[1]);
                self.spread = Some(Spread {
                    first: first.uid,
                    second: second.uid,
                    // A zero spread would make every later ratio infinite.
                    distance: first.pos.distance(&second.pos).max(1.0),
                });
                PinchStep::Began((first.pos + second.pos) * 0.5)
            }
            Some(spread) => match (contact(spread.first), contact(spread.second)) {
                (Some(first), Some(second)) => PinchStep::Moved(
                    first.pos.distance(&second.pos) / spread.distance,
                    (first.pos + second.pos) * 0.5,
                ),
                _ => {
                    self.spread = None;
                    PinchStep::Ended
                }
            },
            None => PinchStep::Nothing,
        }
    }

    pub(crate) fn pinching(&self) -> bool {
        self.spread.is_some()
    }

    /// Whether any finger that started a pinch is still down: its release
    /// must not read as a tap.
    fn holding(&self) -> bool {
        !self.contacts.is_empty()
    }
}

/// One finger followed from the raw touch stream (see GestureView's
/// handle_event): where and when it went down, and what it has become.
#[derive(Clone, Copy, Debug)]
struct TouchPress {
    uid: u64,
    start: DVec2,
    time: f64,
    panning: bool,
    /// Spent on a long press: its release is no tap.
    spent: bool,
    last_sent: DVec2,
}

#[derive(Script, ScriptHook, Widget)]
pub struct GestureView {
    #[deref]
    view: View,
    #[live]
    on_tap: ScriptFnRef,
    #[live]
    on_double_tap: ScriptFnRef,
    #[live]
    on_long_press: ScriptFnRef,
    #[live]
    on_swipe: ScriptFnRef,
    #[live]
    on_pan: ScriptFnRef,
    #[live]
    on_pinch: ScriptFnRef,

    /// This press is ours: no child had already captured it.
    #[rust]
    live: bool,
    /// The press moved past the tap slop: a drag, not a tap.
    #[rust]
    panning: bool,
    /// Last drag deltas sent, so a still finger sends nothing.
    #[rust]
    pan_sent: DVec2,
    #[rust]
    last_tap: Option<(f64, DVec2)>,
    #[rust]
    pinch: PinchTracker,
    #[rust]
    touch: Option<TouchPress>,
    #[rust]
    pinch_sent: f64,
    /// A pinch happened during this press: the fingers' release is not a
    /// tap, a swipe or the end of a drag.
    #[rust]
    pinched: bool,
}

impl GestureView {
    fn call(&self, cx: &mut Cx, script_fn: &ScriptFnRef, args: &[ScriptValue]) {
        if script_fn.as_object() == ScriptObject::ZERO {
            return;
        }
        cx.widget_to_script_call(self.widget_uid(), NIL, self.view.source.clone(), script_fn.clone(), args);
    }

    fn local(&self, cx: &Cx, abs: DVec2) -> DVec2 {
        abs - self.view.area().rect(cx).pos
    }

    fn handle_touches(&mut self, cx: &mut Cx, event: &TouchUpdateEvent) {
        let area = self.view.area();
        let bounds = area.clipped_rect(cx);
        for t in &event.touches {
            match t.state {
                TouchState::Start => {
                    if self.touch.is_some() || !bounds.contains(t.abs) {
                        continue;
                    }
                    let claimed = t.handled.get();
                    if !claimed.is_empty() && claimed != area && is_inside(cx, claimed, area) {
                        continue;
                    }
                    if !self.pinch.holding() || self.pinch.contacts.len() <= 1 {
                        self.pinched = false;
                    }
                    self.touch = Some(TouchPress { uid: t.uid, start: t.abs, time: t.time, panning: false, spent: false, last_sent: DVec2::default() });
                }
                TouchState::Move | TouchState::Stable => {
                    let Some(mut press) = self.touch.filter(|p| p.uid == t.uid) else { continue };
                    if self.pinch.pinching() || self.pinched {
                        continue;
                    }
                    let delta = t.abs - press.start;
                    if !press.panning && delta.length() > TAP_SLOP {
                        press.panning = true;
                        self.call(cx, &self.on_pan.clone(), &[0.0.into(), 0.0.into(), 0.0.into()]);
                    }
                    if press.panning && (delta - press.last_sent).length() >= 1.0 {
                        press.last_sent = delta;
                        self.call(cx, &self.on_pan.clone(), &[delta.x.into(), delta.y.into(), 1.0.into()]);
                    }
                    self.touch = Some(press);
                }
                TouchState::Stop => {
                    let Some(press) = self.touch.filter(|p| p.uid == t.uid) else { continue };
                    self.touch = None;
                    if self.pinched {
                        continue;
                    }
                    let delta = t.abs - press.start;
                    if press.panning {
                        self.call(cx, &self.on_pan.clone(), &[delta.x.into(), delta.y.into(), 2.0.into()]);
                        if let Some((dx, dy)) = swipe(delta.x, delta.y, t.time - press.time) {
                            self.call(cx, &self.on_swipe.clone(), &[dx.into(), dy.into()]);
                        }
                    } else if !press.spent && delta.length() <= TAP_SLOP && bounds.contains(t.abs) {
                        self.tap(cx, t.abs, t.time);
                    }
                }
            }
        }
    }

    fn tap(&mut self, cx: &mut Cx, abs: DVec2, time: f64) {
        let at = self.local(cx, abs);
        self.call(cx, &self.on_tap.clone(), &[at.x.into(), at.y.into()]);
        let double = self
            .last_tap
            .is_some_and(|(t, pos)| time - t <= DOUBLE_TAP_SECONDS && pos.distance(&abs) <= DOUBLE_TAP_DISTANCE);
        if double {
            self.last_tap = None;
            self.call(cx, &self.on_double_tap.clone(), &[at.x.into(), at.y.into()]);
        } else {
            self.last_tap = Some((time, abs));
        }
    }
}

/// Whether `inner` is a widget drawn inside `outer` (not `outer`'s own
/// rect, and not one around it).
fn is_inside(cx: &Cx, inner: Area, outer: Area) -> bool {
    let (rect, own) = (inner.rect(cx), outer.rect(cx));
    rect.is_inside_of(own) && rect.size != own.size
}

impl Widget for GestureView {
    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        self.view.draw_walk(cx, scope, walk)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        self.view.handle_event(cx, event, scope);
        if !self.view.visible {
            return;
        }

        if let Event::TouchUpdate(touches) = event {
            let bounds = self.view.area().clipped_rect(cx);
            match self.pinch.update(touches, bounds) {
                PinchStep::Began(center) => {
                    self.pinched = true;
                    self.pinch_sent = 1.0;
                    if self.panning {
                        // The drag a first finger started gives way to the pinch.
                        let at = self.pan_sent;
                        self.call(cx, &self.on_pan.clone(), &[at.x.into(), at.y.into(), 2.0.into()]);
                        self.panning = false;
                    }
                    let at = self.local(cx, center);
                    self.call(cx, &self.on_pinch.clone(), &[1.0.into(), at.x.into(), at.y.into(), 0.0.into()]);
                }
                PinchStep::Moved(scale, center) => {
                    if (scale / self.pinch_sent - 1.0).abs() >= PINCH_STEP {
                        self.pinch_sent = scale;
                        let at = self.local(cx, center);
                        self.call(cx, &self.on_pinch.clone(), &[scale.into(), at.x.into(), at.y.into(), 1.0.into()]);
                    }
                }
                PinchStep::Ended => {
                    let scale = self.pinch_sent;
                    self.call(cx, &self.on_pinch.clone(), &[scale.into(), 0.0.into(), 0.0.into(), 2.0.into()]);
                }
                PinchStep::Nothing => {}
            }
        }

        // Touch is read from the raw stream and never claimed. Claiming it,
        // even with a capture overload, marks the touch handled, and the
        // scroll view around this one then never starts its drag: a list of
        // GestureView rows would not scroll. A child that claimed the touch
        // (a Button inside) still keeps it.
        match event {
            Event::TouchUpdate(touches) => {
                self.handle_touches(cx, touches);
                return;
            }
            Event::LongPress(press) => {
                let fire = match self.touch.as_mut() {
                    Some(p) if p.uid == press.uid && !p.panning && !p.spent && !self.pinched => {
                        p.spent = true;
                        true
                    }
                    _ => false,
                };
                if fire {
                    let at = self.local(cx, press.abs);
                    self.call(cx, &self.on_long_press.clone(), &[at.x.into(), at.y.into()]);
                }
                return;
            }
            _ => {}
        }

        match event.hits_with_capture_overload(cx, self.view.area(), true) {
            Hit::FingerDown(e) => {
                // A child that already owns this press (a Button) keeps it. A
                // scroll view around us captures every press too; that one is
                // not a child, and the tap is still ours.
                self.live = !child_owns_press(cx, e.digit_id, self.view.area());
                self.panning = false;
                self.pan_sent = DVec2::default();
                if !self.pinch.holding() {
                    self.pinched = false;
                }
            }
            Hit::FingerMove(e) if self.live && !self.pinch.pinching() && !self.pinched => {
                let delta = e.abs - e.abs_start;
                if !self.panning && delta.length() > TAP_SLOP {
                    self.panning = true;
                    self.call(cx, &self.on_pan.clone(), &[0.0.into(), 0.0.into(), 0.0.into()]);
                }
                if self.panning && (delta - self.pan_sent).length() >= 1.0 {
                    self.pan_sent = delta;
                    self.call(cx, &self.on_pan.clone(), &[delta.x.into(), delta.y.into(), 1.0.into()]);
                }
            }
            Hit::FingerLongPress(e) if self.live && !self.panning && !self.pinched => {
                let at = self.local(cx, e.abs);
                // The press is spent on the long press: its release is no tap.
                self.live = false;
                self.call(cx, &self.on_long_press.clone(), &[at.x.into(), at.y.into()]);
            }
            Hit::FingerUp(e) if self.live => {
                self.live = false;
                if self.pinched {
                    return;
                }
                let delta = e.abs - e.abs_start;
                if self.panning {
                    self.panning = false;
                    self.call(cx, &self.on_pan.clone(), &[delta.x.into(), delta.y.into(), 2.0.into()]);
                    if let Some((dx, dy)) = swipe(delta.x, delta.y, e.time - e.capture_time) {
                        self.call(cx, &self.on_swipe.clone(), &[dx.into(), dy.into()]);
                    }
                    return;
                }
                if e.is_over && delta.length() <= TAP_SLOP && !e.has_long_press_occurred {
                    self.tap(cx, e.abs, e.time);
                }
            }
            _ => {}
        }
    }
}

/// Whether a widget inside `area` (not `area` itself, not one around it)
/// captured this press.
fn child_owns_press(cx: &Cx, digit_id: DigitId, area: Area) -> bool {
    cx.fingers
        .digit_capture_areas(digit_id)
        .into_iter()
        .any(|captured| captured != area && is_inside(cx, captured, area))
}

/// Of the sheet's own height: the peek, half and full resting heights.
const PEEK_HEIGHT: f64 = 132.0;
const HALF_FRACTION: f64 = 0.45;
/// What full height leaves above the sheet, so a search bar stays reachable.
const FULL_TOP_GAP: f64 = 96.0;
/// Pixels a second: faster than this at release is a flick to the next rest.
const FLICK_VELOCITY: f64 = 600.0;

/// The three resting heights in a viewport this tall, rising. In a short
/// viewport they close up rather than cross.
fn sheet_heights(viewport_h: f64) -> [f64; 3] {
    let half = viewport_h * HALF_FRACTION;
    [PEEK_HEIGHT.min(half), half, (viewport_h - FULL_TOP_GAP).max(half)]
}

/// Where a released drag comes to rest: the nearest height, or for a flick
/// the next one the way the finger was going (velocity is downwards, px/s).
fn sheet_rest(at: f64, velocity: f64, viewport_h: f64) -> usize {
    let heights = sheet_heights(viewport_h);
    if velocity <= -FLICK_VELOCITY {
        heights.iter().position(|h| *h > at + 0.5).unwrap_or(2)
    } else if velocity >= FLICK_VELOCITY {
        heights.iter().rposition(|h| *h < at - 0.5).unwrap_or(0)
    } else {
        let distance = |i: &usize| (heights[*i] - at).abs();
        (0..3).min_by(|a, b| distance(a).total_cmp(&distance(b))).unwrap_or(0)
    }
}

/// A bottom sheet: fills its parent, draws its children in a panel along the
/// bottom edge at the current detent's height, follows a drag that starts on
/// the panel, and eases to a detent on release. The app hears only where it
/// came to rest (`on_detent`), and can move it with `detent: 0|1|2`.
#[derive(Script, ScriptHook, Widget)]
pub struct SheetView {
    #[deref]
    view: View,
    /// The detent to rest at: 0 peek, 1 half, 2 full. Script may set it.
    #[live(0.0)]
    detent: f64,
    #[live]
    on_detent: ScriptFnRef,

    #[rust]
    height: f64,
    /// While a finger is on the panel: the height it started at, and the
    /// last (time, height) for the release velocity.
    #[rust]
    drag: Option<(f64, f64, f64)>,
    #[rust]
    prev: Option<(f64, f64)>,
    #[rust]
    viewport_h: f64,
    #[rust]
    next_frame: NextFrame,
}

impl SheetView {
    fn target(&self) -> f64 {
        sheet_heights(self.viewport_h)[self.detent.clamp(0.0, 2.0) as usize]
    }
}

impl Widget for SheetView {
    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        let outer = cx.peek_walk_turtle(walk);
        if outer.size.y > 0.0 {
            self.viewport_h = outer.size.y;
        }
        if self.height <= 0.0 {
            self.height = self.target();
        }
        let mut inner = walk;
        inner.height = Size::Fixed(self.height.max(0.0));
        inner.abs_pos = Some(dvec2(outer.pos.x, outer.pos.y + outer.size.y - self.height));
        cx.walk_turtle(walk);
        self.view.draw_walk(cx, scope, inner)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        self.view.handle_event(cx, event, scope);
        if let Some(ne) = self.next_frame.is_event(event) {
            let target = self.target();
            if self.drag.is_none() {
                // Ease a fixed share of the way each frame: settles in ~200 ms.
                let step = (target - self.height) * 0.25;
                if step.abs() < 0.5 {
                    self.height = target;
                } else {
                    self.height += step;
                    self.next_frame = cx.new_next_frame();
                }
                let _ = ne;
                self.view.redraw(cx);
            }
        }
        match event.hits_with_capture_overload(cx, self.view.area(), true) {
            Hit::FingerDown(e) => {
                if !child_owns_press(cx, e.digit_id, self.view.area()) {
                    self.drag = Some((self.height, e.time, self.height));
                    self.prev = None;
                }
            }
            Hit::FingerMove(e) => {
                if let Some((from, _, _)) = self.drag {
                    let [lowest, _, highest] = sheet_heights(self.viewport_h);
                    let now = (from - (e.abs.y - e.abs_start.y)).clamp(lowest, highest);
                    self.prev = self.drag.map(|(_, t, h)| (t, h));
                    self.drag = Some((from, e.time, now));
                    self.height = now;
                    self.view.redraw(cx);
                }
            }
            Hit::FingerUp(e) => {
                if let Some((_, time, at)) = self.drag.take() {
                    // Downwards velocity: the sheet shrinks as the finger descends.
                    let velocity = match self.prev {
                        Some((t0, h0)) if time > t0 => -(at - h0) / (time - t0),
                        _ => 0.0,
                    };
                    let _ = e;
                    let rest = sheet_rest(at, velocity, self.viewport_h) as f64;
                    let moved = rest != self.detent.clamp(0.0, 2.0).floor();
                    self.detent = rest;
                    self.next_frame = cx.new_next_frame();
                    if moved && self.on_detent.as_object() != ScriptObject::ZERO {
                        cx.widget_to_script_call(
                            self.widget_uid(),
                            NIL,
                            self.view.source.clone(),
                            self.on_detent.clone(),
                            &[rest.into()],
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::makepad_draw::makepad_platform::event::TouchPoint;
    use std::cell::Cell;

    fn touches(points: &[(u64, TouchState, f64, f64)]) -> TouchUpdateEvent {
        TouchUpdateEvent {
            time: 1.0,
            window_id: WindowId(0, 0),
            modifiers: KeyModifiers::default(),
            touches: points
                .iter()
                .map(|&(uid, state, x, y)| TouchPoint {
                    uid,
                    state,
                    abs: dvec2(x, y),
                    time: 1.0,
                    rotation_angle: 0.0,
                    force: 1.0,
                    radius: dvec2(1.0, 1.0),
                    handled: Cell::new(Area::Empty),
                    sweep_lock: Cell::new(Area::Empty),
                })
                .collect(),
        }
    }

    #[test]
    fn a_swipe_is_fast_and_along_one_axis() {
        assert_eq!(swipe(-120.0, 10.0, 0.2), Some((-120.0, 0.0)));
        assert_eq!(swipe(5.0, 90.0, 0.2), Some((0.0, 90.0)));
        assert_eq!(swipe(-120.0, 100.0, 0.2), None, "a diagonal is no swipe");
        assert_eq!(swipe(-120.0, 0.0, 1.0), None, "a slow drag is no swipe");
        assert_eq!(swipe(-30.0, 0.0, 0.1), None, "nor is a short one");
    }

    #[test]
    fn two_fingers_inside_the_view_pinch_relative_to_their_start() {
        let bounds = Rect { pos: dvec2(0.0, 0.0), size: dvec2(400.0, 400.0) };
        let mut pinch = PinchTracker::default();
        assert_eq!(pinch.update(&touches(&[(1, TouchState::Start, 100.0, 200.0)]), bounds), PinchStep::Nothing);
        assert_eq!(
            pinch.update(&touches(&[(2, TouchState::Start, 300.0, 200.0)]), bounds),
            PinchStep::Began(dvec2(200.0, 200.0))
        );
        assert_eq!(
            pinch.update(&touches(&[(2, TouchState::Move, 500.0, 200.0)]), bounds),
            PinchStep::Moved(2.0, dvec2(300.0, 200.0))
        );
        assert_eq!(pinch.update(&touches(&[(1, TouchState::Stop, 100.0, 200.0)]), bounds), PinchStep::Ended);
        assert!(pinch.holding(), "a finger is still down: its release is not a tap");
        pinch.update(&touches(&[(2, TouchState::Stop, 500.0, 200.0)]), bounds);
        assert!(!pinch.holding());
    }

    #[test]
    fn a_finger_that_starts_outside_the_view_is_not_part_of_its_pinch() {
        let bounds = Rect { pos: dvec2(0.0, 0.0), size: dvec2(100.0, 100.0) };
        let mut pinch = PinchTracker::default();
        pinch.update(&touches(&[(1, TouchState::Start, 50.0, 50.0)]), bounds);
        assert_eq!(pinch.update(&touches(&[(2, TouchState::Start, 500.0, 50.0)]), bounds), PinchStep::Nothing);
    }

    #[test]
    fn the_sheet_rests_at_the_nearest_detent_or_flicks_to_the_next() {
        assert_eq!(sheet_heights(800.0), [132.0, 360.0, 704.0]);
        assert_eq!(sheet_rest(200.0, 0.0, 800.0), 0);
        assert_eq!(sheet_rest(300.0, 0.0, 800.0), 1);
        assert_eq!(sheet_rest(200.0, -900.0, 800.0), 1, "an upward flick goes to the next detent up");
        assert_eq!(sheet_rest(650.0, 900.0, 800.0), 1, "a downward flick to the next one down");
        let [peek, half, full] = sheet_heights(200.0);
        assert!(peek <= half && half <= full && full <= 200.0);
    }
}
