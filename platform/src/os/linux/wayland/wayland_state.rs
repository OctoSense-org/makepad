#![allow(unused_imports, unused_variables)]
use crate::{
    libc_sys::{self, munmap},
    makepad_math::{dvec2, Vec2d},
    wayland::{wayland_type, xkb_sys},
    Area, KeyEvent, KeyModifiers, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent,
    TextClipboardEvent, TextInputEvent, WindowClosedEvent, WindowDragQueryEvent,
    WindowDragQueryResponse,
};
use std::{
    cell::{Cell, RefCell},
    os::fd::{AsFd, AsRawFd, FromRawFd},
    rc::Rc,
    sync::Arc,
};

use wayland_client::{
    delegate_noop,
    protocol::{
        wl_buffer, wl_callback, wl_compositor, wl_data_device, wl_data_device_manager,
        wl_data_offer, wl_data_source, wl_keyboard, wl_output,
        wl_pointer::{self, ButtonState},
        wl_registry, wl_seat, wl_shm, wl_shm_pool, wl_surface,
    },
    Connection, Dispatch, Proxy, QueueHandle, WEnum,
};
use wayland_protocols::{
    wp::{
        cursor_shape::v1::client::{
            wp_cursor_shape_device_v1,
            wp_cursor_shape_manager_v1::{self, WpCursorShapeManagerV1},
        },
        fractional_scale::v1::client::{wp_fractional_scale_manager_v1, wp_fractional_scale_v1},
        primary_selection::zv1::client::{
            zwp_primary_selection_device_manager_v1, zwp_primary_selection_device_v1,
            zwp_primary_selection_offer_v1, zwp_primary_selection_source_v1,
        },
        text_input::zv3::client::{zwp_text_input_manager_v3, zwp_text_input_v3},
        viewporter::client::{wp_viewport, wp_viewporter},
    },
    xdg::{
        self,
        decoration::zv1::client::{zxdg_decoration_manager_v1, zxdg_toplevel_decoration_v1},
        shell::client::{xdg_popup, xdg_positioner, xdg_surface, xdg_toplevel, xdg_wm_base},
        toplevel_icon::v1::client::{xdg_toplevel_icon_manager_v1, xdg_toplevel_icon_v1},
    },
};

use crate::{
    cx_native::EventFlow,
    event::{PopupDismissReason, PopupDismissedEvent, ScrollEvent, ScrollPhase, WindowGeom},
    select_timer::SelectTimers,
    wayland::wayland_app::WaylandApp,
    x11::xlib_event::XlibEvent,
    KeyCode, WindowCloseRequestedEvent, WindowGeomChangeEvent, WindowId, WindowMovedEvent,
};

use super::super::windowing_backend::PIXELS_PER_WHEEL_DETENT;
use super::opengl_wayland::{WaylandPopupWindow, WaylandWindow};

/// Reserved timer ID for keyboard repeat. Uses a high value to avoid conflicts with app timers.
const KEY_REPEAT_TIMER_ID: u64 = u64::MAX - 1;

/// Whether a pointer frame's scroll came from a wheel-like source: one that ratchets in
/// coarse steps, so its detents drive the delta, it carries no gesture phase, and it is
/// reported to widgets as mouse input.
///
/// `source` is the frame's `wl_pointer::AxisSource`, or `None` when the compositor sent
/// none — the event is optional and only sent when the source is known. With no source,
/// the detents settle it: a device without discrete steps does not generate them, which
/// the spec spells out for `axis_discrete`. Guessing wheel-like is in any case the safe
/// guess, being the one classification that cannot strand a stretched rubber band waiting
/// for a terminator the spec does not promise.
fn scroll_is_wheel_like(source: Option<wl_pointer::AxisSource>, has_detents: bool) -> bool {
    match source {
        Some(wl_pointer::AxisSource::Wheel) | Some(wl_pointer::AxisSource::WheelTilt) => true,
        // A trackpoint or button-held scroll is smooth, so it takes the raw pixel path even
        // though it is not a gesture.
        Some(wl_pointer::AxisSource::Finger) | Some(wl_pointer::AxisSource::Continuous) => false,
        _ => has_detents,
    }
}

/// What one pointer frame's axis events add up to.
struct FrameScroll {
    /// The delta in logical pixels.
    delta: Vec2d,
    phase: ScrollPhase,
    /// Reported as `ScrollEvent::is_mouse`: a wheel that ratchets in steps, which widgets may
    /// ease between. False for every smooth source, which needs no easing.
    is_mouse: bool,
}

/// Resolve a pointer frame's accumulated axis events into one scroll, or `None` when the
/// frame carries nothing worth dispatching.
///
/// `source` is the frame's `wl_pointer::AxisSource` (`None` if the compositor sent none),
/// `gesture_active` whether the previous frame was a live touchpad gesture, and `stopped`
/// whether an `AxisStop` arrived in this frame.
fn frame_scroll(
    source: Option<wl_pointer::AxisSource>,
    gesture_active: bool,
    stopped: bool,
    acc: Vec2d,
    detents: Vec2d,
) -> Option<FrameScroll> {
    let has_detents = detents.x != 0.0 || detents.y != 0.0;
    let has_delta = acc.x != 0.0 || acc.y != 0.0 || has_detents;
    let is_wheel_like = scroll_is_wheel_like(source, has_detents);
    // `axis_source` is per-frame and optional, so a compositor may name the source on a
    // gesture's motion frames and omit it on the lift-off frame. Treating that frame as
    // sourceless would drop the terminator and leave a stretched rubber band with nothing
    // to release it, so a gesture already in flight carries its classification forward --
    // but never over a frame whose detents say it is a wheel.
    let is_finger = match source {
        Some(wl_pointer::AxisSource::Finger) => true,
        None => gesture_active && !is_wheel_like,
        _ => false,
    };
    // A stop alongside live motion is not the end of the gesture. Per the `frame` event:
    // "When a wl_pointer.axis and a wl_pointer.axis_stop event occur within the same frame,
    // this indicates that axis movement in one axis has stopped but continues in the other
    // axis." The lift-off frame that does end the gesture carries its stops alone.
    let gesture_ended = is_finger && stopped && !has_delta;
    if !has_delta && !gesture_ended {
        // Only `Finger` is guaranteed an `AxisStop`; the spec tells clients to treat wheel,
        // wheel_tilt and continuous sequences "as unterminated by default". A bare stop from
        // one of those says nothing, and dispatching a zero-delta `ScrollPhase::None` for it
        // would clear a widget's overscroll and cut short a running bounce.
        return None;
    }
    // Scale wheel detents to a fixed distance each, so slow deliberate clicks and fast spins
    // both move proportionally. Decided per axis: a frame can carry detents on one axis and
    // only a smooth value on the other, and scaling that second axis by a zero detent count
    // would silently drop it.
    //
    // An axis with no detents keeps its raw value. Compositors pair a detent event with every
    // wheel-source axis event — `axis_discrete` is documented as absent only for continuous
    // devices — and the seat binds above the v5 that introduced it, so a physical wheel
    // always brings one. The fallback is for virtual pointers: `zwlr_virtual_pointer_v1` lets
    // a client send a wheel-source axis value with no discrete step, and `wl_pointer.axis`
    // defines that value as a "length of vector in surface-local coordinate space" — already
    // a distance, with no detent count to recover and no units-per-detent constant that could
    // recover one (compositors disagree, and hwdb ships wheels from 10 to 30 degrees a click).
    let axis_scroll = |detent: f64, raw: f64| {
        if detent != 0.0 {
            detent * PIXELS_PER_WHEEL_DETENT
        } else {
            raw
        }
    };
    // Finger-driven (touchpad) scrolling reports `Changed` per frame and `Ended` when the
    // fingers lift, which is what drives the rubber band at a scroll limit. Every other
    // source is a plain delta with no gesture.
    //
    // Note this yields no kinetic scrolling for Wayland touchpads: widgets start their fling
    // on `ScrollPhase::Momentum`, which only macOS emits — there the OS synthesizes that
    // stream, while Wayland compositors do not and neither Linux backend fabricates one.
    let phase = if !is_finger {
        ScrollPhase::None
    } else if gesture_ended {
        ScrollPhase::Ended
    } else {
        ScrollPhase::Changed
    };
    Some(FrameScroll {
        delta: if is_wheel_like {
            dvec2(axis_scroll(detents.x, acc.x), axis_scroll(detents.y, acc.y))
        } else {
            acc
        },
        phase,
        is_mouse: is_wheel_like,
    })
}

/// State for tracking keyboard key repeat.
struct KeyRepeatState {
    key_code: KeyCode,
    text: String,
    /// True while waiting for the initial delay; false during steady-state repeat.
    in_initial_delay: bool,
}

pub(crate) struct ClipboardOffer {
    offer: wl_data_offer::WlDataOffer,
    mime_types: Vec<String>,
}

struct PendingClipboardRead {
    fd: std::os::fd::OwnedFd,
    bytes: Vec<u8>,
}

pub(crate) struct WaylandState {
    pub(crate) compositor: Option<wl_compositor::WlCompositor>,
    pub(crate) wm_base: Option<xdg_wm_base::XdgWmBase>,
    pub(crate) seat: Option<wl_seat::WlSeat>,
    pub(crate) shm: Option<wl_shm::WlShm>,
    pub(crate) data_device_manager: Option<wl_data_device_manager::WlDataDeviceManager>,
    pub(crate) data_device: Option<wl_data_device::WlDataDevice>,
    pub(crate) clipboard_source: Option<wl_data_source::WlDataSource>,
    pub(crate) clipboard_offer: Option<ClipboardOffer>,
    pub(crate) data_offers: Vec<ClipboardOffer>,
    pending_clipboard_read: Option<PendingClipboardRead>,
    pending_paste_text_input: Option<String>,
    /// Queued clipboard copy content waiting for a serial from keyboard/pointer.
    pub(crate) pending_clipboard_copy: Option<String>,
    pub(crate) clipboard_text: String,
    pub(crate) cursor_manager: Option<wp_cursor_shape_manager_v1::WpCursorShapeManagerV1>,
    pub(crate) cursor_shape: Option<wp_cursor_shape_device_v1::WpCursorShapeDeviceV1>,
    pub(crate) pointer: Option<wl_pointer::WlPointer>,
    pub(crate) last_mouse_pos: Vec2d,
    pub(crate) pointer_serial: Option<u32>,
    pub(crate) keyboard_serial: Option<u32>,
    pub(crate) decoration_manager: Option<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1>,
    pub(crate) icon_manager: Option<xdg_toplevel_icon_manager_v1::XdgToplevelIconManagerV1>,
    pub(crate) windows: Vec<WaylandWindow>,
    pub(crate) popups: Vec<WaylandPopupWindow>,
    pub(crate) pointer_window: Option<WindowId>,
    /// The latest un-dispatched pointer motion `(window_id, pos)`, coalesced across a whole
    /// `dispatch_pending` batch. A high-Hz mouse queues many `wl_pointer` motion+frame pairs between
    /// paints; dispatching each as a `MouseMove` runs a redundant hover hit-test across the whole
    /// widget tree, stealing frame budget during a fling. We keep only the latest and flush it once
    /// after the queue is drained (and before any intervening button/leave, to preserve ordering),
    /// mirroring the Windows `coalesce_mouse_move`. See [`Self::flush_pending_motion`].
    pub(crate) pending_motion: Option<(WindowId, Vec2d)>,
    pub(crate) keyboard_window: Option<WindowId>,
    pub(crate) modifiers: KeyModifiers,
    pub(crate) timers: SelectTimers,
    pub(crate) scale_manager: Option<wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1>,
    pub(crate) viewporter: Option<wp_viewporter::WpViewporter>,
    pub(crate) xkb_state: Option<xkb_sys::XkbState>,
    pub(crate) xkb_cx: xkb_sys::XkbContext,
    pub(crate) text_input: Option<zwp_text_input_v3::ZwpTextInputV3>,
    pub(crate) text_input_manager: Option<zwp_text_input_manager_v3::ZwpTextInputManagerV3>,
    /// zwp_text_input_v3 double-buffers preedit/commit; these accumulate the
    /// pending IME state until the matching `Done` event applies it.
    text_input_pending_preedit: Option<String>,
    text_input_pending_commit: Option<String>,
    /// Last composition preview forwarded to the widget, to skip redundant updates.
    text_input_last_preedit: String,
    pub(crate) primary_selection_manager:
        Option<zwp_primary_selection_device_manager_v1::ZwpPrimarySelectionDeviceManagerV1>,
    pub(crate) primary_selection_device:
        Option<zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1>,
    pub(crate) primary_selection_source:
        Option<zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1>,
    pub(crate) primary_selection_text: String,
    pub(crate) last_resize_edge: Option<xdg_toplevel::ResizeEdge>,
    event_callback: Option<Box<dyn FnMut(&mut WaylandState, XlibEvent)>>,

    pub(crate) scroll_accumulator: Vec2d,
    /// Wheel detents accumulated over the current pointer frame, from `AxisValue120`
    /// (fractional detents on high-resolution wheels) or `AxisDiscrete` on pre-v8
    /// compositors. Same sign convention as `scroll_accumulator`.
    pub(crate) scroll_detents: Vec2d,
    /// The `wl_pointer::AxisSource` reported for the current pointer frame, or `None`
    /// when the compositor sent no `AxisSource` event — the event is optional ("If the
    /// source is unknown for a particular axis event sequence, no event is sent") and a
    /// source value newer than this protocol copy is likewise recorded as `None`. Scoped
    /// to one frame, so it resets on every `Frame` and must never be assumed to carry
    /// over. See [`scroll_is_wheel_like`].
    pub(crate) scroll_source: Option<wl_pointer::AxisSource>,
    /// Set when `wl_pointer::AxisStop` arrives in the current pointer frame. It ends the
    /// gesture — sending that frame's Scroll event with `ScrollPhase::Ended`, which springs
    /// a stretched rubber band back and releases the widget's gesture ownership — only on a
    /// finger frame that carries no motion of its own. A stop alongside live motion means
    /// that one axis stopped while the other continues (see the `frame` event), not lift-off,
    /// and a stop from a source that is not a gesture says nothing at all. See
    /// [`frame_scroll`].
    pub(crate) scroll_stopped: bool,
    /// Whether the last dispatched pointer frame was a live touchpad gesture, so that a
    /// lift-off frame on which the compositor omitted its `AxisSource` is still recognised
    /// as the end of that gesture rather than as an unclassified scroll.
    pub(crate) scroll_gesture_active: bool,
    /// Windows whose last presented frame's `wl_surface::frame` callback has not fired
    /// yet. While a window is listed here the compositor is not ready for a new frame
    /// on that surface, so presenting it is skipped (its pass stays dirty). See the
    /// frame-callback pacing in `linux_wayland.rs`.
    frame_callbacks_pending: Vec<WindowId>,
    pub(crate) event_flow: EventFlow,
    pub(crate) event_loop_running: bool,

    /// Keyboard repeat rate in keys per second (0 = disabled).
    key_repeat_rate: i32,
    /// Keyboard repeat delay in milliseconds before repeat starts.
    key_repeat_delay: i32,
    /// Currently repeating key state, if any.
    key_repeat: Option<KeyRepeatState>,
}

impl WaylandState {
    pub fn new(event_callback: Box<dyn FnMut(&mut WaylandState, XlibEvent)>) -> Self {
        Self {
            compositor: None,
            wm_base: None,
            seat: None,
            shm: None,
            data_device_manager: None,
            data_device: None,
            clipboard_source: None,
            clipboard_offer: None,
            data_offers: Vec::new(),
            pending_clipboard_read: None,
            pending_paste_text_input: None,
            pending_clipboard_copy: None,
            clipboard_text: String::new(),
            cursor_manager: None,
            cursor_shape: None,
            pointer: None,
            decoration_manager: None,
            icon_manager: None,
            scale_manager: None,
            viewporter: None,
            windows: Vec::new(),
            popups: Vec::new(),
            pointer_window: None,
            pending_motion: None,
            keyboard_window: None,
            pointer_serial: None,
            keyboard_serial: None,
            modifiers: KeyModifiers::default(),
            xkb_state: None,
            xkb_cx: xkb_sys::XkbContext::new().unwrap(),
            text_input: None,
            text_input_manager: None,
            text_input_pending_preedit: None,
            text_input_pending_commit: None,
            text_input_last_preedit: String::new(),
            primary_selection_manager: None,
            primary_selection_device: None,
            primary_selection_source: None,
            primary_selection_text: String::new(),
            last_mouse_pos: dvec2(0., 0.),
            last_resize_edge: None,
            timers: SelectTimers::new(),
            event_callback: Some(event_callback),
            scroll_accumulator: dvec2(0.0, 0.0),
            scroll_detents: dvec2(0.0, 0.0),
            scroll_source: None,
            scroll_gesture_active: false,
            scroll_stopped: false,
            frame_callbacks_pending: Vec::new(),
            event_flow: EventFlow::Wait,
            event_loop_running: true,
            key_repeat_rate: 25,
            key_repeat_delay: 600,
            key_repeat: None,
        }
    }

    fn window_id_for_surface(&self, surface: &wl_surface::WlSurface) -> Option<WindowId> {
        let surface_id = surface.id();
        self.windows
            .iter()
            .find(|win| win.base_surface.id() == surface_id)
            .map(|win| win.window_id)
            .or_else(|| {
                self.popups
                    .iter()
                    .find(|win| win.base_surface.id() == surface_id)
                    .map(|win| win.window_id)
            })
    }

    pub(crate) fn xdg_surface_for_window(
        &self,
        window_id: WindowId,
    ) -> Option<xdg_surface::XdgSurface> {
        self.windows
            .iter()
            .find(|win| win.window_id == window_id)
            .map(|win| win.xdg_surface.clone())
            .or_else(|| {
                self.popups
                    .iter()
                    .find(|win| win.window_id == window_id)
                    .map(|win| win.xdg_surface.clone())
            })
    }
}

impl Dispatch<wl_registry::WlRegistry, ()> for WaylandState {
    fn event(
        state: &mut Self,
        wl_registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        conn: &Connection,
        qhandle: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_compositor" => {
                    let compositor =
                        wl_registry.bind::<wl_compositor::WlCompositor, _, _>(name, 1, qhandle, ());
                    state.compositor = Some(compositor);
                }
                "xdg_wm_base" => {
                    let wm_base =
                        wl_registry.bind::<xdg_wm_base::XdgWmBase, _, _>(name, 1, qhandle, ());
                    state.wm_base = Some(wm_base);
                }
                "wl_seat" => {
                    // Version 8 adds wl_pointer::AxisValue120 for high-resolution wheel
                    // detents (replacing AxisDiscrete on v8+ compositors). Note the v7+
                    // requirement that keymap fds be mapped MAP_PRIVATE.
                    let seat = wl_registry.bind::<wl_seat::WlSeat, _, _>(
                        name,
                        version.min(9),
                        qhandle,
                        (),
                    );
                    state.seat = Some(seat);
                    state.ensure_data_device(qhandle);
                }
                "wl_data_device_manager" => {
                    let data_device_manager = wl_registry
                        .bind::<wl_data_device_manager::WlDataDeviceManager, _, _>(
                        name,
                        version.min(3),
                        qhandle,
                        (),
                    );
                    state.data_device_manager = Some(data_device_manager);
                    state.ensure_data_device(qhandle);
                }
                "zxdg_decoration_manager_v1" => {
                    let decoration_manager = wl_registry
                        .bind::<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1, _, _>(
                        name,
                        1,
                        qhandle,
                        (),
                    );
                    state.decoration_manager = Some(decoration_manager);
                }
                "wp_cursor_shape_manager_v1" => {
                    let cursor =
                        wl_registry.bind::<WpCursorShapeManagerV1, _, _>(name, 1, qhandle, ());
                    state.cursor_manager = Some(cursor);
                }
                "wp_fractional_scale_manager_v1" => {
                    let scale_manager = wl_registry
                        .bind::<wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1, _, _>(
                        name,
                        1,
                        qhandle,
                        (),
                    );
                    state.scale_manager = Some(scale_manager);
                }
                "wp_viewporter" => {
                    let viewporter =
                        wl_registry.bind::<wp_viewporter::WpViewporter, _, _>(name, 1, qhandle, ());
                    state.viewporter = Some(viewporter);
                }
                "wl_shm" => {
                    let shm = wl_registry.bind::<wl_shm::WlShm, _, _>(name, 1, qhandle, ());
                    state.shm = Some(shm);
                }
                "xdg_toplevel_icon_manager_v1" => {
                    let icon_manager = wl_registry
                        .bind::<xdg_toplevel_icon_manager_v1::XdgToplevelIconManagerV1, _, _>(
                        name,
                        1,
                        qhandle,
                        (),
                    );
                    state.icon_manager = Some(icon_manager);
                }
                "zwp_text_input_manager_v3" => {
                    let text_input_manager = wl_registry
                        .bind::<zwp_text_input_manager_v3::ZwpTextInputManagerV3, _, _>(
                        name,
                        1,
                        qhandle,
                        (),
                    );
                    state.text_input_manager = Some(text_input_manager);
                }
                "zwp_primary_selection_device_manager_v1" => {
                    let manager = wl_registry
                        .bind::<zwp_primary_selection_device_manager_v1::ZwpPrimarySelectionDeviceManagerV1, _, _>(
                        name,
                        1,
                        qhandle,
                        (),
                    );
                    state.primary_selection_manager = Some(manager);
                    state.ensure_primary_selection_device(qhandle);
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for WaylandState {
    fn event(
        state: &mut Self,
        wm_base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        conn: &Connection,
        qhandle: &QueueHandle<Self>,
    ) {
        match event {
            xdg_wm_base::Event::Ping { serial } => wm_base.pong(serial),
            _ => {}
        }
    }
}

impl Dispatch<wp_fractional_scale_v1::WpFractionalScaleV1, WindowId> for WaylandState {
    fn event(
        state: &mut Self,
        fractional_scale: &wp_fractional_scale_v1::WpFractionalScaleV1,
        event: wp_fractional_scale_v1::Event,
        window_id: &WindowId,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wp_fractional_scale_v1::Event::PreferredScale { scale } => {
                if let Some(window) = state
                    .windows
                    .iter_mut()
                    .find(|win| win.window_id == *window_id)
                {
                    let old_geom = window.window_geom.clone();
                    let mut new_geom = window.window_geom.clone();
                    new_geom.dpi_factor = scale as f64 / 120.;
                    state.do_callback(XlibEvent::WindowGeomChange(WindowGeomChangeEvent {
                        window_id: *window_id,
                        old_geom,
                        new_geom,
                    }));
                } else if let Some(window) = state
                    .popups
                    .iter_mut()
                    .find(|win| win.window_id == *window_id)
                {
                    let old_geom = window.window_geom.clone();
                    let mut new_geom = window.window_geom.clone();
                    new_geom.dpi_factor = scale as f64 / 120.;
                    state.do_callback(XlibEvent::WindowGeomChange(WindowGeomChangeEvent {
                        window_id: *window_id,
                        old_geom,
                        new_geom,
                    }));
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, WindowId> for WaylandState {
    fn event(
        state: &mut Self,
        xdg_surface: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        window_id: &WindowId,
        conn: &Connection,
        qhandle: &QueueHandle<Self>,
    ) {
        match event {
            xdg_toplevel::Event::Configure {
                width,
                height,
                states,
            } => {
                if let Some(window) = state.windows.iter().find(|win| win.window_id == *window_id) {
                    let inner_size = if width > 0 && height > 0 {
                        dvec2(width as f64, height as f64)
                    } else {
                        window.window_geom.inner_size
                    };
                    let is_maximized =
                        WaylandState::xdg_toplevel_has_state(&states, 1 /* maximized */);
                    let is_fullscreen =
                        WaylandState::xdg_toplevel_has_state(&states, 2 /* fullscreen */);
                    state.do_callback(XlibEvent::WindowGeomChange(WindowGeomChangeEvent {
                        window_id: *window_id,
                        old_geom: window.window_geom.clone(),
                        new_geom: WindowGeom {
                            dpi_factor: window.window_geom.dpi_factor,
                            can_fullscreen: false,
                            xr_is_presenting: false,
                            is_fullscreen: is_fullscreen || is_maximized,
                            is_topmost: false,
                            position: dvec2(0., 0.),
                            inner_size,
                            outer_size: inner_size,
                            ..Default::default()
                        },
                    }));
                }
            }
            xdg_toplevel::Event::Close => {
                let accept_close = Rc::new(Cell::new(true));
                state.do_callback(XlibEvent::WindowCloseRequested(WindowCloseRequestedEvent {
                    window_id: *window_id,
                    accept_close,
                }))
            }
            _ => {}
        }
    }
}
impl Dispatch<xdg_surface::XdgSurface, WindowId> for WaylandState {
    fn event(
        state: &mut Self,
        xdg_surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        window_id: &WindowId,
        conn: &Connection,
        qhandle: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial, .. } = event {
            xdg_surface.ack_configure(serial);
            let mut first_configure_event = None;
            if let Some(window) = state
                .windows
                .iter_mut()
                .find(|win| win.window_id == *window_id)
            {
                if !window.configured {
                    let mut old_geom = window.window_geom.clone();
                    old_geom.inner_size = dvec2(0., 0.);
                    old_geom.outer_size = dvec2(0., 0.);
                    first_configure_event = Some(WindowGeomChangeEvent {
                        window_id: *window_id,
                        old_geom,
                        new_geom: window.window_geom.clone(),
                    });
                }
                window.configured = true;
            } else if let Some(window) = state
                .popups
                .iter_mut()
                .find(|win| win.window_id == *window_id)
            {
                if !window.configured {
                    let mut old_geom = window.window_geom.clone();
                    old_geom.inner_size = dvec2(0., 0.);
                    old_geom.outer_size = dvec2(0., 0.);
                    first_configure_event = Some(WindowGeomChangeEvent {
                        window_id: *window_id,
                        old_geom,
                        new_geom: window.window_geom.clone(),
                    });
                }
                window.configured = true;
            }
            if let Some(event) = first_configure_event {
                state.do_callback(XlibEvent::WindowGeomChange(event));
            }
        }
    }
}

impl Dispatch<xdg_popup::XdgPopup, WindowId> for WaylandState {
    fn event(
        state: &mut Self,
        _xdg_popup: &xdg_popup::XdgPopup,
        event: xdg_popup::Event,
        window_id: &WindowId,
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            xdg_popup::Event::Configure {
                x,
                y,
                width,
                height,
            } => {
                let mut geom_change = None;
                if let Some(popup) = state
                    .popups
                    .iter_mut()
                    .find(|popup| popup.window_id == *window_id)
                {
                    let old_geom = popup.window_geom.clone();
                    popup.window_geom.position = dvec2(x as f64, y as f64);
                    if width > 0 && height > 0 {
                        popup.window_geom.inner_size = dvec2(width as f64, height as f64);
                        popup.window_geom.outer_size = popup.window_geom.inner_size;
                    }
                    if popup.window_geom != old_geom {
                        geom_change = Some(WindowGeomChangeEvent {
                            window_id: *window_id,
                            old_geom,
                            new_geom: popup.window_geom.clone(),
                        });
                    }
                }
                if let Some(event) = geom_change {
                    state.do_callback(XlibEvent::WindowGeomChange(event));
                }
            }
            xdg_popup::Event::PopupDone => {
                // WindowClosed must fire before PopupDismissed so the
                // platform can access the CxWindow pool entry (valid
                // generation) before the app drops its WindowHandle
                // which frees the pool slot.
                state.do_callback(XlibEvent::WindowClosed(WindowClosedEvent {
                    window_id: *window_id,
                }));
                state.do_callback(XlibEvent::PopupDismissed(PopupDismissedEvent {
                    window_id: *window_id,
                    reason: PopupDismissReason::Compositor,
                }));
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for WaylandState {
    fn event(
        state: &mut Self,
        seat: &wl_seat::WlSeat,
        event: wl_seat::Event,
        _: &(),
        conn: &Connection,
        qhandle: &QueueHandle<Self>,
    ) {
        state.ensure_data_device(qhandle);
        if let Some(input_manager) = state.text_input_manager.as_ref() {
            state.text_input = Some(input_manager.get_text_input(&seat, qhandle, ()));
        }
        if let wl_seat::Event::Capabilities {
            capabilities: WEnum::Value(capabilities),
        } = event
        {
            if capabilities.contains(wl_seat::Capability::Keyboard) {
                seat.get_keyboard(qhandle, ());
            }
            if capabilities.contains(wl_seat::Capability::Pointer) {
                let pointer = seat.get_pointer(qhandle, ());
                if let Some(manager) = state.cursor_manager.as_ref() {
                    state.cursor_shape = Some(manager.get_pointer(&pointer, qhandle, ()));
                }
                state.pointer = Some(pointer);
            }
        }
    }
}

impl Dispatch<wl_data_device::WlDataDevice, ()> for WaylandState {
    fn event(
        state: &mut Self,
        _proxy: &wl_data_device::WlDataDevice,
        event: wl_data_device::Event,
        _: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            wl_data_device::Event::DataOffer { id } => {
                if state.data_offers.iter().all(|entry| entry.offer != id) {
                    state.data_offers.push(ClipboardOffer {
                        offer: id,
                        mime_types: Vec::new(),
                    });
                }
            }
            wl_data_device::Event::Selection { id } => {
                state.clipboard_offer = id.map(|offer| {
                    if let Some(index) = state
                        .data_offers
                        .iter()
                        .position(|entry| entry.offer == offer)
                    {
                        state.data_offers.swap_remove(index)
                    } else {
                        ClipboardOffer {
                            offer,
                            mime_types: Vec::new(),
                        }
                    }
                });
                state.data_offers.clear();
            }
            _ => {}
        }
    }

    fn event_created_child(
        opcode: u16,
        qhandle: &QueueHandle<Self>,
    ) -> Arc<dyn wayland_client::backend::ObjectData> {
        match opcode {
            wl_data_device::EVT_DATA_OFFER_OPCODE => {
                qhandle.make_data::<wl_data_offer::WlDataOffer, ()>(())
            }
            _ => unreachable!("wl_data_device created unknown child for opcode {}", opcode),
        }
    }
}

impl Dispatch<wl_data_offer::WlDataOffer, ()> for WaylandState {
    fn event(
        state: &mut Self,
        proxy: &wl_data_offer::WlDataOffer,
        event: wl_data_offer::Event,
        _: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            wl_data_offer::Event::Offer { mime_type } => {
                if let Some(active_offer) = state.clipboard_offer.as_mut() {
                    if active_offer.offer == *proxy
                        && !active_offer.mime_types.iter().any(|m| m == &mime_type)
                    {
                        active_offer.mime_types.push(mime_type.clone());
                    }
                }
                if let Some(offer) = state
                    .data_offers
                    .iter_mut()
                    .find(|entry| entry.offer == *proxy)
                {
                    if !offer.mime_types.iter().any(|m| m == &mime_type) {
                        offer.mime_types.push(mime_type);
                    }
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_data_source::WlDataSource, ()> for WaylandState {
    fn event(
        state: &mut Self,
        proxy: &wl_data_source::WlDataSource,
        event: wl_data_source::Event,
        _: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            wl_data_source::Event::Send { mime_type, fd } => {
                if Self::is_text_mime_type(&mime_type) {
                    let raw_fd = fd.as_raw_fd();
                    unsafe {
                        let flags = libc_sys::fcntl(raw_fd, libc_sys::F_GETFL, 0);
                        if flags >= 0 {
                            let _ = libc_sys::fcntl(
                                raw_fd,
                                libc_sys::F_SETFL,
                                flags | libc_sys::O_NONBLOCK,
                            );
                        }
                        let bytes = state.clipboard_text.as_bytes();
                        let _ = libc_sys::write(
                            raw_fd,
                            bytes.as_ptr() as *const std::os::raw::c_void,
                            bytes.len(),
                        );
                    }
                }
            }
            wl_data_source::Event::Cancelled => {
                if state
                    .clipboard_source
                    .as_ref()
                    .is_some_and(|source| source == proxy)
                {
                    state.clipboard_source = None;
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_data_device_manager::WlDataDeviceManager, ()> for WaylandState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_data_device_manager::WlDataDeviceManager,
        _event: wl_data_device_manager::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zwp_text_input_v3::ZwpTextInputV3, ()> for WaylandState {
    fn event(
        state: &mut Self,
        proxy: &zwp_text_input_v3::ZwpTextInputV3,
        event: <zwp_text_input_v3::ZwpTextInputV3 as Proxy>::Event,
        data: &(),
        conn: &Connection,
        qhandle: &QueueHandle<Self>,
    ) {
        match event {
            zwp_text_input_v3::Event::Enter { surface } => {}
            zwp_text_input_v3::Event::Leave { surface } => {}
            zwp_text_input_v3::Event::PreeditString {
                text,
                cursor_begin: _,
                cursor_end: _,
            } => {
                // Double-buffered: stash the preedit (composition) text and apply
                // it on the matching `Done`. A `None`/absent preedit means the
                // composition preview should be cleared for this cycle.
                state.text_input_pending_preedit = text;
            }
            zwp_text_input_v3::Event::CommitString { text } => {
                // Double-buffered: stash the committed text and apply on `Done`.
                state.text_input_pending_commit = text;
            }
            zwp_text_input_v3::Event::DeleteSurroundingText {
                before_length: _,
                after_length: _,
            } => {}
            zwp_text_input_v3::Event::Done { serial: _ } => {
                // Apply the IME state accumulated since the previous `Done`, in the
                // protocol-mandated order: commit string first, then preedit. Per
                // spec the pending state resets each cycle, so a `Done` carrying no
                // preedit means the composition preview is cleared.
                if let Some(commit) =
                    state.text_input_pending_commit.take().filter(|t| {
                        !t.is_empty() && !t.chars().all(char::is_control)
                    })
                {
                    // `replace_last = false` commits: replaces any active
                    // composition preview with the text, then clears composition.
                    state.do_callback(XlibEvent::TextInput(TextInputEvent {
                        input: commit,
                        replace_last: false,
                        was_paste: false,
                        ..Default::default()
                    }));
                    // The widget's composition is now cleared by the commit above.
                    state.text_input_last_preedit.clear();
                }
                let preedit = state.text_input_pending_preedit.take().unwrap_or_default();
                if preedit != state.text_input_last_preedit {
                    // `replace_last = true` updates the inline composition preview;
                    // an empty string clears it.
                    state.do_callback(XlibEvent::TextInput(TextInputEvent {
                        input: preedit.clone(),
                        replace_last: true,
                        was_paste: false,
                        ..Default::default()
                    }));
                    state.text_input_last_preedit = preedit;
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1, ()> for WaylandState {
    fn event(
        _state: &mut Self,
        _proxy: &zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1,
        _event: <zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1 as Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        // We only set primary selection, not read it.
    }

    fn event_created_child(
        opcode: u16,
        qhandle: &QueueHandle<Self>,
    ) -> Arc<dyn wayland_client::backend::ObjectData> {
        match opcode {
            zwp_primary_selection_device_v1::EVT_DATA_OFFER_OPCODE => {
                qhandle
                    .make_data::<zwp_primary_selection_offer_v1::ZwpPrimarySelectionOfferV1, ()>(())
            }
            _ => unreachable!(
                "zwp_primary_selection_device_v1 created unknown child for opcode {}",
                opcode
            ),
        }
    }
}

impl Dispatch<zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1, ()> for WaylandState {
    fn event(
        state: &mut Self,
        _proxy: &zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1,
        event: <zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1 as Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            zwp_primary_selection_source_v1::Event::Send { mime_type: _, fd } => {
                use std::io::Write;
                let mut file = std::fs::File::from(fd);
                let _ = file.write_all(state.primary_selection_text.as_bytes());
            }
            zwp_primary_selection_source_v1::Event::Cancelled => {
                state.primary_selection_source = None;
            }
            _ => {}
        }
    }
}

impl Dispatch<zwp_text_input_manager_v3::ZwpTextInputManagerV3, ()> for WaylandState {
    fn event(
        state: &mut Self,
        proxy: &zwp_text_input_manager_v3::ZwpTextInputManagerV3,
        event: <zwp_text_input_manager_v3::ZwpTextInputManagerV3 as Proxy>::Event,
        data: &(),
        conn: &Connection,
        qhandle: &QueueHandle<Self>,
    ) {
        if let Some(seat) = state.seat.as_ref() {
            state.text_input = Some(proxy.get_text_input(seat, qhandle, ()));
        }
    }
}

impl Dispatch<wl_keyboard::WlKeyboard, ()> for WaylandState {
    fn event(
        state: &mut Self,
        keyboard: &wl_keyboard::WlKeyboard,
        event: wl_keyboard::Event,
        _: &(),
        conn: &Connection,
        qhandle: &QueueHandle<Self>,
    ) {
        match event {
            wl_keyboard::Event::Enter {
                serial,
                surface,
                keys: _,
            } => {
                state.keyboard_serial = Some(serial);
                state.flush_pending_clipboard_copy(qhandle, serial);
                if let Some(window_id) = state.window_id_for_surface(&surface) {
                    if state.keyboard_window != Some(window_id) {
                        if let Some(prev) = state.keyboard_window {
                            state.do_callback(XlibEvent::WindowLostFocus(prev));
                        }
                        state.keyboard_window = Some(window_id);
                        state.do_callback(XlibEvent::WindowGotFocus(window_id));
                    }
                }
            }
            wl_keyboard::Event::Leave { serial, surface } => {
                // Cancel any active key repeat when keyboard focus is lost
                state.timers.stop_timer(KEY_REPEAT_TIMER_ID);
                state.key_repeat = None;

                state.keyboard_serial = Some(serial);
                state.flush_pending_clipboard_copy(qhandle, serial);
                if let Some(window_id) = state.window_id_for_surface(&surface) {
                    if state.keyboard_window == Some(window_id) {
                        state.keyboard_window = None;
                        state.do_callback(XlibEvent::WindowLostFocus(window_id));
                    }
                }
                {
                    let popup_ids: Vec<_> =
                        state.popups.iter().rev().map(|p| p.window_id).collect();
                    for window_id in popup_ids {
                        state.do_callback(XlibEvent::PopupDismissed(PopupDismissedEvent {
                            window_id,
                            reason: PopupDismissReason::FocusLost,
                        }));
                    }
                }
            }
            wl_keyboard::Event::Key {
                serial,
                time: _,
                key,
                state: key_state,
            } => {
                if let WEnum::Value(key_state) = key_state {
                    match key_state {
                        wl_keyboard::KeyState::Pressed => {
                            state.keyboard_serial = Some(serial);
                            state.flush_pending_clipboard_copy(qhandle, serial);
                            let (key_code, text_str, should_repeat) =
                                if let Some(xkb_state) = state.xkb_state.as_mut() {
                                    (
                                        xkb_state.keycode_to_makepad_keycode(key + 8),
                                        xkb_state.key_get_utf8(key + 8),
                                        xkb_state.key_repeats(key + 8),
                                    )
                                } else {
                                    return;
                                };

                            let primary_mod = state.modifiers.control || state.modifiers.logo;
                            if primary_mod {
                                match key_code {
                                    KeyCode::KeyV => state.request_clipboard_paste(conn),
                                    KeyCode::KeyC => {
                                        let response = Rc::new(RefCell::new(None));
                                        state.do_callback(XlibEvent::TextCopy(
                                            TextClipboardEvent {
                                                response: response.clone(),
                                            },
                                        ));
                                        let content = response.borrow().clone();
                                        if let Some(content) = content {
                                            state.set_clipboard_text(qhandle, serial, content);
                                        }
                                    }
                                    KeyCode::KeyX => {
                                        let response = Rc::new(RefCell::new(None));
                                        state.do_callback(XlibEvent::TextCut(TextClipboardEvent {
                                            response: response.clone(),
                                        }));
                                        let content = response.borrow().clone();
                                        if let Some(content) = content {
                                            state.set_clipboard_text(qhandle, serial, content);
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            let block_text = primary_mod || state.modifiers.alt;
                            state.do_callback(XlibEvent::KeyDown(KeyEvent {
                                key_code,
                                is_repeat: false,
                                modifiers: state.modifiers,
                                time: state.time_now(),
                            }));

                            if !block_text && text_str.chars().any(|ch| !ch.is_control()) {
                                state.do_callback(XlibEvent::TextInput(TextInputEvent {
                                    input: text_str.clone(),
                                    replace_last: false,
                                    was_paste: false,
                                    ..Default::default()
                                }));
                            }

                            // Start key repeat timer if the key supports it
                            if should_repeat && state.key_repeat_rate > 0 {
                                state.timers.stop_timer(KEY_REPEAT_TIMER_ID);
                                state.key_repeat = Some(KeyRepeatState {
                                    key_code,
                                    text: text_str,
                                    in_initial_delay: true,
                                });
                                let delay_secs = state.key_repeat_delay as f64 / 1000.0;
                                state
                                    .timers
                                    .start_timer(KEY_REPEAT_TIMER_ID, delay_secs, false);
                            }
                        }
                        wl_keyboard::KeyState::Released => {
                            if let Some(xkb_state) = state.xkb_state.as_mut() {
                                let key_code = xkb_state.keycode_to_makepad_keycode(key + 8);

                                // Stop key repeat if this is the key being repeated
                                if state
                                    .key_repeat
                                    .as_ref()
                                    .is_some_and(|r| r.key_code == key_code)
                                {
                                    state.timers.stop_timer(KEY_REPEAT_TIMER_ID);
                                    state.key_repeat = None;
                                }

                                state.do_callback(XlibEvent::KeyUp(KeyEvent {
                                    key_code,
                                    is_repeat: false,
                                    modifiers: state.modifiers,
                                    time: state.time_now(),
                                }));
                            }
                        }
                        _ => {}
                    };
                }
            }
            wl_keyboard::Event::RepeatInfo { rate, delay } => {
                state.key_repeat_rate = rate;
                state.key_repeat_delay = delay;
            }
            wl_keyboard::Event::Modifiers {
                serial: _,
                mods_depressed,
                mods_latched,
                mods_locked,
                group,
            } => {
                if let Some(xkb_state) = state.xkb_state.as_mut() {
                    xkb_state.update_mask(mods_depressed, mods_latched, mods_locked, 0, 0, group);
                    state.modifiers = xkb_state.get_key_modifiers();
                }
            }
            wl_keyboard::Event::Keymap { format, fd, size } => match format {
                WEnum::Value(wl_keyboard::KeymapFormat::XkbV1) => {
                    // wl_seat v7+ requires keymap fds to be mapped MAP_PRIVATE; it is
                    // also valid on older versions since the map is read-only.
                    let map_str = unsafe {
                        libc_sys::mmap(
                            std::ptr::null_mut(),
                            size as libc_sys::size_t,
                            libc_sys::PROT_READ,
                            libc_sys::MAP_PRIVATE,
                            fd.as_raw_fd(),
                            0,
                        )
                    };
                    let keymap = xkb_sys::XkbKeymap::from_cstr(&state.xkb_cx, map_str).unwrap();
                    unsafe {
                        munmap(map_str, size as libc_sys::size_t);
                    }
                    state.xkb_state = xkb_sys::XkbState::new(&keymap);
                }
                _ => {}
            },
            _ => {}
        }
    }
}
impl Dispatch<wl_pointer::WlPointer, ()> for WaylandState {
    fn event(
        state: &mut Self,
        pointer: &wl_pointer::WlPointer,
        event: wl_pointer::Event,
        _: &(),
        conn: &Connection,
        qhandle: &QueueHandle<Self>,
    ) {
        match event {
            wl_pointer::Event::Enter {
                serial,
                surface,
                surface_x: _,
                surface_y: _,
            } => {
                state.pointer_serial = Some(serial);
                state.flush_pending_clipboard_copy(qhandle, serial);
                state.pointer_window = state.window_id_for_surface(&surface);
            }
            wl_pointer::Event::Leave { serial, surface: _ } => {
                // Dispatch any buffered motion before the pointer leaves, so the final hover
                // position is delivered to the right window first.
                state.flush_pending_motion();
                state.pointer_serial = Some(serial);
                state.flush_pending_clipboard_copy(qhandle, serial);
                state.pointer_window = None;
                state.scroll_gesture_active = false;
                state.last_resize_edge = None;
            }
            wl_pointer::Event::Motion {
                time,
                surface_x,
                surface_y,
            } => {
                if let Some(window_id) = state.pointer_window {
                    let pos = dvec2(surface_x as f64, surface_y as f64);
                    state.last_mouse_pos = pos;

                    // Edge-resize detection (matches X11 backend thresholds)
                    let window_size = state
                        .windows
                        .iter()
                        .find(|w| w.window_id == window_id)
                        .map(|w| w.window_geom.inner_size);
                    if let Some(ws) = window_size {
                        let edge = if pos.x < 10.0 && pos.y < 10.0 {
                            Some((
                                xdg_toplevel::ResizeEdge::TopLeft,
                                wp_cursor_shape_device_v1::Shape::NwResize,
                            ))
                        } else if pos.x < 10.0 && pos.y >= ws.y - 10.0 {
                            Some((
                                xdg_toplevel::ResizeEdge::BottomLeft,
                                wp_cursor_shape_device_v1::Shape::SwResize,
                            ))
                        } else if pos.x < 5.0 {
                            Some((
                                xdg_toplevel::ResizeEdge::Left,
                                wp_cursor_shape_device_v1::Shape::WResize,
                            ))
                        } else if pos.x >= ws.x - 10.0 && pos.y < 10.0 {
                            Some((
                                xdg_toplevel::ResizeEdge::TopRight,
                                wp_cursor_shape_device_v1::Shape::NeResize,
                            ))
                        } else if pos.x >= ws.x - 10.0 && pos.y >= ws.y - 10.0 {
                            Some((
                                xdg_toplevel::ResizeEdge::BottomRight,
                                wp_cursor_shape_device_v1::Shape::SeResize,
                            ))
                        } else if pos.x >= ws.x - 5.0 {
                            Some((
                                xdg_toplevel::ResizeEdge::Right,
                                wp_cursor_shape_device_v1::Shape::EResize,
                            ))
                        } else if pos.y < 5.0 {
                            Some((
                                xdg_toplevel::ResizeEdge::Top,
                                wp_cursor_shape_device_v1::Shape::NResize,
                            ))
                        } else if pos.y >= ws.y - 5.0 {
                            Some((
                                xdg_toplevel::ResizeEdge::Bottom,
                                wp_cursor_shape_device_v1::Shape::SResize,
                            ))
                        } else {
                            None
                        };
                        if let Some((resize_edge, cursor_shape)) = edge {
                            state.last_resize_edge = Some(resize_edge);
                            if let (Some(cursor_dev), Some(serial)) =
                                (state.cursor_shape.as_ref(), state.pointer_serial)
                            {
                                cursor_dev.set_shape(serial, cursor_shape);
                            }
                        } else {
                            if state.last_resize_edge.is_some() {
                                if let (Some(cursor_dev), Some(serial)) =
                                    (state.cursor_shape.as_ref(), state.pointer_serial)
                                {
                                    cursor_dev.set_shape(
                                        serial,
                                        wp_cursor_shape_device_v1::Shape::Default,
                                    );
                                }
                            }
                            state.last_resize_edge = None;
                        }
                    }

                    // Buffer this motion instead of dispatching immediately; the latest one is
                    // flushed as a single MouseMove once the whole event batch is drained (or before
                    // an intervening button/leave). The edge-resize cursor above still updates per
                    // motion so the resize cursor stays responsive. See `flush_pending_motion`.
                    state.pending_motion = Some((window_id, pos));
                }
            }
            wl_pointer::Event::Button {
                serial,
                time,
                button,
                state: key_state,
            } => {
                // Dispatch any buffered motion first so a MouseMove precedes this button's
                // down/up (and the WindowDragQuery it triggers) at the correct hover position.
                state.flush_pending_motion();
                state.pointer_serial = Some(serial);
                state.flush_pending_clipboard_copy(qhandle, serial);
                // Outside-click popup dismissal: if press lands on a
                // regular window while popups are open, fire dismiss.
                if let WEnum::Value(ButtonState::Pressed) = key_state {
                    if let Some(win_id) = state.pointer_window {
                        if state.windows.iter().any(|w| w.window_id == win_id)
                            && !state.popups.is_empty()
                        {
                            let popup_ids: Vec<_> =
                                state.popups.iter().rev().map(|p| p.window_id).collect();
                            for popup_wid in popup_ids {
                                state.do_callback(XlibEvent::PopupDismissed(PopupDismissedEvent {
                                    window_id: popup_wid,
                                    reason: PopupDismissReason::OutsideClick,
                                }));
                            }
                        }
                    }
                }
                if let Some(btn) = wayland_type::from_mouse(button) {
                    if let Some(window_id) = state.pointer_window {
                        match key_state {
                            WEnum::Value(ButtonState::Pressed) => {
                                if btn == MouseButton::PRIMARY {
                                    if state.windows.iter().any(|win| win.window_id == window_id) {
                                        // Edge resize takes priority
                                        if let Some(resize_edge) = state.last_resize_edge.take() {
                                            if let (Some(seat), Some(window)) = (
                                                state.seat.as_ref(),
                                                state
                                                    .windows
                                                    .iter()
                                                    .find(|win| win.window_id == window_id),
                                            ) {
                                                window.toplevel.resize(seat, serial, resize_edge);
                                                return;
                                            }
                                        }

                                        let response =
                                            Rc::new(Cell::new(WindowDragQueryResponse::NoAnswer));
                                        state.do_callback(XlibEvent::WindowDragQuery(
                                            WindowDragQueryEvent {
                                                window_id,
                                                abs: state.last_mouse_pos,
                                                response: response.clone(),
                                            },
                                        ));
                                        if matches!(
                                            response.get(),
                                            WindowDragQueryResponse::Caption
                                        ) {
                                            if let (Some(seat), Some(window)) = (
                                                state.seat.as_ref(),
                                                state
                                                    .windows
                                                    .iter()
                                                    .find(|win| win.window_id == window_id),
                                            ) {
                                                window.toplevel._move(seat, serial);
                                                return;
                                            }
                                        }
                                    }
                                }
                                state.do_callback(XlibEvent::MouseDown(MouseDownEvent {
                                    abs: state.last_mouse_pos,
                                    button: btn,
                                    window_id: window_id,
                                    modifiers: state.modifiers,
                                    handled: Cell::new(Area::Empty),
                                    time: state.time_now(),
                                }))
                            }
                            WEnum::Value(ButtonState::Released) => {
                                state.do_callback(XlibEvent::MouseUp(MouseUpEvent {
                                    abs: state.last_mouse_pos,
                                    button: btn,
                                    window_id,
                                    modifiers: state.modifiers,
                                    time: state.time_now(),
                                }));
                            }
                            WEnum::Unknown(_) | WEnum::Value(_) => {}
                        }
                    }
                }
            }
            // Wayland axis values already match Makepad's convention: positive vertical =
            // scroll down = viewport moves DOWN. The spec pins the sign in
            // wl_pointer::axis_relative_direction, whose `identical` case is fingers moving
            // down producing a "vertical_scroll down" axis event; libinput documents the
            // same ("the positive direction being down or right"). So pass the values
            // through untouched — the compositor has already applied the user's
            // natural-scrolling preference to the sign, and negating here would invert both
            // settings. Toolkits that do negate (winit, SDL, Chromium) only do so because
            // their own convention is inverted; GTK, which shares Makepad's, does not.
            wl_pointer::Event::Axis {
                time: _,
                axis,
                value,
            } => match axis {
                WEnum::Value(wl_pointer::Axis::VerticalScroll) => {
                    state.scroll_accumulator.y += value;
                }
                WEnum::Value(wl_pointer::Axis::HorizontalScroll) => {
                    state.scroll_accumulator.x += value;
                }
                _ => {}
            },
            wl_pointer::Event::AxisSource { axis_source } => {
                // A source this protocol copy predates (`AxisSource` is `#[non_exhaustive]`)
                // is as good as no source: record `None` rather than letting it fall through
                // to the finger branch, which is the one classification that can strand a
                // stretched rubber band.
                state.scroll_source = match axis_source {
                    WEnum::Value(source) => Some(source),
                    WEnum::Unknown(_) => None,
                };
            }
            wl_pointer::Event::Frame => {
                let frame = frame_scroll(
                    state.scroll_source,
                    state.scroll_gesture_active,
                    state.scroll_stopped,
                    state.scroll_accumulator,
                    state.scroll_detents,
                );
                if let Some(frame) = &frame {
                    // Tracked whether or not a window is under the pointer, so a gesture that
                    // starts over one window and lifts over another still terminates.
                    state.scroll_gesture_active = frame.phase == ScrollPhase::Changed;
                }
                if let (Some(frame), Some(window_id)) = (frame, state.pointer_window) {
                    // Deliver any buffered motion first so the Scroll event's hover
                    // position is current (Button and Leave already do this).
                    state.flush_pending_motion();
                    let time_now = state.time_now();
                    state.do_callback(XlibEvent::Scroll(ScrollEvent {
                        window_id,
                        scroll: frame.delta,
                        abs: state.last_mouse_pos,
                        modifiers: state.modifiers,
                        is_mouse: frame.is_mouse,
                        handled_x: Cell::new(false),
                        handled_y: Cell::new(false),
                        time: time_now,
                        phase: frame.phase,
                    }));
                }
                state.scroll_accumulator = dvec2(0.0, 0.0);
                state.scroll_detents = dvec2(0.0, 0.0);
                state.scroll_source = None;
                state.scroll_stopped = false;
            }
            wl_pointer::Event::AxisStop { time: _, axis } => {
                // An axis stopped. One flag for the whole frame rather than one per axis:
                // `ScrollEvent` carries a single phase for both axes, so a per-axis mask
                // could not be expressed anyway. `frame_scroll` separates the two cases the
                // protocol defines — "this axis stopped, the other continues" from a real
                // lift-off — by whether the frame also carries motion.
                if matches!(
                    axis,
                    WEnum::Value(wl_pointer::Axis::VerticalScroll)
                        | WEnum::Value(wl_pointer::Axis::HorizontalScroll)
                ) {
                    state.scroll_stopped = true;
                }
            }
            // Wheel detent counts, carrying the same sign convention as the Axis event
            // above: the spec states each expresses its direction in terms of the positive
            // or negative direction of the same axis, never inverted relative to it.
            // AxisDiscrete is only sent by compositors below seat v8; v8+ compositors
            // send AxisValue120 instead (120 units per detent, fractional detents
            // allowed for high-resolution wheels), so the two never double-count.
            wl_pointer::Event::AxisDiscrete { axis, discrete } => match axis {
                WEnum::Value(wl_pointer::Axis::VerticalScroll) => {
                    state.scroll_detents.y += discrete as f64;
                }
                WEnum::Value(wl_pointer::Axis::HorizontalScroll) => {
                    state.scroll_detents.x += discrete as f64;
                }
                _ => {}
            },
            wl_pointer::Event::AxisValue120 { axis, value120 } => match axis {
                WEnum::Value(wl_pointer::Axis::VerticalScroll) => {
                    state.scroll_detents.y += value120 as f64 / 120.0;
                }
                WEnum::Value(wl_pointer::Axis::HorizontalScroll) => {
                    state.scroll_detents.x += value120 as f64 / 120.0;
                }
                _ => {}
            },
            // Purely informational: the physical direction of the entity that caused the
            // axis event. The axis value itself already reflects the user's natural-scrolling
            // setting, so scrolling content must ignore this. It exists for widgets that
            // should follow the physical wheel regardless of that setting — the spec's
            // example is a volume slider — which Makepad has no plumbing for, so drop it.
            wl_pointer::Event::AxisRelativeDirection {
                axis: _,
                direction: _,
            } => {}
            _ => {}
        }
    }
}

impl Dispatch<wl_callback::WlCallback, WindowId> for WaylandState {
    fn event(
        state: &mut Self,
        _callback: &wl_callback::WlCallback,
        event: wl_callback::Event,
        window_id: &WindowId,
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        // The compositor is ready for a new frame on this window's surface. Clear the
        // pending flag; the Paint that follows event dispatch in the event loop presents
        // the window's pass if it is still dirty. The window may have been closed while
        // the callback was in flight, in which case there is nothing left to clear.
        if let wl_callback::Event::Done { .. } = event {
            state.clear_frame_callback_pending(*window_id);
        }
    }
}

impl Dispatch<wp_cursor_shape_manager_v1::WpCursorShapeManagerV1, ()> for WaylandState {
    fn event(
        state: &mut Self,
        cursor_shape_manager: &wp_cursor_shape_manager_v1::WpCursorShapeManagerV1,
        event: wp_cursor_shape_manager_v1::Event,
        _: &(),
        conn: &Connection,
        qhandle: &QueueHandle<Self>,
    ) {
        if let Some(pointer) = state.pointer.as_ref() {
            state.cursor_shape = Some(cursor_shape_manager.get_pointer(pointer, qhandle, ()));
        }
    }
}

delegate_noop!(WaylandState: ignore wp_viewport::WpViewport);
delegate_noop!(WaylandState: ignore wp_viewporter::WpViewporter);
delegate_noop!(WaylandState: ignore wl_surface::WlSurface);
delegate_noop!(WaylandState: ignore wp_cursor_shape_device_v1::WpCursorShapeDeviceV1);
delegate_noop!(WaylandState: ignore wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1);
delegate_noop!(WaylandState: ignore wl_compositor::WlCompositor);
delegate_noop!(WaylandState: ignore zxdg_decoration_manager_v1::ZxdgDecorationManagerV1);
delegate_noop!(WaylandState: ignore zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1);
delegate_noop!(WaylandState: ignore xdg_toplevel_icon_v1::XdgToplevelIconV1);
delegate_noop!(WaylandState: ignore wl_shm::WlShm);
delegate_noop!(WaylandState: ignore wl_shm_pool::WlShmPool);
delegate_noop!(WaylandState: ignore wl_buffer::WlBuffer);
delegate_noop!(WaylandState: ignore xdg_positioner::XdgPositioner);
delegate_noop!(WaylandState: ignore zwp_primary_selection_device_manager_v1::ZwpPrimarySelectionDeviceManagerV1);
delegate_noop!(WaylandState: ignore zwp_primary_selection_offer_v1::ZwpPrimarySelectionOfferV1);

impl Dispatch<xdg_toplevel_icon_manager_v1::XdgToplevelIconManagerV1, ()> for WaylandState {
    fn event(
        _state: &mut Self,
        _proxy: &xdg_toplevel_icon_manager_v1::XdgToplevelIconManagerV1,
        _event: xdg_toplevel_icon_manager_v1::Event,
        _: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        // icon_size events are informational; we ignore them for now
    }
}

impl WaylandState {
    fn ensure_data_device(&mut self, qhandle: &QueueHandle<Self>) {
        if self.data_device.is_none() {
            if let (Some(data_device_manager), Some(seat)) =
                (self.data_device_manager.as_ref(), self.seat.as_ref())
            {
                self.data_device = Some(data_device_manager.get_data_device(seat, qhandle, ()));
            }
        }
    }

    fn ensure_primary_selection_device(&mut self, qhandle: &QueueHandle<Self>) {
        if self.primary_selection_device.is_none() {
            if let (Some(manager), Some(seat)) =
                (self.primary_selection_manager.as_ref(), self.seat.as_ref())
            {
                self.primary_selection_device = Some(manager.get_device(seat, qhandle, ()));
            }
        }
    }

    pub(crate) fn set_primary_selection_text(
        &mut self,
        qhandle: &QueueHandle<Self>,
        serial: u32,
        text: String,
    ) {
        self.primary_selection_text = text;
        if let Some(device) = self.primary_selection_device.as_ref() {
            if let Some(manager) = self.primary_selection_manager.as_ref() {
                let source = manager.create_source(qhandle, ());
                source.offer("text/plain;charset=utf-8".to_string());
                source.offer("text/plain".to_string());
                source.offer("UTF8_STRING".to_string());
                source.offer("STRING".to_string());
                source.offer("TEXT".to_string());
                device.set_selection(Some(&source), serial);
                self.primary_selection_source = Some(source);
            }
        }
    }

    fn is_text_mime_type(mime_type: &str) -> bool {
        matches!(
            mime_type,
            "text/plain;charset=utf-8" | "text/plain" | "UTF8_STRING" | "STRING" | "TEXT"
        )
    }

    fn preferred_clipboard_mime_type(offer: &ClipboardOffer) -> Option<&str> {
        for preferred in [
            "text/plain;charset=utf-8",
            "text/plain",
            "UTF8_STRING",
            "STRING",
            "TEXT",
        ] {
            if let Some(mime_type) = offer.mime_types.iter().find(|m| m.as_str() == preferred) {
                return Some(mime_type.as_str());
            }
        }
        offer.mime_types.first().map(String::as_str)
    }

    pub(crate) fn set_clipboard_text(
        &mut self,
        qhandle: &QueueHandle<Self>,
        serial: u32,
        text: String,
    ) {
        self.ensure_data_device(qhandle);
        if let (Some(data_device_manager), Some(data_device)) =
            (self.data_device_manager.as_ref(), self.data_device.as_ref())
        {
            let source = data_device_manager.create_data_source(qhandle, ());
            source.offer("text/plain;charset=utf-8".to_string());
            source.offer("text/plain".to_string());
            source.offer("UTF8_STRING".to_string());
            source.offer("STRING".to_string());
            source.offer("TEXT".to_string());
            data_device.set_selection(Some(&source), serial);
            self.clipboard_source = Some(source);
            self.clipboard_text = text;
        }
    }

    /// Flush a pending clipboard copy now that a serial is available.
    pub(crate) fn flush_pending_clipboard_copy(
        &mut self,
        qhandle: &QueueHandle<Self>,
        serial: u32,
    ) {
        if let Some(text) = self.pending_clipboard_copy.take() {
            self.set_clipboard_text(qhandle, serial, text);
        }
    }

    fn dispatch_paste_bytes(&mut self, mut bytes: Vec<u8>) {
        while bytes.last() == Some(&0) {
            bytes.pop();
        }
        let input = String::from_utf8_lossy(&bytes).into_owned();
        if !input.is_empty() {
            self.pending_paste_text_input = Some(input);
        }
    }

    pub(crate) fn take_pending_paste_text_input(&mut self) -> Option<String> {
        self.pending_paste_text_input.take()
    }

    pub(crate) fn pump_pending_clipboard_read(&mut self) {
        let mut pending = match self.pending_clipboard_read.take() {
            Some(pending) => pending,
            None => return,
        };

        let read_raw_fd = pending.fd.as_raw_fd();
        let mut readfds = unsafe { std::mem::zeroed::<libc_sys::fd_set>() };
        let mut timeout = libc_sys::timeval {
            tv_sec: 0,
            tv_usec: 0,
        };
        unsafe {
            libc_sys::FD_ZERO(&mut readfds);
            libc_sys::FD_SET(read_raw_fd, &mut readfds);
        }
        let ready = unsafe {
            libc_sys::select(
                read_raw_fd + 1,
                &mut readfds,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut timeout,
            )
        };
        if ready <= 0 {
            self.pending_clipboard_read = Some(pending);
            return;
        }

        loop {
            let mut chunk = [0u8; 4096];
            let count = unsafe {
                libc_sys::read(
                    read_raw_fd,
                    chunk.as_mut_ptr() as *mut std::os::raw::c_void,
                    chunk.len(),
                )
            };
            if count > 0 {
                pending.bytes.extend_from_slice(&chunk[..count as usize]);
                continue;
            }

            if pending.bytes.is_empty() {
                self.pending_clipboard_read = Some(pending);
            } else {
                self.dispatch_paste_bytes(pending.bytes);
            }
            return;
        }
    }

    fn request_clipboard_paste(&mut self, conn: &Connection) {
        if let Some(offer) = self.clipboard_offer.as_ref() {
            if let Some(mime_type) = Self::preferred_clipboard_mime_type(offer) {
                let mut pipe_fds = [0; 2];
                if unsafe { libc_sys::pipe(pipe_fds.as_mut_ptr()) } != 0 {
                    return;
                }
                let read_fd = unsafe { std::os::fd::OwnedFd::from_raw_fd(pipe_fds[0]) };
                let read_raw_fd = read_fd.as_raw_fd();
                let write_fd = unsafe { std::os::fd::OwnedFd::from_raw_fd(pipe_fds[1]) };
                offer.offer.receive(mime_type.to_string(), write_fd.as_fd());
                drop(write_fd);
                let _ = conn.flush();

                unsafe {
                    let flags = libc_sys::fcntl(read_raw_fd, libc_sys::F_GETFL, 0);
                    if flags >= 0 {
                        let _ = libc_sys::fcntl(
                            read_raw_fd,
                            libc_sys::F_SETFL,
                            flags | libc_sys::O_NONBLOCK,
                        );
                    }
                }
                self.pending_clipboard_read = Some(PendingClipboardRead {
                    fd: read_fd,
                    bytes: Vec::new(),
                });
                self.pump_pending_clipboard_read();
            }
        } else if !self.clipboard_text.is_empty() {
            self.do_callback(XlibEvent::TextInput(TextInputEvent {
                input: self.clipboard_text.clone(),
                replace_last: false,
                was_paste: true,
                ..Default::default()
            }));
        }
    }

    pub(crate) fn available(&self) -> bool {
        self.compositor.is_some() && self.wm_base.is_some()
    }

    fn xdg_toplevel_has_state(states: &[u8], needle: u32) -> bool {
        states
            .chunks_exact(4)
            .any(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) == needle)
    }

    fn do_callback(&mut self, event: XlibEvent) {
        if let Some(mut callback) = self.event_callback.take() {
            callback(self, event);
            self.event_callback = Some(callback);
        }
    }

    /// Dispatch the latest coalesced pointer motion (if any) as a single `MouseMove`, then clear it.
    /// Called once after the `wl_pointer` event
    /// batch is drained and before any intervening button/leave, so a high-Hz mouse produces one
    /// hover hit-test per frame instead of one per queued motion. See [`Self::pending_motion`].
    pub(crate) fn flush_pending_motion(&mut self) {
        let Some((window_id, pos)) = self.pending_motion.take() else {
            return;
        };
        // The window may have been closed by an event earlier in this batch;
        // dispatching a motion for a dead window would hit a stale or recycled
        // window pool slot downstream.
        if !self.windows.iter().any(|w| w.window_id == window_id)
            && !self.popups.iter().any(|w| w.window_id == window_id)
        {
            return;
        }
        self.do_callback(XlibEvent::MouseMove(MouseMoveEvent {
                lock_delta: Default::default(),
            abs: pos,
            window_id,
            modifiers: self.modifiers,
            time: self.time_now(),
            handled: Cell::new(Area::Empty),
        }));
    }

    /// True while the given window's last presented frame awaits its `wl_surface::frame`
    /// callback, meaning the compositor is not ready for another frame on that surface.
    pub(crate) fn is_frame_callback_pending(&self, window_id: WindowId) -> bool {
        self.frame_callbacks_pending.contains(&window_id)
    }

    pub(crate) fn set_frame_callback_pending(&mut self, window_id: WindowId) {
        if !self.frame_callbacks_pending.contains(&window_id) {
            self.frame_callbacks_pending.push(window_id);
        }
    }

    /// Clear a window's pending frame callback. Called when the callback fires and when
    /// a window is closed, since the compositor never fires callbacks for a destroyed
    /// surface and a stale entry would keep the window's presents gated forever.
    pub(crate) fn clear_frame_callback_pending(&mut self, window_id: WindowId) {
        self.frame_callbacks_pending.retain(|id| *id != window_id);
    }

    pub(crate) fn any_frame_callback_pending(&self) -> bool {
        !self.frame_callbacks_pending.is_empty()
    }

    /// Called from the event loop when the key repeat timer fires.
    /// Returns true if the timer was handled (i.e., it was the key repeat timer).
    pub(crate) fn handle_key_repeat_timer(&mut self, timer_id: u64) -> bool {
        if timer_id != KEY_REPEAT_TIMER_ID {
            return false;
        }
        if let Some(repeat) = self.key_repeat.as_mut() {
            let key_code = repeat.key_code;
            let text = repeat.text.clone();
            let modifiers = self.modifiers;

            if repeat.in_initial_delay {
                // Initial delay has elapsed; switch to steady-state repeat interval.
                repeat.in_initial_delay = false;
                let interval_secs = 1.0 / self.key_repeat_rate as f64;
                self.timers
                    .start_timer(KEY_REPEAT_TIMER_ID, interval_secs, true);
            }

            self.do_callback(XlibEvent::KeyDown(KeyEvent {
                key_code,
                is_repeat: true,
                modifiers,
                time: self.time_now(),
            }));

            let block_text = modifiers.control || modifiers.logo || modifiers.alt;
            if !block_text && text.chars().any(|ch| !ch.is_control()) {
                self.do_callback(XlibEvent::TextInput(TextInputEvent {
                    input: text,
                    replace_last: false,
                    was_paste: false,
                    ..Default::default()
                }));
            }
        }
        true
    }

    pub fn start_timer(&mut self, id: u64, timeout: f64, repeats: bool) {
        self.timers.start_timer(id, timeout, repeats);
    }

    pub fn stop_timer(&mut self, id: u64) {
        self.timers.stop_timer(id);
    }
    pub fn time_now(&self) -> f64 {
        self.timers.time_now()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wheel_like_sources_take_the_detent_path() {
        // Wheels and wheel tilts ratchet, whether or not this frame carried detents.
        for source in [
            wl_pointer::AxisSource::Wheel,
            wl_pointer::AxisSource::WheelTilt,
        ] {
            assert!(scroll_is_wheel_like(Some(source), true));
            assert!(scroll_is_wheel_like(Some(source), false));
        }
        // A touchpad gesture and a trackpoint / button-held scroll are both smooth.
        for source in [
            wl_pointer::AxisSource::Finger,
            wl_pointer::AxisSource::Continuous,
        ] {
            assert!(!scroll_is_wheel_like(Some(source), false));
            assert!(!scroll_is_wheel_like(Some(source), true));
        }
    }

    #[test]
    fn a_frame_without_an_axis_source_is_classified_by_its_detents() {
        // `axis_source` is optional, and an unknown value is recorded as `None`. Detents
        // then decide, and the sourceless default must not be the finger path.
        assert!(scroll_is_wheel_like(None, true));
        assert!(!scroll_is_wheel_like(None, false));
    }

    /// A frame carrying no stop, from a source with no gesture in flight.
    fn plain_frame(
        source: Option<wl_pointer::AxisSource>,
        acc: Vec2d,
        detents: Vec2d,
    ) -> Option<FrameScroll> {
        frame_scroll(source, false, false, acc, detents)
    }

    #[test]
    fn each_axis_chooses_detents_or_raw_pixels_on_its_own() {
        // A wheel frame with a detented vertical axis and a smooth horizontal one: scaling
        // the horizontal by its zero detent count would drop it entirely.
        let frame = plain_frame(
            Some(wl_pointer::AxisSource::Wheel),
            dvec2(7.5, 15.0),
            dvec2(0.0, 1.0),
        )
        .expect("a frame with a delta dispatches");
        assert_eq!(frame.delta, dvec2(7.5, PIXELS_PER_WHEEL_DETENT));
        assert!(frame.is_mouse);
        assert_eq!(frame.phase, ScrollPhase::None);
    }

    #[test]
    fn a_wheel_frame_without_detents_keeps_its_raw_distance_unscaled() {
        let frame = plain_frame(
            Some(wl_pointer::AxisSource::Wheel),
            dvec2(0.0, 15.0),
            dvec2(0.0, 0.0),
        )
        .expect("a frame with a delta dispatches");
        assert_eq!(frame.delta, dvec2(0.0, 15.0));
    }

    #[test]
    fn a_sourceless_frame_with_detents_takes_the_wheel_path() {
        let frame = plain_frame(None, dvec2(0.0, 15.0), dvec2(0.0, 1.0))
            .expect("a frame with a delta dispatches");
        assert_eq!(frame.delta, dvec2(0.0, PIXELS_PER_WHEEL_DETENT));
        assert!(frame.is_mouse);
        assert_eq!(frame.phase, ScrollPhase::None);
    }

    #[test]
    fn a_touchpad_gesture_reports_changed_then_ended_at_lift_off() {
        let moving = plain_frame(
            Some(wl_pointer::AxisSource::Finger),
            dvec2(0.0, 12.0),
            dvec2(0.0, 0.0),
        )
        .expect("a frame with a delta dispatches");
        assert_eq!(moving.phase, ScrollPhase::Changed);
        assert_eq!(moving.delta, dvec2(0.0, 12.0));
        assert!(!moving.is_mouse);

        // Lift-off: the stops arrive alone, and the zero-delta event is what springs a
        // stretched rubber band back.
        let lifted = frame_scroll(
            Some(wl_pointer::AxisSource::Finger),
            true,
            true,
            dvec2(0.0, 0.0),
            dvec2(0.0, 0.0),
        )
        .expect("a bare stop ends the gesture");
        assert_eq!(lifted.phase, ScrollPhase::Ended);
        assert_eq!(lifted.delta, dvec2(0.0, 0.0));
    }

    #[test]
    fn a_stop_alongside_live_motion_is_one_axis_stopping_not_lift_off() {
        // The `frame` event defines axis + axis_stop in one frame as "movement in one axis
        // has stopped but continues in the other axis".
        let frame = frame_scroll(
            Some(wl_pointer::AxisSource::Finger),
            true,
            true,
            dvec2(0.0, 12.0),
            dvec2(0.0, 0.0),
        )
        .expect("a frame with a delta dispatches");
        assert_eq!(frame.phase, ScrollPhase::Changed);
    }

    #[test]
    fn a_gesture_in_flight_still_ends_when_the_compositor_drops_the_axis_source() {
        // `axis_source` is per-frame and optional, so the lift-off frame may carry none.
        // Losing the terminator would strand a stretched rubber band.
        let frame = frame_scroll(None, true, true, dvec2(0.0, 0.0), dvec2(0.0, 0.0))
            .expect("the in-flight gesture recognises its own lift-off");
        assert_eq!(frame.phase, ScrollPhase::Ended);

        // With no gesture in flight the same frame says nothing and must not dispatch.
        assert!(frame_scroll(None, false, true, dvec2(0.0, 0.0), dvec2(0.0, 0.0)).is_none());
    }

    #[test]
    fn a_bare_stop_from_a_source_with_no_gesture_dispatches_nothing() {
        // Only `Finger` is guaranteed an AxisStop. A zero-delta `ScrollPhase::None` from one
        // of the others would clear a widget's overscroll and cut short a running bounce.
        for source in [
            wl_pointer::AxisSource::Wheel,
            wl_pointer::AxisSource::WheelTilt,
            wl_pointer::AxisSource::Continuous,
        ] {
            assert!(
                frame_scroll(Some(source), true, true, dvec2(0.0, 0.0), dvec2(0.0, 0.0))
                    .is_none(),
                "{source:?} has no gesture to end"
            );
        }
    }

    #[test]
    fn a_trackpoint_scroll_is_a_plain_delta_that_skips_wheel_easing() {
        let frame = plain_frame(
            Some(wl_pointer::AxisSource::Continuous),
            dvec2(0.0, 9.0),
            dvec2(0.0, 0.0),
        )
        .expect("a frame with a delta dispatches");
        assert_eq!(frame.phase, ScrollPhase::None);
        assert_eq!(frame.delta, dvec2(0.0, 9.0));
        assert!(!frame.is_mouse);
    }

    #[test]
    fn an_empty_frame_dispatches_nothing() {
        assert!(plain_frame(None, dvec2(0.0, 0.0), dvec2(0.0, 0.0)).is_none());
    }

}
