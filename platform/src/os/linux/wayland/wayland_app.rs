#![allow(dead_code, unused_imports)]
use std::{
    option,
    os::fd::{AsFd, AsRawFd},
};

use wayland_client::{Connection, EventQueue};

use crate::{
    cx_native::EventFlow, wayland::wayland_state::WaylandState, x11::xlib_event::XlibEvent,
    SignalToUI, TimerEvent,
};

pub(crate) struct WaylandApp {
    connection: Connection,
    pub event_queue: EventQueue<WaylandState>,
    pub state: WaylandState,
    event_callback: Option<Box<dyn FnMut(&mut WaylandApp, XlibEvent) -> EventFlow>>,
}
impl WaylandApp {
    pub fn new(
        connection: Connection,
        event_queue: EventQueue<WaylandState>,
        state: WaylandState,
        event_callback: Box<dyn FnMut(&mut WaylandApp, XlibEvent) -> EventFlow>,
    ) -> Self {
        Self {
            connection,
            event_queue,
            state: state,
            event_callback: Some(event_callback),
        }
    }
    pub fn event_loop(&mut self) {
        self.do_callback(XlibEvent::Paint);
        let mut timer_ids = Vec::new();
        while self.state.event_loop_running {
            match self.state.event_flow {
                EventFlow::Exit => {
                    break;
                }
                EventFlow::Wait => {
                    let time = self.time_now();
                    self.state.timers.update_timers(&mut timer_ids);
                    let mut timer_queued_work = false;
                    for timer_id in &timer_ids {
                        if !self.state.handle_key_repeat_timer(*timer_id) {
                            self.do_callback(XlibEvent::Timer(TimerEvent {
                                timer_id: *timer_id,
                                time: Some(time),
                            }));
                            timer_queued_work |= matches!(self.state.event_flow, EventFlow::Poll);
                        }
                    }
                    if !self.state.event_loop_running {
                        break;
                    }
                    // A timer that redrew or dispatched a signal with work paints
                    // now. Timer 0 idles slowly, so sleeping first would hold that
                    // paint until an unrelated wake.
                    if timer_queued_work {
                        self.state.event_flow = EventFlow::Poll;
                        continue;
                    }
                    // Send any requests queued during event handling (cursor shapes,
                    // frame callback requests, etc.) before blocking, so the compositor
                    // can respond and wake the select below. A WouldBlock just means the
                    // socket buffer is full; the messages stay queued and the next loop
                    // iteration retries, so only a dead connection is fatal.
                    if let Err(err) = self.event_queue.flush() {
                        let transient = matches!(
                            &err,
                            wayland_client::backend::WaylandError::Io(io)
                                if io.kind() == std::io::ErrorKind::WouldBlock
                        );
                        if !transient {
                            crate::warning!("Wayland flush failed: {}", err);
                            self.terminate_event_loop();
                            return;
                        }
                    }
                    let woken = match self.event_queue.prepare_read() {
                        Some(guard) => self.state.timers.select(guard.connection_fd().as_raw_fd()),
                        None => false,
                    };
                    // Another thread raised a signal or queued a `--remote` command.
                    // Service it on this wake, as macOS does for its wake event,
                    // instead of on the next timer-0 tick.
                    if woken || SignalToUI::any_pending() {
                        self.do_callback(XlibEvent::Timer(TimerEvent {
                            timer_id: 0,
                            time: Some(self.time_now()),
                        }));
                    }
                    self.state.event_flow = EventFlow::Poll;
                }
                EventFlow::Poll => {
                    let time = self.time_now();
                    self.state.timers.update_timers(&mut timer_ids);
                    for timer_id in &timer_ids {
                        if !self.state.handle_key_repeat_timer(*timer_id) {
                            self.do_callback(XlibEvent::Timer(TimerEvent {
                                timer_id: *timer_id,
                                time: Some(time),
                            }));
                        }
                    }
                    self.event_loop_poll();
                }
            }
        }
    }
    fn event_loop_poll(&mut self) {
        if let Err(err) = self.event_queue.flush() {
            crate::warning!("Wayland flush failed: {}", err);
            self.terminate_event_loop();
            return;
        }
        if let Some(guard) = self.event_queue.prepare_read() {
            if let Err(err) = guard.read() {
                crate::warning!("Wayland read failed: {}", err);
                self.terminate_event_loop();
                return;
            }
            if let Err(err) = self.event_queue.dispatch_pending(&mut self.state) {
                crate::warning!("Wayland dispatch failed: {}", err);
                self.terminate_event_loop();
                return;
            }
        } else if let Err(err) = self.event_queue.dispatch_pending(&mut self.state) {
            crate::warning!("Wayland dispatch failed: {}", err);
            self.terminate_event_loop();
            return;
        }

        // The whole pointer-event batch is drained; dispatch the single latest coalesced motion
        // (one hover hit-test instead of one per queued motion) before painting.
        self.state.flush_pending_motion();

        self.do_callback(XlibEvent::Paint);
    }
    fn do_callback(&mut self, event: XlibEvent) {
        if let Some(mut callback) = self.event_callback.take() {
            self.state.event_flow = callback(self, event);
            if let EventFlow::Exit = self.state.event_flow {
                self.terminate_event_loop();
            }
            self.event_callback = Some(callback);
        }
    }
    pub fn terminate_event_loop(&mut self) {
        self.state.event_loop_running = false;
    }

    pub fn start_timer(&mut self, id: u64, timeout: f64, repeats: bool) {
        self.state.start_timer(id, timeout, repeats);
    }

    pub fn stop_timer(&mut self, id: u64) {
        self.state.stop_timer(id);
    }
    pub fn time_now(&self) -> f64 {
        self.state.time_now()
    }
}
