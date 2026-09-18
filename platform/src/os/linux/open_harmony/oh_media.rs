use {
    super::oh_camera::{OhCameraAccess, OhCameraPlayer},
    crate::{
        audio::*,
        cx::Cx,
        event::{
            video_playback::{
                VideoPlaybackPreparedEvent, VideoTextureUpdatedEvent, VideoYuvMetadata,
            },
            Event,
        },
        makepad_live_id::LiveId,
        media_api::CxMediaApi,
        midi::*,
        video::*,
    },
    std::{
        collections::HashMap,
        sync::{Arc, Mutex},
    },
};

#[derive(Clone)]
pub struct OsMidiOutput {}

impl OsMidiOutput {
    pub fn send(&self, _port_id: Option<MidiPortId>, _data: MidiData) {}
}

pub struct OsMidiInput {}

impl OsMidiInput {
    pub fn receive(&mut self) -> Option<(MidiPortId, MidiData)> {
        None
    }
}

#[derive(Default)]
pub struct CxOpenHarmonyMedia {
    camera: Option<Arc<Mutex<OhCameraAccess>>>,
    camera_descs_sent: bool,
    /// Video widgets bound to the camera stream, by video id.
    pub(crate) camera_players: HashMap<LiveId, OhCameraPlayer>,
}

impl CxOpenHarmonyMedia {
    /// The camera subsystem, enumerated on first use (no permission needed
    /// for the listing).
    pub fn camera(&mut self) -> Arc<Mutex<OhCameraAccess>> {
        if self.camera.is_none() {
            self.camera = Some(Arc::new(Mutex::new(OhCameraAccess::new())));
        }
        self.camera.as_ref().unwrap().clone()
    }
}

impl Cx {
    pub(crate) fn handle_media_signals(&mut self) {
        // The camera list goes out once, on the first media signal after
        // something asked for it (`video_input` raises the signal).
        if self.os.media.camera.is_some() && !self.os.media.camera_descs_sent {
            self.os.media.camera_descs_sent = true;
            let descs = self.os.media.camera().lock().unwrap().get_updated_descs();
            self.call_event_handler(&Event::VideoInputs(VideoInputsEvent { descs }));
        }
    }

    pub fn reinitialise_media(&mut self) {}

    /// Runs every vsync: upload the newest camera frame of every bound
    /// Video widget and tell it its textures changed.
    pub(crate) fn poll_ohos_camera_players(&mut self) {
        if self.os.media.camera_players.is_empty() || self.os.display.is_none() {
            return;
        }
        let ids: Vec<LiveId> = self.os.media.camera_players.keys().copied().collect();
        let mut events = Vec::new();
        for video_id in ids {
            let Some(player) = self.os.media.camera_players.get_mut(&video_id) else { continue };
            if !player.playing {
                continue;
            }
            let Some(frame) = player.shared.take_latest() else { continue };
            if frame.width == 0 || frame.height == 0 {
                player.shared.recycle(frame);
                continue;
            }
            let gl = &self.os.display.as_ref().unwrap().libgl;
            super::super::gl_video_upload::upload_i420_slices_to_gl(
                gl,
                &mut self.textures,
                player.tex_y,
                player.tex_u,
                player.tex_v,
                &frame.y,
                &frame.u,
                &frame.v,
                frame.width,
                frame.height,
            );
            player.width = frame.width;
            player.height = frame.height;
            player.shared.recycle(frame);
            if !player.prepared {
                player.prepared = true;
                events.push(Event::VideoPlaybackPrepared(VideoPlaybackPreparedEvent {
                    video_id,
                    video_width: player.width,
                    video_height: player.height,
                    duration: 0,
                    is_seekable: false,
                    video_tracks: vec!["camera".to_string()],
                    audio_tracks: vec![],
                }));
            }
            events.push(Event::VideoTextureUpdated(VideoTextureUpdatedEvent {
                video_id,
                current_position_ms: 0,
                yuv: VideoYuvMetadata {
                    enabled: true,
                    matrix: 1.0,
                    biplanar: false,
                    full_range: false,
                    rotation_steps: player.rotation_steps,
                    external: false,
                    array: false,
                },
                rgba_gl_2d: false,
            }));
        }
        for event in events {
            self.call_event_handler(&event);
        }
    }
}

impl CxMediaApi for Cx {
    fn midi_input(&mut self) -> MidiInput {
        MidiInput(Some(OsMidiInput {}))
    }

    fn midi_output(&mut self) -> MidiOutput {
        MidiOutput(Some(OsMidiOutput {}))
    }

    fn midi_reset(&mut self) {}

    fn use_midi_inputs(&mut self, _ports: &[MidiPortId]) {}

    fn use_midi_outputs(&mut self, _ports: &[MidiPortId]) {}

    fn use_audio_inputs(&mut self, _devices: &[AudioDeviceId]) {}

    fn use_audio_outputs(&mut self, _devices: &[AudioDeviceId]) {}

    fn audio_output_box_os(&mut self, _index: usize, _f: AudioOutputFn) {}

    fn audio_input_box(&mut self, _index: usize, _f: AudioInputFn) {}

    fn video_input_box(&mut self, index: usize, f: VideoInputFn) {
        if index >= MAX_VIDEO_DEVICE_INDEX {
            return;
        }
        *self.os.media.camera().lock().unwrap().video_input_cb[index]
            .lock()
            .unwrap() = Some(f);
        crate::thread::SignalToUI::set_ui_signal();
    }

    fn camera_frame_input_box(&mut self, index: usize, f: CameraFrameInputFn) {
        if index >= MAX_VIDEO_DEVICE_INDEX {
            return;
        }
        *self.os.media.camera().lock().unwrap().camera_frame_input_cb[index]
            .lock()
            .unwrap() = Some(f);
        crate::thread::SignalToUI::set_ui_signal();
    }

    fn use_video_input(&mut self, inputs: &[(VideoInputId, VideoFormatId)]) {
        self.os.media.camera().lock().unwrap().use_video_input(inputs);
    }
}
