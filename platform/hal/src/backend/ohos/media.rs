//! OpenHarmony's media service: nothing is wired yet (audio, camera, video and file dialogs
//! answer as a system without them does).
use crate::backend::exec::{Gpu, OsCmd};
use crate::buffers::CmdView;
use crate::cmds::HalCmd;
use crate::session::Session;
use crate::Id;

#[derive(Default)]
pub struct OhosMedia;
impl OhosMedia {
    pub fn command(&mut self, _gpu: &mut dyn Gpu, _s: &mut Session, _c: &HalCmd) -> bool { false }
    pub fn native_frame(&mut self, _s: &mut Session, _window: Id, _packet: u32, _view: &CmdView) {}
    pub fn dialog(&mut self, _sid: u32, _command: OsCmd) -> bool { false }
    pub fn drain(&mut self, _gpu: &mut dyn Gpu, _s: &mut Session) {}
    pub fn next_due(&self, _sid: u32, _now: u64) -> Option<u64> { None }
    pub fn session_closed(&mut self, _sid: u32) {}
}

pub fn can_play_type(_mime: &str) -> &'static str { "" }
