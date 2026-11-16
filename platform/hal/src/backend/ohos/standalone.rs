//! A standalone app HAP: its libmakepad.so is the app itself, natively compiled, with this HAL
//! in the same library. The app's `app_main!` registers the library as the Node-API module
//! `makepad` when it loads ([`register`]); the ability's `start` then sends stdout and stderr to
//! hilog and runs the app on its own thread.
use super::sys;
use std::ffi::{c_int, c_void, CString};
use std::sync::OnceLock;

extern "C" {
    fn pipe(fds: *mut c_int) -> c_int;
    fn dup2(old: c_int, new: c_int) -> c_int;
    fn read(fd: c_int, buf: *mut c_void, n: usize) -> isize;
    fn close(fd: c_int) -> c_int;
    fn fflush(stream: *mut c_void) -> c_int;
    fn _exit(status: c_int) -> !;
}

/// hilog's domain of the app's lines (`hilog | grep makepad`).
const DOMAIN: u32 = 0xCA11;

static APP: OnceLock<fn()> = OnceLock::new();

fn log(text: &str) {
    // hilog drops what is past its line limit
    for part in text.as_bytes().chunks(900) {
        if let Ok(t) = CString::new(String::from_utf8_lossy(part).replace('\0', " ")) {
            unsafe { sys::OH_LOG_Print(sys::LOG_APP, sys::LOG_INFO, DOMAIN, c"makepad".as_ptr(), c"%{public}s".as_ptr(), t.as_ptr()) };
        }
    }
}

/// stdout and stderr go nowhere in an OpenHarmony app: a pipe on both, each line to hilog.
fn redirect_stdio() {
    unsafe {
        let mut fds = [0 as c_int; 2];
        if pipe(fds.as_mut_ptr()) != 0 {
            return;
        }
        dup2(fds[1], 1);
        dup2(fds[1], 2);
        close(fds[1]);
        let rd = fds[0];
        let _ = std::thread::Builder::new().name("makepad-hilog".into()).spawn(move || {
            let mut buf = vec![0u8; 8192];
            let mut line: Vec<u8> = Vec::new();
            loop {
                let n = read(rd, buf.as_mut_ptr().cast(), buf.len());
                if n <= 0 {
                    break;
                }
                for &b in &buf[..n as usize] {
                    if b == b'\n' {
                        log(&String::from_utf8_lossy(&line));
                        line.clear();
                    } else {
                        line.push(b);
                    }
                }
            }
        });
    }
}

/// What `app_main!` runs as the library loads: `app` is the app's entry.
/// # Safety
/// Called once, from the library's constructor.
pub unsafe fn register(app: fn()) {
    let _ = APP.set(app);
    super::napi::register_module();
}

/// The ability exists (its directories and raw files are set): the app starts, once.
pub(super) fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        redirect_stdio();
        std::panic::set_hook(Box::new(|info| {
            log(&format!("panic: {info}"));
            log(&format!("{}", std::backtrace::Backtrace::force_capture()));
        }));
        let Some(app) = APP.get().copied() else { return log("no app registered with this library") };
        // The app runs on a thread of its own, so its thread-locals are dropped before this
        // thread ends the process.
        let spawned = std::thread::Builder::new().name("makepad-host".into()).spawn(move || {
            let run = std::thread::Builder::new().name("makepad-app".into()).stack_size(16 << 20).spawn(app);
            match run.map(|thread| thread.join()) {
                Ok(Ok(())) => {}
                // the panic hook logged it
                Ok(Err(_)) => return,
                Err(error) => return log(&format!("cannot start the app thread: {error}")),
            }
            // The app ended (it quit), so the ability and the process end too.
            log("the app returned; ending the process");
            crate::Hal::global().before_exit();
            let _ = std::io::Write::flush(&mut std::io::stdout());
            unsafe { fflush(std::ptr::null_mut()) };
            super::napi::call("terminate", "");
            std::thread::sleep(std::time::Duration::from_millis(300));
            unsafe { _exit(0) }
        });
        if let Err(error) = spawned {
            log(&format!("cannot start the app's host thread: {error}"));
        }
    });
}
