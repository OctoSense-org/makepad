use crate::cx::Cx;
pub use crate::makepad_error_log::*;
use makepad_studio_protocol::{AppToStudio, StudioLogItem};

#[allow(unused)]
fn log_level_prefix(level: LogLevel) -> &'static str {
    match level {
        LogLevel::Panic => "[!]",
        LogLevel::Error => "[E]",
        LogLevel::Warning => "[W]",
        LogLevel::Log => "[I]",
        LogLevel::Wait => "[.]",
    }
}

#[cfg(target_os = "android")]
fn android_logcat_write(
    file_name: &str,
    line_start: u32,
    column_start: u32,
    message: &str,
    level: LogLevel,
) {
    use std::ffi::c_int;
    extern "C" {
        pub fn __android_log_write(prio: c_int, tag: *const u8, text: *const u8) -> c_int;
    }

    let prio: c_int = match level {
        LogLevel::Error | LogLevel::Panic => 6,
        LogLevel::Warning => 5,
        _ => 4,
    };
    let msg = format!(
        "{}:{}:{} - {}\0",
        file_name,
        line_start + 1,
        column_start + 1,
        message
    );
    unsafe { __android_log_write(prio, "Makepad\0".as_ptr(), msg.as_ptr()) };
}

/// Send a log line to hilog on OpenHarmony.
///
/// Without this, `log!` on OHOS goes nowhere: `log_with_level` dispatches through
/// a function pointer and no OHOS writer was ever installed, so the platform is
/// silent. That made every failure on device undiagnosable — an app that draws
/// nothing and says nothing.
///
/// `OH_LOG_Print` is variadic and treats its `fmt` argument as a printf format
/// string, so the message goes through as a `%s` *argument* rather than as the
/// format itself — otherwise a log line containing `%` would be read as a
/// conversion and could walk off the stack. (`OH_LOG_PrintMsg` takes plain text
/// and would be safer, but it is behind hilog-sys' `api-18` feature.)
#[cfg(target_env = "ohos")]
fn ohos_hilog_write(
    file_name: &str,
    line_start: u32,
    column_start: u32,
    message: &str,
    level: LogLevel,
) {
    use std::ffi::CString;
    let lvl = match level {
        LogLevel::Error | LogLevel::Panic => hilog_sys::LogLevel::LOG_ERROR,
        LogLevel::Warning => hilog_sys::LogLevel::LOG_WARN,
        _ => hilog_sys::LogLevel::LOG_INFO,
    };
    let text = format!(
        "{}{}:{}:{} - {}",
        log_level_prefix(level),
        file_name,
        line_start + 1,
        column_start + 1,
        message
    );
    if let (Ok(tag), Ok(msg)) = (CString::new("Makepad"), CString::new(text)) {
        unsafe {
            hilog_sys::OH_LOG_Print(
                hilog_sys::LogType::LOG_APP,
                lvl,
                0,
                tag.as_ptr(),
                c"%s".as_ptr(),
                msg.as_ptr(),
            );
        }
    }
}

impl Cx {
    pub fn init_log() {
        let mut logger = LOG_WITH_LEVEL.write().expect("Logger lock poisoned");
        *logger = log_with_level_makepad_platform;
    }
}

pub(crate) fn log_with_level_makepad_platform(
    file_name: &str,
    line_start: u32,
    column_start: u32,
    line_end: u32,
    column_end: u32,
    message: String,
    level: LogLevel,
) {
    // lets send out our log message on the studio websocket
    #[cfg(target_arch = "wasm32")]
    {
        #[link(wasm_import_module = "env")]
        extern "C" {
            pub fn js_console_log(u8_ptr: u32, len: u32);
            pub fn js_console_error(u8_ptr: u32, len: u32);
        }
        let msg = format!(
            "{}:{}:{} - {}",
            file_name, line_start, column_start, message
        );
        let buf = msg.as_bytes();
        if let LogLevel::Error = level {
            unsafe { js_console_error(buf.as_ptr() as u32, buf.len() as u32) };
        } else {
            unsafe { js_console_log(buf.as_ptr() as u32, buf.len() as u32) };
        }
    }

    #[cfg(target_os = "android")]
    android_logcat_write(file_name, line_start, column_start, &message, level);
    #[cfg(target_env = "ohos")]
    ohos_hilog_write(file_name, line_start, column_start, &message, level);

    let studio_enabled = Cx::has_studio_web_socket();
    let studio_connected = Cx::has_studio_web_socket_connected();

    if !studio_connected {
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        println!(
            "{} {}:{}:{} - {}",
            log_level_prefix(level),
            file_name,
            line_start + 1,
            column_start + 1,
            message
        );
        #[cfg(target_os = "ios")]
        {
            extern "C" {
                fn NSLog(fmt: crate::os::apple::apple_sys::ObjcId, ...);
            }
            use crate::os::apple::apple_util::str_to_nsstring;
            let msg = format!(
                "{} {}:{}:{} - {}",
                log_level_prefix(level),
                file_name,
                line_start + 1,
                column_start + 1,
                message
            );
            unsafe { NSLog(str_to_nsstring(&msg)) };
        }
        #[cfg(target_env = "ohos")]
        {
            let msg = format!(
                "{} {}:{}:{} - {}\0",
                log_level_prefix(level),
                file_name,
                line_start,
                column_start,
                message
            );
            let hilevel: hilog_sys::LogLevel = match level {
                LogLevel::Warning => hilog_sys::LogLevel::LOG_WARN,
                LogLevel::Error => hilog_sys::LogLevel::LOG_ERROR,
                LogLevel::Log => hilog_sys::LogLevel::LOG_INFO,
                _ => hilog_sys::LogLevel::LOG_INFO,
            };
            unsafe {
                hilog_sys::OH_LOG_Print(
                    hilog_sys::LogType::LOG_APP,
                    hilevel,
                    0x03D00,
                    "makepad-ohos\0".as_ptr().cast(),
                    "%{public}s\0".as_ptr().cast(),
                    msg.as_ptr(),
                )
            };
        }
    }

    if studio_enabled {
        Cx::send_studio_message(AppToStudio::LogItem(StudioLogItem {
            file_name: file_name.to_string(),
            line_start,
            column_start,
            line_end,
            column_end,
            message,
            explanation: None,
            level,
        }));
    }
}

#[cfg(target_arch = "wasm32")]
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
pub fn profile_start() -> Instant {
    Instant::now()
}

#[cfg(target_arch = "wasm32")]
pub struct ProfileStart {
    started_at: f64,
}

#[cfg(target_arch = "wasm32")]
impl ProfileStart {
    pub fn elapsed(&self) -> Duration {
        Duration::from_secs_f64((Cx::time_now() - self.started_at).max(0.0))
    }
}

#[cfg(target_arch = "wasm32")]
pub fn profile_start() -> ProfileStart {
    ProfileStart {
        started_at: Cx::time_now(),
    }
}

#[macro_export]
macro_rules! profile_end {
    ( $ inst: expr) => {
        $crate::log::log_with_level(
            file!(),
            line!(),
            column!(),
            line!(),
            column!() + 4,
            format!(
                "Profile time {} ms",
                ($inst.elapsed().as_nanos() as f64) / 1000000f64
            ),
            $crate::log::LogLevel::Log,
        )
    };
}

#[macro_export]
macro_rules!profile_end_log {
    ( $inst:expr, $ ( $ t: tt) *) => {
        $crate::log::log_with_level(
            file!(),
            line!(),
            column!(),
            line!(),
            column!() + 4,
            format!("Profile time {} {}",( $ inst.elapsed().as_nanos() as f64) / 1000000f64, format!( $ ( $ t) *)),
            $ crate::log::LogLevel::Log
        )
    }
}

#[macro_export]
macro_rules! fmt_over {
    ($dst:expr, $($arg:tt)*) => {
        {
            $dst.clear();
            use std::fmt::Write;
            $dst.write_fmt(std::format_args!($($arg)*)).unwrap();
        }
    };
}

#[macro_export]
macro_rules! fmt_over_ref {
    ($dst:expr, $($arg:tt)*) => {
        {
            $dst.clear();
            use std::fmt::Write;
            $dst.write_fmt(std::format_args!($($arg)*)).unwrap();
            &$dst
        }
    };
}
