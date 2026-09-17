use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The platform's own writable per-app directory, when it has one that the
/// environment does not describe: Android's files directory, handed over
/// by the activity when the process starts. `HOME` on Android is not a
/// directory the app may write, so without this the storage backend and
/// every cache below `makepad_home()` fail with a read-only file system.
static PLATFORM_DATA_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Record the platform's writable per-app directory. The first call wins;
/// an explicit `MAKEPAD_HOME` still takes precedence over it.
pub fn set_platform_data_dir(dir: &Path) {
    if !dir.as_os_str().is_empty() {
        let _ = PLATFORM_DATA_DIR.set(dir.to_path_buf());
    }
}

/// The platform's writable per-app directory, once the platform reported
/// it (Android, before `Event::Startup`). A host that keeps its own state
/// beside Makepad's uses it in place of the user's home directory.
pub fn platform_data_dir() -> Option<PathBuf> {
    PLATFORM_DATA_DIR.get().cloned()
}

/// Returns Makepad's shared per-user state directory.
///
/// `MAKEPAD_HOME` overrides the default. Otherwise it is the platform's own
/// data directory when one was recorded (Android), else `.makepad` below the
/// user's home directory, with the process temporary directory used only
/// when the platform exposes no home directory. The AI hub has an older copy
/// of this rule and should call this helper when its dependency direction
/// permits it.
pub fn makepad_home() -> PathBuf {
    resolve_home(
        std::env::var_os("MAKEPAD_HOME").map(PathBuf::from),
        platform_data_dir().map(|dir| dir.join(".makepad")),
        std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(PathBuf::from),
        std::env::temp_dir,
    )
}

fn resolve_home(
    explicit: Option<PathBuf>,
    platform: Option<PathBuf>,
    user: Option<PathBuf>,
    temp: impl FnOnce() -> PathBuf,
) -> PathBuf {
    explicit
        .or(platform)
        .unwrap_or_else(|| user.unwrap_or_else(temp).join(".makepad"))
}

/// Returns the root used by the native key/value storage backend.
pub fn storage_dir() -> PathBuf {
    makepad_home().join("storage")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_home_wins_then_the_platform_directory_then_the_user_home() {
        let temp = || PathBuf::from("/tmp");
        let p = PathBuf::from;
        assert_eq!(
            resolve_home(Some(p("/custom")), Some(p("/data/app/files/.makepad")), Some(p("/home/me")), temp),
            p("/custom")
        );
        assert_eq!(
            resolve_home(None, Some(p("/data/app/files/.makepad")), Some(p("/")), temp),
            p("/data/app/files/.makepad"),
            "Android: the files directory, not the read-only HOME"
        );
        assert_eq!(resolve_home(None, None, Some(p("/home/me")), temp), p("/home/me/.makepad"));
        assert_eq!(resolve_home(None, None, None, temp), p("/tmp/.makepad"));
    }
}
