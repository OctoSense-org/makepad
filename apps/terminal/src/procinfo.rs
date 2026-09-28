//! What a tab needs to know about the processes in its PTY: the name of the
//! foreground job (tab titles, confirm-before-close) and the shell's working
//! directory (a new tab opens where the current one is). Best effort: every
//! query answers `None` when the platform cannot tell.

use std::path::PathBuf;

/// The working directory of `pid`.
pub fn cwd(pid: i32) -> Option<PathBuf> {
    if pid <= 0 {
        return None;
    }
    platform::cwd(pid)
}

/// The short executable name of `pid` (`vim`, `zsh`). A program installed
/// as a versioned file (Claude Code runs `…/claude/versions/2.1.283`) is
/// named by the directory it lives in.
pub fn name(pid: i32) -> Option<String> {
    if pid <= 0 {
        return None;
    }
    let name = platform::name(pid).filter(|name| !name.is_empty())?;
    if !looks_like_version(&name) {
        return Some(name);
    }
    Some(platform::path(pid).and_then(|path| name_from_path(&path)).unwrap_or(name))
}

fn looks_like_version(name: &str) -> bool {
    name.starts_with(|c: char| c.is_ascii_digit()) && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '+'))
}

/// The program a versioned executable belongs to: the nearest directory
/// up its path that is neither a version nor a packaging directory.
fn name_from_path(path: &std::path::Path) -> Option<String> {
    const PACKAGING: &[&str] = &["versions", "releases", "bin", "current", "libexec", "share", "lib"];
    path.ancestors()
        .skip(1)
        .filter_map(|dir| dir.file_name()?.to_str())
        .find(|dir| !looks_like_version(dir) && !PACKAGING.contains(dir) && !dir.starts_with('.'))
        .map(str::to_owned)
}

#[cfg(target_os = "macos")]
mod platform {
    use std::ffi::{c_int, c_void, CStr};
    use std::path::PathBuf;

    extern "C" {
        fn proc_pidinfo(pid: c_int, flavor: c_int, arg: u64, buffer: *mut c_void, size: c_int) -> c_int;
        fn proc_name(pid: c_int, buffer: *mut c_void, size: u32) -> c_int;
    }

    // Layout checked against <sys/proc_info.h> (macOS 26): sizeof(struct
    // vnode_info) = 152, sizeof(struct proc_vnodepathinfo) = 2352, the cwd
    // path at offset 152, MAXPATHLEN = 1024.
    const PROC_PIDVNODEPATHINFO: c_int = 9;
    /// `struct vnode_info` precedes each path in `vnode_info_path`.
    const VNODE_INFO_SIZE: usize = 152;
    const MAXPATHLEN: usize = 1024;
    /// `struct proc_vnodepathinfo`: the cwd entry, then the root entry.
    const VNODEPATHINFO_SIZE: usize = 2 * (VNODE_INFO_SIZE + MAXPATHLEN);

    pub fn cwd(pid: i32) -> Option<PathBuf> {
        let mut buf = vec![0u8; VNODEPATHINFO_SIZE];
        let n = unsafe {
            proc_pidinfo(pid, PROC_PIDVNODEPATHINFO, 0, buf.as_mut_ptr().cast(), buf.len() as c_int)
        };
        if n as usize != VNODEPATHINFO_SIZE {
            return None;
        }
        let path = CStr::from_bytes_until_nul(&buf[VNODE_INFO_SIZE..VNODE_INFO_SIZE + MAXPATHLEN]).ok()?;
        let path = path.to_str().ok()?;
        (!path.is_empty()).then(|| PathBuf::from(path))
    }

    extern "C" {
        fn proc_pidpath(pid: c_int, buffer: *mut c_void, size: u32) -> c_int;
    }

    pub fn path(pid: i32) -> Option<PathBuf> {
        let mut buf = vec![0u8; 4096];
        let n = unsafe { proc_pidpath(pid, buf.as_mut_ptr().cast(), buf.len() as u32) };
        // The kernel never reports more than it wrote; clamp anyway.
        (n > 0).then(|| PathBuf::from(String::from_utf8_lossy(&buf[..(n as usize).min(buf.len())]).into_owned()))
    }

    pub fn name(pid: i32) -> Option<String> {
        let mut buf = [0u8; 256];
        let n = unsafe { proc_name(pid, buf.as_mut_ptr().cast(), buf.len() as u32) };
        if n <= 0 {
            return None;
        }
        Some(String::from_utf8_lossy(&buf[..(n as usize).min(buf.len())]).into_owned())
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::path::PathBuf;

    pub fn cwd(pid: i32) -> Option<PathBuf> {
        std::fs::read_link(format!("/proc/{pid}/cwd")).ok()
    }

    pub fn path(pid: i32) -> Option<PathBuf> {
        std::fs::read_link(format!("/proc/{pid}/exe")).ok()
    }

    pub fn name(pid: i32) -> Option<String> {
        let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).ok()?;
        Some(comm.trim_end().to_owned())
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod platform {
    use std::path::PathBuf;

    pub fn cwd(_pid: i32) -> Option<PathBuf> {
        None
    }

    pub fn name(_pid: i32) -> Option<String> {
        None
    }

    pub fn path(_pid: i32) -> Option<PathBuf> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn this_process_has_a_cwd_and_a_name() {
        let pid = std::process::id() as i32;
        let want = std::env::current_dir().unwrap().canonicalize().unwrap();
        assert_eq!(cwd(pid).map(|p| p.canonicalize().unwrap()), Some(want));
        assert!(name(pid).is_some_and(|n| !n.is_empty()));
    }

    #[test]
    fn a_versioned_executable_is_named_by_its_directory() {
        use std::path::Path;
        assert!(looks_like_version("2.1.283"));
        assert!(!looks_like_version("claude"));
        assert_eq!(name_from_path(Path::new("/Users/u/.local/share/claude/versions/2.1.283")).as_deref(), Some("claude"));
        assert_eq!(name_from_path(Path::new("/opt/tool/releases/1.0/bin/9")).as_deref(), Some("tool"));
    }

    #[test]
    fn no_process_answers_none() {
        assert_eq!(cwd(0), None);
        assert_eq!(name(-1), None);
    }
}
