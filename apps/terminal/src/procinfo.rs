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

/// The short executable name of `pid` (`vim`, `zsh`).
pub fn name(pid: i32) -> Option<String> {
    if pid <= 0 {
        return None;
    }
    platform::name(pid).filter(|name| !name.is_empty())
}

#[cfg(target_os = "macos")]
mod platform {
    use std::ffi::{c_int, c_void, CStr};
    use std::path::PathBuf;

    extern "C" {
        fn proc_pidinfo(pid: c_int, flavor: c_int, arg: u64, buffer: *mut c_void, size: c_int) -> c_int;
        fn proc_name(pid: c_int, buffer: *mut c_void, size: u32) -> c_int;
    }

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

    pub fn name(pid: i32) -> Option<String> {
        let mut buf = [0u8; 256];
        let n = unsafe { proc_name(pid, buf.as_mut_ptr().cast(), buf.len() as u32) };
        if n <= 0 {
            return None;
        }
        Some(String::from_utf8_lossy(&buf[..n as usize]).into_owned())
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::path::PathBuf;

    pub fn cwd(pid: i32) -> Option<PathBuf> {
        std::fs::read_link(format!("/proc/{pid}/cwd")).ok()
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
    fn no_process_answers_none() {
        assert_eq!(cwd(0), None);
        assert_eq!(name(-1), None);
    }
}
