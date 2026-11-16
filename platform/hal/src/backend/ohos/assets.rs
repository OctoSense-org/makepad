//! The HAP's raw files stay with the native host, independently of the compiled app's lifetime.
use super::sys;
use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicPtr, Ordering};

static MANAGER: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

/// # Safety
/// `manager` came from `OH_ResourceManager_InitNativeResourceManager` and is never released.
pub unsafe fn set_manager(manager: *mut c_void) {
    MANAGER.store(manager, Ordering::Release);
}

/// The raw file's length, with its bytes in `out` when they fit; -1 when it is missing.
pub(crate) fn read(path: &str, out: &mut [u8]) -> i64 {
    let manager = MANAGER.load(Ordering::Acquire);
    let Ok(path) = CString::new(path) else { return -1 };
    if manager.is_null() { return -1; }
    unsafe {
        let file = sys::OH_ResourceManager_OpenRawFile64(manager, path.as_ptr());
        if file.is_null() { return -1; }
        let len = sys::OH_ResourceManager_GetRawFileSize64(file);
        let mut result = len;
        if len >= 0 && out.len() as u64 >= len as u64 {
            let mut offset = 0;
            while offset < len as usize {
                let n = sys::OH_ResourceManager_ReadRawFile64(file, out[offset..].as_mut_ptr().cast(), len - offset as i64);
                if n <= 0 { result = -1; break; }
                offset += n as usize;
            }
        }
        sys::OH_ResourceManager_CloseRawFile64(file);
        result
    }
}

/// A raw file as a descriptor range: the open descriptor (the caller owns and closes it) and
/// `out = [start, length]`; -1 for a missing file.
pub(crate) fn range(path: &str, out: &mut [i64]) -> i64 {
    let manager = MANAGER.load(Ordering::Acquire);
    let Ok(path) = CString::new(path) else { return -1 };
    if manager.is_null() || out.len() < 2 { return -1; }
    unsafe {
        let file = sys::OH_ResourceManager_OpenRawFile64(manager, path.as_ptr());
        if file.is_null() { return -1; }
        let mut descriptor = sys::RawFileDescriptor64 { fd: -1, start: 0, length: 0 };
        let ok = sys::OH_ResourceManager_GetRawFileDescriptor64(file, &mut descriptor);
        sys::OH_ResourceManager_CloseRawFile64(file);
        if !ok || descriptor.fd < 0 { return -1; }
        out[0] = descriptor.start;
        out[1] = descriptor.length;
        descriptor.fd as i64
    }
}

/// A raw file mapped read-only (see [`range`]).
pub(crate) fn map(path: &str) -> std::io::Result<std::sync::Arc<makepad_font_core::MappedBytes>> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    let mut span = [0i64; 2];
    let fd = range(path, &mut span);
    if fd < 0 { return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no raw file of that name in the HAP")); }
    let fd = unsafe { OwnedFd::from_raw_fd(fd as i32) };
    let bytes = makepad_font_core::HostBuf::map_fd_range_named(fd.as_raw_fd(), span[0] as u64, span[1] as usize, path)?;
    bytes.mapped().cloned().ok_or_else(|| std::io::Error::other("raw file not mapped"))
}
