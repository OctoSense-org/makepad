// Clone for Box<Path>, Box<OsStr> and Box<CStr> (stable 1.29): cc 1.8's `CompilerFlag` derives
// Clone over a `Box<Path>` field, and blake3's build script reaches it on x86_64 through
// `is_flag_supported`, which parked on E0277 (`Box<Path>: Clone`).

use std::ffi::{CStr, CString, OsStr};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
struct CompilerFlag {
    compiler: Box<Path>,
    flag: Box<OsStr>,
}

#[test]
fn boxed_paths_and_strings_clone_into_their_own_allocation() {
    let path: Box<Path> = Path::new("/usr/bin/cc").into();
    let copy = path.clone();
    assert_eq!(&*copy, Path::new("/usr/bin/cc"));
    assert_ne!(copy.as_os_str().as_encoded_bytes().as_ptr(), path.as_os_str().as_encoded_bytes().as_ptr());

    let os: Box<OsStr> = OsStr::new("-mavx512f").into();
    let os_copy = os.clone();
    assert_eq!(&*os_copy, OsStr::new("-mavx512f"));
    assert_ne!(os_copy.as_encoded_bytes().as_ptr(), os.as_encoded_bytes().as_ptr());

    let c: Box<CStr> = CString::new("cc").unwrap().into_boxed_c_str();
    let c_copy = c.clone();
    assert_eq!(c_copy.to_bytes_with_nul(), b"cc\0");
    assert_ne!(c_copy.as_ptr(), c.as_ptr());

    // as cc uses it: a derived Clone over the boxed fields
    let key = CompilerFlag { compiler: path, flag: os };
    let other = key.clone();
    drop(key);
    assert_eq!(other, CompilerFlag { compiler: Path::new("/usr/bin/cc").into(), flag: OsStr::new("-mavx512f").into() });
}
