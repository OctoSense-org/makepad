// Debug for MaybeUninit<T> (stable since 1.41): the value may be uninitialised, so it prints as
// its type from `MaybeUninit` on. socket2 0.6's `MaybeUninitSlice` (src/lib.rs) formats its
// buffer with `fmt::Debug::fmt(self.0.as_slice(), fmt)`, a `&[MaybeUninit<u8>]`, which failed
// for want of the element's impl.
use std::fmt;
use std::mem::MaybeUninit;

// socket2's shape
struct MaybeUninitSlice<'a>(&'a mut [MaybeUninit<u8>]);
impl<'a> fmt::Debug for MaybeUninitSlice<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, fmt)
    }
}

#[derive(Debug)]
struct RecvBuf {
    filled: usize,
    storage: MaybeUninit<[u8; 4]>,
}

#[test]
fn maybe_uninit_prints_its_type_not_its_bytes() {
    assert_eq!(format!("{:?}", MaybeUninit::<u8>::uninit()), "MaybeUninit<u8>");
    assert_eq!(format!("{:?}", MaybeUninit::new(7u32)), "MaybeUninit<u32>");
    assert_eq!(format!("{:?}", MaybeUninit::<(u8, bool)>::zeroed()), "MaybeUninit<(u8, bool)>");
    assert_eq!(format!("{:?}", MaybeUninit::new("text")), "MaybeUninit<&str>");
    assert_eq!(format!("{:?}", MaybeUninit::<[f64; 3]>::uninit()), "MaybeUninit<[f64; 3]>");
    assert_eq!(format!("{:?}", MaybeUninit::<*const i16>::uninit()), "MaybeUninit<*const i16>");
    // it pads, unlike a derived unit struct
    assert_eq!(format!("[{:>18?}]", MaybeUninit::<u8>::uninit()), "[   MaybeUninit<u8>]");
    assert_eq!(format!("[{:-<17?}]", MaybeUninit::<i8>::uninit()), "[MaybeUninit<i8>--]");
    assert_eq!(format!("{:.11?}", MaybeUninit::<u8>::uninit()), "MaybeUninit");
}

#[test]
fn slices_and_fields_of_maybe_uninit() {
    let mut buf = [MaybeUninit::<u8>::uninit(); 3];
    assert_eq!(format!("{:?}", MaybeUninitSlice(&mut buf)), "[MaybeUninit<u8>, MaybeUninit<u8>, MaybeUninit<u8>]");
    assert_eq!(format!("{:?}", MaybeUninitSlice(&mut [])), "[]");
    assert_eq!(format!("{:#?}", &buf[..1]), "[\n    MaybeUninit<u8>,\n]");
    assert_eq!(format!("{:?}", buf), "[MaybeUninit<u8>, MaybeUninit<u8>, MaybeUninit<u8>]");
    let v: Vec<MaybeUninit<u16>> = vec![MaybeUninit::new(1), MaybeUninit::uninit()];
    assert_eq!(format!("{:?}", v), "[MaybeUninit<u16>, MaybeUninit<u16>]");
    let r = RecvBuf { filled: 2, storage: MaybeUninit::uninit() };
    assert_eq!(format!("{:?}", r), "RecvBuf { filled: 2, storage: MaybeUninit<[u8; 4]> }");
    assert_eq!(format!("{:?}", Some(&r.storage)), "Some(MaybeUninit<[u8; 4]>)");
    assert_eq!(r.filled, 2);
}
