// core's `#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]` on
// PhantomPinned: the std here stopped at Clone, Copy and Default. Every pin-project-lite struct
// with a `_pin: PhantomPinned` field and a hand-written or derived Debug needs the Debug impl:
// tokio's `Read` future (tokio/src/io/util/read.rs) and about thirty more of its I/O futures did
// not compile ("expected `&dyn core::fmt::Debug`, found `&&PhantomPinned`").
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::marker::PhantomPinned;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Read {
    pos: u32,
    _pin: PhantomPinned,
}

// tokio's shape: Debug written by hand through debug_struct, the marker as a field
struct ReadFuture<'a> {
    buf: &'a [u8],
    _pin: PhantomPinned,
}
impl std::fmt::Debug for ReadFuture<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Read").field("buf", &self.buf).field("_pin", &self._pin).finish()
    }
}

fn hash_of<T: Hash>(t: &T) -> u64 {
    let mut h = DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

#[test]
fn phantom_pinned_formats_as_its_name() {
    assert_eq!(format!("{:?}", PhantomPinned), "PhantomPinned");
    assert_eq!(format!("{:#?}", PhantomPinned), "PhantomPinned");
    // the derive writes the name with write_str: width and fill are ignored
    assert_eq!(format!("{:>15?}|", PhantomPinned), "PhantomPinned|");
    assert_eq!(format!("{:?}", Read { pos: 3, _pin: PhantomPinned }), "Read { pos: 3, _pin: PhantomPinned }");
    assert_eq!(format!("{:?}", ReadFuture { buf: &[1, 2], _pin: PhantomPinned }), "Read { buf: [1, 2], _pin: PhantomPinned }");
    assert_eq!(format!("{:#?}", ReadFuture { buf: &[], _pin: PhantomPinned }), "Read {\n    buf: [],\n    _pin: PhantomPinned,\n}");
}

#[test]
fn phantom_pinned_compares_equal_and_hashes_nothing() {
    use std::cmp::Ordering;
    assert!(PhantomPinned == PhantomPinned);
    assert_eq!(PhantomPinned.cmp(&PhantomPinned), Ordering::Equal);
    assert_eq!(PhantomPinned.partial_cmp(&PhantomPinned), Some(Ordering::Equal));
    assert_eq!(hash_of(&PhantomPinned), hash_of(&()));
    let a = Read { pos: 1, _pin: PhantomPinned };
    let b = Read { pos: 2, _pin: PhantomPinned };
    assert!(a < b && a != b && a == a.clone());
    assert_eq!(a.max(b).pos, 2);
    assert_eq!(hash_of(&a), hash_of(&1u32));
    assert_eq!(Read::default(), Read { pos: 0, _pin: PhantomPinned });
}
