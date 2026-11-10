// core/src/slice/mod.rs: `<[T]>::split_at_unchecked` and `split_at_mut_unchecked` (stable 1.79).
// cmov 0.5 splits with them in `slice_as_chunks`, which every constant-time slice comparison of
// ctutils goes through (the AEAD tag check of chacha20poly1305).
#[inline(never)]
fn chunks<const N: usize>(s: &[u8]) -> (&[u8], &[u8]) {
    let mid = s.len() / N * N;
    // SAFETY: mid <= s.len()
    unsafe { s.split_at_unchecked(mid) }
}

#[test]
fn split_at_unchecked_splits_where_asked() {
    let v = std::hint::black_box([1u8, 2, 3, 4, 5, 6, 7]);
    let (head, tail) = chunks::<3>(&v);
    assert_eq!(head, [1, 2, 3, 4, 5, 6]);
    assert_eq!(tail, [7]);
    let (all, none) = unsafe { v.split_at_unchecked(v.len()) };
    assert_eq!(all.len(), 7);
    assert!(none.is_empty());
    let words = std::hint::black_box([10u32, 20, 30]);
    let (a, b) = unsafe { words.split_at_unchecked(1) };
    assert_eq!((a, b), (&[10u32][..], &[20u32, 30][..]));
}

#[test]
fn split_at_mut_unchecked_gives_two_disjoint_halves() {
    let mut v = std::hint::black_box([1u16, 2, 3, 4, 5]);
    let (a, b) = unsafe { v.split_at_mut_unchecked(2) };
    a[0] = 100;
    b[2] = 500;
    std::mem::swap(&mut a[1], &mut b[0]);
    assert_eq!(v, [100, 3, 2, 4, 500]);
}
