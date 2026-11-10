// `str::from_utf8` and friends as associated functions of the primitive type (Rust 1.87),
// named without `core::str` in scope: yoke-derive 0.8's `str::from_utf8(&underscores)`.
#[test]
fn a01_associated_from_utf8() {
    let bytes = vec![b'_'; 3];
    let s = str::from_utf8(&bytes).expect("ascii");
    println!("| {s}yoke");
    println!("| {:?}", str::from_utf8(&[0xff]).is_err());
}
#[test]
fn a02_associated_from_utf8_mut_and_unchecked() {
    let mut m = *b"ab";
    let t = str::from_utf8_mut(&mut m).unwrap();
    t.make_ascii_uppercase();
    println!("| {t}");
    println!("| {}", unsafe { str::from_utf8_unchecked(b"ok") });
}
