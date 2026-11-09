// `str::parse::<PathBuf>()` (FromStr, Err = Infallible): comrak 0.55's build script writes
// `env::var("OUT_DIR").unwrap().parse().unwrap()` into a `PathBuf`.
use std::path::PathBuf;
#[test]
fn a01_parse_pathbuf() {
    let p: PathBuf = "target/out".parse().unwrap();
    println!("| {}", p.join("entitydata.rs").display());
    let q = "".parse::<PathBuf>();
    println!("| {:?}", q.map(|x| x.as_os_str().is_empty()));
}
