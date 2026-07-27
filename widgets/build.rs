use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let path = Path::new(&out_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let cwd = std::env::current_dir().unwrap();
    let mut file = File::create(path.join("makepad-widgets.path")).unwrap();
    file.write_all(&format!("{}", cwd.display()).as_bytes())
        .unwrap();

    // `mobile` — Android and OpenHarmony share a phone-shaped shell: a native
    // composer overlay instead of a docked one, a soft keyboard, a sandboxed
    // per-app HOME, and no desktop window chrome. Gate that shared behaviour on
    // `mobile` rather than repeating
    // `any(target_os = "android", target_env = "ohos")` at every site.
    //
    // NOTE iOS is deliberately NOT included: it has its own shell and its own
    // backend, and folding it in here would silently change its behaviour at
    // every one of these sites. Add it only with per-site review.
    println!("cargo:rustc-check-cfg=cfg(mobile)");
    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "android"
        || env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() == "ohos"
    {
        println!("cargo:rustc-cfg=mobile");
    }

    println!("cargo:rustc-check-cfg=cfg(ignore_query, panic_query, force_whisper)");
    println!("cargo:rerun-if-env-changed=MAKEPAD");
    if let Ok(configs) = env::var("MAKEPAD") {
        for config in configs.split(['+', ',']) {
            match config {
                "ignore_query" => println!("cargo:rustc-cfg=ignore_query"),
                "panic_query" => println!("cargo:rustc-cfg=panic_query"),
                "whisper" => println!("cargo:rustc-cfg=force_whisper"),
                _ => {}
            }
        }
    }
}
