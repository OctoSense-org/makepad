//! The fonts installed on the machine, for the terminal's font settings:
//! a primary font (any family — monospace ones are listed first) and a CJK
//! fallback (PingFang on macOS, Noto/Source Han CJK on Linux, found by
//! `auto_cjk`).
//!
//! The scan reads only each file's table directory, `name` and `post`
//! tables. Font collections (`.ttc`) hold several faces and the text
//! engine loads the first face of a file, so a face deeper in a collection
//! is copied out once into a standalone font under the terminal's cache
//! (`standalone_path`).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// The font-file reader lives in the text engine, shared with every app's
// system-font fallback.
pub use makepad_widgets::makepad_draw::text::system_fonts::{
    face_loadable, font_dirs, group, loadable, read_faces, scan, standalone_path, Face, Family,
};

static FAMILIES: OnceLock<Vec<Family>> = OnceLock::new();
static SCANNING: std::sync::Once = std::sync::Once::new();

/// Start the scan on a thread (about a second on macOS, where fonts
/// downloaded on demand live in many directories). When it is done the
/// settings generation moves, so every terminal re-applies its fonts.
pub fn warm() {
    SCANNING.call_once(|| {
        let _ = std::thread::Builder::new().name("terminal-font-scan".into()).spawn(|| {
            FAMILIES.get_or_init(scan_usable);
            crate::settings::bump_generation();
            makepad_widgets::makepad_platform::thread::SignalToUI::set_ui_signal();
        });
    });
}

/// The installed families whose regular face the text engine can load.
fn scan_usable() -> Vec<Family> {
    let mut families = group(scan(&font_dirs()));
    families.retain(|family| face_loadable(&family.regular));
    for family in &mut families {
        if family.bold.as_ref().is_some_and(|bold| !face_loadable(bold)) {
            family.bold = None;
        }
    }
    families
}

/// Every installed family, monospace first, then by name; empty until the
/// scan `warm` started has finished.
pub fn families() -> &'static [Family] {
    warm();
    FAMILIES.get().map(Vec::as_slice).unwrap_or(&[])
}

/// Whether the scan has finished.
pub fn ready() -> bool {
    FAMILIES.get().is_some()
}

pub fn find(name: &str) -> Option<&'static Family> {
    families().iter().find(|family| family.name.eq_ignore_ascii_case(name))
}

/// The CJK fallback to use when the setting says `auto`.
pub fn auto_cjk() -> Option<&'static Family> {
    const PREFERRED: &[&str] = &[
        "PingFang SC",
        "Hiragino Sans GB",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
    ];
    PREFERRED.iter().find_map(|name| find(name))
}

/// A face's loadable file, prepared off the UI thread: copying a face out
/// of a large collection and checking it takes up to a second (PingFang,
/// STHeiti, the Toppan and Yu families). `Ready(None)`: it cannot be used.
#[derive(Clone, Debug, PartialEq)]
pub enum Prepared {
    Ready(Option<PathBuf>),
    Pending,
}

pub fn prepared_path(face: &Face, cache_dir: &Path) -> Prepared {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static PREPARED: Mutex<Option<HashMap<(PathBuf, u32), Prepared>>> = Mutex::new(None);
    let key = (face.path.clone(), face.index);
    {
        let mut map = PREPARED.lock().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(HashMap::new);
        if let Some(state) = map.get(&key) {
            return state.clone();
        }
        map.insert(key.clone(), Prepared::Pending);
    }
    let (face, cache_dir) = (face.clone(), cache_dir.to_path_buf());
    let spawned = std::thread::Builder::new().name("terminal-font-prepare".into()).spawn(move || {
        let path = standalone_path(&face, &cache_dir);
        PREPARED.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(key, Prepared::Ready(path));
        // Every terminal re-applies its fonts, now with this one ready.
        crate::settings::bump_generation();
        makepad_widgets::makepad_platform::thread::SignalToUI::set_ui_signal();
    });
    if spawned.is_err() {
        return Prepared::Ready(None);
    }
    Prepared::Pending
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_has_menlo_and_a_cjk_fallback() {
        FAMILIES.get_or_init(scan_usable);
        let menlo = find("Menlo").expect("Menlo ships with macOS");
        assert!(menlo.monospace);
        assert!(find("GB18030 Bitmap").is_none(), "a bitmap-only font is not offered");
        let dir = std::env::temp_dir().join(format!("terminal-menlo-{}", std::process::id()));
        assert!(standalone_path(&menlo.regular, &dir).is_some(), "Menlo loads");
        std::fs::remove_dir_all(&dir).ok();
        if let Some(cjk) = auto_cjk() {
            let dir = std::env::temp_dir().join(format!("terminal-cjk-{}", std::process::id()));
            let path = standalone_path(&cjk.regular, &dir).expect("loadable");
            assert_eq!(read_faces(&path).unwrap()[0].family, cjk.name);
            std::fs::remove_dir_all(&dir).ok();
        }
    }
}
