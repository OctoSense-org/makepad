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

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// One face of an installed font file.
#[derive(Clone, Debug, PartialEq)]
pub struct Face {
    pub family: String,
    pub style: String,
    pub path: PathBuf,
    /// The face's index in a collection; 0 for a single-face file.
    pub index: u32,
    pub monospace: bool,
}

/// An installed family: the face to use for regular text and, when the
/// family has one, for bold.
#[derive(Clone, Debug)]
pub struct Family {
    pub name: String,
    pub monospace: bool,
    pub regular: Face,
    pub bold: Option<Face>,
}

static FAMILIES: OnceLock<Vec<Family>> = OnceLock::new();
static SCANNING: std::sync::Once = std::sync::Once::new();

/// Start the scan on a thread (about a second on macOS, where fonts
/// downloaded on demand live in many directories). When it is done the
/// settings generation moves, so every terminal re-applies its fonts.
pub fn warm() {
    SCANNING.call_once(|| {
        let _ = std::thread::Builder::new().name("terminal-font-scan".into()).spawn(|| {
            FAMILIES.get_or_init(|| group(scan(&font_dirs())));
            crate::settings::bump_generation();
            makepad_widgets::makepad_platform::thread::SignalToUI::set_ui_signal();
        });
    });
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

/// The directories fonts are installed in on this platform.
pub fn font_dirs() -> Vec<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut dirs = Vec::new();
    if cfg!(target_os = "macos") {
        dirs.push(PathBuf::from("/System/Library/Fonts"));
        dirs.push(PathBuf::from("/Library/Fonts"));
        if let Some(home) = &home {
            dirs.push(home.join("Library/Fonts"));
        }
        // Fonts macOS downloads on demand (PingFang among them).
        if let Ok(entries) = std::fs::read_dir("/System/Library/AssetsV2") {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().starts_with("com_apple_MobileAsset_Font") {
                    dirs.push(entry.path());
                }
            }
        }
    } else if cfg!(windows) {
        if let Some(windir) = std::env::var_os("WINDIR") {
            dirs.push(PathBuf::from(windir).join("Fonts"));
        }
    } else {
        dirs.push(PathBuf::from("/usr/share/fonts"));
        dirs.push(PathBuf::from("/usr/local/share/fonts"));
        if let Some(home) = &home {
            dirs.push(home.join(".local/share/fonts"));
            dirs.push(home.join(".fonts"));
        }
    }
    dirs
}

/// Every face in font files under `dirs` (recursively).
pub fn scan(dirs: &[PathBuf]) -> Vec<Face> {
    let mut faces = Vec::new();
    let mut stack: Vec<PathBuf> = dirs.to_vec();
    let mut visited = 0;
    while let Some(dir) = stack.pop() {
        visited += 1;
        if visited > 10_000 {
            break;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() {
                stack.push(path);
            } else if is_font_file(&path) {
                faces.extend(read_faces(&path).unwrap_or_default());
            }
        }
    }
    faces
}

fn is_font_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "ttf" | "otf" | "ttc" | "otc"))
}

/// Faces grouped into families, monospace first, then by name.
pub fn group(faces: Vec<Face>) -> Vec<Family> {
    let mut by_name: BTreeMap<String, Vec<Face>> = BTreeMap::new();
    for face in faces {
        if face.family.is_empty() || face.family.starts_with('.') || face.family == "LastResort" {
            continue;
        }
        by_name.entry(face.family.clone()).or_default().push(face);
    }
    let mut out: Vec<Family> = by_name
        .into_iter()
        .map(|(name, faces)| {
            let pick = |styles: &[&str]| {
                styles.iter().find_map(|want| faces.iter().find(|f| f.style.eq_ignore_ascii_case(want)).cloned())
            };
            let regular = pick(&["Regular", "Book", "Roman", "Normal", "Text", "Medium"]).unwrap_or_else(|| faces[0].clone());
            let bold = pick(&["Bold", "Semibold", "SemiBold", "Demibold", "Heavy"]);
            // Colour-bitmap emoji fonts have one advance but are no text font.
            let monospace = regular.monospace && !name.contains("Emoji");
            Family { monospace, name, regular, bold }
        })
        .collect();
    out.sort_by(|a, b| b.monospace.cmp(&a.monospace).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())));
    out
}

fn be16(b: &[u8], at: usize) -> Option<u16> {
    b.get(at..at + 2).map(|s| u16::from_be_bytes([s[0], s[1]]))
}

fn be32(b: &[u8], at: usize) -> Option<u32> {
    b.get(at..at + 4).map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

fn read_at(file: &mut File, offset: u64, len: usize) -> Option<Vec<u8>> {
    if len > 16 << 20 {
        return None;
    }
    let mut buf = vec![0; len];
    file.seek(SeekFrom::Start(offset)).ok()?;
    file.read_exact(&mut buf).ok()?;
    Some(buf)
}

/// A face's table directory: (tag, offset, length) per table.
fn tables(file: &mut File, face_offset: u32) -> Option<(u32, Vec<([u8; 4], u32, u32)>)> {
    let head = read_at(file, face_offset as u64, 12)?;
    let version = be32(&head, 0)?;
    let count = be16(&head, 4)? as usize;
    if count == 0 || count > 256 {
        return None;
    }
    let records = read_at(file, face_offset as u64 + 12, count * 16)?;
    let tables = (0..count)
        .filter_map(|i| {
            let r = &records[i * 16..i * 16 + 16];
            Some(([r[0], r[1], r[2], r[3]], be32(r, 8)?, be32(r, 12)?))
        })
        .collect();
    Some((version, tables))
}

/// The start of each face in a font file.
fn face_offsets(file: &mut File) -> Option<Vec<u32>> {
    let head = read_at(file, 0, 12)?;
    if &head[0..4] == b"ttcf" {
        let count = (be32(&head, 8)? as usize).min(256);
        let offsets = read_at(file, 12, count * 4)?;
        return Some((0..count).filter_map(|i| be32(&offsets, i * 4)).collect());
    }
    Some(vec![0])
}

pub fn read_faces(path: &Path) -> Option<Vec<Face>> {
    let mut file = File::open(path).ok()?;
    let offsets = face_offsets(&mut file)?;
    let mut faces = Vec::new();
    for (index, offset) in offsets.into_iter().enumerate() {
        let Some((_, tables)) = tables(&mut file, offset) else {
            continue;
        };
        let find = |tag: &[u8; 4]| tables.iter().find(|(t, _, _)| t == tag).map(|(_, o, l)| (*o, *l));
        let Some((name_off, name_len)) = find(b"name") else {
            continue;
        };
        let Some(name) = read_at(&mut file, name_off as u64, name_len as usize) else {
            continue;
        };
        let (family, style) = names(&name);
        let fixed_flag = find(b"post")
            .and_then(|(o, _)| read_at(&mut file, o as u64 + 12, 4))
            .and_then(|b| be32(&b, 0))
            .is_some_and(|fixed| fixed != 0);
        let monospace = fixed_flag || uniform_advances(&mut file, find(b"hhea"), find(b"hmtx"));
        if let Some(family) = family {
            faces.push(Face {
                family,
                style: style.unwrap_or_else(|| "Regular".into()),
                path: path.to_path_buf(),
                index: index as u32,
                monospace,
            });
        }
    }
    Some(faces)
}

/// Whether the glyphs share one advance (a CJK monospace font may also
/// have a double-width one). Reads the horizontal metrics directly: many
/// monospace fonts (Monaco among them) leave `post.isFixedPitch` clear.
fn uniform_advances(file: &mut File, hhea: Option<(u32, u32)>, hmtx: Option<(u32, u32)>) -> bool {
    let (Some((hhea, _)), Some((hmtx, hmtx_len))) = (hhea, hmtx) else {
        return false;
    };
    let Some(count) = read_at(file, hhea as u64 + 34, 2).and_then(|b| be16(&b, 0)) else {
        return false;
    };
    let count = (count as usize).min(hmtx_len as usize / 4).min(1024);
    if count <= 1 {
        return count == 1;
    }
    let Some(metrics) = read_at(file, hmtx as u64, count * 4) else {
        return false;
    };
    let mut widths: Vec<u16> = (0..count).filter_map(|i| be16(&metrics, i * 4)).filter(|&w| w != 0).collect();
    widths.sort_unstable();
    widths.dedup();
    match widths.as_slice() {
        [_] => true,
        [narrow, wide] => *wide == narrow * 2,
        _ => false,
    }
}

/// (family, style) from a `name` table: the typographic names (16/17) when
/// present, else the legacy ones (1/2); English Windows names preferred.
fn names(table: &[u8]) -> (Option<String>, Option<String>) {
    let Some(count) = be16(table, 2) else {
        return (None, None);
    };
    let Some(strings) = be16(table, 4) else {
        return (None, None);
    };
    let mut best: BTreeMap<u16, (u8, String)> = BTreeMap::new();
    for i in 0..count as usize {
        let r = 6 + i * 12;
        let (Some(platform), Some(encoding), Some(language), Some(name_id), Some(len), Some(off)) = (
            be16(table, r),
            be16(table, r + 2),
            be16(table, r + 4),
            be16(table, r + 6),
            be16(table, r + 8),
            be16(table, r + 10),
        ) else {
            break;
        };
        if !matches!(name_id, 1 | 2 | 16 | 17) {
            continue;
        }
        let start = strings as usize + off as usize;
        let Some(bytes) = table.get(start..start + len as usize) else {
            continue;
        };
        let (rank, text) = match (platform, encoding) {
            (3, 1) | (3, 10) | (0, _) => {
                let units: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
                let rank = if platform == 3 && language == 0x409 { 3 } else { 1 };
                (rank, String::from_utf16_lossy(&units))
            }
            (1, 0) => (2, bytes.iter().map(|&b| if b < 128 { b as char } else { '?' }).collect()),
            _ => continue,
        };
        if best.get(&name_id).is_none_or(|(have, _)| rank > *have) {
            best.insert(name_id, (rank, text.trim().to_owned()));
        }
    }
    let get = |id| best.get(&id).map(|(_, s)| s.clone()).filter(|s| !s.is_empty());
    (get(16).or_else(|| get(1)), get(17).or_else(|| get(2)))
}

/// A file holding just `face`, loadable by the text engine (which reads the
/// first face of a file): the file itself for a single-face font, else a
/// copy of the face's tables under `cache_dir`, written once.
pub fn standalone_path(face: &Face, cache_dir: &Path) -> Option<PathBuf> {
    let mut file = File::open(&face.path).ok()?;
    let offsets = face_offsets(&mut file)?;
    if offsets.len() == 1 && face.index == 0 {
        return Some(face.path.clone());
    }
    let (version, tables) = tables(&mut file, *offsets.get(face.index as usize)?)?;
    let ext = if version == u32::from_be_bytes(*b"OTTO") { "otf" } else { "ttf" };
    let safe: String = format!("{}-{}", face.family, face.style)
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    let out = cache_dir.join(format!("{safe}.{ext}"));
    if out.is_file() {
        return Some(out);
    }

    let count = tables.len();
    let mut data = Vec::new();
    data.extend_from_slice(&version.to_be_bytes());
    let entry_selector = (usize::BITS - 1 - count.leading_zeros()) as u16;
    let search_range = (1u16 << entry_selector) * 16;
    data.extend_from_slice(&(count as u16).to_be_bytes());
    data.extend_from_slice(&search_range.to_be_bytes());
    data.extend_from_slice(&entry_selector.to_be_bytes());
    data.extend_from_slice(&((count as u16) * 16 - search_range).to_be_bytes());
    let mut body = Vec::new();
    let mut records = Vec::new();
    let mut offset = 12 + count * 16;
    let mut sorted = tables.clone();
    sorted.sort_by_key(|(tag, _, _)| *tag);
    for (tag, src, len) in sorted {
        let bytes = read_at(&mut file, src as u64, len as usize)?;
        let checksum = bytes
            .chunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                u32::from_be_bytes(w)
            })
            .fold(0u32, u32::wrapping_add);
        records.extend_from_slice(&tag);
        records.extend_from_slice(&checksum.to_be_bytes());
        records.extend_from_slice(&(offset as u32).to_be_bytes());
        records.extend_from_slice(&len.to_be_bytes());
        body.extend_from_slice(&bytes);
        while body.len() % 4 != 0 {
            body.push(0);
        }
        offset = 12 + count * 16 + body.len();
    }
    data.extend_from_slice(&records);
    data.extend_from_slice(&body);
    std::fs::create_dir_all(cache_dir).ok()?;
    let tmp = out.with_extension("tmp");
    std::fs::write(&tmp, &data).ok()?;
    std::fs::rename(&tmp, &out).ok()?;
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A font with one face whose name table says `family`/`style`.
    fn tiny_font(family: &str, style: &str, fixed: bool) -> Vec<u8> {
        let enc = |s: &str| s.encode_utf16().flat_map(|u| u.to_be_bytes()).collect::<Vec<u8>>();
        let (f, s) = (enc(family), enc(style));
        let mut name = Vec::new();
        name.extend_from_slice(&0u16.to_be_bytes());
        name.extend_from_slice(&2u16.to_be_bytes());
        name.extend_from_slice(&(6u16 + 24).to_be_bytes());
        for (id, off, len) in [(1u16, 0u16, f.len() as u16), (2, f.len() as u16, s.len() as u16)] {
            for v in [3u16, 1, 0x409, id, len, off] {
                name.extend_from_slice(&v.to_be_bytes());
            }
        }
        name.extend_from_slice(&f);
        name.extend_from_slice(&s);
        let mut post = vec![0u8; 32];
        post[12..16].copy_from_slice(&(fixed as u32).to_be_bytes());
        let mut out = Vec::new();
        out.extend_from_slice(&0x00010000u32.to_be_bytes());
        out.extend_from_slice(&2u16.to_be_bytes());
        out.extend_from_slice(&[0; 6]);
        let name_at = 12 + 32;
        let post_at = name_at + name.len();
        for (tag, at, len) in [(b"name", name_at, name.len()), (b"post", post_at, post.len())] {
            out.extend_from_slice(tag);
            out.extend_from_slice(&0u32.to_be_bytes());
            out.extend_from_slice(&(at as u32).to_be_bytes());
            out.extend_from_slice(&(len as u32).to_be_bytes());
        }
        out.extend_from_slice(&name);
        out.extend_from_slice(&post);
        out
    }

    #[test]
    fn a_font_file_reads_its_family_style_and_pitch() {
        let dir = std::env::temp_dir().join(format!("terminal-fonts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.ttf"), tiny_font("Test Mono", "Regular", true)).unwrap();
        std::fs::write(dir.join("b.otf"), tiny_font("Test Mono", "Bold", true)).unwrap();
        std::fs::write(dir.join("c.ttf"), tiny_font("Test Sans", "Regular", false)).unwrap();
        std::fs::write(dir.join("notes.txt"), "not a font").unwrap();
        let families = group(scan(&[dir.clone()]));
        let names: Vec<&str> = families.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["Test Mono", "Test Sans"], "monospace first");
        assert!(families[0].monospace && !families[1].monospace);
        assert_eq!(families[0].regular.style, "Regular");
        assert_eq!(families[0].bold.as_ref().map(|f| f.style.as_str()), Some("Bold"));
        // A single-face file is used as it is.
        assert_eq!(standalone_path(&families[0].regular, &dir), Some(dir.join("a.ttf")));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_face_is_copied_out_of_a_collection() {
        let dir = std::env::temp_dir().join(format!("terminal-ttc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b) = (tiny_font("Coll A", "Regular", false), tiny_font("Coll B", "Regular", true));
        // A collection of the two: faces keep their table offsets relative
        // to the file, so rebase b's tables after a.
        let header = 12 + 8;
        let mut ttc = Vec::new();
        ttc.extend_from_slice(b"ttcf");
        ttc.extend_from_slice(&0x00010000u32.to_be_bytes());
        ttc.extend_from_slice(&2u32.to_be_bytes());
        ttc.extend_from_slice(&(header as u32).to_be_bytes());
        ttc.extend_from_slice(&((header + a.len()) as u32).to_be_bytes());
        for (font, base) in [(&a, header), (&b, header + a.len())] {
            let mut f = font.clone();
            for i in 0..2 {
                let at = 12 + i * 16 + 8;
                let off = u32::from_be_bytes(f[at..at + 4].try_into().unwrap()) + base as u32;
                f[at..at + 4].copy_from_slice(&off.to_be_bytes());
            }
            ttc.extend_from_slice(&f);
        }
        std::fs::write(dir.join("both.ttc"), &ttc).unwrap();
        let faces = read_faces(&dir.join("both.ttc")).unwrap();
        assert_eq!(faces.iter().map(|f| f.family.as_str()).collect::<Vec<_>>(), ["Coll A", "Coll B"]);
        let cache = dir.join("cache");
        let out = standalone_path(&faces[1], &cache).expect("copied out");
        let copied = read_faces(&out).unwrap();
        assert_eq!(copied.len(), 1);
        assert_eq!(copied[0].family, "Coll B");
        assert!(copied[0].monospace);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_has_menlo_and_a_cjk_fallback() {
        FAMILIES.get_or_init(|| group(scan(&font_dirs())));
        let menlo = find("Menlo").expect("Menlo ships with macOS");
        assert!(menlo.monospace);
        if let Some(cjk) = auto_cjk() {
            let dir = std::env::temp_dir().join(format!("terminal-cjk-{}", std::process::id()));
            let path = standalone_path(&cjk.regular, &dir).expect("loadable");
            assert_eq!(read_faces(&path).unwrap()[0].family, cjk.name);
            std::fs::remove_dir_all(&dir).ok();
        }
    }
}
