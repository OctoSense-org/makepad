//! Robustness tests: the terminal's inputs from outside the process — font
//! files, the process table, the control socket, the settings file, the
//! byte stream a program writes — fed with random, truncated and corrupt
//! data. A panic here is a crashed terminal (and, hosted in-process, a
//! crashed OctoSense shell). Deterministic: a failing seed reproduces.

use crate::control;
use crate::fonts;
use crate::panes::{self, Dir, Node};
use crate::procinfo;
use crate::settings::Settings;
use crate::term::stream::Stream;
use crate::term::terminal::Terminal;
use makepad_widgets::{dvec2, Rect};
use std::path::PathBuf;

/// xorshift64*: deterministic noise, no dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("terminal-robust-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Real font files to corrupt: the bundled ones, and a system collection.
fn sample_fonts() -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for path in [
        manifest.join("../../widgets/resources/jetbrains_mono_variable.ttf"),
        manifest.join("../../widgets/resources/NotoSans-Regular.ttf"),
        PathBuf::from("/System/Library/Fonts/Menlo.ttc"),
    ] {
        if let Ok(bytes) = std::fs::read(&path) {
            out.push(bytes);
        }
    }
    assert!(!out.is_empty(), "no sample fonts found");
    out
}

struct Sink;

impl ttf_parser::OutlineBuilder for Sink {
    fn move_to(&mut self, _: f32, _: f32) {}
    fn line_to(&mut self, _: f32, _: f32) {}
    fn quad_to(&mut self, _: f32, _: f32, _: f32, _: f32) {}
    fn curve_to(&mut self, _: f32, _: f32, _: f32, _: f32, _: f32, _: f32) {}
    fn close(&mut self) {}
}

/// What drawing text does with a face: map characters, outline glyphs,
/// read bitmap glyphs and metrics, and shape a line.
fn render_with(data: &[u8], index: u32) {
    use makepad_widgets::makepad_draw::text::font_face::FontFace;
    use makepad_widgets::makepad_platform::SharedBytes;
    let Some(face) = FontFace::from_data_and_index(SharedBytes::from_owned(std::rc::Rc::new(data.to_vec())), index) else {
        return;
    };
    face.with_ttf_parser_face(|f| {
        let _ = (f.units_per_em(), f.ascender(), f.descender(), f.line_gap(), f.number_of_glyphs());
        let mut ids: Vec<ttf_parser::GlyphId> =
            "AaM0 \u{4e2d}\u{e4}\u{2192}\u{2502}\u{2588}\u{1F600}".chars().filter_map(|c| f.glyph_index(c)).collect();
        let n = f.number_of_glyphs();
        ids.extend((0..n.min(48)).map(ttf_parser::GlyphId));
        ids.push(ttf_parser::GlyphId(n.saturating_sub(1)));
        ids.push(ttf_parser::GlyphId(u16::MAX));
        for id in ids {
            let _ = f.outline_glyph(id, &mut Sink);
            let _ = f.glyph_hor_advance(id);
            let _ = f.glyph_bounding_box(id);
            let _ = f.glyph_raster_image(id, 64);
            let _ = f.glyph_svg_image(id);
        }
    });
    face.with_rustybuzz_face(|f| {
        let mut buffer = rustybuzz::UnicodeBuffer::new();
        buffer.push_str("Hello \u{4e2d}\u{6587} \u{2192} fi \u{e4} \u{1F600}");
        let _ = rustybuzz::shape(f, &[], buffer);
    });
}

/// Every font code path, on one file: scanning, copying a face out, the
/// engine's own check, and drawing with whatever it accepts. None may
/// panic, whatever the bytes.
fn exercise_font_file(path: &std::path::Path, cache: &std::path::Path) {
    if let Some(faces) = fonts::read_faces(path) {
        for face in faces.iter().take(4) {
            let _ = fonts::face_loadable(face);
            let _ = fonts::standalone_path(face, cache);
        }
        let _ = fonts::group(faces);
    }
    let _ = fonts::loadable(path);
    if let Ok(data) = std::fs::read(path) {
        for index in 0..3 {
            render_with(&data, index);
        }
    }
}

#[test]
fn corrupt_font_files_never_panic() {
    let dir = scratch("fonts");
    let cache = dir.join("cache");
    let samples = sample_fonts();
    let seed: u64 = std::env::var("TERMINAL_FUZZ_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(0x5eed_f0f0);
    let mut rng = Rng::new(seed);
    let rounds: usize = std::env::var("TERMINAL_FUZZ_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    for round in 0..rounds {
        let base = &samples[round % samples.len()];
        let mut bytes = match round % 5 {
            // Truncated anywhere, including inside the headers.
            0 => base[..rng.below(base.len().min(4096) + 1)].to_vec(),
            1 => base[..rng.below(base.len()) + 1].to_vec(),
            // Bit flips, concentrated where the offsets and lengths live.
            2 | 3 => {
                let mut b = base.clone();
                for _ in 0..1 + rng.below(40) {
                    let at = if round % 2 == 0 { rng.below(b.len().min(1024)) } else { rng.below(b.len()) };
                    b[at] ^= 1 << rng.below(8);
                }
                b
            }
            // Pure noise, sometimes behind a valid-looking tag.
            _ => {
                let n = rng.below(2048);
                let mut b = rng.bytes(n);
                if rng.below(2) == 0 && b.len() >= 12 {
                    let tag: &[u8] = [b"ttcf" as &[u8], &[0, 1, 0, 0], b"OTTO"][rng.below(3)];
                    b[..4].copy_from_slice(tag);
                }
                b
            }
        };
        if bytes.is_empty() {
            bytes.push(0);
        }
        let path = dir.join(format!("f{round}.ttf"));
        std::fs::write(&path, &bytes).unwrap();
        exercise_font_file(&path, &cache);
        let _ = std::fs::remove_file(&path);
    }
    // Directory entries that are not fonts, or unreadable, are skipped.
    std::fs::create_dir_all(dir.join("sub.ttf")).unwrap();
    std::fs::write(dir.join("empty.otf"), b"").unwrap();
    let _ = fonts::scan(&[dir.clone(), dir.join("missing")]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_collection_whose_offsets_point_anywhere_is_safe() {
    let dir = scratch("ttc");
    let mut rng = Rng::new(7);
    for round in 0..200 {
        // A ttcf header claiming many faces at random offsets, then noise.
        let mut b = Vec::new();
        b.extend_from_slice(b"ttcf");
        b.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        let faces = [0u32, 1, 3, 255, 256, u32::MAX][rng.below(6)];
        b.extend_from_slice(&faces.to_be_bytes());
        for _ in 0..faces.min(8) {
            let off = [0u32, 12, 16, u32::MAX, rng.next() as u32][rng.below(5)];
            b.extend_from_slice(&off.to_be_bytes());
        }
        let noise = rng.below(512);
        b.extend(rng.bytes(noise));
        let path = dir.join(format!("c{round}.ttc"));
        std::fs::write(&path, &b).unwrap();
        exercise_font_file(&path, &dir.join("cache"));
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn process_queries_take_any_pid() {
    for pid in [i32::MIN, -1, 0, 1, i32::MAX, 999_999] {
        let _ = procinfo::cwd(pid);
        let _ = procinfo::name(pid);
    }
    // A child that has exited and been reaped: its pid names nothing now
    // (or, recycled, something else) — never a crash.
    let mut child = std::process::Command::new("/bin/sh").arg("-c").arg("exit 0").spawn().unwrap();
    let pid = child.id() as i32;
    child.wait().unwrap();
    let _ = procinfo::cwd(pid);
    let _ = procinfo::name(pid);
    // A zombie (exited, not yet reaped).
    let child = std::process::Command::new("/bin/sh").arg("-c").arg("exit 0").spawn().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let _ = procinfo::cwd(child.id() as i32);
    let _ = procinfo::name(child.id() as i32);
    drop(child);
}

#[test]
fn the_control_socket_takes_any_line() {
    let _serial = control::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    control::set_enabled_for_tests(true);
    let mut rng = Rng::new(0xc0de);
    let mut lines: Vec<String> = vec![
        String::new(),
        "{".into(),
        "[]".into(),
        "null".into(),
        r#"{"cmd":null}"#.into(),
        r#"{"cmd":"read","pane":null}"#.into(),
        r#"{"cmd":"read","pane":"."}"#.into(),
        r#"{"cmd":"read","pane":"-1.-1"}"#.into(),
        r#"{"cmd":"read","pane":"99999999999999999999.1"}"#.into(),
        r#"{"cmd":"read","pane":"1","lines":-5}"#.into(),
        r#"{"cmd":"read","pane":"1","lines":1e308}"#.into(),
        r#"{"cmd":"prompt","pane":"1","text":"\u0000\u001b[31m"}"#.into(),
        format!(r#"{{"cmd":"list","x":"{}"}}"#, "a".repeat(100_000)),
        "[".repeat(10_000),
        "{\"a\":".repeat(5_000),
    ];
    for _ in 0..300 {
        let len = rng.below(200);
        lines.push(String::from_utf8_lossy(&rng.bytes(len)).into_owned());
    }
    for line in &lines {
        let reply = control::answer(line);
        // Every reply is an object saying ok or not.
        assert!(reply.get("ok").is_some(), "{line:.60?} -> {reply:?}");
    }
}

#[test]
fn any_settings_text_parses() {
    let mut rng = Rng::new(42);
    let keys = [
        "theme", "font-family", "cjk-font", "font-size", "line-height", "background-opacity", "cursor-shape",
        "term", "scrollback-lines", "shell", "profile", "tab-bar", "external-control",
    ];
    let values = ["", "=", "nan", "inf", "-inf", "-1", "1e400", "18446744073709551616", "\u{0}", "a\u{202e}b", "\n"];
    for _ in 0..2000 {
        let mut text = String::new();
        for _ in 0..rng.below(20) {
            match rng.below(3) {
                0 => text.push_str(&format!("{} = {}\n", keys[rng.below(keys.len())], values[rng.below(values.len())])),
                1 => {
                    let n = rng.below(40);
                    text.push_str(&String::from_utf8_lossy(&rng.bytes(n)))
                }
                _ => text.push_str(&format!("{}={}\n", keys[rng.below(keys.len())], rng.next())),
            }
        }
        let s = Settings::parse(&text);
        assert!(s.font_size.is_finite() && s.line_height.is_finite());
        // What parses writes back and parses to the same thing.
        assert_eq!(Settings::parse(&s.to_text()), s);
    }
}

#[test]
fn random_split_and_close_keep_the_pane_tree_whole() {
    let mut rng = Rng::new(99);
    let area = Rect { pos: dvec2(10.0, 20.0), size: dvec2(1200.0, 800.0) };
    for _ in 0..200 {
        let mut tree = Node::Leaf(1);
        let mut next = 2;
        for _ in 0..rng.below(40) {
            let leaves = tree.leaves();
            let at = leaves[rng.below(leaves.len())];
            if rng.below(3) == 0 && leaves.len() > 1 {
                tree = tree.remove(at).unwrap();
            } else {
                assert!(tree.split(at, next, rng.below(2) == 0));
                next += 1;
            }
            let leaves = tree.leaves();
            let mut unique = leaves.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), leaves.len(), "a pane twice");
            let layout = tree.layout(area);
            assert_eq!(layout.len(), leaves.len());
            for (i, (_, a)) in layout.iter().enumerate() {
                assert!(a.size.x >= 0.0 && a.size.y >= 0.0);
                assert!(a.pos.x >= area.pos.x - 0.5 && a.pos.y >= area.pos.y - 0.5);
                assert!(a.pos.x + a.size.x <= area.pos.x + area.size.x + 0.5);
                for (_, b) in layout.iter().skip(i + 1) {
                    let overlap_x = (a.pos.x + a.size.x).min(b.pos.x + b.size.x) - a.pos.x.max(b.pos.x);
                    let overlap_y = (a.pos.y + a.size.y).min(b.pos.y + b.size.y) - a.pos.y.max(b.pos.y);
                    assert!(overlap_x <= 0.5 || overlap_y <= 0.5, "panes overlap");
                }
            }
            for id in &leaves {
                for dir in [Dir::Left, Dir::Right, Dir::Up, Dir::Down] {
                    if let Some(n) = panes::neighbour(&layout, *id, dir) {
                        assert!(leaves.contains(&n) && n != *id);
                    }
                }
            }
            for d in tree.dividers(area) {
                let mut t = tree.clone();
                t.set_ratio(&d.path, panes::ratio_at(&d, (rng.next() % 3000) as f64 - 1000.0));
                assert_eq!(t.layout(area).len(), leaves.len());
            }
        }
    }
}

#[test]
fn the_emulator_takes_any_byte_stream() {
    let mut rng = Rng::new(0xfeed);
    // Fragments that reach deep parser states: CSI with many and huge
    // params, OSC/DCS/APC strings, charset designations, wide and
    // combining characters, the kitty keyboard and graphics escapes.
    let pieces: &[&[u8]] = &[
        b"\x1b[", b"\x1b]", b"\x1bP", b"\x1b_", b"\x1b^", b"\x1bX", b"\x9b", b"\x9d", b"\x90", b"\x07", b"\x1b\\",
        b";", b":", b"?", b">", b"<", b"=", b"999999999999", b"65535", b"0", b"1", b"m", b"H", b"J", b"K", b"r",
        b"h", b"l", b"q", b"u", b"t", b"S", b"T", b"L", b"M", b"@", b"P", b"X", b"b", b"c", b"n", b"g", b"z",
        b"\x1b(0", b"\x1b)B", b"\x1b#8", b"\x1bD", b"\x1bM", b"\x1bE", b"\x1b7", b"\x1b8", b"\x1bc",
        "中文".as_bytes(), "e\u{301}".as_bytes(), "\u{1F600}".as_bytes(), "\u{200d}".as_bytes(), "\u{FE0F}".as_bytes(),
        b"\r\n", b"\t", b"\x08", b"\x7f", b"\xff\xfe", b"\xc3", b"\xe4\xb8",
        b"0;title\x07", b"52;c;aGVsbG8=\x07", b"7;file:///tmp\x07", b"8;;http://x\x07", b"133;A\x07",
        b"Gf=100,a=T;AAAA\x1b\\", b"1337;File=:AAAA\x07",
    ];
    for seed in 0..60 {
        let (cols, rows) = (1 + rng.below(200), 1 + rng.below(80));
        let mut terminal = Terminal::with_scrollback(cols, rows, rng.below(2000));
        let mut stream = Stream::new();
        for _ in 0..400 {
            let chunk: Vec<u8> = if rng.below(3) == 0 {
                let n = rng.below(256);
                rng.bytes(n)
            } else {
                (0..1 + rng.below(12)).flat_map(|_| pieces[rng.below(pieces.len())].iter().copied()).collect()
            };
            stream.process(&chunk, &mut terminal);
            if rng.below(40) == 0 {
                terminal.resize(1 + rng.below(250), 1 + rng.below(90));
            }
            if rng.below(60) == 0 {
                terminal.set_scrollback(rng.below(500));
            }
            let _ = terminal.take_events();
        }
        let screen = terminal.screen();
        assert!(screen.cursor.x <= screen.cols && screen.cursor.y < screen.rows.max(1), "seed {seed}: cursor off the grid");
        for row in 0..screen.total_rows() {
            let _ = screen.row_virtual(row).map(|r| r.text());
        }
    }
}
