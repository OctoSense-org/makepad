// (makepad's HAL in the graph is what makes the run a hosted one: this program's main runs in an
// app domain; it opens no window of its own)
use makepad_widgets as _;

#[derive(relax_live_beside_count::Count)]
struct Thing;

/// The program's own statics, written once: every word must stay what the program wrote.
static mut MARKS: [u64; 1 << 19] = [0; 1 << 19];

struct Held(u64);
impl Drop for Held {
    fn drop(&mut self) {}
}
thread_local! {
    // drop glue: a runtime key, created on first use and kept in a static of this program
    static HELD: std::cell::RefCell<Held> = std::cell::RefCell::new(Held(0));
}

fn value() -> u32 {
    1
}

fn main() {
    let marks = unsafe { &mut *(&raw mut MARKS) };
    for m in marks.iter_mut() {
        *m = 0x1111_1111_1111_1111;
    }
    HELD.with(|h| h.borrow_mut().0 = 77);
    let t0 = std::time::Instant::now();
    loop {
        let changed = marks.iter().filter(|m| **m != 0x1111_1111_1111_1111).count();
        // (the app domain's id: 0 would be a program run in the unit's compile domain)
        println!("out tick {} {} marks-changed {} held {} domain {}", Thing::count(), value(), changed, HELD.with(|h| h.borrow().0), std::relax::domain());
        if std::path::Path::new("stop").exists() || t0.elapsed().as_secs() > 120 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}
