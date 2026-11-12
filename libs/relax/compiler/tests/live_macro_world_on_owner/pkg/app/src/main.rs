#[derive(relax_live_world_count::Count)]
struct Thing;

fn value() -> u32 {
    1
}

fn main() {
    let t0 = std::time::Instant::now();
    loop {
        println!("out tick {} {}", Thing::count(), value());
        if std::path::Path::new("stop").exists() || t0.elapsed().as_secs() > 120 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}
