//! The shared GPU rig runs real gpu_jobs Steps: the deterministic exclusive scan, byte-identical
//! with a CPU scan whatever the job size, a counted output, and host steps behind empty passes.
#[path = "support/gpu_rig.rs"]
mod rig;
use makepad_sdf_engine::gpu_jobs::{Counts, Elements, Finished, JobClass, JobError, Output, Pool, ScanKernels, Step, SCAN_BLOCK};

#[test]
fn the_exclusive_scan_matches_the_cpu() {
    rig::with_gpu(|cx| {
        let scan = ScanKernels::new(cx).unwrap();
        // nothing; one block in jobs of 3 elements; three blocks and their sums in jobs of 2
        for (n, forced) in [(0u32, None), (SCAN_BLOCK, Some(3)), (SCAN_BLOCK * 2 + 1, Some(2))] {
            let values: Vec<u32> = (0..n).map(|i| i.wrapping_mul(2654435761) >> 22).collect();
            let mut expected = Vec::with_capacity(n as usize);
            let mut sum = 0u32;
            for v in &values {
                expected.push(sum);
                sum += v;
            }
            rig::force_elements(cx, forced);
            let mut pool = Pool::new(32);
            let input = pool.buffer("scan input", n.max(1) as u64 * 4);
            if n > 0 { pool.upload(input, 0, rig::bytes(&values)); }
            let output = pool.readback("scan output", n.max(1) as u64 * 4);
            let total = pool.claim("scan total", 1).unwrap();
            let mut steps = scan.exclusive(&mut pool, input, output, n, total);
            if n > 0 { steps.push(Output::fixed(output, n as u64 * 4)); }
            let (counts, output) = match rig::run_steps(cx, pool, steps, JobClass::Visible) {
                Finished::Done { counts, output, .. } => (counts, output),
                Finished::Failed { error, .. } => panic!("scan {n} (jobs {forced:?}): {error}"),
            };
            if n > 0 {
                let got = rig::words(&output.expect("the scan's output"));
                assert_eq!(got.len(), expected.len(), "scan {n} (jobs {forced:?}) length");
                if let Some(i) = (0..expected.len()).find(|&i| got[i] != expected[i]) {
                    panic!("scan {n} (jobs {forced:?}) value {i}: {} != {}", got[i], expected[i]);
                }
            }
            assert_eq!(counts.word(total), sum, "scan {n} (jobs {forced:?}) total");
        }
        rig::force_elements(cx, None);
    });
}

/// A counted output is cut to its count word, read in the same submission as the passes that
/// wrote it; a count past its capacity answers Overflow, naming the buffer.
#[test]
fn a_counted_output_is_cut_to_its_count() {
    rig::with_gpu(|cx| {
        let scan = ScanKernels::new(cx).unwrap();
        // ones: the scan's total counts the values, its output holds 0..n
        let n = SCAN_BLOCK * 3 + 5;
        for cap in [n + 100, n - 1] {
            let mut pool = Pool::new(32);
            let input = pool.buffer("ones", n as u64 * 4);
            pool.upload(input, 0, rig::bytes(&vec![1u32; n as usize]));
            let output = pool.readback("scan output", (n + 100) as u64 * 4);
            let total = pool.claim("scan total", 1).unwrap();
            let mut steps = scan.exclusive(&mut pool, input, output, n, total);
            steps.push(Output::counted(output, total, 4, cap));
            match (cap >= n, rig::run_steps(cx, pool, steps, JobClass::Visible)) {
                (true, Finished::Done { output, .. }) => {
                    let got = rig::words(&output.expect("the counted output"));
                    assert_eq!(got, (0..n).collect::<Vec<u32>>(), "the records up to the count");
                }
                (false, Finished::Failed { error: JobError::Overflow { buffer, count, cap: c }, .. }) => {
                    assert_eq!((buffer, count, c), ("scan output", n, cap), "the overflow names its buffer, count and capacity");
                }
                (fits, Finished::Done { .. }) => panic!("capacity {cap} (fits {fits}): answered Done"),
                (fits, Finished::Failed { error, .. }) => panic!("capacity {cap} (fits {fits}): {error}"),
            }
        }
    });
}

/// A pass with no elements before a host step never stalls its item: nothing carries it to the
/// GPU, so the host step after it runs in the same pump (here twice in a row, which the rig
/// fails as an item stuck with nothing on the GPU).
#[test]
fn an_empty_pass_before_a_host_step_runs_on() {
    rig::with_gpu(|cx| {
        let scan = ScanKernels::new(cx).unwrap();
        let mut pool = Pool::new(32);
        let input = pool.buffer("scan input", 4);
        let total = pool.claim("scan total", 1).unwrap();
        let mut empty = || {
            let mut steps = scan.exclusive(&mut pool, input, input, 0, total);
            if let Some(Step::Pass(p)) = steps.first_mut() { p.elements = Elements::Fixed(0); }
            steps
        };
        let (first, second) = (empty(), empty());
        let last = Step::Host(Box::new(|_: &Counts, _: &mut Pool| Ok(Vec::new())));
        let then = Step::Host(Box::new(move |_: &Counts, _: &mut Pool| Ok(second.into_iter().chain([last]).collect())));
        let steps = vec![Step::Host(Box::new(move |_: &Counts, _: &mut Pool| Ok(first.into_iter().chain([then]).collect())))];
        match rig::run_steps(cx, pool, steps, JobClass::Visible) {
            Finished::Done { .. } => {}
            Finished::Failed { error, .. } => panic!("empty passes between host steps: {error}"),
        }
    });
}
