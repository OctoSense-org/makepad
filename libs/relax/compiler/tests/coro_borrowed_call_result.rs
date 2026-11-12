//! A call's result that a later call borrows must live across a suspension the borrower lives
//! through, also when the result has no drop glue (nothing else reads its slot afterwards).
//! rustc_mir_transform/src/coroutine/layout.rs::locals_live_across_suspend_points saves a
//! local that is storage-live and maybe-borrowed at the suspension point.
//!
//! deadpool's `Pool::get` is the first case: `self.timeout_get(&self.timeouts()).await`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

struct YieldOnce(bool);
impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            Poll::Pending
        }
    }
}

struct Noop;
impl Wake for Noop {
    fn wake(self: Arc<Self>) {}
}

/// Overwrites the stack below the caller, where a poll's frame was.
#[inline(never)]
fn scribble(depth: u32) -> u64 {
    let a = std::hint::black_box([0xDEAD_BEEF_DEAD_BEEFu64; 96]);
    let mut s = 0u64;
    for v in a.iter() {
        s = s.wrapping_add(*v);
    }
    if depth > 0 {
        s = s.wrapping_add(scribble(depth - 1));
    }
    std::hint::black_box(s)
}

/// Polls at a different stack depth each time.
#[inline(never)]
fn poll_deeper<F: Future>(f: Pin<&mut F>, cx: &mut Context<'_>, depth: u32) -> Poll<F::Output> {
    let pad = std::hint::black_box([depth as u64; 24]);
    let r = if depth == 0 { f.poll(cx) } else { poll_deeper(f, cx, depth - 1) };
    std::hint::black_box(pad);
    r
}

fn run<F: Future>(f: F) -> F::Output {
    let mut fut = Box::pin(f);
    let waker = Waker::from(Arc::new(Noop));
    let mut cx = Context::from_waker(&waker);
    let mut polls = 0u32;
    loop {
        match poll_deeper(fut.as_mut(), &mut cx, polls) {
            Poll::Ready(v) => break v,
            Poll::Pending => {
                polls += 1;
                scribble(12);
            }
        }
    }
}

#[inline(never)]
fn four(x: u64) -> [u64; 4] {
    [x, x + 1, x + 2, x + 3]
}

#[inline(never)]
async fn read_after_yield(v: &[u64; 4]) -> u64 {
    YieldOnce(false).await;
    v[0] + v[3]
}

async fn temporary_argument() -> u64 {
    read_after_yield(&four(10)).await
}

async fn named_local() -> u64 {
    let t = four(10);
    read_after_yield(&t).await
}

async fn extended_temporary() -> u64 {
    let r = &four(10);
    read_after_yield(r).await
}

async fn iterator_into_temporary() -> u64 {
    let mut s = 0u64;
    for x in four(10).iter() {
        YieldOnce(false).await;
        s += *x;
    }
    s
}

struct Holder<'a>(&'a [u64; 4]);

#[inline(never)]
fn hold(v: &[u64; 4]) -> Holder<'_> {
    Holder(v)
}

async fn match_on_holder() -> u64 {
    match hold(&four(10)) {
        Holder(v) => {
            YieldOnce(false).await;
            v[0] + v[3]
        }
    }
}

#[test]
fn borrowed_call_result_lives_across_a_suspension() {
    assert_eq!(run(temporary_argument()), 23);
    assert_eq!(run(named_local()), 23);
    assert_eq!(run(extended_temporary()), 23);
    assert_eq!(run(iterator_into_temporary()), 46);
    assert_eq!(run(match_on_holder()), 23);
}

#[inline(never)]
fn sum_held(h: Holder<'_>, k: u64) -> u64 {
    h.0[0] + h.0[3] + k
}

#[inline(never)]
async fn sum_held_later(h: Holder<'_>, k: u64) -> u64 {
    YieldOnce(false).await;
    h.0[0] + h.0[3] + k
}

async fn borrower_awaited_in_a_later_statement() -> u64 {
    let t = four(10);
    let fut = read_after_yield(&t);
    fut.await
}

async fn local_borrowed_in_an_inner_block() -> u64 {
    let t = four(10);
    let h = {
        let k = 1u64;
        sum_held(hold(&t), k);
        hold(&t)
    };
    YieldOnce(false).await;
    h.0[0] + h.0[3]
}

/// The borrower is an operand already evaluated when a later operand's block runs its own
/// statements and suspends.
async fn later_operand_suspends() -> u64 {
    sum_held(hold(&four(10)), {
        let k = 1u64;
        YieldOnce(false).await;
        k
    })
}

async fn later_operand_has_statements() -> u64 {
    sum_held_later(hold(&four(10)), {
        let k = 1u64;
        k
    })
    .await
}

async fn made_on_one_path(c: bool) -> u64 {
    let n = if c {
        sum_held(hold(&four(10)), {
            YieldOnce(false).await;
            1
        })
    } else {
        YieldOnce(false).await;
        0
    };
    n + (c && sum_held(hold(&four(20)), { YieldOnce(false).await; 0 }) == 43) as u64
}

async fn early_exit(fail: bool) -> Result<u64, u64> {
    let v = sum_held(hold(&four(10)), {
        YieldOnce(false).await;
        if fail {
            return Err(7);
        }
        1
    });
    Ok(v)
}

#[derive(Clone, Copy)]
struct Wrap {
    inner: [u64; 4],
    tail: u64,
}

#[inline(never)]
fn wrap(x: u64) -> Wrap {
    Wrap { inner: four(x), tail: x }
}

async fn binding_takes_over_a_field() -> u64 {
    match wrap(10) {
        Wrap { inner, tail } => {
            let fut = read_after_yield(&inner);
            fut.await + tail
        }
    }
}

#[inline(never)]
fn hold_below(v: &[u64; 4], end: u64) -> Option<Holder<'_>> {
    if v[0] < end {
        Some(Holder(v))
    } else {
        None
    }
}

async fn scrutinee_of_a_while_let() -> u64 {
    let mut n = 0u64;
    let mut s = 0u64;
    while let Some(Holder(v)) = hold_below(&four(n), 3) {
        YieldOnce(false).await;
        s += v[3];
        n += 1;
    }
    s
}

async fn local_of_a_loop_body() -> u64 {
    let mut s = 0u64;
    for i in 0..3u64 {
        let t = four(i);
        let h = hold(&t);
        YieldOnce(false).await;
        s += h.0[3];
    }
    s
}

#[inline(never)]
async fn sum_iter(it: std::slice::Iter<'_, u64>) -> u64 {
    YieldOnce(false).await;
    it.copied().sum()
}

async fn autoref_receiver() -> u64 {
    sum_iter(four(10).iter()).await
}

#[inline(never)]
async unsafe fn read_raw(p: *const u64) -> u64 {
    YieldOnce(false).await;
    unsafe { *p + *p.add(3) }
}

async fn raw_pointer_into_temporary() -> u64 {
    unsafe { read_raw(four(10).as_ptr()).await }
}

struct Owned<'a> {
    refs: Vec<&'a [u64; 4]>,
    name: String,
}

#[inline(never)]
fn owned(v: &[u64; 4]) -> Owned<'_> {
    Owned { refs: vec![v], name: String::from("n") }
}

#[inline(never)]
fn sum_refs(r: Vec<&[u64; 4]>, k: u64) -> u64 {
    r[0][0] + r[0][3] + k
}

/// The field moved out of the temporary still borrows the call result.
async fn field_moved_out_of_a_temporary() -> u64 {
    sum_refs(owned(&four(10)).refs, {
        YieldOnce(false).await;
        1
    })
}

async fn other_field_of_the_temporary() -> u64 {
    owned(&four(10)).name.len() as u64 + read_after_yield(&four(10)).await
}

async fn ref_binding_of_a_scrutinee() -> u64 {
    match four(10) {
        ref v => {
            let fut = read_after_yield(v);
            fut.await
        }
    }
}

async fn scrutinee_borrowed_in_an_arm() -> u64 {
    match wrap(10) {
        w => {
            let fut = read_after_yield(&w.inner);
            fut.await + w.tail
        }
    }
}

#[inline(never)]
fn some_four(x: u64) -> Option<[u64; 4]> {
    Some(four(x))
}

async fn ref_binding_of_an_if_let() -> u64 {
    if let Some(ref v) = some_four(10) {
        let fut = read_after_yield(v);
        fut.await
    } else {
        0
    }
}

#[test]
fn borrowed_call_result_is_kept_to_the_end_of_its_storage() {
    assert_eq!(run(borrower_awaited_in_a_later_statement()), 23);
    assert_eq!(run(local_borrowed_in_an_inner_block()), 23);
    assert_eq!(run(later_operand_suspends()), 24);
    assert_eq!(run(later_operand_has_statements()), 24);
    assert_eq!(run(made_on_one_path(true)), 25);
    assert_eq!(run(made_on_one_path(false)), 0);
    assert_eq!(run(early_exit(false)), Ok(24));
    assert_eq!(run(early_exit(true)), Err(7));
    assert_eq!(run(binding_takes_over_a_field()), 33);
    assert_eq!(run(scrutinee_of_a_while_let()), 12);
    assert_eq!(run(local_of_a_loop_body()), 12);
    assert_eq!(run(autoref_receiver()), 46);
    assert_eq!(run(raw_pointer_into_temporary()), 23);
    assert_eq!(run(field_moved_out_of_a_temporary()), 24);
    assert_eq!(run(other_field_of_the_temporary()), 24);
    assert_eq!(run(ref_binding_of_a_scrutinee()), 23);
    assert_eq!(run(scrutinee_borrowed_in_an_arm()), 33);
    assert_eq!(run(ref_binding_of_an_if_let()), 23);
}

#[derive(Clone, Copy)]
struct Timeouts {
    wait: Option<Duration>,
    create: Option<Duration>,
    recycle: Option<Duration>,
}

struct Pool {
    timeouts: Timeouts,
}

impl Pool {
    #[inline(never)]
    fn timeouts(&self) -> Timeouts {
        self.timeouts
    }
    async fn get(&self) -> (bool, bool, bool) {
        self.timeout_get(&self.timeouts()).await
    }
    async fn timeout_get(&self, timeouts: &Timeouts) -> (bool, bool, bool) {
        let wait = timeouts.wait.is_none();
        YieldOnce(false).await;
        let create = timeouts.create.is_none();
        (wait, create, self.try_recycle(timeouts).await)
    }
    #[inline]
    async fn try_recycle(&self, timeouts: &Timeouts) -> bool {
        YieldOnce(false).await;
        apply(Some(1), timeouts.recycle).await
    }
}

async fn apply(runtime: Option<u8>, duration: Option<Duration>) -> bool {
    match (runtime, duration) {
        (_, None) => true,
        (Some(_), Some(d)) => d.as_secs() == u64::MAX,
        (None, Some(_)) => false,
    }
}

#[test]
fn pool_timeouts_survive_waiting_for_a_slot() {
    let unset = Pool { timeouts: Timeouts { wait: None, create: None, recycle: None } };
    assert_eq!(run(unset.get()), (true, true, true));
    let set = Pool {
        timeouts: Timeouts { wait: Some(Duration::new(3, 7)), create: Some(Duration::new(5, 11)), recycle: Some(Duration::new(u64::MAX, 13)) },
    };
    assert_eq!(run(set.get()), (false, false, true));
}
