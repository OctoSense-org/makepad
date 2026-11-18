// A tail after a diverging statement still types an async body's output: rustc coerces the
// body's tail to the return type whether or not the code before it returns, so in
// `panic!(); Ok(())` the error type comes from where the output is used. Relax left the tail
// out once the body had diverged (E0283: cannot satisfy `?E: Sized`).
use std::future::Future;

struct Task<T>(Option<T>);

fn spawn<F: Future>(f: F) -> Task<F::Output> {
    drop(f);
    Task(None)
}

struct Owner {
    task: Option<Task<Result<(), String>>>,
}

fn with_job<F, Fut>(f: F) -> bool
where
    F: FnOnce(u8) -> Fut,
    Fut: Future<Output = Result<u8, String>>,
{
    drop(f(1));
    true
}

#[allow(unreachable_code)]
async fn opaque_after_panic() -> impl std::fmt::Debug {
    panic!("never polled");
    7u8
}

#[test]
fn a_tail_after_a_diverging_statement_types_the_output() {
    // the error type is known only from the field the task is stored in
    let task = spawn(async move {
        panic!("never polled");
        #[allow(unreachable_code)]
        Ok(())
    });
    let owner = Owner { task: Some(task) };
    assert!(owner.task.unwrap().0.is_none());
    // and from the bound on a closure's future
    assert!(with_job(|_| async {
        panic!("never polled");
        #[allow(unreachable_code)]
        Ok(3)
    }));
    // an async fn's opaque output is defined by such a tail
    drop(opaque_after_panic());
}
