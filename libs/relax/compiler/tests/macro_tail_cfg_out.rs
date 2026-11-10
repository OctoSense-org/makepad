// A macro call that ends its block without `;` is the block's trailing expression statement:
// rustc expands it as an optional expression (rustc_expand flat_map_stmt, AstFragmentKind::OptExpr),
// so a `#[cfg]` on the expansion that is false removes it and the block's value is `()`.
// hyper 1.11's `trace!` (src/trace.rs) is `#[cfg(feature = "tracing")] { tracing::trace!(..); }`
// and proto/h1/conn.rs calls it as a match arm's tail: `=> { trace!("body drained") }`.
#![allow(unused_macros, unused_braces)]
use std::cell::RefCell;

thread_local! { static LOG: RefCell<Vec<String>> = Default::default(); }
fn log(s: String) {
    LOG.with(|l| l.borrow_mut().push(s));
}

// hyper's macros, as written there (no `tracing` crate exists here: the block is never resolved)
macro_rules! trace {
    ($($arg:tt)*) => {
        #[cfg(feature = "tracing")]
        {
            tracing::trace!($($arg)+);
        }
    }
}
macro_rules! debug {
    ($($arg:tt)+) => {
        #[cfg(feature = "tracing")]
        {
            tracing::debug!($($arg)+);
        }
    }
}
macro_rules! trace_span {
    ($($arg:tt)*) => {
        {
            #[cfg(feature = "tracing")]
            {
                let _span = tracing::trace_span!($($arg)+);
                _span.entered()
            }
        }
    }
}
macro_rules! kept {
    ($($arg:tt)*) => {
        #[cfg(all())]
        {
            log(format!($($arg)+));
        }
    }
}
macro_rules! gone_value { () => { #[cfg(any())] 5 } }
macro_rules! kept_value { () => { #[cfg(all())] { 5 } } }
macro_rules! allowed_value { () => { #[allow(unused_braces)] { 6 } } }
macro_rules! through { ($m:ident) => { $m!() } }
macro_rules! through_trace { ($($arg:tt)*) => { trace!($($arg)*) } }

enum Reading { Init, KeepAlive, Body }

struct Conn { reading: Reading, closed: bool }

impl Conn {
    // hyper-1.11.1/src/proto/h1/conn.rs:869
    fn poll_drain_or_close_read(&mut self) {
        match self.reading {
            Reading::Init | Reading::KeepAlive => {
                trace!("body drained")
            }
            _ => self.close_read(),
        }
    }
    fn close_read(&mut self) {
        self.closed = true;
    }
}

fn fn_tail() {
    debug!("fn tail {}", 1)
}
fn stmt() -> u8 {
    trace!("statement");
    let _entered = trace_span!("span");
    7
}
fn if_tails(x: u8) {
    if x == 0 {
        trace!("zero")
    } else if x == 1 {
        kept!("one")
    } else {
        through_trace!("more {}", x)
    }
}
fn arm_kept(x: u8) {
    match x {
        2 => {
            kept!("arm {}", x)
        }
        _ => {
            #[cfg(any())]
            {
                log("written cfg'd out".into());
            }
        }
    }
}
fn gone() {
    gone_value!()
}
fn gone_through() {
    through!(gone_value)
}
fn kept_five() -> i32 {
    kept_value!()
}
fn kept_through() -> i32 {
    through!(kept_value)
}
fn six() -> i32 {
    allowed_value!()
}

#[test]
fn a_trailing_macro_whose_expansion_is_cfgd_out_leaves_unit() {
    for reading in [Reading::Init, Reading::KeepAlive] {
        let mut c = Conn { reading, closed: false };
        c.poll_drain_or_close_read();
        assert!(!c.closed);
    }
    let mut c = Conn { reading: Reading::Body, closed: false };
    c.poll_drain_or_close_read();
    assert!(c.closed);
    fn_tail();
    assert_eq!(stmt(), 7);
    for x in 0..4 {
        if_tails(x);
        arm_kept(x);
    }
    gone();
    gone_through();
    assert_eq!((kept_five(), kept_through(), six()), (5, 5, 6));
    assert_eq!(LOG.with(|l| l.borrow().clone()), ["one", "arm 2"]);
}
