//! Regression tests for OctoSense-org/OctoSense#492: an
//! `if … { … for … } else { … }` whose true branch contains a `for` loop
//! left the script VM stack unbalanced; the next call in the same body
//! trapped with `pop_stack_value on empty stack`, and when the body was an
//! `on_render` handler the whole output was silently discarded (the view
//! rendered nothing).
//!
//! Root cause: the value-handoff contract of conditional jumps. Every arm of
//! an if/elif/else — and of try/catch and match, which desugar onto the same
//! IF_TEST/IF_ELSE joins — must hand the join exactly one value, because the
//! statement-end `set_pop_to_me` following the construct lands its pop on the
//! join (a standalone POP_TO_ME at the jump target) or just past it. Arms
//! whose tail produced no value — `for`/`while` (FOR_END), `let`, `break`/
//! `continue`, an empty body, or a `;` tail whose commit already ran inside
//! the arm — handed nothing, so the join's pop read through the base of the
//! stack. The fix stands in a NIL for those arms (parser `close_branch` and
//! the try-block close paths).
//!
//! Most tests evaluate a helper `f()` and then call `next_call(f())`: the
//! crash used to surface on that *next* call, possibly far from the helper
//! that corrupted the stack.

use makepad_script::*;

fn test_vm() -> ScriptVm<'static> {
    let host = Box::leak(Box::new(ScriptVmHost::new(0i32, ())));
    let mut vm = ScriptVm {
        host,
        bx: Box::new(ScriptVmBase::new()),
    };
    vm.bx.captured_errors = Some(Vec::new());
    vm
}

fn eval_str(vm: &mut ScriptVm, name: &str, code: &str) -> ScriptValue {
    // Bounded: a regression must fail the test, not wedge the suite.
    vm.with_instruction_limit(500_000, |vm| {
        vm.eval(ScriptMod {
            cargo_manifest_path: String::new(),
            module_path: String::new(),
            file: format!("if_for_else_stack_{name}"),
            line: 0,
            column: 0,
            code: code.to_string(),
            values: vec![],
        })
    })
}

/// Run `body` as the body of `f()`, then call `next_call(f())` — the shape
/// the bug surfaced in. Returns `f() + 1`; asserts a clean run (no captured
/// errors) and a numeric result.
fn run_body(name: &str, body: &str) -> f64 {
    let vm = &mut test_vm();
    let code =
        format!("fn f(){{\n{body}\n}}\nfn next_call(x){{ x + 1 }}\nlet a = f()\nnext_call(a)");
    let v = eval_str(vm, name, &code);
    let errors = vm.take_errors();
    assert!(errors.is_empty(), "{name}: unexpected errors: {errors:?}");
    v.as_number()
        .unwrap_or_else(|| panic!("{name}: expected a number, got {v:?}"))
}

// ---------------------------------------------------------------------------
// The exact shape from the issue
// ---------------------------------------------------------------------------

/// The issue's minimal repro: the helper computes fine; the trap surfaced on
/// the next call in the caller.
#[test]
fn issue_492_exact_shape() {
    let vm = &mut test_vm();
    let v = eval_str(
        vm,
        "issue492",
        r#"
fn bad(){
    let total = 0
    if total >= 0 {
        for j in 2 { total += j }
    } else { total = 100 }
    total
}
fn next_call(x){ x + 1 }
let a = bad()
next_call(a)
"#,
    );
    let errors = vm.take_errors();
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    assert_eq!(v.as_number(), Some(2.0), "got {v:?}");
}

/// Removing the `else` made the crash go away before the fix; both stay
/// working.
#[test]
fn without_else_still_works() {
    assert_eq!(
        run_body(
            "no_else",
            "    let t = 0\n    if t >= 0 { for j in 2 { t += j } }\n    t"
        ),
        2.0
    );
}

/// The untaken-true-branch variant (the else branch runs) stayed working.
#[test]
fn untaken_true_branch_runs_the_else() {
    assert_eq!(
        run_body(
            "untaken",
            "    let t = 100\n    if t >= 1000 { for j in 2 { t += j } } else { t = 100 }\n    t"
        ),
        101.0
    );
}

/// A statement after the loop inside the true branch kept the arm
/// value-ful, so this shape worked before the fix and must keep working.
#[test]
fn statement_after_the_loop_in_the_true_branch() {
    assert_eq!(
        run_body(
            "after_loop",
            "    let t = 0\n    if t >= 0 { for j in 2 { t += j }\n        t += 10 } else { t = 100 }\n    t"
        ),
        12.0
    );
}

// ---------------------------------------------------------------------------
// The value-less-tail family: every shape used to trap (or, in expression
// contexts, drop the `let` binding); each hand-off now stands in a NIL
// ---------------------------------------------------------------------------

#[test]
fn value_less_tails_hand_the_join_a_value() {
    for (name, body, expect) in [
        (
            "for_tail",
            "    let t = 0\n    if t >= 0 { for j in 2 { t += j } } else { t = 100 }\n    t",
            2.0,
        ),
        (
            "while_tail",
            "    let t = 0\n    if t >= 0 { while t < 2 { t += 1 } } else { t = 9 }\n    t",
            3.0,
        ),
        (
            "let_tail",
            "    let t = 0\n    if t >= 0 { let q = 1 } else { t = 2 }\n    t",
            1.0,
        ),
        (
            "semicolon_tail",
            "    let t = 0\n    if t >= 0 { t += 1; } else { t = 2 }\n    t",
            2.0,
        ),
        (
            "empty_branch",
            "    let t = 0\n    if t >= 0 { } else { t = 2 }\n    t",
            1.0,
        ),
        (
            "nested_if_tail",
            "    let t = 0\n    if t >= 0 { if t >= 0 { t += 1 } else { t = 2 } } else { t = 3 }\n    t",
            2.0,
        ),
        (
            "both_branches_for",
            "    let t = 0\n    if t >= 0 { for j in 2 { t += j } } else { for j in 2 { t += j } }\n    t",
            2.0,
        ),
        (
            "no_else_mid_statement",
            "    let t = 0\n    if t >= 0 { for j in 2 { t += j } }\n    let q = 1\n    t + q",
            3.0,
        ),
        (
            "no_else_semicolon_tail",
            "    let t = 0\n    if t >= 0 { t += 1; }\n    t",
            2.0,
        ),
        (
            "plain_tail_control",
            "    let t = 0\n    if t >= 0 { t += 1 } else { t = 2 }\n    t",
            2.0,
        ),
    ] {
        assert_eq!(run_body(name, body), expect, "{name}");
    }
}

// ---------------------------------------------------------------------------
// try/catch, match and elif desugar onto the same joins
// ---------------------------------------------------------------------------

#[test]
fn try_match_and_elif_hand_the_join_a_value() {
    for (name, body, expect) in [
        (
            "try_for_tail",
            "    let t = 0\n    try { for j in 2 { t += j } } catch { t = 5 }\n    t",
            2.0,
        ),
        (
            "try_for_then_statement",
            "    let t = 0\n    try { for j in 2 { t += j } } catch { t = 5 }\n    let q = 1\n    t + q",
            3.0,
        ),
        (
            "try_semicolon_tail",
            "    let t = 0\n    try { t += 1; } catch { t = 5 }\n    t",
            2.0,
        ),
        (
            "catch_for_not_taken",
            "    let t = 0\n    try { t += 1 } catch { for j in 2 { t += j } }\n    t",
            2.0,
        ),
        (
            "catch_for_taken",
            "    let t = 0\n    try { t = undefined_thing.x } catch { for j in 2 { t += j } }\n    t",
            2.0,
        ),
        (
            "match_for_tail",
            "    let t = 0\n    match 1 {\n        1 => { for j in 2 { t += j } }\n        _ => { t = 5 }\n    }\n    t",
            2.0,
        ),
        (
            "match_semicolon_tail",
            "    let t = 0\n    match 1 {\n        1 => { t += 1; }\n        _ => { t = 5 }\n    }\n    t",
            2.0,
        ),
        (
            "match_single_arm_for",
            "    let t = 0\n    match 1 {\n        1 => { for j in 2 { t += j } }\n    }\n    t",
            2.0,
        ),
        (
            "elif_for_tail",
            "    let t = 0\n    if t >= 3 { t = 1 } elif t >= 0 { for j in 2 { t += j } } else { t = 5 }\n    t",
            2.0,
        ),
    ] {
        assert_eq!(run_body(name, body), expect, "{name}");
    }
}

// ---------------------------------------------------------------------------
// Expression contexts: a value-less branch used to drop the `let` binding
// ("variable x not found" after the trap)
// ---------------------------------------------------------------------------

#[test]
fn if_expression_assignment_with_value_less_branch() {
    for (name, body, expect) in [
        (
            "expr_plain",
            "    let t = 0\n    let x = if t >= 0 { t += 1 } else { 2 }\n    (x) + 0",
            1.0,
        ),
        (
            "expr_for",
            "    let t = 0\n    let x = if t >= 0 { for j in 2 { t += j } } else { 2 }\n    (x) + 0",
            1.0,
        ),
        (
            "expr_empty",
            "    let t = 0\n    let x = if t >= 0 { } else { 5 }\n    (x) + 0",
            1.0,
        ),
    ] {
        assert_eq!(run_body(name, body), expect, "{name}");
    }
}

// ---------------------------------------------------------------------------
// The on_render emission contract stays intact for the fixed shapes
// ---------------------------------------------------------------------------

/// The same if/for/else shape inside an object-literal body must still
/// commit its emitted child (in the issue this is the case that lost the
/// whole output silently).
#[test]
fn if_for_else_branch_emits_into_literal() {
    let vm = &mut test_vm();
    let v = eval_str(
        vm,
        "emit",
        r#"{
    let n = 0
    if n >= 0 {
        for j in 2 { n += j }
        {v: n}
    } else { {v: 9} }
}"#,
    );
    let errors = vm.take_errors();
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    let obj = v.as_object().expect("expected an object result");
    let len = vm.bx.heap.iter_len(obj);
    let mut vs = Vec::new();
    for i in 0..len {
        let kv = vm.bx.heap.iter_key_value(obj, i, vm.trap());
        if let Some(child) = kv.value.as_object() {
            if let Some(n) = vm.bx.heap.value(child, id!(v).into(), vm.trap()).as_number() {
                vs.push(Some(n));
            }
        }
    }
    assert_eq!(vs, vec![Some(1.0)], "emitted children: {vs:?}");
}
