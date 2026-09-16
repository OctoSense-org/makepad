//! A proto-inherit (`+:`) resolves its field with a probe first and falls back
//! to the type-check default. The probe's miss is expected and discarded, so it
//! must not build a "Did you mean" list; a real miss still reports one.
use makepad_script::*;

fn test_vm() -> ScriptVm<'static> {
    let host = Box::leak(Box::new(ScriptVmHost::new((), ())));
    ScriptVm { host, bx: Box::new(ScriptVmBase::new()) }
}

fn eval(vm: &mut ScriptVm, file: &str, code: &str) -> Vec<String> {
    vm.bx.captured_errors = Some(Vec::new());
    vm.with_instruction_limit(500_000, |vm| {
        vm.eval(ScriptMod {
            cargo_manifest_path: String::new(),
            module_path: String::new(),
            file: file.to_string(),
            line: 0,
            column: 0,
            code: code.to_string(),
            values: vec![],
        })
    });
    vm.take_errors()
}

// One test body: the counter is process-wide, so the cases must not run in parallel.
#[test]
fn a_probe_miss_builds_no_suggestion_and_a_real_miss_still_does() {
    let mut vm = test_vm();

    // The field exists on the prototype: the probe hits, nothing is reported.
    let before = suggest::suggestions_built();
    let errors = eval(&mut vm, "probe_hit", "let base = {sub: {x: 1}, other: 2}\nlet o = base{sub +: {y: 2}}\n");
    assert!(errors.is_empty(), "a +: over an inherited field errored: {errors:?}");
    assert_eq!(suggest::suggestions_built(), before, "a hit must not build suggestions");

    // Neither on the prototype nor in a type-check: a real error. Before the
    // probe passed NoTrap, this built two lists (the discarded probe's and the
    // reported one); now only the reported one.
    let before = suggest::suggestions_built();
    let errors = eval(&mut vm, "real_miss", "let base = {sub: {x: 1}, other: 2}\nlet o = base{sbu +: {y: 2}}\n");
    let built = suggest::suggestions_built() - before;
    assert!(!errors.is_empty(), "a +: on a missing field must still report an error");
    assert!(errors.iter().any(|e| e.contains("Did you mean")), "the real error keeps its suggestion: {errors:?}");
    assert_eq!(built, 1, "only the reported miss builds a suggestion list (the probe built one too before)");
}
