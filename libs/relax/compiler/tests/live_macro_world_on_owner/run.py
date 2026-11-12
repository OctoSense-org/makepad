#!/usr/bin/env python3
"""A live edit whose file derives with a hosted proc macro, after the macro world has gone.

The program's load frees the macro world once its files are parsed (a warm start never builds
it), so the first live edit of a file with a hosted derive rebuilds the world, on the compiler's
owner: the edit is an owner request. The world here holds a library (`shape`, a dependency of the
proc-macro crate `count`) that itself derives with another proc-macro crate (`tag`). Parsed by the
loader's workers, that library's derives were requests to the owner, which waited for those very
files: the session hung in "patching" for good (a daemon under `makepad run` stopped answering).
Past that, the borrow checker's carry-over of the edit read the rebuilt world's verdicts as
definitions of the program (an index past its last one): "internal compiler error applying the
edit", and the app restarted.

usage: run.py --makepad <makepad binary> [--warm]   (MAKEPAD_RUN_STD in the environment)

  1. app/src/main.rs: `value` returns 2 -> "patched", the program prints the new value
  2. the same again with 3 (the world is alive now)        -> "patched", the new value
"""
import os, shutil, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
relax = os.path.abspath(sys.argv[sys.argv.index("--makepad") + 1])
warm = "--warm" in sys.argv
tmp = tempfile.mkdtemp(prefix="live-macro-world-")
pkg = os.path.join(tmp, "pkg")
shutil.copytree(os.path.join(HERE, "pkg"), pkg)
main_rs = os.path.join(pkg, "app", "src", "main.rs")
base = open(main_rs).read()
assert base.count("    1\n}") == 1
APP = "relax-live-world-app"
# `shape` gets MODS more files, each with a derive: more than the loader parses itself before a
# worker of its pool takes one
MODS = 24
more = os.path.join(pkg, "shape", "src", "more")
os.mkdir(more)
for i in range(MODS):
    with open(os.path.join(more, "m%d.rs" % i), "w") as f:
        f.write("#[derive(relax_live_world_tag::Tag)]\npub struct T;\n")
with open(os.path.join(pkg, "shape", "src", "more.rs"), "w") as f:
    f.write("use crate::Tagged;\n" + "".join("mod m%d;\n" % i for i in range(MODS)))
    f.write("pub fn extra() -> u32 {\n    0%s\n}\n" % "".join(" + <m%d::T as Tagged>::TAG" % i for i in range(MODS)))
# shape::base(): "Seven77" has 7 letters, each "T" 1
COUNT = 700 + MODS

base_env = dict(os.environ, MAKEPAD_RUN_AHEAD="minimal")
if warm:
    env = dict(base_env, MAKEPAD_RUN_CACHE_DIR=os.path.join(tmp, "cache"), MAKEPAD_RUN_TARGET_DIR=os.path.join(tmp, "target"))
    open(os.path.join(pkg, "stop"), "w").close()
    fill = subprocess.run([relax, "run", "--quiet", APP], cwd=pkg, env=env, capture_output=True, text=True, timeout=600)
    os.remove(os.path.join(pkg, "stop"))
    print("%s warm-up run fills the cache (exit %d)" % ("PASS" if fill.returncode == 0 else "FAIL", fill.returncode))
    if fill.returncode != 0:
        print(fill.stderr[-3000:]); sys.exit(1)
else:
    env = dict(base_env, MAKEPAD_RUN_CACHE="0")
p = subprocess.Popen([relax, "run", "--verbose", APP], cwd=pkg, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
out, err = [], []
def pump(f, acc):
    for line in f:
        acc.append(line.rstrip("\n"))
threading.Thread(target=pump, args=(p.stdout, out), daemon=True).start()
threading.Thread(target=pump, args=(p.stderr, err), daemon=True).start()

def wait(pred, secs=30):
    t = time.time()
    while time.time() - t < secs:
        if pred():
            return True
        time.sleep(0.05)
    return False

fails = 0
def step(name, ok):
    global fails
    print("%s %s" % ("PASS" if ok else "FAIL", name))
    if not ok:
        fails += 1

def save(value):
    with open(main_rs, "w") as f:
        f.write(base.replace("    1\n}", "    %d\n}" % value))
    st = os.stat(main_rs)
    os.utime(main_rs, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

def edit(value, what):
    n_err, n_out = len(err), len(out)
    save(value)
    step("%s: patched" % what, wait(lambda: any("patched" in l for l in err[n_err:])))
    line = "out tick %d %d" % (COUNT, value)
    step("%s: the program prints `%s`" % (what, line), wait(lambda: line in out[n_out:], 10))

step("starts", wait(lambda: out.count("out tick %d 1" % COUNT) > 2 and any("watching" in l for l in err), 180))
edit(2, "first edit (the macro world is rebuilt)")
edit(3, "second edit (the macro world is alive)")
ice = [l for l in err if "internal compiler error" in l or "panicked" in l]
step("no internal compiler error%s" % ((": " + ice[0][:300]) if ice else ""), not ice)
open(os.path.join(pkg, "stop"), "w").close()
ok = wait(lambda: p.poll() is not None)
step("program exits", ok and p.returncode == 0)
if p.poll() is None:
    p.kill()
if fails:
    print("---- stderr"); print("\n".join(err[-60:]))
shutil.rmtree(tmp, ignore_errors=True)
print("live macro world on owner: %d fail" % fails)
sys.exit(1 if fails else 0)
