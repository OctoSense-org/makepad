#!/usr/bin/env python3
"""A live edit that rebuilds the macro world beside a program running in an app domain.

A hosted program (makepad's HAL in its crate graph) runs in a domain of its own, not in the unit's
compile domain: its statics are in that domain's block. The program's load frees the macro world,
so the first live edit of a file with a hosted derive rebuilds the world while the program runs,
and the world lays out its own statics then. Each of them was copied into every live domain block
of the process, the program's among them, at the world's offset: whatever static of the program
had that offset was overwritten (a tokio daemon under `makepad run`: its thread-local keys read as
never created, "there is no reactor running", or a fault on a pointer that was another static's
image). The program here fills a large static at its start and keeps a thread-local with drop
glue; the proc-macro crate has a static whose image is not zeros.

usage: run.py --makepad <makepad binary>   (MAKEPAD_RUN_STD in the environment)

  1. app/src/main.rs: `value` returns 2 -> "patched"; the program prints the new value, no word of
     its static changed and its thread-local still holds what it stored
  2. the same again with 3 (the world is alive now)
"""
import os, shutil, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../../.."))
relax = os.path.abspath(sys.argv[sys.argv.index("--makepad") + 1])
# under the workspace: cargo reads .cargo/config.toml from the folder it runs in
os.makedirs(os.path.join(ROOT, "target"), exist_ok=True)
tmp = tempfile.mkdtemp(prefix="live-beside-domain-", dir=os.path.join(ROOT, "target"))
pkg = os.path.join(tmp, "pkg")
shutil.copytree(os.path.join(HERE, "pkg"), pkg)
# makepad-widgets (the HAL comes with it, with the features the makepad binary's copy was built
# with), and the workspace's [patch.crates-io] (the copy is a workspace of its own)
manifest = os.path.join(pkg, "app", "Cargo.toml")
with open(manifest, "a") as f:
    f.write('makepad-widgets = { path = "%s", default-features = false }\n' % os.path.join(ROOT, "widgets"))
patch, on = [], False
for line in open(os.path.join(ROOT, "Cargo.toml")):
    if line.startswith("["):
        on = line.strip() == "[patch.crates-io]"
    elif on and 'path = "' in line:
        name, rest = line.split("=", 1)
        patch.append('%s = { path = "%s" }' % (name.strip(), os.path.join(ROOT, rest.split('path = "', 1)[1].split('"', 1)[0])))
if patch:
    with open(os.path.join(pkg, "Cargo.toml"), "a") as f:
        f.write("\n[patch.crates-io]\n" + "\n".join(patch) + "\n")
main_rs = os.path.join(pkg, "app", "src", "main.rs")
base = open(main_rs).read()
assert base.count("    1\n}") == 1
APP = "relax-live-beside-app"

env = dict(os.environ, MAKEPAD_RUN_AHEAD="minimal", MAKEPAD_RUN_CACHE="0", MAKEPAD_HIDE_WINDOWS="1", MAKEPAD_MUTE="1")
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
        if p.poll() is not None:
            time.sleep(0.2)
            return pred()
        time.sleep(0.05)
    return False

fails = 0
def step(name, ok):
    global fails
    print("%s %s" % ("PASS" if ok else "FAIL", name), flush=True)
    if not ok:
        fails += 1

def tick(value):
    return "out tick 7 %d marks-changed 0 held 77 domain " % value

def ticked(value, lines):
    return [l for l in lines if l.startswith(tick(value))]

def save(value):
    with open(main_rs, "w") as f:
        f.write(base.replace("    1\n}", "    %d\n}" % value))
    st = os.stat(main_rs)
    os.utime(main_rs, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

def edit(value, what):
    n_err, n_out = len(err), len(out)
    save(value)
    step("%s: patched" % what, wait(lambda: any("patched" in l for l in err[n_err:])))
    step("%s: the program prints `%s..`" % (what, tick(value)), wait(lambda: ticked(value, out[n_out:]), 10))
    # every line since the edit: the statics as the program wrote them
    time.sleep(0.5)
    odd = [l for l in out[n_out:] if l.startswith("out tick ") and " marks-changed 0 held 77 domain " not in l]
    step("%s: the program's statics are its own%s" % (what, (": " + odd[0]) if odd else ""), not odd)

step("starts", wait(lambda: len(ticked(1, out)) > 2 and any("watching" in l for l in err), 300))
first = ticked(1, out)
step("runs in an app domain%s" % ((": " + first[0]) if first else ""), bool(first) and not first[0].endswith(" domain 0"))
edit(2, "first edit (the macro world is rebuilt)")
edit(3, "second edit (the macro world is alive)")
bad = [l for l in err if "internal compiler error" in l or "panicked" in l or "POISONED" in l or "fatal fault" in l]
step("no internal compiler error, panic or fault%s" % ((": " + bad[0][:300]) if bad else ""), not bad)
open(os.path.join(pkg, "stop"), "w").close()
# a hosted program's process goes on after its main returned (the host's loop): ended here
t_end = time.time()
while p.poll() is None and time.time() - t_end < 5:
    time.sleep(0.05)
if p.poll() is None:
    p.terminate()
    try:
        p.wait(10)
    except subprocess.TimeoutExpired:
        p.kill()
if fails:
    print("---- stdout"); print("\n".join(out[-8:]))
    print("---- stderr"); print("\n".join(err[-60:]))
shutil.rmtree(tmp, ignore_errors=True)
print("live macro world beside app domain: %d fail" % fails)
sys.exit(1 if fails else 0)
