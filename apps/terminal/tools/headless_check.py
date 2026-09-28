#!/usr/bin/env python3
"""Headless end-to-end check of the terminal app, driven through Makepad's
remote bridge (MAKEPAD_REMOTE, windows hidden) and read back through the
terminal's own control socket (terminal-ctl).

After every scenario it checks the process is alive, its log has no panic,
and the terminal still answers: a marker echoed in the focused pane is read
back from the emulator. Exit status 0 when every scenario passes.

    python3 tools/headless_check.py [--app PATH] [--ctl PATH] [--port N]
        [--only NAME[,NAME]] [--fonts N] [--octosense PATH]

--octosense runs the same scenarios in the OctoSense desktop (the terminal
as an in-process module) instead of the standalone app.
"""

import argparse
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


class Harness:
    def __init__(self, args):
        self.args = args
        self.home = tempfile.mkdtemp(prefix="term-hl-", dir="/tmp")
        self.port = args.port
        self.log_path = os.path.join(self.home, "app.log")
        self.proc = None
        self.failures = []
        self.marker_n = 0

    # --- process -----------------------------------------------------------
    def settings_dir(self):
        return os.path.join(self.home, "terminal")

    def write_settings(self, text):
        os.makedirs(self.settings_dir(), exist_ok=True)
        with open(os.path.join(self.settings_dir(), "settings.conf"), "w") as f:
            f.write(text)

    def start(self):
        env = dict(os.environ)
        env.update({"MAKEPAD_HIDE_WINDOWS": "1", "MAKEPAD_REMOTE": str(self.port)})
        if self.args.octosense:
            env["OCTOSENSE_HOME"] = self.home
            cmd = [self.args.octosense]
        else:
            env["MAKEPAD_HOME"] = self.home
            cmd = [self.args.app]
        env.pop("NO_COLOR", None)
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT, cwd=self.home)
        for _ in range(200):
            try:
                if '"i":0' in self.get("/s"):
                    break
            except Exception:
                pass
            time.sleep(0.1)
        time.sleep(2.0)
        if self.args.octosense:
            self.key("ReturnKey", ctrl=1, alt=1)  # Super+Return: open a terminal
            time.sleep(4.0)
        self.wait_ready()

    def stop(self):
        try:
            self.get("/quit")
        except Exception:
            pass
        if self.proc:
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def alive(self):
        return self.proc.poll() is None

    def panics(self):
        with open(self.log_path, errors="replace") as f:
            text = f.read()
        return [l for l in text.splitlines() if "panicked at" in l or "PANIC" in l or "AddressSanitizer" in l]

    # --- remote bridge -----------------------------------------------------
    def get(self, path):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=20) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"{path[:60]} -> {e.code} {body[:200]}") from None

    def key(self, code, **mods):
        q = "".join(f"&{k}=1" for k, v in mods.items() if v)
        self.get(f"/k?k=down&c={code}{q}")
        self.get(f"/k?k=up&c={code}{q}")
        time.sleep(0.05)

    def type(self, text):
        self.get("/k?t=" + urllib.parse.quote(text, safe=""))

    def line(self, text):
        self.type(text)
        self.key("ReturnKey")

    def mouse(self, kind, x, y, **extra):
        q = "".join(f"&{k}={v}" for k, v in extra.items())
        self.get(f"/m?k={kind}&x={x}&y={y}{q}")

    # --- control socket ----------------------------------------------------
    def ctl(self, *args):
        out = subprocess.run([self.args.ctl, "--home", self.home, *args], capture_output=True, text=True, timeout=30)
        return out.returncode, out.stdout, out.stderr

    def panes(self):
        code, out, _ = self.ctl("list", "--json")
        return json.loads(out) if code == 0 and out.strip() else []

    def focused(self):
        for p in self.panes():
            if p.get("focused"):
                return p
        return None

    def read(self, pane_id, lines=0):
        code, out, _ = self.ctl("read", pane_id, "--lines", str(lines))
        return out if code == 0 else ""

    def wait_ready(self):
        # OctoSense builds the terminal on its first launch in a fresh home
        # (about a minute), so its first pane can take far longer than the
        # standalone app's.
        for _ in range(1800 if self.args.octosense else 100):
            if self.focused():
                return True
            time.sleep(0.1)
        return False

    def responsive(self):
        """Echo a marker in the focused pane and read it back."""
        self.marker_n += 1
        marker = f"hl-marker-{self.marker_n}-{random.randint(0, 1 << 30)}"
        self.line(f"echo {marker}")
        for _ in range(60):
            p = self.focused()
            if p and marker in self.read(p["id"], 200).split("echo")[-1]:
                return True
            time.sleep(0.1)
        return False

    # --- scenario runner ---------------------------------------------------
    def check(self, name, ok, detail=""):
        status = "ok  " if ok else "FAIL"
        print(f"  [{status}] {name}{(' - ' + detail) if detail and not ok else ''}", flush=True)
        if not ok:
            self.failures.append(f"{name}: {detail}")

    def after(self, scenario):
        alive = self.alive()
        self.check(f"{scenario}: process alive", alive)
        panics = self.panics()
        self.check(f"{scenario}: no panic in log", not panics, "; ".join(panics[:3]))
        if alive:
            # Leave whatever the scenario left open: a list in the settings
            # panel, the panel, a close dialog, a job.
            for _ in range(3):
                self.key("Escape")
            self.key("KeyC", ctrl=1)
            self.check(f"{scenario}: terminal responds", self.responsive())


def count_panes(h):
    return len(h.panes())


def count_tabs(h):
    return len({p["tab"] for p in h.panes()})


# ------------------------------------------------------------------------------
def s_basic(h):
    h.check("echo round trip", h.responsive())
    h.line("printf '\\033[31mred\\033[0m \\xe4\\xb8\\xad\\xe6\\x96\\x87 \\xf0\\x9f\\x98\\x80\\n'")
    time.sleep(0.5)
    text = h.read(h.focused()["id"], 5)
    h.check("utf-8 and colour output read back", "red" in text and "中文" in text, repr(text[-120:]))


def s_tabs(h):
    start = count_tabs(h)
    for _ in range(5):
        h.key("KeyT", ctrl=1, shift=1)
        time.sleep(0.6)
    time.sleep(1.5)
    h.check("five new tabs", count_tabs(h) == start + 5, f"{count_tabs(h)} tabs")
    h.key("Key3", alt=1)
    time.sleep(0.5)
    f = h.focused()
    h.check("Alt+3 selects the third tab", f and f["tab"] == 3, str(f and f["tab"]))
    for _ in range(7):
        h.key("Tab", ctrl=1)
    for _ in range(3):
        h.key("Tab", ctrl=1, shift=1)
    h.key("PageDown", ctrl=1)
    h.key("PageUp", ctrl=1)
    h.key("Key9", alt=1)
    time.sleep(0.5)
    f = h.focused()
    h.check("Alt+9 selects the last tab", f and f["tab"] == count_tabs(h), str(f and f["tab"]))
    h.line("exit")
    time.sleep(1.5)
    h.check("exit closes its tab", count_tabs(h) == start + 4, f"{count_tabs(h)} tabs")
    for _ in range(4):
        h.key("KeyW", ctrl=1, shift=1)
        time.sleep(0.6)
    time.sleep(1.0)
    h.check("Ctrl+Shift+W closes tabs", count_tabs(h) == start, f"{count_tabs(h)} tabs")


def s_panes(h):
    start = count_panes(h)
    for i in range(6):
        h.key("KeyD" if i % 2 == 0 else "KeyE", ctrl=1, shift=1)
        time.sleep(0.7)
    time.sleep(1.5)
    h.check("six splits", count_panes(h) == start + 6, f"{count_panes(h)} panes")
    # The newest pane is at the bottom right: left and up lead away from
    # it, right and down lead back.
    before = h.focused()["id"]
    h.key("ArrowLeft", ctrl=1, shift=1)
    time.sleep(0.2)
    left = h.focused()["id"]
    h.check("Ctrl+Shift+Left moves focus", left != before, f"{before} -> {left}")
    h.key("ArrowRight", ctrl=1, shift=1)
    time.sleep(0.2)
    h.check("Ctrl+Shift+Right moves focus", h.focused()["id"] != left, h.focused()["id"])
    h.key("ArrowUp", ctrl=1, shift=1)
    time.sleep(0.2)
    h.check("Ctrl+Shift+Up moves focus", h.focused()["id"] != before)
    h.check("typing reaches the focused pane", h.responsive())
    h.key("KeyZ", ctrl=1, shift=1)
    h.check("zoomed pane responds", h.responsive())
    h.key("KeyZ", ctrl=1, shift=1)
    # Divider drags across the window, including past the edges.
    for x in [300, 10, 2000, 500]:
        h.mouse("down", 490, 300)
        h.mouse("move", x, 300)
        h.mouse("up", x, 300)
    h.line("exit")
    time.sleep(1.2)
    h.check("exit closes its pane", count_panes(h) == start + 5, f"{count_panes(h)} panes")
    for _ in range(5):
        h.key("KeyW", ctrl=1, shift=1)
        time.sleep(0.6)
    time.sleep(1.0)
    h.check("Ctrl+Shift+W closes panes", count_panes(h) == start, f"{count_panes(h)} panes")


def s_confirm_close(h):
    h.key("KeyT", ctrl=1, shift=1)
    time.sleep(1.2)
    tabs = count_tabs(h)
    h.line("sleep 347")
    time.sleep(1.5)
    h.key("KeyW", ctrl=1, shift=1)
    time.sleep(0.6)
    h.check("a busy tab asks first", count_tabs(h) == tabs)
    h.key("Escape")
    time.sleep(0.4)
    h.check("Esc keeps it", count_tabs(h) == tabs)
    h.key("KeyW", ctrl=1, shift=1)
    time.sleep(0.4)
    h.key("ReturnKey")
    time.sleep(1.5)
    h.check("Enter closes it", count_tabs(h) == tabs - 1, f"{count_tabs(h)} tabs")
    left = subprocess.run(["pgrep", "-f", "sleep 347"], capture_output=True, text=True).stdout.split()
    h.check("its job is gone", not left, str(left))


def s_settings_panel(h):
    h.key("Comma", ctrl=1)
    time.sleep(0.8)
    rows = 23
    for _ in range(rows + 2):
        for _ in range(3):
            h.key("ArrowRight")
        for _ in range(3):
            h.key("ArrowLeft")
        h.key("ArrowDown")
    for _ in range(rows + 2):
        h.key("ArrowUp")
    # The long lists: open, filter, move, cancel (restores) or pick.
    for down, text in [(0, "no"), (1, "m"), (2, "ping"), (13, "zsh")]:
        for _ in range(rows + 2):
            h.key("ArrowUp")
        for _ in range(down):
            h.key("ArrowDown")
        h.key("ReturnKey")
        h.type(text)
        for _ in range(3):
            h.key("ArrowDown")
        h.key("Backspace")
        h.key("Escape")
    h.key("Escape")
    conf = open(os.path.join(h.settings_dir(), "settings.conf")).read()
    h.check("settings file still readable", "theme" in conf and "external-control = true" in conf, conf[:200])


def s_all_fonts(h):
    """Apply fonts one after another from the panel and draw with each."""
    h.key("Comma", ctrl=1)
    time.sleep(0.8)
    h.key("ArrowDown")  # Font
    sample = "echo 'AaMm0Oil1 {}[] -> 中文 │█ é \U0001F600'"
    n = h.args.fonts
    for i in range(n):
        h.key("ArrowRight")
        time.sleep(0.12)
        if i % 25 == 24:
            h.key("Escape")
            h.line(sample)
            time.sleep(0.3)
            if not h.alive():
                break
            h.key("Comma", ctrl=1)
            time.sleep(0.3)
            h.key("ArrowDown")
    h.key("Escape")
    h.check(f"stepped through {n} fonts", h.alive())
    # Leave the terminal on the bundled font.
    conf_path = os.path.join(h.settings_dir(), "settings.conf")
    conf = open(conf_path).read()
    font = [l for l in conf.splitlines() if l.startswith("font-family")]
    print(f"      last font: {font}")


def wait_for(cond, seconds=5.0):
    """Poll `cond`: keys reach a hosted terminal later than a standalone one."""
    for _ in range(int(seconds / 0.1)):
        if cond():
            return True
        time.sleep(0.1)
    return cond()


def s_profiles(h):
    h.key("Comma", ctrl=1)
    time.sleep(0.5)
    for _ in range(30):
        h.key("ArrowUp")
    for _ in range(20):
        h.key("ArrowDown")  # Save as profile
    h.key("ReturnKey")
    h.type("hl-test")
    h.key("ReturnKey")
    profile = os.path.join(h.settings_dir(), "profiles", "hl-test.conf")
    saved = wait_for(lambda: os.path.exists(profile))
    h.check("profile saved", saved)
    h.key("ReturnKey")  # save again, bad name
    for _ in range(10):
        h.key("Backspace")
    h.type("a/b")
    h.key("ReturnKey")
    h.key("Escape")
    h.check("a bad name is not saved", not os.path.exists(os.path.join(h.settings_dir(), "profiles", "a")))
    h.key("ArrowDown")  # Delete profile
    h.key("ReturnKey")
    h.key("ReturnKey")
    h.check("profile deleted", wait_for(lambda: not os.path.exists(profile)))
    h.key("Escape")


def s_emulator_stress(h):
    h.line("seq 1 200000")
    time.sleep(3)
    h.line("head -c 3000000 /dev/urandom")
    time.sleep(4)
    h.key("KeyC", ctrl=1)
    h.line("reset")
    time.sleep(1.5)
    h.line("printf '\\033[?1049h\\033[2J\\033[999;999Hx\\033[?1049l\\033[?25l\\033[?25h\\033[?2004h\\033[?1000;1006h\\033[?1000;1006l'")
    h.line("printf '\\033]0;%s\\007' $(head -c 20000 /dev/zero | tr '\\0' x)")
    h.line("printf '\\033[>1u\\033[?u\\033[<u\\033[?1u'")
    h.line("less /etc/services")
    time.sleep(1)
    for k in ["KeyJ", "KeyJ", "Space", "KeyG", "KeyQ"]:
        h.key(k)
    h.line("vim -u NONE -N")
    time.sleep(1.5)
    h.type("ihello")
    h.key("Escape")
    h.line(":q!")
    time.sleep(1)
    h.line("top -l 1 | head -5")
    time.sleep(2)


def s_key_storm(h):
    h.line("cat > /dev/null")
    time.sleep(0.4)
    rnd = random.Random(7)
    keys = ["KeyA", "KeyZ", "Key1", "Space", "Tab", "Backspace", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
            "Home", "End", "PageUp", "PageDown", "F1", "F5", "Delete", "Insert", "Comma", "Period", "Slash"]
    for _ in range(400):
        mods = {m: 1 for m in ["shift", "alt", "ctrl"] if rnd.random() < 0.2}
        # Stay away from what closes things or quits.
        if mods.get("ctrl") and mods.get("shift"):
            mods.pop("shift")
        # In OctoSense, Ctrl+Alt is Super: the window manager's, not ours.
        if h.args.octosense and mods.get("ctrl") and mods.get("alt"):
            mods.pop("alt")
        h.key(rnd.choice(keys), **mods)
    h.type("".join(chr(rnd.choice([rnd.randint(0x20, 0x7e), rnd.randint(0xa0, 0x2fff), 0x4e2d])) for _ in range(300)))
    h.key("KeyC", ctrl=1)


def s_mouse(h):
    h.line("seq 1 500")
    time.sleep(0.8)
    h.mouse("down", 50, 200)
    for x, y in [(300, 250), (600, 400), (5, 5), (2000, 2000)]:
        h.mouse("move", x, y)
    h.mouse("up", 400, 300)
    for dy in [-50, 50, -500, 500]:
        h.mouse("scroll", 300, 300, dy=dy)
    for _ in range(3):
        h.get("/click?x=300&y=300")


def s_control_abuse(h):
    code, _, err = h.ctl("prompt", h.focused()["id"], "y")
    h.check("a lone approval key is refused", code == 1, err)
    code, _, _ = h.ctl("read", "1.1")
    h.check("another process's pane is refused", code == 1)
    sock_dir = os.path.join(h.home, "terminal", "control")
    socks = [os.path.join(sock_dir, n) for n in os.listdir(sock_dir) if n.endswith(".sock")]
    socks += [open(os.path.join(sock_dir, n)).read().strip() for n in os.listdir(sock_dir) if n.endswith(".path")]
    rnd = random.Random(3)
    payloads = [b"\n", b"{\n", os.urandom(5000) + b"\n", b"[" * 100000 + b"\n", b'{"cmd":"list"}' * 1000 + b"\n",
                b"x" * (3 << 20) + b"\n", b'{"cmd":"read","pane":"%d.1"}\n' % h.proc.pid]
    for path in socks:
        for p in payloads:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(5)
                s.connect(path)
                s.sendall(p)
                try:
                    s.recv(1 << 16)
                except socket.timeout:
                    pass
                s.close()
            except OSError:
                pass
        # Many clients at once.
        procs = [subprocess.Popen([h.args.ctl, "--home", h.home, "list", "--json"], stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL) for _ in range(20)]
        for p in procs:
            p.wait(30)
        # A client that connects and never writes, and one that half-writes.
        idle = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        idle.connect(path)
        half = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        half.connect(path)
        half.sendall(b'{"cmd":"li')
        code, out, _ = h.ctl("list", "--json")
        h.check("still answers while clients hang", code == 0 and out.strip().startswith("["))
        idle.close()
        half.close()


def s_churn(h):
    # The standalone app quits when its last tab closes: keep one spare.
    h.key("KeyT", ctrl=1, shift=1)
    time.sleep(1.0)
    shells_before = len(subprocess.run(["pgrep", "-P", str(h.proc.pid)], capture_output=True, text=True).stdout.split())
    for _ in range(25):
        h.key("KeyT", ctrl=1, shift=1)
        h.key("KeyD", ctrl=1, shift=1)
        time.sleep(0.3)
        h.key("KeyW", ctrl=1, shift=1)
        h.key("KeyW", ctrl=1, shift=1)
        time.sleep(0.2)
    time.sleep(3)
    shells_after = len(subprocess.run(["pgrep", "-P", str(h.proc.pid)], capture_output=True, text=True).stdout.split())
    h.check("no shells left behind", shells_after <= shells_before, f"{shells_before} -> {shells_after}")


SCENARIOS = [
    ("basic", s_basic),
    ("tabs", s_tabs),
    ("panes", s_panes),
    ("confirm-close", s_confirm_close),
    ("settings-panel", s_settings_panel),
    ("profiles", s_profiles),
    ("emulator-stress", s_emulator_stress),
    ("key-storm", s_key_storm),
    ("mouse", s_mouse),
    ("control-abuse", s_control_abuse),
    ("churn", s_churn),
    ("all-fonts", s_all_fonts),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default=os.path.join(REPO, "target", "release", "terminal"))
    ap.add_argument("--ctl", default=os.path.join(REPO, "target", "release", "terminal-ctl"))
    ap.add_argument("--octosense")
    ap.add_argument("--port", type=int, default=8460)
    ap.add_argument("--only")
    ap.add_argument("--fonts", type=int, default=400)
    args = ap.parse_args()
    # The app runs in a scratch directory: make every path absolute.
    args.app, args.ctl = os.path.abspath(args.app), os.path.abspath(args.ctl)
    if args.octosense:
        args.octosense = os.path.abspath(args.octosense)
    h = Harness(args)
    h.write_settings("external-control = true\ntab-bar = always\nconfirm-close-running = true\n")
    print(f"home {h.home}, log {h.log_path}")
    h.start()
    try:
        for name, fn in SCENARIOS:
            if args.only and name not in args.only.split(","):
                continue
            print(f"- {name}", flush=True)
            if not h.alive():
                h.check(f"{name}: skipped, the app is gone", False)
                continue
            try:
                fn(h)
            except Exception as e:  # a harness error is a failure too
                import traceback
                traceback.print_exc()
                h.check(f"{name}: harness", False, repr(e))
            h.after(name)
    finally:
        h.stop()
    print(f"\n{len(h.failures)} failure(s)")
    for f in h.failures:
        print("  " + f)
    if not h.failures:
        shutil.rmtree(h.home, ignore_errors=True)
    return 1 if h.failures else 0


if __name__ == "__main__":
    sys.exit(main())
