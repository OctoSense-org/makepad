#!/usr/bin/env python3
"""Marks the Linux desktop sites of a crate, so an OpenHarmony build leaves them out.

OpenHarmony is `target_os = "linux"` with `target_env = "ohos"`: a phone system on a Linux
kernel, like Android, with none of the desktop's libraries (X11, Wayland, ALSA, D-Bus).
The tree already writes what Linux and Android share as
`any(target_os = "linux", target_os = "android")`, so a `target_os = "linux"` that stands
without Android beside it is a desktop site. This rewrites each of those to
`all(target_os = "linux", not(target_env = "ohos"))` and leaves the shared sites alone.

    cfg_linux_desktop.py [--check] [--all | --phone] <file or directory>...

It reads `cfg(..)`, `cfg_attr(..)` and `cfg!(..)` in .rs files and the `cfg(..)` of Cargo.toml
target tables. Running it twice changes nothing. `--check` lists the files it would change.

`--all` is for a crate whose shared Linux and Android code is not OpenHarmony's either (the
video codecs): there every `target_os = "linux"` is marked, also beside Android.

`--phone` does the other half for the files named: their `target_os = "android"` sites are a
phone's rather than JNI's (the GLES core's), so OpenHarmony joins them as
`any(target_os = "android", target_env = "ohos")`.
"""
import os
import re
import sys

LINUX = ('target_os', 'linux')
ANDROID = ('target_os', 'android')
OHOS = ('target_env', 'ohos')
START = re.compile(r'\bcfg(_attr)?(!)?\s*\(')


class Node:
    def __init__(self, kind, start, end, key=None, value=None, children=None):
        self.kind, self.start, self.end = kind, start, end
        self.key, self.value, self.children = key, value, children or []

    def atom(self):
        return (self.key, self.value) if self.kind == 'atom' else None


def skip_space(text, i):
    while i < len(text) and text[i] in ' \t\r\n':
        i += 1
    return i


def parse(text, i):
    """One predicate starting at `i`; None when the text there is not one."""
    i = skip_space(text, i)
    m = re.compile(r'[A-Za-z_][A-Za-z0-9_]*').match(text, i)
    if not m:
        return None
    name, j = m.group(0), skip_space(text, m.end())
    if j < len(text) and text[j] == '(' and name in ('any', 'all', 'not'):
        children, k = [], j + 1
        while True:
            k = skip_space(text, k)
            if k < len(text) and text[k] == ')':
                return Node(name, i, k + 1, children=children)
            child = parse(text, k)
            if child is None:
                return None
            children.append(child)
            k = skip_space(text, child.end)
            if k < len(text) and text[k] == ',':
                k += 1
            elif k >= len(text) or text[k] != ')':
                return None
    if j < len(text) and text[j] == '=':
        k = skip_space(text, j + 1)
        s = re.compile(r'"([^"\\]*)"').match(text, k)
        if not s:
            return None
        return Node('atom', i, s.end(), key=name, value=s.group(1))
    return Node('atom', i, m.end(), key=name, value=None)


def edits(node, parent, out, mode):
    if node.kind == 'atom':
        siblings = [c.atom() for c in parent.children] if parent else []
        nots = [c.children[0].atom() for c in parent.children if c.kind == 'not' and len(c.children) == 1] if parent else []
        if mode == 'phone':
            if node.atom() != ANDROID or OHOS in siblings:
                return
            if parent and parent.kind == 'any':
                out.append((node.start, node.end, 'target_os = "android", target_env = "ohos"'))
            else:
                out.append((node.start, node.end, 'any(target_os = "android", target_env = "ohos")'))
            return
        if node.atom() != LINUX:
            return
        if parent and parent.kind == 'any' and ((ANDROID in siblings and mode != 'all') or OHOS in siblings):
            return
        if parent and parent.kind == 'all' and (OHOS in siblings or OHOS in nots):
            return
        if parent and parent.kind == 'all':
            out.append((node.start, node.end, 'target_os = "linux", not(target_env = "ohos")'))
        else:
            out.append((node.start, node.end, 'all(target_os = "linux", not(target_env = "ohos"))'))
        return
    for child in node.children:
        edits(child, node, out, mode)


def rewrite(text, mode=None):
    out = []
    for m in START.finditer(text):
        line_start = text.rfind('\n', 0, m.start()) + 1
        if text[line_start:m.start()].lstrip().startswith('//'):
            continue
        node = parse(text, m.end())
        if node is not None:
            edits(node, None, out, mode)
    out = sorted(set(out), reverse=True)
    for start, end, new in out:
        text = text[:start] + new + text[end:]
    return text, len(out)


def files(paths):
    for path in paths:
        if os.path.isfile(path):
            yield path
            continue
        for root, dirs, names in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ('target', '.git')]
            for name in names:
                if name.endswith('.rs') or name == 'Cargo.toml':
                    yield os.path.join(root, name)


def main():
    args = sys.argv[1:]
    check = '--check' in args
    mode = 'phone' if '--phone' in args else 'all' if '--all' in args else None
    paths = [a for a in args if not a.startswith('--')]
    if not paths:
        sys.exit(__doc__)
    total = changed = 0
    for path in files(paths):
        with open(path, encoding='utf-8') as f:
            text = f.read()
        new, n = rewrite(text, mode)
        if n:
            total += n
            changed += 1
            if check:
                print(f'{n:4} {path}')
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new)
    print(f'{total} sites in {changed} files' + (' (not written)' if check else ''))


if __name__ == '__main__':
    main()
