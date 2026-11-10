# compat

Crates that apps on the OctoSense fork of Makepad depend on and upstream `work` has no
equivalent for, ported so those apps run under relax (hagency-desktop / Rinx):

- `makepad-markdown`: the fork's `libs/markdown` (Markdown parsing and passive HTML rendering),
  copied unchanged from OctoSense-org/makepad a0ad1839c.
- `makepad-markdown-widgets`: the fork's `libs/markdown_widgets`, ported to this tree's widgets.

They are not workspace members.
