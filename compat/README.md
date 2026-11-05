# compat: the OctoSense jit line's crate names

OctoSense (and Octoscript-Makepad) name Makepad's apps by the fork's crate names
(`makepad-calculator`, `makepad-terminal`, …); upstream renamed them
`makepad-app-<name>`. Each crate here is that old name re-exporting the new
crate, with its `standalone` feature forwarded. `makepad-ai`, `makepad-markdown`
and `makepad-markdown-widgets` are empty: the OctoSense features that need them
(Rinx, the article editor, the markdown editor) are off on this line until those
libraries are ported to the HAL widgets.
