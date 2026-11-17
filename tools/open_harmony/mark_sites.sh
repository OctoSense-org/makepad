#!/bin/sh
# Re-applies the OpenHarmony site marking (cfg_linux_desktop.py) to the crates an OpenHarmony
# app builds from. Run it from the repository root after a merge from upstream brings new
# `target_os = "linux"` or `target_os = "android"` sites; it changes nothing that is marked.
set -eu
mark="python3 $(dirname "$0")/cfg_linux_desktop.py"

# Linux desktop sites are not OpenHarmony's: the crates of an app's build closure.
closure="
code_editor draw libs/filesystem_watcher libs/font_core libs/git libs/remote libs/shared_bytes
platform platform/hal platform/network platform/studio platform/video
widgets/core widgets/families/data widgets/families/extras widgets/families/media
widgets/families/pickers widgets/families/rich_text
"
paths=""
for crate in $closure; do
    for part in src Cargo.toml build.rs; do
        [ -e "$crate/$part" ] && paths="$paths $crate/$part"
    done
done
$mark $paths platform/hal/tests platform/hal/examples

# The video codecs' shared Linux and Android code is not OpenHarmony's either.
$mark --all platform/video/src platform/video/Cargo.toml

# Where the Android sites are a phone's, not JNI's, OpenHarmony joins them.
$mark --phone platform/hal/src/backend/gl platform/network/src platform/network/Cargo.toml platform/src \
    draw/src/shader/draw_text.rs widgets/core/src libs/remote/src libs/font_core/src/font_scan.rs
