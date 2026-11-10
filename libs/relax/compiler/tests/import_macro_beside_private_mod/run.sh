#!/bin/bash
# usage: run.sh <makepad binary>: `use dep::name;` where crate `dep` has an exported macro `name`
# and a private module `name` (zerovec's `zerovec!` and `mod zerovec;`). rustc binds the import in
# the macro namespace only, so later `dep`-named paths of the importing module (`name!(..)`,
# `#[name::attr]`, `name::m::f()`, `name::other!()`) still go through the crate. Prints
# "import_macro_beside_private_mod: ok" or "import_macro_beside_private_mod: FAILED" (exit 1).
H=$(cd "$(dirname "$0")" && pwd)
export MAKEPAD_RUN_CACHE=off
ERR=$(mktemp)
OUT=$(cd "$H" && MAKEPAD_RUN_STD=$H/../../std "$1" run --quiet ibp-app 2>"$ERR")
echo "$OUT"
if [ "$OUT" = "import beside private mod 19 7 9" ]; then rm -f "$ERR"; echo "import_macro_beside_private_mod: ok"; else tail -5 "$ERR"; rm -f "$ERR"; echo "import_macro_beside_private_mod: FAILED"; exit 1; fi
