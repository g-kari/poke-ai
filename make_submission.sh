#!/usr/bin/env bash
# Bundle the PTCGABC submission as submission.tar.gz.
#
# Layout inside the tarball (must NOT be nested under a top-level directory;
# Kaggle mounts the contents at /kaggle_simulations/agent/):
#   main.py            entrypoint with agent()
#   deck.csv           60 card IDs
#   cg/                engine wrappers + libcg.so + cg.dll
#   train/             policy.npz + mlp_policy.pt + .py modules
#                      (main.py tries MLP first, falls back to LinearPolicy)
#
# Usage:
#   ./make_submission.sh                  # -> submission.tar.gz
#   ./make_submission.sh out/foo.tar.gz   # -> out/foo.tar.gz

set -euo pipefail

OUT="${1:-submission.tar.gz}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ---- sanity checks ---------------------------------------------------------
require() {
    [ -e "$1" ] || { echo "missing: $1" >&2; exit 1; }
}
require main.py
require deck.csv
require cg/__init__.py
require cg/api.py
require cg/game.py
require cg/sim.py
require cg/utils.py
require cg/libcg.so
require cg/cg.dll
require train/__init__.py
require train/policy.py
require train/features.py
require train/mlp_policy.py
require train/ensemble_policy.py

deck_count=$(grep -c . deck.csv)
if [ "$deck_count" -ne 60 ]; then
    echo "deck.csv has $deck_count lines, expected 60" >&2
    exit 1
fi

# policy.npz is optional — the agent falls back to engine-order prior if absent.
if [ ! -e train/policy.npz ]; then
    echo "warning: train/policy.npz missing — submission will use untrained fallback" >&2
fi

# ---- pack ------------------------------------------------------------------
mkdir -p "$(dirname "$OUT")"
tar --owner=0 --group=0 \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -czvf "$OUT" \
    main.py \
    deck.csv \
    cg \
    train/__init__.py \
    train/policy.py \
    train/features.py \
    train/mlp_policy.py \
    train/ensemble_policy.py \
    $( [ -e train/policy.npz ] && echo train/policy.npz ) \
    $(ls train/mlp_policy*.pt 2>/dev/null)

echo
echo "wrote: $OUT  ($(du -h "$OUT" | cut -f1))"
echo
echo "contents:"
tar -tzf "$OUT"

# Verify the tar.gz loads under the same exec()-based call shape Kaggle
# uses (catches __file__ / sys.path bugs that import-based checks miss).
# Skip when SKIP_VERIFY=1 — useful for iteration during dev when torch
# isn't available in the calling shell.
if [ "${SKIP_VERIFY:-0}" != "1" ]; then
    echo
    echo "verifying with scripts/check_main_exec.py..."
    if "$ROOT/scripts/run.sh" python3 "$ROOT/scripts/check_main_exec.py" --tar-gz "$OUT"; then
        echo "verification: OK"
    else
        rc=$?
        echo "verification FAILED; tar.gz is at $OUT but should NOT be submitted" >&2
        exit $rc
    fi
fi
