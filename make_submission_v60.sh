#!/usr/bin/env bash
# Bundle the V60 single-policy submission as submission_v60.tar.gz.
#
# Layout:
#   main.py                      <- main_v60.py
#   deck.csv                     unchanged (= our learned-policy training deck)
#   cg/                          engine wrappers
#   train/__init__.py
#   train/features.py            (features_v60 imports from it)
#   train/features_v60.py
#   train/mlp_policy.py          (legacy import paths)
#   train/mlp_policy_v60.py
#   train/mlp_policy_v60_*.pt    latest V60 weights
#
# Usage:
#   ./make_submission_v60.sh                  # -> submission_v60.tar.gz
#   ./make_submission_v60.sh out/foo.tar.gz   # -> out/foo.tar.gz

set -euo pipefail
OUT="${1:-submission_v60.tar.gz}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

require() {
    [ -e "$1" ] || { echo "missing: $1" >&2; exit 1; }
}
require main_v60.py
require deck.csv
require cg/__init__.py
require cg/api.py
require cg/game.py
require cg/sim.py
require cg/utils.py
require cg/libcg.so
require cg/cg.dll
require train/__init__.py
require train/features.py
require train/features_v60.py
require train/mlp_policy.py
require train/mlp_policy_v60.py
require train/ensemble_policy_v60.py

# Need at least one V60 weight file.
weights=$(ls train/mlp_policy_v60*.pt 2>/dev/null || true)
if [ -z "$weights" ]; then
    echo "no V60 weights found in train/" >&2
    exit 1
fi

deck_count=$(grep -c . deck.csv)
if [ "$deck_count" -ne 60 ]; then
    echo "deck.csv has $deck_count lines, expected 60" >&2
    exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp main_v60.py "$STAGE/main.py"
cp deck.csv "$STAGE/deck.csv"
cp -r cg "$STAGE/cg"
mkdir -p "$STAGE/train"
cp train/__init__.py "$STAGE/train/"
cp train/features.py "$STAGE/train/"
cp train/features_v60.py "$STAGE/train/"
cp train/mlp_policy.py "$STAGE/train/"
cp train/mlp_policy_v60.py "$STAGE/train/"
cp train/ensemble_policy_v60.py "$STAGE/train/"
# Ship all V60 weight files (main_v60 picks the newest).
for w in $weights; do
    cp "$w" "$STAGE/train/"
done

mkdir -p "$(dirname "$OUT")"
( cd "$STAGE" && tar --owner=0 --group=0 \
    --exclude='__pycache__' --exclude='*.pyc' \
    -czvf "$ROOT/$OUT" main.py deck.csv train cg )

echo
echo "wrote: $OUT  ($(du -h "$OUT" | cut -f1))"
echo
tar -tzf "$OUT" | head -20

if [ "${SKIP_VERIFY:-0}" != "1" ]; then
    echo
    echo "verifying with scripts/check_main_exec.py..."
    if "$ROOT/scripts/run.sh" python3 "$ROOT/scripts/check_main_exec.py" \
        --tar-gz "$OUT" --no-policy; then
        echo "verification: OK"
    else
        rc=$?
        echo "verification FAILED" >&2
        exit $rc
    fi
fi
