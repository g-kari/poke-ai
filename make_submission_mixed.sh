#!/usr/bin/env bash
# Bundle the mixed 3-MLP submission (= seed=0 ext + seed=2 base + seed=100 base).
#
# Lab winrate: 20.4% on 7-opp suite (vs 3-MLP base 18.9%, +1.5pp).
# Expected LB: ~700-740 (using ratio 35.9 from 3-MLP base reference).
#
# Usage:
#   ./make_submission_mixed.sh                  # -> submission_mixed.tar.gz

set -euo pipefail
OUT="${1:-submission_mixed.tar.gz}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

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
require train/mlp_policy_ext.pt
require train/mlp_policy_seed2.pt
require train/mlp_policy_seed100.pt

deck_count=$(grep -c . deck.csv)
if [ "$deck_count" -ne 60 ]; then
    echo "deck.csv has $deck_count lines, expected 60" >&2
    exit 1
fi

# Stage only the 3 weights we want (= mixed ensemble).
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp main.py "$STAGE/main.py"
cp deck.csv "$STAGE/deck.csv"
cp -r cg "$STAGE/cg"
mkdir -p "$STAGE/train"
cp train/__init__.py "$STAGE/train/"
cp train/policy.py "$STAGE/train/"
cp train/features.py "$STAGE/train/"
cp train/mlp_policy.py "$STAGE/train/"
cp train/ensemble_policy.py "$STAGE/train/"
if [ -e train/policy.npz ]; then
    cp train/policy.npz "$STAGE/train/"
fi
# Only the mixed 3-MLP weights.
cp train/mlp_policy_ext.pt "$STAGE/train/"
cp train/mlp_policy_seed2.pt "$STAGE/train/"
cp train/mlp_policy_seed100.pt "$STAGE/train/"

mkdir -p "$(dirname "$OUT")"
( cd "$STAGE" && tar --owner=0 --group=0 \
    --exclude='__pycache__' --exclude='*.pyc' \
    -czvf "$ROOT/$OUT" main.py deck.csv train cg )

echo
echo "wrote: $OUT  ($(du -h "$OUT" | cut -f1))"
echo
tar -tzf "$OUT"

if [ "${SKIP_VERIFY:-0}" != "1" ]; then
    echo
    echo "verifying with scripts/check_main_exec.py..."
    if "$ROOT/scripts/run.sh" python3 "$ROOT/scripts/check_main_exec.py" --tar-gz "$OUT"; then
        echo "verification: OK"
    else
        rc=$?
        echo "verification FAILED" >&2
        exit $rc
    fi
fi
