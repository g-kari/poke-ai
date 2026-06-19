#!/usr/bin/env bash
# Bundle the CHAMPION 3-MLP base ensemble (seed=0/2/100, all base).
#
# This is the exact configuration of ref 53778627 (LB 679.6) — DL
# champion. Use this when you need to re-submit the champion exactly
# (preservation, control experiment, A/B baseline).
#
# Lab winrate: 18.9% on 7-opp suite. Ratio 35.9 = our highest LB efficiency.

set -euo pipefail
OUT="${1:-submission_3mlp_base.tar.gz}"
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
require train/mlp_policy.pt
require train/mlp_policy_seed2.pt
require train/mlp_policy_seed100.pt

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
# ONLY the 3 base seeds — NO v60, NO ext, NO 4-MLP additions.
cp train/mlp_policy.pt "$STAGE/train/"
cp train/mlp_policy_seed2.pt "$STAGE/train/"
cp train/mlp_policy_seed100.pt "$STAGE/train/"

mkdir -p "$(dirname "$OUT")"
( cd "$STAGE" && tar --owner=0 --group=0 \
    --exclude='__pycache__' --exclude='*.pyc' \
    -czvf "$ROOT/$OUT" main.py deck.csv train cg )

echo
echo "wrote: $OUT  ($(du -h "$OUT" | cut -f1))"
echo
tar -tzf "$OUT" | head -20

if [ "${SKIP_VERIFY:-0}" != "1" ]; then
    if "$ROOT/scripts/run.sh" python3 "$ROOT/scripts/check_main_exec.py" --tar-gz "$OUT"; then
        echo "verification: OK"
    else
        rc=$?
        echo "verification FAILED" >&2
        exit $rc
    fi
fi
