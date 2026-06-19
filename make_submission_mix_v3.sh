#!/usr/bin/env bash
# Bundle the Mix v3 ensemble: seed=0 base + seed=2 base + seed=100 ext.
#
# Lab winrate: 19.4% on 7-opp suite — balanced profile with 5 opponents
# at 20-25% (no peaks like Mix v1's Mega-Aboma 37.5%).
# Hypothesis: balanced profile yields higher LB ratio than Mix v1 (32.3).

set -euo pipefail
OUT="${1:-submission_mix_v3.tar.gz}"
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
require train/mlp_policy_seed100_ext.pt

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
cp train/mlp_policy.pt "$STAGE/train/"
cp train/mlp_policy_seed2.pt "$STAGE/train/"
cp train/mlp_policy_seed100_ext.pt "$STAGE/train/"

mkdir -p "$(dirname "$OUT")"
( cd "$STAGE" && tar --owner=0 --group=0 \
    --exclude='__pycache__' --exclude='*.pyc' \
    -czvf "$ROOT/$OUT" main.py deck.csv train cg )

echo
echo "wrote: $OUT  ($(du -h "$OUT" | cut -f1))"
tar -tzf "$OUT"

if [ "${SKIP_VERIFY:-0}" != "1" ]; then
    if "$ROOT/scripts/run.sh" python3 "$ROOT/scripts/check_main_exec.py" --tar-gz "$OUT"; then
        echo "verification: OK"
    else
        rc=$?
        echo "verification FAILED" >&2
        exit $rc
    fi
fi
