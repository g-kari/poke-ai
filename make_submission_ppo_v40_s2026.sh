#!/usr/bin/env bash
# Bundle PPO_v40 base s100 + PPO seed=2026 — lab 22.9% on 7-opp 700g (= our highest
# single-policy lab achievement). Expected LB ~800 at ratio 35.
#
# Layout uses the standard make_submission.sh shape but with only
# mlp_policy_ppo_v40_s100_t2026.pt as the v40 weight (= single policy, not ensemble).

set -euo pipefail
OUT="${1:-submission_ppo_v40_s2026.tar.gz}"
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
require train/mlp_policy_ppo_v40_s100_t2026.pt

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
# Only the PPO_v40 seed=100 weight (= main.py picks single via glob).
cp train/mlp_policy_ppo_v40_s100_t2026.pt "$STAGE/train/"

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
