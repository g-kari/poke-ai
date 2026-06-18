#!/usr/bin/env bash
# Full BC pipeline orchestrator: bench BC weights, optionally bundle.
#
# Usage:
#   scripts/bc_pipeline.sh                       # bench only, 10g/side
#   scripts/bc_pipeline.sh 20                    # bench 20g/side (= 40g/opp)
#   SHIP=1 scripts/bc_pipeline.sh 20             # also bundle + verify

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GAMES="${1:-10}"
WEIGHTS="${WEIGHTS:-train/mlp_policy_v60_bc.pt}"

if [ ! -e "$WEIGHTS" ]; then
    echo "missing: $WEIGHTS — train BC first via train/bc_train.py" >&2
    exit 1
fi

echo "=== BC bench: $WEIGHTS @ $((2 * GAMES))g/opp ==="
scripts/run.sh python3 scripts/bench_v60.py --weights "$WEIGHTS" --games "$GAMES" \
    | tee /tmp/bc_bench.log

# Extract overall winrate from log.
overall=$(awk '/^overall:/ {print $NF}' /tmp/bc_bench.log | tr -d '%' || true)
echo
echo "overall winrate parsed: $overall%"

if [ "${SHIP:-0}" = "1" ]; then
    echo
    echo "=== Bundling submission_v60_bc.tar.gz ==="
    BC_PT="$WEIGHTS" ./make_submission_v60_bc.sh
    echo
    echo "Ready: .venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \\"
    echo "    -f submission_v60_bc.tar.gz -m \"BC from V6 (lab ${overall}% @ $((2 * GAMES))g/opp)\""
fi
