#!/usr/bin/env bash
# Bundle the rule-based Mega Lucario submission as submission_rule_based.tar.gz.
#
# Layout inside the tarball:
#   main.py            <- main_rule_based.py (rule-based agent wrapper)
#   deck.csv           <- deck_mega_lucario.csv (Mega Lucario deck)
#   rule_based_agent.py <- scripts/rule_based_agent.py (Kiyota Mega Lucario logic)
#   cg/                engine wrappers
#
# Verified lab bench (80g/opp, 2026-06-18):
#   overall 46.5% vs 23.3% for the 3-MLP submission.
#
# Usage:
#   ./make_submission_rule_based.sh                       # -> submission_rule_based.tar.gz
#   ./make_submission_rule_based.sh out/foo.tar.gz        # -> out/foo.tar.gz

set -euo pipefail

OUT="${1:-submission_rule_based.tar.gz}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ---- sanity checks ---------------------------------------------------------
require() {
    [ -e "$1" ] || { echo "missing: $1" >&2; exit 1; }
}
require main_rule_based.py
require deck_mega_lucario.csv
require scripts/rule_based_agent.py
require cg/__init__.py
require cg/api.py
require cg/game.py
require cg/sim.py
require cg/utils.py
require cg/libcg.so
require cg/cg.dll

deck_count=$(grep -c . deck_mega_lucario.csv)
if [ "$deck_count" -ne 60 ]; then
    echo "deck_mega_lucario.csv has $deck_count lines, expected 60" >&2
    exit 1
fi

# ---- stage rename files into a tmp dir, then tar from there --------------
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp main_rule_based.py "$STAGE/main.py"
cp deck_mega_lucario.csv "$STAGE/deck.csv"
cp scripts/rule_based_agent.py "$STAGE/rule_based_agent.py"
cp -r cg "$STAGE/cg"

mkdir -p "$(dirname "$OUT")"
( cd "$STAGE" && tar --owner=0 --group=0 \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -czvf "$ROOT/$OUT" \
    main.py \
    deck.csv \
    rule_based_agent.py \
    cg )

echo
echo "wrote: $OUT  ($(du -h "$OUT" | cut -f1))"
echo
echo "contents:"
tar -tzf "$OUT"

# Verify the bundle loads under Kaggle's exec() shape.
if [ "${SKIP_VERIFY:-0}" != "1" ]; then
    echo
    echo "verifying with scripts/check_main_exec.py..."
    if "$ROOT/scripts/run.sh" python3 "$ROOT/scripts/check_main_exec.py" --tar-gz "$OUT" --no-policy; then
        echo "verification: OK"
    else
        rc=$?
        echo "verification FAILED; tar.gz is at $OUT but should NOT be submitted" >&2
        exit $rc
    fi
fi
