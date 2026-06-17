#!/usr/bin/env bash
# Bundle the rule-based Iono submission as submission_rule_based_iono.tar.gz.
#
# Layout inside the tarball:
#   main.py              <- main_rule_based_iono.py (rule-based Iono wrapper)
#   deck.csv             <- deck_iono.csv (Iono deck)
#   rule_based_iono.py   <- scripts/rule_based_iono.py (Kiyota Iono logic)
#   cg/                  engine wrappers
#
# Verified lab bench (80g/opp, 2026-06-18):
#   overall 64.0% vs 23.3% for the 3-MLP, vs 46.5% for rule-based(Lucario).
#
# Usage:
#   ./make_submission_rule_based_iono.sh                       # -> submission_rule_based_iono.tar.gz
#   ./make_submission_rule_based_iono.sh out/foo.tar.gz        # -> out/foo.tar.gz

set -euo pipefail

OUT="${1:-submission_rule_based_iono.tar.gz}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

require() {
    [ -e "$1" ] || { echo "missing: $1" >&2; exit 1; }
}
require main_rule_based_iono.py
require deck_iono.csv
require scripts/rule_based_iono.py
require cg/__init__.py
require cg/api.py
require cg/game.py
require cg/sim.py
require cg/utils.py
require cg/libcg.so
require cg/cg.dll

deck_count=$(grep -c . deck_iono.csv)
if [ "$deck_count" -ne 60 ]; then
    echo "deck_iono.csv has $deck_count lines, expected 60" >&2
    exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp main_rule_based_iono.py "$STAGE/main.py"
cp deck_iono.csv "$STAGE/deck.csv"
cp scripts/rule_based_iono.py "$STAGE/rule_based_iono.py"
cp -r cg "$STAGE/cg"

mkdir -p "$(dirname "$OUT")"
( cd "$STAGE" && tar --owner=0 --group=0 \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -czvf "$ROOT/$OUT" \
    main.py \
    deck.csv \
    rule_based_iono.py \
    cg )

echo
echo "wrote: $OUT  ($(du -h "$OUT" | cut -f1))"
echo
echo "contents:"
tar -tzf "$OUT"

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
