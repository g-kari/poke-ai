#!/usr/bin/env bash
# Bundle the rule-based Crustle Dashimaki submission as
# submission_rule_based_crustle_dashimaki.tar.gz.
#
# Layout inside the tarball:
#   main.py                          <- main_rule_based_crustle_dashimaki.py
#   deck.csv                         <- deck_crustle_dashimaki.csv
#   rule_based_crustle_dashimaki.py  <- scripts/rule_based_crustle_dashimaki.py
#   cg/                              engine wrappers
#
# Verified lab bench (80g/opp, 2026-06-18): overall 67.3%, strongest of
# all 6 candidate subjects. Single weakness: vs Iono 11.2%.

set -euo pipefail

OUT="${1:-submission_rule_based_crustle_dashimaki.tar.gz}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

require() {
    [ -e "$1" ] || { echo "missing: $1" >&2; exit 1; }
}
require main_rule_based_crustle_dashimaki.py
require deck_crustle_dashimaki.csv
require scripts/rule_based_crustle_dashimaki.py
require cg/__init__.py
require cg/api.py
require cg/game.py
require cg/sim.py
require cg/utils.py
require cg/libcg.so
require cg/cg.dll

deck_count=$(grep -c . deck_crustle_dashimaki.csv)
if [ "$deck_count" -ne 60 ]; then
    echo "deck_crustle_dashimaki.csv has $deck_count lines, expected 60" >&2
    exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp main_rule_based_crustle_dashimaki.py "$STAGE/main.py"
cp deck_crustle_dashimaki.csv "$STAGE/deck.csv"
cp scripts/rule_based_crustle_dashimaki.py "$STAGE/rule_based_crustle_dashimaki.py"
cp -r cg "$STAGE/cg"

mkdir -p "$(dirname "$OUT")"
( cd "$STAGE" && tar --owner=0 --group=0 \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -czvf "$ROOT/$OUT" \
    main.py \
    deck.csv \
    rule_based_crustle_dashimaki.py \
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
