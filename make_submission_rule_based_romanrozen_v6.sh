#!/usr/bin/env bash
# Bundle the romanrozen V6 submission as submission_rule_based_romanrozen_v6.tar.gz.

set -euo pipefail

OUT="${1:-submission_rule_based_romanrozen_v6.tar.gz}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

require() {
    [ -e "$1" ] || { echo "missing: $1" >&2; exit 1; }
}
require main_rule_based_romanrozen_v6.py
require deck_romanrozen_v6.csv
require scripts/rule_based_romanrozen_v6.py
require cg/__init__.py
require cg/api.py
require cg/game.py
require cg/sim.py
require cg/utils.py
require cg/libcg.so
require cg/cg.dll

deck_count=$(grep -c . deck_romanrozen_v6.csv)
if [ "$deck_count" -ne 60 ]; then
    echo "deck_romanrozen_v6.csv has $deck_count lines, expected 60" >&2
    exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp main_rule_based_romanrozen_v6.py "$STAGE/main.py"
cp deck_romanrozen_v6.csv "$STAGE/deck.csv"
cp scripts/rule_based_romanrozen_v6.py "$STAGE/rule_based_romanrozen_v6.py"
cp -r cg "$STAGE/cg"

mkdir -p "$(dirname "$OUT")"
( cd "$STAGE" && tar --owner=0 --group=0 \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -czvf "$ROOT/$OUT" \
    main.py \
    deck.csv \
    rule_based_romanrozen_v6.py \
    cg )

echo
echo "wrote: $OUT  ($(du -h "$OUT" | cut -f1))"
echo
tar -tzf "$OUT"

if [ "${SKIP_VERIFY:-0}" != "1" ]; then
    echo
    echo "verifying with scripts/check_main_exec.py..."
    if "$ROOT/scripts/run.sh" python3 "$ROOT/scripts/check_main_exec.py" --tar-gz "$OUT" --no-policy; then
        echo "verification: OK"
    else
        rc=$?
        echo "verification FAILED" >&2
        exit $rc
    fi
fi
