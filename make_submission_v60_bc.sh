#!/usr/bin/env bash
# Bundle the V60 BC (Behavioral Cloning from V6) submission as
# submission_v60_bc.tar.gz.
#
# Layout: identical to make_submission_v60.sh except only the BC pt/npz
# is shipped — main_v60.py picks the lexically-last .npz, so we must
# avoid shipping ext3 alongside BC.
#
# Usage:
#   ./make_submission_v60_bc.sh                                    # default BC v1 weights
#   ./make_submission_v60_bc.sh out/foo.tar.gz                     # custom tar path
#   BC_PT=train/mlp_policy_v60_bc_v2.pt ./make_submission_v60_bc.sh # use v2 weights

set -euo pipefail
OUT="${1:-submission_v60_bc.tar.gz}"
BC_PT="${BC_PT:-train/mlp_policy_v60_bc.pt}"
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
require train/mlp_policy_v60_numpy.py
require "$BC_PT"
require scripts/extract_v60_weights.py

# Extract BC weights to npz.
BC_NPZ="${BC_PT%.pt}.npz"
if [ ! -e "$BC_NPZ" ] || [ "$BC_PT" -nt "$BC_NPZ" ]; then
    echo "extracting $BC_PT -> $BC_NPZ"
    "$ROOT/scripts/run.sh" python3 "$ROOT/scripts/extract_v60_weights.py" \
        --pt "$BC_PT" --out "$BC_NPZ"
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
cp train/mlp_policy_v60_numpy.py "$STAGE/train/"
# Only ship the BC npz so main_v60.py's lexically-last glob picks it.
cp "$BC_NPZ" "$STAGE/train/"

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
