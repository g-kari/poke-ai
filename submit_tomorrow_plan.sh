#!/usr/bin/env bash
# Tomorrow's 4-slot UTC-reset submission plan (2026-06-20 00:00 UTC).
#
# Usage:
#   bash submit_tomorrow_plan.sh            # dry-run: print commands
#   bash submit_tomorrow_plan.sh --execute  # actually submit (use AFTER UTC reset)
#
# 残り 1 slot は LB 着地点を見てから adaptive submit (= このスクリプトでは
# 投入しない)。 LB の表示は数十分~数時間遅れがあるので、 1 件 submit 後に
# 5-10 分待ってから次を実行する運用が安全。
#
# 4 枠投入の目的:
#   1. ratio 35 仮説の検証 (= 23% 級 PPO が LB 815 に届くか)
#   2. matchup specialization 効果の測定 (= s100 vs s2026 で LB 差)
#   3. lab → LB 線形性の校正 (= mid-tier s500 を加える)
#   4. baseline 安定性の確認 (= 3-MLP base の re-submit)

set -euo pipefail

DRYRUN=true
if [ "${1:-}" = "--execute" ]; then
    DRYRUN=false
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Verify bundles exist before any submit
for bundle in submission_ppo_v40_s100.tar.gz submission_ppo_v40_s2026.tar.gz \
              submission_ppo_v40_s500.tar.gz submission_3mlp_base.tar.gz; do
    [ -e "$bundle" ] || { echo "missing: $bundle" >&2; exit 1; }
done

declare -a PLANS=(
    "submission_ppo_v40_s100.tar.gz|PPO_v40 seed=100 single: lab 23.3% (n=4 PEAK, Crustle Wall 特化, 700g). Test ratio 35 hypothesis - expected LB ~815."
    "submission_ppo_v40_s2026.tar.gz|PPO_v40 seed=2026 single: lab 22.9% (n=4 2nd PEAK, V6 特化, 700g). Specialization-vs-s100 test - expected LB ~800."
    "submission_ppo_v40_s500.tar.gz|PPO_v40 seed=500 single: lab 18.6% (n=4 median mid-tier, 700g). Linearity calibration - expected LB ~650."
    "submission_3mlp_base.tar.gz|3-MLP base re-submit: settling control (known LB 679.6 @ 6/17, ratio 35.9, lab 18.9% @ 700g). Confirm baseline stability."
)

i=1
for plan in "${PLANS[@]}"; do
    bundle="${plan%%|*}"
    msg="${plan#*|}"
    echo "=== Slot $i: $bundle ==="
    echo "    msg: $msg"
    if [ "$DRYRUN" = true ]; then
        echo "    [DRY RUN] .venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle -f $bundle"
    else
        .venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \
            -f "$bundle" -m "$msg"
        echo "    submitted. Wait 5-10 min and check LB before next slot."
        if [ "$i" -lt "${#PLANS[@]}" ]; then
            read -p "    Press Enter to continue to next slot..."
        fi
    fi
    echo
    i=$((i + 1))
done

if [ "$DRYRUN" = true ]; then
    echo "Dry run complete. Re-run with --execute AFTER UTC reset (2026-06-20 00:00 UTC)."
else
    echo "4 slots submitted. Remaining 1 slot: adaptive based on LB landings."
fi
