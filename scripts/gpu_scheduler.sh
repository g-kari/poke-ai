#!/usr/bin/env bash
# Overnight GPU scheduler: queue heavy experiments so the GPU stays busy
# while we're away. Each job writes its result to data/sweep/<job-id>/ and
# logs to /tmp/poke-ai-sweep.log.
#
# Phases (run sequentially — GPU is single-occupant):
#   1. Seed sweep:  V60 pool5 across multiple seeds (verify variance)
#   2. GA loop:     deck-builder evolution (1-card swap on the v4 top deck)
#   3. PIMC value:  self-distillation training (placeholder)
#
# Usage:
#   nohup scripts/gpu_scheduler.sh > /tmp/poke-ai-sweep.log 2>&1 &
#   tail -f /tmp/poke-ai-sweep.log
#
# Stop:
#   pkill -f gpu_scheduler.sh
#
# Designed to be safe to restart — each job creates a marker file when
# done; existing markers cause that job to be skipped.

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SWEEP_DIR="$ROOT/data/sweep"
mkdir -p "$SWEEP_DIR"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*"; }

run_job() {
    local job_id="$1"
    shift
    local marker="$SWEEP_DIR/${job_id}.done"
    if [ -f "$marker" ]; then
        log "SKIP $job_id (marker exists)"
        return 0
    fi
    log "START $job_id"
    if "$@"; then
        log "DONE  $job_id"
        touch "$marker"
    else
        log "FAIL  $job_id (exit $?)"
    fi
}

# ---------------------------------------------------------------------------
# Phase 1: Seed sweep — V60 pool5 across 4 seeds. Each run ~16 min on GPU.
# ---------------------------------------------------------------------------
seed_sweep() {
    local seed="$1"
    local out="$SWEEP_DIR/v60_seed${seed}.pt"
    local metrics="$SWEEP_DIR/v60_seed${seed}.json"
    "$ROOT/scripts/run.sh" python3 -m train.mlp_train_v60 \
        --episodes 2500 --lr 5e-4 --seed "$seed" \
        --opponent-pool "rule_based_iono,rule_based_crustle_dashimaki,rule_based_romanrozen_v6,rule_based_agent,rule_based_dragapult" \
        --out "$out" --metrics-out "$metrics" --log-every 500
}

# ---------------------------------------------------------------------------
# Phase 2: GA loop — evolve deck_builder_v4_top.csv by 1-card swap.
# Each generation evaluates a candidate deck @ 10g/opp × 5 opps = 100 games.
# Slow but deterministic; produces data/sweep/ga_history.json.
# ---------------------------------------------------------------------------
ga_loop() {
    "$ROOT/scripts/run.sh" python3 "$ROOT/scripts/ga_deck.py" \
        --seed-deck "$ROOT/deck_builder_v4_top.csv" \
        --generations 20 --games-per-eval 5 \
        --out "$SWEEP_DIR/ga_history.json"
}

# ---------------------------------------------------------------------------
# Phase 3: PIMC value-head self-distillation (placeholder skeleton).
# Implementation TBD next cycle; for now we just touch the marker.
# ---------------------------------------------------------------------------
pimc_value_train() {
    log "  (placeholder — real training script lands in next cycle)"
    sleep 1
}

log "GPU scheduler starting"

# Phase 1: 4 seeds × ~16 min ≈ 1h
for s in 7 13 21 99; do
    run_job "v60_seed_${s}" seed_sweep "$s"
done

# Phase 2: GA loop (~2-3h depending on game length).
run_job "ga_deck" ga_loop

# Phase 3: PIMC value training placeholder.
run_job "pimc_value" pimc_value_train

log "GPU scheduler done"
