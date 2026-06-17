"""Bench the MLP policy against linear / rule-based / random.

Loads train/mlp_policy.pt and runs side-swapped match-ups so each
configuration plays both P0 and P1. The linear opponent loads through
the existing main.agent, so it picks up POKEAI_PIMC if set.

Usage:
    scripts/run.sh python3 scripts/bench_mlp.py [N_per_side]
"""

from __future__ import annotations

import os
import random
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("RULE_DECK_PATH", str(ROOT / "deck_mega_lucario.csv"))
sys.modules.setdefault("litellm", types.ModuleType("litellm"))

import rule_based_agent  # noqa: E402
from kaggle_environments import make  # noqa: E402

import main  # noqa: E402 — linear baseline
from agent import random_agent  # noqa: E402
from train.mlp_policy import MlpPolicy  # noqa: E402

POLICY_PATH = ROOT / "train" / "mlp_policy.pt"


def make_mlp_agent():
    policy = MlpPolicy.try_load(str(POLICY_PATH))
    if policy is None:
        raise SystemExit(f"could not load {POLICY_PATH}")
    deck_path = ROOT / "deck.csv"
    with open(deck_path) as f:
        deck = [int(line.strip()) for line in f if line.strip()]

    def _agent(obs):
        sel = obs.get("select")
        if sel is None:
            return list(deck)
        opts = sel.get("option") or []
        max_c = int(sel.get("maxCount") or 0)
        min_c = int(sel.get("minCount") or 0)
        if not opts or max_c == 0:
            return []
        import numpy as np  # noqa: PLC0415

        logits = policy.logits(obs, sel)
        order = np.argsort(-logits)
        k = max(min_c, 1) if max_c >= 1 else min_c
        k = min(k, max_c, len(opts))
        return [int(x) for x in order[:k].tolist()]

    return _agent


def play(a, b):
    env = make("cabt")
    env.run([a, b])
    return env.steps[-1][0].reward, env.steps[-1][1].reward


def matchup(a, b, n: int, la: str, lb: str):
    random.seed(0)
    t0 = time.monotonic()
    a_wins = b_wins = draws = 0
    for _ in range(n):
        r0, _ = play(a, b)
        if r0 == 1:
            a_wins += 1
        elif r0 == -1:
            b_wins += 1
        else:
            draws += 1
    for _ in range(n):
        _, r1 = play(b, a)
        if r1 == 1:
            a_wins += 1
        elif r1 == -1:
            b_wins += 1
        else:
            draws += 1
    dt = time.monotonic() - t0
    total = 2 * n
    print(f"{la} vs {lb}: {total} games in {dt:.1f}s ({dt / total:.2f}s/game)")
    print(f"  {la}: {a_wins}-{b_wins}-{draws}  ({a_wins / total:.1%})")


def main_bench(n: int = 20):
    mlp = make_mlp_agent()
    matchup(mlp, main.agent, n, "mlp(ours)", "linear(ours)")
    matchup(mlp, rule_based_agent.agent, n, "mlp(ours)", "rule_based(Lucario)")
    matchup(mlp, random_agent, n, "mlp(ours)", "random")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main_bench(n)
