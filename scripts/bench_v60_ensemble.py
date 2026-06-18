"""Bench an ensemble of V60 policies (logit average) vs the 7 opponents.

Different policies trained on different objectives/data may have
anti-correlated weaknesses — averaging their logits can give a few
percentage points without retraining.

Usage:
    scripts/run.sh python3 scripts/bench_v60_ensemble.py \\
        --weights train/mlp_policy_v60_ext3.pt,train/mlp_policy_v60_bcrl1.pt \\
        --games 20
"""

from __future__ import annotations

import argparse
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

import numpy as np  # noqa: E402
import rule_based_abomasnow  # noqa: E402
import rule_based_agent  # noqa: E402
import rule_based_crustle  # noqa: E402
import rule_based_crustle_dashimaki  # noqa: E402
import rule_based_dragapult  # noqa: E402
import rule_based_iono  # noqa: E402
import rule_based_romanrozen_v6  # noqa: E402
from kaggle_environments import make  # noqa: E402

from train.mlp_policy_v60 import MlpPolicyV60  # noqa: E402

OPPS = [
    ("Mega Lucario", rule_based_agent.agent),
    ("Dragapult ex", rule_based_dragapult.agent),
    ("Iono", rule_based_iono.agent),
    ("Mega Abomasnow", rule_based_abomasnow.agent),
    ("Crustle Wall", rule_based_crustle.agent),
    ("Crustle Dashi", rule_based_crustle_dashimaki.agent),
    ("V6", rule_based_romanrozen_v6.agent),
]


def make_ensemble_agent(policies: list[MlpPolicyV60], deck: list[int]):
    rng = np.random.default_rng(0)

    def agent(obs):
        sel = obs.get("select")
        if sel is None:
            return list(deck)
        opts = sel.get("option") or []
        if not opts:
            return []
        max_c = int(sel.get("maxCount") or 0)
        min_c = int(sel.get("minCount") or 0)
        if max_c == 0:
            return []
        # Average logits across all member policies.
        logits_sum = None
        for p in policies:
            ls = p.logits(obs, sel)
            logits_sum = ls if logits_sum is None else logits_sum + ls
        logits = logits_sum / len(policies)
        if sel.get("type") == 0 and max_c == 1:
            z = logits - logits.max()
            e = np.exp(z)
            probs = e / e.sum()
            return [int(rng.choice(len(opts), p=probs))]
        if max_c >= 1:
            order = np.argsort(-logits)
            k = max(min_c, 1)
            k = min(k, max_c, len(opts))
            return [int(x) for x in order[:k].tolist()]
        return []

    return agent


def play(a, b):
    env = make("cabt")
    env.run([a, b])
    return env.steps[-1][0].reward, env.steps[-1][1].reward


def bench(ensemble_agent, opp, n: int) -> tuple[int, int, int]:
    random.seed(0)
    aw = bw = d = 0
    for _ in range(n):
        r0, _ = play(ensemble_agent, opp)
        aw += r0 == 1
        bw += r0 == -1
        d += r0 == 0
    for _ in range(n):
        _, r1 = play(opp, ensemble_agent)
        aw += r1 == 1
        bw += r1 == -1
        d += r1 == 0
    return aw, bw, d


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Comma-separated paths to V60 .pt weights")
    p.add_argument("--games", type=int, default=20, help="games per side, default 20")
    args = p.parse_args()

    paths = [s.strip() for s in args.weights.split(",") if s.strip()]
    if len(paths) < 2:
        print("ensemble needs >= 2 weights", file=sys.stderr)
        return 1

    with open(ROOT / "deck.csv") as f:
        deck = [int(x.strip()) for x in f if x.strip()]

    policies = []
    for path in paths:
        policy = MlpPolicyV60.load(path)
        policy.eval()
        policies.append(policy)
        print(f"  loaded {path}")

    pa = make_ensemble_agent(policies, deck)
    print(f"\nensemble bench ({len(policies)} policies) @ {2 * args.games}g/opp")
    t0 = time.monotonic()
    total_w = total_l = 0
    for label, opp in OPPS:
        aw, bw, d = bench(pa, opp, args.games)
        n = 2 * args.games
        print(f"  vs {label:18s}: {aw}-{bw}-{d}  ({aw / n:.1%})")
        total_w += aw
        total_l += bw
    total = total_w + total_l
    print(
        f"\noverall: {total_w}-{total_l} ({total_w / total:.1%}) "
        f"across {len(OPPS) * 2 * args.games}g  [{time.monotonic() - t0:.0f}s]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
