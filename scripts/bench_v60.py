"""Bench a V60 policy weight file vs the 7 vendored opponents.

Usage:
    scripts/run.sh python3 scripts/bench_v60.py \\
        --weights train/mlp_policy_v60_ext3.pt --games 20

20 games per side = 40g per opp = 280g total. Approx 90-120s on CPU.
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


def make_policy_agent(policy: MlpPolicyV60, deck: list[int]):
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
        if sel.get("type") == 0 and max_c == 1:
            p = policy.probs(obs, sel)
            return [int(rng.choice(len(opts), p=p))]
        if max_c >= 1:
            logits = policy.logits(obs, sel)
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


def bench(policy_agent, opp, n: int) -> tuple[int, int, int]:
    random.seed(0)
    aw = bw = d = 0
    for _ in range(n):
        r0, _ = play(policy_agent, opp)
        aw += r0 == 1
        bw += r0 == -1
        d += r0 == 0
    for _ in range(n):
        _, r1 = play(opp, policy_agent)
        aw += r1 == 1
        bw += r1 == -1
        d += r1 == 0
    return aw, bw, d


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--games", type=int, default=20, help="games per side, default 20 (= 40g/opp)")
    args = p.parse_args()

    with open(ROOT / "deck.csv") as f:
        deck = [int(x.strip()) for x in f if x.strip()]
    policy = MlpPolicyV60.load(str(args.weights))
    policy.eval()
    pa = make_policy_agent(policy, deck)

    print(f"V60 bench: {args.weights} @ {2 * args.games}g/opp")
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
