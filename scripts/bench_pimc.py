"""Bench the minimal 1-ply PIMC agent against the 6 meta-deck rule-based opps.

Uses our own deck (deck.csv) and assumes opponent runs Iono (the LB-prevalent
deck guess). Outputs same matchup matrix as bench_meta.py for direct compare.
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

import rule_based_abomasnow  # noqa: E402
import rule_based_agent  # noqa: E402
import rule_based_crustle  # noqa: E402
import rule_based_crustle_dashimaki  # noqa: E402
import rule_based_dragapult  # noqa: E402
import rule_based_iono  # noqa: E402
from kaggle_environments import make  # noqa: E402

from train.pimc_agent import make_pimc_agent  # noqa: E402


def _read_deck(path: Path) -> list[int]:
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def play(a, b):
    env = make("cabt")
    env.run([a, b])
    return env.steps[-1][0].reward, env.steps[-1][1].reward


def matchup(a, b, n: int, la: str, lb: str):
    random.seed(0)
    t0 = time.monotonic()
    aw = bw = d = 0
    for _ in range(n):
        r0, _ = play(a, b)
        if r0 == 1:
            aw += 1
        elif r0 == -1:
            bw += 1
        else:
            d += 1
    for _ in range(n):
        _, r1 = play(b, a)
        if r1 == 1:
            aw += 1
        elif r1 == -1:
            bw += 1
        else:
            d += 1
    dt = time.monotonic() - t0
    total = 2 * n
    print(f"  {la} vs {lb}: {aw}-{bw}-{d} ({aw / total:.1%}) in {dt:.1f}s")
    return aw, bw, d


def main_bench(n: int = 10):
    our_deck = _read_deck(ROOT / "deck.csv")
    iono_deck = _read_deck(ROOT / "deck_iono.csv")
    pimc = make_pimc_agent(our_deck, opp_deck_assumption=iono_deck, seed=0)

    opps = [
        ("Mega Lucario", rule_based_agent.agent),
        ("Dragapult ex", rule_based_dragapult.agent),
        ("Iono's", rule_based_iono.agent),
        ("Mega Abomasnow", rule_based_abomasnow.agent),
        ("Crustle Wall", rule_based_crustle.agent),
        ("Crustle Dashimaki", rule_based_crustle_dashimaki.agent),
    ]
    print(f"PIMC (1-ply, our deck, Iono assumed opp) vs Kiyota meta ({2 * n} games each)\n")
    rows = []
    for label, opp in opps:
        aw, bw, d = matchup(pimc, opp, n, "PIMC", label)
        rows.append((label, aw, bw, d))
    total_w = sum(r[1] for r in rows)
    total_l = sum(r[2] for r in rows)
    total = total_w + total_l + sum(r[3] for r in rows)
    print(f"\noverall: {total_w}-{total_l} ({total_w / total:.1%}) across {total} games")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main_bench(n)
