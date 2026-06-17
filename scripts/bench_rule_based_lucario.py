"""Bench the Kiyota rule-based Mega Lucario agent against the meta deck pool.

Why: our 3-MLP submission (LB 666.3) gets crushed vs Crustle Dashimaki (23.8%)
and Iono (11.2%). The rule-based Mega Lucario agent was previously measured
to beat the 4-Kiyota-meta lineup at ~70% (vs random; from NOTES.md). If it
also handles Crustle Dashimaki and Iono well, switching the submission from
"our deck + 3-MLP" to "Mega Lucario deck + rule-based" could be a major
LB jump.

Usage:
    scripts/run.sh python3 scripts/bench_rule_based_lucario.py [N_per_side]
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
os.environ["RULE_DECK_PATH"] = str(ROOT / "deck_mega_lucario.csv")
sys.modules.setdefault("litellm", types.ModuleType("litellm"))

import rule_based_abomasnow  # noqa: E402
import rule_based_agent  # noqa: E402  # Mega Lucario (subject)
import rule_based_crustle  # noqa: E402  # harukiharada Crustle Wall
import rule_based_crustle_dashimaki  # noqa: E402  # dashimaki Day-1 #1 Crustle
import rule_based_dragapult  # noqa: E402
import rule_based_iono  # noqa: E402
from kaggle_environments import make  # noqa: E402


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
    pct = a_wins / total
    print(f"  {la} vs {lb}: {a_wins}-{b_wins}-{draws} ({pct:.1%}) in {dt:.1f}s")
    return a_wins, b_wins, draws


def main_bench(n: int = 10):
    opps = [
        ("Mega Lucario (mirror)", rule_based_agent.agent),
        ("Dragapult ex", rule_based_dragapult.agent),
        ("Iono's", rule_based_iono.agent),
        ("Mega Abomasnow", rule_based_abomasnow.agent),
        ("Crustle Wall", rule_based_crustle.agent),
        ("Crustle Dashimaki", rule_based_crustle_dashimaki.agent),
    ]
    print(f"rule_based(Mega Lucario) vs Kiyota meta opponents ({2 * n} games each)\n")
    rows: list[tuple[str, int, int, int]] = []
    for label, opp in opps:
        a_w, b_w, d = matchup(rule_based_agent.agent, opp, n, "rule_based(Lucario)", label)
        rows.append((label, a_w, b_w, d))
    total_w = sum(r[1] for r in rows)
    total_l = sum(r[2] for r in rows)
    total = total_w + total_l + sum(r[3] for r in rows)
    print(f"\noverall: {total_w}-{total_l} ({total_w / total:.1%}) across {total} games")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main_bench(n)
