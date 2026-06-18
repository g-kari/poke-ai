"""Bench an arbitrary rule-based agent (subject) against the full meta pool.

Why: 2026-06-18 bench discovered rule_based(Mega Lucario) gets 46.5% overall
across the 6-opp meta pool (vs our 3-MLP submission at 23.3%). This script
lets us test alternate "subject" agents (rule_based_dragapult,
rule_based_iono, etc.) to find which deck/bot pair gives the strongest
overall lab score — that becomes the next LB submit candidate.

Usage:
    scripts/run.sh python3 scripts/bench_rule_based_lucario.py [N_per_side] [subject]

subject = lucario | dragapult | iono | abomasnow | crustle | crustle_dashimaki
(default: lucario)
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
# rule_based_agent (Mega Lucario) reads RULE_DECK_PATH; the other Kiyota
# bots each set their own deck path via env vars defined in their modules
# (e.g. RULE_DECK_PATH_DRAGAPULT). Setting all of them up-front lets us
# load every subject from the same process without re-importing.
os.environ["RULE_DECK_PATH"] = str(ROOT / "deck_mega_lucario.csv")
sys.modules.setdefault("litellm", types.ModuleType("litellm"))

import rule_based_abomasnow  # noqa: E402
import rule_based_agent  # noqa: E402  # Mega Lucario (subject)
import rule_based_crustle  # noqa: E402  # harukiharada Crustle Wall
import rule_based_crustle_dashimaki  # noqa: E402  # dashimaki Day-1 #1 Crustle
import rule_based_dragapult  # noqa: E402
import rule_based_iono  # noqa: E402
import rule_based_kojimar  # noqa: E402  # kojimar validated Lucario+Hariyama
import rule_based_romanrozen_v6  # noqa: E402  # romanrozen V6 LB 860+ hybrid
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


SUBJECTS = {
    "lucario": ("rule_based(Lucario)", rule_based_agent.agent),
    "dragapult": ("rule_based(Dragapult)", rule_based_dragapult.agent),
    "iono": ("rule_based(Iono)", rule_based_iono.agent),
    "abomasnow": ("rule_based(Abomasnow)", rule_based_abomasnow.agent),
    "crustle": ("rule_based(CrustleWall)", rule_based_crustle.agent),
    "crustle_dashimaki": (
        "rule_based(CrustleDashimaki)",
        rule_based_crustle_dashimaki.agent,
    ),
    "romanrozen": ("rule_based(RomanrozenV6)", rule_based_romanrozen_v6.agent),
    "kojimar": ("rule_based(Kojimar)", rule_based_kojimar.agent),
}


def main_bench(n: int = 10, subject_key: str = "lucario"):
    if subject_key not in SUBJECTS:
        print(f"unknown subject {subject_key!r}; pick one of {list(SUBJECTS)}", file=sys.stderr)
        sys.exit(2)
    subject_label, subject_fn = SUBJECTS[subject_key]
    opps = [
        ("Mega Lucario", rule_based_agent.agent),
        ("Dragapult ex", rule_based_dragapult.agent),
        ("Iono's", rule_based_iono.agent),
        ("Mega Abomasnow", rule_based_abomasnow.agent),
        ("Crustle Wall", rule_based_crustle.agent),
        ("Crustle Dashimaki", rule_based_crustle_dashimaki.agent),
    ]
    print(f"{subject_label} vs Kiyota meta opponents ({2 * n} games each)\n")
    rows: list[tuple[str, int, int, int]] = []
    for label, opp in opps:
        a_w, b_w, d = matchup(subject_fn, opp, n, subject_label, label)
        rows.append((label, a_w, b_w, d))
    total_w = sum(r[1] for r in rows)
    total_l = sum(r[2] for r in rows)
    total = total_w + total_l + sum(r[3] for r in rows)
    print(f"\noverall: {total_w}-{total_l} ({total_w / total:.1%}) across {total} games")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    subject = sys.argv[2] if len(sys.argv) > 2 else "lucario"
    main_bench(n, subject)
