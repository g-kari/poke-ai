"""Three-way bench: our linear agent vs the Mega Lucario rule-based agent.

The Mega Lucario ex deck and the rule-based policy come from kaggle.com/code/
kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck. The deck is saved
as deck_mega_lucario.csv and the agent as scripts/rule_based_agent.py.

Three configurations are tested in 20-game match-ups (10 per side swap):
  1. rule_based (Mega Lucario) vs linear_agent (our deck.csv)
  2. rule_based (Mega Lucario) vs random_agent
  3. linear_agent (our deck.csv) vs random_agent  (sanity-check baseline)

Usage:
    scripts/run.sh python3 scripts/bench_rule_based.py [N_per_side]
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
# Point the rule-based agent at the Mega Lucario deck via env var.
os.environ.setdefault("RULE_DECK_PATH", str(ROOT / "deck_mega_lucario.csv"))
sys.modules.setdefault("litellm", types.ModuleType("litellm"))

import rule_based_agent  # noqa: E402  – Mega Lucario rule-based agent
from kaggle_environments import make  # noqa: E402

import main  # noqa: E402  – our linear agent
from agent import random_agent  # noqa: E402


def play(a, b):
    env = make("cabt")
    env.run([a, b])
    return env.steps[-1][0].reward, env.steps[-1][1].reward


def matchup(a, b, n: int, label_a: str, label_b: str):
    random.seed(0)
    t0 = time.monotonic()
    a_wins = b_wins = draws = 0
    for _ in range(n):
        r0, r1 = play(a, b)
        if r0 == 1:
            a_wins += 1
        elif r0 == -1:
            b_wins += 1
        else:
            draws += 1
    for _ in range(n):
        r0, r1 = play(b, a)
        if r0 == 1:
            b_wins += 1
        elif r0 == -1:
            a_wins += 1
        else:
            draws += 1
    dt = time.monotonic() - t0
    total = 2 * n
    print(f"{label_a} vs {label_b}: {total} games in {dt:.1f}s ({dt / total:.2f}s/game)")
    print(f"  {label_a}: {a_wins}-{b_wins}-{draws}  ({a_wins / total:.1%})")


def main_bench(n: int = 10):
    matchup(rule_based_agent.agent, main.agent, n, "rule_based(Lucario)", "linear(ours)")
    matchup(rule_based_agent.agent, random_agent, n, "rule_based(Lucario)", "random")
    matchup(main.agent, random_agent, n, "linear(ours)", "random")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main_bench(n)
