"""Self-battle bench: two flavors of main.agent against each other.

By default this pits main.agent with POKEAI_PIMC=1 against main.agent with
POKEAI_PIMC=0 — the diagnostic we need to know whether PIMC's choices
are genuinely better than the linear-policy-with-engine-prior baseline.
vs random_agent the engine prior is hard to beat because random is so
weak; in a mirror match the comparison is fair.

Usage:
    scripts/run.sh python3 scripts/bench_selfbattle.py [N_per_side] [opp]

  opp:
    pimc-off (default)  main.agent flavor with PIMC disabled
    first               always-pick-index-0 baseline (the engine prior)
    random              random_agent (sanity check)
"""

from __future__ import annotations

import os
import random
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.modules.setdefault("litellm", types.ModuleType("litellm"))

from kaggle_environments import make  # noqa: E402

# We want main.agent in PIMC-on mode for "A" and PIMC-off for "B".
# Because main.py reads POKEAI_PIMC at import time, importing it twice
# would just reuse the cached module. So we instantiate the on/off flavors
# by importing once with PIMC=1, then patching the singleton flag at runtime.
os.environ["POKEAI_PIMC"] = "1"
import main  # noqa: E402
from agent import DECK, random_agent  # noqa: E402


def make_pimc_agent(on: bool):
    """Return an agent fn that calls main.agent with PIMC forced on/off."""

    def _agent(obs):
        original = main._PIMC_ENABLED
        main._PIMC_ENABLED = on
        try:
            return main.agent(obs)
        finally:
            main._PIMC_ENABLED = original

    return _agent


def first_agent(obs):
    """Engine-prior baseline: always pick the first option(s)."""
    sel = obs.get("select")
    if sel is None:
        return list(DECK)
    opts = sel.get("option") or []
    max_c = int(sel.get("maxCount") or 0)
    min_c = int(sel.get("minCount") or 0)
    if not opts or max_c == 0:
        return []
    k = max(min_c, 1) if max_c >= 1 else min_c
    k = min(k, max_c, len(opts))
    return list(range(k))


def play(a, b):
    env = make("cabt")
    env.run([a, b])
    return env.steps[-1][0].reward, env.steps[-1][1].reward


def bench(n: int, opp_name: str):
    a_on = make_pimc_agent(True)
    if opp_name == "pimc-off":
        b = make_pimc_agent(False)
    elif opp_name == "first":
        b = first_agent
    elif opp_name == "random":
        b = random_agent
    else:
        raise SystemExit(f"unknown opp: {opp_name}")

    random.seed(0)
    t0 = time.monotonic()
    w0 = l0 = d = w1 = l1 = 0
    for _ in range(n):
        r0, _ = play(a_on, b)
        if r0 == 1:
            w0 += 1
        elif r0 == -1:
            l0 += 1
        else:
            d += 1
    for _ in range(n):
        _, r1 = play(b, a_on)
        if r1 == 1:
            w1 += 1
        elif r1 == -1:
            l1 += 1
        else:
            d += 1
    dt = time.monotonic() - t0
    total_w = w0 + w1
    total_l = l0 + l1
    total = 2 * n
    print(f"PIMC-ON vs {opp_name}: {total} games in {dt:.1f}s ({dt / total:.2f}s/game)")
    print(f"  PIMC-ON as P0: {w0}-{l0}")
    print(f"  PIMC-ON as P1: {w1}-{l1}")
    print(f"  draws: {d}")
    print(f"  total: {total_w}-{total_l} ({total_w / total:.1%})")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    opp = sys.argv[2] if len(sys.argv) > 2 else "pimc-off"
    bench(n, opp)
