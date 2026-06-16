"""Benchmark main.agent (the actual submission entrypoint, including PIMC)
against random_agent. Mirrors selfplay_test.py but uses main.agent instead
of agent.agent so that the PIMC look-ahead path is exercised.

Usage:
    scripts/run.sh python3 scripts/bench_main.py [N_per_side]
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

# Run from anywhere — make sure the repo root is on sys.path so `main` etc.
# resolve to the submission entrypoint rather than something on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.modules.setdefault("litellm", types.ModuleType("litellm"))

import random  # noqa: E402

from kaggle_environments import make  # noqa: E402

# Use main.agent — the actual submission entrypoint.
import main  # noqa: E402
from agent import random_agent  # noqa: E402


def play(a, b):
    env = make("cabt")
    env.run([a, b])
    return env.steps[-1][0].reward, env.steps[-1][1].reward


def bench(n: int = 20) -> None:
    random.seed(0)
    t0 = time.monotonic()
    w0 = l0 = d = w1 = l1 = 0
    for _ in range(n):
        r0, _ = play(main.agent, random_agent)
        if r0 == 1:
            w0 += 1
        elif r0 == -1:
            l0 += 1
        else:
            d += 1
    for _ in range(n):
        _, r1 = play(random_agent, main.agent)
        if r1 == 1:
            w1 += 1
        elif r1 == -1:
            l1 += 1
        else:
            d += 1
    dt = time.monotonic() - t0
    print(f"{2 * n} games in {dt:.1f}s ({dt / (2 * n):.2f}s per game)")
    print(f"  main.agent as P0 vs random: {w0}-{l0}")
    print(f"  main.agent as P1 vs random: {w1}-{l1}")
    print(f"  draws: {d}")
    total_w = w0 + w1
    total_l = l0 + l1
    total = 2 * n
    print(f"  total: {total_w}-{total_l} ({total_w / total:.1%})")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    bench(n)
