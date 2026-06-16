"""Smoke test + benchmark vs random.

Run: `python3 selfplay_test.py [N]`  (default N=8 games per side).
"""

from __future__ import annotations

import sys
import time

# litellm is broken on this image and breaks the werewolf env import.
sys.modules.setdefault("litellm", type(sys)("litellm"))

import random

from kaggle_environments import make

from agent import agent as my_agent
from agent import random_agent


def play(a, b):
    env = make("cabt")
    env.run([a, b])
    return env.steps[-1][0].reward, env.steps[-1][1].reward


def bench(n: int = 8) -> None:
    random.seed(0)
    t0 = time.monotonic()
    w0 = l0 = d = w1 = l1 = 0
    for _ in range(n):
        r0, _ = play(my_agent, random_agent)
        if r0 == 1:
            w0 += 1
        elif r0 == -1:
            l0 += 1
        else:
            d += 1
    for _ in range(n):
        _, r1 = play(random_agent, my_agent)
        if r1 == 1:
            w1 += 1
        elif r1 == -1:
            l1 += 1
        else:
            d += 1
    dt = time.monotonic() - t0
    print(f"{2 * n} games in {dt:.1f}s")
    print(f"  agent as P0 vs random: {w0}-{l0}")
    print(f"  agent as P1 vs random: {w1}-{l1}")
    print(f"  draws: {d}")
    return w0 + w1, l0 + l1, d


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    bench(n)
