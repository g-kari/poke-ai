"""AlphaZero mirror match: AZ-ON s500 vs AZ-OFF s500, 100 games at n_sims=100.

補遺 23 (50g で 54%) の追試 — CI を ±14pp から ±10pp に狭める。
補遺 24 (n=200, 50g, 42%) を受けて、 n=100 が真に lift しているか
それとも noise だったかを区別。

期待: 100g で 50-55% 帯なら sweet spot 仮説、 45-50% なら noise 仮説。
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

sys.modules.setdefault("litellm", type(sys)("litellm"))
import numpy as np  # noqa: E402
from kaggle_environments import make  # noqa: E402

from train.alphazero import alphazero_choose  # noqa: E402
from train.mlp_policy import MlpPolicy  # noqa: E402

with open("deck.csv") as _f:
    DECK = [int(line.strip()) for line in _f if line.strip()]

policy = MlpPolicy.load("train/mlp_policy_ppo_v40_s100_t500.pt")  # Mode B s500
rng_on = np.random.default_rng(42)
rng_off = np.random.default_rng(42)
N_SIMS = 100


def fallback_pick(obs, sel, rng):
    """PPO_v40 s500 sampling (= AZ-OFF agent's behavior)."""
    opts = sel["option"]
    if not opts:
        return []
    max_c = int(sel.get("maxCount") or 0)
    min_c = int(sel.get("minCount") or 0)
    if sel["type"] == 0 and max_c == 1:
        probs = policy.probs(obs, sel)
        return [int(rng.choice(len(opts), p=probs))]
    if max_c >= 1:
        logits = policy.logits(obs, sel)
        order = np.argsort(-logits)
        k = max(min_c, 1)
        k = min(k, max_c, len(opts))
        return [int(x) for x in order[:k].tolist()]
    return []


def az_on_agent(obs):
    sel = obs.get("select")
    if sel is None:
        return list(DECK)
    opts = sel.get("option") or []
    max_c = int(sel.get("maxCount") or 0)
    if sel.get("type") == 0 and max_c == 1 and len(opts) >= 2:
        result = alphazero_choose(obs, DECK, policy, rng_on, n_sims=N_SIMS)
        if result is not None:
            return result
    return fallback_pick(obs, sel, rng_on)


def az_off_agent(obs):
    sel = obs.get("select")
    if sel is None:
        return list(DECK)
    return fallback_pick(obs, sel, rng_off)


def main():
    N = 100
    on_w = off_w = draws = 0
    t0 = time.monotonic()
    print(f"AlphaZero mirror match (AZ-ON vs AZ-OFF, both s500), {N} games")
    print("  policy: PPO_v40 s500 (= Mode B)")
    print(f"  n_sims: {N_SIMS}")
    print()

    for i in range(N):
        env = make("cabt")
        if i < N // 2:
            env.run([az_on_agent, az_off_agent])
            r = env.steps[-1][0].reward
            if r == 1:
                on_w += 1
            elif r == -1:
                off_w += 1
            else:
                draws += 1
        else:
            env.run([az_off_agent, az_on_agent])
            r = env.steps[-1][0].reward
            if r == 1:
                off_w += 1
            elif r == -1:
                on_w += 1
            else:
                draws += 1
        elapsed = time.monotonic() - t0
        print(
            f"  game {i + 1}: ON {on_w} / OFF {off_w} / D {draws}  "
            f"[{elapsed:.0f}s = {elapsed / (i + 1):.1f}s/game]",
            flush=True,
        )

    elapsed = time.monotonic() - t0
    print()
    total = on_w + off_w + draws
    print(f"Final ({N} games, {elapsed:.0f}s = {elapsed / N:.1f}s/game):")
    print(f"  AZ-ON wins: {on_w}/{total} ({on_w / total:.1%})")
    print(f"  AZ-OFF wins: {off_w}/{total} ({off_w / total:.1%})")
    print(f"  Draws: {draws}/{total} ({draws / total:.1%})")
    decisive = on_w + off_w
    if decisive > 0:
        print(f"  AZ-ON in decisive: {on_w}/{decisive} ({on_w / decisive:.1%})")


if __name__ == "__main__":
    main()
