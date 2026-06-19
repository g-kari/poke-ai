"""PIMC v6 mirror match: PIMC-ON s500 vs PIMC-OFF s500, 50 games.

Uses existing train/pimc.pick_best_option with PPO_v40 s500 MlpPolicy
as rollout. Compares to PIMC-OFF (= same policy, no lookahead).

Existing PIMC v5 (linear rollout) showed 51-49 tie at 100 games.
This test checks if PPO_v40 s500 rollout (= Mode B, 3-MLP base に 77.5%
で勝つ強い policy) gives a different result.

50 games chosen for time budget (~20-40 min). Wilson 95% CI ±14pp.
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

from train.mlp_policy import MlpPolicy  # noqa: E402
from train.pimc import pick_best_option  # noqa: E402

with open("deck.csv") as _f:
    DECK = [int(line.strip()) for line in _f if line.strip()]

policy = MlpPolicy.load("train/mlp_policy_ppo_v40_s100_t500.pt")  # Mode B s500
rng_pimc = np.random.default_rng(42)
rng_off = np.random.default_rng(42)


def fallback_pick(obs, sel, rng):
    """Same logic as PIMC-OFF agent (policy.probs sampling for MAIN)."""
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


def pimc_on_agent(obs):
    sel = obs.get("select")
    if sel is None:
        return list(DECK)
    opts = sel.get("option") or []
    max_c = int(sel.get("maxCount") or 0)
    # PIMC for MAIN single-choice with >=2 options
    if sel.get("type") == 0 and max_c == 1 and len(opts) >= 2:
        idx = pick_best_option(obs, sel, DECK, policy)
        if idx is not None:
            return [idx]
    return fallback_pick(obs, sel, rng_pimc)


def pimc_off_agent(obs):
    sel = obs.get("select")
    if sel is None:
        return list(DECK)
    return fallback_pick(obs, sel, rng_off)


def main():
    N = 50
    on_w = off_w = draws = 0
    t0 = time.monotonic()
    print(f"PIMC v6 mirror match (PIMC-ON vs PIMC-OFF, both s500), {N} games")
    print("  rollout policy: PPO_v40 s500 (= Mode B)")
    print()

    for i in range(N):
        env = make("cabt")
        if i < N // 2:
            env.run([pimc_on_agent, pimc_off_agent])
            r = env.steps[-1][0].reward
            if r == 1:
                on_w += 1
            elif r == -1:
                off_w += 1
            else:
                draws += 1
        else:
            env.run([pimc_off_agent, pimc_on_agent])
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
    print(f"  PIMC-ON wins: {on_w}/{total} ({on_w / total:.1%})")
    print(f"  PIMC-OFF wins: {off_w}/{total} ({off_w / total:.1%})")
    print(f"  Draws: {draws}/{total} ({draws / total:.1%})")
    decisive = on_w + off_w
    if decisive > 0:
        print(f"  PIMC-ON in decisive: {on_w}/{decisive} ({on_w / decisive:.1%})")


if __name__ == "__main__":
    main()
