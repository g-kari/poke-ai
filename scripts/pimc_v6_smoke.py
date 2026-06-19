"""PIMC v6 smoke test: pick_best_option(MlpPolicy s500) on a mid-game obs.

Verifies the existing train/pimc.py works with our Mode B (PPO_v40 s500)
MlpPolicy as rollout policy, not just LinearPolicy.
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

policy = MlpPolicy.load("train/mlp_policy_ppo_v40_s100_t500.pt")  # Mode B

with open("deck.csv") as _f:
    DECK = [int(line.strip()) for line in _f if line.strip()]
rng = np.random.default_rng(42)


def policy_agent(obs):
    sel = obs.get("select")
    if sel is None:
        return list(DECK)
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


def smoke():
    print("=== PIMC v6 smoke test (PPO_v40 s500 MlpPolicy rollout) ===")
    print("  policy: PPO_v40 s500 (= Mode B, lab 18.6%, vs 3-MLP base 77.5%)")
    print(f"  deck: {len(DECK)} cards")
    print()

    # Capture a mid-game MAIN single-choice obs
    print("--- Capturing mid-game obs ---")
    captured = [None]
    step = [0]
    n_opts = [0]

    def capturing(obs):
        sel = obs.get("select")
        if sel is not None and sel.get("type") == 0:
            mc = int(sel.get("maxCount") or 0)
            opts = sel.get("option") or []
            if mc == 1 and len(opts) >= 2 and captured[0] is None and step[0] > 5:
                captured[0] = (obs, sel)
                n_opts[0] = len(opts)
        step[0] += 1
        return policy_agent(obs)

    env = make("cabt")
    env.run([capturing, policy_agent])

    if captured[0] is None:
        print("  ERROR: no MAIN single-choice obs captured")
        return False

    obs, sel = captured[0]
    print(f"  captured at step {step[0]}, n_options={n_opts[0]}")
    print()

    # Run pick_best_option with MlpPolicy
    print("--- Calling pick_best_option(MlpPolicy s500) ---")
    t0 = time.monotonic()
    try:
        result = pick_best_option(obs, sel, DECK, policy)
        elapsed_ms = (time.monotonic() - t0) * 1000
        print(f"  result: {result} (type={type(result).__name__})")
        print(f"  elapsed: {elapsed_ms:.0f} ms")

        if result is None:
            print("  → PIMC returned None (fall back to rollout policy)")
            print("  This may be expected (e.g., search_begin failed)")
            return False

        if not isinstance(result, int):
            print(f"  ERROR: expected int, got {type(result).__name__}")
            return False

        if result < 0 or result >= n_opts[0]:
            print(f"  ERROR: result {result} out of range [0, {n_opts[0]})")
            return False

        print(f"  → PIMC picked option {result} of {n_opts[0]} (= valid)")
    except Exception as e:
        elapsed_ms = (time.monotonic() - t0) * 1000
        print(f"  ERROR: {type(e).__name__}: {e}")
        print(f"  elapsed: {elapsed_ms:.0f} ms")
        return False

    print()
    print("=== PIMC v6 smoke test PASSED ===")
    return True


if __name__ == "__main__":
    sys.exit(0 if smoke() else 1)
