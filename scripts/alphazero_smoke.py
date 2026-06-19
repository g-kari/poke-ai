"""AlphaZero MCTS smoke test (Phase 2 verification).

Captures a mid-game MAIN single-choice obs and runs alphazero_choose with
PPO_v40 s500 policy + calibrated value head.
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

policy = MlpPolicy.load("train/mlp_policy_ppo_v40_s100_t500.pt")

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


def main():
    print("=== AlphaZero MCTS smoke test ===")
    print("  policy: PPO_v40 s500 (Mode B)")
    print(f"  deck: {len(DECK)} cards")
    print()

    # Capture a mid-game MAIN single-choice obs
    print("--- Capturing mid-game obs ---")
    captured = [None]
    step_box = [0]

    def make_capturing():
        def _agent(obs):
            sel = obs.get("select")
            cur_step = step_box[0]
            if (
                sel is not None
                and sel.get("type") == 0
                and int(sel.get("maxCount") or 0) == 1
                and len(sel.get("option") or []) >= 2
                and captured[0] is None
                and cur_step > 5
            ):
                captured[0] = obs
            step_box[0] = cur_step + 1
            return policy_agent(obs)

        return _agent

    env = make("cabt")
    env.run([make_capturing(), policy_agent])

    if captured[0] is None:
        print("  ERROR: no MAIN single-choice obs captured")
        return 1

    obs = captured[0]
    n_opts = len(obs["select"]["option"])
    print(f"  captured at step {step_box[0]}, n_options={n_opts}")
    print()

    for n_sims in (5, 20):
        print(f"--- alphazero_choose(n_sims={n_sims}) ---")
        rng2 = np.random.default_rng(0)
        t0 = time.monotonic()
        try:
            result = alphazero_choose(obs, DECK, policy, rng2, n_sims=n_sims)
            elapsed_ms = (time.monotonic() - t0) * 1000
            print(f"  result: {result}")
            print(f"  elapsed: {elapsed_ms:.0f} ms ({elapsed_ms / n_sims:.0f} ms/sim)")
            if result is None:
                print("  → returned None (fallback to policy)")
            elif not isinstance(result, list) or len(result) != 1:
                print(f"  → unexpected shape: {result}")
            else:
                idx = result[0]
                if 0 <= idx < n_opts:
                    print(f"  → picked option {idx} of {n_opts} (= valid)")
                else:
                    print("  → out of range")
        except Exception as e:  # noqa: BLE001
            elapsed_ms = (time.monotonic() - t0) * 1000
            print(f"  EXCEPTION: {type(e).__name__}: {e}")
            print(f"  elapsed: {elapsed_ms:.0f} ms")
            import traceback

            traceback.print_exc()
        print()

    print("=== AlphaZero smoke complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
