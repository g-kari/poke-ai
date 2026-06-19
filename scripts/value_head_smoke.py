"""Value head smoke test: verify PPO_v40 s500 value head returns sensible V(state).

Plays a few games, captures observations across game phases (early/mid/late),
checks V(obs) is in [-1, +1] and tracks the eventual outcome.

If V predicts the eventual outcome better than chance, value head is usable
for AlphaZero MCTS (= Phase 2). If V is uniformly ~0, value head learned
nothing useful and we need to train it separately (= Phase 1).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

sys.modules.setdefault("litellm", type(sys)("litellm"))
import numpy as np  # noqa: E402
from kaggle_environments import make  # noqa: E402

from train.mlp_policy import MlpPolicy  # noqa: E402

with open("deck.csv") as _f:
    DECK = [int(line.strip()) for line in _f if line.strip()]

policy = MlpPolicy.load("train/mlp_policy_ppo_v40_s100_t500.pt")  # Mode B
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
    print("=== Value head smoke test (PPO_v40 s500) ===")
    print()

    # Run 5 games, capture V at early/mid/late, compare to outcome
    N_GAMES = 5
    samples = []  # list of (phase, V, outcome)

    def make_capturing(captured_list, step_box):
        def _agent(obs):
            sel = obs.get("select")
            cur_step = step_box[0]
            if (
                sel is not None
                and obs.get("current") is not None
                and sel.get("type") == 0
                and cur_step % 5 == 0
            ):
                try:
                    v = policy.value(obs)
                    captured_list.append((cur_step, v))
                except Exception as e:  # noqa: BLE001
                    captured_list.append((cur_step, None, str(e)))
            step_box[0] = cur_step + 1
            return policy_agent(obs)

        return _agent

    for game_idx in range(N_GAMES):
        env = make("cabt")
        captured = []
        step = [0]
        capturing = make_capturing(captured, step)
        env.run([capturing, policy_agent])
        # outcome: reward of P0
        outcome = env.steps[-1][0].reward  # +1/0/-1
        print(f"Game {game_idx + 1}: outcome P0={outcome:+d}, total steps={step[0]}")
        for s, v in captured:
            phase = "early" if s < 10 else ("mid" if s < 30 else "late")
            agree = "✓" if (v > 0 and outcome > 0) or (v < 0 and outcome < 0) else "✗"
            print(f"  step {s:3d} ({phase}): V={v:+.3f}  outcome={outcome:+d}  {agree}")
            samples.append((phase, v, outcome))

    # Aggregate stats
    print()
    print("=== Aggregate stats ===")
    if samples:
        Vs = np.array([s[1] for s in samples])
        Os = np.array([s[2] for s in samples])
        print(f"  N samples: {len(samples)}")
        print(
            f"  V range: [{Vs.min():.3f}, {Vs.max():.3f}], mean: {Vs.mean():+.3f}, std: {Vs.std():.3f}"
        )
        # Correlation with outcome
        correct = np.sum((Vs > 0) == (Os > 0))
        print(
            f"  V sign matches outcome sign: {correct}/{len(samples)} ({correct / len(samples):.1%})"
        )
        # Per-phase
        for phase in ["early", "mid", "late"]:
            p_samples = [s for s in samples if s[0] == phase]
            if p_samples:
                p_Vs = np.array([s[1] for s in p_samples])
                p_Os = np.array([s[2] for s in p_samples])
                p_correct = np.sum((p_Vs > 0) == (p_Os > 0))
                print(
                    f"  {phase}: N={len(p_samples)}, V mean={p_Vs.mean():+.3f}, "
                    f"sign match={p_correct}/{len(p_samples)} ({p_correct / len(p_samples):.0%})"
                )


if __name__ == "__main__":
    main()
