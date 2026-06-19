"""Value head calibration (AlphaZero Phase 1a).

Measures the systematic bias in PPO_v40 s500's value head by playing N
games, recording V(state) across game phases, and comparing to terminal
outcomes. Saves bias statistics as JSON for use during inference.

Output: train/value_calibration.json

  {
    "policy": "mlp_policy_ppo_v40_s100_t500.pt",
    "n_games": 20,
    "n_samples": 75,
    "v_mean": -0.30,
    "v_median": -0.28,
    "outcome_mean": 0.5,
    "bias_mean": -0.80,  // v_mean - outcome_mean
    "bias_median": -0.78,
    "per_phase": {
      "early": {...}, "mid": {...}, "late": {...}
    }
  }
"""

import json
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

POLICY_PATH = "train/mlp_policy_ppo_v40_s100_t500.pt"
OUT_PATH = "train/value_calibration.json"
N_GAMES = 20

policy = MlpPolicy.load(POLICY_PATH)
rng = np.random.default_rng(42)

with open("deck.csv") as _f:
    DECK = [int(line.strip()) for line in _f if line.strip()]


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


def phase_of(step, total):
    """Bucket step into early/mid/late based on fraction of game length."""
    if total <= 0:
        return "mid"
    frac = step / total
    if frac < 0.33:
        return "early"
    if frac < 0.67:
        return "mid"
    return "late"


def make_capturing(captured_list, step_box):
    def _agent(obs):
        sel = obs.get("select")
        cur_step = step_box[0]
        if (
            sel is not None
            and obs.get("current") is not None
            and sel.get("type") == 0
            and cur_step % 3 == 0  # every 3rd MAIN obs
        ):
            try:
                v = policy.value(obs)
                captured_list.append((cur_step, float(v)))
            except Exception:  # noqa: BLE001
                pass
        step_box[0] = cur_step + 1
        return policy_agent(obs)

    return _agent


def main():
    print("=== Value head calibration ===")
    print(f"  policy: {POLICY_PATH}")
    print(f"  games: {N_GAMES}, capture every 3rd MAIN obs")
    print()

    samples = []  # list of (step, v, total_steps, outcome)

    for game_idx in range(N_GAMES):
        env = make("cabt")
        captured = []
        step = [0]
        capturing = make_capturing(captured, step)
        env.run([capturing, policy_agent])
        total = step[0]
        outcome = float(env.steps[-1][0].reward)
        for s, v in captured:
            samples.append((s, v, total, outcome))
        print(
            f"  game {game_idx + 1}/{N_GAMES}: outcome={outcome:+.0f}, "
            f"steps={total}, samples={len(captured)}",
            flush=True,
        )

    print()
    print("=== Stats ===")
    if not samples:
        print("  No samples collected!")
        return 1

    Vs = np.array([s[1] for s in samples])
    Os = np.array([s[3] for s in samples])
    v_mean = float(Vs.mean())
    v_median = float(np.median(Vs))
    o_mean = float(Os.mean())
    bias_mean = v_mean - o_mean  # how much V over-/under-predicts
    bias_median = float(np.median(Vs - Os))
    sign_match = int(np.sum((Vs > 0) == (Os > 0)))

    print(f"  N samples: {len(samples)}")
    print(f"  V range: [{Vs.min():.3f}, {Vs.max():.3f}]")
    print(f"  V mean: {v_mean:+.3f}, median: {v_median:+.3f}")
    print(f"  Outcome mean: {o_mean:+.3f}")
    print(f"  Bias (V - O): mean={bias_mean:+.3f}, median={bias_median:+.3f}")
    print(f"  Sign match: {sign_match}/{len(samples)} ({sign_match / len(samples):.1%})")

    # Per-phase stats
    per_phase = {}
    for phase_name in ["early", "mid", "late"]:
        ps = [s for s in samples if phase_of(s[0], s[2]) == phase_name]
        if not ps:
            continue
        pVs = np.array([p[1] for p in ps])
        pOs = np.array([p[3] for p in ps])
        per_phase[phase_name] = {
            "n": len(ps),
            "v_mean": float(pVs.mean()),
            "v_median": float(np.median(pVs)),
            "o_mean": float(pOs.mean()),
            "bias_mean": float(pVs.mean() - pOs.mean()),
            "bias_median": float(np.median(pVs - pOs)),
            "sign_match": int(np.sum((pVs > 0) == (pOs > 0))),
        }
        pm = per_phase[phase_name]
        print(
            f"  {phase_name}: N={pm['n']}, V mean={pm['v_mean']:+.3f}, "
            f"O mean={pm['o_mean']:+.3f}, bias mean={pm['bias_mean']:+.3f}, "
            f"sign match {pm['sign_match']}/{pm['n']} ({pm['sign_match'] / pm['n']:.0%})"
        )

    out = {
        "policy": POLICY_PATH,
        "n_games": N_GAMES,
        "n_samples": len(samples),
        "v_mean": v_mean,
        "v_median": v_median,
        "outcome_mean": o_mean,
        "bias_mean": bias_mean,
        "bias_median": bias_median,
        "sign_match_rate": sign_match / len(samples),
        "per_phase": per_phase,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved calibration to {OUT_PATH}")
    print()
    print("Usage in MCTS: V_calibrated = policy.value(obs) - bias_mean")


if __name__ == "__main__":
    main()
