"""Direct match between 3-MLP base ensemble (LB 679.6 champion) and PPO_v40 s500.

Compares "DL champion" vs "PPO median policy" in same-deck 1v1.
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

from train.ensemble_policy import EnsemblePolicy  # noqa: E402
from train.mlp_policy import MlpPolicy  # noqa: E402

mlp0 = MlpPolicy.load("train/mlp_policy.pt")
mlp2 = MlpPolicy.load("train/mlp_policy_seed2.pt")
mlp100 = MlpPolicy.load("train/mlp_policy_seed100.pt")
ensemble = EnsemblePolicy([mlp0, mlp2, mlp100])
policy_s100 = MlpPolicy.load("train/mlp_policy_ppo_v40_s100.pt")

with open("deck.csv") as _f:
    DECK = [int(line.strip()) for line in _f if line.strip()]
rng = np.random.default_rng(42)


def make_agent(policy):
    def _agent(obs):
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

    return _agent


a_3mlp = make_agent(ensemble)
a_s100 = make_agent(policy_s100)

mlp_w = s100_w = draws = 0
for i in range(40):
    env = make("cabt")
    if i < 20:
        env.run([a_3mlp, a_s100])
        r = env.steps[-1][0].reward
        if r == 1:
            mlp_w += 1
        elif r == -1:
            s100_w += 1
        else:
            draws += 1
    else:
        env.run([a_s100, a_3mlp])
        r = env.steps[-1][0].reward
        if r == 1:
            s100_w += 1
        elif r == -1:
            mlp_w += 1
        else:
            draws += 1
    if (i + 1) % 10 == 0:
        print(f"  after {i + 1}: 3-MLP {mlp_w}, s100 {s100_w}, draws {draws}")

print(f"\nFinal (40 games): 3-MLP {mlp_w}, s100 {s100_w}, draws {draws}")
print(f"3-MLP winrate: {mlp_w / 40:.1%}")
print(f"s100 winrate: {s100_w / 40:.1%}")
