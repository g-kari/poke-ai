"""REINFORCE self-play training loop.

How it works
------------
1. Roll out one game with two stochastic agents driven by the current policy.
2. For each decision a player made, record (state_feat, option_feat_for_pick,
   chosen_index, candidate option_feats). The reward is the game's final
   reward (+1 / 0 / -1).
3. After each episode, do a single SGD step on the policy parameters:
     grad_logp(s, i) = phi(s, i) - sum_j p_j * phi(s, j)
   where phi(s, i) concatenates state and option features. The advantage is
   the final reward (no baseline yet — easy to add later).
4. Save weights to `train/policy.npz`; the production `agent.py` will pick
   them up automatically.

Usage:
    python3 -m train.reinforce --episodes 200 --lr 0.05 --out train/policy.npz

Cost: ~0.4 s per game on this image, so 200 episodes ≈ 80 s.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.modules.setdefault("litellm", type(sys)("litellm"))

from kaggle_environments import make

from .features import OPTION_DIM, STATE_DIM, option_features, state_features
from .policy import DEFAULT_PATH, LinearPolicy

# avoid circular import; only need DECK
from agent import DECK


@dataclass
class Step:
    sf: np.ndarray            # (STATE_DIM,)
    of_picked: np.ndarray     # (OPTION_DIM,)
    of_all: np.ndarray        # (n_opts, OPTION_DIM)
    probs: np.ndarray         # (n_opts,)


def make_training_agent(policy: LinearPolicy, rng: np.random.Generator,
                        trace: list[Step], my_index_holder: list[int]):
    """Returns a Kaggle agent fn that uses `policy` and appends to `trace`.

    Only MAIN decisions (sel.type == 0) are trained on for now — they are
    by far the most common decision and the most strategically meaningful.
    """

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
            # Stochastic policy choice + record gradient inputs.
            probs = policy.probs(obs, sel)
            i = int(rng.choice(len(opts), p=probs))
            sf = state_features(obs)
            of_all = np.stack([option_features(o, obs, sel) for o in opts])
            trace.append(Step(sf=sf, of_picked=of_all[i],
                              of_all=of_all, probs=probs))
            my_index_holder[0] = obs["current"]["yourIndex"]
            return [i]

        # Non-MAIN / multi-select: argmax of policy (no gradient).
        if max_c >= 1:
            logits = policy.logits(obs, sel)
            order = np.argsort(-logits)
            k = max(min_c, 1)
            k = min(k, max_c, len(opts))
            return [int(x) for x in order[:k].tolist()]
        return []

    return _agent


def run_episode(policy: LinearPolicy, rng: np.random.Generator):
    """Run one self-play episode. Returns (trace_p0, trace_p1, r0, r1)."""
    trace0, trace1 = [], [],
    trace1 = []
    idx0 = [0]
    idx1 = [1]
    a0 = make_training_agent(policy, rng, trace0, idx0)
    a1 = make_training_agent(policy, rng, trace1, idx1)
    env = make("cabt")
    env.run([a0, a1])
    r0 = env.steps[-1][0].reward
    r1 = env.steps[-1][1].reward
    return trace0, trace1, r0, r1


def reinforce_update(policy: LinearPolicy, trace: list[Step],
                     reward: float, lr: float) -> None:
    """One on-policy gradient step over all decisions in `trace`."""
    if not trace or reward == 0:
        return
    g_state = np.zeros(STATE_DIM, dtype=np.float32)
    g_opt = np.zeros(OPTION_DIM, dtype=np.float32)
    for s in trace:
        # grad log p(i|s) for option features = of_picked - E_p[of]
        expected_of = s.probs @ s.of_all
        g_opt += reward * (s.of_picked - expected_of)
        # state features are shared by all options so their gradient under
        # a softmax over options is zero. We still let the bias-on-order
        # term drift via a tiny gradient on b_order through option ordering.
    # No per-step state-feature update under this parameterization; rely on
    # option features. (If you add per-option state-conditioned terms,
    # update g_state here.)
    policy.w_state += lr * g_state / max(1, len(trace))
    policy.w_opt += lr * g_opt / max(1, len(trace))


def train(episodes: int, lr: float, out: str, seed: int = 0,
          start_from: str | None = None, log_every: int = 20,
          metrics_out: str | None = None) -> None:
    if log_every <= 0:
        raise ValueError("log_every must be > 0")
    rng = np.random.default_rng(seed)
    policy = LinearPolicy()
    if start_from:
        try:
            policy = LinearPolicy.load(start_from)
            print(f"loaded warm start from {start_from}")
        except FileNotFoundError:
            print(f"no warm start at {start_from}, training from scratch")
    # Engine-order prior so the untrained agent already matches `first_agent`.
    policy.b_order = 2.0

    t0 = time.monotonic()
    wins = losses = draws = 0
    # Rolling-window stats (last `log_every` episodes) so we can see if
    # winrate trends up over time rather than just cumulative.
    recent = []
    metrics = []
    for ep in range(1, episodes + 1):
        trace0, trace1, r0, r1 = run_episode(policy, rng)
        # Engine occasionally returns reward=None when a player's action is
        # rejected by the C side (status=INVALID/ERROR). Skip the gradient
        # for that player so the run keeps going.
        if r0 is not None:
            reinforce_update(policy, trace0, float(r0), lr)
        if r1 is not None:
            reinforce_update(policy, trace1, float(r1), lr)
        if r0 == 1:
            wins += 1
        elif r0 == -1:
            losses += 1
        elif r0 == 0:
            draws += 1
        recent.append(r0 if r0 is not None else 0)
        if len(recent) > log_every:
            recent.pop(0)
        if ep % log_every == 0 or ep == episodes:
            dt = time.monotonic() - t0
            win_rate_recent = sum(1 for r in recent if r == 1) / max(1, len(recent))
            row = {
                "ep": ep,
                "wins": wins, "losses": losses, "draws": draws,
                "win_rate_recent": round(win_rate_recent, 3),
                "w_opt_norm": round(float(np.linalg.norm(policy.w_opt)), 4),
                "w_state_norm": round(float(np.linalg.norm(policy.w_state)), 4),
                "elapsed_s": round(dt, 1),
            }
            metrics.append(row)
            print(f"ep {ep:4d}  cum W/L/D = {wins}/{losses}/{draws}  "
                  f"recent {win_rate_recent:.2f}  "
                  f"|w_opt|={row['w_opt_norm']:.3f}  {dt:.1f}s")
    policy.save(out)
    print(f"saved policy to {out}")
    if metrics_out:
        import json as _json
        with open(metrics_out, "w") as f:
            _json.dump(metrics, f, indent=2)
        print(f"saved metrics to {metrics_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--out", default=DEFAULT_PATH)
    p.add_argument("--warm-start", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--metrics-out", default=None)
    args = p.parse_args()
    train(args.episodes, args.lr, args.out, args.seed, args.warm_start,
          args.log_every, args.metrics_out)


if __name__ == "__main__":
    main()
