"""REINFORCE self-play training for MlpPolicyV60 (features_v60).

Identical to mlp_train.py but uses MlpPolicyV60 / features_v60. Kept as
a sibling so the v40 pipeline (mlp_train.py) stays untouched and the
3-MLP submission (40-d) keeps working.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.modules.setdefault("litellm", type(sys)("litellm"))

from kaggle_environments import make  # noqa: E402

from .features_v60 import option_features, state_features  # noqa: E402
from .mlp_policy_v60 import DEFAULT_PATH, MlpPolicyV60  # noqa: E402


def _read_deck() -> list[int]:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "deck.csv")
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


DECK = _read_deck()


@dataclass
class Step:
    sf: np.ndarray
    of_all: np.ndarray
    picked: int


def make_training_agent(policy: MlpPolicyV60, rng: np.random.Generator, trace: list[Step]):
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
            i = int(rng.choice(len(opts), p=probs))
            sf = state_features(obs)
            of_all = np.stack([option_features(o, obs, sel) for o in opts])
            trace.append(Step(sf=sf, of_all=of_all, picked=i))
            return [i]

        if max_c >= 1:
            logits = policy.logits(obs, sel)
            order = np.argsort(-logits)
            k = max(min_c, 1)
            k = min(k, max_c, len(opts))
            return [int(x) for x in order[:k].tolist()]
        return []

    return _agent


def run_episode(
    policy: MlpPolicyV60,
    rng: np.random.Generator,
    opponent_pool: list | None = None,
):
    trace0: list[Step] = []
    a0 = make_training_agent(policy, rng, trace0)
    env = make("cabt")
    if not opponent_pool:
        trace1: list[Step] = []
        a1 = make_training_agent(policy, rng, trace1)
        env.run([a0, a1])
        r0 = env.steps[-1][0].reward
        r1 = env.steps[-1][1].reward
        return trace0, trace1, r0, r1
    opp = opponent_pool[int(rng.integers(len(opponent_pool)))]
    if rng.random() < 0.5:
        env.run([a0, opp])
        return trace0, [], env.steps[-1][0].reward, None
    env.run([opp, a0])
    return trace0, [], env.steps[-1][1].reward, None


def reinforce_update(policy, optimizer, trace, reward, linear_value: bool = False):
    """REINFORCE with value baseline. linear_value=True drops the tanh clip
    on V(s) — lets the value function express negative-magnitude expected
    rewards in hard matchups (= where prior tanh saturated at -1 and broke
    the advantage gradient)."""
    if not trace:
        return None
    device = policy.device
    policy_loss = torch.zeros(1, device=device)
    value_loss = torch.zeros(1, device=device)
    reward_t = torch.tensor(reward, device=device, dtype=torch.float32)
    n_decisions = len(trace)
    for s in trace:
        sf = torch.from_numpy(s.sf).to(device)
        of_all = torch.from_numpy(s.of_all).to(device)
        n = of_all.shape[0]
        x = torch.cat([sf.unsqueeze(0).expand(n, -1), of_all], dim=1)
        logits = policy.pi(x).squeeze(-1)
        ranks = torch.arange(n, device=device, dtype=torch.float32)
        logits = logits + policy.b_order * (n - 1 - ranks) / max(1, n - 1)
        log_probs = torch.log_softmax(logits, dim=0)
        v_raw = policy.v(sf.unsqueeze(0)).squeeze()
        v_pred = v_raw if linear_value else torch.tanh(v_raw)
        advantage = (reward_t - v_pred).detach()
        policy_loss = policy_loss - advantage * log_probs[s.picked]
        value_loss = value_loss + (v_pred - reward_t).pow(2)
    loss = policy_loss / n_decisions + value_loss / n_decisions
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return None


def _load_opponent(name: str):
    if not name:
        return None
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module(name).agent


def train(
    episodes: int,
    lr: float,
    out: str,
    seed: int,
    log_every: int,
    metrics_out: str | None,
    opponent_pool: str,
    warm_start: str | None = None,
    linear_value: bool = False,
    hidden_pi: tuple[int, ...] | None = None,
    hidden_v: tuple[int, ...] | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    kwargs = {}
    if hidden_pi is not None:
        kwargs["hidden_pi"] = hidden_pi
    if hidden_v is not None:
        kwargs["hidden_v"] = hidden_v
    policy = MlpPolicyV60(**kwargs)
    if warm_start and os.path.exists(warm_start):
        try:
            policy = MlpPolicyV60.load(warm_start, device=policy.device)
            print(f"loaded warm start from {warm_start}")
        except Exception as exc:
            print(f"warm-start load failed ({exc}); training from scratch")
    print(f"policy v60: pi={policy.hidden_pi} v={policy.hidden_v} device={policy.device}")
    pool = [_load_opponent(n.strip()) for n in opponent_pool.split(",") if n.strip()]
    pool = [p for p in pool if p is not None]
    print(f"opponent pool: {len(pool)} agents [{opponent_pool}]")

    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    t0 = time.monotonic()
    wins = losses = draws = 0
    recent: list[int] = []
    metrics: list[dict] = []
    print(f"value head: {'linear (no tanh)' if linear_value else 'tanh-bounded'}")
    for ep in range(1, episodes + 1):
        trace0, _, r0, _ = run_episode(policy, rng, pool or None)
        if r0 is not None:
            reinforce_update(policy, optimizer, trace0, float(r0), linear_value)
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
            win_rate = sum(1 for x in recent if x == 1) / max(1, len(recent))
            metrics.append(
                {
                    "ep": ep,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "win_rate_recent": round(win_rate, 3),
                    "elapsed_s": round(dt, 1),
                }
            )
            print(f"ep {ep:4d}  W/L/D={wins}/{losses}/{draws}  recent {win_rate:.2f}  {dt:.1f}s")
    policy.eval()
    policy.save(out)
    print(f"saved policy to {out}")
    if metrics_out:
        with open(metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"saved metrics to {metrics_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=DEFAULT_PATH)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--metrics-out", default=None)
    p.add_argument(
        "--opponent-pool",
        default="",
        help="Comma-separated opponent modules (e.g. rule_based_iono,rule_based_crustle_dashimaki)",
    )
    p.add_argument("--warm-start", default=None, help="Path to existing .pt to warm-start from")
    p.add_argument(
        "--linear-value",
        action="store_true",
        help="Drop tanh on V(s) so it can express negative-magnitude expected rewards.",
    )
    p.add_argument(
        "--hidden-pi",
        default=None,
        help="Comma-separated policy MLP widths, e.g. '128,64'. Default: 64,32.",
    )
    p.add_argument(
        "--hidden-v",
        default=None,
        help="Comma-separated value MLP widths, e.g. '64,32'. Default: 32.",
    )
    args = p.parse_args()

    def _parse(spec):
        return tuple(int(x) for x in spec.split(",")) if spec else None

    train(
        args.episodes,
        args.lr,
        args.out,
        args.seed,
        args.log_every,
        args.metrics_out,
        args.opponent_pool,
        args.warm_start,
        args.linear_value,
        _parse(args.hidden_pi),
        _parse(args.hidden_v),
    )


if __name__ == "__main__":
    main()
