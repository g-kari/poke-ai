"""Fine-tune the MLP policy with REINFORCE against a fixed external opponent.

Mirrors scripts/train_vs_opponent.py (which fine-tunes the linear policy)
but for the PyTorch MlpPolicy. Always uses the advantage baseline
(adv = reward - V(state)), gradient-clips at 1.0, and updates only OUR
side of the match-up.

Usage:
    scripts/run.sh python3 scripts/train_mlp_vs_opponent.py \\
        --episodes 1500 \\
        --opponent rule_based \\
        --warm-start train/mlp_policy.pt \\
        --out train/mlp_policy.pt \\
        --metrics-out train/metrics_mlp_vs_rule.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("RULE_DECK_PATH", str(ROOT / "deck_mega_lucario.csv"))
sys.modules.setdefault("litellm", types.ModuleType("litellm"))

from kaggle_environments import make  # noqa: E402

from agent import random_agent  # noqa: E402
from train.features import option_features, state_features  # noqa: E402
from train.mlp_policy import DEFAULT_PATH, MlpPolicy  # noqa: E402


def _read_deck() -> list[int]:
    path = ROOT / "deck.csv"
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


DECK = _read_deck()


@dataclass
class Step:
    sf: np.ndarray
    of_all: np.ndarray
    picked: int


def make_training_agent(policy: MlpPolicy, rng: np.random.Generator, trace: list[Step]):
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


def load_opponent(name: str):
    if name == "rule_based":
        import rule_based_agent  # noqa: PLC0415

        return rule_based_agent.agent
    if name == "random":
        return random_agent
    raise ValueError(f"unknown opponent: {name}")


def run_episode_vs(policy: MlpPolicy, opp_agent, rng: np.random.Generator, our_side: int):
    trace: list[Step] = []
    our_agent = make_training_agent(policy, rng, trace)
    env = make("cabt")
    if our_side == 0:
        env.run([our_agent, opp_agent])
    else:
        env.run([opp_agent, our_agent])
    our_reward = env.steps[-1][our_side].reward
    return trace, our_reward


def reinforce_update(
    policy: MlpPolicy,
    optimizer: torch.optim.Optimizer,
    trace: list[Step],
    reward: float,
):
    if not trace:
        return
    device = policy.device
    reward_t = torch.tensor(reward, device=device, dtype=torch.float32)
    policy_loss = torch.zeros(1, device=device)
    value_loss = torch.zeros(1, device=device)
    for s in trace:
        sf = torch.from_numpy(s.sf).to(device)
        of_all = torch.from_numpy(s.of_all).to(device)
        n = of_all.shape[0]
        x = torch.cat([sf.unsqueeze(0).expand(n, -1), of_all], dim=1)
        logits = policy.pi(x).squeeze(-1)
        ranks = torch.arange(n, device=device, dtype=torch.float32)
        logits = logits + policy.b_order * (n - 1 - ranks) / max(1, n - 1)
        log_probs = torch.log_softmax(logits, dim=0)
        v_pred = torch.tanh(policy.v(sf.unsqueeze(0)).squeeze())
        advantage = (reward_t - v_pred).detach()
        policy_loss = policy_loss - advantage * log_probs[s.picked]
        value_loss = value_loss + (v_pred - reward_t).pow(2)
    loss = policy_loss / len(trace) + value_loss / len(trace)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()


def train(
    episodes: int,
    lr: float,
    out: str,
    seed: int,
    start_from: str | None,
    log_every: int,
    metrics_out: str | None,
    opponent_name: str,
) -> None:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    policy = MlpPolicy()
    if start_from and os.path.exists(start_from):
        policy = MlpPolicy.load(start_from, device=policy.device)
        print(f"loaded warm start from {start_from}")
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    opp = load_opponent(opponent_name)

    t0 = time.monotonic()
    wins = losses = draws = 0
    recent: list[int] = []
    metrics: list[dict] = []
    for ep in range(1, episodes + 1):
        our_side = ep % 2
        trace, r = run_episode_vs(policy, opp, rng, our_side)
        if r is not None:
            reinforce_update(policy, optimizer, trace, float(r))
        if r == 1:
            wins += 1
        elif r == -1:
            losses += 1
        elif r == 0:
            draws += 1
        recent.append(r if r is not None else 0)
        if len(recent) > log_every:
            recent.pop(0)
        if ep % log_every == 0 or ep == episodes:
            dt = time.monotonic() - t0
            win_rate_recent = sum(1 for x in recent if x == 1) / max(1, len(recent))
            row = {
                "ep": ep,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate_recent": round(win_rate_recent, 3),
                "elapsed_s": round(dt, 1),
            }
            metrics.append(row)
            print(
                f"ep {ep:4d}  cum W/L/D = {wins}/{losses}/{draws}  "
                f"recent {win_rate_recent:.2f}  {dt:.1f}s"
            )
    policy.eval()
    policy.save(out)
    print(f"saved policy to {out}")
    if metrics_out:
        with open(metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"saved metrics to {metrics_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--out", default=DEFAULT_PATH)
    p.add_argument("--warm-start", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--opponent", default="rule_based", choices=["rule_based", "random"])
    args = p.parse_args()
    train(
        args.episodes,
        args.lr,
        args.out,
        args.seed,
        args.warm_start,
        args.log_every,
        args.metrics_out,
        args.opponent,
    )


if __name__ == "__main__":
    main()
