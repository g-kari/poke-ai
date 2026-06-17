"""REINFORCE self-play training for the MLP policy.

Mirrors train.reinforce.train() but uses MlpPolicy and PyTorch autograd
for both the policy gradient and the value-head MSE. Always uses the
advantage baseline (advantage = reward - V(state)) because the MLP has
more capacity than the linear policy and benefits more from the
variance-reduction baseline.

Usage:
    scripts/run.sh python3 -m train.mlp_train \\
        --episodes 2000 \\
        --lr 1e-3 \\
        --out train/mlp_policy.pt \\
        --metrics-out train/metrics_mlp.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

sys.modules.setdefault("litellm", type(sys)("litellm"))

from kaggle_environments import make  # noqa: E402

from .features import option_features, state_features  # noqa: E402
from .mlp_policy import DEFAULT_PATH, MlpPolicy  # noqa: E402


def _read_deck() -> list[int]:
    """Read deck.csv from the repo root."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "deck.csv")
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


DECK = _read_deck()


@dataclass
class Step:
    sf: np.ndarray  # (STATE_DIM,)
    of_all: np.ndarray  # (n_opts, OPTION_DIM)
    picked: int  # chosen option index


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


def run_episode(policy: MlpPolicy, rng: np.random.Generator):
    trace0: list[Step] = []
    trace1: list[Step] = []
    a0 = make_training_agent(policy, rng, trace0)
    a1 = make_training_agent(policy, rng, trace1)
    env = make("cabt")
    env.run([a0, a1])
    r0 = env.steps[-1][0].reward
    r1 = env.steps[-1][1].reward
    return trace0, trace1, r0, r1


def reinforce_update(
    policy: MlpPolicy,
    optimizer: torch.optim.Optimizer,
    trace: list[Step],
    reward: float,
) -> tuple[float, float] | None:
    """REINFORCE policy gradient + value MSE on a single episode's trace.
    Returns (policy_loss, value_loss) for logging, or None if trace empty."""
    if not trace:
        return None
    device = policy.device

    # Batch all decisions in this trace.
    n_decisions = len(trace)
    # Build per-decision tensors. Different decisions have different
    # n_opts, so we can't stack naively; iterate but accumulate the loss.
    policy_loss = torch.zeros(1, device=device)
    value_loss = torch.zeros(1, device=device)
    reward_t = torch.tensor(reward, device=device, dtype=torch.float32)

    for s in trace:
        sf = torch.from_numpy(s.sf).to(device)
        of_all = torch.from_numpy(s.of_all).to(device)
        n = of_all.shape[0]
        # Logits for every option.
        x = torch.cat([sf.unsqueeze(0).expand(n, -1), of_all], dim=1)
        logits = policy.pi(x).squeeze(-1)
        ranks = torch.arange(n, device=device, dtype=torch.float32)
        logits = logits + policy.b_order * (n - 1 - ranks) / max(1, n - 1)
        log_probs = torch.log_softmax(logits, dim=0)

        # Value baseline.
        v_pred = torch.tanh(policy.v(sf.unsqueeze(0)).squeeze())
        advantage = (reward_t - v_pred).detach()

        # REINFORCE: -advantage * log_pi(picked|s)
        policy_loss = policy_loss - advantage * log_probs[s.picked]
        # Value MSE.
        value_loss = value_loss + (v_pred - reward_t).pow(2)

    loss = policy_loss / n_decisions + value_loss / n_decisions
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return float(policy_loss.detach().item() / n_decisions), float(
        value_loss.detach().item() / n_decisions
    )


def train(
    episodes: int,
    lr: float,
    out: str,
    seed: int = 0,
    start_from: str | None = None,
    log_every: int = 100,
    metrics_out: str | None = None,
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
    policy = MlpPolicy(**kwargs)
    if start_from and os.path.exists(start_from):
        try:
            policy = MlpPolicy.load(start_from, device=policy.device)
            print(f"loaded warm start from {start_from}")
        except Exception as exc:
            print(f"warm-start load failed ({exc}); training from scratch")
    print(f"policy: pi={policy.hidden_pi} v={policy.hidden_v} device={policy.device}")
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    t0 = time.monotonic()
    wins = losses = draws = 0
    recent: list[int] = []
    metrics: list[dict] = []
    for ep in range(1, episodes + 1):
        trace0, trace1, r0, r1 = run_episode(policy, rng)
        if r0 is not None:
            reinforce_update(policy, optimizer, trace0, float(r0))
        if r1 is not None:
            reinforce_update(policy, optimizer, trace1, float(r1))
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
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default=DEFAULT_PATH)
    p.add_argument("--warm-start", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--metrics-out", default=None)
    p.add_argument(
        "--hidden-pi",
        default=None,
        help="Comma-separated MLP policy widths, e.g. '128,64,32'",
    )
    p.add_argument(
        "--hidden-v",
        default=None,
        help="Comma-separated MLP value-head widths, e.g. '64,32'",
    )
    args = p.parse_args()

    def _parse(spec):
        return tuple(int(x) for x in spec.split(",")) if spec else None

    train(
        args.episodes,
        args.lr,
        args.out,
        args.seed,
        args.warm_start,
        args.log_every,
        args.metrics_out,
        _parse(args.hidden_pi),
        _parse(args.hidden_v),
    )


if __name__ == "__main__":
    main()
