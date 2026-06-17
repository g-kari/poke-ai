"""Train the MLP against a rotation of the four Kiyota meta-deck rule-based
agents (Mega Lucario, Dragapult ex, Iono's, Mega Abomasnow ex).

The single-opponent finetune we tried earlier overfit to that one
opponent's play patterns and regressed on everything else. Sampling
uniformly across multiple opponents per episode keeps the gradient
signal diverse — each rule-based agent has its own strategy (energy
ramp, Phantom Dive board sweep, Wattrel chain, mirror wall) so the
policy can't collapse onto a single counter.

Always uses the advantage baseline (`adv = reward - V(state)`); the
training reward distribution would otherwise be heavily skewed
negative (we win ~22% overall against this set) and the no-baseline
gradient would just push the policy toward uniform.

Logs per-opponent winrate so we can spot whether one opponent is
dragging the whole signal.

Usage:
    scripts/run.sh python3 scripts/train_mlp_vs_meta.py \\
        --episodes 2500 \\
        --warm-start train/mlp_policy.pt \\
        --out train/mlp_policy_meta.pt \\
        --metrics-out train/metrics_mlp_vs_meta.json
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
sys.modules.setdefault("litellm", types.ModuleType("litellm"))

from kaggle_environments import make  # noqa: E402

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


def load_opponents() -> list[tuple[str, callable]]:
    """Load the four meta-deck rule-based agents."""
    import rule_based_abomasnow  # noqa: PLC0415
    import rule_based_agent  # Mega Lucario  # noqa: PLC0415
    import rule_based_dragapult  # noqa: PLC0415
    import rule_based_iono  # noqa: PLC0415

    return [
        ("mega_lucario", rule_based_agent.agent),
        ("dragapult", rule_based_dragapult.agent),
        ("iono", rule_based_iono.agent),
        ("abomasnow", rule_based_abomasnow.agent),
    ]


def run_episode_vs(policy, opp_agent, rng, our_side: int):
    trace: list[Step] = []
    our_agent = make_training_agent(policy, rng, trace)
    env = make("cabt")
    if our_side == 0:
        env.run([our_agent, opp_agent])
    else:
        env.run([opp_agent, our_agent])
    our_reward = env.steps[-1][our_side].reward
    return trace, our_reward


def reinforce_update(policy, optimizer, trace, reward: float):
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
) -> None:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    policy = MlpPolicy()
    if start_from and os.path.exists(start_from):
        policy = MlpPolicy.load(start_from, device=policy.device)
        print(f"loaded warm start from {start_from}")
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    opps = load_opponents()
    n_opps = len(opps)
    print(f"opponents: {[name for name, _ in opps]}")

    t0 = time.monotonic()
    per_opp: dict[str, list[int]] = {name: [] for name, _ in opps}
    recent_all: list[int] = []
    metrics: list[dict] = []
    for ep in range(1, episodes + 1):
        idx = rng.integers(n_opps)
        opp_name, opp_agent = opps[idx]
        our_side = ep % 2
        trace, r = run_episode_vs(policy, opp_agent, rng, our_side)
        if r is not None:
            reinforce_update(policy, optimizer, trace, float(r))
        outcome = 1 if r == 1 else 0
        per_opp[opp_name].append(outcome)
        recent_all.append(outcome)
        if len(recent_all) > log_every:
            recent_all.pop(0)
        for k in per_opp:
            if len(per_opp[k]) > 2 * log_every:
                per_opp[k] = per_opp[k][-2 * log_every :]

        if ep % log_every == 0 or ep == episodes:
            dt = time.monotonic() - t0
            per_str = "  ".join(
                f"{name[:4]}={sum(per_opp[name][-log_every:]) / max(1, len(per_opp[name][-log_every:])):.2f}"
                for name, _ in opps
            )
            row = {
                "ep": ep,
                "recent_overall": round(sum(recent_all) / max(1, len(recent_all)), 3),
                "per_opp": {
                    name: round(
                        sum(per_opp[name][-log_every:]) / max(1, len(per_opp[name][-log_every:])),
                        3,
                    )
                    for name, _ in opps
                },
                "elapsed_s": round(dt, 1),
            }
            metrics.append(row)
            print(f"ep {ep:4d}  recent={row['recent_overall']:.2f}  [{per_str}]  {dt:.1f}s")
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
    p.add_argument("--out", default=DEFAULT_PATH)
    p.add_argument("--warm-start", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--metrics-out", default=None)
    args = p.parse_args()
    train(
        args.episodes,
        args.lr,
        args.out,
        args.seed,
        args.warm_start,
        args.log_every,
        args.metrics_out,
    )


if __name__ == "__main__":
    main()
