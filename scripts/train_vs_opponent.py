"""Train our linear policy with REINFORCE against a fixed external opponent.

Self-play has the limitation that the gradient signal is computed against
a copy of the same policy, so the policy converges to mirror-match
equilibrium with no incentive to learn moves that beat a different
opponent. Training against a stronger fixed opponent (e.g., the
rule-based Mega Lucario agent) forces our policy to find moves that
actually punish the opponent's mistakes.

Only OUR policy is updated. The opponent is treated as a black-box
agent function — we record traces only for our side.

Usage:
    scripts/run.sh python3 scripts/train_vs_opponent.py \\
        --episodes 2000 \\
        --opponent rule_based \\
        --warm-start train/policy.npz \\
        --out train/policy.npz \\
        --metrics-out train/metrics_vs_rule_based.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("RULE_DECK_PATH", str(ROOT / "deck_mega_lucario.csv"))
sys.modules.setdefault("litellm", types.ModuleType("litellm"))

from kaggle_environments import make  # noqa: E402

from agent import random_agent  # noqa: E402
from train.features import option_features, state_features  # noqa: E402
from train.policy import DEFAULT_PATH, LinearPolicy  # noqa: E402
from train.reinforce import Step, reinforce_update  # noqa: E402


def _read_deck() -> list[int]:
    path = ROOT / "deck.csv"
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


DECK = _read_deck()


def make_training_agent(policy, rng, trace, my_side):
    """Same shape as the self-play training agent but records only when
    sel.type == 0 and maxCount == 1, matching train.reinforce."""

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
            trace.append(
                Step(sf=sf, of_picked=of_all[i], of_all=of_all, probs=probs),
            )
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


def run_episode_vs(policy, opp_agent, rng, our_side: int):
    """Play one episode with our policy on `our_side` (0 or 1) and the
    fixed opponent on the other side. Return (trace, our_reward)."""
    trace: list[Step] = []
    our_agent = make_training_agent(policy, rng, trace, our_side)
    env = make("cabt")
    if our_side == 0:
        env.run([our_agent, opp_agent])
    else:
        env.run([opp_agent, our_agent])
    our_reward = env.steps[-1][our_side].reward
    return trace, our_reward


def train(
    episodes: int,
    lr: float,
    lr_value: float,
    out: str,
    seed: int,
    start_from: str | None,
    log_every: int,
    metrics_out: str | None,
    opponent_name: str,
    use_advantage: bool,
) -> None:
    rng = np.random.default_rng(seed)
    policy = LinearPolicy()
    if start_from and os.path.exists(start_from):
        policy = LinearPolicy.load(start_from)
        print(f"loaded warm start from {start_from}")
    policy.b_order = 2.0
    opp = load_opponent(opponent_name)

    t0 = time.monotonic()
    wins = losses = draws = 0
    recent: list[int] = []
    metrics = []
    for ep in range(1, episodes + 1):
        our_side = ep % 2  # alternate P0 / P1
        trace, r = run_episode_vs(policy, opp, rng, our_side)
        if r is not None:
            reinforce_update(policy, trace, float(r), lr, lr_value, use_advantage=use_advantage)
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
                "w_opt_norm": round(float(np.linalg.norm(policy.w_opt)), 4),
                "w_state_norm": round(float(np.linalg.norm(policy.w_state)), 4),
                "w_value_norm": round(float(np.linalg.norm(policy.w_value)), 4),
                "elapsed_s": round(dt, 1),
            }
            metrics.append(row)
            print(
                f"ep {ep:4d}  cum W/L/D = {wins}/{losses}/{draws}  "
                f"recent {win_rate_recent:.2f}  "
                f"|w_opt|={row['w_opt_norm']:.3f}  {dt:.1f}s"
            )
    policy.save(out)
    print(f"saved policy to {out}")
    if metrics_out:
        with open(metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"saved metrics to {metrics_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr-value", type=float, default=0.05)
    p.add_argument("--out", default=DEFAULT_PATH)
    p.add_argument("--warm-start", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--opponent", default="rule_based", choices=["rule_based", "random"])
    p.add_argument(
        "--use-advantage",
        action="store_true",
        default=True,
        help="Subtract V(state) from reward before scaling the policy gradient. "
        "Default on for vs-opponent training; pass --no-use-advantage to disable.",
    )
    p.add_argument("--no-use-advantage", dest="use_advantage", action="store_false")
    args = p.parse_args()
    train(
        args.episodes,
        args.lr,
        args.lr_value,
        args.out,
        args.seed,
        args.warm_start,
        args.log_every,
        args.metrics_out,
        args.opponent,
        args.use_advantage,
    )


if __name__ == "__main__":
    main()
