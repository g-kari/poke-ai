"""PPO Phase 4: full training loop integrating Phases 1-3.

Reusable bone-structure:
1. Rollout: collect N episodes with rollout-time log_prob + value
2. GAE: compute per-episode advantages + return targets
3. PPO update: k_epochs of mini-batch clipped surrogate + value MSE + entropy
4. Repeat

Designed to **warm-start from BCRL2** (lab 16.1%, LB 570.4) and push to
lab 20%+ via PPO's variance control — predicted LB 700+.

Usage:
    scripts/run.sh python3 -m train.ppo_train \\
        --warm-start train/mlp_policy_v60_bcrl2.pt \\
        --out train/mlp_policy_v60_ppo1.pt \\
        --batch-size 64 --total-iterations 50 \\
        --opponent-pool rule_based_agent,rule_based_dragapult,...
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.modules.setdefault("litellm", type(sys)("litellm"))

from kaggle_environments import make  # noqa: E402

from .features_v60 import option_features, state_features  # noqa: E402
from .mlp_policy_v60 import MlpPolicyV60  # noqa: E402
from .ppo_buffer import PPOBuffer, PPOStep, compute_log_prob_and_value  # noqa: E402
from .ppo_gae import compute_gae_for_buffer, normalize_advantages  # noqa: E402
from .ppo_loss import ppo_update  # noqa: E402


def _read_deck() -> list[int]:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    path = os.path.join(root, "deck.csv")
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


DECK = _read_deck()


def _load_opponent(name: str):
    if not name:
        return None
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module(name).agent


def make_ppo_agent(policy: MlpPolicyV60, rng: np.random.Generator, trace: list[PPOStep]):
    """Agent that records (sf, of_all, picked, log_prob_old, value) per MAIN
    single-choice select — exactly what PPO needs as rollout."""

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
            log_prob_old, value = compute_log_prob_and_value(policy, sf, of_all, i)
            trace.append(
                PPOStep(sf=sf, of_all=of_all, picked=i, log_prob=log_prob_old, value=value)
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


def run_ppo_episode(
    policy: MlpPolicyV60, rng: np.random.Generator, opp_pool: list
) -> tuple[list[PPOStep], float]:
    trace: list[PPOStep] = []
    a0 = make_ppo_agent(policy, rng, trace)
    env = make("cabt")
    if not opp_pool:
        # Mirror self-play (= rare for PPO; expected to be opp pool)
        trace1: list[PPOStep] = []
        a1 = make_ppo_agent(policy, rng, trace1)
        env.run([a0, a1])
        return trace, float(env.steps[-1][0].reward)
    opp = opp_pool[int(rng.integers(len(opp_pool)))]
    if rng.random() < 0.5:
        env.run([a0, opp])
        return trace, float(env.steps[-1][0].reward)
    env.run([opp, a0])
    return trace, float(env.steps[-1][1].reward)


def train(
    warm_start: str | None,
    out: str,
    batch_size: int,
    total_iterations: int,
    opponent_pool: str,
    seed: int,
    lr: float,
    k_epochs: int,
    mb_size: int,
    eps_clip: float,
    value_coef: float,
    entropy_coef: float,
    clip_grad: float,
    gamma: float,
    lam: float,
    log_every: int,
    metrics_out: str | None,
) -> None:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    policy = MlpPolicyV60()
    if warm_start and os.path.exists(warm_start):
        policy = MlpPolicyV60.load(warm_start, device=policy.device)
        print(f"warm-started from {warm_start}")
    else:
        print("fresh policy (no warm-start)")
    print(f"device={policy.device}, lr={lr}, eps_clip={eps_clip}, k_epochs={k_epochs}")

    pool = [_load_opponent(n.strip()) for n in opponent_pool.split(",") if n.strip()]
    pool = [p for p in pool if p is not None]
    print(f"opponent pool: {len(pool)} agents")

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    t0 = time.monotonic()
    history: list[dict] = []
    cum_w = cum_l = cum_d = 0

    for it in range(1, total_iterations + 1):
        buffer = PPOBuffer()
        ep_wins = ep_losses = ep_draws = 0
        for _ in range(batch_size):
            trace, reward = run_ppo_episode(policy, rng, pool)
            buffer.add_episode(trace, reward)
            if reward == 1:
                ep_wins += 1
            elif reward == -1:
                ep_losses += 1
            else:
                ep_draws += 1
        cum_w += ep_wins
        cum_l += ep_losses
        cum_d += ep_draws

        flat = buffer.flatten()
        advantages, returns = compute_gae_for_buffer(flat, gamma=gamma, lam=lam)
        advantages = normalize_advantages(advantages)

        metrics = ppo_update(
            policy,
            optimizer,
            flat,
            advantages,
            returns,
            k_epochs=k_epochs,
            mb_size=mb_size,
            eps_clip=eps_clip,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            clip_grad=clip_grad,
        )

        recent_winrate = ep_wins / max(1, ep_wins + ep_losses + ep_draws)
        elapsed = time.monotonic() - t0
        history.append(
            {
                "iter": it,
                "ep_wins": ep_wins,
                "ep_losses": ep_losses,
                "ep_draws": ep_draws,
                "recent_winrate": recent_winrate,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
                "elapsed_s": elapsed,
            }
        )

        if it % log_every == 0 or it == 1:
            print(
                f"iter {it:3d}/{total_iterations}  "
                f"cum W/L/D={cum_w}/{cum_l}/{cum_d}  recent {recent_winrate:.2f}  "
                f"pl={metrics['policy_loss']:.4f} vl={metrics['value_loss']:.4f} "
                f"ent={metrics['entropy']:.3f}  [{elapsed:.0f}s]"
            )

    policy.save(out)
    print(f"saved policy to {out}")

    if metrics_out:
        with open(metrics_out, "w") as f:
            json.dump(history, f, indent=2)
        print(f"saved metrics to {metrics_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm-start", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=64, help="episodes per PPO iteration")
    p.add_argument("--total-iterations", type=int, default=50)
    p.add_argument(
        "--opponent-pool",
        default="rule_based_agent,rule_based_dragapult,rule_based_iono,"
        "rule_based_abomasnow,rule_based_crustle_dashimaki,rule_based_romanrozen_v6",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--k-epochs", type=int, default=4)
    p.add_argument("--mb-size", type=int, default=32)
    p.add_argument("--eps-clip", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--clip-grad", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--metrics-out", default=None)
    args = p.parse_args()

    train(
        warm_start=args.warm_start,
        out=args.out,
        batch_size=args.batch_size,
        total_iterations=args.total_iterations,
        opponent_pool=args.opponent_pool,
        seed=args.seed,
        lr=args.lr,
        k_epochs=args.k_epochs,
        mb_size=args.mb_size,
        eps_clip=args.eps_clip,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        clip_grad=args.clip_grad,
        gamma=args.gamma,
        lam=args.lam,
        log_every=args.log_every,
        metrics_out=args.metrics_out,
    )


if __name__ == "__main__":
    main()
