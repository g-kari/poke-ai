"""PPO Phase 1: trajectory buffer with log-probabilities for v60 policy.

Stores per-step (state, action, log_prob, value) tuples plus episode-level
reward, ready for PPO update with GAE advantage (Phase 2 next).

Why log_prob storage matters: PPO needs the OLD policy's log_prob(a|s)
to compute the ratio `pi_new(a)/pi_old(a)` for the clipped surrogate
objective. REINFORCE could recompute it from current policy weights,
but PPO does k_epochs of updates between rollouts, so the rollout-time
log_prob must be cached.

Usage:
    buffer = PPOBuffer()
    for ep in range(batch_size):
        trace = run_episode_with_logprobs(policy, opp_pool, rng)
        buffer.add_episode(trace, reward)
    # ... PPO update uses buffer.samples()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PPOStep:
    """One decision point within an episode."""

    sf: np.ndarray  # state features (60-d for v60)
    of_all: np.ndarray  # option features stacked (n_opts, 40)
    picked: int  # chosen index in [0, n_opts)
    log_prob: float  # log pi_old(picked | state)
    value: float  # V(s) at decision time (= advantage baseline)


@dataclass
class PPOEpisode:
    """One full trajectory + terminal reward."""

    steps: list[PPOStep]
    reward: float  # +1 / 0 / -1 (terminal only for sparse-reward PTCG)


class PPOBuffer:
    """In-memory rollout buffer for one PPO update batch.

    Holds N episodes; each episode is a list of PPOStep. Phase 2 will
    compute GAE advantages here; Phase 3 will mini-batch over the
    flattened (sf, of_all, picked, log_prob, value, advantage, return)
    tuples for the clipped surrogate loss.
    """

    def __init__(self) -> None:
        self.episodes: list[PPOEpisode] = []

    def add_episode(self, steps: list[PPOStep], reward: float) -> None:
        if steps:
            self.episodes.append(PPOEpisode(steps=steps, reward=float(reward)))

    def clear(self) -> None:
        self.episodes.clear()

    def __len__(self) -> int:
        return len(self.episodes)

    def total_steps(self) -> int:
        return sum(len(ep.steps) for ep in self.episodes)

    def flatten(self) -> dict[str, np.ndarray | list]:
        """Concatenate all steps for indexing. Returns a dict of arrays
        suitable for mini-batch sampling (Phase 3)."""
        sf_list = []
        of_list = []  # ragged — kept as Python list of arrays
        picked = []
        log_probs = []
        values = []
        ep_idx = []  # which episode each step came from (for GAE per-ep)
        for i, ep in enumerate(self.episodes):
            for s in ep.steps:
                sf_list.append(s.sf)
                of_list.append(s.of_all)
                picked.append(s.picked)
                log_probs.append(s.log_prob)
                values.append(s.value)
                ep_idx.append(i)
        return {
            "sf": np.stack(sf_list) if sf_list else np.zeros((0, 60), dtype=np.float32),
            "of_all": of_list,  # ragged
            "picked": np.array(picked, dtype=np.int64),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
            "ep_idx": np.array(ep_idx, dtype=np.int64),
            "rewards": np.array([ep.reward for ep in self.episodes], dtype=np.float32),
        }


def compute_log_prob_and_value(
    policy, sf: np.ndarray, of_all: np.ndarray, picked: int
) -> tuple[float, float]:
    """Run a forward pass to record rollout-time log_prob and value.

    Used by the agent function during episode collection to seal in the
    'old' policy's log_prob (PPO needs this for the ratio computation
    later).
    """
    probs = policy.probs_from_arrays(sf, of_all) if hasattr(policy, "probs_from_arrays") else None
    if probs is None:
        # Fallback: recompute via logits().
        # Build a minimal obs+sel stub for policy.logits(); not great but works.
        # In practice the caller (=make_training_agent_ppo) will pass sf/of_all
        # already computed, so we just do the math here.
        import torch  # noqa: PLC0415

        from train.features_v60 import option_features as _ofn  # noqa: F401, PLC0415

        device = policy.device
        sf_t = torch.from_numpy(sf).to(device)
        of_t = torch.from_numpy(of_all).to(device)
        n = of_t.shape[0]
        x = torch.cat([sf_t.unsqueeze(0).expand(n, -1), of_t], dim=1)
        with torch.no_grad():
            logits = policy.pi(x).squeeze(-1)
            ranks = torch.arange(n, device=device, dtype=torch.float32)
            logits = logits + policy.b_order * (n - 1 - ranks) / max(1, n - 1)
            log_probs = torch.log_softmax(logits, dim=0)
            v_raw = policy.v(sf_t.unsqueeze(0)).squeeze()
            v_pred = torch.tanh(v_raw).item()
            log_prob = float(log_probs[picked].item())
        return log_prob, v_pred
    return float(np.log(probs[picked] + 1e-12)), 0.0
