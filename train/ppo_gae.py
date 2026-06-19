"""PPO Phase 2: GAE (Generalized Advantage Estimation) for sparse-reward PTCG.

GAE blends 1-step to n-step advantages with a lambda parameter:
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t = delta_t + gamma * lambda * A_{t+1}

For sparse-reward games (= reward only at terminal), this reduces to:
    A_t = sum over future deltas weighted by (gamma * lambda)^k

Compared to plain REINFORCE (= advantage = R - V(s)), GAE:
- variance is tunable via lambda (1.0 = pure MC, 0.0 = pure 1-step TD)
- bias is bounded by V(s) quality
- LP-friendly for high-noise sparse-reward setting

Reference: Schulman et al. 2016, "High-Dimensional Continuous Control
Using Generalized Advantage Estimation"

For PTCG (= sparse +/-1 terminal), recommended hyperparameters:
- gamma = 0.99 (= small discount, 30+ turn games)
- lambda = 0.95 (= smoothness vs variance)
"""

from __future__ import annotations

import numpy as np


def compute_gae_per_episode(
    values: np.ndarray,
    reward: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE for ONE episode in PTCG (= sparse terminal reward).

    Args:
        values: V(s_t) for each decision step, shape (T,). NOTE: not
            (T+1,) — for sparse PTCG we treat the implicit V(s_T) = 0
            because the game ends after the last step.
        reward: terminal reward (+1 win, -1 loss, 0 draw).
        gamma: discount factor (default 0.99).
        lam: GAE lambda (default 0.95).

    Returns:
        advantages: shape (T,) with A_t
        returns: shape (T,) with V_target = A_t + V(s_t) for value loss
    """
    T = len(values)
    if T == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    advantages = np.zeros(T, dtype=np.float32)
    last_advantage = 0.0
    # PTCG sparse-reward: r_t = 0 for all t < T-1, r_{T-1} = reward.
    # V(s_T) (= post-terminal) = 0.
    for t in reversed(range(T)):
        r_t = reward if t == T - 1 else 0.0
        v_next = 0.0 if t == T - 1 else float(values[t + 1])
        delta = r_t + gamma * v_next - float(values[t])
        last_advantage = delta + gamma * lam * last_advantage
        advantages[t] = last_advantage

    returns = advantages + values
    return advantages, returns


def compute_gae_for_buffer(
    buffer_flat: dict,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Run GAE for all episodes in a PPOBuffer.flatten() dict.

    Args:
        buffer_flat: output of PPOBuffer.flatten() with keys
            ep_idx, values, rewards.

    Returns:
        all_advantages: shape (total_steps,) — advantage per step
        all_returns: shape (total_steps,) — V_target per step
    """
    ep_idx = buffer_flat["ep_idx"]
    values = buffer_flat["values"]
    rewards = buffer_flat["rewards"]
    n_eps = len(rewards)

    all_advantages = np.zeros_like(values)
    all_returns = np.zeros_like(values)

    for e in range(n_eps):
        mask = ep_idx == e
        ep_values = values[mask]
        adv, ret = compute_gae_per_episode(ep_values, float(rewards[e]), gamma=gamma, lam=lam)
        all_advantages[mask] = adv
        all_returns[mask] = ret

    return all_advantages, all_returns


def normalize_advantages(advantages: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standard PPO trick: zero-mean unit-std advantages per batch.

    Helps gradient scaling stability across batches with very different
    reward magnitudes."""
    if len(advantages) == 0:
        return advantages
    mean = advantages.mean()
    std = advantages.std() + eps
    return (advantages - mean) / std
