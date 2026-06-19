"""PPO Phase 3: clipped surrogate loss + value MSE + entropy bonus.

The PPO objective replaces REINFORCE's `-A * log_pi(a|s)` with:

    L_CLIP = -E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)]
        where ratio = exp(log_pi_new(a|s) - log_pi_old(a|s))

This caps the policy update per batch — if a single sample would push
the policy too far (= ratio > 1+eps and advantage positive), the clip
limits the gradient. Prevents the catastrophic-collapse pattern we
observed in V60 EXT4, s200ext, BCRL3.

Full loss:
    L = L_CLIP + value_coef * L_V - entropy_coef * H(pi)

with k_epochs of mini-batch updates between rollouts.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_logits_and_value(
    policy, sf: torch.Tensor, of_all: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward pass: returns (logits over options, value scalar).

    Args:
        sf: (60,) state features tensor on policy.device
        of_all: (n_opts, 40) option features tensor on policy.device
    """
    n = of_all.shape[0]
    x = torch.cat([sf.unsqueeze(0).expand(n, -1), of_all], dim=1)
    logits = policy.pi(x).squeeze(-1)
    ranks = torch.arange(n, device=policy.device, dtype=torch.float32)
    logits = logits + policy.b_order * (n - 1 - ranks) / max(1, n - 1)
    v_raw = policy.v(sf.unsqueeze(0)).squeeze()
    value = torch.tanh(v_raw)
    return logits, value


def ppo_loss_per_step(
    logits: torch.Tensor,
    value: torch.Tensor,
    picked: int,
    log_prob_old: float,
    advantage: float,
    return_target: float,
    eps_clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-step PPO loss components.

    Returns:
        loss (scalar): total loss for backprop
        policy_loss (scalar)
        value_loss (scalar)
        entropy (scalar) — for logging
    """
    log_probs_new = torch.log_softmax(logits, dim=0)
    log_prob_new = log_probs_new[picked]
    ratio = torch.exp(log_prob_new - log_prob_old)

    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantage
    policy_loss = -torch.min(surr1, surr2)

    value_loss = (value - return_target).pow(2)

    probs = log_probs_new.exp()
    entropy = -(probs * log_probs_new).sum()

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return loss, policy_loss.detach(), value_loss.detach(), entropy.detach()


def ppo_update(
    policy,
    optimizer: torch.optim.Optimizer,
    buffer_flat: dict,
    advantages: np.ndarray,
    returns: np.ndarray,
    k_epochs: int = 4,
    mb_size: int = 32,
    eps_clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    clip_grad: float = 0.5,
) -> dict:
    """Run k_epochs of mini-batch PPO updates on the rollout buffer.

    Args:
        policy: MlpPolicyV60 instance
        optimizer: torch.optim.Adam(policy.parameters())
        buffer_flat: PPOBuffer.flatten() output
        advantages, returns: from compute_gae_for_buffer + normalize_advantages
        k_epochs: number of passes over the buffer
        mb_size: mini-batch size (in steps)
        eps_clip, value_coef, entropy_coef, clip_grad: standard PPO hp

    Returns:
        metrics dict (mean policy_loss, value_loss, entropy)
    """
    sf_all = buffer_flat["sf"]
    of_all_list = buffer_flat["of_all"]
    picked = buffer_flat["picked"]
    log_probs_old = buffer_flat["log_probs"]

    n = len(picked)
    device = policy.device

    metrics = {"policy_loss": [], "value_loss": [], "entropy": []}

    for _ in range(k_epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, mb_size):
            mb = idx[start : start + mb_size]
            optimizer.zero_grad()
            mb_loss = torch.zeros(1, device=device)
            pl_acc, vl_acc, ent_acc = 0.0, 0.0, 0.0
            for j in mb:
                sf = torch.from_numpy(sf_all[j]).to(device)
                of = torch.from_numpy(of_all_list[j]).to(device)
                logits, value = compute_logits_and_value(policy, sf, of)
                loss_j, pl, vl, ent = ppo_loss_per_step(
                    logits,
                    value,
                    picked=int(picked[j]),
                    log_prob_old=float(log_probs_old[j]),
                    advantage=float(advantages[j]),
                    return_target=float(returns[j]),
                    eps_clip=eps_clip,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                )
                mb_loss = mb_loss + loss_j
                pl_acc += float(pl.item())
                vl_acc += float(vl.item())
                ent_acc += float(ent.item())
            n_mb = len(mb)
            mb_loss = mb_loss / max(1, n_mb)
            mb_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
            optimizer.step()
            metrics["policy_loss"].append(pl_acc / n_mb)
            metrics["value_loss"].append(vl_acc / n_mb)
            metrics["entropy"].append(ent_acc / n_mb)

    return {
        "policy_loss": float(np.mean(metrics["policy_loss"])) if metrics["policy_loss"] else 0.0,
        "value_loss": float(np.mean(metrics["value_loss"])) if metrics["value_loss"] else 0.0,
        "entropy": float(np.mean(metrics["entropy"])) if metrics["entropy"] else 0.0,
    }


__all__ = [
    "compute_logits_and_value",
    "ppo_loss_per_step",
    "ppo_update",
]
