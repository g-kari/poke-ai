"""PyTorch MLP policy & value head, matching LinearPolicy's interface.

logits[i] = mlp_pi(concat(state_features, option_features_i))
value      = tanh(mlp_v(state_features))

Forward passes are batched over options per call. Save/load uses PyTorch
state_dict in a .pt file. The hidden-layer widths are deliberately small
(64 / 32) so the per-call latency stays well under 1ms — the cabt engine
is the bottleneck at ~0.1-0.3s per game, so the policy doesn't need to
be fast, just expressive.

The policy keeps the same engine-order bias as LinearPolicy
(self.b_order * (n-1-i) / (n-1)) so an untrained MlpPolicy already
matches first_agent and beats random ~7-1.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .features import OPTION_DIM, STATE_DIM, option_features, state_features

DEFAULT_PATH = str(Path(__file__).resolve().parent / "mlp_policy.pt")
DEFAULT_HIDDEN_PI = (64, 32)
DEFAULT_HIDDEN_V = (32,)


class MlpPolicy(nn.Module):
    """MLP-based policy + value head with LinearPolicy.logits()-compatible API."""

    def __init__(
        self,
        hidden_pi: tuple[int, ...] = DEFAULT_HIDDEN_PI,
        hidden_v: tuple[int, ...] = DEFAULT_HIDDEN_V,
        b_order: float = 2.0,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.b_order = float(b_order)
        self.hidden_pi = tuple(hidden_pi)
        self.hidden_v = tuple(hidden_v)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Policy head: input = state + option features.
        in_dim = STATE_DIM + OPTION_DIM
        layers: list[nn.Module] = []
        prev = in_dim
        for w in self.hidden_pi:
            layers.append(nn.Linear(prev, w))
            layers.append(nn.ReLU())
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.pi = nn.Sequential(*layers)

        # Value head: input = state features only.
        layers = []
        prev = STATE_DIM
        for w in self.hidden_v:
            layers.append(nn.Linear(prev, w))
            layers.append(nn.ReLU())
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.v = nn.Sequential(*layers)

        self.to(self.device)

    # ------------------------------------------------------------------
    # LinearPolicy-compatible API
    # ------------------------------------------------------------------

    def logits(self, obs: dict, sel: dict) -> np.ndarray:
        sf = state_features(obs)
        opts = sel["option"]
        n = len(opts)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        of_all = np.stack([option_features(o, obs, sel) for o in opts])
        with torch.no_grad():
            x = torch.cat(
                [
                    torch.from_numpy(np.broadcast_to(sf, (n, STATE_DIM)).copy()),
                    torch.from_numpy(of_all),
                ],
                dim=1,
            ).to(self.device)
            logits = self.pi(x).squeeze(-1).cpu().numpy().astype(np.float32)
        # Engine-order prior, identical to LinearPolicy.logits().
        if n > 1:
            ranks = np.arange(n, dtype=np.float32)
            logits = logits + self.b_order * (n - 1 - ranks) / (n - 1)
        return logits

    def value(self, obs: dict) -> float:
        sf = state_features(obs)
        with torch.no_grad():
            x = torch.from_numpy(sf[None, :]).to(self.device)
            v = torch.tanh(self.v(x)).item()
        return float(v)

    def probs(self, obs: dict, sel: dict) -> np.ndarray:
        z = self.logits(obs, sel)
        if z.size == 0:
            return z
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    # ------------------------------------------------------------------
    # Save / load — separate from LinearPolicy's policy.npz to avoid
    # confusion and let main.py try MLP first, fall back to linear.
    # ------------------------------------------------------------------

    def save(self, path: str = DEFAULT_PATH) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hidden_pi": list(self.hidden_pi),
                "hidden_v": list(self.hidden_v),
                "b_order": self.b_order,
            },
            path,
        )

    @classmethod
    def load(cls, path: str = DEFAULT_PATH, device: str | None = None) -> MlpPolicy:
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=True)
        p = cls(
            hidden_pi=tuple(ckpt["hidden_pi"]),
            hidden_v=tuple(ckpt["hidden_v"]),
            b_order=ckpt["b_order"],
            device=device,
        )
        p.load_state_dict(ckpt["state_dict"])
        p.eval()
        return p

    @classmethod
    def try_load(cls, path: str = DEFAULT_PATH, device: str | None = None) -> MlpPolicy | None:
        if not os.path.exists(path):
            return None
        try:
            return cls.load(path, device=device)
        except Exception:
            return None
