"""MlpPolicyV60: same shape as MlpPolicy but driven by features_v60.

Kept as a sibling module so 40-d trained policies (mlp_policy.pt et al.)
still load cleanly via MlpPolicy. Pre-trained 40-d and 60-d weights can
coexist in train/ as long as the loader picks the right class.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features_v60 import OPTION_DIM, STATE_DIM, option_features, state_features

DEFAULT_PATH = str(Path(__file__).resolve().parent / "mlp_policy_v60.pt")
DEFAULT_HIDDEN_PI = (64, 32)
DEFAULT_HIDDEN_V = (32,)


class MlpPolicyV60(nn.Module):
    """MLP policy + value head over the 60-d (v60) state vector."""

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

        in_dim = STATE_DIM + OPTION_DIM
        layers: list[nn.Module] = []
        prev = in_dim
        for w in self.hidden_pi:
            layers.append(nn.Linear(prev, w))
            layers.append(nn.ReLU())
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.pi = nn.Sequential(*layers)

        layers = []
        prev = STATE_DIM
        for w in self.hidden_v:
            layers.append(nn.Linear(prev, w))
            layers.append(nn.ReLU())
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.v = nn.Sequential(*layers)

        self.to(self.device)

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
            raw = self.pi(x).squeeze(-1).cpu().numpy()
        ranks = np.arange(n, dtype=np.float32)
        bias = self.b_order * (n - 1 - ranks) / max(1, n - 1)
        return raw + bias

    def probs(self, obs: dict, sel: dict) -> np.ndarray:
        z = self.logits(obs, sel)
        if z.size == 0:
            return z
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def value(self, obs: dict) -> float:
        sf = state_features(obs)
        with torch.no_grad():
            x = torch.from_numpy(sf).unsqueeze(0).to(self.device)
            v = self.v(x).squeeze().item()
        return float(np.tanh(v))

    def save(self, path: str = DEFAULT_PATH) -> None:
        torch.save(
            {
                "hidden_pi": list(self.hidden_pi),
                "hidden_v": list(self.hidden_v),
                "b_order": self.b_order,
                "state_dim": STATE_DIM,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str = DEFAULT_PATH, device: str | None = None) -> MlpPolicyV60:
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
    def try_load(cls, path: str = DEFAULT_PATH, device: str | None = None) -> MlpPolicyV60 | None:
        if not os.path.exists(path):
            return None
        try:
            return cls.load(path, device=device)
        except Exception:
            return None
