"""Ensemble of MlpPolicyV60 checkpoints. Mirrors ensemble_policy.py.

Averages logits across N V60 members. Used by main_v60.py when more than
one V60 .pt is present in the bundle (= EXT1 + EXT3 ensemble candidate).
"""

from __future__ import annotations

import os

import numpy as np

from .mlp_policy_v60 import MlpPolicyV60


class EnsemblePolicyV60:
    def __init__(self, members: list[MlpPolicyV60]) -> None:
        if not members:
            raise ValueError("EnsemblePolicyV60 needs at least one member")
        self.members = members
        self.b_order = float(members[0].b_order)

    def logits(self, obs: dict, sel: dict) -> np.ndarray:
        per_member = [m.logits(obs, sel) for m in self.members]
        return np.mean(np.stack(per_member, axis=0), axis=0).astype(np.float32)

    def value(self, obs: dict) -> float:
        return float(np.mean([m.value(obs) for m in self.members]))

    def probs(self, obs: dict, sel: dict) -> np.ndarray:
        z = self.logits(obs, sel)
        if z.size == 0:
            return z
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    @classmethod
    def try_load(cls, paths: list[str], device: str | None = None) -> EnsemblePolicyV60 | None:
        """Best-effort load: skip files that don't exist or fail to load."""
        members: list[MlpPolicyV60] = []
        for p in paths:
            if not os.path.exists(p):
                continue
            try:
                members.append(MlpPolicyV60.load(p, device=device))
            except Exception:
                continue
        if not members:
            return None
        return cls(members)
