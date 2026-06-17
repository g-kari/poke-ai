"""Ensemble of MlpPolicy checkpoints. Averages logits across N models.

A single MLP trained on self-play tends to narrow its policy distribution
toward whatever pattern wins the mirror match; that's how the 2000ep MLP
beat linear but lost to rule_based. Averaging across independently-trained
MLPs cancels the per-seed quirks while preserving the moves all the
seeds agree on, which is the standard cheap fix for the over-confidence
problem.

The ensemble exposes the same `.logits(obs, sel)` / `.value(obs)` /
`.probs(obs, sel)` interface as MlpPolicy so main.py can drop it in
without further changes.

Construction:
    ens = EnsemblePolicy([MlpPolicy.load(p) for p in paths])
or
    ens = EnsemblePolicy.try_load(["train/mlp_policy.pt",
                                   "train/mlp_policy_seed2.pt"])
"""

from __future__ import annotations

import os

import numpy as np

from .mlp_policy import MlpPolicy


class EnsemblePolicy:
    def __init__(self, members: list[MlpPolicy]) -> None:
        if not members:
            raise ValueError("EnsemblePolicy needs at least one member")
        self.members = members
        # Engine-order bias is the same on every member, but expose it so
        # callers that look at .b_order (e.g. logging) still work.
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
    def try_load(cls, paths: list[str], device: str | None = None) -> EnsemblePolicy | None:
        members: list[MlpPolicy] = []
        for path in paths:
            if not os.path.exists(path):
                continue
            m = MlpPolicy.try_load(path, device=device)
            if m is not None:
                members.append(m)
        if not members:
            return None
        return cls(members)
