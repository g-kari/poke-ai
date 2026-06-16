"""Tiny linear policy over (state, option) features. Numpy-only.

logit_i = state_features(obs) @ W_state + option_features(opt_i, obs, sel) @ W_opt
p = softmax(logits)

Weights are saved to / loaded from `policy.npz`. The submission agent in
`agent.py` will use these weights if the file is present.
"""

from __future__ import annotations

import os

import numpy as np

from .features import OPTION_DIM, STATE_DIM, option_features, state_features

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "policy.npz")


class LinearPolicy:
    def __init__(
        self,
        w_state: np.ndarray | None = None,
        w_opt: np.ndarray | None = None,
        b: float = 0.0,
        w_value: np.ndarray | None = None,
    ):
        self.w_state = w_state if w_state is not None else np.zeros(STATE_DIM, dtype=np.float32)
        self.w_opt = w_opt if w_opt is not None else np.zeros(OPTION_DIM, dtype=np.float32)
        # Engine-order prior: small positive weight on "earlier option" so we
        # default to the engine's recommended ordering when untrained.
        self.b_order = float(b)
        # Value head: state_features @ w_value -> predicted terminal reward
        # from the perspective of the player about to choose. Used by PIMC
        # look-ahead as a calibrated cross-state value function (policy
        # logits cannot be reused for that — they're within-state preferences).
        self.w_value = w_value if w_value is not None else np.zeros(STATE_DIM, dtype=np.float32)

    def logits(self, obs: dict, sel: dict) -> np.ndarray:
        sf = state_features(obs)
        opts = sel["option"]
        base = float(sf @ self.w_state)
        n = len(opts)
        out = np.zeros(n, dtype=np.float32)
        for i, opt in enumerate(opts):
            of = option_features(opt, obs, sel)
            out[i] = base + float(of @ self.w_opt) + self.b_order * (n - 1 - i) / max(1, n - 1)
        return out

    def value(self, obs: dict) -> float:
        """Predicted terminal reward in roughly [-1, 1] from obs's currently-
        selecting player's perspective. Used by PIMC look-ahead."""
        sf = state_features(obs)
        # tanh-bound so an untrained value head returns ~0, not arbitrary scale
        return float(np.tanh(sf @ self.w_value))

    def probs(self, obs: dict, sel: dict) -> np.ndarray:
        z = self.logits(obs, sel)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def save(self, path: str = DEFAULT_PATH) -> None:
        np.savez(
            path,
            w_state=self.w_state,
            w_opt=self.w_opt,
            b_order=np.float32(self.b_order),
            w_value=self.w_value,
        )

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> LinearPolicy:
        d = np.load(path)
        # w_value is optional for backward compat with older policy.npz.
        w_value = d["w_value"] if "w_value" in d.files else None
        return cls(d["w_state"], d["w_opt"], float(d["b_order"]), w_value)

    @classmethod
    def try_load(cls, path: str = DEFAULT_PATH) -> LinearPolicy | None:
        if not os.path.exists(path):
            return None
        p = cls.load(path)
        # Reject weights from a previous feature-dim layout — agent.py will
        # then fall back to the engine-order baseline instead of crashing.
        if p.w_state.shape[0] != STATE_DIM or p.w_opt.shape[0] != OPTION_DIM:
            return None
        return p
