"""Pure-numpy inference for MlpPolicyV60. No torch required at runtime.

Loads weights from a `.npz` extracted by scripts/extract_v60_weights.py
(or directly from a torch `.pt` if torch is available). Exposes the same
.logits() / .value() / .probs() API as MlpPolicyV60 so main_v60 can drop
it in.

Why: the Kaggle cabt submission runtime does not ship torch. Our previous
V60 submission (53810836) ERRORed because MlpPolicyV60 loading silently
failed and we fell back to engine-prior, which can't sustain a real game.
This module sidesteps torch entirely.
"""

from __future__ import annotations

import os

import numpy as np

from .features_v60 import OPTION_DIM, STATE_DIM, option_features, state_features


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class MlpPolicyV60Numpy:
    """Numpy implementation of MlpPolicyV60 inference.

    Weights are dicts with keys "pi.{i}.weight" / "pi.{i}.bias" /
    "v.{i}.weight" / "v.{i}.bias" mirroring the torch state_dict layout
    (linear layers indexed at 0, 2, 4, ... after every ReLU).
    """

    def __init__(
        self,
        pi_layers: list[tuple[np.ndarray, np.ndarray]],
        v_layers: list[tuple[np.ndarray, np.ndarray]],
        b_order: float = 2.0,
    ) -> None:
        self.pi_layers = pi_layers  # list of (W, b)
        self.v_layers = v_layers
        self.b_order = float(b_order)

    # ----- forward helpers -----
    def _pi_forward(self, x: np.ndarray) -> np.ndarray:
        for i, (w, b) in enumerate(self.pi_layers):
            x = x @ w.T + b
            if i < len(self.pi_layers) - 1:
                x = _relu(x)
        return x

    def _v_forward(self, sf: np.ndarray) -> np.ndarray:
        x = sf
        for i, (w, b) in enumerate(self.v_layers):
            x = x @ w.T + b
            if i < len(self.v_layers) - 1:
                x = _relu(x)
        return x

    # ----- LinearPolicy-compatible API -----
    def logits(self, obs: dict, sel: dict) -> np.ndarray:
        sf = state_features(obs)
        opts = sel["option"]
        n = len(opts)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        of_all = np.stack([option_features(o, obs, sel) for o in opts]).astype(np.float32)
        sf_b = np.broadcast_to(sf, (n, STATE_DIM)).astype(np.float32)
        x = np.concatenate([sf_b, of_all], axis=1)
        raw = self._pi_forward(x).squeeze(-1)
        ranks = np.arange(n, dtype=np.float32)
        bias = self.b_order * (n - 1 - ranks) / max(1, n - 1)
        return (raw + bias).astype(np.float32)

    def probs(self, obs: dict, sel: dict) -> np.ndarray:
        z = self.logits(obs, sel)
        if z.size == 0:
            return z
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def value(self, obs: dict) -> float:
        sf = state_features(obs).astype(np.float32)
        v = self._v_forward(sf.reshape(1, -1)).squeeze()
        return float(np.tanh(v))

    # ----- weight loading -----
    @classmethod
    def from_state_dict(cls, state_dict: dict, b_order: float = 2.0) -> MlpPolicyV60Numpy:
        """Convert a torch state_dict dump (dict of numpy arrays) into the
        layered representation we use. Linear modules are at sequential
        indices 0, 2, 4, ... because of the ReLU interleaving."""
        pi_keys = sorted(
            int(k.split(".")[1])
            for k in state_dict
            if k.startswith("pi.") and k.endswith(".weight")
        )
        v_keys = sorted(
            int(k.split(".")[1]) for k in state_dict if k.startswith("v.") and k.endswith(".weight")
        )
        pi_layers = [(state_dict[f"pi.{i}.weight"], state_dict[f"pi.{i}.bias"]) for i in pi_keys]
        v_layers = [(state_dict[f"v.{i}.weight"], state_dict[f"v.{i}.bias"]) for i in v_keys]
        # sanity
        assert pi_layers[0][0].shape[1] == STATE_DIM + OPTION_DIM, (
            f"pi input shape mismatch: {pi_layers[0][0].shape}"
        )
        assert v_layers[0][0].shape[1] == STATE_DIM, (
            f"v input shape mismatch: {v_layers[0][0].shape}"
        )
        return cls(pi_layers, v_layers, b_order=b_order)

    @classmethod
    def load_npz(cls, path: str) -> MlpPolicyV60Numpy:
        z = np.load(path, allow_pickle=False)
        state_dict = {k: np.asarray(z[k]) for k in z.files if not k.startswith("__")}
        b_order = float(z["__b_order__"]) if "__b_order__" in z.files else 2.0
        return cls.from_state_dict(state_dict, b_order=b_order)

    @classmethod
    def try_load(cls, path: str) -> MlpPolicyV60Numpy | None:
        if not os.path.exists(path):
            return None
        try:
            return cls.load_npz(path)
        except Exception:
            return None
