"""Extended 60-d state features with opponent deck-type fingerprint.

State layout (delta vs v40):
  f[0..39]: identical to features.state_features() (40-d core)
  f[40..55]: opp deck-type hash buckets (16 each)
            - active card-IDs (weight 2.0)
            - bench card-IDs  (weight 1.0)
            - discard card-IDs (weight 0.5)
            normalized so total = 1.0 across the source
  f[56..59]: own active card-id 4-bucket hash (mirror sense)

This is the "deck fingerprint" addition motivated by NOTES.md — the pool
training showed the policy can't distinguish Iono from Crustle without
opp deck-ID signal in the state, so we add it here.

Kept in a separate module from features.py so existing 40-d trained
policies (mlp_policy.pt et al.) still load correctly and the 3-MLP
submission stays alive.
"""

from __future__ import annotations

import numpy as np

from .features import (
    OPTION_DIM,
    _accumulate_card_buckets,
)
from .features import (
    option_features as _option_features_v40,
)
from .features import (
    state_features as _state_features_v40,
)

STATE_DIM = 60


def state_features(obs: dict) -> np.ndarray:
    """Compose v40 core + opp deck fingerprint into a 60-d vector."""
    f = np.zeros(STATE_DIM, dtype=np.float32)
    core = _state_features_v40(obs)
    f[:40] = core
    cur = obs.get("current")
    if cur is None:
        return f
    you = cur["yourIndex"]
    me = cur["players"][you]
    opp = cur["players"][1 - you]

    # Opponent deck fingerprint: 16 buckets, each gets normalized weighted
    # hits from active(2.0) + bench(1.0) + discard(0.5).
    _accumulate_card_buckets(f, 40, opp.get("active") or [], weight=2.0)
    _accumulate_card_buckets(f, 40, opp.get("bench") or [], weight=1.0)
    _accumulate_card_buckets(f, 40, opp.get("discard") or [], weight=0.5)
    # Our own active card-id 4-bucket hash for mirror sense.
    _accumulate_card_buckets(f, 56, me.get("active") or [], weight=2.0, n_buckets=4)
    return f


def option_features(opt: dict, obs: dict, sel: dict) -> np.ndarray:
    """Reuse v40 option features (OPTION_DIM unchanged)."""
    return _option_features_v40(opt, obs, sel)


__all__ = ["STATE_DIM", "OPTION_DIM", "state_features", "option_features"]
