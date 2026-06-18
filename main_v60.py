"""Submission entry point for the V60 learned single-policy agent.

Built from train/mlp_policy_v60_pool5_ext*.pt (60-d features with opp
deck-id fingerprint). This is the (C) deep-learning route — same deck
as our 3-MLP submission, but driven by the single V60 policy that learns
opponent-aware routing from features_v60's bucket hash.

Lab status (latest EXT — see NOTES):
  V60 EXT1 (5500ep): solo overall ~21% @ 30g
  V60 EXT3 target: 25%+ to beat 3-MLP (23.3%)

How to ship this:
  bash make_submission_v60.sh
  .venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \
      -f submission_v60.tar.gz -m "..."

Falls back to engine-prior (option 0) on any inference error, so it
never crashes at runtime.
"""

from __future__ import annotations

import contextlib
import glob
import sys
from pathlib import Path


def _resolve_here() -> Path:
    with contextlib.suppress(NameError):
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


_HERE = _resolve_here()
sys.path.insert(0, str(_HERE))


def _read_deck() -> list[int]:
    deck_path = _HERE / "deck.csv"
    with open(deck_path) as f:
        return [int(line.strip()) for line in f if line.strip()]


_DECK = _read_deck()


def _try_load_v60():
    """Load V60 policies from the bundle.

    Decision stack:
      1. EnsemblePolicyV60 if 2+ V60 .pt files exist (= EXT1 + EXT3 etc.)
      2. Single MlpPolicyV60 if exactly one exists
      3. None on any failure (caller will fall back to engine-prior)
    """
    candidates = sorted(glob.glob(str(_HERE / "train" / "mlp_policy_v60*.pt")))
    if not candidates:
        return None, None
    try:
        import numpy as np  # noqa: PLC0415

        from train.mlp_policy_v60 import MlpPolicyV60  # noqa: PLC0415

        if len(candidates) >= 2:
            from train.ensemble_policy_v60 import EnsemblePolicyV60  # noqa: PLC0415

            ens = EnsemblePolicyV60.try_load(candidates)
            if ens is not None and len(ens.members) >= 2:
                return ens, np.random.default_rng(0)
        # Fall through to single-policy.
        policy = MlpPolicyV60.load(candidates[-1])
        policy.eval()
        return policy, np.random.default_rng(0)
    except Exception:
        return None, None


_POLICY, _RNG = _try_load_v60()


def agent(obs: dict) -> list[int]:
    sel = obs.get("select")
    if sel is None:
        return list(_DECK)
    opts = sel.get("option") or []
    if not opts:
        return []
    max_c = int(sel.get("maxCount") or 0)
    min_c = int(sel.get("minCount") or 0)
    if max_c == 0:
        return []

    if _POLICY is None:
        # Engine-prior fallback.
        k = max(min_c, 1)
        k = min(k, max_c, len(opts))
        return list(range(k))

    try:
        if sel.get("type") == 0 and max_c == 1:
            import numpy as np  # noqa: PLC0415

            probs = _POLICY.probs(obs, sel)
            return [int(_RNG.choice(len(opts), p=probs))]
        if max_c >= 1:
            import numpy as np  # noqa: PLC0415

            logits = _POLICY.logits(obs, sel)
            order = np.argsort(-logits)
            k = max(min_c, 1)
            k = min(k, max_c, len(opts))
            return [int(x) for x in order[:k].tolist()]
    except Exception:
        pass

    k = max(min_c, 1)
    k = min(k, max_c, len(opts))
    return list(range(k))
