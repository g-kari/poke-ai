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


def _candidate_roots() -> list[Path]:
    """All directories we'll try for deck.csv + train/ resolution.

    Kaggle's runtime path layout varies (the previous V60 ERRORs at
    53810836 / 53812115 / 53812882 didn't have /kaggle_simulations/agent
    accessible the way we expected). We aggressively scan: __file__ parent,
    cwd, and any path we can find that contains a deck.csv.
    """
    roots: list[Path] = []
    with contextlib.suppress(NameError):
        roots.append(Path(__file__).resolve().parent)
    roots.append(Path.cwd().resolve())
    # Common Kaggle mounts.
    for fixed in (
        "/kaggle_simulations/agent",
        "/kaggle_simulations",
        "/kaggle/working",
        "/kaggle/input",
    ):
        roots.append(Path(fixed))
    # Walk sys.path for any entry that has a deck.csv (= our bundle).
    for p in sys.path:
        if not p:
            continue
        roots.append(Path(p))
    # Dedupe while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


_ROOTS = _candidate_roots()
# Ensure all candidate roots are importable so `from train.x import ...` works.
for _r in _ROOTS:
    if _r.exists() and str(_r) not in sys.path:
        sys.path.insert(0, str(_r))


def _read_deck() -> list[int]:
    for r in _ROOTS:
        p = r / "deck.csv"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return [int(line.strip()) for line in f if line.strip()]
    raise FileNotFoundError("deck.csv not found in any candidate root")


_DECK = _read_deck()


def _try_load_v60():
    """Load V60 policy from the bundle. Pure-numpy inference (no torch
    required at runtime). The bundle ships .npz files extracted from .pt
    by scripts/extract_v60_weights.py at build time."""
    candidates: list[str] = []
    for r in _ROOTS:
        candidates.extend(sorted(glob.glob(str(r / "train" / "mlp_policy_v60*.npz"))))
    if not candidates:
        return None, None
    try:
        import numpy as np  # noqa: PLC0415

        from train.mlp_policy_v60_numpy import MlpPolicyV60Numpy  # noqa: PLC0415

        policy = MlpPolicyV60Numpy.load_npz(candidates[-1])
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
