"""PTCGABC submission entrypoint. Place at the top of the tar.gz; Kaggle mounts
the bundle at /kaggle_simulations/agent/.

Decision stack (top wins, falls through on failure):
  1. PIMC 1-ply lookahead on MAIN selects (single mirror-deck sample)
  2. Linear policy logits (trained numpy weights in train/policy.npz)
  3. Engine option-order prior (always-pick-index-0) — beats random 7-1
"""

from __future__ import annotations

import os
import random
from typing import Any

_RNG = random.Random(20260616)
# Turn PIMC on/off without re-deploying. Defaults to OFF: the current
# heuristic value drops the agent from 95% to 55% vs random_agent because
# the engine option-order prior is already very strong on a random opponent.
# Re-enable with POKEAI_PIMC=1 when testing against stronger opponents
# (where look-ahead actually pays off).
_PIMC_ENABLED = os.environ.get("POKEAI_PIMC", "0") == "1"


def _read_deck() -> list[int]:
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    deck = [int(x) for x in lines[:60]]
    if len(deck) != 60:
        raise ValueError(f"deck.csv must contain 60 card IDs (got {len(deck)})")
    return deck


_DECK = _read_deck()


def _try_load_policy():
    try:
        from train.policy import LinearPolicy  # noqa: PLC0415

        return LinearPolicy.try_load()
    except Exception:
        return None


_POLICY = _try_load_policy()


def _pimc_pick(obs: dict[str, Any], sel: dict[str, Any]) -> int | None:
    """Return a PIMC look-ahead pick for MAIN selects, or None to fall through."""
    if not _PIMC_ENABLED or _POLICY is None:
        return None
    if sel.get("type") != 0:  # MAIN only
        return None
    try:
        from train.pimc import pick_best_option  # noqa: PLC0415

        return pick_best_option(obs, sel, _DECK, _POLICY)
    except Exception:
        return None


def agent(obs: dict[str, Any]) -> list[int]:
    sel = obs.get("select")
    if sel is None:
        return list(_DECK)

    options = sel.get("option") or []
    max_c = int(sel.get("maxCount") or 0)
    min_c = int(sel.get("minCount") or 0)

    if not options or max_c == 0:
        return []

    # PIMC look-ahead for MAIN single-pick selects.
    if max_c == 1:
        pimc_pick = _pimc_pick(obs, sel)
        if pimc_pick is not None:
            return [int(pimc_pick)]

    if _POLICY is not None:
        import numpy as np  # noqa: PLC0415

        logits = _POLICY.logits(obs, sel)
        order = np.argsort(-logits)
        k = max(min_c, 1) if max_c >= 1 else min_c
        k = min(k, max_c, len(options))
        return [int(x) for x in order[:k].tolist()]

    k = max(min_c, 1) if max_c >= 1 else min_c
    k = min(k, max_c, len(options))
    return list(range(k))
