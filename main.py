"""PTCGABC submission entrypoint. Place at the top of the tar.gz; Kaggle mounts
the bundle at /kaggle_simulations/agent/.

The agent uses a trained numpy linear policy (train/policy.npz) if present and
otherwise falls back to the engine's option-order prior, which already beats
random ~7-1 in mirror self-play.
"""

from __future__ import annotations

import os
import random
from typing import Any

_RNG = random.Random(20260616)


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


def agent(obs: dict[str, Any]) -> list[int]:
    sel = obs.get("select")
    if sel is None:
        return list(_DECK)

    options = sel.get("option") or []
    max_c = int(sel.get("maxCount") or 0)
    min_c = int(sel.get("minCount") or 0)

    if not options or max_c == 0:
        return []

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
