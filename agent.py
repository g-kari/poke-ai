"""
PTCGABC (cabt) baseline agent.

Engine import paths (verified against kaggle-environments==1.30.1):
  - The env's Python wrappers live at
    `kaggle_environments.envs.cabt.cg.game` (battle_start / battle_select / ...)
    and `...cg.sim` (ctypes bindings + Battle singleton).
  - There is NO `cabt` top-level package, NO `cabt.api` module, and the
    Python layer does NOT expose `all_card_data() / all_attack() /
    to_observation_class() / search_begin() / search_step() / search_end() /
    search_release()` that the HANDOVER assumed. The symbols exist in
    `libcg.so` (`AllCard`, `AllAttack`, `SearchBegin`, `SearchStep`,
    `SearchEnd`, `SearchRelease`) but the calling conventions are not
    documented and not wired through Python in the shipped env.
  - obs is a plain `dict`, not a class. Keys: `select`, `logs`, `current`,
    `search_begin_input`. `search_begin_input` is an opaque ~80-char ASCII
    string (serialized state for SearchBegin).

Observation shape (empirical, mirror-deck self-play):
  obs["current"]["yourIndex"] -> 0 or 1
  obs["current"]["result"]    -> -1 while playing, 0/1=winner, 2=draw
  obs["current"]["players"][i] keys: active, bench, benchMax, deckCount,
    discard, prize, handCount, hand (None for opponent), poisoned, burned,
    asleep, paralyzed, confused.
  obs["select"]: {type, context, minCount, maxCount, option, ...} or None.
  Most-frequent (type, context) pairs in self-play:
    (0, 0)   MAIN/MAIN              ~65 % of decisions
    (1, 3)   CARD/DISCARD
    (1, 7)   CARD/...               (area=DECK looking)
    (1, 1)   CARD/SETUP_ACTIVE
    (1, 2)   CARD/SETUP_BENCH
    (1, 4)   CARD/SWITCH
    (1, 22)  CARD/...               (area=DISCARD)
    (8, 38)  COIN_HEAD or NUMBER    options have {type:0, number:N}
    (9, 41)  YES_NO/IS_FIRST

MAIN OptionType values observed:
   7 = PLAY    (item/supporter from hand: fields {index})
   8 = ATTACH  (energy attach: {area, index, inPlayArea, inPlayIndex})
   9 = EVOLVE  (evolution: {area, index, inPlayArea, inPlayIndex})
  13 = ATTACK  (declare attack: {attackId})
  14 = END     (end turn: no fields)
  Other values (ABILITY / DISCARD / RETREAT) may appear but were not hit in
  baseline self-play.

Why this baseline is "pick option 0":
  The C++ engine emits options in a sensible order — the bundled `first_agent`
  (always indices `range(maxCount)`) beats `random_agent` 7-1 in 8 mirror
  games, while a naive ATTACK-first heuristic loses 0-8 (it attacks before
  building energy). Until a learned or search-driven policy is wired in, the
  index-0 default is the strongest cheap baseline.

Returns:
  list[int] of option indices, length in [minCount, maxCount]. If select is
  None (initial step), return the full 60-card deck list.
"""

from __future__ import annotations

import random
from typing import Any

# Submission decklist. Replace with your tuned 60-card list (card IDs from
# the engine's AllCard catalog). The current list is the sample shipped with
# the env so self-play is reproducible out of the box.
DECK: list[int] = [
    721, 721, 722, 722, 722, 722, 723, 723, 723, 723,
    1092, 1121, 1121, 1145, 1145, 1163, 1163, 1219, 1219, 1219, 1219,
    1227, 1227, 1227, 1227, 1262, 1262,
] + [3] * 33


_RNG = random.Random(20260616)


def _try_load_policy():
    """Best-effort load of trained weights. Returns None if unavailable."""
    try:
        from train.policy import LinearPolicy  # noqa: PLC0415
        return LinearPolicy.try_load()
    except Exception:
        return None


_POLICY = _try_load_policy()


def agent(obs: dict[str, Any]) -> list[int]:
    sel = obs.get("select")
    if sel is None:
        return list(DECK)

    options = sel.get("option") or []
    max_c = int(sel.get("maxCount") or 0)
    min_c = int(sel.get("minCount") or 0)

    if not options or max_c == 0:
        return []

    # Trained linear policy: rank options by predicted logit.
    if _POLICY is not None:
        import numpy as np  # noqa: PLC0415

        logits = _POLICY.logits(obs, sel)
        order = np.argsort(-logits)
        k = max(min_c, 1) if max_c >= 1 else min_c
        k = min(k, max_c, len(options))
        return [int(x) for x in order[:k].tolist()]

    # Untrained baseline: trust the engine's option ordering (it puts the
    # "best" candidate first), which beats random ~7-1 in mirror self-play.
    k = max(min_c, 1) if max_c >= 1 else min_c
    k = min(k, max_c, len(options))
    return list(range(k))


# Engine baselines for benchmarking; mirror of cabt.cabt.agents.
def random_agent(obs: dict[str, Any]) -> list[int]:
    sel = obs.get("select")
    if sel is None:
        return list(DECK)
    n = len(sel["option"])
    k = min(sel["maxCount"], n)
    return _RNG.sample(range(n), k) if k else []
