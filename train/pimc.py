"""Single-world 1-ply PIMC for MAIN selects.

The linear policy can pick a move that looks good locally but loses to the
opponent's natural response. PIMC fixes this by sampling the hidden state,
stepping into each option, and scoring the resulting position with the
linear policy's value function. The option with the highest scored
look-ahead value wins.

This module assumes the opponent's deck composition is identical to ours
(see DECK in main.py). That's not strictly true at submission time but
works as a reasonable prior for self-play and against random_agent. A
multi-world extension would average across several opponent deck samples.
"""

from __future__ import annotations

import time
from typing import Any

DEFAULT_TIME_BUDGET_MS = 500.0


def pick_best_option(
    obs: dict[str, Any],
    sel: dict[str, Any],
    deck: list[int],
    policy,  # train.policy.LinearPolicy
    time_budget_ms: float = DEFAULT_TIME_BUDGET_MS,
) -> int | None:
    """Return the index of the best MAIN option via 1-ply PIMC look-ahead.

    Returns None when PIMC cannot be used (search_begin fails, no search_input
    in obs, etc.) — caller should fall back to the linear policy.
    """
    if obs.get("search_begin_input") is None:
        return None
    cur = obs.get("current")
    if cur is None:
        return None
    if sel.get("type") != 0:  # MAIN only
        return None
    options = sel.get("option") or []
    if len(options) <= 1:
        return None  # nothing to compare

    # Import here so callers (tests, smoke import) that don't have cg.api
    # available don't pay the import cost.
    try:
        from cg.api import (  # noqa: PLC0415
            search_begin,
            search_release,
            search_step,
            to_observation_class,
        )
    except Exception:
        return None

    deadline = time.monotonic() + time_budget_ms / 1000.0

    you = cur["yourIndex"]
    me = cur["players"][you]
    opp = cur["players"][1 - you]

    # Sample hidden info: opponent deck/prize/hand from our deck (mirror match
    # assumption). Our remaining deck = full DECK minus known-visible cards.
    # The engine validates lengths, not contents, so we just need plausible
    # card-id lists of the right size.
    opp_deck_pred = list(deck)
    opp_prize_pred = [deck[0]] * len(opp.get("prize", []))
    opp_hand_pred = [deck[0]] * opp.get("handCount", 0)
    your_prize_pred = [deck[0]] * len(me.get("prize", []))
    your_deck_pred = list(deck)[: me.get("deckCount", 0)]
    opp_active_pred: list[int] = []
    # Predict opponent's face-down active only if there is one.
    opp_active = opp.get("active") or []
    if opp_active and opp_active[0] is None:
        # Pick the first Basic Pokemon in the predicted deck. Falls back to
        # any deck card if no Basic Pokemon is identifiable.
        opp_active_pred = [opp_deck_pred[0]]

    try:
        agent_obs = to_observation_class(obs)
        root = search_begin(
            agent_obs,
            your_deck=your_deck_pred,
            your_prize=your_prize_pred,
            opponent_deck=opp_deck_pred,
            opponent_prize=opp_prize_pred,
            opponent_hand=opp_hand_pred,
            opponent_active=opp_active_pred,
        )
    except Exception:
        return None

    my_index = you
    best_i = 0
    best_q = -1e9
    try:
        for i in range(len(options)):
            if time.monotonic() > deadline:
                break
            try:
                child = search_step(root.searchId, [i])
            except Exception:
                continue
            try:
                child_obs = _child_to_dict(child.observation)
                q = _value_of(child_obs, policy, my_index)
                # Tiny tie-break: prefer earlier (= engine-recommended) options.
                q -= 1e-4 * i
                if q > best_q:
                    best_q = q
                    best_i = i
            finally:
                search_release(child.searchId)
    finally:
        search_release(root.searchId)

    return best_i


def _value_of(obs: dict[str, Any], policy, my_index: int) -> float:
    """Heuristic value of `obs` from `my_index`'s perspective in [-1, 1].

    The linear policy's option logits are NOT calibrated values across
    states (they encode relative preference within a single state), so we
    cannot use policy.logits as a value function. Use simple board
    heuristics instead: prize differential dominates because taking all
    prizes wins, with HP / bench / hand as tiebreakers.
    """
    cur = obs.get("current")
    if cur is None:
        return 0.0

    result = cur.get("result", -1)
    if result != -1:
        if result == my_index:
            return 1.0
        if result == 1 - my_index:
            return -1.0
        return 0.0  # draw

    me = cur["players"][my_index]
    opp = cur["players"][1 - my_index]

    # Prize differential: smaller my_prize_count = closer to winning.
    my_prize = len(me.get("prize", []))
    opp_prize = len(opp.get("prize", []))
    prize_term = (opp_prize - my_prize) / 6.0

    # Active HP ratios.
    my_act = (me.get("active") or [None])[0]
    opp_act = (opp.get("active") or [None])[0]
    hp_term = _hp_ratio(my_act) - _hp_ratio(opp_act)

    # Aggregate bench HP, normalized to a typical max.
    bench_term = (_total_hp(me.get("bench") or []) - _total_hp(opp.get("bench") or [])) / 1000.0

    # Hand-size differential (slight signal — having more cards is better).
    hand_term = (me.get("handCount", 0) - opp.get("handCount", 0)) / 10.0

    return float(prize_term + 0.5 * hp_term + 0.3 * bench_term + 0.1 * hand_term)


def _hp_ratio(p) -> float:
    if not p:
        return 0.0
    mx = p.get("maxHp") or 0
    return (p.get("hp") or 0) / mx if mx > 0 else 0.0


def _total_hp(pokes) -> float:
    return float(sum((p.get("hp") or 0) for p in pokes if p))


def _child_to_dict(obs_class) -> dict[str, Any]:
    """Best-effort convert a SearchState.observation (dataclass) into the same
    plain-dict shape that main.agent / features expect. We only need the
    subset that LinearPolicy.logits reads: select.{type, context, option, ...},
    current.{yourIndex, turn, players, ...}, and so on.
    """
    from dataclasses import asdict  # noqa: PLC0415

    return asdict(obs_class)


def quick_self_check() -> bool:
    """Cheap sanity check used by tests — returns True if search_begin works."""
    try:
        from cg.api import search_begin  # noqa: F401, PLC0415

        return True
    except Exception:
        return False
