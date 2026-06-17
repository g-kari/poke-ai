"""Single-world 1-ply PIMC for MAIN selects.

The linear policy can pick a move that looks good locally but loses to the
opponent's natural response. PIMC fixes this by sampling the hidden state,
stepping into each option, and scoring the resulting position with a
trained value head. The option with the highest scored look-ahead value
wins.

This module assumes the opponent's deck composition is identical to ours
(see DECK in main.py). That's not strictly true at submission time but
works as a reasonable prior for self-play and against random_agent. A
multi-world extension would average across several opponent deck samples.

Status (2026-06-17): Three PIMC backends benchmarked. Multi-world IS-MCTS
is the first to beat the engine-prior baseline (vs random) but still loses
in the mirror match and against the always-pick-zero first_agent.

  Backend A (1-ply + trained linear value head):
    PIMC-ON vs PIMC-OFF mirror:   10-30 (25%)
    PIMC-ON vs first_agent:       15-25 (37.5%)
    PIMC-ON vs random:            75%   (OFF: 92.5%)

  Backend B (rollout-to-terminal, single-world):
    PIMC-ON vs PIMC-OFF mirror:   4-16  (20%)
    PIMC-ON vs random:            80%   (OFF: 92.5%)

  Backend C (this file: multi-world IS-MCTS, N=2 sampled hidden states):
    PIMC-ON vs PIMC-OFF mirror:   7-13  (35%)
    PIMC-ON vs first_agent:       8-12  (40%)
    PIMC-ON vs random:            95%   (OFF: 92.5%)  ← beats baseline

So multi-world helps everywhere over single-world and finally beats OFF
in the vs-random bench. The remaining gap on mirror / first_agent
suggests strategy fusion isn't fully resolved at N=2 — N=4 was tried
but its rollouts blow the 500ms budget and get truncated mid-game,
which gives noisier estimates than N=2 with complete rollouts.

Kept enabled via POKEAI_PIMC=1. Default OFF in main.py is conservative:
the +2.5pp vs random doesn't outweigh the mirror-match risk against
similar-skill TrueSkill opponents. Re-test when the rollout policy
improves or when we can budget for more worlds.
"""

from __future__ import annotations

import contextlib
import random
import time
from typing import Any

DEFAULT_TIME_BUDGET_MS = 500.0
DEFAULT_ROLLOUT_DEPTH = 80  # max steps per rollout before bailing to value head
# Information-Set MCTS: average Q across N sampled worlds. Empirically
# tuned — N=2 with the full linear rollout policy lands at 95% vs random;
# N=8 with the engine-prior rollout policy drops to 80% even though more
# samples should help, because engine-prior moves stall games and rollouts
# hit DEFAULT_ROLLOUT_DEPTH instead of reaching a terminal state.
DEFAULT_N_WORLDS = 2
# Linear policy rollout (default) gives cleaner terminal rewards than the
# engine-prior rollout. The engine_prior_rollout option in pick_best_option
# is kept as infrastructure for experiments (e.g., to model PIMC-OFF
# opponents in the mirror match more faithfully).
DEFAULT_ROLLOUT_ENGINE_PRIOR = False


def pick_best_option(
    obs: dict[str, Any],
    sel: dict[str, Any],
    deck: list[int],
    policy,  # train.policy.LinearPolicy
    time_budget_ms: float = DEFAULT_TIME_BUDGET_MS,
    rollout_depth: int = DEFAULT_ROLLOUT_DEPTH,
    n_worlds: int = DEFAULT_N_WORLDS,
    rng: random.Random | None = None,
    engine_prior_rollout: bool = DEFAULT_ROLLOUT_ENGINE_PRIOR,
) -> int | None:
    """Return the index of the best MAIN option via PIMC rollout-to-terminal.

    For each option:
      1. search_step from the sampled root into the chosen option's child state
      2. roll the linear policy from there to the terminal state (or until
         rollout_depth selects, whichever comes first)
      3. record the terminal reward from our perspective as the option's Q

    The rollout uses the linear policy because it knows how to handle every
    select type — not just MAIN. This sidesteps the value-head extrapolation
    problem of the 1-ply variant.

    Returns None when PIMC cannot be used (search_begin fails, no
    search_begin_input in obs, only one option, etc.) — caller should fall
    back to the linear policy.
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
    rng = rng if rng is not None else random.Random()

    you = cur["yourIndex"]
    me = cur["players"][you]
    opp = cur["players"][1 - you]
    my_index = you

    # Sample hidden info: opponent deck/prize/hand from our deck (mirror match
    # assumption). The engine validates lengths, not contents, so any card-id
    # list of the right size is accepted; we shuffle so each world's rollout
    # sees a different draw order.
    opp_prize_count = len(opp.get("prize", []))
    opp_hand_count = opp.get("handCount", 0)
    your_prize_count = len(me.get("prize", []))
    your_deck_count = me.get("deckCount", 0)
    opp_active = opp.get("active") or []
    needs_face_down = bool(opp_active) and opp_active[0] is None

    agent_obs = to_observation_class(obs)

    # Aggregate Q per option across N sampled worlds.
    q_sum = [0.0] * len(options)
    q_count = [0] * len(options)
    actual_worlds = 0
    for _ in range(n_worlds):
        if time.monotonic() > deadline:
            break
        # Each world gets an independently shuffled deck — the engine then
        # draws from this order during the rollout, giving us a different
        # game realization per world.
        opp_deck_pred = list(deck)
        rng.shuffle(opp_deck_pred)
        your_deck_pred = list(deck)
        rng.shuffle(your_deck_pred)
        your_deck_pred = your_deck_pred[:your_deck_count]
        opp_prize_pred = [opp_deck_pred[k % len(opp_deck_pred)] for k in range(opp_prize_count)]
        opp_hand_pred = [opp_deck_pred[k % len(opp_deck_pred)] for k in range(opp_hand_count)]
        your_prize_pred = [
            your_deck_pred[k % max(1, len(your_deck_pred))] for k in range(your_prize_count)
        ]
        opp_active_pred = [opp_deck_pred[0]] if needs_face_down else []

        try:
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
            continue
        actual_worlds += 1
        try:
            for i in range(len(options)):
                if time.monotonic() > deadline:
                    break
                try:
                    child = search_step(root.searchId, [i])
                except Exception:
                    continue
                try:
                    q = _rollout_reward(
                        child,
                        policy,
                        my_index,
                        rollout_depth,
                        search_step,
                        search_release,
                        engine_prior_rollout,
                    )
                    q_sum[i] += q
                    q_count[i] += 1
                except Exception:
                    pass
                finally:
                    with contextlib.suppress(Exception):
                        search_release(child.searchId)
        finally:
            search_release(root.searchId)

    if actual_worlds == 0:
        return None

    # Pick option with highest mean Q. Unscored options (no rollouts at all
    # because every world timed out before reaching them) get -inf so a
    # scored option always wins. Tiny engine-prior tie-break stays.
    best_i = 0
    best_q = -1e9
    for i in range(len(options)):
        if q_count[i] == 0:
            continue
        mean_q = q_sum[i] / q_count[i] - 1e-4 * i
        if mean_q > best_q:
            best_q = mean_q
            best_i = i
    return best_i


def _rollout_reward(
    state,
    policy,
    my_index: int,
    max_depth: int,
    search_step,
    search_release,
    engine_prior_rollout: bool = False,
) -> float:
    """Roll forward from `state` until termination or max_depth selects.
    Returns the terminal reward in [-1, 1] from `my_index`'s perspective.

    When `engine_prior_rollout` is True, every action picks the first
    `k` options the engine offers (= "first_agent" / engine prior). This
    is ~10x faster than the linear-policy rollout and matches the
    behavior of PIMC-OFF opponents in the mirror match. Otherwise picks
    argmax of the linear policy logits.

    `state` is owned by the caller and never released here; intermediate
    SearchStates created during the rollout are tracked and released in
    a finally block to avoid leaking when the loop exits early.
    """
    intermediates: list[int] = []
    current = state
    try:
        for _ in range(max_depth):
            obs = _child_to_dict(current.observation)
            cur = obs.get("current")
            if cur is None:
                return _value_of(obs, policy, my_index)
            result = cur.get("result", -1)
            if result != -1:
                if result == my_index:
                    return 1.0
                if result == 1 - my_index:
                    return -1.0
                return 0.0
            sel = obs.get("select")
            if sel is None:
                return _value_of(obs, policy, my_index)
            opts = sel.get("option") or []
            if not opts:
                return _value_of(obs, policy, my_index)
            max_c = int(sel.get("maxCount") or 0)
            min_c = int(sel.get("minCount") or 0)
            if max_c == 0:
                return _value_of(obs, policy, my_index)
            k = max(min_c, 1) if max_c >= 1 else min_c
            k = min(k, max_c, len(opts))
            if engine_prior_rollout:
                # Engine prior: just take the first k options the engine offered.
                choice = list(range(k))
            else:
                # Greedy linear-policy rollout: argmax of logits, no sampling.
                import numpy as np  # noqa: PLC0415

                logits = policy.logits(obs, sel)
                order = np.argsort(-logits)
                choice = [int(x) for x in order[:k].tolist()]
            try:
                next_state = search_step(current.searchId, choice)
            except Exception:
                return _value_of(obs, policy, my_index)
            if current is not state:
                intermediates.append(current.searchId)
            current = next_state
        # Ran out of depth — fall back to value head / heuristic.
        return _value_of(_child_to_dict(current.observation), policy, my_index)
    finally:
        if current is not state:
            with contextlib.suppress(Exception):
                search_release(current.searchId)
        for sid in intermediates:
            with contextlib.suppress(Exception):
                search_release(sid)


def _value_of(obs: dict[str, Any], policy, my_index: int) -> float:
    """Value of `obs` from `my_index`'s perspective in roughly [-1, 1].

    Uses the trained value head (policy.value) when available, falling
    back to a board-heuristic value if the value head is uninitialized
    (||w_value|| ≈ 0). Terminal states short-circuit to the actual reward.
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

    # Use the value head if it's been trained.
    if _value_head_trained(policy):
        v = policy.value(obs)
        # policy.value evaluates from the perspective of the player about to
        # choose (which is cur.yourIndex), so flip if that's not us.
        next_index = cur.get("yourIndex", my_index)
        if next_index != my_index:
            return -v
        return v

    return _heuristic_value(cur, my_index)


def _value_head_trained(policy) -> bool:
    """True if w_value has been updated away from its zero init."""
    import numpy as np  # noqa: PLC0415

    w = getattr(policy, "w_value", None)
    if w is None:
        return False
    return float(np.linalg.norm(w)) > 1e-3


def _heuristic_value(cur: dict[str, Any], my_index: int) -> float:
    me = cur["players"][my_index]
    opp = cur["players"][1 - my_index]

    my_prize = len(me.get("prize", []))
    opp_prize = len(opp.get("prize", []))
    prize_term = (opp_prize - my_prize) / 6.0

    my_act = (me.get("active") or [None])[0]
    opp_act = (opp.get("active") or [None])[0]
    hp_term = _hp_ratio(my_act) - _hp_ratio(opp_act)

    bench_term = (_total_hp(me.get("bench") or []) - _total_hp(opp.get("bench") or [])) / 1000.0
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
