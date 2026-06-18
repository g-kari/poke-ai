"""Minimal 1-ply PIMC agent.

For each option at the root decision, sample the opponent's hidden info
(hand + active if face-down), call search_begin, advance one search_step,
then score the resulting position by prize differential. Return argmax.

Trade-offs:
- 1-ply only (no deep tree), and prize-delta heuristic value function
  (no learned value head yet). This is the simplest thing that lets PIMC
  beat the engine-order baseline; AlphaZero-style PIMC+NN is next.
- Opponent info sampling is "assume opponent plays an Iono-deck mix,
  sample hand uniformly from leftover deck pool". Wrong but stable enough
  for a first pass.
- Falls back to engine-order option 0 on any exception, so it can never
  crash on Kaggle.
"""

from __future__ import annotations

import contextlib
import random
import time
from collections.abc import Callable

from cg.api import (
    Observation,
    search_begin,
    search_release,
    search_step,
    to_observation_class,
)

# How long we allow per decision; Kaggle's hard limit is ~3s.
_TIME_BUDGET_S = 1.5
# Cap number of options we score (root branching can blow up).
_MAX_OPTIONS_TO_SCORE = 8
# v5: how many additional search_step iterations to push past the 1-ply
# evaluation. Each step is ~0.5ms so 8 steps is well under our budget.
_ROLLOUT_DEPTH = 8


def _sample_opp_hand(opp_deck: list[int], hand_count: int, rng: random.Random) -> list[int]:
    """Best-effort sample of opponent's hand from their deck.

    Uniform random pick; in real PIMC we'd weight by what we've seen
    them play, but uniform is a fine starting point.
    """
    if hand_count <= 0:
        return []
    pool = list(opp_deck)
    rng.shuffle(pool)
    return pool[:hand_count]


def _pick_opp_active(opp_deck: list[int], obs_class: Observation, your_index: int) -> list[int]:
    """If the opponent has a face-down active, predict one Pokémon card from
    their deck. Returns [] when no prediction is needed."""
    opp_active = obs_class.current.players[1 - your_index].active
    if len(opp_active) > 0 and opp_active[0] is None:
        # Pick the first reasonable Basic Pokémon ID we can find. We have
        # no card metadata loaded here, so just return the first deck card;
        # the engine will complain if it's not a Pokémon.
        return [opp_deck[0]]
    return []


def _prize_delta(obs_class: Observation, your_index: int) -> float:
    """v2 board-state value heuristic. Higher = better for us.

    Components (weights chosen so prize delta is dominant but field state
    matters at the margin):
      - prize delta (×100): we want opp to have fewer remaining prizes
      - active HP ratio: we want our active healthy, theirs hurt
      - bench fill: setup advantage = more bench Pokemon ready
      - energy on our active: ability to attack next turn
    """
    cur = obs_class.current
    me = cur.players[your_index]
    opp = cur.players[1 - your_index]
    my_taken = 6 - len(me.prize or [])
    opp_taken = 6 - len(opp.prize or [])
    score = (opp_taken - my_taken) * 100.0

    # Active HP ratio (max HP comes from card data when available;
    # fallback: assume 200 if missing).
    def _hp_ratio(pkmn_list) -> float:
        if not pkmn_list:
            return 0.0
        p = pkmn_list[0]
        if p is None:
            return 0.0
        hp = getattr(p, "hp", None) or 0
        max_hp = getattr(p, "maxHp", None) or 200
        if max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, hp / max_hp))

    score += _hp_ratio(me.active or []) * 10.0
    score -= _hp_ratio(opp.active or []) * 10.0

    # Bench fill (more = better setup options).
    score += len([p for p in (me.bench or []) if p]) * 2.0
    score -= len([p for p in (opp.bench or []) if p]) * 2.0

    # Energy on our active (= ability to attack soon).
    def _energy_count(pkmn_list) -> int:
        if not pkmn_list:
            return 0
        p = pkmn_list[0]
        if p is None:
            return 0
        eng = getattr(p, "energy", None) or []
        return len(eng)

    score += _energy_count(me.active or []) * 1.5
    score -= _energy_count(opp.active or []) * 1.5
    return float(score)


def _try_score_option(
    obs_dict: dict,
    obs_class: Observation,
    option_idx: int,
    your_deck: list[int],
    opp_deck: list[int],
    rng: random.Random,
    nn_value_fn: Callable[[dict, int], float] | None = None,
    nn_weight: float = 30.0,
) -> float | None:
    """Run search_begin + search_step([option_idx]) and score the result.
    Returns None on any failure; caller treats None as 'cannot evaluate'.

    If nn_value_fn is provided (AlphaZero-style PIMC v3), the heuristic
    score is augmented with the learned value estimate of the resulting
    position.
    """
    cur = obs_class.current
    your_index = cur.yourIndex
    opp_player = cur.players[1 - your_index]
    opp_hand_count = opp_player.handCount or 0
    opp_prize_remaining = len(opp_player.prize) if opp_player.prize else 6

    my_prize_remaining = len(cur.players[your_index].prize) if cur.players[your_index].prize else 6

    try:
        ss = search_begin(
            agent_observation=obs_class,
            your_deck=your_deck,
            your_prize=[your_deck[0]] * my_prize_remaining,
            opponent_deck=opp_deck,
            opponent_prize=[opp_deck[0]] * opp_prize_remaining,
            opponent_hand=_sample_opp_hand(opp_deck, opp_hand_count, rng),
            opponent_active=_pick_opp_active(opp_deck, obs_class, your_index),
            manual_coin=False,
        )
        sid = ss.searchId
    except Exception:  # noqa: BLE001
        return None

    try:
        next_state = search_step(sid, [option_idx])
        next_obs = next_state.observation
        next_obs_class = to_observation_class(next_obs) if isinstance(next_obs, dict) else next_obs
        if next_obs_class.current is None:
            return 0.0
        # v5: rollout further (greedy: always pick option 0 = engine-prior).
        # Doing this for a few more steps lets prize-delta accumulate signal.
        # Each step is ~0.5ms so we can afford 5-10 steps cheap.
        for _ in range(_ROLLOUT_DEPTH):
            cur = next_obs_class.current
            if cur is None or (cur.result is not None and cur.result >= 0):
                break
            try:
                ns = search_step(sid, [0])  # greedy engine-order
                next_obs = ns.observation
                next_obs_class = (
                    to_observation_class(next_obs) if isinstance(next_obs, dict) else next_obs
                )
            except Exception:  # noqa: BLE001
                break
        score = _prize_delta(next_obs_class, your_index) if next_obs_class.current else 0.0
        if nn_value_fn is not None and isinstance(next_obs, dict):
            with contextlib.suppress(Exception):
                v = nn_value_fn(next_obs, your_index)
                score += v * nn_weight
        return score
    except Exception:  # noqa: BLE001
        return None
    finally:
        with contextlib.suppress(Exception):
            search_release(sid)


def _seen_opp_card_ids(obs_class: Observation, your_index: int) -> set[int]:
    """Collect card IDs we've seen on opponent's side (active, bench, discard).

    These are public knowledge — engine reveals them in obs. We use the
    multiset for deck-fingerprinting.
    """
    seen: set[int] = set()
    opp = obs_class.current.players[1 - your_index]
    for area in (opp.active, opp.bench, opp.discard):
        for p in area or []:
            if p is None:
                continue
            cid = getattr(p, "id", None)
            if cid is None and isinstance(p, dict):
                cid = p.get("id")
            if cid:
                seen.add(int(cid))
    return seen


def infer_opp_deck(
    obs_class: Observation,
    your_index: int,
    candidate_decks: dict[str, list[int]],
    default_deck: list[int],
) -> tuple[list[int], str]:
    """Score each candidate deck by overlap with seen opponent cards.

    Returns (best_deck, best_label). Falls back to default_deck if no card
    has been seen yet (very early game)."""
    seen = _seen_opp_card_ids(obs_class, your_index)
    if not seen:
        return default_deck, "default"
    best_name = "default"
    best_deck = default_deck
    best_score = -1
    for name, deck in candidate_decks.items():
        deck_set = set(deck)
        overlap = len(seen & deck_set)
        if overlap > best_score:
            best_score = overlap
            best_deck = deck
            best_name = name
    return best_deck, best_name


def make_v60_value_fn(policy_path: str = "train/mlp_policy_v60_pool5_ext.pt"):
    """Build a callable nn_value_fn(obs_dict, your_index) -> float in [-1, 1]
    backed by the V60 EXT policy's tanh-bounded value head."""
    from train.mlp_policy_v60 import MlpPolicyV60  # local import — torch heavy

    policy = MlpPolicyV60.load(policy_path)
    policy.eval()

    def _v(obs_dict: dict, _your_index: int) -> float:
        # policy.value() already applies tanh internally.
        with contextlib.suppress(Exception):
            return float(policy.value(obs_dict))
        return 0.0

    return _v


def make_pimc_agent(
    deck: list[int],
    opp_deck_assumption: list[int],
    seed: int = 0,
    time_budget_s: float = _TIME_BUDGET_S,
    nn_value_fn: Callable[[dict, int], float] | None = None,
    nn_weight: float = 30.0,
    candidate_opp_decks: dict[str, list[int]] | None = None,
):
    """Build an agent(obs) function that does 1-ply PIMC scoring.

    deck: our 60-card deck (returned on initial deck-submission step).
    opp_deck_assumption: default guess for opponent (used when nothing seen yet).
    nn_value_fn: optional V(s) function (= AlphaZero-style PIMC v3).
    nn_weight: weight on NN value when blending with prize-delta.
    candidate_opp_decks: optional {label: deck} pool. If given, each turn
        the inferred deck (highest overlap with seen opp cards) replaces
        opp_deck_assumption for that decision (= PIMC v4 dynamic inference).
    """
    rng = random.Random(seed)

    def agent(obs: dict) -> list[int]:
        sel = obs.get("select")
        if sel is None:
            return list(deck)
        opts = sel.get("option") or []
        if not opts:
            return []
        max_c = int(sel.get("maxCount") or 0)
        min_c = int(sel.get("minCount") or 0)

        if max_c == 0:
            return []

        # Single-choice MAIN-context decisions are where PIMC pays off;
        # multi-pick selections (energy assign, etc.) fall through to
        # engine-order for now.
        if max_c != 1 or len(opts) <= 1:
            k = max(min_c, 1)
            k = min(k, max_c, len(opts))
            return list(range(k))

        try:
            obs_class = to_observation_class(obs)
        except Exception:  # noqa: BLE001
            return [0]

        # PIMC v4: infer opp deck from observed cards.
        if candidate_opp_decks:
            try:
                your_index = obs_class.current.yourIndex
                opp_deck_for_search, _ = infer_opp_deck(
                    obs_class, your_index, candidate_opp_decks, opp_deck_assumption
                )
            except Exception:  # noqa: BLE001
                opp_deck_for_search = opp_deck_assumption
        else:
            opp_deck_for_search = opp_deck_assumption

        # Score the first N options under a time budget.
        n_to_score = min(_MAX_OPTIONS_TO_SCORE, len(opts))
        t0 = time.monotonic()
        scores: list[tuple[int, float]] = []
        for i in range(n_to_score):
            if time.monotonic() - t0 > time_budget_s:
                break
            s = _try_score_option(
                obs,
                obs_class,
                i,
                deck,
                opp_deck_for_search,
                rng,
                nn_value_fn=nn_value_fn,
                nn_weight=nn_weight,
            )
            if s is not None:
                scores.append((i, s))

        if not scores:
            return [0]  # fallback: engine order

        # argmax. Stable: prefer earlier options on ties (engine-order prior).
        best_i, _ = max(scores, key=lambda kv: (kv[1], -kv[0]))
        return [best_i]

    return agent
