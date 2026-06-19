"""AlphaZero MCTS for PTCG ABC (Phase 2 minimum).

Builds on:
- Existing train/pimc.py for sample_opp_state + cg.api integration patterns
- train/value_calibration.json for value head bias correction
- Existing MlpPolicy (policy + value head, same model)

Design (= docs/ALPHAZERO_DESIGN.md):
- Uses PPO_v40 s500 (= Mode B) as policy_net for PUCT priors
- Uses same model's value head for leaf evaluation (calibrated)
- Shallow MCTS (n_sims=20-50) within Kaggle 5s budget

Key difference vs PIMC v1-v6:
- PIMC: 1-ply lookahead per option, rollout to terminal
- AlphaZero: deep search via PUCT, learnt value at leaf (no rollout)

Phase 2 = minimum implementation, only for MAIN single-choice selects.
Phase 3 = main.py integration with POKEAI_ALPHAZERO=1 env var.
"""

from __future__ import annotations

import contextlib
import json
import math
import os

import numpy as np

# Calibration loaded lazily (file may not exist in fresh checkouts)
_CALIBRATION: dict | None = None


def _load_calibration() -> dict:
    global _CALIBRATION
    if _CALIBRATION is not None:
        return _CALIBRATION
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "value_calibration.json")
    if not os.path.exists(path):
        _CALIBRATION = {"bias_mean": 0.0}  # no calibration
        return _CALIBRATION
    with open(path) as f:
        _CALIBRATION = json.load(f)
    return _CALIBRATION


class MCTSNode:
    """One node in the MCTS tree."""

    __slots__ = ("P", "N", "W", "Q", "children", "expanded")

    def __init__(self, prior_p: float) -> None:
        self.P = prior_p
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children: dict[int, MCTSNode] = {}
        self.expanded = False

    def select_child(self, c_puct: float = 1.4) -> tuple[int, MCTSNode]:
        """Pick child maximizing PUCT score: Q + c * P * sqrt(parent.N) / (1+N)."""
        sqrt_N = math.sqrt(self.N) if self.N > 0 else 1.0
        return max(
            self.children.items(),
            key=lambda kv: kv[1].Q + c_puct * kv[1].P * sqrt_N / (1 + kv[1].N),
        )

    def expand(self, priors: np.ndarray) -> None:
        """Create child for each action with given prior."""
        if self.expanded:
            return
        for i in range(len(priors)):
            self.children[i] = MCTSNode(prior_p=float(priors[i]))
        self.expanded = True

    def backup(self, value: float) -> None:
        """Update visit count + value."""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N


def _calibrated_value(policy, obs: dict) -> float:
    """V_calibrated(s) = V(s) - bias_mean. Clamped to [-1, +1]."""
    raw = policy.value(obs)
    bias = _load_calibration().get("bias_mean", 0.0)
    return float(np.clip(raw - bias, -1.0, 1.0))


def alphazero_choose(
    obs: dict,
    deck: list[int],
    policy,
    rng: np.random.Generator,
    n_sims: int = 20,
    c_puct: float = 1.4,
    max_depth: int = 20,
) -> list[int] | None:
    """MCTS-based action selection for MAIN single-choice obs.

    Returns option index as [int], or None if fallback should be used.
    """
    sel = obs.get("select")
    if sel is None:
        return None
    if sel.get("type") != 0 or int(sel.get("maxCount") or 0) != 1:
        return None
    opts = sel.get("option") or []
    if len(opts) <= 1:
        return [0] if opts else []

    # Import cg.api lazily — same pattern as train/pimc.py
    try:
        from cg.api import (  # noqa: PLC0415
            search_begin,
            search_release,
            search_step,
            to_observation_class,
        )
    except Exception:
        return None

    if obs.get("search_begin_input") is None:
        return None

    try:
        agent_obs = to_observation_class(obs)
    except Exception:
        return None

    # Build root + expand with policy priors
    priors = policy.probs(obs, sel)
    root = MCTSNode(prior_p=1.0)
    root.expand(priors)

    n_opts = len(opts)  # noqa: F841
    successful_sims = 0

    for _sim in range(n_sims):
        # 1. Sample opponent world (inline, same as scripts/pimc_smoke_test.py)
        try:
            ydp, ypp, op_d, op_p, op_h, op_a = _sample_world(obs, deck, rng)
        except Exception:
            continue

        # 2. search_begin to get root SearchState
        try:
            root_state = search_begin(agent_obs, ydp, ypp, op_d, op_p, op_h, op_a)
        except Exception:
            continue

        sid = root_state.searchId
        path: list[MCTSNode] = [root]

        # 3. Selection: PUCT down to a leaf
        try:
            current_node = root
            current_state = root_state
            depth = 0

            # Pick root action via PUCT
            action_idx, child_node = current_node.select_child(c_puct)
            path.append(child_node)

            # Apply that action
            current_state = search_step(sid, [action_idx])
            depth = 1

            # If terminal, use terminal reward as value
            cur = current_state.observation.current
            if cur is not None and cur.result != -1:
                # outcome from our perspective
                your_idx = cur.yourIndex
                value = 0.0 if cur.result == 2 else (1.0 if cur.result == your_idx else -1.0)
            else:
                # Continue down the tree, expanding leaves on the fly
                while depth < max_depth:
                    next_sel = current_state.observation.select
                    if next_sel is None:
                        break

                    # Need an obs-like dict for policy — reconstruct minimally
                    obs_dict = _state_to_obs_dict(current_state)
                    if (
                        obs_dict["select"].get("type") != 0
                        or int(obs_dict["select"].get("maxCount") or 0) != 1
                    ):
                        # Non-MAIN selects: pick by policy (sampling), no MCTS branching
                        try:
                            n_next_opts = len(obs_dict["select"]["option"])
                            if n_next_opts == 0:
                                break
                            min_c = int(obs_dict["select"].get("minCount") or 0)
                            max_c = int(obs_dict["select"].get("maxCount") or 0)
                            if max_c == 0:
                                break
                            if obs_dict["select"].get("type") == 0 and max_c == 1:
                                pp = policy.probs(obs_dict, obs_dict["select"])
                                action = [int(rng.choice(n_next_opts, p=pp))]
                            else:
                                # Multi-pick fallback: top-k by logits
                                z = policy.logits(obs_dict, obs_dict["select"])
                                order = np.argsort(-z)
                                k = max(min_c, 1)
                                k = min(k, max_c, n_next_opts)
                                action = [int(x) for x in order[:k].tolist()]
                            current_state = search_step(sid, action)
                        except Exception:
                            break
                        depth += 1
                        cur = current_state.observation.current
                        if cur is not None and cur.result != -1:
                            your_idx = cur.yourIndex
                            value = (
                                0.0
                                if cur.result == 2
                                else (1.0 if cur.result == your_idx else -1.0)
                            )
                            break
                        continue

                    # MAIN single-choice: branch by PUCT
                    if not child_node.expanded:
                        try:
                            next_priors = policy.probs(obs_dict, obs_dict["select"])
                            child_node.expand(next_priors)
                        except Exception:
                            break
                    # Pick child
                    try:
                        next_action_idx, next_child = child_node.select_child(c_puct)
                    except Exception:
                        break
                    try:
                        current_state = search_step(sid, [next_action_idx])
                    except Exception:
                        break
                    path.append(next_child)
                    child_node = next_child
                    depth += 1

                    cur = current_state.observation.current
                    if cur is not None and cur.result != -1:
                        your_idx = cur.yourIndex
                        value = (
                            0.0 if cur.result == 2 else (1.0 if cur.result == your_idx else -1.0)
                        )
                        break
                else:
                    # Hit max_depth: use value head
                    obs_dict = _state_to_obs_dict(current_state)
                    try:
                        value = _calibrated_value(policy, obs_dict)
                    except Exception:
                        value = 0.0

                # If we broke out without value set, use value head
                if "value" not in locals():
                    obs_dict = _state_to_obs_dict(current_state)
                    try:
                        value = _calibrated_value(policy, obs_dict)
                    except Exception:
                        value = 0.0

        except Exception:
            with contextlib.suppress(Exception):
                search_release(sid)
            continue
        finally:
            with contextlib.suppress(Exception):
                search_release(sid)

        # 4. Backup
        for node in path:
            node.backup(value if "value" in dir() else 0.0)
        successful_sims += 1
        # Reset value for next sim
        if "value" in locals():
            del value

    if successful_sims == 0:
        return None

    # Pick most-visited root child
    return [max(root.children.items(), key=lambda kv: kv[1].N)[0]]


def _state_to_obs_dict(search_state) -> dict:
    """Convert SearchState back to dict-like obs for policy call.

    Policy expects the dict shape kaggle_environments delivers. Recurses
    through dataclasses to rebuild it minimally.
    """
    return {
        "select": _dc_to_dict(search_state.observation.select),
        "current": _dc_to_dict(search_state.observation.current),
        "logs": [],
        "search_begin_input": search_state.observation.search_begin_input or "",
    }


def _dc_to_dict(obj):
    if obj is None:
        return None
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dc_to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, list):
        return [_dc_to_dict(x) for x in obj]
    return obj


def _sample_world(
    obs: dict, our_deck: list[int], rng: np.random.Generator
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    """Sample (your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active).

    Naive: assumes opponent's deck = our deck. Excludes visible opp cards.
    Same pattern as scripts/pimc_smoke_test.py.
    """
    state = obs["current"]
    your_idx = state["yourIndex"]
    your = state["players"][your_idx]
    opp = state["players"][1 - your_idx]

    # Our prize
    your_prize_pred: list[int] = []
    for prize_card in your.get("prize", []) or []:
        if prize_card is not None and isinstance(prize_card, dict) and "id" in prize_card:
            your_prize_pred.append(prize_card["id"])
        else:
            your_prize_pred.append(our_deck[0] if our_deck else 0)

    # Opp visible
    visible: list[int] = []
    for area_name in ["discard", "bench", "active"]:
        for card in opp.get(area_name, []) or []:
            if card is not None and isinstance(card, dict) and "id" in card:
                visible.append(card["id"])

    remaining = list(our_deck)
    for v in visible:
        if v in remaining:
            remaining.remove(v)
    rng.shuffle(remaining)

    hand_count = int(opp.get("handCount", 0))
    opp_hand = remaining[:hand_count]
    remaining = remaining[hand_count:]

    prize_count = len(opp.get("prize", []) or [])
    opp_prize = remaining[:prize_count]
    remaining = remaining[prize_count:]

    return list(our_deck), your_prize_pred, remaining, opp_prize, opp_hand, []
