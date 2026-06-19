"""PIMC Phase 1: smoke test for cg.api.search_begin/step/release.

Verifies the basic API works on a single observation captured from a
self-play episode. Does NOT implement full PIMC yet — just confirms the
API can be called without errors and returns sensible state.

Usage:
    scripts/run.sh python3 scripts/pimc_smoke_test.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

sys.modules.setdefault("litellm", type(sys)("litellm"))
import numpy as np  # noqa: E402
from kaggle_environments import make  # noqa: E402

from cg.api import (  # noqa: E402
    search_begin,
    search_release,
    to_observation_class,
)
from train.mlp_policy import MlpPolicy  # noqa: E402

with open("deck.csv") as _f:
    DECK = [int(line.strip()) for line in _f if line.strip()]

policy = MlpPolicy.load("train/mlp_policy_ppo_v40_s100_t500.pt")  # Mode B rollout
rng = np.random.default_rng(42)


def policy_agent(obs):
    """Standard policy agent (no search) — used as both player and rollout."""
    sel = obs.get("select")
    if sel is None:
        return list(DECK)
    opts = sel["option"]
    if not opts:
        return []
    max_c = int(sel.get("maxCount") or 0)
    min_c = int(sel.get("minCount") or 0)
    if sel["type"] == 0 and max_c == 1:
        probs = policy.probs(obs, sel)
        return [int(rng.choice(len(opts), p=probs))]
    if max_c >= 1:
        logits = policy.logits(obs, sel)
        order = np.argsort(-logits)
        k = max(min_c, 1)
        k = min(k, max_c, len(opts))
        return [int(x) for x in order[:k].tolist()]
    return []


def sample_opp_state(obs):
    """Naive opp-state sampler (PIMC v1).

    Assumes opponent uses the same deck as us. Excludes visible cards
    (discard, bench, active, our hand prediction). Returns the args
    needed for search_begin.
    """
    state = obs["current"]
    your_idx = state["yourIndex"]
    opp = state["players"][1 - your_idx]

    # 1. Opp deck = our deck (= 60 cards naive assumption)
    opp_full = list(DECK)

    # 2. Visible opp cards (= discard + bench + active known cards)
    visible = []
    for area_name in ["discard", "bench", "active"]:
        for card in opp.get(area_name, []) or []:
            if card is not None and isinstance(card, dict) and "id" in card:
                visible.append(card["id"])

    # 3. Remove visible from full pool
    remaining = list(opp_full)
    for v in visible:
        if v in remaining:
            remaining.remove(v)

    rng.shuffle(remaining)

    # 4. Hand (handCount cards)
    hand_count = int(opp.get("handCount", 0))
    opp_hand = remaining[:hand_count]
    remaining = remaining[hand_count:]

    # 5. Prize (= len(prize) cards from remaining)
    prize_count = len(opp.get("prize", []))
    opp_prize = remaining[:prize_count]
    remaining = remaining[prize_count:]

    # 6. Remaining = opp deck
    opp_deck = remaining

    # 7. Face-down active: not handled in smoke test (= rare, mid-game obs only)
    opp_active = []

    return opp_deck, opp_hand, opp_prize, opp_active


def smoke_test():
    """Capture a single mid-game observation, then try search_begin."""
    print("=== PIMC smoke test ===")
    print("  rollout policy: PPO_v40 s500 (= Mode B)")
    print(f"  deck size: {len(DECK)}")

    # 1. Run a few steps of self-play to get a mid-game obs
    print("\n--- Capturing mid-game observation ---")
    env = make("cabt")
    captured = [None]
    step_count = [0]

    def capturing_agent(obs):
        sel = obs.get("select")
        if sel is not None and sel.get("type") == 0:
            mc = int(sel.get("maxCount") or 0)
            # Capture an obs where we have a MAIN single-choice (= the kind
            # PIMC would actually evaluate).
            if mc == 1 and captured[0] is None and step_count[0] > 5:
                captured[0] = obs
        step_count[0] += 1
        return policy_agent(obs)

    env.run([capturing_agent, policy_agent])
    print(f"  captured obs at step {step_count[0]} (game finished)")

    if captured[0] is None:
        print("  ERROR: no mid-game obs captured (game too short)")
        return False

    obs = captured[0]
    print(
        f"  obs type: select.type={obs['select'].get('type')}, "
        f"maxCount={obs['select'].get('maxCount')}, "
        f"options={len(obs['select']['option'])}"
    )

    # 2. Convert to Observation dataclass
    print("\n--- Converting to Observation dataclass ---")
    try:
        agent_obs = to_observation_class(obs)
        sbi_len = len(agent_obs.search_begin_input) if agent_obs.search_begin_input else 0
        print(f"  search_begin_input length: {sbi_len}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False

    # 3. Sample opp state
    print("\n--- Sampling opp state ---")
    opp_deck, opp_hand, opp_prize, opp_active = sample_opp_state(obs)
    print(f"  opp_deck: {len(opp_deck)} cards")
    print(f"  opp_hand: {len(opp_hand)} cards")
    print(f"  opp_prize: {len(opp_prize)} cards")
    print(f"  opp_active: {len(opp_active)} cards")

    # 4. Prepare our deck / prize args
    state = obs["current"]
    your_idx = state["yourIndex"]
    your = state["players"][your_idx]
    # your_deck: remaining cards in our deck (= deck.csv minus discard/bench/active/hand)
    your_deck_pred = list(DECK)
    # your_prize: ID list of our prize cards
    your_prize_pred = []
    for prize_card in your.get("prize", []) or []:
        if prize_card is not None and isinstance(prize_card, dict) and "id" in prize_card:
            your_prize_pred.append(prize_card["id"])
        else:
            # Face-down prize — sample any card
            your_prize_pred.append(DECK[0])

    print(f"  your_deck size: {len(your_deck_pred)}, your_prize size: {len(your_prize_pred)}")

    # 5. search_begin
    print("\n--- Calling search_begin ---")
    try:
        state = search_begin(
            agent_obs,
            your_deck_pred,
            your_prize_pred,
            opp_deck,
            opp_prize,
            opp_hand,
            opp_active,
        )
        print("  OK! Got SearchState (= root)")
        print(f"  observation.search_begin_input: {state.observation.search_begin_input}")
        sid = (
            state.observation.current.searchId
            if hasattr(state.observation.current, "searchId")
            else "?"
        )
        print(f"  search_id available: {sid}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False

    # 6. Cleanup
    print("\n--- Releasing search ---")
    try:
        search_release(0)  # release all? or specific id?
        print("  OK")
    except Exception as e:
        print(f"  WARN: {type(e).__name__}: {e} (may be expected if no id tracked)")

    print("\n=== Smoke test PASSED ===")
    return True


if __name__ == "__main__":
    sys.exit(0 if smoke_test() else 1)
