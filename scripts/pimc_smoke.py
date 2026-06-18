"""PIMC smoke test: verify cg.api.search_begin/step/release actually fire.

What we want to learn:
  1. Does search_begin return a usable SearchState in our environment?
  2. How long does one search_begin + a few search_step + search_release take?
     (per-turn budget on Kaggle is ~3s; PIMC must fit.)
  3. What does SearchState.observation look like for the SAME decision point —
     do its options match the engine's actual options?

For this smoke we cheat with full information (we know our own and
opponent's decks because both seats are us in self-play). Real PIMC samples
the opponent's unknown info; that's the next milestone.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
import types
from pathlib import Path

sys.modules.setdefault("litellm", types.ModuleType("litellm"))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("RULE_DECK_PATH_IONO", str(ROOT / "deck_iono.csv"))

from cg.api import search_begin, search_release, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


def read_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def main() -> int:
    our_deck = read_deck(str(ROOT / "deck.csv"))
    opp_deck = read_deck(str(ROOT / "deck_iono.csv"))
    print(f"our deck: {len(our_deck)} cards, head={our_deck[:5]}")
    print(f"opp deck: {len(opp_deck)} cards, head={opp_deck[:5]}")

    # Start a battle.
    obs, start = battle_start(our_deck, opp_deck)
    if obs is None:
        print(f"battle_start failed: {start}")
        return 1
    print(f"battle started; result.errorType={getattr(start, 'errorType', '?')}")

    steps_taken = 0
    pimc_calls = 0
    total_search_time = 0.0
    max_steps = 30  # smoke: just probe a few decisions

    try:
        for _ in range(max_steps):
            o = to_observation_class(obs)
            cur = o.current
            if cur is None:
                # Initial deck submission step — return the deck.
                obs = battle_select(our_deck if steps_taken % 2 == 0 else opp_deck)
                steps_taken += 1
                continue
            if cur.result is not None and cur.result >= 0:
                print(f"battle finished: result={cur.result}")
                break

            # Only the seat whose turn it is gets to act.
            seat = cur.yourIndex
            deck_for_me = our_deck if seat == 0 else opp_deck
            deck_for_opp = opp_deck if seat == 0 else our_deck

            # Look at the actual option count to compare with PIMC's view.
            sel = o.select
            n_opts = len(sel.option) if sel and sel.option else 0
            max_c = sel.maxCount if sel else 0

            # PIMC PROBE: try search_begin with FULL INFORMATION (we know
            # both decks because both seats are us). Fill prize/hand/active
            # from the public observation.
            opp_player = cur.players[1 - seat]
            opp_prize = list(opp_player.prize) if opp_player.prize else []
            opp_hand_count = opp_player.handCount or 0
            # We don't know opp's hand cards; smoke uses arbitrary basics
            # from their deck. Real PIMC samples this set.
            opp_hand = deck_for_opp[:opp_hand_count]
            # Active card if face-down; otherwise empty list.
            opp_active: list[int] = []

            try:
                t0 = time.monotonic()
                # docstring says "pass obs as is" but the C side reads
                # .search_begin_input as an attribute, so we must convert
                # to the Observation dataclass first.
                search_state = search_begin(
                    agent_observation=to_observation_class(obs),
                    your_deck=deck_for_me,
                    your_prize=[deck_for_me[0]] * 6,  # dummy prize fillers
                    opponent_deck=deck_for_opp,
                    opponent_prize=opp_prize if len(opp_prize) == 6 else [deck_for_opp[0]] * 6,
                    opponent_hand=opp_hand,
                    opponent_active=opp_active,
                    manual_coin=False,
                )
                dt = time.monotonic() - t0
                total_search_time += dt
                pimc_calls += 1
                sid = search_state.searchId
                sim_o = search_state.observation
                sim_opts = (
                    sim_o.get("select", {}).get("option", []) if isinstance(sim_o, dict) else []
                )
                print(
                    f"  step {steps_taken:2d} seat={seat} engine opts={n_opts} maxC={max_c} "
                    f"PIMC opts={len(sim_opts)} sid={sid} dt={dt * 1000:.1f}ms"
                )
                search_release(sid)
            except Exception as exc:  # noqa: BLE001
                print(f"  step {steps_taken} PIMC failed: {type(exc).__name__}: {exc}")

            # Advance: take a trivial action (option 0 if any options).
            action = [0] if max_c >= 1 else []
            obs = battle_select(action)
            steps_taken += 1

    finally:
        with contextlib.suppress(Exception):
            battle_finish()

    if pimc_calls > 0:
        print(
            f"\nPIMC stats: {pimc_calls} successful calls, "
            f"avg {total_search_time / pimc_calls * 1000:.1f}ms each"
        )
    else:
        print("\nPIMC never succeeded (or smoke didn't reach a decision point).")
    print(f"total steps: {steps_taken}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
