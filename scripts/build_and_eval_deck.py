"""Deck-builder v4 (Task #107): generate decks + measure real bench fitness.

Pipeline:
  1. For each (target_type, allow_stage2) combo, call build_deck() to
     construct a 60-card deck.
  2. Pair the deck with a generic policy_agent (= a heuristic that just
     evolves + attacks — same shape as rule_based_agent but type-neutral).
  3. Play 10-20 games each vs N rule-based opponents.
  4. Report (deck spec, overall winrate).

This lets us compare deck constructions without running 80g/opp benches.
The fitness signal here is the seed for GA-style deck evolution later.

Usage:
    scripts/run.sh python3 scripts/build_and_eval_deck.py --games 10
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("RULE_DECK_PATH", str(ROOT / "deck_mega_lucario.csv"))
sys.modules.setdefault("litellm", types.ModuleType("litellm"))

import rule_based_abomasnow  # noqa: E402
import rule_based_agent  # noqa: E402
import rule_based_crustle_dashimaki  # noqa: E402
import rule_based_dragapult  # noqa: E402
import rule_based_iono  # noqa: E402
from build_deck import build_deck, load_cards  # noqa: E402
from kaggle_environments import make  # noqa: E402

# ---------------------------------------------------------------------------
# Type-neutral heuristic agent (built decks use varying types, so we can't
# reuse a Mega-Lucario-specific bot). This is a tiny rule-based player that:
#   1. Returns deck on initial select.
#   2. For MAIN-context single-choice decisions, picks the first ATTACH /
#      EVOLVE / PLAY / ABILITY / ATTACK option, in that priority order.
#   3. For other selects, returns engine-prior (option 0).
# ---------------------------------------------------------------------------


def make_generic_agent(deck: list[int]):
    OPTION_PRIORITY = {
        # PTCG cabt OptionType values (verified vs NOTES.md):
        8: 0,  # ATTACH
        9: 1,  # EVOLVE
        7: 2,  # PLAY
        10: 3,  # ABILITY
        13: 4,  # ATTACK
        12: 5,  # RETREAT
        14: 6,  # END turn
    }

    def agent(obs):
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

        if max_c == 1 and len(opts) > 1:
            ranked = sorted(
                enumerate(opts),
                key=lambda kv: (OPTION_PRIORITY.get(kv[1].get("type", -1), 99), kv[0]),
            )
            return [ranked[0][0]]

        k = max(min_c, 1)
        k = min(k, max_c, len(opts))
        return list(range(k))

    return agent


def play(a, b):
    env = make("cabt")
    env.run([a, b])
    return env.steps[-1][0].reward, env.steps[-1][1].reward


def measure_fitness(deck: list[int], opps: list, games_per_opp: int) -> dict:
    """Return {opp_label: winrate} + overall winrate from playing both seats."""
    agent = make_generic_agent(deck)
    rows = {}
    total_w = total_l = total = 0
    for label, opp in opps:
        random.seed(0)
        w = lo = d = 0
        for _ in range(games_per_opp):
            r0, _ = play(agent, opp)
            if r0 == 1:
                w += 1
            elif r0 == -1:
                lo += 1
            else:
                d += 1
        for _ in range(games_per_opp):
            _, r1 = play(opp, agent)
            if r1 == 1:
                w += 1
            elif r1 == -1:
                lo += 1
            else:
                d += 1
        n = 2 * games_per_opp
        rows[label] = w / n if n else 0.0
        total_w += w
        total_l += lo
        total += n
    rows["_overall"] = total_w / max(1, total)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cards", type=Path, default=ROOT / "data" / "cards.json")
    p.add_argument("--games", type=int, default=10, help="Games per (deck, opp) side")
    args = p.parse_args()

    cards = load_cards(args.cards)
    opps = [
        ("Mega Lucario", rule_based_agent.agent),
        ("Dragapult", rule_based_dragapult.agent),
        ("Iono", rule_based_iono.agent),
        ("Aboma", rule_based_abomasnow.agent),
        ("Crustle Dashi", rule_based_crustle_dashimaki.agent),
    ]

    # Try multiple deck specs.
    specs = [
        {"target_type": "R", "allow_stage2": False, "label": "Fire / Stage1"},
        {"target_type": "R", "allow_stage2": True, "label": "Fire / Stage2 OK"},
        {"target_type": "F", "allow_stage2": False, "label": "Fighting / Stage1"},
        {"target_type": "F", "allow_stage2": True, "label": "Fighting / Stage2 OK"},
        {"target_type": "L", "allow_stage2": False, "label": "Lightning / Stage1"},
        {"target_type": None, "allow_stage2": True, "label": "Default / Stage2 OK"},
    ]

    print(f"Evaluating {len(specs)} deck specs × {len(opps)} opps × 2 × {args.games}g\n")
    t0 = time.monotonic()
    results = []
    for spec in specs:
        deck = build_deck(
            cards,
            target_weakness=spec["target_type"],
            allow_stage2=spec["allow_stage2"],
        )
        winrates = measure_fitness(deck, opps, args.games)
        overall = winrates["_overall"]
        results.append((overall, spec["label"], winrates, deck[:8]))
        print(f"  [{spec['label']:30s}] overall={overall:.1%}  first attackers={deck[:4]}")

    results.sort(reverse=True)
    print(f"\nDone in {time.monotonic() - t0:.1f}s")
    print("\nRanking:")
    for ov, label, wr, _head in results:
        per_opp = ", ".join(f"{k}={v:.0%}" for k, v in wr.items() if not k.startswith("_"))
        print(f"  {ov:.1%}  {label:30s}  | {per_opp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
