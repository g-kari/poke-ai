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


# Crustle line card IDs (from NOTES.md and bench data).
_CRUSTLE_IDS = {344, 345}  # Crustib (Basic), Crustle (Stage 1 — "Mysterious Rock Inn")


def _opp_has_crustle(obs: dict) -> bool:
    """Return True if opponent shows a Crustle-line Pokemon anywhere on board."""
    cur = obs.get("current")
    if not cur:
        return False
    you = cur.get("yourIndex", 0)
    players = cur.get("players") or []
    if len(players) <= 1 - you:
        return False
    opp = players[1 - you]
    for area_name in ("active", "bench", "discard"):
        for p in opp.get(area_name) or []:
            if not p:
                continue
            cid = p.get("id") if isinstance(p, dict) else getattr(p, "id", None)
            if cid in _CRUSTLE_IDS:
                return True
    return False


def _my_secondary_count(obs: dict, secondary_card_ids: set[int]) -> int:
    """Count how many secondary-chain Pokemon we already have on the field
    (active + bench). Used by v11 proactive-deploy logic."""
    cur = obs.get("current")
    if not cur:
        return 0
    you = cur.get("yourIndex", 0)
    players = cur.get("players") or []
    if you >= len(players):
        return 0
    me = players[you]
    count = 0
    for area in (me.get("active") or [], me.get("bench") or []):
        for p in area:
            if not p:
                continue
            cid = p.get("id") if isinstance(p, dict) else getattr(p, "id", None)
            if cid in secondary_card_ids:
                count += 1
    return count


def _my_active_card_id(obs: dict) -> int | None:
    """Return the card ID of our currently-active Pokemon, or None."""
    cur = obs.get("current")
    if not cur:
        return None
    you = cur.get("yourIndex", 0)
    players = cur.get("players") or []
    if you >= len(players):
        return None
    me = players[you]
    active = me.get("active") or []
    if not active or active[0] is None:
        return None
    p = active[0]
    return p.get("id") if isinstance(p, dict) else getattr(p, "id", None)


def make_generic_agent(deck: list[int], secondary_card_ids: set[int] | None = None):
    """Generic heuristic agent. With `secondary_card_ids`, switches to anti-
    Crustle routing when an opponent Crustle is detected: ATTACH / EVOLVE /
    ATTACK options targeting a secondary-chain card jump to the top priority.

    v10 adds rotation routing: when opp has Crustle AND our active is NOT
    a secondary-chain card, prefer RETREAT (option type 12) to swap in a
    secondary attacker. On the follow-up "which bench to swap to" select,
    prefer the option referencing a secondary card ID."""
    secondary_card_ids = secondary_card_ids or set()

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

    def _option_targets_secondary(opt: dict, sel: dict, _obs: dict) -> bool:
        """Heuristic: True if the option appears to deploy a secondary card.

        We check the option payload for an inPlay* hint or a contextCard ID,
        and also fall back to the select.deck contents (= card IDs in the
        hand for PLAY/EVOLVE options)."""
        if not secondary_card_ids:
            return False
        # contextCard: cabt sometimes attaches the card being acted on.
        ctx = opt.get("contextCard") or sel.get("contextCard")
        if isinstance(ctx, dict) and ctx.get("id") in secondary_card_ids:
            return True
        # cardId field (some option types carry it directly). toolIndex/
        # energyIndex/etc. reference indices into select.deck; we don't
        # follow those here, contextCard cover is enough.
        return opt.get("cardId") in secondary_card_ids

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
            crustle_mode = secondary_card_ids and _opp_has_crustle(obs)
            # v11: proactive secondary deployment.
            # Whenever we have NO secondary on the field yet, prioritize any
            # option that places one (PLAY/EVOLVE/ATTACH targeting secondary).
            # Once one is set up, fall back to normal priority unless Crustle
            # is detected (then keep boosting secondary).
            need_proactive_deploy = False
            if secondary_card_ids:
                sec_count = _my_secondary_count(obs, secondary_card_ids)
                if sec_count == 0:
                    need_proactive_deploy = True

            def _rank_key(kv):
                idx, opt = kv
                base = OPTION_PRIORITY.get(opt.get("type", -1), 99)
                touches_secondary = _option_targets_secondary(opt, sel, obs)
                # v11: proactive — strong boost when no secondary is up yet.
                if need_proactive_deploy and touches_secondary:
                    base -= 20  # higher than crustle_mode boost
                # v8 (kept): Crustle mode boost for any secondary-touching opt.
                if crustle_mode and touches_secondary:
                    base -= 10
                return (base, idx)

            ranked = sorted(enumerate(opts), key=_rank_key)
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

    # Try multiple deck specs. v5 adds non-ex variants (anti-Crustle route).
    specs = [
        {"target_type": "R", "allow_stage2": False, "label": "Fire / Stage1"},
        {"target_type": "F", "allow_stage2": False, "label": "Fighting / Stage1"},
        {"target_type": "L", "allow_stage2": False, "label": "Lightning / Stage1"},
        {"target_type": "P", "allow_stage2": False, "label": "Psychic / Stage1"},
        # v5: non-ex specs (V6-style Hariyama route, immune to Crustle ex lock)
        {
            "target_type": "F",
            "allow_stage2": False,
            "require_non_ex": True,
            "label": "Fighting / Stage1 / non-ex",
        },
        {
            "target_type": "R",
            "allow_stage2": False,
            "require_non_ex": True,
            "label": "Fire / Stage1 / non-ex",
        },
        {
            "target_type": None,
            "allow_stage2": False,
            "require_non_ex": True,
            "label": "Default / Stage1 / non-ex",
        },
        {"target_type": None, "allow_stage2": True, "label": "Default / Stage2 OK (baseline)"},
    ]

    print(f"Evaluating {len(specs)} deck specs × {len(opps)} opps × 2 × {args.games}g\n")
    t0 = time.monotonic()
    results = []
    for spec in specs:
        deck = build_deck(
            cards,
            target_weakness=spec["target_type"],
            allow_stage2=spec["allow_stage2"],
            require_non_ex=spec.get("require_non_ex", False),
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
