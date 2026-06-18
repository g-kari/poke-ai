"""Heuristic deck-builder agent prototype (Task #107).

Reads data/cards.json (produced by analyze_cards.py --json) and emits a
60-card deck.csv using simple efficiency heuristics:

  - Pokemon: pick ONE evolution line (Basic + Stage1) with high HP/retreat
    AND high damage/energy. Want 4 Basic + 4 Stage1 = 8 attacker slots.
  - Energy: 18 Basic Energy of the attacker's type.
  - Trainer/Item/Tool/Stadium/Supporter: 34 supporting slots, filled with
    generic "always good" picks plus weakness-targeted filler.

Constraints (PTCG):
  - Total 60 cards.
  - Each card ID max 4 copies, except ACE SPEC = 1.
  - Need at least 1 Basic Pokemon (engine rules).

Usage:
    scripts/run.sh python3 scripts/build_deck.py \\
        --cards data/cards.json --out deck_builder_v1.csv --target-type R
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Energy-type Basic Energy card IDs (from kaggle_data/EN_Card_Data.csv).
ENERGY_BASIC = {
    "G": 1,
    "R": 2,
    "W": 3,
    "L": 4,
    "P": 5,
    "F": 6,
    "D": 7,
    "M": 8,
}
# Reasonable trainer staples (verified IDs from existing decks).
# These appear across multiple meta decks so they're "always good" picks.
TRAINER_STAPLES = [
    # ID, max copies, description
    (1086, 4, "Buddy-Buddy Poffin"),  # search Basic Pokemon
    (1102, 4, "Dusk Ball"),  # search any Pokemon
    (1123, 2, "Switch"),  # active rotation
    (1182, 2, "Boss's Orders"),  # bench gust supporter
    (1192, 3, "Carmine"),  # draw supporter
    (1227, 3, "Lillie's Determination"),  # draw supporter
    (1252, 2, "Gravity Mountain"),  # stadium
    (1159, 1, "Hero's Cape"),  # ACE SPEC tool
]


def load_cards(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _energy_type(weak: str | None) -> str | None:
    """Strip braces from a weakness string '{R}' -> 'R'."""
    if not weak:
        return None
    return weak.strip("{}").upper()[:1] if weak else None


def _build_name_index(pkmn_db_list: list[dict]) -> dict[str, list[dict]]:
    """Group cards by Pokemon Name so evolves_from can resolve to a list of
    candidate pre-evolution cards (multiple printings of e.g. 'Riolu')."""
    idx: dict[str, list[dict]] = {}
    for p in pkmn_db_list:
        idx.setdefault(p["name"], []).append(p)
    return idx


def _pick_lowest_basic(name_to_cards: dict[str, list[dict]], name: str) -> dict | None:
    """Pick the lowest-ID Basic-stage printing for a Pokemon name."""
    cards = name_to_cards.get(name) or []
    basics = [c for c in cards if c.get("stage") == "Basic Pokémon"]
    if not basics:
        return None
    basics.sort(key=lambda c: c.get("card_id", 9999))
    return basics[0]


def _pick_lowest_stage1(name_to_cards: dict[str, list[dict]], name: str) -> dict | None:
    """Pick the lowest-ID Stage 1 printing for a Pokemon name."""
    cards = name_to_cards.get(name) or []
    s1s = [c for c in cards if c.get("stage") == "Stage 1 Pokémon"]
    if not s1s:
        return None
    s1s.sort(key=lambda c: c.get("card_id", 9999))
    return s1s[0]


def _resolve_chain(name_to_cards: dict[str, list[dict]], attacker: dict) -> list[dict] | None:
    """Walk evolves_from upward and return [basic, stage1?, attacker].

    Returns None if any link is unresolvable.
    """
    stage = attacker.get("stage")
    if stage == "Basic Pokémon":
        return [attacker]
    if stage == "Stage 1 Pokémon":
        ev = attacker.get("evolves_from")
        if not ev:
            return None
        basic = _pick_lowest_basic(name_to_cards, ev)
        return [basic, attacker] if basic else None
    if stage == "Stage 2 Pokémon":
        ev1 = attacker.get("evolves_from")
        if not ev1:
            return None
        # ev1 is a Stage 1 Pokemon Name; find a Stage 1 printing
        stage1 = _pick_lowest_stage1(name_to_cards, ev1)
        if not stage1:
            return None
        ev2 = stage1.get("evolves_from")
        if not ev2:
            return None
        basic = _pick_lowest_basic(name_to_cards, ev2)
        return [basic, stage1, attacker] if basic else None
    return None


def pick_attacker_chain(
    cards: dict,
    target_weakness: str | None = None,
    avoid_self_weakness: bool = True,
    allow_stage2: bool = True,
    require_non_ex: bool = False,
) -> list[dict] | None:
    """Pick an attacker chain [basic, (stage1?), attacker].

    v5: target_weakness bonus boosted from 30 → 200 (now actually dominant
    over raw HP/retreat efficiency). ex penalty added because ex pokemon
    are vulnerable to Crustle's "Mysterious Rock Inn" lock. Optional
    require_non_ex strict filter for "wall-bypass" specs (V6-style).
    """
    pkmn_list = cards["pokemon_db"]
    name_index = _build_name_index(pkmn_list)
    weak_dist = cards.get("weakness_distribution", {})

    candidates: list[dict] = [p for p in pkmn_list if p.get("stage") == "Stage 1 Pokémon"]
    if allow_stage2:
        candidates += [p for p in pkmn_list if p.get("stage") == "Stage 2 Pokémon"]

    scored: list[tuple[float, list[dict]]] = []
    for atk in candidates:
        if require_non_ex and atk.get("ex"):
            continue
        chain = _resolve_chain(name_index, atk)
        if not chain:
            continue
        hp = atk.get("hp") or 0
        retreat = atk.get("retreat")
        if retreat is None:
            continue
        eff = hp / max(retreat + 1, 1)
        atk_type = _energy_type(atk.get("type"))
        target_count = weak_dist.get(f"{{{atk_type}}}", 0)
        bonus = target_count / 20.0
        # v6 (reverted from v5 mistake): modest target-type bonus, no ex
        # penalty — HP/(retreat+1) eff should dominate so the best Stage1
        # ex (= v4's anti-meta discovery) can win.
        if target_weakness and atk_type == target_weakness:
            bonus += 30.0
        if avoid_self_weakness:
            my_weak = _energy_type(atk.get("weakness"))
            for w, count in weak_dist.items():
                if my_weak and w.strip("{}") == my_weak and count > 50:
                    bonus -= count / 30.0
        # Penalty per evolution step (harder to set up).
        bonus -= (len(chain) - 1) * 5.0
        scored.append((eff + bonus, chain))
    if not scored:
        return None
    scored.sort(key=lambda kv: -kv[0])
    return scored[0][1]


# Backwards-compatible alias for v2 callers
def pick_attacker_line(
    cards: dict,
    target_weakness: str | None = None,
    avoid_self_weakness: bool = True,
) -> tuple[dict, dict] | None:
    chain = pick_attacker_chain(
        cards,
        target_weakness=target_weakness,
        avoid_self_weakness=avoid_self_weakness,
        allow_stage2=False,
    )
    return (chain[0], chain[-1]) if chain and len(chain) == 2 else None


def pick_attacker_chains(
    cards: dict,
    target_weakness: str | None = None,
    avoid_self_weakness: bool = True,
    allow_stage2: bool = True,
) -> tuple[list[dict], list[dict]] | None:
    """v7: pick TWO complementary attacker chains.

    - Primary: best by HP/(retreat+1) + meta-fit (= same as v6/v4).
    - Secondary: best NON-EX whose type is *different* from primary
      (= V6-style fallback for Crustle "Mysterious Rock Inn" lock).

    Returns (primary_chain, secondary_chain) or None on resolution failure.
    The secondary chain shares Basic-only Pokemon space so the deck still
    fits in 60 cards.
    """
    primary = pick_attacker_chain(
        cards,
        target_weakness=target_weakness,
        avoid_self_weakness=avoid_self_weakness,
        allow_stage2=allow_stage2,
    )
    if not primary:
        return None
    primary_type = _energy_type(primary[-1].get("type"))
    # Find best non-ex attacker of a different type.
    # We do a constrained search on Stage 1 only (to keep n_each_stage modest).
    pkmn_list = cards["pokemon_db"]
    name_index = _build_name_index(pkmn_list)
    weak_dist = cards.get("weakness_distribution", {})
    candidates = [
        p
        for p in pkmn_list
        if p.get("stage") == "Stage 1 Pokémon"
        and not p.get("ex")
        and _energy_type(p.get("type")) != primary_type
    ]
    scored: list[tuple[float, list[dict]]] = []
    for atk in candidates:
        chain = _resolve_chain(name_index, atk)
        if not chain:
            continue
        hp = atk.get("hp") or 0
        retreat = atk.get("retreat")
        if retreat is None:
            continue
        eff = hp / max(retreat + 1, 1)
        atk_type = _energy_type(atk.get("type"))
        target_count = weak_dist.get(f"{{{atk_type}}}", 0)
        bonus = target_count / 20.0
        # No target_weakness override for secondary; just take strongest.
        bonus -= (len(chain) - 1) * 5.0
        scored.append((eff + bonus, chain))
    if not scored:
        return None
    scored.sort(key=lambda kv: -kv[0])
    secondary = scored[0][1]
    return primary, secondary


def build_hybrid_deck(
    cards: dict,
    target_weakness: str | None = None,
    n_primary_each: int = 3,
    n_secondary_each: int = 2,
    n_energy: int = 16,
    primary_allow_stage2: bool = True,
) -> list[int]:
    """v7: build a 60-card hybrid deck with two attacker lines.

    Defaults: 3x primary chain (each card) + 2x secondary chain + 16 energy
    + trainer staples. v9 callers should use n_secondary_each=4 (and force
    primary Stage 1) so the anti-Crustle line is drawable.
    """
    pair = pick_attacker_chains(
        cards, target_weakness=target_weakness, allow_stage2=primary_allow_stage2
    )
    if not pair:
        raise RuntimeError("no hybrid chain pair found")
    primary, secondary = pair
    deck: list[int] = []
    for c in primary:
        deck += [c["card_id"]] * n_primary_each
    for c in secondary:
        deck += [c["card_id"]] * n_secondary_each

    # Primary-type energy.
    atk_type = _energy_type(primary[-1].get("type")) or "C"
    energy_id = ENERGY_BASIC.get(atk_type, ENERGY_BASIC["F"])
    deck += [energy_id] * n_energy

    used_counts: Counter = Counter(deck)
    for cid, max_n, _name in TRAINER_STAPLES:
        slots = 60 - len(deck)
        if slots <= 0:
            break
        take = min(max_n - used_counts[cid], max_n, slots)
        if take <= 0:
            continue
        deck += [cid] * take
        used_counts[cid] += take
    while len(deck) < 60:
        deck.append(energy_id)
    return deck[:60]


def build_deck(
    cards: dict,
    target_weakness: str | None = None,
    n_each_stage: int = 4,
    n_energy: int = 18,
    allow_stage2: bool = True,
    require_non_ex: bool = False,
) -> list[int]:
    """Construct a 60-card deck.

    Picks an attacker chain (Basic + Stage1 [+ Stage2]) and includes
    n_each_stage copies of each stage card, plus energy + trainer staples.
    """
    chain = pick_attacker_chain(
        cards,
        target_weakness=target_weakness,
        allow_stage2=allow_stage2,
        require_non_ex=require_non_ex,
    )
    if not chain:
        raise RuntimeError("no attacker chain found")
    deck: list[int] = []
    for card in chain:
        deck += [card["card_id"]] * n_each_stage
    # Energy: Basic of attacker (top of chain) type.
    atk_type = _energy_type(chain[-1].get("type")) or "C"
    energy_id = ENERGY_BASIC.get(atk_type, ENERGY_BASIC["F"])
    deck += [energy_id] * n_energy
    # Trainer staples until we fill 60.
    used_counts: Counter = Counter(deck)
    for cid, max_n, _name in TRAINER_STAPLES:
        slots = 60 - len(deck)
        if slots <= 0:
            break
        take = min(max_n - used_counts[cid], max_n, slots)
        if take <= 0:
            continue
        deck += [cid] * take
        used_counts[cid] += take
    while len(deck) < 60:
        deck.append(energy_id)
    return deck[:60]


def evaluate_deck(cards: dict, deck: list[int]) -> dict:
    """Compute basic stats about the deck."""
    pkmn_db = {p["card_id"]: p for p in cards["pokemon_db"]}
    cat = Counter()
    types_count = Counter()
    weaknesses = Counter()
    hp_total = 0
    for cid in deck:
        p = pkmn_db.get(cid)
        if p is None:
            cat["?nonpokemon"] += 1
            continue
        cat[p.get("stage", "?")] += 1
        t = p.get("type")
        if t:
            types_count[t] += 1
        w = p.get("weakness")
        if w:
            weaknesses[w] += 1
        hp = p.get("hp")
        if hp:
            hp_total += hp
    return {
        "size": len(deck),
        "category_counts": dict(cat),
        "type_counts": dict(types_count),
        "weakness_counts": dict(weaknesses),
        "hp_total_across_all_pokemon_copies": hp_total,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cards",
        type=Path,
        default=ROOT / "data" / "cards.json",
        help="data/cards.json input (built by analyze_cards.py --json).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "deck_builder_v1.csv",
        help="Output deck.csv path.",
    )
    p.add_argument(
        "--target-type",
        default=None,
        help="Optional energy type to push (e.g. 'R' for Fire to counter the "
        "361 Fire-weak Pokemon in the meta).",
    )
    p.add_argument(
        "--no-stage2",
        action="store_true",
        help="Disable Stage 2 attackers (force Stage 1 only — faster setup).",
    )
    p.add_argument(
        "--non-ex",
        action="store_true",
        help="Require non-ex attacker (= V6-style anti-Crustle Hariyama route).",
    )
    p.add_argument(
        "--hybrid",
        action="store_true",
        help="v7: build a hybrid deck with primary ex + secondary non-ex chain.",
    )
    args = p.parse_args()

    cards = load_cards(args.cards)
    print(f"Loaded {cards['metadata']['total_cards']} cards from {args.cards}")

    if args.hybrid:
        deck = build_hybrid_deck(cards, target_weakness=args.target_type)
    else:
        deck = build_deck(
            cards,
            target_weakness=args.target_type,
            allow_stage2=not args.no_stage2,
            require_non_ex=args.non_ex,
        )
    assert len(deck) == 60, f"deck size != 60: {len(deck)}"
    with open(args.out, "w") as f:
        for cid in deck:
            f.write(f"{cid}\n")
    print(f"\nwrote {args.out} ({len(deck)} cards)")

    stats = evaluate_deck(cards, deck)
    print("\nDeck stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
