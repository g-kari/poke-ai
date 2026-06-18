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


def pick_attacker_line(
    cards: dict,
    target_weakness: str | None = None,
    avoid_self_weakness: bool = True,
) -> tuple[dict, dict] | None:
    """Pick a (Basic, Stage1-attacker) evolution chain.

    NEW v2 semantics: we pick a Stage 1 (or Stage 1 ex/mega) ATTACKER first
    by damage-efficiency / meta-fit, then resolve `evolves_from` to find a
    Basic printing whose Name matches. Returns (basic, stage1_attacker) or
    None if no chain resolves.
    """
    pkmn_list = cards["pokemon_db"]
    name_index = _build_name_index(pkmn_list)
    weak_dist = cards.get("weakness_distribution", {})

    stage1s = [p for p in pkmn_list if p.get("stage") == "Stage 1 Pokémon"]
    scored: list[tuple[float, dict, dict]] = []
    for s1 in stage1s:
        if not s1.get("evolves_from"):
            continue
        basic = _pick_lowest_basic(name_index, s1["evolves_from"])
        if basic is None:
            continue
        # Score: HP_basic / (retreat+1) + meta-fit + damage_eff-of-stage1
        hp = s1.get("hp") or 0
        # Use stage1 HP / (retreat+1) since that's the attacker on the field
        retreat = s1.get("retreat")
        if retreat is None:
            continue
        eff = hp / max(retreat + 1, 1)
        atk_type = _energy_type(s1.get("type"))
        target_count = weak_dist.get(f"{{{atk_type}}}", 0)
        bonus = target_count / 20.0
        if target_weakness and atk_type == target_weakness:
            bonus += 30.0
        if avoid_self_weakness:
            my_weak = _energy_type(s1.get("weakness"))
            for w, count in weak_dist.items():
                if my_weak and w.strip("{}") == my_weak and count > 50:
                    bonus -= count / 30.0
        scored.append((eff + bonus, basic, s1))
    if not scored:
        return None
    scored.sort(key=lambda kv: -kv[0])
    _, best_basic, best_stage1 = scored[0]
    return best_basic, best_stage1


def build_deck(
    cards: dict,
    target_weakness: str | None = None,
    n_basic_attacker: int = 4,
    n_stage1: int = 4,
    n_energy: int = 18,
) -> list[int]:
    """Construct a 60-card deck."""
    pick = pick_attacker_line(cards, target_weakness=target_weakness)
    if not pick:
        raise RuntimeError("no attacker line found")
    basic, stage1 = pick
    deck: list[int] = []
    deck += [basic["card_id"]] * n_basic_attacker
    if stage1:
        deck += [stage1["card_id"]] * n_stage1
    # Energy: Basic of attacker's type.
    atk_type = _energy_type(basic.get("type")) or "C"
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
    # If still short, top up with more energy.
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
    args = p.parse_args()

    cards = load_cards(args.cards)
    print(f"Loaded {cards['metadata']['total_cards']} cards from {args.cards}")

    deck = build_deck(cards, target_weakness=args.target_type)
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
