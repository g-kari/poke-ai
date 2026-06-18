"""First step toward a deck-building agent (Task #107).

Loads kaggle_data/EN_Card_Data.csv, categorizes cards, and computes basic
efficiency stats. Lets us answer questions like:
  - which Basic Pokemon have the best HP-per-retreat?
  - which attacks have the best damage-per-energy?
  - what's the weakness distribution across the meta?

Usage:
    scripts/run.sh python3 scripts/analyze_cards.py [--deck deck.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "kaggle_data" / "EN_Card_Data.csv"


def _parse_int(s: str | None) -> int | None:
    if not s or s.strip() in ("n/a", ""):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _energy_count(cost: str | None) -> int:
    """Count number of energy symbols like '{F}{F}{C}' -> 3."""
    if not cost or cost.strip() in ("n/a", ""):
        return 0
    return len(re.findall(r"\{[A-Z]+\}", cost))


def load_cards(path: Path = DB_PATH) -> list[dict]:
    cards = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cards.append(row)
    return cards


def categorize(cards: list[dict]) -> dict[str, list[dict]]:
    """Group cards by their Stage/Type column."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in cards:
        cat = (c.get("Stage (Pokémon)/Type (Energy and Trainer)") or "?").strip()
        by_cat[cat].append(c)
    return by_cat


def _is_pokemon(cat: str) -> bool:
    return cat in ("Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon")


def hp_efficiency(cards: list[dict], limit: int = 10) -> None:
    """Rank Basic Pokemon by HP / (retreat + 1)."""
    rows: list[tuple[float, str, int, int]] = []
    for c in cards:
        cat = (c.get("Stage (Pokémon)/Type (Energy and Trainer)") or "").strip()
        if cat != "Basic Pokémon":
            continue
        hp = _parse_int(c.get("HP"))
        retreat = _parse_int(c.get("Retreat"))
        if hp is None or retreat is None:
            continue
        eff = hp / max(retreat + 1, 1)
        rows.append((eff, c["Card Name"], hp, retreat))
    rows.sort(reverse=True)
    print(f"\nTop {limit} Basic Pokemon by HP/(retreat+1):")
    for eff, name, hp, rc in rows[:limit]:
        print(f"  {eff:6.1f}  {name:30s}  HP={hp}  retreat={rc}")


def damage_efficiency(cards: list[dict], limit: int = 10) -> None:
    """Rank attacks by damage per energy."""
    rows: list[tuple[float, str, str, int, int]] = []
    for c in cards:
        dmg = _parse_int(c.get("Damage"))
        cost_str = c.get("Cost")
        if dmg is None or dmg <= 0 or not cost_str:
            continue
        ec = _energy_count(cost_str)
        if ec == 0:
            continue
        eff = dmg / ec
        rows.append((eff, c["Card Name"], c.get("Move Name") or "?", dmg, ec))
    rows.sort(reverse=True)
    print(f"\nTop {limit} attacks by damage/energy:")
    for eff, name, move, dmg, ec in rows[:limit]:
        print(f"  {eff:6.1f}  {name:25s}  {move:30s}  dmg={dmg}/energy={ec}")


def weakness_distribution(cards: list[dict]) -> None:
    """Count Pokemon by weakness type — tells us which attack types are
    super-effective against the meta."""
    weak_counter: Counter[str] = Counter()
    for c in cards:
        cat = (c.get("Stage (Pokémon)/Type (Energy and Trainer)") or "").strip()
        if not _is_pokemon(cat):
            continue
        weak = (c.get("Weakness") or "").strip()
        if not weak or weak in ("n/a", ""):
            continue
        weak_counter[weak] += 1
    print("\nWeakness distribution (Pokemon count by weakness type):")
    for w, n in weak_counter.most_common():
        print(f"  {w:8s}  {n}")


def deck_analysis(deck_path: Path) -> None:
    """Look at a deck.csv and summarize what's in it using the card DB."""
    cards = load_cards()
    by_id = {int(c["Card ID"]): c for c in cards if c["Card ID"].isdigit()}
    with open(deck_path) as f:
        deck_ids = [int(x.strip()) for x in f if x.strip()]
    print(f"\nDeck analysis: {deck_path} ({len(deck_ids)} cards)")
    cat_counter: Counter[str] = Counter()
    hp_total = 0
    pokemons: dict[int, int] = Counter()
    for cid in deck_ids:
        c = by_id.get(cid)
        if c is None:
            cat_counter["?unknown"] += 1
            continue
        cat = (c.get("Stage (Pokémon)/Type (Energy and Trainer)") or "?").strip()
        cat_counter[cat] += 1
        hp = _parse_int(c.get("HP"))
        if hp:
            hp_total += hp
        if _is_pokemon(cat):
            pokemons[cid] += 1
    print("  by category:")
    for cat, n in cat_counter.most_common():
        print(f"    {cat:25s}  {n}")
    if hp_total:
        print(f"  total HP across all Pokemon-card copies: {hp_total}")
    print(f"  unique Pokemon card IDs: {len(pokemons)}")
    for cid, n in pokemons.most_common(10):
        c = by_id.get(cid)
        if c:
            print(
                f"    {n}x  id={cid}  {c['Card Name']:25s}  "
                f"HP={c.get('HP', '?'):>4}  type={c.get('Type', '?')}"
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--deck",
        type=Path,
        default=None,
        help="Optional deck.csv to analyze (e.g. deck_iono.csv).",
    )
    p.add_argument("--limit", type=int, default=10, help="Top-N for ranking tables.")
    args = p.parse_args()

    cards = load_cards()
    by_cat = categorize(cards)
    print(f"Loaded {len(cards)} cards from {DB_PATH.name}")
    print("\nCategory breakdown:")
    for cat, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        print(f"  {cat:30s}  {len(items)}")

    hp_efficiency(cards, args.limit)
    damage_efficiency(cards, args.limit)
    weakness_distribution(cards)

    if args.deck:
        deck_analysis(args.deck)
    return 0


if __name__ == "__main__":
    sys.exit(main())
