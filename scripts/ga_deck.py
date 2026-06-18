"""GA loop for the deck-builder (Task #107).

Start from a seed deck. Each generation:
  1. Mutate by swapping one card for a candidate from a pool.
  2. Evaluate vs N rule-based opponents at K games/side.
  3. Keep the mutation if its overall winrate >= parent's.

This is a very simple hill-climber, not a real GA — single-parent,
single-child per generation. Designed to run unattended (overnight) and
write its history to JSON for offline review.

Usage:
    scripts/run.sh python3 scripts/ga_deck.py \\
        --seed-deck deck_builder_v4_top.csv --generations 20 \\
        --games-per-eval 5 --out data/sweep/ga_history.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import types
from collections import Counter
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
from build_and_eval_deck import measure_fitness  # noqa: E402

# Trainer + Energy pool used as mutation candidates. Curated from existing
# decks so each card is engine-legal.
MUTATION_POOL = [
    # Energies
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,  # Basic G/R/W/L/P/F/D/M
    11,
    14,
    18,
    19,  # Special Energies
    # Items
    1086,
    1102,
    1147,
    1159,
    1212,
    1224,
    1264,
    # Supporters
    1142,
    1182,
    1192,
    1227,
    # Stadiums
    1252,
    # Tools
    1141,
    1123,
]


def _load_deck(path: Path) -> list[int]:
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


def _save_deck(deck: list[int], path: Path) -> None:
    with open(path, "w") as f:
        for c in deck:
            f.write(f"{c}\n")


def _mutate(deck: list[int], rng: random.Random) -> list[int]:
    """Replace one randomly chosen card with a candidate from the pool.

    Avoid invalid states (60-card length preserved; max 4 copies per
    non-energy ID; ACE SPEC = 1 is best-effort, not enforced here)."""
    out = list(deck)
    idx = rng.randrange(len(out))
    counts = Counter(out)
    # Try until we find a swap that doesn't break the 4-copy rule.
    for _ in range(50):
        cand = rng.choice(MUTATION_POOL)
        new_count = counts[cand] + (0 if out[idx] == cand else 1) - (1 if out[idx] == cand else 0)
        # Allow energy stacks > 4 only for Basic energies (IDs 1..8).
        is_basic_energy = cand <= 8
        if not is_basic_energy and new_count > 4:
            continue
        out[idx] = cand
        return out
    return out  # giving up after 50 tries


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-deck", type=Path, required=True)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--games-per-eval", type=int, default=5)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "sweep" / "ga_history.json")
    args = p.parse_args()

    rng = random.Random(args.rng_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    opps = [
        ("Mega Lucario", rule_based_agent.agent),
        ("Dragapult", rule_based_dragapult.agent),
        ("Iono", rule_based_iono.agent),
        ("Aboma", rule_based_abomasnow.agent),
        ("Crustle Dashi", rule_based_crustle_dashimaki.agent),
    ]

    parent = _load_deck(args.seed_deck)
    print(f"seed deck: {len(parent)} cards from {args.seed_deck}")
    parent_fit = measure_fitness(parent, opps, args.games_per_eval)["_overall"]
    print(f"  parent fitness: {parent_fit:.1%}")

    history = [
        {
            "gen": 0,
            "deck": parent,
            "fitness": parent_fit,
            "accepted": True,
            "swap": None,
        }
    ]
    t0 = time.monotonic()
    for gen in range(1, args.generations + 1):
        child = _mutate(parent, rng)
        # Find what changed (1 position by construction).
        swap = None
        for i, (a, b) in enumerate(zip(parent, child, strict=False)):
            if a != b:
                swap = {"idx": i, "from": a, "to": b}
                break
        child_fit = measure_fitness(child, opps, args.games_per_eval)["_overall"]
        accepted = child_fit >= parent_fit
        dt = time.monotonic() - t0
        print(
            f"  gen {gen:3d}  swap[{swap['idx'] if swap else '?'}]: "
            f"{swap['from'] if swap else '?'} -> {swap['to'] if swap else '?'}  "
            f"fit={child_fit:.1%} (parent {parent_fit:.1%})  "
            f"{'✓ accept' if accepted else '✗ reject'}  "
            f"[{dt:.0f}s elapsed]"
        )
        history.append(
            {
                "gen": gen,
                "swap": swap,
                "fitness": child_fit,
                "accepted": accepted,
            }
        )
        if accepted:
            parent = child
            parent_fit = child_fit
        # Persist after every generation so we don't lose progress.
        with open(args.out, "w") as f:
            json.dump(
                {
                    "seed_deck": str(args.seed_deck),
                    "rng_seed": args.rng_seed,
                    "games_per_eval": args.games_per_eval,
                    "generations": args.generations,
                    "best_fitness": parent_fit,
                    "best_deck": parent,
                    "history": history,
                },
                f,
                indent=2,
            )

    print(f"\nbest fitness after {args.generations} gens: {parent_fit:.1%}")
    print(f"history saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
