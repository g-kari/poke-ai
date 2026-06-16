"""Fail commit if deck.csv is not exactly 60 well-formed card IDs."""

from __future__ import annotations

import sys
from pathlib import Path

DECK = Path(__file__).resolve().parent.parent / "deck.csv"


def main() -> int:
    if not DECK.exists():
        print(f"deck.csv missing at {DECK}", file=sys.stderr)
        return 1
    lines = [ln.strip() for ln in DECK.read_text().splitlines() if ln.strip()]
    if len(lines) != 60:
        print(f"deck.csv must have 60 card IDs, got {len(lines)}", file=sys.stderr)
        return 1
    counts: dict[int, int] = {}
    for i, ln in enumerate(lines, 1):
        try:
            cid = int(ln)
        except ValueError:
            print(f"deck.csv line {i}: not an int: {ln!r}", file=sys.stderr)
            return 1
        if cid <= 0:
            print(f"deck.csv line {i}: card id must be > 0", file=sys.stderr)
            return 1
        counts[cid] = counts.get(cid, 0) + 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
