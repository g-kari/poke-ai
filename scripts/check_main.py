"""Smoke-test main.py: it must be importable, expose agent(), and return a
60-card list for the initial deck-submission step. Avoids numpy / engine
imports because those may not work in the nix host shell."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    spec = importlib.util.spec_from_file_location("submission_main", ROOT / "main.py")
    if spec is None or spec.loader is None:
        print("could not load main.py spec", file=sys.stderr)
        return 1
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        print(f"main.py failed to import: {exc}", file=sys.stderr)
        return 1
    if not callable(getattr(mod, "agent", None)):
        print("main.agent() not defined or not callable", file=sys.stderr)
        return 1
    deck = mod.agent({"select": None})
    if not isinstance(deck, list) or len(deck) != 60:
        print(f"main.agent(select=None) must return 60 cards, got {len(deck)}", file=sys.stderr)
        return 1
    if not all(isinstance(x, int) and x > 0 for x in deck):
        print("main.agent(select=None) must return positive int card IDs", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
