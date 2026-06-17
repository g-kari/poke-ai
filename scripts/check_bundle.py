"""Fail commit if any file required by make_submission.sh is missing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    "main.py",
    "deck.csv",
    "cg/__init__.py",
    "cg/api.py",
    "cg/game.py",
    "cg/sim.py",
    "cg/utils.py",
    "cg/libcg.so",
    "cg/cg.dll",
    "train/__init__.py",
    "train/policy.py",
    "train/features.py",
    "train/mlp_policy.py",
    "make_submission.sh",
]


def main() -> int:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        print("missing submission files:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
