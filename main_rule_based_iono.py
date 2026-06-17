"""Submission entry point for the rule-based Iono agent.

This is candidate #2 vs the Lucario rule-based wrapper (main_rule_based.py).
Bench results (80g/opp, 2026-06-18):

  3-MLP submission             overall 23.3%
  rule_based(Mega Lucario)     overall 46.5%
  rule_based(Iono)             overall 64.0%  ← THIS

Strengths: Crustle Wall 97.5%, Crustle Dashimaki 92.5%, Abomasnow 68.8%.
Weakness: Mega Lucario 21.2% (single failure mode).

How to ship this:
  bash make_submission_rule_based_iono.sh
  .venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \
      -f submission_rule_based_iono.tar.gz -m "Switch to rule-based Iono"

The bundle uses Iono deck (deck_iono.csv -> deck.csv) and ships only
main.py + deck.csv + rule_based_iono.py + cg/ (no numpy/torch required).
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path


def _resolve_here() -> Path:
    """Resolve the bundle root. Under Kaggle's exec() entry point, __file__
    is undefined; fall back to the current working directory (Kaggle mounts
    the bundle at /kaggle_simulations/agent and chdir's into it)."""
    with contextlib.suppress(NameError):
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


_HERE = _resolve_here()

# rule_based_iono reads the deck via RULE_DECK_PATH_IONO with fallback.
# The bundle places deck.csv next to main.py, so point env var there.
os.environ.setdefault("RULE_DECK_PATH_IONO", str(_HERE / "deck.csv"))

# Local dev: rule_based_iono.py lives under scripts/. In the Kaggle bundle
# (built by make_submission_rule_based_iono.sh) it sits next to main.py.
for _candidate in (_HERE, _HERE / "scripts"):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from rule_based_iono import agent  # noqa: E402,F401  re-exported as main.agent
