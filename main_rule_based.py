"""Submission entry point for the rule-based Mega Lucario agent.

This is a switch candidate vs the 3-MLP entry point in `main.py`. Bench
results (80g per opponent, 2026-06-18):

  3-MLP submission             overall 23.3%
  rule_based(Mega Lucario)     overall 46.5% (+23.2pp)

Big wins: Iono 11.2%->83.8%, Dragapult 20%->53.8%, Aboma 21.2%->47.5%.
Concession: Crustle Wall 38.8%->25%. Crustle Dashimaki ~unchanged.

How to ship this:
  bash make_submission_rule_based.sh
  .venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \
      -f submission_rule_based.tar.gz -m "Switch to rule-based Mega Lucario"

The bundle uses Mega Lucario deck (deck_mega_lucario.csv -> deck.csv) and
ships only main.py + deck.csv + cg/ (no numpy/torch required).
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

# rule_based_agent reads the deck via RULE_DECK_PATH env var with fallback.
# The bundle places deck.csv next to main.py, so point env var there.
os.environ.setdefault("RULE_DECK_PATH", str(_HERE / "deck.csv"))

# Local dev: rule_based_agent.py lives under scripts/. In the Kaggle bundle
# (built by make_submission_rule_based.sh) it sits next to main.py. Both
# paths must resolve.
for _candidate in (_HERE, _HERE / "scripts"):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from rule_based_agent import agent  # noqa: E402,F401  re-exported as main.agent
