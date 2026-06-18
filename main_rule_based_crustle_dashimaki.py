"""Submission entry point for the rule-based Crustle Dashimaki agent.

This is candidate #3 vs Lucario / Iono wrappers. Bench results
(80g/opp, 2026-06-18):

  3-MLP submission             overall 23.3%
  rule_based(Mega Lucario)     overall 46.5%
  rule_based(Iono)             overall 64.0%
  rule_based(CrustleDashimaki) overall 67.3%  ← THIS (strongest)

Strengths: Dragapult 100%, Crustle Wall 85.0%, Mega Lucario 83.8%,
Abomasnow 73.8%, mirror 50.0%.
Weakness: Iono 11.2% (single failure mode — directly opposite Iono's
weakness vs Lucario, so picking between Iono and CrustleDashi is about
which deck is more common on the LB).

How to ship this:
  bash make_submission_rule_based_crustle_dashimaki.sh
  .venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \
      -f submission_rule_based_crustle_dashimaki.tar.gz \
      -m "Switch to rule-based Crustle Dashimaki"
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path


def _resolve_here() -> Path:
    with contextlib.suppress(NameError):
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


_HERE = _resolve_here()

# rule_based_crustle_dashimaki reads the deck via
# RULE_DECK_PATH_CRUSTLE_DASHIMAKI with fallback. The bundle places
# deck.csv next to main.py, so point env var there.
os.environ.setdefault("RULE_DECK_PATH_CRUSTLE_DASHIMAKI", str(_HERE / "deck.csv"))

for _candidate in (_HERE, _HERE / "scripts"):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from rule_based_crustle_dashimaki import agent  # noqa: E402,F401  re-exported as main.agent
