"""Submission entry point for romanrozen V6 (Crustle+Lucario hybrid).

Vendored from Kaggle kernel (36 votes, claims LB 860+):
  romanrozen/strong-start-crustle-lucario-agent-v6-lb-860

Lab bench (80g/opp, 2026-06-18): overall 57.9% across 6-opp meta pool.
- Mega Lucario 43.8% (lowest, single soft spot)
- Dragapult 57.5%, Iono 76.2%, Aboma 53.8%
- Crustle Wall 51.2%, Crustle Dashimaki 65.0%

Lower peak than CrustleDashi (67.3%) or Iono (64.0%) but **no critical
weakness** (lab minimum 43.8% vs Iono's 21.2% and CrustleDashi's 11.2%).
Lower variance = safer pick if LB deck mix is unknown.

How to ship:
  bash make_submission_rule_based_romanrozen_v6.sh
  .venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \
      -f submission_rule_based_romanrozen_v6.tar.gz -m "..."
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

os.environ.setdefault("RULE_DECK_PATH_ROMANROZEN_V6", str(_HERE / "deck.csv"))

for _candidate in (_HERE, _HERE / "scripts"):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from rule_based_romanrozen_v6 import agent  # noqa: E402,F401  re-exported as main.agent
