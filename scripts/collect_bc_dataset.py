"""Collect (obs, V6 choice) pairs by running V6 agent vs varied opponents.

This is the BC (Behavioral Cloning) dataset generator. Each (state_features,
option_features for all options, V6's picked index) tuple becomes one
supervised training sample.

Why V6: it's our LB best (= 921-926) and combines deck + agent in a tight
30+ line CRUSTLE_AWARE routine. We can't extract its logic, but we CAN
collect its decisions and learn to imitate them with a generic policy.

Usage:
    scripts/run.sh python3 scripts/collect_bc_dataset.py \\
        --games 200 --out data/sweep/bc_v6_dataset.npz
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("RULE_DECK_PATH", str(ROOT / "deck_mega_lucario.csv"))
sys.modules.setdefault("litellm", types.ModuleType("litellm"))

import numpy as np  # noqa: E402
import rule_based_abomasnow  # noqa: E402
import rule_based_agent  # noqa: E402
import rule_based_crustle_dashimaki  # noqa: E402
import rule_based_dragapult  # noqa: E402
import rule_based_iono  # noqa: E402
import rule_based_romanrozen_v6  # noqa: E402
from kaggle_environments import make  # noqa: E402

from train.features_v60 import OPTION_DIM, STATE_DIM, option_features, state_features  # noqa: E402


def make_logging_v6_agent(samples: list):
    """Wrap V6 agent so each MAIN single-choice select is logged as a sample."""
    inner = rule_based_romanrozen_v6.agent

    def agent(obs):
        choice = inner(obs)
        sel = obs.get("select")
        # Log only single-choice MAIN selects where policy will be applied.
        if sel is not None and len(choice) == 1:
            opts = sel.get("option") or []
            max_c = int(sel.get("maxCount") or 0)
            if max_c == 1 and len(opts) > 1 and sel.get("type") == 0:
                try:
                    sf = state_features(obs)
                    of_all = np.stack([option_features(o, obs, sel) for o in opts]).astype(
                        np.float32
                    )
                    samples.append(
                        {
                            "sf": sf.astype(np.float32),
                            "of_all": of_all,
                            "picked": int(choice[0]),
                            "n_opts": len(opts),
                        }
                    )
                except Exception:
                    pass
        return choice

    return agent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "sweep" / "bc_v6_dataset.npz")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    opps = [
        rule_based_agent.agent,
        rule_based_dragapult.agent,
        rule_based_iono.agent,
        rule_based_abomasnow.agent,
        rule_based_crustle_dashimaki.agent,
    ]

    samples: list = []
    random.seed(0)
    t0 = time.monotonic()
    for g in range(args.games):
        opp = random.choice(opps)
        v6_logging = make_logging_v6_agent(samples)
        env = make("cabt")
        # Alternate seats to balance
        seats = [v6_logging, opp] if g % 2 == 0 else [opp, v6_logging]
        env.run(seats)
        if (g + 1) % 25 == 0:
            print(
                f"  game {g + 1}/{args.games}, samples so far: {len(samples)} "
                f"[{time.monotonic() - t0:.0f}s]"
            )

    print(f"\ncollected {len(samples)} BC samples from {args.games} games")

    # Pack as ragged arrays (n_opts varies per sample). Save sf stacked
    # (every sample is STATE_DIM wide) and pickled lists for of_all.
    sf_all = np.stack([s["sf"] for s in samples]).astype(np.float32)
    picked_all = np.array([s["picked"] for s in samples], dtype=np.int32)
    n_opts_all = np.array([s["n_opts"] for s in samples], dtype=np.int32)
    # of_all is ragged; we flatten with a CSR-style indptr.
    of_flat_list = [s["of_all"] for s in samples]
    of_flat = np.concatenate(of_flat_list, axis=0).astype(np.float32)
    indptr = np.cumsum([0] + [a.shape[0] for a in of_flat_list]).astype(np.int32)

    np.savez(
        args.out,
        sf=sf_all,
        of_flat=of_flat,
        of_indptr=indptr,
        picked=picked_all,
        n_opts=n_opts_all,
        state_dim=np.asarray(STATE_DIM),
        option_dim=np.asarray(OPTION_DIM),
    )
    print(f"wrote {args.out}")
    print(
        f"  sf: {sf_all.shape}, of_flat: {of_flat.shape} "
        f"(total options: {sum(s['n_opts'] for s in samples)}), "
        f"picked: {picked_all.shape}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
