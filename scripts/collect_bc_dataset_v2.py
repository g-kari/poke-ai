"""V2 BC dataset: only V6's WINNING-game decisions are kept.

bc_v60_v1 (= collect_bc_dataset.py) trained on V6's choices regardless of
game outcome, giving lab 5.7% — BC overfit to the median V6 decision but
distribution-shifted itself into states V6 never explored. By filtering
to games V6 actually won, we drop ~40-50% of noisy/losing trajectories
and concentrate on V6's competent regions.

Usage:
    scripts/run.sh python3 scripts/collect_bc_dataset_v2.py \\
        --games 400 --out data/sweep/bc_v6_wins_only.npz
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


def make_logging_v6_agent(per_game_samples: list):
    """Wrap V6 agent so each MAIN single-choice select is logged per-game."""
    inner = rule_based_romanrozen_v6.agent

    def agent(obs):
        choice = inner(obs)
        sel = obs.get("select")
        if sel is not None and len(choice) == 1:
            opts = sel.get("option") or []
            max_c = int(sel.get("maxCount") or 0)
            if max_c == 1 and len(opts) > 1 and sel.get("type") == 0:
                try:
                    sf = state_features(obs)
                    of_all = np.stack([option_features(o, obs, sel) for o in opts]).astype(
                        np.float32
                    )
                    per_game_samples.append(
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
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "sweep" / "bc_v6_wins_only.npz")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    opps = [
        rule_based_agent.agent,
        rule_based_dragapult.agent,
        rule_based_iono.agent,
        rule_based_abomasnow.agent,
        rule_based_crustle_dashimaki.agent,
    ]

    samples_kept: list = []
    n_wins = n_losses = n_draws = 0
    random.seed(0)
    t0 = time.monotonic()
    for g in range(args.games):
        opp = random.choice(opps)
        per_game: list = []
        v6_logging = make_logging_v6_agent(per_game)
        env = make("cabt")
        v6_seat = g % 2
        seats = [v6_logging, opp] if v6_seat == 0 else [opp, v6_logging]
        env.run(seats)
        # Inspect V6's final reward.
        try:
            v6_reward = env.steps[-1][v6_seat].reward
        except Exception:
            v6_reward = 0
        if v6_reward == 1:
            samples_kept.extend(per_game)
            n_wins += 1
        elif v6_reward == -1:
            n_losses += 1
        else:
            n_draws += 1
        if (g + 1) % 50 == 0:
            print(
                f"  game {g + 1}/{args.games}  wins={n_wins} losses={n_losses} "
                f"draws={n_draws}  samples kept: {len(samples_kept)} "
                f"[{time.monotonic() - t0:.0f}s]"
            )

    print(
        f"\nKept {len(samples_kept)} BC samples from "
        f"{n_wins}/{args.games} winning games "
        f"(losses={n_losses}, draws={n_draws})"
    )

    if not samples_kept:
        print("no winning samples — aborting save", file=sys.stderr)
        return 1

    sf_all = np.stack([s["sf"] for s in samples_kept]).astype(np.float32)
    picked_all = np.array([s["picked"] for s in samples_kept], dtype=np.int32)
    n_opts_all = np.array([s["n_opts"] for s in samples_kept], dtype=np.int32)
    of_flat_list = [s["of_all"] for s in samples_kept]
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
    print(f"  sf: {sf_all.shape}, of_flat: {of_flat.shape}, picked: {picked_all.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
