"""A/B benchmark two policy.npz checkpoints against random_agent.

Usage:
    scripts/run.sh python3 scripts/ab_bench.py <new.npz> <baseline.npz> [games_per_side]

Backs up the current train/policy.npz, swaps each candidate in, runs
selfplay_test.bench(N), and reports both win-loss-draw counts side by side.
Always restores the working tree's policy.npz at exit.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "train" / "policy.npz"


def run_bench(n: int) -> tuple[int, int, int]:
    """Run selfplay_test.bench(n) and parse its stdout."""
    out = subprocess.check_output(
        ["python3", str(ROOT / "selfplay_test.py"), str(n)],
        cwd=str(ROOT),
    ).decode()
    w_total = l_total = d = 0
    for raw in out.splitlines():
        line = raw.strip()
        if "agent as P0 vs random:" in line or "agent as P1 vs random:" in line:
            wins, losses = line.split(":")[1].strip().split("-")
            w_total += int(wins)
            l_total += int(losses)
        elif "draws:" in line:
            d = int(line.split(":")[1].strip())
    return w_total, l_total, d


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    new = Path(sys.argv[1]).resolve()
    base = Path(sys.argv[2]).resolve()
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    if not new.exists() or not base.exists():
        print(f"missing input: new={new.exists()} base={base.exists()}", file=sys.stderr)
        return 2

    # Save the live policy so we can put it back at exit.
    live_backup = LIVE.with_suffix(".npz.live-backup")
    shutil.copy(LIVE, live_backup)

    try:
        shutil.copy(new, LIVE)
        new_w, new_l, new_d = run_bench(n)
        shutil.copy(base, LIVE)
        base_w, base_l, base_d = run_bench(n)
    finally:
        shutil.move(live_backup, LIVE)

    total = 2 * n
    print(f"A/B vs random_agent, {total} games per policy:")
    print(f"  new      : {new_w}-{new_l}-{new_d}  ({new_w / total:.1%})")
    print(f"  baseline : {base_w}-{base_l}-{base_d}  ({base_w / total:.1%})")
    print(f"  delta    : {new_w - base_w:+d} wins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
