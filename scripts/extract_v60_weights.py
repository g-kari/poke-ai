"""Extract V60 .pt weights into a torch-free .npz file.

Used at build time (= make_submission_v60.sh) to produce a numpy-only
checkpoint that main_v60 can load without torch present on Kaggle.

Usage:
    scripts/run.sh python3 scripts/extract_v60_weights.py \\
        --pt train/mlp_policy_v60_ext3.pt --out train/mlp_policy_v60_ext3.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    ckpt = torch.load(args.pt, map_location="cpu", weights_only=True)
    arrays: dict[str, np.ndarray] = {
        k: v.detach().cpu().numpy().astype(np.float32) for k, v in ckpt["state_dict"].items()
    }
    arrays["__b_order__"] = np.asarray(ckpt.get("b_order", 2.0), dtype=np.float32)
    np.savez(args.out, **arrays)
    print(f"wrote {args.out} ({len(arrays)} arrays)")
    print(
        f"  pi layers: {sorted(int(k.split('.')[1]) for k in arrays if k.startswith('pi.') and k.endswith('.weight'))}"
    )
    print(
        f"  v layers: {sorted(int(k.split('.')[1]) for k in arrays if k.startswith('v.') and k.endswith('.weight'))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
