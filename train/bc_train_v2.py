"""Vectorized BC trainer — 10x faster than bc_train.py.

bc_train.py was Python-loop heavy (n_batches * batch_size forward passes
per epoch). For features_v60 with 8-d median n_opts and tiny networks,
GPU was idle most of the time.

bc_train_v2 pads each minibatch to max_n_opts then does ONE forward pass
per minibatch via a (B, max_n_opts, in_dim) → (B, max_n_opts) tensor. The
masked cross-entropy ignores padding logits with -inf. Empirically ~10x
faster (500 epoch in ~5 min vs ~45 min).

Usage:
    scripts/run.sh python3 -m train.bc_train_v2 \\
        --dataset data/sweep/bc_v6_dataset.npz \\
        --out train/mlp_policy_v60_bc_v2.pt \\
        --epochs 1000 --lr 1e-3 --batch-size 128 --log-every 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from train.features_v60 import OPTION_DIM, STATE_DIM
from train.mlp_policy_v60 import MlpPolicyV60


def load_dataset(path: Path) -> dict:
    d = np.load(path)
    sf = d["sf"].astype(np.float32)
    of_flat = d["of_flat"].astype(np.float32)
    of_indptr = d["of_indptr"].astype(np.int64)
    picked = d["picked"].astype(np.int64)
    n_opts = d["n_opts"].astype(np.int64)
    assert sf.shape[1] == STATE_DIM
    assert of_flat.shape[1] == OPTION_DIM
    return {
        "sf": sf,
        "of_flat": of_flat,
        "of_indptr": of_indptr,
        "picked": picked,
        "n_opts": n_opts,
    }


def make_minibatch_tensors(
    rng: np.random.Generator, ds: dict, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample batch_size items and pad option dim to local max.

    Returns:
        sf: (B, STATE_DIM)
        of: (B, max_n, OPTION_DIM) — padded with zeros
        mask: (B, max_n) — 1 for valid options, 0 for padding
        picked: (B,) target index per row
    """
    n = len(ds["picked"])
    idx = rng.choice(n, size=batch_size, replace=False)
    ks = ds["n_opts"][idx]
    max_n = int(ks.max())

    sf = ds["sf"][idx]
    of_pad = np.zeros((batch_size, max_n, OPTION_DIM), dtype=np.float32)
    mask = np.zeros((batch_size, max_n), dtype=np.float32)
    for j, i in enumerate(idx):
        start = ds["of_indptr"][i]
        end = ds["of_indptr"][i + 1]
        k = end - start
        of_pad[j, :k] = ds["of_flat"][start:end]
        mask[j, :k] = 1.0

    return (
        torch.from_numpy(sf).to(device),
        torch.from_numpy(of_pad).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(ds["picked"][idx]).to(device),
    )


def epoch_loss(
    policy: MlpPolicyV60, ds: dict, rng: np.random.Generator, batch_size: int, optimizer
) -> tuple[float, float]:
    n = len(ds["picked"])
    losses, hits = [], []
    n_batches = max(1, n // batch_size)
    device = policy.device
    for _ in range(n_batches):
        sf, of, mask, picked = make_minibatch_tensors(rng, ds, batch_size, device)
        b, max_n, _ = of.shape
        sf_exp = sf.unsqueeze(1).expand(b, max_n, STATE_DIM)
        x = torch.cat([sf_exp, of], dim=-1).reshape(b * max_n, -1)
        raw = policy.pi(x).squeeze(-1).reshape(b, max_n)
        # mask out padded positions
        logits = raw.masked_fill(mask == 0, float("-inf"))
        loss = F.cross_entropy(logits, picked)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        hits.append(float((logits.argmax(dim=-1) == picked).float().mean().item()))
    return float(np.mean(losses)), float(np.mean(hits))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warm-start", type=Path, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--also-train-value", action="store_true", help="train v head on same loss")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ds = load_dataset(args.dataset)
    n = len(ds["picked"])
    print(
        f"loaded {n} samples, n_opts median={int(np.median(ds['n_opts']))}, "
        f"max={int(ds['n_opts'].max())}",
        flush=True,
    )

    if args.warm_start and args.warm_start.exists():
        policy = MlpPolicyV60.load(str(args.warm_start))
        print(f"warm-start from {args.warm_start}", flush=True)
    else:
        policy = MlpPolicyV60()
        print(f"fresh policy on device={policy.device}", flush=True)

    optimizer = optim.Adam(policy.pi.parameters(), lr=args.lr)

    t0 = time.monotonic()
    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        loss, acc = epoch_loss(policy, ds, rng, args.batch_size, optimizer)
        if acc > best_acc:
            best_acc = acc
        if ep % args.log_every == 0 or ep == 1:
            dt = time.monotonic() - t0
            print(
                f"ep {ep:5d}/{args.epochs} loss={loss:.4f} acc={acc:.3f} "
                f"best={best_acc:.3f} [{dt:.0f}s]",
                flush=True,
            )

    policy.save(str(args.out))
    print(f"\nsaved {args.out} (best acc: {best_acc:.3f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
