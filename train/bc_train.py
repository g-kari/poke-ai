"""Behavioral Cloning trainer for MlpPolicyV60 from collected V6 decisions.

Reads a ragged .npz produced by scripts/collect_bc_dataset.py — each sample is
(state_features, option_features for n_opts options, picked index) — and trains
the pi head with cross-entropy loss over the option logits.

Why BC: V6 is our LB best (~921-926) but is a hand-tuned rule-based agent that
can't be deployed across decks. By cloning its (state, picks) into a generic
MLP policy operating on v60 features, we test whether the supervised signal
is strong enough to recover V6-level decision quality in a learnable form.

Usage:
    scripts/run.sh python3 -m train.bc_train \\
        --dataset data/sweep/bc_v6_dataset.npz \\
        --out train/mlp_policy_v60_bc.pt \\
        --epochs 2000 --lr 1e-3 --batch-size 64
"""

from __future__ import annotations

import argparse
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
    assert sf.shape[1] == STATE_DIM, f"sf dim {sf.shape[1]} != {STATE_DIM}"
    assert of_flat.shape[1] == OPTION_DIM, f"of dim {of_flat.shape[1]} != {OPTION_DIM}"
    return {
        "sf": sf,
        "of_flat": of_flat,
        "of_indptr": of_indptr,
        "picked": picked,
        "n_opts": n_opts,
    }


def sample_minibatch(rng: np.random.Generator, ds: dict, batch_size: int) -> list[dict]:
    n = len(ds["picked"])
    idx = rng.choice(n, size=batch_size, replace=False)
    samples = []
    for i in idx:
        start = ds["of_indptr"][i]
        end = ds["of_indptr"][i + 1]
        samples.append(
            {
                "sf": ds["sf"][i],
                "of": ds["of_flat"][start:end],
                "picked": int(ds["picked"][i]),
            }
        )
    return samples


def forward_logits(policy: MlpPolicyV60, sf: torch.Tensor, of: torch.Tensor) -> torch.Tensor:
    """Run pi head with (sf broadcast to n_opts) + of concatenated."""
    n = of.shape[0]
    x = torch.cat([sf.unsqueeze(0).expand(n, -1), of], dim=1)
    return policy.pi(x).squeeze(-1)


def epoch_loss(
    policy: MlpPolicyV60, ds: dict, rng: np.random.Generator, batch_size: int, optimizer
) -> tuple[float, float]:
    """One epoch = N/batch_size minibatches over the dataset."""
    n = len(ds["picked"])
    losses, hits = [], []
    n_batches = max(1, n // batch_size)
    for _ in range(n_batches):
        samples = sample_minibatch(rng, ds, batch_size)
        loss_terms = []
        for s in samples:
            sf = torch.from_numpy(s["sf"]).to(policy.device)
            of = torch.from_numpy(s["of"]).to(policy.device)
            logits = forward_logits(policy, sf, of)
            target = torch.tensor([s["picked"]], device=policy.device)
            loss_terms.append(F.cross_entropy(logits.unsqueeze(0), target))
            hits.append(int(logits.argmax().item() == s["picked"]))
        loss = torch.stack(loss_terms).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)), float(np.mean(hits))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warm-start", type=Path, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ds = load_dataset(args.dataset)
    n = len(ds["picked"])
    print(f"loaded {n} samples from {args.dataset}")
    print(
        f"  sf={ds['sf'].shape}, of_flat={ds['of_flat'].shape}, "
        f"n_opts median={int(np.median(ds['n_opts']))}, max={int(ds['n_opts'].max())}"
    )

    if args.warm_start and args.warm_start.exists():
        policy = MlpPolicyV60.load(str(args.warm_start))
        print(f"warm-start from {args.warm_start}")
    else:
        policy = MlpPolicyV60()
        print(f"fresh policy on device={policy.device}")

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
                f"best={best_acc:.3f} [{dt:.0f}s]"
            )

    policy.save(str(args.out))
    print(f"\nsaved {args.out} (best acc: {best_acc:.3f})")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
