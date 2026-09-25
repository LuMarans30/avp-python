#!/usr/bin/env python3
"""Train a small MLP token-embedding translator and evaluate held-out metrics.

Reads a cached GGUF projection map (``avp.rosetta.gguf_map``) containing paired
shared-token embeddings ``[N, D_src]`` / ``[N, D_tgt]``, trains an MLP with a
cosine loss plus an optional vector-space-preservation (VSP) penalty, and
compares held-out cosine / top-1 against linear and vocab-overlap baselines.

Example:
    python experiments/token_translator.py --map ~/.avp/gguf_maps/abc_def.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


def top1(projected: torch.Tensor, targets: torch.Tensor, candidates: torch.Tensor) -> float:
    """Fraction of projections whose nearest candidate is the correct target."""
    pred = (projected @ candidates.T).argmax(-1)
    return (pred == torch.arange(len(projected))).float().mean().item()


class Translator(nn.Module):
    def __init__(self, d_src: int, d_tgt: int, hidden: int = 2048) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_src, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, d_tgt),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(
    x: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    epochs: int,
    hidden: int,
    batch_size: int,
    vsp: float,
    lr: float = 1e-3,
) -> Translator:
    model = Translator(x.shape[1], y.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for epoch in range(epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        for i in range(0, len(order), batch_size):
            b = order[i : i + batch_size]
            xb, yb = x[b], y[b]
            pred = model(xb)
            loss = (1 - cosine(pred, yb)).mean()
            if vsp > 0:
                pn = F.normalize(pred, dim=-1)
                yn = F.normalize(yb, dim=-1)
                loss = loss + vsp * ((pn @ pn.T) - (yn @ yn.T)).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True, help="Path to a gguf_maps *.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--vsp", type=float, default=0.0)
    ap.add_argument("--split", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = np.load(args.map)
    x = torch.tensor(data["src_shared"], dtype=torch.float32)
    y = torch.tensor(data["tgt_shared"], dtype=torch.float32)
    n = x.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    train_idx, test_idx = perm[: int(args.split * n)], perm[int(args.split * n) :]
    x_tr, y_tr, x_te, y_te = x[train_idx], y[train_idx], x[test_idx], y[test_idx]
    print(f"pairs={n} D_src={x.shape[1]} D_tgt={y.shape[1]} test={len(test_idx)}")

    # --- baselines ---
    a = torch.cat([x_tr, torch.ones(len(train_idx), 1)], 1)
    ridge = 1e-3 * torch.eye(x.shape[1] + 1)
    sol = torch.linalg.solve(a.T @ a + ridge, a.T @ y_tr)
    lin = torch.cat([x_te, torch.ones(len(test_idx), 1)], 1) @ sol
    vo = torch.softmax(x_te @ x_tr.T, dim=-1) @ y_tr
    cand = y_te  # top-1 among held-out targets (same convention as the design note)
    print(f"linear        cos={cosine(lin, y_te).mean().item():.4f} top1={top1(lin, y_te, cand):.4f}")
    print(f"vocab_overlap cos={cosine(vo, y_te).mean().item():.4f} top1={top1(vo, y_te, cand):.4f}")

    model = train(
        x, y, train_idx, test_idx,
        epochs=args.epochs, hidden=args.hidden, batch_size=args.batch_size, vsp=args.vsp,
    )
    model.eval()
    with torch.no_grad():
        mlp = model(x_te)
    print(f"MLP(vsp={args.vsp}) cos={cosine(mlp, y_te).mean().item():.4f} top1={top1(mlp, y_te, cand):.4f}")


if __name__ == "__main__":
    main()
