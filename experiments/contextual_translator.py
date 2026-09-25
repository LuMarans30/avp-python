#!/usr/bin/env python3
"""Collect contextual embeddings from two GGUF models and train a translator.

Phase 1 (optional) computes a last-token hidden state for each text with both
models and caches them. Phase 2 trains an MLP ``D_src -> D_tgt`` and reports
held-out cosine / top-1.

Corpus format: a JSON list of strings. A small wiki corpus can be fetched from
the HF datasets-server, e.g.:

    curl 'https://datasets-server.huggingface.co/rows?dataset=Salesforce/wikitext\
    &config=wikitext-2-raw-v1&split=train&offset=0&length=100'

Example:
    python experiments/contextual_translator.py \
      --corpus /tmp/corpus.json --ling /models/Ling.gguf --gemma /models/Gemma.gguf \
      --embeddings /tmp/ctx_embeddings.npz --limit 300 --epochs 300
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch import nn


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


def top1(projected: torch.Tensor, candidates: torch.Tensor) -> float:
    pred = (projected @ candidates.T).argmax(-1)
    return (pred == torch.arange(len(projected))).float().mean().item()


def collect(corpus: list[str], ling_path: str, gemma_path: str, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Last-token hidden state of each text under both models."""
    from avp import LlamaCppConnector
    from avp.types import OutputType

    ling = LlamaCppConnector.from_pretrained(ling_path, n_ctx=512, n_gpu_layers=0)
    gemma = LlamaCppConnector.from_pretrained(gemma_path, n_ctx=512, n_gpu_layers=0)
    src, tgt = [], []
    start = time.time()
    for i, text in enumerate(corpus[:limit]):
        src.append(ling.think(text, steps=0, output=OutputType.HIDDEN_STATE).last_hidden_state[0])
        tgt.append(gemma.think(text, steps=0, output=OutputType.HIDDEN_STATE).last_hidden_state[0])
        if i % 50 == 0:
            print(f"  collected {i}/{limit} ({time.time() - start:.0f}s)", flush=True)
    return np.stack(src).astype(np.float32), np.stack(tgt).astype(np.float32)


def train(
    x: np.ndarray, y: np.ndarray, *, epochs: int, hidden: int, seed: int = 0
) -> tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    xs = torch.tensor(x, dtype=torch.float32)
    ys = torch.tensor(y, dtype=torch.float32)
    n = len(xs)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n) :]
    mu_s, sd_s = xs[tr].mean(0), xs[tr].std(0) + 1e-6
    mu_t, sd_t = ys[tr].mean(0), ys[tr].std(0) + 1e-6
    xn, yn = (xs - mu_s) / sd_s, (ys - mu_t) / sd_t
    x_tr, y_tr, x_te, y_te = xn[tr], yn[tr], xn[te], yn[te]

    model = nn.Sequential(
        nn.Linear(x.shape[1], hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        nn.Linear(hidden, y.shape[1]),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for epoch in range(epochs):
        model.train()
        loss = (1 - cosine(model(x_tr), y_tr)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                print(f"  ep {epoch:3d} val_cos {cosine(model(x_te), y_te).mean().item():.4f}")
    return model, x_te, y_te


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", help="JSON list of texts (required to collect)")
    ap.add_argument("--ling", help="Source GGUF path (required to collect)")
    ap.add_argument("--gemma", help="Target GGUF path (required to collect)")
    ap.add_argument("--embeddings", required=True, help="Cached .npz; collected if absent")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=512)
    args = ap.parse_args()

    import os

    if os.path.exists(args.embeddings):
        d = np.load(args.embeddings)
        x, y = d["S"], d["T"]
    else:
        if not (args.corpus and args.ling and args.gemma):
            ap.error("--corpus/--ling/--gemma are required when --embeddings is absent")
        with open(args.corpus) as f:
            corpus = json.load(f)
        x, y = collect(corpus, args.ling, args.gemma, args.limit)
        np.savez(args.embeddings, S=x, T=y)
    print(f"embeddings: S={x.shape} T={y.shape}")

    # linear baseline
    model, x_te, y_te = train(x, y, epochs=args.epochs, hidden=args.hidden)
    model.eval()
    with torch.no_grad():
        mlp = model(x_te)
    print(f"MLP held-out cos={cosine(mlp, y_te).mean().item():.4f} top1={top1(mlp, y_te):.4f}")
    print(f"random top-1 = {1 / len(y_te):.4f}")


if __name__ == "__main__":
    main()
