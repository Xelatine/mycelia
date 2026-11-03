#!/usr/bin/env python
"""Dense baseline for comparison: a small GPT trained on the same French data.

    python scripts/train_baseline.py --config configs/demo.yaml --steps 2000

It is deliberately conventional (stacked self-attention, no graph, no
lifecycle) so the README can compare "same data, same steps" perplexities and
show what the emergent wiring buys — or fails to buy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mycelia.config import load_config  # noqa: E402
from mycelia.data.corpus import resolve_corpus  # noqa: E402
from mycelia.data.dataset import TokenStore  # noqa: E402
from mycelia.data.tokenizer import resolve_tokenizer  # noqa: E402
from mycelia.train.loop import pick_device, seed_everything  # noqa: E402


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ctx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))
        self.register_buffer("mask", torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        ctx = x.shape[1]
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:ctx, :ctx], need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class DenseGPT(nn.Module):
    def __init__(self, vocab: int, dim: int = 128, layers: int = 2,
                 heads: int = 4, ctx: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(ctx, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, ctx) for _ in range(layers)])
        self.ln = nn.LayerNorm(dim)
        self.vocab_size = vocab

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.emb(ids) + self.pos(pos).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.ln(x) @ self.emb.weight.t()


def main() -> int:
    parser = argparse.ArgumentParser(description="Dense baseline")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--out", default="runs/baseline")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, seed=args.seed)
    if args.seed is not None:
        cfg.seed = args.seed
    steps = int(args.steps or cfg.train.steps)
    seed_everything(cfg.seed)
    device = pick_device(cfg.train.device)

    corpus = resolve_corpus(cfg.data)
    tokenizer, _ = resolve_tokenizer(corpus.path, cfg.data.vocab_size,
                                    Path(cfg.data.cache_dir))
    store = TokenStore.build(corpus.path, tokenizer, Path(cfg.data.cache_dir),
                             cfg.data.max_chars)
    rng = np.random.default_rng(cfg.seed)

    model = DenseGPT(tokenizer.vocab_size, dim=args.dim, layers=args.layers,
                     ctx=cfg.train.ctx_len).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    print(f"[baseline] {params/1e6:.2f} M params on {device} for {steps} steps", flush=True)

    val = store.fixed_batches(cfg.train.batch_size, cfg.train.ctx_len, 4, seed=cfg.seed)
    best = float("inf")
    start = time.time()
    model.train()
    for step in range(1, steps + 1):
        xb, yb = store.batch(cfg.train.batch_size, cfg.train.ctx_len, rng)
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
        opt.step()
        if step % 100 == 0 or step == 1:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for vx, vy in val:
                    total += float(F.cross_entropy(
                        model(vx.to(device)).reshape(-1, tokenizer.vocab_size),
                        vy.to(device).reshape(-1)))
            model.train()
            vloss = total / max(1, len(val))
            best = min(best, vloss)
            print(f"[baseline {step:5d}] loss {float(loss):.3f} "
                  f"val {math.exp(vloss):7.2f} ppl ({time.time()-start:.0f}s)", flush=True)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": tokenizer.vocab_size,
                "dim": args.dim, "layers": args.layers,
                "step": steps, "val_loss": best}, out_dir / "checkpoint.pt")
    (out_dir / "state.json").write_text(json.dumps({
        "kind": "dense-baseline", "step": steps, "params": params,
        "val_loss": best, "val_ppl": math.exp(min(20, best)),
        "seconds": round(time.time() - start, 1),
        "dim": args.dim, "layers": args.layers,
    }, indent=2), encoding="utf-8")
    print(f"[baseline] done · best val ppl {math.exp(min(20, best)):.2f} "
          f"· {params/1e6:.2f} M params", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
