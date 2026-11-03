"""The number without which no result is interpretable: a small Transformer.

Same corpus (LeCarnet), same tokenizer, same budget, same seed, same sequence
length as the text run of the colony. It is a conventional 2-layer, 128-wide,
4-head decoder (~2 M parameters, embeddings tied), trained with the same
AdamW/step budget. Its validation perplexity is written to
``baseline_ppl.txt``; the colony has to be read against that number, not in a
vacuum.

    python scripts/train_baseline.py --steps 10000
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mycelia.config import load_config  # noqa: E402
from mycelia.train import get_text_dataset  # noqa: E402


class Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        causal = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attn(x, x, x, attn_mask=causal, need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.mlp(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int, dim: int = 128, heads: int = 4, layers: int = 2, seq: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(seq, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight  # tied: keeps the budget ~2 M

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed(tokens) + self.position(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def main() -> None:
    parser = argparse.ArgumentParser(description="train the transformer baseline")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=200)
    parser.add_argument("--out", type=Path, default=ROOT / "baseline_ppl.txt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_text_dataset(args.seq_len)
    vocab = dataset.tokenizer.get_vocab_size()
    model = TinyTransformer(vocab, seq=args.seq_len).to(device)
    print(f"baseline: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters | {device}")
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)

    started = time.time()
    for step in range(1, args.steps + 1):
        x, y = dataset.sample_batch(args.batch_size, generator=generator)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step % args.report_every == 0:
            print(
                f"step {step:6d} | perte {loss.item():.4f} | ppl {math.exp(min(loss.item(), 20)):8.2f} "
                f"| {(time.time() - started) / step * 1000:.0f} ms/step"
            )

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in dataset.windows(8, "val"):
            logits = model(x.to(device))
            total += float(F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1).to(device)))
            count += 1
    ppl = math.exp(min(total / max(1, count), 20))
    args.out.write_text(
        f"transformer baseline | {sum(p.numel() for p in model.parameters())} params | "
        f"steps {args.steps} | seed {args.seed} | validation perplexity {ppl:.3f}\n",
        encoding="utf-8",
    )
    print(f"validation perplexity: {ppl:.3f} → {args.out}")


if __name__ == "__main__":
    main()
