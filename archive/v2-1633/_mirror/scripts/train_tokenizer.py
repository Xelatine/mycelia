#!/usr/bin/env python
"""Train the dedicated French BPE tokenizer (10 000 tokens by default).

    python scripts/train_tokenizer.py --vocab 10000

The output lands in the data cache as ``tokenizer-fr-10k.json`` and is reused by
training, generation and the cockpit.  It is never an English tokenizer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mycelia.config import load_config  # noqa: E402
from mycelia.data.corpus import resolve_corpus  # noqa: E402
from mycelia.data.tokenizer import FrenchBPE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the French tokenizer")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--vocab", type=int, default=10000)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-chars", type=int, default=8_000_000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.corpus:
        cfg.data.corpus = args.corpus
    corpus = resolve_corpus(cfg.data, ROOT)
    text = corpus.path.read_text(encoding="utf-8", errors="replace")[: args.max_chars]
    docs = [line for line in text.split("\n") if len(line.strip()) > 20]
    print(f"[tokenizer] corpus {corpus.path.name} ({corpus.source}, "
          f"{corpus.chars:,} chars, {len(docs):,} documents)")

    out = Path(args.out) if args.out else (
        ROOT / cfg.data.cache_dir / f"tokenizer-fr-{args.vocab // 1000}k.json")
    tokenizer = FrenchBPE.train(docs, vocab_size=args.vocab, path=out)
    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "kind": "bpe-fr", "vocab_size": tokenizer.vocab_size,
        "specials": ["<pad>", "<bos>", "<eos>", "<unk>"],
        "trained_on": corpus.source, "docs": len(docs),
        "chars": min(len(text), args.max_chars),
    }, indent=2), encoding="utf-8")

    probe = "Il était une fois un petit garçon qui aimait les histoires."
    ids = tokenizer.encode(probe)
    print(f"[tokenizer] {tokenizer.vocab_size} tokens → {out}")
    print(f"[tokenizer] round-trip ok: {tokenizer.decode(ids) == probe} "
          f"({len(ids)} tokens for {len(probe)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
