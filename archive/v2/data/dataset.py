"""Corpus -> token stream -> batches, with an offline token cache.

The corpus is encoded once and stored as ``int32`` so restarts do not re-encode
tens of megabytes.  Documents are separated by ``<eos>`` so the brain learns
where a story ends.  The last 5% of documents form the validation split, which
is what the tier controller watches.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from mycelia.data.tokenizer import BaseTokenizer, BOS, EOS


@dataclass
class Tokens:
    train: np.ndarray
    val: np.ndarray
    vocab_size: int

    @property
    def total(self) -> int:
        return int(self.train.size + self.val.size)


class TokenStore:
    def __init__(self, tokens: Tokens):
        self.tokens = tokens

    # ----------------------------------------------------------------- io
    @classmethod
    def build(cls, corpus_path: Path, tokenizer: BaseTokenizer, cache_dir: Path,
              max_chars: Optional[int] = None) -> "TokenStore":
        stat = Path(corpus_path).stat()
        key = hashlib.sha1(
            f"{corpus_path.name}:{stat.st_size}:{tokenizer.vocab_size}:{max_chars}".encode()
        ).hexdigest()[:10]
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_path = cache_dir / f"tokens-fr-train-{key}.npy"
        val_path = cache_dir / f"tokens-fr-val-{key}.npy"
        if train_path.exists() and val_path.exists():
            train = np.load(train_path)
            val = np.load(val_path)
            return cls(Tokens(train, val, tokenizer.vocab_size))

        text = Path(corpus_path).read_text(encoding="utf-8", errors="replace")
        if max_chars:
            text = text[: int(max_chars)]
        docs = [d.strip() for d in text.split("\n") if len(d.strip()) > 20]
        cut = max(1, int(len(docs) * 0.95))
        train_ids: List[int] = []
        for doc in docs[:cut]:
            train_ids.append(BOS)
            train_ids.extend(tokenizer.encode(doc)[:512])
            train_ids.append(EOS)
        val_ids: List[int] = []
        for doc in docs[cut:]:
            val_ids.append(BOS)
            val_ids.extend(tokenizer.encode(doc)[:512])
            val_ids.append(EOS)
        train = np.asarray(train_ids, dtype=np.int32)
        val = np.asarray(val_ids or train_ids[:2048], dtype=np.int32)
        np.save(train_path, train)
        np.save(val_path, val)
        return cls(Tokens(train, val, tokenizer.vocab_size))

    # ------------------------------------------------------------- batches
    def batch(self, batch_size: int, ctx_len: int, rng: np.random.Generator,
              split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.tokens.train if split == "train" else self.tokens.val
        n = data.size
        span = min(int(ctx_len), max(2, n - 2))
        starts = rng.integers(0, max(1, n - span - 1), size=int(batch_size))
        x = np.stack([data[s: s + span] for s in starts]).astype(np.int64)
        y = np.stack([data[s + 1: s + 1 + span] for s in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def fixed_batches(self, batch_size: int, ctx_len: int, count: int,
                      seed: int = 0) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Deterministic validation batches (same metric across restarts)."""
        rng = np.random.default_rng(seed)
        return [self.batch(batch_size, ctx_len, rng, split="val")
                for _ in range(int(count))]

    def info(self) -> Dict[str, int]:
        return {"train_tokens": int(self.tokens.train.size),
                "val_tokens": int(self.tokens.val.size),
                "vocab_size": int(self.tokens.vocab_size)}
