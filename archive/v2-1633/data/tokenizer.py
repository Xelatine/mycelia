"""Dedicated French BPE tokenizer (10 000 tokens), loaded or trained locally.

The tokenizer is a Hugging Face ``tokenizers`` byte-level BPE trained **on the
French corpus**, never an English tokenizer reused as-is.  It is serialised as
plain JSON so a run is reproducible offline; ``scripts/train_tokenizer.py``
rebuilds it from scratch.

Special ids are fixed everywhere in the project:

====  =========
0     ``<pad>``
1     ``<bos>``
2     ``<eos>``
3     ``<unk>``
====  =========
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class BaseTokenizer:
    """Minimal interface the rest of the project relies on."""

    vocab_size: int = 0

    def encode(self, text: str) -> List[int]:  # pragma: no cover - interface
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:  # pragma: no cover
        raise NotImplementedError


class FrenchBPE(BaseTokenizer):
    def __init__(self, backend, vocab_size: int):
        self._tok = backend
        self.vocab_size = int(vocab_size)

    # ----------------------------------------------------------------- io
    @classmethod
    def load(cls, path: str | Path) -> "FrenchBPE":
        from tokenizers import Tokenizer

        backend = Tokenizer.from_file(str(path))
        return cls(backend, backend.get_vocab_size())

    def save(self, path: str | Path) -> None:
        self._tok.save(str(path))

    # ------------------------------------------------------------- encode
    def encode(self, text: str) -> List[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: Sequence[int]) -> str:
        clean = [int(i) for i in ids if int(i) >= len(SPECIALS)]
        if not clean:
            return ""
        return self._tok.decode(clean, skip_special_tokens=False)

    def id_to_token(self, idx: int) -> str:
        return self._tok.id_to_token(int(idx)) or ""

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)

    # --------------------------------------------------------------- train
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 10_000,
              path: Optional[str | Path] = None) -> "FrenchBPE":
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        backend = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True,
                                       continuing_subword_prefix="",
                                       end_of_word_suffix=""))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                        use_regex=True)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=int(vocab_size), special_tokens=SPECIALS,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        backend.train_from_iterator(texts, trainer=trainer)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            backend.save(str(path))
        return cls(backend, backend.get_vocab_size())


def resolve_tokenizer(corpus_path: Path, vocab_size: int, cache_dir: Path,
                      retrain: bool = False,
                      max_chars: int = 8_000_000) -> tuple[BaseTokenizer, str]:
    """Load the cached French tokenizer or train a new one on the corpus."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    preferred = cache_dir / f"tokenizer-fr-{vocab_size // 1000}k.json"
    if not retrain and preferred.exists():
        try:
            return FrenchBPE.load(preferred), f"cached {preferred.name}"
        except Exception:
            pass
    generic = cache_dir / f"tokenizer-fr-{vocab_size}.json"
    if not retrain and generic.exists():
        try:
            return FrenchBPE.load(generic), f"cached {generic.name}"
        except Exception:
            pass
    text = corpus_path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    docs = [line for line in text.split("\n") if len(line) > 20]
    tok = FrenchBPE.train(docs, vocab_size=vocab_size, path=preferred)
    return tok, f"trained on {corpus_path.name} ({len(docs)} docs)"


def tokenizer_info(path: Path) -> dict:
    meta = path.with_suffix(".meta.json")
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return {}
