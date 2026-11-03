"""French corpus resolution with an explicit, documented fallback chain.

Order:

1. an explicit ``data.corpus`` path;
2. a cached French corpus (``.cache/corpus-fr-*.txt``), so a machine that has
   already prepared data works fully offline;
3. Hugging Face TinyStories-fr (needs ``datasets`` + network);
4. Hugging Face TinyStories (English) — the documented fallback when no French
   corpus is reachable;
5. a small bundled French text so unit tests and demos never explode.

The chosen source and its size are always reported, because the user's success
criterion is French output and the corpus is what makes it possible.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from mycelia.config import DataConfig


@dataclass
class Corpus:
    path: Path
    source: str
    chars: int
    docs: int

    def text(self) -> str:
        return self.path.read_text(encoding="utf-8", errors="replace")


BUNDLED_FR = """\
Il était une fois une petite fille qui aimait courir dans la forêt.
Chaque matin, elle prenait son panier et partait cueillir des fraises.
Un jour, elle rencontra un renard qui avait perdu sa famille.
Elle lui parla doucement et le renard la suivit jusqu'à la maison.
Sa mère prépara une soupe chaude et ils mangèrent ensemble près du feu.
Le lendemain, le renard retrouva ses petits au bord de la rivière.
Il était heureux et la petite fille sourit en rentrant chez elle.
"""


def _cache_candidates(cache: Path, max_chars: int) -> List[Path]:
    found = sorted(cache.glob("corpus-fr-*.txt"),
                   key=lambda p: p.stat().st_size, reverse=True)
    return [p for p in found if p.stat().st_size > 10_000]


def _write_cache(text: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    path = cache / f"corpus-fr-{digest}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _from_huggingface(name: str, max_chars: int) -> Optional[str]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    try:
        ds = load_dataset(name, split="train")
    except Exception as exc:  # network, gated dataset, ...
        print(f"[data] Hugging Face {name} unavailable: {exc}", file=sys.stderr)
        return None
    parts: List[str] = []
    total = 0
    for row in ds:
        text = row.get("text") or row.get("translation", {}).get("fr") or ""
        if not isinstance(text, str) or len(text) < 20:
            continue
        parts.append(text.strip())
        total += len(text)
        if total >= max_chars:
            break
    return "\n\n".join(parts) if parts else None


def resolve_corpus(cfg: DataConfig, root: Optional[Path] = None) -> Corpus:
    root = Path(root or ".")
    cache = root / cfg.cache_dir

    if cfg.corpus and cfg.corpus != "auto":
        path = Path(cfg.corpus)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"corpus not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        return Corpus(path, "explicit", len(text), text.count("\n"))

    for candidate in _cache_candidates(cache, cfg.max_chars):
        text = candidate.read_text(encoding="utf-8", errors="replace")
        if len(text) > 10_000:
            return Corpus(candidate, "cached-fr", len(text), text.count("\n"))

    text = _from_huggingface(cfg.hf_dataset, cfg.max_chars)
    source = "hf-fr"
    if text is None:
        text = _from_huggingface("tiny_stories", cfg.max_chars)
        source = "hf-en-fallback"
    if text is None:
        text = BUNDLED_FR * 200
        source = "bundled-fr"
    path = _write_cache(text, cache)
    return Corpus(path, source, len(text), text.count("\n"))
