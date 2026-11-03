"""Data: the delayed-copy toy task, then real French stories.

Two datasets live here and they never mix.

The toy task exists because every architecture decision must be provable in
seconds. It is a **delayed copy** (D2): ``y[t] = x[t-1]``. The old "next token
of random integers" was impossible (nothing to learn) and asked nothing of the
state; the delayed copy forces information to cross the graph *and* to survive
from one token position to the next. A brain that only looks at the token it is
reading cannot solve it.

The text task is the real goal: the same Brain, fed French children stories,
must predict the next token. There is no shift in the dataset because the brain
already reads before injecting: ``logits[t]`` is computed from the state built
by tokens ``0..t-1``, so comparing it with token ``t`` *is* next-token
prediction.
"""

from __future__ import annotations

from pathlib import Path

import torch

VOCAB = 100
FIRST_TOKEN = 2  # 0 and 1 stay free: they are reserved control slots
LAST_TOKEN = 99

DEFAULT_SEQ_LEN = 128
ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = ROOT / "data" / "tokenizer.json"
CORPUS_PATH = ROOT / "data" / "corpus_fr.txt"


def clean_text(text: str) -> str:
    """Make a decoded sample printable, at display time only (C8).

    A byte-level BPE can end on an incomplete UTF-8 sequence, so ``decode``
    sometimes yields U+FFFD or a stray control character. That is a display
    artefact, not a defect of the brain: nothing is rewritten in the model, the
    characters are simply not shown.
    """
    import unicodedata

    kept = []
    for char in text:
        if char == "\ufffd":
            continue
        if char in "\n\t ":
            kept.append(char)
            continue
        if unicodedata.category(char).startswith("C"):
            continue
        kept.append(char)
    return "".join(kept)


def get_batch(
    batch_size: int, seq_len: int = 10, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x, y) of shape [batch, seq_len] as int64 token ids.

    ``y[t] = x[t-1]``: the target is the token that was just read. Position 0
    has no predecessor, so it repeats ``x[0]``; training ignores it anyway.
    """
    x = torch.randint(FIRST_TOKEN, LAST_TOKEN + 1, (batch_size, seq_len), generator=generator)
    y = torch.empty_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1:] = x[:, :-1]
    return x, y


def load_tokenizer(path: Path | str = TOKENIZER_PATH):
    """Load the BPE model trained by scripts/train_tokenizer.py."""
    from tokenizers import Tokenizer

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no tokenizer at {path}: run `python scripts/train_tokenizer.py` first"
        )
    return Tokenizer.from_file(str(path))


class TextDataset:
    """The corpus as one continuous stream, with 5 % held out for validation.

    Why one flat stream and not one sequence per story: the brain has no notion
    of document boundaries, it only sees tokens arriving. Cutting the stream
    into fixed windows gives every step the same shape and lets a story start in
    the middle of another one, which is exactly the training signal a small
    language model needs (it must handle any context, not only clean openings).

    Why 5 % held out: growth is triggered by the *validation* curve (F8). A
    manager that reads the training loss would keep growing on noise.
    """

    def __init__(
        self,
        corpus_path: Path | str = CORPUS_PATH,
        tokenizer=None,
        seq_len: int = DEFAULT_SEQ_LEN,
        val_fraction: float = 0.05,
        max_lines: int | None = None,
    ) -> None:
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"no corpus at {corpus_path}: run `python scripts/train_tokenizer.py` first"
            )
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer()
        self.seq_len = int(seq_len)
        text = corpus_path.read_text(encoding="utf-8")
        if max_lines is not None:
            text = "\n".join(text.splitlines()[: int(max_lines)])

        ids = self.tokenizer.encode(text).ids
        stream = torch.tensor(ids, dtype=torch.long)
        cut = int(stream.numel() * (1.0 - float(val_fraction)))
        # One token of context overlap at the boundary: a window needs to start
        # before the split to read the first held-out token in context.
        self.train = stream[:cut]
        self.val = stream[max(0, cut - self.seq_len) :]
        self.n_train_windows = max(0, self.train.numel() // self.seq_len)
        self.n_val_windows = max(0, self.val.numel() // self.seq_len)

    def __len__(self) -> int:
        return self.n_train_windows

    def sample_batch(
        self,
        batch_size: int,
        split: str = "train",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random windows from the stream, as [batch, seq_len] pairs."""
        stream, windows = (self.train, self.n_train_windows) if split == "train" else (
            self.val,
            self.n_val_windows,
        )
        if windows == 0:
            raise ValueError(f"the {split} split is shorter than one window")
        starts = torch.randint(0, windows, (batch_size,), generator=generator) * self.seq_len
        index = starts.unsqueeze(1) + torch.arange(self.seq_len).unsqueeze(0)
        x = stream[index]
        # The target is the window itself: the state already carries the past.
        return x, x

    def windows(self, batch_size: int, split: str = "val"):
        """Deterministic windows for validation (same batches at every measure)."""
        stream, windows = (self.train, self.n_train_windows) if split == "train" else (
            self.val,
            self.n_val_windows,
        )
        if windows == 0:
            raise ValueError(f"the {split} split is shorter than one window")
        for start in range(0, min(windows, batch_size) * self.seq_len, self.seq_len):
            index = start + torch.arange(self.seq_len)
            x = stream[index].unsqueeze(0)
            yield x, x


_CACHE: dict[int, TextDataset] = {}


def get_text_batch(
    batch_size: int,
    seq_len: int = DEFAULT_SEQ_LEN,
    dataset: TextDataset | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a random French batch. The dataset is built once and reused."""
    if dataset is None:
        dataset = _CACHE.get(seq_len)
        if dataset is None:
            dataset = TextDataset(seq_len=seq_len)
            _CACHE[seq_len] = dataset
    return dataset.sample_batch(batch_size, generator=generator)
