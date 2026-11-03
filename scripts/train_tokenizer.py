"""Build the French corpus and train the 10 000-token BPE on it.

Run from anywhere:

    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --max-chars 20000000   # quicker pass

Corpus: **MaxLSB/LeCarnet**, about two million small stories written natively
in French, with a deliberately restricted vocabulary (TinyStories applied to
French). A ~21 M-parameter model trained on it produces perfectly readable
French, so the colony has a realistic target. If the Hub is unreachable the
script keeps the local ``data/corpus_fr.txt`` (and says so) instead of silently
training on English.

Why a tokenizer trained on the corpus itself and not a generic 50 000-token
one: a small brain has a fixed budget of embedding rows, and 10 000 tokens
learned on these stories spend all of it on the words these stories actually
use. A generic tokenizer would waste most rows on code, English and rare
suffixes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DATASET_ID = "MaxLSB/LeCarnet"
CORPUS_PATH = DATA / "corpus_fr.txt"
TOKENIZER_PATH = DATA / "tokenizer.json"
VOCAB_SIZE = 10_000

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

SAMPLES = [
    "Il était une fois une petite fille qui vivait près de la forêt.",
    "Le chat noir dormait sur le toit de la maison.",
    "Tom et Sam ont trouvé une balle rouge dans le jardin.",
    "Elle était très triste, alors son ami lui a offert un gâteau.",
    "Les enfants ont appris qu'il faut partager avec les autres.",
]

# LeCarnet is native French, but copy-pasted text can still carry a non-breaking
# space or a typographic apostrophe. Normalising them keeps the vocabulary small.
SPACE = re.compile(r"[ \t\u00a0]+")
BLANK = re.compile(r"\s*\n\s*")


def normalise(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u00a0", " ").replace("\u2026", "...")
    text = SPACE.sub(" ", text)
    text = BLANK.sub(" ", text)
    return text.strip()


def download_corpus(max_chars: int, max_stories: int | None = None) -> int:
    """Stream LeCarnet into data/corpus_fr.txt, one story per line."""
    from datasets import load_dataset

    print(f"corpus: streaming {DATASET_ID} …")
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    written = 0
    stories = 0
    DATA.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = normalise(str(row.get("text", "")))
            if len(text) < 40:
                continue
            handle.write(text + "\n")
            written += len(text) + 1
            stories += 1
            if written >= max_chars or (max_stories and stories >= max_stories):
                break
    print(f"corpus: {stories} histoires, {written / 1e6:.1f} M caractères → {CORPUS_PATH}")
    return stories


def main() -> None:
    parser = argparse.ArgumentParser(description="build the French corpus and its tokenizer")
    parser.add_argument("--max-chars", type=int, default=30_000_000)
    parser.add_argument("--max-stories", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--reuse-corpus", action="store_true", help="do not download, tokenize the local file")
    parser.add_argument("--reuse-tokenizer", action="store_true", help="keep an existing data/tokenizer.json")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    if not args.reuse_corpus:
        try:
            download_corpus(args.max_chars, args.max_stories)
        except Exception as error:  # network, quota, rate limit
            if not CORPUS_PATH.exists():
                raise
            print(f"téléchargement impossible ({type(error).__name__}: {error})")
            print(f"corpus local conservé : {CORPUS_PATH}")

    if args.reuse_tokenizer and TOKENIZER_PATH.exists():
        print(f"tokenizer conservé : {TOKENIZER_PATH}")
        return

    from tokenizers import ByteLevelBPETokenizer

    print(f"BPE: entraînement sur {CORPUS_PATH} (vocab {args.vocab_size}) …")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(CORPUS_PATH)],
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save(str(TOKENIZER_PATH))
    print(f"tokenizer: {tokenizer.get_vocab_size()} tokens → {TOKENIZER_PATH}")

    # The visual check: French in, the same French out, accents and elisions.
    print("\nallers-retours :")
    for sentence in SAMPLES:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded.ids)
        status = "ok" if decoded.strip() == sentence.strip() else "DIFFÉRENT"
        print(f"  [{status}] {sentence}")
        print(f"           {encoded.tokens[:14]}{' …' if len(encoded.tokens) > 14 else ''}")


if __name__ == "__main__":
    main()
