"""Speak: load a trained genome and let it continue a French sentence.

Run from anywhere:

    python scripts/generate.py --prompt "Il était une fois" --tokens 100

The text is only half of the output. The script also prints the circulation
trace of the last token: which neurons spoke to which, hop by hop, and the chain
that carried the signal from the door (node 0) to the reader (node 1), in the
form the brief asks for:

    token 'petite' : 0 → 5 → 17 → 1

That chain is the first window on who does what, i.e. the specialisation that
emerges on its own (F9).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mycelia.brain import extract_path, hop_summary  # noqa: E402
from mycelia.checkpoint import brain_from_genome, load_genome  # noqa: E402
from mycelia.data import TOKENIZER_PATH, clean_text, load_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generate French with a trained brain")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "genome.pt")
    parser.add_argument("--prompt", type=str, default="Il était une fois")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hops", type=int, default=None)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    if not args.checkpoint.exists():
        raise SystemExit(
            f"no genome at {args.checkpoint}: train first with "
            "`python mycelia/train.py --data text`"
        )

    blob = load_genome(args.checkpoint)
    brain = brain_from_genome(blob)
    brain.eval()
    if args.hops:
        brain.max_hops = int(args.hops)
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer.get_vocab_size() != brain.vocab:
        raise SystemExit(
            f"tokenizer has {tokenizer.get_vocab_size()} tokens but the brain has {brain.vocab}"
        )

    prompt_ids = tokenizer.encode(args.prompt).ids
    ids, trace = brain.generate(
        prompt_ids,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=tokenizer.token_to_id("<eos>"),
        seed=args.seed,
    )

    last_token = clean_text(tokenizer.decode([ids[-1]])).strip() or repr(ids[-1])
    print(f"prompt : {args.prompt!r}")
    print(
        f"cerveau : {brain.n_nodes} nœuds, {brain.ne} synapses, {brain.max_hops} tours, "
        f"cimetière {len(brain.cemetery)}"
    )
    # C8: the byte-level BPE can leave U+FFFD and control characters; they are
    # dropped for display, never rewritten inside the brain.
    print(f"\n{clean_text(tokenizer.decode(ids))}\n")

    path = extract_path(trace)
    if path:
        chain = " → ".join(str(node) for node in path)
        print(f"token {last_token!r} : {chain}")
        print(f"  ({len(path) - 2} nœuds intermédiaires, {len(path) - 1} synapses)")
    else:
        print(f"token {last_token!r} : aucun chemin n'a atteint le nœud 1")

    print(f"\ncirculation : {hop_summary(trace)}")
    for hop in sorted(trace):
        edges = trace[hop]
        detail = ", ".join(f"{s}->{t}" for s, t in edges[:8])
        more = "" if len(edges) <= 8 else f" … (+{len(edges) - 8})"
        print(f"  tour {hop}: {len(edges):3d} synapses | {detail}{more}")


if __name__ == "__main__":
    main()
