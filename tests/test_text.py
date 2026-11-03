"""Stage 3 guarantees: faithful French tokenizer, bounded speech, no peeking.

These tests never download anything. They use the tokenizer built by
``scripts/train_tokenizer.py`` when it exists, and a tiny brain for generation,
so the suite stays offline and fast.
"""

from __future__ import annotations

import pytest
import torch

from mycelia.brain import Brain, apply_repetition_penalty
from mycelia.data import TOKENIZER_PATH, VOCAB, get_batch, load_tokenizer

FRENCH = [
    "Il était une fois une petite fille qui vivait près de la forêt.",
    "Le chat noir dormait sur le toit de la maison, très tranquille.",
    "Tom et Sam ont trouvé une balle rouge dans le jardin de l'école.",
    "Elle était triste, alors son ami lui a offert un gâteau au citron.",
    "Où est passé le petit chien ? demanda la mère, un peu inquiète.",
]


def test_tokenizer_roundtrip() -> None:
    """French in, the same French out: accents, apostrophes, cedillas."""
    if not TOKENIZER_PATH.exists():
        pytest.skip("run scripts/train_tokenizer.py to build data/tokenizer.json")
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    assert tokenizer.get_vocab_size() == 10_000
    for sentence in FRENCH:
        ids = tokenizer.encode(sentence).ids
        assert all(0 <= token < tokenizer.get_vocab_size() for token in ids)
        assert tokenizer.decode(ids).strip() == sentence.strip()


def test_causality() -> None:
    """Position t must not see token t: otherwise next-token prediction is a cheat."""
    torch.manual_seed(0)
    brain = Brain(n_nodes=16, max_hops=4, max_births=0, max_edges=256)
    base = torch.tensor([[4, 5, 6, 7]])
    changed_last = torch.tensor([[4, 5, 6, 9]])  # only the token being predicted
    changed_earlier = torch.tensor([[4, 9, 6, 7]])

    with torch.no_grad():
        a, _ = brain(base, grow=False)
        b, _ = brain(changed_last, grow=False)
        c, _ = brain(changed_earlier, grow=False)

    assert torch.allclose(a[0, -1], b[0, -1])  # own token is invisible
    assert not torch.allclose(a[0, -1], c[0, -1])  # earlier tokens are not


def test_generate_shapes() -> None:
    """generate() speaks inside the vocabulary and reports a well-formed trace."""
    torch.manual_seed(0)
    brain = Brain(n_nodes=16, vocab=100, max_hops=4, max_births=0, max_edges=256)
    prompt = [5, 6, 7]
    ids, trace = brain.generate(prompt, max_new_tokens=6, temperature=0.9, seed=0)

    assert len(ids) == len(prompt) + 6
    assert ids[: len(prompt)] == prompt
    assert all(0 <= token < 100 for token in ids[len(prompt) :])
    assert isinstance(trace, dict)
    assert all(isinstance(hop, int) for hop in trace)
    for edges in trace.values():
        assert all(isinstance(s, int) and isinstance(t, int) for s, t in edges)
        assert all(0 <= s < 16 and 0 <= t < 16 for s, t in edges)


def test_repetition_penalty() -> None:
    """The token that would close a loop already heard three times is taxed."""
    logits = torch.zeros(VOCAB)
    logits[1] = 5.0
    logits[2] = 4.0
    # (2, 3) is followed by 1 three times: predicting 1 again closes the loop.
    context = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]

    taxed = apply_repetition_penalty(logits, context)
    assert taxed[1].item() == pytest.approx(logits[1].item() - 2.0)
    assert taxed[2].item() == pytest.approx(logits[2].item())
    assert taxed[0].item() == pytest.approx(logits[0].item())

    untouched = apply_repetition_penalty(logits, [1, 2, 3, 4, 5])
    assert torch.equal(untouched, logits)


def test_toy_still_works() -> None:
    """The delayed copy still learns: stages 1 and 2 are not broken."""
    from mycelia.train import Trainer

    trainer = Trainer(
        data="toy",
        device="cpu",
        seed=0,
        n_nodes=16,
        max_hops=3,
        max_births=0,
        max_edges=256,
        batch_size=16,
        seq_len=10,
        enable_growth=False,
        enable_lifecycle=False,
    )
    trainer.brain.allow_node_growth = False
    first = None
    losses = []
    for _ in range(120):
        metrics = trainer.train_step()
        first = metrics["loss"] if first is None else first
        losses.append(metrics["loss"])

    assert all(torch.isfinite(torch.tensor(value)) for value in losses[-5:])
    assert sum(losses[-10:]) / 10 < first
