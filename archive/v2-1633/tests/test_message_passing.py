"""Message passing: energy must decay, hops must exist, traces must be real."""

from __future__ import annotations

import torch

from mycelia.gen.sampling import generate
from mycelia.models.brain import straight_through


def test_forward_returns_logits_and_energy(make_brain):
    brain = make_brain(vocab=64)
    ids = torch.randint(0, 64, (2, 6))
    out = brain(ids, tau=0.5, gumbel=False)
    assert out["logits"].shape == (2, 6, 64)
    assert torch.isfinite(out["logits"]).all()
    assert 0.0 <= out["energy"] <= brain.mcfg.init_energy + 1e-6
    assert out["aux"]["hops"] > 0


def test_energy_dissipates_over_turns(make_brain):
    """More turns must never *increase* the residual energy: dissipation < 1."""
    brain = make_brain(vocab=64)
    ids = torch.randint(0, 64, (1, 4))
    short = brain(ids, gumbel=False)
    brain.mcfg.max_turns = brain.mcfg.max_turns * 3
    long = brain(ids, gumbel=False)
    assert long["energy"] <= short["energy"] + 1e-6


def test_no_signal_reaches_s_degrades_gracefully(make_brain):
    """A brain wired nowhere still returns finite logits (no NaN, no crash)."""
    brain = make_brain(vocab=64)
    brain.adj.subset(torch.zeros(brain.adj.n_edges, dtype=torch.bool))
    brain.adj.rebuild_tables(brain.nodes.n_nodes)
    out = brain(torch.randint(0, 64, (1, 4)), gumbel=False)
    assert torch.isfinite(out["logits"]).all()


def test_trace_edges_exist_in_the_graph(make_brain):
    brain = make_brain(vocab=64)
    out = brain(torch.randint(0, 64, (1, 3)), track=True, gumbel=False)
    trace = out["trace"]
    assert trace["turns"], "a diffusion that visits nodes must record turns"
    edges = {(int(a), int(b)) for a, b in
             zip(brain.adj.src[: brain.adj.n_edges], brain.adj.dst[: brain.adj.n_edges])}
    for turn in trace["turns"]:
        for edge in turn["edges"]:
            assert (edge["src"], edge["dst"]) in edges
    assert all(0 <= node < brain.nodes.n_nodes for node in trace["nodes"])


def test_straight_through_matches_hard_forward():
    mask = torch.tensor([0.1, 0.4, 0.6, 0.9], requires_grad=True)
    hard = straight_through(mask, 0.5)
    assert torch.equal((hard > 0.5), (mask > 0.5))
    hard.sum().backward()
    assert torch.allclose(mask.grad, torch.ones_like(mask))


def test_generate_returns_tokens_and_path(make_brain, toy_corpus):
    from mycelia.data.tokenizer import resolve_tokenizer

    brain = make_brain(vocab=10_000)
    tokenizer, _ = resolve_tokenizer(toy_corpus, 10_000,
                                    toy_corpus.parent / "cache")
    out = generate(brain, tokenizer, prompt="Il était une fois", max_new_tokens=4,
                   temperature=0.0, device="cpu", track=True, seed=1)
    assert out["text"]
    assert all(0 <= t["token_id"] < tokenizer.vocab_size for t in out["tokens"])
    assert all(isinstance(t["edges"], list) for t in out["tokens"])
