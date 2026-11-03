"""Stage 1 guarantees: sparse wiring, circulation, silence, fatigue, learning."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mycelia.brain import INPUT_NODE, OUTPUT_NODE, Brain, extract_path
from mycelia.data import VOCAB, get_batch
from mycelia.node import GATE_THRESHOLD

SEED_PAIRS = {(INPUT_NODE, t) for t in range(2, 8)} | {(s, OUTPUT_NODE) for s in range(2, 8)}


def used_pairs(brain: Brain) -> set[tuple[int, int]]:
    return {
        (int(s), int(t))
        for s, t in brain.edge_index[:, : brain.ne].t().tolist()
    }


def test_forward_shape() -> None:
    brain = Brain(n_nodes=16, max_hops=4)
    x, _ = get_batch(2)
    logits, trace = brain(x, grow=False)
    assert logits.shape == (2, 10, VOCAB)
    assert isinstance(trace, dict)
    assert torch.isfinite(logits).all()


def test_sparse_init() -> None:
    """The seed is exactly D1's 12 edges, and the buffer stays a capacity.

    The count is read from ``ne``, never from the buffer's shape (the buffer is
    preallocated to 20 000 columns, which is a capacity, not a wiring).
    """
    brain = Brain()
    assert brain.ne == 12
    assert brain.edge_index.shape == (2, brain.max_edges)
    assert brain.edge_mask.numel() == 12
    assert used_pairs(brain) == SEED_PAIRS

    wired = set(brain.edge_index[:, :12].flatten().tolist())
    assert wired == set(range(8))  # 0..7 are wired, 8..511 start mute
    assert brain.n_nodes - len(wired) == 504

    # No dense adjacency anywhere: the edge list is the only wiring.
    for name, tensor in list(brain.named_buffers()) + list(brain.named_parameters()):
        assert tuple(tensor.shape) != (brain.n_nodes, brain.n_nodes), name


def test_signal_reaches_output() -> None:
    brain = Brain(n_nodes=16, max_hops=4)
    x, _ = get_batch(2)
    _, trace = brain(x, grow=False)

    assert float(brain.last_out_acc.abs().sum()) > 0.0
    path = extract_path(trace)
    assert path is not None
    assert path[0] == INPUT_NODE and path[-1] == OUTPUT_NODE
    assert len(path) >= 3  # door, at least one middle node, reader


def test_gate_silence() -> None:
    """A closed door sends nothing: the colony stays dark, prediction frozen."""
    brain = Brain(n_nodes=16, max_hops=4)
    with torch.no_grad():
        brain.bank.gate_w[INPUT_NODE].zero_()
        brain.bank.gate_b[INPUT_NODE].fill_(-10.0)

    x, _ = get_batch(2)
    logits, trace = brain(x, grow=False)

    assert torch.sigmoid(torch.tensor(-10.0)) < GATE_THRESHOLD
    assert all(len(edges) == 0 for edges in trace.values())
    assert brain.last_firing_nodes == []
    assert torch.allclose(logits[:, 0], logits[:, -1])


def test_dissipation() -> None:
    """An artificial cycle loses energy at every traversal: fatigue, not a ban."""
    brain = Brain(n_nodes=8, max_hops=6, max_edges=64, max_births=0)
    brain.set_edges(torch.tensor([[0, 2], [2, 0]], dtype=torch.long))

    with torch.no_grad():
        for node_id in (0, 2):
            brain.bank.gate_w[node_id].zero_()
            brain.bank.gate_b[node_id].fill_(10.0)
            brain.bank.q_w[node_id].zero_()
            brain.bank.q_b[node_id].fill_(10.0)
            brain.bank.k_w[node_id].zero_()
            brain.bank.k_b[node_id].fill_(10.0)
            # Near-identity body: the only decay left is the medium itself.
            brain.bank.b1_w[node_id].zero_()
            brain.bank.b1_b[node_id].zero_()
            brain.bank.b2_w[node_id].zero_()
            brain.bank.b2_b[node_id].zero_()

    x, _ = get_batch(1)
    brain(x, grow=False)

    energy = brain.last_energy_by_hop
    assert len(energy) >= 4
    assert energy[0] > 0.0
    assert all(later < earlier for earlier, later in zip(energy, energy[1:]))


def test_chunking_equivalence() -> None:
    """D2: chunking the scatter changes nothing but the order of additions."""
    torch.manual_seed(0)
    brain = Brain(n_nodes=64, max_hops=3, max_births=0, max_edges=1024)
    pairs = torch.stack(
        [torch.randint(2, 64, (300,)), torch.randint(2, 64, (300,))]
    )
    brain.set_edges(pairs)
    x = torch.randint(2, 99, (3, 8))

    with torch.no_grad():
        brain.edge_chunk_size = 100_000  # one single chunk: the v3 behaviour
        whole, _ = brain(x, grow=False)
        brain.edge_chunk_size = 32
        chunked, _ = brain(x, grow=False)

    assert torch.allclose(whole, chunked, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the memory wall only exists on the GPU")
def test_chunking_memory() -> None:
    """D2: 5 000 edges, batch 32, sequence 128 must fit under 6 GiB.

    Two hops rather than eight keep the test runnable; the point is the peak,
    which follows ``edge_chunk_size`` and not the size of the wiring.
    """
    torch.manual_seed(0)
    brain = Brain(
        n_nodes=512, dim=32, vocab=100, max_hops=2, max_births=0,
        max_edges=8192, edge_chunk_size=256,
    ).cuda()
    edges = 5000
    pairs = torch.stack(
        [torch.randint(2, 512, (edges,)), torch.randint(2, 512, (edges,))]
    ).cuda()
    brain.set_edges(pairs)
    x = torch.randint(2, 99, (32, 128), device="cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    logits, _ = brain(x, grow=False)
    logits.float().sum().backward()
    peak = torch.cuda.max_memory_allocated() / 2**30

    assert peak < 6.0, f"pic mémoire {peak:.2f} GiB à 5 000 arêtes"


def test_learning() -> None:
    """500 steps must visibly bend the delayed-copy loss (D2)."""
    brain = Brain(n_nodes=16, max_hops=4, max_births=0, max_edges=256)
    brain.allow_node_growth = False
    optimiser = torch.optim.AdamW(brain.parameters(), lr=1e-3)

    first = None
    losses = []
    for step in range(500):
        x, y = get_batch(16)
        logits, _ = brain(x)
        loss = F.cross_entropy(logits[:, 1:].reshape(-1, VOCAB), y[:, 1:].reshape(-1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if first is None:
            first = float(loss.detach())
        losses.append(float(loss.detach()))

    assert all(torch.isfinite(torch.tensor(value)) for value in losses[-5:])
    # Not "1 % better": a visible bend, measured against the last 20 steps.
    assert sum(losses[-20:]) / 20 < first * 0.9
