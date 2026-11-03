"""Phase 1 guarantees: sparse wiring, circulation, silence, fatigue, learning."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Tests run from the repository root; make the package importable without an
# installed distribution or a conftest.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mycelia.brain import INPUT_NODE, OUTPUT_NODE, Brain, extract_path  # noqa: E402
from mycelia.data import VOCAB, get_batch  # noqa: E402

SEED_EDGES = 13  # 6 into the middle + 1 bridge + 6 into the reader


def test_forward_shape() -> None:
    torch.manual_seed(0)
    brain = Brain(n_nodes=16)
    x, _ = get_batch(2)
    logits, trace = brain(x)
    assert logits.shape == (2, 10, VOCAB)
    assert isinstance(trace, dict)


def test_sparse_init() -> None:
    brain = Brain()
    assert brain.edge_index.shape == (2, SEED_EDGES)
    assert brain.edge_mask.shape == (SEED_EDGES,)

    # 14 nodes are wired, so 50 of the 64 neurons start mute.
    wired = set(brain.edge_index.flatten().tolist())
    assert len(wired) == 14
    assert brain.n_nodes - len(wired) == 50

    # No dense adjacency anywhere: the edge list is the only wiring.
    for name, tensor in list(brain.named_buffers()) + list(brain.named_parameters()):
        assert tuple(tensor.shape) != (brain.n_nodes, brain.n_nodes), name


def test_signal_reaches_output() -> None:
    torch.manual_seed(0)
    brain = Brain()
    x, _ = get_batch(2)
    _, trace = brain(x)

    # The reader accumulated something, so the signal crossed the graph.
    assert brain.last_out_acc is not None
    assert float(brain.last_out_acc.abs().sum()) > 0.0

    # And it crossed along a real path: door, at least two middle nodes, reader.
    path = extract_path(trace)
    assert path is not None
    assert path[0] == INPUT_NODE and path[-1] == OUTPUT_NODE
    assert len(path) - 2 >= 2


def test_gate_silence() -> None:
    torch.manual_seed(0)
    brain = Brain()
    with torch.no_grad():
        # Force the door shut: its gate is below the firing threshold.
        brain.nodes[INPUT_NODE].gate.weight.zero_()
        brain.nodes[INPUT_NODE].gate.bias.fill_(-10.0)

    x, _ = get_batch(2)
    _, trace = brain(x)

    # A silent node sends nothing, so no neighbour state is touched at all.
    assert float(brain.last_out_acc.abs().sum()) == 0.0
    assert all(not edges for edges in trace.values())
    assert brain.last_firing_nodes == []


def test_dissipation() -> None:
    torch.manual_seed(0)
    brain = Brain(n_nodes=4, max_hops=6)
    # An artificial two-node cycle: the signal can go round and round.
    brain.set_edges(torch.tensor([[0, 1], [1, 0]], dtype=torch.long))

    dim = brain.dim
    with torch.no_grad():
        brain.embedding.weight.fill_(1.0)  # a purely positive signal
        for node_id in (0, 1):
            node = brain.nodes[node_id]
            node.gate.weight.zero_()
            node.gate.bias.fill_(10.0)
            node.router.weight.zero_()
            node.router.bias.fill_(10.0)
            # Near-identity body: the only decay left is the medium itself.
            node.body[0].weight.zero_()
            node.body[0].bias.zero_()
            node.body[0].weight[:dim, :] = torch.eye(dim)
            node.body[2].weight.zero_()
            node.body[2].bias.zero_()
            node.body[2].weight[:, :dim] = torch.eye(dim)

    x, _ = get_batch(1)
    brain(x)

    energy = brain.last_energy_by_hop
    assert energy[0] > 0.0
    # Fatigue, not an anti-loop rule: each traversal costs 10% of the energy.
    assert all(later < earlier for earlier, later in zip(energy, energy[1:]))


def test_learning() -> None:
    torch.manual_seed(0)
    brain = Brain()
    optimiser = torch.optim.AdamW(brain.parameters(), lr=1e-3)

    first = None
    last = None
    for step in range(500):
        x, y = get_batch(32)
        logits, _ = brain(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), y[:, :-1].reshape(-1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert first is not None and last is not None
    assert torch.isfinite(torch.tensor(last))
    assert last < first
