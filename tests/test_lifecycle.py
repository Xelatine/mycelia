"""Stage 2 guarantees: births, pruning, death, exact revival, anti-copy."""

from __future__ import annotations

import torch

from mycelia.brain import Brain
from mycelia.data import get_batch
from mycelia.lifecycle import (
    anti_identity_penalty,
    anti_redundancy_penalty,
    kill_isolated_nodes,
    prune_edges,
    revive_node,
)


def edge_pairs(brain: Brain) -> set[tuple[int, int]]:
    return {
        (int(s), int(t)) for s, t in brain.edge_index[:, : brain.ne].t().tolist()
    }


def test_birth(monkeypatch) -> None:
    """A high router score toward an unwired peer opens a synapse, in the pass."""
    torch.manual_seed(0)
    brain = Brain(n_nodes=32, max_hops=3, max_edges=128, birth_candidates=32)
    brain.allow_node_growth = False
    brain.set_edges(torch.tensor([[0], [2]], dtype=torch.long))

    with torch.no_grad():
        # Node 2 wants node 14 intensely; everybody else keeps a neutral score.
        brain.bank.q_b[2].zero_()
        brain.bank.q_b[2, 0] = 10.0
        brain.bank.k_b[14].zero_()
        brain.bank.k_b[14, 0] = 10.0

    monkeypatch.setattr(
        "mycelia.brain._gumbel_like", lambda tensor: torch.zeros_like(tensor)
    )

    before = brain.ne
    x, _ = get_batch(1)
    brain(x)  # training mode: births allowed

    assert (2, 14) in edge_pairs(brain)
    assert brain.ne > before
    assert brain.births > 0
    assert brain.edge_mask.numel() == brain.ne


def test_prune() -> None:
    """Starved synapses are cut; busy ones survive with their strength."""
    brain = Brain(n_nodes=16, max_edges=64)
    # Neither pair belongs to the seed path, which pruning always spares.
    brain.set_edges(
        torch.tensor([[8, 10], [9, 11]], dtype=torch.long),
        edge_mask=torch.tensor([0.2, 0.9]),
    )
    with torch.no_grad():
        brain.edge_usage[0] = 0.001  # starved
        brain.edge_usage[1] = 0.5  # busy

    dead = prune_edges(brain)

    assert dead == 1
    assert brain.ne == 1
    assert edge_pairs(brain) == {(10, 11)}
    assert torch.allclose(brain.edge_mask, torch.tensor([0.9]))
    assert brain.edge_deaths == 1


def test_prune_is_bounded_and_spares_the_brainstem() -> None:
    """A pass may starve a tenth of the wiring, never 92 % of it (live bug)."""
    brain = Brain(n_nodes=64, max_edges=256)
    pairs = torch.tensor([[i % 60, (i * 7 + 3) % 60] for i in range(120)], dtype=torch.long)
    brain.set_edges(pairs.t())
    with torch.no_grad():
        # A skew like the real one: most links are cold, a few carry everything.
        brain.edge_usage[:120] = torch.linspace(0.0, 1.0, 120) ** 3

    dead = prune_edges(brain)

    assert 0 < dead <= 12
    assert brain.ne == 120 - dead
    # The busiest links survive with their history: none of them is in the cut.
    assert float(brain.edge_usage[: brain.ne].max()) > 0.99

    # A fresh colony has only the 12 brainstem edges, all still cold: a naive
    # absolute cut would amputate the only path to the output.
    young = Brain(n_nodes=32)
    with torch.no_grad():
        young.edge_usage[: young.ne].zero_()
    assert prune_edges(young) == 0
    assert young.ne == 12


def test_node_death() -> None:
    """A neuron with no synapse is buried with its exact weights."""
    brain = Brain(n_nodes=16)
    before = {k: v.clone() for k, v in brain.bank.row_state(14).items()}

    dead = kill_isolated_nodes(brain, step=1000, grace=300)

    assert 14 in dead
    assert not bool(brain.node_alive[14])
    assert set(brain.cemetery[14]) == set(before)
    for name, tensor in before.items():
        assert torch.equal(brain.cemetery[14][name], tensor)
    # The firing organs are cleared, not left computing in a corner.
    assert brain.bank.compute_row_is_zero(14)
    # The doors never die.
    assert bool(brain.node_alive[0]) and bool(brain.node_alive[1])
    assert brain.node_deaths == len(dead)

    # A young node keeps its grace period.
    brain2 = Brain(n_nodes=16)
    dead2 = kill_isolated_nodes(brain2, step=10, grace=300)
    assert dead2 == []


def test_revival_exact() -> None:
    """A buried node comes back bit for bit: 'it works again like before'."""
    brain = Brain(n_nodes=16)
    probe = torch.randn(1, 4, brain.dim)
    ids = torch.tensor([14])
    with torch.no_grad():
        expected, _ = brain.bank.forward_rows(probe, ids)

    kill_isolated_nodes(brain, step=1000, grace=300)
    assert not bool(brain.node_alive[14])

    assert revive_node(brain, 14, source=2)
    with torch.no_grad():
        restored, _ = brain.bank.forward_rows(probe, ids)

    assert bool(brain.node_alive[14])
    assert 14 not in brain.cemetery
    assert (2, 14) in edge_pairs(brain)
    assert torch.equal(restored, expected)


def test_anti_identity() -> None:
    """Copying the input is taxed; transforming it is free (F4)."""
    brain = Brain(n_nodes=16)
    signal = torch.randn(2, 3, brain.dim)

    copy = anti_identity_penalty(brain, states=[(signal, signal.clone())])
    transformed = anti_identity_penalty(brain, states=[(signal, -signal)])

    assert float(copy) > 0.99
    assert float(transformed) == 0.0


def test_anti_redundancy() -> None:
    """Two nodes saying the same thing pay; orthogonal voices do not (F9)."""
    brain = Brain(n_nodes=16)
    outputs = torch.zeros(16, brain.dim)
    outputs[0] = torch.ones(brain.dim)
    outputs[1] = torch.ones(brain.dim)  # a clone of node 0
    outputs[2] = -torch.ones(brain.dim)

    redundant = anti_redundancy_penalty(brain, outputs=outputs, sample=16)
    assert float(redundant) > 0.0

    orthogonal = torch.zeros(16, brain.dim)
    for i in range(4):
        orthogonal[i, i] = 1.0
    free = anti_redundancy_penalty(brain, outputs=orthogonal, sample=16)
    assert float(free) == 0.0
