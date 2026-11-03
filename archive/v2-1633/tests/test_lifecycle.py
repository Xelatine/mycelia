"""Lifecycle: links die before nodes, and the cemetery resurrects exactly."""

from __future__ import annotations

import torch

from mycelia.models.lifecycle import Lifecycle


def test_links_die_before_nodes(make_brain, tiny_cfg):
    brain = make_brain(vocab=64)
    events = []
    life = Lifecycle(brain, tiny_cfg, seed=1, emit=lambda **kw: events.append(kw))
    victim = 5

    # Starve every link touching node 5 and make them old enough to die.
    adj = brain.adj
    keep = torch.ones(adj.n_edges, dtype=torch.bool)
    for i in range(adj.n_edges):
        if int(adj.src[i]) == victim or int(adj.dst[i]) == victim:
            keep[i] = False
    adj.usage[: adj.n_edges] = 0.0
    adj.born[: adj.n_edges] = 0
    adj.subset(keep)
    adj.rebuild_tables(brain.nodes.n_nodes)

    # First pass: the orphan is only isolated (links die first).
    brain.step = 100
    life.tick(100)
    assert victim in life.isolated
    assert not bool(brain.nodes.dead_mask[victim])
    assert any(e["type"] == "isolated" for e in events)

    # Second pass, past the node grace period: now the node is buried.
    brain.step = 400
    life.tick(400)
    assert bool(brain.nodes.dead_mask[victim])
    assert victim in life.cemetery
    assert life.counters["deaths"] >= 1


def test_minimum_wiring_is_protected(make_brain, tiny_cfg):
    brain = make_brain(vocab=64)
    life = Lifecycle(brain, tiny_cfg, seed=1, emit=lambda **kw: None)
    before = brain.adj.n_edges
    brain.adj.usage[: brain.adj.n_edges] = 0.0
    brain.adj.born[: brain.adj.n_edges] = 0
    tiny_cfg.life.min_live_edges = before
    life.prune_links(step=10_000)
    assert brain.adj.n_edges == before, "the vital minimum must never be cut"


def test_pruning_is_progressive(make_brain, tiny_cfg):
    brain = make_brain(vocab=64)
    life = Lifecycle(brain, tiny_cfg, seed=1, emit=lambda **kw: None)
    before = brain.adj.n_edges
    brain.adj.usage[: before] = 0.0
    brain.adj.born[: before] = 0
    tiny_cfg.life.max_cut_fraction = 0.25
    tiny_cfg.life.min_live_edges = 0
    cut = life.prune_links(step=10_000)
    assert 0 < cut <= int(0.25 * before) + 1


def test_cemetery_wake_is_exact(make_brain, tiny_cfg):
    brain = make_brain(vocab=64)
    life = Lifecycle(brain, tiny_cfg, seed=1, emit=lambda **kw: None)
    victim = 9
    before = {name: getattr(brain.nodes, name).data[victim].clone()
              for name in ("W1", "W2", "b1", "b2", "profile", "conf", "diss")}

    life.bury(victim, step=50)
    assert bool(brain.nodes.dead_mask[victim])
    assert victim in life.free_slots

    assert life.wake(victim, step=60) is True
    assert not bool(brain.nodes.dead_mask[victim])
    for name, value in before.items():
        assert torch.equal(getattr(brain.nodes, name).data[victim], value), name
    assert victim not in life.cemetery


def test_birth_creates_a_quiet_node_with_links(make_brain, tiny_cfg):
    brain = make_brain(vocab=64)
    life = Lifecycle(brain, tiny_cfg, seed=1, emit=lambda **kw: None)
    n_before = brain.nodes.n_nodes
    node = life.spawn(parent=3, step=10)
    assert node == n_before
    assert brain.nodes.n_nodes == n_before + 1
    # Newborn: closed gate, zero body output, so it cannot perturb the brain.
    assert float(brain.nodes.W2[node].abs().sum()) == 0.0
    assert float(brain.nodes.gate_b[node]) <= -1.9
    edges = [(int(a), int(b)) for a, b in
             zip(brain.adj.src[: brain.adj.n_edges], brain.adj.dst[: brain.adj.n_edges])]
    assert (3, node) in edges
    assert (node, tiny_cfg.model.s_node) in edges


def test_recycled_slot_loses_the_previous_identity(make_brain, tiny_cfg):
    brain = make_brain(vocab=64)
    life = Lifecycle(brain, tiny_cfg, seed=1, emit=lambda **kw: None)
    life.bury(7, step=10)
    with torch.no_grad():
        brain.nodes.conf[7] = 0.99
    node = life.spawn(parent=3, step=20)
    assert node == 7, "a dead slot is recycled before the pool grows"
    assert life.counters["recycled"] >= 1
    assert abs(float(brain.nodes.conf[node]) - 0.5) < 1e-6
