"""Pool reallocation must be lossless: weights, links, states, cemetery survive."""

from __future__ import annotations

import torch

from mycelia.models.lifecycle import Lifecycle


def _fingerprint(brain):
    bank = brain.nodes
    n = bank.n_nodes
    return {
        "W1": bank.W1.data[:n].clone(),
        "W2": bank.W2.data[:n].clone(),
        "profile": bank.profile.data[:n].clone(),
        "conf": bank.conf.data[:n].clone(),
        "usage": bank.usage[:n].clone(),
        "src": brain.adj.src[: brain.adj.n_edges].clone(),
        "dst": brain.adj.dst[: brain.adj.n_edges].clone(),
        "logit": brain.adj.logit.data[: brain.adj.n_edges].clone(),
    }


def test_reallocation_preserves_everything(make_brain, tiny_cfg):
    brain = make_brain(vocab=64)
    life = Lifecycle(brain, tiny_cfg, seed=1, emit=lambda **kw: None)
    life.bury(11, step=5)
    cemetery_before = dict(life.cemetery)

    before = _fingerprint(brain)
    nodes_before = brain.nodes.n_nodes
    capacity_before = brain.nodes.max_nodes

    life.reallocate(16)

    assert brain.nodes.max_nodes == capacity_before + 16
    assert brain.adj.cand_edge.shape[0] == brain.nodes.max_nodes
    after = _fingerprint(brain)
    for key in ("W1", "W2", "profile", "conf", "usage", "src", "dst", "logit"):
        assert after[key].shape == before[key].shape, key
        assert torch.equal(after[key], before[key]), key
    assert brain.nodes.n_nodes == nodes_before + 16
    assert set(life.cemetery) == set(cemetery_before)
    assert life.counters["reallocations"] == 1


def test_brain_still_trains_after_reallocation(make_brain, tiny_cfg):
    brain = make_brain(vocab=64)
    brain.train()
    life = Lifecycle(brain, tiny_cfg, seed=1, emit=lambda **kw: None)
    life.reallocate(8)
    ids = torch.randint(0, 64, (1, 4))
    out = brain(ids, gumbel=True)
    loss = out["logits"].sum() + out["aux"]["mask_l1"]
    loss.backward()
    assert torch.isfinite(out["logits"]).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in brain.parameters())
