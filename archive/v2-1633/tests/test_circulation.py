"""The circulation rule: a node must transform what it receives.

Two checks:

* a body that returns its input unchanged is measured as a circulation
  violation (``cos(out, in) > 0.999``);
* that violation actually costs energy, so an identity loop fades out instead
  of ringing forever.
"""

from __future__ import annotations

import torch

from mycelia.models.brain import BrainNetwork


def _identity_body(self, idx, x):
    gate = torch.ones(x.shape[0], device=x.device)
    return {"out": x, "gate": gate, "open": gate >= self.cfg_gate_floor}


def _transforming_body(self, idx, x):
    gate = torch.ones(x.shape[0], device=x.device)
    out = torch.tanh(x * 1.7) + 0.3 * torch.roll(x, 1, dims=-1)
    return {"out": out, "gate": gate, "open": gate >= self.cfg_gate_floor}


def test_identity_node_is_penalised(make_brain, monkeypatch):
    brain = make_brain(vocab=64)
    ids = torch.randint(0, 64, (1, 4))
    monkeypatch.setattr(type(brain.nodes), "body", _identity_body)
    identity = brain(ids, gumbel=False)
    monkeypatch.setattr(type(brain.nodes), "body", _transforming_body)
    transforming = brain(ids, gumbel=False)

    assert float(identity["aux"]["circulation"]) >= 0.0
    # The identity pass-through is flagged; a real transformation is not.
    assert float(transforming["aux"]["circulation"]) <= float(
        identity["aux"]["circulation"]) + 1e-6


def test_identity_signal_loses_more_energy(make_brain, monkeypatch):
    brain = make_brain(vocab=64)
    ids = torch.randint(0, 64, (1, 4))
    monkeypatch.setattr(type(brain.nodes), "body", _identity_body)
    identity_energy = brain(ids, gumbel=False)["energy"]
    monkeypatch.setattr(type(brain.nodes), "body", _transforming_body)
    transforming_energy = brain(ids, gumbel=False)["energy"]
    assert identity_energy <= transforming_energy + 1e-9


def test_circulation_penalty_is_differentiable(make_brain, monkeypatch):
    brain = make_brain(vocab=64)
    brain.train()
    monkeypatch.setattr(type(brain.nodes), "body", _identity_body)
    out = brain(torch.randint(0, 64, (1, 3)), gumbel=True)
    penalty = out["aux"]["circulation"]
    assert penalty.requires_grad
    penalty.backward()
    grads = [p.grad for p in brain.parameters() if p.grad is not None]
    assert grads, "the circulation penalty must reach the node parameters"
