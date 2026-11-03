"""Scale 2: 3 000 nodes with a strictly sparse adjacency (no [N, N] tensor)."""

from __future__ import annotations

import torch

from mycelia.models.brain import BrainNetwork


def test_scale_two_is_sparse(tiny_cfg):
    cfg = tiny_cfg
    cfg.model.nodes_initial = 3000
    cfg.model.nodes_max = 3000
    cfg.model.dim = 32
    cfg.model.candidates = 4
    cfg.model.beam = 2
    cfg.model.max_turns = 3
    cfg.model.seed_edges_per_node = 0.5

    brain = BrainNetwork(cfg, 128)
    brain.eval()

    # No dense N x N allocation anywhere in the parameters or buffers.
    for name, tensor in list(brain.named_parameters()) + list(brain.named_buffers()):
        assert tensor.dim() <= 3, name
        if tensor.dim() == 2:
            assert tensor.shape[0] != tensor.shape[1] or tensor.shape[0] <= 1024, name
    assert brain.adj.cand_edge.shape == (3000, cfg.model.candidates)
    assert brain.adj.src.numel() >= brain.adj.n_edges

    ids = torch.randint(0, 128, (1, 4))
    out = brain(ids, gumbel=False)
    assert out["logits"].shape == (1, 4, 128)
    assert torch.isfinite(out["logits"]).all()
    assert brain.adj.n_edges < 20_000
