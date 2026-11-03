"""Stage 5 guarantees: a genome round-trips, and a resumed run is the same run.

``test_resume_continuity`` is the central test of the whole project: 50 steps,
save, reload, 50 more steps, and every loss must match the uninterrupted run to
1e-6. Its tolerance is not a knob to loosen; an unexplained gap is a bug.
"""

from __future__ import annotations

import torch

from mycelia.brain import Brain
from mycelia.checkpoint import (
    brain_from_genome,
    load_genome,
    save_genome,
)
from mycelia.data import get_batch
from mycelia.growth import GrowthManager
from mycelia.lifecycle import kill_isolated_nodes, revive_node
from mycelia.train import Trainer


def test_roundtrip(tmp_path) -> None:
    """Weights, wiring (ne), population and cemetery all come back."""
    torch.manual_seed(0)
    brain = Brain(n_nodes=16, max_hops=3, max_edges=128)
    x, _ = get_batch(2)
    with torch.no_grad():
        before, _ = brain(x, grow=False)

    kill_isolated_nodes(brain, step=1000, grace=300)
    path = save_genome(tmp_path / "genome.pt", brain, step=7)
    blob = load_genome(path)
    assert blob["format_version"] == 3
    assert blob["ne"] == brain.ne
    assert blob["step"] == 7

    restored = brain_from_genome(blob)
    with torch.no_grad():
        after, _ = restored(x, grow=False)

    assert restored.n_nodes == brain.n_nodes
    assert restored.ne == brain.ne
    assert torch.equal(before, after)
    assert set(restored.cemetery) == set(brain.cemetery)


def test_cemetery_survives(tmp_path) -> None:
    """A node buried before the save wakes up exactly after the reload."""
    torch.manual_seed(0)
    brain = Brain(n_nodes=16, max_hops=2, max_edges=64)
    probe = torch.randn(1, 3, brain.dim)
    ids = torch.tensor([14])
    with torch.no_grad():
        expected, _ = brain.bank.forward_rows(probe, ids)

    kill_isolated_nodes(brain, step=1000, grace=300)
    path = save_genome(tmp_path / "cemetery.pt", brain, step=11)
    restored = brain_from_genome(load_genome(path))

    assert 14 in restored.cemetery
    assert not bool(restored.node_alive[14])
    assert revive_node(restored, 14, source=2)
    with torch.no_grad():
        revived, _ = restored.bank.forward_rows(probe, ids)
    assert torch.equal(revived, expected)


def test_resume_continuity(tmp_path) -> None:
    """50 steps, save, 50 more: identical to the uninterrupted 100 to 1e-6."""
    kwargs = dict(
        data="toy",
        device="cpu",
        seed=42,
        n_nodes=16,
        max_hops=3,
        max_births=4,
        max_edges=256,
        batch_size=8,
        seq_len=10,
        enable_growth=False,
        enable_lifecycle=False,
    )
    trainer = Trainer(**kwargs)
    trainer.brain.allow_node_growth = False

    losses = [trainer.train_step()["loss"] for _ in range(50)]
    path = trainer.save(tmp_path / "genome.pt", latest=False)
    losses += [trainer.train_step()["loss"] for _ in range(50)]

    resumed = Trainer.resume(path, config=trainer.config, **kwargs)
    after = [resumed.train_step()["loss"] for _ in range(50)]

    assert len(after) == 50
    for index, (expected, got) in enumerate(zip(losses[50:], after)):
        assert abs(expected - got) < 1e-6, (
            f"step {51 + index}: uninterrupted {expected!r} vs resumed {got!r}"
        )


def test_growth_manager_survives(tmp_path) -> None:
    brain = Brain(n_nodes=16, max_edges=128)
    manager = GrowthManager(every=500, reallocate=256)
    manager.update(100.0, 0, brain.n_nodes, brain)
    manager.update(90.0, 500, brain.n_nodes, brain)
    manager.update(89.8, 1000, brain.n_nodes, brain)

    path = save_genome(tmp_path / "growth.pt", brain, step=1000, growth=manager)
    blob = load_genome(path)
    clone = GrowthManager(every=500)
    clone.load_state_dict(blob["growth"])

    assert clone.last_val == manager.last_val
    assert clone.suspended_until == manager.suspended_until
    assert clone.tiers == manager.tiers
    assert clone.history == manager.history
