"""Stage 4 guarantees: growth is real, lossless, and driven by measurements."""

from __future__ import annotations

import time

import torch

import mycelia.brain as brain_module
from mycelia.brain import Brain
from mycelia.data import get_batch
from mycelia.growth import (
    GrowthManager,
    apply_pending_node_births,
    reallocate_pool,
    try_birth_node,
)
from mycelia.lifecycle import kill_isolated_nodes


def edges(brain: Brain) -> set[tuple[int, int]]:
    return {(int(s), int(t)) for s, t in brain.edge_index[:, : brain.ne].t().tolist()}


def test_birth_from_message() -> None:
    """A message addressed beyond the pool makes a slot exist at the next stop."""
    torch.manual_seed(0)
    brain = Brain(
        n_nodes=16,
        max_hops=2,
        max_edges=128,
        max_births=8,
        max_node_births=2,
        node_birth_interval=1,
        node_margin=4,
    )
    brain.set_edges(torch.tensor([[0], [2]], dtype=torch.long))
    brain.clock = 0

    x, _ = get_batch(1)
    brain(x)  # training mode: the door may wish for a node that does not exist

    assert brain.pending_node_requests, "no address was sent beyond the pool"
    created = apply_pending_node_births(brain)

    assert created
    assert brain.n_nodes > 16
    new_id = created[0]
    assert new_id >= 16
    assert bool(brain.node_alive[new_id])
    assert brain.node_born_step[new_id] == brain.clock
    assert (0, new_id) in edges(brain)
    # The vertex was allocated outside the forward, never inside it.
    assert not brain.pending_node_requests


def test_reallocate_no_loss() -> None:
    """New slots change nothing about the old colony, bit for bit."""
    torch.manual_seed(0)
    brain = Brain(n_nodes=16, max_hops=3, max_births=0, max_edges=128)
    x, _ = get_batch(2)
    with torch.no_grad():
        before, _ = brain(x, grow=False)
    old_rows = {name: getattr(brain.bank, name).detach().clone() for name in ("gate_w", "b1_w", "k_b")}

    old_n, new_n = reallocate_pool(brain, add=8)
    assert (old_n, new_n) == (16, 24)
    assert brain.n_nodes == 24

    with torch.no_grad():
        after, _ = brain(x, grow=False)

    assert torch.equal(before, after)
    for name, row in old_rows.items():
        assert torch.equal(getattr(brain.bank, name).detach()[:16], row)
    # The new rows are weak noise, never exact zeros: a zero node is invisible.
    assert not brain.bank.row_is_zero(16)


def test_revive_from_cemetery_via_growth(monkeypatch) -> None:
    """The cemetery returns exact weights, and the calling edge is wired."""
    torch.manual_seed(0)
    brain = Brain(n_nodes=32, max_hops=3, max_edges=128, birth_candidates=32)
    # Node 2 has both an entry and an exit, so it survives the burial and can
    # be the one that addresses the cemetery.
    brain.set_edges(torch.tensor([[0, 2], [2, 1]], dtype=torch.long))
    brain.clock = 0

    probe = torch.randn(1, 3, brain.dim)
    ids = torch.tensor([14])
    with torch.no_grad():
        expected, _ = brain.bank.forward_rows(probe, ids)

    kill_isolated_nodes(brain, step=1000, grace=300)
    assert not bool(brain.node_alive[14])

    with torch.no_grad():
        # The door addresses the buried node; the address survives burial, which
        # is precisely how a message can still call a dead node back (F3).
        brain.bank.q_w[0].zero_()
        brain.bank.q_b[0].zero_()
        brain.bank.q_b[0, 0] = 10.0
        brain.bank.k_b[14].zero_()
        brain.bank.k_b[14, 0] = 10.0
    monkeypatch.setattr(
        "mycelia.brain._gumbel_like", lambda tensor: torch.zeros_like(tensor)
    )

    x, _ = get_batch(1)
    brain(x)

    assert bool(brain.node_alive[14]), "the addressed buried node did not wake"
    assert 14 not in brain.cemetery
    assert any(target == 14 for _, target in edges(brain)), "no edge called it back"
    with torch.no_grad():
        restored, _ = brain.bank.forward_rows(probe, ids)
    assert torch.equal(restored, expected)


def test_newborn_gradients_frozen() -> None:
    """A newborn keeps its weak noise for 200 steps, then may be shaped."""
    torch.manual_seed(0)
    brain = Brain(n_nodes=16, max_hops=2, max_births=0, max_edges=64, node_freeze=200)
    brain.set_edges(torch.tensor([[0], [2]], dtype=torch.long))
    assert int(brain.freeze_newborn_gradients()) == 0  # initial nodes are not frozen

    reallocate_pool(brain, add=4)
    assert all(int(value) == 200 for value in brain.node_frozen_until[16:])
    # C3: the allocated slots are a reserve, not a population. The birth that
    # makes node 16 exist has to come from a message, not from the allocation.
    assert not bool(brain.node_alive[16])
    assert try_birth_node(brain, 0, 16)
    assert bool(brain.node_alive[16])
    assert int(brain.node_births) == 1

    brain.add_edge(16, 1)  # make the newborn part of the readout path
    x, _ = get_batch(1)
    brain.zero_grad(set_to_none=True)
    logits, _ = brain(x, grow=False)
    logits.sum().backward()
    assert float(brain.bank.gate_w.grad[16].abs().sum()) > 0.0

    frozen = brain.freeze_newborn_gradients()
    assert frozen >= 1
    assert float(brain.bank.gate_w.grad[16].abs().sum()) == 0.0
    assert float(brain.bank.gate_w.grad[0].abs().sum()) > 0.0


def test_growth_manager_transitions() -> None:
    """The tier manager reacts to measurements, never to a counter (F8)."""
    brain = Brain(n_nodes=16, max_edges=128)
    manager = GrowthManager(every=500, reallocate=256, suspend_births=5000)

    first = manager.update(100.0, step=0, pool=brain.n_nodes, brain=brain)
    assert first["event"] == "hold"  # no previous measure yet

    growth = manager.update(90.0, step=500, pool=brain.n_nodes, brain=brain)
    assert growth["event"] == "grow"

    optimise = manager.update(89.8, step=1000, pool=brain.n_nodes, brain=brain)
    assert optimise["event"] == "optimize"
    assert manager.births_suspended(1000)
    assert not manager.births_suspended(7000)

    plateau = manager.update(89.8, step=6500, pool=brain.n_nodes, brain=brain)
    assert plateau["event"] == "reallocate"
    assert brain.n_nodes == 16 + 256
    assert "PALIER" in plateau["message"]
    assert manager.tiers[-1]["to"] == brain.n_nodes

    state = manager.state_dict()
    clone = GrowthManager(every=500)
    clone.load_state_dict(state)
    assert clone.tiers == manager.tiers
    assert clone.last_val == manager.last_val


def test_birth_no_grad(monkeypatch) -> None:
    """D1.1: the birth *decision* never enters the autograd graph.

    The edge that a birth creates carries its trainable ``edge_mask`` like any
    other synapse, but the choice of whom to address is a discrete event: it is
    taken with autograd disabled, so nothing of the Gumbel/top-k machinery is
    retained for the backward pass.
    """
    torch.manual_seed(0)
    brain = Brain(n_nodes=32, max_hops=2, max_edges=128, max_births=4, birth_candidates=32)
    # A complete path to the reader: without it the loss does not depend on the
    # wiring at all, and the gradient of every edge mask is exactly zero.
    brain.set_edges(torch.tensor([[0, 2], [2, 1]], dtype=torch.long))
    seen: dict[str, bool] = {}
    # The Gumbel draw happens *inside* the birth pass, so it observes the
    # decorator's no_grad context. A spy on _birth_edges itself would record
    # the caller's context, which is autograd-enabled by design.
    original_gumbel = brain_module._gumbel_like

    def spy(tensor):
        seen["grad"] = torch.is_grad_enabled()
        return original_gumbel(tensor)

    monkeypatch.setattr(brain_module, "_gumbel_like", spy)
    before = brain.ne
    x, _ = get_batch(1)
    brain.zero_grad(set_to_none=True)
    logits, _ = brain(x)  # training mode: births allowed

    assert seen, "the birth pass never ran"
    assert seen["grad"] is False, "the birth decision ran with autograd enabled"
    assert brain.ne > before

    # The newborn synapse is trainable like any other: the gradient reaches it
    # on the forward that actually carries a message through it (D1.2 puts the
    # birth at the end of the forward, so it is used from the next one on).
    brain.zero_grad(set_to_none=True)
    logits, _ = brain(x)
    logits.sum().backward()
    # edge_mask is a property returning a slice view: the gradient accumulates
    # on the leaf parameter itself. The newborn syllables are part of that same
    # parameter, so they are trainable exactly like the seed wiring once they
    # carry a message.
    assert brain._edge_mask.requires_grad
    assert brain._edge_mask.grad is not None
    assert torch.isfinite(brain._edge_mask.grad).all()
    assert float(brain._edge_mask.grad.abs().sum()) > 0.0
    assert brain._edge_mask.grad.numel() == brain.max_edges


def test_birth_once_per_forward(monkeypatch) -> None:
    """D1.2: eight hops do not mean eight birth decisions."""
    torch.manual_seed(0)
    brain = Brain(n_nodes=64, max_hops=8, max_edges=512, max_births=8, birth_candidates=32)
    brain.set_edges(torch.tensor([[0], [2]], dtype=torch.long))
    calls = {"n": 0}
    original = Brain._birth_edges

    def counted(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Brain, "_birth_edges", counted)
    x, _ = get_batch(1)
    brain(x)

    assert calls["n"] == 1, f"{calls['n']} passes de naissance pour un forward à 8 tours"


def test_birth_candidates_bounded(monkeypatch) -> None:
    """D1.3: the birth pass costs the same at 512 and 3 000 nodes."""

    def birth_seconds(n_nodes: int) -> float:
        torch.manual_seed(0)
        brain = Brain(
            n_nodes=n_nodes, max_hops=2, max_edges=4096, max_births=8, birth_candidates=64
        )
        brain.set_edges(torch.tensor([[0], [2]], dtype=torch.long))
        x, _ = get_batch(1)
        brain(x)  # warm-up
        times: list[float] = []
        original = Brain._birth_edges

        def timed(self, *args, **kwargs):
            started = time.perf_counter()
            result = original(self, *args, **kwargs)
            times.append(time.perf_counter() - started)
            return result

        monkeypatch.setattr(Brain, "_birth_edges", timed)
        brain(x)
        monkeypatch.undo()
        return min(times) if times else 0.0

    small = birth_seconds(512)
    large = birth_seconds(3000)
    assert large <= 3.0 * small + 0.005, (
        f"512 nœuds: {small * 1000:.2f} ms, 3000 nœuds: {large * 1000:.2f} ms "
        "— le coût de naissance dépend de la population"
    )


def test_allocated_slots_dead() -> None:
    """C3.1: allocating slots creates no neuron; a message wakes exactly one.

    Before v3, ``grow_population`` set ``alive = torch.ones(new_n)``: a single
    message addressed to node 531 allocated 20 slots and 19 of them existed at
    once, with a birth date and a freeze, violating F3 (the message precedes
    existence) and inflating ``node_alive``, which feeds the redundancy sample
    and the vital signs.
    """
    torch.manual_seed(0)
    brain = Brain(n_nodes=16, max_hops=2, max_births=0, max_edges=128)
    alive_before = int(brain.node_alive.sum())

    allocated = brain.grow_population(36)
    assert allocated == 20
    assert brain.n_nodes == 36
    assert int(brain.node_alive.sum()) == alive_before  # a reserve, not a tier
    assert not bool(brain.node_alive[20])
    assert int(brain.node_births) == 0

    assert try_birth_node(brain, 0, 20)
    assert int(brain.node_alive.sum()) == alive_before + 1
    assert int(brain.node_births) == 1
    assert (0, 20) in edges(brain)


def test_scaling_edges() -> None:
    """C4.4: four times the wiring must not cost more than three times.

    The cost of a step follows the number of *live* edges the wavefront
    reaches, not the population. A super-linear jump here means an accidental
    dense operation has crept into the hot path.
    """

    def best_of(edge_count: int, repeats: int = 3) -> float:
        torch.manual_seed(0)
        brain = Brain(n_nodes=128, max_hops=4, max_births=0, max_edges=4096)
        sources = torch.arange(edge_count) % 126 + 2
        targets = (torch.arange(edge_count) * 7 + 3) % 126 + 2
        brain.set_edges(torch.stack([sources, targets]))
        x = torch.randint(2, 99, (2, 6))
        best = float("inf")
        for _ in range(repeats):
            started = time.perf_counter()
            with torch.no_grad():
                brain(x, grow=False)
            best = min(best, time.perf_counter() - started)
        return best

    small = best_of(500)
    large = best_of(2000)
    assert large <= 3.0 * small + 0.02, (
        f"500 arêtes: {small * 1000:.1f} ms, 2000 arêtes: {large * 1000:.1f} ms "
        "— la croissance n'est pas quasi linéaire"
    )


def test_scale_2_forward() -> None:
    """3 000 nodes stay sparse and cheap: one forward under five CPU seconds."""
    torch.manual_seed(0)
    brain = Brain(n_nodes=3000, max_hops=4, max_births=0, max_edges=4000)
    x, _ = get_batch(2, seq_len=4)
    started = time.time()
    with torch.no_grad():
        logits, _ = brain(x, grow=False)
    elapsed = time.time() - started
    assert logits.shape == (2, 4, 100)
    assert elapsed < 5.0, f"3 000 nodes took {elapsed:.2f}s"
    assert brain.edge_index.shape == (2, 4000)
