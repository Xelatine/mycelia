"""Growth: a bigger pool, brand-new nodes, and the tier manager that decides.

The colony does not grow because a counter reached a number. It grows because
the *measurements* say it is stuck (F8): improving validation means the current
brain is still learning and should be left alone; a plateau means optimisation;
a flat or worsening curve means the pool itself is too small, so slots are
reallocated and the colony gets a new tier.

All allocations happen **between two steps**. Growing a node bank replaces
parameters, and an autograd graph that saved those parameters cannot survive
that; doing it inside a forward is the in-place trap in yet another costume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mycelia.brain import Brain
    from mycelia.train import Trainer

EVENTS = ("hold", "grow", "optimize", "reallocate")


def try_birth_node(brain: "Brain", source: int, target_id: int) -> bool:
    """Allocate the slot a message asked for, then wire it.

    ``target_id`` may be beyond the population: that is the whole point of F3.
    The router can address a node that does not exist yet; the slot is created
    at the next checkpoint, with very weak noise (never exact zero: a zero node
    would be invisible and no signal could ever shape it).
    """
    target_id = int(target_id)
    if target_id >= brain.max_population or target_id < 2:
        return False
    if target_id >= brain.n_nodes:
        brain.grow_population(target_id + 1, step=brain.clock)
    brain.node_alive[target_id] = True
    brain.node_born_step[target_id] = brain.clock
    brain.node_frozen_until[target_id] = brain.clock + brain.node_freeze
    if int(source) != target_id:
        brain.add_edge(int(source), target_id, count=False)
    brain.births += 1
    # A real birth: this slot was *woken by a message*, not merely allocated.
    brain.node_births += 1
    return True


def reallocate_pool(brain: "Brain", add: int = 256, step: int | None = None) -> tuple[int, int]:
    """Append node slots, changing nothing about the ones already alive.

    The first ``n`` rows are copied bit for bit, so every existing output is
    unchanged until the new slots are actually reached by a message. That is
    what ``test_reallocate_no_loss`` checks with ``torch.equal``.
    """
    old = brain.n_nodes
    new = min(old + int(add), brain.max_population)
    step = brain.clock if step is None else int(step)
    brain.grow_population(new, step=step)
    return old, brain.n_nodes


def apply_pending_node_births(brain: "Brain", max_new: int | None = None) -> list[int]:
    """Honour the addresses the router tried to reach during the last forward."""
    requests = list(brain.pending_node_requests)
    brain.pending_node_requests = []
    if not requests:
        return []
    created: list[int] = []
    seen: set[int] = set()
    for source, target in requests[: max_new or len(requests)]:
        target = int(target)
        if target in seen:
            continue
        seen.add(target)
        if try_birth_node(brain, source, target):
            created.append(target)
    return created


class GrowthManager:
    """Reads validation, decides the next tier, and records why.

    Improvement is the relative drop of validation perplexity since the last
    check:

    * above 1 %  -> **grow**: the brain is still learning, let it wire more;
    * under 0.5 % (but positive) -> **optimise**: births are suspended for
      5 000 steps, the graph is allowed to settle;
    * flat or worse -> **reallocate**: the pool is the bottleneck, add 256
      slots and log the tier.

    The thresholds come from ``configs/config.yaml`` and are meant to be
    loosened before anything else when a run refuses to grow.
    """

    def __init__(
        self,
        every: int = 500,
        grow_threshold: float = 0.01,
        optimize_threshold: float = 0.005,
        reallocate: int = 256,
        suspend_births: int = 5000,
        max_population: int = 140_000,
    ) -> None:
        self.every = int(every)
        self.grow_threshold = float(grow_threshold)
        self.optimize_threshold = float(optimize_threshold)
        self.reallocate = int(reallocate)
        self.suspend_births = int(suspend_births)
        self.max_population = int(max_population)
        self.last_val: float | None = None
        self.suspended_until = 0
        self.last_check = -1
        self.tiers: list[dict] = []
        self.history: list[dict] = []

    def births_suspended(self, step: int) -> bool:
        return int(step) < self.suspended_until

    def update(self, val_ppl: float, step: int, pool: int, brain: "Brain | None" = None) -> dict:
        """One measurement, one decision. Always returns an event."""
        val = float(val_ppl)
        event = "hold"
        improvement = 0.0
        if self.last_val is not None and self.last_val > 0:
            improvement = (self.last_val - val) / self.last_val
            if improvement > self.grow_threshold:
                event = "grow"
            elif improvement <= 0:
                event = "reallocate"
            elif improvement <= self.optimize_threshold:
                event = "optimize"
        # Every event says exactly what it does (C6). "grow" opens the births,
        # it does not create a tier; only "reallocate" touches the pool.
        message = ""
        if event == "hold" and self.last_val is None:
            message = (
                f"référence de validation posée à step {step} (val={val:.1f}) — "
                f"première décision au pas {int(step) + self.every}"
            )
        elif event == "optimize":
            self.suspended_until = int(step) + self.suspend_births
            message = (
                f"phase OPTIMISATION à step {step} (val={val:.1f}) : naissances suspendues"
            )
        elif event == "reallocate":
            old = pool
            if brain is not None and brain.n_nodes < brain.max_population:
                old, new = reallocate_pool(brain, self.reallocate, step=step)
                self.tiers.append({"step": int(step), "from": old, "to": new, "val": val})
                message = f"PALIER : pool {old} → {new} à step {step} (val={val:.1f})"
            else:
                message = f"pool au plafond ({pool}) à step {step} (val={val:.1f})"
        elif event == "grow":
            message = (
                f"phase CROISSANCE à step {step} (val={val:.1f}, gain {improvement:.1%}) : "
                "naissances ouvertes"
            )

        record = {
            "step": int(step),
            "event": event,
            "improvement": improvement,
            "val": val,
            "pool": int(pool),
            "message": message,
        }
        self.history.append(record)
        self.last_val = val
        self.last_check = int(step)
        return record

    def state_dict(self) -> dict:
        return {
            "last_val": self.last_val,
            "suspended_until": self.suspended_until,
            "last_check": self.last_check,
            "tiers": self.tiers,
            "history": self.history,
        }

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        self.last_val = state.get("last_val")
        self.suspended_until = int(state.get("suspended_until", 0))
        self.last_check = int(state.get("last_check", -1))
        self.tiers = list(state.get("tiers", []))
        self.history = list(state.get("history", []))


def apply_growth(trainer: "Trainer", val_ppl: float) -> dict:
    """Glue used by the training loop and the dashboard."""
    manager = trainer.growth
    record = manager.update(val_ppl, trainer.step, trainer.brain.n_nodes, trainer.brain)
    trainer.brain.allow_node_growth = not manager.births_suspended(trainer.step)
    return record
