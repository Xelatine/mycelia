"""Life and death of the colony: usage, pruning, burying, exact revival, taxes.

A brain that keeps its wiring forever is a matrix with a nicer name. Three
rules make this one alive, and nothing else does:

* a synapse whose average traffic falls under the threshold is cut;
* a neuron left with no synapse at all is buried, weights and all, so that it
  can come back *exactly* as it was;
* a buried neuron wakes up when a live one addresses it (F3: the message
  precedes existence).

Nothing here detects, forbids or breaks loops. A loop is a legitimate accident
of local routing; the dissipation of the medium is what makes it fade.

Two soft taxes complete the rules of circulation. They are penalties, never
constraints: a node that returns its input unchanged (F4) or that repeats what
other nodes already say (F9) is not forbidden, it is simply less rewarded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import torch
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mycelia.brain import Brain

INPUT_NODE = 0
OUTPUT_NODE = 1
IMMORTAL_NODES = (INPUT_NODE, OUTPUT_NODE)

# Starvation is judged against the traffic of the links that work, not against
# the mean of the whole table. Measured in the live text cockpit: the mean is
# dominated by a handful of hot synapses, so "under a tenth of the mean" chose
# 92 % of the wiring and cut 772 edges in one pass (loss rose from 8.33 to
# 9.12). An absolute 0.02 did the same on text, where message magnitudes are
# an order of magnitude smaller than on the toy task (121 edges and 55 neurons
# lost in one beat). A colony has to be judged against itself, and a single
# pass may never remove more than a tenth of it.
DEFAULT_PRUNE_RATIO = 0.1
DEFAULT_PRUNE_FRACTION = 0.1


class EdgeUsage:
    """Exponential moving average of the signal each synapse carries.

    Why an average: a synapse atrophies from being unused over time, not from
    one silent hop. Decay 0.99 means a link must be fed for tens of forwards
    before it looks healthy, and slowly fades once it stops being used.
    """

    def __init__(self, decay: float = 0.99) -> None:
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, usage: torch.Tensor, edge_ids: torch.Tensor, magnitudes: torch.Tensor) -> None:
        """Fold this hop's traffic into the average, outside the graph."""
        usage.mul_(self.decay)
        if edge_ids.numel():
            usage.index_add_(0, edge_ids, magnitudes * (1.0 - self.decay))


@torch.no_grad()
def prune_edges(
    brain: "Brain", threshold: float | None = None, fraction: float | None = None
) -> int:
    """Cut starved synapses. First time of death (F7).

    By default an edge dies when its average usage is under ``prune_ratio``
    (a tenth) of the traffic of the busiest decile, never more than
    ``prune_fraction`` (a tenth) of the table at once, and never if it is part
    of the seed path 0 -> 2..7 -> 1. That last rule is the same instinct as
    the immortal doors: whatever else starves, a signal must still be able to
    reach the output. Passing a number switches to an absolute cut, which is
    what the tests and a manual debugging session use.

    The surviving links keep their strength and their history; the edge buffer
    is compacted in place, so ``ne`` shrinks and the slots past it are recycled
    by the next birth. This runs between two steps, never during a forward:
    compacting the buffer under a live autograd graph is exactly the in-place
    trap the spec warns about.
    """
    ne = brain.ne
    if ne == 0:
        return 0
    usage = brain.edge_usage[:ne].clone()

    if threshold is None:
        ratio = float(getattr(brain, "prune_ratio", DEFAULT_PRUNE_RATIO))
        share = float(fraction if fraction is not None else getattr(brain, "prune_fraction", DEFAULT_PRUNE_FRACTION))
        busiest = max(1, ne // 10)
        cut = float(usage.topk(busiest).values.mean()) * ratio
        sources = brain.edge_index[0, :ne]
        targets = brain.edge_index[1, :ne]
        protected = torch.zeros(ne, dtype=torch.bool, device=usage.device)
        for source, target in getattr(brain, "seed_pairs", ()):
            protected |= (sources == source) & (targets == target)
        starved = (usage < cut) & ~protected
        budget = max(1, int(round(ne * max(0.0, min(1.0, share)))))
        if int(starved.sum()) > budget:
            # Keep the least starved of the eligible ones: the pass is bounded,
            # so a colony can starve but never collapse in one beat.
            silent = usage.masked_fill(~starved, float("inf"))
            chosen = silent.topk(budget, largest=False).indices
            starved = torch.zeros_like(starved)
            starved[chosen] = True
        keep = ~starved
    else:
        keep = usage >= float(threshold)

    dead = int((~keep).sum())
    if dead == 0:
        return 0

    alive = int(keep.sum())
    brain.edge_index.data[:, :alive].copy_(brain.edge_index[:, :ne][:, keep].data)
    brain._edge_mask.data[:alive].copy_(brain._edge_mask[:ne][keep].detach())
    brain.edge_usage[:alive].copy_(usage[keep])
    # Freed slots restart neutral so a future newborn starts half-open.
    brain._edge_mask.data[alive:ne].fill_(brain.birth_strength)
    brain.edge_usage[alive:ne].zero_()
    brain.ne = alive
    # The buffer was compacted: the host-side membership has to follow, or a
    # later birth would refuse a pair that no longer exists (or the reverse).
    brain._rebuild_edge_keys()
    brain.edge_deaths += dead
    return dead


@torch.no_grad()
def kill_isolated_nodes(brain: "Brain", step: int | None = None, grace: int | None = None) -> list[int]:
    """Bury every node without incoming *or* without outgoing synapse.

    The grace period is what stops the brain from executing its whole newborn
    population on the first checkpoint: a node born 20 steps ago has not had
    the time to be discovered by the router yet. Nodes 0 and 1 are the doors
    and are never buried, whatever the wiring does.
    """
    step = brain.clock if step is None else int(step)
    grace = brain.isolate_grace if grace is None else int(grace)
    touched = torch.zeros(brain.n_nodes, dtype=torch.bool, device=brain.node_alive.device)
    if brain.ne:
        touched[brain.edge_index[0, : brain.ne]] = True
        touched[brain.edge_index[1, : brain.ne]] = True
    has_in = torch.zeros_like(touched)
    has_out = torch.zeros_like(touched)
    if brain.ne:
        has_out[brain.edge_index[0, : brain.ne]] = True
        has_in[brain.edge_index[1, : brain.ne]] = True

    isolated = brain.node_alive & (~has_in | ~has_out)
    old_enough = (step - brain.node_born_step) >= grace
    isolated = isolated & old_enough
    for node_id in IMMORTAL_NODES:
        isolated[node_id] = False

    dead = isolated.nonzero().flatten().tolist()
    for node_id in dead:
        brain.cemetery[int(node_id)] = brain.bank.row_state(int(node_id))
        # Empty the slot: keeping the old weights around would hide a node
        # that still computes while pretending to be dead.
        # Only the firing organs are cleared. The address (query/key) stays in
        # the slot so that a live node can still address this one: that is how
        # a buried node is called back instead of being lost forever.
        brain.bank.zero_compute_row(int(node_id))
        brain.node_alive[node_id] = False
        brain.mean_out[node_id] = 0
        brain.node_deaths += 1
    return dead


@torch.no_grad()
def revive_node(brain: "Brain", node_id: int, source: int | None = None) -> bool:
    """Wake a buried node, bit for bit, and wire it back.

    The promise of the cemetery is not "it starts over": it is "it works again
    like before". Restoring the rows and then comparing the output with
    ``torch.equal`` is exactly what ``test_revival_exact`` does.
    """
    node_id = int(node_id)
    if node_id >= brain.n_nodes or bool(brain.node_alive[node_id]):
        return False
    archived = brain.cemetery.get(node_id)
    if archived is None:
        return False
    brain.bank.load_row(node_id, archived)
    brain.node_alive[node_id] = True
    brain.node_born_step[node_id] = brain.clock
    brain.node_frozen_until[node_id] = brain.clock + brain.node_freeze
    brain.cemetery.pop(node_id, None)
    if source is not None and int(source) != node_id and bool(brain.node_alive[int(source)]):
        # The caller accounts for the birth; this edge is not a second one.
        brain.add_edge(int(source), node_id, count=False)
    return True


def anti_identity_penalty(
    brain: "Brain",
    states: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    threshold: float | None = None,
) -> torch.Tensor:
    """Tax a node that sends back what it received (F4).

    The cosine is taken between each node's input and its output. Only pairs
    above ``threshold`` (0.999 by default) pay, so a genuine transform costs
    nothing and a copy costs its own cosine. Soft by design: penalised, never
    forbidden.
    """
    threshold = brain.identity_cosine if threshold is None else float(threshold)
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]] = brain.last_node_io if states is None else states
    cosines = [F.cosine_similarity(out, inp, dim=-1).reshape(-1) for inp, out in pairs]
    if not cosines:
        return brain.readout_norm.weight.new_zeros(())
    values = torch.cat(cosines)
    copies = values[values > threshold]
    if copies.numel() == 0:
        return values.sum() * 0.0  # keeps the graph, worth exactly zero
    return copies.mean()


def anti_redundancy_penalty(
    brain: "Brain",
    outputs: torch.Tensor | None = None,
    threshold: float | None = None,
    sample: int | None = None,
) -> torch.Tensor:
    """Tax nodes that say the same thing as their neighbours (F9).

    Each node keeps an average of its own output; the penalty is the mean
    positive cosine between those averages, above ``threshold``. Sampling
    keeps it quadratic in a few dozen nodes instead of all 140 000.
    """
    threshold = brain.redundancy_cosine if threshold is None else float(threshold)
    sample = brain.redundancy_sample if sample is None else int(sample)
    means = brain.mean_out if outputs is None else outputs
    alive = brain.node_alive.nonzero().flatten()
    if alive.numel() < 2:
        return means.new_zeros(())
    norms = means[alive].norm(dim=-1)
    active = alive[norms > 1e-6]
    if active.numel() < 2:
        return means.new_zeros(())
    if active.numel() > sample:
        order = means[active].norm(dim=-1).topk(sample).indices
        active = active[order]

    rows = means[active]
    rows = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cosines = rows @ rows.t()
    upper = torch.triu(cosines, diagonal=1)
    excess = torch.relu(upper - threshold)
    count = max(1, active.numel() * (active.numel() - 1) // 2)
    return excess.sum() / count


@torch.no_grad()
def apply_lifecycle(brain: "Brain", step: int | None = None, cfg: dict | None = None) -> dict[str, int]:
    """One beat of the colony: cut, bury, then wake what is wanted again.

    Order matters: cutting starved synapses is what isolates a neuron, so the
    burial has to read the graph the cut just produced. The revived nodes then
    get their edge, and the next forward can reach them.
    """
    life = (cfg or {}).get("lifecycle", cfg or {})
    prune_first = bool(life.get("prune", True))
    dead_edges = prune_edges(brain) if prune_first else 0
    dead_nodes = kill_isolated_nodes(brain, step=step)
    revived = 0
    for node_id in list(brain.pending_revivals):
        if revive_node(brain, node_id):
            revived += 1
    brain.pending_revivals = []
    # pending_node_requests are *new slots*, allocated by growth between two
    # steps (never here, and never inside a forward). They are left untouched.
    return {"edge_deaths": dead_edges, "node_deaths": len(dead_nodes), "revived": revived}
