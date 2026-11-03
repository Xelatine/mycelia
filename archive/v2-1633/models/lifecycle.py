"""Lifecycle: links die first, then nodes, and both can be born again.

Order of operations, every ``life.every`` steps:

1. **Link death** — any link whose EMA usage stays under
   ``link_usage_threshold`` after its grace period is cut, keeping a vital
   minimum wired.
2. **Node isolation then burial** — a node with no incoming *and* no outgoing
   link is marked isolated; if it is still orphaned after
   ``node_grace_steps`` it is buried.  Its parameters go to the cemetery, so a
   wake-up rebuilds the node exactly as it was.
3. **Births** — nodes with a high learned growth score spawn children.  Dead
   slots are recycled first; a newborn starts with zero output influence so it
   cannot destabilise the running network.
4. **Pool reallocation** — when the pool is full the capacity itself grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from mycelia.config import Config


@dataclass
class CemeteryEntry:
    node: int
    step: int
    rows: Dict[str, torch.Tensor]
    birth_step: int
    edges: List[Tuple[int, int, float, float]] = field(default_factory=list)


class Lifecycle:
    def __init__(self, brain, cfg: Config, seed: Optional[int] = None,
                 emit: Optional[Callable[..., None]] = None):
        self.brain = brain
        self.cfg = cfg
        self.lc = cfg.life
        self.emit = emit or (lambda **kwargs: None)
        self.rng = torch.Generator().manual_seed(int(seed if seed is not None else cfg.seed))
        self.isolated: Dict[int, int] = {}
        self.cemetery: Dict[int, CemeteryEntry] = {}
        self.free_slots: List[int] = []
        self.wake_steps: Dict[int, int] = {}
        self.counters = {"births": 0, "deaths": 0, "link_cuts": 0,
                         "isolations": 0, "wakes": 0, "reallocations": 0,
                         "recycled": 0, "edges_born": 0}
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------- helpers
    @property
    def n_nodes(self) -> int:
        return self.brain.nodes.n_nodes

    def is_special(self, node: int) -> bool:
        return node in (self.cfg.model.e_node, self.cfg.model.s_node)

    def live_ids(self) -> List[int]:
        bank = self.brain.nodes
        return [i for i in range(self.n_nodes) if not bool(bank.dead_mask[i])]

    # ---------------------------------------------------------------- tick
    def tick(self, step: int) -> Dict[str, Any]:
        if not self.lc.enabled:
            return {}
        summary: Dict[str, Any] = {}
        summary["link_cuts"] = self.prune_links(step)
        summary["deaths"] = self.bury_orphans(step)
        summary["births"] = self.births(step)
        if any(summary.values()):
            self.history.append({"step": step, **summary})
        return summary

    # --------------------------------------------------------------- links
    def prune_links(self, step: int) -> int:
        adj = self.brain.adj
        n = adj.n_edges
        if n == 0:
            return 0
        usage = adj.usage[:n].clone()
        born = adj.born[:n]
        age = int(step) - born
        weak = (usage < float(self.lc.link_usage_threshold)) & \
               (age > int(self.lc.link_grace_steps))
        protect = adj.dst[:n] == self.cfg.model.s_node
        cuttable = weak & ~protect
        allowed = max(0, n - int(self.lc.min_live_edges))
        share = int(max(1, float(self.lc.max_cut_fraction) * n))
        cut_count = int(min(int(cuttable.sum()), allowed, share))
        if cut_count <= 0:
            return 0
        order = torch.argsort(usage + protect.to(usage.dtype) * 1e9)
        cut_idx = order[:cut_count]
        keep = torch.ones(n, dtype=torch.bool)
        keep[cut_idx] = False
        examples = [(int(adj.src[i]), int(adj.dst[i])) for i in cut_idx[:3]]
        removed = adj.subset(keep)
        adj.rebuild_tables(self.n_nodes, generator=self.rng)
        self.counters["link_cuts"] += removed
        self.emit(type="link_death", step=int(step), cut=removed,
                  remaining=int(adj.n_edges), examples=examples,
                  message=f"pruned {removed} unused links ({adj.n_edges} left)")
        return removed

    # --------------------------------------------------------------- nodes
    def bury_orphans(self, step: int) -> int:
        adj = self.brain.adj
        n = self.n_nodes
        out_deg, in_deg = adj.degree(n)
        buried = 0
        live = len(self.live_ids())
        for node in range(2, n):
            if bool(self.brain.nodes.dead_mask[node]):
                continue
            orphan = int(out_deg[node]) == 0 and int(in_deg[node]) == 0
            if orphan:
                first = self.isolated.get(node)
                if first is None:
                    first = self.isolated[node] = int(step)
                    self.counters["isolations"] += 1
                if int(step) - first > int(self.lc.node_grace_steps):
                    if live <= int(self.lc.min_live_nodes):
                        continue
                    self.emit(type="isolated", step=int(step), node=node,
                              since=first,
                              message=f"node {node} has no link left — awaiting burial")
                    self.bury(node, int(step))
                    buried += 1
                    live -= 1
            elif node in self.isolated:
                since = self.isolated.pop(node)
                self.emit(type="revived", step=int(step), node=node,
                          message=f"node {node} was re-wired after isolation")
                _ = since
        return buried

    def bury(self, node: int, step: int) -> None:
        """Move a node to the cemetery (weights kept for an exact wake)."""
        bank = self.brain.nodes
        adj = self.brain.adj
        n = adj.n_edges
        incident = []
        for i in range(n):
            a, b = int(adj.src[i]), int(adj.dst[i])
            if a == node or b == node:
                incident.append((a, b, float(adj.logit[i]), float(adj.usage[i])))
        if incident:
            keep = torch.ones(n, dtype=torch.bool)
            for a, b, _, _ in incident:
                keep[(adj.src[:n] == a) & (adj.dst[:n] == b)] = False
            adj.subset(keep)
            adj.rebuild_tables(self.n_nodes, generator=self.rng)
        self.cemetery[node] = CemeteryEntry(
            node=node, step=int(step), rows=bank.snapshot_rows(torch.tensor([node])),
            birth_step=int(bank.birth_step[node]), edges=incident)
        with torch.no_grad():
            bank.dead_mask[node] = True
            bank.usage[node] = 0.0
        self.free_slots.append(node)
        self.isolated.pop(node, None)
        self.counters["deaths"] += 1
        self.emit(type="node_death", step=int(step), node=node,
                  message=f"node {node} buried after losing every link")

    def wake(self, node: int, step: int) -> bool:
        """Resurrect a buried node *exactly* as it was."""
        entry = self.cemetery.pop(node, None)
        if entry is None:
            return False
        bank = self.brain.nodes
        bank.restore_rows(node, {k: v[0] for k, v in entry.rows.items()})
        with torch.no_grad():
            bank.dead_mask[node] = False
            bank.birth_step[node] = entry.birth_step
        if node in self.free_slots:
            self.free_slots.remove(node)
        if entry.edges:
            src = torch.tensor([a for a, _, _, _ in entry.edges])
            dst = torch.tensor([b for _, b, _, _ in entry.edges])
            before = self.brain.adj.n_edges
            self.brain.adj.add(src, dst, mask_init=0.0, step=int(step))
            start = before
            with torch.no_grad():
                for offset, (_, _, logit, usage) in enumerate(entry.edges):
                    self.brain.adj.logit.data[start + offset] = logit
                    self.brain.adj.usage[start + offset] = usage
            self.brain.adj.rebuild_tables(self.n_nodes, generator=self.rng)
        self.wake_steps[node] = int(step)
        self.counters["wakes"] += 1
        self.emit(type="wake", step=int(step), node=node,
                  message=f"node {node} woken from the cemetery (exact weights)")
        return True

    # -------------------------------------------------------------- births
    def growth_scores(self) -> torch.Tensor:
        bank = self.brain.nodes
        n = self.n_nodes
        with torch.no_grad():
            growth = torch.sigmoid(bank.growth[:n])
            usage = torch.log1p(bank.usage[:n].clamp(min=0.0))
            novelty = self.brain.novelty[:n].clamp(min=0.0)
            score = growth * (0.25 + usage) * (1.0 + 4.0 * novelty)
            score[:2] = 0.0
            if bank.dead_mask[:n].any():
                score = score.masked_fill(bank.dead_mask[:n], 0.0)
            return score

    def births(self, step: int) -> int:
        if int(step) % max(1, int(self.lc.birth_every)) != 0:
            return 0
        scores = self.growth_scores()
        threshold = float(self.lc.birth_threshold)
        eligible = (scores > threshold).nonzero(as_tuple=False).flatten().tolist()
        if not eligible:
            # Keep the brain explorative: the best-placed node still gets a
            # chance, but only if it has actually earned some usage, otherwise
            # the pool would grow without any selection pressure.
            top = int(torch.argmax(scores).item())
            if float(scores[top]) > 0.15:
                eligible = [top]
        made = 0
        for parent in eligible[: int(self.lc.max_births_per_round)]:
            if self.spawn(int(parent), int(step)) >= 0:
                made += 1
        return made

    def spawn(self, parent: int, step: int) -> int:
        """Create a node addressed by ``parent``; returns the new node id."""
        bank = self.brain.nodes
        adj = self.brain.adj
        if self.free_slots:
            node = self.free_slots.pop(0)
            bank.init_rows(torch.tensor([node]))
            with torch.no_grad():
                bank.dead_mask[node] = False
            self.counters["recycled"] += 1
        else:
            if self.n_nodes >= bank.max_nodes:
                self.reallocate(int(self.lc.birth_every))
            node = bank.n_nodes
            if bank.grow(1) != 1:
                return -1
        bank.zero_output_rows(torch.tensor([node]))
        with torch.no_grad():
            bank.birth_step[node] = int(step)
            bank.usage[node] = 0.0
            self.brain.novelty[node] = 0.0
        n = self.n_nodes
        # A newborn is addressed by its parent, talks to [s] so it has a way out,
        # and reaches one *busy* node: random wiring would keep reviving dead
        # regions of the graph instead of deepening the useful ones.
        with torch.no_grad():
            usage = bank.usage[:n].clone()
            usage[:2] = -1.0
            usage[node] = -1.0
            k = min(n, 8)
            top = torch.topk(usage, k=k).indices.tolist()
        peer = int(top[int(torch.randint(0, max(1, len(top)), (1,),
                                        generator=self.rng))])
        src = torch.tensor([parent, node, node])
        dst = torch.tensor([node, self.cfg.model.s_node, peer])
        added = adj.add(src, dst, mask_init=2.5, step=int(step))
        adj.rebuild_tables(n, generator=self.rng)
        self.counters["births"] += 1
        self.counters["edges_born"] += added
        self.emit(type="birth", step=int(step), node=node, parent=parent,
                  edges=added, pool=n,
                  message=f"node {node} born from {parent} (+{added} links, pool {n})")
        return node

    # -------------------------------------------------------- reallocation
    def reallocate(self, extra: int) -> int:
        """Grow the pool capacity without losing a single weight."""
        extra = int(extra)
        if extra <= 0:
            return 0
        before = self.brain.nodes.max_nodes
        self.brain.nodes.expand_capacity(extra)
        self.brain.adj.resize_nodes(self.brain.nodes.max_nodes)
        self.brain.novelty = torch.cat([
            self.brain.novelty,
            torch.zeros(extra, device=self.brain.novelty.device)])
        self.brain.nodes.grow(extra)
        self.brain.adj.rebuild_tables(self.n_nodes, generator=self.rng)
        self.counters["reallocations"] += 1
        self.emit(type="reallocation", step=self.brain.step, extra=extra,
                  capacity=self.brain.nodes.max_nodes, pool=self.n_nodes,
                  message=f"pool reallocated +{extra} slots "
                          f"({before} → {self.brain.nodes.max_nodes}), pool {self.n_nodes}")
        return extra

    # ---------------------------------------------------------------- state
    def state_dict(self) -> Dict[str, Any]:
        return {
            "isolated": dict(self.isolated),
            "free_slots": list(self.free_slots),
            "counters": dict(self.counters),
            "history": list(self.history[-200:]),
            "cemetery": {
                int(k): {"step": v.step, "birth_step": v.birth_step,
                         "edges": v.edges,
                         "rows": {name: t.clone() for name, t in v.rows.items()}}
                for k, v in self.cemetery.items()
            },
            "wake_steps": dict(self.wake_steps),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.isolated = {int(k): int(v) for k, v in state.get("isolated", {}).items()}
        self.free_slots = [int(v) for v in state.get("free_slots", [])]
        self.counters.update(state.get("counters", {}))
        self.history = list(state.get("history", []))
        self.wake_steps = {int(k): int(v) for k, v in state.get("wake_steps", {}).items()}
        self.cemetery = {}
        for node, entry in state.get("cemetery", {}).items():
            self.cemetery[int(node)] = CemeteryEntry(
                node=int(node), step=int(entry["step"]),
                birth_step=int(entry.get("birth_step", 0)),
                rows={name: t.clone() for name, t in entry["rows"].items()},
                edges=[tuple(e) for e in entry.get("edges", [])])
