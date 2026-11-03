"""Sparse, growable adjacency: an edge list with a continuous mask per link.

There is never an ``[N, N]`` dense matrix here.  An edge is a triple
``(src, dst, logit)`` where ``sigmoid(logit)`` is the structural mask used with
a straight-through estimator, plus two bookkeeping buffers (EMA usage and birth
step).  Links are created, pruned and rewired by rebuilding the flat arrays,
which keeps everything differentiable and cheap.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn


class SparseEdges(nn.Module):
    def __init__(self, capacity: int = 65_536, candidates: int = 6,
                 max_nodes: int = 512):
        super().__init__()
        self.capacity = int(max(capacity, 8))
        self.candidates = int(candidates)
        self.n_edges = 0
        self.register_buffer("src", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("dst", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("usage", torch.zeros(self.capacity))
        self.register_buffer("born", torch.zeros(self.capacity, dtype=torch.long))
        self.logit = nn.Parameter(torch.zeros(self.capacity))
        # Candidate table: for each node, the edge indices of its outgoing links.
        self.register_buffer("cand_edge", torch.full((max_nodes, self.candidates),
                                                     -1, dtype=torch.long))
        self.register_buffer("cand_dst", torch.zeros(max_nodes, self.candidates,
                                                     dtype=torch.long))
        self._dirty = True
        self.resized = False

    # ------------------------------------------------------------------ io
    def mask(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.logit if idx is None else self.logit[idx]
        return torch.sigmoid(logits)

    def _reserve(self, extra: int) -> None:
        need = self.n_edges + int(extra)
        if need <= self.capacity:
            return
        new_cap = max(need, int(self.capacity * 1.6), 8)
        pad = new_cap - self.capacity

        def grow(buf: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
            return torch.cat([buf, torch.full((pad,), fill, dtype=buf.dtype,
                                              device=buf.device)])

        self.src = grow(self.src, 0).long()
        self.dst = grow(self.dst, 0).long()
        self.usage = grow(self.usage, 0.0)
        self.born = grow(self.born, 0).long()
        old_logit = self.logit.data
        self.logit = nn.Parameter(torch.cat([old_logit, torch.zeros(
            pad, dtype=old_logit.dtype, device=old_logit.device)]))
        self.capacity = new_cap
        # The mask Parameter was replaced: the trainer must rebuild its optimiser
        # so the new tensor gets its own moments (rare, only past capacity).
        self.resized = True

    def add(self, src: torch.Tensor, dst: torch.Tensor,
            mask_init: float = 2.0, step: int = 0) -> int:
        """Append links; duplicates already present are skipped."""
        if src.numel() == 0:
            return 0
        src = src.long().flatten()
        dst = dst.long().flatten()
        keep_s, keep_d = [], []
        seen = set()
        if self.n_edges:
            existing = set(zip(self.src[: self.n_edges].tolist(),
                               self.dst[: self.n_edges].tolist()))
        else:
            existing = set()
        for a, b in zip(src.tolist(), dst.tolist()):
            if a == b or (a, b) in existing or (a, b) in seen:
                continue
            seen.add((a, b))
            keep_s.append(a)
            keep_d.append(b)
        if not keep_s:
            return 0
        self._reserve(len(keep_s))
        start = self.n_edges
        stop = start + len(keep_s)
        with torch.no_grad():
            self.src[start:stop] = torch.tensor(keep_s, dtype=torch.long,
                                                device=self.src.device)
            self.dst[start:stop] = torch.tensor(keep_d, dtype=torch.long,
                                                device=self.dst.device)
            self.usage[start:stop] = 0.0
            self.born[start:stop] = int(step)
            self.logit.data[start:stop] = float(mask_init)
        self.n_edges = stop
        self._dirty = True
        return len(keep_s)

    def subset(self, keep: torch.Tensor) -> int:
        """Keep only the selected edges; everything else is deleted.

        The buffers and the mask Parameter are compacted *in place*: replacing
        ``self.logit`` with a new Parameter would silently detach it from the
        optimiser's state, so the storage is kept and only ``n_edges`` moves.
        """
        keep = keep.bool()
        before = self.n_edges
        if before == 0:
            return 0
        idx = torch.nonzero(keep[: before], as_tuple=False).flatten()
        kept = int(idx.numel())
        with torch.no_grad():
            for name in ("src", "dst", "usage", "born"):
                buf = getattr(self, name)
                buf[:kept] = buf[idx]
            self.logit.data[:kept] = self.logit.data[idx].clone()
        self.n_edges = kept
        self._dirty = True
        return before - kept

    # --------------------------------------------------------------- tables
    def rebuild_tables(self, n_nodes: int, generator: Optional[torch.Generator] = None,
                       rotate: bool = True) -> None:
        """Rebuild the per-node candidate table (CSR-like, padded)."""
        self._dirty = False
        self.cand_edge.fill_(-1)
        if self.n_edges == 0:
            return
        src = self.src[: self.n_edges]
        order = torch.argsort(src, stable=True)
        src_sorted = src[order]
        counts = torch.bincount(src_sorted, minlength=n_nodes)
        starts = torch.cumsum(counts, 0) - counts
        if rotate and self.candidates > 0:
            # Rotate the candidate order per rebuild so a node with many links
            # does not always expose the same subset to its router.
            offsets = torch.randint(0, max(1, self.candidates), (n_nodes,),
                                    generator=generator)
        else:
            offsets = torch.zeros(n_nodes, dtype=torch.long)
        rows = min(n_nodes, self.cand_edge.shape[0])
        for node in range(rows):
            count = int(counts[node])
            if count == 0:
                continue
            take = min(count, self.candidates)
            start = int(starts[node])
            rot = int(offsets[node]) % count
            picks = (torch.arange(take) + rot) % count + start
            edges = order[picks]
            self.cand_edge[node, :take] = edges
            self.cand_dst[node, :take] = self.dst[edges]

    def resize_nodes(self, new_max: int) -> None:
        """Widen the candidate table after a pool reallocation."""
        new_max = int(new_max)
        if new_max <= self.cand_edge.shape[0]:
            return
        pad = new_max - self.cand_edge.shape[0]
        self.cand_edge = torch.cat([
            self.cand_edge,
            torch.full((pad, self.candidates), -1, dtype=torch.long,
                       device=self.cand_edge.device),
        ])
        self.cand_dst = torch.cat([
            self.cand_dst,
            torch.zeros((pad, self.candidates), dtype=torch.long,
                        device=self.cand_dst.device),
        ])

    # ---------------------------------------------------------------- stats
    def degree(self, n_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        out_deg = torch.bincount(self.src[: self.n_edges], minlength=n_nodes)
        in_deg = torch.bincount(self.dst[: self.n_edges], minlength=n_nodes)
        return out_deg, in_deg

    def stats(self) -> Dict[str, float]:
        if self.n_edges == 0:
            return {"edges": 0, "mean_mask": 0.0}
        m = self.mask()[: self.n_edges].detach()
        return {
            "edges": int(self.n_edges),
            "mean_mask": float(m.mean()),
            "strong_edges": int((m > 0.5).sum()),
        }
