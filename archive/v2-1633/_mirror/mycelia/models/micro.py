"""The micro-neuron: one small, self-contained AI per node.

A node is deliberately tiny — a gate, a two-layer body with a residual path and
per-node layer norm, a router profile, plus three learned scalars (growth,
confidence, dissipation).  Every tensor is stored with a leading ``max_nodes``
axis so the pool can be reallocated without rebuilding the module or losing a
single weight.

Nodes 0 and 1 are special and immortal:

* ``[e]`` injects the token embedding (no body);
* ``[s]`` aggregates by confidence-weighted attention (no body).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


class NodeBank(nn.Module):
    """Flat parameter store for the whole node pool."""

    def __init__(self, max_nodes: int, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.max_nodes = int(max_nodes)
        self.n_nodes = 0            # live pool size (includes [e] and [s])
        self.dead: torch.Tensor     # bool mask of buried slots

        d, d2 = self.dim, 2 * self.dim
        self.W1 = nn.Parameter(torch.empty(self.max_nodes, d, d2))
        self.b1 = nn.Parameter(torch.zeros(self.max_nodes, d2))
        self.W2 = nn.Parameter(torch.empty(self.max_nodes, d2, d))
        self.b2 = nn.Parameter(torch.zeros(self.max_nodes, d))
        self.ln_w = nn.Parameter(torch.ones(self.max_nodes, d))
        self.ln_b = nn.Parameter(torch.zeros(self.max_nodes, d))
        self.gate_w = nn.Parameter(torch.zeros(self.max_nodes, d))
        self.gate_b = nn.Parameter(torch.full((self.max_nodes,), -1.0))
        self.profile = nn.Parameter(torch.empty(self.max_nodes, d))
        self.conf = nn.Parameter(torch.full((self.max_nodes,), 0.5))
        self.diss = nn.Parameter(torch.zeros(self.max_nodes))     # sigmoid -> <1
        self.growth = nn.Parameter(torch.zeros(self.max_nodes))
        self.register_buffer("birth_step", torch.zeros(self.max_nodes,
                                                       dtype=torch.long))
        self.register_buffer("usage", torch.zeros(self.max_nodes))
        self.register_buffer("dead_mask", torch.zeros(self.max_nodes,
                                                     dtype=torch.bool))
        self.reset_parameters()

    # ------------------------------------------------------------------ init
    def reset_parameters(self) -> None:
        d = self.dim
        with torch.no_grad():
            # Small, signed init: the body starts as a near-identity transform
            # so fresh nodes cannot blow up the signal before they are trained.
            self.W1.normal_(0.0, 1.0 / math.sqrt(d))
            self.W2.normal_(0.0, 1.0 / math.sqrt(2 * d))
            self.profile.normal_(0.0, 1.0)
            self.diss.fill_(1.2)                      # sigmoid(1.2) ~ 0.77: a signal survives hops
            self.gate_b.fill_(0.5)
            self.growth.zero_()
            if self.max_nodes > 2:
                # [e] and [s] are never gated away.
                self.gate_b[0] = 4.0
                self.gate_b[1] = 4.0

    def set_pool(self, n: int) -> None:
        n = int(min(max(n, 2), self.max_nodes))
        self.n_nodes = n

    # ----------------------------------------------------------------- grow
    def grow(self, extra: int) -> int:
        """Enlarge the pool inside the pre-allocated capacity.

        New rows are freshly initialised (zero output influence, neutral gate)
        and marked alive; existing weights are untouched, which is what makes
        reallocation lossless.
        """
        room = self.max_nodes - self.n_nodes
        add = int(min(extra, room))
        if add <= 0:
            return 0
        start = self.n_nodes
        stop = start + add
        with torch.no_grad():
            d = self.dim
            self.W1[start:stop].normal_(0.0, 1.0 / math.sqrt(d))
            self.W2[start:stop].normal_(0.0, 1.0 / math.sqrt(2 * d))
            self.b1[start:stop].zero_()
            self.b2[start:stop].zero_()
            self.ln_w[start:stop].fill_(1.0)
            self.ln_b[start:stop].zero_()
            self.gate_w[start:stop].zero_()
            self.gate_b[start:stop].fill_(0.5)
            self.profile[start:stop].normal_(0.0, 1.0)
            self.conf[start:stop].fill_(0.5)
            self.diss[start:stop].fill_(1.2)
            self.growth[start:stop].zero_()
            self.usage[start:stop].zero_()
            self.dead_mask[start:stop] = False
        self.n_nodes = stop
        return add

    def expand_capacity(self, extra: int) -> int:
        """Grow the *capacity* itself (pool reallocation beyond the buffer)."""
        extra = int(extra)
        if extra <= 0:
            return 0
        old = self.max_nodes
        new = old + extra
        d = self.dim

        def pad(param: nn.Parameter, fill: float = 0.0) -> nn.Parameter:
            shape = (new,) + tuple(param.shape[1:])
            out = torch.empty(shape, dtype=param.dtype, device=param.device)
            out[:old] = param.detach()
            out[old:].fill_(fill)
            return nn.Parameter(out)

        self.W1 = pad(self.W1)
        self.W2 = pad(self.W2)
        self.b1 = pad(self.b1)
        self.b2 = pad(self.b2)
        self.ln_w = pad(self.ln_w, 1.0)
        self.ln_b = pad(self.ln_b)
        self.gate_w = pad(self.gate_w)
        self.gate_b = pad(self.gate_b, 0.5)
        self.profile = pad(self.profile)
        self.conf = pad(self.conf, 0.5)
        self.diss = pad(self.diss, 1.2)
        self.growth = pad(self.growth)
        self.birth_step = F.pad(self.birth_step, (0, extra), value=0)
        self.usage = F.pad(self.usage, (0, extra), value=0.0)
        self.dead_mask = F.pad(self.dead_mask, (0, extra), value=False)
        self.max_nodes = new
        with torch.no_grad():
            self.profile[old:].normal_(0.0, 1.0)
            self.W1[old:].normal_(0.0, 1.0 / math.sqrt(d))
            self.W2[old:].normal_(0.0, 1.0 / math.sqrt(2 * d))
        return extra

    # -------------------------------------------------------------- forward
    def body(self, idx: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the body of the selected nodes on their input signals.

        ``idx`` is ``[R]`` and ``x`` is ``[R, dim]``.  Rows whose gate closes
        return their input untouched and are reported so the caller can skip
        their routing cost.
        """
        gate_logit = (x * self.gate_w[idx]).sum(-1) / math.sqrt(self.dim)
        gate = torch.sigmoid(gate_logit + self.gate_b[idx])
        h = torch.bmm(x.unsqueeze(1), self.W1[idx]).squeeze(1) + self.b1[idx]
        h = F.gelu(h)
        h = torch.bmm(h.unsqueeze(1), self.W2[idx]).squeeze(1) + self.b2[idx]
        mixed = x + h
        mean = mixed.mean(-1, keepdim=True)
        var = mixed.var(-1, keepdim=True, unbiased=False)
        normed = (mixed - mean) / torch.sqrt(var + 1e-5)
        normed = normed * self.ln_w[idx] + self.ln_b[idx]
        # Gate interpolation: a closed gate is a pass-through, never a new signal.
        out = x + gate.unsqueeze(-1) * (normed - x)
        return {"out": out, "gate": gate, "open": gate >= self.cfg_gate_floor}

    cfg_gate_floor: float = 0.1

    def gate_bias(self, idx: torch.Tensor) -> torch.Tensor:
        """Confidence scalar used by [s] to weight incoming signals."""
        return self.conf[idx]

    def dissipation(self, idx: torch.Tensor) -> torch.Tensor:
        """Learned energy factor in (0, 1) applied at every crossing."""
        return torch.sigmoid(self.diss[idx])

    def snapshot_rows(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Copy every learned tensor of the given nodes (cemetery entry)."""
        idx = idx.detach().cpu()
        return {
            "W1": self.W1.data[idx].clone(),
            "W2": self.W2.data[idx].clone(),
            "b1": self.b1.data[idx].clone(),
            "b2": self.b2.data[idx].clone(),
            "ln_w": self.ln_w.data[idx].clone(),
            "ln_b": self.ln_b.data[idx].clone(),
            "gate_w": self.gate_w.data[idx].clone(),
            "gate_b": self.gate_b.data[idx].clone(),
            "profile": self.profile.data[idx].clone(),
            "conf": self.conf.data[idx].clone(),
            "diss": self.diss.data[idx].clone(),
            "growth": self.growth.data[idx].clone(),
        }

    def restore_rows(self, idx: int, snap: Dict[str, torch.Tensor]) -> None:
        """Exact resurrection: the node computes as it did before burial."""
        with torch.no_grad():
            for name, value in snap.items():
                tensor = getattr(self, name)
                tensor.data[idx] = value.to(tensor.device)

    def init_rows(self, idx: torch.Tensor) -> None:
        """Fresh parameters for a reused slot (a recycled newborn)."""
        d = self.dim
        with torch.no_grad():
            self.W1[idx] = torch.randn(len(idx), d, 2 * d) / math.sqrt(d)
            self.W2[idx] = torch.randn(len(idx), 2 * d, d) / math.sqrt(2 * d)
            self.b1[idx] = 0.0
            self.b2[idx] = 0.0
            self.ln_w[idx] = 1.0
            self.ln_b[idx] = 0.0
            self.gate_w[idx] = 0.0
            self.gate_b[idx] = 0.5
            self.profile[idx] = torch.randn(len(idx), d)
            self.conf[idx] = 0.5
            self.diss[idx] = 1.2
            self.growth[idx] = 0.0
            self.usage[idx] = 0.0

    def zero_output_rows(self, idx: torch.Tensor) -> None:
        """Birth initialisation: a newborn must not perturb the network."""
        with torch.no_grad():
            # Zero body output + a nearly closed gate: the newborn passes the
            # signal through untouched until training opens it.
            self.W2[idx] = 0.0
            self.b2[idx] = 0.0
            self.ln_w[idx] = 1.0
            self.ln_b[idx] = 0.0
            self.gate_w[idx] = 0.0
            self.gate_b[idx] = -2.0

    def describe(self, idx: int) -> Dict[str, float]:
        return {
            "gate_bias": float(self.gate_b[idx]),
            "confidence": float(torch.sigmoid(self.conf[idx])),
            "dissipation": float(torch.sigmoid(self.diss[idx])),
            "growth": float(torch.sigmoid(self.growth[idx])),
            "usage": float(self.usage[idx]),
            "alive": bool(self.n_nodes > idx and not self.dead_mask[idx]),
        }
