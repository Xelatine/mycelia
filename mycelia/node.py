"""One neuron of the colony, and the batched bank that holds thousands of them.

A node is not a layer. It is a small autonomous cell of width ``dim`` that
receives a state vector, decides whether it is worth firing, transforms what it
received, and chooses whom to address. Three organs:

* ``gate``  — a firing threshold. Under 0.1 the node answers with an exact zero
  vector and costs nothing downstream. This is the economy (F10) that makes
  140 000 nodes imaginable.
* ``body``  — the transform. ``dim -> 2*dim -> dim``, GELU, residual, layer
  norm. This is where F4 lives: a node that returns its input unchanged is
  taxed by a soft penalty, never forbidden.
* ``router`` — a *query* and a *key*, never a ``Linear(dim, n_nodes)``. Each
  node emits a query, each node owns a key (its address); the score of an edge
  is ``sigmoid(q . k / sqrt(dim))``. Cost O(dim) per address instead of O(N)
  parameters, and no dense N x N matrix anywhere (D3).

Why two implementations of the same maths: :class:`Node` is the readable
reference (one cell, used in tests and in the documentation), :class:`NodeBank`
is the same equations applied to a gathered batch of cells. The training code
never loops over nodes: at 3 000 cells a Python loop is already too slow, and at
140 000 it is impossible. Both call the same helpers so they cannot drift apart.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Below this firing score a node is silent: exact zeros, no message, no cost.
GATE_THRESHOLD = 0.1
DEFAULT_DIM = 32
LN_EPS = 1e-5

# One bank row per node; the router costs 2 * dim * dim of them.
BANK_SHAPES = ("gate_w", "gate_b", "b1_w", "b1_b", "b2_w", "b2_b", "ln_w", "ln_b", "q_w", "q_b", "k_w", "k_b")
# Organs that compute. The address (query/key) is *not* here: a buried node must
# stay reachable, otherwise no message could ever call it back (F3).
COMPUTE_PARAMS = ("gate_w", "gate_b", "b1_w", "b1_b", "b2_w", "b2_b", "ln_w", "ln_b")


def _blinear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply one separate ``Linear`` per node, without a Python loop.

    ``x`` is [A, B, in], ``weight`` [A, out, in], ``bias`` [A, out]. The batch
    dimension alternates between (nodes, tokens) and (tokens, dim) depending on
    the call, which is why the bias is expanded to [A, 1, out].
    """
    return torch.baddbmm(bias.unsqueeze(-1), weight, x.transpose(1, 2)).transpose(1, 2)


def _layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """LayerNorm over the last dim, with one gain/bias pair per node.

    Written by hand because ``F.layer_norm`` expects 1-D parameters and here
    each row of the batch is a different node with its own gain. LayerNorm is
    what keeps a node's voice bounded when its membrane has accumulated signal
    for thousands of tokens: without it the body would saturate and every node
    would say the same saturated thing.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    normalised = (x - mean) * torch.rsqrt(var + LN_EPS)
    return normalised * weight.unsqueeze(1) + bias.unsqueeze(1)


def bank_shapes(dim: int) -> dict[str, tuple[int, ...]]:
    return {
        "gate_w": (1, dim),
        "gate_b": (1,),
        "b1_w": (2 * dim, dim),
        "b1_b": (2 * dim,),
        "b2_w": (dim, 2 * dim),
        "b2_b": (dim,),
        "ln_w": (dim,),
        "ln_b": (dim,),
        "q_w": (dim, dim),
        "q_b": (dim,),
        "k_w": (dim, dim),
        "k_b": (dim,),
    }


class Node(nn.Module):
    """A single node, as a module. The readable reference implementation."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        super().__init__()
        self.dim = dim
        self.gate = nn.Linear(dim, 1)
        self.body = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
        # The address: a node emits a query, every node owns a key.
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)

    def gate_value(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform what arrived, scaled by the gate, silent below threshold.

        The residual carries the input *un-normalised*: the norm only bounds
        the learned transform. Normalising the whole sum would restore the
        amplitude at every hop, and the dissipation of the medium would never
        be visible -- a loop would ring forever at constant energy.
        """
        gate = self.gate_value(x)
        transformed = x + self.norm(self.body(x))
        out = transformed * gate
        return torch.where(gate >= GATE_THRESHOLD, out, torch.zeros_like(out))

    def route(self, x: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Score every candidate address. sigmoid, never softmax: F2.

        A node may want none of its peers, or several: no distribution is
        imposed, so no central router decides who speaks to whom.
        """
        query = self.query(x)
        return torch.sigmoid((query @ keys.transpose(-1, -2)) / math.sqrt(self.dim))

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class NodeBank(nn.Module):
    """All nodes as stacked tensors, so the hot path is batched matmuls.

    Why not a ``ModuleList`` of Nodes: replacing a parameter is how the pool
    grows, and a list of 140 000 Python modules costs more than the maths it
    does. Growing a bank is one tensor copy per organ, and the first
    ``n`` rows are preserved bit for bit (proved by ``test_reallocate_no_loss``).
    """

    def __init__(self, n_nodes: int, dim: int = DEFAULT_DIM, weak_noise: float = 0.02) -> None:
        super().__init__()
        self.dim = int(dim)
        self.weak_noise = float(weak_noise)
        self.n_nodes = int(n_nodes)
        for name, shape in bank_shapes(self.dim).items():
            self.register_parameter(name, nn.Parameter(torch.empty(self.n_nodes, *shape)))
        self._init_rows(0, self.n_nodes, weak=False)

    # ------------------------------------------------------------------
    # Initialisation and growth
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _init_rows(self, start: int, end: int, weak: bool) -> None:
        """Initialise rows ``[start, end)``.

        ``weak=True`` is the newborn rule: very small noise, never an exact
        zero. An exactly zero node is invisible (its output is zero, nothing
        notices it, no signal shapes it), so it would never specialise.
        """
        if end <= start:
            return
        dim = self.dim
        bound = 1.0 / math.sqrt(dim)
        rows = end - start

        def uniform(shape, limit=bound):
            tensor = torch.empty((rows, *shape), device=self.gate_w.device, dtype=self.gate_w.dtype)
            return tensor.uniform_(-limit, limit)

        def normal(shape, std):
            tensor = torch.empty((rows, *shape), device=self.gate_w.device, dtype=self.gate_w.dtype)
            return tensor.normal_(0.0, std)

        noise = self.weak_noise
        for name in BANK_SHAPES:
            param = getattr(self, name)
            shape = param.shape[1:]
            if weak:
                if name == "ln_w":
                    value = torch.ones((rows, *shape), device=param.device, dtype=param.dtype)
                elif name.endswith("_b"):
                    value = torch.zeros((rows, *shape), device=param.device, dtype=param.dtype)
                else:
                    value = normal(shape, noise)
            else:
                if name == "ln_w":
                    value = torch.ones((rows, *shape), device=param.device, dtype=param.dtype)
                elif name.endswith("_b"):
                    value = torch.zeros((rows, *shape), device=param.device, dtype=param.dtype)
                elif name in ("b1_w", "b2_w"):
                    fan_in = shape[1]
                    limit = 1.0 / math.sqrt(fan_in)
                    value = uniform(shape, limit)
                else:
                    value = uniform(shape)
            param[start:end].copy_(value)

    def grow(self, new_n: int) -> int:
        """Enlarge the bank, keeping every existing row bit for bit."""
        new_n = int(new_n)
        old_n = self.n_nodes
        if new_n <= old_n:
            return 0
        with torch.no_grad():
            for name in BANK_SHAPES:
                old = getattr(self, name)
                new = old.new_empty((new_n, *old.shape[1:]))
                new[:old_n].copy_(old)
                self._parameters[name] = nn.Parameter(new)
            self.n_nodes = new_n
            self._init_rows(old_n, new_n, weak=True)
        return new_n - old_n

    # ------------------------------------------------------------------
    # The hot path
    # ------------------------------------------------------------------
    def forward_rows(self, x: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fire the gathered nodes ``ids`` on their states ``x`` [A, B, dim]."""
        gate = torch.sigmoid(_blinear(x, self.gate_w[ids], self.gate_b[ids]))
        hidden = F.gelu(_blinear(x, self.b1_w[ids], self.b1_b[ids]))
        transformed = _blinear(hidden, self.b2_w[ids], self.b2_b[ids])
        # Residual outside the norm: what a node receives keeps its amplitude,
        # so a loop genuinely loses 10 % of it at every traversal (F5).
        out = (x + _layernorm(transformed, self.ln_w[ids], self.ln_b[ids])) * gate
        # Silence is exact: no faint echo keeps travelling for free.
        out = torch.where(gate >= GATE_THRESHOLD, out, torch.zeros_like(out))
        return out, gate

    def queries(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        return _blinear(x, self.q_w[ids], self.q_b[ids])

    def keys(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        return _blinear(x, self.k_w[ids], self.k_b[ids])

    # ------------------------------------------------------------------
    # Per-node access, for the cemetery, the dashboard and the tests
    # ------------------------------------------------------------------
    def row_state(self, node_id: int) -> dict[str, torch.Tensor]:
        """Exact detached weights of one node, on the CPU."""
        return {
            name: getattr(self, name)[node_id].detach().to("cpu").clone()
            for name in BANK_SHAPES
        }

    def load_row(self, node_id: int, weights: dict[str, torch.Tensor]) -> None:
        """Restore one node's rows. The cemetery promises bit-for-bit revival."""
        with torch.no_grad():
            for name, value in weights.items():
                getattr(self, name)[node_id].copy_(value.to(getattr(self, name).device))

    def zero_row(self, node_id: int) -> None:
        with torch.no_grad():
            for name in BANK_SHAPES:
                getattr(self, name)[node_id].zero_()

    def zero_compute_row(self, node_id: int) -> None:
        """Silence a node while keeping its address: it can still be called."""
        with torch.no_grad():
            for name in COMPUTE_PARAMS:
                getattr(self, name)[node_id].zero_()

    def row_is_zero(self, node_id: int) -> bool:
        with torch.no_grad():
            return all(bool((getattr(self, name)[node_id] == 0).all()) for name in BANK_SHAPES)

    def compute_row_is_zero(self, node_id: int) -> bool:
        with torch.no_grad():
            return all(
                bool((getattr(self, name)[node_id] == 0).all()) for name in COMPUTE_PARAMS
            )
