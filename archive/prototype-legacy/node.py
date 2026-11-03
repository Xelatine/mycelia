"""A single neuron of the colony.

A node is a small autonomous network of width ``dim``. It has no global view:
it reads whatever messages reached it, decides whether it is worth firing, and
routes its own output. The three modules below mirror a biological neuron.

Why the gate threshold: firing costs energy, so a node that received nothing
interesting must cost nothing. At 140 000 nodes, skipping silent nodes is what
keeps the whole brain affordable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# A node whose gate stays under this value produces a zero vector and spends
# no compute. It is the firing threshold of the artificial neuron.
GATE_THRESHOLD = 0.1

DEFAULT_DIM = 32
DEFAULT_K = 8


class Node(nn.Module):
    """One micro-neuron: gate (threshold), body (transform), router (choices)."""

    def __init__(self, dim: int = DEFAULT_DIM, k: int = DEFAULT_K) -> None:
        super().__init__()
        self.dim = dim
        self.k = k

        # Gate: a single scalar per input line. sigmoid keeps it in (0, 1),
        # which reads as "how alive is this node right now".
        self.gate = nn.Linear(dim, 1)

        # Body: inverted bottleneck dim -> 2*dim -> dim. The extra width buys
        # capacity for the transform without growing the per-node state, which
        # is what explodes at brain scale. GELU supplies the non-linearity.
        self.body = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        # Router: K independent scores. sigmoid, never softmax, because a real
        # neuron may address several targets at once or none at all. This is
        # the whole decentralisation of the project: nobody routes globally.
        # Later phases reuse these scores to address a node that does not exist
        # yet, i.e. to make a connection be born.
        self.router = nn.Linear(dim, k)

    def gate_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return the firing score, shaped like the input's trailing dims + 1."""
        return torch.sigmoid(self.gate(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the incoming signal, scaled by the gate.

        Below the threshold the node answers with an exact zero vector, not a
        small one: silence must be free and must not leak into the graph.
        """
        gate = self.gate_value(x)
        out = self.body(x) * gate
        return torch.where(gate >= GATE_THRESHOLD, out, torch.zeros_like(out))

    def route(self, x: torch.Tensor) -> torch.Tensor:
        """Score each of the K outgoing slots: 'how much I want to talk'."""
        return torch.sigmoid(self.router(x))
