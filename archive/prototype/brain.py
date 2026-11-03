"""The brain: a sparse graph of nodes whose only conversation is messages.

There is no layer stack here. The structure is a graph stored as an edge list
``edge_index`` of shape [2, E]; a dense N x N matrix is never allocated. Why:
the target is 140 000 nodes, which would be 19.6 billion dense cells, while the
edge list only stores synapses that actually exist.

The signal is not computed in one pass. It is diffused hop by hop, so a message
travels the graph one synapse per hop. That is what lets loops appear naturally
and what makes learning a routing problem rather than a matrix product.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mycelia.node import GATE_THRESHOLD, Node

DEFAULT_NODES = 64  # enough for structure to emerge, small enough for seconds
DEFAULT_DIM = 32
DEFAULT_VOCAB = 100
DEFAULT_K = 8  # maximum outgoing neighbours a single node can address
DEFAULT_HOPS = 8
DEFAULT_DISSIPATION = 0.9

INPUT_NODE = 0  # the door: only this node receives the token embeddings
OUTPUT_NODE = 1  # the reader: every signal it receives is accumulated


def _batched_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply one separate Linear per row of ``x`` in a single batched matmul.

    ``x`` is [A, M, in], ``weight`` [A, out, in], ``bias`` [A, out]. Grouping
    the nodes this way keeps the per-node Python loop out of the hot path: at
    140 000 nodes a node-at-a-time loop would never finish.
    """
    out = torch.baddbmm(bias.unsqueeze(-1), weight, x.transpose(1, 2))
    return out.transpose(1, 2)


class Brain(nn.Module):
    """A colony of nodes, its sparse wiring, and the diffusion of signal."""

    def __init__(
        self,
        n_nodes: int = DEFAULT_NODES,
        dim: int = DEFAULT_DIM,
        vocab: int = DEFAULT_VOCAB,
        k: int = DEFAULT_K,
        max_hops: int = DEFAULT_HOPS,
        dissipation: float = DEFAULT_DISSIPATION,
        edge_mask_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.dim = dim
        self.vocab = vocab
        self.k = k
        self.max_hops = max_hops

        # Every node is alive from the start; training connects them, it does
        # not create them. Nodes 0 and 1 are the permanent doors of the brain
        # and later phases must never delete them.
        self.nodes = nn.ModuleList(Node(dim=dim, k=k) for _ in range(n_nodes))

        # Tokens become signals here, and signals become token predictions here.
        self.embedding = nn.Embedding(vocab, dim)
        self.output_head = nn.Linear(dim, vocab)

        # The one and only representation of the wiring. A buffer (not a
        # parameter) so it follows .to(device) but is never touched by Adam.
        self.register_buffer("edge_index", self._seed_edges())

        # One strength per synapse. A Parameter because phases 2 and 4 grow it
        # (use it) or kill it (starve it). Its L1 penalty pulls useless links
        # toward zero, which is the preparation for that death.
        self.edge_mask = nn.Parameter(torch.full((self.edge_index.shape[1],), float(edge_mask_init)))

        # Rank of each edge among the outgoing edges of its source: the router
        # slot k it must read. Recomputed whenever the wiring changes.
        self.edge_rank: torch.Tensor
        self.register_buffer("edge_rank", torch.zeros(self.edge_index.shape[1], dtype=torch.long))

        # Resistance of the medium. One traversal costs this factor. It is a
        # buffer today because phase 1 fixes it; later phases learn it.
        self.register_buffer("dissipation", torch.tensor(float(dissipation)))
        self._refresh_edge_rank()

        # Telemetry for the training script, the tests and, later, the
        # dashboard. Detached on purpose: these are readings, not state.
        self.last_out_acc: torch.Tensor | None = None
        self.last_firing_nodes: list[int] = []
        self.last_energy_by_hop: list[float] = []

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _seed_edges(self) -> torch.Tensor:
        """Birth state of the brain: a minimal path, and a silent majority.

        The entry talks to six middle nodes, six middle nodes talk to the exit,
        and one bridge links the two groups. Without that bridge the listed
        seeds leave node 0 and node 1 in two disconnected components, so no
        signal could ever cross and the brain could not learn anything. Nodes
        14..63 start with zero connections: 80% of the colony is still mute,
        waiting for the wiring phases.
        """
        src: list[int] = []
        dst: list[int] = []
        for target in range(2, 8):  # the door speaks to six middle nodes
            src.append(INPUT_NODE)
            dst.append(target)
        src.append(7)  # the single bridge between the two seeded groups
        dst.append(8)
        for source in range(8, 14):  # six middle nodes speak to the reader
            src.append(source)
            dst.append(OUTPUT_NODE)
        return torch.tensor([src, dst], dtype=torch.long)

    def _refresh_edge_rank(self) -> None:
        """Give every edge the rank of its k-th outgoing slot, vectorised."""
        src = self.edge_index[0]
        e = int(src.numel())
        rank = torch.zeros(e, dtype=torch.long, device=src.device)
        if e:
            order = torch.argsort(src, stable=True)
            sorted_src = src[order]
            counts = torch.bincount(sorted_src, minlength=self.n_nodes)
            starts = torch.cumsum(counts, 0) - counts
            positions = torch.arange(e, dtype=torch.long, device=src.device)
            rank_sorted = positions - starts[sorted_src]
            if int(rank_sorted.max()) >= self.k:
                raise ValueError(f"a node has more than k={self.k} outgoing edges")
            rank[order] = rank_sorted
        self.edge_rank = rank

    def set_edges(self, edge_index: torch.Tensor, edge_mask: torch.Tensor | None = None) -> None:
        """Replace the wiring in place. Call this before building an optimizer.

        Phases 2 and 4 mutate the graph constantly, so growing and shrinking
        the edge list has to be a first-class, cheap operation.
        """
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=self.edge_index.device)
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_index.numel() and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= self.n_nodes
        ):
            raise ValueError("edge_index refers to a node outside the colony")
        self.edge_index = edge_index
        mask = (
            torch.ones(edge_index.shape[1], device=edge_index.device)
            if edge_mask is None
            else torch.as_tensor(edge_mask, dtype=torch.float32, device=edge_index.device)
        )
        self.edge_mask = nn.Parameter(mask)
        self._refresh_edge_rank()

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    # ------------------------------------------------------------------
    # Diffusion
    # ------------------------------------------------------------------
    def _node_banks(self) -> dict[str, torch.Tensor]:
        """Stack the per-node weights once per forward pass.

        The nodes stay individual modules (each owns its own body and router),
        but the hot loop needs the weights as one block so that all awake nodes
        are processed by bmm instead of a Python loop.
        """
        nodes = self.nodes
        return {
            "gate_w": torch.stack([n.gate.weight for n in nodes]),
            "gate_b": torch.stack([n.gate.bias for n in nodes]),
            "b1_w": torch.stack([n.body[0].weight for n in nodes]),
            "b1_b": torch.stack([n.body[0].bias for n in nodes]),
            "b2_w": torch.stack([n.body[2].weight for n in nodes]),
            "b2_b": torch.stack([n.body[2].bias for n in nodes]),
            "route_w": torch.stack([n.router.weight for n in nodes]),
            "route_b": torch.stack([n.router.bias for n in nodes]),
        }

    def forward(
        self, tokens: torch.Tensor, max_hops: int | None = None
    ) -> tuple[torch.Tensor, dict[int, list[tuple[int, int]]]]:
        """Diffuse the signal for ``max_hops`` hops and read the exit node.

        Returns the logits [batch, seq_len, vocab] and a trace of every edge
        that actually carried a message, hop by hop, for the traffic view.
        """
        if tokens.dim() != 2:
            raise ValueError("tokens must be a [batch, seq_len] LongTensor")
        hops = self.max_hops if max_hops is None else int(max_hops)
        device = self.edge_index.device
        tokens = tokens.to(device)
        batch, seq_len = tokens.shape
        flat = batch * seq_len

        # The entry signal. Positions are independent for now: each token is
        # its own line of traffic through the same graph.
        h0 = self.embedding(tokens).reshape(flat, self.dim)
        banks = self._node_banks()

        # Only nodes that hold signal are kept: an absent key means "state is
        # zero". This sparsity is what will keep 140 000 mostly-silent nodes
        # affordable.
        state: dict[int, torch.Tensor] = {INPUT_NODE: h0}
        out_acc = torch.zeros(flat, self.dim, device=device, dtype=h0.dtype)
        trace: dict[int, list[tuple[int, int]]] = {}
        fired: set[int] = set()
        energy: list[float] = []

        for hop in range(hops):
            trace[hop] = []
            if not state:
                energy.append(0.0)
                continue

            ids = torch.tensor(sorted(state), dtype=torch.long, device=device)
            x = torch.stack([state[i] for i in sorted(state)])  # [A, flat, dim]

            # The gate is the cheap test: it costs one dot product per node and
            # decides who spends energy on a body at all.
            gate = torch.sigmoid(
                _batched_linear(x, banks["gate_w"][ids], banks["gate_b"][ids])
            )
            awake = (gate >= GATE_THRESHOLD).any(dim=1).squeeze(-1)

            if bool(awake.any()):
                f_ids = ids[awake]
                f_x = x[awake]
                f_gate = gate[awake]

                hidden = F.gelu(_batched_linear(f_x, banks["b1_w"][f_ids], banks["b1_b"][f_ids]))
                out = _batched_linear(hidden, banks["b2_w"][f_ids], banks["b2_b"][f_ids])
                # Where the gate is silent the answer is exactly zero, never a
                # faint echo that would keep travelling for free.
                out = out * f_gate * (f_gate >= GATE_THRESHOLD)
                scores = torch.sigmoid(
                    _batched_linear(f_x, banks["route_w"][f_ids], banks["route_b"][f_ids])
                )

                src, dst = self.edge_index[0], self.edge_index[1]
                # Map node id -> row in the awake batch; -1 means "was silent".
                row_of = torch.full((self.n_nodes,), -1, dtype=torch.long, device=device)
                row_of[f_ids] = torch.arange(f_ids.numel(), device=device)
                rows = row_of[src]
                keep = rows >= 0

                delivered = torch.zeros(
                    self.n_nodes, flat, self.dim, device=device, dtype=h0.dtype
                )
                if bool(keep.any()):
                    rows = rows[keep]
                    strength = self.edge_mask[keep].to(h0.dtype).view(-1, 1, 1)
                    # One message per (edge, position): transformed state,
                    # times the router's wish for that slot, times the synapse
                    # strength. Scatter-add, never an edge loop: 50 million
                    # edges have to stay a single kernel.
                    msg = out[rows] * scores[rows, :, self.edge_rank[keep]].unsqueeze(-1)
                    msg = msg * strength
                    delivered.index_add_(0, dst[keep], msg)
                    trace[hop] = [
                        (int(s), int(t)) for s, t in self.edge_index[:, keep].t().tolist()
                    ]

                # The reader accumulates raw signal; it is read only at the end.
                out_acc = out_acc + delivered[OUTPUT_NODE]

                # A node spends its potential when it fires. What survives the
                # hop is the delivered signal, multiplied by the resistance of
                # the medium. That single factor is what makes a loop fade out
                # by fatigue instead of resonating forever, with no anti-loop
                # rule anywhere.
                resisted = delivered * self.dissipation
                live = (delivered.abs().amax(dim=(1, 2)) > 0).nonzero().flatten().tolist()
                state = {int(n): resisted[int(n)] for n in live}
                energy.append(float(delivered.detach().abs().sum()) * float(self.dissipation))
                fired.update(int(i) for i in f_ids.tolist())
            else:
                state = {}
                energy.append(0.0)

        logits = self.output_head(out_acc).reshape(batch, seq_len, self.vocab)

        self.last_out_acc = out_acc.detach().reshape(batch, seq_len, self.dim)
        self.last_firing_nodes = sorted(fired)
        self.last_energy_by_hop = energy
        return logits, trace


def extract_path(
    trace: dict[int, list[tuple[int, int]]], source: int = INPUT_NODE, target: int = OUTPUT_NODE
) -> list[int] | None:
    """Shortest path that respects hop order, using only edges that fired.

    Used by the training script (and later by the dashboard) to show that the
    signal genuinely crossed the graph rather than skipping it.
    """
    best: dict[int, list[int]] = {source: [source]}
    for hop in sorted(trace):
        reached = dict(best)
        for s, t in trace[hop]:
            if s in best:
                candidate = best[s] + [t]
                if t not in reached or len(candidate) < len(reached[t]):
                    reached[t] = candidate
        best = reached
    return best.get(target)
