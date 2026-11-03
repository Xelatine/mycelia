"""The brain: a sparse graph of nodes whose only conversation is messages.

There is no layer stack here. The structure is an edge list ``edge_index`` of
shape [2, E]; a dense N x N matrix is never allocated. Why: the target is
140 000 nodes, which would be 19.6 billion dense cells, while the edge list only
stores synapses that exist.

The signal is not computed in one global pass. It is diffused hop by hop: node 0
injects the token embedding, a node that receives a message transforms it and
addresses its own targets, the message arrives at the next hop, and the medium
takes 10 % of the energy at every traversal. That is what lets loops appear
naturally and what makes learning a routing problem rather than a matrix
product.

Between two tokens the membrane of each node *persists*. Without that memory the
brain could only ever see the token it is reading; with it, position ``t``
predicts from everything that reached the colony before ``t``. This is the whole
recurrence of the project: no attention, no positional encoding, no layer stack
-- the hops are the time axis, the state is the memory.

Two rules are implemented here rather than elsewhere because they happen inside
the forward, while the signal is moving:

* **a message chooses its destination** (F2): the sender's query is matched
  against the candidates' keys, and the score multiplies the message;
* **speaking makes a node exist** (F3): a target with no synapse yet gets one,
  a buried target is revived, and an address beyond the population is recorded
  so the slot can be allocated between two steps.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mycelia.lifecycle import EdgeUsage, anti_identity_penalty  # noqa: F401  (re-exported)
from mycelia.node import COMPUTE_PARAMS, DEFAULT_DIM, NodeBank

INPUT_NODE = 0  # the door: only this node receives the token embeddings
OUTPUT_NODE = 1  # the reader: its state is what the colony has understood
DEFAULT_NODES = 512
DEFAULT_VOCAB = 100
DEFAULT_HOPS = 8
DEFAULT_EDGES = 20_000
MAX_POPULATION = 140_000


def _gumbel_like(tensor: torch.Tensor) -> torch.Tensor:
    """Gumbel noise: makes a sampled choice instead of an argmax.

    Why not plain top-k: connections must be born by luck, like an axon growing
    toward whatever it bumps into. Without noise every node would always pick
    the same four neighbours and the graph would freeze on the first forward.
    """
    uniform = torch.rand_like(tensor).clamp_min(1e-20)
    return -torch.log(-torch.log(uniform))


class Brain(nn.Module):
    """A colony of nodes, its sparse wiring, the diffusion and the speech."""

    def __init__(
        self,
        n_nodes: int = DEFAULT_NODES,
        dim: int = DEFAULT_DIM,
        vocab: int = DEFAULT_VOCAB,
        max_hops: int = DEFAULT_HOPS,
        dissipation: float = 0.9,
        state_decay: float = 0.995,
        max_edges: int = DEFAULT_EDGES,
        max_births: int = 8,
        max_node_births: int = 2,
        birth_k: int = 4,
        birth_candidates: int = 64,
        init_bias: float = -8.0,
        edge_mask_init: float = 0.5,
        birth_strength: float = 0.5,
        weak_noise: float = 0.02,
        node_margin: int = 32,
        node_birth_interval: int = 500,
        node_freeze: int = 200,
        trace_limit: int = 2000,
        usage_decay: float = 0.99,
        isolate_grace: int = 300,
        prune_threshold: float = 0.02,
        prune_ratio: float = 0.1,
        prune_fraction: float = 0.1,
        identity_cosine: float = 0.999,
        redundancy_cosine: float = 0.5,
        redundancy_sample: int = 64,
        max_population: int = MAX_POPULATION,
        edge_chunk_size: int = 512,
        mmap: bool = False,
    ) -> None:
        super().__init__()
        if n_nodes < 8:
            raise ValueError("a colony needs at least 8 nodes (the 12 seed edges use 0..7)")
        if max_edges < 16:
            raise ValueError("max_edges must leave room for the seed edges")
        self._init_kwargs = {
            "n_nodes": int(n_nodes),
            "dim": int(dim),
            "vocab": int(vocab),
            "max_hops": int(max_hops),
            "dissipation": float(dissipation),
            "state_decay": float(state_decay),
            "max_edges": int(max_edges),
            "max_births": int(max_births),
            "max_node_births": int(max_node_births),
            "birth_k": int(birth_k),
            "birth_candidates": int(birth_candidates),
            "init_bias": float(init_bias),
            "edge_mask_init": float(edge_mask_init),
            "birth_strength": float(birth_strength),
            "weak_noise": float(weak_noise),
            "node_margin": int(node_margin),
            "node_birth_interval": int(node_birth_interval),
            "node_freeze": int(node_freeze),
            "trace_limit": int(trace_limit),
            "usage_decay": float(usage_decay),
            "isolate_grace": int(isolate_grace),
            "prune_threshold": float(prune_threshold),
            "prune_ratio": float(prune_ratio),
            "prune_fraction": float(prune_fraction),
            "identity_cosine": float(identity_cosine),
            "redundancy_cosine": float(redundancy_cosine),
            "redundancy_sample": int(redundancy_sample),
            "max_population": int(max_population),
            "edge_chunk_size": int(edge_chunk_size),
            "mmap": bool(mmap),
        }

        self.n_nodes = int(n_nodes)
        self.dim = int(dim)
        self.vocab = int(vocab)
        self.max_hops = int(max_hops)
        self.max_edges = int(max_edges)
        self.max_births = int(max_births)
        self.max_node_births = int(max_node_births)
        self.birth_k = int(birth_k)
        self.birth_candidates = int(birth_candidates)
        self.init_bias = float(init_bias)
        self.birth_strength = float(birth_strength)
        self.node_margin = int(node_margin)
        self.node_birth_interval = int(node_birth_interval)
        self.node_freeze = int(node_freeze)
        self.trace_limit = int(trace_limit)
        self.isolate_grace = int(isolate_grace)
        self.prune_threshold = float(prune_threshold)
        self.prune_ratio = float(prune_ratio)
        self.prune_fraction = float(prune_fraction)
        self.identity_cosine = float(identity_cosine)
        self.redundancy_cosine = float(redundancy_cosine)
        self.redundancy_sample = int(redundancy_sample)
        self.max_population = int(max_population)
        # D2: messages are accumulated by bounded edge chunks. The peak memory
        # of a hop follows this number, not the size of the wiring.
        self.edge_chunk_size = max(1, int(edge_chunk_size))
        self.birth_passes = 0
        self.mmap = bool(mmap)
        self.dissipation = float(dissipation)
        # A membrane also cools at every tour, for two reasons: a merge of old
        # and new signal must fade, and a state that only accumulates would grow
        # without bound. 0.995 per hop is ~0.96 per token: about twenty tokens
        # of memory. The wavefront is not decayed here (the message pays its own
        # 0.9 per tour), so the loop stability does not depend on this value.
        self.state_decay = float(state_decay)
        self.inv_sqrt_dim = 1.0 / (self.dim**0.5)
        self.allow_node_growth = True

        # All node organs as stacked matrices: one tensor per organ, growable.
        self.bank = NodeBank(self.n_nodes, self.dim, weak_noise=weak_noise)

        # Tokens become signals here, and signals become token predictions here.
        self.embedding = nn.Embedding(self.vocab, self.dim)
        self.output_head = nn.Linear(self.dim, self.vocab)
        # The readout is normalised: node 1's membrane is a sum of everything it
        # ever received, so its scale must not depend on how long we trained.
        self.readout_norm = nn.LayerNorm(self.dim)

        # The one and only representation of the wiring: a preallocated buffer
        # plus a used count. Tests must look at ne, never at the buffer's shape.
        self.register_buffer("edge_index", torch.zeros(2, self.max_edges, dtype=torch.long))
        self.register_buffer("edge_usage", torch.zeros(self.max_edges))
        # One trainable strength per slot. Slots past ne already hold the birth
        # value, so appending an edge never writes into a saved tensor.
        self.register_parameter(
            "_edge_mask", nn.Parameter(torch.full((self.max_edges,), float(edge_mask_init)))
        )
        self.ne = 0
        self.births = 0
        # Real node births (a message woke a slot), never the slots that were
        # merely allocated: C5 keeps the two counters apart on purpose.
        self.node_births = 0
        # Host-side membership of the wiring, so the hot loop never needs a
        # sort-based torch.isin to know whether a pair already exists.
        self._edge_keys: set[int] = set()
        self.edge_deaths = 0
        self.node_deaths = 0
        self.clock = 0
        self.cemetery: dict[int, dict[str, torch.Tensor]] = {}

        # Life bookkeeping. Buffers grow with the population, between steps.
        self.register_buffer("node_alive", torch.ones(self.n_nodes, dtype=torch.bool))
        self.register_buffer("node_born_step", torch.zeros(self.n_nodes, dtype=torch.long))
        # Until this step, a newborn's gradients are masked: its weak noise must
        # not be sharpened by the first loss it happens to see. Initial nodes
        # start unfrozen (0), only real births get the cooling period.
        self.register_buffer("node_frozen_until", torch.zeros(self.n_nodes, dtype=torch.long))
        is_io = torch.zeros(self.n_nodes, dtype=torch.bool)
        is_io[INPUT_NODE] = True
        is_io[OUTPUT_NODE] = True
        self.register_buffer("is_io", is_io)
        # Running average of each node's output: the material of F9.
        self.register_buffer("mean_out", torch.zeros(self.n_nodes, self.dim))

        self.usage_rule = EdgeUsage(decay=usage_decay)
        self.pending_revivals: list[int] = []
        self.pending_node_requests: list[tuple[int, int]] = []
        self._node_request_step = -1
        self.last_births = 0

        # Telemetry (detached readings, never state).
        self.last_out_acc: torch.Tensor | None = None
        self.last_node_io: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.last_energy_by_hop: list[float] = []
        self.last_firing_nodes: list[int] = []
        self.last_hot: torch.Tensor = torch.zeros(0, dtype=torch.long)
        self.last_trace: dict[int, list[tuple[int, int]]] = {}
        self._last_out: torch.Tensor | None = None
        self._last_hot: torch.Tensor = torch.zeros(0, dtype=torch.long)

        self._seed_wiring()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _seed_wiring(self) -> None:
        """The minimal path of D1: 0 -> 2..7 -> 1, exactly 12 edges.

        The old seed (0 -> 2..7 and 8..13 -> 1) left node 0 and node 1 in two
        disconnected components: the signal could never reach the output. The
        corrected seed lets the door speak to six middle nodes, and those six
        speak to the reader. Nodes 8..N-1 start with zero connection: most of
        the colony is mute, waiting for the wiring to reach it (F1).
        """
        mids = list(range(2, 8))
        self.ne = 0
        pairs: list[tuple[int, int]] = []
        for target in mids:
            if self.add_edge(INPUT_NODE, target, count=False):
                pairs.append((INPUT_NODE, target))
        for source in mids:
            if self.add_edge(source, OUTPUT_NODE, count=False):
                pairs.append((source, OUTPUT_NODE))
        # The brainstem. Pruning never cuts these twelve: a colony that loses
        # its only entry->exit path is not a sparse brain, it is a mute one.
        self.seed_pairs = tuple(pairs)

    @property
    def edge_mask(self) -> torch.Tensor:
        """Trainable strength of the synapses that currently exist."""
        return self._edge_mask[: self.ne]

    @property
    def num_edges(self) -> int:
        return self.ne

    def add_edge(self, source: int, target: int, count: bool = True) -> bool:
        """Append one synapse in the free slot. Never touches a used slot.

        The strength of a new link is already in its slot (birth value), so
        nothing trainable is written here: this is the antidote to the
        "modified by an inplace operation" trap.
        """
        source, target = int(source), int(target)
        if self.ne >= self.max_edges:
            return False
        if not (0 <= source < self.n_nodes and 0 <= target < self.n_nodes):
            return False
        if source == target:
            return False
        self.edge_index.data[0, self.ne] = source
        self.edge_index.data[1, self.ne] = target
        # The stride is the population ceiling, not n_nodes: the pool grows and
        # a key must keep meaning the same pair after it does.
        self._edge_keys.add(source * self.max_population + target)
        self.ne += 1
        if count:
            self.births += 1
        return True

    def add_edges(self, sources: Sequence[int], targets: Sequence[int]) -> int:
        created = 0
        for source, target in zip(sources, targets):
            created += int(self.add_edge(source, target))
        return created

    def set_edges(self, edge_index: torch.Tensor, edge_mask: torch.Tensor | None = None) -> None:
        """Replace the wiring. Used by tests, checkpoint loading and the dashboard."""
        tensor = torch.as_tensor(edge_index, dtype=torch.long)
        if tensor.dim() != 2 or tensor.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if tensor.shape[1] > self.max_edges:
            raise ValueError(f"{tensor.shape[1]} edges exceed capacity {self.max_edges}")
        self.ne = int(tensor.shape[1])
        with torch.no_grad():
            self.edge_index[:, : self.ne] = tensor.to(self.edge_index.device)
            if edge_mask is None:
                self._edge_mask[: self.ne].fill_(self.birth_strength)
            else:
                self._edge_mask[: self.ne].copy_(
                    torch.as_tensor(edge_mask, dtype=self._edge_mask.dtype)
                )
            self.edge_usage[: self.ne].zero_()
        self._rebuild_edge_keys()
        self._refresh_birth_counter()

    def _rebuild_edge_keys(self) -> None:
        """Rebuild the host-side membership after a wholesale wiring change."""
        stride = self.max_population
        self._edge_keys = {
            int(source) * stride + int(target)
            for source, target in self.edge_index[:, : self.ne].t().tolist()
        }

    def has_edge(self, source: int, target: int) -> bool:
        return int(source) * self.max_population + int(target) in self._edge_keys

    def _refresh_birth_counter(self) -> None:
        # The seed wiring is not "growth"; only later additions are telemetry.
        self.births = max(0, self.ne - 12)

    # ------------------------------------------------------------------
    # Population growth (between two steps only)
    # ------------------------------------------------------------------
    def grow_population(self, new_n: int, step: int | None = None) -> int:
        """Add node slots, keeping every existing row bit for bit.

        This must never be called inside a forward: it replaces parameters, and
        an autograd graph that saved those parameters would be invalidated.
        """
        new_n = min(int(new_n), self.max_population)
        old_n = self.n_nodes
        if new_n <= old_n:
            return 0
        step = self.clock if step is None else int(step)
        self.bank.grow(new_n)
        with torch.no_grad():
            # C3: the added slots are a *virgin reserve*, not a population. A
            # message that reaches id 531 allocates the slots up to it, and that
            # allocation must not make 19 neurons exist: F3 says the message
            # precedes existence, so only the addressed slot is woken, later, by
            # growth.try_birth_node. Marking them alive here also inflated
            # node_alive, which feeds the redundancy sample and the vitals.
            alive = torch.zeros(new_n, dtype=torch.bool, device=self.node_alive.device)
            alive[:old_n] = self.node_alive
            born = torch.full((new_n,), step, dtype=torch.long, device=self.node_born_step.device)
            born[:old_n] = self.node_born_step
            frozen = torch.zeros(new_n, dtype=torch.long, device=self.node_frozen_until.device)
            frozen[:old_n] = self.node_frozen_until
            frozen[old_n:] = step + self.node_freeze
            is_io = torch.zeros(new_n, dtype=torch.bool, device=self.is_io.device)
            is_io[:old_n] = self.is_io
            mean = torch.zeros(new_n, self.dim, device=self.mean_out.device, dtype=self.mean_out.dtype)
            mean[:old_n] = self.mean_out
        self.register_buffer("node_alive", alive)
        self.register_buffer("node_born_step", born)
        self.register_buffer("node_frozen_until", frozen)
        self.register_buffer("is_io", is_io)
        self.register_buffer("mean_out", mean)
        self.n_nodes = new_n
        return new_n - old_n

    def config(self) -> dict:
        return dict(self._init_kwargs)

    @torch.no_grad()
    def freeze_newborn_gradients(self) -> int:
        """Mask the gradients of nodes born less than ``node_freeze`` steps ago.

        Called between the backward pass and the optimizer step. Without it, a
        newborn's very weak noise is immediately overwritten by whatever the
        current loss wants, which is the opposite of "let it be discovered".
        """
        frozen = (self.node_frozen_until > self.clock).nonzero().flatten()
        if frozen.numel() == 0:
            return 0
        for name in COMPUTE_PARAMS + ("q_w", "q_b", "k_w", "k_b"):
            grad = getattr(self.bank, name).grad
            if grad is not None:
                grad[frozen] = 0
        return int(frozen.numel())

    # ------------------------------------------------------------------
    # Diffusion
    # ------------------------------------------------------------------
    def _fresh_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(
            self.n_nodes,
            batch,
            self.dim,
            device=self.embedding.weight.device,
            dtype=self.embedding.weight.dtype,
        )

    def _fit_state(self, state: torch.Tensor, batch: int) -> torch.Tensor:
        """Adapt a carried state to a population that grew between two steps."""
        if state.shape[0] == self.n_nodes and state.shape[1] == batch:
            return state
        state = state.to(self.embedding.weight.device)
        if state.shape[0] < self.n_nodes:
            pad = state.new_zeros(self.n_nodes - state.shape[0], *state.shape[1:])
            state = torch.cat([state, pad], dim=0)
        elif state.shape[0] > self.n_nodes:
            state = state[: self.n_nodes]
        if state.shape[1] != batch:
            pad = state.new_zeros(self.n_nodes, batch - state.shape[1], self.dim)
            state = torch.cat([state, pad], dim=1)
        return state

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Read the colony's understanding: the reader's membrane, normalised."""
        return self.output_head(self.readout_norm(state[OUTPUT_NODE]))

    @torch.no_grad()
    def _birth_edges(
        self, state: torch.Tensor, out: torch.Tensor, ids: torch.Tensor, budget: int
    ) -> tuple[int, list[tuple[int, int]]]:
        """Open synapses toward peers this node was not wired to (F2, F3).

        Two rules bound where a new synapse may land: the reader never speaks,
        and no synapse lands on either door, so the structure that learns is the
        middle of the brain. The candidate set is *sampled* -- scoring all N
        peers at 140 000 would be the dense matrix the project forbids -- and
        the budget is global to the pass, which is what stops the cascade of
        births that used to explode a run on its first hundred steps.
        """
        if budget <= 0 or self.ne >= self.max_edges or ids.numel() == 0:
            return 0, []
        device = self.edge_index.device
        n = self.n_nodes
        count = min(self.birth_candidates, n)
        cand = torch.randint(0, n, (count,), device=device)
        cand = cand[(cand != INPUT_NODE) & (cand != OUTPUT_NODE)]
        cand = torch.unique(cand)
        if cand.numel() == 0:
            return 0, []

        query = self.bank.queries(out, ids).mean(dim=1)  # [A, dim]
        keys = self.bank.keys(state[cand], cand).mean(dim=1)  # [C, dim]
        scores = torch.sigmoid(query @ keys.t() * self.inv_sqrt_dim + self.init_bias)

        # No self-loop is a definition, not an anti-loop rule. Loops between
        # distinct nodes stay allowed, and are neither detected nor broken.
        # Existing pairs are filtered out of the *selection* below, against the
        # host-side set: a torch.isin over the whole edge table on every hop was
        # a sort per hop, and the profile showed it in the hot path for nothing.
        allowed = ids.unsqueeze(1) != cand.unsqueeze(0)
        noisy = scores.masked_fill(~allowed, -1e9) + _gumbel_like(scores)
        take = min(self.birth_k, cand.numel())
        chosen = noisy.topk(take, dim=1).indices
        rows = torch.arange(ids.numel(), device=device).unsqueeze(1).expand_as(chosen)
        pairs = torch.stack([ids[rows.reshape(-1)], cand[chosen.reshape(-1)]], dim=1)
        # The budget must go to the strongest wishes, not to the lowest ids:
        # a lexicographic cut would silently drop the synapse a node wanted most.
        priority = noisy.gather(1, chosen).reshape(-1)
        order = torch.argsort(priority, descending=True)
        selected: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for source, target in pairs[order].tolist():
            pair = (int(source), int(target))
            if pair in seen or self.has_edge(pair[0], pair[1]):
                continue
            seen.add(pair)
            selected.append(pair)
            if len(selected) >= budget:
                break

        created = 0
        revived = 0
        for source, target in selected:
            if not bool(self.node_alive[target]):
                # The message precedes existence: address a buried node and it
                # is restored, exact weights and all, with the edge that called it.
                from mycelia.lifecycle import revive_node

                if revive_node(self, int(target), source=int(source)):
                    revived += 1
                continue
            if self.add_edge(int(source), int(target), count=False):
                created += 1

        self.births += created + revived
        return created + revived, []

    def forward(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None,
        hops: int | None = None,
        grow: bool | None = None,
        return_state: bool = False,
        trace: bool = True,
        births: int | None = None,
    ):
        """Diffuse the signal over the sequence and read the exit node.

        ``tokens`` is [batch, seq_len]. The state is carried from one position to
        the next, so ``logits[b, t]`` is what the colony predicts *at* position
        ``t`` after having read positions ``0..t-1``. This is why next-token
        prediction needs no shift here, and why the toy copy task (predict the
        previous token) forces the state to survive.

        Returns ``(logits, trace)``, or ``(logits, state, trace)`` when
        ``return_state`` is set (truncated backprop and generation both need it).
        """
        if tokens.dim() != 2:
            raise ValueError("tokens must be a [batch, seq_len] LongTensor")
        device = self.edge_index.device
        tokens = tokens.to(device).long()
        batch, seq_len = tokens.shape
        hops = self.max_hops if hops is None else int(hops)
        grow = self.training if grow is None else bool(grow)
        state = self._fresh_state(batch) if state is None else self._fit_state(state, batch)

        predictions: list[torch.Tensor] = []
        trace_out: dict[int, list[tuple[int, int]]] = {}
        energy_tensors: list[torch.Tensor] = []
        io_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        fired_nodes = torch.zeros(self.n_nodes, dtype=torch.bool, device=device)
        # Nodes that actually emitted a non-silent message during this forward.
        # They are the senders the birth pass (D1.2) may birth *from*; a shut
        # gate emits nothing, so it neither births nor appears as firing.
        emitted = torch.zeros(self.n_nodes, dtype=torch.bool, device=device)
        # The birth budget belongs to the *step*, not to the call: a text step
        # is eight truncated forwards, and resetting the budget at each of them
        # would multiply births by eight (the explosion in slow motion).
        births_left = (self.max_births if births is None else int(births)) if grow else 0
        created_total = 0
        node_requests: list[tuple[int, int]] = []
        idx0 = torch.zeros(1, dtype=torch.long, device=device)

        # Addressing a node that does not exist yet is allowed (F3), but the
        # slot can only be allocated between two steps. The wish is therefore
        # recorded here, rarely (once every node_birth_interval forwards), and
        # honoured by growth.apply_pending_node_births after the optimizer
        # step. One slot per request: a cascade of wished-for nodes would be
        # the unbudgeted-birth explosion in a new costume.
        if (
            grow
            and self.allow_node_growth
            and self.max_node_births > 0
            and self.node_birth_interval > 0
            and self.clock % self.node_birth_interval == 0
            and self.clock != self._node_request_step
            and self.n_nodes < self.max_population
            and births_left > 0
        ):
            room = min(self.max_node_births, births_left, self.max_population - self.n_nodes)
            if room > 0:
                unborn = self.n_nodes + torch.randint(
                    0, max(1, self.node_margin), (room,), device=device
                )
                node_requests = [(INPUT_NODE, int(u)) for u in torch.unique(unborn).tolist()]
                births_left -= len(node_requests)
                self._node_request_step = self.clock

        for position in range(seq_len):
            # Read before injecting: the prediction of token t must not see it.
            predictions.append(self.predict(state))

            injection = self.embedding(tokens[:, position])

            # The door injects into the wavefront, not into the membrane: what
            # circulates is what was just received. A node emits its input, so
            # a loop loses energy instead of accumulating it forever.
            hot = idx0
            wave = injection.unsqueeze(0)
            position_trace: dict[int, list[tuple[int, int]]] = {}
            position_energy: list[torch.Tensor] = []
            position_io: list[tuple[torch.Tensor, torch.Tensor]] = []

            for hop in range(hops):
                if hot.numel() == 0:
                    break
                x = wave
                out, _gate = self.bank.forward_rows(x, hot)
                position_io.append((x, out))
                # Keep the pair (ids, outputs) together: the last hop may have
                # no successor, and the telemetry must not mix two hops.
                self._last_out = out
                self._last_hot = hot

                # Cloned views: the buffer is written by births, and autograd
                # must not see those writes (the in-place trap of the spec).
                src_all = self.edge_index[0, : self.ne].clone()
                tgt_all = self.edge_index[1, : self.ne].clone()
                if src_all.numel() == 0:
                    break

                row_of = torch.full((self.n_nodes,), -1, dtype=torch.long, device=device)
                row_of[hot] = torch.arange(hot.numel(), device=device)
                rows = row_of[src_all]
                # A node whose gate is shut sends exactly nothing: it must not
                # appear in the trace as if it had spoken.
                fired = out.detach().abs().amax(dim=(1, 2)) > 0
                emitted[hot[fired]] = True
                # A buried node receives nothing (its firing organs are zero and
                # it must not wake as a zombie); it is only reachable through a
                # *birth*, which is the one door the cemetery honours.
                keep = (rows >= 0) & fired[rows.clamp_min(0)] & self.node_alive[tgt_all]
                if not bool(keep.any()):
                    break

                eids = keep.nonzero(as_tuple=True)[0]
                rows_e = rows[eids]
                src_e = src_all[eids]
                tgt_e = tgt_all[eids]
                mask_e = self._edge_mask[eids]

                receivers, inverse = torch.unique(tgt_e, return_inverse=True)
                inbox = out.new_zeros(receivers.numel(), batch, self.dim)
                # D2: the scatter runs on bounded edge chunks, so a hop never
                # materialises [E, B, dim] for the whole wiring at once. The
                # maths is unchanged; only the order of the additions moves,
                # which is why the equivalence test uses 1e-6 and not torch.equal.
                chunk = self.edge_chunk_size
                for start in range(0, int(eids.numel()), chunk):
                    stop = min(start + chunk, int(eids.numel()))
                    rows_c = rows_e[start:stop]
                    inv_c = inverse[start:stop]
                    mask_c = mask_e[start:stop]
                    recv_c = receivers[inv_c]
                    key_c = self.bank.keys(state[recv_c], recv_c)
                    query_c = self.bank.queries(out[rows_c], src_e[start:stop])
                    score_c = torch.sigmoid(
                        (query_c * key_c).sum(dim=-1, keepdim=True) * self.inv_sqrt_dim
                    )
                    message_c = (
                        out[rows_c] * score_c * mask_c.view(-1, 1, 1) * self.dissipation
                    )
                    inbox.index_add_(0, inv_c, message_c)
                    if grow:
                        magnitudes = message_c.detach().abs().mean(dim=(1, 2)) * mask_c.detach()
                        self.usage_rule.update(self.edge_usage, eids[start:stop], magnitudes)

                # A signal that received nothing dies here. No sync on values:
                # the decision stays a tensor comparison.
                alive_inbox = inbox.detach().abs().amax(dim=(1, 2)) > 0
                hot = receivers[alive_inbox]
                fired_nodes[hot] = True
                # The membrane cools, then integrates what just arrived. This
                # line is the memory of the colony; the wave itself is replaced
                # by what was delivered, which is the fatigue of the medium.
                state = state * self.state_decay
                if hot.numel():
                    state = state.index_add(0, hot, inbox[alive_inbox])
                wave = inbox[alive_inbox]

                position_energy.append(inbox.detach().abs().sum())
                if trace:
                    pairs = torch.stack([src_e, tgt_e], dim=1)[: self.trace_limit].tolist()
                    position_trace[hop] = [(int(s), int(t)) for s, t in pairs]

            trace_out = position_trace
            energy_tensors = position_energy
            io_pairs = position_io

        # D1.2: ONE birth pass per forward, on the aggregated state of the
        # nodes that fired, never one per hop. A birth is a structural event
        # (F3), and growth is a phase (F8), not a pulse in every tour. The pass
        # is already inside torch.no_grad (D1.1): the decision is discrete, the
        # new edge carries its trainable edge_mask like any other synapse.
        if grow and births_left > 0 and bool(emitted.any()):
            senders = emitted.nonzero(as_tuple=True)[0]
            aggregate = self.mean_out[senders].unsqueeze(1)
            created, requests = self._birth_edges(state, aggregate, senders, births_left)
            births_left -= created + len(requests)
            created_total += created
            node_requests.extend(requests)
            if created or requests:
                self.birth_passes += 1

        logits = torch.stack(predictions, dim=1)
        self.last_out_acc = state[OUTPUT_NODE].detach()
        self.last_node_io = io_pairs
        self.last_energy_by_hop = [float(t) for t in energy_tensors]
        self.last_firing_nodes = sorted(int(i) for i in fired_nodes.nonzero().flatten().tolist())
        self.last_trace = trace_out
        self.last_births = created_total
        if grow and self._last_out is not None and self._last_hot.numel():
            with torch.no_grad():
                average = self._last_out.detach().mean(dim=1)
                slots = self._last_hot
                self.mean_out.index_copy_(
                    0, slots, 0.99 * self.mean_out[slots] + 0.01 * average
                )
        if node_requests:
            self.pending_node_requests.extend(node_requests)
        if return_state:
            return logits, state, trace_out
        return logits, trace_out

    # ------------------------------------------------------------------
    # Speech
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        context_len: int = 128,
        eos_id: int | None = None,
        seed: int | None = None,
        repetition_window: int = 20,
    ) -> tuple[list[int], dict[int, list[tuple[int, int]]]]:
        """Speak, one token at a time, through the living graph.

        The state accumulates as the prompt is read, then every sampled token is
        injected back. The window slides at ``context_len``: when it overflows,
        the state is rebuilt from the last window, so generation never grows
        memory without bound.

        Returns the full token list and the circulation trace of the last token,
        which is what lights up the dashboard and what scripts/generate.py
        prints as "token 'petite' : 0 -> 5 -> 17 -> 1".
        """
        ids = [int(t) for t in prompt_ids] or [0]
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))

        window = ids[-int(context_len) :]
        tokens = torch.tensor([window], dtype=torch.long, device=self.edge_index.device)
        _logits, state, trace = self.forward(tokens, grow=False, return_state=True)

        for _ in range(int(max_new_tokens)):
            step_logits = self.predict(state)[0].float().cpu()
            recent = ids[-int(repetition_window) :]
            step_logits = apply_repetition_penalty(step_logits, recent)
            if temperature <= 0:
                next_id = int(torch.argmax(step_logits))
            else:
                probs = nucleus_probabilities(step_logits / float(temperature), top_p)
                next_id = int(torch.multinomial(probs, 1, generator=generator))
            if not 0 <= next_id < self.vocab:
                next_id = 0
            ids.append(next_id)
            if eos_id is not None and next_id == int(eos_id):
                break
            token = torch.tensor([[next_id]], dtype=torch.long, device=self.edge_index.device)
            _logits, state, trace = self.forward(token, state=state, grow=False, return_state=True)
            if len(ids) > int(context_len):
                window = ids[-int(context_len) :]
                tokens = torch.tensor([window], dtype=torch.long, device=self.edge_index.device)
                _logits, state, trace = self.forward(tokens, grow=False, return_state=True)
        return ids, trace

    @torch.no_grad()
    def generate_token(self, token_id: int, state: torch.Tensor | None, seed: int | None = None):
        """One generation step, exposed for the dashboard's streaming chat."""
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        if state is None:
            state = self._fresh_state(1)
        logits = self.predict(state)[0].float().cpu()
        probs = nucleus_probabilities(logits, 0.9)
        token = int(torch.multinomial(probs, 1, generator=generator))
        new_state, _ = self.forward(
            torch.tensor([[token]], device=self.edge_index.device), state=state, grow=False
        ), None
        return token, new_state[0]


# ----------------------------------------------------------------------
# Sampling helpers (kept out of the model: generation policy, not anatomy)
# ----------------------------------------------------------------------
def apply_repetition_penalty(
    logits: torch.Tensor,
    context: Sequence[int],
    ngram: int = 3,
    threshold: int = 3,
    penalty: float = 2.0,
) -> torch.Tensor:
    """Lower the logit of the token that would close a loop we already hear.

    The rule is the tight one of the spec: only the token that would extend the
    current (n-1)-token suffix into an n-gram already heard ``threshold`` times
    is taxed. Normal text keeps its probabilities; a degenerate loop is simply
    refused the easy exit. This lives at generation, never in the brain: we do
    not cheat by teaching the model about repetition.
    """
    if ngram <= 1 or len(context) < ngram:
        return logits
    suffix = tuple(int(t) for t in context[-(ngram - 1) :])
    counts: dict[int, int] = {}
    for start in range(len(context) - ngram + 1):
        window = tuple(int(t) for t in context[start : start + ngram - 1])
        if window == suffix:
            following = int(context[start + ngram - 1])
            counts[following] = counts.get(following, 0) + 1

    out = logits.clone()
    for token_id, count in counts.items():
        if count >= threshold and 0 <= token_id < out.numel():
            out[token_id] -= penalty
    return out


def nucleus_probabilities(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of tokens that holds ``top_p`` of the mass."""
    probs = F.softmax(logits, dim=-1)
    if not 0.0 < top_p < 1.0:
        return probs
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = (cumulative - sorted_probs) < top_p
    keep[0] = True  # always leave one token to draw from
    kept = torch.zeros_like(probs).scatter(0, sorted_ids, sorted_probs * keep)
    return kept / kept.sum().clamp_min(1e-12)


def extract_path(
    trace: dict[int, list[tuple[int, int]]],
    source: int = INPUT_NODE,
    target: int = OUTPUT_NODE,
) -> list[int] | None:
    """Shortest path that respects hop order, using only edges that fired.

    Used by the training script, the CLI and the dashboard to show that the
    signal genuinely crossed the graph rather than skipping it.
    """
    hops = sorted(trace)
    if not hops:
        return None
    best: dict[int, list[int]] = {source: [source]}
    for hop in range(hops[0], hops[-1] + 1):
        reached = dict(best)
        for s, t in trace.get(hop, []):
            if s in best:
                candidate = best[s] + [t]
                if t not in reached or len(candidate) < len(reached[t]):
                    reached[t] = candidate
        best = reached
    return best.get(target)


def hop_summary(trace: dict[int, list[tuple[int, int]]]) -> dict[int, int]:
    """Number of synapses that carried a message, per hop."""
    return {hop: len(edges) for hop, edges in sorted(trace.items())}
