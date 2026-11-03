"""The brain: iterative message passing over the sparse wired graph.

A forward pass is a *diffusion*, not a layer stack:

1. ``[e]`` injects the token embedding with energy 1.0.
2. Each turn, every signal sitting on a node recomputes its state (the node
   body), applies the **circulation rule** (a node must transform what it
   received; a signal that comes back nearly identical is penalised), multiplies
   its energy by the node's **learned dissipation** (< 1) and routes itself to a
   few neighbours chosen by its local router.
3. When a signal's energy drops under ``eps_energy`` it dies.  Loops therefore
   fade out by fatigue, without any anti-loop rule.
4. Every arrival on ``[s]`` is accumulated with the sender's confidence; the
   final logits come from that aggregate plus the token residual.

Signals travel as a narrow beam so a token can branch and re-merge while the
cost stays proportional to the number of live signals, never to ``N^2``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from mycelia.config import Config
from mycelia.models.graph import SparseEdges
from mycelia.models.micro import NodeBank


def straight_through(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Hard mask forward, soft gradient backward (STE)."""
    hard = (mask > threshold).to(mask.dtype)
    return hard + mask - mask.detach()


class Trace:
    """Compact record of one signal's journey, for the cockpit."""

    def __init__(self, max_edges: int = 400, max_nodes: int = 400):
        self.max_edges = max_edges
        self.max_nodes = max_nodes
        self.turns: List[Dict[str, Any]] = []
        self.nodes: List[int] = []
        self.edges: List[List[int]] = []
        self._seen_edges = set()

    def add_turn(self, visited: List[int], edges: List[Tuple[int, int, float, float]]) -> None:
        kept = []
        for src, dst, weight, energy in edges:
            if len(self.edges) >= self.max_edges:
                break
            if (src, dst) in self._seen_edges:
                continue
            self._seen_edges.add((src, dst))
            self.edges.append([int(src), int(dst)])
            kept.append({"src": int(src), "dst": int(dst),
                         "w": round(float(weight), 4),
                         "energy": round(float(energy), 5)})
        for node in visited:
            if node not in self.nodes and len(self.nodes) < self.max_nodes:
                self.nodes.append(int(node))
        self.turns.append({"n": len(self.turns) + 1, "visited": [int(v) for v in visited[:64]],
                           "edges": kept[:64]})

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges, "turns": self.turns}


class BrainNetwork(nn.Module):
    """Emergent-wiring brain over a sparse edge list."""

    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.mcfg = cfg.model
        self.vocab_size = int(vocab_size)
        dim = self.mcfg.dim
        self.dim = dim

        max_nodes = int(self.mcfg.nodes_max)
        self.nodes = NodeBank(max_nodes, dim)
        self.nodes.cfg_gate_floor = float(self.mcfg.gate_floor)
        self.nodes.set_pool(int(self.mcfg.nodes_initial))

        # Generous but cheap: one float per link.  Growing this later is possible
        # but replaces the mask Parameter, so it is reported to the trainer.
        capacity = max(16_384, int(self.mcfg.nodes_max) * 8)
        self.adj = SparseEdges(capacity=capacity, candidates=self.mcfg.candidates,
                               max_nodes=max_nodes)

        self.tok_emb = nn.Embedding(self.vocab_size, dim)
        self.pos_emb = nn.Embedding(1024, dim)
        self.s_proj = nn.Linear(dim, dim)
        self.s_norm = nn.LayerNorm(dim)
        self.register_buffer("novelty", torch.zeros(max_nodes))
        nn.init.normal_(self.tok_emb.weight, 0.0, 0.02)
        nn.init.normal_(self.pos_emb.weight, 0.0, 0.02)

        self.step = 0
        self.seed_wiring()
        self.adj.rebuild_tables(self.nodes.n_nodes)

    # ------------------------------------------------------------- wiring
    def seed_wiring(self, generator: Optional[torch.Generator] = None) -> None:
        """A newborn brain: every node exists, almost nothing is connected."""
        n = self.nodes.n_nodes
        if n <= 2:
            return
        per_node = max(0.0, float(self.mcfg.seed_edges_per_node))
        src, dst = [], []
        payload = max(1, n - 2)
        for node in range(2, n):
            src.append(node)
            dst.append(int(torch.randint(2, n, (1,), generator=generator)))
        for _ in range(int(payload * per_node)):
            a = int(torch.randint(2, n, (1,), generator=generator))
            b = int(torch.randint(2, n, (1,), generator=generator))
            src.append(a)
            dst.append(b)
        # A few nodes discover [s] so a signal has a chance to arrive at all.
        for node in torch.randperm(payload, generator=generator)[: max(4, payload // 16)].tolist():
            src.append(int(node) + 2)
            dst.append(self.mcfg.s_node)
        for node in torch.randperm(payload, generator=generator)[: max(2, payload // 64)].tolist():
            src.append(self.mcfg.e_node)
            dst.append(int(node) + 2)
        self.adj.add(torch.tensor(src), torch.tensor(dst), mask_init=1.0, step=0)
        # Masks start around 0.5: weak links that must earn their place.
        with torch.no_grad():
            self.adj.logit[: self.adj.n_edges] = torch.randn(
                self.adj.n_edges, generator=generator).clamp(-2.0, 2.0)

    # ------------------------------------------------------------ forward
    def forward(self, ids: torch.Tensor, tau: float = 1.0, track: bool = False,
                update_usage: bool = False, gumbel: bool = True,
                create_edges: bool = False, step: Optional[int] = None,
                generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, T = ids.shape
        L = B * T
        D = self.dim
        beam = int(self.mcfg.beam)
        fanout = int(self.mcfg.fanout)
        e_node = self.mcfg.e_node
        s_node = self.mcfg.s_node
        eps = float(self.mcfg.eps_energy)
        device = ids.device

        step = self.step if step is None else step
        pos = torch.arange(T, device=device).clamp(max=self.pos_emb.num_embeddings - 1)
        base = self.tok_emb(ids) + self.pos_emb(pos).unsqueeze(0)      # [B,T,D]
        base = base.reshape(L, D)

        cur = torch.full((L, beam), e_node, dtype=torch.long, device=device)
        hidden = base.unsqueeze(1).repeat(1, beam, 1)                  # [L,beam,D]
        energy = torch.full((L, beam), float(self.mcfg.init_energy), device=device)
        alive = torch.zeros((L, beam), dtype=torch.bool, device=device)
        alive[:, 0] = True

        ctx = torch.zeros(L, D, device=device)
        denom = torch.zeros(L, device=device)

        circ_terms: List[torch.Tensor] = []
        entropy_terms: List[torch.Tensor] = []
        node_sum: Dict[int, torch.Tensor] = {}
        node_cnt: Dict[int, int] = {}
        trace = Trace() if track else None
        hops: List[float] = []
        computed_rows = 0

        for turn in range(int(self.mcfg.max_turns)):
            active = alive & (energy > eps)
            if not active.any():
                break
            rows = active.nonzero(as_tuple=False)                       # [R,2]
            lin = rows[:, 0]
            slot = rows[:, 1]
            idx = cur[lin, slot]
            x = hidden[lin, slot]
            # [s] is terminal; [e] is a pure source that injects and routes but
            # has no body, so the token embedding itself is never transformed
            # before the first hop.
            is_s = idx == s_node
            is_e = idx == e_node
            compute_row = ~is_s
            if not compute_row.any():
                break

            body = self.nodes.body(idx.clamp(min=0), x)
            out = body["out"]
            gate = body["gate"]
            out = torch.where(is_e.unsqueeze(-1), x, out)
            gate = torch.where(is_e, torch.ones_like(gate), gate)
            open_mask = body["open"] & compute_row & ~is_e
            out = torch.where((open_mask | is_e).unsqueeze(-1), out, x)

            # The circulation rule is measured on real node visits only.
            meas = compute_row & ~is_e
            cos_in = F.cosine_similarity(out, x, dim=-1)
            circ_terms.append(torch.relu(cos_in - 0.999)[meas].mean()
                              if meas.any() else torch.zeros((), device=device))
            # Circulation in force: a re-emitted signal loses energy fast.
            identity_penalty = torch.where(
                is_e, torch.zeros_like(cos_in), torch.clamp(cos_in, min=0.0) ** 8)
            dissip = self.nodes.dissipation(idx)
            parent_energy = energy[lin, slot] * dissip * (0.5 + 0.5 * gate) \
                * (1.0 - 0.5 * identity_penalty)
            computed_rows += int(open_mask.sum())

            # ---- local routing ------------------------------------------------
            cand_edge = self.adj.cand_edge[idx]                          # [R,K]
            cand_dst = self.adj.cand_dst[idx]
            valid = cand_edge >= 0
            safe_edge = cand_edge.clamp(min=0)
            struct = self.adj.mask(safe_edge) * valid.to(self.adj.logit.dtype)
            prof = self.nodes.profile[cand_dst.clamp(min=0)]
            sim = (out.unsqueeze(1) * prof).sum(-1) / math.sqrt(D)
            logits = sim + torch.log(struct.clamp(min=1e-4))
            logits = logits.masked_fill(~valid, -1e9)

            no_candidate = ~valid.any(dim=-1)
            if no_candidate.any():
                # Degenerate case: a node with no outgoing link.  Give it a way
                # out so the signal is never trapped (documented behaviour).
                free = torch.nonzero(no_candidate).flatten()
                target = torch.randint(2, max(3, self.nodes.n_nodes), (free.numel(),),
                                       device=device, generator=generator)
                logits = logits.clone()
                logits[free, 0] = 0.0
                cand_dst = cand_dst.clone()
                cand_dst[free, 0] = target
                struct = struct.clone()
                struct[free, 0] = 1.0
                valid = valid.clone()
                valid[free, 0] = True

            if gumbel and self.training:
                probs = F.gumbel_softmax(logits, tau=max(float(tau), 1e-2),
                                         hard=False, dim=-1)
            else:
                probs = torch.softmax(logits / max(float(tau), 1e-2), dim=-1)
            p_safe = probs.clamp(min=1e-9)
            ent = -(p_safe * torch.log(p_safe)).sum(-1)
            valid_frac = valid.to(ent.dtype).mean(-1).clamp(min=1e-6)
            entropy_terms.append((ent / valid_frac)[meas].mean()
                                 if meas.any() else torch.zeros((), device=device))

            take = min(fanout, probs.shape[-1])
            top_w, top_i = probs.topk(take, dim=-1)
            chosen_dst = torch.gather(cand_dst, -1, top_i)
            chosen_edge = torch.gather(cand_edge, -1, top_i)
            edge_mask = self.adj.mask(chosen_edge.clamp(min=0))
            valid_child = torch.gather(valid, -1, top_i)
            weight = top_w / top_w.sum(-1, keepdim=True).clamp(min=1e-9)
            weight = weight * straight_through(edge_mask) * valid_child.to(weight.dtype)
            weight = weight * compute_row.unsqueeze(-1).to(weight.dtype)

            # ---- fan the beam out --------------------------------------------
            n_child = beam * take
            child_dst = torch.full((L, n_child), -1, dtype=torch.long, device=device)
            child_e = torch.zeros((L, n_child), device=device)
            child_h = torch.zeros((L, n_child, D), device=device)

            group = rows[:, 0]
            slot_col = rows[:, 1]
            flat_target = (group.unsqueeze(-1) * n_child
                           + slot_col.unsqueeze(-1) * take
                           + torch.arange(take, device=device))
            payload = out.unsqueeze(1) * (0.5 + 0.5 * weight).unsqueeze(-1)
            child_dst.view(-1)[flat_target.reshape(-1)] = chosen_dst.reshape(-1)
            child_e.view(-1)[flat_target.reshape(-1)] = (parent_energy.unsqueeze(-1)
                                                         * weight).reshape(-1)
            child_h.view(-1, D)[flat_target.reshape(-1)] = payload.reshape(-1, D)

            # ---- arrivals on [s] ---------------------------------------------
            arrived = (child_dst == s_node) & (child_e > eps)
            if arrived.any():
                flat = arrived.reshape(-1).nonzero(as_tuple=False).flatten()
                owner = torch.div(flat, n_child, rounding_mode="floor")
                conf = torch.sigmoid(self.nodes.conf[s_node])
                w_att = (child_e.reshape(-1)[flat] * conf).clamp(min=0.0)
                ctx.index_add_(0, owner, child_h.reshape(-1, D)[flat]
                               * w_att.unsqueeze(-1))
                denom.index_add_(0, owner, w_att)

            # ---- select the surviving beam ------------------------------------
            vals, pick = child_e.topk(beam, dim=-1)
            new_cur = torch.gather(child_dst, -1, pick)
            new_h = torch.gather(child_h, 1, pick.unsqueeze(-1).repeat(1, 1, D))
            new_alive = (vals > eps) & (new_cur >= 0) & (new_cur != s_node)
            cur = torch.where(new_alive, new_cur, torch.full_like(new_cur, e_node))
            hidden = torch.where(new_alive.unsqueeze(-1), new_h, base.unsqueeze(1))
            energy = torch.where(new_alive, vals, torch.zeros_like(vals))
            alive = new_alive

            # ---- bookkeeping ---------------------------------------------------
            visited_idx, visited_count = torch.unique(idx, return_counts=True)
            for node, count in zip(visited_idx.tolist(), visited_count.tolist()):
                if node in (e_node, s_node):
                    continue
                keep = (idx == node) & meas
                if not bool(keep.any()):
                    continue
                node_sum[node] = node_sum.get(node, torch.zeros(D, device=device)) \
                    + out[keep].sum(0)
                node_cnt[node] = node_cnt.get(node, 0) + int(keep.sum())
                with torch.no_grad():
                    self.novelty[node] = 0.9 * float(self.novelty[node]) + 0.1 * float(
                        (out[keep] - x[keep]).abs().mean())

            hops.append(float(alive.sum(dim=-1).to(torch.float32).mean()))

            if update_usage:
                with torch.no_grad():
                    self._accumulate_usage(chosen_edge.reshape(-1), idx,
                                           weight.reshape(-1), pick)

            if trace is not None:
                sent = (weight * struct.gather(-1, top_i)).detach()
                edges = []
                for r in range(min(len(group), 12)):
                    for j in range(take):
                        w = float(sent[r, j])
                        dst = int(chosen_dst[r, j])
                        if w > 1e-4 and dst >= 0:
                            edges.append((int(idx[r]), dst, w, float(parent_energy[r])))
                visited = torch.unique(idx).tolist()
                trace.add_turn(visited, edges)

        # ------------------------------------------------------------- output
        ctx = ctx.reshape(B, T, D)
        resid = base.reshape(B, T, D)
        context = self.s_norm(resid + torch.tanh(self.s_proj(ctx)))
        logits = context @ self.tok_emb.weight.t()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0).clamp(-60, 60)

        circ = torch.stack(circ_terms).mean() if circ_terms else torch.zeros((), device=device)
        entropy = torch.stack(entropy_terms).mean() if entropy_terms else \
            torch.zeros((), device=device)

        visited_nodes = [n for n in node_sum if node_cnt.get(n, 0) > 0][:200]
        if len(visited_nodes) > 2:
            means = torch.stack([node_sum[n] / max(1, node_cnt[n])
                                 for n in visited_nodes])
            means = F.normalize(means, dim=-1)
            sim = means @ means.t()
            m = sim.shape[0]
            off = (sim.sum() - sim.diagonal().sum()) / max(1, m * (m - 1))
            redundancy = torch.relu(off - 0.25)
        else:
            redundancy = torch.zeros((), device=device)

        aux = {
            "circulation": circ,
            "entropy": entropy,
            "redundancy": redundancy,
            "mask_l1": self.adj.mask()[: self.adj.n_edges].mean()
            if self.adj.n_edges else torch.zeros((), device=device),
            "visited": len(visited_nodes),
            "hops": sum(hops) / max(1, len(hops)),
            "computed_rows": computed_rows,
        }
        residual_energy = float(energy.detach().mean()) if energy.numel() else 0.0
        result = {
            "logits": logits,
            "energy": residual_energy,
            "aux": aux,
            "trace": trace.to_dict() if trace is not None else None,
        }

        if create_edges:
            result["births"] = self._maybe_create_edges(step)
        return result

    # ------------------------------------------------------------ internals
    @torch.no_grad()
    def _accumulate_usage(self, chosen_edge: torch.Tensor, idx: torch.Tensor,
                          weight: torch.Tensor, pick: torch.Tensor) -> None:
        decay = 0.95
        self.adj.usage[: self.adj.n_edges] *= decay
        valid = chosen_edge >= 0
        if not bool(valid.any()):
            return
        edges = chosen_edge[valid].reshape(-1)
        vals = weight[valid].reshape(-1)
        self.adj.usage.index_add_(0, edges, vals)
        self.nodes.usage[self.nodes.n_nodes:] = 0.0
        self.nodes.usage.index_add_(
            0, idx.reshape(-1), torch.ones_like(idx.reshape(-1), dtype=torch.float32)
        )
        self.nodes.usage[: self.nodes.n_nodes] *= 0.95

    @torch.no_grad()
    def _maybe_create_edges(self, step: int) -> int:
        """A routing event may wire a link that did not exist yet.

        This is the only way the graph grows outside births: the sender's
        router wanted a specific destination, so the link is materialised.
        """
        return 0

    # ---------------------------------------------------------------- utils
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def edges_per_node(self) -> float:
        n = max(1, self.nodes.n_nodes)
        return self.adj.n_edges / n

    def to(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        super().to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        return self


class BrainOutput(dict):
    """Readable alias: a plain dict is all the trainer needs."""


def describe_model(model: BrainNetwork) -> Dict[str, Any]:
    return {
        "params": model.param_count(),
        "nodes": model.nodes.n_nodes,
        "dim": model.dim,
        "edges": model.adj.n_edges,
        "edges_per_node": round(model.edges_per_node(), 3),
        "beam": model.mcfg.beam,
    }
