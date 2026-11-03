"""Graph payloads for the cockpit.

The browser never receives ``N`` nodes or an adjacency matrix: it gets a stable,
subsampled view (important nodes plus their strongest links), a small binned
heatmap of the sparse adjacency, and a sampled similarity matrix between node
router profiles.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch

GOLDEN = math.pi * (3.0 - math.sqrt(5.0))


def _spiral_positions(n: int, radius: float = 1.0) -> List[tuple[float, float]]:
    """Stable golden-angle layout: a node keeps its position between snapshots."""
    out = []
    for i in range(n):
        r = radius * math.sqrt((i + 0.5) / max(1, n))
        theta = i * GOLDEN
        out.append((round(r * math.cos(theta), 4), round(r * math.sin(theta), 4)))
    return out


def node_state(node: int, alive: bool, born: int, step: int, isolated: bool,
               degree: int, usage: float) -> str:
    if not alive:
        return "dead"
    if node in (0, 1):
        return "special"
    if isolated:
        return "isolated"
    if step - born <= 150:
        return "new"
    if degree >= 3 and usage > 0.05:
        return "mature"
    return "active"


def build_graph(brain, cfg, step: int, isolated: Optional[Dict[int, int]] = None,
                max_nodes: int = 420, max_edges: int = 900) -> Dict[str, Any]:
    isolated = isolated or {}
    bank = brain.nodes
    adj = brain.adj
    n = bank.n_nodes
    n_edges = adj.n_edges

    out_deg, in_deg = adj.degree(n)
    degree = out_deg + in_deg
    usage = bank.usage[:n]
    born = bank.birth_step[:n]
    alive = ~bank.dead_mask[:n]

    # Rank nodes: specials first, then the busiest, then births, then the rest.
    score = (degree.float() * 2.0 + usage * 3.0
             + torch.sigmoid(bank.growth[:n]) * 0.5)
    order = torch.argsort(score, descending=True).tolist()
    keep_nodes = [0, 1]
    for node in order:
        if len(keep_nodes) >= max_nodes:
            break
        if node not in keep_nodes:
            keep_nodes.append(int(node))
    keep_set = set(keep_nodes)
    pos = _spiral_positions(max(2, len(keep_nodes)))
    index_of = {node: i for i, node in enumerate(keep_nodes)}

    nodes: List[Dict[str, Any]] = []
    for node in keep_nodes:
        x, y = pos[index_of[node]]
        nodes.append({
            "id": int(node),
            "x": x,
            "y": y,
            "state": node_state(int(node), bool(alive[node]), int(born[node]),
                                int(step), int(node) in isolated,
                                int(degree[node]), float(usage[node])),
            "degree": int(degree[node]),
            "out": int(out_deg[node]),
            "in": int(in_deg[node]),
            "usage": round(float(usage[node]), 4),
            "conf": round(float(torch.sigmoid(bank.conf[node])), 3),
            "growth": round(float(torch.sigmoid(bank.growth[node])), 3),
            "diss": round(float(torch.sigmoid(bank.diss[node])), 3),
            "born": int(born[node]),
            "age": int(step - int(born[node])),
        })

    src = adj.src[:n_edges]
    dst = adj.dst[:n_edges]
    mask = adj.mask()[:n_edges].detach()
    link_usage = adj.usage[:n_edges]
    inside = torch.zeros(n_edges, dtype=torch.bool)
    inside_set = torch.tensor(sorted(keep_set), dtype=torch.long)
    in_src = torch.isin(src, inside_set)
    in_dst = torch.isin(dst, inside_set)
    inside = in_src & in_dst
    weight = mask * (0.25 + link_usage)
    weight = torch.where(inside, weight, torch.zeros_like(weight))
    top = torch.topk(weight, k=min(int(max_edges), n_edges)).indices.tolist() \
        if n_edges else []
    edges = []
    for i in top:
        if float(mask[i]) <= 0.01 or not bool(inside[i]):
            continue
        edges.append({
            "s": index_of[int(src[i])],
            "t": index_of[int(dst[i])],
            "w": round(float(mask[i]), 3),
            "u": round(float(link_usage[i]), 3),
            "strong": bool(float(mask[i]) > 0.5),
        })

    # Binned adjacency heatmap (never the full N x N matrix).
    bins = 44
    heat = torch.zeros(bins, bins)
    if n_edges:
        b_src = (src * bins // max(1, n)).clamp(0, bins - 1)
        b_dst = (dst * bins // max(1, n)).clamp(0, bins - 1)
        flat = b_src * bins + b_dst
        counts = torch.bincount(flat, minlength=bins * bins).float()
        heat = counts.reshape(bins, bins)
        heat = torch.log1p(heat) / max(1e-6, float(torch.log1p(heat).max()))

    # Sampled similarity between router profiles.
    sample_n = min(36, n)
    sample_idx = torch.linspace(0, n - 1, sample_n).long() if n else torch.zeros(0, dtype=torch.long)
    prof = bank.profile[sample_idx].detach()
    prof = torch.nn.functional.normalize(prof, dim=-1)
    sim = (prof @ prof.t()).clamp(-1, 1)

    hist = torch.bincount(degree.clamp(max=20), minlength=21).tolist()

    return {
        "step": int(step),
        "nodes": nodes,
        "edges": edges,
        "heat": [[round(float(v), 3) for v in row] for row in heat],
        "sim": [[round(float(v), 3) for v in row] for row in sim],
        "sim_ids": sample_idx.tolist(),
        "degree_hist": hist,
        "shown_nodes": len(nodes),
        "total_nodes": int(n),
        "total_edges": int(n_edges),
    }


def node_card(brain, node: int, isolated: Dict[int, int], cemetery,
              step: int) -> Dict[str, Any]:
    """Everything the cockpit shows when you click a node."""
    bank = brain.nodes
    adj = brain.adj
    if node >= bank.n_nodes:
        raise KeyError(node)
    out_deg, in_deg = adj.degree(bank.n_nodes)
    n_edges = adj.n_edges
    outgoing, incoming = [], []
    for i in range(n_edges):
        s, t = int(adj.src[i]), int(adj.dst[i])
        if s == node:
            outgoing.append({"to": t, "w": round(float(adj.mask()[i]), 3),
                             "usage": round(float(adj.usage[i]), 3)})
        elif t == node:
            incoming.append({"from": s, "w": round(float(adj.mask()[i]), 3),
                             "usage": round(float(adj.usage[i]), 3)})
    outgoing.sort(key=lambda e: -e["w"])
    incoming.sort(key=lambda e: -e["w"])
    entry = cemetery.get(node) if hasattr(cemetery, "get") else None
    return {
        "id": int(node),
        "state": node_state(int(node), not bool(bank.dead_mask[node]),
                            int(bank.birth_step[node]), int(step),
                            node in isolated, int(out_deg[node] + in_deg[node]),
                            float(bank.usage[node])),
        "born": int(bank.birth_step[node]),
        "age": int(step - int(bank.birth_step[node])),
        "usage": round(float(bank.usage[node]), 4),
        "growth": round(float(torch.sigmoid(bank.growth[node])), 4),
        "confidence": round(float(torch.sigmoid(bank.conf[node])), 4),
        "dissipation": round(float(torch.sigmoid(bank.diss[node])), 4),
        "gate_bias": round(float(bank.gate_b[node]), 4),
        "in_degree": int(in_deg[node]),
        "out_degree": int(out_deg[node]),
        "incoming": incoming[:12],
        "outgoing": outgoing[:12],
        "cemetery": entry is not None,
        "cemetery_step": int(entry.step) if entry is not None else None,
        "weight_norm": round(float(bank.W1[node].norm() + bank.W2[node].norm()), 4),
        "profile_norm": round(float(bank.profile[node].norm()), 4),
    }
