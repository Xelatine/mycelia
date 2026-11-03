"""Full-genome checkpoints: weights, wiring, lifecycle, tier, data position.

A checkpoint is everything needed to continue *exactly*: node parameters for the
live pool, every edge with its mask and usage, the cemetery, the tier state, the
optimiser moments, both RNG streams and the step counter.  Nothing is
recomputed from scratch on ``--resume``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

NODE_PARAMS = ("W1", "W2", "b1", "b2", "ln_w", "ln_b", "gate_w", "gate_b",
               "profile", "conf", "diss", "growth")
NODE_BUFFERS = ("birth_step", "usage", "dead_mask")


def _slice_nodes(brain) -> Dict[str, torch.Tensor]:
    bank = brain.nodes
    n = bank.n_nodes
    out: Dict[str, torch.Tensor] = {}
    for name in NODE_PARAMS:
        out[name] = getattr(bank, name).data[:n].detach().cpu().clone()
    for name in NODE_BUFFERS:
        out[name] = getattr(bank, name)[:n].detach().cpu().clone()
    return out


def rng_state() -> Dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    try:
        torch.set_rng_state(state["torch"])
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
    except Exception:
        pass


def save_checkpoint(path: str | Path, brain, optimizer=None, lifecycle=None,
                    tier=None, step: int = 0, tau: float = 1.0,
                    data_rng: Optional[np.random.Generator] = None,
                    extra: Optional[Dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adj = brain.adj
    payload: Dict[str, Any] = {
        "format": "mycelia-genome-v1",
        "step": int(step),
        "tau": float(tau),
        "dim": brain.dim,
        "vocab_size": brain.vocab_size,
        "config": brain.cfg.to_dict(),
        "n_nodes": int(brain.nodes.n_nodes),
        "capacity": int(brain.nodes.max_nodes),
        "nodes": _slice_nodes(brain),
        "novelty": brain.novelty[: brain.nodes.n_nodes].detach().cpu().clone(),
        "edges": {
            "n": int(adj.n_edges),
            "src": adj.src[: adj.n_edges].detach().cpu().clone(),
            "dst": adj.dst[: adj.n_edges].detach().cpu().clone(),
            "usage": adj.usage[: adj.n_edges].detach().cpu().clone(),
            "born": adj.born[: adj.n_edges].detach().cpu().clone(),
            "logit": adj.logit.data[: adj.n_edges].detach().cpu().clone(),
        },
        "globals": {
            "tok_emb": brain.tok_emb.weight.detach().cpu().clone(),
            "pos_emb": brain.pos_emb.weight.detach().cpu().clone(),
            "s_proj_w": brain.s_proj.weight.detach().cpu().clone(),
            "s_proj_b": brain.s_proj.bias.detach().cpu().clone(),
            "s_norm_w": brain.s_norm.weight.detach().cpu().clone(),
            "s_norm_b": brain.s_norm.bias.detach().cpu().clone(),
        },
        "lifecycle": lifecycle.state_dict() if lifecycle else None,
        "tier": tier.state_dict() if tier else None,
        "rng": rng_state(),
        "data_rng": data_rng.bit_generator.state if data_rng is not None else None,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def read_checkpoint(path: str | Path) -> Dict[str, Any]:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def apply_payload(brain, payload: Dict[str, Any], optimizer=None,
                  lifecycle=None, tier=None) -> Dict[str, Any]:
    """Restore a payload into ``brain`` in place (keeps Parameter objects)."""
    bank = brain.nodes
    need = int(payload["n_nodes"])
    capacity = int(payload.get("capacity", need))
    if max(need, capacity) > bank.max_nodes:
        bank.expand_capacity(max(need, capacity) - bank.max_nodes)
        brain.adj.resize_nodes(bank.max_nodes)
    bank.set_pool(need)

    with torch.no_grad():
        for name, value in payload["nodes"].items():
            target = getattr(bank, name)
            target.data[:need] = value.to(target.device).to(target.dtype)
        brain.novelty[:need] = payload["novelty"].to(brain.novelty.device)
        if brain.novelty.numel() > need:
            brain.novelty[need:] = 0.0

    edges = payload["edges"]
    adj = brain.adj
    n_edges = int(edges["n"])
    adj._reserve(n_edges)
    with torch.no_grad():
        adj.src[:n_edges] = edges["src"].to(adj.src.device)
        adj.dst[:n_edges] = edges["dst"].to(adj.dst.device)
        adj.usage[:n_edges] = edges["usage"].to(adj.usage.device)
        adj.born[:n_edges] = edges["born"].to(adj.born.device)
        adj.logit.data[:n_edges] = edges["logit"].to(adj.logit.device)
    adj.n_edges = n_edges
    adj.rebuild_tables(bank.n_nodes)

    g = payload["globals"]
    with torch.no_grad():
        brain.tok_emb.weight.copy_(g["tok_emb"])
        brain.pos_emb.weight.copy_(g["pos_emb"])
        brain.s_proj.weight.copy_(g["s_proj_w"])
        brain.s_proj.bias.copy_(g["s_proj_b"])
        brain.s_norm.weight.copy_(g["s_norm_w"])
        brain.s_norm.bias.copy_(g["s_norm_b"])

    if lifecycle is not None and payload.get("lifecycle"):
        lifecycle.load_state_dict(payload["lifecycle"])
    if tier is not None and payload.get("tier"):
        tier.load_state_dict(payload["tier"])
    if optimizer is not None and payload.get("optimizer"):
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except Exception:
            pass
    restore_rng(payload.get("rng"))
    return payload


def load_checkpoint(path: str | Path, brain, optimizer=None, lifecycle=None,
                    tier=None) -> Dict[str, Any]:
    """Restore in place, keeping every Parameter object (so AdamW stays valid)."""
    payload = read_checkpoint(path)
    return apply_payload(brain, payload, optimizer, lifecycle, tier)
