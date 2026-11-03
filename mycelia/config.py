"""One place for every threshold, loaded from ``configs/config.yaml``.

Why a file and not constants: a run that refuses to wire, prune or grow is
almost always a threshold problem, and the person debugging it should be able
to loosen a number without touching (or re-reading) the code. The defaults
below are the v2 values; the YAML only has to override what changes.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"

# Node budget per scale. 302 is a wink at C. elegans; 140 000 is the fly.
SCALE_NODES = {0: 512, 1: 302, 2: 3000, 3: 140_000}
SCALE_NAMES = {
    0: "prototype (RTX 3070, 8 Go)",
    1: "C. elegans",
    2: "fly larva",
    3: "fly (long term, mmap)",
}

DEFAULTS: dict[str, Any] = {
    "brain": {
        "scale": 0,
        "dim": 32,
        "max_hops": 8,
        "dissipation": 0.9,
        "state_decay": 0.995,
        "max_edges": 20000,
        "max_births": 8,
        "max_node_births": 2,
        "birth_k": 4,
        "birth_candidates": 64,
        "init_bias": -8.0,
        "edge_mask_init": 0.5,
        "birth_strength": 0.5,
        "weak_noise": 0.02,
        "node_margin": 32,
        "node_birth_interval": 500,
        "trace_limit": 2000,
        "edge_chunk_size": 512,
    },
    "lifecycle": {
        "prune_every": 100,
        "prune_ratio": 0.1,
        "prune_fraction": 0.1,
        "prune_threshold": 0.02,
        "isolate_grace": 300,
        "newborn_grace": 300,
        "identity_weight": 0.01,
        "identity_cosine": 0.999,
        "redundancy_weight": 0.02,
        "redundancy_cosine": 0.5,
        "redundancy_sample": 64,
    },
    "training": {
        "toy_steps": 2000,
        "toy_batch": 32,
        "toy_seq": 10,
        "text_steps": 10000,
        "text_batch": 16,
        "text_seq": 128,
        "tbptt": 16,
        "lr": 0.001,
        "text_lr": 0.0003,
        "warmup": 1000,
        "l1_weight": 0.0001,
        "clip": 1.0,
        "report_every": 100,
        "text_report_every": 200,
        "save_seconds": 600,
        "first_save_step": 10,
        "val_fraction": 0.05,
        "val_batch": 8,
        "seed": 42,
    },
    "growth": {
        "check_every": 500,
        "grow_threshold": 0.01,
        "optimize_threshold": 0.005,
        "reallocate": 256,
        "suspend_births": 5000,
        "node_freeze": 200,
    },
    "dashboard": {
        "host": "127.0.0.1",
        "port": 8080,
        "snapshot_ms": 500,
        "max_draw_nodes": 1500,
        "events": 500,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML, merge it over the defaults, and return plain dicts."""
    config = copy.deepcopy(DEFAULTS)
    path = Path(path) if path is not None else CONFIG_PATH
    if path.exists():
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config = _merge(config, loaded)
    return config


def nodes_for_scale(scale: int) -> int:
    if int(scale) not in SCALE_NODES:
        raise ValueError(f"unknown scale {scale!r}: expected one of {sorted(SCALE_NODES)}")
    return SCALE_NODES[int(scale)]
