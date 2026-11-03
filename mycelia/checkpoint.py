"""Pause, resume, and the promise of bit-for-bit continuity.

A run of this project is long. Losing it to a Ctrl+C, or resuming it into a
slightly different trajectory, would make every measurement meaningless. The
genome therefore stores *everything* that influences the next step:

* the node bank, the wiring that is actually used (``ne``, not the capacity),
  the trainable strengths, the usage averages;
* the cemetery, with the exact weights of every buried neuron;
* the growth manager's state and the tier history;
* the AdamW moments, keyed by parameter name (not by position: the pool grows,
  and positional keys would silently attach an edge's momentum to a node);
* the complete RNG state -- torch, CUDA, numpy, Python -- plus the batch
  sampler's generator.

The central test, ``test_resume_continuity``, compares the 50 losses *after* a
reload with the 50 losses of the uninterrupted run to 1e-6. Its tolerance is not
negotiable: an unexplained gap is a signal, not a nuisance.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mycelia.brain import Brain
    from mycelia.growth import GrowthManager

# 3: node_births (real births) is a counter of its own, and the membrane is
# saved as the single [N, batch, dim] tensor it has always been at runtime.
FORMAT_VERSION = 3
READABLE_VERSIONS = (2, 3)


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover - numpy is a test dependency
        state["numpy"] = None
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        import numpy as np

        np.random.set_state(state["numpy"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def purge_zombies(brain: "Brain") -> int:
    """Remove dead nodes that still hold weights but have no cemetery entry.

    A zombie is tolerated during a run (it costs nothing: its rows are never
    gathered) but it must not be written to the checkpoint: on reload it would
    be a dead node with weights, which is not a state the brain can be in.
    """
    purged = 0
    dead = (~brain.node_alive).nonzero().flatten().tolist()
    for node_id in dead:
        if int(node_id) in brain.cemetery:
            continue
        if not brain.bank.row_is_zero(int(node_id)):
            brain.bank.zero_row(int(node_id))
            purged += 1
        brain.mean_out[int(node_id)] = 0
    return purged


def capture_optimizer_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None) -> dict:
    """Snapshot AdamW by *name*, with shapes, so growth can re-attach it."""
    if optimizer is None:
        return {}
    names = {name: param for name, param in model.named_parameters()}
    by_id = {id(param): name for name, param in names.items()}
    captured: dict[str, Any] = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            name = by_id.get(id(param))
            if name is None:
                continue
            moment = optimizer.state.get(param, {})
            captured[name] = {
                key: (value.detach().cpu().clone() if torch.is_tensor(value) else value)
                for key, value in moment.items()
            }
    return captured


def restore_optimizer_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, captured: dict | None
) -> int:
    """Re-attach the moments. Parameters whose shape changed start fresh."""
    if not captured:
        return 0
    restored = 0
    for name, param in model.named_parameters():
        saved = captured.get(name)
        if not saved:
            continue
        compatible = all(
            not torch.is_tensor(value)
            or value.dim() == 0  # AdamW's step counter is a scalar, not a moment
            or tuple(value.shape) == tuple(param.shape)
            for value in saved.values()
        )
        if not compatible:
            continue
        optimizer.state[param] = {
            key: (value.to(param.device) if torch.is_tensor(value) else value)
            for key, value in saved.items()
        }
        restored += 1
    return restored


def save_genome(
    path: str | Path,
    brain: "Brain",
    step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    growth: "GrowthManager | None" = None,
    tokenizer: str | Path | None = None,
    history: list[dict] | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a complete, versioned genome."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    zombies = purge_zombies(brain)
    blob = {
        "format_version": FORMAT_VERSION,
        "step": int(step),
        "config": brain.config(),
        "n_nodes": int(brain.n_nodes),
        "ne": int(brain.ne),
        "state_dict": {key: value.detach().cpu().clone() for key, value in brain.state_dict().items()},
        "cemetery": {
            int(node): {name: tensor.clone() for name, tensor in weights.items()}
            for node, weights in brain.cemetery.items()
        },
        "counters": {
            "clock": int(brain.clock),
            "births": int(brain.births),
            "node_births": int(getattr(brain, "node_births", 0)),
            "edge_deaths": int(brain.edge_deaths),
            "node_deaths": int(brain.node_deaths),
        },
        "optimizer": capture_optimizer_state(brain, optimizer),
        "growth": growth.state_dict() if growth is not None else None,
        "rng": rng_state(),
        "tokenizer": str(tokenizer) if tokenizer is not None else None,
        "history": list(history or []),
        "zombies_purged": zombies,
        "extra": dict(extra or {}),
    }
    torch.save(blob, path)
    return path


def load_genome(path: str | Path, map_location: str = "cpu", mmap: bool = False) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no genome at {path}")
    blob = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
        mmap=mmap,
    )
    version = blob.get("format_version")
    if version not in READABLE_VERSIONS:
        raise ValueError(
            f"genome format {version!r} is not supported "
            f"(expected one of {READABLE_VERSIONS}, written as {FORMAT_VERSION})"
        )
    return blob


def brain_from_genome(blob: dict) -> "Brain":
    """Rebuild the exact brain, wiring and population included."""
    from mycelia.brain import Brain

    config = dict(blob["config"])
    config["n_nodes"] = int(blob["n_nodes"])
    brain = Brain(**config)
    brain.load_state_dict(blob["state_dict"])
    brain.ne = int(blob["ne"])
    # The host-side membership is derived state, not a saved tensor: rebuild it
    # from the wiring that was just restored, or the resumed run would think a
    # pair is free and open a duplicate (test_resume_continuity catches it).
    brain._rebuild_edge_keys()
    brain.clock = int(blob["counters"]["clock"])
    brain.births = int(blob["counters"]["births"])
    brain.node_births = int(blob["counters"].get("node_births", 0))
    brain.edge_deaths = int(blob["counters"]["edge_deaths"])
    brain.node_deaths = int(blob["counters"]["node_deaths"])
    brain.cemetery = {
        int(node): {name: tensor.clone() for name, tensor in weights.items()}
        for node, weights in blob.get("cemetery", {}).items()
    }
    return brain


def optimizer_for(brain: "Brain", lr: float = 1e-3) -> torch.optim.Optimizer:
    return torch.optim.AdamW(brain.parameters(), lr=lr)
