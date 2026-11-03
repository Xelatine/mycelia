"""Configuration objects for the Mycelia brain.

Every knob the project talks about lives here, so a run is fully described by
its YAML file plus the command-line overrides.  Nothing is hard-coded in the
model: thresholds, budgets and loss weights all come from these dataclasses.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    """Shape of the brain."""

    scale: int = 0
    dim: int = 128
    nodes_initial: int = 512
    nodes_max: int = 4096
    # A signal may walk at most this many hops before it is considered spent.
    max_turns: int = 16
    # Width of the beam: how many parallel signals one token may keep in flight.
    beam: int = 8
    # Candidate neighbours examined by a router (its outgoing links).
    candidates: int = 6
    # How many neighbours one node actually talks to per visit.
    fanout: int = 2
    # Signals below this energy are dead and stop computing.
    eps_energy: float = 1e-3
    # Initial energy injected by [e].
    init_energy: float = 1.0
    # Gate value under which a node's body is skipped entirely.
    gate_floor: float = 0.1
    # Fraction of random long-range links seeded at birth (kept tiny: the brain
    # starts almost unwired on purpose).
    seed_edges_per_node: float = 1.5
    dropout: float = 0.0

    @property
    def e_node(self) -> int:
        return 0

    @property
    def s_node(self) -> int:
        return 1


@dataclass
class LossConfig:
    """Soft auxiliary losses; the main cross-entropy always dominates."""

    l1_edges: float = 1e-4
    routing_entropy: float = 0.05
    anti_redundancy: float = 0.02
    circulation: float = 0.05
    growth: float = 0.01


@dataclass
class LifecycleConfig:
    """Death and birth rules."""

    enabled: bool = True
    every: int = 50                 # pruning pass period (steps)
    link_usage_threshold: float = 0.02
    link_usage_decay: float = 0.95   # EMA of link usage
    link_grace_steps: int = 100      # a link must be old before it can die
    node_grace_steps: int = 150      # same for a node losing its last link
    min_live_edges: int = 32         # keep a vital minimum wired
    min_live_nodes: int = 64
    birth_threshold: float = 0.55    # growth score needed to spawn a node
    birth_every: int = 25
    max_births_per_round: int = 4
    max_isolated_events: int = 3
    cemetery_capacity: int = 4096


@dataclass
class PhaseConfig:
    """Growth / optimisation / reallocation tiers."""

    enabled: bool = True
    window: int = 100                # steps per measurement window
    growth_gain: float = 0.01        # >1% validation gain per window = growth
    optimise_gain: float = 0.005     # under 0.5% after optimisation = reallocate
    optimise_steps: int = 400        # steps of optimisation before deciding
    realloc_nodes: int = 64          # pool growth on reallocation
    max_realloc: int = 8


@dataclass
class TrainConfig:
    """Optimisation and data."""

    steps: int = 2000
    batch_size: int = 8
    ctx_len: int = 48
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup: int = 100
    weight_decay: float = 0.01
    clip: float = 1.0
    bf16: bool = False
    grad_checkpoint: bool = False
    tau_start: float = 1.0
    tau_end: float = 0.1
    tau_anneal_steps: int = 1200
    val_every: int = 100
    val_batches: int = 8
    checkpoint_every: int = 100
    sample_every: int = 200
    log_every: int = 10
    device: str = "auto"


@dataclass
class DataConfig:
    """French corpus and tokenizer."""

    corpus: str = "auto"            # auto | path to a .txt file
    tokenizer: str = "auto"         # auto | path to a tokenizers .json
    vocab_size: int = 10000
    cache_dir: str = ".cache"
    hf_dataset: str = "tiny_stories_fr"
    max_chars: int = 32_000_000
    min_docs: int = 400


@dataclass
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    max_events: int = 4000
    graph_nodes: int = 420          # nodes shipped to the browser per snapshot
    top_edges: int = 900            # strongest links shipped per snapshot
    push_interval: float = 0.5      # websocket period (seconds)


@dataclass
class Config:
    seed: int = 42
    run_dir: str = "runs/mycelia-demo"
    name: str = "mycelia"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    life: LifecycleConfig = field(default_factory=LifecycleConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Scale presets.  ``scale`` in the YAML selects one and explicit YAML keys
# override individual fields afterwards.
SCALE_PRESETS: Dict[int, Dict[str, Any]] = {
    0: dict(dim=128, nodes_initial=512, nodes_max=4096, max_turns=16, beam=8,
            candidates=6, fanout=2),
    1: dict(dim=32, nodes_initial=302, nodes_max=302, max_turns=12, beam=6,
            candidates=4, fanout=2),
    2: dict(dim=128, nodes_initial=3000, nodes_max=6000, max_turns=16, beam=6,
            candidates=5, fanout=2),
    3: dict(dim=128, nodes_initial=140_000, nodes_max=140_000, max_turns=16,
            beam=4, candidates=4, fanout=2),
}


def _coerce(value: Any) -> Any:
    """YAML gives strings for numbers in some hand-written configs."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes"):
            return True
        if lowered in ("false", "no"):
            return False
        try:
            return int(lowered)
        except ValueError:
            pass
        try:
            return float(lowered)
        except ValueError:
            return value
    return value


def _apply(obj: Any, values: Dict[str, Any]) -> None:
    known = {f.name: f for f in fields(obj)}
    for key, raw in values.items():
        if key not in known:
            raise KeyError(f"unknown config key {key!r} for {type(obj).__name__}")
        current = getattr(obj, key)
        value = _coerce(raw)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _apply(current, value)
        else:
            if current is not None and value is not None:
                target = type(current)
                if target is int and isinstance(value, str):
                    value = int(float(value))
                elif target is float and isinstance(value, int):
                    value = float(value)
            setattr(obj, key, value)


def load_config(path: Optional[str] = None, **overrides: Any) -> Config:
    """Load a YAML config, expand the scale preset, then apply overrides.

    ``overrides`` may contain dotted names such as ``model.dim`` or nested
    dicts; ``None`` values are ignored so CLI flags can stay optional.
    """
    raw: Dict[str, Any] = {}
    if path:
        text = Path(path).read_text(encoding="utf-8")
        raw = yaml.safe_load(text) or {}
        raw = {k: v for k, v in raw.items()}

    cfg = Config()
    if "scale" in raw:
        scale = int(_coerce(raw.pop("scale")))
        preset = SCALE_PRESETS.get(scale)
        if preset is None:
            raise ValueError(f"scale must be one of {sorted(SCALE_PRESETS)}")
        cfg.model.scale = scale
        cfg.model.nodes_max = max(scale, 0) + 0  # placeholder, replaced below
        for key, value in preset.items():
            setattr(cfg.model, key, value)
        cfg.model.nodes_max = preset["nodes_max"]

    _apply(cfg, raw)

    flat: Dict[str, Any] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        if "." in key:
            section, leaf = key.split(".", 1)
            flat.setdefault(section, {})[leaf] = value
        elif "." not in key and key in {f.name for f in fields(Config)}:
            flat[key] = value
        else:
            # bare name: route to the section that knows it
            placed = False
            for section in ("model", "train", "data", "life", "loss", "phase",
                            "dashboard"):
                if key in {f.name for f in fields(getattr(cfg, section))}:
                    flat.setdefault(section, {})[key] = value
                    placed = True
                    break
            if not placed:
                raise KeyError(f"unknown override {key!r}")
    if flat:
        _apply(cfg, flat)
    if cfg.model.nodes_max < cfg.model.nodes_initial:
        cfg.model.nodes_max = cfg.model.nodes_initial
    return cfg


def clone_config(cfg: Config) -> Config:
    return copy.deepcopy(cfg)


def config_from_dict(payload: Dict[str, Any]) -> Config:
    """Rebuild a Config from ``Config.to_dict()`` (used by checkpoints)."""
    cfg = Config()
    _apply(cfg, payload)
    return cfg
