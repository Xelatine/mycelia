"""Shared fixtures: tiny configurations, a toy French corpus, a small brain."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mycelia.config import Config, load_config  # noqa: E402
from mycelia.models.brain import BrainNetwork  # noqa: E402

TOY_FR = [
    "Il était une fois une petite fille qui aimait les histoires de la forêt.",
    "Un jour, un garçon trouva un chien perdu près de la rivière du village.",
    "La mère prépara une soupe chaude et toute la famille mangea ensemble.",
    "Le renard courut dans les bois jusqu'à la maison de ses petits.",
    "Elle ouvrit la porte et découvrit un jardin rempli de fleurs jaunes.",
    "Le vieil homme raconta comment il avait traversé la grande montagne.",
    "Les enfants construisirent une cabane avec des planches et de la paille.",
    "Le chat dormait au soleil pendant que la pluie tombait sur le toit.",
    "Nous avons marché longtemps avant de trouver un endroit pour dormir.",
    "Le petit garçon apprit à nager dans le lac derrière la maison.",
    "Sa sœur lisait un livre pendant qu'il préparait le repas du soir.",
    "La ville était calme et les rues sentaient le pain chaud du matin.",
]


@pytest.fixture(scope="session")
def toy_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("corpus") / "toy-fr.txt"
    lines = []
    for i in range(60):
        for sentence in TOY_FR:
            lines.append(f"{sentence} (histoire {i})")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture()
def tiny_cfg(toy_corpus: Path) -> Config:
    cfg = load_config(ROOT / "configs" / "demo.yaml")
    cfg.model.dim = 16
    cfg.model.nodes_initial = 40
    cfg.model.nodes_max = 64
    cfg.model.beam = 3
    cfg.model.max_turns = 4
    cfg.model.candidates = 4
    cfg.model.seed_edges_per_node = 2.0
    cfg.train.batch_size = 2
    cfg.train.ctx_len = 8
    cfg.train.steps = 12
    cfg.train.warmup = 2
    cfg.train.val_every = 6
    cfg.train.val_batches = 1
    cfg.train.checkpoint_every = 6
    cfg.train.sample_every = 0
    cfg.train.log_every = 3
    cfg.data.corpus = str(toy_corpus)
    cfg.data.max_chars = 200_000
    cfg.life.every = 4
    cfg.life.birth_every = 4
    cfg.life.link_grace_steps = 4
    cfg.life.node_grace_steps = 4
    cfg.life.min_live_edges = 8
    cfg.life.min_live_nodes = 8
    cfg.life.max_cut_fraction = 0.5
    cfg.phase.window = 6
    cfg.phase.optimise_steps = 6
    cfg.phase.realloc_nodes = 8
    cfg.dashboard.max_events = 200
    return cfg


@pytest.fixture()
def make_brain(tiny_cfg: Config) -> Callable[..., BrainNetwork]:
    def _make(vocab: int = 64, **overrides) -> BrainNetwork:
        cfg = tiny_cfg
        for key, value in overrides.items():
            setattr(cfg.model, key, value)
        torch.manual_seed(0)
        brain = BrainNetwork(cfg, vocab)
        brain.eval()
        return brain

    return _make
