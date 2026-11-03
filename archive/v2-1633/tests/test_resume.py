"""Pause/resume must be exact: a resumed run equals an uninterrupted one."""

from __future__ import annotations

import json

import torch

from mycelia.train.loop import train


def test_resume_matches_uninterrupted_run(tiny_cfg, tmp_path):
    base = tmp_path / "run"
    cfg = tiny_cfg
    cfg.model.nodes_max = 72
    cfg.train.steps = 12

    # Reference: 12 uninterrupted steps.
    train(cfg, steps=12, run_dir=str(base), quiet=True)
    ref = torch.load(base / "checkpoint.pt", map_location="cpu", weights_only=False)
    ref_state = json.loads((base / "state.json").read_text(encoding="utf-8"))

    # Split: 6 steps, pause, then resume for 6 more.
    part = tmp_path / "resume"
    train(cfg, steps=6, run_dir=str(part), quiet=True)
    first = torch.load(part / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert int(first["step"]) == 6

    train(cfg, steps=12, resume=True, run_dir=str(part), quiet=True)
    resumed = torch.load(part / "checkpoint.pt", map_location="cpu", weights_only=False)
    resumed_state = json.loads((part / "state.json").read_text(encoding="utf-8"))

    assert int(resumed["step"]) == int(ref["step"]) == 12
    assert int(resumed["n_nodes"]) == int(ref["n_nodes"])
    assert int(resumed["edges"]["n"]) == int(ref["edges"]["n"])
    for name, value in ref["nodes"].items():
        assert torch.equal(resumed["nodes"][name], value), name
    assert torch.equal(resumed["globals"]["tok_emb"], ref["globals"]["tok_emb"])
    assert resumed_state["phase"] == ref_state["phase"]
    assert abs(float(resumed_state["best_val"]) - float(ref_state["best_val"])) < 1e-6


def test_pause_file_stops_the_loop(tiny_cfg, tmp_path):
    run = tmp_path / "paused"
    run.mkdir(parents=True)
    (run / ".pause").touch()
    cfg = tiny_cfg
    cfg.train.steps = 8
    result = train(cfg, steps=8, run_dir=str(run), quiet=True)
    assert result["state"] == "paused"
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert state["trainer_state"] == "paused"
    assert (run / "checkpoint.pt").exists()
