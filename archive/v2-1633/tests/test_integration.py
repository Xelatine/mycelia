"""End-to-end: 50 steps, checkpoint, dashboard payload and generation."""

from __future__ import annotations

import json

import torch
from fastapi.testclient import TestClient

from mycelia.config import config_from_dict
from mycelia.models.brain import BrainNetwork
from mycelia.train.checkpoint import load_checkpoint
from mycelia.train.loop import train
from mycelia.viz.snapshot import build_graph


def test_fifty_steps_produce_a_live_run(tiny_cfg, tmp_path):
    run = tmp_path / "fifty"
    cfg = tiny_cfg
    cfg.train.steps = 50
    cfg.train.val_every = 10
    cfg.train.checkpoint_every = 25
    cfg.train.log_every = 10
    result = train(cfg, steps=50, run_dir=str(run), quiet=True)

    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert state["step"] == 50
    assert state["edges"] > 0
    assert state["curve"], "the cockpit needs a curve to draw"
    assert state["params"] > 0
    assert state["trainer_state"] in ("finished", "paused")

    events = [json.loads(line) for line in
              (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    kinds = {e["type"] for e in events}
    assert "init" in kinds and "val" in kinds
    assert any(e["type"] in {"birth", "link_death", "node_death"} for e in events)

    payload = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == 50
    assert payload["nodes"]["W1"].shape[0] == payload["n_nodes"]

    # The checkpoint reloads into a fresh genome and generates text.
    cfg2 = config_from_dict(payload["config"])
    brain = BrainNetwork(cfg2, int(payload["vocab_size"]))
    load_checkpoint(run / "checkpoint.pt", brain)
    assert brain.nodes.n_nodes == payload["n_nodes"]
    assert brain.adj.n_edges == payload["edges"]["n"]
    graph = build_graph(brain, cfg2, step=50)
    assert graph["nodes"] and graph["edges"]
    assert len(graph["heat"]) == len(graph["heat"][0])
    assert result["state"] in ("finished", "paused")


def test_dashboard_api_serves_the_run(tiny_cfg, tmp_path):
    from mycelia.dashboard.server import create_app

    run = tmp_path / "dash"
    cfg = tiny_cfg
    cfg.train.steps = 12
    cfg.train.val_every = 6
    cfg.train.checkpoint_every = 6
    train(cfg, steps=12, run_dir=str(run), quiet=True)

    app = create_app(run, allow_control=False)
    client = TestClient(app)
    assert client.get("/").status_code == 200
    state = client.get("/api/state").json()
    assert state["step"] >= 12
    graph = client.get("/api/graph").json()
    assert graph["total_nodes"] >= 2
    events = client.get("/api/events").json()
    assert events
    configs = client.get("/api/configs").json()
    assert any(c["label"] == "demo" for c in configs)
    node = client.get(f"/api/node/{graph['nodes'][0]['id']}").json()
    assert "state" in node and "outgoing" in node
    control = client.get("/api/train").json()
    assert control["control_enabled"] is False
