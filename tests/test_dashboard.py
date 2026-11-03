"""Stage 6 guarantees: the cockpit's state machine, snapshots, chat, sliders."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from dashboard.runner import TrainingRunner
from dashboard.server import create_app
from mycelia.config import load_config

TINY = dict(
    n_nodes=16,
    max_hops=2,
    max_births=2,
    max_edges=128,
    batch_size=4,
    seq_len=6,
    enable_growth=False,
    enable_lifecycle=False,
)


def make_runner(max_steps: int = 400) -> TrainingRunner:
    config = load_config()
    runner = TrainingRunner(
        config=config,
        data="toy",
        device="cpu",
        max_steps=max_steps,
        trainer_kwargs=dict(TINY),
    )
    runner.trainer.brain.allow_node_growth = False
    return runner


def wait_for(predicate, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def runner():
    instance = make_runner()
    yield instance
    instance.stop()


def test_state_transitions(runner) -> None:
    """stopped → running → paused → running → stopped, always written."""
    assert runner.status == "stopped"
    runner.start()
    assert runner.status == "running"
    assert wait_for(lambda: runner.trainer.step >= 1)

    runner.pause()
    assert runner.status == "paused"
    assert runner.snapshot()["status"] == "paused"

    runner.resume()
    assert runner.status == "running"

    runner.stop()
    assert runner.status == "stopped"


def test_training_done_is_written_by_finally() -> None:
    runner = make_runner(max_steps=2)
    try:
        runner.start()
        assert wait_for(lambda: runner.status == "training_done", timeout=20.0)
        assert runner.snapshot()["status"] == "training_done"
    finally:
        runner.stop()


def test_ws_snapshot(runner) -> None:
    client = TestClient(create_app(runner))
    with client.websocket_connect("/ws") as socket:
        snapshot = socket.receive_json()

    assert snapshot["status"] == "stopped"
    assert {"nodes", "edges", "trace", "stats", "events", "history"} <= set(snapshot)
    assert len(snapshot["nodes"]) == runner.trainer.brain.n_nodes
    assert len(snapshot["edges"]) == runner.trainer.brain.ne
    assert all(len(edge) == 3 for edge in snapshot["edges"])
    # The edges are the ones *used*, never the 20 000 slots of the buffer.
    assert len(snapshot["edges"]) < runner.trainer.brain.max_edges


def test_chat_endpoint(runner) -> None:
    client = TestClient(create_app(runner))
    response = client.post("/chat", json={"prompt": "bonjour", "max_new_tokens": 4})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["tokens"]) == 4
    assert payload["text"]
    for token in payload["tokens"]:
        assert 0 <= token["id"] < 100
        assert "trace" in token and "path" in token


def test_hyperparams(runner) -> None:
    client = TestClient(create_app(runner))
    response = client.post(
        "/hyperparams", json={"temperature": 1.2, "lr": 0.0005, "l1": 0.0002}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["temperature"] == pytest.approx(1.2)
    assert runner.trainer.hparams["lr"] == pytest.approx(0.0005)
    assert runner.trainer.hparams["l1"] == pytest.approx(0.0002)


def test_node_endpoint(runner) -> None:
    client = TestClient(create_app(runner))
    response = client.get("/node/0")
    assert response.status_code == 200
    info = response.json()
    assert info["id"] == 0
    assert info["io"] is True
    assert client.get("/node/99999").status_code == 404
