"""FastAPI + WebSocket cockpit, on localhost, no build step.

The WebSocket pushes a full snapshot every 500 ms: nodes, the edges that are
*used* (never the 20 000 slots of the buffer), the trace of the last token, the
vital signs and the journal. The frontend filters above 1 500 nodes; the server
never sends a dense structure.

Endpoints:
    GET  /                 the dashboard
    WS   /ws               snapshot every dashboard.snapshot_ms
    POST /control/{action} start | pause | resume | stop
    POST /hyperparams      temperature, lr, l1, top_p (live, no restart)
    POST /chat             speech from the living brain, token by token
    GET  /node/{id}        one neuron's file
    GET  /health           liveness of the server, not of the training
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dashboard.runner import TrainingRunner

STATIC = Path(__file__).resolve().parent / "static"


class ControlBody(BaseModel):
    action: str | None = None


class HyperBody(BaseModel):
    """pydantic v2: serialised with model_dump(), never the deprecated .dict()."""

    temperature: float | None = Field(default=None, ge=0.0, le=3.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    lr: float | None = Field(default=None, gt=0.0, le=1.0)
    l1: float | None = Field(default=None, ge=0.0, le=1.0)


class ChatBody(BaseModel):
    prompt: str = "Il était une fois"
    max_new_tokens: int = Field(default=40, ge=1, le=300)
    temperature: float | None = Field(default=None, ge=0.0, le=3.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)


def create_app(runner: TrainingRunner) -> FastAPI:
    app = FastAPI(title="Mycelia", version="2.0", docs_url="/api/docs")
    app.state.runner = runner

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        page = STATIC / "index.html"
        if not page.exists():
            raise HTTPException(status_code=500, detail="dashboard/static/index.html is missing")
        return FileResponse(page)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "status": runner.status, "step": runner.trainer.step}

    @app.websocket("/ws")
    async def websocket(socket: WebSocket) -> None:
        await socket.accept()
        delay = float(runner.config["dashboard"]["snapshot_ms"]) / 1000.0
        try:
            while True:
                await socket.send_json(runner.snapshot())
                await asyncio.sleep(delay)
        except (WebSocketDisconnect, RuntimeError):
            return

    @app.post("/control/{action}")
    def control(action: str) -> dict:
        try:
            return {"status": runner.control(action)}
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/control")
    def control_body(body: ControlBody) -> dict:
        if not body.action:
            raise HTTPException(status_code=400, detail="missing action")
        return control(body.action)

    @app.post("/hyperparams")
    def hyperparams(body: HyperBody) -> dict:
        return runner.set_hyperparams(body.model_dump())

    @app.post("/chat")
    def chat(body: ChatBody) -> dict:
        try:
            return runner.chat(
                body.prompt,
                max_new_tokens=body.max_new_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
            )
        except RuntimeError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/node/{node_id}")
    def node(node_id: int) -> dict:
        try:
            return runner.node_info(int(node_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"no node {node_id}") from error

    return app
