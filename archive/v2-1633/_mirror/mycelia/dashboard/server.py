"""Cockpit backend: FastAPI + WebSocket over a run directory.

The server only ever *reads* the run directory (``state.json``,
``events.jsonl``, ``checkpoint.pt``); commands are passed to the detached
trainer process through small files (``.pause``, ``.stop``, ``.save``,
``control.json``), so closing the page never kills a run.

Three panels need the live genome — the graph, the node sheet and the chat
generation — and they share one lazily reloaded copy of the checkpoint.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mycelia.config import Config, config_from_dict, load_config
from mycelia.data.tokenizer import FrenchBPE
from mycelia.gen.sampling import generate
from mycelia.models.brain import BrainNetwork
from mycelia.train.checkpoint import apply_payload, read_checkpoint
from mycelia.viz.snapshot import build_graph, node_card

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"


# --------------------------------------------------------------- payloads
class StartRequest(BaseModel):
    config: str = Field(default="configs/demo.yaml")
    seed: Optional[int] = None
    steps: Optional[int] = Field(default=None, ge=1, le=10_000_000)
    fresh: bool = False


class ParamsRequest(BaseModel):
    lr: Optional[float] = Field(default=None, gt=0, le=1.0)
    l1_scale: Optional[float] = Field(default=None, ge=0, le=1000)
    tau_scale: Optional[float] = Field(default=None, gt=0, le=10)
    temperature: Optional[float] = Field(default=None, gt=0, le=2.0)
    top_p: Optional[float] = Field(default=None, gt=0, le=1.0)


class GenerateRequest(BaseModel):
    prompt: str = Field(default="Il était une fois", max_length=400)
    tokens: int = Field(default=32, ge=1, le=300)
    temperature: float = Field(default=0.85, gt=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.15, ge=0.5, le=3.0)


# ------------------------------------------------------------ brain holder
class BrainHolder:
    """Keeps one in-memory copy of the run's genome, reloaded on change."""

    def __init__(self, run_dir: Path, device: str = "cpu"):
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self.brain: Optional[BrainNetwork] = None
        self.tokenizer = None
        self.cfg: Optional[Config] = None
        self.payload: Dict[str, Any] = {}
        self.mtime: float = 0.0
        self.error: Optional[str] = None
        self.loaded_at: float = 0.0

    @property
    def checkpoint(self) -> Path:
        return self.run_dir / "checkpoint.pt"

    def maybe_reload(self) -> bool:
        try:
            stamp = self.checkpoint.stat().st_mtime
        except OSError:
            self.error = "no checkpoint yet — start or resume a run"
            return False
        if self.brain is not None and stamp <= self.mtime:
            return True
        for attempt in range(3):
            try:
                payload = read_checkpoint(self.checkpoint)
                cfg = config_from_dict(payload["config"])
                brain = BrainNetwork(cfg, int(payload["vocab_size"]))
                _restore_genome(brain, payload)
                brain.to(self.device)
                brain.eval()
                self.brain = brain
                self.cfg = cfg
                self.payload = payload
                self.tokenizer = self._find_tokenizer(cfg)
                self.mtime = stamp
                self.error = None
                self.loaded_at = time.time()
                return True
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.4 * (attempt + 1))
        return self.brain is not None

    def _find_tokenizer(self, cfg: Config):
        cache = ROOT / cfg.data.cache_dir
        for candidate in sorted(cache.glob(f"tokenizer-fr-{cfg.data.vocab_size // 1000}k.json")):
            try:
                return FrenchBPE.load(candidate)
            except Exception:
                continue
        for candidate in sorted(cache.glob("tokenizer-fr-*.json")):
            try:
                return FrenchBPE.load(candidate)
            except Exception:
                continue
        return None


def _restore_genome(brain: BrainNetwork, payload: Dict[str, Any]) -> None:
    """Restore an already-read checkpoint payload into a fresh brain."""
    apply_payload(brain, payload)


# --------------------------------------------------------------- controller
class TrainingController:
    """Spawns and steers a detached ``scripts/train.py`` process."""

    def __init__(self, run_dir: Path, allow_control: bool = True,
                 python: Optional[str] = None):
        self.run_dir = Path(run_dir)
        self.allow_control = allow_control
        self.python = python or sys.executable
        self.process: Optional[subprocess.Popen] = None
        self.log_path = self.run_dir / "trainer.out"
        self.started_at: float = 0.0
        self.error: Optional[str] = None
        self.last_config: str = "configs/demo.yaml"

    # ------------------------------------------------------------- helpers
    def _flag(self, name: str) -> Path:
        return self.run_dir / name

    def status(self) -> Dict[str, Any]:
        running = self.process is not None and self.process.poll() is None
        state = "idle"
        if running:
            state = "running"
        elif self._flag(".pause").exists():
            state = "paused"
        elif self.process is not None and self.process.returncode is not None:
            state = "finished" if self.process.returncode == 0 else "error"
        elif self.process is None and (self.run_dir / "checkpoint.pt").exists():
            state = "idle"
        lines = self.tail_log()
        return {
            "state": state,
            "error": self.error or (None if running else self._run_error()),
            "pid": self.process.pid if running else None,
            "log": str(self.log_path),
            "log_exists": self.log_path.exists(),
            "last_lines": lines[-12:],
            "control_enabled": self.allow_control,
            "can_start": self.allow_control and not running,
            "can_pause": self.allow_control and running,
            "can_resume": self.allow_control and not running,
            "can_stop": self.allow_control and (running or self._flag(".pause").exists()),
            "started_at": self.started_at,
            "run_dir": str(self.run_dir),
        }

    def _run_error(self) -> Optional[str]:
        if self.process is None or self.process.returncode in (None, 0):
            return None
        lines = self.tail_log()
        for line in reversed(lines):
            if "Error" in line or "Traceback" in line:
                return line.strip()
        return f"trainer exited with code {self.process.returncode}"

    def tail_log(self, lines: int = 12) -> List[str]:
        if not self.log_path.exists():
            return []
        try:
            content = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return content.splitlines()[-lines:]

    def spawn(self, config: str, seed: Optional[int] = None,
              steps: Optional[int] = None, resume: bool = False) -> Dict[str, Any]:
        if not self.allow_control:
            raise HTTPException(403, "control disabled (--no-control)")
        if self.process is not None and self.process.poll() is None:
            raise HTTPException(409, "a trainer is already running")
        if resume and not (self.run_dir / "checkpoint.pt").exists():
            raise HTTPException(400, "no checkpoint to resume from")
        if not resume and (self.run_dir / "checkpoint.pt").exists():
            raise HTTPException(409, "checkpoint already exists")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for flag in (".pause", ".stop", ".save"):
            self._flag(flag).unlink(missing_ok=True)
        cmd = [self.python, str(ROOT / "scripts" / "train.py"),
               "--config", config, "--run-dir", str(self.run_dir)]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        if steps is not None:
            cmd += ["--steps", str(steps)]
        if resume:
            cmd += ["--resume"]
        log = self.log_path.open("a", encoding="utf-8")
        log.write(f"\n=== {'resume' if resume else 'fresh'} "
                  f"{time.strftime('%H:%M:%S')} :: {' '.join(cmd)}\n")
        log.flush()
        creation = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=log,
                                        stdin=subprocess.DEVNULL,
                                        creationflags=creation, start_new_session=(os.name != "nt"))
        self.started_at = time.time()
        self.error = None
        self.last_config = config
        return self.status()

    # -------------------------------------------------------------- actions
    def pause(self) -> Dict[str, Any]:
        if not self.allow_control:
            raise HTTPException(403, "control disabled")
        self._flag(".pause").touch()
        return self.status()

    def stop(self) -> Dict[str, Any]:
        if not self.allow_control:
            raise HTTPException(403, "control disabled")
        running = self.process is not None and self.process.poll() is None
        if running:
            self._flag(".stop").touch()
        return self.status()

    def save(self) -> Dict[str, Any]:
        if not self.allow_control:
            raise HTTPException(403, "control disabled")
        self._flag(".save").touch()
        return self.status()

    def write_control(self, params: ParamsRequest) -> Dict[str, Any]:
        path = self.run_dir / "control.json"
        current: Dict[str, Any] = {}
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                current = {}
        for key, value in params.model_dump().items():
            if value is not None:
                current[key] = value
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current), encoding="utf-8")
        return {"control": current}


# --------------------------------------------------------------------- app
def create_app(run_dir: str | Path, allow_control: bool = True,
               push_interval: float = 0.5) -> FastAPI:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    holder = BrainHolder(run_path)
    controller = TrainingController(run_path, allow_control=allow_control)

    app = FastAPI(title="Mycelia cockpit", version="1.0")
    app.state.holder = holder
    app.state.controller = controller
    app.state.run_dir = run_path

    def state() -> Dict[str, Any]:
        path = run_path / "state.json"
        if not path.exists():
            return {"trainer_state": "idle", "step": 0, "steps_total": 0,
                    "curve": [], "nodes": {}, "edges": 0,
                    "message": "waiting for the trainer's first snapshot"}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"trainer_state": "loading"}

    def events(limit: int = 120, since: int = 0) -> List[Dict[str, Any]]:
        path = run_path / "events.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        rows: List[Dict[str, Any]] = []
        for line in lines[max(0, int(since)):]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows[-int(limit):]

    # ------------------------------------------------------------- reading
    @app.get("/api/state")
    def api_state() -> Dict[str, Any]:
        data = state()
        data["trainer"] = controller.status()
        return data

    @app.get("/api/graph")
    def api_graph(max_nodes: int = 420, max_edges: int = 900) -> Dict[str, Any]:
        holder.maybe_reload()
        if holder.brain is None:
            raise HTTPException(503, holder.error or "genome unavailable")
        st = state()
        return build_graph(holder.brain, holder.cfg, int(st.get("step", 0)),
                           isolated={n: 0 for n in range(0)}, max_nodes=max_nodes,
                           max_edges=max_edges)

    @app.get("/api/events")
    def api_events(limit: int = 120, since: int = 0) -> List[Dict[str, Any]]:
        return events(limit, since)

    @app.get("/api/node/{node_id}")
    def api_node(node_id: int) -> Dict[str, Any]:
        holder.maybe_reload()
        if holder.brain is None:
            raise HTTPException(503, holder.error or "genome unavailable")
        st = state()
        try:
            card = node_card(holder.brain, node_id, {}, {}, int(st.get("step", 0)))
        except KeyError:
            raise HTTPException(404, f"node {node_id} does not exist")
        card["events"] = [e for e in events(400) if e.get("node") == node_id][-20:]
        return card

    @app.get("/api/configs")
    def api_configs() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for path in sorted((ROOT / "configs").glob("*.yaml")):
            try:
                cfg = load_config(path)
                note = f"scale {cfg.model.scale} · {cfg.model.nodes_initial} nodes · dim {cfg.model.dim}"
            except Exception as exc:
                note = f"unreadable: {exc}"
            out.append({"name": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "label": path.stem, "description": note})
        return out

    @app.get("/api/train")
    def api_train() -> Dict[str, Any]:
        return controller.status()

    @app.get("/api/corpus")
    def api_corpus() -> Dict[str, Any]:
        st = state()
        return {"data": st.get("data", {}), "config": st.get("config", {})}

    # ------------------------------------------------------------ commands
    @app.post("/api/train/start")
    def api_start(req: StartRequest) -> Dict[str, Any]:
        return controller.spawn(req.config, req.seed, req.steps, resume=False)

    @app.post("/api/train/resume")
    def api_resume(req: StartRequest) -> Dict[str, Any]:
        return controller.spawn(req.config, req.seed, req.steps, resume=True)

    @app.post("/api/train/pause")
    def api_pause() -> Dict[str, Any]:
        controller.pause()
        return controller.status()

    @app.post("/api/train/stop")
    def api_stop() -> Dict[str, Any]:
        controller.stop()
        return controller.status()

    @app.post("/api/train/save")
    def api_save() -> Dict[str, Any]:
        controller.save()
        return controller.status()

    @app.post("/api/params")
    def api_params(req: ParamsRequest) -> Dict[str, Any]:
        return controller.write_control(req)

    @app.post("/api/generate")
    def api_generate(req: GenerateRequest) -> Dict[str, Any]:
        holder.maybe_reload()
        if holder.brain is None or holder.tokenizer is None:
            raise HTTPException(503, holder.error or "genome/tokenizer unavailable")
        st = state()
        out = generate(holder.brain, holder.tokenizer, prompt=req.prompt,
                       max_new_tokens=req.tokens, temperature=req.temperature,
                       top_p=req.top_p, repetition_penalty=req.repetition_penalty,
                       device=str(holder.device), track=True,
                       max_ctx=int(holder.cfg.train.ctx_len),
                       seed=int(time.time()) % 100000)
        out["step"] = int(st.get("step", 0))
        out["checkpoint_loaded_at"] = holder.loaded_at
        return out

    # ------------------------------------------------------------ websocket
    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        import asyncio

        last_event = 0
        try:
            while True:
                payload: Dict[str, Any] = {"type": "tick", "state": state(),
                                           "trainer": controller.status()}
                holder.maybe_reload()
                if holder.brain is not None:
                    st = payload["state"]
                    payload["graph"] = build_graph(
                        holder.brain, holder.cfg, int(st.get("step", 0)),
                        isolated={}, max_nodes=420, max_edges=900)
                payload["events"] = events(80, last_event)
                all_events = events(400)
                last_event = max(0, len(all_events) - 80)
                await websocket.send_json(payload)
                await asyncio.sleep(float(push_interval))
        except WebSocketDisconnect:
            return
        except Exception:
            return

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    return app
