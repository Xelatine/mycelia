"""The live runner: training in a thread, tensors readable at any moment.

Why a thread and not a subprocess: the dashboard has to read the *living*
tensors -- the current gate of node 7, the wiring that just changed, the trace
of the token being predicted. A subprocess would only offer a checkpoint.

The first debugging of this project died on a thread that raised and left the
status at "running" forever. Hence the non-negotiable shape of ``_loop``:
``try / except / finally``, with the status written in the ``finally`` whatever
happened, and ``stop()`` forcing ``stopped`` even if the thread is wedged.

Training and speech share one compute lock. Chat therefore waits for at most
one training step, which is the price of speaking with the *live* brain instead
of a copy: the brief asks for the brain in the browser, not a snapshot of it.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from typing import Any

import torch

from mycelia.brain import (
    apply_repetition_penalty,
    extract_path,
    nucleus_probabilities,
)
from mycelia.checkpoint import brain_from_genome, load_genome
from mycelia.config import load_config, nodes_for_scale
from mycelia.data import clean_text
from mycelia.train import Trainer

IMMORTAL = (0, 1)


class TrainingRunner:
    """Owns one Trainer, one thread, and every reading the cockpit needs."""

    def __init__(
        self,
        config: dict | None = None,
        data: str = "text",
        scale: int = 0,
        device: str | None = None,
        max_steps: int | None = None,
        checkpoint: str | None = None,
        resume: bool = False,
        observe: bool = False,
        batch_size: int | None = None,
        seq_len: int | None = None,
        trainer_kwargs: dict | None = None,
    ) -> None:
        self.config = config or load_config()
        self.data_mode = data
        self.scale = int(scale)
        self.observe = bool(observe)
        self.max_steps = int(
            max_steps
            if max_steps is not None
            else (
                self.config["training"]["text_steps"]
                if data == "text"
                else self.config["training"]["toy_steps"]
            )
        )

        if observe:
            if not checkpoint:
                raise ValueError("observation mode needs a checkpoint")
            self.trainer = self._trainer_from_checkpoint(checkpoint)
            self.status = "observation"
        elif resume and checkpoint:
            self.trainer = Trainer.resume(
                checkpoint, config=self.config, device=device, batch_size=batch_size, seq_len=seq_len
            )
            self.status = "stopped"
        else:
            extra = dict(trainer_kwargs or {})
            if batch_size is None:
                batch_size = extra.pop("batch_size", None)
            else:
                extra.pop("batch_size", None)
            if seq_len is None:
                seq_len = extra.pop("seq_len", None)
            else:
                extra.pop("seq_len", None)
            self.trainer = Trainer(
                config=self.config,
                data=data,
                scale=scale,
                device=device,
                batch_size=batch_size,
                seq_len=seq_len,
                **extra,
            )
            self.status = "stopped"

        self.hyperparams: dict[str, float] = {
            "temperature": 0.8,
            "top_p": 0.9,
            "lr": float(self.trainer.optimizer.param_groups[0]["lr"]),
            "l1": float(self.trainer.l1_weight),
        }
        self.events: deque[dict] = deque(
            maxlen=int(self.config["dashboard"]["events"])
        )
        self.history: list[dict] = []
        self.error: str | None = None
        self.started_at: float | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._compute = threading.RLock()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._snapshot: dict[str, Any] = {}
        self._event_cursor = 0
        self._publish({}, force=True)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _trainer_from_checkpoint(self, path: str) -> Trainer:
        blob = load_genome(path)
        brain = brain_from_genome(blob)
        data = blob.get("extra", {}).get("data", "text")
        trainer = Trainer(config=self.config, data=data, brain=brain, step=int(blob["step"]))
        from mycelia.checkpoint import restore_optimizer_state

        restore_optimizer_state(trainer.brain, trainer.optimizer, blob.get("optimizer"))
        trainer.history = list(blob.get("history", []))
        return trainer

    # ------------------------------------------------------------------
    # Thread control
    # ------------------------------------------------------------------
    def start(self) -> str:
        if self.observe:
            return self.status
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._paused.clear()
                self.status = "running"
                return self.status
            self._stop.clear()
            self._paused.clear()
            self.error = None
            self.status = "running"
            self.started_at = self.started_at or time.time()
            self._thread = threading.Thread(target=self._loop, name="mycelia-train", daemon=True)
        self._thread.start()
        self._publish({}, force=True)
        return self.status

    def pause(self) -> str:
        if self.observe:
            return self.status
        self._paused.set()
        if self._thread is not None and self._thread.is_alive():
            self.status = "paused"
            self._publish({}, force=True)
        return self.status

    def resume(self) -> str:
        return self.start()

    def stop(self) -> str:
        self._stop.set()
        self._paused.clear()
        self.status = "stopped"
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
        self._publish({}, force=True)
        return self.status

    def control(self, action: str) -> str:
        action = str(action).lower()
        if action == "start":
            return self.start()
        if action in ("pause", "paused"):
            return self.pause()
        if action == "resume":
            return self.resume()
        if action in ("stop", "stopped"):
            return self.stop()
        raise ValueError(f"unknown action {action!r}")

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    time.sleep(0.05)
                    continue
                with self._compute:
                    metrics = self.trainer.train_step()
                self._publish(metrics)
                if self.trainer.step >= self.max_steps:
                    break
        except Exception as error:  # noqa: BLE001 - the point is to survive anything
            self.error = f"{type(error).__name__}: {error}"
            self.events.append(
                {"step": self.trainer.step, "kind": "error", "text": self.error}
            )
            traceback.print_exc()
        finally:
            with self._lock:
                # The status is ALWAYS written, even when the thread died.
                if self._stop.is_set():
                    self.status = "stopped"
                elif self.error:
                    self.status = "error"
                else:
                    self.status = "training_done"
            self._publish({}, force=True)
            self._thread = None

    # ------------------------------------------------------------------
    # Readings
    # ------------------------------------------------------------------
    def _publish(self, metrics: dict, force: bool = False) -> None:
        brain = self.trainer.brain
        if metrics or force:
            with self._compute:
                gates = torch.sigmoid(brain.bank.gate_b.detach().float()).flatten().tolist()
                alive = brain.node_alive.detach().tolist()
                age = (int(brain.clock) - brain.node_born_step).clamp_min(0).tolist()
                edges = brain.edge_index[:, : brain.ne].t().tolist()
                masks = brain.edge_mask.detach().float().tolist()
                trace = {
                    str(hop): [[int(s), int(t)] for s, t in pairs]
                    for hop, pairs in brain.last_trace.items()
                }
                last_births = int(getattr(brain, "last_births", 0))
                node_count = brain.n_nodes
                edge_count = brain.ne
                alive_count = int(brain.node_alive.sum())
                cemetery = len(brain.cemetery)
                clock = int(brain.clock)

            if metrics:
                self.history.append(
                    {
                        "step": int(metrics["step"]),
                        "loss": round(float(metrics["loss"]), 4),
                        "ppl": round(float(metrics["ppl"]), 3),
                        "edges": int(metrics["edges"]),
                        "nodes": int(metrics["nodes"]),
                        "pool": int(metrics["pool"]),
                    }
                )
                del self.history[:-4000]
                while self._event_cursor < len(self.trainer.events):
                    self.events.append(
                        {
                            "step": int(metrics["step"]),
                            "kind": "growth",
                            "text": self.trainer.events[self._event_cursor],
                        }
                    )
                    self._event_cursor += 1

            snapshot = {
                "status": self.status,
                "error": self.error,
                "step": int(metrics.get("step", self.trainer.step)) if metrics else self.trainer.step,
                "max_steps": self.max_steps,
                "stats": {
                    "loss": float(metrics["loss"]) if metrics else None,
                    "ppl": float(metrics["ppl"]) if metrics else None,
                    "val_ppl": float(metrics["val_ppl"]) if metrics and "val_ppl" in metrics else None,
                    "edges": edge_count,
                    "nodes": alive_count,
                    "pool": node_count,
                    "births": int(getattr(brain, "births", 0)),
                    "edge_deaths": int(getattr(brain, "edge_deaths", 0)),
                    "node_deaths": int(getattr(brain, "node_deaths", 0)),
                    "cemetery": cemetery,
                    "last_births": last_births,
                    "lr": float(self.trainer.optimizer.param_groups[0]["lr"]),
                    "device": str(self.trainer.device),
                    "data": self.data_mode,
                },
                "nodes": [
                    {
                        "id": node_id,
                        "alive": bool(alive[node_id]),
                        "gate": round(float(gates[node_id]), 4),
                        "age": int(age[node_id]),
                        "io": node_id in IMMORTAL,
                    }
                    for node_id in range(node_count)
                ],
                "edges": [
                    [int(s), int(t), round(float(mask), 4)]
                    for (s, t), mask in zip(edges, masks)
                ],
                "trace": trace,
                "events": list(self.events)[-120:],
                "history": self.history[-600:],
                "hyperparams": dict(self.hyperparams),
                "uptime": round(time.time() - (self.started_at or time.time()), 1),
                "clock": clock,
            }
            with self._lock:
                self._snapshot = snapshot
                # No status write here: a step that finishes *after* a pause
                # must not resurrect "running" behind the user's back.

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot

    def set_hyperparams(self, values: dict) -> dict:
        for key in ("temperature", "lr", "l1", "top_p"):
            if key in values and values[key] is not None:
                self.hyperparams[key] = float(values[key])
        # The trainer reads them at every step: no restart, no reload.
        self.trainer.hparams["lr"] = self.hyperparams["lr"]
        self.trainer.hparams["l1"] = self.hyperparams["l1"]
        with self._lock:
            if self._snapshot:
                self._snapshot["hyperparams"] = dict(self.hyperparams)
        self.events.append(
            {
                "step": self.trainer.step,
                "kind": "hyperparams",
                "text": (
                    f"hyperparams : temperature {self.hyperparams['temperature']:.2f}, "
                    f"lr {self.hyperparams['lr']:.2e}, L1 {self.hyperparams['l1']:.4g}"
                ),
            }
        )
        return dict(self.hyperparams)

    # ------------------------------------------------------------------
    # Speech with the living brain
    # ------------------------------------------------------------------
    def chat(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict:
        trainer = self.trainer
        tokenizer = trainer.tokenizer
        brain = trainer.brain
        prompt = prompt or "Il était une fois"
        if tokenizer is None:
            # Toy brain: the cockpit still has to be testable without a corpus.
            ids = [(ord(char) % max(1, brain.vocab - 2)) + 2 for char in prompt[:16]]

            def decode(token_ids: list[int]) -> str:
                return " ".join(str(token) for token in token_ids)
        else:
            ids = tokenizer.encode(prompt).ids
            raw_decode = tokenizer.decode

            # C8: a byte-level BPE leaves U+FFFD and control characters on an
            # incomplete sequence. The cockpit hides them; the brain is intact.
            def decode(token_ids: list[int]) -> str:
                return clean_text(raw_decode(token_ids))
        temperature = float(self.hyperparams["temperature"] if temperature is None else temperature)
        top_p = float(self.hyperparams["top_p"] if top_p is None else top_p)
        brain = trainer.brain
        steps: list[dict] = []
        with self._compute:
            brain.eval()
            tokens = torch.tensor([ids], dtype=torch.long, device=brain.edge_index.device)
            _logits, state, trace = brain(tokens, grow=False, return_state=True)
            for _ in range(int(max_new_tokens)):
                logits = brain.predict(state)[0].float().cpu()
                logits = apply_repetition_penalty(logits, ids[-20:])
                if temperature <= 0:
                    next_id = int(torch.argmax(logits))
                else:
                    probs = nucleus_probabilities(logits / temperature, top_p)
                    next_id = int(torch.multinomial(probs, 1))
                if not 0 <= next_id < brain.vocab:
                    next_id = 0
                ids.append(next_id)
                text = decode([next_id])
                _logits, state, trace = brain(
                    torch.tensor([[next_id]], dtype=torch.long, device=brain.edge_index.device),
                    state=state,
                    grow=False,
                    return_state=True,
                )
                path = extract_path(trace) or []
                steps.append(
                    {
                        "id": next_id,
                        "text": text,
                        "trace": {
                            str(hop): [[int(s), int(t)] for s, t in pairs]
                            for hop, pairs in trace.items()
                        },
                        "path": [int(node) for node in path],
                    }
                )
            brain.train()
        return {
            "text": decode(ids),
            "tokens": steps,
        }

    # ------------------------------------------------------------------
    # One node's file
    # ------------------------------------------------------------------
    def node_info(self, node_id: int) -> dict:
        brain = self.trainer.brain
        if not 0 <= node_id < brain.n_nodes:
            raise KeyError(node_id)
        with self._compute:
            edges = brain.edge_index[:, : brain.ne]
            incoming = edges[:, edges[1] == node_id].t().tolist()
            outgoing = edges[:, edges[0] == node_id].t().tolist()
            gate = float(torch.sigmoid(brain.bank.gate_b[node_id].detach().float()))
            info = {
                "id": int(node_id),
                "alive": bool(brain.node_alive[node_id]),
                "io": bool(brain.is_io[node_id]),
                "gate": round(gate, 5),
                "age": int(max(0, brain.clock - int(brain.node_born_step[node_id]))),
                "born_step": int(brain.node_born_step[node_id]),
                "incoming": [[int(s), int(t)] for s, t in incoming],
                "outgoing": [[int(s), int(t)] for s, t in outgoing],
                "parents": sorted({int(s) for s, _ in incoming}),
                "mean_out_norm": round(float(brain.mean_out[node_id].norm()), 4),
                "in_cemetery": int(node_id) in brain.cemetery,
                "events": [
                    event
                    for event in list(self.events)[-80:]
                    if str(node_id) in str(event.get("text", ""))
                ],
            }
        return info
