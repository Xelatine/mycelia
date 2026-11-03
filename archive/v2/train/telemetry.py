"""Live telemetry: one JSON snapshot plus an append-only event journal.

``state.json`` is what the cockpit reads for the panels and curves;
``events.jsonl`` is the human-readable journal (births, deaths, wakes,
reallocations, tier transitions).  Both are written atomically so a reader never
sees a half-written file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class Telemetry:
    def __init__(self, run_dir: str | Path, max_events: int = 4000):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.max_events = int(max_events)
        self._events_written = 0
        self._buffer: List[Dict[str, Any]] = []
        self.started = time.time()

    # ------------------------------------------------------------- events
    def emit(self, kind: Optional[str] = None, message: str = "", **fields: Any) -> None:
        event: Dict[str, Any] = {"t": round(time.time() - self.started, 2),
                                 "type": kind or fields.pop("type", "event")}
        if message:
            event["message"] = message
        for key, value in fields.items():
            if key in ("type", "message"):
                continue
            if isinstance(value, (int, float, str, bool)) or value is None:
                event[key] = value
            else:
                event[key] = str(value)
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._events_written += 1
        if self._events_written > self.max_events * 2:
            self.compact()

    def compact(self) -> None:
        """Keep the newest events so the journal never grows without bound."""
        if not self.events_path.exists():
            return
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        keep = lines[-self.max_events:]
        self.events_path.write_text("\n".join(keep) + "\n", encoding="utf-8")
        self._events_written = len(keep)

    def events(self, limit: int = 200, since: int = 0) -> List[Dict[str, Any]]:
        if not self.events_path.exists():
            return []
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        out: List[Dict[str, Any]] = []
        for line in lines[max(0, since):]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out[-int(limit):]

    # -------------------------------------------------------------- state
    def write_state(self, payload: Dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        payload = dict(payload)
        payload.setdefault("updated_at", round(time.time() - self.started, 2))
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def read_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
