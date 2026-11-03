"""Tier controller: growth -> optimisation -> pool reallocation -> growth.

The rule is the one written in the project philosophy: while validation keeps
improving by more than 1% per window the brain is allowed to grow.  When it
stagnates, births are suspended and the existing wiring is optimised for a
fixed number of steps; if that still does not buy 0.5%, the pool itself is
enlarged and growth resumes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from mycelia.config import Config


class TierController:
    def __init__(self, cfg: Config, brain, lifecycle, emit=None):
        self.cfg = cfg
        self.pc = cfg.phase
        self.brain = brain
        self.lifecycle = lifecycle
        self.emit = emit or (lambda **kwargs: None)
        self.phase = "growth"
        self.best_val = float("inf")
        self.last_val = float("inf")
        self.window_start_val = float("inf")
        self.window_start_step = 0
        self.optimise_start_val = float("inf")
        self.optimise_start_step = 0
        self.reallocations = 0
        self.transitions: list[Dict[str, Any]] = []

    # ------------------------------------------------------------- helpers
    @property
    def births_allowed(self) -> bool:
        return self.phase == "growth"

    def _transition(self, phase: str, step: int, reason: str, **extra: Any) -> None:
        self.phase = phase
        self.transitions.append({"step": step, "phase": phase, "reason": reason})
        self.emit(type="phase", step=step, phase=phase, reason=reason,
                  message=f"phase → {phase} ({reason})", **extra)

    # ---------------------------------------------------------------- tick
    def on_validation(self, step: int, val_loss: float) -> str:
        """Feed one validation measurement; returns the current phase."""
        if not self.pc.enabled:
            return self.phase
        val_loss = float(val_loss)
        self.last_val = val_loss
        if val_loss < self.best_val:
            self.best_val = val_loss
        if self.window_start_step == 0:
            self.window_start_step = int(step)
            self.window_start_val = val_loss
            self.optimise_start_step = int(step)
            self.optimise_start_val = val_loss
            return self.phase

        if self.phase == "growth":
            if int(step) - self.window_start_step >= int(self.pc.window):
                gain = (self.window_start_val - val_loss) / max(1e-9, self.window_start_val)
                self.window_start_step = int(step)
                self.window_start_val = val_loss
                if gain < float(self.pc.growth_gain):
                    self.optimise_start_step = int(step)
                    self.optimise_start_val = val_loss
                    self._transition("optimise", int(step),
                                     f"validation gain {gain * 100:.2f}% < "
                                     f"{self.pc.growth_gain * 100:.1f}%")
        else:
            if int(step) - self.optimise_start_step >= int(self.pc.optimise_steps):
                gain = (self.optimise_start_val - val_loss) / max(
                    1e-9, self.optimise_start_val)
                if gain > float(self.pc.optimise_gain):
                    self.window_start_step = int(step)
                    self.window_start_val = val_loss
                    self._transition("growth", int(step),
                                     f"optimisation paid off ({gain * 100:.2f}%)")
                elif self.reallocations >= int(self.pc.max_realloc):
                    self.window_start_step = int(step)
                    self.window_start_val = val_loss
                    self._transition("growth", int(step),
                                     "reallocation budget exhausted")
                else:
                    extra = int(self.pc.realloc_nodes)
                    self.lifecycle.reallocate(extra)
                    self.reallocations += 1
                    self.window_start_step = int(step)
                    self.window_start_val = val_loss
                    self._transition("growth", int(step),
                                     f"no gain after optimisation → pool +{extra}")
        return self.phase

    # ---------------------------------------------------------------- state
    def state_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "best_val": self.best_val,
            "last_val": self.last_val,
            "window_start_val": self.window_start_val,
            "window_start_step": self.window_start_step,
            "optimise_start_val": self.optimise_start_val,
            "optimise_start_step": self.optimise_start_step,
            "reallocations": self.reallocations,
            "transitions": self.transitions[-50:],
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        self.phase = state.get("phase", "growth")
        self.best_val = float(state.get("best_val", float("inf")))
        self.last_val = float(state.get("last_val", float("inf")))
        self.window_start_val = float(state.get("window_start_val", float("inf")))
        self.window_start_step = int(state.get("window_start_step", 0))
        self.optimise_start_val = float(state.get("optimise_start_val", float("inf")))
        self.optimise_start_step = int(state.get("optimise_start_step", 0))
        self.reallocations = int(state.get("reallocations", 0))
        self.transitions = list(state.get("transitions", []))
