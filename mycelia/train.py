"""Training: the delayed-copy toy task, then real French.

Run from the repository root:

    python mycelia/train.py                              # toy, 2000 steps
    python mycelia/train.py --data text --scale 0        # French, 10 000 steps
    python mycelia/train.py --data text --resume         # continue a run

The same :class:`Trainer` drives the CLI, the dashboard and the tests, so a
checkpoint written by one is resumable by the others. Three things it must get
right, because each one already failed a real run:

* **truncated backprop by blocks of 16 tokens**: the state is detached between
  blocks, never the graph of a block under a live optimizer step;
* **OOM retry outside the except block**: the failed tensors must be released
  *before* the smaller batch is tried, otherwise the retry keeps the memory of
  the attempt that just failed;
* **the life cycle runs between steps**: pruning compacts the edge buffer and
  growth replaces parameters, and both are forbidden in the middle of a forward.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.brain import Brain, extract_path, hop_summary  # noqa: E402
from mycelia.checkpoint import (  # noqa: E402
    brain_from_genome,
    capture_optimizer_state,
    load_genome,
    restore_optimizer_state,
    save_genome,
    set_rng_state,
)
from mycelia.config import load_config, nodes_for_scale  # noqa: E402
from mycelia.data import VOCAB, TextDataset, get_batch, load_tokenizer  # noqa: E402
from mycelia.growth import (  # noqa: E402
    GrowthManager,
    apply_growth,
    apply_pending_node_births,
)
from mycelia.lifecycle import (  # noqa: E402
    anti_identity_penalty,
    anti_redundancy_penalty,
    apply_lifecycle,
)

IGNORE = -100
TEXT_PROMPT = "Il était une fois"
CHECKPOINT_PATH = ROOT / "data" / "genome.pt"
LATEST_PATH = ROOT / "data" / "genome_latest.pt"
LOCK_PATH = ROOT / "data" / "train.lock"
HISTORY_PATH = ROOT / "history.json"

_DATASET_CACHE: dict[tuple, TextDataset] = {}


def get_text_dataset(seq_len: int = 128, val_fraction: float = 0.05, **kwargs) -> TextDataset:
    """Build the token stream once per process: encoding 18 MB takes seconds."""
    key = (int(seq_len), float(val_fraction))
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = TextDataset(seq_len=seq_len, val_fraction=val_fraction, **kwargs)
    return _DATASET_CACHE[key]


def pid_alive(pid: int) -> bool:
    """Is this PID a live process?

    Never ``os.kill(pid, 0)`` on Windows: there sig is an *exit code* and the
    call terminates the process instead of probing it.
    """
    pid = int(pid)
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def process_memory() -> dict[str, float]:
    """This process's memory, in MiB: working set, peak, commit charge.

    Three numbers, because they answer three different questions. The working
    set is what is physically resident; the peak is the worst instant since
    start (the autograd graph inside one step); the commit charge is what
    Windows reserved, and it is the one that kills a run with no traceback.

    Read with ``ctypes`` and not ``psutil``: psutil is not a dependency of this
    project, and the diagnostic has to run on a machine whose memory is already
    exhausted, where importing a new package is exactly what fails.
    """
    mib = 1024.0 * 1024.0
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        # argtypes are not optional here: without them ctypes passes the
        # pseudo-handle returned by GetCurrentProcess as a 32-bit int, the
        # 64-bit handle is truncated, and the call fails with no error at all.
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return {"working_set": 0.0, "peak": 0.0, "commit": 0.0}
        return {
            "working_set": counters.WorkingSetSize / mib,
            "peak": counters.PeakWorkingSetSize / mib,
            "commit": counters.PagefileUsage / mib,
        }
    values = {"working_set": 0.0, "peak": 0.0, "commit": 0.0}
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                amount = rest.strip().split()
                if not amount:
                    continue
                if key == "VmRSS":
                    values["working_set"] = float(amount[0]) / 1024.0
                elif key == "VmHWM":
                    values["peak"] = float(amount[0]) / 1024.0
                elif key == "VmSize":
                    values["commit"] = float(amount[0]) / 1024.0
    except OSError:
        pass
    return values


def acquire_lock(path: Path = LOCK_PATH) -> Path:
    """One trainer per checkpoint (C1).

    Two processes once wrote ``genome.pt`` in parallel (PIDs 13440 and 9972) and
    shared the GPU: 29 s/step instead of 8.5, and a corruptible checkpoint. On a
    run of a day, that is the difference between a result and nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
        try:
            other = int(raw.split()[0])
        except (ValueError, IndexError):
            other = 0
        if other and other != os.getpid() and pid_alive(other):
            raise SystemExit(
                f"Un entraînement tourne déjà (PID {other}). "
                f"Supprimez {path} si c'est une erreur."
            )
        print(f"verrou périmé (PID {other or '?'} n'est plus vivant) : reprise")
    path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    return path


def release_lock(path: Path = LOCK_PATH) -> None:
    """Remove the lock, but never someone else's."""
    try:
        if not path.exists():
            return
        owner = path.read_text(encoding="utf-8", errors="replace").strip().split()[0]
        if owner == str(os.getpid()):
            path.unlink()
    except (OSError, IndexError, ValueError):
        pass


def write_history(history: list[dict], path: Path = HISTORY_PATH) -> None:
    """C2: incremental and atomic.

    The cockpit reads ``history.json`` while training runs, so it is written to
    a sibling ``.tmp`` and moved into place: a reader never sees half a file,
    and a crash between two reports loses nothing already reported.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scale_label(scale: int, nodes: int) -> str:
    from mycelia.config import SCALE_NAMES

    return f"scale {scale} — {nodes} nodes ({SCALE_NAMES.get(int(scale), 'custom')})"


class Trainer:
    """One training loop, shared by the CLI, the dashboard and the tests."""

    def __init__(
        self,
        config: dict | None = None,
        data: str = "toy",
        device: str | torch.device | None = None,
        seed: int | None = None,
        scale: int = 0,
        brain: Brain | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        step: int = 0,
        history: list[dict] | None = None,
        growth: GrowthManager | None = None,
        tokenizer=None,
        dataset: TextDataset | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        enable_growth: bool = True,
        enable_lifecycle: bool = True,
        **brain_overrides,
    ) -> None:
        self.config = config or load_config()
        self.data_mode = data
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = int(self.config["training"]["seed"] if seed is None else seed)
        self.enable_growth = bool(enable_growth)
        self.enable_lifecycle = bool(enable_lifecycle)
        self.step = int(step)
        self.history = list(history or [])
        self.events: list[str] = []
        self.scale = int(scale)

        training = self.config["training"]
        brain_cfg = self.config["brain"]
        life = self.config["lifecycle"]
        self.tbptt = int(training["tbptt"])
        self.l1_weight = float(training["l1_weight"])
        self.clip = float(training["clip"])
        self.base_lr = float(training["text_lr"] if data == "text" else training["lr"])
        self.hparams: dict[str, float] = {}

        if brain is None:
            seed_everything(self.seed)
            self.tokenizer = tokenizer or (load_tokenizer() if data == "text" else None)
            vocab = self.tokenizer.get_vocab_size() if data == "text" else VOCAB
            kwargs = {
                "n_nodes": nodes_for_scale(self.scale),
                "dim": int(brain_cfg["dim"]),
                "vocab": vocab,
                "max_hops": int(brain_cfg["max_hops"]),
                "dissipation": float(brain_cfg["dissipation"]),
                "state_decay": float(brain_cfg["state_decay"]),
                "max_edges": int(brain_cfg["max_edges"]),
                "max_births": int(brain_cfg["max_births"]),
                "max_node_births": int(brain_cfg["max_node_births"]),
                "birth_k": int(brain_cfg["birth_k"]),
                "birth_candidates": int(brain_cfg["birth_candidates"]),
                "init_bias": float(brain_cfg["init_bias"]),
                "edge_mask_init": float(brain_cfg["edge_mask_init"]),
                "birth_strength": float(brain_cfg["birth_strength"]),
                "weak_noise": float(brain_cfg["weak_noise"]),
                "node_margin": int(brain_cfg["node_margin"]),
                "node_birth_interval": int(brain_cfg["node_birth_interval"]),
                "node_freeze": int(self.config["growth"]["node_freeze"]),
                "trace_limit": int(brain_cfg["trace_limit"]),
                "isolate_grace": int(life["isolate_grace"]),
                "prune_threshold": float(life["prune_threshold"]),
                "prune_ratio": float(life["prune_ratio"]),
                "prune_fraction": float(life["prune_fraction"]),
                "identity_cosine": float(life["identity_cosine"]),
                "redundancy_cosine": float(life["redundancy_cosine"]),
                "redundancy_sample": int(life["redundancy_sample"]),
                "edge_chunk_size": int(brain_cfg.get("edge_chunk_size", 512)),
                "mmap": self.scale == 3,
            }
            kwargs.update(brain_overrides)
            self.brain = Brain(**kwargs).to(self.device)
        else:
            self.brain = brain.to(self.device)
            self.tokenizer = tokenizer or (load_tokenizer() if data == "text" else None)
        self.brain.clock = self.step

        self.batch_size = int(
            batch_size
            if batch_size is not None
            else (training["text_batch"] if data == "text" else training["toy_batch"])
        )
        self.seq_len = int(
            seq_len
            if seq_len is not None
            else (training["text_seq"] if data == "text" else training["toy_seq"])
        )
        self.dataset = None
        if data == "text":
            self.dataset = dataset or get_text_dataset(
                self.seq_len, float(training["val_fraction"])
            )

        self.optimizer = optimizer or torch.optim.AdamW(
            self.brain.parameters(), lr=self.base_lr
        )
        self.generator = torch.Generator().manual_seed(self.seed)
        self.growth = growth or GrowthManager(
            every=int(self.config["growth"]["check_every"]),
            grow_threshold=float(self.config["growth"]["grow_threshold"]),
            optimize_threshold=float(self.config["growth"]["optimize_threshold"]),
            reallocate=int(self.config["growth"]["reallocate"]),
            suspend_births=int(self.config["growth"]["suspend_births"]),
            max_population=self.brain.max_population,
        )
        self.report_every = int(
            training["text_report_every"] if data == "text" else training["report_every"]
        )
        # C2: saving follows the clock, not the step counter. At 8.5 s/step a
        # thousand steps is more than two hours of work; and a step is not a
        # unit of time at all, since its cost follows the wiring.
        self.save_seconds = float(training.get("save_seconds", 600))
        self.first_save_step = int(training.get("first_save_step", 10))
        self._optimizer_pool = self.brain.n_nodes

    # ------------------------------------------------------------------
    # Hyperparameters, learning rate
    # ------------------------------------------------------------------
    def _apply_learning_rate(self) -> float:
        if self.hparams.get("lr"):
            lr = float(self.hparams["lr"])
        elif self.data_mode == "text":
            warmup = max(1, int(self.config["training"]["warmup"]))
            total = max(1, int(self.config["training"]["text_steps"]))
            if self.step < warmup:
                lr = self.base_lr * (self.step + 1) / warmup
            else:
                progress = min(1.0, (self.step - warmup) / max(1, total - warmup))
                lr = self.base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            lr = self.base_lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------
    def _release_memory(self) -> None:
        """Everything the failed attempt holds, released *outside* the except."""
        self.optimizer.zero_grad(set_to_none=True)
        self.brain.last_node_io = []
        self.brain.last_out_acc = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def step_once(self) -> dict:
        lr = self._apply_learning_rate()
        self.brain.clock = self.step
        self.brain.train()
        if self.data_mode == "toy":
            x, y = get_batch(self.batch_size, self.seq_len, generator=self.generator)
        else:
            x, y = self.dataset.sample_batch(self.batch_size, generator=self.generator)
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        blocks = max(1, math.ceil(self.seq_len / self.tbptt))
        births_left = self.brain.max_births
        state = None
        total_ce = 0.0
        penalty_value = 0.0
        identity_value = 0.0
        redundancy_value = 0.0
        for block in range(blocks):
            start = block * self.tbptt
            xc = x[:, start : start + self.tbptt]
            yc = y[:, start : start + self.tbptt]
            logits, state, _trace = self.brain(
                xc, state=state, return_state=True, trace=False, births=births_left
            )
            births_left = max(0, births_left - int(self.brain.last_births))
            targets = yc.reshape(-1).clone()
            if self.data_mode == "toy" and block == 0:
                # Position 0 of the sequence has no predecessor: no signal.
                targets[:: xc.shape[1]] = IGNORE
            ce = F.cross_entropy(logits.reshape(-1, self.brain.vocab), targets, ignore_index=IGNORE)
            if block == 0:
                # The three taxes ride on the *first* block's graph: the last
                # block's graph is freed by its own backward, and a second
                # backward through it would be the "backward twice" error.
                identity = anti_identity_penalty(self.brain)
                redundancy = anti_redundancy_penalty(self.brain)
                l1 = self.brain.edge_mask.float().abs().sum()
                penalty = (
                    float(self.hparams.get("l1", self.l1_weight)) * l1
                    + float(self.config["lifecycle"]["identity_weight"]) * identity
                    + float(self.config["lifecycle"]["redundancy_weight"]) * redundancy
                )
                (ce / blocks + penalty).backward()
                penalty_value = float(penalty.detach())
                identity_value = float(identity.detach())
                redundancy_value = float(redundancy.detach())
            else:
                (ce / blocks).backward()
            total_ce += float(ce.detach())
            state = state.detach() if state is not None else None

        # A newborn keeps its weak noise for 200 steps; only then is it shaped.
        self.brain.freeze_newborn_gradients()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), self.clip)
        self.optimizer.step()
        self.step += 1
        self.brain.clock = self.step

        ce_value = total_ce / blocks
        metrics = {
            "step": self.step,
            "loss": ce_value,
            "loss_total": ce_value + penalty_value,
            "ppl": math.exp(min(ce_value, 20.0)),
            "lr": lr,
            "edges": self.brain.ne,
            "nodes": int(self.brain.node_alive.sum()),
            "pool": self.brain.n_nodes,
            # C5: edges born and nodes born are two different lives.
            "edge_births": self.brain.births,
            "node_births": self.brain.node_births,
            "edge_deaths": self.brain.edge_deaths,
            "node_deaths": self.brain.node_deaths,
            "cemetery": len(self.brain.cemetery),
            "identity": identity_value,
            "redundancy": redundancy_value,
        }

        # A message addressed a node that did not exist: allocate the slot
        # between two steps, then re-attach the optimizer by name.
        if self.brain.pending_node_requests:
            before = self.brain.n_nodes
            captured = capture_optimizer_state(self.brain, self.optimizer)
            created = apply_pending_node_births(self.brain)
            if self.brain.n_nodes != before:
                self._rebuild_optimizer(captured)
            if created:
                self.log(
                    f"  {len(created)} nœud(s) né(s) d'un message : pool {before} → "
                    f"{self.brain.n_nodes} (nœuds_nés {self.brain.node_births})"
                )

        if self.enable_lifecycle and self.step % int(self.config["lifecycle"]["prune_every"]) == 0:
            stats = apply_lifecycle(self.brain, step=self.step, cfg=self.config)
            if stats["edge_deaths"]:
                self.log(f"  {stats['edge_deaths']} arêtes coupées, {stats['node_deaths']} nœuds enterrés")

        if self.enable_growth and self.step % int(self.config["growth"]["check_every"]) == 0:
            val_ppl = self.validate()
            metrics["val_ppl"] = val_ppl
            self._maybe_grow(val_ppl, metrics)

        # Do not keep the autograd graph of this step alive through telemetry.
        self.brain.last_node_io = []
        # Measured at the very end of the step, after the life cycle and the
        # growth: this is the memory the next step starts from, not the peak of
        # the graph that has just been freed.
        memory = process_memory()
        metrics["ram_mb"] = round(memory["working_set"], 1)
        metrics["ram_peak_mb"] = round(memory["peak"], 1)
        metrics["commit_mb"] = round(memory["commit"], 1)
        if self.device.type == "cuda":
            # Reserved, not allocated: the caching allocator keeps whole
            # segments, and on WDDM each segment is backed by host commit —
            # which is how a run can die with a flat working set.
            metrics["gpu_alloc_mb"] = round(torch.cuda.memory_allocated() / 1048576.0, 1)
            metrics["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / 1048576.0, 1)
        return metrics

    def train_step(self) -> dict:
        """One optimizer step, with the OOM retry placed *outside* the except."""
        attempts = 0
        while True:
            try:
                return self.step_once()
            except RuntimeError as error:
                if "out of memory" not in str(error).lower() or self.batch_size <= 1:
                    raise
            # The except block is over: the failed graph is no longer pinned by
            # the exception, so the memory can actually be returned.
            attempts += 1
            if attempts >= 4:
                raise RuntimeError("out of memory even at batch size 1")
            self._release_memory()
            self.batch_size = max(1, self.batch_size // 2)
            self.log(f"  OOM: batch réduit à {self.batch_size} (le graphe reste creux)")

    def _maybe_grow(self, val_ppl: float, metrics: dict) -> None:
        before = self.brain.n_nodes
        captured = capture_optimizer_state(self.brain, self.optimizer)
        record = apply_growth(self, val_ppl)
        if self.brain.n_nodes != before:
            self._rebuild_optimizer(captured)
        metrics["event"] = record["event"]
        if record["message"]:
            self.log(record["message"])
            self.events.append(record["message"])
        self.history.append(
            {
                "step": self.step,
                "event": record["event"],
                "val_ppl": round(val_ppl, 3),
                "pool": self.brain.n_nodes,
            }
        )

    def _rebuild_optimizer(self, captured: dict) -> None:
        """After a pool growth: same hyperparameters, moments re-attached by name."""
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer = torch.optim.AdamW(self.brain.parameters(), lr=lr)
        restored = restore_optimizer_state(self.brain, self.optimizer, captured)
        self._optimizer_pool = self.brain.n_nodes
        self.log(f"  optimiseur reconstruit après croissance ({restored} tenseurs re-attachés)")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> float:
        self.brain.eval()
        generator = torch.Generator().manual_seed(1234)
        batches = []
        if self.data_mode == "toy":
            x, y = get_batch(
                int(self.config["training"]["val_batch"]), self.seq_len, generator=generator
            )
            batches.append((x, y))
        else:
            for x, y in self.dataset.windows(int(self.config["training"]["val_batch"]), "val"):
                batches.append((x, y))
        if not batches:
            return float("nan")
        total = 0.0
        count = 0
        for x, y in batches:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.brain(x, grow=False)
            targets = y.reshape(-1)
            if self.data_mode == "toy":
                logits = logits[:, 1:]
                targets = y[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits.reshape(-1, self.brain.vocab), targets)
            total += float(loss)
            count += 1
        mean = total / max(1, count)
        self.brain.train()
        return math.exp(min(mean, 20.0))

    # ------------------------------------------------------------------
    # Reports, speech, persistence
    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        print(message, flush=True)

    def vitals(self, metrics: dict) -> str:
        return (
            f"perte {metrics['loss']:.4f} | ppl {metrics['ppl']:7.1f} | "
            f"arêtes {metrics['edges']:5d} | nœuds {metrics['nodes']:5d} | "
            f"arêtes_nées {metrics['edge_births']:5d} | nœuds_nés {metrics['node_births']:4d} | "
            f"morts-arêtes {metrics['edge_deaths']:4d} | "
            f"morts-nœuds {metrics['node_deaths']:4d} | cimetière {metrics['cemetery']:4d} | "
            f"RAM {metrics.get('ram_mb', 0.0):.0f} Mo "
            f"(pic {metrics.get('ram_peak_mb', 0.0):.0f}, commit {metrics.get('commit_mb', 0.0):.0f})"
            + (
                f" | GPU {metrics['gpu_alloc_mb']:.0f}/{metrics['gpu_reserved_mb']:.0f} Mo"
                if "gpu_reserved_mb" in metrics
                else ""
            )
        )

    def speak(self, prompt: str = TEXT_PROMPT, tokens: int = 40, seed: int | None = None):
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer: speech needs --data text")
        ids, trace = self.brain.generate(
            self.tokenizer.encode(prompt).ids,
            max_new_tokens=tokens,
            seed=seed,
            eos_id=self.tokenizer.token_to_id("<eos>"),
        )
        return self.tokenizer.decode(ids), trace

    def save(self, path: str | Path, latest: bool = True) -> Path:
        path = Path(path)
        saved = save_genome(
            path,
            self.brain,
            step=self.step,
            optimizer=self.optimizer,
            growth=self.growth,
            tokenizer=getattr(self.tokenizer, "name", None) or "data/tokenizer.json",
            history=self.history,
            extra={
                "data": self.data_mode,
                "scale": self.scale,
                "data_rng": self.generator.get_state(),
                "events": self.events[-200:],
            },
        )
        if latest:
            save_genome(
                LATEST_PATH,
                self.brain,
                step=self.step,
                optimizer=self.optimizer,
                growth=self.growth,
                tokenizer="data/tokenizer.json",
                history=self.history,
                extra={
                    "data": self.data_mode,
                    "scale": self.scale,
                    "data_rng": self.generator.get_state(),
                    "events": self.events[-200:],
                },
            )
        return saved

    @classmethod
    def resume(
        cls,
        path: str | Path = CHECKPOINT_PATH,
        config: dict | None = None,
        device: str | torch.device | None = None,
        **kwargs,
    ) -> "Trainer":
        blob = load_genome(path)
        brain = brain_from_genome(blob)
        data = kwargs.pop("data", blob.get("extra", {}).get("data", "text"))
        seed = int(kwargs.pop("seed", blob.get("extra", {}).get("seed", 42)))
        kwargs.pop("brain", None)
        kwargs.pop("step", None)
        kwargs.pop("history", None)
        trainer = cls(
            config=config,
            data=data,
            device=device,
            seed=seed,
            brain=brain,
            step=int(blob["step"]),
            history=blob.get("history", []),
            **kwargs,
        )
        restore_optimizer_state(trainer.brain, trainer.optimizer, blob.get("optimizer"))
        trainer.growth.load_state_dict(blob.get("growth"))
        trainer.events = list(blob.get("extra", {}).get("events", []))
        set_rng_state(blob.get("rng"))
        data_rng = blob.get("extra", {}).get("data_rng")
        if data_rng is not None:
            trainer.generator.set_state(data_rng)
        trainer.log(
            f"reprise à step {trainer.step} | pool {trainer.brain.n_nodes} | "
            f"arêtes {trainer.brain.ne} | cimetière {len(trainer.brain.cemetery)}"
        )
        return trainer

    def run(self, steps: int, callback=None) -> dict:
        last: dict = {}
        for _ in range(int(steps)):
            last = self.train_step()
            if callback is not None:
                callback(self, last)
            if self.report_every and self.step % self.report_every == 0:
                self.log(f"step {self.step:6d} | {self.vitals(last)}")
                self.history.append(
                    {
                        "step": self.step,
                        "loss": round(last["loss"], 4),
                        "ppl": round(last["ppl"], 2),
                        "edges": last["edges"],
                        "nodes": last["nodes"],
                        "pool": last["pool"],
                        "ram_mb": last.get("ram_mb"),
                    }
                )
        return last


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train the mycelia brain")
    parser.add_argument("--data", choices=["toy", "text"], default="toy")
    parser.add_argument("--scale", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--hops", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", nargs="?", const=str(CHECKPOINT_PATH), default=None)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--report-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    config = load_config(args.config)
    training = config["training"]
    steps = args.steps if args.steps is not None else int(
        training["text_steps"] if args.data == "text" else training["toy_steps"]
    )

    # C1: one writer per checkpoint. A second trainer does not just steal the
    # GPU, it writes the same genome.pt and genome_latest.pt.
    lock = acquire_lock()

    if args.resume:
        trainer = Trainer.resume(args.resume, config=config, device=args.device)
    else:
        trainer = Trainer(
            config=config,
            data=args.data,
            device=args.device,
            seed=args.seed,
            scale=args.scale,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            **({"max_hops": args.hops} if args.hops else {}),
        )
    if args.report_every:
        trainer.report_every = args.report_every
    # --save-every stays available as an extra trigger; the clock is the rule.

    brain = trainer.brain
    print(f"device: {trainer.device} | {scale_label(args.scale, brain.n_nodes)}")
    print(
        f"dim {brain.dim} | hops {brain.max_hops} | arêtes utilisées {brain.ne} "
        f"/ capacité {brain.max_edges} | vocab {brain.vocab}"
    )
    print(
        f"verrou {lock} (PID {os.getpid()}) | checkpoint toutes les "
        f"{trainer.save_seconds / 60:.0f} min, premier au pas {trainer.first_save_step}"
    )

    started = time.time()
    last_save = started
    # C5: the rolling mean of the last steps, never elapsed/steps, which can
    # only rise and hides every slow interval behind the fastest one.
    window: deque[float] = deque(maxlen=20)
    write_history(trainer.history)

    def checkpoint(reason: str) -> None:
        nonlocal last_save
        trainer.save(args.checkpoint)
        write_history(trainer.history)
        last_save = time.time()
        trainer.log(
            f"  checkpoint ({reason}) à step {trainer.step}, "
            f"{time.time() - started:.0f} s de run"
        )

    try:
        for _ in range(steps):
            step_started = time.perf_counter()
            metrics = trainer.train_step()
            window.append(time.perf_counter() - step_started)

            if trainer.report_every and trainer.step % trainer.report_every == 0:
                rolling = 1000.0 * sum(window) / len(window)
                trainer.log(
                    f"step {trainer.step:6d} | {trainer.vitals(metrics)} | "
                    f"{rolling:.0f} ms/step (moyenne des {len(window)} derniers)"
                )
                if trainer.data_mode == "text":
                    text, _ = trainer.speak(
                        tokens=int(training["text_seq"] // 3), seed=trainer.step
                    )
                    trainer.log(f"  « {text.strip()} »")
                trainer.history.append(
                    {
                        "step": trainer.step,
                        "loss": round(metrics["loss"], 4),
                        "ppl": round(metrics["ppl"], 2),
                        "edges": metrics["edges"],
                        "nodes": metrics["nodes"],
                        "pool": metrics["pool"],
                        "edge_births": metrics["edge_births"],
                        "node_births": metrics["node_births"],
                        "ram_mb": metrics.get("ram_mb"),
                    }
                )
                # Diagnostic: does a collection return anything? If yes, objects
                # were dead but unreachable through reference counting alone.
                if os.environ.get("MYCELIA_NO_GC_PROBE") != "1":
                    before_gc = process_memory()["working_set"]
                    collected = gc.collect()
                    after_gc = process_memory()["working_set"]
                    trainer.log(
                        f"  gc.collect() : {before_gc:.0f} → {after_gc:.0f} Mo "
                        f"({after_gc - before_gc:+.0f}, {collected} objets)"
                    )
                # C2: atomic and incremental, so the cockpit never reads a stale
                # or half-written history.
                write_history(trainer.history)

            if trainer.step == trainer.first_save_step:
                checkpoint("premier")
            elif trainer.save_seconds > 0 and time.time() - last_save >= trainer.save_seconds:
                checkpoint("horloge")
            if args.save_every and trainer.step % args.save_every == 0:
                checkpoint(f"tous les {args.save_every} pas")
    except KeyboardInterrupt:
        # Between two steps, so the state is clean by construction.
        checkpoint("interruption")
        print(f"\ninterrompu proprement à step {trainer.step}.")
        print(f"reprenez avec : python mycelia/train.py --data {trainer.data_mode} --resume")
        return
    finally:
        release_lock(lock)

    checkpoint("fin de run")

    print("\ntrace de circulation du dernier token")
    if trainer.data_mode == "text":
        sample_x, _ = trainer.dataset.sample_batch(1, generator=trainer.generator)
    else:
        sample_x, _ = get_batch(1, trainer.seq_len)
    with torch.no_grad():
        _, trace = brain(sample_x.to(trainer.device), grow=False)
    print(f"  {hop_summary(trace)}")
    path = extract_path(trace)
    print(f"  chemin : {' -> '.join(map(str, path)) if path else 'aucun'}")
    if path:
        print(f"  nœuds intermédiaires : {len(path) - 2}")
    print(f"\ngenome : {args.checkpoint}")
    print("suivant : python scripts/generate.py --prompt \"Il était une fois\"")


if __name__ == "__main__":
    main()
