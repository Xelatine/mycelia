"""Training loop: growth, plateaus, clean pause/resume, live telemetry.

The loop owns four moving parts:

* the brain forward (message passing) and the soft auxiliary losses;
* the lifecycle tick (link deaths, node deaths, births, reallocations);
* the tier controller (growth ↔ optimisation) driven by validation;
* the telemetry the cockpit reads, checkpointing, and the pause/stop files.

``Ctrl+C`` (or the dashboard's stop button) writes the full genome and exits
cleanly; ``--resume`` restores the same step, RNG and data position.
"""

from __future__ import annotations

import json
import math
import random
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from mycelia.config import Config
from mycelia.data.corpus import resolve_corpus
from mycelia.data.dataset import TokenStore
from mycelia.data.tokenizer import resolve_tokenizer
from mycelia.gen.sampling import generate
from mycelia.models.brain import BrainNetwork
from mycelia.models.lifecycle import Lifecycle
from mycelia.train.checkpoint import (load_checkpoint, save_checkpoint)
from mycelia.train.phases import TierController
from mycelia.train.telemetry import Telemetry

PROMPTS = [
    "Il était une fois",
    "Un jour, un petit garçon",
    "La petite fille ouvrit la porte et",
]


def pick_device(name: str = "auto") -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at(step: int, cfg: Config) -> float:
    warm = max(1, int(cfg.train.warmup))
    if step < warm:
        return cfg.train.lr * (step + 1) / warm
    progress = min(1.0, (step - warm) / max(1, cfg.train.steps - warm))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = cfg.train.min_lr_ratio
    return cfg.train.lr * (floor + (1.0 - floor) * cosine)


def tau_at(step: int, cfg: Config) -> float:
    span = max(1, int(cfg.train.tau_anneal_steps))
    frac = min(1.0, step / span)
    return float(cfg.train.tau_start + (cfg.train.tau_end - cfg.train.tau_start) * frac)


def build_brain(cfg: Config, vocab_size: int, device: torch.device) -> BrainNetwork:
    return BrainNetwork(cfg, vocab_size).to(device)


def train(cfg: Config, steps: Optional[int] = None, resume: bool = False,
          run_dir: Optional[str] = None, device: Optional[str] = None,
          quiet: bool = False) -> Dict[str, Any]:
    run_path = Path(run_dir or cfg.run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    total_steps = int(steps or cfg.train.steps)
    if steps:
        cfg.train.steps = int(steps)
    dev = pick_device(device or cfg.train.device)
    seed_everything(cfg.seed)

    corpus = resolve_corpus(cfg.data)
    tokenizer, tok_note = resolve_tokenizer(corpus.path, cfg.data.vocab_size,
                                           Path(cfg.data.cache_dir))
    store = TokenStore.build(corpus.path, tokenizer, Path(cfg.data.cache_dir),
                            cfg.data.max_chars)
    data_rng = np.random.default_rng(cfg.seed)

    telemetry = Telemetry(run_path, max_events=cfg.dashboard.max_events)
    brain = build_brain(cfg, tokenizer.vocab_size, dev)
    optimizer = torch.optim.AdamW(brain.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay, betas=(0.9, 0.95))
    lifecycle = Lifecycle(brain, cfg, seed=cfg.seed, emit=telemetry.emit)
    tier = TierController(cfg, brain, lifecycle, emit=telemetry.emit)

    ckpt_path = run_path / "checkpoint.pt"
    step = 0
    best_val = float("inf")
    curve: List[Dict[str, Any]] = []
    sample: Dict[str, Any] = {}
    start_time = time.time()
    tau = cfg.train.tau_start

    if resume:
        if ckpt_path.exists():
            payload = load_checkpoint(ckpt_path, brain, optimizer, lifecycle, tier)
            step = int(payload.get("step", 0))
            tau = float(payload.get("tau", cfg.train.tau_start))
            extra = payload.get("extra", {})
            best_val = float(extra.get("best_val", float("inf")))
            curve = list(extra.get("curve", []))
            sample = dict(extra.get("sample", {}))
            if payload.get("data_rng"):
                data_rng.bit_generator.state = payload["data_rng"]
            telemetry.emit(type="resume", step=step,
                           message=f"resumed from step {step} "
                                   f"({brain.nodes.n_nodes} nodes, {brain.adj.n_edges} links)")
        else:
            telemetry.emit(type="warn", step=0,
                           message="no checkpoint found — starting a fresh genome")

    if not resume:
        for name in ("events.jsonl", "train.jsonl", "state.json"):
            (run_path / name).unlink(missing_ok=True)
        telemetry.emit(type="init", step=0,
                       message=f"{brain.nodes.n_nodes} nodes seeded, "
                               f"{brain.adj.n_edges} links "
                               f"({brain.adj.n_edges / max(1, brain.nodes.n_nodes):.2f} links/node)",
                       pool=brain.nodes.n_nodes, edges=brain.adj.n_edges,
                       scale=cfg.model.scale, dim=cfg.model.dim,
                       vocab=tokenizer.vocab_size, corpus=corpus.source,
                       corpus_chars=corpus.chars, tokenizer=tok_note)

    control_path = run_path / "control.json"
    pause_path = run_path / ".pause"
    stop_path = run_path / ".stop"
    save_path = run_path / ".save"
    status = {"trainer": "running", "error": None, "reason": None}
    interrupt = {"flag": False}

    def handle_signal(signum, frame):  # noqa: ARG001
        interrupt["flag"] = True
        status["reason"] = "signal"

    old_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, handle_signal)
    except Exception:
        pass

    def read_control() -> Dict[str, Any]:
        if not control_path.exists():
            return {}
        try:
            return json.loads(control_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    last_metrics: Dict[str, Any] = {}

    def counters() -> Dict[str, Any]:
        out = dict(lifecycle.counters)
        out["phase"] = tier.phase
        return out

    def write_state(extra: Optional[Dict[str, Any]] = None) -> None:
        stats = brain.adj.stats()
        out_deg, in_deg = brain.adj.degree(brain.nodes.n_nodes)
        alive = int((~brain.nodes.dead_mask[: brain.nodes.n_nodes]).sum())
        payload: Dict[str, Any] = {
            "step": step,
            "steps_total": total_steps,
            "phase": tier.phase,
            "trainer_state": status["trainer"],
            "error": status["error"],
            "best_val": best_val,
            "params": brain.param_count(),
            "nodes": {
                "total": int(brain.nodes.n_nodes),
                "alive": alive,
                "dead": int(brain.nodes.dead_mask[: brain.nodes.n_nodes].sum()),
                "isolated": len(lifecycle.isolated),
                "capacity": int(brain.nodes.max_nodes),
            },
            "cemetery": len(lifecycle.cemetery),
            "edges": int(brain.adj.n_edges),
            "mean_mask": round(stats.get("mean_mask", 0.0), 4),
            "edges_per_node": round(brain.adj.n_edges / max(1, brain.nodes.n_nodes), 3),
            "degree": {"max_in": int(in_deg.max()) if in_deg.numel() else 0,
                       "max_out": int(out_deg.max()) if out_deg.numel() else 0},
            "counters": counters(),
            "curve": curve[-400:],
            "sample": sample,
            "phase_history": tier.transitions[-40:],
            "config": {"scale": cfg.model.scale, "dim": cfg.model.dim,
                       "beam": cfg.model.beam, "turns": cfg.model.max_turns,
                       "lr": round(cfg.train.lr, 8), "l1": cfg.loss.l1_edges,
                       "tau": round(tau, 4),
                       "corpus": corpus.source, "corpus_chars": corpus.chars,
                       "tokenizer": tok_note, "device": str(dev)},
            "runtime": {"seconds": round(time.time() - start_time, 1)},
            "data": store.info(),
            "live": last_metrics,
        }
        if extra:
            payload.update(extra)
        telemetry.write_state(payload)

    validate_every = max(1, int(cfg.train.val_every))
    val_batches = store.fixed_batches(cfg.train.batch_size, cfg.train.ctx_len,
                                      cfg.train.val_batches, seed=cfg.seed)

    def run_generation(tag: str) -> Dict[str, Any]:
        try:
            out = generate(brain, tokenizer,
                           prompt=PROMPTS[len(curve) % len(PROMPTS)],
                           max_new_tokens=24, temperature=0.85, top_p=0.9,
                           device=str(dev), track=True, max_ctx=cfg.train.ctx_len,
                           seed=cfg.seed + step)
            return {"text": out["text"], "step": step, "tag": tag,
                    "nodes_per_token": out["nodes_per_token"]}
        except Exception as exc:  # a sample must never kill a run
            return {"text": f"(sample failed: {exc})", "step": step, "tag": tag}

    def validate() -> float:
        brain.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in val_batches:
                out = brain(xb.to(dev), tau=tau, track=False, update_usage=False,
                            gumbel=False, step=step)
                total += float(F.cross_entropy(
                    out["logits"].reshape(-1, tokenizer.vocab_size),
                    yb.to(dev).reshape(-1)))
        brain.train()
        return total / max(1, len(val_batches))

    write_state()
    telemetry.emit(type="train_start", step=step, steps_total=total_steps,
                   message=f"training {max(0, total_steps - step)} steps on {dev}")

    try:
        for _ in range(total_steps - step):
            step += 1
            t0 = time.time()
            ctrl = read_control()
            if "lr" in ctrl:
                try:
                    cfg.train.lr = float(ctrl["lr"])
                except (TypeError, ValueError):
                    pass
            l1_scale = float(ctrl.get("l1_scale", 1.0) or 1.0)
            tau_scale = float(ctrl.get("tau_scale", 1.0) or 1.0)
            tau = tau_at(step, cfg) * max(0.1, min(3.0, tau_scale))
            lr = lr_at(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            xb, yb = store.batch(cfg.train.batch_size, cfg.train.ctx_len, data_rng)
            xb, yb = xb.to(dev), yb.to(dev)
            brain.train()
            out = brain(xb, tau=tau, track=False, update_usage=True,
                        gumbel=True, step=step)
            logits = out["logits"]
            ce = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size),
                                 yb.reshape(-1))
            aux = out["aux"]
            loss = (ce
                    + cfg.loss.l1_edges * l1_scale * aux["mask_l1"]
                    + cfg.loss.routing_entropy * aux["entropy"]
                    + cfg.loss.anti_redundancy * aux["redundancy"]
                    + cfg.loss.circulation * aux["circulation"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                brain.parameters(), cfg.train.clip))
            optimizer.step()

            # Growth score: a node is a birth candidate when it is used and when
            # its output actually differs from what it received.
            with torch.no_grad():
                n = brain.nodes.n_nodes
                usage = brain.nodes.usage[:n]
                novelty = brain.novelty[:n]
                target = torch.logit(
                    (0.15 + 0.85 * torch.sigmoid(2.0 * usage * (1.0 + 6.0 * novelty)))
                    .clamp(1e-4, 1 - 1e-4))
                brain.nodes.growth[:n] += 0.05 * (target - brain.nodes.growth[:n])

            if cfg.life.enabled and step % max(1, cfg.life.every) == 0:
                brain.step = step
                lifecycle.tick(step)
                if getattr(brain.adj, "resized", False):
                    brain.adj.resized = False
                    optimizer = torch.optim.AdamW(
                        brain.parameters(), lr=lr,
                        weight_decay=cfg.train.weight_decay, betas=(0.9, 0.95))
                    telemetry.emit(type="warn", step=step,
                                   message="edge capacity exceeded — mask parameter "
                                           "resized, optimiser moments reset")

            step_time = time.time() - t0
            last_metrics = {
                "loss": round(float(ce.detach()), 4),
                "aux": {"circ": round(float(aux["circulation"]), 4),
                        "entropy": round(float(aux["entropy"]), 4),
                        "redundancy": round(float(aux["redundancy"]), 5),
                        "mask_l1": round(float(aux["mask_l1"]), 5)},
                "energy": round(float(out["energy"]), 5),
                "hops": round(float(aux["hops"]), 2),
                "visited": int(aux["visited"]),
                "rows": int(aux["computed_rows"]),
                "flops_per_token": int(aux["computed_rows"] * 12 * brain.dim ** 2),
                "grad_norm": round(grad_norm, 3),
                "tau": round(tau, 4),
                "lr": round(lr, 7),
                "step_time": round(step_time, 3),
            }

            if step % max(1, cfg.train.log_every) == 0 or step == 1:
                ppl = math.exp(min(20.0, float(ce.detach())))
                curve.append({"step": step, "loss": round(float(ce.detach()), 4),
                              "ppl": round(ppl, 2), "edges": int(brain.adj.n_edges),
                              "nodes": int(brain.nodes.n_nodes),
                              "phase": tier.phase})
                with (run_path / "train.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"step": step, **last_metrics}) + "\n")
                if not quiet:
                    print(f"[{step:5d}/{total_steps}] loss {float(ce.detach()):.3f} "
                          f"ppl {ppl:7.2f} | nodes {brain.nodes.n_nodes} "
                          f"edges {brain.adj.n_edges} | {tier.phase} | "
                          f"{step_time * 1000:.0f} ms/step", flush=True)
                write_state()

            if step % validate_every == 0:
                val = validate()
                val_ppl = math.exp(min(20.0, val))
                best_val = min(best_val, val)
                phase_before = tier.phase
                tier.on_validation(step, val)
                last_metrics["val_loss"] = round(val, 4)
                last_metrics["val_ppl"] = round(val_ppl, 2)
                telemetry.emit(type="val", step=step, val_loss=round(val, 4),
                               ppl=round(val_ppl, 2), phase=tier.phase,
                               message=f"validation ppl {val_ppl:.2f}")
                curve.append({"step": step, "val_ppl": round(val_ppl, 2),
                              "phase": tier.phase})
                if tier.phase != phase_before:
                    telemetry.emit(type="tier", step=step, phase=tier.phase,
                                   message=f"nouvelle phase : {tier.phase}")
                write_state()

            if cfg.train.sample_every and step % int(cfg.train.sample_every) == 0:
                sample = run_generation("periodic")
                telemetry.emit(type="sample", step=step, text=sample["text"],
                               message="échantillon : " + sample["text"][:120])
                write_state()

            if cfg.train.checkpoint_every and step % int(cfg.train.checkpoint_every) == 0:
                save_checkpoint(ckpt_path, brain, optimizer, lifecycle, tier,
                                step=step, tau=tau, data_rng=data_rng,
                                extra={"best_val": best_val, "curve": curve,
                                       "sample": sample})
                telemetry.emit(type="checkpoint", step=step,
                               message=f"checkpoint sauvegardé au step {step}")

            if save_path.exists():
                save_path.unlink(missing_ok=True)
                save_checkpoint(ckpt_path, brain, optimizer, lifecycle, tier,
                                step=step, tau=tau, data_rng=data_rng,
                                extra={"best_val": best_val, "curve": curve,
                                       "sample": sample})
                telemetry.emit(type="checkpoint", step=step,
                               message="sauvegarde manuelle")

            if stop_path.exists():
                status["trainer"] = "stopped"
                status["reason"] = "stop-file"
                break
            if pause_path.exists():
                status["trainer"] = "paused"
                status["reason"] = "pause-file"
                break
            if interrupt["flag"]:
                status["trainer"] = "paused"
                status["reason"] = "signal"
                break
    except Exception as exc:  # keep the genome, surface the failure
        status["trainer"] = "error"
        status["error"] = f"{type(exc).__name__}: {exc}"
        telemetry.emit(type="error", step=step, message=status["error"])
        save_checkpoint(ckpt_path, brain, optimizer, lifecycle, tier, step=step,
                        tau=tau, data_rng=data_rng,
                        extra={"best_val": best_val, "curve": curve, "sample": sample})
        write_state({"trainer_state": "error"})
        raise
    finally:
        try:
            signal.signal(signal.SIGINT, old_handler)
        except Exception:
            pass

    if status["trainer"] != "error":
        sample = run_generation("final")
    save_checkpoint(ckpt_path, brain, optimizer, lifecycle, tier, step=step,
                    tau=tau, data_rng=data_rng,
                    extra={"best_val": best_val, "curve": curve, "sample": sample})
    if status["trainer"] == "running":
        status["trainer"] = "finished"
    write_state({"trainer_state": status["trainer"],
                 "stopped_reason": status["reason"], "best_val": best_val})
    telemetry.emit(type="train_end", step=step, state=status["trainer"],
                   message=f"entraînement {status['trainer']} au step {step}")
    return {"step": step, "state": status["trainer"], "best_val": best_val,
            "sample": sample, "nodes": int(brain.nodes.n_nodes),
            "edges": int(brain.adj.n_edges),
            "counters": dict(lifecycle.counters)}
