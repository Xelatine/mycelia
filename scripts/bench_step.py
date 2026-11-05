"""Benchmark of one training step, on the *exact* path a run takes.

Why this file exists: the acceptance figure ``0,283 s/pas à 2 000 arêtes``
(docs/RAPPORT-v4.md §6.3) was produced by an ad-hoc measurement that no file in
the repository reproduces — it does not say whether it timed a forward, a
forward with backward, or a whole step with the life cycle, the validation and
the checkpoint. A number nobody can reproduce is not a measurement, it is a
memory. This script measures the three layers separately, in the same process,
on the same wiring, so the difference between them is attributable.

Usage (from the repository root):

    python scripts/bench_step.py --batch 16 --seq 10 --edges 2200
    python scripts/bench_step.py --batch 16 --seq 128 --edges 2000 --warmup 3

What is timed, in order, on the same brain:

1. ``forward``      — diffusion alone, ``torch.no_grad``, no births;
2. ``fwd+bwd``      — diffusion and backward, no optimizer step;
3. ``step_once``    — the real thing: births, penalties, backward, clip,
   optimizer step, life cycle, growth decision when due.

The median of the timed iterations is reported, never the mean: one OOM retry
or one ``gc`` would drag an average and hide the regime.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.config import load_config, nodes_for_scale  # noqa: E402
from mycelia.train import Trainer, process_memory  # noqa: E402


def build_trainer(
    config: dict, batch: int, seq: int, device: str | None, data: str = "toy"
) -> Trainer:
    return Trainer(
        config=config,
        data=data,
        device=device,
        scale=int(config["brain"]["scale"]),
        batch_size=batch,
        seq_len=seq,
    )


def wire_to(brain, edges: int, seed: int = 0) -> int:
    """Add random synapses until the wiring holds ``edges`` of them."""
    generator = torch.Generator().manual_seed(seed)
    n = brain.n_nodes
    added = 0
    while brain.ne < min(edges, brain.max_edges) and added < 10 * edges:
        source = int(torch.randint(0, n, (1,), generator=generator))
        target = int(torch.randint(0, n, (1,), generator=generator))
        if brain.add_edge(source, target):
            added += 1
    return brain.ne


def timeit(call, iterations: int) -> tuple[float, float, float]:
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), min(samples), max(samples)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="time one training step, layer by layer")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--edges", type=int, default=2200)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data", choices=["toy", "text"], default="toy")
    # One process per measurement when the point is memory: torch's CUDA cache
    # keeps the high-water mark of everything measured before it.
    parser.add_argument("--only", choices=["all", "step"], default="all")
    # Overrides the config the way a run would, so the effect of a knob is a
    # measurement and not an opinion.
    parser.add_argument("--tbptt", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    config = load_config()
    default_seq = config["training"]["text_seq" if args.data == "text" else "toy_seq"]
    seq = int(args.seq if args.seq is not None else default_seq)
    trainer = build_trainer(config, args.batch, seq, args.device, args.data)
    if args.tbptt is not None:
        trainer.tbptt = int(args.tbptt)
    brain = trainer.brain
    reachable = wire_to(brain, args.edges)
    print(
        f"device {trainer.device} | batch {args.batch} | seq {seq} | "
        f"arêtes {reachable} | nœuds {brain.n_nodes} | dim {brain.dim} | hops {brain.max_hops} | "
        f"tbptt {trainer.tbptt} | data {args.data}"
    )

    tokens = torch.randint(0, brain.vocab, (args.batch, seq), device=trainer.device)

    def forward_only() -> None:
        # trace=False on purpose: the real step never builds the telemetry
        # trace, and building it costs a .tolist() of up to trace_limit pairs
        # per hop — a benchmark that leaves it on measures Python, not the brain.
        with torch.no_grad():
            brain(tokens, grow=False, trace=False)

    state = None
    if args.only != "step":
        _logits, _state, _trace = brain(tokens, state=None, return_state=True, trace=False)
        state = _state.detach()

    def forward_backward() -> None:
        logits, carried, _ = brain(tokens, state=state, return_state=True, trace=False)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, brain.vocab),
            torch.zeros(args.batch * seq, dtype=torch.long, device=trainer.device),
        )
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _ = carried

    def full_step() -> None:
        trainer.step_once()

    # Growth and validation are periodic; the benchmark must time the regime,
    # so the step counter is parked far from every period.
    trainer.step = 10_007

    layers = [("step_once", full_step)]
    if args.data == "toy" and args.only != "step":
        # A text sequence of 128 positions in ONE graph is exactly what the run
        # never does (it truncates by blocks); timing it would measure a bug.
        layers = [("forward", forward_only), ("fwd+bwd", forward_backward), ("step_once", full_step)]

    for name, call in layers:
        for _ in range(args.warmup):
            call()
        before = process_memory()
        median, low, high = timeit(call, args.iters)
        memory = process_memory()
        gpu = ""
        if trainer.device.type == "cuda":
            gpu = (
                f" | GPU {torch.cuda.memory_allocated() / 1048576.0:.0f}"
                f"/{torch.cuda.memory_reserved() / 1048576.0:.0f} Mo"
            )
        # The commit drift per call is the number that matters: a positive
        # slope at fixed wiring is a reservation leak, a flat one is not.
        drift = (memory["commit"] - before["commit"]) / max(1, args.iters)
        print(
            f"{name:9s} : médiane {median * 1000:7.0f} ms/pas "
            f"[{low * 1000:.0f}–{high * 1000:.0f}] | "
            f"RAM {memory['working_set']:.0f} Mo (pic {memory['peak']:.0f}, "
            f"commit {memory['commit']:.0f}, +{drift:.0f}/pas){gpu}"
        )


if __name__ == "__main__":
    main()
