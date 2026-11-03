#!/usr/bin/env python
"""Train the Mycelia brain.

    python scripts/train.py --config configs/demo.yaml            # fresh run
    python scripts/train.py --config configs/demo.yaml --resume   # exact resume
    python scripts/train.py --scale 2 --steps 500                 # CLI overrides

``Ctrl+C`` writes the complete genome (weights, wiring, cemetery, phase, data
position) and exits cleanly, so ``--resume`` continues at the same step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mycelia.config import load_config  # noqa: E402
from mycelia.train.loop import train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Mycelia brain")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--scale", type=int, default=None, choices=[0, 1, 2, 3])
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=None, help="initial pool size")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true",
                        help="delete any existing checkpoint and start over")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config,
                      scale=args.scale,
                      dim=args.dim,
                      seed=args.seed,
                      device=args.device,
                      **({"model.nodes_initial": args.nodes} if args.nodes else {}))
    run_dir = Path(args.run_dir or cfg.run_dir)
    if args.fresh:
        for name in ("checkpoint.pt", "state.json", "events.jsonl", "train.jsonl"):
            (run_dir / name).unlink(missing_ok=True)
        args.resume = False

    print(f"[mycelia] config={args.config} run={run_dir} "
          f"scale={cfg.model.scale} dim={cfg.model.dim} nodes={cfg.model.nodes_initial}",
          flush=True)
    result = train(cfg, steps=args.steps, resume=args.resume, run_dir=str(run_dir),
                   device=args.device, quiet=args.quiet)
    print(f"[mycelia] {result['state']} at step {result['step']} · "
          f"{result['nodes']} nodes · {result['edges']} edges", flush=True)
    sample = (result.get("sample") or {}).get("text")
    if sample:
        print(f"[mycelia] sample: {sample}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
