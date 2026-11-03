"""Launch the cockpit.

    python scripts/dashboard.py --train --data text --scale 0
    python scripts/dashboard.py --checkpoint data/genome.pt     # observation

Then open http://127.0.0.1:8080. The training thread starts immediately in
--train mode: the point is to watch the colony wire itself, not to click start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from dashboard.runner import TrainingRunner  # noqa: E402
from dashboard.server import create_app  # noqa: E402
from mycelia.checkpoint import load_genome  # noqa: E402
from mycelia.config import load_config, nodes_for_scale  # noqa: E402
from mycelia.data import load_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mycelia live cockpit")
    parser.add_argument("--train", action="store_true", help="train live (default)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="observe a saved genome instead")
    parser.add_argument("--data", choices=["toy", "text"], default="text")
    parser.add_argument("--scale", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    config = load_config()
    observe = args.checkpoint is not None and not args.train
    checkpoint = str(args.checkpoint) if args.checkpoint else None

    if observe:
        blob = load_genome(checkpoint)
        data = blob.get("extra", {}).get("data", args.data)
    else:
        data = args.data

    runner = TrainingRunner(
        config=config,
        data=data,
        scale=args.scale,
        device=args.device,
        max_steps=args.steps,
        checkpoint=checkpoint if (observe or args.resume) else None,
        resume=args.resume,
        observe=observe,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    if not observe:
        runner.start()

    host = args.host or config["dashboard"]["host"]
    port = int(args.port or config["dashboard"]["port"])
    brain = runner.trainer.brain
    print(f"Mycelia cockpit — {nodes_for_scale(args.scale)} nodes planned, pool {brain.n_nodes}")
    print(f"  mode      : {'observation' if observe else 'entraînement en direct'}")
    print(f"  cerveau   : {brain.n_nodes} nœuds, {brain.ne} arêtes utilisées, {brain.max_hops} tours")
    print(f"  corpus    : {data}")
    print(f"  URL       : http://{host}:{port}")
    uvicorn.run(create_app(runner), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
