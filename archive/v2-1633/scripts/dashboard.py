#!/usr/bin/env python
"""Start the cockpit on http://127.0.0.1:8080

    python scripts/dashboard.py                    # watches runs/mycelia-demo
    python scripts/dashboard.py --run runs/other --port 8081
    python scripts/dashboard.py --no-control       # read-only mode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mycelia.dashboard.server import create_app  # noqa: E402


def newest_run() -> Path:
    runs = ROOT / "runs"
    candidates = [p for p in runs.glob("*") if p.is_dir()] if runs.exists() else []
    with_ckpt = [p for p in candidates if (p / "checkpoint.pt").exists()]
    pool = with_ckpt or candidates
    if not pool:
        return ROOT / "runs" / "mycelia-demo"
    return max(pool, key=lambda p: (p / "checkpoint.pt").stat().st_mtime
               if (p / "checkpoint.pt").exists() else p.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mycelia dashboard")
    parser.add_argument("--run", default=None, help="run directory to observe/drive")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-control", action="store_true",
                        help="disable start/pause/stop buttons")
    args = parser.parse_args()

    run_dir = Path(args.run) if args.run else ROOT / "runs" / "mycelia-demo"
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if args.run is None and not run_dir.exists():
        run_dir = newest_run()
    run_dir.mkdir(parents=True, exist_ok=True)

    app = create_app(run_dir, allow_control=not args.no_control)
    print(f"[dashboard] watching {run_dir}", flush=True)
    print(f"[dashboard] cockpit → http://{args.host}:{args.port}", flush=True)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
