#!/usr/bin/env python
"""Generate French text from a trained checkpoint and print the neural path.

    python scripts/generate.py --path runs/mycelia-demo/checkpoint.pt \
        --prompt "Il était une fois" --tokens 48
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mycelia.config import config_from_dict  # noqa: E402
from mycelia.data.tokenizer import FrenchBPE  # noqa: E402
from mycelia.gen.sampling import generate  # noqa: E402
from mycelia.models.brain import BrainNetwork  # noqa: E402
from mycelia.train.checkpoint import load_checkpoint  # noqa: E402


def find_tokenizer(cfg, root: Path) -> FrenchBPE:
    cache = root / cfg.data.cache_dir
    for candidate in sorted(cache.glob("tokenizer-fr-*.json")):
        try:
            return FrenchBPE.load(candidate)
        except Exception:
            continue
    raise FileNotFoundError("no tokenizer found in .cache — run scripts/train.py once")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate French text")
    parser.add_argument("--path", default="runs/mycelia-demo/checkpoint.pt")
    parser.add_argument("--prompt", default="Il était une fois")
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-path", action="store_true", help="hide the node path")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_absolute():
        path = ROOT / path
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = config_from_dict(payload["config"])
    brain = BrainNetwork(cfg, int(payload["vocab_size"]))
    load_checkpoint(path, brain)
    brain.to(torch.device(args.device))
    brain.eval()
    tokenizer = find_tokenizer(cfg, ROOT)

    out = generate(brain, tokenizer, prompt=args.prompt, max_new_tokens=args.tokens,
                   temperature=args.temperature, top_p=args.top_p,
                   device=args.device, track=True, max_ctx=cfg.train.ctx_len,
                   seed=args.seed)

    print("\n" + out["text"] + "\n")
    print(f"step {payload.get('step')} · {len(out['tokens'])} tokens · "
          f"{out['nodes_per_token']} nœuds/token"
          + (" · garde-fou anti-répétition" if out["guard_fired"] else ""))
    if not args.no_path:
        for i, tok in enumerate(out["tokens"], 1):
            edges = " ".join(f"{a}→{b}" for a, b in tok["edges"][:6])
            print(f"{i:3d}. {tok['token']!r:>12}  {len(tok['nodes'])} nœuds  {edges}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
