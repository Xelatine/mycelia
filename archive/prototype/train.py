"""Phase 1 training: does the skeleton learn, and does the signal circulate?

Run from the repository root with ``python mycelia/train.py``. The script also
works from any directory because it puts the repository root on sys.path before
importing the package.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

# Running this file directly makes mycelia/ the import root, so the package
# itself has to be reachable from its parent directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mycelia.brain import Brain, extract_path  # noqa: E402
from mycelia.data import VOCAB, get_batch  # noqa: E402

STEPS = 2000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
L1_WEIGHT = 1e-4
REPORT_EVERY = 100
EDGE_ACTIVE = 0.5

torch.manual_seed(0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    brain = Brain().to(device)
    optimiser = torch.optim.AdamW(brain.parameters(), lr=LEARNING_RATE)
    print(f"nodes: {brain.n_nodes} | edges: {brain.num_edges} | hops: {brain.max_hops}")

    for step in range(1, STEPS + 1):
        x, y = get_batch(BATCH_SIZE)
        x, y = x.to(device), y.to(device)

        logits, _ = brain(x)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), y[:, :-1].reshape(-1))
        # L1 on the synapse strengths: useless links are pushed to zero, which
        # is exactly what phase 2 needs to let them die.
        l1 = L1_WEIGHT * brain.edge_mask.abs().sum()
        loss = ce + l1

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        if step % REPORT_EVERY == 0:
            active_edges = int((brain.edge_mask.detach() > EDGE_ACTIVE).sum())
            print(
                f"step {step:5d} | loss {loss.item():.4f} | ce {ce.item():.4f} | "
                f"active edges {active_edges}/{brain.num_edges} | "
                f"active nodes {len(brain.last_firing_nodes)}/{brain.n_nodes}"
            )

    # Final proof: one example sequence, one forward pass, the traffic printed
    # hop by hop, then the path the signal took from the door to the reader.
    x, _ = get_batch(1)
    with torch.no_grad():
        _, trace = brain(x.to(device))

    print("\ncirculation trace (hop -> sender: receivers)")
    for hop in sorted(trace):
        receivers_of: dict[int, list[int]] = defaultdict(list)
        for source, target in trace[hop]:
            receivers_of[source].append(target)
        parts = [f"{s} -> {sorted(receivers_of[s])}" for s in sorted(receivers_of)]
        print(f"  hop {hop}: " + ("; ".join(parts) if parts else "(silence)"))

    path = extract_path(trace)
    if path is None:
        print("path: none (the signal never reached node 1)")
    else:
        print(f"path: {' -> '.join(str(n) for n in path)}")
        print(f"intermediate nodes: {len(path) - 2}")


if __name__ == "__main__":
    main()
