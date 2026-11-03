"""Toy data. No real corpus yet: phase 3 brings the French text.

The task is a copy task. It is the simplest task that forces information to
cross the whole graph while still requiring the brain to actually work.

Why same-token copy and not the one-step shift of a language model: every
operation in this architecture is per position (gate, body, router, message
passing), so position t never sees position t+1. A shifted target would be
information-theoretically impossible here and the loss would only settle at the
uniform baseline (~4.58). Same-token copy is learnable through the graph, which
is the one thing this phase has to prove. The shift becomes meaningful once
sequence wiring exists in a later phase.
"""

from __future__ import annotations

import torch

VOCAB = 100
FIRST_TOKEN = 2  # 0 and 1 stay free: they are reserved control slots
LAST_TOKEN = 99


def get_batch(
    batch_size: int, seq_len: int = 10, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x, y) of shape [batch_size, seq_len] as int64 token ids."""
    x = torch.randint(FIRST_TOKEN, LAST_TOKEN + 1, (batch_size, seq_len), generator=generator)
    # A copy target: predict the token that was just read. The signal must go
    # in through node 0, cross the sparse middle, and come out at node 1.
    y = x.clone()
    return x, y
