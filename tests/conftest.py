"""Make the project importable from the tests, and keep them fast and offline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _deterministic_cpu():
    """Every test starts from a known RNG and never touches the GPU.

    The central continuity test compares losses to 1e-6: that is only
    meaningful on a deterministic device, so the suite stays on the CPU.
    """
    torch.manual_seed(0)
    # Bit-for-bit continuity can only be checked on a deterministic device:
    # a multi-threaded CPU reduction may sum in a different order from one run
    # to the next, which shows up as a 1e-6 wobble unrelated to the reload.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
