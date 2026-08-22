"""Explicit data-loader RNG ownership for reproducible source decomposition."""

from __future__ import annotations

import random

import torch


def data_loader_generator(seed: int) -> torch.Generator:
    """Create the CPU generator owned by sampler/worker data-side draws."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("data-loader seed must be a non-negative integer")
    return torch.Generator(device="cpu").manual_seed(seed)


def seed_data_loader_worker(worker_id: int) -> None:
    """Make Python/NumPy worker sources descendants of DataLoader's seed.

    PyTorch seeds the worker torch RNG before invoking this hook.  Mirroring
    that derived seed into Python and NumPy makes the ownership explicit for
    datasets that may use either library, while preserving the default
    DataLoader worker-seed distribution.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass
