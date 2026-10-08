"""Teacher-only per-sample entropy measurements."""

from __future__ import annotations

import math

import torch

from ard.device_checks import require


def shannon_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return finite per-sample Shannon entropy in nats for [batch, class] logits."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("entropy requires logits with shape [batch, class>=2]")
    probabilities = torch.softmax(logits.float(), dim=1)
    log_probabilities = torch.log_softmax(logits.float(), dim=1)
    entropy = -(probabilities * log_probabilities).sum(dim=1)
    # require(): a host check, or a device assert inside a captured CUDA graph (plan 0105).
    require(torch.isfinite(entropy).all(), "teacher entropy is non-finite", FloatingPointError)
    # Entropy can differ from the mathematical range by only roundoff; do not
    # clamp because the policy must expose scientific mismatches.
    upper_bound = math.log(logits.shape[1])
    require(
        ~((entropy < -1e-6).any() | (entropy > upper_bound + 1e-5).any()),
        "teacher entropy is outside Shannon bounds",
        FloatingPointError,
    )
    return entropy
