"""Teacher-only per-sample entropy measurements."""

from __future__ import annotations

import math

import torch


def shannon_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return finite per-sample Shannon entropy in nats for [batch, class] logits."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("entropy requires logits with shape [batch, class>=2]")
    probabilities = torch.softmax(logits.float(), dim=1)
    log_probabilities = torch.log_softmax(logits.float(), dim=1)
    entropy = -(probabilities * log_probabilities).sum(dim=1)
    if not torch.isfinite(entropy).all():
        raise FloatingPointError("teacher entropy is non-finite")
    # Entropy can differ from the mathematical range by only roundoff; do not
    # clamp because the policy must expose scientific mismatches.
    upper_bound = math.log(logits.shape[1])
    if bool((entropy < -1e-6).any()) or bool((entropy > upper_bound + 1e-5).any()):
        raise FloatingPointError("teacher entropy is outside Shannon bounds")
    return entropy
