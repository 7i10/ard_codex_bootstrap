"""Numerical contracts shared by teacher-target construction and KL loss."""

from __future__ import annotations

import torch

from ard.device_checks import require


def validate_probability_distribution(probabilities: torch.Tensor, *, name: str = "target probabilities") -> None:
    """Reject malformed distributions without rewriting a valid FP32 target."""
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(f"{name} must be a [batch, class] tensor with at least two classes")
    if not probabilities.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    # require(): a host check, or a device assert inside a captured CUDA graph (plan 0105).
    require(torch.isfinite(probabilities).all(), f"{name} must be finite", FloatingPointError)
    require(~(probabilities < 0).any(), f"{name} must be non-negative")
    # Four ULPs per class permits normal FP reduction error but not a material
    # probability-mass discrepancy.  This is validation only: risk-zero rows
    # must retain exact baseline softmax bits.  (``isclose(...).all()`` is
    # ``torch.allclose`` without its host read.)
    row_sum_atol = 4.0 * torch.finfo(probabilities.dtype).eps * probabilities.shape[1]
    require(
        torch.isclose(probabilities.sum(dim=1), torch.ones_like(probabilities[:, 0]), rtol=0, atol=row_sum_atol).all(),
        f"{name} must sum to one",
    )
