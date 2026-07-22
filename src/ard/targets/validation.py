"""Numerical contracts shared by teacher-target construction and KL loss."""

from __future__ import annotations

import torch


def validate_probability_distribution(probabilities: torch.Tensor, *, name: str = "target probabilities") -> None:
    """Reject malformed distributions without rewriting a valid FP32 target."""
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(f"{name} must be a [batch, class] tensor with at least two classes")
    if not probabilities.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError(f"{name} must be finite")
    if bool((probabilities < 0).any()):
        raise ValueError(f"{name} must be non-negative")
    # Four ULPs per class permits normal FP reduction error but not a material
    # probability-mass discrepancy.  This is validation only: risk-zero rows
    # must retain exact baseline softmax bits.
    row_sum_atol = 4.0 * torch.finfo(probabilities.dtype).eps * probabilities.shape[1]
    if not torch.allclose(probabilities.sum(dim=1), torch.ones_like(probabilities[:, 0]), rtol=0, atol=row_sum_atol):
        raise ValueError(f"{name} must sum to one")
