"""Explicit per-sample KL objectives used by the M2 baseline methods."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ard.targets.validation import validate_probability_distribution


def target_to_student_kl(
    *,
    student_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float,
    temperature_squared: bool,
) -> torch.Tensor:
    """Return ``KL(target || student)`` for each sample.

    The target is detached so teacher logits and the TRADES clean target do
    not receive outer-objective gradients.  The direction is intentionally
    written in the function name because ``torch.kl_div`` takes log-Q first.
    """
    if student_logits.shape != target_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and target logits must be matching [batch, class] tensors")
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    target = F.softmax(target_logits.detach() / temperature, dim=1)
    values = F.kl_div(log_student, target, reduction="none").sum(dim=1)
    return values * (temperature * temperature) if temperature_squared else values


def probabilities_to_student_kl(
    *,
    student_logits: torch.Tensor,
    target_probabilities: torch.Tensor,
    temperature: float,
    temperature_squared: bool,
) -> torch.Tensor:
    """Return KL(target probabilities || student) with an explicit detach contract."""
    if student_logits.shape != target_probabilities.shape or student_logits.ndim != 2:
        raise ValueError("student logits and target probabilities must be matching [batch, class] tensors")
    target = target_probabilities.detach()
    validate_probability_distribution(target)
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    values = F.kl_div(log_student, target, reduction="none").sum(dim=1)
    return values * (temperature * temperature) if temperature_squared else values
