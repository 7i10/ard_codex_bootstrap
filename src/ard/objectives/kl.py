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
    detach_target: bool = True,
) -> torch.Tensor:
    """Return ``KL(target || student)`` for each sample.

    The direction is intentionally written in the function name because
    ``torch.kl_div`` takes log-Q first.

    ``detach_target`` defaults to True because the usual target here is a frozen
    teacher, for which detaching changes nothing and states the intent.  It must
    be False when the target is part of the network being trained, as it is in
    TRADES, where the KL term exists precisely to pull the clean and adversarial
    predictions towards each other and needs gradients through both.  Detaching
    it there silently trains the clean branch from the cross-entropy term alone;
    see docs/debugging/0028-trades-clean-target-detached.md, which measured a
    58 per cent difference in the gradient.
    """
    if student_logits.shape != target_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and target logits must be matching [batch, class] tensors")
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    scaled_target = (target_logits.detach() if detach_target else target_logits) / temperature
    if detach_target:
        target = F.softmax(scaled_target, dim=1)
        values = F.kl_div(log_student, target, reduction="none").sum(dim=1)
    else:
        # The same quantity in log space.  ``F.softmax`` can underflow to exactly
        # zero under mixed precision, and the target-side gradient then needs
        # ``log(0)``; ``log_softmax`` returns a large finite negative instead, so
        # ``p * log p`` stays finite.  Only the non-detached branch needs this,
        # because a detached target carries no gradient through ``log(target)``.
        log_target = F.log_softmax(scaled_target, dim=1)
        values = (log_target.exp() * (log_target - log_student)).sum(dim=1)
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
