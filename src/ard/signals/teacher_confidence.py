"""Primitive teacher-confidence measurements for offline signal construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .teacher_entropy import shannon_entropy


@dataclass(frozen=True)
class TeacherConfidenceBatch:
    """Detached FP32 primitives; no proposed risk or gate is encoded here."""

    entropy: torch.Tensor
    true_probability: torch.Tensor
    max_wrong_probability: torch.Tensor
    prediction: torch.Tensor
    correct: torch.Tensor


def teacher_confidence_primitives(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> TeacherConfidenceBatch:
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("teacher confidence requires logits with shape [batch, class>=2]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("teacher confidence labels must match the logits batch")
    if valid_mask.shape != labels.shape or valid_mask.dtype != torch.bool:
        raise ValueError("teacher confidence valid_mask must match labels and be bool")
    if bool(((labels < 0) | (labels >= logits.shape[1])).any()):
        raise ValueError("teacher confidence labels are outside the class range")
    detached = logits.detach().float()
    probabilities = torch.softmax(detached, dim=1)
    true_probability = probabilities.gather(1, labels.reshape(-1, 1)).reshape(-1)
    wrong_probabilities = probabilities.clone()
    wrong_probabilities.scatter_(1, labels.reshape(-1, 1), float("-inf"))
    max_wrong_probability = wrong_probabilities.max(dim=1).values
    prediction = detached.argmax(dim=1)
    values = TeacherConfidenceBatch(
        entropy=shannon_entropy(detached).detach(),
        true_probability=true_probability.detach(),
        max_wrong_probability=max_wrong_probability.detach(),
        prediction=prediction.detach(),
        correct=prediction.eq(labels).detach(),
    )
    finite = torch.stack((values.entropy, values.true_probability, values.max_wrong_probability), dim=1)
    if bool((~torch.isfinite(finite[valid_mask])).any()):
        raise FloatingPointError("teacher confidence contains non-finite valid values")
    if bool(((finite[valid_mask, 1:] < 0.0) | (finite[valid_mask, 1:] > 1.0)).any()):
        raise FloatingPointError("teacher confidence probability is outside [0,1]")
    return values
