"""Detached, normalized teacher-target calibration for adversarial student KD."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .validation import validate_probability_distribution


@dataclass(frozen=True)
class TeacherTargetOutput:
    """Calibrated probabilities and their detached per-sample mixing strength."""

    probabilities: torch.Tensor
    rho: torch.Tensor

    def __post_init__(self) -> None:
        probabilities, rho = self.probabilities, self.rho
        if probabilities.ndim != 2 or rho.ndim != 1 or probabilities.shape[0] != rho.shape[0]:
            raise ValueError("target probabilities must be [batch, class] with a matching rho vector")
        if probabilities.shape[1] < 2:
            raise ValueError("teacher targets require at least two classes")
        if probabilities.requires_grad or rho.requires_grad:
            raise ValueError("teacher target probabilities and rho must be detached")
        validate_probability_distribution(probabilities, name="teacher target probabilities")
        if not torch.isfinite(rho).all():
            raise FloatingPointError("teacher target rho must be finite")
        if bool((rho < 0).any()) or bool((rho > 1).any()):
            raise ValueError("teacher target rho must be in [0, 1]")


class TeacherTargetPolicy(ABC):
    """Interface for detached teacher-target construction.

    This owns only target construction.  It deliberately does not decide KD or
    hard-label loss weights, so method identity cleanly separates target and
    reduction policy semantics.
    """

    requires_labels = False

    @abstractmethod
    def __call__(
        self,
        *,
        teacher_logits: torch.Tensor,
        risk: torch.Tensor,
        temperature: float,
        labels: torch.Tensor | None = None,
    ) -> TeacherTargetOutput:
        raise NotImplementedError


class UniformSofteningTeacherTargetPolicy(TeacherTargetPolicy):
    """Mix detached teacher-clean probabilities with uniform mass by detached risk."""

    def __init__(self, *, rho_max: float) -> None:
        if not 0 <= rho_max <= 1:
            raise ValueError("rho_max must be in [0, 1]")
        self.rho_max = float(rho_max)

    def __call__(
        self,
        *,
        teacher_logits: torch.Tensor,
        risk: torch.Tensor,
        temperature: float,
        labels: torch.Tensor | None = None,
    ) -> TeacherTargetOutput:
        if labels is not None:
            raise ValueError("uniform teacher-target policy does not consume labels")
        if teacher_logits.ndim != 2 or teacher_logits.shape[1] < 2:
            raise ValueError("teacher logits must be [batch, class] with at least two classes")
        if risk.ndim != 1 or risk.shape[0] != teacher_logits.shape[0]:
            raise ValueError("teacher target risk must be a batch-aligned vector")
        if temperature <= 0:
            raise ValueError("teacher target temperature must be positive")
        with torch.no_grad():
            detached_logits = teacher_logits.detach()
            detached_risk = risk.detach()
            if not torch.isfinite(detached_logits).all() or not torch.isfinite(detached_risk).all():
                raise FloatingPointError("teacher target logits and risk must be finite")
            rho = (self.rho_max * detached_risk.clamp(0.0, 1.0)).detach()
            teacher_probabilities = F.softmax(detached_logits / temperature, dim=1)
            uniform = torch.full_like(teacher_probabilities, 1.0 / teacher_probabilities.shape[1])
            mixed_probabilities = (1.0 - rho[:, None]) * teacher_probabilities + rho[:, None] * uniform
            probabilities = torch.where((rho == 0)[:, None], teacher_probabilities, mixed_probabilities).detach()
        return TeacherTargetOutput(probabilities=probabilities, rho=rho)


class IdentityTeacherTargetPolicy(UniformSofteningTeacherTargetPolicy):
    """Explicit identity target transform for branch-parity tests and future IDs."""

    def __init__(self) -> None:
        super().__init__(rho_max=0.0)


class TrueLabelMixTeacherTargetPolicy(TeacherTargetPolicy):
    """Mix selected detached teacher-clean targets with the true label.

    ``risk`` is an immutable binary intervention mask.  A selected sample
    (risk=1) receives exactly half teacher probability and half one-hot true
    label; an unselected sample (risk=0) is bitwise-equivalent to the ordinary
    detached teacher target.  No gradient can enter either the teacher logits,
    labels, or mask through this calibration boundary.
    """

    MIXING = 0.5
    requires_labels = True

    def __call__(
        self,
        *,
        teacher_logits: torch.Tensor,
        risk: torch.Tensor,
        temperature: float,
        labels: torch.Tensor | None = None,
    ) -> TeacherTargetOutput:
        if teacher_logits.ndim != 2 or teacher_logits.shape[1] < 2:
            raise ValueError("teacher logits must be [batch, class] with at least two classes")
        if risk.ndim != 1 or risk.shape[0] != teacher_logits.shape[0]:
            raise ValueError("teacher target risk must be a batch-aligned vector")
        if labels is None or labels.ndim != 1 or labels.shape[0] != teacher_logits.shape[0]:
            raise ValueError("true-label teacher-target policy requires batch-aligned labels")
        if temperature <= 0:
            raise ValueError("teacher target temperature must be positive")
        with torch.no_grad():
            detached_logits = teacher_logits.detach()
            detached_risk = risk.detach()
            detached_labels = labels.detach()
            if not torch.isfinite(detached_logits).all() or not torch.isfinite(detached_risk).all():
                raise FloatingPointError("teacher target logits and risk must be finite")
            if detached_labels.dtype.is_floating_point or bool(
                (detached_labels < 0).any() | (detached_labels >= detached_logits.shape[1]).any()
            ):
                raise ValueError("true-label teacher-target labels are outside the class range")
            if not bool(((detached_risk == 0) | (detached_risk == 1)).all()):
                raise ValueError("true-label teacher-target risk must be an immutable binary mask")
            teacher_probabilities = F.softmax(detached_logits / temperature, dim=1)
            one_hot = F.one_hot(detached_labels.long(), num_classes=teacher_logits.shape[1]).to(teacher_probabilities)
            rho = (self.MIXING * detached_risk).detach()
            mixed = (1.0 - rho[:, None]) * teacher_probabilities + rho[:, None] * one_hot
            probabilities = torch.where((rho == 0)[:, None], teacher_probabilities, mixed).detach()
        return TeacherTargetOutput(probabilities=probabilities, rho=rho)
