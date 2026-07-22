"""RSLAD KD total plus a separately policy-controlled adversarial CE fallback."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import DistillationObjective, ObjectiveTerms
from .kl import probabilities_to_student_kl, target_to_student_kl


class RSLADObjective(DistillationObjective):
    """Complete 5/6 adversarial plus 1/6 clean RSLAD KD before policy reduction."""

    ADVERSARIAL_COEFFICIENT = 5.0 / 6.0
    CLEAN_COEFFICIENT = 1.0 / 6.0
    requires_clean_student_logits = True
    requires_teacher_clean_logits = True

    def __init__(self, *, temperature: float = 1.0, temperature_squared: bool = True) -> None:
        self.temperature = temperature
        self.temperature_squared = temperature_squared

    def __call__(
        self,
        *,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
        clean_student_logits: torch.Tensor | None = None,
        adversarial_target_probabilities: torch.Tensor | None = None,
    ) -> ObjectiveTerms:
        if teacher_logits is None or clean_student_logits is None:
            raise ValueError("RSLAD requires clean teacher and clean student logits")
        if adversarial_target_probabilities is None:
            adversarial_kd = target_to_student_kl(
                student_logits=student_logits,
                target_logits=teacher_logits,
                temperature=self.temperature,
                temperature_squared=self.temperature_squared,
            )
        else:
            adversarial_kd = probabilities_to_student_kl(
                student_logits=student_logits,
                target_probabilities=adversarial_target_probabilities,
                temperature=self.temperature,
                temperature_squared=self.temperature_squared,
            )
        clean_kd = target_to_student_kl(
            student_logits=clean_student_logits,
            target_logits=teacher_logits,
            temperature=self.temperature,
            temperature_squared=self.temperature_squared,
        )
        return ObjectiveTerms(
            hard=F.cross_entropy(student_logits, labels, reduction="none"),
            kd=self.ADVERSARIAL_COEFFICIENT * adversarial_kd + self.CLEAN_COEFFICIENT * clean_kd,
            regularization=torch.zeros_like(adversarial_kd),
            adversarial_kd=adversarial_kd,
            clean_kd=clean_kd,
        )
