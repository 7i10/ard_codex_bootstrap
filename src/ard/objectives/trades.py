"""TRADES outer objective: clean CE plus beta times clean-to-adversarial KL."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import DistillationObjective, ObjectiveTerms
from .kl import target_to_student_kl


class TRADESObjective(DistillationObjective):
    requires_clean_student_logits = True

    def __init__(self, *, beta: float, temperature: float = 1.0, temperature_squared: bool = True) -> None:
        self.beta = beta
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
        del teacher_logits, adversarial_target_probabilities
        if clean_student_logits is None:
            raise ValueError("TRADES requires student logits on the clean input")
        hard = F.cross_entropy(clean_student_logits, labels, reduction="none")
        kd = self.beta * target_to_student_kl(
            student_logits=student_logits,
            target_logits=clean_student_logits,
            temperature=self.temperature,
            temperature_squared=self.temperature_squared,
        )
        return ObjectiveTerms(hard=hard, kd=kd, regularization=torch.zeros_like(hard))
