"""Standard adversarial-training hard-label objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import DistillationObjective, ObjectiveTerms


class PGDATObjective(DistillationObjective):
    def __call__(
        self,
        *,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
        clean_student_logits: torch.Tensor | None = None,
    ) -> ObjectiveTerms:
        hard = F.cross_entropy(student_logits, labels, reduction="none")
        zeros = torch.zeros_like(hard)
        return ObjectiveTerms(hard=hard, kd=zeros, regularization=zeros)
