"""Standard adversarial-training hard-label objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import DistillationObjective, ObjectiveTerms


class PGDATObjective(DistillationObjective):
    """Plan 0102 Workstream A: ``label_smoothing`` reproduces Singh/Croce/Hein
    2023's own pretrained-init ImageNet recipe (arXiv:2303.01870, Appendix
    A.1: "label smoothing coefficient of 0.1"). Default 0.0 reproduces
    today's exact behavior (plain hard-label CE) for every existing caller.
    Deliberately NOT applied when ``adversarial_target_probabilities`` is
    supplied (a soft/mixed target, e.g. from CutMix/MixUp) -- their own code
    (``revisiting-at/main.py``) does the same: label smoothing is folded
    into the mixing transform's own smoothing parameter instead of stacking
    a second smoothing on top, once mixing is active.
    """

    def __init__(self, *, label_smoothing: float = 0.0) -> None:
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must lie in [0, 1)")
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        *,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
        clean_student_logits: torch.Tensor | None = None,
        adversarial_target_probabilities: torch.Tensor | None = None,
        rectified_target_probabilities: torch.Tensor | None = None,
    ) -> ObjectiveTerms:
        del teacher_logits, clean_student_logits, rectified_target_probabilities
        if adversarial_target_probabilities is not None:
            log_probabilities = F.log_softmax(student_logits, dim=1)
            hard = -(adversarial_target_probabilities * log_probabilities).sum(dim=1)
        else:
            hard = F.cross_entropy(student_logits, labels, reduction="none", label_smoothing=self.label_smoothing)
        zeros = torch.zeros_like(hard)
        return ObjectiveTerms(hard=hard, kd=zeros, regularization=zeros)
