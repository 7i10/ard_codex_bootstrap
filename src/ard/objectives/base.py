"""Outer-objective contracts retain per-sample components until policy reduction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ard.policies import PolicyWeights


@dataclass(frozen=True)
class ObjectiveTerms:
    hard: torch.Tensor
    kd: torch.Tensor
    regularization: torch.Tensor
    adversarial_kd: torch.Tensor | None = None
    clean_kd: torch.Tensor | None = None

    def __post_init__(self) -> None:
        shapes = {self.hard.shape, self.kd.shape, self.regularization.shape}
        if len(shapes) != 1 or self.hard.ndim != 1:
            raise ValueError("objective components must be same-shape unreduced vectors")
        for branch in (self.adversarial_kd, self.clean_kd):
            if branch is not None and branch.shape != self.kd.shape:
                raise ValueError("exposed KD branches must match the unreduced objective batch")

    @property
    def total(self) -> torch.Tensor:
        return self.hard + self.kd + self.regularization

    def apply_policy(self, weights: PolicyWeights) -> ObjectiveTerms:
        if self.hard.shape != weights.hard_weight.shape:
            raise ValueError("policy weights do not match objective batch")
        return ObjectiveTerms(
            hard=self.hard * weights.hard_weight,
            kd=self.kd * weights.kd_weight,
            regularization=self.regularization,
            adversarial_kd=self.adversarial_kd,
            clean_kd=self.clean_kd,
        )


class DistillationObjective(ABC):
    requires_clean_student_logits = False
    requires_teacher_clean_logits = False

    @abstractmethod
    def __call__(
        self,
        *,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
        clean_student_logits: torch.Tensor | None = None,
        adversarial_target_probabilities: torch.Tensor | None = None,
    ) -> ObjectiveTerms:
        raise NotImplementedError
