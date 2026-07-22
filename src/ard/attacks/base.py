"""Typed boundary between training and inner maximization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AttackRequest:
    inputs: torch.Tensor
    labels: torch.Tensor
    student: nn.Module
    teacher: nn.Module | None = None
    target_logits: torch.Tensor | None = None
    generator: torch.Generator | None = None


@dataclass(frozen=True)
class AttackResult:
    adversarial: torch.Tensor
    initial_delta: torch.Tensor
    step_losses: tuple[float, ...]
    max_abs_delta: float


class AttackGenerator(ABC):
    @property
    def requires_teacher_clean_target(self) -> bool:
        """Whether this attack consumes a detached clean-teacher target."""
        return False

    @abstractmethod
    def generate(self, request: AttackRequest) -> AttackResult:
        raise NotImplementedError
