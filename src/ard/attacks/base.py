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
    # Required only by the versioned sample-keyed random-start contract.
    source_ids: torch.Tensor | None = None
    epoch: int | None = None
    attack_seed: int | None = None
    stream_tag: str = "train_pgd"
    restart_index: int = 0
    capture_step: int | None = None
    # Optional per-example treatment budget.  These are only used by an
    # explicitly registered mixed-budget screen; when absent the resolved
    # AttackConfig scalar budget is used for every sample.
    epsilon_override: torch.Tensor | None = None
    step_size_override: torch.Tensor | None = None


@dataclass(frozen=True)
class AttackResult:
    adversarial: torch.Tensor
    initial_delta: torch.Tensor
    step_losses: tuple[float, ...]
    max_abs_delta: float
    captured_adversarial: torch.Tensor | None = None


class AttackGenerator(ABC):
    @property
    def requires_teacher_clean_target(self) -> bool:
        """Whether this attack consumes a detached clean-teacher target."""
        return False

    @abstractmethod
    def generate(self, request: AttackRequest) -> AttackResult:
        raise NotImplementedError
