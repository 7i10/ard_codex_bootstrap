"""Contracts for detached per-sample training measurements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SignalBatch:
    """One unreduced signal vector and the rows that are real samples."""

    values: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("signal values must be a one-dimensional vector")
        if self.valid_mask.ndim != 1 or self.valid_mask.dtype != torch.bool:
            raise ValueError("signal valid_mask must be a one-dimensional bool vector")
        if self.values.shape != self.valid_mask.shape:
            raise ValueError("signal values and valid_mask must have the same shape")
        if bool((~torch.isfinite(self.values) & self.valid_mask).any()):
            raise FloatingPointError("valid signal values must be finite")


class SampleSignal(ABC):
    """A named, detached measurement computed before the optimizer update."""

    @abstractmethod
    def compute(
        self,
        *,
        student_adv_logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> SignalBatch:
        raise NotImplementedError
