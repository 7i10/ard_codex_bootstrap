"""Policy outputs remain explicit until the trainer applies them to terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PolicyContext:
    """Batch validity plus the injected cross-rank scalar-min operation."""

    valid_mask: torch.Tensor
    global_min: Callable[[torch.Tensor], torch.Tensor]

    def __post_init__(self) -> None:
        if self.valid_mask.ndim != 1 or self.valid_mask.dtype != torch.bool:
            raise ValueError("policy valid_mask must be a one-dimensional bool tensor")


@dataclass(frozen=True)
class PolicyWeights:
    """Pre-reduction weights and the explicit risk that produced them."""

    hard_weight: torch.Tensor
    kd_weight: torch.Tensor
    joint_risk: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.hard_weight.shape != self.kd_weight.shape or self.kd_weight.ndim != 1:
            raise ValueError("policy weights must be same-shape unreduced vectors")
        if not torch.isfinite(self.hard_weight).all() or not torch.isfinite(self.kd_weight).all():
            raise FloatingPointError("policy weights must be finite")
        risk = self.joint_risk
        if risk is None:
            risk = torch.zeros_like(self.kd_weight)
            object.__setattr__(self, "joint_risk", risk)
        if risk.shape != self.kd_weight.shape or not torch.isfinite(risk).all():
            raise ValueError("policy joint_risk must be a finite unreduced vector")


# The prior name remains a source-compatible alias for integrations that used
# the proposal-stage terminology.
PolicyOutput = PolicyWeights


class WeightPolicy(ABC):
    required_signals: frozenset[str] = frozenset()

    @abstractmethod
    def compute(
        self,
        signals: Mapping[str, torch.Tensor],
        *,
        context: PolicyContext,
        num_classes: int,
    ) -> PolicyWeights:
        raise NotImplementedError

    def weights(
        self,
        signals: Mapping[str, torch.Tensor],
        *,
        context: PolicyContext,
        num_classes: int,
    ) -> PolicyWeights:
        """Compatibility alias; new callers should use :meth:`compute`."""
        return self.compute(signals, context=context, num_classes=num_classes)
