"""Student-only and student/teacher joint risk mappings."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from .base import PolicyContext, PolicyWeights, WeightPolicy


def student_risk_from_margin(margin_ema: torch.Tensor) -> torch.Tensor:
    """Map a probability margin in ``[-1, 1]`` to risk in ``[0, 1]``."""
    if margin_ema.ndim != 1 or not torch.isfinite(margin_ema).all():
        raise ValueError("student margin EMA must be a finite vector")
    return ((1.0 - margin_ema.detach()) / 2.0).clamp(0.0, 1.0)


def teacher_risk_from_entropy(entropy: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    if entropy.ndim != 1 or num_classes < 2 or not torch.isfinite(entropy).all():
        raise ValueError("teacher entropy must be a finite vector and num_classes at least two")
    return (1.0 - entropy.detach() / math.log(num_classes)).clamp(0.0, 1.0)


class _RiskPolicy(WeightPolicy):
    signal_name: str

    def compute(
        self,
        signals: Mapping[str, torch.Tensor],
        *,
        context: PolicyContext,
        num_classes: int,
    ) -> PolicyWeights:
        if set(signals) != self.required_signals:
            raise ValueError(f"{type(self).__name__} requires exactly {sorted(self.required_signals)}")
        risk = signals[self.signal_name].detach()
        if risk.ndim != 1 or risk.shape != context.valid_mask.shape:
            raise ValueError("risk vector must match the objective batch")
        if bool((~torch.isfinite(risk) & context.valid_mask).any()):
            raise FloatingPointError("policy received non-finite valid risk")
        risk = risk.clamp(0.0, 1.0)
        valid = context.valid_mask.to(device=risk.device)
        risk = torch.where(valid, risk, torch.zeros_like(risk)).detach()
        return PolicyWeights(hard_weight=risk, kd_weight=(1.0 - risk) * valid.to(risk.dtype), joint_risk=risk)


class StudentRiskPolicy(_RiskPolicy):
    """Blend complete RSLAD KD with CE according to robust student risk."""

    required_signals = frozenset({"student_risk"})
    signal_name = "student_risk"


class JointRiskPolicy(_RiskPolicy):
    """Use student risk times teacher overconfidence as the hard-label risk."""

    required_signals = frozenset({"joint_risk"})
    signal_name = "joint_risk"
