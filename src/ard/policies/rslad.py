"""RSLAD baseline mapping with no adversarial hard-label fallback."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import PolicyContext, PolicyWeights, WeightPolicy


class RSLADBaselinePolicy(WeightPolicy):
    """Preserve baseline RSLAD as complete KD only: hard=0, KD=1."""

    def compute(
        self,
        signals: Mapping[str, torch.Tensor],
        *,
        context: PolicyContext,
        num_classes: int,
    ) -> PolicyWeights:
        if signals:
            raise ValueError("RSLAD baseline policy does not consume signals")
        kd = context.valid_mask.to(dtype=torch.float32)
        return PolicyWeights(hard_weight=torch.zeros_like(kd), kd_weight=kd)
