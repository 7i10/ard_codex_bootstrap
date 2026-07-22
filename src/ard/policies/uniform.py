"""No-op policy for baseline methods without sample-wise weighting."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import PolicyContext, PolicyWeights, WeightPolicy


class UniformPolicy(WeightPolicy):
    def compute(
        self,
        signals: Mapping[str, torch.Tensor],
        *,
        context: PolicyContext,
        num_classes: int,
    ) -> PolicyWeights:
        if signals:
            raise ValueError("uniform policy does not consume signals")
        weight = torch.ones(context.valid_mask.shape, device=context.valid_mask.device, dtype=torch.float32)
        return PolicyWeights(hard_weight=weight, kd_weight=weight)
