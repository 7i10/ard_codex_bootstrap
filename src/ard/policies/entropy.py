"""SAAD-compatible entropy-only weight mapping."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from .base import PolicyContext, PolicyWeights, WeightPolicy


class EntropyOnlyPolicy(WeightPolicy):
    """Map Shannon entropy to the unnormalized SAAD entropy-only weight.

    This deliberately has no upper clipping, mean preservation, or hard-label
    fallback.  The coefficient is the method identity: exactly 5.
    """

    required_signals = frozenset({"teacher_entropy"})

    def compute(
        self,
        signals: Mapping[str, torch.Tensor],
        *,
        context: PolicyContext,
        num_classes: int,
    ) -> PolicyWeights:
        if set(signals) != self.required_signals:
            raise ValueError("entropy-only policy requires exactly the teacher_entropy signal")
        entropy = signals["teacher_entropy"].detach()
        if entropy.ndim != 1 or num_classes < 2:
            raise ValueError("entropy policy requires [batch] entropy and at least two classes")
        if entropy.shape != context.valid_mask.shape:
            raise ValueError("entropy policy signal batch size does not match objective batch size")
        valid_mask = context.valid_mask.to(device=entropy.device)
        if bool((~torch.isfinite(entropy) & valid_mask).any()):
            raise FloatingPointError("entropy policy received non-finite valid entropy")
        local_candidate = entropy.masked_fill(~valid_mask, float("inf")).min()
        global_min = context.global_min(local_candidate).detach()
        if global_min.ndim != 0 or not bool(torch.isfinite(global_min)):
            raise FloatingPointError("entropy policy has no finite valid entropy across ranks")
        weights = torch.where(valid_mask, 5.0 * (entropy - global_min), torch.zeros_like(entropy)).detach()
        if not torch.isfinite(weights).all():
            raise FloatingPointError("entropy policy produced non-finite weights")
        upper_bound = 5.0 * math.log(num_classes)
        if bool((weights < -1e-6).any()) or bool((weights > upper_bound + 1e-5).any()):
            raise FloatingPointError("entropy policy weight is outside its Shannon range")
        # RSLAD now exposes its complete KD total separately from the
        # adversarial hard-label fallback.  Entropy retains the M2 contract by
        # weighting that complete KD total exactly, with no fallback CE.
        return PolicyWeights(hard_weight=torch.zeros_like(weights), kd_weight=weights)
