"""Student robust-margin measurement in adversarial-logit space."""

from __future__ import annotations

import torch

from .base import SampleSignal, SignalBatch


class RobustMarginSignal(SampleSignal):
    """Compute ``p(y | x_adv) - max_{c != y} p(c | x_adv)`` in detached FP32."""

    def compute(
        self,
        *,
        student_adv_logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> SignalBatch:
        if student_adv_logits.ndim != 2 or student_adv_logits.shape[1] < 2:
            raise ValueError("robust margin requires logits with shape [batch, class>=2]")
        if labels.ndim != 1 or labels.shape[0] != student_adv_logits.shape[0]:
            raise ValueError("robust margin labels must match the logits batch")
        if valid_mask.shape != labels.shape or valid_mask.dtype != torch.bool:
            raise ValueError("robust margin valid_mask must match labels and be bool")
        if bool(((labels < 0) | (labels >= student_adv_logits.shape[1])).any()):
            raise ValueError("robust margin labels are outside the class range")
        logits = student_adv_logits.detach().float()
        probabilities = torch.softmax(logits, dim=1)
        true_probability = probabilities.gather(1, labels.reshape(-1, 1)).reshape(-1)
        other_probabilities = probabilities.clone()
        other_probabilities.scatter_(1, labels.reshape(-1, 1), float("-inf"))
        margin = (true_probability - other_probabilities.max(dim=1).values).detach()
        return SignalBatch(values=margin, valid_mask=valid_mask.detach())
