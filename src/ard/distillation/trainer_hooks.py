"""Trainer-side distillation target hooks (plan 0103 Phase 2, batch D).

Kept out of ``ard.engine.trainer`` so the Trainer only gains two call sites:

* the teacher-clean target, which a :class:`SoftLabelBankTeacher` answers from
  the bank (crop keys checked) instead of a teacher forward;
* the ``rslad_advt`` adversarial target ``softmax(T(x')/tau)``, built from the
  Trainer's one cached, counted teacher forward on the training adversarial
  example.  When the clean target comes from a top-K bank, the same
  storage-and-reconstruction truncation is applied to ``T(x')`` so the two RSLAD
  terms are on the same footing.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .soft_label_bank import SoftLabelBankTeacher, kl_rows, truncate_like_bank


def bank_clean_logits(teacher: Any, batch: Any, *, epoch: int) -> torch.Tensor | None:
    """Bank pseudo-logits for a bank teacher; ``None`` means "run the teacher as usual"."""
    if isinstance(teacher, SoftLabelBankTeacher):
        return teacher.clean_logits(batch, epoch=epoch).float()
    return None


class DistillationTargetHooks:
    METRIC_PREFIX = "advt_"

    def __init__(self, *, adversarial_teacher_target: bool, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("distillation temperature must be positive")
        self.adversarial_teacher_target = adversarial_teacher_target
        self.temperature = float(temperature)
        self._totals: torch.Tensor | None = None

    def reset_epoch(self, device: torch.device) -> None:
        # Teacher-adversarial correct, sum of KL(T(x') || T(x)), valid examples.
        self._totals = torch.zeros(3, dtype=torch.float64, device=device)

    def adversarial_target(
        self,
        *,
        teacher: Any,
        teacher_adversarial_logits: torch.Tensor,
        teacher_clean_logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            target = F.softmax(teacher_adversarial_logits.detach().float() / self.temperature, dim=1)
            if isinstance(teacher, SoftLabelBankTeacher):
                # Config validation fixes temperature 1 in bank mode, so this is
                # the same distribution the bank would have stored for x'.
                target = truncate_like_bank(target, teacher.top_k)
            clean = F.softmax(teacher_clean_logits.detach().float() / self.temperature, dim=1)
            weights = mask.detach().to(torch.float64)
            if self._totals is not None:
                self._totals += torch.stack(
                    [
                        ((target.argmax(1) == labels).to(torch.float64) * weights).sum(),
                        (kl_rows(target, clean).to(torch.float64) * weights).sum(),
                        weights.sum(),
                    ]
                )
            if not bool(torch.isfinite(target).all()):
                raise FloatingPointError("rslad_advt adversarial teacher target is non-finite")
        return target.detach()

    def epoch_metrics(self, reduce_sums: Any) -> dict[str, float]:
        if not self.adversarial_teacher_target or self._totals is None:
            return {}
        totals = reduce_sums(self._totals)
        count = max(float(totals[2].item()), 1.0)
        return {
            f"{self.METRIC_PREFIX}teacher_adversarial_accuracy": float(totals[0].item()) / count,
            f"{self.METRIC_PREFIX}teacher_adversarial_clean_kl": float(totals[1].item()) / count,
        }
