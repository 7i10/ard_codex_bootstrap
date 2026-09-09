"""ADR: EMA-of-student rectified-label self-distillation.

Wu, Wang & Chen, "Annealing Self-Distillation Rectification Improves
Adversarial Training", ICLR 2024, arXiv:2305.12118.  Official code:
github.com/yuyuwu5/ADR.

ADR is not a KD loss term added on top of hard-label CE, unlike every other
objective in this package.  It is a label replacement: an EMA copy of the
student's own weights produces a temperature-softened clean prediction, which
is blended with the one-hot label into a single rectified soft target
``P(x)``.  ``P(x)`` is computed once per batch, before the attack runs, and
substitutes for the true label in *both* the PGD attack (see
``ard.attacks.pgd``'s ``kl_target="rectified"`` path, which consumes the same
tensor via ``AttackRequest.target_probabilities`` so the attack and the loss
below never disagree about the target) and the training loss.

Both this module and the attack reuse the existing KL machinery
(``probabilities_to_student_kl``, ``target_to_student_kl``) rather than a
dedicated cross-entropy-against-a-probability-vector primitive.  The target
is fully detached in every case, and cross-entropy against a fixed target
differs from KL-divergence against it only by the target's own constant
entropy term -- the gradient with respect to the student is identical.  This
project already trusts those primitives' numerics (they carry their own
tests), so no new loss math is introduced here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import DistillationObjective, ObjectiveTerms
from .kl import probabilities_to_student_kl, target_to_student_kl


def rectify_label(
    *,
    ema_clean_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    lambda_floor: float,
) -> torch.Tensor:
    """Paper Eq. 3-5: the per-sample rectified target ``P(x)``.

    ``ema_clean_logits`` must be the EMA-of-student's forward on the *clean*
    image, already detached -- this function does not detach anything
    itself, since it has no way to know whether a caller's tensor is meant
    to carry gradient; the EMA module's own ``requires_grad_(False)`` plus a
    ``torch.no_grad()`` forward is what actually guarantees detachment (see
    ``Trainer._ema_clean_logits``).  Rejecting a tensor that still requires
    grad here is a cheap, explicit second check, not the real guarantee.

    Eq. 3: ``P_t(x) = softmax(f_theta_t(x) / temperature)``.
    Eq. 4: per-sample ``lambda_i = clip[0,1](lambda_floor - (P_t(x)[argmax] -
    P_t(x)[y]))`` -- no discount when the EMA's top class already matches the
    true label; otherwise ``lambda_floor`` is reduced by exactly the size of
    the EMA's mistake.
    Eq. 5: ``P(x) = lambda_i * P_t(x) + (1 - lambda_i) * one_hot(y)``.
    """
    if ema_clean_logits.ndim != 2:
        raise ValueError("ema_clean_logits must be a [batch, class] tensor")
    if labels.shape != ema_clean_logits.shape[:1]:
        raise ValueError("labels must be a [batch] tensor matching ema_clean_logits")
    if ema_clean_logits.requires_grad:
        raise ValueError(
            "rectify_label requires a detached EMA forward; the rectified target must never be differentiated through"
        )
    if not 0.0 <= lambda_floor <= 1.0:
        raise ValueError("lambda_floor must lie in [0, 1]")
    teacher_probabilities = F.softmax(ema_clean_logits.float() / temperature, dim=1)
    top_probability = teacher_probabilities.amax(dim=1)
    true_label_probability = teacher_probabilities.gather(1, labels[:, None]).squeeze(1)
    per_sample_lambda = (lambda_floor - (top_probability - true_label_probability)).clamp(0.0, 1.0)
    one_hot = F.one_hot(labels, num_classes=ema_clean_logits.shape[1]).to(teacher_probabilities.dtype)
    rectified = per_sample_lambda[:, None] * teacher_probabilities + (1.0 - per_sample_lambda[:, None]) * one_hot
    return rectified


class ADRObjective(DistillationObjective):
    """PGD-AT base: ``ell(f_theta_s(x'), P(x))``. No separate KD/hard split;
    ADR does not add a term, it replaces the label the existing term uses."""

    requires_rectified_target_probabilities = True

    def __call__(
        self,
        *,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
        clean_student_logits: torch.Tensor | None = None,
        adversarial_target_probabilities: torch.Tensor | None = None,
        rectified_target_probabilities: torch.Tensor | None = None,
    ) -> ObjectiveTerms:
        del teacher_logits, clean_student_logits, adversarial_target_probabilities
        if rectified_target_probabilities is None:
            raise ValueError("adr requires a rectified target")
        hard = probabilities_to_student_kl(
            student_logits=student_logits,
            target_probabilities=rectified_target_probabilities,
            temperature=1.0,
            temperature_squared=False,
        )
        zeros = torch.zeros_like(hard)
        return ObjectiveTerms(hard=hard, kd=zeros, regularization=zeros)


class ADRTRADESObjective(DistillationObjective):
    """TRADES base: the rectified label replaces the label in the natural
    (clean) CE term only.  The KL robustness term -- adversarial pulled
    towards clean, gradients through both -- is untouched, identical to
    ``TRADESObjective``."""

    requires_clean_student_logits = True
    requires_rectified_target_probabilities = True

    def __init__(self, *, beta: float, temperature: float = 1.0, temperature_squared: bool = True) -> None:
        self.beta = beta
        self.temperature = temperature
        self.temperature_squared = temperature_squared

    def __call__(
        self,
        *,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
        clean_student_logits: torch.Tensor | None = None,
        adversarial_target_probabilities: torch.Tensor | None = None,
        rectified_target_probabilities: torch.Tensor | None = None,
    ) -> ObjectiveTerms:
        del teacher_logits, adversarial_target_probabilities
        if clean_student_logits is None:
            raise ValueError("adr_trades requires student logits on the clean input")
        if rectified_target_probabilities is None:
            raise ValueError("adr_trades requires a rectified target")
        hard = probabilities_to_student_kl(
            student_logits=clean_student_logits,
            target_probabilities=rectified_target_probabilities,
            temperature=1.0,
            temperature_squared=False,
        )
        kd = self.beta * target_to_student_kl(
            student_logits=student_logits,
            target_logits=clean_student_logits,
            temperature=self.temperature,
            temperature_squared=self.temperature_squared,
            detach_target=False,
        )
        return ObjectiveTerms(hard=hard, kd=kd, regularization=torch.zeros_like(hard))
