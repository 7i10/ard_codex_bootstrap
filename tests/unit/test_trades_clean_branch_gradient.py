"""TRADES must send gradients through its clean branch.

TRADES minimises ``CE(f(x), y) + beta * KL(f(x') || f(x))`` where both branches
are the same network.  The KL term exists to pull the clean and adversarial
predictions towards each other, which it can only do if gradients reach both.

This engine routes TRADES through a helper shared with distillation, and that
helper detaches its target by default because the usual target is a frozen
teacher.  Applied to TRADES the default is wrong, and it cost four points of
AutoAttack accuracy without failing anything:
docs/debugging/0028-trades-clean-target-detached.md.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ard.objectives.kl import target_to_student_kl
from ard.objectives.trades import TRADESObjective


def _clean_branch_gradient(objective: TRADESObjective) -> torch.Tensor:
    torch.manual_seed(0)
    layer = torch.nn.Linear(6, 10)
    clean_features, adversarial_features = torch.randn(8, 6), torch.randn(8, 6)
    labels = torch.randint(0, 10, (8,))
    clean_logits = layer(clean_features)
    adversarial_logits = layer(adversarial_features)
    # Only the KL term may contribute, so the cross-entropy term is excluded:
    # it reaches the clean branch either way and would mask the defect.
    terms = objective(
        student_logits=adversarial_logits, labels=labels, clean_student_logits=clean_logits
    )
    (grad,) = torch.autograd.grad(terms.kd.mean(), clean_logits, retain_graph=True)
    return grad


def test_trades_kl_term_reaches_the_clean_branch() -> None:
    grad = _clean_branch_gradient(TRADESObjective(beta=6.0))
    assert torch.linalg.vector_norm(grad) > 0, (
        "the TRADES KL term produced no gradient on the clean logits; the clean target is "
        "detached, so the clean branch is trained by cross-entropy alone and this is not TRADES"
    )


def test_detaching_the_target_removes_that_gradient() -> None:
    """The guard above only means something if the defective form would fail it."""
    torch.manual_seed(0)
    layer = torch.nn.Linear(6, 10)
    clean_logits = layer(torch.randn(8, 6))
    adversarial_logits = layer(torch.randn(8, 6))
    detached = target_to_student_kl(
        student_logits=adversarial_logits,
        target_logits=clean_logits,
        temperature=1.0,
        temperature_squared=True,
        detach_target=True,
    )
    (grad,) = torch.autograd.grad(detached.mean(), clean_logits, allow_unused=True, materialize_grads=True)
    assert torch.linalg.vector_norm(grad) == 0


def test_a_frozen_teacher_target_is_unaffected_by_the_flag() -> None:
    """Distillation call sites keep the default, and it changes nothing for them."""
    torch.manual_seed(0)
    student = torch.randn(8, 10, requires_grad=True)
    teacher = torch.randn(8, 10)  # frozen: no grad either way
    kwargs = {"student_logits": student, "target_logits": teacher, "temperature": 1.0, "temperature_squared": True}
    assert torch.allclose(
        target_to_student_kl(**kwargs, detach_target=True),
        target_to_student_kl(**kwargs, detach_target=False),
    )
