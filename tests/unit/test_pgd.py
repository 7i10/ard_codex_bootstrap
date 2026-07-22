from __future__ import annotations

import pytest
import torch
from torch import nn

from ard.attacks import AttackRequest, LinfPGD, teacher_input_gradient
from ard.config.schema import AttackConfig
from ard.objectives import PGDATObjective

pytestmark = pytest.mark.t2


def linear_model(classes: int = 3) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, classes))


def test_pgd_projection_clamp_mode_diagnostics_and_no_parameter_grads() -> None:
    torch.manual_seed(3)
    model = linear_model()
    model.train()
    inputs = torch.rand(4, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 0])
    config = AttackConfig(epsilon="8/255", step_size="2/255", steps=3, random_start=True, student_mode="eval")
    result = LinfPGD(config).generate(
        AttackRequest(inputs=inputs, labels=labels, student=model, generator=torch.Generator().manual_seed(9))
    )
    assert model.training
    assert result.adversarial.dtype == torch.float32
    assert result.adversarial.min() >= 0 and result.adversarial.max() <= 1
    assert (result.adversarial - inputs).abs().max() <= 8 / 255 + 1e-7
    assert result.max_abs_delta <= 8 / 255 + 1e-7
    assert len(result.step_losses) == 3
    assert all(parameter.grad is None for parameter in model.parameters())


def test_kl_pgd_and_frozen_teacher_input_gradient_contract() -> None:
    student = linear_model()
    teacher = linear_model()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    inputs = torch.rand(2, 3, 4, 4)
    labels = torch.tensor([0, 1])
    config = AttackConfig(
        loss="kl",
        kl_target="teacher_clean",
        epsilon="1/255",
        step_size="1/255",
        steps=1,
        random_start=False,
    )
    result = LinfPGD(config).generate(AttackRequest(inputs=inputs, labels=labels, student=student, teacher=teacher))
    gradient = teacher_input_gradient(teacher, inputs, labels)
    assert result.adversarial.shape == inputs.shape
    assert teacher.training
    assert gradient.shape == inputs.shape and torch.isfinite(gradient).all()
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters())


def test_pgd_at_objective_is_unreduced() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    terms = PGDATObjective()(student_logits=logits, labels=torch.tensor([0, 1]))
    assert terms.hard.shape == terms.kd.shape == terms.regularization.shape == (2,)
    assert torch.equal(terms.kd, torch.zeros(2))
    terms.total.mean().backward()
    assert logits.grad is not None
