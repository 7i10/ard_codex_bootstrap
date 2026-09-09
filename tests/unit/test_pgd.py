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


def test_pgd_projection_clamp_mode_without_trace_collection_and_no_parameter_grads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(3)
    model = linear_model()
    model.train()
    inputs = torch.rand(4, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 0])
    config = AttackConfig(epsilon="8/255", step_size="2/255", steps=3, random_start=True, student_mode="eval")
    original_cpu = torch.Tensor.cpu
    cpu_calls = 0

    def count_cpu(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal cpu_calls
        cpu_calls += 1
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", count_cpu)
    result = LinfPGD(config).generate(
        AttackRequest(inputs=inputs, labels=labels, student=model, generator=torch.Generator().manual_seed(9))
    )
    assert model.training
    assert result.adversarial.dtype == torch.float32
    assert result.adversarial.min() >= 0 and result.adversarial.max() <= 1
    assert (result.adversarial - inputs).abs().max() <= 8 / 255 + 1e-7
    assert result.max_abs_delta <= 8 / 255 + 1e-7
    assert result.step_losses == ()
    # The final max-delta summary is one bounded scalar transfer.  Disabled
    # tracing must not add one transfer for every PGD step.
    assert cpu_calls == 1
    assert all(parameter.grad is None for parameter in model.parameters())


def test_pgd_trace_collection_is_explicit_and_exact() -> None:
    torch.manual_seed(3)
    model = linear_model()
    inputs = torch.rand(4, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 0])
    config = AttackConfig(
        epsilon="8/255", step_size="2/255", steps=3, random_start=True, student_mode="eval", trace_step_losses=True
    )
    result = LinfPGD(config).generate(
        AttackRequest(inputs=inputs, labels=labels, student=model, generator=torch.Generator().manual_seed(9))
    )
    assert len(result.step_losses) == config.steps
    assert all(isinstance(loss, float) and torch.isfinite(torch.tensor(loss)) for loss in result.step_losses)


def test_pgd_mixed_per_sample_budget_keeps_each_example_within_its_bound() -> None:
    model = linear_model()
    inputs = torch.rand(3, 3, 4, 4)
    labels = torch.tensor([0, 1, 2])
    config = AttackConfig(epsilon="8/255", step_size="2/255", steps=3, random_start=True, student_mode="eval")
    result = LinfPGD(config).generate(
        AttackRequest(
            inputs=inputs,
            labels=labels,
            student=model,
            generator=torch.Generator().manual_seed(4),
            epsilon_override=torch.tensor([8 / 255, 4 / 255, 2 / 255]),
            step_size_override=torch.tensor([2 / 255, 1 / 255, 0.5 / 255]),
        )
    )
    delta = (result.adversarial - inputs).abs().flatten(1).amax(dim=1)
    assert torch.all(delta <= torch.tensor([8 / 255, 4 / 255, 2 / 255]) + 1e-7)


def test_pgd_captured_prefix_is_from_the_same_random_start_trajectory() -> None:
    torch.manual_seed(31)
    model = linear_model()
    inputs, labels = torch.rand(4, 3, 4, 4), torch.tensor([0, 1, 2, 0])
    config = AttackConfig(epsilon="8/255", step_size="2/255", steps=10, random_start=True, student_mode="eval")
    prefix = LinfPGD(config).generate(
        AttackRequest(
            inputs=inputs,
            labels=labels,
            student=model,
            generator=torch.Generator().manual_seed(17),
            capture_step=5,
        )
    )
    five = LinfPGD(config.model_copy(update={"steps": 5})).generate(
        AttackRequest(inputs=inputs, labels=labels, student=model, generator=torch.Generator().manual_seed(17))
    )
    assert prefix.captured_adversarial is not None
    assert torch.equal(prefix.captured_adversarial, five.adversarial)
    assert (prefix.captured_adversarial - inputs).abs().max() <= 8 / 255 + 1e-7


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


def test_rectified_kl_pgd_uses_supplied_probabilities_directly_and_matches_hand_computed_gradient() -> None:
    """``kl_target='rectified'`` must ascend a plain soft cross-entropy against the
    exact probabilities supplied on the request -- no internal softmax, no teacher,
    no student_clean resolution -- and the target must stay fixed across all steps."""
    torch.manual_seed(11)
    student = linear_model()
    inputs = torch.rand(2, 3, 4, 4)
    labels = torch.tensor([0, 1])
    rectified = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    config = AttackConfig(
        loss="kl",
        kl_target="rectified",
        temperature=1.0,
        epsilon="2/255",
        step_size="2/255",
        steps=1,
        random_start=False,
    )
    result = LinfPGD(config).generate(
        AttackRequest(inputs=inputs, labels=labels, student=student, target_probabilities=rectified)
    )
    # Reconstruct the one-step update by hand: ascend the sign of the gradient
    # of a plain (temperature=1) cross-entropy against the fixed distribution.
    probe = inputs.clone().requires_grad_(True)
    logits = student(probe)
    manual_loss = -(rectified * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
    expected_gradient = torch.autograd.grad(manual_loss, probe)[0]
    expected = (inputs + (2.0 / 255.0) * expected_gradient.sign()).clamp(0, 1)
    assert torch.allclose(result.adversarial, expected, atol=1e-6)


def test_rectified_kl_pgd_rejects_target_logits_and_plain_kl_rejects_target_probabilities() -> None:
    student = linear_model()
    inputs = torch.rand(2, 3, 4, 4)
    labels = torch.tensor([0, 1])
    rectified = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    rectified_config = AttackConfig(loss="kl", kl_target="rectified", steps=1, random_start=False)
    with pytest.raises(ValueError, match="takes target_probabilities, not target_logits"):
        LinfPGD(rectified_config).generate(
            AttackRequest(
                inputs=inputs,
                labels=labels,
                student=student,
                target_logits=torch.zeros(2, 3),
                target_probabilities=rectified,
            )
        )
    with pytest.raises(ValueError, match="rectified KL PGD requires target_probabilities"):
        LinfPGD(rectified_config).generate(AttackRequest(inputs=inputs, labels=labels, student=student))

    student_clean_config = AttackConfig(loss="kl", kl_target="student_clean", steps=1, random_start=False)
    with pytest.raises(ValueError, match="only accepted when kl_target='rectified'"):
        LinfPGD(student_clean_config).generate(
            AttackRequest(inputs=inputs, labels=labels, student=student, target_probabilities=rectified)
        )


def test_pgd_at_objective_is_unreduced() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    terms = PGDATObjective()(student_logits=logits, labels=torch.tensor([0, 1]))
    assert terms.hard.shape == terms.kd.shape == terms.regularization.shape == (2,)
    assert torch.equal(terms.kd, torch.zeros(2))
    terms.total.mean().backward()
    assert logits.grad is not None
