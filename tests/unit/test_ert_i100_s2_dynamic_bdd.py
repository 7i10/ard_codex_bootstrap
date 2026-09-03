from __future__ import annotations

import torch
from torch import nn

from ard.analysis.ert_stage_a_runtime import StageATreatment
from ard.data import IndexedBatch
from ard.engine import Trainer


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


def _trainer(mode: str) -> Trainer:
    student = _Linear()
    teacher = _Linear()
    with torch.no_grad():
        student.linear.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
        teacher.linear.weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 2.0], [-1.0, -1.0]]))
    for p in teacher.parameters():
        p.requires_grad_(False)
    trainer = Trainer.__new__(Trainer)
    trainer.model = student
    trainer.teacher = teacher
    trainer.device = torch.device("cpu")
    trainer.boundary_intervention = mode
    trainer.boundary_coefficient = 1.0
    trainer.boundary_epsilon = 1e-12
    trainer._boundary_epoch_stats = {
        "boundary_active_count": 0.0,
        "boundary_gate_positive_count": 0.0,
        "boundary_zero_rho_count": 0.0,
        "boundary_input_gradient_calls": 0.0,
    }
    trainer._teacher_adversarial_forward_calls = 0.0
    return trainer


def _inputs() -> tuple[IndexedBatch, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])
    labels = torch.tensor([0, 1])
    batch = IndexedBatch(images, labels, torch.tensor([10, 11]), torch.tensor([True, True]))
    student_adv = torch.tensor([[1.2, 0.6, -0.2], [0.3, 1.1, -0.4]], requires_grad=True)
    student_clean = torch.tensor([[1.0, 0.8, -0.2], [0.2, 1.0, -0.4]], requires_grad=True)
    teacher_adv = torch.tensor([[1.5, 0.7, -0.3], [0.4, 1.3, -0.4]])
    teacher_clean = torch.tensor([[1.2, 0.9, -0.3], [0.3, 1.1, -0.4]])
    adversarial = images + 0.1
    return batch, adversarial, student_adv, student_clean, teacher_clean, teacher_adv


def test_dynamic_pair_uses_strongest_non_true_logit() -> None:
    logits = torch.tensor([[0.1, 2.0, 1.5], [2.0, 0.3, 1.7]], requires_grad=True)
    margin, rival = Trainer._dynamic_pair_margins(logits, torch.tensor([0, 1]))
    assert rival.tolist() == [1, 0]
    assert torch.allclose(margin, torch.tensor([-1.9, -1.7]))
    assert not rival.requires_grad


def test_stage_treatment_accepts_three_boundary_modes() -> None:
    for mode in ("pair_margin", "detached_boundary_distance", "secant_boundary_distance"):
        t = StageATreatment(
            arm=mode,
            mask_key="s2_t1",
            kind="broad",
            boundary_intervention=mode,
            boundary_coefficient=0.25,
        )
        assert t.boundary_intervention == mode


def test_dpm_teacher_gate_and_gradient() -> None:
    trainer = _trainer("pair_margin")
    batch, adv, student_adv, student_clean, teacher_clean, teacher_adv = _inputs()
    values = trainer._boundary_terms(
        batch=batch,
        adversarial=adv,
        logits=student_adv,
        clean_student_logits=student_clean,
        teacher_clean_logits=teacher_clean,
        teacher_adversarial_logits=teacher_adv,
        treatment_risk=torch.ones(2),
    )
    assert values.shape == (2,)
    assert trainer._boundary_epoch_stats["boundary_active_count"] == 2.0
    values.sum().backward()
    assert student_adv.grad is not None
    assert all(p.grad is None for p in trainer.teacher.parameters())


def test_secant_zero_radius_is_skipped_without_input_gradients() -> None:
    trainer = _trainer("secant_boundary_distance")
    batch, adv, student_adv, student_clean, teacher_clean, teacher_adv = _inputs()
    values = trainer._boundary_terms(
        batch=batch,
        adversarial=batch.images,
        logits=student_adv,
        clean_student_logits=student_clean,
        teacher_clean_logits=teacher_clean,
        teacher_adversarial_logits=teacher_adv,
        treatment_risk=torch.ones(2),
    )
    assert torch.equal(values, torch.zeros_like(values))
    assert trainer._boundary_epoch_stats["boundary_zero_rho_count"] == 2.0
    assert trainer._boundary_epoch_stats["boundary_input_gradient_calls"] == 0.0


def _secant_loss(
    trainer: Trainer,
    batch: IndexedBatch,
    adversarial: torch.Tensor,
    *,
    detach_student_q: bool,
) -> torch.Tensor:
    """Independent formula oracle for the Student secant graph regression."""
    student_adv = trainer.model(adversarial)
    student_clean = trainer.model(batch.images)
    with torch.no_grad():
        teacher_adv = trainer.teacher(adversarial)
        teacher_clean = trainer.teacher(batch.images)
        teacher_margin, rival = Trainer._dynamic_pair_margins(teacher_adv, batch.labels)
        teacher_clean_margin, _ = Trainer._dynamic_pair_margins(teacher_clean, batch.labels, rival)
    student_margin, _ = Trainer._dynamic_pair_margins(student_adv, batch.labels, rival)
    student_clean_margin, _ = Trainer._dynamic_pair_margins(student_clean, batch.labels, rival)
    rho = (adversarial.detach() - batch.images.detach()).abs().flatten(1).amax(dim=1)
    q_student = (student_margin - student_clean_margin).abs() / (rho + trainer.boundary_epsilon)
    q_teacher = (teacher_margin - teacher_clean_margin).abs() / (rho + trainer.boundary_epsilon)
    if detach_student_q:
        q_student = q_student.detach()
    return (
        0.5
        * torch.relu(
            teacher_margin / (q_teacher + trainer.boundary_epsilon)
            - student_margin / (q_student + trainer.boundary_epsilon)
        ).square()
        * (teacher_margin > 0).to(dtype=student_margin.dtype)
        * (rho > trainer.boundary_epsilon).to(dtype=student_margin.dtype)
    )


def test_secant_student_denominator_retains_parameter_graph() -> None:
    """S-BDD must differ from the obsolete detached-qS approximation.

    The test deliberately uses real forwards through a parameterized Student,
    so matching values alone cannot hide an incorrect autograd graph.
    """
    trainer = _trainer("secant_boundary_distance")
    images = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])
    batch = IndexedBatch(images, torch.tensor([0, 1]), torch.tensor([10, 11]), torch.tensor([True, True]))
    adversarial = images + 0.1

    actual = trainer._boundary_terms(
        batch=batch,
        adversarial=adversarial,
        logits=trainer.model(adversarial),
        clean_student_logits=trainer.model(images),
        teacher_clean_logits=trainer.teacher(images),
        teacher_adversarial_logits=trainer.teacher(adversarial),
        treatment_risk=torch.ones(2),
    )
    actual_gradient = torch.autograd.grad(actual.sum(), trainer.model.linear.weight)[0]

    expected = _secant_loss(trainer, batch, adversarial, detach_student_q=False)
    expected_gradient = torch.autograd.grad(expected.sum(), trainer.model.linear.weight)[0]
    legacy = _secant_loss(trainer, batch, adversarial, detach_student_q=True)
    legacy_gradient = torch.autograd.grad(legacy.sum(), trainer.model.linear.weight)[0]

    assert torch.allclose(actual_gradient, expected_gradient, atol=1e-7, rtol=1e-6)
    assert not torch.allclose(actual_gradient, legacy_gradient, atol=1e-7, rtol=1e-6)
    assert trainer._boundary_epoch_stats["boundary_input_gradient_calls"] == 0.0
    assert all(parameter.grad is None for parameter in trainer.teacher.parameters())


def test_secant_calibration_uses_one_even_sample_median_convention() -> None:
    """The frozen ratio must hit .25 under the same estimator it reports."""
    from scripts.recalibrate_ert_i100_s2_secant_boundary_distance import _freeze_coefficient, _median

    base = torch.ones(4, dtype=torch.float64)
    secant = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    coefficient = _freeze_coefficient(base, secant)
    achieved = coefficient * secant / base
    assert torch.isclose(_median(achieved), torch.tensor(0.25, dtype=torch.float64), atol=1e-12, rtol=0.0)


def test_detached_bdd_uses_first_order_input_gradients_only() -> None:
    trainer = _trainer("detached_boundary_distance")
    batch, adv, student_adv, student_clean, teacher_clean, teacher_adv = _inputs()
    values = trainer._boundary_terms(
        batch=batch,
        adversarial=adv,
        logits=student_adv,
        clean_student_logits=student_clean,
        teacher_clean_logits=teacher_clean,
        teacher_adversarial_logits=teacher_adv,
        treatment_risk=torch.ones(2),
    )
    assert torch.isfinite(values).all()
    assert trainer._boundary_epoch_stats["boundary_input_gradient_calls"] == 4.0
    values.sum().backward()
    assert all(p.grad is None for p in trainer.teacher.parameters())
