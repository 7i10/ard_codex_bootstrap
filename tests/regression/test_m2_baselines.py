from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError
from torch import nn

from ard.attacks import LinfPGD
from ard.config.schema import AttackConfig, ExperimentConfig
from ard.data import IndexedBatch
from ard.engine import Trainer
from ard.engine.distributed import reduce_min
from ard.objectives import PGDATObjective, RSLADObjective, TRADESObjective
from ard.objectives.base import ObjectiveTerms
from ard.policies import EntropyOnlyPolicy, PolicyContext, RSLADBaselinePolicy
from ard.signals import shannon_entropy

pytestmark = pytest.mark.t2


def _kl_target_to_student(student: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    return (
        F.kl_div(
            F.log_softmax(student / temperature, dim=1),
            F.softmax(target / temperature, dim=1),
            reduction="none",
        ).sum(dim=1)
        * temperature**2
    )


def test_fixed_batch_pgd_at_trades_rslad_formula_direction_temperature_and_t2() -> None:
    labels = torch.tensor([0, 2])
    student_adv = torch.tensor([[1.2, -0.5, 0.1], [-0.1, 0.3, 1.8]], requires_grad=True)
    student_clean = torch.tensor([[0.4, 0.2, -0.6], [0.5, -0.7, 0.8]], requires_grad=True)
    teacher_clean = torch.tensor([[2.0, -0.3, 0.4], [-0.2, 1.4, 0.7]], requires_grad=True)
    temperature = 2.0

    pgd_terms = PGDATObjective()(student_logits=student_adv, labels=labels)
    assert torch.allclose(pgd_terms.hard, F.cross_entropy(student_adv, labels, reduction="none"))
    assert torch.isfinite(pgd_terms.total).all()

    trades = TRADESObjective(beta=3.0, temperature=temperature, temperature_squared=True)(
        student_logits=student_adv,
        clean_student_logits=student_clean,
        labels=labels,
    )
    expected_trades_kl = 3.0 * _kl_target_to_student(student_adv, student_clean, temperature)
    assert torch.allclose(trades.hard, F.cross_entropy(student_clean, labels, reduction="none"), atol=1e-7, rtol=0)
    assert torch.equal(trades.kd, expected_trades_kl)

    rslad = RSLADObjective(temperature=temperature, temperature_squared=True)(
        student_logits=student_adv,
        clean_student_logits=student_clean,
        teacher_logits=teacher_clean,
        labels=labels,
    )
    expected_adv = _kl_target_to_student(student_adv, teacher_clean, temperature)
    expected_clean = _kl_target_to_student(student_clean, teacher_clean, temperature)
    expected_kd = (5.0 / 6.0) * expected_adv + (1.0 / 6.0) * expected_clean
    assert torch.equal(rslad.kd, expected_kd)
    assert torch.equal(rslad.hard, F.cross_entropy(student_adv, labels, reduction="none"))
    baseline = RSLADBaselinePolicy().compute(
        {},
        context=PolicyContext(valid_mask=torch.ones(2, dtype=torch.bool), global_min=reduce_min),
        num_classes=3,
    )
    assert torch.allclose(rslad.apply_policy(baseline).total, expected_kd)
    rslad.apply_policy(baseline).total.mean().backward()
    assert teacher_clean.grad is None
    assert student_adv.grad is not None and student_clean.grad is not None


def test_entropy_policy_is_shannon_unclipped_and_weights_complete_rslad_loss() -> None:
    teacher_adv = torch.tensor([[6.0, -3.0, -2.0], [0.0, 0.0, 0.0], [2.0, 1.0, -4.0]])
    entropy = shannon_entropy(teacher_adv)
    policy = EntropyOnlyPolicy()
    weights = policy.weights(
        {"teacher_entropy": entropy},
        context=PolicyContext(valid_mask=torch.ones(3, dtype=torch.bool), global_min=reduce_min),
        num_classes=3,
    )
    expected = 5.0 * (entropy - entropy.min())
    assert torch.equal(weights.hard_weight, torch.zeros_like(expected))
    assert torch.allclose(weights.kd_weight, expected, atol=1e-7, rtol=0)
    assert float(weights.kd_weight.min()) == pytest.approx(0.0)
    assert float(weights.kd_weight.max()) <= 5.0 * math.log(3) + 1e-6

    terms = RSLADObjective()(
        student_logits=torch.tensor([[0.3, 0.1, -0.1], [0.2, -0.4, 0.5], [0.0, 0.4, -0.2]], requires_grad=True),
        clean_student_logits=torch.tensor([[0.4, 0.0, -0.3], [0.1, -0.2, 0.7], [0.2, 0.3, -0.5]], requires_grad=True),
        teacher_logits=torch.tensor([[1.0, -0.5, 0.2], [0.3, 0.2, 0.1], [-0.1, 0.6, 0.0]]),
        labels=torch.tensor([0, 2, 1]),
    )
    weighted = terms.apply_policy(weights)
    assert torch.allclose(weighted.total, expected * terms.kd, atol=1e-7, rtol=0)
    assert torch.isfinite(weighted.total).all()


def test_entropy_policy_uses_global_valid_min_with_padded_two_rank_shards() -> None:
    seen_candidates: list[float] = []

    def global_point_one(local_candidate: torch.Tensor) -> torch.Tensor:
        seen_candidates.append(float(local_candidate))
        return local_candidate.new_tensor(0.1)

    policy = EntropyOnlyPolicy()
    rank0 = policy.weights(
        {"teacher_entropy": torch.tensor([0.1], requires_grad=True)},
        context=PolicyContext(valid_mask=torch.tensor([True]), global_min=global_point_one),
        num_classes=3,
    )
    rank1 = policy.weights(
        {"teacher_entropy": torch.tensor([0.4, 0.01], requires_grad=True)},
        context=PolicyContext(valid_mask=torch.tensor([True, False]), global_min=global_point_one),
        num_classes=3,
    )
    assert seen_candidates == pytest.approx([0.1, 0.4])
    assert torch.equal(rank0.kd_weight, torch.tensor([0.0]))
    assert torch.allclose(rank1.kd_weight, torch.tensor([1.5, 0.0]), atol=1e-7, rtol=0)
    assert not rank0.kd_weight.requires_grad and not rank1.kd_weight.requires_grad

    theta0 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    theta1 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    raw0 = ObjectiveTerms(
        torch.zeros(1, dtype=torch.float64), theta0.square().reshape(1), torch.zeros(1, dtype=torch.float64)
    )
    raw1 = ObjectiveTerms(
        torch.zeros(2, dtype=torch.float64),
        torch.stack((theta1.square(), (theta1 + 10).square())),
        torch.zeros(2, dtype=torch.float64),
    )
    weighted0 = raw0.apply_policy(rank0)
    weighted1 = raw1.apply_policy(rank1)
    # W/N = 2/2 on each rank; DDP then averages the replica gradients.
    local0 = (weighted0.total * torch.tensor([1.0], dtype=torch.float64)).sum()
    local1 = (weighted1.total * torch.tensor([1.0, 0.0], dtype=torch.float64)).sum()
    grad0 = torch.autograd.grad(local0, theta0)[0]
    grad1 = torch.autograd.grad(local1, theta1)[0]
    reduced_loss = (local0.detach() + local1.detach()) / 2.0
    ddp_gradient = (grad0 + grad1) / 2.0
    assert float(reduced_loss) == pytest.approx(0.03, abs=1e-15)
    assert float(ddp_gradient) == pytest.approx(0.3, abs=1e-15)

    source = torch.tensor(0.4)
    reduced = reduce_min(source)
    reduced.add_(1.0)
    assert torch.equal(source, torch.tensor(0.4))


@pytest.mark.parametrize(
    ("method", "temperature_squared", "expected_loss", "expected_gradient"),
    [
        ("trades", False, 1.0978097948679981, 0.3798930973442602),
        ("trades", True, 1.1315866808684356, 0.12806653131986745),
        ("rslad", False, 0.015829451434101395, -0.05364875992800346),
        ("rslad", True, 0.06331780573640558, -0.21459503971201385),
    ],
)
def test_frozen_scalar_objective_loss_and_gradient(
    method: str,
    temperature_squared: bool,
    expected_loss: float,
    expected_gradient: float,
) -> None:
    theta = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    student_adv = torch.stack((theta, -0.3 * theta)).reshape(1, 2)
    student_clean = torch.stack((0.5 * theta + 0.4, -0.2 * theta - 0.1)).reshape(1, 2)
    labels = torch.tensor([1])
    if method == "trades":
        terms = TRADESObjective(beta=3.0, temperature=2.0, temperature_squared=temperature_squared)(
            student_logits=student_adv,
            clean_student_logits=student_clean,
            labels=labels,
        )
        loss = terms.total.sum()
    else:
        terms = RSLADObjective(temperature=2.0, temperature_squared=temperature_squared)(
            student_logits=student_adv,
            clean_student_logits=student_clean,
            teacher_logits=torch.tensor([[0.7, -0.4]], dtype=torch.float64),
            labels=labels,
        )
        baseline = RSLADBaselinePolicy().compute(
            {},
            context=PolicyContext(valid_mask=torch.ones(1, dtype=torch.bool), global_min=reduce_min),
            num_classes=2,
        )
        loss = terms.apply_policy(baseline).total.sum()
    gradient = torch.autograd.grad(loss, theta)[0]
    assert float(loss.detach()) == pytest.approx(expected_loss, abs=1e-14)
    assert float(gradient.detach()) == pytest.approx(expected_gradient, abs=1e-14)


def test_rslad_trainer_freezes_teacher_and_changes_student_once(tmp_path: pytest.TempPathFactory) -> None:
    torch.manual_seed(17)
    student = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 3))
    teacher = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 3))
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)
    attack_config = AttackConfig(
        loss="kl",
        kl_target="teacher_clean",
        epsilon="1/255",
        step_size="1/255",
        steps=1,
        random_start=False,
    )
    trainer = Trainer(
        model=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        attack=LinfPGD(attack_config),
        selection_attack=LinfPGD(AttackConfig(epsilon="1/255", step_size="1/255", steps=1, random_start=False)),
        objective=RSLADObjective(),
        policy=EntropyOnlyPolicy(),
        device=torch.device("cpu"),
        output_dir=tmp_path / "run",
        config_hash="m2-fixed-batch",
        seed=17,
    )
    before = next(student.parameters()).detach().clone()
    batch = IndexedBatch(
        images=torch.rand(3, 3, 4, 4),
        labels=torch.tensor([0, 1, 2]),
        sample_ids=torch.tensor([10, 11, 12]),
    )
    metrics = trainer.train_epoch([batch])  # type: ignore[arg-type]
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters())
    assert not torch.equal(before, next(student.parameters()).detach())
    assert all(math.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize(
    ("name", "attack"),
    [
        ("pgd_at", {"loss": "ce"}),
        ("trades", {"loss": "kl", "kl_target": "student_clean"}),
        ("rslad", {"loss": "kl", "kl_target": "teacher_clean"}),
        ("rslad_entropy", {"loss": "kl", "kl_target": "teacher_clean"}),
    ],
)
def test_each_m2_method_config_is_selectable(name: str, attack: dict[str, str]) -> None:
    data: dict[str, object] = {
        "dataset": {"name": "synthetic_cifar", "num_classes": 3, "num_samples": 8, "image_size": 4},
        "student": {"architecture": "fixture_cnn", "num_classes": 3},
        "method": {"name": name, "attack": {**attack, "epsilon": "1/255", "step_size": "1/255", "steps": 1}},
        "output_dir": "outputs/m2-config-test",
    }
    if name.startswith("rslad"):
        data["teacher"] = {"source": "fixture", "architecture": "fixture_cnn", "num_classes": 3}
    config = ExperimentConfig.model_validate(data)
    assert config.method.name == name
    assert config.method.selection_attack is not None and config.method.selection_attack.loss == "ce"


def test_rslad_entropy_rejects_configurable_scale_even_when_five() -> None:
    with pytest.raises(ValidationError, match="entropy_scale"):
        ExperimentConfig.model_validate(
            {
                "dataset": {"name": "synthetic_cifar", "num_classes": 3, "num_samples": 8, "image_size": 4},
                "student": {"architecture": "fixture_cnn", "num_classes": 3},
                "teacher": {"source": "fixture", "architecture": "fixture_cnn", "num_classes": 3},
                "method": {
                    "name": "rslad_entropy",
                    "entropy_scale": 5,
                    "attack": {
                        "loss": "kl",
                        "kl_target": "teacher_clean",
                        "epsilon": "1/255",
                        "step_size": "1/255",
                        "steps": 1,
                    },
                },
            }
        )
