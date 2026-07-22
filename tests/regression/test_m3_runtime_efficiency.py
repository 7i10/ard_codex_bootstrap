from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import SGD

from ard.attacks import AttackGenerator, AttackRequest, AttackResult, LinfPGD
from ard.config.schema import AttackConfig
from ard.data import IndexedBatch
from ard.engine.trainer import Trainer
from ard.objectives import DistillationObjective, PGDATObjective, RSLADObjective
from ard.policies import PolicyContext, RSLADBaselinePolicy

pytestmark = pytest.mark.t2


class CountingTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 3))
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.network(inputs)


class ModeRecordingTeacher(CountingTeacher):
    def __init__(self) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(3)
        self.modes: list[bool] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.modes.append(self.training)
        return self.batch_norm(super().forward(inputs))


class TargetRecordingAttack(AttackGenerator):
    def __init__(self) -> None:
        self.request: AttackRequest | None = None

    @property
    def requires_teacher_clean_target(self) -> bool:
        return True

    def generate(self, request: AttackRequest) -> AttackResult:
        self.request = request
        return AttackResult(
            adversarial=request.inputs.detach().float(),
            initial_delta=torch.zeros_like(request.inputs),
            step_losses=(),
            max_abs_delta=0.0,
        )


class TargetRecordingRSLAD(RSLADObjective):
    def __init__(self) -> None:
        super().__init__()
        self.teacher_logits: torch.Tensor | None = None

    def __call__(self, **kwargs: torch.Tensor | None) -> object:  # type: ignore[override]
        self.teacher_logits = kwargs["teacher_logits"]
        return super().__call__(**kwargs)  # type: ignore[arg-type]


def _student() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 3))


def _batch() -> IndexedBatch:
    return IndexedBatch(
        images=torch.tensor(
            [
                [[[0.1] * 4] * 4] * 3,
                [[[0.3] * 4] * 4] * 3,
                [[[0.6] * 4] * 4] * 3,
            ],
            dtype=torch.float32,
        ),
        labels=torch.tensor([0, 1, 2]),
        sample_ids=torch.tensor([4, 7, 9]),
    )


def _rslad_attack() -> AttackConfig:
    return AttackConfig(
        loss="kl",
        kl_target="teacher_clean",
        epsilon="1/255",
        step_size="1/255",
        steps=2,
        random_start=True,
    )


def _trainer(student: nn.Module, teacher: nn.Module, output: Path, *, objective: DistillationObjective) -> Trainer:
    attack = LinfPGD(
        _rslad_attack()
        if isinstance(objective, RSLADObjective)
        else AttackConfig(epsilon="1/255", step_size="1/255", steps=2, random_start=True)
    )
    return Trainer(
        model=student,
        teacher=teacher,
        optimizer=SGD(student.parameters(), lr=0.05),
        scheduler=None,
        scaler=None,
        attack=attack,
        selection_attack=LinfPGD(AttackConfig(epsilon="1/255", step_size="1/255", steps=1, random_start=False)),
        objective=objective,
        policy=RSLADBaselinePolicy() if isinstance(objective, RSLADObjective) else None,
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="m3-test",
        seed=17,
    )


def test_rslad_reuses_one_clean_teacher_target_with_legacy_two_forward_parity(tmp_path: Path) -> None:
    torch.manual_seed(81)
    original_student = _student()
    original_teacher = CountingTeacher().eval()
    for parameter in original_teacher.parameters():
        parameter.requires_grad_(False)
    optimized_student = copy.deepcopy(original_student)
    optimized_teacher = copy.deepcopy(original_teacher)
    trainer = _trainer(optimized_student, optimized_teacher, tmp_path / "optimized", objective=RSLADObjective())
    batch = _batch()

    metrics = trainer.train_epoch([batch])  # type: ignore[arg-type]
    assert optimized_teacher.calls == 1

    legacy_student = copy.deepcopy(original_student)
    legacy_teacher = copy.deepcopy(original_teacher).eval()
    legacy_optimizer = SGD(legacy_student.parameters(), lr=0.05)
    attack = LinfPGD(_rslad_attack())
    legacy_attack = attack.generate(
        AttackRequest(
            inputs=batch.images,
            labels=batch.labels,
            student=legacy_student,
            teacher=legacy_teacher,
            generator=torch.Generator().manual_seed(17),
        )
    )
    adversarial_logits = legacy_student(legacy_attack.adversarial)
    clean_student_logits = legacy_student(batch.images)
    with torch.no_grad():
        legacy_teacher_logits = legacy_teacher(batch.images).detach().float()
    terms = RSLADObjective()(
        student_logits=adversarial_logits,
        clean_student_logits=clean_student_logits,
        teacher_logits=legacy_teacher_logits,
        labels=batch.labels,
    )
    policy = RSLADBaselinePolicy().compute(
        {},
        context=PolicyContext(valid_mask=torch.ones(3, dtype=torch.bool), global_min=lambda value: value),
        num_classes=3,
    )
    legacy_loss = terms.apply_policy(policy).total.mean()
    legacy_loss.backward()
    legacy_optimizer.step()

    assert legacy_teacher.calls == 2
    assert metrics["loss"] == pytest.approx(float(legacy_loss.detach()), abs=1e-7, rel=0)
    for name, parameter in legacy_student.state_dict().items():
        assert torch.allclose(parameter, optimized_student.state_dict()[name], atol=1e-7, rtol=0), name


def test_pgd_at_does_not_forward_an_unused_teacher(tmp_path: Path) -> None:
    torch.manual_seed(19)
    teacher = CountingTeacher().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    trainer = _trainer(_student(), teacher, tmp_path, objective=PGDATObjective())
    trainer.train_epoch([_batch()])  # type: ignore[arg-type]
    assert teacher.calls == 0


def test_configless_attack_receives_the_exact_outer_teacher_target(tmp_path: Path) -> None:
    torch.manual_seed(22)
    teacher = CountingTeacher().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    attack = TargetRecordingAttack()
    objective = TargetRecordingRSLAD()
    student = _student()
    trainer = Trainer(
        model=student,
        teacher=teacher,
        optimizer=SGD(student.parameters(), lr=0.05),
        scheduler=None,
        scaler=None,
        attack=attack,
        selection_attack=LinfPGD(AttackConfig(epsilon="1/255", step_size="1/255", steps=1, random_start=False)),
        objective=objective,
        policy=RSLADBaselinePolicy(),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        config_hash="m3-test",
        seed=17,
    )
    trainer.train_epoch([_batch()])  # type: ignore[arg-type]
    assert teacher.calls == 1
    assert attack.request is not None
    assert attack.request.target_logits is objective.teacher_logits


def test_teacher_clean_kl_is_eval_only_and_preserves_batch_norm_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="teacher_clean KL attacks require teacher_mode=eval"):
        AttackConfig(loss="kl", kl_target="teacher_clean", teacher_mode="train")
    teacher = ModeRecordingTeacher().train()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    trainer = _trainer(_student(), teacher, tmp_path, objective=RSLADObjective())
    trainer.train_epoch([_batch()])  # type: ignore[arg-type]
    assert teacher.modes == [False]
    assert teacher.training
