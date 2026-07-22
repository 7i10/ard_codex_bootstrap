from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import SGD

from ard.attacks import AttackGenerator, AttackRequest, AttackResult
from ard.data import IndexedBatch
from ard.engine import trainer as trainer_module
from ard.engine.trainer import Trainer, _reduce_epoch_observability
from ard.objectives import DistillationObjective, ObjectiveTerms
from ard.testing.impact import select

pytestmark = pytest.mark.t1


class CountingTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(3 * 2 * 2, 2))
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.network(inputs)


class CleanTargetAttack(AttackGenerator):
    @property
    def requires_teacher_clean_target(self) -> bool:
        return True

    def generate(self, request: AttackRequest) -> AttackResult:
        assert request.target_logits is not None and not request.target_logits.requires_grad
        return AttackResult(request.inputs.detach(), torch.zeros_like(request.inputs), (), 0.0)


class TeacherTargetObjective(DistillationObjective):
    requires_teacher_clean_logits = True

    def __call__(
        self,
        *,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
        clean_student_logits: torch.Tensor | None = None,
        adversarial_target_probabilities: torch.Tensor | None = None,
    ) -> ObjectiveTerms:
        del clean_student_logits, adversarial_target_probabilities
        assert teacher_logits is not None and not teacher_logits.requires_grad
        hard = torch.nn.functional.cross_entropy(student_logits, labels, reduction="none")
        zeros = torch.zeros_like(hard)
        return ObjectiveTerms(hard=hard, kd=zeros, regularization=zeros)


def _batch(*, valid: tuple[bool, bool]) -> IndexedBatch:
    return IndexedBatch(
        images=torch.rand(2, 3, 2, 2),
        labels=torch.tensor([0, 1]),
        sample_ids=torch.tensor([0, 1]),
        state_update_mask=torch.tensor(valid),
    )


def test_cpu_epoch_observability_counts_only_actual_valid_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    student = nn.Sequential(nn.Flatten(), nn.Linear(3 * 2 * 2, 2))
    teacher = CountingTeacher()
    trainer = Trainer(
        model=student,
        teacher=teacher,
        optimizer=SGD(student.parameters(), lr=0.01),
        scheduler=None,
        scaler=None,
        attack=CleanTargetAttack(),
        selection_attack=CleanTargetAttack(),
        objective=TeacherTargetObjective(),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        config_hash="observability-test",
        seed=7,
    )
    clock = iter((10.0, 12.0))
    monkeypatch.setattr(trainer_module.time, "perf_counter", lambda: next(clock))

    metrics = trainer.train_epoch([_batch(valid=(True, False)), _batch(valid=(True, True))])  # type: ignore[arg-type]

    assert teacher.calls == 2
    assert metrics["valid_examples"] == 3.0
    assert metrics["teacher_clean_forward_calls"] == 2.0
    assert metrics["seconds"] == 2.0
    assert metrics["images_per_second"] == 1.5
    assert metrics["cuda_peak_allocated_bytes"] == 0.0


def test_two_rank_observability_reduction_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    local_totals = torch.tensor([10.0, 2.0, 1.0, 3.0, 2.0], dtype=torch.float64)
    remote_totals = torch.tensor([20.0, 4.0, 3.0, 5.0, 4.0], dtype=torch.float64)

    monkeypatch.setattr(trainer_module, "reduce_sums", lambda values: values + remote_totals)
    monkeypatch.setattr(
        trainer_module,
        "reduce_max",
        lambda values: torch.maximum(values, torch.tensor([4.0, 80.0], dtype=torch.float64)),
    )

    global_totals, metrics = _reduce_epoch_observability(
        local_totals,
        local_seconds=2.0,
        local_cuda_peak_allocated_bytes=100,
    )

    assert torch.equal(global_totals, local_totals + remote_totals)
    assert metrics == {
        "valid_examples": 8.0,
        "seconds": 4.0,
        "images_per_second": 2.0,
        "cuda_peak_allocated_bytes": 100.0,
        "teacher_clean_forward_calls": 6.0,
    }


@pytest.mark.parametrize("changed", ("src/ard/engine/trainer.py", "src/ard/engine/distributed.py"))
def test_engine_impact_selects_both_observability_contract_tests(changed: str) -> None:
    available = ("tests/unit/test_distributed.py", "tests/unit/test_pilot_observability.py")
    assert select((changed,), available).tests == available
