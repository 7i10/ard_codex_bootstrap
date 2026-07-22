from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ard.attacks.base import AttackResult
from ard.config.schema import ExperimentConfig
from ard.data import EpochShuffleSampler, IndexedBatch, IndexedDataset, SyntheticCIFAR, collate_indexed
from ard.engine import Trainer
from ard.engine.distributed import reduce_min
from ard.objectives import RSLADObjective
from ard.policies import (
    HardFallbackPolicy,
    JointDownweightPolicy,
    JointRiskPolicy,
    PolicyContext,
    RSLADBaselinePolicy,
    StudentRiskPolicy,
    student_risk_from_margin,
    teacher_risk_from_entropy,
)
from ard.signals import RobustMarginSignal, shannon_entropy
from ard.state import SampleStateStore
from ard.targets import UniformSofteningTeacherTargetPolicy

pytestmark = pytest.mark.t2


def _context(valid: torch.Tensor) -> PolicyContext:
    return PolicyContext(valid_mask=valid, global_min=reduce_min)


def _v2_experiment(*, method: dict[str, object], num_classes: int = 3, image_size: int = 4) -> dict[str, object]:
    return {
        "schema_version": 2,
        "protocol": {"id": "synthetic_smoke_v2"},
        "seeds": {
            "split": 0,
            "model_init": 0,
            "data_order": 0,
            "augmentation": 0,
            "train_attack": 0,
            "evaluation_attack": 0,
            "qualitative_panel": 0,
        },
        "dataset": {"name": "synthetic_cifar", "num_samples": 8, "num_classes": num_classes, "image_size": image_size},
        "student": {"architecture": "fixture_cnn", "num_classes": num_classes},
        "teacher": {"source": "fixture", "architecture": "fixture_cnn", "num_classes": num_classes},
        "method": method,
        "optimizer": {"id": "sgd", "learning_rate": 0.01, "momentum": 0.0, "weight_decay": 0.0, "nesterov": False},
        "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
        "training": {"epochs": 1, "per_rank_batch_size": 2, "global_batch_size": 2},
    }


def _target_policy() -> dict[str, object]:
    return {
        "id": "teacher_target_uniform_mix",
        "version": 1,
        "risk_transform": "identity",
        "mixing": "uniform",
        "apply_to": "adversarial_student_kd",
        "rho_max": 0.5,
    }


def test_robust_margin_formula_is_detached_fp32_and_excludes_padding() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    signal = RobustMarginSignal().compute(
        student_adv_logits=logits,
        labels=torch.tensor([0, 1]),
        valid_mask=torch.tensor([True, False]),
    )
    probabilities = torch.softmax(logits.detach().float(), dim=1)
    expected = torch.stack((probabilities[0, 0] - probabilities[0, 1], probabilities[1, 1] - probabilities[1, 2]))
    assert signal.values.dtype == torch.float32
    assert not signal.values.requires_grad
    assert torch.allclose(signal.values, expected, atol=0, rtol=0)
    assert torch.equal(signal.valid_mask, torch.tensor([True, False]))


def test_store_ema_correctness_forgetting_padding_and_exact_roundtrip() -> None:
    store = SampleStateStore(ema_decay=0.9)
    store.record_pending(
        sample_ids=torch.tensor([9, 10]),
        margins=torch.tensor([0.4, -0.2]),
        robust_correct=torch.tensor([True, False]),
        valid_mask=torch.tensor([True, False]),
        update=3,
    )
    store.merge_pending([store.pending_state()])
    assert set(store.records) == {9}
    record = store.records[9]
    assert record.margin_ema == pytest.approx(0.4)
    assert record.seen == 1 and record.robust_correct_count == 1
    assert record.robust_correct_frequency == pytest.approx(1.0)
    assert record.forgetting_count == 0 and record.last_update == 3

    store.record_pending(
        sample_ids=torch.tensor([9]),
        margins=torch.tensor([-0.6]),
        robust_correct=torch.tensor([False]),
        valid_mask=torch.tensor([True]),
        update=4,
    )
    store.merge_pending([store.pending_state()])
    record = store.records[9]
    assert record.margin_ema == pytest.approx(0.9 * 0.4 + 0.1 * -0.6)
    assert record.seen == 2 and record.robust_correct_count == 1
    assert record.robust_correct_frequency == pytest.approx(0.5)
    assert record.previous_robust_correct is False and record.forgetting_count == 1 and record.last_update == 4

    snapshot = store.state_dict()
    restored = SampleStateStore(ema_decay=0.9)
    restored.load_state_dict(copy.deepcopy(snapshot))
    assert restored.state_dict() == snapshot


def test_pending_rank_merge_is_stable_and_duplicate_safe() -> None:
    store = SampleStateStore()
    rank0 = [{"sample_id": 4, "margin": 0.2, "robust_correct": True, "update": 1, "rank": 0, "order": 0}]
    rank1 = [
        {"sample_id": 4, "margin": -0.7, "robust_correct": False, "update": 1, "rank": 1, "order": 0},
        {"sample_id": 2, "margin": -0.4, "robust_correct": False, "update": 1, "rank": 1, "order": 1},
    ]
    store.merge_pending([rank1, rank0])
    assert list(store.records) == [2, 4]
    # Rank zero's canonical duplicate is applied once, not twice.
    assert store.records[4].margin_ema == pytest.approx(0.2)
    assert store.records[4].seen == 1 and store.records[4].forgetting_count == 0
    assert store.pending == []


def test_student_joint_risk_ranges_monotonicity_and_warmup_is_exact_baseline_rslad() -> None:
    margins = torch.tensor([-1.0, 0.0, 1.0])
    student = student_risk_from_margin(margins)
    assert torch.equal(student, torch.tensor([1.0, 0.5, 0.0]))
    teacher = teacher_risk_from_entropy(torch.tensor([0.0, torch.log(torch.tensor(3.0))]), num_classes=3)
    assert torch.allclose(teacher, torch.tensor([1.0, 0.0]), atol=1e-7, rtol=0)

    student_weights = StudentRiskPolicy().compute(
        {"student_risk": student}, context=_context(torch.ones(3, dtype=torch.bool)), num_classes=3
    )
    assert torch.equal(student_weights.hard_weight, torch.zeros_like(student))
    assert torch.equal(student_weights.kd_weight, torch.ones_like(student))
    assert torch.all((student_weights.hard_weight >= 0) & (student_weights.hard_weight <= 1))
    joint = torch.tensor([0.0, 0.25, 1.0])
    joint_weights = JointRiskPolicy().compute(
        {"joint_risk": joint}, context=_context(torch.ones(3, dtype=torch.bool)), num_classes=3
    )
    assert torch.allclose(joint_weights.joint_risk, joint)
    assert torch.equal(joint_weights.hard_weight, torch.zeros(3))
    assert torch.equal(joint_weights.kd_weight, torch.ones(3))
    # The Trainer, rather than sample observation count, owns warmup.  Epoch
    # zero is exact baseline RSLAD while margins are still observed elsewhere.
    trainer = object.__new__(Trainer)
    trainer.policy = StudentRiskPolicy()
    trainer.sample_store = SampleStateStore()
    trainer.current_epoch = 0
    trainer.policy_warmup_epochs = 1
    batch = IndexedBatch(
        images=torch.zeros(2, 3, 1, 1),
        labels=torch.tensor([0, 1]),
        sample_ids=torch.tensor([101, 202]),
    )
    warmup = Trainer._policy_weights(
        trainer,
        batch=batch,
        adversarial=batch.images,
        logits=torch.zeros(2, 3),
        valid_mask=torch.tensor([True, True]),
        student_signals={"student_risk": torch.tensor([0.0, 1.0])},
    )
    assert warmup is not None
    assert torch.equal(warmup.hard_weight, torch.zeros(2))
    assert torch.equal(warmup.kd_weight, torch.ones(2))
    assert torch.equal(warmup.joint_risk, torch.zeros(2))


def test_rslad_policy_fallback_identities() -> None:
    labels = torch.tensor([0, 2])
    student_adv = torch.tensor([[0.2, -0.1, 0.6], [0.3, 0.1, -0.2]], requires_grad=True)
    student_clean = torch.tensor([[0.5, 0.0, 0.1], [0.2, 0.7, -0.4]], requires_grad=True)
    teacher = torch.tensor([[1.0, -0.2, 0.3], [-0.1, 0.4, 0.2]])
    terms = RSLADObjective()(
        student_logits=student_adv, clean_student_logits=student_clean, teacher_logits=teacher, labels=labels
    )
    baseline = RSLADBaselinePolicy().compute({}, context=_context(torch.ones(2, dtype=torch.bool)), num_classes=3)
    assert torch.equal(terms.apply_policy(baseline).total, terms.kd)
    risk = torch.tensor([0.0, 1.0])
    uniform = terms.apply_policy(
        StudentRiskPolicy().compute(
            {"student_risk": risk}, context=_context(torch.ones(2, dtype=torch.bool)), num_classes=3
        )
    )
    assert torch.equal(uniform.total, terms.kd)
    downweighted = terms.apply_policy(
        JointDownweightPolicy().compute(
            {"joint_risk": risk}, context=_context(torch.ones(2, dtype=torch.bool)), num_classes=3
        )
    )
    assert torch.equal(downweighted.total, torch.stack((terms.kd[0], torch.zeros_like(terms.kd[1]))))
    fallback = terms.apply_policy(
        HardFallbackPolicy().compute(
            {"joint_risk": risk}, context=_context(torch.ones(2, dtype=torch.bool)), num_classes=3
        )
    )
    assert torch.equal(fallback.total, torch.stack((terms.kd[0], terms.hard[1])))


@pytest.mark.parametrize(
    "name",
    ["rslad", "rslad_entropy", "rslad_student", "rslad_joint", "rslad_joint_downweight", "rslad_hard_fallback"],
)
def test_rslad_method_ids_are_config_only_switches(name: str) -> None:
    method: dict[str, object] = {
        "id": name,
        "version": 1,
        "attack": {
            "loss": "kl",
            "kl_target": "teacher_clean",
            "epsilon": "1/255",
            "step_size": "1/255",
            "steps": 1,
        },
    }
    if name in {"rslad_student", "rslad_joint"}:
        method["target_policy"] = _target_policy()
    config = ExperimentConfig.model_validate(_v2_experiment(method=method))
    assert config.method.id == name


def test_oracle_mask_is_dev_only_and_student_aware_only() -> None:
    base = _v2_experiment(
        method={
            "id": "rslad_hard_fallback",
            "version": 1,
            "oracle_mask": True,
            "attack": {
                "loss": "kl",
                "kl_target": "teacher_clean",
                "epsilon": "1/255",
                "step_size": "1/255",
                "steps": 1,
            },
        }
    )
    assert ExperimentConfig.model_validate({**base, "tier": "dev"}).method.oracle_mask
    with pytest.raises(ValidationError, match="scientific/dev-only"):
        ExperimentConfig.model_validate({**base, "tier": "smoke"})
    invalid = {**base, "method": {**base["method"], "id": "rslad", "oracle_mask": True}}
    with pytest.raises(ValidationError, match="only defined"):
        ExperimentConfig.model_validate(invalid)


@pytest.mark.parametrize("name", ["rslad_student", "rslad_joint"])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("student_ema_decay", 0.8, "canonical EMA=0.9"),
        ("student_policy_warmup_epochs", 2, "canonical one-epoch-warmup"),
    ],
)
def test_canonical_student_aware_method_rejects_unversioned_variants(
    name: str,
    field: str,
    value: float | int,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(
            {
                **_v2_experiment(
                    num_classes=2,
                    image_size=2,
                    method={
                        "id": name,
                        "version": 1,
                        "target_policy": _target_policy(),
                        field: value,
                        "attack": {
                            "loss": "kl",
                            "kl_target": "teacher_clean",
                            "epsilon": "1/255",
                            "step_size": "1/255",
                            "steps": 1,
                        },
                    },
                ),
            }
        )


class _IdentityAttack:
    def generate(self, request: object) -> AttackResult:
        inputs = request.inputs  # type: ignore[attr-defined]
        return AttackResult(inputs, torch.zeros_like(inputs), (), 0.0)


def _m3_loaders(seed: int) -> tuple[DataLoader, DataLoader, EpochShuffleSampler]:
    dataset = IndexedDataset(SyntheticCIFAR(size=4, num_classes=2, image_size=2, seed=seed))
    sampler = EpochShuffleSampler(len(dataset), seed=seed)
    validation_sampler = EpochShuffleSampler(len(dataset), seed=seed, shuffle=False)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, collate_fn=collate_indexed)
    validation_loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=validation_sampler,
        collate_fn=collate_indexed,
    )
    return loader, validation_loader, sampler


def _m3_trainer(
    output: Path, *, method: str, captures: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Trainer:
    torch.manual_seed(321)
    student = nn.Sequential(nn.Flatten(), nn.Linear(3 * 2 * 2, 2))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(654)
        teacher = nn.Sequential(nn.Flatten(), nn.Linear(3 * 2 * 2, 2))
    optimizer = SGD(student.parameters(), lr=0.03, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    policy = StudentRiskPolicy() if method == "rslad_student" else JointRiskPolicy()
    trainer = Trainer(
        model=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        attack=_IdentityAttack(),
        selection_attack=_IdentityAttack(),
        objective=RSLADObjective(),
        policy=policy,
        sample_store=SampleStateStore(ema_decay=0.9),
        target_policy=UniformSofteningTeacherTargetPolicy(rho_max=0.5),
        policy_warmup_epochs=1,
        device=torch.device("cpu"),
        output_dir=output,
        # Deliberately constant across the one-epoch first leg and two-epoch
        # target; epochs is execution position, not scientific configuration.
        config_hash="m3-student-aware-two-epoch",
        seed=73,
    )
    original_policy_weights = trainer._policy_weights

    def record_policy_weights(**kwargs: object):
        weights = original_policy_weights(**kwargs)
        assert weights is not None
        batch = kwargs["batch"]
        assert isinstance(batch, IndexedBatch)
        valid = kwargs["valid_mask"]
        assert isinstance(valid, torch.Tensor)
        if trainer.current_epoch == 0:
            assert torch.equal(weights.hard_weight, torch.zeros_like(weights.hard_weight))
            assert torch.equal(weights.kd_weight, valid.to(dtype=weights.kd_weight.dtype))
            assert torch.equal(weights.joint_risk, torch.zeros_like(weights.joint_risk))
        else:
            prior_student_risk = student_risk_from_margin(trainer.sample_store.margin_ema(batch.sample_ids))
            if method == "rslad_student":
                expected_risk = prior_student_risk
            else:
                adversarial = kwargs["adversarial"]
                logits = kwargs["logits"]
                assert isinstance(adversarial, torch.Tensor) and isinstance(logits, torch.Tensor)
                with torch.no_grad():
                    entropy = shannon_entropy(trainer.teacher(adversarial))
                expected_risk = prior_student_risk * teacher_risk_from_entropy(
                    entropy,
                    num_classes=logits.shape[1],
                )
            torch.testing.assert_close(weights.joint_risk, expected_risk, rtol=0, atol=0)
            torch.testing.assert_close(weights.hard_weight, torch.zeros_like(expected_risk), rtol=0, atol=0)
            torch.testing.assert_close(weights.kd_weight, torch.ones_like(expected_risk), rtol=0, atol=0)
        captures.append(
            (
                trainer.current_epoch,
                weights.hard_weight.detach().clone(),
                weights.kd_weight.detach().clone(),
                weights.joint_risk.detach().clone(),
            )
        )
        return weights

    trainer._policy_weights = record_policy_weights  # type: ignore[method-assign]
    return trainer


@pytest.mark.t3
@pytest.mark.parametrize("method", ["rslad_student", "rslad_joint"])
def test_student_aware_two_epoch_resume_matches_uninterrupted_exactly(tmp_path: Path, method: str) -> None:
    full_captures: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    full = _m3_trainer(tmp_path / method / "full", method=method, captures=full_captures)
    full_loader, full_validation, _ = _m3_loaders(seed=73)
    full.fit(full_loader, validation_loader=full_validation, epochs=2)

    resumed_captures: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    first_leg = _m3_trainer(tmp_path / method / "resumed", method=method, captures=resumed_captures)
    first_loader, first_validation, _ = _m3_loaders(seed=73)
    first_leg.fit(first_loader, validation_loader=first_validation, epochs=1)
    resumed = _m3_trainer(tmp_path / method / "resumed", method=method, captures=resumed_captures)
    resumed_loader, resumed_validation, resumed_sampler = _m3_loaders(seed=73)
    state = resumed.resume(tmp_path / method / "resumed" / "last.pt", sampler=resumed_sampler)
    assert state.next_epoch == 1
    resumed.fit(resumed_loader, validation_loader=resumed_validation, epochs=2, start_epoch=state.next_epoch)

    assert {entry[0] for entry in full_captures} == {0, 1}
    assert {entry[0] for entry in resumed_captures} == {0, 1}
    for name, expected in full.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[name]), name
    assert full.sample_store.state_dict() == resumed.sample_store.state_dict()
    assert full.global_step == resumed.global_step == 4
    assert full.scheduler.state_dict() == resumed.scheduler.state_dict()
