from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from ard.analysis.ert_dynamic_s3_recovery import (
    DynamicS3Router,
    _transition_summary,
    active_s3_recovery_mask,
)
from ard.attacks import AttackGenerator, AttackRequest, AttackResult
from ard.config.schema import AttackConfig
from ard.data import IndexedBatch
from ard.engine import Trainer
from ard.objectives import RSLADObjective
from ard.policies import RSLADBaselinePolicy
from ard.targets import UniformSofteningTeacherTargetPolicy


def _logits(predictions: list[int]) -> torch.Tensor:
    return torch.tensor([[3.0, -1.0] if prediction == 0 else [-1.0, 3.0] for prediction in predictions])


class _FlipAttack(AttackGenerator):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: AttackRequest) -> AttackResult:
        self.calls += 1
        adversarial = request.inputs.clone()
        adversarial[:, 0, 0, 0] = 1.0 - adversarial[:, 0, 0, 0]
        return AttackResult(
            adversarial=adversarial,
            initial_delta=torch.zeros_like(adversarial),
            step_losses=(),
            max_abs_delta=1.0,
        )


class _BNStateStudent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(1)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        value = self.bn(images[:, 0, 0, 0, None]).squeeze(1)
        return torch.stack((-self.scale * value, self.scale * value), dim=1)


class _CorrectTeacher(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.stack((self.bias.expand(images.shape[0]), -self.bias.expand(images.shape[0])), dim=1)


def test_active_s3_recovery_truth_table_is_exact_and_detached() -> None:
    labels = torch.tensor([0, 0, 0, 0])
    values = active_s3_recovery_mask(
        # clean wrong, adv correct, teacher wrong, and the only active row.
        student_clean_logits=_logits([1, 0, 0, 0]),
        student_adversarial_logits=_logits([1, 0, 1, 1]),
        teacher_adversarial_logits=_logits([0, 0, 1, 0]),
        labels=labels,
        valid_mask=torch.tensor([True, True, True, True]),
    )
    assert values.tolist() == [False, False, False, True]
    assert not values.requires_grad


def _observe(router: DynamicS3Router, epoch: int, *, clean: list[int], adversarial: list[int], teacher: list[int]):
    labels = torch.tensor([0, 0])
    return router.observe(
        epoch=epoch,
        sample_ids=torch.tensor([11, 31]),
        labels=labels,
        valid_mask=torch.tensor([True, True]),
        student_clean_logits=_logits(clean),
        student_adversarial_logits=_logits(adversarial),
        teacher_clean_logits=_logits([0, 0]),
        teacher_adversarial_logits=_logits(teacher),
    )


def test_epoch80_capture_is_shared_then_fixed_is_immutable_and_dynamic_recomputes(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    labels = {11: 0, 31: 0}
    fixed = DynamicS3Router(arm="fixed", train_labels=labels, output_dir=tmp_path / "fixed")
    dynamic = DynamicS3Router(arm="dynamic", train_labels=labels, output_dir=tmp_path / "dynamic")
    # At epoch 80 both receive exactly the same same-step action: only ID 11
    # is S3×Teacher-correct.
    fixed80 = _observe(fixed, 80, clean=[0, 0], adversarial=[1, 0], teacher=[0, 0])
    dynamic80 = _observe(dynamic, 80, clean=[0, 0], adversarial=[1, 0], teacher=[0, 0])
    assert fixed80.action_active.tolist() == dynamic80.action_active.tolist() == [True, False]
    fixed.flush_epoch(80)
    dynamic.flush_epoch(80)
    assert fixed.capture_ids == dynamic.capture_ids == (11,)
    # ID 11 recovered and ID 31 newly failed: fixed keeps the captured action,
    # whereas dynamic exits/enters in the same next visit without an epoch lag.
    fixed81 = _observe(fixed, 81, clean=[0, 0], adversarial=[0, 1], teacher=[0, 0])
    dynamic81 = _observe(dynamic, 81, clean=[0, 0], adversarial=[0, 1], teacher=[0, 0])
    assert fixed81.current_active.tolist() == dynamic81.current_active.tolist() == [False, True]
    assert fixed81.action_active.tolist() == [True, False]
    assert dynamic81.action_active.tolist() == [False, True]
    fixed.flush_epoch(81)
    dynamic.flush_epoch(81)
    assert fixed.finalize()["selected_count"] == 1
    assert dynamic.finalize()["selected_count"] == 1


def test_capture_rejects_missing_or_duplicate_train_ids(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    router = DynamicS3Router(arm="fixed", train_labels={11: 0, 31: 0}, output_dir=tmp_path)
    _observe(router, 80, clean=[0, 0], adversarial=[1, 0], teacher=[0, 0])
    # A second same-epoch visit proves the capture is not exactly once.
    _observe(router, 80, clean=[0, 0], adversarial=[1, 0], teacher=[0, 0])
    with pytest.raises(RuntimeError, match="duplicate"):
        router.flush_epoch(80)


def test_capture_checkpoint_state_restores_without_recomputing_history(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = DynamicS3Router(arm="fixed", train_labels={11: 0, 31: 0}, output_dir=tmp_path / "source")
    _observe(source, 80, clean=[0, 0], adversarial=[1, 0], teacher=[0, 0])
    source.flush_epoch(80)
    restored = DynamicS3Router(arm="fixed", train_labels={11: 0, 31: 0}, output_dir=tmp_path / "restored")
    restored.load_state_dict(source.state_dict())
    # The epoch-81 action comes from the immutable captured ID, even though
    # the current state is the opposite.  A resume cannot silently recapture.
    decision = _observe(restored, 81, clean=[0, 0], adversarial=[0, 1], teacher=[0, 0])
    assert decision.action_active.tolist() == [True, False]


def test_capture_only_state_can_be_adopted_by_a_child_with_its_own_lineage(tmp_path: Path) -> None:
    """FIXED/DYNAMIC must fork one immutable capture-only epoch-80 action map."""
    pytest.importorskip("pyarrow")
    labels = {11: 0, 31: 0}
    prefix = DynamicS3Router(arm="capture", train_labels=labels, output_dir=tmp_path / "prefix")
    _observe(prefix, 80, clean=[0, 0], adversarial=[1, 0], teacher=[0, 0])
    prefix.flush_epoch(80)
    prefix_state = prefix.state_dict()
    prefix_epoch_state = Path(prefix_state["state_paths"]["80"]["path"])

    child = DynamicS3Router(arm="dynamic", train_labels=labels, output_dir=tmp_path / "child")
    with pytest.raises(RuntimeError, match="incompatible lineage"):
        child.load_state_dict(prefix_state)
    child.adopt_capture_state(prefix_state)
    child_epoch_state = child.output_dir / "dynamic-state" / "epoch-80.parquet"
    child_epoch_state.parent.mkdir(parents=True)
    child_epoch_state.write_bytes(prefix_epoch_state.read_bytes())
    child.register_existing_epoch_state(epoch=80, path=child_epoch_state)
    capture = yaml.safe_load((child.output_dir / "routing-capture-mask.json").read_text(encoding="utf-8"))
    assert capture["arm"] == "dynamic"
    assert capture["selected_ids"] == [11]
    assert child.state_dict()["arm"] == "dynamic"
    # The child starts at epoch 81 and uses its current, same-step predicate;
    # it never recalculates the historical capture action.
    decision = _observe(child, 81, clean=[0, 0], adversarial=[0, 1], teacher=[0, 0])
    assert decision.action_active.tolist() == [False, True]


def test_target_policy_only_rslad_does_not_require_a_fixed_mask(tmp_path: Path) -> None:
    """rslad_student/joint target shaping is policy-driven, not fixed-ID routing."""
    student = torch.nn.Linear(2, 2)
    teacher = torch.nn.Linear(2, 2)
    trainer = Trainer(
        model=student,
        optimizer=torch.optim.SGD(student.parameters(), lr=0.1),
        scheduler=None,
        scaler=None,
        attack=object(),
        selection_attack=object(),
        objective=RSLADObjective(),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        config_hash="a" * 64,
        seed=1,
        teacher=teacher,
        policy=RSLADBaselinePolicy(),
        target_policy=UniformSofteningTeacherTargetPolicy(rho_max=0.5),
    )
    assert trainer.intervention_mask is None


def test_dynamic_router_is_same_attack_baseline_equivalent_when_ignored_and_only_active_gets_advce(
    tmp_path: Path,
) -> None:
    batch = IndexedBatch(
        images=torch.tensor([[[[0.0]]], [[[1.0]]]]),
        labels=torch.tensor([0, 0]),
        sample_ids=torch.tensor([11, 31]),
        state_update_mask=torch.tensor([True, True]),
    )
    initial = _BNStateStudent()

    def make(*, router: DynamicS3Router | None, coefficient: float | None, directory: Path):
        student, teacher = _BNStateStudent(), _CorrectTeacher()
        student.load_state_dict(initial.state_dict())
        attack = _FlipAttack()
        trainer = Trainer(
            model=student,
            optimizer=torch.optim.SGD(student.parameters(), lr=0.1),
            scheduler=None,
            scaler=None,
            attack=attack,
            selection_attack=_FlipAttack(),
            objective=RSLADObjective(),
            device=torch.device("cpu"),
            output_dir=directory,
            config_hash="b" * 64,
            seed=7,
            teacher=teacher,
            policy=RSLADBaselinePolicy(),
            dynamic_s3_router=router,
            adversarial_ce_coefficient=coefficient,
        )
        trainer.current_epoch = 80
        trainer.train_epoch([batch])
        return trainer, attack

    baseline, baseline_attack = make(router=None, coefficient=None, directory=tmp_path / "baseline")
    observed_router = DynamicS3Router(arm="baseline", train_labels={11: 0, 31: 0}, output_dir=tmp_path / "observed")
    observed, observed_attack = make(router=observed_router, coefficient=None, directory=tmp_path / "observed")
    observed_router.flush_epoch(80)
    assert baseline_attack.calls == observed_attack.calls == 1
    torch.testing.assert_close(baseline.model.scale, observed.model.scale, rtol=0, atol=0)
    torch.testing.assert_close(baseline.model.bn.running_mean, observed.model.bn.running_mean, rtol=0, atol=0)
    assert observed_router.epoch_statistics[80]["current_active_count"] == 1
    assert observed_router.epoch_statistics[80]["active_count"] == 0
    assert all(parameter.grad is None for parameter in observed.teacher.parameters())

    active_router = DynamicS3Router(arm="dynamic", train_labels={11: 0, 31: 0}, output_dir=tmp_path / "active")
    active, active_attack = make(router=active_router, coefficient=0.075, directory=tmp_path / "active")
    active_router.flush_epoch(80)
    assert active_attack.calls == 1
    assert active_router.epoch_statistics[80]["active_count"] == 1
    assert not torch.equal(active.model.scale, baseline.model.scale)
    assert all(parameter.grad is None for parameter in active.teacher.parameters())


def test_checked_in_dynamic_attack_contracts_exactly_match_canonical_parent_identities() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/analysis/ert_dynamic_s3_recovery_v1.yaml").read_text(encoding="utf-8"))
    assert config["training_attack"] == AttackConfig(loss="kl", kl_target="teacher_clean", steps=10).identity()
    assert config["endpoint_attack"] == AttackConfig(loss="ce", steps=20).identity()


def test_reentry_counts_a_sample_that_is_active_on_its_initial_visit() -> None:
    rows = [
        {
            "epoch": 80,
            "sample_id": 11,
            "student_clean_correct": True,
            "student_adv_correct": False,
            "teacher_adv_correct": True,
            "action_active": True,
            "current_active": True,
        },
        {
            "epoch": 81,
            "sample_id": 11,
            "student_clean_correct": True,
            "student_adv_correct": True,
            "teacher_adv_correct": True,
            "action_active": False,
            "current_active": False,
        },
        {
            "epoch": 82,
            "sample_id": 11,
            "student_clean_correct": True,
            "student_adv_correct": False,
            "teacher_adv_correct": True,
            "action_active": True,
            "current_active": True,
        },
    ]
    summary = _transition_summary(rows)
    assert summary["action_events"] == {
        "entries": 1,
        "exits": 1,
        "reentries": 1,
        "switches": 2,
        "short_cycle_reentries": 1,
    }


def test_history_router_uses_clean_correct_s3_and_two_correct_exit(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    labels = {11: 0}
    router = DynamicS3Router(arm="majority3_exit2", train_labels=labels, output_dir=tmp_path)
    def observe_one(epoch: int, clean: list[int], adv: list[int]):
        return router.observe(
            epoch=epoch,
            sample_ids=torch.tensor([11]),
            labels=torch.tensor([0]),
            valid_mask=torch.tensor([True]),
            student_clean_logits=_logits(clean),
            student_adversarial_logits=_logits(adv),
            teacher_clean_logits=_logits([0]),
            teacher_adversarial_logits=_logits([0]),
        )
    # Two clean-correct/adv-wrong visits enter at epoch 82.  Clean-wrong is
    # not an exit-correct visit and resets the streak; two clean+adv-correct
    # visits then exit the state.
    for epoch, clean, adv in (
        (80, [0], [1]),
        (81, [0], [1]),
        (82, [0], [0]),
        (83, [1], [0]),
        (84, [0], [0]),
        (85, [0], [0]),
    ):
        decision = observe_one(epoch, clean, adv)
        if epoch < 82:
            assert decision.action_active.tolist() == [False]
        if epoch == 82:
            assert decision.action_active.tolist() == [True]
        if epoch == 83:
            assert decision.action_active.tolist() == [False]
        if epoch == 84:
            assert decision.action_active.tolist() == [True]
        if epoch == 85:
            assert decision.action_active.tolist() == [False]
        router.flush_epoch(epoch)
    assert router.capture_complete
