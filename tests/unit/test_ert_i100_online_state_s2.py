from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from ard.analysis.ert_i100_online_state_s2 import (
    OnlineStateS2Router,
    OnlineStateS2RoutingError,
    freeze_online_thresholds,
    global_logit_margin,
)
from ard.analysis.ert_stage_a_runtime import (
    StageARuntimeError,
    StageATreatment,
    _CanaryEpochSubset,
    _stable_train_labels,
    _validate_online_state_s2_treatment,
)
from ard.data import EpochShuffleSampler, IndexedDataset, SampleRef, SyntheticCIFAR
from ard.data.datasets import SourceIndexedSubset
from ard.engine import Trainer
from ard.objectives import RSLADObjective
from ard.policies import RSLADBaselinePolicy


def _manifest_builder_module():
    path = Path(__file__).parents[2] / "scripts" / "build_ert_i100_online_state_s2_manifest.py"
    spec = importlib.util.spec_from_file_location("ert_i100_online_state_manifest_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _logits(margins: list[float], *, wrong: set[int] | None = None) -> torch.Tensor:
    """Three-class logits with label zero and a requested global margin."""
    wrong = set() if wrong is None else wrong
    values = []
    for index, margin in enumerate(margins):
        if index in wrong:
            values.append([margin, 0.0, -2.0])
        else:
            values.append([margin, 0.0, -2.0])
    return torch.tensor(values, dtype=torch.float32)


def _write_prefix_and_thresholds(tmp_path: Path) -> tuple[OnlineStateS2Router, Path, dict[int, int]]:
    pytest.importorskip("pyarrow")
    labels = {11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}
    prefix = OnlineStateS2Router(
        arm="prefix",
        train_labels=labels,
        output_dir=tmp_path / "prefix",
        original_parent_checkpoint_sha256="a" * 64,
    )
    # IDs 11/15/16 are low-positive Student margins.  Teacher margin makes
    # them T1/T2/T3 respectively once the frozen scalar q10 is bound.
    clean = _logits([1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
    student_adv = _logits([0.10, 0.30, -0.20, 0.10, 0.10, 0.10])
    teacher_adv = _logits([0.70, 0.80, 0.70, 0.70, 0.05, -0.20])
    prefix.observe(
        epoch=100,
        sample_ids=torch.tensor(list(labels)),
        labels=torch.zeros(len(labels), dtype=torch.long),
        valid_mask=torch.ones(len(labels), dtype=torch.bool),
        student_clean_logits=clean,
        student_adversarial_logits=student_adv,
        teacher_adversarial_logits=teacher_adv,
    )
    prefix.flush_epoch(100)
    checkpoint = tmp_path / "prefix-e100.pt"
    checkpoint.write_bytes(b"frozen-prefix-checkpoint")
    thresholds = tmp_path / "thresholds.json"
    artifact = freeze_online_thresholds(
        prefix_router_state=prefix.state_dict(),
        prefix_checkpoint=checkpoint,
        output_path=thresholds,
        source_git_sha="b" * 40,
        training_attack_identity_sha256="c" * 64,
    )
    assert artifact["population"]["student"]["equal_threshold_count"] == 4
    assert artifact["population"]["student"]["at_or_below_threshold_count"] == 4
    return prefix, thresholds, labels


def _observe_child(router: OnlineStateS2Router, *, epoch: int = 101):
    labels = torch.zeros(6, dtype=torch.long)
    return router.observe(
        epoch=epoch,
        sample_ids=torch.tensor([11, 12, 13, 14, 15, 16]),
        labels=labels,
        valid_mask=torch.ones(6, dtype=torch.bool),
        student_clean_logits=_logits([1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
        student_adversarial_logits=_logits([0.10, 0.30, -0.20, 0.10, 0.10, 0.10]),
        teacher_adversarial_logits=_logits([0.70, 0.80, 0.70, 0.70, 0.05, -0.20]),
    )


def test_global_logit_margin_uses_each_models_own_strongest_nontrue() -> None:
    logits = torch.tensor([[0.25, 0.20, 0.24], [0.40, 0.50, -1.0]], requires_grad=True)
    values = global_logit_margin(logits, torch.tensor([0, 0]))
    assert values.tolist() == pytest.approx([0.01, -0.10])
    assert not values.requires_grad


def test_canary_subset_maps_sampler_positions_to_noncontiguous_stable_ids() -> None:
    """The public canary must support the real SampleRef sampler protocol."""
    indexed = IndexedDataset(SyntheticCIFAR(size=12, num_classes=10, seed=7))
    # Source IDs are intentionally non-contiguous and not in sorted order.
    full_view = SourceIndexedSubset(indexed, [9, 2, 11, 5])
    bounded = _CanaryEpochSubset(full_view, positions=[2, 0], source_ids=[11, 9])
    references = list(EpochShuffleSampler(len(bounded), seed=3, shuffle=False))
    assert all(isinstance(reference, SampleRef) for reference in references)

    items = [bounded[reference] for reference in references]
    assert [item[2] for item in items] == [11, 9]
    assert [item[3] for item in items] == [True, True]
    assert [item[4] for item in items] == [1, 1]


def test_online_state_manifest_binds_phase_epochs_and_static_cli_to_a_real_job(tmp_path: Path) -> None:
    builder = _manifest_builder_module()
    manifest = builder.build_manifest(
        source_sha="a" * 40,
        campaign_root=tmp_path / "campaign",
        requested_at="2026-09-04T00:00:00+00:00",
    )
    static = manifest["canary"]["static_cli"]
    static_command = [
        str(builder.PYTHON),
        str(builder.ROOT / "scripts/run_ert_i100_online_state_s2.py"),
        "--help",
    ]
    assert static == [
        {
            "job_id": "prefix-dev-1",
            "commands": [static_command],
            "timeout_seconds": 30,
            "parallel_safe": False,
        }
    ]
    jobs = {job["job_id"]: job for job in manifest["jobs"]}
    assert jobs["prefix-dev-1"]["epoch_binding"] == {
        "scientific_start_epoch": 100,
        "scientific_final_epoch": 100,
    }
    assert jobs["prefix-dev-1"]["command"][-1] == "101"
    assert jobs["arm-dev-1-control"]["epoch_binding"] == {
        "scientific_start_epoch": 101,
        "scientific_final_epoch": 114,
    }
    assert jobs["arm-dev-1-control"]["command"][-1] == "115"


def test_canary_subset_label_recovery_is_independent_of_wrapper_depth() -> None:
    class TargetsDataset:
        targets = [3, 4, 5, 6, 7, 8]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(3, 2, 2), self.targets[index]

    full_view = SourceIndexedSubset(IndexedDataset(TargetsDataset()), [5, 1, 4])
    bounded = _CanaryEpochSubset(full_view, positions=[2, 0], source_ids=[4, 5])
    assert _stable_train_labels(bounded) == {4: 7, 5: 8}


def test_frozen_q10_ties_and_branch_priority_are_explicit(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    prefix, thresholds, labels = _write_prefix_and_thresholds(tmp_path)
    child = OnlineStateS2Router(
        arm="pmp",
        train_labels=labels,
        output_dir=tmp_path / "pmp",
        original_parent_checkpoint_sha256="a" * 64,
        thresholds_path=thresholds,
    )
    child.adopt_prefix_state(prefix.state_dict())
    decision = _observe_child(child)
    # Only ID 11 is Online-S2×T1.  ID 12 is S1; 13 is S3; 14 is Clean
    # Wrong; and 15/16 are Online-S2×T2/T3.
    assert decision.eligible_s2_t1.tolist() == [True, False, False, False, False, False]
    assert decision.action_active.tolist() == [True, False, False, False, False, False]
    child.flush_epoch(101)
    import pyarrow.parquet as pq

    state_path = tmp_path / "pmp" / "online-state" / "epoch-101.parquet"
    rows = {row["sample_id"]: row for row in pq.read_table(state_path).to_pylist()}
    assert {sample_id: row["branch"] for sample_id, row in rows.items()} == {
        11: "S2_T1",
        12: "S1",
        13: "S3",
        14: "CW",
        15: "S2_T2",
        16: "S2_T3",
    }
    assert all(not row["action_active"] or row["branch"] == "S2_T1" for row in rows.values())


def test_control_observes_same_state_but_never_activates_loss(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    prefix, thresholds, labels = _write_prefix_and_thresholds(tmp_path)
    control = OnlineStateS2Router(
        arm="control",
        train_labels=labels,
        output_dir=tmp_path / "control",
        original_parent_checkpoint_sha256="a" * 64,
        thresholds_path=thresholds,
    )
    control.adopt_prefix_state(prefix.state_dict())
    decision = _observe_child(control)
    assert decision.eligible_s2_t1.tolist() == [True, False, False, False, False, False]
    assert not bool(decision.action_active.any())
    control.flush_epoch(101)
    assert control.epoch_statistics[101]["online_s2_t1_count"] == 1
    assert control.epoch_statistics[101]["active_treatment_count"] == 0


def test_online_router_binds_finite_boundary_runtime_metrics_into_checkpoint_state(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    prefix, thresholds, labels = _write_prefix_and_thresholds(tmp_path)
    child = OnlineStateS2Router(
        arm="dbdp",
        train_labels=labels,
        output_dir=tmp_path / "dbdp",
        original_parent_checkpoint_sha256="a" * 64,
        thresholds_path=thresholds,
    )
    child.adopt_prefix_state(prefix.state_dict())
    _observe_child(child)
    child.flush_epoch(101)
    child.record_runtime_metrics(
        epoch=101,
        metrics={
            "boundary_active_count": 1.0,
            "boundary_loss_count": 1.0,
            "boundary_input_gradient_calls": 2.0,
            "loss": 0.5,
        },
    )
    state = child.state_dict()["epoch_statistics"]["101"]
    assert state["boundary_active_count"] == 1.0
    assert state["boundary_loss_count"] == 1.0
    assert state["boundary_input_gradient_calls"] == 2.0
    assert "loss" not in state
    with pytest.raises(OnlineStateS2RoutingError, match="non-finite"):
        child.record_runtime_metrics(epoch=101, metrics={"boundary_active_count": float("nan")})


def test_control_records_state_reentry_independently_of_action(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    prefix, thresholds, labels = _write_prefix_and_thresholds(tmp_path)
    control = OnlineStateS2Router(
        arm="control",
        train_labels=labels,
        output_dir=tmp_path / "control",
        original_parent_checkpoint_sha256="a" * 64,
        thresholds_path=thresholds,
    )
    control.adopt_prefix_state(prefix.state_dict())
    _observe_child(control, epoch=101)
    control.flush_epoch(101)
    # ID 11 leaves S2×T1 at e102.
    control.observe(
        epoch=102,
        sample_ids=torch.tensor([11, 12, 13, 14, 15, 16]),
        labels=torch.zeros(6, dtype=torch.long),
        valid_mask=torch.ones(6, dtype=torch.bool),
        student_clean_logits=_logits([1.0] * 6),
        student_adversarial_logits=_logits([0.30, 0.30, -0.20, 0.30, 0.10, 0.10]),
        teacher_adversarial_logits=_logits([0.70, 0.80, 0.70, 0.70, 0.05, -0.20]),
    )
    control.flush_epoch(102)
    # It re-enters e103; the Control arm must still report zero action but a
    # nonzero state re-entry.
    _observe_child(control, epoch=103)
    control.flush_epoch(103)
    assert control.epoch_statistics[103]["s2_t1_state_reentry_count"] == 1
    assert control.epoch_statistics[103]["action_reentry_count"] == 0


def test_child_rejects_prefix_state_or_threshold_with_foreign_lineage(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    prefix, thresholds, labels = _write_prefix_and_thresholds(tmp_path)
    raw = json.loads(thresholds.read_text(encoding="utf-8"))
    raw["original_parent_checkpoint_sha256"] = "f" * 64
    thresholds.write_text(json.dumps(raw), encoding="utf-8")
    thresholds.with_name(thresholds.name + ".sha256").write_text(
        hashlib.sha256(thresholds.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    with pytest.raises(OnlineStateS2RoutingError, match="incompatible lineage"):
        OnlineStateS2Router(
            arm="pmp",
            train_labels=labels,
            output_dir=tmp_path / "bad",
            original_parent_checkpoint_sha256="a" * 64,
            thresholds_path=thresholds,
        )
    assert prefix.state_dict()["arm"] == "prefix"


def test_child_rejects_threshold_from_a_different_e100_prefix_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    prefix, thresholds, labels = _write_prefix_and_thresholds(tmp_path)
    child = OnlineStateS2Router(
        arm="pmp",
        train_labels=labels,
        output_dir=tmp_path / "child",
        original_parent_checkpoint_sha256="a" * 64,
        thresholds_path=thresholds,
    )
    with pytest.raises(OnlineStateS2RoutingError, match="not bound to this exact e100 child lineage"):
        child.adopt_prefix_state(
            prefix.state_dict(),
            prefix_checkpoint_sha256="d" * 64,
            source_git_sha="b" * 40,
            training_attack_identity_sha256="c" * 64,
        )


def test_child_accepts_only_sha_verified_materialized_prefix_state(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    prefix, thresholds, labels = _write_prefix_and_thresholds(tmp_path)
    original = tmp_path / "prefix" / "online-state" / "epoch-100.parquet"
    materialized = tmp_path / "materialized-e100.parquet"
    materialized.write_bytes(original.read_bytes())
    child = OnlineStateS2Router(
        arm="pmp",
        train_labels=labels,
        output_dir=tmp_path / "child",
        original_parent_checkpoint_sha256="a" * 64,
        thresholds_path=thresholds,
    )
    child.adopt_prefix_state(prefix.state_dict(), materialized_state_path=materialized)
    assert child.state_dict()["prefix_state"]["materialized_path"] == str(materialized.resolve())
    materialized.write_bytes(b"not-a-parquet-file")
    rejected = OnlineStateS2Router(
        arm="pmp",
        train_labels=labels,
        output_dir=tmp_path / "rejected",
        original_parent_checkpoint_sha256="a" * 64,
        thresholds_path=thresholds,
    )
    with pytest.raises(OnlineStateS2RoutingError, match="path/hash"):
        rejected.adopt_prefix_state(prefix.state_dict(), materialized_state_path=materialized)


def test_child_checkpoint_state_preserves_prior_online_action_for_resume(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    prefix, thresholds, labels = _write_prefix_and_thresholds(tmp_path)
    child = OnlineStateS2Router(
        arm="pmp",
        train_labels=labels,
        output_dir=tmp_path / "child",
        original_parent_checkpoint_sha256="a" * 64,
        thresholds_path=thresholds,
    )
    child.adopt_prefix_state(prefix.state_dict())
    _observe_child(child, epoch=101)
    child.flush_epoch(101)
    resumed = OnlineStateS2Router(
        arm="pmp",
        train_labels=labels,
        output_dir=tmp_path / "resumed",
        original_parent_checkpoint_sha256="a" * 64,
        thresholds_path=thresholds,
    )
    resumed.load_state_dict(child.state_dict())
    decision = resumed.observe(
        epoch=102,
        sample_ids=torch.tensor([11, 12, 13, 14, 15, 16]),
        labels=torch.zeros(6, dtype=torch.long),
        valid_mask=torch.ones(6, dtype=torch.bool),
        student_clean_logits=_logits([1.0] * 6),
        # ID 11 leaves Online-S2×T1 for S1.  The resumed state must register
        # one exit instead of treating this as a fresh, unobserved visit.
        student_adversarial_logits=_logits([0.30, 0.30, -0.20, 0.30, 0.10, 0.10]),
        teacher_adversarial_logits=_logits([0.70, 0.80, 0.70, 0.70, 0.05, -0.20]),
    )
    assert not bool(decision.action_active[0])
    resumed.flush_epoch(102)
    assert resumed.epoch_statistics[102]["action_exit_count"] == 1


def test_online_boundary_treatment_is_allowed_only_with_explicit_router_opt_in() -> None:
    with pytest.raises(RuntimeError, match="fixed selected mask"):
        StageATreatment(
            arm="bad",
            mask_key=None,
            kind="broad",
            boundary_intervention="pair_margin",
            boundary_coefficient=0.1,
        )
    accepted = StageATreatment(
        arm="OS-PMP",
        mask_key=None,
        kind="broad",
        boundary_intervention="pair_margin",
        boundary_coefficient=0.1,
        online_state_s2=True,
    )
    assert accepted.online_state_s2


def test_online_router_rejects_clean_wrong_or_margin_component_stacking() -> None:
    stacked = StageATreatment(
        arm="bad-online-stack",
        mask_key=None,
        kind="broad",
        boundary_intervention="pair_margin",
        boundary_coefficient=0.1,
        extra_clean_ce=0.15,
        online_state_s2=True,
    )
    with pytest.raises(StageARuntimeError, match="may not stack"):
        _validate_online_state_s2_treatment(arm="pmp", treatment=stacked)


def test_online_router_accepts_only_registered_pmp_and_dbdp_boundaries() -> None:
    pmp = StageATreatment(
        arm="OS-PMP",
        mask_key=None,
        kind="broad",
        boundary_intervention="pair_margin",
        boundary_coefficient=0.05380932585058825,
        online_state_s2=True,
    )
    dbdp = StageATreatment(
        arm="OS-DBDP",
        mask_key=None,
        kind="broad",
        boundary_intervention="detached_boundary_distance",
        boundary_coefficient=31.649566509850324,
        online_state_s2=True,
    )
    _validate_online_state_s2_treatment(arm="pmp", treatment=pmp)
    _validate_online_state_s2_treatment(arm="dbdp", treatment=dbdp)
    wrong_coefficient = StageATreatment(
        arm="OS-PMP-wrong-coefficient",
        mask_key=None,
        kind="broad",
        boundary_intervention="pair_margin",
        boundary_coefficient=0.05380932585058826,
        online_state_s2=True,
    )
    with pytest.raises(StageARuntimeError, match="registered boundary"):
        _validate_online_state_s2_treatment(arm="pmp", treatment=wrong_coefficient)
    with pytest.raises(StageARuntimeError, match="registered boundary"):
        _validate_online_state_s2_treatment(arm="pmp", treatment=dbdp)


def test_trainer_allows_boundary_only_for_one_online_router_and_rejects_mode_stacking(tmp_path: Path) -> None:
    student = torch.nn.Linear(2, 2)
    teacher = torch.nn.Linear(2, 2)
    common = {
        "model": student,
        "optimizer": torch.optim.SGD(student.parameters(), lr=0.1),
        "scheduler": None,
        "scaler": None,
        "attack": object(),
        "selection_attack": object(),
        "objective": RSLADObjective(),
        "device": torch.device("cpu"),
        "output_dir": tmp_path,
        "config_hash": "d" * 64,
        "seed": 1,
        "teacher": teacher,
        "policy": RSLADBaselinePolicy(),
    }
    router = object()
    trainer = Trainer(
        **common,
        boundary_intervention="pair_margin",
        boundary_coefficient=0.1,
        online_state_s2_router=router,
    )
    assert trainer.online_state_s2_router is router
    with pytest.raises(ValueError, match="mutually exclusive"):
        Trainer(**common, dynamic_s3_router=object(), online_state_s2_router=object())
