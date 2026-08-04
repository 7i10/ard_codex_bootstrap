from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from ard.analysis import prescriptive_v3
from ard.analysis.schedule_control_fork import sha256_file
from ard.attacks import AttackGenerator, AttackRequest, AttackResult
from ard.cli import train as train_cli
from ard.cli.train import _attach_prescriptive_v3_input_artifacts
from ard.config import load_config
from ard.config.loader import resolved_config_dict
from ard.data import EpochShuffleSampler, IndexedBatch, IndexedDataset, SyntheticCIFAR, collate_indexed
from ard.engine import Trainer
from ard.engine.checkpoint import config_digest
from ard.objectives import RSLADObjective
from ard.policies import FixedInterventionMask, RSLADBaselinePolicy, selected_ids_sha256
from ard.state import SampleStateStore
from ard.targets import AnchoredTeacherTargetPolicy
from ard.tracking import LocalTracker, TrackingError
from tests.unit.test_schedule_control_fork import _child_config, _equal, _inputs, _raw_config, _spec

pytestmark = pytest.mark.t2


def test_pf_anchor_target_is_selected_only_and_normalized() -> None:
    teacher = torch.tensor([[3.0, 0.0, -1.0], [0.0, 2.0, -1.0]], requires_grad=True)
    anchor = torch.tensor([[0.0, 3.0, -1.0], [2.0, 0.0, -1.0]], requires_grad=True)
    output = AnchoredTeacherTargetPolicy()(
        teacher_logits=teacher,
        anchor_logits=anchor,
        risk=torch.tensor([1.0, 0.0]),
        temperature=1.0,
    )
    expected_teacher = torch.softmax(teacher.detach(), dim=1)
    expected_anchor = torch.softmax(anchor.detach(), dim=1)
    assert torch.allclose(output.probabilities[0], 0.75 * expected_teacher[0] + 0.25 * expected_anchor[0])
    assert torch.equal(output.probabilities[1], expected_teacher[1])
    assert torch.equal(output.rho, torch.tensor([0.25, 0.0]))
    assert not output.probabilities.requires_grad
    assert teacher.grad is None and anchor.grad is None


def test_pf_anchor_target_rejects_nonbinary_selection() -> None:
    with pytest.raises(ValueError, match="binary"):
        AnchoredTeacherTargetPolicy()(
            teacher_logits=torch.zeros(1, 2),
            anchor_logits=torch.zeros(1, 2),
            risk=torch.tensor([0.5]),
            temperature=1.0,
        )


class _ArtifactTracker(LocalTracker):
    """Small durable manifest fixture for fork-bound artifact tests."""

    def __init__(self, root: Path, *, run_id: str) -> None:
        self.run_id = run_id
        self.bundle_dir = root / "run-bundle"
        self.bundle_dir.mkdir(parents=True)
        self.manifest = {"artifacts": []}

    def log_artifact(self, path: Path, *, name: str, artifact_type: str, aliases: tuple[str, ...] = ()) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        local = self.bundle_dir / "artifacts" / name / digest
        local.mkdir(parents=True)
        shutil.copy2(path, local / path.name)
        self.manifest["artifacts"].append(
            {
                "name": name,
                "type": artifact_type,
                "aliases": list(aliases),
                "sha256": digest,
                "local_path": str(local.relative_to(self.bundle_dir)),
            }
        )


def _v3_artifact_config(tmp_path: Path, arm: str) -> SimpleNamespace:
    selector, mask, anchor = tmp_path / "selector.json", tmp_path / "mask.json", tmp_path / "anchor.pt"
    selector.write_text("selector", encoding="utf-8")
    mask.write_text("mask", encoding="utf-8")
    anchor.write_text("anchor", encoding="utf-8")
    return SimpleNamespace(
        prescriptive_v3=SimpleNamespace(
            arm=arm,
            parent=SimpleNamespace(epoch=79),
            selector_bundle_path=selector,
            selector_bundle_sha256=hashlib.sha256(selector.read_bytes()).hexdigest(),
            mask=SimpleNamespace(path=mask, sha256=hashlib.sha256(mask.read_bytes()).hexdigest()),
            anchor_checkpoint=anchor,
            anchor_checkpoint_sha256=hashlib.sha256(anchor.read_bytes()).hexdigest(),
        )
    )


@pytest.mark.parametrize(
    ("arm", "expected_suffixes"),
    (("PF_RET_H", ("selector", "mask", "anchor")), ("NR_PFX_R", ("selector", "mask"))),
)
def test_v3_initial_fork_artifacts_are_boundary_based_idempotent_and_route_conditional(
    tmp_path: Path, arm: str, expected_suffixes: tuple[str, ...]
) -> None:
    config = _v3_artifact_config(tmp_path, arm)
    lineage = {
        "kind": "prescriptive_v3_intervention_v1",
        "arm": arm,
        "parent_epoch": 79,
        "child_tracker_run_id": "v3-run",
    }
    initial, later = tmp_path / "initial.pt", tmp_path / "later.pt"
    torch.save({"epoch": 79, "tracker_run_id": "v3-run", "fork_lineage": lineage}, initial)
    torch.save({"epoch": 80, "tracker_run_id": "v3-run", "fork_lineage": lineage}, later)
    tracker = _ArtifactTracker(tmp_path / "initial", run_id="v3-run")

    def rank_zero(active: LocalTracker, *, phase: str, action: object) -> None:
        assert phase == "prescriptive v3 input artifacts"
        assert callable(action)
        action(active)

    _attach_prescriptive_v3_input_artifacts(tracker=tracker, config=config, resume=initial, coordinator=rank_zero)
    assert [entry["name"] for entry in tracker.manifest["artifacts"]] == [
        f"prescriptive-v3-{suffix}-v3-run" for suffix in expected_suffixes
    ]
    _attach_prescriptive_v3_input_artifacts(tracker=tracker, config=config, resume=initial, coordinator=rank_zero)
    _attach_prescriptive_v3_input_artifacts(tracker=tracker, config=config, resume=later, coordinator=rank_zero)
    assert len(tracker.manifest["artifacts"]) == len(expected_suffixes)
    absent = _ArtifactTracker(tmp_path / "later-absent", run_id="v3-run")
    with pytest.raises(TrackingError, match="later resume lacks exact"):
        _attach_prescriptive_v3_input_artifacts(tracker=absent, config=config, resume=later, coordinator=rank_zero)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _v3_mask_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, Path, dict[int, tuple[int, bool]]]:
    """Compact sparse-ID fixture; parent integrity is covered by schedule-control tests."""
    parent = {
        101: (0, True),
        102: (0, True),
        103: (0, True),
        104: (0, True),
        301: (0, False),
        302: (0, False),
        303: (0, False),
        304: (0, False),
    }
    payload = {"tracker_run_id": "online-run", "config_hash": "c" * 64}
    monkeypatch.setattr(
        prescriptive_v3,
        "_state",
        lambda _path: (parent, "p" * 64, "s" * 64, payload),
    )
    online = tmp_path / "online.parquet"
    feature = tmp_path / "feature.parquet"
    online_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for sample_id, (class_id, correct) in parent.items():
        # Within each epoch-34 route, the first sparse ID has the worst
        # frequency and EMA margin.  It must survive the epoch-79 refresh.
        high = sample_id in {101, 301}
        online_rows.append(
            {
                "sample_id": sample_id,
                "class_id": class_id,
                "anchor_epoch": 34,
                "current_robust_correct": correct,
                "robust_correct_frequency_inclusive": 0.0 if high else 1.0,
                "margin_ema": -1.0 if high else 1.0,
            }
        )
        feature_rows.append(
            {
                "sample_id": sample_id,
                "class_id": class_id,
                "epoch": 34,
                "namespace": "train",
                "teacher_adversarial_correct": True,
            }
        )
    _write_rows(online, online_rows)
    _write_rows(feature, feature_rows)
    identity = {
        "run_id": "online-run",
        "config_hash": "c" * 64,
        "teacher": {"checkpoint_sha256": "t" * 64},
        "dataset_identity": {"name": "cifar10", "split": "train"},
        "attack_identity": {"steps": 10, "epsilon": "8/255"},
    }
    online_lineage = tmp_path / "online-lineage.json"
    online_lineage.write_text(
        json.dumps(
            {
                "contract": "pre39_online_state_candidate_v1",
                "anchor_epoch": 34,
                "observations_sha256": prescriptive_v3.sha256_file(online),
                "feature_observations_sha256": prescriptive_v3.sha256_file(feature),
                **identity,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    feature_lineage = tmp_path / "feature-lineage.json"
    feature_lineage.write_text(
        json.dumps(
            {
                "observation_schema_version": 2,
                "feature_observations_sha256": prescriptive_v3.sha256_file(feature),
                **identity,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return online, online_lineage, feature, feature_lineage, tmp_path / "parent.pt", parent


def test_v3_midrank_and_masks_use_sparse_ids_epoch34_rank_epoch79_intersection_and_exact_strata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ranks = prescriptive_v3._midrank({900: 3.0, 10: 1.0, 50: 1.0})
    assert ranks[10] == ranks[50] == pytest.approx(1.0 / 3.0)
    assert ranks[900] == pytest.approx(5.0 / 6.0)
    online, online_lineage, feature, feature_lineage, parent_path, parent = _v3_mask_inputs(tmp_path, monkeypatch)
    created = prescriptive_v3.build_prescriptive_v3_masks(
        online_observations=online,
        online_lineage=online_lineage,
        feature_observations=feature,
        feature_lineage=feature_lineage,
        parent_checkpoint=parent_path,
        output_dir=tmp_path / "one",
    )
    repeated = prescriptive_v3.build_prescriptive_v3_masks(
        online_observations=online,
        online_lineage=online_lineage,
        feature_observations=feature,
        feature_lineage=feature_lineage,
        parent_checkpoint=parent_path,
        output_dir=tmp_path / "two",
    )
    assert created["bundle"].read_bytes() == repeated["bundle"].read_bytes()
    bundle = json.loads(created["bundle"].read_text(encoding="utf-8"))
    assert bundle["routes"]["PF_RET"]["history"] == [101]
    assert bundle["routes"]["NR_PFX"]["history"] == [301]
    for route, expected_state in (("PF_RET", True), ("NR_PFX", False)):
        value = bundle["routes"][route]
        history, random = value["history"], value["random"]
        assert len(history) == len(random) == 1
        assert set(history).isdisjoint(random)
        assert all(parent[sample_id][1] is expected_state for sample_id in history + random)
        history_mask = json.loads(created[f"{route}_H"].read_text(encoding="utf-8"))
        random_mask = json.loads(created[f"{route}_R"].read_text(encoding="utf-8"))
        assert history_mask["selected_count"] == random_mask["selected_count"]
        assert history_mask["selected_class_counts"] == random_mask["selected_class_counts"]
        assert history_mask["selected_ids_sha256"] == selected_ids_sha256(tuple(history))
    with pytest.raises(FileExistsError, match="overwrite"):
        prescriptive_v3.build_prescriptive_v3_masks(
            online_observations=online,
            online_lineage=online_lineage,
            feature_observations=feature,
            feature_lineage=feature_lineage,
            parent_checkpoint=parent_path,
            output_dir=tmp_path / "one",
        )


@pytest.mark.parametrize(
    ("target", "key", "value", "message"),
    (
        ("online", "attack_identity", {"steps": 9}, "identity drifted"),
        ("feature", "teacher", {"checkpoint_sha256": "x" * 64}, "identity drifted"),
    ),
)
def test_v3_masks_reject_online_feature_lineage_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    key: str,
    value: object,
    message: str,
) -> None:
    online, online_lineage, feature, feature_lineage, parent_path, _ = _v3_mask_inputs(tmp_path, monkeypatch)
    path = online_lineage if target == "online" else feature_lineage
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw[key] = value
    path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    with pytest.raises(prescriptive_v3.PrescriptiveV3Error, match=message):
        prescriptive_v3.build_prescriptive_v3_masks(
            online_observations=online,
            online_lineage=online_lineage,
            feature_observations=feature,
            feature_lineage=feature_lineage,
            parent_checkpoint=parent_path,
            output_dir=tmp_path / "out",
        )


@pytest.mark.parametrize("which", ("online", "feature"))
def test_v3_masks_reject_duplicate_sparse_source_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    online, online_lineage, feature, feature_lineage, parent_path, _ = _v3_mask_inputs(tmp_path, monkeypatch)
    path = online if which == "online" else feature
    rows = pq.read_table(path).to_pylist()
    rows.append(dict(rows[0]))
    _write_rows(path, rows)
    # Reseal the relevant content hash to ensure duplicate rejection is a
    # stable-ID contract, rather than merely a stale-file-hash failure.
    if which == "online":
        raw = json.loads(online_lineage.read_text(encoding="utf-8"))
        raw["observations_sha256"] = prescriptive_v3.sha256_file(online)
        online_lineage.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    else:
        for lineage_path, key in (
            (feature_lineage, "feature_observations_sha256"),
            (online_lineage, "feature_observations_sha256"),
        ):
            raw = json.loads(lineage_path.read_text(encoding="utf-8"))
            raw[key] = prescriptive_v3.sha256_file(feature)
            lineage_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    with pytest.raises(prescriptive_v3.PrescriptiveV3Error, match="sparse-ID/class join drifted"):
        prescriptive_v3.build_prescriptive_v3_masks(
            online_observations=online,
            online_lineage=online_lineage,
            feature_observations=feature,
            feature_lineage=feature_lineage,
            parent_checkpoint=parent_path,
            output_dir=tmp_path / "out",
        )


def _write_v3_config_inputs(tmp_path: Path, fields: dict[str, object]) -> tuple[Path, Path]:
    """Write exact H/R mask/bundle inputs for the schedule-control fixture."""
    masks = tmp_path / "masks"
    masks.mkdir()
    parent_sha = str(fields["checkpoint_sha256"])
    state_sha = str(fields["sample_state_sha256"])
    bundle = tmp_path / "selector.json"
    selected = {
        "PF_RET_H": [1],
        "PF_RET_R": [2],
        "NR_PFX_H": [3],
        "NR_PFX_R": [4],
    }
    bundle_payload = {
        "schema_version": 1,
        "kind": "prescriptive_v3_epoch34_online_epoch79_route_v1",
        "parent": {"checkpoint_sha256": parent_sha, "sample_state_sha256": state_sha, "epoch": 79},
        "routes": {
            "PF_RET": {
                "history_ids_sha256": selected_ids_sha256(tuple(selected["PF_RET_H"])),
                "random_ids_sha256": selected_ids_sha256(tuple(selected["PF_RET_R"])),
            },
            "NR_PFX": {
                "history_ids_sha256": selected_ids_sha256(tuple(selected["NR_PFX_H"])),
                "random_ids_sha256": selected_ids_sha256(tuple(selected["NR_PFX_R"])),
            },
        },
    }
    bundle.write_text(json.dumps(bundle_payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    bundle_sha = sha256_file(bundle)
    history_paths: dict[str, Path] = {}
    for arm, ids in selected.items():
        route = "peak_failure" if arm.startswith("PF_") else "non_recovery"
        current = arm.startswith("PF_")
        source = "prescriptive_v3_online_history" if arm.endswith("_H") else "prescriptive_v3_matched_random"
        provenance: dict[str, object] = {
            "source": source,
            "parent_checkpoint_sha256": parent_sha,
            "parent_sample_state_sha256": state_sha,
            "route": route,
            "anchor_robust_correct": current,
        }
        if arm.endswith("_H"):
            provenance.update(
                {"approved_selector_spec_sha256": bundle_sha, "selector_spec_path": str(bundle.resolve())}
            )
        else:
            history_arm = arm[:-1] + "H"
            history_path = history_paths[history_arm]
            history_payload = json.loads(history_path.read_text(encoding="utf-8"))
            provenance.update(
                {
                    "random_seed": 0,
                    "generator": "sha256",
                    "generator_version": "prescriptive-v3-v1",
                    "reference_history_mask_sha256": sha256_file(history_path),
                    "reference_selected_count": len(selected[history_arm]),
                    "reference_selected_class_counts": history_payload["selected_class_counts"],
                    "reference_history_selector_spec_sha256": bundle_sha,
                }
            )
        payload = {
            "schema_version": 1,
            "namespace": "train",
            "num_classes": 10,
            "selected_ids": ids,
            "selected_ids_sha256": selected_ids_sha256(tuple(ids)),
            "selected_count": len(ids),
            "selected_class_counts": {"0": len(ids)},
            "provenance": provenance,
        }
        route_key = "pf_ret" if arm.startswith("PF_") else "nr_pfx"
        suffix = "h" if arm.endswith("_H") else "r"
        path = masks / f"{route_key}_{suffix}.json"
        path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        if arm.endswith("_H"):
            history_paths[arm] = path
    return bundle, masks


def test_v3_generates_four_strict_arms_and_atomically_forks_exact_epoch79_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, resolved, manifest, inventory, attestation, _, fields = _inputs(tmp_path)
    delayed_config = _child_config(tmp_path, _raw_config(tmp_path), fields)
    spec = _spec(tmp_path, fields)
    bundle, masks = _write_v3_config_inputs(tmp_path, fields)
    config_dir, output_root = tmp_path / "configs", tmp_path / "screen"
    arm_paths = prescriptive_v3.write_prescriptive_v3_arm_configs(
        delayed_config=delayed_config,
        schedule_spec=spec,
        parent_checkpoint=checkpoint,
        selector_bundle=bundle,
        masks_dir=masks,
        config_dir=config_dir,
        output_root=output_root,
        run_prefix="screen",
    )
    assert tuple(arm_paths) == prescriptive_v3.ARMS
    arms = {name: load_config(path) for name, path in arm_paths.items()}
    assert all(arm.protocol.id == "controlled_cifar10_r18_prescriptive_v3_v1" for arm in arms.values())
    assert all(arm.scheduler.milestones == (120, 170) and arm.intervention is None for arm in arms.values())
    assert {arm.tracking.run_id for arm in arms.values()} == {f"screen-{name.lower()}" for name in arms}
    assert all(arm.prescriptive_v3 is not None for arm in arms.values())
    assert {arm.prescriptive_v3.arm for arm in arms.values() if arm.prescriptive_v3 is not None} == set(
        prescriptive_v3.ARMS
    )

    # Forks must refuse a dirty launch before a screen directory is created.
    with pytest.raises(prescriptive_v3.PrescriptiveV3Error, match="clean addressable"):
        prescriptive_v3.create_prescriptive_v3_forks(
            parent_checkpoint=checkpoint,
            parent_resolved_config=resolved,
            parent_manifest=manifest,
            artifact_inventory=inventory,
            artifact_attestation=attestation,
            schedule_spec=spec,
            arm_config_paths=list(arm_paths.values()),
            root=Path.cwd(),
            git_state_collector=lambda _: {"sha": "b" * 40, "dirty": True},
        )
    assert not output_root.exists()
    created = prescriptive_v3.create_prescriptive_v3_forks(
        parent_checkpoint=checkpoint,
        parent_resolved_config=resolved,
        parent_manifest=manifest,
        artifact_inventory=inventory,
        artifact_attestation=attestation,
        schedule_spec=spec,
        arm_config_paths=list(arm_paths.values()),
        root=Path.cwd(),
        git_state_collector=lambda _: {"sha": "b" * 40, "dirty": False},
    )
    assert set(created) == set(prescriptive_v3.ARMS)
    children = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in created.items()}
    parent = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert {child["config_hash"] for child in children.values()} == {
        config_digest(resolved_config_dict(arm)) for arm in arms.values()
    }
    assert len({child["tracker_run_id"] for child in children.values()}) == 4
    assert all(child["tracker_run_id"] != parent["tracker_run_id"] for child in children.values())
    for name, child in children.items():
        assert child["scheduler"]["milestones"] == {120: 1, 170: 1}
        assert child["best_metric"] == float("-inf")
        assert child["selection_metadata"]["scope"] == "post_fork_best"
        lineage = child["fork_lineage"]
        assert lineage["kind"] == "prescriptive_v3_intervention_v1"
        assert lineage["arm"] == name
        assert lineage["parent_checkpoint_sha256"] == fields["checkpoint_sha256"]
        assert lineage["parent_sample_state_sha256"] == fields["sample_state_sha256"]
        assert lineage["parent_scheduler"]["milestones"] == [100, 150]
        assert lineage["child_scheduler"]["milestones"] == [120, 170]
        for key in (
            "model",
            "optimizer",
            "scaler",
            "rng",
            "sampler_epoch",
            "sampler_state",
            "sample_state",
            "global_step",
        ):
            assert _equal(child[key], parent[key]), key
    monkeypatch.setattr(train_cli, "collect_git_state", lambda _: {"sha": "b" * 40, "dirty": False})
    first_name = "PF_RET_H"
    first_config = arms[first_name]
    first_hash = config_digest(resolved_config_dict(first_config))
    train_cli._validate_intervention_resume(created[first_name], first_config, config_hash=first_hash)
    # The initial fork is immutable evidence; changing one byte must fail
    # before it can be launched.  Later epoch-boundary resumes are mutable.
    with created[first_name].open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="initial screen fork checkpoint bytes"):
        train_cli._validate_intervention_resume(created[first_name], first_config, config_hash=first_hash)
    with pytest.raises(prescriptive_v3.PrescriptiveV3Error, match="outputs must be distinct and absent"):
        prescriptive_v3.create_prescriptive_v3_forks(
            parent_checkpoint=checkpoint,
            parent_resolved_config=resolved,
            parent_manifest=manifest,
            artifact_inventory=inventory,
            artifact_attestation=attestation,
            schedule_spec=spec,
            arm_config_paths=list(arm_paths.values()),
            root=Path.cwd(),
            git_state_collector=lambda _: {"sha": "b" * 40, "dirty": False},
        )


class _PrefixProbeAttack(AttackGenerator):
    requires_teacher_clean_target = True

    def __init__(self) -> None:
        self.capture_steps: list[int | None] = []

    def generate(self, request: AttackRequest) -> AttackResult:
        self.capture_steps.append(request.capture_step)
        final = request.inputs.detach() + 0.2
        prefix = request.inputs.detach() + 0.1 if request.capture_step is not None else None
        return AttackResult(
            adversarial=final,
            initial_delta=torch.zeros_like(final),
            step_losses=(),
            max_abs_delta=0.2,
            captured_adversarial=prefix,
        )


class _InputSpy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(3 * 4 * 4, 3)
        self.inputs: list[torch.Tensor] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.inputs.append(value.detach().clone())
        return self.layer(value.flatten(1))


def test_v3_trainer_epoch_boundaries_keep_pf_inactive_and_use_nr_prefix_only_for_selected_samples(
    tmp_path: Path,
) -> None:
    mask = FixedInterventionMask(frozenset({101}), "m" * 64, {0: 1})
    student, teacher, anchor = _InputSpy(), _InputSpy(), _InputSpy()
    pf = Trainer(
        model=student,
        optimizer=torch.optim.SGD(student.parameters(), lr=0.1),
        scheduler=None,
        scaler=None,
        attack=_PrefixProbeAttack(),
        selection_attack=_PrefixProbeAttack(),
        objective=RSLADObjective(),
        device=torch.device("cpu"),
        output_dir=tmp_path / "pf",
        config_hash="c" * 64,
        seed=7,
        teacher=teacher,
        policy=RSLADBaselinePolicy(),
        intervention_mask=mask,
        anchor_model=anchor,
        target_policy=AnchoredTeacherTargetPolicy(),
        prescriptive_v3_route="pf_retention",
    )
    assert [pf._prescriptive_active() for pf.current_epoch in (79, 80, 129, 130)] == [False, True, True, False]
    assert not anchor.training and all(
        not parameter.requires_grad and parameter.grad is None for parameter in anchor.parameters()
    )
    pf_batch = IndexedBatch(
        images=torch.full((2, 3, 4, 4), 0.2),
        labels=torch.tensor([0, 1]),
        sample_ids=torch.tensor([101, 999]),
        state_update_mask=torch.tensor([True, True]),
    )
    pf.current_epoch = 79
    pf.train_epoch([pf_batch])
    assert anchor.inputs == []  # inactive PF is exact ordinary-teacher target construction
    pf.current_epoch = 80
    pf.train_epoch([pf_batch])
    assert len(anchor.inputs) == 1
    torch.testing.assert_close(anchor.inputs[0], pf_batch.images, rtol=0, atol=0)

    attack = _PrefixProbeAttack()
    nr_student, nr_teacher = _InputSpy(), _InputSpy()
    nr = Trainer(
        model=nr_student,
        optimizer=torch.optim.SGD(nr_student.parameters(), lr=0.1),
        scheduler=None,
        scaler=None,
        attack=attack,
        selection_attack=_PrefixProbeAttack(),
        objective=RSLADObjective(),
        device=torch.device("cpu"),
        output_dir=tmp_path / "nr",
        config_hash="c" * 64,
        seed=7,
        teacher=nr_teacher,
        policy=RSLADBaselinePolicy(),
        intervention_mask=mask,
        prescriptive_v3_route="nr_prefix",
    )
    batch = IndexedBatch(
        images=torch.full((2, 3, 4, 4), 0.2),
        labels=torch.tensor([0, 1]),
        sample_ids=torch.tensor([101, 999]),
        state_update_mask=torch.tensor([True, True]),
    )
    nr.current_epoch = 80
    nr.train_epoch([batch])
    assert attack.capture_steps == [5]
    torch.testing.assert_close(nr_student.inputs[0][0], batch.images[0] + 0.1, rtol=0, atol=0)
    torch.testing.assert_close(nr_student.inputs[0][1], batch.images[1] + 0.2, rtol=0, atol=0)
    nr_student.inputs.clear()
    nr.current_epoch = 100
    nr.train_epoch([batch])
    assert attack.capture_steps == [5, None]
    torch.testing.assert_close(nr_student.inputs[0], batch.images + 0.2, rtol=0, atol=0)


class _ExactAttack(AttackGenerator):
    requires_teacher_clean_target = True

    def __init__(self) -> None:
        self.targets: list[torch.Tensor] = []

    def generate(self, request: AttackRequest) -> AttackResult:
        if request.target_logits is not None:
            self.targets.append(request.target_logits.detach().clone())
        return AttackResult(
            adversarial=request.inputs.detach().clone(),
            initial_delta=torch.zeros_like(request.inputs),
            step_losses=(),
            max_abs_delta=0.0,
            captured_adversarial=request.inputs.detach().clone() if request.capture_step is not None else None,
        )


class _BNStudent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(48)
        self.fc = torch.nn.Linear(48, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bn(value.flatten(1)))


def _exact_trainer(
    tmp_path: Path, *, route: str | None, with_store: bool = False
) -> tuple[Trainer, _ExactAttack, _BNStudent, _BNStudent, _BNStudent | None]:
    torch.manual_seed(811)
    student, teacher = _BNStudent(), _BNStudent()
    anchor = _BNStudent() if route == "pf_retention" else None
    attack = _ExactAttack()
    mask = FixedInterventionMask(frozenset({1}), "x" * 64, {0: 1}) if route else None
    optimizer = torch.optim.SGD(student.parameters(), lr=0.03, momentum=0.9)
    trainer = Trainer(
        model=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 170]),
        scaler=None,
        attack=attack,
        selection_attack=attack,
        objective=RSLADObjective(),
        device=torch.device("cpu"), output_dir=tmp_path, config_hash="p" * 64, seed=4,
        policy=RSLADBaselinePolicy(), intervention_mask=mask, anchor_model=anchor,
        target_policy=AnchoredTeacherTargetPolicy() if route == "pf_retention" else None,
        prescriptive_v3_route=route,
        sample_store=SampleStateStore(ema_decay=0.9) if with_store else None,
    )
    return trainer, attack, student, teacher, anchor


@pytest.mark.parametrize("route", ("pf_retention", "nr_prefix"))
def test_v3_inactive_paths_are_bitwise_delayed_rslad_parity(tmp_path: Path, route: str) -> None:
    baseline, _, base_student, _, _ = _exact_trainer(tmp_path / "base", route=None)
    v3, _, v3_student, _, _ = _exact_trainer(tmp_path / "v3", route=route)
    baseline.current_epoch = v3.current_epoch = 79
    batch = IndexedBatch(torch.full((3, 3, 4, 4), 0.2), torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    left = baseline.train_epoch([batch])
    left_rng = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    right = v3.train_epoch([batch])
    right_rng = torch.get_rng_state().clone()
    assert {key: value for key, value in left.items() if key not in {"seconds", "images_per_second"}} == {
        key: value for key, value in right.items() if key not in {"seconds", "images_per_second"}
    }
    assert torch.equal(left_rng, right_rng)
    for left_value, right_value in zip(base_student.state_dict().values(), v3_student.state_dict().values()):
        assert torch.equal(left_value, right_value)
    for left_parameter, right_parameter in zip(base_student.parameters(), v3_student.parameters()):
        assert left_parameter.grad is not None and right_parameter.grad is not None
        assert torch.equal(left_parameter.grad, right_parameter.grad)


def test_v3_active_pf_changes_only_adv_target_and_keeps_anchor_frozen(tmp_path: Path) -> None:
    baseline, base_attack, _, base_teacher, _ = _exact_trainer(tmp_path / "base", route=None)
    v3, v3_attack, _, v3_teacher, anchor = _exact_trainer(tmp_path / "v3", route="pf_retention")
    assert anchor is not None
    v3.current_epoch = baseline.current_epoch = 80
    batch = IndexedBatch(torch.full((3, 3, 4, 4), 0.2), torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
    bn_before = anchor.bn.running_mean.clone(), anchor.bn.running_var.clone(), anchor.bn.num_batches_tracked.clone()
    torch.manual_seed(99)
    baseline.train_epoch([batch])
    torch.manual_seed(99)
    v3.train_epoch([batch])
    assert torch.equal(base_attack.targets[0], v3_attack.targets[0])  # unchanged inner teacher-clean PGD target
    assert all(parameter.grad is None for parameter in base_teacher.parameters())
    assert all(parameter.grad is None for parameter in v3_teacher.parameters())
    assert all(parameter.grad is None for parameter in anchor.parameters())
    assert torch.equal(anchor.bn.running_mean, bn_before[0])
    assert torch.equal(anchor.bn.running_var, bn_before[1])
    assert torch.equal(anchor.bn.num_batches_tracked, bn_before[2])


def _exact_loaders() -> tuple[DataLoader[IndexedBatch], DataLoader[IndexedBatch], EpochShuffleSampler]:
    dataset = IndexedDataset(SyntheticCIFAR(size=6, num_classes=3, image_size=4, seed=19))
    sampler = EpochShuffleSampler(len(dataset), seed=23)
    validation_sampler = EpochShuffleSampler(len(dataset), seed=29, shuffle=False)
    loader = DataLoader(dataset, batch_size=3, sampler=sampler, collate_fn=collate_indexed)
    validation = DataLoader(
        dataset,
        batch_size=3,
        sampler=validation_sampler,
        collate_fn=collate_indexed,
    )
    return loader, validation, sampler


def test_v3_nr_resume_across_epoch99_boundary_matches_uninterrupted_exactly(tmp_path: Path) -> None:
    """Checkpoint/resume must not alter the active-99 to inactive-100 transition."""
    uninterrupted, _, _, _, _ = _exact_trainer(
        tmp_path / "uninterrupted", route="nr_prefix", with_store=True
    )
    full_loader, full_validation, _ = _exact_loaders()
    full_history = uninterrupted.fit(
        full_loader,
        validation_loader=full_validation,
        epochs=101,
        start_epoch=99,
    )
    full_rng = torch.get_rng_state().clone()

    first_leg, _, _, _, _ = _exact_trainer(tmp_path / "resumed", route="nr_prefix", with_store=True)
    first_loader, first_validation, _ = _exact_loaders()
    first_history = first_leg.fit(
        first_loader,
        validation_loader=first_validation,
        epochs=100,
        start_epoch=99,
    )

    resumed, _, _, _, _ = _exact_trainer(tmp_path / "resumed", route="nr_prefix", with_store=True)
    resumed_loader, resumed_validation, resumed_sampler = _exact_loaders()
    state = resumed.resume(tmp_path / "resumed" / "last.pt", sampler=resumed_sampler)
    assert state.next_epoch == 100
    second_history = resumed.fit(
        resumed_loader,
        validation_loader=resumed_validation,
        epochs=101,
        start_epoch=state.next_epoch,
    )
    resumed_rng = torch.get_rng_state().clone()

    assert _equal(uninterrupted.model.state_dict(), resumed.model.state_dict())
    assert _equal(uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert uninterrupted.scheduler.state_dict() == resumed.scheduler.state_dict()
    assert uninterrupted.sample_state == resumed.sample_state
    assert uninterrupted.global_step == resumed.global_step
    assert uninterrupted.best_metric == resumed.best_metric
    assert uninterrupted.selection_metadata == resumed.selection_metadata
    assert torch.equal(full_rng, resumed_rng)
    deterministic = (
        "train_loss",
        "train_clean_accuracy",
        "train_robust_accuracy",
        "train_valid_examples",
        "train_teacher_clean_forward_calls",
        "train_teacher_adversarial_forward_calls",
        "val_clean_accuracy",
        "val_pgd_accuracy",
    )
    for expected, actual in zip(full_history, first_history + second_history, strict=True):
        assert {key: expected[key] for key in deterministic} == {key: actual[key] for key in deterministic}
