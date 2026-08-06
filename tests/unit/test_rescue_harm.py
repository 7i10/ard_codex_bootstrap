from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ard.analysis.rescue_harm import (
    EPOCHS,
    SOURCE_ARM,
    V3_ARMS,
    V3_EPOCHS,
    V3_OBSERVATION_COLUMNS,
    RescueHarmError,
    _load_v3_inventory,
    _teacher_response_kl,
    build_v3_checkpoint_inventory,
    load_checkpoint_inventory,
    merge_epoch_replays,
    merge_v3_epoch_replays,
    replay_v3_inventory,
    report_rescue_harm,
    report_v3_rescue_harm,
    smoke_v3_report,
)
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS
from ard.analysis.signal_audit import CheckpointInventory, sha256_file
from ard.cli.rescue_harm import main as rescue_harm_main
from ard.engine.checkpoint import REQUIRED_KEYS
from ard.state.sample_store import SampleRecord, SampleStateStore

pytestmark = pytest.mark.t1


def test_rescue_harm_inventory_cli_prints_its_contract(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "ard.cli.rescue_harm.build_checkpoint_inventory",
        lambda **_: {"schema_version": 1},
    )
    assert (
        rescue_harm_main(
            [
                "inventory",
                "--manifest",
                "manifest.json",
                "--resolved-config",
                "config.yaml",
                "--arm",
                "control",
                "--seed",
                "1",
                "--output",
                "inventory.json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"contract": "completed_v2_checkpoint_inventory_v1"}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(epoch: int, *, run_id: str, config_hash: str) -> dict[str, object]:
    payload = {key: {} for key in REQUIRED_KEYS}
    payload.update({"epoch": epoch, "config_hash": config_hash, "tracker_run_id": run_id})
    return payload


@pytest.mark.parametrize("arm", ["control", "PF_TA", "PF_R", "NR_TA", "NR_R"])
def test_rescue_harm_inventory_accepts_real_arms_and_rejects_payload_epoch_drift(tmp_path: Path, arm: str) -> None:
    run_id, config_hash = "run", "a" * 64
    entries = []
    for epoch in EPOCHS:
        path = tmp_path / f"{epoch}.pt"
        torch.save(_checkpoint(epoch, run_id=run_id, config_hash=config_hash), path)
        entries.append({"epoch": epoch, "path": str(path), "sha256": _digest(path), "scientific_git_sha": "b" * 40})
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "arm": arm,
                "seed": 1,
                "teacher": {"registry_id": "t"},
                "config_hash": config_hash,
                "checkpoints": entries,
            }
        ),
        encoding="utf-8",
    )
    assert [item.epoch for item in load_checkpoint_inventory(inventory).checkpoints] == list(EPOCHS)
    changed = json.loads(inventory.read_text(encoding="utf-8"))
    changed["checkpoints"][0]["epoch"] = 98
    inventory.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RescueHarmError, match="epoch/config/run"):
        load_checkpoint_inventory(inventory)


def _teacher(prefix: str, *, class_id: int, correct: bool, probability: float) -> dict[str, object]:
    prediction = class_id if correct else (class_id + 1) % 10
    wrong = 1 - probability
    return {
        f"{prefix}_prediction": prediction,
        f"{prefix}_correct": correct,
        f"{prefix}_true_probability": probability,
        f"{prefix}_max_wrong_probability": wrong,
        f"{prefix}_wrong_confidence": wrong - probability,
        f"{prefix}_probability_margin": probability - wrong,
        f"{prefix}_entropy_normalized": 0.4,
    }


def _feature_row(sample_id: int, epoch: int) -> dict[str, object]:
    class_id, probability = sample_id % 10, 0.2 + sample_id / 100
    clean = _teacher("teacher_clean", class_id=class_id, correct=sample_id % 2 == 0, probability=probability)
    adversarial = _teacher(
        "teacher_adversarial", class_id=class_id, correct=sample_id % 3 != 0, probability=probability
    )
    return {
        "namespace": "train",
        "sample_id": sample_id,
        "class_id": class_id,
        "epoch": epoch,
        "observation_schema_version": 2,
        "teacher_entropy_normalized": 0.4,
        "student_probability_margin": 0.5,
        "student_margin_risk": 0.25,
        "robust_correct": sample_id not in {21, 34},
        **clean,
        **adversarial,
        "teacher_clean_to_adversarial_prediction_flip": clean["teacher_clean_prediction"]
        != adversarial["teacher_adversarial_prediction"],
        "teacher_clean_to_adversarial_true_probability_delta": 0.0,
        "teacher_clean_to_adversarial_margin_delta": 0.0,
        "student_clean_prediction": class_id,
        "student_clean_correct": True,
        "student_clean_probability_margin": 0.5,
    }


def _write_feature(tmp_path: Path, ids: tuple[int, ...]) -> tuple[Path, Path]:
    observations, lineage = tmp_path / "feature.parquet", tmp_path / "feature.json"
    pq.write_table(
        pa.Table.from_pylist([_feature_row(sample_id, epoch) for epoch in FEATURE_EPOCHS for sample_id in ids]),
        observations,
    )
    lineage.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observation_schema_version": 2,
                "run_id": "run",
                "config_hash": "a" * 64,
                "scientific_git_sha": "b" * 40,
                "train_expected_count": len(ids),
                "feature_observations_sha256": sha256_file(observations),
                "attack_identity": {"steps": 20},
                "dataset_identity": {"name": "cifar10"},
                "teacher": {"registry_id": "teacher", "checkpoint_sha256": "e" * 64},
                "feature_protocol": {"domain": "feature"},
            }
        ),
        encoding="utf-8",
    )
    return observations, lineage


def _observation_row(*, arm: str, epoch: int, sample_id: int, correct: bool) -> dict[str, object]:
    return {
        "namespace": "train",
        "run_id": "run",
        "arm": arm,
        "seed": 7,
        "epoch": epoch,
        "sample_id": sample_id,
        "class_id": sample_id % 10,
        "clean_prediction": sample_id % 10,
        "clean_correct": True,
        "clean_probability_margin": 0.6,
        "robust_prediction": sample_id % 10 if correct else (sample_id + 1) % 10,
        "robust_correct": correct,
        "robust_probability_margin": 0.4 if correct else -0.4,
    }


def _write_arm(tmp_path: Path, arm: str, ids: tuple[int, ...], *, control: bool) -> tuple[Path, Path]:
    source_arm = SOURCE_ARM[arm]
    observations, lineage = tmp_path / f"{source_arm}.parquet", tmp_path / f"{source_arm}.json"
    rows = []
    for epoch in EPOCHS:
        for sample_id in ids:
            base = sample_id in {2, 8}
            correct = (
                base if control else ({2: True, 5: True, 8: False}.get(sample_id, base) if arm == "PF_H" else base)
            )
            rows.append(_observation_row(arm=source_arm, epoch=epoch, sample_id=sample_id, correct=correct))
    pq.write_table(pa.Table.from_pylist(rows), observations)
    lineage.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "completed_v2_rescue_harm_replay_v1",
                "observations_sha256": sha256_file(observations),
                "run_id": "run",
                "arm": source_arm,
                "seed": 7,
                "teacher": {"registry_id": "teacher", "checkpoint_sha256": "e" * 64},
                "config_sha256": "a" * 64,
                "dataset_identity": {"name": "cifar10"},
                "attack_identity": {"loss": "ce", "steps": 20},
                "analysis_seed": 99,
                "student_identity": {"architecture": "resnet18"},
                "row_count": len(rows),
            }
        ),
        encoding="utf-8",
    )
    return observations, lineage


def _parent_checkpoint(tmp_path: Path, ids: tuple[int, ...]) -> tuple[Path, str, str]:
    store = SampleStateStore(ema_decay=0.9)
    # Deliberately unlike `_feature_row` robustness: the selector's epoch-39
    # eligibility is the online state, never the common-PGD replay result.
    online_correct = {2, 5, 21, 34}
    store.records = {
        sample_id: SampleRecord(
            margin_ema=0.2,
            seen=40,
            robust_correct_count=20,
            previous_robust_correct=sample_id in online_correct,
            forgetting_count=0,
            last_update=39,
            last_margin=0.2,
            true_label=sample_id % 10,
        )
        for sample_id in ids
    }
    payload = {key: {} for key in REQUIRED_KEYS}
    payload.update({"epoch": 39, "epoch_boundary": "end", "sample_state": store.state_dict()})
    checkpoint = tmp_path / "parent-epoch39.pt"
    torch.save(payload, checkpoint)
    return (
        checkpoint,
        _digest(checkpoint),
        hashlib.sha256(
            json.dumps(payload["sample_state"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
    )


def _report_inputs(tmp_path: Path) -> tuple[dict[str, tuple[Path, Path]], Path, Path, Path, Path]:
    ids = (2, 5, 8, 13, 21, 34)
    observations = {"control": _write_arm(tmp_path, "control", ids, control=True)}
    observations.update(
        {arm: _write_arm(tmp_path, arm, ids, control=False) for arm in ("PF_H", "PF_R", "NR_H", "NR_R")}
    )
    parent_checkpoint, checkpoint_sha, state_sha = _parent_checkpoint(tmp_path, ids)
    masks = tmp_path / "history-routing-v2-bundle.json"
    parent = {"checkpoint_sha256": checkpoint_sha, "sample_state_sha256": state_sha, "epoch": 39}
    paths = {
        "peak_failure": {"history": "/remote/pf-history.json", "random": "/remote/pf-random.json"},
        "non_recovery": {"history": "/remote/nr-history.json", "random": "/remote/nr-random.json"},
    }
    selection = {
        "peak_failure": {"selected_count": 2, "selected_class_counts": {"2": 1, "5": 1}, "eligible_count": 4},
        "non_recovery": {"selected_count": 1, "selected_class_counts": {"8": 1}, "eligible_count": 2},
    }
    masks.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "history_routing_v2_online_selector_v1",
                "parent": parent,
                "selection": selection,
                "mask_paths": paths,
            }
        ),
        encoding="utf-8",
    )
    bundle_sha = sha256_file(masks)
    mask_specs = {
        "pf-history.json": ([2, 5], "peak_failure", "history"),
        "pf-random.json": ([2, 5], "peak_failure", "random"),
        "nr-history.json": ([8], "non_recovery", "history"),
        "nr-random.json": ([8], "non_recovery", "random"),
    }
    labels = {sample_id: sample_id % 10 for sample_id in ids}
    for name, (selected, route, kind) in mask_specs.items():
        counts = {
            str(labels[sample_id]): sum(labels[item] == labels[sample_id] for item in selected)
            for sample_id in selected
        }
        provenance = {
            "parent_checkpoint_sha256": parent["checkpoint_sha256"],
            "parent_sample_state_sha256": parent["sample_state_sha256"],
            "route": route,
            "anchor_robust_correct": route == "peak_failure",
        }
        provenance[
            "approved_selector_spec_sha256" if kind == "history" else "reference_history_selector_spec_sha256"
        ] = bundle_sha
        (tmp_path / name).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "namespace": "train",
                    "num_classes": 10,
                    "selected_ids": selected,
                    "selected_ids_sha256": hashlib.sha256(
                        json.dumps(selected, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "selected_count": len(selected),
                    "selected_class_counts": counts,
                    "provenance": provenance,
                }
            ),
            encoding="utf-8",
        )
    feature, feature_lineage = _write_feature(tmp_path, ids)
    return observations, masks, feature, feature_lineage, parent_checkpoint


def test_rescue_harm_report_categories_spillover_mix_formula_and_row_order(tmp_path: Path) -> None:
    observations, masks, feature, feature_lineage, parent_checkpoint = _report_inputs(tmp_path)
    # Delayed children have distinct resolved-config hashes; paired replay
    # identity is instead student/data/attack/teacher/seed based.
    child_lineage = json.loads(observations["PF_H"][1].read_text(encoding="utf-8"))
    child_lineage["config_sha256"] = "f" * 64
    observations["PF_H"][1].write_text(json.dumps(child_lineage), encoding="utf-8")
    result = report_rescue_harm(
        observations=observations,
        mask_bundle=masks,
        feature_observations=feature,
        feature_lineage=feature_lineage,
        parent_checkpoint=parent_checkpoint,
        output=tmp_path / "report.json",
        expected_count=6,
    )
    pf = result["epochs_report"]["99"]["PF_H"]
    assert pf["categories"]["all"]["categories"] == {
        "rescued": 1,
        "harmed": 1,
        "stable_correct": 1,
        "unchanged_failure": 3,
    }
    assert pf["categories"]["all"]["net_rescue"] == 0
    assert pf["categories"]["selected"]["count"] == 2
    assert pf["categories"]["non_selected"]["count"] == 4
    # Online PF contains 21/34 while the fixed replay calls those samples
    # wrong; the selector domain must remain the settled parent state.
    assert pf["categories"]["eligible"]["count"] == 4
    assert result["input_identity"]["eligibility_domain"] == "epoch39_parent_sample_state.previous_robust_correct"
    assert result["input_identity"]["outcome_domain"] == "fixed_checkpoint_common_ce_pgd20_replay.robust_correct"
    assert pf["true_label_mix_l1_distance"]["selected"]["mean"] == pytest.approx(((1 - 0.22) + (1 - 0.25)) / 2)
    assert result["diagnostics"]["kl_js"] == "not_available_without_full_distribution"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        report_rescue_harm(
            observations=observations,
            mask_bundle=masks,
            feature_observations=feature,
            feature_lineage=feature_lineage,
            parent_checkpoint=parent_checkpoint,
            output=tmp_path / "report.json",
            expected_count=6,
        )
    table = pq.read_table(observations["control"][0])
    pq.write_table(table.take(list(reversed(range(table.num_rows)))), observations["control"][0])
    control_lineage = json.loads(observations["control"][1].read_text(encoding="utf-8"))
    control_lineage["observations_sha256"] = sha256_file(observations["control"][0])
    observations["control"][1].write_text(json.dumps(control_lineage), encoding="utf-8")
    rerun = report_rescue_harm(
        observations=observations,
        mask_bundle=masks,
        feature_observations=feature,
        feature_lineage=feature_lineage,
        parent_checkpoint=parent_checkpoint,
        output=tmp_path / "rerun.json",
        expected_count=6,
    )
    assert rerun["epochs_report"] == result["epochs_report"]


def test_rescue_harm_rejects_attack_lineage_drift_and_frozen_mask_future_fields(tmp_path: Path) -> None:
    observations, masks, feature, feature_lineage, parent_checkpoint = _report_inputs(tmp_path)
    bad = json.loads(observations["PF_R"][1].read_text(encoding="utf-8"))
    bad["attack_identity"] = {"loss": "ce", "steps": 10}
    observations["PF_R"][1].write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RescueHarmError, match="attack/student/teacher/seed"):
        report_rescue_harm(
            observations=observations,
            mask_bundle=masks,
            feature_observations=feature,
            feature_lineage=feature_lineage,
            parent_checkpoint=parent_checkpoint,
            output=tmp_path / "bad.json",
            expected_count=6,
        )
    # Selection IDs are accepted only from the frozen mask bundle; it has no
    # future-outcome field from which report code could rebuild a mask.
    assert "outcome" not in json.loads(masks.read_text(encoding="utf-8"))


def test_rescue_harm_rejects_wrong_parent_checkpoint_and_sample_state_hash(tmp_path: Path) -> None:
    observations, masks, feature, feature_lineage, parent_checkpoint = _report_inputs(tmp_path)
    payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    payload["global_step"] = 1
    wrong_checkpoint = tmp_path / "wrong-parent.pt"
    torch.save(payload, wrong_checkpoint)
    with pytest.raises(RescueHarmError, match="checkpoint SHA"):
        report_rescue_harm(
            observations=observations,
            mask_bundle=masks,
            feature_observations=feature,
            feature_lineage=feature_lineage,
            parent_checkpoint=wrong_checkpoint,
            output=tmp_path / "wrong-parent.json",
            expected_count=6,
        )
    bundle = json.loads(masks.read_text(encoding="utf-8"))
    bundle["parent"]["sample_state_sha256"] = "0" * 64
    masks.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(RescueHarmError, match="sample-state SHA"):
        report_rescue_harm(
            observations=observations,
            mask_bundle=masks,
            feature_observations=feature,
            feature_lineage=feature_lineage,
            parent_checkpoint=parent_checkpoint,
            output=tmp_path / "wrong-state.json",
            expected_count=6,
        )


def test_rescue_harm_merge_turns_single_epoch_replays_into_formal_resumable_panel(tmp_path: Path) -> None:
    observations, _, _, _, _ = _report_inputs(tmp_path)
    full_observations, full_lineage = observations["control"]
    table = pq.read_table(full_observations)
    base_lineage = json.loads(full_lineage.read_text(encoding="utf-8"))
    base_lineage["analysis_provenance"] = {"git": {"sha": "b" * 40, "dirty": False}}
    inputs: dict[int, tuple[Path, Path]] = {}
    for epoch in EPOCHS:
        path, lineage = tmp_path / f"epoch{epoch}.parquet", tmp_path / f"epoch{epoch}.json"
        rows = [row for row in table.to_pylist() if row["epoch"] == epoch]
        pq.write_table(pa.Table.from_pylist(rows), path)
        value = {
            **base_lineage,
            "observations_sha256": sha256_file(path),
            "row_count": len(rows),
            "checkpoints": [{"epoch": epoch, "sha256": "f" * 64}],
        }
        lineage.write_text(json.dumps(value), encoding="utf-8")
        inputs[epoch] = (path, lineage)
    merged = merge_epoch_replays(
        inputs=inputs, output_parquet=tmp_path / "merged.parquet", output_lineage=tmp_path / "merged.json"
    )
    assert merged["row_count"] == 24
    assert [item["epoch"] for item in merged["checkpoints"]] == list(EPOCHS)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        merge_epoch_replays(
            inputs=inputs, output_parquet=tmp_path / "merged.parquet", output_lineage=tmp_path / "other.json"
        )
    drifted = pq.read_table(inputs[104][0]).to_pylist()
    drifted[0]["sample_id"] = 999
    pq.write_table(pa.Table.from_pylist(drifted), inputs[104][0])
    drift_lineage = json.loads(inputs[104][1].read_text(encoding="utf-8"))
    drift_lineage["observations_sha256"] = sha256_file(inputs[104][0])
    inputs[104][1].write_text(json.dumps(drift_lineage), encoding="utf-8")
    with pytest.raises(RescueHarmError, match="stable ID/class join"):
        merge_epoch_replays(
            inputs=inputs, output_parquet=tmp_path / "drift.parquet", output_lineage=tmp_path / "drift.json"
        )


def _v3_row(*, arm: str, epoch: int, sample_id: int, robust: bool) -> dict[str, object]:
    selected = arm != "C" and sample_id in {2, 5}
    pf = arm.startswith("PF-")
    return {
        "namespace": "train",
        "run_id": f"run-{arm}",
        "arm": arm,
        "seed": 7,
        "epoch": epoch,
        "sample_id": sample_id,
        "class_id": sample_id % 10,
        "clean_prediction": sample_id % 10,
        "clean_correct": True,
        "clean_probability_margin": 0.5,
        "robust_prediction": sample_id % 10 if robust else (sample_id + 1) % 10,
        "robust_correct": robust,
        "robust_probability_margin": 0.4 if robust else -0.4,
        "teacher_clean_prediction": sample_id % 10,
        "teacher_clean_correct": True,
        "teacher_clean_true_probability": 0.6,
        "teacher_clean_probability_margin": 0.2,
        "teacher_clean_entropy_normalized": 0.4,
        "teacher_adversarial_prediction": sample_id % 10,
        "teacher_adversarial_correct": True,
        "teacher_adversarial_true_probability": 0.5,
        "teacher_adversarial_probability_margin": 0.1,
        "teacher_adversarial_entropy_normalized": 0.5,
        "teacher_clean_to_adversarial_kl": 0.02,
        "teacher_clean_to_adversarial_prediction_flip": False,
        "teacher_clean_to_adversarial_true_probability_delta": -0.1,
        "teacher_clean_to_adversarial_margin_delta": -0.1,
        "route": "control" if arm == "C" else "PF" if pf else "NR",
        "mask_selected": selected,
        "intervention_active": selected
        and ((pf and 80 <= epoch <= 129) or (arm.startswith("NR-") and 80 <= epoch <= 99)),
        "intervention_identity": "control"
        if arm == "C"
        else "pf_teacher_0.75_anchor_0.25_epochs80_129"
        if pf
        else "nr_prefix_pgd5_selected_epochs80_99_else_pgd10",
        "pf_anchor_clean_probability_margin": 0.3 if pf else None,
        "pf_anchor_adversarial_probability_margin": 0.2 if pf else None,
    }


def _write_v3_arm(tmp_path: Path, arm: str, ids: tuple[int, ...]) -> tuple[Path, Path]:
    observations, lineage = tmp_path / f"v3-{arm}.parquet", tmp_path / f"v3-{arm}.json"
    rows = []
    for epoch in V3_EPOCHS if arm == "C" else V3_EPOCHS[1:]:
        for sample_id in ids:
            control = sample_id in {2, 8}
            robust = control if arm == "C" else {2: False, 5: True, 8: True}.get(sample_id, control)
            rows.append(_v3_row(arm=arm, epoch=epoch, sample_id=sample_id, robust=robust))
    pq.write_table(pa.Table.from_pylist(rows), observations)
    parent_sha, state_sha = "c" * 64, "d" * 64
    checkpoints = [{"epoch": 79, "sha256": parent_sha, "sample_state_sha256": state_sha}] if arm == "C" else []
    parent: dict[str, object] = {
        "kind": "explicit_shared_epoch79_parent_v1",
        "checkpoint_path": "parent.pt",
        "checkpoint_sha256": "b" * 64,
    }
    if arm != "C":
        parent = {
            "kind": "prescriptive_v3_epoch79_parent_v1",
            "checkpoint_sha256": parent_sha,
            "sample_state_sha256": state_sha,
        }
    lineage.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "contract": "prescriptive_v3_rescue_harm_replay_v1",
                "observations_sha256": sha256_file(observations),
                "inventory_sha256": "a" * 64,
                "run_id": f"run-{arm}",
                "arm": arm,
                "seed": 7,
                "requested_epochs": list(V3_EPOCHS if arm == "C" else V3_EPOCHS[1:]),
                "parent_epochs": list(V3_EPOCHS),
                "config_sha256": "b" * 64,
                "scientific_git_sha": "e" * 40,
                "teacher": {"checkpoint_sha256": "f" * 64},
                "dataset_identity": {"name": "cifar10"},
                "attack_identity": {"loss": "ce", "steps": 20},
                "source_manifest_sha256": "1" * 64,
                "source_resolved_config_sha256": "2" * 64,
                "parent": parent,
                "checkpoints": checkpoints,
                "analysis_seed": 99,
                "row_count": len(rows),
                "observation_columns": list(V3_OBSERVATION_COLUMNS),
                "analysis_provenance": {"git": {"sha": "3" * 40, "dirty": False}},
            }
        ),
        encoding="utf-8",
    )
    return observations, lineage


def test_v3_report_common_epochs_categories_parent_and_sparse_row_order(tmp_path: Path) -> None:
    ids = (2, 5, 8, 13)
    observations = {arm: _write_v3_arm(tmp_path, arm, ids) for arm in V3_ARMS}
    assert 79 not in pq.read_table(observations["PF-H"][0]).column("epoch").to_pylist()
    result = report_v3_rescue_harm(observations=observations, output=tmp_path / "v3-report.json", expected_count=4)
    categories = result["epochs_report"]["99"]["PF-H"]["categories"]
    assert categories["all"]["categories"] == {
        "rescued": 1,
        "harmed": 1,
        "stable_correct": 1,
        "unchanged_failure": 1,
    }
    assert categories["selected"]["count"] == 2
    assert result["input_identity"]["shared_parent_checkpoint_sha256"] == "c" * 64
    assert result["epochs_report"]["99"]["PF-H"]["teacher_response"]["selected"]["margin_delta"]["count"] == 2
    assert result["epochs_report"]["99"]["PF-H"]["pf_anchor_alignment"]["selected"]["anchor_clean_margin"]["count"] == 2
    assert result["epochs_report"]["99"]["NR-H"]["intervention"]["nr_phase"] == "active_window"
    assert result["epochs_report"]["79"]["NR-H"]["shared_parent_baseline"] is True
    assert result["epochs_report"]["119"]["NR-H"]["intervention"]["nr_phase"] == "post_window"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        report_v3_rescue_harm(observations=observations, output=tmp_path / "v3-report.json", expected_count=4)
    table = pq.read_table(observations["C"][0])
    pq.write_table(table.take(list(reversed(range(table.num_rows)))), observations["C"][0])
    control_lineage = json.loads(observations["C"][1].read_text(encoding="utf-8"))
    control_lineage["observations_sha256"] = sha256_file(observations["C"][0])
    observations["C"][1].write_text(json.dumps(control_lineage), encoding="utf-8")
    rerun = report_v3_rescue_harm(observations=observations, output=tmp_path / "v3-rerun.json", expected_count=4)
    assert rerun["epochs_report"] == result["epochs_report"]


def test_v3_report_allows_preregistered_control_child_git_difference(tmp_path: Path) -> None:
    observations = {arm: _write_v3_arm(tmp_path, arm, (2, 5)) for arm in V3_ARMS}
    child = json.loads(observations["C"][1].read_text(encoding="utf-8"))
    child["scientific_git_sha"] = "9" * 40
    observations["C"][1].write_text(json.dumps(child), encoding="utf-8")
    result = report_v3_rescue_harm(observations=observations, output=tmp_path / "git.json", expected_count=2)
    assert result["input_identity"]["arm_scientific_git_sha"]["C"] == "9" * 40
    assert result["input_identity"]["arm_scientific_git_sha"]["PF-H"] == "e" * 40


def test_v3_report_rejects_child_git_or_analysis_provenance_drift(tmp_path: Path) -> None:
    observations = {arm: _write_v3_arm(tmp_path, arm, (2, 5)) for arm in V3_ARMS}
    lineage = json.loads(observations["PF-R"][1].read_text(encoding="utf-8"))
    lineage["scientific_git_sha"] = "9" * 40
    observations["PF-R"][1].write_text(json.dumps(lineage), encoding="utf-8")
    with pytest.raises(RescueHarmError, match="PF/NR children"):
        report_v3_rescue_harm(observations=observations, output=tmp_path / "git-drift.json", expected_count=2)
    observations = {arm: _write_v3_arm(tmp_path, arm, (2, 5)) for arm in V3_ARMS}
    lineage = json.loads(observations["NR-R"][1].read_text(encoding="utf-8"))
    lineage["analysis_provenance"] = {"git": {"sha": "9" * 40, "dirty": False}}
    observations["NR-R"][1].write_text(json.dumps(lineage), encoding="utf-8")
    with pytest.raises(RescueHarmError, match="identity drifted"):
        report_v3_rescue_harm(observations=observations, output=tmp_path / "provenance-drift.json", expected_count=2)


def test_teacher_response_kl_is_finite_for_extreme_logits() -> None:
    clean = torch.tensor([[1000.0, -1000.0, -1000.0], [-1000.0, 1000.0, -1000.0]])
    adversarial = torch.tensor([[-1000.0, 1000.0, -1000.0], [1000.0, -1000.0, -1000.0]])
    result = _teacher_response_kl(clean, adversarial)
    assert torch.isfinite(result).all()
    assert (result > 0).all()


def test_v3_cli_smoke_epoch_is_forwarded_to_replay_and_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    replay_epochs: list[tuple[int, ...] | None] = []
    report_epochs: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        "ard.cli.rescue_harm.replay_v3_inventory",
        lambda **kwargs: (
            replay_epochs.append(kwargs["expected_epochs"]) or {"contract": "prescriptive_v3_rescue_harm_replay_v1"}
        ),
    )
    assert (
        rescue_harm_main(
            [
                "replay",
                "--contract",
                "v3",
                "--resolved-config",
                "config.yaml",
                "--inventory",
                "inventory.json",
                "--observations",
                "rows.parquet",
                "--lineage",
                "rows.json",
                "--device",
                "cpu",
                "--batch-size",
                "2",
                "--analysis-seed",
                "1",
                "--epoch",
                "99",
            ]
        )
        == 0
    )
    assert replay_epochs == [(79, 99)]
    monkeypatch.setattr(
        "ard.cli.rescue_harm.report_v3_rescue_harm",
        lambda **kwargs: (
            report_epochs.append(tuple(kwargs["report_epochs"]))
            or {"contract": "prescriptive_v3_rescue_harm_report_v1"}
        ),
    )
    observations = [item for arm in V3_ARMS for item in ("--observations", f"{arm}=rows-{arm}.parquet")]
    lineages = [item for arm in V3_ARMS for item in ("--lineage", f"{arm}=rows-{arm}.json")]
    assert (
        rescue_harm_main(
            [
                "report",
                "--contract",
                "v3",
                *observations,
                *lineages,
                "--output",
                str(tmp_path / "report.json"),
                "--expected-count",
                "2",
                "--smoke-epoch",
                "99",
            ]
        )
        == 0
    )
    assert report_epochs == [(99,)]
    assert "prescriptive_v3_rescue_harm_report_v1" in capsys.readouterr().out
    monkeypatch.setattr(
        "ard.cli.rescue_harm.smoke_v3_report",
        lambda **_: {"contract": "prescriptive_v3_rescue_harm_smoke_report_v1"},
    )
    assert (
        rescue_harm_main(
            [
                "smoke-report",
                "--observations",
                "rows.parquet",
                "--lineage",
                "rows.json",
                "--arm",
                "PF-H",
                "--epoch",
                "99",
                "--expected-count",
                "2",
                "--output",
                str(tmp_path / "smoke.json"),
            ]
        )
        == 0
    )


def test_v3_inventory_uses_payload_epochs_control_bytes_and_child_shared_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_hash = "a" * 64
    paths: dict[int, Path] = {}
    for epoch in V3_EPOCHS:
        path = tmp_path / f"epoch-{epoch}.pt"
        torch.save(_checkpoint(epoch, run_id="run", config_hash=config_hash), path)
        paths[epoch] = path
    parent_sha = sha256_file(paths[79])
    state_sha = hashlib.sha256(b"{}").hexdigest()
    for epoch in V3_EPOCHS[1:]:
        payload = _checkpoint(epoch, run_id="run", config_hash=config_hash)
        payload["fork_lineage"] = {
            "parent_epoch": 79,
            "parent_checkpoint_sha256": parent_sha,
            "parent_sample_state_sha256": state_sha,
        }
        torch.save(payload, paths[epoch])
    entries = tuple(
        CheckpointInventory(
            run_id="run",
            artifact_name=f"opaque-{epoch}",
            aliases=("last",),
            publication_order=epoch,
            path=str(paths[epoch]),
            sha256=sha256_file(paths[epoch]),
            epoch=epoch,
            sample_state_present=True,
            sample_state_count=0,
            config_hash=config_hash,
            scientific_git_sha="b" * 40,
        )
        for epoch in V3_EPOCHS
    )
    teacher = SimpleNamespace(model_dump=lambda **_: {"checkpoint_sha256": "c" * 64})
    attack = SimpleNamespace(model_dump=lambda **_: {"loss": "ce", "steps": 20})
    config = SimpleNamespace(teacher=teacher, method=SimpleNamespace(selection_attack=attack))
    monkeypatch.setattr(
        "ard.analysis.rescue_harm.load_resolved_config_for_evaluation",
        lambda _: SimpleNamespace(config=config, raw_config_hash=config_hash),
    )
    monkeypatch.setattr("ard.analysis.rescue_harm.inventory_run_bundle", lambda _: entries)
    monkeypatch.setattr("ard.analysis.rescue_harm.logical_dataset_identity", lambda *_args, **_kwargs: {"train": "x"})
    monkeypatch.setattr("ard.analysis.rescue_harm.resolved_config_dict", lambda _: {})
    manifest, resolved = tmp_path / "manifest.json", tmp_path / "resolved.yaml"
    manifest.write_text("{}", encoding="utf-8")
    resolved.write_text("{}", encoding="utf-8")
    control = build_v3_checkpoint_inventory(
        manifest=manifest,
        resolved_config=resolved,
        arm="C",
        seed=1,
        output=tmp_path / "control.json",
        shared_parent_checkpoint=paths[79],
    )
    assert [row["epoch"] for row in control["checkpoints"]] == list(V3_EPOCHS[1:])
    assert control["parent"]["checkpoint_sha256"] == parent_sha
    _load_v3_inventory(tmp_path / "control.json")
    changed_payload = _checkpoint(79, run_id="run", config_hash=config_hash)
    changed_payload["global_step"] = 1
    torch.save(changed_payload, paths[79])
    with pytest.raises(RescueHarmError, match="control shared epoch-79 parent bytes drifted"):
        _load_v3_inventory(tmp_path / "control.json")
    parent = {"kind": "parent", "checkpoint_sha256": "d" * 64, "sample_state_sha256": "e" * 64}
    monkeypatch.setattr("ard.analysis.rescue_harm._v3_mask_and_parent", lambda *_args, **_kwargs: parent)
    child = build_v3_checkpoint_inventory(
        manifest=manifest,
        resolved_config=resolved,
        arm="PF-H",
        seed=1,
        output=tmp_path / "child.json",
    )
    assert 79 not in [row["epoch"] for row in child["checkpoints"]]
    assert child["parent"] == parent


@pytest.mark.parametrize("arm,expect_anchor", [("PF-H", True), ("NR-H", False), ("C", False)])
def test_v3_replay_mocked_small_teacher_anchor_schema_and_nonoverwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str, expect_anchor: bool
) -> None:
    class Batch:
        images = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        labels = torch.tensor([0, 1])
        sample_ids = torch.tensor([101, 707])

        def to(self, _device: torch.device) -> Batch:
            return self

    class Loader:
        dataset = [None, None]

        def __iter__(self):
            return iter((Batch(),))

    class Student(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.cat((inputs[:, :1] * 4, inputs[:, 1:2] * 4, torch.zeros((len(inputs), 8))), dim=1)

    class Teacher(Student):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    class Attack:
        def __init__(self, _config: object) -> None:
            pass

        def generate(self, request: object) -> SimpleNamespace:
            return SimpleNamespace(adversarial=request.inputs, max_abs_delta=0.0)

    attack = SimpleNamespace(
        loss="ce", steps=20, norm="linf", input_domain="pixel_0_1", student_mode="eval", epsilon_value=0.1
    )
    teacher_config = SimpleNamespace(model_dump=lambda **_: {"checkpoint_sha256": "c" * 64})
    config = SimpleNamespace(
        teacher=teacher_config,
        dataset=SimpleNamespace(name="cifar10", split="train"),
        method=SimpleNamespace(selection_attack=attack),
        tier="dev",
        student=SimpleNamespace(),
    )
    checkpoints = [{"epoch": 99, "path": "child.pt", "sha256": "a" * 64}]
    if arm == "C":
        checkpoints = [{"epoch": 79, "path": "parent.pt", "sha256": "b" * 64}, *checkpoints]
    parent: dict[str, object] = {
        "kind": "explicit_shared_epoch79_parent_v1",
        "checkpoint_path": "parent.pt",
        "checkpoint_sha256": "b" * 64,
    }
    if arm != "C":
        parent = {"checkpoint_path": "parent.pt", "checkpoint_sha256": "b" * 64}
    inventory = {
        "config_sha256": "d" * 64,
        "teacher": teacher_config.model_dump(),
        "parent": parent,
        "checkpoints": checkpoints,
        "arm": arm,
        "requested_epochs": [79, 99],
        "run_id": "run",
        "seed": 1,
        "scientific_git_sha": "e" * 40,
        "dataset_identity": {"name": "cifar10"},
        "attack_identity": {"loss": "ce", "steps": 20},
        "source_manifest_sha256": "f" * 64,
        "source_resolved_config_sha256": "0" * 64,
    }
    monkeypatch.setattr("ard.analysis.rescue_harm._tracked_clean_provenance", lambda: {"git": {"dirty": False}})
    monkeypatch.setattr("ard.analysis.rescue_harm._load_v3_inventory", lambda _: inventory)
    monkeypatch.setattr(
        "ard.analysis.rescue_harm.load_resolved_config_for_evaluation",
        lambda _: SimpleNamespace(config=config, raw_config_hash="d" * 64),
    )
    monkeypatch.setattr("ard.analysis.rescue_harm.build_replay_loader", lambda *_args, **_kwargs: Loader())
    monkeypatch.setattr("ard.analysis.rescue_harm._v3_selected_ids", lambda _: {101})
    teachers: list[Teacher] = []

    def build_mock_teacher(*_args: object, **_kwargs: object) -> Teacher:
        value = Teacher()
        teachers.append(value)
        return value

    monkeypatch.setattr("ard.analysis.rescue_harm.build_teacher", build_mock_teacher)
    monkeypatch.setattr("ard.analysis.rescue_harm.build_student", lambda *_args, **_kwargs: Student())
    monkeypatch.setattr(
        "ard.analysis.rescue_harm.load_saved_student_checkpoint",
        lambda *_args, **_kwargs: _checkpoint(
            79 if _args[0].name == "parent.pt" else 99, run_id="run", config_hash="d" * 64
        ),
    )
    monkeypatch.setattr("ard.analysis.rescue_harm.LinfPGD", Attack)
    output, lineage = tmp_path / f"{arm}.parquet", tmp_path / f"{arm}.json"
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text("{}", encoding="utf-8")
    replay_v3_inventory(
        resolved_config=tmp_path / "resolved.yaml",
        inventory_path=inventory_path,
        output_parquet=output,
        output_lineage=lineage,
        device=torch.device("cpu"),
        batch_size=2,
        analysis_seed=3,
        expected_count=2,
    )
    table = pq.read_table(output)
    assert tuple(table.column_names) == V3_OBSERVATION_COLUMNS
    assert set(table.column("sample_id").to_pylist()) == {101, 707}
    anchor = table.column("pf_anchor_clean_probability_margin").to_pylist()
    assert all(value is not None for value in anchor) if expect_anchor else all(value is None for value in anchor)
    assert len(teachers) == 1 and teachers[0].training is False
    assert all(parameter.requires_grad is False and parameter.grad is None for parameter in teachers[0].parameters())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        replay_v3_inventory(
            resolved_config=tmp_path / "resolved.yaml",
            inventory_path=inventory_path,
            output_parquet=output,
            output_lineage=tmp_path / "other.json",
            device=torch.device("cpu"),
            batch_size=2,
            analysis_seed=3,
            expected_count=2,
        )


def test_v3_single_arm_smoke_report_validates_one_emitted_epoch(tmp_path: Path) -> None:
    observations, lineage = _write_v3_arm(tmp_path, "PF-H", (101, 707))
    table = pq.read_table(observations)
    pq.write_table(table.filter(pa.compute.equal(table["epoch"], 99)), observations)
    meta = json.loads(lineage.read_text(encoding="utf-8"))
    meta["requested_epochs"] = [99]
    meta["parent_epochs"] = list(V3_EPOCHS)
    meta["row_count"] = 2
    meta["observations_sha256"] = sha256_file(observations)
    lineage.write_text(json.dumps(meta), encoding="utf-8")
    result = smoke_v3_report(
        observations=observations,
        lineage=lineage,
        arm="PF-H",
        epoch=99,
        expected_count=2,
        output=tmp_path / "smoke.json",
    )
    assert result["count"] == 2
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        smoke_v3_report(
            observations=observations,
            lineage=lineage,
            arm="PF-H",
            epoch=99,
            expected_count=2,
            output=tmp_path / "smoke.json",
        )


def test_v3_report_rejects_common_epoch_and_shared_parent_drift(tmp_path: Path) -> None:
    observations = {arm: _write_v3_arm(tmp_path, arm, (2, 5)) for arm in V3_ARMS}
    lineage = json.loads(observations["PF-H"][1].read_text(encoding="utf-8"))
    lineage["requested_epochs"] = [79, 99]
    observations["PF-H"][1].write_text(json.dumps(lineage), encoding="utf-8")
    with pytest.raises(RescueHarmError, match="common epoch"):
        report_v3_rescue_harm(observations=observations, output=tmp_path / "bad-epochs.json", expected_count=2)
    observations = {arm: _write_v3_arm(tmp_path, arm, (2, 5)) for arm in V3_ARMS}
    lineage = json.loads(observations["NR-H"][1].read_text(encoding="utf-8"))
    lineage["parent"]["sample_state_sha256"] = "0" * 64
    observations["NR-H"][1].write_text(json.dumps(lineage), encoding="utf-8")
    with pytest.raises(RescueHarmError, match="shared epoch-79 parent/state"):
        report_v3_rescue_harm(observations=observations, output=tmp_path / "bad-parent.json", expected_count=2)


def test_v3_merge_rejects_contract_mixing_and_preserves_sparse_panel(tmp_path: Path) -> None:
    observation, lineage = _write_v3_arm(tmp_path, "C", (2, 5))
    inputs = {epoch: (observation, lineage) for epoch in V3_EPOCHS}
    merged = merge_v3_epoch_replays(
        inputs=inputs,
        output_parquet=tmp_path / "v3-merged.parquet",
        output_lineage=tmp_path / "v3-merged.json",
    )
    assert merged["requested_epochs"] == list(V3_EPOCHS)
    assert merged["row_count"] == 2 * len(V3_EPOCHS)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        merge_v3_epoch_replays(
            inputs=inputs,
            output_parquet=tmp_path / "v3-merged.parquet",
            output_lineage=tmp_path / "elsewhere.json",
        )
    v2 = json.loads(lineage.read_text(encoding="utf-8"))
    v2["contract"] = "completed_v2_rescue_harm_replay_v1"
    lineage.write_text(json.dumps(v2), encoding="utf-8")
    with pytest.raises(RescueHarmError, match="mixes a v2"):
        merge_v3_epoch_replays(
            inputs=inputs,
            output_parquet=tmp_path / "mixed.parquet",
            output_lineage=tmp_path / "mixed.json",
        )
