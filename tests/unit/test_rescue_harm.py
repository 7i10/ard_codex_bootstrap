from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ard.analysis.rescue_harm import (
    EPOCHS,
    SOURCE_ARM,
    RescueHarmError,
    load_checkpoint_inventory,
    merge_epoch_replays,
    report_rescue_harm,
)
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS
from ard.analysis.signal_audit import sha256_file
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
