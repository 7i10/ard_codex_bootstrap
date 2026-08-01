"""Focused contracts for the frozen H2 logging-only prediction boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ard.analysis.logging_only_prediction import (
    EXPECTED_COUNT,
    EXPECTED_EXTERNAL_COMMITS,
    EXPECTED_MANIFEST_LINEAGE,
    EXPECTED_RUN_IDS,
    LoggingOnlyPredictionError,
    _features,
    _FrozenDesign,
    _prepare_rows,
    _trajectory_report,
    _trajectory_reports,
    _validate_frozen_export_lineage,
    _validate_manifest_lineage,
    analyze_logging_only_exports,
    load_frozen_design,
    load_state_export_with_provenance,
)
from ard.cli import logging_only_prediction as prediction_cli


def _export(
    *,
    run_id: str = "run-1",
    duplicate: bool = False,
    noncontiguous: bool = False,
    leakage: bool = False,
    outcome_mismatch: bool = False,
) -> dict[str, object]:
    source_files = {"analysis": "a" * 64, "cli": "b" * 64}
    source_sha256 = hashlib.sha256(json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rows = []
    for sample_id in range(EXPECTED_COUNT):
        forgetting = sample_id % 3 == 0
        row: dict[str, object] = {
            "sample_id": (
                0
                if duplicate and sample_id == EXPECTED_COUNT - 1
                else 49999
                if noncontiguous and sample_id == EXPECTED_COUNT - 1
                else sample_id
            ),
            "true_label": sample_id % 10,
            "anchor_teacher_adversarial_entropy": 0.5,
            "anchor_previous_robust_correct": sample_id % 2 == 0,
            "anchor_last_margin": 0.2,
            "anchor_robust_correct_frequency": 0.5,
            "anchor_margin_ema": -0.2,
            "anchor_forgetting_count": 2,
            "final_forgetting_count": 3 if forgetting else 2,
            "subsequent_forgetting_increment": 1 if forgetting else 0,
            "future_online_forgetting": int(forgetting),
        }
        if leakage:
            row["official_test_auroc"] = 1.0
        if outcome_mismatch and sample_id == 0:
            row["future_online_forgetting"] = 0
        rows.append(row)
    return {
        "identity": {
            "contract": "logging_only_exact_state_anchor99_final199_v1",
            "expected_count": EXPECTED_COUNT,
            "row_count": EXPECTED_COUNT,
            "run_id": run_id,
            "config_hash": "c" * 64,
            "scientific_git_sha": "2" * 40,
            "world_size": 1,
            "run_bundle_manifest_sha256": "3" * 64,
            "run_bundle_completion_sha256": "4" * 64,
            "checkpoint_artifacts": {
                "anchor": {
                    "artifact_name": "anchor-last",
                    "artifact_local_path": "artifacts/anchor",
                    "sha256": "d" * 64,
                },
                "final": {
                    "artifact_name": "final-last",
                    "artifact_local_path": "artifacts/final",
                    "sha256": "f" * 64,
                },
            },
            "anchor": {"epoch": 99, "checkpoint_sha256": "d" * 64, "sample_state_sha256": "e" * 64},
            "final": {"epoch": 199, "checkpoint_sha256": "f" * 64, "sample_state_sha256": "0" * 64},
            "analysis_provenance": {
                "git_sha": "1" * 40,
                "dirty": False,
                "source_files": source_files,
                "source_sha256": source_sha256,
            },
        },
        "rows": rows,
    }


def _manifest_for_export(tmp_path: Path, *, label: str, export: dict[str, object]) -> tuple[Path, Path]:
    identity = export["identity"]
    assert isinstance(identity, dict)
    expected = EXPECTED_MANIFEST_LINEAGE[label]
    identity["scientific_git_sha"] = expected["scientific_git_sha"]
    teacher = {
        "source": "robustbench",
        "registry_id": expected["teacher_registry_id"],
        "checkpoint_sha256": expected["teacher_checkpoint_sha256"],
        "checkpoint_actual_sha256": expected["teacher_checkpoint_sha256"],
        "external_commit": EXPECTED_EXTERNAL_COMMITS["robustbench"],
    }
    manifest = {
        "run_id": EXPECTED_RUN_IDS[label],
        "config_hash": identity["config_hash"],
        "world_size": 1,
        "seed": expected["seed"],
        "training_seed": expected["seed"],
        "git": {"sha": expected["scientific_git_sha"], "dirty": False},
        "teacher": teacher,
        "external": {
            "repositories": {
                name: {"commit": commit, "checkout": {"head": commit, "status": ""}}
                for name, commit in EXPECTED_EXTERNAL_COMMITS.items()
            }
        },
        "artifacts": [
            {
                "name": "anchor-last",
                "type": "model",
                "aliases": ["last"],
                "local_path": "artifacts/anchor",
                "sha256": "d" * 64,
            },
            {
                "name": "final-last",
                "type": "model",
                "aliases": ["last"],
                "local_path": "artifacts/final",
                "sha256": "f" * 64,
            },
        ],
    }
    path = tmp_path / f"{label}-manifest.json"
    encoded = json.dumps(manifest, sort_keys=True).encode()
    path.write_bytes(encoded)
    identity["run_bundle_manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    completion = path.parent / "completion.json"
    completion_bytes = json.dumps(
        {"status": "completed", "output_dir": f"/outputs/{EXPECTED_RUN_IDS[label]}"}, sort_keys=True
    ).encode()
    completion.write_bytes(completion_bytes)
    identity["run_bundle_completion_sha256"] = hashlib.sha256(completion_bytes).hexdigest()
    return path, completion


def _rebind_frozen_lineage(monkeypatch: pytest.MonkeyPatch, *, label: str, export: dict[str, object]) -> None:
    identity = export["identity"]
    assert isinstance(identity, dict)
    lineage = EXPECTED_MANIFEST_LINEAGE[label]
    monkeypatch.setitem(lineage, "config_hash", identity["config_hash"])
    monkeypatch.setitem(lineage, "manifest_sha256", identity["run_bundle_manifest_sha256"])
    monkeypatch.setitem(lineage, "completion_sha256", identity["run_bundle_completion_sha256"])
    for checkpoint in ("anchor", "final"):
        checkpoint_identity = identity[checkpoint]
        assert isinstance(checkpoint_identity, dict)
        monkeypatch.setitem(lineage, f"{checkpoint}_checkpoint_sha256", checkpoint_identity["checkpoint_sha256"])


def test_frozen_design_hash_contract_loads() -> None:
    design = load_frozen_design()
    assert design.design_sha256 == "d653d9ef08cfa94976a0e3279166b47543d16f3eaadb69810769470b77838c12"
    assert design.bootstrap_replicates == 1000


@pytest.mark.parametrize("kwargs", [{"duplicate": True}, {"leakage": True}, {"outcome_mismatch": True}])
def test_state_export_rejects_duplicate_leakage_or_outcome_drift(kwargs: dict[str, bool]) -> None:
    with pytest.raises(LoggingOnlyPredictionError):
        _prepare_rows(_export(**kwargs))


def test_state_export_rejects_tampered_hash_bound_identity() -> None:
    export = _export()
    identity = export["identity"]
    assert isinstance(identity, dict)
    identity["world_size"] = 2
    with pytest.raises(LoggingOnlyPredictionError, match="world size"):
        _prepare_rows(export)


@pytest.mark.parametrize(
    "drift", ["run_id", "seed", "config", "git", "teacher", "checkpoint", "manifest_bytes", "completion_status"]
)
def test_manifest_lineage_rejects_run_id_only_or_tampered_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    export = _export(run_id=EXPECTED_RUN_IDS["L1"])
    path, completion = _manifest_for_export(tmp_path, label="L1", export=export)
    identity = export["identity"]
    assert isinstance(identity, dict)
    _rebind_frozen_lineage(monkeypatch, label="L1", export=export)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if drift == "run_id":
        manifest["run_id"] = EXPECTED_RUN_IDS["L2"]
    elif drift == "seed":
        manifest["seed"] = 2
    elif drift == "config":
        manifest["config_hash"] = "9" * 64
    elif drift == "git":
        manifest["git"]["sha"] = "8" * 40
    elif drift == "teacher":
        manifest["teacher"]["registry_id"] = "chen2021_ltd_wrn34_10"
    elif drift == "checkpoint":
        manifest["teacher"]["checkpoint_sha256"] = "7" * 64
        manifest["teacher"]["checkpoint_actual_sha256"] = "7" * 64
    elif drift == "completion_status":
        completion_bytes = json.dumps(
            {"status": "failed", "output_dir": f"/outputs/{EXPECTED_RUN_IDS['L1']}"}, sort_keys=True
        ).encode()
        completion.write_bytes(completion_bytes)
        identity["run_bundle_completion_sha256"] = hashlib.sha256(completion_bytes).hexdigest()
        monkeypatch.setitem(
            EXPECTED_MANIFEST_LINEAGE["L1"], "completion_sha256", identity["run_bundle_completion_sha256"]
        )
        with pytest.raises(LoggingOnlyPredictionError, match="not completed"):
            _validate_manifest_lineage(label="L1", path=path, export_identity=identity)
        return
    else:
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(LoggingOnlyPredictionError):
            _validate_manifest_lineage(label="L1", path=path, export_identity=identity)
        return
    encoded = json.dumps(manifest, sort_keys=True).encode()
    path.write_bytes(encoded)
    identity["run_bundle_manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(LoggingOnlyPredictionError):
        _validate_manifest_lineage(label="L1", path=path, export_identity=identity)


def test_manifest_lineage_records_resolved_path_hash_and_validated_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _export(run_id=EXPECTED_RUN_IDS["L3"])
    path, completion = _manifest_for_export(tmp_path, label="L3", export=export)
    identity = export["identity"]
    assert isinstance(identity, dict)
    _rebind_frozen_lineage(monkeypatch, label="L3", export=export)
    result = _validate_manifest_lineage(label="L3", path=path, export_identity=identity)
    assert result["path"] == str(path.resolve())
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["completion_path"] == str(completion.resolve())
    assert result["completion_sha256"] == hashlib.sha256(completion.read_bytes()).hexdigest()
    assert result["identity"]["teacher"]["registry_id"] == "bartoldson2024_adversarial_wrn94_16"


def test_frozen_export_lineage_rejects_exact_checkpoint_drift() -> None:
    export = _export(run_id=EXPECTED_RUN_IDS["L1"])
    identity = export["identity"]
    assert isinstance(identity, dict)
    identity["config_hash"] = EXPECTED_MANIFEST_LINEAGE["L1"]["config_hash"]
    identity["run_bundle_manifest_sha256"] = EXPECTED_MANIFEST_LINEAGE["L1"]["manifest_sha256"]
    identity["run_bundle_completion_sha256"] = EXPECTED_MANIFEST_LINEAGE["L1"]["completion_sha256"]
    for checkpoint in ("anchor", "final"):
        value = identity[checkpoint]
        assert isinstance(value, dict)
        value["checkpoint_sha256"] = EXPECTED_MANIFEST_LINEAGE["L1"][f"{checkpoint}_checkpoint_sha256"]
    identity["anchor"]["checkpoint_sha256"] = "9" * 64
    with pytest.raises(LoggingOnlyPredictionError, match="anchor checkpoint"):
        _validate_frozen_export_lineage(label="L1", export_identity=identity)


def test_state_export_accepts_noncontiguous_cifar_source_ids() -> None:
    _, rows, _ = _prepare_rows(_export(noncontiguous=True))
    assert len(rows) == EXPECTED_COUNT
    assert max(int(row["sample_id"]) for row in rows) == 49999
    assert 44999 not in {int(row["sample_id"]) for row in rows}


def test_frozen_feature_sets_and_history_current_decision_shape() -> None:
    row = {
        "teacher": 0.4,
        "previous_correctness": 1.0,
        "last_margin_risk": 0.6,
        "robust_correct_frequency": 0.7,
        "margin_ema_risk": 0.3,
    }
    assert _features(row, "teacher_only") == [0.4]
    assert _features(row, "student_only") == [0.7, 0.3]
    assert _features(row, "main_effects_plus_product") == pytest.approx([0.4, 0.7, 0.3, 0.28, 0.12])

    rows = [
        {
            **row,
            "namespace": "train",
            "sample_id": index,
            "class_id": index % 10,
            "outcome": int(index % 2 == 0),
            "teacher": 0.1 + 0.8 * (index % 2),
            "robust_correct_frequency": 0.2 + 0.6 * (index % 2),
            "margin_ema_risk": 0.1 + 0.7 * (index % 2),
        }
        for index in range(100)
    ]
    report = _trajectory_report(
        rows,
        design=_FrozenDesign("test", "a" * 64, "b" * 64, split_seed=1, bootstrap_seed=2, bootstrap_replicates=1),
    )
    assert set(report["models"]) == {
        "teacher_only",
        "student_only",
        "main_effects",
        "main_effects_plus_product",
        "current_correctness",
        "instantaneous_margin",
        "current_only",
    }
    assert report["history_vs_best_current"]["history_model"] == "student_only"
    assert set(report["paired_comparisons"]) == {
        "teacher_only_to_student_only",
        "student_only_to_main_effects",
        "main_effects_to_main_effects_plus_product",
    }


def test_parallel_trajectory_reports_match_sequential_fixture() -> None:
    rows = [
        {
            "namespace": "train",
            "sample_id": index,
            "class_id": index % 10,
            "outcome": int(index % 2 == 0),
            "teacher": 0.1 + 0.8 * (index % 2),
            "previous_correctness": float(index % 2),
            "last_margin_risk": 0.2 + 0.6 * (index % 2),
            "robust_correct_frequency": 0.2 + 0.6 * (index % 2),
            "margin_ema_risk": 0.1 + 0.7 * (index % 2),
        }
        for index in range(100)
    ]
    design = _FrozenDesign("test", "a" * 64, "b" * 64, split_seed=1, bootstrap_seed=2, bootstrap_replicates=1)
    prepared = {label: (label, rows, {}) for label in ("L1", "L2", "L3", "L4")}
    assert _trajectory_reports(prepared, design=design, workers=4) == _trajectory_reports(
        prepared, design=design, workers=1
    )


def test_block_rejects_missing_or_swapped_frozen_run_ids() -> None:
    with pytest.raises(LoggingOnlyPredictionError, match="exactly L1"):
        analyze_logging_only_exports({"L1": {"identity": {"run_id": EXPECTED_RUN_IDS["L1"]}}})
    swapped = {label: {"identity": {"run_id": EXPECTED_RUN_IDS[label]}} for label in ("L1", "L2", "L3", "L4")}
    swapped["L1"] = {"identity": {"run_id": EXPECTED_RUN_IDS["L2"]}}
    with pytest.raises(LoggingOnlyPredictionError, match="L1 does not bind"):
        analyze_logging_only_exports(swapped)


@pytest.mark.parametrize("drift", ["membership", "label"])
def test_block_rejects_cross_run_source_membership_or_label_drift(monkeypatch: pytest.MonkeyPatch, drift: str) -> None:
    exports = {label: _export(run_id=EXPECTED_RUN_IDS[label]) for label in ("L1", "L2", "L3", "L4")}
    for label, export in exports.items():
        _rebind_frozen_lineage(monkeypatch, label=label, export=export)
    rows = exports["L2"]["rows"]
    assert isinstance(rows, list) and isinstance(rows[-1], dict)
    if drift == "membership":
        rows[-1]["sample_id"] = 49999
    else:
        rows[-1]["true_label"] = (int(rows[-1]["true_label"]) + 1) % 10
    with pytest.raises(LoggingOnlyPredictionError, match="sample-ID to true-label"):
        analyze_logging_only_exports(exports)


def test_cli_report_binds_exact_consumed_input_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = {}
    arguments = []
    manifest_arguments = []
    for label in ("L1", "L2", "L3", "L4"):
        path = tmp_path / f"{label}.json"
        payload = {"identity": {"run_id": EXPECTED_RUN_IDS[label]}, "rows": []}
        encoded = json.dumps(payload, sort_keys=True).encode()
        path.write_bytes(encoded)
        inputs[label] = (path.resolve(), hashlib.sha256(encoded).hexdigest())
        arguments.extend(["--run", f"{label}={path}"])
        manifest = tmp_path / f"{label}-manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        manifest_arguments.extend(["--manifest", f"{label}={manifest}"])
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        prediction_cli,
        "analyze_logging_only_exports",
        lambda exports, manifest_paths: {"runs": {}, "manifest_inputs": {}},
    )
    assert prediction_cli.main([*arguments, *manifest_arguments, "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["source_inputs"] == {
        label: {"path": str(path), "sha256": digest} for label, (path, digest) in inputs.items()
    }
    parsed, provenance = load_state_export_with_provenance(inputs["L1"][0])
    assert parsed["identity"]["run_id"] == EXPECTED_RUN_IDS["L1"]
    assert provenance == {"path": str(inputs["L1"][0]), "sha256": inputs["L1"][1]}
