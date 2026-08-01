"""Focused contracts for the post-H2 fixed selector boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ard.analysis import intervention_selector as selector
from ard.analysis.signal_audit import _fit_logistic, _predict_logistic, binary_metrics, deterministic_hash_split

pytestmark = pytest.mark.unit


def _feature_row(sample_id: int, class_id: int) -> dict[str, object]:
    row: dict[str, object] = {column: 0.0 for column in selector.FEATURE_COLUMNS}
    row.update(
        {
            "namespace": "train",
            "sample_id": sample_id,
            "class_id": class_id,
            "feature_epoch": selector.ANCHOR_EPOCH,
            "teacher_entropy_normalized": 0.5,
            "student_robust_correct_epoch99": 1,
            "student_robust_correct_frequency": (sample_id % 7) / 6,
            "student_margin_historical_ema": 0.0,
            "student_margin_historical_risk": (sample_id % 5) / 4,
            "student_margin_instantaneous_epoch99": 0.0,
            "student_margin_panel_ema": 0.0,
            "student_margin_panel_risk": 0.0,
            "student_margin_epoch99": 0.0,
            "student_margin_risk_epoch99": 0.0,
        }
    )
    return row


def _outcome_row(sample_id: int, class_id: int) -> dict[str, object]:
    return {
        "namespace": "train",
        "sample_id": sample_id,
        "class_id": class_id,
        "outcome_start_epoch": selector.ANCHOR_EPOCH,
        "outcome_end_epoch": 199,
        "checkpoint_panel_forgetting": int(sample_id % 3 == 0),
        "checkpoint_panel_transition_count": 0,
        "final_robust_error": 0,
        "persistent_wrong": 0,
        "post_anchor_robust_correct_frequency": 1.0,
    }


def _fixture_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> selector.SelectorFiles:
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(selector, "TRAIN_COUNT", 30)
    monkeypatch.setattr(selector, "K", 6)
    monkeypatch.setattr(selector, "NUM_CLASSES", 3)
    analysis_provenance = {
        "git": {"sha": "f" * 40, "dirty": False},
        "source_files": {"replay": "1" * 64},
        "source_sha256": "2" * 64,
    }
    monkeypatch.setattr(selector, "tracked_clean_analysis_provenance", lambda: analysis_provenance)
    feature_rows = [_feature_row(index, index % 3) for index in range(30)]
    outcome_rows = [_outcome_row(index, index % 3) for index in range(30)]
    feature, outcome, l3_feature = (
        tmp_path / "seed0-feature.parquet",
        tmp_path / "seed0-outcome.parquet",
        tmp_path / "l3-feature.parquet",
    )
    # Arrow infers the heterogeneous concrete schema correctly; selector
    # checks names, rather than an implementation-specific physical type.
    pq.write_table(pa.Table.from_pylist(feature_rows), feature)
    pq.write_table(pa.Table.from_pylist(outcome_rows), outcome)
    pq.write_table(pa.Table.from_pylist(feature_rows), l3_feature)
    rows = [
        {
            "namespace": "train",
            "sample_id": row["sample_id"],
            "class_id": row["class_id"],
            "student_robust_correct_frequency": row["student_robust_correct_frequency"],
            "student_margin_historical_risk": row["student_margin_historical_risk"],
            "outcome": int(outcome_rows[index]["checkpoint_panel_forgetting"]),
        }
        for index, row in enumerate(feature_rows)
    ]
    train_ids, held_ids = deterministic_hash_split(
        rows, seed=selector.SPLIT_SEED, held_out_fraction=selector.HELD_OUT_FRACTION
    )
    train, held = (
        [row for row in rows if row["sample_id"] in train_ids],
        [row for row in rows if row["sample_id"] in held_ids],
    )
    fit = _fit_logistic(
        [[float(row[field]) for field in selector.FEATURE_NAMES] for row in train],
        [int(row["outcome"]) for row in train],
    )
    metrics = binary_metrics(
        [int(row["outcome"]) for row in held],
        _predict_logistic(fit, [[float(row[field]) for field in selector.FEATURE_NAMES] for row in held]),
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "outcome": "checkpoint_panel_forgetting",
                "split_identity": {"train_sample_ids": list(train_ids), "held_out_sample_ids": list(held_ids)},
                "models": {"history_only": metrics},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lineage = tmp_path / "seed0-lineage.json"
    feature_domain = {
        "attack_identity": {"steps": 10, "loss": "kl"},
        "feature_protocol": {
            "base_seed": 2026073103,
            "seed_domain": "feature",
            "seed_formula": "frozen",
            "backend": "torch-cuda",
            "precision": "fp32",
            "batch_size": 128,
            "epochs": [4, 99],
            "panel_ema_beta": 0.5904900000000001,
        },
        "checkpoint_training": {"world_size": 1, "execution_identity": {"world_size": 1, "per_rank_batch_size": 128}},
        "dataset_identity": {"dataset": {"name": "cifar10"}, "split_seed": 20260722},
        "teacher": {
            "registry_id": "bartoldson2024_adversarial_wrn94_16",
            "checkpoint_sha256": selector.L3_TEACHER_CHECKPOINT_SHA256,
        },
    }
    lineage.write_text(
        json.dumps(
            {
                "output_parquet_sha256": {
                    "feature_panel": selector.sha256_file(feature),
                    "outcome_panel": selector.sha256_file(outcome),
                },
                "predictive_audit_sha256": selector.sha256_file(report),
                "train_expected_count": 30,
                **feature_domain,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    l3_lineage = tmp_path / "l3-lineage.json"
    l3_lineage.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "l3_checkpoint_panel_feature_source_v1",
                "run_id": "bart-rslad-observed-s2-confirm-v2",
                "teacher_registry_id": "bartoldson2024_adversarial_wrn94_16",
                "seed": 2,
                "scientific_git_sha": selector.L3_SCIENTIFIC_GIT_SHA,
                "config_hash": selector.L3_CONFIG_SHA256,
                "parent_epoch": 99,
                "parent_checkpoint_sha256": selector.L3_PARENT_CHECKPOINT_SHA256,
                "parent_sample_state_sha256": selector.L3_PARENT_SAMPLE_STATE_SHA256,
                "parent_raw_config_sha256": selector.L3_CONFIG_SHA256,
                "feature_panel_sha256": selector.sha256_file(l3_feature),
                "train_expected_count": 30,
                "analysis_provenance": analysis_provenance,
                **feature_domain,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(selector, "SEED0_FEATURE_PANEL_SHA256", selector.sha256_file(feature))
    monkeypatch.setattr(selector, "SEED0_OUTCOME_PANEL_SHA256", selector.sha256_file(outcome))
    monkeypatch.setattr(selector, "SEED0_REPORT_SHA256", selector.sha256_file(report))
    monkeypatch.setattr(selector, "SEED0_LINEAGE_SHA256", selector.sha256_file(lineage))
    return selector.SelectorFiles(feature, outcome, report, lineage, l3_feature, l3_lineage)


def test_selector_reproduces_fit_masks_and_class_matched_random_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _fixture_files(tmp_path, monkeypatch)
    created = selector.build_selector_bundle(files=files, output_dir=tmp_path / "out")
    verified = selector.verify_selector_bundle(
        bundle_path=created["bundle"],
        expected_parent={
            "parent_checkpoint_sha256": selector.L3_PARENT_CHECKPOINT_SHA256,
            "parent_sample_state_sha256": selector.L3_PARENT_SAMPLE_STATE_SHA256,
            "parent_raw_config_sha256": selector.L3_CONFIG_SHA256,
        },
    )
    assert verified["selection"]["k"] == 6
    assert verified["history"]["selected_count"] == verified["random"]["selected_count"] == 6
    assert verified["history"]["selected_class_counts"] == verified["random"]["selected_class_counts"]
    bundle = json.loads(created["bundle"].read_text(encoding="utf-8"))
    assert len(bundle["seed0_fit"]["weights"]) == 3
    assert len(bundle["seed0_fit"]["coefficients_sha256"]) == 64
    assert bundle["selection"]["order"] == "descending_probability_then_sample_id_ascending"
    assert bundle["random_control"]["numpy_rng"] is False


def test_selector_rejects_seed0_hash_drift_leakage_and_same_budget_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _fixture_files(tmp_path, monkeypatch)
    created = selector.build_selector_bundle(files=files, output_dir=tmp_path / "out")
    history = json.loads(created["history_mask"].read_text(encoding="utf-8"))
    labels = {row["sample_id"]: row["class_id"] for row in pq.read_table(files.l3_feature_panel).to_pylist()}
    replacement = next(
        sample_id
        for sample_id, label in labels.items()
        if label == labels[history["selected_ids"][0]] and sample_id not in history["selected_ids"]
    )
    history["selected_ids"][0] = replacement
    history["selected_ids"].sort()
    history["selected_ids_sha256"] = hashlib.sha256(selector.canonical_json(history["selected_ids"])).hexdigest()
    created["history_mask"].write_bytes(selector.canonical_json(history))
    with pytest.raises(selector.SelectorBundleError, match="history mask"):
        selector.verify_selector_bundle(bundle_path=created["bundle"])
    feature = files.seed0_feature_panel
    feature.write_bytes(feature.read_bytes() + b"drift")
    with pytest.raises(selector.SelectorBundleError, match="seed-0 feature panel"):
        selector.build_selector_bundle(files=files, output_dir=tmp_path / "another")


def test_selector_rejects_l3_protocol_or_hex_identity_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = _fixture_files(tmp_path, monkeypatch)
    lineage = json.loads(files.l3_lineage.read_text(encoding="utf-8"))
    # The line is deliberately reserialized (hence re-hashed) before the
    # builder sees it.  Domain parity, not a stale lineage-file hash, rejects
    # an attack change.
    lineage["attack_identity"]["steps"] = 9
    files.l3_lineage.write_text(json.dumps(lineage, sort_keys=True), encoding="utf-8")
    with pytest.raises(selector.SelectorBundleError, match="attack_identity"):
        selector.build_selector_bundle(files=files, output_dir=tmp_path / "protocol-drift")

    files = _fixture_files(tmp_path / "second", monkeypatch)
    lineage = json.loads(files.l3_lineage.read_text(encoding="utf-8"))
    lineage["config_hash"] = "A" * 64
    files.l3_lineage.write_text(json.dumps(lineage, sort_keys=True), encoding="utf-8")
    with pytest.raises(selector.SelectorBundleError, match="registered source"):
        selector.build_selector_bundle(files=files, output_dir=tmp_path / "hex-drift")

    files = _fixture_files(tmp_path / "third", monkeypatch)
    lineage = json.loads(files.l3_lineage.read_text(encoding="utf-8"))
    lineage["analysis_provenance"]["source_sha256"] = "3" * 64
    # Reserializing the lineage also reseals its file bytes.  The rejection is
    # therefore against provenance self-declaration rather than a stale hash.
    files.l3_lineage.write_text(json.dumps(lineage, sort_keys=True), encoding="utf-8")
    with pytest.raises(selector.SelectorBundleError, match="analysis provenance"):
        selector.build_selector_bundle(files=files, output_dir=tmp_path / "provenance-drift")
