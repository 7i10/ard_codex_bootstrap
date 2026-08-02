from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ard.analysis.h4a_taxonomy import H4aTaxonomyError, analyze_h4a_collection, analyze_h4a_taxonomy, write_h4a_outputs
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS, OUTCOME_EPOCHS
from ard.analysis.signal_audit import sha256_file
from ard.cli.h4a_taxonomy import main as h4a_main

pytestmark = pytest.mark.t1


def _correct(sample_id: int, epoch: int, *, domain: str) -> bool:
    if sample_id == 1:
        return epoch < (59 if domain == "feature" else 114)
    if sample_id == 2:
        return False
    if sample_id == 3:
        return epoch >= (59 if domain == "feature" else 114)
    if sample_id == 4:
        recover, relapse = (59, 79) if domain == "feature" else (114, 124)
        return recover <= epoch < relapse
    return True


def _row(sample_id: int, epoch: int, *, domain: str) -> dict[str, object]:
    class_id = sample_id % 2
    clean_teacher_correct = sample_id % 3 != 0
    adv_teacher_correct = sample_id % 4 != 0

    def teacher(prefix: str, correct: bool) -> dict[str, object]:
        prediction = class_id if correct else 1 - class_id
        true, wrong = (0.8, 0.1) if correct else (0.1, 0.8)
        return {
            f"{prefix}_prediction": prediction,
            f"{prefix}_correct": correct,
            f"{prefix}_true_probability": true,
            f"{prefix}_max_wrong_probability": wrong,
            f"{prefix}_wrong_confidence": wrong - true,
            f"{prefix}_probability_margin": true - wrong,
            f"{prefix}_entropy_normalized": 0.4,
        }

    robust_correct = _correct(sample_id, epoch, domain=domain)
    margin = 0.4 if robust_correct else -0.4
    clean = teacher("teacher_clean", clean_teacher_correct)
    adversarial = teacher("teacher_adversarial", adv_teacher_correct)
    return {
        "namespace": "train",
        "sample_id": sample_id,
        "class_id": class_id,
        "epoch": epoch,
        "observation_schema_version": 2,
        "teacher_entropy_normalized": adversarial["teacher_adversarial_entropy_normalized"],
        "student_probability_margin": margin,
        "student_margin_risk": (1 - margin) / 2,
        "robust_correct": robust_correct,
        **clean,
        **adversarial,
        "teacher_clean_to_adversarial_prediction_flip": (
            clean["teacher_clean_prediction"] != adversarial["teacher_adversarial_prediction"]
        ),
        "teacher_clean_to_adversarial_true_probability_delta": (
            adversarial["teacher_adversarial_true_probability"] - clean["teacher_clean_true_probability"]
        ),
        "teacher_clean_to_adversarial_margin_delta": (
            adversarial["teacher_adversarial_probability_margin"] - clean["teacher_clean_probability_margin"]
        ),
        "student_clean_prediction": class_id,
        "student_clean_correct": True,
        "student_clean_probability_margin": 0.3,
    }


def _rows(epochs: tuple[int, ...], *, domain: str) -> list[dict[str, object]]:
    return [_row(sample_id, epoch, domain=domain) for epoch in epochs for sample_id in range(10)]


def _lineage(path: Path, observations: Path, *, key: str, protocol: str, run_id: str = "run") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observation_schema_version": 2,
                "run_id": run_id,
                "config_hash": "a" * 64,
                "scientific_git_sha": "b" * 40,
                "train_expected_count": 10,
                key: sha256_file(observations),
                "attack_identity": {"steps": 10},
                "dataset_identity": {"dataset": "cifar10"},
                "teacher": {"registry_id": "teacher"},
                protocol: {"seed_domain": "feature" if protocol == "feature_protocol" else "outcome"},
            }
        ),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    feature, outcome = tmp_path / "feature.parquet", tmp_path / "outcome.parquet"
    feature_lineage, outcome_lineage = tmp_path / "feature.json", tmp_path / "outcome.json"
    pq.write_table(pa.Table.from_pylist(_rows(FEATURE_EPOCHS, domain="feature")), feature)
    pq.write_table(pa.Table.from_pylist(_rows(OUTCOME_EPOCHS, domain="outcome")), outcome)
    _lineage(feature_lineage, feature, key="feature_observations_sha256", protocol="feature_protocol")
    _lineage(outcome_lineage, outcome, key="outcome_observations_sha256", protocol="outcome_protocol")
    return feature, outcome, feature_lineage, outcome_lineage


def _analyze(inputs: tuple[Path, Path, Path, Path]) -> dict[str, object]:
    return analyze_h4a_taxonomy(
        feature_observations=inputs[0],
        outcome_observations=inputs[1],
        feature_lineage=inputs[2],
        outcome_lineage=inputs[3],
        expected_count=10,
        analysis_provenance={"test": True},
    )


def test_h4a_primary_groups_are_exhaustive_disjoint_and_domain_separated(tmp_path: Path) -> None:
    report = _analyze(_inputs(tmp_path))
    early = report["early"]["anchors"]["39"]["primary_groups"]
    assert {group: values["count"] for group, values in early.items()} == {
        "stable_correct": 6,
        "future_forgetting": 1,
        "persistent_wrong": 1,
        "recovered_stable": 1,
        "recovered_relapsed": 1,
    }
    assert all(values["same_panel_oracle_headroom"]["denominator_train_panel"] == 10 for values in early.values())
    assert early["future_forgetting"]["pre_anchor_student_margin_trend"]["available"]
    late = report["late"]["anchors"]["99"]["primary_groups"]
    assert all(not values["pre_anchor_student_margin_trend"]["available"] for values in late.values())
    assert report["input_identity"]["feature_attack_domain"]["seed_domain"] == "feature"
    assert report["input_identity"]["outcome_attack_domain"]["seed_domain"] == "outcome"
    group = early["future_forgetting"]
    assert set(group["teacher_continuous"]) == {
        "adversarial_wrong_confidence",
        "adversarial_probability_margin",
        "clean_wrong_confidence",
        "clean_probability_margin",
    }
    assert "stable_sample_ids_sha256" in group and "class_counts" in group and "cross_tabs" in group


def test_h4a_is_row_order_invariant_and_never_uses_cross_domain_transition(tmp_path: Path) -> None:
    feature, outcome, feature_lineage, outcome_lineage = _inputs(tmp_path)
    rows = pq.read_table(feature).to_pylist()
    # Feature-domain epoch-99 disagreement is a feature outcome only: late
    # taxonomy still starts from the independent outcome-domain epoch-99 row.
    for row in rows:
        if row["sample_id"] == 0 and row["epoch"] == 99:
            row["robust_correct"] = False
            row["student_probability_margin"] = -0.4
            row["student_margin_risk"] = 0.7
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), feature)
    _lineage(feature_lineage, feature, key="feature_observations_sha256", protocol="feature_protocol")
    report = _analyze((feature, outcome, feature_lineage, outcome_lineage))
    assert report["early"]["anchors"]["79"]["primary_groups"]["future_forgetting"]["count"] == 1
    assert report["late"]["anchors"]["99"]["primary_groups"]["stable_correct"]["count"] == 6


def test_h4a_accepts_sparse_original_dataset_ids_with_exact_panel_joins(tmp_path: Path) -> None:
    feature, outcome, feature_lineage, outcome_lineage = _inputs(tmp_path)
    for path, lineage, key, protocol in (
        (feature, feature_lineage, "feature_observations_sha256", "feature_protocol"),
        (outcome, outcome_lineage, "outcome_observations_sha256", "outcome_protocol"),
    ):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row["sample_id"] = int(row["sample_id"]) + 40
        pq.write_table(pa.Table.from_pylist(rows), path)
        _lineage(lineage, path, key=key, protocol=protocol)
    report = _analyze((feature, outcome, feature_lineage, outcome_lineage))
    assert report["early"]["anchors"]["39"]["primary_groups"]["stable_correct"]["count"] == 6


def test_h4a_rejects_schema_algebra_and_lineage_drift(tmp_path: Path) -> None:
    feature, outcome, feature_lineage, outcome_lineage = _inputs(tmp_path)
    rows = pq.read_table(feature).to_pylist()
    rows[0]["teacher_adversarial_wrong_confidence"] = 0.99
    pq.write_table(pa.Table.from_pylist(rows), feature)
    _lineage(feature_lineage, feature, key="feature_observations_sha256", protocol="feature_protocol")
    with pytest.raises(H4aTaxonomyError, match="wrong confidence algebra"):
        _analyze((feature, outcome, feature_lineage, outcome_lineage))

    feature, outcome, feature_lineage, outcome_lineage = _inputs(tmp_path / "lineage")
    bad = json.loads(outcome_lineage.read_text())
    bad["config_hash"] = "c" * 64
    outcome_lineage.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(H4aTaxonomyError, match="identity drifted"):
        _analyze((feature, outcome, feature_lineage, outcome_lineage))


def test_h4a_collection_jaccard_and_blinded_manifest_are_local_diagnostics(tmp_path: Path) -> None:
    report = _analyze(_inputs(tmp_path))
    collection = analyze_h4a_collection({"seed1": report, "seed2": report})
    jaccard = collection["cross_seed_jaccard"]["teacher"]
    assert jaccard["status"] == "available"
    assert all(value in {None, 1.0} for value in jaccard["pairs"][0]["groups"].values())
    assert collection["blinded_panel"]
    assert all("class_id" not in row and row["diagnostic_only"] for row in collection["blinded_panel"])
    paths = write_h4a_outputs(output_dir=tmp_path / "output", collection=collection)
    manifest = json.loads(paths["blinded_manifest"].read_text())
    assert manifest["contains_images"] is False
    assert manifest["contains_label_corrections"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        write_h4a_outputs(output_dir=tmp_path / "output", collection=collection)


def test_h4a_cli_requires_the_exact_cifar10_train_panel_size(tmp_path: Path) -> None:
    with pytest.raises(H4aTaxonomyError, match="exactly 45,000"):
        h4a_main(
            [
                "--feature-observations",
                "L1=feature.parquet",
                "--outcome-observations",
                "L1=outcome.parquet",
                "--feature-lineage",
                "L1=feature.json",
                "--outcome-lineage",
                "L1=outcome.json",
                "--expected-count",
                "10",
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
