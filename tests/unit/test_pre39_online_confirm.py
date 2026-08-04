from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ard.analysis.pre39_online_confirm import (
    Pre39OnlineConfirmError,
    analyze_pre39_online_confirm,
    run_pre39_online_bootstrap,
    write_pre39_online_confirm,
)
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS, OUTCOME_EPOCHS
from ard.analysis.signal_audit import sha256_file

pytestmark = pytest.mark.t1


def _teacher(prefix: str, class_id: int) -> dict[str, object]:
    return {
        f"{prefix}_prediction": class_id,
        f"{prefix}_correct": True,
        f"{prefix}_true_probability": 0.8,
        f"{prefix}_max_wrong_probability": 0.1,
        f"{prefix}_wrong_confidence": -0.7,
        f"{prefix}_probability_margin": 0.7,
        f"{prefix}_entropy_normalized": 0.2 + class_id / 20,
    }


def _row(sample_id: int, epoch: int, *, correct: bool) -> dict[str, object]:
    class_id = sample_id % 10
    return {
        "namespace": "train",
        "sample_id": sample_id,
        "class_id": class_id,
        "epoch": epoch,
        "observation_schema_version": 2,
        "teacher_entropy_normalized": 0.2 + class_id / 20,
        "student_probability_margin": 0.7 if correct else -0.7,
        "student_margin_risk": 0.15 if correct else 0.85,
        "robust_correct": correct,
        **_teacher("teacher_clean", class_id),
        **_teacher("teacher_adversarial", class_id),
        "teacher_clean_to_adversarial_prediction_flip": False,
        "teacher_clean_to_adversarial_true_probability_delta": 0.0,
        "teacher_clean_to_adversarial_margin_delta": 0.0,
        "student_clean_prediction": class_id,
        "student_clean_correct": True,
        "student_clean_probability_margin": 0.7,
    }


def _lineage(path: Path, *, kind: str, observations: Path, count: int) -> None:
    key = f"{kind}_observations_sha256"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observation_schema_version": 2,
                "run_id": "run",
                "config_hash": "a" * 64,
                "scientific_git_sha": "b" * 40,
                "seed": 1,
                "train_expected_count": count,
                key: sha256_file(observations),
                "attack_identity": {"steps": 10},
                "dataset_identity": {"name": "cifar10"},
                "teacher": {"registry_id": "teacher", "checkpoint_sha256": "e" * 64},
                f"{kind}_protocol": {"domain": kind},
            }
        ),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ids = tuple(range(20, 40))
    online_current = {sample_id: sample_id % 2 == 0 for sample_id in ids}
    future_wrong = {
        (epoch, sample_id): (sample_id % 4 == 0 if epoch in {99, 104} else sample_id % 6 == 0)
        for epoch in OUTCOME_EPOCHS
        for sample_id in ids
    }
    feature = tmp_path / "feature.parquet"
    outcome = tmp_path / "outcome.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [_row(item, epoch, correct=online_current[item]) for epoch in FEATURE_EPOCHS for item in ids]
        ),
        feature,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_row(item, epoch, correct=not future_wrong[epoch, item]) for epoch in OUTCOME_EPOCHS for item in ids]
        ),
        outcome,
    )
    feature_lineage, outcome_lineage = tmp_path / "feature.json", tmp_path / "outcome.json"
    _lineage(feature_lineage, kind="feature", observations=feature, count=len(ids))
    _lineage(outcome_lineage, kind="outcome", observations=outcome, count=len(ids))
    online = tmp_path / "online.parquet"
    online_rows = [
        {
            "namespace": "train",
            "sample_id": item,
            "class_id": item % 10,
            "anchor_epoch": 34,
            "robust_correct_count": (item - 20) % 35,
            "robust_correct_frequency_inclusive": ((item - 20) % 35) / 35,
            "margin_ema": -0.9 + (item - 20) / 11,
            "last_margin": 0.1,
            "current_robust_correct": online_current[item],
        }
        for item in ids
    ]
    pq.write_table(pa.Table.from_pylist(online_rows), online)
    online_lineage = tmp_path / "online.json"
    online_lineage.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "pre39_online_state_candidate_v1",
                "run_id": "run",
                "config_hash": "a" * 64,
                "world_size": 1,
                "scientific_git_sha": "b" * 40,
                "seed": 1,
                "teacher": {"registry_id": "teacher", "checkpoint_sha256": "e" * 64},
                "dataset_identity": {"name": "cifar10"},
                "attack_identity": {"steps": 10},
                "anchor_epoch": 34,
                "expected_count": len(ids),
                "row_count": len(ids),
                "checkpoint_sha256": "f" * 64,
                "feature_observations_sha256": sha256_file(feature),
                "feature_lineage_sha256": sha256_file(feature_lineage),
                "analysis_provenance": {"test": True},
                "observations_sha256": sha256_file(online),
            }
        ),
        encoding="utf-8",
    )
    return {
        "online_observations": online,
        "online_lineage": online_lineage,
        "feature_observations": feature,
        "feature_lineage": feature_lineage,
        "outcome_observations": outcome,
        "outcome_lineage": outcome_lineage,
    }


def _report(paths: dict[str, Path]) -> dict[str, object]:
    return analyze_pre39_online_confirm(**paths, expected_count=20, analysis_provenance={"test": True})


def test_exact_online_confirmation_scores_frozen_outcomes_and_nonoverwrites(tmp_path: Path) -> None:
    report = _report(_inputs(tmp_path))
    assert report["anchor_epoch"] == 34
    assert set(report["strata"]) == {"PF", "NR"}
    for stratum in report["strata"].values():
        assert set(stratum["models"]) == {"exact_online_student", "instantaneous_margin", "teacher_entropy"}
        assert stratum["models"]["exact_online_student"]["q"] == 0.10
    output = tmp_path / "report.json"
    assert write_pre39_online_confirm(output=output, report=report) == output
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_pre39_online_confirm(output=output, report=report)
    assert "_bootstrap_rows" not in json.loads(output.read_text(encoding="utf-8"))


def test_exact_online_confirmation_rejects_hash_and_temporal_drift(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    meta = json.loads(paths["online_lineage"].read_text(encoding="utf-8"))
    meta["observations_sha256"] = "0" * 64
    paths["online_lineage"].write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(Pre39OnlineConfirmError, match="observations hash"):
        _report(paths)
    paths = _inputs(tmp_path / "temporal")
    meta = json.loads(paths["online_lineage"].read_text(encoding="utf-8"))
    meta["anchor_epoch"] = 29
    paths["online_lineage"].write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(Pre39OnlineConfirmError, match="anchor/count"):
        _report(paths)


def test_exact_online_paired_bootstrap_resumes_deterministically(tmp_path: Path) -> None:
    report = _report(_inputs(tmp_path))
    progress = tmp_path / "progress.json"
    partial = run_pre39_online_bootstrap(
        report=report,
        stratum="PF",
        baseline="instantaneous_margin",
        output=tmp_path / "partial.json",
        progress=progress,
        max_replicates=5,
    )
    complete = run_pre39_online_bootstrap(
        report=report,
        stratum="PF",
        baseline="instantaneous_margin",
        output=tmp_path / "complete.json",
        progress=progress,
    )
    assert partial["partial"] is True
    assert complete["completed_replicates"] <= 2000
    assert complete["partial"] is False
