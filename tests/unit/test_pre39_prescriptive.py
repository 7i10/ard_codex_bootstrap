from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ard.analysis.pre39_prescriptive import (
    ANCHORS,
    MODEL_NAMES,
    Pre39PrescriptiveError,
    _history_values,
    _scores,
    _top_q,
    analyze_pre39_prescriptive,
    run_pre39_bootstrap,
    write_pre39_report,
)
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS, OUTCOME_EPOCHS, PANEL_EMA_BETA
from ard.analysis.signal_audit import sha256_file

pytestmark = pytest.mark.t1


def _row(sample_id: int, epoch: int, *, domain: str) -> dict[str, object]:
    class_id = sample_id % 10
    anchor_wrong = sample_id in {109, 211} and domain == "feature"
    future_wrong = (sample_id in {3, 17} and domain == "outcome" and epoch in {99, 104}) or (
        sample_id == 109 and domain == "outcome"
    )
    robust_correct = not (anchor_wrong or future_wrong)
    margin = 0.7 if robust_correct else -0.7
    clean_correct, adversarial_correct = sample_id % 3 != 0, sample_id % 4 != 0

    def teacher(prefix: str, correct: bool) -> dict[str, object]:
        true, wrong = (0.8, 0.1) if correct else (0.1, 0.8)
        prediction = class_id if correct else (class_id + 1) % 10
        return {
            f"{prefix}_prediction": prediction,
            f"{prefix}_correct": correct,
            f"{prefix}_true_probability": true,
            f"{prefix}_max_wrong_probability": wrong,
            f"{prefix}_wrong_confidence": wrong - true,
            f"{prefix}_probability_margin": true - wrong,
            f"{prefix}_entropy_normalized": sample_id % 5 / 5,
        }

    clean, adversarial = teacher("teacher_clean", clean_correct), teacher("teacher_adversarial", adversarial_correct)
    return {
        "namespace": "train",
        "sample_id": sample_id,
        "class_id": class_id,
        "epoch": epoch,
        "observation_schema_version": 2,
        "teacher_entropy_normalized": sample_id % 5 / 5,
        "student_probability_margin": margin,
        "student_margin_risk": (1 - margin) / 2,
        "robust_correct": robust_correct,
        **clean,
        **adversarial,
        "teacher_clean_to_adversarial_prediction_flip": clean["teacher_clean_prediction"]
        != adversarial["teacher_adversarial_prediction"],
        "teacher_clean_to_adversarial_true_probability_delta": float(
            adversarial["teacher_adversarial_true_probability"]
        )
        - float(clean["teacher_clean_true_probability"]),
        "teacher_clean_to_adversarial_margin_delta": float(adversarial["teacher_adversarial_probability_margin"])
        - float(clean["teacher_clean_probability_margin"]),
        "student_clean_prediction": class_id,
        "student_clean_correct": True,
        "student_clean_probability_margin": 0.6,
    }


def _lineage(path: Path, observations: Path, *, key: str, protocol: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observation_schema_version": 2,
                "run_id": "L1-seed0",
                "config_hash": "a" * 64,
                "scientific_git_sha": "b" * 40,
                "train_expected_count": 12,
                key: sha256_file(observations),
                "attack_identity": {"epsilon": "8/255", "steps": 10},
                "dataset_identity": {"partition": "train"},
                "teacher": {"registry_id": "teacher-a"},
                protocol: {"seed_domain": "feature" if protocol == "feature_protocol" else "outcome"},
            }
        ),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ids = (3, 17, 29, 41, 53, 67, 79, 83, 97, 109, 211, 307)
    feature = tmp_path / "features.parquet"
    outcome = tmp_path / "outcomes.parquet"
    feature_lineage, outcome_lineage = tmp_path / "features.json", tmp_path / "outcomes.json"
    pq.write_table(
        pa.Table.from_pylist(
            [_row(sample_id, epoch, domain="feature") for epoch in FEATURE_EPOCHS for sample_id in ids]
        ),
        feature,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_row(sample_id, epoch, domain="outcome") for epoch in OUTCOME_EPOCHS for sample_id in ids]
        ),
        outcome,
    )
    _lineage(feature_lineage, feature, key="feature_observations_sha256", protocol="feature_protocol")
    _lineage(outcome_lineage, outcome, key="outcome_observations_sha256", protocol="outcome_protocol")
    return feature, outcome, feature_lineage, outcome_lineage


def _report(tmp_path: Path) -> dict[str, object]:
    feature, outcome, feature_lineage, outcome_lineage = _inputs(tmp_path)
    return analyze_pre39_prescriptive(
        feature_observations=feature,
        outcome_observations=outcome,
        feature_lineage=feature_lineage,
        outcome_lineage=outcome_lineage,
        expected_count=12,
        analysis_provenance={"test": True},
    )


def test_pre39_inclusive_history_lead_time_strata_sparse_ids_and_future_field_exclusion(tmp_path: Path) -> None:
    report = _report(tmp_path)
    assert report["anchors"] == list(ANCHORS)
    anchor4 = report["anchor_reports"]["4"]
    assert anchor4["prior_anchor_only_lead_time_diagnostic"] is None
    assert anchor4["inclusive_history_primary"]["PF"]["models"]["student_history"]["count"] == 10
    assert anchor4["inclusive_history_primary"]["NR"]["models"]["student_history"]["count"] == 2
    anchor9 = report["anchor_reports"]["9"]
    assert anchor9["prior_anchor_only_lead_time_diagnostic"] is not None
    # The routing score declaration and the score-only model fields make the
    # prospective labels audit-only, while PF/NR labels remain report targets.
    assert report["routing_score_contract"] == "outcome_free_midrank_aggregation_v1"
    assert set(anchor9["inclusive_history_primary"]["PF"]["models"]) == set(MODEL_NAMES)
    feature, outcome, feature_lineage, outcome_lineage = _inputs(tmp_path / "leakage")
    before = analyze_pre39_prescriptive(
        feature_observations=feature,
        outcome_observations=outcome,
        feature_lineage=feature_lineage,
        outcome_lineage=outcome_lineage,
        expected_count=12,
        analysis_provenance={"test": True},
    )
    outcome_rows = pq.read_table(outcome).to_pylist()
    for row in outcome_rows:
        if row["sample_id"] == 3 and row["epoch"] in {99, 104}:
            row["robust_correct"] = True
    pq.write_table(pa.Table.from_pylist(outcome_rows), outcome)
    _lineage(outcome_lineage, outcome, key="outcome_observations_sha256", protocol="outcome_protocol")
    after = analyze_pre39_prescriptive(
        feature_observations=feature,
        outcome_observations=outcome,
        feature_lineage=feature_lineage,
        outcome_lineage=outcome_lineage,
        expected_count=12,
        analysis_provenance={"test": True},
    )
    before_models = before["anchor_reports"]["9"]["inclusive_history_primary"]["PF"]["models"]
    after_models = after["anchor_reports"]["9"]["inclusive_history_primary"]["PF"]["models"]
    assert {name: value["top_q_sample_ids_sha256"] for name, value in before_models.items()} == {
        name: value["top_q_sample_ids_sha256"] for name, value in after_models.items()
    }
    assert before_models["student_history"]["prevalence"] != after_models["student_history"]["prevalence"]


def test_pre39_is_row_order_invariant_and_ties_use_stable_sparse_ids(tmp_path: Path) -> None:
    feature, outcome, feature_lineage, outcome_lineage = _inputs(tmp_path)
    original = analyze_pre39_prescriptive(
        feature_observations=feature,
        outcome_observations=outcome,
        feature_lineage=feature_lineage,
        outcome_lineage=outcome_lineage,
        expected_count=12,
        analysis_provenance={"test": True},
    )
    table = pq.read_table(feature)
    pq.write_table(table.take(list(reversed(range(table.num_rows)))), feature)
    _lineage(feature_lineage, feature, key="feature_observations_sha256", protocol="feature_protocol")
    reordered = analyze_pre39_prescriptive(
        feature_observations=feature,
        outcome_observations=outcome,
        feature_lineage=feature_lineage,
        outcome_lineage=outcome_lineage,
        expected_count=12,
        analysis_provenance={"test": True},
    )
    assert original["anchor_reports"] == reordered["anchor_reports"]
    assert _top_q({307: 1.0, 3: 1.0, 109: 0.5, 17: 0.5, 29: 0.0}) == {3}


def test_pre39_predictor_direction_interaction_lineage_and_nonoverwrite(tmp_path: Path) -> None:
    report = _report(tmp_path)
    feature, outcome, feature_lineage, outcome_lineage = _inputs(tmp_path / "second")
    # High entropy is a risk direction and the product interaction is a
    # distinct fixed, outcome-free routing score.
    from ard.analysis.pre39_prescriptive import _validate_inputs

    panel, _, _, _ = _validate_inputs(
        feature_observations=feature,
        outcome_observations=outcome,
        feature_lineage=feature_lineage,
        outcome_lineage=outcome_lineage,
        expected_count=12,
    )
    panel[9][3]["robust_correct"] = False
    panel[9][3]["student_margin"] = -0.7
    inclusive_history = _history_values(panel, anchor=9, inclusive=True)
    lead_history = _history_values(panel, anchor=9, inclusive=False)
    assert inclusive_history is not None and lead_history is not None
    assert inclusive_history["frequency_risk"][3] == pytest.approx(0.5)
    assert lead_history["frequency_risk"][3] == pytest.approx(0.0)
    replay_panel_ema = PANEL_EMA_BETA * 0.7 + (1 - PANEL_EMA_BETA) * -0.7
    assert inclusive_history["margin_ema_risk"][3] == pytest.approx((1 - replay_panel_ema) / 2)
    scores = _scores(panel, anchor=34, inclusive=True)
    assert scores is not None
    assert scores["teacher_entropy"][79] > scores["teacher_entropy"][3]
    assert scores["student_teacher_product"] != scores["student_teacher_additive"]
    output = tmp_path / "report.json"
    write_pre39_report(output=output, report=report)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_pre39_report(output=output, report=report)
    bad = json.loads(feature_lineage.read_text(encoding="utf-8"))
    bad["feature_observations_sha256"] = "0" * 64
    feature_lineage.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(Pre39PrescriptiveError, match="lineage byte"):
        analyze_pre39_prescriptive(
            feature_observations=feature,
            outcome_observations=outcome,
            feature_lineage=feature_lineage,
            outcome_lineage=outcome_lineage,
            expected_count=12,
            analysis_provenance={"test": True},
        )


def test_pre39_bootstrap_is_deterministic_and_resumes(tmp_path: Path) -> None:
    report = _report(tmp_path)
    progress = tmp_path / "progress.json"
    partial = run_pre39_bootstrap(
        report=report,
        anchor=4,
        stratum="PF",
        baseline="instantaneous_margin",
        candidate="student_history",
        output=tmp_path / "partial.json",
        progress=progress,
        max_replicates=50,
    )
    resumed = run_pre39_bootstrap(
        report=report,
        anchor=4,
        stratum="PF",
        baseline="instantaneous_margin",
        candidate="student_history",
        output=tmp_path / "resumed.json",
        progress=progress,
    )
    fresh = run_pre39_bootstrap(
        report=report,
        anchor=4,
        stratum="PF",
        baseline="instantaneous_margin",
        candidate="student_history",
        output=tmp_path / "fresh.json",
        progress=tmp_path / "fresh-progress.json",
    )
    assert partial["partial"]
    assert resumed["partial"] is False
    assert {key: resumed[key] for key in ("completed_replicates", "lower", "upper")} == {
        key: fresh[key] for key in ("completed_replicates", "lower", "upper")
    }
