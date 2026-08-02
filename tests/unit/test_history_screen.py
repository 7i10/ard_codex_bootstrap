from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ard.analysis import history_screen
from ard.analysis.history_screen import HistoryScreenError, analyze_history_screen, build_late_collection
from ard.analysis.intervention_selector import FEATURE_COLUMNS
from ard.analysis.signal_audit import canonical_json, sha256_file
from ard.cli.history_screen import main

pytestmark = pytest.mark.t1


def _feature(sample_id: int) -> dict[str, object]:
    return {
        "namespace": "train",
        "sample_id": sample_id,
        "class_id": sample_id % 2,
        "feature_epoch": 99,
        "teacher_entropy_normalized": 0.5,
        "student_robust_correct_epoch99": int(sample_id < 10),
        "student_robust_correct_frequency": 1.0 - sample_id / 12.0,
        "student_margin_historical_ema": 0.0,
        "student_margin_historical_risk": sample_id / 12.0,
        "student_margin_instantaneous_epoch99": 0.0,
        "student_margin_panel_ema": 0.0,
        "student_margin_panel_risk": 0.0,
        "student_margin_epoch99": 0.0,
        "student_margin_risk_epoch99": 0.0,
    }


def _state_rows() -> list[dict[str, object]]:
    return [
        {
            "namespace": "train",
            "sample_id": sample_id,
            "true_label": sample_id % 2,
            "anchor_epoch": 99,
            "final_epoch": 199,
            "future_online_forgetting": int(sample_id in {8, 9}),
            "anchor_seen": 100,
            "anchor_robust_correct_count": 80,
            "anchor_previous_robust_correct": sample_id < 10,
            "anchor_margin_ema": 0.1,
            "anchor_last_margin": 0.2,
            "subsequent_forgetting_increment": int(sample_id in {8, 9}),
            "anchor_forgetting_count": 2,
            "final_forgetting_count": 2 + int(sample_id in {8, 9}),
        }
        for sample_id in range(12)
    ]


def _fit(domain: dict[str, object]) -> dict[str, object]:
    weights, means, scales = [0.0, -1.0, 2.0], [0.0, 0.0], [1.0, 1.0]
    fit_payload = {
        "feature_names": ["student_robust_correct_frequency", "student_margin_historical_risk"],
        "weights": weights,
        "means": means,
        "scales": scales,
    }
    return {
        "schema_version": 1,
        "contract": "h5_frozen_fixed_fit_bundle_v1",
        "feature_names": fit_payload["feature_names"],
        "weights": weights,
        "means": means,
        "scales": scales,
        "predictor_spec_sha256": "a" * 64,
        "coefficients_sha256": hashlib.sha256(canonical_json(weights)).hexdigest(),
        "preprocessing_sha256": hashlib.sha256(canonical_json({"means": means, "scales": scales})).hexdigest(),
        "fit_sha256": hashlib.sha256(canonical_json(fit_payload)).hexdigest(),
        "seed0_input_lineage_hashes": {"feature_panel": "b" * 64, "outcome_panel": "c" * 64},
        "fit_domain": domain,
        "training_outcome": "checkpoint_panel_forgetting",
    }


def _files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    features = [_feature(sample_id) for sample_id in range(12)]
    feature_path, state_path, lineage_path, fit_path = (
        tmp_path / "feature.parquet",
        tmp_path / "state.json",
        tmp_path / "lineage.json",
        tmp_path / "fit.json",
    )
    pq.write_table(pa.Table.from_pylist(features).select(FEATURE_COLUMNS), feature_path)
    domain = {
        "attack_identity": {"loss": "kl", "steps": 10},
        "dataset_identity": {"dataset": "cifar10"},
        "teacher": {"registry_id": "teacher", "checkpoint_sha256": "d" * 64},
    }
    lineage = {
        "schema_version": 1,
        "run_id": "run-1",
        "config_hash": "e" * 64,
        "scientific_git_sha": "f" * 40,
        "seed": 1,
        "train_expected_count": 12,
        "output_parquet_sha256": {"feature_panel": sha256_file(feature_path)},
        "checkpoints": [{"epoch": 99, "sha256": "1" * 64}],
        "analysis_provenance": {"git": {"sha": "2" * 40, "dirty": False}},
        **domain,
    }
    state = {
        "identity": {
            "contract": "logging_only_exact_state_anchor99_final199_v1",
            "run_id": "run-1",
            "config_hash": "e" * 64,
            "expected_count": 12,
            "row_count": 12,
            "anchor": {"epoch": 99, "checkpoint_sha256": "1" * 64},
            "final": {"epoch": 199, "checkpoint_sha256": "3" * 64},
        },
        "rows": _state_rows(),
    }
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    fit_path.write_text(json.dumps(_fit(domain)), encoding="utf-8")
    return feature_path, state_path, lineage_path, fit_path


def _provenance() -> dict[str, object]:
    return {"git": {"sha": "4" * 40, "dirty": False}, "source_files": {"history": "5" * 64}, "source_sha256": "6" * 64}


def _late_cohort(path: Path) -> dict[str, dict[str, object]]:
    runs = {
        "L1": {
            "run_id": "l1",
            "config_hash": "1" * 64,
            "scientific_git_sha": "a" * 40,
            "seed": 1,
            "teacher_registry_id": "bartoldson2024_adversarial_wrn94_16",
        },
        "L2": {
            "run_id": "l2",
            "config_hash": "2" * 64,
            "scientific_git_sha": "b" * 40,
            "seed": 1,
            "teacher_registry_id": "chen2021_ltd_wrn34_10",
        },
        "L3": {
            "run_id": "l3",
            "config_hash": "3" * 64,
            "scientific_git_sha": "c" * 40,
            "seed": 2,
            "teacher_registry_id": "bartoldson2024_adversarial_wrn94_16",
        },
        "L4": {
            "run_id": "l4",
            "config_hash": "4" * 64,
            "scientific_git_sha": "d" * 40,
            "seed": 2,
            "teacher_registry_id": "chen2021_ltd_wrn34_10",
        },
    }
    path.write_text(json.dumps({"schema_version": 1, "contract": "h5_confirmatory_cohort_inventory_v1", "runs": runs}))
    return runs


def _late_reports(identities: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    def metrics(auroc: float, precision: float) -> dict[str, float]:
        return {"auroc": auroc, "precision_at_k": precision}

    return {
        label: {
            "input_identity": identity,
            "fixed_score": {"metrics": metrics(0.80, 0.50)},
            "deployable_online_scores": {
                "online_rank_inclusive": {"metrics": metrics(0.82, 0.50)},
                "online_instantaneous_margin": {"metrics": metrics(0.79, 0.50)},
            },
            "_bootstrap_rows": [
                {"sample_id": i, "class_id": i % 10, "outcome": int(i < 10), "baseline": i / 30, "candidate": i / 25}
                for i in range(30)
            ],
        }
        for label, identity in identities.items()
    }


def test_late_point_gate_requires_all_runs_and_creates_only_paired_bartoldson_tasks(tmp_path: Path) -> None:
    identities = _late_cohort(tmp_path / "cohort.json")
    reports = _late_reports(identities)
    gate, tasks, _ = build_late_collection(cohort_inventory=tmp_path / "cohort.json", reports=reports)
    assert gate["status"] == "point_gate_pass_bootstrap_pending"
    assert {task["run"] for task in tasks} == {"L1", "L3"}
    reports["L2"]["deployable_online_scores"]["online_rank_inclusive"]["metrics"]["auroc"] = 0.70
    gate, tasks, _ = build_late_collection(cohort_inventory=tmp_path / "cohort.json", reports=reports)
    assert gate["status"] == "no_go_point_gate"
    assert tasks == []


def test_late_empty_anchor_correct_population_reports_degenerate_metrics(tmp_path: Path) -> None:
    feature, state, lineage, fit = _files(tmp_path)
    raw = json.loads(state.read_text())
    for row in raw["rows"]:
        row["anchor_previous_robust_correct"] = False
    state.write_text(json.dumps(raw))
    report = analyze_history_screen(
        feature_panel=feature,
        online_state_export=state,
        replay_lineage=lineage,
        frozen_fit=fit,
        expected_count=12,
        analysis_provenance=_provenance(),
    )
    assert report["population"]["anchor_online_correct_rows"] == 0
    assert report["deployable_online_scores"]["online_rank_inclusive"]["metrics"]["auroc"] is None


def test_h5_late_uses_one_replay_feature_panel_and_online_forgetting_outcome(tmp_path: Path) -> None:
    feature, state, lineage, fit = _files(tmp_path)
    report = analyze_history_screen(
        feature_panel=feature,
        online_state_export=state,
        replay_lineage=lineage,
        frozen_fit=fit,
        expected_count=12,
        analysis_provenance=_provenance(),
    )
    assert report["evaluation_outcome"] == "online_future_forgetting"
    assert report["fixed_score"]["training_outcome"] == "checkpoint_panel_forgetting"
    assert report["population"] == {
        "all_rows": 12,
        "anchor_online_correct_rows": 10,
        "anchor_wrong_excluded_rows": 2,
        "k": 1,
    }
    assert report["fixed_score"]["metrics"]["precision_at_k"] == 1.0
    assert report["adaptive_score"]["metrics"]["precision_at_k"] == 1.0
    assert report["overlap"]["groups"]["common"]["count"] == 1


def test_h5_late_rejects_state_checkpoint_mismatch_and_fit_domain_drift(tmp_path: Path) -> None:
    feature, state, lineage, fit = _files(tmp_path)
    state_json = json.loads(state.read_text(encoding="utf-8"))
    state_json["identity"]["anchor"]["checkpoint_sha256"] = "0" * 64
    state.write_text(json.dumps(state_json), encoding="utf-8")
    with pytest.raises(HistoryScreenError, match="share the epoch-99 checkpoint"):
        analyze_history_screen(
            feature_panel=feature,
            online_state_export=state,
            replay_lineage=lineage,
            frozen_fit=fit,
            expected_count=12,
            analysis_provenance=_provenance(),
        )

    feature, state, lineage, fit = _files(tmp_path / "second")
    fit_json = json.loads(fit.read_text(encoding="utf-8"))
    fit_json["fit_domain"]["teacher"] = {"registry_id": "wrong"}
    fit.write_text(json.dumps(fit_json), encoding="utf-8")
    with pytest.raises(HistoryScreenError, match="teacher does not match"):
        analyze_history_screen(
            feature_panel=feature,
            online_state_export=state,
            replay_lineage=lineage,
            frozen_fit=fit,
            expected_count=12,
            analysis_provenance=_provenance(),
        )


def test_h5_late_accepts_actual_feature_only_lineage_and_rejects_hash_binding_ambiguity(tmp_path: Path) -> None:
    feature, state, lineage, fit = _files(tmp_path)
    lineage_json = json.loads(lineage.read_text(encoding="utf-8"))
    lineage_json.pop("output_parquet_sha256")
    lineage_json["kind"] = "l3_checkpoint_panel_feature_source_v1"
    lineage_json["feature_panel_sha256"] = sha256_file(feature)
    lineage.write_text(json.dumps(lineage_json), encoding="utf-8")
    report = analyze_history_screen(
        feature_panel=feature,
        online_state_export=state,
        replay_lineage=lineage,
        frozen_fit=fit,
        expected_count=12,
        analysis_provenance=_provenance(),
    )
    assert report["input_identity"]["run_id"] == "run-1"

    lineage_json["output_parquet_sha256"] = {"feature_panel": sha256_file(feature)}
    lineage.write_text(json.dumps(lineage_json), encoding="utf-8")
    with pytest.raises(HistoryScreenError, match="exactly one top-level"):
        analyze_history_screen(
            feature_panel=feature,
            online_state_export=state,
            replay_lineage=lineage,
            frozen_fit=fit,
            expected_count=12,
            analysis_provenance=_provenance(),
        )


def test_h5_late_rejects_out_of_namespace_feature_and_state_ids(tmp_path: Path) -> None:
    feature, state, lineage, fit = _files(tmp_path)
    table = pq.read_table(feature).to_pylist()
    table[0]["sample_id"] = 50_000
    pq.write_table(pa.Table.from_pylist(table).select(FEATURE_COLUMNS), feature)
    lineage_json = json.loads(lineage.read_text(encoding="utf-8"))
    lineage_json["output_parquet_sha256"]["feature_panel"] = sha256_file(feature)
    lineage.write_text(json.dumps(lineage_json), encoding="utf-8")
    with pytest.raises(HistoryScreenError, match="epoch-99 stable-ID"):
        analyze_history_screen(
            feature_panel=feature,
            online_state_export=state,
            replay_lineage=lineage,
            frozen_fit=fit,
            expected_count=12,
            analysis_provenance=_provenance(),
        )

    feature, state, lineage, fit = _files(tmp_path / "state")
    state_json = json.loads(state.read_text(encoding="utf-8"))
    state_json["rows"][0]["true_label"] = 10
    state.write_text(json.dumps(state_json), encoding="utf-8")
    with pytest.raises(HistoryScreenError, match="exact 99-to-199 stable-ID"):
        analyze_history_screen(
            feature_panel=feature,
            online_state_export=state,
            replay_lineage=lineage,
            frozen_fit=fit,
            expected_count=12,
            analysis_provenance=_provenance(),
        )


def test_h5_late_uses_stable_sample_id_ties_and_is_feature_row_order_invariant(tmp_path: Path) -> None:
    feature, state, lineage, fit = _files(tmp_path)
    fit_json = json.loads(fit.read_text(encoding="utf-8"))
    fit_json["weights"] = [0.0, 0.0, 0.0]
    fit_json["coefficients_sha256"] = hashlib.sha256(canonical_json(fit_json["weights"])).hexdigest()
    fit_payload = {
        "feature_names": fit_json["feature_names"],
        "weights": fit_json["weights"],
        "means": fit_json["means"],
        "scales": fit_json["scales"],
    }
    fit_json["fit_sha256"] = hashlib.sha256(canonical_json(fit_payload)).hexdigest()
    fit.write_text(json.dumps(fit_json), encoding="utf-8")
    report = analyze_history_screen(
        feature_panel=feature,
        online_state_export=state,
        replay_lineage=lineage,
        frozen_fit=fit,
        expected_count=12,
        analysis_provenance=_provenance(),
    )
    assert report["fixed_score"]["selected_ids_sha256"] == hashlib.sha256(canonical_json([0])).hexdigest()

    table = pq.read_table(feature)
    reversed_path = tmp_path / "feature-reversed.parquet"
    pq.write_table(table.take(list(reversed(range(table.num_rows)))), reversed_path)
    lineage_json = json.loads(lineage.read_text(encoding="utf-8"))
    lineage_json["output_parquet_sha256"]["feature_panel"] = sha256_file(reversed_path)
    reversed_lineage = tmp_path / "lineage-reversed.json"
    reversed_lineage.write_text(json.dumps(lineage_json), encoding="utf-8")
    reordered = analyze_history_screen(
        feature_panel=reversed_path,
        online_state_export=state,
        replay_lineage=reversed_lineage,
        frozen_fit=fit,
        expected_count=12,
        analysis_provenance=_provenance(),
    )
    assert reordered["fixed_score"]["selected_ids_sha256"] == report["fixed_score"]["selected_ids_sha256"]
    assert reordered["adaptive_score"] == report["adaptive_score"]
    assert reordered["overlap"] == report["overlap"]


def test_h5_late_cli_repeated_labeled_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    feature, state, lineage, fit = _files(tmp_path)
    output = tmp_path / "report.json"
    monkeypatch.setattr(history_screen, "_tracked_clean_provenance", _provenance)
    with pytest.raises(SystemExit):
        main(
            [
                "--feature-panel",
                f"L1={feature}",
                "--online-state-export",
                f"L1={state}",
                "--replay-lineage",
                f"L1={lineage}",
                "--frozen-fit",
                f"L1={fit}",
                "--expected-count",
                "12",
                "--output",
                str(output),
            ]
        )
