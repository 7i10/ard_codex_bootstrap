from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ard.analysis import history_early
from ard.analysis.history_early import (
    OBS_COLS,
    HistoryEarlyError,
    analyze_history_early,
    build_online_bootstrap_tasks,
    collection_gate,
)
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS, OUTCOME_EPOCHS, build_feature_panel
from ard.analysis.signal_audit import sha256_file

pytestmark = pytest.mark.t1


def test_history_early_module_execution_invokes_the_cli_guard(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["ard.cli.history_early", "--help"])
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_module("ard.cli.history_early", run_name="__main__")
    assert exit_status.value.code == 0
    assert "--feature-observations" in capsys.readouterr().out


def _rows(epochs: tuple[int, ...]) -> list[dict[str, object]]:
    out = []
    for e in epochs:
        for sid in range(12):
            # IDs 8/9 are a reproducible high-risk post-peak/peak group.
            wrong = sid in {8, 9} and e in {104, 109, 114}
            out.append(
                {
                    "namespace": "train",
                    "sample_id": sid,
                    "class_id": sid % 2,
                    "epoch": e,
                    "teacher_entropy_normalized": 0.5,
                    "student_probability_margin": -sid / 12,
                    "student_margin_risk": (1 + sid / 12) / 2,
                    "robust_correct": not wrong,
                }
            )
    return out


def _lineage(path: Path, obs: Path, key: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observation_schema_version": 2,
                "run_id": "run",
                "config_hash": "a" * 64,
                "scientific_git_sha": "b" * 40,
                "seed": 1,
                "train_expected_count": 12,
                key: sha256_file(obs),
                "attack_identity": {"steps": 10},
                "dataset_identity": {"dataset": "cifar10"},
                "teacher": {"registry_id": "teacher"},
                "feature_protocol" if key.startswith("feature") else "outcome_protocol": {
                    "seed_domain": "feature" if key.startswith("feature") else "outcome"
                },
            }
        )
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    f, o, fl, ol = (tmp_path / "f.parquet", tmp_path / "o.parquet", tmp_path / "fl.json", tmp_path / "ol.json")
    pq.write_table(pa.Table.from_pylist(_rows(FEATURE_EPOCHS)).select(OBS_COLS), f)
    pq.write_table(pa.Table.from_pylist(_rows(OUTCOME_EPOCHS)).select(OBS_COLS), o)
    _lineage(fl, f, "feature_observations_sha256")
    _lineage(ol, o, "outcome_observations_sha256")
    return f, o, fl, ol


def test_history_early_exact_anchors_peak_postpeak_and_order_invariance(tmp_path: Path) -> None:
    f, o, fl, ol = _inputs(tmp_path)
    r = analyze_history_early(
        feature_observations=f,
        outcome_observations=o,
        feature_lineage=fl,
        outcome_lineage=ol,
        expected_count=12,
        analysis_provenance={"test": True},
    )
    assert r["anchors"] == [39, 59, 79, 99]
    assert r["outcomes"]["peak_window_error"] == "wrong_at_least_two_of_99_104_109"
    assert r["tables"]["99"]["peak_window_error"]["adaptive_history"]["auroc"] is not None
    # At epoch 109 only ten samples are eligible for post-peak forgetting.
    assert r["tables"]["39"]["post_peak_forgetting"]["adaptive_history"]["prevalence"] is not None
    table = pq.read_table(f)
    rev = tmp_path / "fr.parquet"
    pq.write_table(table.take(list(reversed(range(table.num_rows)))), rev)
    _lineage(fl, rev, "feature_observations_sha256")
    rr = analyze_history_early(
        feature_observations=rev,
        outcome_observations=o,
        feature_lineage=fl,
        outcome_lineage=ol,
        expected_count=12,
        analysis_provenance={"test": True},
    )
    assert rr["tables"] == r["tables"]


def test_history_early_rejects_lineage_and_epoch_drift(tmp_path: Path) -> None:
    f, o, fl, ol = _inputs(tmp_path)
    bad = json.loads(ol.read_text())
    bad["config_hash"] = "c" * 64
    ol.write_text(json.dumps(bad))
    with pytest.raises(HistoryEarlyError, match="identity drifted"):
        analyze_history_early(
            feature_observations=f,
            outcome_observations=o,
            feature_lineage=fl,
            outcome_lineage=ol,
            expected_count=12,
            analysis_provenance={"test": True},
        )
    f, o, fl, ol = _inputs(tmp_path / "x")
    rows = pq.read_table(f).to_pylist()
    rows[0]["epoch"] = 3
    pq.write_table(pa.Table.from_pylist(rows).select(OBS_COLS), f)
    _lineage(fl, f, "feature_observations_sha256")
    with pytest.raises(HistoryEarlyError, match="epoch/ID contract"):
        analyze_history_early(
            feature_observations=f,
            outcome_observations=o,
            feature_lineage=fl,
            outcome_lineage=ol,
            expected_count=12,
            analysis_provenance={"test": True},
        )


def test_collection_gate_is_predeclared_pending_ci() -> None:
    report = {
        "tables": {
            "39": {
                "post_peak_forgetting": {
                    n: {"auroc": 0.7 if n == "adaptive_history" else 0.6}
                    for n in (
                        "adaptive_history",
                        "current_correctness",
                        "instantaneous_margin",
                        "outcome_free_current_rank",
                    )
                }
            }
        }
    }
    gate = collection_gate({"L1": report, "L3": report})
    assert gate["status"] == "sequential_no_automatic_go"
    assert gate["anchors"]["39"]["point_threshold_met"]
    assert gate["anchors"]["39"]["status"] == "point_gate_pass_bootstrap_task_required"


def test_collection_gate_rejects_point_fail_without_running_a_bootstrap() -> None:
    report = {
        "tables": {
            "39": {
                "post_peak_forgetting": {
                    "adaptive_history": {"auroc": 0.60},
                    "outcome_free_current_rank": {"auroc": 0.60},
                }
            }
        }
    }

    gate = collection_gate({"L1": report, "L3": report})
    assert gate["anchors"]["39"]["status"] == "no_go_point_gate"
    assert gate["anchors"]["39"]["paired_lower_bound"] is None


def test_empty_early_stratum_is_degenerate_no_go_without_division() -> None:
    assert history_early._metric({}, {}) == {"auroc": None, "auprc": None, "prevalence": None, "count": 0}


def test_online_bootstrap_requires_both_bartoldson_confirmations_and_excludes_diagnostics() -> None:
    outcomes = {sample_id: int(sample_id < 10) for sample_id in range(20)}
    raw = {
        "peak": outcomes,
        "non_recovery": outcomes,
        "online_rank": {sample_id: float(outcomes[sample_id]) for sample_id in outcomes},
        "baseline": {sample_id: float(1 - outcomes[sample_id]) for sample_id in outcomes},
        "class_id": {sample_id: sample_id % 10 for sample_id in outcomes},
    }
    reports = {label: {"_bootstrap_inputs": {"39": raw}} for label in ("L1", "L2", "L3", "L4")}
    tasks, gate = build_online_bootstrap_tasks(reports)
    assert {task["run"] for task in tasks} == {"L1", "L3"}
    assert {task["outcome"] for task in tasks} == {"peak_failure", "non_recovery"}
    assert gate["epoch39-peak_failure"]["pass"]
    assert gate["epoch39-peak_failure"]["diagnostic_only_runs"] == ["L2", "L4"]

    failed = {**reports, "L3": {"_bootstrap_inputs": {"39": {**raw, "online_rank": raw["baseline"]}}}}
    tasks, gate = build_online_bootstrap_tasks(failed)
    assert tasks == []
    assert not gate["epoch39-peak_failure"]["pass"]


def test_anchor99_history_features_match_the_frozen_h5_feature_panel() -> None:
    rows = _rows(FEATURE_EPOCHS)
    panel = history_early._panel(rows, FEATURE_EPOCHS, 12, "feature")
    scores = history_early._score(panel, 99)
    frozen = {int(row["sample_id"]): row for row in build_feature_panel(rows, expected_count=12)}
    for sample_id, row in frozen.items():
        assert scores["frequency_only"][sample_id] == pytest.approx(1 - row["student_robust_correct_frequency"])
        assert scores["margin_only"][sample_id] == pytest.approx(row["student_margin_historical_risk"])
        assert scores["instantaneous_margin"][sample_id] == pytest.approx(row["student_margin_risk_epoch99"])


def test_feature_outcome_domains_are_joined_only_by_stable_id_not_as_a_transition(tmp_path: Path) -> None:
    f, o, fl, ol = _inputs(tmp_path)
    feature_rows = pq.read_table(f).to_pylist()
    # A feature-domain attack outcome at epoch 99 must not be treated as the
    # predecessor of the independent outcome-domain attack at epoch 99.
    feature_rows[0]["robust_correct"] = False
    pq.write_table(pa.Table.from_pylist(feature_rows).select(OBS_COLS), f)
    _lineage(fl, f, "feature_observations_sha256")
    result = analyze_history_early(
        feature_observations=f,
        outcome_observations=o,
        feature_lineage=fl,
        outcome_lineage=ol,
        expected_count=12,
        analysis_provenance={"test": True},
    )
    assert result["input_identity"]["feature_attack_domain"]["seed_domain"] == "feature"
    assert result["input_identity"]["outcome_attack_domain"]["seed_domain"] == "outcome"
