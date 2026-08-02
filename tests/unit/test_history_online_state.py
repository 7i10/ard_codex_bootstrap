from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ard.analysis.history_early import HistoryEarlyError
from ard.analysis.history_online_state import (
    HistoryOnlineStateError,
    OnlineStateExport,
    export_online_anchors,
    write_online_anchors,
)
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS, OUTCOME_EPOCHS
from ard.analysis.signal_audit import sha256_file
from ard.cli.history_early import main as early_main

pytestmark = pytest.mark.t1


def _checkpoint(path: Path, epoch: int, *, run: str = "run", config: str = "a" * 64) -> str:
    records = {
        str(i + 40): {
            "true_label": i % 2,
            "seen": epoch + 1,
            "robust_correct_count": epoch // 2,
            "previous_robust_correct": bool(i % 2),
            "margin_ema": 0.2,
            "last_margin": 0.3,
        }
        for i in range(10)
    }
    torch.save(
        {
            "epoch": epoch,
            "epoch_boundary": "end",
            "tracker_run_id": run,
            "config_hash": config,
            "world_size": 1,
            "sample_state": {"format_version": 3, "pending": [], "records": records},
        },
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[dict[int, Path], Path]:
    checkpoints = {e: tmp_path / f"{e}.pt" for e in (39, 59, 79)}
    hashes = {e: _checkpoint(p, e) for e, p in checkpoints.items()}
    lineage = tmp_path / "replay.json"
    lineage.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run",
                "config_hash": "a" * 64,
                "scientific_git_sha": "b" * 40,
                "seed": 1,
                "attack_identity": {"steps": 10},
                "dataset_identity": {"name": "cifar10"},
                "teacher": {"registry_id": "teacher"},
                "checkpoints": [{"epoch": e, "sha256": h} for e, h in hashes.items()],
            }
        )
    )
    return checkpoints, lineage


def _outcome_rows() -> list[dict[str, object]]:
    return [
        {
            "namespace": "train",
            "sample_id": sid + 40,
            "class_id": sid % 2,
            "epoch": epoch,
            "teacher_entropy_normalized": 0.5,
            "student_probability_margin": 0.2,
            "student_margin_risk": 0.4,
            "robust_correct": not (sid == 0 and epoch in {99, 104, 109}),
        }
        for epoch in OUTCOME_EPOCHS
        for sid in range(10)
    ]


def _feature_rows(*, positive_high_risk: bool) -> list[dict[str, object]]:
    return [
        {
            "namespace": "train",
            "sample_id": sid + 40,
            "class_id": sid % 2,
            "epoch": epoch,
            "teacher_entropy_normalized": 0.5,
            "student_probability_margin": (-0.8 if positive_high_risk else 0.8) if sid == 0 else 0.0,
            "student_margin_risk": (0.9 if positive_high_risk else 0.1) if sid == 0 else 0.5,
            "robust_correct": True,
        }
        for epoch in FEATURE_EPOCHS
        for sid in range(10)
    ]


def _replay_lineage(path: Path, observations: Path, *, key: str, protocol: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observation_schema_version": 2,
                "run_id": "run",
                "config_hash": "a" * 64,
                "scientific_git_sha": "b" * 40,
                "seed": 1,
                "train_expected_count": 10,
                key: sha256_file(observations),
                "attack_identity": {"steps": 10},
                "dataset_identity": {"name": "cifar10"},
                "teacher": {"registry_id": "teacher"},
                protocol: {},
            }
        )
    )


def _cohort(path: Path) -> dict[str, dict[str, object]]:
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


def _bound_lineage(source: Path, destination: Path, *, identity: dict[str, object]) -> Path:
    value = json.loads(source.read_text())
    value.update(
        {
            "run_id": identity["run_id"],
            "config_hash": identity["config_hash"],
            "scientific_git_sha": identity["scientific_git_sha"],
            "seed": identity["seed"],
            "teacher": {"registry_id": identity["teacher_registry_id"]},
        }
    )
    destination.write_text(json.dumps(value))
    return destination


def test_online_export_exact_sparse_shuffled_ids_and_nonoverwrite(tmp_path: Path) -> None:
    checkpoints, lineage = _inputs(tmp_path)
    export = export_online_anchors(
        checkpoints={79: checkpoints[79], 39: checkpoints[39], 59: checkpoints[59]},
        replay_lineage=lineage,
        expected_count=10,
        analysis_provenance={"test": True},
    )
    assert [r["sample_id"] for r in export.rows[:2]] == [40, 41]
    assert export.rows[0]["robust_correct_frequency_inclusive"] == pytest.approx(19 / 40)
    paths = write_online_anchors(output_dir=tmp_path / "out", export=export)
    assert paths["observations"].is_file()
    with pytest.raises(FileExistsError):
        write_online_anchors(output_dir=tmp_path / "out", export=export)


def test_early_cli_requires_online_primary_or_explicit_legacy_opt_in(tmp_path: Path) -> None:
    common = [
        "--expected-count",
        "10",
        "--cohort-inventory",
        str(tmp_path / "cohort.json"),
        "--output",
        str(tmp_path / "out.json"),
    ]
    with pytest.raises(HistoryEarlyError, match="corrected H5-Early requires --online-states"):
        early_main(common)
    with pytest.raises(SystemExit):
        early_main([*common, "--legacy-retrospective-ro", "--online-states", "L1=state.json"])


def test_early_cli_explicit_legacy_marks_retrospective_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ard.cli.history_early.analyze_history_early",
        lambda **_: {"tables": {}, "_post_peak_gate_rows": {"future_conditioned": True}},
    )
    monkeypatch.setattr("ard.cli.history_early.bind_early_collection_to_cohort", lambda **_: "a" * 64)
    output = tmp_path / "legacy.json"
    assert (
        early_main(
            [
                "--legacy-retrospective-ro",
                "--expected-count",
                "10",
                "--cohort-inventory",
                str(tmp_path / "cohort.json"),
                "--output",
                str(output),
                "--feature-observations",
                "L1=feature.parquet",
                "--feature-lineage",
                "L1=feature.json",
                "--outcome-observations",
                "L1=outcome.parquet",
                "--outcome-lineage",
                "L1=outcome.json",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["contract"] == "h5_early_legacy_retrospective_ro_collection_v1"
    assert report["scientific_status"] == "retrospective_ro_diagnostic_only"
    assert "primary_selection_gate" not in report


def test_online_export_rejects_missing_sha_identity_and_seen_drift(tmp_path: Path) -> None:
    checkpoints, lineage = _inputs(tmp_path)
    with pytest.raises(HistoryOnlineStateError, match="checkpoint schedule"):
        export_online_anchors(
            checkpoints={39: checkpoints[39]}, replay_lineage=lineage, expected_count=10, analysis_provenance={}
        )
    _checkpoint(checkpoints[59], 59, run="wrong")
    with pytest.raises(HistoryOnlineStateError, match="SHA"):
        export_online_anchors(
            checkpoints=checkpoints, replay_lineage=lineage, expected_count=10, analysis_provenance={}
        )


def test_public_online_export_to_early_point_cli_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    epochs = (39, 59, 79)
    checkpoints = {epoch: tmp_path / f"{epoch}.pt" for epoch in epochs}
    hashes = {epoch: _checkpoint(path, epoch) for epoch, path in checkpoints.items()}
    replay_lineage = tmp_path / "replay-all.json"
    replay_lineage.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run",
                "config_hash": "a" * 64,
                "scientific_git_sha": "b" * 40,
                "seed": 1,
                "attack_identity": {"steps": 10},
                "dataset_identity": {"name": "cifar10"},
                "teacher": {"registry_id": "teacher"},
                "checkpoints": [{"epoch": epoch, "sha256": digest} for epoch, digest in hashes.items()],
            }
        )
    )
    export = export_online_anchors(
        checkpoints=checkpoints,
        replay_lineage=replay_lineage,
        expected_count=10,
        analysis_provenance={"test": True},
    )
    # An accidental online "future" field must not become an outcome source.
    # The replay panel alone marks sample 40 wrong at 99/104/109.
    export = OnlineStateExport(
        lineage=export.lineage,
        rows=tuple(
            {
                **row,
                "previous_robust_correct": True if row["sample_id"] == 40 else row["previous_robust_correct"],
                "online_future_robust_correct": True,
            }
            for row in export.rows
        ),
    )
    paths = write_online_anchors(output_dir=tmp_path / "online", export=export)
    feature = tmp_path / "feature.parquet"
    pq.write_table(pa.Table.from_pylist(_feature_rows(positive_high_risk=True)), feature)
    feature_lineage = tmp_path / "feature-lineage.json"
    _replay_lineage(
        feature_lineage,
        feature,
        key="feature_observations_sha256",
        protocol="feature_protocol",
    )
    outcome = tmp_path / "outcome.parquet"
    outcome_rows = _outcome_rows()
    pq.write_table(pa.Table.from_pylist(outcome_rows), outcome)
    outcome_lineage = tmp_path / "outcome-lineage.json"
    _replay_lineage(
        outcome_lineage,
        outcome,
        key="outcome_observations_sha256",
        protocol="outcome_protocol",
    )
    monkeypatch.setattr("ard.analysis.history_early._tracked_clean_provenance", lambda: {"test": True})
    cohort = tmp_path / "cohort.json"
    cohort_runs = _cohort(cohort)
    lineages = {
        label: (
            _bound_lineage(paths["lineage"], tmp_path / f"{label}-online.json", identity=identity),
            _bound_lineage(feature_lineage, tmp_path / f"{label}-feature.json", identity=identity),
            _bound_lineage(outcome_lineage, tmp_path / f"{label}-outcome.json", identity=identity),
        )
        for label, identity in cohort_runs.items()
    }
    report = tmp_path / "point.json"
    argv = ["--expected-count", "10", "--cohort-inventory", str(cohort), "--output", str(report)]
    for label in ("L1", "L2", "L3", "L4"):
        online_lineage, feature_bound, outcome_bound = lineages[label]
        argv.extend(
            [
                "--online-states",
                f"{label}={paths['observations']}",
                "--online-lineage",
                f"{label}={online_lineage}",
                "--feature-observations",
                f"{label}={feature}",
                "--feature-lineage",
                f"{label}={feature_bound}",
                "--outcome-observations",
                f"{label}={outcome}",
                "--outcome-lineage",
                f"{label}={outcome_bound}",
            ]
        )
    assert early_main(argv) == 0
    result = json.loads(report.read_text())
    assert result["contract"] == "h5_early_online_collection_v1"
    assert result["reports"]["L1"]["anchors"] == [39, 59, 79]
    assert "99" not in result["reports"]["L1"]["reports"]
    assert result["bootstrap_tasks"] == []  # L1 alone is diagnostic, never a bootstrap gate.
    peak = result["reports"]["L1"]["reports"]["39"]["anchor_correct_peak_failure"]
    assert peak["online_rank"]["prevalence"] == pytest.approx(1 / 6)
    assert "top_10pct_mask_overlap" in peak
    assert peak["replay_pre_anchor_rank_diagnostic"]["max_feature_epoch"] == 34

    feature_changed = tmp_path / "feature-changed.parquet"
    pq.write_table(pa.Table.from_pylist(_feature_rows(positive_high_risk=False)), feature_changed)
    feature_changed_lineage = tmp_path / "feature-changed-lineage.json"
    _replay_lineage(
        feature_changed_lineage,
        feature_changed,
        key="feature_observations_sha256",
        protocol="feature_protocol",
    )
    changed_report = tmp_path / "point-changed.json"
    changed_argv = list(argv)
    changed_argv[changed_argv.index(str(report))] = str(changed_report)
    l1_feature_lineage = _bound_lineage(
        feature_changed_lineage,
        tmp_path / "L1-feature-changed.json",
        identity=cohort_runs["L1"],
    )
    changed_argv[changed_argv.index(f"L1={feature}")] = f"L1={feature_changed}"
    changed_argv[changed_argv.index(f"L1={lineages['L1'][1]}")] = f"L1={l1_feature_lineage}"
    assert early_main(changed_argv) == 0
    changed = json.loads(changed_report.read_text())
    changed_peak = changed["reports"]["L1"]["reports"]["39"]["anchor_correct_peak_failure"]
    assert changed["bootstrap_tasks"] == result["bootstrap_tasks"]
    assert changed_peak["online_rank"] == peak["online_rank"]
    assert changed_peak["replay_pre_anchor_rank_diagnostic"] != peak["replay_pre_anchor_rank_diagnostic"]

    historical_outcome = json.loads(lineages["L1"][2].read_text())
    historical_outcome.pop("seed")
    historical_outcome_path = tmp_path / "L1-outcome-historical-no-seed.json"
    historical_outcome_path.write_text(json.dumps(historical_outcome))
    historical_report = tmp_path / "point-historical-outcome-seed.json"
    historical_argv = list(argv)
    historical_argv[historical_argv.index(str(report))] = str(historical_report)
    historical_argv[historical_argv.index(f"L1={lineages['L1'][2]}")] = f"L1={historical_outcome_path}"
    assert early_main(historical_argv) == 0
    assert (
        json.loads(historical_report.read_text())["reports"]["L1"]["input_identity"]["outcome_seed_compatibility"]
        == {"status": "historical_missing_seed", "effective_seed": 1}
    )

    mismatched_outcome = json.loads(lineages["L1"][2].read_text())
    mismatched_outcome["seed"] = 999
    mismatched_outcome_path = tmp_path / "L1-outcome-mismatched-seed.json"
    mismatched_outcome_path.write_text(json.dumps(mismatched_outcome))
    mismatched_argv = list(argv)
    mismatched_argv[mismatched_argv.index(str(report))] = str(tmp_path / "point-mismatched-outcome-seed.json")
    mismatched_argv[mismatched_argv.index(f"L1={lineages['L1'][2]}")] = f"L1={mismatched_outcome_path}"
    with pytest.raises(HistoryEarlyError, match="online/replay lineage identity drifted"):
        early_main(mismatched_argv)
