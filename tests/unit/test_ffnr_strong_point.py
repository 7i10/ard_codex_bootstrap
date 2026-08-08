from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ard.analysis.ffnr_strong_point as strong_point
from ard.analysis.ffnr_strong_point import (
    StrongPointError,
    _strong_lineage,
    _strong_panel,
    analyze_strong_run,
    write_strong_point_report,
)
from ard.analysis.ffnr_strong_replay import expected_selection_attack
from ard.analysis.signal_audit import canonical_json, sha256_file

pytestmark = pytest.mark.unit


def _analysis_provenance() -> dict[str, object]:
    source_files = {
        "ffnr_strong_point": "1" * 64,
        "ffnr_strong_point_cli": "2" * 64,
        "ffnr_forecasting": "3" * 64,
        "ffnr_strong_replay": "4" * 64,
    }
    return {
        "git": {"sha": "a" * 40, "dirty": False},
        "source_files": source_files,
        "source_sha256": hashlib.sha256(canonical_json(source_files)).hexdigest(),
    }


def _universe(ids: tuple[int, ...]) -> str:
    return hashlib.sha256(
        canonical_json(
            [{"sample_id": sample_id, "class_id": index % 10} for index, sample_id in enumerate(sorted(ids))]
        )
    ).hexdigest()


def _row(sample_id: int, class_id: int, epoch: int, *, wrong: bool) -> dict[str, object]:
    teacher = [0.01] * 10
    teacher[class_id] = 0.91
    return {
        "namespace": "train",
        "sample_id": sample_id,
        "class_id": class_id,
        "epoch": epoch,
        "observation_schema_version": 1,
        "student_robust_correct": not wrong,
        "student_adversarial_probability_margin": -0.3 if wrong else 0.4,
        "student_adversarial_logit_margin": -1.0 if wrong else 2.0,
        "student_adversarial_ce": 2.0 if wrong else 0.1,
        "student_clean_probability_margin": -0.1 if wrong else 0.6,
        "student_clean_logit_margin": -0.5 if wrong else 3.0,
        "student_clean_correct": not wrong,
        "student_clean_to_adversarial_prediction_flip": wrong,
        "student_clean_to_adversarial_true_probability_delta": -0.2,
        "student_clean_to_adversarial_probability_margin_delta": -0.2,
        "student_clean_to_adversarial_logit_margin_delta": -0.5 if wrong else -1.0,
        "teacher_clean_probabilities": teacher,
        "teacher_adversarial_probabilities": teacher,
        "teacher_clean_adversarial_js": 0.0,
    }


def _strong_lineage_payload(
    observations: Path, *, role: str, epochs: tuple[int, ...], expected_count: int, universe: str, inventory: Path
) -> dict[str, object]:
    return {
        "contract": "ffnr_strong_replay_ce_pgd20_v1",
        "schema_version": 1,
        "semantic_role": role,
        "observations_sha256": sha256_file(observations),
        "train_expected_count": expected_count,
        "row_count": expected_count * len(epochs),
        "requested_epochs": list(epochs),
        "run_id": "chen-run",
        "saved_resolved_config_mapping_sha256": "a" * 64,
        "manifest_sha256": "0" * 64,
        "checkpoint_inventory": str(inventory),
        "checkpoint_inventory_sha256": sha256_file(inventory),
        "teacher": {"registry_id": "chen2021_ltd_wrn34_10"},
        "dataset_identity": {"dataset": {"name": "cifar10", "split": "train"}},
        "attack_identity": expected_selection_attack(),
        "attack_identity_sha256": hashlib.sha256(canonical_json(expected_selection_attack())).hexdigest(),
        "analysis_provenance": {"source_sha256": "c" * 64},
        "runtime": {
            "deterministic_backend": {
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
            }
        },
        "stable_id_class_universe": {"count": expected_count, "sha256": universe},
        "checkpoints": [{"epoch": epoch, "path": f"/checkpoint-{epoch}.pt", "sha256": "d" * 64} for epoch in epochs],
    }


def _inputs(tmp_path: Path) -> tuple[dict[str, Path], str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ids = (7, 19, 31, 47)
    universe = _universe(ids)
    feature_epochs, outcome_epochs = (39, 59, 79), (179, 184, 189)
    feature, outcome = tmp_path / "feature.parquet", tmp_path / "outcome.parquet"
    feature_rows = [
        _row(sample_id, index % 10, epoch, wrong=sample_id == 7)
        for epoch in feature_epochs
        for index, sample_id in enumerate(ids)
    ]
    outcome_rows = [
        _row(sample_id, index % 10, epoch, wrong=sample_id in {7, 19})
        for epoch in outcome_epochs
        for index, sample_id in enumerate(ids)
    ]
    pq.write_table(pa.Table.from_pylist(feature_rows), feature)
    pq.write_table(pa.Table.from_pylist(outcome_rows), outcome)
    inventory = tmp_path / "checkpoint-inventory.json"
    inventory.write_text("{\"contract\":\"fixture\"}\n")
    feature_lineage, outcome_lineage = tmp_path / "feature.json", tmp_path / "outcome.json"
    feature_lineage.write_text(
        json.dumps(
            _strong_lineage_payload(
                feature, role="feature", epochs=feature_epochs, expected_count=len(ids), universe=universe, inventory=inventory
            )
        )
    )
    outcome_lineage.write_text(
        json.dumps(
            _strong_lineage_payload(
                outcome, role="outcome", epochs=outcome_epochs, expected_count=len(ids), universe=universe, inventory=inventory
            )
        )
    )
    online, online_lineage = tmp_path / "online.parquet", tmp_path / "online.json"
    online_rows = []
    for epoch in feature_epochs:
        for index, sample_id in enumerate(ids):
            online_rows.append(
                {
                    "namespace": "train",
                    "sample_id": sample_id,
                    "anchor_epoch": epoch,
                    "true_label": index % 10,
                    "robust_correct_count": epoch,
                    "previous_robust_correct": sample_id != 19,
                    "margin_ema": -0.2 if sample_id in {7, 19} else 0.4,
                    "last_margin": -0.3 if sample_id in {7, 19} else 0.5,
                    "robust_correct_frequency_inclusive": epoch / (epoch + 1),
                }
            )
    pq.write_table(pa.Table.from_pylist(online_rows), online)
    online_lineage.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "h5_online_state_anchor_v1",
                "expected_count": len(ids),
                "observations_sha256": sha256_file(online),
                "run_id": "chen-run",
                "config_hash": "a" * 64,
                "scientific_git_sha": "b" * 40,
                "seed": 1,
                "attack_identity": {"x": "online"},
                "dataset_identity": {"dataset": "cifar10"},
                "teacher": {"registry_id": "chen2021_ltd_wrn34_10"},
            }
        )
    )
    history, manifest = tmp_path / "metrics.jsonl", tmp_path / "manifest.json"
    history.write_text(
        "\n".join(
            json.dumps(
                {"epoch": epoch, "val_pgd_accuracy": 0.8 if epoch == 184 else 0.79 if epoch in {179, 189} else 0.4}
            )
            for epoch in range(200)
        )
        + "\n"
    )
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "chen-run",
                "config_hash": "a" * 64,
                "git": {"sha": "b" * 40},
                "seed": 1,
                "teacher": {"registry_id": "chen2021_ltd_wrn34_10"},
            }
        )
    )
    for lineage_path in (feature_lineage, outcome_lineage):
        payload = json.loads(lineage_path.read_text())
        payload["manifest_sha256"] = sha256_file(manifest)
        lineage_path.write_text(json.dumps(payload))
    return {
        "feature_observations": feature,
        "feature_lineage": feature_lineage,
        "outcome_observations": outcome,
        "outcome_lineage": outcome_lineage,
        "online_states": online,
        "online_lineage": online_lineage,
        "validation_history": history,
        "validation_manifest": manifest,
    }, universe


def test_lineage_source_drift_is_rejected(tmp_path: Path) -> None:
    inputs, universe = _inputs(tmp_path)
    payload = json.loads(inputs["feature_lineage"].read_text())
    payload["analysis_provenance"] = {}
    inputs["feature_lineage"].write_text(json.dumps(payload))
    with pytest.raises(StrongPointError, match="source SHA"):
        _strong_lineage(
            path=inputs["feature_lineage"],
            observations=inputs["feature_observations"],
            role="feature",
            expected_count=4,
            expected_universe_sha256=universe,
        )


def test_id_class_join_temporal_leakage_metrics_and_nonoverwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, universe = _inputs(tmp_path)
    legacy_strong_lineage = json.loads(inputs["feature_lineage"].read_text())
    assert {"config_hash", "scientific_git_sha", "seed"}.isdisjoint(legacy_strong_lineage)
    report = analyze_strong_run(
        label="L2",
        expected_count=4,
        expected_universe_sha256=universe,
        deltas_pp=(1.0,),
        window_sizes=(3,),
        thresholds=("majority",),
        **inputs,
    )
    assert report["input_identity"]["config_hash"] == "a" * 64
    assert report["input_identity"]["scientific_git_sha"] == "b" * 40
    assert report["input_identity"]["seed"] == 1
    assert set(report["input_identity"]["input_bytes"]) == {
        "feature_observations",
        "feature_lineage",
        "outcome_observations",
        "outcome_lineage",
        "online_states",
        "online_lineage",
        "validation_history",
        "validation_manifest",
    }
    assert report["input_identity"]["strong_replay_contracts"]["feature"]["attack_identity"] == expected_selection_attack()
    assert report["score_definitions"]["D_teacher_signed_wrong_class_dominance_margin"] == {
        "definition": "max_{c!=y} p_T(c|x_adv)-p_T(y|x_adv)",
        "range": "[-1,1]",
        "direction": "larger=higher-risk",
    }
    history_rows = [json.loads(line) for line in inputs["validation_history"].read_text().splitlines()]
    history_rows[0]["val_pgd_accuracy"] = 0.401
    inputs["validation_history"].write_text("\n".join(json.dumps(row) for row in history_rows) + "\n")
    changed_history_report = analyze_strong_run(
        label="L2",
        expected_count=4,
        expected_universe_sha256=universe,
        deltas_pp=(1.0,),
        window_sizes=(3,),
        thresholds=("majority",),
        **inputs,
    )
    assert (
        changed_history_report["input_identity"]["input_bytes"]["validation_history"]["sha256"]
        != report["input_identity"]["input_bytes"]["validation_history"]["sha256"]
    )
    candidate = next(item for item in report["candidates"] if not item["censored"])
    metrics = candidate["anchors"]["39"]["FF"]["L_adversarial_cross_entropy"]
    assert metrics["positive_count"] == 1 and metrics["auroc"] is not None
    rows = pq.read_table(inputs["feature_observations"]).to_pylist()
    changed = [dict(row) for row in rows]
    changed[0]["class_id"] = 9
    with pytest.raises(StrongPointError, match="universe hash"):
        _strong_panel(changed, epochs=(39, 59, 79), expected_count=4, expected_universe_sha256=universe)
    leaked = json.loads(inputs["outcome_lineage"].read_text())
    leaked["requested_epochs"] = [79]
    leaked["checkpoints"] = [{**leaked["checkpoints"][-1], "epoch": 79}]
    leaked["row_count"] = 4
    inputs["outcome_lineage"].write_text(json.dumps(leaked))
    with pytest.raises(StrongPointError, match="temporal leakage"):
        analyze_strong_run(
            label="L2",
            expected_count=4,
            expected_universe_sha256=universe,
            deltas_pp=(1.0,),
            window_sizes=(3,),
            thresholds=("majority",),
            **inputs,
        )
    inputs, universe = _inputs(tmp_path / "second")
    left = analyze_strong_run(
        label="L2",
        expected_count=4,
        expected_universe_sha256=universe,
        deltas_pp=(1.0,),
        window_sizes=(3,),
        thresholds=("majority",),
        **inputs,
    )
    shared = next(key for key in left["_formula_masks"] if key.startswith("future_failure:"))
    right = {
        **left,
        "label": "L4",
        "_formula_masks": {
            **left["_formula_masks"],
            shared: left["_formula_masks"][shared],
            "future_failure:l4_only": {19},
        },
    }
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n")
    output = tmp_path / "output"
    provenance = _analysis_provenance()
    monkeypatch.setattr(strong_point, "_tracked_clean_provenance", lambda: provenance)
    paths = write_strong_point_report(output_dir=output, reports={"L2": left, "L4": right}, config_path=config)
    payload = json.loads(paths["report"].read_text())
    assert payload["formula_level_cross_seed_jaccard"] == {shared: 1.0}
    assert payload["analysis_provenance"] == provenance
    within = payload["predictor_mask_stability"]["within_run_consecutive_anchor"]["L2"]
    assert within and {item["from_anchor"] for item in within} == {39, 59}
    assert all("from_realized_fraction" in item and "to_realized_count" in item for item in within)
    cross_seed = payload["predictor_mask_stability"]["cross_seed_top10"]
    assert cross_seed and {item["candidate_id"] for item in cross_seed} == {shared.removeprefix("future_failure:")}
    assert all(item["jaccard"] == 1.0 and item["entry"] == item["exit"] == 0 for item in cross_seed)
    with pytest.raises(StrongPointError, match="overwrite"):
        write_strong_point_report(output_dir=output, reports={"L2": left, "L4": right}, config_path=config)


def test_report_provenance_rejects_dirty_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, stdout="a" * 40 + "\n")
        if "status" in command:
            return subprocess.CompletedProcess(command, 0, stdout=" M src/ard/analysis/ffnr_strong_point.py\n")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(strong_point.subprocess, "run", fake_run)
    with pytest.raises(StrongPointError, match="tracked-clean"):
        strong_point._tracked_clean_provenance()
