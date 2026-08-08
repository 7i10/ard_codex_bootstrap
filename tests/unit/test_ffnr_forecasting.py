from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ard.analysis.ffnr_forecasting import (
    FFNRForecastingError,
    _metric,
    _replay_components,
    _tie_inclusive_selection,
    analyze_ffnr_run,
    deterministic_midranks,
    plateau_candidates,
    split_ff_nr,
    write_ffnr_report,
)
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS, OUTCOME_EPOCHS
from ard.analysis.signal_audit import sha256_file

pytestmark = pytest.mark.unit


def _observations(epochs: tuple[int, ...], ids: tuple[int, ...]) -> list[dict[str, object]]:
    rows = []
    for epoch in epochs:
        for index, sample_id in enumerate(ids):
            # IDs 103 and 211 will fail at the plateau.  The latter is online
            # wrong, so it exercises the current-wrong future-failure stratum.
            wrong = sample_id in {103, 211} and epoch in {99, 104, 109}
            margin = -0.8 if sample_id in {103, 211} else 0.4 - 0.01 * index
            rows.append(
                {
                    "namespace": "train",
                    "sample_id": sample_id,
                    "class_id": index % 10,
                    "epoch": epoch,
                    "teacher_entropy_normalized": 0.2,
                    "student_probability_margin": margin,
                    "student_margin_risk": (1.0 - margin) / 2.0,
                    "robust_correct": not wrong,
                    "teacher_adversarial_entropy_normalized": 0.2,
                    "teacher_adversarial_probability_margin": 0.5,
                    "teacher_clean_probability_margin": 0.8,
                    "student_clean_probability_margin": 0.6,
                    "teacher_clean_to_adversarial_margin_delta": -0.1,
                }
            )
    return rows


def _lineage(path: Path, observations: Path, *, key: str, seed: int | None) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observation_schema_version": 2,
                "run_id": "bart-rslad-logging-only-s1-confirm-v1",
                "config_hash": "a" * 64,
                "scientific_git_sha": "b" * 40,
                "train_expected_count": 12,
                key: sha256_file(observations),
                "attack_identity": {
                    "norm": "linf",
                    "input_domain": "pixel_0_1",
                    "epsilon": "8/255",
                    "epsilon_value": 8 / 255,
                    "step_size": "2/255",
                    "step_size_value": 2 / 255,
                    "steps": 10,
                    "random_start": True,
                    "loss": "kl",
                    "kl_target": "teacher_clean",
                    "temperature": 1.0,
                    "temperature_squared": True,
                    "student_mode": "eval",
                    "teacher_mode": "eval",
                },
                "dataset_identity": {"dataset": "cifar10", "split": "train"},
                "teacher": {"registry_id": "bartoldson2024_adversarial_wrn94_16"},
                "feature_protocol" if key.startswith("feature") else "outcome_protocol": {"seed_domain": "test"},
            }
        )
    )
    if seed is not None:
        value = json.loads(path.read_text())
        value["seed"] = seed
        path.write_text(json.dumps(value))


def _online(path: Path, lineage: Path, ids: tuple[int, ...]) -> None:
    rows = []
    for epoch in (39, 59, 79):
        for index, sample_id in enumerate(ids):
            rows.append(
                {
                    "namespace": "train",
                    "sample_id": sample_id,
                    "anchor_epoch": epoch,
                    "true_label": index % 10,
                    "robust_correct_count": epoch // 2,
                    "previous_robust_correct": sample_id != 211,
                    "margin_ema": -0.7 if sample_id in {103, 211} else 0.3,
                    "last_margin": -0.8 if sample_id in {103, 211} else 0.4,
                    "robust_correct_frequency_inclusive": (epoch // 2) / (epoch + 1),
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path)
    lineage.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "h5_online_state_anchor_v1",
                "run_id": "bart-rslad-logging-only-s1-confirm-v1",
                "config_hash": "a" * 64,
                "scientific_git_sha": "b" * 40,
                "seed": 1,
                "attack_identity": {
                    "norm": "linf",
                    "input_domain": "pixel_0_1",
                    "epsilon": "8/255",
                    "epsilon_value": 8 / 255,
                    "step_size": "2/255",
                    "step_size_value": 2 / 255,
                    "steps": 10,
                    "random_start": True,
                    "loss": "kl",
                    "kl_target": "teacher_clean",
                    "temperature": 1.0,
                    "temperature_squared": True,
                    "student_mode": "eval",
                    "teacher_mode": "eval",
                },
                "dataset_identity": {"dataset": "cifar10", "split": "train"},
                "teacher": {"registry_id": "bartoldson2024_adversarial_wrn94_16"},
                "expected_count": len(ids),
                "observations_sha256": sha256_file(path),
            }
        )
    )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ids = (3, 7, 11, 29, 41, 58, 79, 103, 131, 173, 211, 401)
    feature, outcome = tmp_path / "feature.parquet", tmp_path / "outcome.parquet"
    pq.write_table(pa.Table.from_pylist(_observations(FEATURE_EPOCHS, ids)), feature)
    pq.write_table(pa.Table.from_pylist(_observations(OUTCOME_EPOCHS, ids)), outcome)
    feature_lineage, outcome_lineage = tmp_path / "feature.json", tmp_path / "outcome.json"
    _lineage(feature_lineage, feature, key="feature_observations_sha256", seed=1)
    # Historical outcome lineage may not declare seed; the analysis supports
    # this only as a labelled compatibility path.
    _lineage(outcome_lineage, outcome, key="outcome_observations_sha256", seed=None)
    online, online_lineage = tmp_path / "online.parquet", tmp_path / "online.json"
    _online(online, online_lineage, ids)
    validation = tmp_path / "metrics.jsonl"
    validation.write_text(
        "\n".join(
            json.dumps(
                {"epoch": epoch, "val_pgd_accuracy": 0.8 if epoch == 109 else 0.799 if epoch in {104, 114} else 0.4}
            )
            for epoch in range(200)
        )
        + "\n"
    )
    # An all-epoch maximum off the replay grid establishes that the report uses
    # the explicitly labelled best *available* replay checkpoint.
    rows = [json.loads(line) for line in validation.read_text().splitlines()]
    rows[110]["val_pgd_accuracy"] = 0.81
    validation.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    validation_manifest = tmp_path / "manifest.json"
    validation_manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "bart-rslad-logging-only-s1-confirm-v1",
                "config_hash": "a" * 64,
                "git": {"sha": "b" * 40},
                "seed": 1,
                "teacher": {"registry_id": "bartoldson2024_adversarial_wrn94_16"},
            }
        )
    )
    return {
        "feature_observations": feature,
        "feature_lineage": feature_lineage,
        "outcome_observations": outcome,
        "outcome_lineage": outcome_lineage,
        "online_states": online,
        "online_lineage": online_lineage,
        "validation_history": validation,
        "validation_manifest": validation_manifest,
    }


def test_plateau_candidates_use_saved_grid_and_censor_terminal_window() -> None:
    history = {epoch: 0.5 for epoch in range(200)}
    history[110] = 0.9  # all-epoch maximum, not a saved replay point
    history[109] = 0.8
    history[104] = history[114] = 0.799
    candidates = plateau_candidates(
        history,
        saved_epochs=OUTCOME_EPOCHS,
        scheduler_stages=((0, 99), (100, 149), (150, 199)),
        deltas_pp=(0.25,),
        window_sizes=(3, 7),
        thresholds=("majority",),
    )
    k3 = next(candidate for candidate in candidates if candidate["window_size"] == 3)
    assert k3["raw_validation_best_epoch"] == 110
    assert k3["best_available_replay_epoch"] == 109
    assert k3["window_epochs"] == [104, 109, 114]
    assert not k3["censored"]
    # A terminal component is censored; it is never shifted into an asymmetric window.
    history[199] = 0.95
    terminal = plateau_candidates(
        history,
        saved_epochs=OUTCOME_EPOCHS,
        scheduler_stages=((0, 99), (100, 149), (150, 199)),
        deltas_pp=(0.25,),
        window_sizes=(3,),
        thresholds=("all",),
    )[0]
    assert terminal["censored"]
    assert terminal["window_epochs"] == []


def test_ff_current_wrong_partition_and_deterministic_midranks() -> None:
    ff, current_wrong = split_ff_nr(
        future_failure={3: 1, 7: 1, 11: 0}, online_current_correct={3: True, 7: False, 11: True}
    )
    assert ff == {3: 1, 7: 0, 11: 0}
    assert current_wrong == {3: 0, 7: 1, 11: 0}
    assert deterministic_midranks({7: 1.0, 3: 1.0, 11: 2.0}) == {
        3: pytest.approx(0.5),
        7: pytest.approx(0.5),
        11: pytest.approx(1.0),
    }


def test_cached_metric_ranking_preserves_tie_aware_auc() -> None:
    scores = {1: 0.9, 2: 0.9, 3: 0.1, 4: 0.0}
    targets = {1: 1, 2: 0, 3: 1, 4: 0}
    ranked = tuple(sorted(scores, key=lambda sample_id: (-scores[sample_id], sample_id)))
    result = _metric(scores, targets, ranked=ranked)
    assert result["auroc"] == pytest.approx(0.625)
    assert result["auprc"] == pytest.approx(7.0 / 12.0)


def test_boundary_ties_are_not_split_by_sample_id() -> None:
    scores = {1: 0.9, 2: 0.8, 3: 0.8, 4: 0.1}
    targets = {1: 1, 2: 0, 3: 1, 4: 0}
    selected, top = _tie_inclusive_selection(scores, (1, 2, 3, 4), nominal_count=2)
    assert selected == (1, 2, 3)
    assert top == {
        "nominal_count": 2,
        "realized_count": 3,
        "realized_fraction": pytest.approx(0.75),
        "boundary_score": pytest.approx(0.8),
        "boundary_tie_count": 2,
    }
    result = _metric(scores, targets)
    selection = result["top_percent"]["0.2"]
    assert selection["nominal_count"] == 1
    assert selection["realized_count"] == 1
    oracle = result["oracle_m"]
    assert oracle["nominal_count"] == 2
    assert oracle["realized_count"] == 3
    assert oracle["boundary_tie_count"] == 2


def test_full_analysis_uses_sparse_ids_train_only_and_conditional_strata(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    report = analyze_ffnr_run(
        label="L1",
        expected_count=12,
        scheduler_stages=((0, 99), (100, 149), (150, 199)),
        anchors=(39, 59, 79),
        analysis_provenance={"test": True},
        **inputs,
    )
    candidate = next(item for item in report["candidates"] if not item["censored"] and item["window_size"] == 3)
    anchor = candidate["anchors"]["79"]
    assert anchor["feature_domain"] == "five_epoch_replay"
    assert anchor["online_domain"] == "exact_online_sample_state"
    assert "L_logit_margin_risk" in anchor["score_availability"]["unavailable"]
    assert "D_teacher_js_response" in anchor["score_availability"]["unavailable"]
    assert set(anchor["strata"]) == {"FF", "current_wrong_future_failure"}
    ff = anchor["strata"]["FF"]["scores"]["online_L"]
    assert ff["count"] == 11 and ff["positive_count"] == 1
    assert set(ff["top_percent"]) == {"0.01", "0.05", "0.1", "0.2"}
    stable_selection = report["score_mask_stability"]["selection"][79]["FF:online_L"]
    assert stable_selection["nominal_count"] == 2
    assert stable_selection["realized_count"] >= stable_selection["nominal_count"]
    assert stable_selection["realized_fraction"] == pytest.approx(stable_selection["realized_count"] / 11)
    assert report["ground_truth_attack_status"].startswith("five_epoch_KL_PGD10")
    assert all(row["sample_id"] in {3, 7, 11, 29, 41, 58, 79, 103, 131, 173, 211, 401} for row in report["_score_rows"])
    assert all("candidate_id" not in row for row in report["_score_rows"])
    assert "seed" not in json.loads(inputs["outcome_lineage"].read_text())


def test_teacher_response_directions_use_the_stored_closed_form() -> None:
    panel = {epoch: {sample_id: {"margin": 0.2, "correct": True} for sample_id in (7, 8)} for epoch in (4, 9)}
    panel[4][7]["margin"] = 0.4
    panel[9][7].update(
        {
            "teacher_clean_to_adversarial_margin_delta": -0.1,
            "teacher_adversarial_entropy_normalized": 0.2,
            "student_clean_probability_margin": 0.7,
            "teacher_clean_probability_margin": 0.8,
            "teacher_adversarial_probability_margin": 0.5,
        }
    )
    panel[9][8].update(
        {
            "teacher_clean_to_adversarial_margin_delta": -0.5,
            "teacher_adversarial_entropy_normalized": 0.2,
            "student_clean_probability_margin": 0.7,
            "teacher_clean_probability_margin": 0.8,
            "teacher_adversarial_probability_margin": 0.5,
        }
    )
    values, _ = _replay_components(panel, 9)
    assert values["D_teacher_nonresponse_risk"][7] == pytest.approx(-0.1)
    assert values["D_teacher_nonresponse_risk"][7] > values["D_teacher_nonresponse_risk"][8]
    assert values["D_student_teacher_response_gap"][7] == pytest.approx((0.7 - 0.2) - (0.8 - 0.5))


def test_rejects_attack_identity_drift_and_writes_nonduplicated_score_and_gt_rows(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    bad = json.loads(inputs["feature_lineage"].read_text())
    bad["attack_identity"]["steps"] = 9
    inputs["feature_lineage"].write_text(json.dumps(bad))
    with pytest.raises(FFNRForecastingError, match="KL teacher_clean PGD10"):
        analyze_ffnr_run(
            label="L1",
            expected_count=12,
            scheduler_stages=((0, 99), (100, 149), (150, 199)),
            analysis_provenance={"test": True},
            **inputs,
        )
    _lineage(inputs["feature_lineage"], inputs["feature_observations"], key="feature_observations_sha256", seed=1)
    report = analyze_ffnr_run(
        label="L1",
        expected_count=12,
        scheduler_stages=((0, 99), (100, 149), (150, 199)),
        analysis_provenance={"test": True},
        **inputs,
    )
    reports = {}
    cohort_runs = {}
    for label, seed, teacher, token in (
        ("L1", 1, "bartoldson2024_adversarial_wrn94_16", "1"),
        ("L2", 1, "chen2021_ltd_wrn34_10", "2"),
        ("L3", 2, "bartoldson2024_adversarial_wrn94_16", "3"),
        ("L4", 2, "chen2021_ltd_wrn34_10", "4"),
    ):
        identity = {
            **report["input_identity"],
            "run_id": label,
            "config_hash": token * 64,
            "scientific_git_sha": token * 40,
            "seed": seed,
            "teacher_registry_id": teacher,
        }
        reports[label] = {**report, "input_identity": identity}
        cohort_runs[label] = {
            "run_id": label,
            "config_hash": identity["config_hash"],
            "scientific_git_sha": identity["scientific_git_sha"],
            "seed": seed,
            "teacher_registry_id": teacher,
        }
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps({"schema_version": 1, "contract": "h5_confirmatory_cohort_inventory_v1", "runs": cohort_runs})
    )
    config = tmp_path / "config.yaml"
    config.write_text("test: true\n")
    formula_a, formula_b = sorted(report["_formula_candidate_positive_masks"])[:2]
    reports["L1"] = {
        **reports["L1"],
        "_formula_candidate_positive_masks": {formula_a: {3}, formula_b: {3}},
        "_representative_candidate_positive_masks": {formula_a: {3}},
    }
    reports["L3"] = {
        **reports["L3"],
        "_formula_candidate_positive_masks": {formula_a: {3}, formula_b: {7}},
        "_representative_candidate_positive_masks": {formula_a: {3}, formula_b: {7}},
    }
    paths = write_ffnr_report(output_dir=tmp_path / "out", reports=reports, config_path=config, cohort_inventory=cohort)
    assert pq.read_table(paths["score_rows"]).num_rows == 4 * len(report["_score_rows"])
    expected_gt_rows = 12 * sum(len(item["_representative_candidate_positive_masks"]) for item in reports.values())
    assert pq.read_table(paths["ground_truth_rows"]).num_rows == expected_gt_rows
    written = json.loads(paths["report"].read_text())
    cross_seed = written["gt_sensitivity"]["same_teacher_seed_jaccard"]["L1_vs_L3"]
    assert {row["candidate_id"] for row in cross_seed} == {formula_a, formula_b}
    assert "_formula_candidate_positive_masks" not in written["reports"]["L1"]
    with pytest.raises(FileExistsError):
        write_ffnr_report(output_dir=tmp_path / "out", reports=reports, config_path=config, cohort_inventory=cohort)
    bad_identity = {**reports["L3"]["input_identity"], "stable_id_class_universe_sha256": "f" * 64}
    bad_reports = {**reports, "L3": {**reports["L3"], "input_identity": bad_identity}}
    bad_output = tmp_path / "bad-universe"
    with pytest.raises(FFNRForecastingError, match="stable-ID/class universe"):
        write_ffnr_report(output_dir=bad_output, reports=bad_reports, config_path=config, cohort_inventory=cohort)
    assert not bad_output.exists()


def test_rejects_nontrain_input_before_any_metric(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    rows = pq.read_table(inputs["feature_observations"]).to_pylist()
    rows[0]["namespace"] = "test"
    pq.write_table(pa.Table.from_pylist(rows), inputs["feature_observations"])
    _lineage(inputs["feature_lineage"], inputs["feature_observations"], key="feature_observations_sha256", seed=1)
    with pytest.raises(FFNRForecastingError, match="official-test"):
        analyze_ffnr_run(
            label="L1",
            expected_count=12,
            scheduler_stages=((0, 99), (100, 149), (150, 199)),
            analysis_provenance={"test": True},
            **inputs,
        )


def test_rejects_incomplete_validation_history_or_manifest(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    rows = inputs["validation_history"].read_text().splitlines()
    inputs["validation_history"].write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(FFNRForecastingError, match="exact epoch 0..199"):
        analyze_ffnr_run(
            label="L1",
            expected_count=12,
            scheduler_stages=((0, 99), (100, 149), (150, 199)),
            analysis_provenance={"test": True},
            **inputs,
        )
    inputs = _inputs(tmp_path / "manifest")
    manifest = json.loads(inputs["validation_manifest"].read_text())
    manifest["status"] = "running"
    inputs["validation_manifest"].write_text(json.dumps(manifest))
    with pytest.raises(FFNRForecastingError, match="must be completed"):
        analyze_ffnr_run(
            label="L1",
            expected_count=12,
            scheduler_stages=((0, 99), (100, 149), (150, 199)),
            analysis_provenance={"test": True},
            **inputs,
        )
    inputs = _inputs(tmp_path / "identity")
    manifest = json.loads(inputs["validation_manifest"].read_text())
    manifest["config_hash"] = "c" * 64
    inputs["validation_manifest"].write_text(json.dumps(manifest))
    with pytest.raises(FFNRForecastingError, match="manifest and replay identity"):
        analyze_ffnr_run(
            label="L1",
            expected_count=12,
            scheduler_stages=((0, 99), (100, 149), (150, 199)),
            analysis_provenance={"test": True},
            **inputs,
        )


def test_rejects_missing_requested_online_anchor(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    rows = [row for row in pq.read_table(inputs["online_states"]).to_pylist() if row["anchor_epoch"] != 79]
    pq.write_table(pa.Table.from_pylist(rows), inputs["online_states"])
    lineage = json.loads(inputs["online_lineage"].read_text())
    lineage["observations_sha256"] = sha256_file(inputs["online_states"])
    inputs["online_lineage"].write_text(json.dumps(lineage))
    with pytest.raises(FFNRForecastingError, match="requested anchor"):
        analyze_ffnr_run(
            label="L1",
            expected_count=12,
            scheduler_stages=((0, 99), (100, 149), (150, 199)),
            analysis_provenance={"test": True},
            **inputs,
        )
