"""Frozen, read-only H2 predictions from logging-only state exports.

The only model-fitting inputs are the epoch-99 detached state primitives named
in ``logging_only_history_confirmatory_v1``.  This module never opens training
checkpoints, test results, or evaluation artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ard.analysis.signal_audit import (
    SignalAuditError,
    _fit_logistic,
    _predict_logistic,
    binary_metrics,
    bootstrap_binary_metric_intervals,
    bootstrap_metric_delta,
    deterministic_hash_split,
)


class LoggingOnlyPredictionError(ValueError):
    """Raised when an H2 input cannot satisfy the frozen analysis contract."""


EXPORT_CONTRACT = "logging_only_exact_state_anchor99_final199_v1"
PREDICTION_CONTRACT = "logging_only_history_prediction_v1"
EXPECTED_COUNT = 45000
NUM_CLASSES = 10
RUN_LABELS = ("L1", "L2", "L3", "L4")
CURRENT_BASELINES = ("current_correctness", "instantaneous_margin", "current_only")
MODEL_ORDER = ("teacher_only", "student_only", "main_effects", "main_effects_plus_product", *CURRENT_BASELINES)
EXPECTED_RUN_IDS = {
    "L1": "bart-rslad-logging-only-s1-confirm-v1",
    "L2": "chen-rslad-observed-s1-confirm-v2",
    "L3": "bart-rslad-observed-s2-confirm-v2",
    "L4": "chen-rslad-observed-s2-confirm-v2",
}


@dataclass(frozen=True)
class _FrozenDesign:
    design_id: str
    design_sha256: str
    block_sha256: str
    split_seed: int
    bootstrap_seed: int
    bootstrap_replicates: int


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_design_paths() -> tuple[Path, Path]:
    root = _repository_root()
    return (
        root / "tools/internal/history_replication/provenance/logging_only_history_confirmatory_v1.yaml",
        root / "configs/analysis/history_confirmatory_block_v2.yaml",
    )


def _load_yaml_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LoggingOnlyPredictionError(f"cannot read frozen {name}") from exc
    if not isinstance(value, dict):
        raise LoggingOnlyPredictionError(f"frozen {name} must be a mapping")
    return value


def _require_int(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise LoggingOnlyPredictionError(f"{name} must be an integer{suffix}")
    return value


def _expect(value: object, expected: object, *, name: str) -> None:
    if value != expected:
        raise LoggingOnlyPredictionError(f"frozen design drift in {name}")


def load_frozen_design(*, design_path: Path | None = None, block_path: Path | None = None) -> _FrozenDesign:
    """Validate every H2-relevant frozen field before reading run outcomes."""
    default_design, default_block = default_design_paths()
    design_path = default_design if design_path is None else design_path
    block_path = default_block if block_path is None else block_path
    design_sha256, block_sha256 = _sha256_file(design_path), _sha256_file(block_path)
    design = _load_yaml_mapping(design_path, name="v1 design")
    block = _load_yaml_mapping(block_path, name="block design")
    _expect(design.get("schema_version"), 1, name="schema_version")
    _expect(design.get("design_id"), "logging_only_history_confirmatory_v1", name="design_id")
    scope, trajectory, predictor, metrics, history_gate = (
        design.get("scope"),
        design.get("trajectory"),
        design.get("predictor"),
        design.get("metrics"),
        design.get("history_gate"),
    )
    if not all(isinstance(value, Mapping) for value in (scope, trajectory, predictor, metrics, history_gate)):
        raise LoggingOnlyPredictionError("frozen v1 design is missing its H2 mappings")
    assert isinstance(scope, Mapping) and isinstance(trajectory, Mapping)
    assert isinstance(predictor, Mapping) and isinstance(metrics, Mapping) and isinstance(history_gate, Mapping)
    _expect(scope.get("train_partition_count"), EXPECTED_COUNT, name="train_partition_count")
    _expect(trajectory.get("anchor_epoch"), 99, name="anchor_epoch")
    _expect(trajectory.get("final_epoch"), 199, name="final_epoch")
    _expect(trajectory.get("primary_outcome"), "future_online_forgetting", name="primary_outcome")
    _expect(
        trajectory.get("primary_outcome_definition"),
        "final_forgetting_count > anchor_forgetting_count",
        name="outcome",
    )
    _expect(predictor.get("implementation"), "ard.analysis.signal_audit._fit_logistic", name="logistic implementation")
    _expect(predictor.get("preprocessing"), "train_fold_mean_and_population_std", name="preprocessing")
    _expect(predictor.get("held_out_fraction"), 0.2, name="held_out_fraction")
    split = predictor.get("split")
    bootstrap = metrics.get("bootstrap")
    if not isinstance(split, Mapping) or not isinstance(bootstrap, Mapping):
        raise LoggingOnlyPredictionError("frozen split/bootstrap definitions are missing")
    _expect(split.get("method"), "true_class_stratified_deterministic_hash", name="split method")
    split_seed = _require_int(split.get("seed"), name="split seed")
    _expect(bootstrap.get("method"), "paired_true_class_stratified_sample_bootstrap", name="bootstrap method")
    bootstrap_seed = _require_int(bootstrap.get("seed"), name="bootstrap seed")
    bootstrap_replicates = _require_int(bootstrap.get("replicates"), name="bootstrap replicates", minimum=1)
    _expect(metrics.get("primary"), ["auroc", "log_loss"], name="primary metrics")
    _expect(
        metrics.get("secondary"),
        ["auprc", "positive_prevalence", "brier_score", "fixed_10_bin_ece"],
        name="secondary metrics",
    )
    _expect(
        history_gate.get("go"),
        {
            "delta_auroc_vs_best_current_state_at_least": 0.02,
            "paired_auroc_lower_bound_greater_than": 0.0,
            "held_out_log_loss_must_improve": True,
        },
        name="history go threshold",
    )
    _expect(
        history_gate.get("no_go"),
        {
            "paired_auroc_upper_bound_below": 0.01,
            "or_held_out_log_loss_does_not_improve": True,
        },
        name="history no-go threshold",
    )
    _expect(block.get("schema_version"), 1, name="block schema_version")
    _expect(block.get("design_id"), "history_confirmatory_block_v2", name="block design_id")
    contract = block.get("scientific_contract")
    analysis_lock = block.get("analysis_lock")
    rules = block.get("decision_rules")
    if not isinstance(contract, Mapping) or not isinstance(analysis_lock, Mapping) or not isinstance(rules, Mapping):
        raise LoggingOnlyPredictionError("frozen block is missing H2 contract mappings")
    _expect(contract.get("inherited_design_sha256"), design_sha256, name="inherited design SHA-256")
    _expect(
        contract.get("predictor_features_outcomes_splits_metrics_and_thresholds"),
        "unchanged",
        name="block inheritance",
    )
    _expect(contract.get("train_partition_count"), EXPECTED_COUNT, name="block train partition")
    _expect(contract.get("anchor_epoch"), 99, name="block anchor epoch")
    _expect(contract.get("final_epoch"), 199, name="block final epoch")
    _expect(
        analysis_lock.get("required_table"),
        ["teacher_only", "student_only", "main_effects", "main_effects_plus_product"],
        name="required table",
    )
    _expect(
        rules.get("bartoldson_history_selector_admissible_for_screen"),
        {"required_runs": ["L1", "L3"], "rule": "both_runs_must_meet_per_trajectory_history_go"},
        name="Bartoldson decision rule",
    )
    _expect(
        rules.get("cross_teacher_history_claim"),
        {
            "required_runs": ["L1", "L2", "L3", "L4"],
            "rule": "all_runs_must_have_positive_delta_auroc_and_improved_log_loss",
        },
        name="cross-teacher decision rule",
    )
    return _FrozenDesign(
        design_id="logging_only_history_confirmatory_v1",
        design_sha256=design_sha256,
        block_sha256=block_sha256,
        split_seed=split_seed,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )


def _tracked_clean_analysis_provenance() -> dict[str, Any]:
    root = _repository_root()
    paths = {"analysis": Path(__file__).resolve(), "cli": root / "src/ard/cli/logging_only_prediction.py"}
    try:
        relative = [str(path.relative_to(root)) for path in paths.values()]
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", *relative], check=True, capture_output=True
        )
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise LoggingOnlyPredictionError("prediction analysis requires tracked source files") from exc
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha) or dirty:
        raise LoggingOnlyPredictionError("prediction analysis requires a tracked-clean Git revision")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    return {
        "git_sha": sha,
        "dirty": False,
        "source_files": hashes,
        "source_sha256": hashlib.sha256(_canonical_json(hashes)).hexdigest(),
    }


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise LoggingOnlyPredictionError(f"{name} must be finite")
    return float(value)


def _binary(value: object, *, name: str) -> int:
    if value not in {0, 1, False, True}:
        raise LoggingOnlyPredictionError(f"{name} must be binary")
    return int(value)


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LoggingOnlyPredictionError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_export_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if identity.get("contract") != EXPORT_CONTRACT or identity.get("expected_count") != EXPECTED_COUNT:
        raise LoggingOnlyPredictionError("state export contract or expected count does not match frozen H2")
    _sha256(identity.get("config_hash"), name="state export config hash")
    _expect(identity.get("world_size"), 1, name="state export world size")
    scientific_git_sha = identity.get("scientific_git_sha")
    if (
        not isinstance(scientific_git_sha, str)
        or len(scientific_git_sha) != 40
        or any(character not in "0123456789abcdef" for character in scientific_git_sha)
    ):
        raise LoggingOnlyPredictionError("state export scientific Git SHA is invalid")
    for checkpoint in ("anchor", "final"):
        value = identity.get(checkpoint)
        if not isinstance(value, Mapping):
            raise LoggingOnlyPredictionError(f"state export lacks hash-bound {checkpoint} identity")
        _expect(value.get("epoch"), 99 if checkpoint == "anchor" else 199, name=f"{checkpoint} epoch")
        _sha256(value.get("checkpoint_sha256"), name=f"{checkpoint} checkpoint hash")
        _sha256(value.get("sample_state_sha256"), name=f"{checkpoint} sample-state hash")
    provenance = identity.get("analysis_provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("dirty") is not False
        or not isinstance(provenance.get("git_sha"), str)
        or len(provenance["git_sha"]) != 40
        or any(character not in "0123456789abcdef" for character in provenance["git_sha"])
    ):
        raise LoggingOnlyPredictionError("state export provenance is not hash-bound and tracked-clean")
    source_files = provenance.get("source_files")
    if not isinstance(source_files, Mapping) or not source_files:
        raise LoggingOnlyPredictionError("state export provenance lacks source hashes")
    hashes = {str(name): _sha256(digest, name="state export source hash") for name, digest in source_files.items()}
    if provenance.get("source_sha256") != hashlib.sha256(_canonical_json(hashes)).hexdigest():
        raise LoggingOnlyPredictionError("state export provenance aggregate hash is invalid")
    return dict(identity)


def _reject_feature_leakage(row: Mapping[str, Any]) -> None:
    forbidden = ("official_test", "autoattack", "evaluation", "test_", "aa_")
    if any(token in key.lower() for key in row if isinstance(key, str) for token in forbidden):
        raise LoggingOnlyPredictionError("logging-only prediction rejects official-test or AutoAttack feature leakage")


def _prepare_rows(export: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    identity, rows = export.get("identity"), export.get("rows")
    if not isinstance(identity, Mapping) or not isinstance(rows, list):
        raise LoggingOnlyPredictionError("state export requires identity and rows")
    source_identity = _validate_export_identity(identity)
    run_id = identity.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise LoggingOnlyPredictionError("state export has a missing run ID")
    if identity.get("row_count") != EXPECTED_COUNT or len(rows) != EXPECTED_COUNT:
        raise LoggingOnlyPredictionError("H2 accepts exactly 45000 state-export rows")
    seen: set[int] = set()
    prepared: list[dict[str, Any]] = []
    log_classes = math.log(NUM_CLASSES)
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise LoggingOnlyPredictionError("state export rows must be mappings")
        _reject_feature_leakage(raw)
        sample_id = _require_int(raw.get("sample_id"), name="sample_id", minimum=0)
        class_id = _require_int(raw.get("true_label"), name="true_label", minimum=0)
        if sample_id in seen or sample_id >= EXPECTED_COUNT or class_id >= NUM_CLASSES:
            raise LoggingOnlyPredictionError("state export has duplicate/out-of-range stable IDs or class labels")
        seen.add(sample_id)
        entropy = _finite(raw.get("anchor_teacher_adversarial_entropy"), name="anchor teacher adversarial entropy")
        previous_correct = _binary(raw.get("anchor_previous_robust_correct"), name="anchor previous correctness")
        last_margin = _finite(raw.get("anchor_last_margin"), name="anchor last margin")
        frequency = _finite(raw.get("anchor_robust_correct_frequency"), name="anchor robust-correct frequency")
        margin_ema = _finite(raw.get("anchor_margin_ema"), name="anchor margin EMA")
        outcome = _binary(raw.get("future_online_forgetting"), name="future_online_forgetting")
        increment = _require_int(
            raw.get("subsequent_forgetting_increment"), name="subsequent forgetting increment", minimum=0
        )
        if outcome != int(increment > 0):
            raise LoggingOnlyPredictionError("future_online_forgetting does not match the frozen outcome definition")
        anchor_forgetting = raw.get("anchor_forgetting_count")
        final_forgetting = raw.get("final_forgetting_count")
        if (
            not isinstance(anchor_forgetting, int)
            or isinstance(anchor_forgetting, bool)
            or not isinstance(final_forgetting, int)
            or isinstance(final_forgetting, bool)
        ):
            raise LoggingOnlyPredictionError("state export lacks exact anchor/final forgetting counters")
        if final_forgetting - anchor_forgetting != increment or outcome != int(final_forgetting > anchor_forgetting):
            raise LoggingOnlyPredictionError("state export outcome counters do not match future_online_forgetting")
        if not (
            0.0 <= entropy <= log_classes
            and -1.0 <= last_margin <= 1.0
            and 0.0 <= frequency <= 1.0
            and -1.0 <= margin_ema <= 1.0
        ):
            raise LoggingOnlyPredictionError("frozen H2 feature is outside its documented range")
        prepared.append(
            {
                "namespace": "train",
                "sample_id": sample_id,
                "class_id": class_id,
                "outcome": outcome,
                "teacher": entropy / log_classes,
                "previous_correctness": float(previous_correct),
                "last_margin_risk": (1.0 - last_margin) / 2.0,
                "robust_correct_frequency": frequency,
                "margin_ema_risk": (1.0 - margin_ema) / 2.0,
            }
        )
    if seen != set(range(EXPECTED_COUNT)):
        raise LoggingOnlyPredictionError("state export stable IDs must be exactly the 45000-source train partition")
    return run_id, prepared, source_identity


def _features(row: Mapping[str, float | int], model: str) -> list[float]:
    teacher = float(row["teacher"])
    history = [float(row["robust_correct_frequency"]), float(row["margin_ema_risk"])]
    if model == "teacher_only":
        return [teacher]
    if model == "student_only":
        return history
    if model == "main_effects":
        return [teacher, *history]
    if model == "main_effects_plus_product":
        return [teacher, *history, teacher * history[0], teacher * history[1]]
    if model == "current_correctness":
        return [float(row["previous_correctness"])]
    if model == "instantaneous_margin":
        return [float(row["last_margin_risk"])]
    if model == "current_only":
        return [float(row["previous_correctness"]), float(row["last_margin_risk"])]
    raise AssertionError(f"unknown frozen model {model}")


def _calibration_metrics(targets: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    if len(targets) != len(scores) or not targets:
        raise LoggingOnlyPredictionError("calibration targets and scores must align")
    brier = sum((target - score) ** 2 for target, score in zip(targets, scores, strict=True)) / len(targets)
    bins: list[list[int]] = [[] for _ in range(10)]
    for index, score in enumerate(scores):
        bins[min(9, int(score * 10.0))].append(index)
    ece = sum(
        (len(indices) / len(targets))
        * abs(
            sum(scores[index] for index in indices) / len(indices)
            - sum(targets[index] for index in indices) / len(indices)
        )
        for indices in bins
        if indices
    )
    return {"brier_score": brier, "fixed_10_bin_ece": ece}


def _select_best_current(metrics: Mapping[str, Mapping[str, Any]]) -> str:
    return min(
        CURRENT_BASELINES,
        key=lambda name: (-float(metrics[name]["auroc"]), float(metrics[name]["log_loss"]), name),
    )


def _trajectory_report(rows: Sequence[dict[str, Any]], *, design: _FrozenDesign) -> dict[str, Any]:
    try:
        train_ids, held_out_ids = deterministic_hash_split(rows, seed=design.split_seed, held_out_fraction=0.2)
    except SignalAuditError as exc:
        raise LoggingOnlyPredictionError("frozen deterministic split failed") from exc
    train_set, held_out_set = set(train_ids), set(held_out_ids)
    train = [row for row in rows if int(row["sample_id"]) in train_set]
    held_out = [row for row in rows if int(row["sample_id"]) in held_out_set]
    if len(train) + len(held_out) != len(rows) or not train or not held_out:
        raise LoggingOnlyPredictionError("frozen deterministic split did not partition the source population")
    targets = [int(row["outcome"]) for row in held_out]
    reports: dict[str, dict[str, Any]] = {}
    predictions: dict[str, list[float]] = {}
    for model in MODEL_ORDER:
        try:
            fit = _fit_logistic([_features(row, model) for row in train], [int(row["outcome"]) for row in train])
            scores = _predict_logistic(fit, [_features(row, model) for row in held_out])
            reports[model] = {
                **binary_metrics(targets, scores),
                **_calibration_metrics(targets, scores),
                "bootstrap_95": bootstrap_binary_metric_intervals(
                    held_out, scores=scores, seed=design.bootstrap_seed, replicates=design.bootstrap_replicates
                ),
            }
            predictions[model] = scores
        except SignalAuditError as exc:
            raise LoggingOnlyPredictionError(f"frozen logistic analysis failed for {model}") from exc
    best_current = _select_best_current(reports)
    try:
        bootstrap = bootstrap_metric_delta(
            held_out,
            baseline=predictions[best_current],
            candidate=predictions["student_only"],
            seed=design.bootstrap_seed,
            replicates=design.bootstrap_replicates,
        )
    except SignalAuditError as exc:
        raise LoggingOnlyPredictionError("paired history-vs-current bootstrap failed") from exc
    delta = float(reports["student_only"]["auroc"]) - float(reports[best_current]["auroc"])
    log_loss_improved = float(reports["student_only"]["log_loss"]) < float(reports[best_current]["log_loss"])
    if delta >= 0.02 and float(bootstrap["lower"]) > 0.0 and log_loss_improved:
        decision = "go"
    elif float(bootstrap["upper"]) < 0.01 or not log_loss_improved:
        decision = "no_go"
    else:
        decision = "inconclusive"
    paired_comparisons: dict[str, Any] = {}
    for baseline_name, candidate_name in (
        ("teacher_only", "student_only"),
        ("student_only", "main_effects"),
        ("main_effects", "main_effects_plus_product"),
    ):
        try:
            interval = bootstrap_metric_delta(
                held_out,
                baseline=predictions[baseline_name],
                candidate=predictions[candidate_name],
                seed=design.bootstrap_seed,
                replicates=design.bootstrap_replicates,
            )
        except SignalAuditError as exc:
            raise LoggingOnlyPredictionError(
                f"paired comparison failed for {baseline_name} to {candidate_name}"
            ) from exc
        paired_comparisons[f"{baseline_name}_to_{candidate_name}"] = {
            "baseline": baseline_name,
            "candidate": candidate_name,
            "point_deltas": {
                metric: float(reports[candidate_name][metric]) - float(reports[baseline_name][metric])
                for metric in ("auroc", "auprc", "log_loss")
            },
            "paired_auroc_bootstrap_95": interval,
        }
    return {
        "outcome": "future_online_forgetting",
        "train_sample_count": len(train),
        "held_out_sample_count": len(held_out),
        "models": reports,
        "paired_comparisons": paired_comparisons,
        "history_vs_best_current": {
            "history_model": "student_only",
            "best_current_model": best_current,
            "delta_auroc": delta,
            "paired_auroc_bootstrap_95": bootstrap,
            "held_out_log_loss_improved": log_loss_improved,
            "decision": decision,
        },
    }


def analyze_logging_only_exports(
    exports: Mapping[str, Mapping[str, Any]],
    *,
    design_path: Path | None = None,
    block_path: Path | None = None,
    analysis_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen H2 contract to labeled, hash-bound state exports."""
    if not exports:
        raise LoggingOnlyPredictionError("at least one H2 state export is required")
    if set(exports) != set(RUN_LABELS):
        raise LoggingOnlyPredictionError("H2 analysis is fail-closed until exactly L1, L2, L3, and L4 are present")
    for label, export in exports.items():
        identity = export.get("identity") if isinstance(export, Mapping) else None
        if not isinstance(identity, Mapping) or identity.get("run_id") != EXPECTED_RUN_IDS[label]:
            raise LoggingOnlyPredictionError(f"H2 label {label} does not bind its frozen expected run ID")
    design = load_frozen_design(design_path=design_path, block_path=block_path)
    run_ids: set[str] = set()
    reports: dict[str, Any] = {}
    for label in sorted(exports):
        run_id, rows, source_identity = _prepare_rows(exports[label])
        if run_id in run_ids:
            raise LoggingOnlyPredictionError("H2 state exports contain duplicate run IDs")
        run_ids.add(run_id)
        reports[label] = {
            "run_id": run_id,
            "source_identity": source_identity,
            **_trajectory_report(rows, design=design),
        }
    provenance = _tracked_clean_analysis_provenance() if analysis_provenance is None else dict(analysis_provenance)
    bartoldson_ready = all(
        reports.get(label, {}).get("history_vs_best_current", {}).get("decision") == "go" for label in ("L1", "L3")
    )
    available = [reports[label] for label in RUN_LABELS if label in reports]
    cross_teacher_ready = len(available) == len(RUN_LABELS) and all(
        item["history_vs_best_current"]["delta_auroc"] > 0.0
        and item["history_vs_best_current"]["held_out_log_loss_improved"]
        for item in available
    )
    return {
        "identity": {
            "schema_version": 1,
            "contract": PREDICTION_CONTRACT,
            "inherited_design_id": design.design_id,
            "inherited_design_sha256": design.design_sha256,
            "block_design_sha256": design.block_sha256,
            "analysis_provenance": provenance,
        },
        "runs": reports,
        "block": {
            "required_identities": {
                "L1": {"teacher": "bartoldson2024_adversarial_wrn94_16", "seed": 1},
                "L2": {"teacher": "chen2021_ltd_wrn34_10", "seed": 1},
                "L3": {"teacher": "bartoldson2024_adversarial_wrn94_16", "seed": 2},
                "L4": {"teacher": "chen2021_ltd_wrn34_10", "seed": 2},
            },
            "status": "complete",
            "bartoldson_selector_admissible": bartoldson_ready,
            "cross_teacher_history_claim": cross_teacher_ready,
            "decision_rules": {
                "bartoldson_selector": "L1 and L3 must both be Go",
                "cross_teacher_claim": "all L1-L4 require positive AUROC delta and improved log loss",
            },
        },
    }


def load_state_export(path: Path) -> dict[str, Any]:
    value, _ = load_state_export_with_provenance(path)
    return value


def load_state_export_with_provenance(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Parse one exact JSON byte stream and preserve its resolved-path digest."""
    resolved = path.resolve()
    try:
        raw = resolved.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LoggingOnlyPredictionError(f"cannot read state export {resolved}") from exc
    if not isinstance(value, dict):
        raise LoggingOnlyPredictionError("state export JSON must be an object")
    return value, {"path": str(resolved), "sha256": hashlib.sha256(raw).hexdigest()}
