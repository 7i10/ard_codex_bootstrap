"""Fail-closed frozen selector for the post-H2 factorial screen.

This is deliberately an offline boundary.  It fits only the immutable seed-0
checkpoint-panel data, scores only the hash-bound L3 checkpoint-panel input,
and emits fixed train-ID masks.  It neither opens a training checkpoint nor
accepts online ``SampleStateStore`` values.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ard.analysis.rslad_signal_replay import RSLADSignalReplayError, tracked_clean_analysis_provenance
from ard.analysis.signal_audit import (
    SignalAuditError,
    _fit_logistic,
    _predict_logistic,
    binary_metrics,
    deterministic_hash_split,
)


class SelectorBundleError(ValueError):
    """A selector input cannot prove the registered prospective contract."""


SEED0_FEATURE_PANEL_SHA256 = "21df19cc431c6be0343d7074d658d208825f40d6f0ed13cede28591aa273effa"
SEED0_OUTCOME_PANEL_SHA256 = "a4f1dcb77d9a0f0df9792022fd0d6c4f368f943ed6928d527a5503c16bfaf4a3"
SEED0_REPORT_SHA256 = "d44ee166f8866b77067ebd07757d394a060242c9cf1cdc5d4513f127897981f8"
SEED0_LINEAGE_SHA256 = "9b6ea091dc9ed4ff81bb579bf05d6650ac8e6d4ab6104981c446f29069e4a64e"
CONFIRMATORY_DESIGN_SHA256 = "a0a7fe0e70fcc8aaf519440012900c7bd8e6db92a8f0143d06892fca1146dd38"
PREDICTOR_SPEC_SHA256 = "d653d9ef08cfa94976a0e3279166b47543d16f3eaadb69810769470b77838c12"
L3_SCIENTIFIC_GIT_SHA = "8254a8899ae7373c2f541d108593e5c8185b26f5"
L3_CONFIG_SHA256 = "bc9fe4223e00c00a3add329166dc4a7441273fcc94f670f93def3927e805f054"
L3_PARENT_CHECKPOINT_SHA256 = "44ac2edb9526917aa3ba1e0f9bd92a3355ed0a93a4a4fce541600a1fc71eb501"
L3_PARENT_SAMPLE_STATE_SHA256 = "dd79c805ff3838cedb174c003f6a8804a75224e536b3498e6c285a8e83f86356"
L3_TEACHER_CHECKPOINT_SHA256 = "56bbad8ad748df86e67c24dba4f59a9e7d285e583251460b2ed154017a18cb0b"
TRAIN_COUNT = 45_000
NUM_CLASSES = 10
ANCHOR_EPOCH = 99
K = 3566
SPLIT_SEED = 2026073101
HELD_OUT_FRACTION = 0.2
CLASS_MATCHED_RANDOM_SEED = 2026080201
FEATURE_NAMES = ("student_robust_correct_frequency", "student_margin_historical_risk")
FEATURE_COLUMNS = (
    "namespace",
    "sample_id",
    "class_id",
    "feature_epoch",
    "teacher_entropy_normalized",
    "student_robust_correct_epoch99",
    "student_robust_correct_frequency",
    "student_margin_historical_ema",
    "student_margin_historical_risk",
    "student_margin_instantaneous_epoch99",
    "student_margin_panel_ema",
    "student_margin_panel_risk",
    "student_margin_epoch99",
    "student_margin_risk_epoch99",
)
OUTCOME_COLUMNS = (
    "namespace",
    "sample_id",
    "class_id",
    "outcome_start_epoch",
    "outcome_end_epoch",
    "checkpoint_panel_forgetting",
    "checkpoint_panel_transition_count",
    "final_robust_error",
    "persistent_wrong",
    "post_anchor_robust_correct_frequency",
)


@dataclass(frozen=True)
class SelectorFiles:
    """Exact source paths used to construct the immutable selector bundle."""

    seed0_feature_panel: Path
    seed0_outcome_panel: Path
    seed0_report: Path
    seed0_lineage: Path
    l3_feature_panel: Path
    l3_lineage: Path


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SelectorBundleError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SelectorBundleError(f"{name} must be finite")
    return float(value)


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorBundleError(f"{name} is unreadable JSON") from exc
    if not isinstance(value, dict):
        raise SelectorBundleError(f"{name} must be a JSON mapping")
    return value


def _read_parquet(path: Path, *, columns: Sequence[str], name: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SelectorBundleError(f"{name} is missing")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise SelectorBundleError("selector bundle requires pyarrow for exact Parquet inputs") from exc
    try:
        table = pq.read_table(path)
    except Exception as exc:  # pragma: no cover - pyarrow error types differ by version
        raise SelectorBundleError(f"{name} is not readable Parquet") from exc
    if tuple(table.column_names) != tuple(columns):
        raise SelectorBundleError(f"{name} column contract drifted")
    return [dict(row) for row in table.to_pylist()]


def _validate_feature_rows(rows: Sequence[Mapping[str, Any]], *, name: str) -> list[dict[str, Any]]:
    if len(rows) != TRAIN_COUNT:
        raise SelectorBundleError(f"{name} must contain exactly {TRAIN_COUNT} train rows")
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        sample_id = _integer(row.get("sample_id"), name=f"{name} sample_id")
        class_id = _integer(row.get("class_id"), name=f"{name} class_id")
        if row.get("namespace") != "train" or class_id >= NUM_CLASSES or sample_id >= 50_000 or sample_id in seen:
            raise SelectorBundleError(f"{name} violates the exact CIFAR train namespace")
        if row.get("feature_epoch") != ANCHOR_EPOCH:
            raise SelectorBundleError(f"{name} is not an epoch-{ANCHOR_EPOCH} checkpoint panel")
        values = {field: _finite(row.get(field), name=f"{name} {field}") for field in FEATURE_NAMES}
        if not all(0.0 <= value <= 1.0 for value in values.values()):
            raise SelectorBundleError(f"{name} history feature is outside [0, 1]")
        seen.add(sample_id)
        output.append({"namespace": "train", "sample_id": sample_id, "class_id": class_id, **values})
    return sorted(output, key=lambda row: int(row["sample_id"]))


def _validate_outcome_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, tuple[int, int]]:
    if len(rows) != TRAIN_COUNT:
        raise SelectorBundleError(f"seed-0 outcome panel must contain exactly {TRAIN_COUNT} train rows")
    output: dict[int, tuple[int, int]] = {}
    for row in rows:
        sample_id = _integer(row.get("sample_id"), name="outcome sample_id")
        class_id = _integer(row.get("class_id"), name="outcome class_id")
        outcome = row.get("checkpoint_panel_forgetting")
        if (
            row.get("namespace") != "train"
            or class_id >= NUM_CLASSES
            or sample_id >= 50_000
            or sample_id in output
            or row.get("outcome_start_epoch") != ANCHOR_EPOCH
            or row.get("outcome_end_epoch") != 199
            or outcome not in {0, 1, False, True}
        ):
            raise SelectorBundleError("seed-0 outcome panel violates the frozen checkpoint-panel contract")
        output[sample_id] = (class_id, int(outcome))
    return output


def _seed0_fit(files: SelectorFiles) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = (
        (files.seed0_feature_panel, SEED0_FEATURE_PANEL_SHA256, "seed-0 feature panel"),
        (files.seed0_outcome_panel, SEED0_OUTCOME_PANEL_SHA256, "seed-0 outcome panel"),
        (files.seed0_report, SEED0_REPORT_SHA256, "seed-0 predictive report"),
        (files.seed0_lineage, SEED0_LINEAGE_SHA256, "seed-0 lineage"),
    )
    for path, digest, name in expected:
        if not path.is_file() or sha256_file(path) != digest:
            raise SelectorBundleError(f"{name} bytes do not match the frozen SHA-256")
    lineage = _read_json(files.seed0_lineage, name="seed-0 lineage")
    output_hashes = lineage.get("output_parquet_sha256")
    if (
        not isinstance(output_hashes, Mapping)
        or output_hashes.get("feature_panel") != SEED0_FEATURE_PANEL_SHA256
        or output_hashes.get("outcome_panel") != SEED0_OUTCOME_PANEL_SHA256
    ):
        raise SelectorBundleError("seed-0 lineage does not bind both frozen panel hashes")
    if (
        lineage.get("predictive_audit_sha256") != SEED0_REPORT_SHA256
        or lineage.get("train_expected_count") != TRAIN_COUNT
    ):
        raise SelectorBundleError("seed-0 lineage does not bind the frozen predictive audit")
    features = _validate_feature_rows(
        _read_parquet(files.seed0_feature_panel, columns=FEATURE_COLUMNS, name="seed-0 feature panel"),
        name="seed-0 feature panel",
    )
    outcomes = _validate_outcome_rows(
        _read_parquet(files.seed0_outcome_panel, columns=OUTCOME_COLUMNS, name="seed-0 outcome panel")
    )
    rows: list[dict[str, Any]] = []
    for row in features:
        outcome = outcomes.get(int(row["sample_id"]))
        if outcome is None or outcome[0] != row["class_id"]:
            raise SelectorBundleError("seed-0 feature/outcome panels do not have identical train identity")
        rows.append({**row, "outcome": outcome[1]})
    if len(outcomes) != len(rows):
        raise SelectorBundleError("seed-0 feature/outcome panels do not have identical train IDs")
    train_ids, held_out_ids = deterministic_hash_split(rows, seed=SPLIT_SEED, held_out_fraction=HELD_OUT_FRACTION)
    train_id_set = set(train_ids)
    held_out_id_set = set(held_out_ids)
    train = [row for row in rows if int(row["sample_id"]) in train_id_set]
    held_out = [row for row in rows if int(row["sample_id"]) in held_out_id_set]
    try:
        fit = _fit_logistic(
            [[float(row[field]) for field in FEATURE_NAMES] for row in train], [int(row["outcome"]) for row in train]
        )
        held_scores = _predict_logistic(fit, [[float(row[field]) for field in FEATURE_NAMES] for row in held_out])
        metrics = binary_metrics([int(row["outcome"]) for row in held_out], held_scores)
    except SignalAuditError as exc:
        raise SelectorBundleError("frozen seed-0 history fit failed") from exc
    report = _read_json(files.seed0_report, name="seed-0 predictive report")
    if report.get("outcome") != "checkpoint_panel_forgetting":
        raise SelectorBundleError("seed-0 report outcome is not checkpoint_panel_forgetting")
    report_split = report.get("split_identity")
    report_model = report.get("models", {}).get("history_only") if isinstance(report.get("models"), Mapping) else None
    if not isinstance(report_split, Mapping) or not isinstance(report_model, Mapping):
        raise SelectorBundleError("seed-0 report lacks frozen history-only split/metrics")
    if (
        tuple(report_split.get("train_sample_ids", ())) != train_ids
        or tuple(report_split.get("held_out_sample_ids", ())) != held_out_ids
    ):
        raise SelectorBundleError("seed-0 report split does not reproduce the frozen deterministic partition")
    for name in ("auroc", "auprc", "prevalence", "log_loss"):
        observed = report_model.get(name)
        if not isinstance(observed, (int, float)) or abs(float(observed) - float(metrics[name])) > 1e-12:
            raise SelectorBundleError(f"seed-0 held-out history-only {name} does not exactly reproduce")
    weights, means, scales = fit
    fit_payload = {
        "feature_names": list(FEATURE_NAMES),
        "weights": list(weights),
        "means": list(means),
        "scales": list(scales),
    }
    return {
        "feature_names": list(FEATURE_NAMES),
        "outcome": "checkpoint_panel_forgetting",
        "split": {
            "method": "true_class_stratified_deterministic_hash",
            "seed": SPLIT_SEED,
            "held_out_fraction": HELD_OUT_FRACTION,
            "train_ids_sha256": hashlib.sha256(canonical_json(list(train_ids))).hexdigest(),
            "held_out_ids_sha256": hashlib.sha256(canonical_json(list(held_out_ids))).hexdigest(),
        },
        "weights": list(weights),
        "means": list(means),
        "scales": list(scales),
        "coefficients_sha256": hashlib.sha256(canonical_json(list(weights))).hexdigest(),
        "preprocessing_sha256": hashlib.sha256(
            canonical_json({"means": list(means), "scales": list(scales)})
        ).hexdigest(),
        "fit_sha256": hashlib.sha256(canonical_json(fit_payload)).hexdigest(),
        "held_out_metrics": metrics,
    }, rows


def _validate_l3_lineage(
    path: Path, *, expected_feature_sha256: str, seed0_lineage: Mapping[str, Any]
) -> dict[str, Any]:
    lineage = _read_json(path, name="L3 feature lineage")
    required = {
        "schema_version": 1,
        "kind": "l3_checkpoint_panel_feature_source_v1",
        "run_id": "bart-rslad-observed-s2-confirm-v2",
        "teacher_registry_id": "bartoldson2024_adversarial_wrn94_16",
        "seed": 2,
        "parent_epoch": ANCHOR_EPOCH,
    }
    if any(lineage.get(key) != value for key, value in required.items()):
        raise SelectorBundleError("L3 feature lineage identity drifted")
    if (
        lineage.get("scientific_git_sha") != L3_SCIENTIFIC_GIT_SHA
        or lineage.get("config_hash") != L3_CONFIG_SHA256
        or lineage.get("parent_raw_config_sha256") != L3_CONFIG_SHA256
        or lineage.get("parent_checkpoint_sha256") != L3_PARENT_CHECKPOINT_SHA256
        or lineage.get("parent_sample_state_sha256") != L3_PARENT_SAMPLE_STATE_SHA256
    ):
        raise SelectorBundleError("L3 run/parent/teacher identity does not match the registered source")
    if lineage.get("feature_panel_sha256") != expected_feature_sha256:
        raise SelectorBundleError("L3 lineage does not bind the supplied feature-panel bytes")
    for field in (
        "attack_identity",
        "feature_protocol",
        "checkpoint_training",
        "dataset_identity",
        "teacher",
    ):
        seed0_value = seed0_lineage.get(field)
        if not isinstance(seed0_value, Mapping) or lineage.get(field) != seed0_value:
            raise SelectorBundleError(f"L3 {field} does not exactly match the seed-0 feature domain")
    if lineage.get("train_expected_count") != TRAIN_COUNT or lineage.get("train_expected_count") != seed0_lineage.get(
        "train_expected_count"
    ):
        raise SelectorBundleError("L3 train partition count does not match the seed-0 feature domain")
    teacher = lineage["teacher"]
    assert isinstance(teacher, Mapping)
    if teacher.get("checkpoint_sha256") != L3_TEACHER_CHECKPOINT_SHA256:
        raise SelectorBundleError("L3 teacher metadata does not bind the registered checkpoint")
    try:
        current_provenance = tracked_clean_analysis_provenance()
    except RSLADSignalReplayError as exc:
        raise SelectorBundleError("L3 selector requires current tracked-clean replay provenance") from exc
    if lineage.get("analysis_provenance") != current_provenance:
        raise SelectorBundleError("L3 analysis provenance does not match the current replay implementation")
    return lineage


def _l3_scores(files: SelectorFiles, fit_mapping: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not files.l3_feature_panel.is_file() or not files.l3_lineage.is_file():
        raise SelectorBundleError("L3 scoring inputs are missing")
    panel_hash = sha256_file(files.l3_feature_panel)
    lineage_hash = sha256_file(files.l3_lineage)
    lineage = _validate_l3_lineage(
        files.l3_lineage,
        expected_feature_sha256=panel_hash,
        seed0_lineage=_read_json(files.seed0_lineage, name="seed-0 lineage"),
    )
    rows = _validate_feature_rows(
        _read_parquet(files.l3_feature_panel, columns=FEATURE_COLUMNS, name="L3 feature panel"), name="L3 feature panel"
    )
    fit = (
        tuple(float(value) for value in fit_mapping["weights"]),
        tuple(float(value) for value in fit_mapping["means"]),
        tuple(float(value) for value in fit_mapping["scales"]),
    )
    scores = _predict_logistic(fit, [[float(row[field]) for field in FEATURE_NAMES] for row in rows])
    scored = [{**row, "score": score} for row, score in zip(rows, scores, strict=True)]
    return scored, {
        "feature_panel_path": str(files.l3_feature_panel.resolve()),
        "feature_panel_sha256": panel_hash,
        "lineage_path": str(files.l3_lineage.resolve()),
        "lineage_sha256": lineage_hash,
        "identity": lineage,
    }


def _history_selection(scored: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(scored, key=lambda row: (-float(row["score"]), int(row["sample_id"])))
    if len(ordered) != TRAIN_COUNT or K > len(ordered):
        raise SelectorBundleError("L3 score population cannot support the fixed K")
    chosen = [dict(row) for row in ordered[:K]]
    threshold = float(chosen[-1]["score"])
    boundary = [row for row in ordered if float(row["score"]) == threshold]
    class_counts = {str(key): value for key, value in sorted(Counter(int(row["class_id"]) for row in chosen).items())}
    score_order_ids = [int(row["sample_id"]) for row in chosen]
    return chosen, {
        "k": K,
        "order": "descending_probability_then_sample_id_ascending",
        "threshold": threshold,
        "boundary_tie_count": len(boundary),
        "boundary_selected_count": sum(float(row["score"]) == threshold for row in chosen),
        "selected_score_order_sha256": hashlib.sha256(canonical_json(score_order_ids)).hexdigest(),
        "selected_ids_sha256": hashlib.sha256(canonical_json(sorted(score_order_ids))).hexdigest(),
        "selected_class_counts": class_counts,
    }


def _mask_payload(
    *, selected_ids: Sequence[int], labels: Mapping[int, int], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    ids = list(sorted(selected_ids))
    class_counts = {
        str(label): count for label, count in sorted(Counter(labels[sample_id] for sample_id in ids).items())
    }
    return {
        "schema_version": 1,
        "namespace": "train",
        "num_classes": NUM_CLASSES,
        "selected_ids": ids,
        "selected_ids_sha256": hashlib.sha256(canonical_json(ids)).hexdigest(),
        "selected_count": len(ids),
        "selected_class_counts": class_counts,
        "provenance": dict(provenance),
    }


def _random_selection(*, labels: Mapping[int, int], class_counts: Mapping[str, int]) -> list[int]:
    selected: list[int] = []
    for raw_class, count in sorted(class_counts.items(), key=lambda item: int(item[0])):
        class_id, expected = int(raw_class), int(count)
        candidates = [sample_id for sample_id, label in labels.items() if label == class_id]
        ranked = sorted(
            candidates,
            key=lambda sample_id: hashlib.sha256(
                f"{CLASS_MATCHED_RANDOM_SEED}:{class_id}:{sample_id}".encode()
            ).digest(),
        )
        if len(ranked) < expected:
            raise SelectorBundleError("class-matched random selection exceeds L3 class population")
        selected.extend(ranked[:expected])
    return sorted(selected)


def build_selector_bundle(*, files: SelectorFiles, output_dir: Path) -> dict[str, Path]:
    """Build a fixed selector specification and its two reproducible mask bytes."""
    fit, _ = _seed0_fit(files)
    scored, l3_input = _l3_scores(files, fit)
    chosen, selection = _history_selection(scored)
    labels = {int(row["sample_id"]): int(row["class_id"]) for row in scored}
    history_ids = [int(row["sample_id"]) for row in chosen]
    random_ids = _random_selection(labels=labels, class_counts=selection["selected_class_counts"])
    output_dir.mkdir(parents=True, exist_ok=False)
    history_path, random_path, bundle_path = (
        output_dir / "history-mask.json",
        output_dir / "random-mask.json",
        output_dir / "selector-bundle.json",
    )
    bundle = {
        "schema_version": 1,
        "kind": "post_h2_prospective_intervention_selector_v2",
        "confirmatory_design_sha256": CONFIRMATORY_DESIGN_SHA256,
        "predictor_spec_sha256": PREDICTOR_SPEC_SHA256,
        "seed0_inputs": {
            "feature_panel_path": str(files.seed0_feature_panel.resolve()),
            "feature_panel_sha256": SEED0_FEATURE_PANEL_SHA256,
            "outcome_panel_path": str(files.seed0_outcome_panel.resolve()),
            "outcome_panel_sha256": SEED0_OUTCOME_PANEL_SHA256,
            "predictive_report_path": str(files.seed0_report.resolve()),
            "predictive_report_sha256": SEED0_REPORT_SHA256,
            "lineage_path": str(files.seed0_lineage.resolve()),
            "lineage_sha256": SEED0_LINEAGE_SHA256,
        },
        "seed0_fit": fit,
        "l3_scoring_input": l3_input,
        "selection": selection,
        "random_control": {
            "seed": CLASS_MATCHED_RANDOM_SEED,
            "ranking": "sha256(seed:class_id:sample_id)",
            "numpy_rng": False,
        },
        "mask_paths": {"history": str(history_path.resolve()), "random": str(random_path.resolve())},
    }
    bundle_path.write_bytes(canonical_json(bundle))
    bundle_hash = sha256_file(bundle_path)
    parent = l3_input["identity"]
    history = _mask_payload(
        selected_ids=history_ids,
        labels=labels,
        provenance={
            "source": "seed0_bartoldson_frozen_predictor",
            "approved_selector_spec_sha256": bundle_hash,
            "selector_spec_path": str(bundle_path.resolve()),
            "parent_checkpoint_sha256": parent["parent_checkpoint_sha256"],
            "parent_sample_state_sha256": parent["parent_sample_state_sha256"],
            "random_seed": None,
            "generator": None,
            "generator_version": None,
            "reference_history_mask_sha256": None,
            "reference_selected_count": None,
            "reference_selected_class_counts": None,
            "reference_history_selector_spec_sha256": None,
        },
    )
    history_path.write_bytes(canonical_json(history))
    history_hash = sha256_file(history_path)
    random = _mask_payload(
        selected_ids=random_ids,
        labels=labels,
        provenance={
            "source": "class_matched_random",
            "approved_selector_spec_sha256": None,
            "selector_spec_path": None,
            "parent_checkpoint_sha256": parent["parent_checkpoint_sha256"],
            "parent_sample_state_sha256": parent["parent_sample_state_sha256"],
            "random_seed": CLASS_MATCHED_RANDOM_SEED,
            "generator": "sha256_rank",
            "generator_version": "seed_class_sample_id_v1",
            "reference_history_mask_sha256": history_hash,
            "reference_selected_count": K,
            "reference_selected_class_counts": selection["selected_class_counts"],
            "reference_history_selector_spec_sha256": bundle_hash,
        },
    )
    random_path.write_bytes(canonical_json(random))
    return {"bundle": bundle_path, "history_mask": history_path, "random_mask": random_path}


def _load_mask(path: Path, *, name: str) -> dict[str, Any]:
    value = _read_json(path, name=name)
    if set(value) != {
        "schema_version",
        "namespace",
        "num_classes",
        "selected_ids",
        "selected_ids_sha256",
        "selected_count",
        "selected_class_counts",
        "provenance",
    }:
        raise SelectorBundleError(f"{name} schema drifted")
    return value


def verify_selector_bundle(
    *,
    bundle_path: Path,
    history_mask_path: Path | None = None,
    random_mask_path: Path | None = None,
    expected_parent: Mapping[str, Any] | None = None,
    expected_train_labels: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Recompute the entire selector and reject byte-consistent forged masks."""
    bundle = _read_json(bundle_path, name="selector bundle")
    required = {
        "schema_version",
        "kind",
        "confirmatory_design_sha256",
        "predictor_spec_sha256",
        "seed0_inputs",
        "seed0_fit",
        "l3_scoring_input",
        "selection",
        "random_control",
        "mask_paths",
    }
    if (
        set(bundle) != required
        or bundle.get("schema_version") != 1
        or bundle.get("kind") != "post_h2_prospective_intervention_selector_v2"
        or bundle.get("confirmatory_design_sha256") != CONFIRMATORY_DESIGN_SHA256
        or bundle.get("predictor_spec_sha256") != PREDICTOR_SPEC_SHA256
    ):
        raise SelectorBundleError("selector bundle identity drifted")
    source, l3 = bundle.get("seed0_inputs"), bundle.get("l3_scoring_input")
    if not isinstance(source, Mapping) or not isinstance(l3, Mapping):
        raise SelectorBundleError("selector bundle source mappings are missing")
    files = SelectorFiles(
        seed0_feature_panel=Path(str(source.get("feature_panel_path"))),
        seed0_outcome_panel=Path(str(source.get("outcome_panel_path"))),
        seed0_report=Path(str(source.get("predictive_report_path"))),
        seed0_lineage=Path(str(source.get("lineage_path"))),
        l3_feature_panel=Path(str(l3.get("feature_panel_path"))),
        l3_lineage=Path(str(l3.get("lineage_path"))),
    )
    fit, _ = _seed0_fit(files)
    if bundle.get("seed0_fit") != fit:
        raise SelectorBundleError("selector bundle seed-0 fit/provenance is not reproducible")
    scored, reconstructed_l3 = _l3_scores(files, fit)
    if l3 != reconstructed_l3:
        raise SelectorBundleError("selector bundle L3 scoring input is not reproducible")
    parent = reconstructed_l3["identity"]
    if expected_parent is not None:
        for field in ("parent_checkpoint_sha256", "parent_sample_state_sha256", "parent_raw_config_sha256"):
            if parent[field] != expected_parent.get(field):
                raise SelectorBundleError(f"selector bundle L3 parent {field} does not match fork parent")
    chosen, selection = _history_selection(scored)
    if bundle.get("selection") != selection or bundle.get("random_control") != {
        "seed": CLASS_MATCHED_RANDOM_SEED,
        "ranking": "sha256(seed:class_id:sample_id)",
        "numpy_rng": False,
    }:
        raise SelectorBundleError("selector bundle selection rule is not reproducible")
    labels = {int(row["sample_id"]): int(row["class_id"]) for row in scored}
    if expected_train_labels is not None and dict(expected_train_labels) != labels:
        raise SelectorBundleError("L3 checkpoint-panel labels do not exactly match the fork train partition")
    paths = bundle.get("mask_paths")
    if not isinstance(paths, Mapping):
        raise SelectorBundleError("selector bundle mask paths are missing")
    history_path = Path(str(paths.get("history"))) if history_mask_path is None else history_mask_path
    random_path = Path(str(paths.get("random"))) if random_mask_path is None else random_mask_path
    history = _load_mask(history_path, name="history mask")
    random = _load_mask(random_path, name="random mask")
    bundle_hash = sha256_file(bundle_path)
    expected_history = _mask_payload(
        selected_ids=[int(row["sample_id"]) for row in chosen],
        labels=labels,
        provenance={
            "source": "seed0_bartoldson_frozen_predictor",
            "approved_selector_spec_sha256": bundle_hash,
            "selector_spec_path": str(bundle_path.resolve()),
            "parent_checkpoint_sha256": parent["parent_checkpoint_sha256"],
            "parent_sample_state_sha256": parent["parent_sample_state_sha256"],
            "random_seed": None,
            "generator": None,
            "generator_version": None,
            "reference_history_mask_sha256": None,
            "reference_selected_count": None,
            "reference_selected_class_counts": None,
            "reference_history_selector_spec_sha256": None,
        },
    )
    if history != expected_history:
        raise SelectorBundleError("history mask IDs or bytes do not reproduce the frozen selector")
    expected_random = _mask_payload(
        selected_ids=_random_selection(labels=labels, class_counts=selection["selected_class_counts"]),
        labels=labels,
        provenance={
            "source": "class_matched_random",
            "approved_selector_spec_sha256": None,
            "selector_spec_path": None,
            "parent_checkpoint_sha256": parent["parent_checkpoint_sha256"],
            "parent_sample_state_sha256": parent["parent_sample_state_sha256"],
            "random_seed": CLASS_MATCHED_RANDOM_SEED,
            "generator": "sha256_rank",
            "generator_version": "seed_class_sample_id_v1",
            "reference_history_mask_sha256": sha256_file(history_path),
            "reference_selected_count": K,
            "reference_selected_class_counts": selection["selected_class_counts"],
            "reference_history_selector_spec_sha256": bundle_hash,
        },
    )
    if random != expected_random:
        raise SelectorBundleError("random mask IDs or bytes do not reproduce the frozen class-matched control")
    return {
        "bundle_sha256": bundle_hash,
        "history_mask_sha256": sha256_file(history_path),
        "random_mask_sha256": sha256_file(random_path),
        "history": history,
        "random": random,
        "selection": selection,
        "l3_identity": parent,
    }
