"""Fail-closed H5-Late screen on the frozen common-PGD replay panels."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ard.analysis.intervention_selector import FEATURE_COLUMNS, FEATURE_NAMES
from ard.analysis.rslad_signal_replay import repository_root_from_source
from ard.analysis.signal_audit import SignalAuditError, _predict_logistic, binary_metrics, canonical_json, sha256_file


class HistoryScreenError(ValueError):
    """An H5-Late input cannot prove the fixed common-PGD contract."""


FIT_CONTRACT = "h5_frozen_fixed_fit_bundle_v1"
REPORT_CONTRACT = "h5_late_common_pgd_history_screen_v1"
ANCHOR_EPOCH = 99
OUTCOME_END_EPOCH = 199
Q = 0.10


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _hash_ids(ids: Sequence[int]) -> str:
    return _sha256(list(sorted(ids)))


def _sha(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise HistoryScreenError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HistoryScreenError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str, lower: float = -float("inf"), upper: float = float("inf")) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise HistoryScreenError(f"{name} must be finite")
    result = float(value)
    if not lower <= result <= upper:
        raise HistoryScreenError(f"{name} is outside its documented range")
    return result


def _binary(value: object, *, name: str) -> int:
    if value not in {0, 1, False, True}:
        raise HistoryScreenError(f"{name} must be binary")
    return int(value)


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoryScreenError(f"cannot read {name}") from exc
    if not isinstance(value, dict):
        raise HistoryScreenError(f"{name} must be a JSON object")
    return value


def _read_parquet(path: Path, *, columns: Sequence[str], name: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - explicit optional dependency boundary
        raise HistoryScreenError("H5-Late requires pyarrow") from exc
    if not path.is_file():
        raise HistoryScreenError(f"{name} is missing")
    try:
        table = pq.read_table(path)
    except Exception as exc:  # pragma: no cover - Arrow exception types differ by version
        raise HistoryScreenError(f"{name} is not readable Parquet") from exc
    if tuple(table.column_names) != tuple(columns):
        raise HistoryScreenError(f"{name} column contract drifted")
    return [dict(row) for row in table.to_pylist()]


def _tracked_clean_provenance() -> dict[str, Any]:
    root = repository_root_from_source()
    paths = {
        "history_screen": Path(__file__).resolve(),
        "history_screen_cli": root / "src/ard/cli/history_screen.py",
        "rslad_signal_replay": root / "src/ard/analysis/rslad_signal_replay.py",
        "intervention_selector": root / "src/ard/analysis/intervention_selector.py",
    }
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
        raise HistoryScreenError("H5-Late analysis requires tracked source files and Git identity") from exc
    if len(sha) != 40 or dirty:
        raise HistoryScreenError("H5-Late analysis requires a tracked-clean revision")
    source_files = {name: sha256_file(path) for name, path in paths.items()}
    return {"git": {"sha": sha, "dirty": False}, "source_files": source_files, "source_sha256": _sha256(source_files)}


def _midrank_percentiles(values: Mapping[int, float]) -> dict[int, float]:
    count = len(values)
    if count < 1:
        raise HistoryScreenError("cannot rank an empty anchor-correct population")
    ordered = sorted((float(value), int(sample_id)) for sample_id, value in values.items())
    output: dict[int, float] = {}
    start = 0
    while start < count:
        end = start + 1
        while end < count and ordered[end][0] == ordered[start][0]:
            end += 1
        percentile = (start + (end - start) / 2.0) / count
        for _, sample_id in ordered[start:end]:
            output[sample_id] = percentile
        start = end
    return output


def _top_k(scores: Mapping[int, float], *, k: int) -> set[int]:
    if k < 1 or k > len(scores):
        raise HistoryScreenError("top-k does not fit the eligible population")
    return {sample_id for sample_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]}


def _validate_feature_rows(rows: Sequence[Mapping[str, Any]], *, expected_count: int) -> dict[int, dict[str, Any]]:
    if len(rows) != expected_count:
        raise HistoryScreenError("feature panel count differs from expected_count")
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        sample_id = _integer(row.get("sample_id"), name="feature sample_id")
        class_id = _integer(row.get("class_id"), name="feature class_id")
        if (
            row.get("namespace") != "train"
            or sample_id >= 50_000
            or class_id >= 10
            or sample_id in output
            or row.get("feature_epoch") != ANCHOR_EPOCH
        ):
            raise HistoryScreenError("feature panel violates exact epoch-99 stable-ID contract")
        values = {name: _finite(row.get(name), name=f"feature {name}", lower=0.0, upper=1.0) for name in FEATURE_NAMES}
        correct = _binary(row.get("student_robust_correct_epoch99"), name="epoch-99 robust correctness")
        output[sample_id] = {"class_id": class_id, "correct": correct, **values}
    return output


def _validate_online_outcomes(
    value: Mapping[str, Any], *, expected_count: int
) -> tuple[dict[str, Any], dict[int, tuple[int, int]]]:
    if set(value) != {"identity", "rows"}:
        raise HistoryScreenError("online state export schema must contain identity and rows")
    identity, rows = value.get("identity"), value.get("rows")
    if not isinstance(identity, Mapping) or not isinstance(rows, list):
        raise HistoryScreenError("online state export identity/rows are malformed")
    if identity.get("contract") != "logging_only_exact_state_anchor99_final199_v1":
        raise HistoryScreenError("online outcome export is not the exact epoch-99/199 contract")
    if (
        _integer(identity.get("expected_count"), name="state expected_count", minimum=1) != expected_count
        or _integer(identity.get("row_count"), name="state row_count", minimum=1) != expected_count
    ):
        raise HistoryScreenError("online state export count differs from expected_count")
    anchor, final = identity.get("anchor"), identity.get("final")
    if (
        not isinstance(anchor, Mapping)
        or not isinstance(final, Mapping)
        or anchor.get("epoch") != ANCHOR_EPOCH
        or final.get("epoch") != OUTCOME_END_EPOCH
    ):
        raise HistoryScreenError("online state export has a feature/outcome temporal mismatch")
    anchor_sha = _sha(anchor.get("checkpoint_sha256"), name="state anchor checkpoint SHA-256")
    _sha(final.get("checkpoint_sha256"), name="state final checkpoint SHA-256")
    if (
        not isinstance(identity.get("run_id"), str)
        or not identity.get("run_id")
        or not isinstance(identity.get("config_hash"), str)
    ):
        raise HistoryScreenError("online state export lacks run/config identity")
    output: dict[int, tuple[int, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise HistoryScreenError("online state-export row must be a mapping")
        sample_id = _integer(row.get("sample_id"), name="state sample_id")
        class_id = _integer(row.get("true_label"), name="state true_label")
        if (
            row.get("namespace") != "train"
            or sample_id >= 50_000
            or class_id >= 10
            or sample_id in output
            or row.get("anchor_epoch") != ANCHOR_EPOCH
            or row.get("final_epoch") != OUTCOME_END_EPOCH
        ):
            raise HistoryScreenError("online state export violates exact 99-to-199 stable-ID contract")
        outcome = _binary(row.get("future_online_forgetting"), name="online future forgetting")
        anchor_forgetting = _integer(row.get("anchor_forgetting_count"), name="anchor forgetting count")
        final_forgetting = _integer(row.get("final_forgetting_count"), name="final forgetting count")
        increment = _integer(row.get("subsequent_forgetting_increment"), name="subsequent forgetting increment")
        if (
            final_forgetting < anchor_forgetting
            or final_forgetting - anchor_forgetting != increment
            or outcome != int(final_forgetting > anchor_forgetting)
        ):
            raise HistoryScreenError("online state export forgetting outcome is temporally inconsistent")
        output[sample_id] = (class_id, outcome)
    if len(output) != expected_count:
        raise HistoryScreenError("online state export row count differs from expected_count")
    return {
        "run_id": identity["run_id"],
        "config_hash": identity["config_hash"],
        "anchor_checkpoint_sha256": anchor_sha,
    }, output


def _validate_lineage(*, lineage: Mapping[str, Any], feature_path: Path, expected_count: int) -> dict[str, Any]:
    if lineage.get("schema_version") != 1 or not isinstance(lineage.get("run_id"), str) or not lineage.get("run_id"):
        raise HistoryScreenError("replay lineage lacks schema/run identity")
    if _integer(lineage.get("train_expected_count"), name="lineage train_expected_count", minimum=1) != expected_count:
        raise HistoryScreenError("replay lineage train count differs from expected_count")
    full_hashes, feature_only_hash = lineage.get("output_parquet_sha256"), lineage.get("feature_panel_sha256")
    feature_only = lineage.get("kind") == "l3_checkpoint_panel_feature_source_v1"
    if feature_only:
        if full_hashes is not None or feature_only_hash != sha256_file(feature_path):
            raise HistoryScreenError("feature-only lineage must bind exactly one top-level feature-panel hash")
    elif (
        feature_only_hash is not None
        or not isinstance(full_hashes, Mapping)
        or full_hashes.get("feature_panel") != sha256_file(feature_path)
    ):
        raise HistoryScreenError("full replay lineage must bind exactly one output feature-panel hash")
    checkpoints = lineage.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise HistoryScreenError("replay lineage lacks periodic checkpoint inventory")
    anchors = [item for item in checkpoints if isinstance(item, Mapping) and item.get("epoch") == ANCHOR_EPOCH]
    if len(anchors) != 1:
        raise HistoryScreenError("replay lineage does not contain one epoch-99 checkpoint")
    _sha(anchors[0].get("sha256"), name="replay lineage epoch-99 checkpoint SHA-256")
    for field in (
        "config_hash",
        "scientific_git_sha",
        "attack_identity",
        "dataset_identity",
        "teacher",
        "analysis_provenance",
    ):
        if field not in lineage:
            raise HistoryScreenError(f"replay lineage lacks {field}")
    return dict(lineage)


def _validate_fit(
    fit: Mapping[str, Any], *, lineage: Mapping[str, Any]
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    required = {
        "schema_version",
        "contract",
        "feature_names",
        "weights",
        "means",
        "scales",
        "predictor_spec_sha256",
        "coefficients_sha256",
        "preprocessing_sha256",
        "fit_sha256",
        "seed0_input_lineage_hashes",
        "fit_domain",
        "training_outcome",
    }
    if set(fit) != required or fit.get("schema_version") != 1 or fit.get("contract") != FIT_CONTRACT:
        raise HistoryScreenError("frozen fit bundle schema drifted")
    if (
        tuple(fit.get("feature_names", ())) != FEATURE_NAMES
        or fit.get("training_outcome") != "checkpoint_panel_forgetting"
    ):
        raise HistoryScreenError("frozen fit feature names do not match H2 selector semantics")
    weights, means, scales = fit.get("weights"), fit.get("means"), fit.get("scales")
    if not all(isinstance(values, list) for values in (weights, means, scales)):
        raise HistoryScreenError("frozen fit weights/means/scales must be lists")
    try:
        parsed = tuple(
            tuple(_finite(value, name="frozen fit value") for value in values) for values in (weights, means, scales)
        )
    except TypeError as exc:  # pragma: no cover - defensive, lists checked above
        raise HistoryScreenError("frozen fit values are malformed") from exc
    parsed_weights, parsed_means, parsed_scales = parsed
    if (
        len(parsed_weights) != len(FEATURE_NAMES) + 1
        or len(parsed_means) != len(FEATURE_NAMES)
        or len(parsed_scales) != len(FEATURE_NAMES)
    ):
        raise HistoryScreenError("frozen fit dimensionality does not match the H2 selector")
    if any(scale <= 0.0 for scale in parsed_scales):
        raise HistoryScreenError("frozen fit preprocessing scale must be positive")
    for field in ("predictor_spec_sha256", "coefficients_sha256", "preprocessing_sha256", "fit_sha256"):
        _sha(fit.get(field), name=field)
    if fit["coefficients_sha256"] != _sha256(list(parsed_weights)):
        raise HistoryScreenError("frozen fit coefficient hash does not reproduce")
    if fit["preprocessing_sha256"] != _sha256({"means": list(parsed_means), "scales": list(parsed_scales)}):
        raise HistoryScreenError("frozen fit preprocessing hash does not reproduce")
    fit_payload = {
        "feature_names": list(FEATURE_NAMES),
        "weights": list(parsed_weights),
        "means": list(parsed_means),
        "scales": list(parsed_scales),
    }
    if fit["fit_sha256"] != _sha256(fit_payload):
        raise HistoryScreenError("frozen fit hash does not reproduce")
    seed0 = fit.get("seed0_input_lineage_hashes")
    domain = fit.get("fit_domain")
    if not isinstance(seed0, Mapping) or not seed0 or not isinstance(domain, Mapping):
        raise HistoryScreenError("frozen fit lacks seed-0 lineage or domain identity")
    for name, digest in seed0.items():
        if not isinstance(name, str):
            raise HistoryScreenError("seed-0 lineage key must be a string")
        _sha(digest, name=f"seed-0 {name}")
    for field in ("attack_identity", "dataset_identity", "teacher"):
        if domain.get(field) != lineage.get(field):
            raise HistoryScreenError(f"frozen fit {field} does not match run replay lineage")
    return parsed_weights, parsed_means, parsed_scales


def _summary(ids: set[int], outcomes: Mapping[int, int]) -> dict[str, Any]:
    positives = sum(outcomes[sample_id] for sample_id in ids)
    return {"count": len(ids), "outcome_count": positives, "outcome_rate": positives / len(ids) if ids else None}


def _score_metrics(*, scores: Mapping[int, float], outcomes: Mapping[int, int], k: int) -> dict[str, float]:
    ordered_ids = sorted(scores)
    try:
        metrics = binary_metrics(
            [outcomes[sample_id] for sample_id in ordered_ids], [scores[sample_id] for sample_id in ordered_ids]
        )
    except SignalAuditError as exc:
        raise HistoryScreenError("H5-Late score metrics require both outcome classes") from exc
    selected = _top_k(scores, k=k)
    positives = sum(outcomes[sample_id] for sample_id in selected)
    precision = positives / k
    return {
        "auroc": float(metrics["auroc"]),
        "auprc": float(metrics["auprc"]),
        "precision_at_k": precision,
        "recall_at_k": positives / sum(outcomes.values()),
        "lift_at_k": precision / float(metrics["prevalence"]),
        "positive_prevalence": float(metrics["prevalence"]),
    }


def analyze_history_screen(
    *,
    feature_panel: Path,
    online_state_export: Path,
    replay_lineage: Path,
    frozen_fit: Path,
    expected_count: int,
    analysis_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare frozen and run-adaptive H5 scores on one exact replay panel."""
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise HistoryScreenError("expected_count must be a positive integer")
    feature_rows = _validate_feature_rows(
        _read_parquet(feature_panel, columns=FEATURE_COLUMNS, name="feature panel"), expected_count=expected_count
    )
    state_identity, outcome_rows = _validate_online_outcomes(
        _read_json(online_state_export, name="online state export"), expected_count=expected_count
    )
    if set(feature_rows) != set(outcome_rows) or any(
        feature_rows[key]["class_id"] != outcome_rows[key][0] for key in feature_rows
    ):
        raise HistoryScreenError("replay features and online outcomes do not have an exact stable-ID/class join")
    lineage = _validate_lineage(
        lineage=_read_json(replay_lineage, name="replay lineage"),
        feature_path=feature_panel,
        expected_count=expected_count,
    )
    if lineage["run_id"] != state_identity["run_id"] or lineage["config_hash"] != state_identity["config_hash"]:
        raise HistoryScreenError("replay lineage and online state export do not share one run identity")
    epoch99_sha = next(item["sha256"] for item in lineage["checkpoints"] if item.get("epoch") == ANCHOR_EPOCH)
    if epoch99_sha != state_identity["anchor_checkpoint_sha256"]:
        raise HistoryScreenError("replay feature and online outcome exports do not share the epoch-99 checkpoint")
    weights, means, scales = _validate_fit(_read_json(frozen_fit, name="frozen fit"), lineage=lineage)
    eligible = {sample_id: row for sample_id, row in feature_rows.items() if row["correct"] == 1}
    if len(eligible) < 2:
        raise HistoryScreenError("anchor-correct population is too small")
    outcomes = {sample_id: outcome_rows[sample_id][1] for sample_id in eligible}
    if not any(outcomes.values()) or all(outcomes.values()):
        raise HistoryScreenError("anchor-correct population must contain both outcome classes")
    k = math.floor(Q * len(eligible))
    if k < 1:
        raise HistoryScreenError("q=0.10 yields an empty risk set")
    feature_matrix = [[float(row[name]) for name in FEATURE_NAMES] for _, row in sorted(eligible.items())]
    try:
        fixed_values = _predict_logistic((weights, means, scales), feature_matrix)
    except SignalAuditError as exc:
        raise HistoryScreenError("frozen H2 predictor cannot score the replay feature panel") from exc
    fixed_scores = {
        sample_id: score for (sample_id, _), score in zip(sorted(eligible.items()), fixed_values, strict=True)
    }
    frequency_rank = _midrank_percentiles(
        {sample_id: 1.0 - float(row[FEATURE_NAMES[0]]) for sample_id, row in eligible.items()}
    )
    margin_rank = _midrank_percentiles({sample_id: float(row[FEATURE_NAMES[1]]) for sample_id, row in eligible.items()})
    adaptive_scores = {sample_id: (frequency_rank[sample_id] + margin_rank[sample_id]) / 2.0 for sample_id in eligible}
    fixed_ids, adaptive_ids = _top_k(fixed_scores, k=k), _top_k(adaptive_scores, k=k)
    common, fixed_only, adaptive_only = fixed_ids & adaptive_ids, fixed_ids - adaptive_ids, adaptive_ids - fixed_ids
    provenance = dict(_tracked_clean_provenance() if analysis_provenance is None else analysis_provenance)
    fixed_metrics, adaptive_metrics = (
        _score_metrics(scores=fixed_scores, outcomes=outcomes, k=k),
        _score_metrics(scores=adaptive_scores, outcomes=outcomes, k=k),
    )
    return {
        "schema_version": 1,
        "contract": REPORT_CONTRACT,
        "diagnostic_only": True,
        "q": Q,
        "analysis_provenance": provenance,
        "input_identity": {
            "run_id": lineage["run_id"],
            "config_hash": lineage["config_hash"],
            "feature_panel_sha256": sha256_file(feature_panel),
            "online_state_export_sha256": sha256_file(online_state_export),
            "lineage_sha256": sha256_file(replay_lineage),
            "frozen_fit_sha256": sha256_file(frozen_fit),
        },
        "population": {
            "all_rows": expected_count,
            "anchor_correct_rows": len(eligible),
            "anchor_wrong_excluded_rows": expected_count - len(eligible),
            "k": k,
        },
        "fixed_score": {
            "training_outcome": "checkpoint_panel_forgetting",
            "metrics": fixed_metrics,
            "selected_ids_sha256": _hash_ids(sorted(fixed_ids)),
        },
        "adaptive_score": {
            "definition": "equal_weight_percentile_midrank(1-frequency,historical_margin_risk)",
            "metrics": adaptive_metrics,
            "selected_ids_sha256": _hash_ids(sorted(adaptive_ids)),
        },
        "evaluation_outcome": "online_future_forgetting",
        "deltas_adaptive_minus_fixed": {
            name: adaptive_metrics[name] - fixed_metrics[name]
            for name in ("auroc", "auprc", "precision_at_k", "recall_at_k", "lift_at_k")
        },
        "overlap": {
            "diagnostic_only": True,
            "jaccard": len(common) / len(fixed_ids | adaptive_ids),
            "top_q_overlap": len(common) / k,
            "groups": {
                "common": _summary(common, outcomes),
                "fixed_only": _summary(fixed_only, outcomes),
                "adaptive_only": _summary(adaptive_only, outcomes),
                "neither": _summary(set(eligible) - fixed_ids - adaptive_ids, outcomes),
            },
        },
    }
