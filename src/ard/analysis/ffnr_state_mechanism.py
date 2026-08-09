"""Fail-closed, read-only FF/NR Student--Teacher margin mechanism analysis.

The analysis consumes only the existing Chen selection-attack replay and the
separate online-state anchors.  It cannot launch an attack, train a model, or
choose a routing threshold.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ard.analysis.ffnr_strong_point import (
    StrongPointError,
    _online_panel,
    _strong_lineage,
)
from ard.analysis.ffnr_strong_replay import EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256
from ard.analysis.rslad_signal_replay import repository_root_from_source
from ard.analysis.signal_audit import binary_metrics, canonical_json, sha256_file


class FFNRStateMechanismError(StrongPointError):
    """Raised when a read-only state/mechanism input drifts from its contract."""


CONTRACT = "ffnr_state_mechanism_v1"
ANCHORS = (39, 59, 79)
TERMINAL_EPOCHS = (189, 194, 199)
ENDPOINTS = {"majority": 2, "all": 3}
QUANTILES = (0.10, 0.20, 0.25, 1.0 / 3.0)

_FEATURE_COLUMNS = {
    "namespace",
    "sample_id",
    "class_id",
    "epoch",
    "student_robust_correct",
    "student_adversarial_probability_margin",
    "student_clean_probability_margin",
    "student_clean_correct",
    "student_clean_to_adversarial_probability_margin_delta",
    "teacher_clean_probabilities",
    "teacher_adversarial_probabilities",
}
_OUTCOME_COLUMNS = {"namespace", "sample_id", "class_id", "epoch", "student_robust_correct"}


def _fit_logistic_fast(features: Sequence[Sequence[float]], targets: Sequence[int]) -> tuple[Any, Any, Any]:
    """Vectorized fixed-iteration logistic fit for the cross-seed point table."""
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency contract
        raise FFNRStateMechanismError("numpy is required for cross-seed mechanism models") from exc
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or len(np.unique(y)) != 2:
        raise FFNRStateMechanismError("cross-seed fit requires a two-class matrix")
    means = x.mean(axis=0)
    scales = np.maximum(x.std(axis=0), 1e-12)
    z = (x - means) / scales
    design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(120):
        logits = np.clip(design @ weights, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - y) / len(y)
        gradient[1:] += 0.001 * weights[1:]
        weights -= 0.15 * gradient
    return weights, means, scales


def _predict_logistic_fast(fit: tuple[Any, Any, Any], features: Sequence[Sequence[float]]) -> list[float]:
    import numpy as np

    weights, means, scales = fit
    x = (np.asarray(features, dtype=np.float64) - means) / scales
    logits = np.clip(weights[0] + x @ weights[1:], -35.0, 35.0)
    return (1.0 / (1.0 + np.exp(-logits))).tolist()


def _read_compact_observations(
    path: Path, *, epochs: Sequence[int], expected_count: int, feature: bool
) -> dict[int, dict[int, dict[str, Any]]]:
    """Read only the columns/epochs needed by this analysis.

    The generic strong-point reader materializes every row as a Python mapping
    and is correct for its point report, but two such panels exceed the small
    worker memory limit.  This reader preserves the same schema/ID checks while
    retaining only compact per-epoch records and discarding Arrow buffers.
    """
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - dependency contract
        raise FFNRStateMechanismError("pyarrow is required for state mechanism analysis") from exc
    columns = sorted(_FEATURE_COLUMNS if feature else _OUTCOME_COLUMNS)
    try:
        table = pq.read_table(path, columns=columns)
    except Exception as exc:
        raise FFNRStateMechanismError(f"state mechanism Parquet is unreadable: {path}") from exc
    if set(table.column_names) != set(columns):
        raise FFNRStateMechanismError("state mechanism Parquet schema drifted")
    values = {name: table[name].to_pylist() for name in columns}
    result: dict[int, dict[int, dict[str, Any]]] = {int(epoch): {} for epoch in epochs}
    for index, epoch_value in enumerate(values["epoch"]):
        if epoch_value not in result:
            continue
        if values["namespace"][index] != "train":
            raise FFNRStateMechanismError("state mechanism namespace drifted")
        sample_id = values["sample_id"][index]
        class_id = values["class_id"][index]
        if (
            not isinstance(sample_id, int)
            or isinstance(sample_id, bool)
            or not isinstance(class_id, int)
            or isinstance(class_id, bool)
        ):
            raise FFNRStateMechanismError("state mechanism stable-ID/class type drifted")
        if sample_id in result[int(epoch_value)]:
            raise FFNRStateMechanismError("state mechanism duplicate stable ID")
        row = {name: values[name][index] for name in columns if name not in {"namespace", "epoch"}}
        result[int(epoch_value)][sample_id] = row
    if any(len(result[int(epoch)]) != expected_count for epoch in epochs):
        raise FFNRStateMechanismError("state mechanism lacks exact epoch coverage")
    reference = result[int(epochs[0])]
    pairs = [{"sample_id": item, "class_id": reference[item]["class_id"]} for item in sorted(reference)]
    if hashlib.sha256(canonical_json(pairs)).hexdigest() != EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256:
        raise FFNRStateMechanismError("state mechanism stable-ID/class universe drifted")
    for epoch in epochs:
        if set(result[int(epoch)]) != set(reference) or any(
            result[int(epoch)][item]["class_id"] != reference[item]["class_id"] for item in reference
        ):
            raise FFNRStateMechanismError("state mechanism stable-ID/class universe changed across epochs")
    return result


def _finite(value: object, name: str, *, lo: float = -math.inf, hi: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FFNRStateMechanismError(f"{name} must be finite")
    value = float(value)
    if not lo <= value <= hi:
        raise FFNRStateMechanismError(f"{name} is outside its contract")
    return value


def _probability_margin(value: object, label: int, name: str) -> tuple[float, bool]:
    if not isinstance(value, list) or len(value) != 10:
        raise FFNRStateMechanismError(f"{name} probabilities must have ten entries")
    probabilities = tuple(_finite(item, name, lo=0.0, hi=1.0) for item in value)
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise FFNRStateMechanismError(f"{name} probabilities must sum to one")
    maximum = max(probabilities)
    if probabilities.count(maximum) != 1:
        raise FFNRStateMechanismError(f"{name} argmax ties are outside the contract")
    margin = probabilities[label] - max(item for index, item in enumerate(probabilities) if index != label)
    return margin, probabilities.index(maximum) == label


def _raw_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    epochs: Sequence[int],
    reference: Mapping[int, Mapping[int, Mapping[str, Any]]],
) -> dict[int, dict[int, dict[str, Any]]]:
    panel: dict[int, dict[int, dict[str, Any]]] = {epoch: {} for epoch in epochs}
    for row in rows:
        epoch, sample_id = row.get("epoch"), row.get("sample_id")
        if epoch not in panel:
            continue
        if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id in panel[epoch]:
            raise FFNRStateMechanismError("replay epoch/sample-ID schema drifted")
        if sample_id not in reference[epoch] or row.get("class_id") != reference[epoch][sample_id]["class_id"]:
            raise FFNRStateMechanismError("replay stable-ID/class join drifted")
        panel[epoch][sample_id] = row
    if any(set(panel[epoch]) != set(reference[epoch]) for epoch in panel):
        raise FFNRStateMechanismError("replay raw panel coverage drifted")
    return panel


def _reference_panel(
    rows: Sequence[Mapping[str, Any]], *, epochs: Sequence[int], expected_count: int, expected_universe_sha256: str
) -> dict[int, dict[int, dict[str, Any]]]:
    """Validate schema/ID/class coverage without retaining replay primitives twice."""
    reference: dict[int, dict[int, dict[str, Any]]] = {epoch: {} for epoch in epochs}
    for row in rows:
        epoch, sample_id, class_id = row.get("epoch"), row.get("sample_id"), row.get("class_id")
        if epoch not in reference:
            continue
        if (
            row.get("namespace") != "train"
            or isinstance(sample_id, bool)
            or not isinstance(sample_id, int)
            or isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or not 0 <= class_id < 10
            or sample_id in reference[epoch]
        ):
            raise FFNRStateMechanismError("replay epoch/sample-ID/class schema drifted")
        reference[epoch][sample_id] = {"class_id": class_id}
    if any(len(reference[epoch]) != expected_count for epoch in reference):
        raise FFNRStateMechanismError("replay lacks exact stable-ID coverage")
    first = reference[epochs[0]]
    pairs = [{"sample_id": item, "class_id": first[item]["class_id"]} for item in sorted(first)]
    if hashlib.sha256(canonical_json(pairs)).hexdigest() != expected_universe_sha256:
        raise FFNRStateMechanismError("replay stable-ID/class universe hash drifted")
    if any(
        set(values) != set(first) or any(values[item]["class_id"] != first[item]["class_id"] for item in first)
        for values in reference.values()
    ):
        raise FFNRStateMechanismError("replay stable-ID/class universe changed across epochs")
    return reference


def _compact_outcome(
    rows: Sequence[Mapping[str, Any]],
    *,
    epochs: Sequence[int],
    reference: Mapping[int, Mapping[int, Mapping[str, Any]]],
) -> dict[int, dict[int, dict[str, Any]]]:
    result: dict[int, dict[int, dict[str, Any]]] = {epoch: {} for epoch in epochs}
    for row in rows:
        epoch, sample_id = row.get("epoch"), row.get("sample_id")
        if epoch not in result:
            continue
        if sample_id not in reference[epoch] or row.get("class_id") != reference[epoch][sample_id]["class_id"]:
            raise FFNRStateMechanismError("outcome stable-ID/class join drifted")
        correct = row.get("student_robust_correct")
        if not isinstance(correct, bool) or sample_id in result[epoch]:
            raise FFNRStateMechanismError("outcome robust correctness schema drifted")
        result[epoch][sample_id] = {"class_id": row["class_id"], "student_robust_correct": correct}
    if any(set(result[epoch]) != set(reference[epoch]) for epoch in result):
        raise FFNRStateMechanismError("outcome compact panel coverage drifted")
    return result


def _margin_rows(raw: Mapping[int, Mapping[int, Mapping[str, Any]]], *, anchor: int) -> dict[int, dict[str, Any]]:
    if anchor not in raw:
        raise FFNRStateMechanismError("requested anchor is absent from replay")
    result: dict[int, dict[str, Any]] = {}
    for sample_id, row in raw[anchor].items():
        label = row.get("class_id")
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 10:
            raise FFNRStateMechanismError("replay class ID is invalid")
        ms_clean = _finite(row.get("student_clean_probability_margin"), "mS_clean", lo=-1, hi=1)
        ms_adv = _finite(row.get("student_adversarial_probability_margin"), "mS_adv", lo=-1, hi=1)
        if not isinstance(row.get("student_clean_correct"), bool) or not isinstance(
            row.get("student_robust_correct"), bool
        ):
            raise FFNRStateMechanismError("Student correctness flags must be boolean")
        # A zero probability margin has no unique true-label side and therefore
        # is rejected instead of silently choosing a tie convention.
        if ms_clean == 0.0 or ms_adv == 0.0:
            raise FFNRStateMechanismError("Student zero margin is outside the sign-consistency contract")
        if row["student_clean_correct"] != (ms_clean > 0.0) or row["student_robust_correct"] != (ms_adv > 0.0):
            raise FFNRStateMechanismError("Student correctness and probability-margin sign disagree")
        replay_delta = _finite(
            row.get("student_clean_to_adversarial_probability_margin_delta"), "Student replay delta", lo=-2, hi=2
        )
        if not math.isclose(replay_delta, ms_adv - ms_clean, rel_tol=0.0, abs_tol=1e-6):
            raise FFNRStateMechanismError("Student clean-to-adversarial margin algebra drifted")
        mt_clean, teacher_clean_correct = _probability_margin(
            row.get("teacher_clean_probabilities"), label, "teacher clean"
        )
        mt_adv, teacher_adv_correct = _probability_margin(
            row.get("teacher_adversarial_probabilities"), label, "teacher adv"
        )
        if mt_clean == 0.0 or mt_adv == 0.0:
            raise FFNRStateMechanismError("Teacher zero margin is outside the sign-consistency contract")
        if teacher_clean_correct != (mt_clean > 0.0) or teacher_adv_correct != (mt_adv > 0.0):
            raise FFNRStateMechanismError("Teacher correctness and probability-margin sign disagree")
        result[sample_id] = {
            "class_id": label,
            "student_clean_correct": row["student_clean_correct"],
            "student_robust_correct": row["student_robust_correct"],
            "teacher_clean_correct": teacher_clean_correct,
            "teacher_adv_correct": teacher_adv_correct,
            "mS_clean": ms_clean,
            "mS_adv": ms_adv,
            "mT_clean": mt_clean,
            "mT_adv": mt_adv,
            "DeltaS": ms_clean - ms_adv,
            "DeltaT": mt_clean - mt_adv,
        }
    return result


def _endpoint(raw: Mapping[int, Mapping[int, Mapping[str, Any]]], endpoint: str) -> dict[int, int]:
    if endpoint not in ENDPOINTS or not set(TERMINAL_EPOCHS).issubset(raw):
        raise FFNRStateMechanismError("endpoint requires frozen CE-PGD20 terminal epochs")
    ids = set(raw[TERMINAL_EPOCHS[0]])
    if any(set(raw[epoch]) != ids for epoch in TERMINAL_EPOCHS):
        raise FFNRStateMechanismError("terminal endpoint stable-ID universe drifted")
    values: dict[int, int] = {}
    for sample_id in ids:
        flags = []
        for epoch in TERMINAL_EPOCHS:
            value = raw[epoch][sample_id].get("student_robust_correct")
            if not isinstance(value, bool):
                raise FFNRStateMechanismError("terminal robust correctness must be boolean")
            flags.append(value)
        values[sample_id] = int(sum(not item for item in flags) >= ENDPOINTS[endpoint])
    return values


def _quantile_bins(values: Mapping[int, float], bins: int) -> dict[int, int]:
    if bins < 2 or not values:
        raise FFNRStateMechanismError("quantile bins require a non-empty population")
    ordered = sorted(values, key=lambda item: (values[item], item))
    return {sample_id: min(bins - 1, index * bins // len(ordered)) for index, sample_id in enumerate(ordered)}


def _rate_summary(ids: Sequence[int], target: Mapping[int, int]) -> dict[str, float | int | None]:
    count = len(ids)
    positives = sum(target[item] for item in ids)
    return {
        "count": count,
        "future_failure_count": positives,
        "future_failure_rate": positives / count if count else None,
    }


def _risk_curves(rows: Mapping[int, Mapping[str, Any]], target: Mapping[int, int]) -> list[dict[str, Any]]:
    fields = {
        "mS_clean_risk": {item: -float(row["mS_clean"]) for item, row in rows.items()},
        "mS_adv_risk": {item: -float(row["mS_adv"]) for item, row in rows.items()},
        "mT_clean_risk": {item: -float(row["mT_clean"]) for item, row in rows.items()},
        "mT_adv_risk": {item: -float(row["mT_adv"]) for item, row in rows.items()},
        "DeltaS": {item: float(row["DeltaS"]) for item, row in rows.items()},
        "DeltaT": {item: float(row["DeltaT"]) for item, row in rows.items()},
    }
    result: list[dict[str, Any]] = []
    for name, values in fields.items():
        assignment = _quantile_bins(values, 10)
        for bin_index in range(10):
            members = [item for item in sorted(values) if assignment[item] == bin_index]
            result.append(
                {
                    "measure": name,
                    "bin": bin_index,
                    **_rate_summary(members, target),
                    "mean": sum(values[item] for item in members) / len(members) if members else None,
                }
            )
    return result


def _surface(rows: Mapping[int, Mapping[str, Any]], target: Mapping[int, int]) -> list[dict[str, Any]]:
    student = _quantile_bins({item: -float(row["mS_adv"]) for item, row in rows.items()}, 5)
    teacher = _quantile_bins({item: -float(row["mT_adv"]) for item, row in rows.items()}, 5)
    result = []
    for student_bin in range(5):
        for teacher_bin in range(5):
            members = [item for item in sorted(rows) if student[item] == student_bin and teacher[item] == teacher_bin]
            result.append(
                {"student_risk_bin": student_bin, "teacher_risk_bin": teacher_bin, **_rate_summary(members, target)}
            )
    return result


def _state_summaries(
    rows: Mapping[int, Mapping[str, Any]], target: Mapping[int, int], delta_threshold: float
) -> dict[str, Any]:
    two = {item: "robust_correct" if row["mS_adv"] > 0 else "robust_wrong" for item, row in rows.items()}
    three = {
        item: (
            "clean_wrong"
            if row["mS_clean"] < 0
            else "clean_correct_high_response"
            if row["DeltaS"] >= delta_threshold
            else "clean_correct_low_response"
        )
        for item, row in rows.items()
    }

    def summarize(values: Mapping[int, str]) -> dict[str, dict[str, float | int | None]]:
        return {
            state: _rate_summary([item for item in values if values[item] == state], target)
            for state in sorted(set(values.values()))
        }

    return {
        "two_state": summarize(two),
        "three_state": summarize(three),
        "three_state_deltaS_threshold": delta_threshold,
    }


def _threshold_tables(rows: Mapping[int, Mapping[str, Any]], target: Mapping[int, int]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for measure in ("DeltaS", "DeltaT"):
        values = {item: float(row[measure]) for item, row in rows.items()}
        ordered = sorted(values, key=lambda item: (values[item], item))
        for fraction in QUANTILES:
            nominal = max(1, math.ceil(fraction * len(ordered)))
            chosen = set(ordered[-nominal:])
            threshold = values[ordered[-nominal]]
            result.append(
                {
                    "measure": measure,
                    "top_fraction": fraction,
                    "threshold": threshold,
                    "high_response": _rate_summary(sorted(chosen), target),
                    "other": _rate_summary([item for item in ordered if item not in chosen], target),
                }
            )
    return result


def _state_threshold_candidates(
    rows: Mapping[int, Mapping[str, Any]], target: Mapping[int, int]
) -> list[dict[str, Any]]:
    """Report preregistered quantile thresholds without selecting one."""
    result: list[dict[str, Any]] = []
    for measure in ("mS_adv", "mT_adv"):
        positive = sorted(float(row[measure]) for row in rows.values() if float(row[measure]) > 0)
        for fraction in QUANTILES:
            if not positive:
                continue
            index = min(len(positive) - 1, max(0, math.ceil(fraction * len(positive)) - 1))
            tau = positive[index]
            states = {
                "safe_correct": [item for item, row in rows.items() if float(row[measure]) > tau],
                "fragile_correct": [item for item, row in rows.items() if 0 < float(row[measure]) <= tau],
                "wrong": [item for item, row in rows.items() if float(row[measure]) <= 0],
            }
            result.append(
                {
                    "measure": measure,
                    "fragile_positive_fraction": fraction,
                    "tau": tau,
                    "states": {name: _rate_summary(ids, target) for name, ids in states.items()},
                }
            )
    return result


def _candidate_thresholds(rows: Mapping[int, Mapping[str, Any]], measure: str) -> list[tuple[float, float]]:
    positive = sorted(float(row[measure]) for row in rows.values() if float(row[measure]) > 0)
    if not positive:
        return []
    result = []
    for fraction in QUANTILES:
        index = min(len(positive) - 1, max(0, math.ceil(fraction * len(positive)) - 1))
        result.append((fraction, positive[index]))
    return result


def _state_grid(
    rows: Mapping[int, Mapping[str, Any]], target: Mapping[int, int], tau_s: float, tau_t: float
) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str], list[int]] = {}

    def classify(value: float, tau: float) -> str:
        return "wrong" if value <= 0 else "fragile_correct" if value <= tau else "safe_correct"

    for item, row in rows.items():
        cells.setdefault((classify(float(row["mS_adv"]), tau_s), classify(float(row["mT_adv"]), tau_t)), []).append(
            item
        )
    return [
        {"student_state": student, "teacher_state": teacher, **_rate_summary(cells.get((student, teacher), []), target)}
        for student in ("safe_correct", "fragile_correct", "wrong")
        for teacher in ("safe_correct", "fragile_correct", "wrong")
    ]


def _teacher_decomposition(rows: Mapping[int, Mapping[str, Any]], target: Mapping[int, int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for clean in (False, True):
        for adv in (False, True):
            members = [
                item
                for item, row in rows.items()
                if row["teacher_clean_correct"] == clean and row["teacher_adv_correct"] == adv
            ]
            result[f"teacher_clean_{int(clean)}_adv_{int(adv)}"] = {
                **_rate_summary(members, target),
                "student_response": {
                    "mean_DeltaS": sum(float(rows[item]["DeltaS"]) for item in members) / len(members)
                    if members
                    else None,
                    "mean_mS_adv": sum(float(rows[item]["mS_adv"]) for item in members) / len(members)
                    if members
                    else None,
                },
                "teacher_response": {
                    "mean_DeltaT": sum(float(rows[item]["DeltaT"]) for item in members) / len(members)
                    if members
                    else None,
                    "mean_mT_adv": sum(float(rows[item]["mT_adv"]) for item in members) / len(members)
                    if members
                    else None,
                },
            }
    return result


def _model_rows(
    fit_label: str,
    eval_label: str,
    fit: Mapping[int, Mapping[str, Any]],
    evaluate: Mapping[int, Mapping[str, Any]],
    fit_target: Mapping[int, int],
    eval_target: Mapping[int, int],
) -> list[dict[str, Any]]:
    columns = {
        # M0 is the strong current Student state.  M0_history is retained as
        # the deployable online-history reference, without using future labels.
        "M0": ("mS_adv",),
        "M0_history": ("online_margin_risk",),
        "M1_student_plus_teacher_clean": ("mS_adv", "mT_clean"),
        "M2_student_plus_teacher_response": ("mS_adv", "DeltaT"),
        "M3_student_plus_both_teacher_parts": ("mS_adv", "mT_clean", "DeltaT"),
        "M4_student_plus_teacher_adv": ("mS_adv", "mT_adv"),
    }
    if set(fit) != set(fit_target) or set(evaluate) != set(eval_target):
        raise FFNRStateMechanismError("cross-seed model IDs and targets differ")
    output: list[dict[str, Any]] = []
    for name, fields in columns.items():
        if name == "M0":
            probability = sum(fit_target.values()) / len(fit_target)
            scores = [probability] * len(evaluate)
        else:
            train_ids, eval_ids = sorted(fit), sorted(evaluate)
            fitted = _fit_logistic_fast(
                [[float(fit[item][field]) for field in fields] for item in train_ids],
                [fit_target[item] for item in train_ids],
            )
            scores = _predict_logistic_fast(
                fitted, [[float(evaluate[item][field]) for field in fields] for item in eval_ids]
            )
        eval_ids = sorted(evaluate)
        metrics = binary_metrics([eval_target[item] for item in eval_ids], scores)
        brier = sum((score - eval_target[item]) ** 2 for item, score in zip(eval_ids, scores, strict=True)) / len(
            eval_ids
        )
        output.append(
            {
                "fit_run": fit_label,
                "eval_run": eval_label,
                "model": name,
                "fields": list(fields),
                **metrics,
                "brier": brier,
            }
        )
    baseline = {row["model"]: row for row in output}
    for row in output:
        reference = "M3" if row["model"] == "M4" else "M0"
        row["delta_auroc_vs_" + reference] = row["auroc"] - baseline[reference]["auroc"]
        row["delta_log_loss_vs_" + reference] = row["log_loss"] - baseline[reference]["log_loss"]
    return output


def _tracked_clean_provenance() -> dict[str, Any]:
    root = repository_root_from_source()
    paths = {"analysis": Path(__file__).resolve(), "cli": root / "src/ard/cli/ffnr_state_mechanism.py"}
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
        raise FFNRStateMechanismError("state mechanism analysis requires tracked source and Git identity") from exc
    if len(sha) != 40 or dirty:
        raise FFNRStateMechanismError("state mechanism analysis requires a tracked-clean revision")
    source_files = {name: sha256_file(path) for name, path in paths.items()}
    return {
        "git": {"sha": sha, "dirty": False},
        "source_files": source_files,
        "source_sha256": hashlib.sha256(canonical_json(source_files)).hexdigest(),
    }


def analyze_run(
    *,
    label: str,
    feature_observations: Path,
    feature_lineage: Path,
    outcome_observations: Path,
    outcome_lineage: Path,
    online_states: Path,
    online_lineage: Path,
    expected_count: int = 45000,
    expected_universe_sha256: str = EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256,
) -> dict[str, Any]:
    if label not in {"L2", "L4"}:
        raise FFNRStateMechanismError("analysis is frozen to Chen L2/L4")
    feature_meta = _strong_lineage(
        path=feature_lineage,
        observations=feature_observations,
        role="feature",
        expected_count=expected_count,
        expected_universe_sha256=expected_universe_sha256,
    )
    outcome_meta = _strong_lineage(
        path=outcome_lineage,
        observations=outcome_observations,
        role="outcome",
        expected_count=expected_count,
        expected_universe_sha256=expected_universe_sha256,
    )
    if tuple(feature_meta.get("requested_epochs", ())) != ANCHORS or not set(TERMINAL_EPOCHS).issubset(
        outcome_meta.get("requested_epochs", ())
    ):
        raise FFNRStateMechanismError("replay anchor or frozen terminal-epoch coverage drifted")
    for key in (
        "run_id",
        "teacher",
        "dataset_identity",
        "attack_identity",
        "saved_resolved_config_mapping_sha256",
        "manifest_sha256",
    ):
        if feature_meta.get(key) != outcome_meta.get(key):
            raise FFNRStateMechanismError("feature/outcome replay lineage identity drifted")
    online, online_meta = _online_panel(online_states, online_lineage, expected_count)
    for key in ("run_id", "config_hash", "teacher", "dataset_identity"):
        replay_value = (
            feature_meta.get("saved_resolved_config_mapping_sha256") if key == "config_hash" else feature_meta.get(key)
        )
        if replay_value != online_meta.get(key):
            raise FFNRStateMechanismError("replay/online lineage identity drifted")
    # Use a column-pruned reader: feature rows retain only the two teacher
    # probability vectors needed to derive margins, while terminal outcomes
    # retain only correctness.  This avoids the multi-gigabyte Python row
    # materialization used by the generic replay point reader.
    feature = _read_compact_observations(
        feature_observations,
        epochs=ANCHORS,
        expected_count=expected_count,
        feature=True,
    )
    outcome = _read_compact_observations(
        outcome_observations,
        epochs=outcome_meta["requested_epochs"],
        expected_count=expected_count,
        feature=False,
    )
    ids = set(feature[ANCHORS[0]])
    if (
        any(set(feature[anchor]) != ids or set(online[anchor]) != ids for anchor in ANCHORS)
        or set(outcome[TERMINAL_EPOCHS[0]]) != ids
    ):
        raise FFNRStateMechanismError("feature/outcome/online stable-ID coverage drifted")
    if any(
        feature[39][item]["class_id"] != online[39][item]["class_id"]
        or feature[39][item]["class_id"] != outcome[TERMINAL_EPOCHS[0]][item]["class_id"]
        for item in ids
    ):
        raise FFNRStateMechanismError("feature/outcome/online stable-ID class join drifted")
    endpoints = {name: _endpoint(outcome, name) for name in ENDPOINTS}
    analyses: dict[str, Any] = {}
    cross_seed: dict[str, Any] = {}
    for anchor in ANCHORS:
        rows = _margin_rows(feature, anchor=anchor)
        eligible = {item: rows[item] for item in ids if online[anchor][item]["current_correct"]}
        if not eligible:
            raise FFNRStateMechanismError("online-current-correct FF cohort is empty")
        analyses[str(anchor)] = {}
        compact_rows = {
            item: {
                **rows[item],
                "online_margin_risk": float(online[anchor][item]["margin_risk"]),
                "online_frequency_risk": float(online[anchor][item]["frequency_risk"]),
            }
            for item in eligible
        }
        for endpoint, target in endpoints.items():
            cohort_target = {item: target[item] for item in eligible}
            delta_values = sorted(float(row["DeltaS"]) for row in eligible.values())
            median = delta_values[(len(delta_values) - 1) // 2]
            analyses[str(anchor)][endpoint] = {
                "eligibility": {
                    "online_current_correct": len(eligible),
                    "online_current_wrong": len(ids) - len(eligible),
                },
                "risk_curves": _risk_curves(eligible, cohort_target),
                "student_teacher_surface": _surface(eligible, cohort_target),
                "threshold_candidates": _threshold_tables(eligible, cohort_target),
                "state_threshold_candidates": _state_threshold_candidates(eligible, cohort_target),
                "state_grids": [
                    {
                        "student_fraction": sf,
                        "teacher_fraction": tf,
                        "cells": _state_grid(eligible, cohort_target, st, tt),
                    }
                    for sf, st in _candidate_thresholds(eligible, "mS_adv")
                    for tf, tt in _candidate_thresholds(eligible, "mT_adv")
                ],
                "state_summaries": _state_summaries(eligible, cohort_target, median),
                "teacher_conditional_decomposition": _teacher_decomposition(eligible, cohort_target),
            }
            cross_seed.setdefault(str(anchor), {"rows": compact_rows, "targets": {}})["targets"][endpoint] = (
                cohort_target
            )
    # Only the compact scalar FF rows and targets are needed by cross-seed
    # fitting.  Release the decoded replay probability vectors before L4 loads.
    del feature, outcome, online, endpoints
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "scientific_status": "read_only_train_split_point_analysis_no_intervention_training_official_test_autoattack_or_threshold_selection",
        "label": label,
        "endpoint": {"epochs": list(TERMINAL_EPOCHS), "primary": "majority", "secondary": "all"},
        "margin_definitions": {
            "mS_clean": "pS(y|x)-max_{c!=y}pS(c|x)",
            "mS_adv": "pS(y|x_adv)-max_{c!=y}pS(c|x_adv)",
            "mT_clean": "pT(y|x)-max_{c!=y}pT(c|x)",
            "mT_adv": "pT(y|x_adv)-max_{c!=y}pT(c|x_adv)",
            "DeltaS": "mS_clean-mS_adv",
            "DeltaT": "mT_clean-mT_adv",
        },
        "input_identity": {
            "run_id": feature_meta["run_id"],
            "config_hash": feature_meta["saved_resolved_config_mapping_sha256"],
            "teacher": feature_meta["teacher"],
            "dataset_identity": feature_meta["dataset_identity"],
            "attack_identity": feature_meta["attack_identity"],
            "stable_id_class_universe_sha256": expected_universe_sha256,
            "input_sha256": {
                "feature_observations": sha256_file(feature_observations),
                "feature_lineage": sha256_file(feature_lineage),
                "outcome_observations": sha256_file(outcome_observations),
                "outcome_lineage": sha256_file(outcome_lineage),
                "online_states": sha256_file(online_states),
                "online_lineage": sha256_file(online_lineage),
            },
        },
        "anchors": analyses,
        "_cross_seed": cross_seed,
    }


def write_outputs(
    *,
    output_dir: Path,
    reports: Mapping[str, Mapping[str, Any]],
    cross_seed_models: Sequence[Mapping[str, Any]],
    config_path: Path,
) -> dict[str, Path]:
    paths = {
        "report": output_dir / "ffnr-state-mechanism-report.json",
        "model_points": output_dir / "ffnr-state-mechanism-model-points.json",
    }
    if output_dir.exists() or any(path.exists() for path in paths.values()) or set(reports) != {"L2", "L4"}:
        raise FFNRStateMechanismError("refusing to overwrite or write an incomplete state-mechanism report")
    output_dir.mkdir(parents=True, exist_ok=False)
    points = list(cross_seed_models)
    paths["model_points"].write_text(json.dumps(points, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "contract": CONTRACT,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "analysis_provenance": _tracked_clean_provenance(),
        "reports": {
            label: {key: value for key, value in report.items() if not key.startswith("_")}
            for label, report in reports.items()
        },
        "cross_seed_models": points,
        "model_points_sha256": sha256_file(paths["model_points"]),
    }
    paths["report"].write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return paths


def cross_seed_models(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Fit M0--M4 on one Chen seed's FF cohort and evaluate on the other."""
    if set(reports) != {"L2", "L4"}:
        raise FFNRStateMechanismError("cross-seed models require exactly L2/L4 inputs")
    result: list[dict[str, Any]] = []
    for fit_label, eval_label in (("L2", "L4"), ("L4", "L2")):
        for anchor in ANCHORS:
            fit_payload = reports[fit_label].get("_cross_seed", {}).get(str(anchor))
            eval_payload = reports[eval_label].get("_cross_seed", {}).get(str(anchor))
            if not isinstance(fit_payload, Mapping) or not isinstance(eval_payload, Mapping):
                raise FFNRStateMechanismError("cross-seed compact anchor payload is missing")
            fit_rows, eval_rows = fit_payload.get("rows"), eval_payload.get("rows")
            fit_targets, eval_targets = fit_payload.get("targets"), eval_payload.get("targets")
            if (
                not isinstance(fit_rows, Mapping)
                or not isinstance(eval_rows, Mapping)
                or not isinstance(fit_targets, Mapping)
                or not isinstance(eval_targets, Mapping)
            ):
                raise FFNRStateMechanismError("cross-seed compact payload schema drifted")
            for endpoint in ENDPOINTS:
                fit_target, eval_target = fit_targets.get(endpoint), eval_targets.get(endpoint)
                if not isinstance(fit_target, Mapping) or not isinstance(eval_target, Mapping):
                    raise FFNRStateMechanismError("cross-seed compact target payload is missing")
                result.extend(
                    {"anchor_epoch": anchor, "endpoint": endpoint, **row}
                    for row in _model_rows(
                        fit_label,
                        eval_label,
                        fit_rows,
                        eval_rows,
                        fit_target,
                        eval_target,
                    )
                )
    return result
