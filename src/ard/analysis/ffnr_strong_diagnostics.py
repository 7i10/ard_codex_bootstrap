"""Read-only D3/D4/D5 diagnostics for CE-PGD20 FF/current-wrong panels.

This module deliberately consumes immutable replay outputs.  It never launches
an attack, changes a cohort, or selects an intervention.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import math
import subprocess
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ard.analysis.ffnr_forecasting import _metric, deterministic_midranks
from ard.analysis.ffnr_strong_point import (
    StrongPointError,
    _file_identity,
    _normalized_identity,
    _online_panel,
    _read_parquet,
    _strong_lineage,
    _strong_panel,
)
from ard.analysis.ffnr_strong_replay import EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256
from ard.analysis.rslad_signal_replay import repository_root_from_source
from ard.analysis.sample_stats import write_sample_parquet
from ard.analysis.signal_audit import canonical_json, sha256_file


class StrongDiagnosticsError(StrongPointError):
    """Raised when diagnostic inputs drift from the frozen CE-PGD20 contract."""


CONTRACT = "ffnr_strong_diagnostics_v1"
ANCHORS = (39, 59, 79)
PRIMARY_ENDPOINT = "majority"
SECONDARY_ENDPOINT = "all"
MAX_BLIND_PAIRS_PER_CLASS = 5


def _wilson(successes: int, count: int) -> dict[str, float | int | None]:
    if count < 0 or successes < 0 or successes > count:
        raise StrongDiagnosticsError("Wilson input counts are invalid")
    if not count:
        return {"count": 0, "rate": None, "lower": None, "upper": None}
    z = 1.959963984540054
    rate = successes / count
    denominator = 1 + z * z / count
    centre = (rate + z * z / (2 * count)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return {"count": count, "rate": rate, "lower": max(0.0, centre - half), "upper": min(1.0, centre + half)}


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("mean", "std", "min", "q05", "q25", "median", "q75", "q95", "max")}
    ordered = sorted(float(value) for value in values)

    def q(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    mean = sum(ordered) / len(ordered)
    return {
        "mean": mean,
        "std": math.sqrt(sum((value - mean) ** 2 for value in ordered) / len(ordered)),
        "min": ordered[0], "q05": q(.05), "q25": q(.25), "median": q(.5), "q75": q(.75), "q95": q(.95), "max": ordered[-1],
    }


def _spearman(left: Mapping[int, float], right: Mapping[int, float]) -> float | None:
    if set(left) != set(right) or len(left) < 2:
        return None
    a, b = deterministic_midranks(left), deterministic_midranks(right)
    ma, mb = sum(a.values()) / len(a), sum(b.values()) / len(b)
    denominator = math.sqrt(sum((a[key] - ma) ** 2 for key in a) * sum((b[key] - mb) ** 2 for key in b))
    return None if denominator == 0 else sum((a[key] - ma) * (b[key] - mb) for key in a) / denominator


def _teacher(row: Mapping[str, Any], *, clean: bool = False) -> dict[str, float | bool]:
    probabilities = row["teacher_clean_probabilities" if clean else "teacher_adversarial_probabilities"]
    if not isinstance(probabilities, list) or len(probabilities) != 10:
        raise StrongDiagnosticsError("teacher probability vector is invalid")
    label = int(row["class_id"])
    values = [float(value) for value in probabilities]
    if not math.isclose(sum(values), 1.0, abs_tol=1e-5) or any(value < 0 or value > 1 for value in values):
        raise StrongDiagnosticsError("teacher probability vector drifted")
    maxima = [index for index, value in enumerate(values) if value == max(values)]
    if len(maxima) != 1:
        raise StrongDiagnosticsError("teacher probability argmax tie is outside the diagnostic contract")
    prediction = maxima[0]
    wrong = max(value for index, value in enumerate(values) if index != label)
    return {
        "correct": prediction == label,
        "true_probability": values[label],
        "max_wrong_probability": wrong,
        "dominance": wrong - values[label],
        "entropy": -sum(value * math.log(value) for value in values if value > 0),
    }


def _raw_strong_panel(
    rows: Sequence[Mapping[str, Any]], *, epochs: Sequence[int], reference: Mapping[int, Mapping[int, Mapping[str, Any]]]
) -> dict[int, dict[int, dict[str, Any]]]:
    """Retain raw teacher and student primitives after the strict point-panel validation."""
    result: dict[int, dict[int, dict[str, Any]]] = {epoch: {} for epoch in epochs}
    for row in rows:
        epoch, sample_id = row.get("epoch"), row.get("sample_id")
        if epoch in result and isinstance(sample_id, int) and not isinstance(sample_id, bool):
            if sample_id in result[epoch]:
                raise StrongDiagnosticsError("raw strong replay has duplicate stable IDs")
            result[epoch][sample_id] = dict(row)
    for epoch in result:
        if set(result[epoch]) != set(reference[epoch]):
            raise StrongDiagnosticsError("raw strong replay stable-ID coverage drifted")
        if any(int(result[epoch][item]["class_id"]) != int(reference[epoch][item]["class_id"]) for item in result[epoch]):
            raise StrongDiagnosticsError("raw strong replay class IDs drifted")
    return result


def _endpoint(outcome: Mapping[int, Mapping[int, Mapping[str, Any]]], endpoint: str) -> dict[int, int]:
    epochs = tuple(sorted(outcome))[-3:]
    if len(epochs) != 3 or endpoint not in {PRIMARY_ENDPOINT, SECONDARY_ENDPOINT}:
        raise StrongDiagnosticsError("endpoint requires exactly three frozen plateau checkpoints")
    required = 2 if endpoint == PRIMARY_ENDPOINT else 3
    return {sample_id: int(sum(not bool(outcome[epoch][sample_id]["correct"]) for epoch in epochs) >= required) for sample_id in outcome[epochs[0]]}


def _snapshot_taxonomy(sequence: Sequence[bool]) -> str:
    if len(sequence) < 2:
        return "other_insufficient"
    flips = sum(left != right for left, right in zip(sequence, sequence[1:]))
    if flips >= 2:
        return "oscillating"
    suffix = 0
    for value in reversed(sequence):
        if not value:
            break
        suffix += 1
    if suffix == 1:
        return "transient_correct"
    if suffix >= 2:
        return "stable_then_forgotten"
    return "other_insufficient"


def _dense_non_recovery(sequence: Sequence[bool]) -> str:
    if not sequence or sequence[0]:
        raise StrongDiagnosticsError("dense NR taxonomy requires a strong-domain current-wrong sequence")
    if not any(sequence[1:]):
        return "persistent_wrong"
    recovery = sequence.index(True)
    return "recovered_stable" if all(sequence[recovery:]) else "recovered_relapsed"


def _merge_dense_chunks(
    chunks: Sequence[Mapping[str, Path]], *, reference: Mapping[str, Any], expected_count: int, expected_universe_sha256: str
) -> dict[int, dict[int, dict[str, Any]]]:
    merged: dict[int, dict[int, dict[str, Any]]] = {}
    for chunk in chunks:
        observations, lineage_path = chunk.get("observations"), chunk.get("lineage")
        if not isinstance(observations, Path) or not isinstance(lineage_path, Path):
            raise StrongDiagnosticsError("dense chunk paths are invalid")
        meta = _strong_lineage(path=lineage_path, observations=observations, role="feature", expected_count=expected_count, expected_universe_sha256=expected_universe_sha256)
        for key in ("run_id", "teacher", "dataset_identity", "attack_identity", "saved_resolved_config_mapping_sha256", "manifest_sha256"):
            if meta.get(key) != reference.get(key):
                raise StrongDiagnosticsError("dense chunk lineage drifted from sparse feature replay")
        panel = _strong_panel(_read_parquet(observations), epochs=meta["requested_epochs"], expected_count=expected_count, expected_universe_sha256=expected_universe_sha256)
        if set(merged) & set(panel):
            raise StrongDiagnosticsError("dense chunks overlap an epoch")
        merged.update(panel)
    return merged


def _class_stratified_folds(ids: Sequence[int], classes: Mapping[int, int], *, folds: int = 5) -> dict[int, int]:
    """Hash-sort each class then round-robin it exactly across the frozen folds."""
    if folds != 5 or not ids or len(set(ids)) != len(ids):
        raise StrongDiagnosticsError("D3 requires unique IDs and exactly five folds")
    grouped: dict[int, list[int]] = {}
    for item in ids:
        class_id = classes.get(item)
        if not isinstance(class_id, int) or not 0 <= class_id < 10:
            raise StrongDiagnosticsError("D3 class-stratified fold class is invalid")
        grouped.setdefault(class_id, []).append(item)
    assignments: dict[int, int] = {}
    for class_id, members in grouped.items():
        ordered = sorted(
            members,
            key=lambda item: hashlib.sha256(f"ffnr-d3-fold-v1:{class_id}:{item}".encode()).digest(),
        )
        for position, item in enumerate(ordered):
            assignments[item] = position % folds
    if set(assignments) != set(ids):
        raise StrongDiagnosticsError("D3 class-stratified fold coverage drifted")
    return assignments


def _fit_empirical_rank(values: Sequence[float]) -> tuple[float, ...]:
    if not values or any(not math.isfinite(value) for value in values):
        raise StrongDiagnosticsError("D3 rank transform requires finite training values")
    return tuple(sorted(float(value) for value in values))


def _apply_empirical_rank(sorted_train: Sequence[float], value: float) -> float:
    """Fold-local empirical midrank/percentile; held-out values never alter the fit."""
    if not sorted_train or not math.isfinite(value):
        raise StrongDiagnosticsError("D3 empirical rank input is invalid")
    lower, upper = bisect_left(sorted_train, value), bisect_right(sorted_train, value)
    return (lower + upper) / (2.0 * len(sorted_train))


def _fold_rank_vectors(
    *, train: Sequence[int], test: Sequence[int], columns: Sequence[str], features: Mapping[str, Mapping[int, float]]
) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]]]:
    """Fit every empirical transform from train IDs only, then transform OOF IDs."""
    transforms = [_fit_empirical_rank([float(features[column][item]) for item in train]) for column in columns]
    return (
        [
            tuple(
                _apply_empirical_rank(transform, float(features[column][item]))
                for column, transform in zip(columns, transforms, strict=True)
            )
            for item in train
        ],
        [
            tuple(
                _apply_empirical_rank(transform, float(features[column][item]))
                for column, transform in zip(columns, transforms, strict=True)
            )
            for item in test
        ],
    )


def _vectorized_logistic_predict(train_vectors: Sequence[Sequence[float]], targets: Sequence[int], test_vectors: Sequence[Sequence[float]]) -> list[float]:
    """Fixed float64 vectorization of the repository's 400-step logistic fit."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - torch environments normally provide NumPy
        raise StrongDiagnosticsError("D3 OOF requires NumPy for the bounded vectorized fit") from exc
    if not train_vectors or len(train_vectors) != len(targets) or len(set(targets)) != 2:
        raise StrongDiagnosticsError("D3 vectorized logistic fit requires both outcome classes")
    train = np.asarray(train_vectors, dtype=np.float64)
    test = np.asarray(test_vectors, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise StrongDiagnosticsError("D3 vectorized logistic shape drifted")
    means = train.mean(axis=0)
    scales = np.maximum(1e-12, np.sqrt(((train - means) ** 2).mean(axis=0)))
    normalized = (train - means) / scales
    weights = np.zeros(train.shape[1] + 1, dtype=np.float64)
    for _ in range(400):
        logits = np.clip(weights[0] + normalized @ weights[1:], -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - labels
        gradient = np.concatenate((np.asarray([error.sum()]), normalized.T @ error))
        prior = weights.copy()
        weights -= 0.15 * gradient / len(normalized)
        weights[1:] -= 0.001 * prior[1:]
    values = np.clip(weights[0] + ((test - means) / scales) @ weights[1:], -35.0, 35.0)
    return (1.0 / (1.0 + np.exp(-values))).tolist()


def _blinded_candidate_rows(
    label: str,
    *,
    taxonomy: Mapping[int, str],
    classes: Mapping[int, int],
    teacher: Mapping[int, Mapping[str, float | bool]],
) -> list[dict[str, int | str]]:
    """Make deterministic class-matched blind target/control pairs from dense taxonomy only."""
    target = {item for item, kind in taxonomy.items() if kind == "persistent_wrong" and not bool(teacher[item]["correct"])}
    controls = {item for item, kind in taxonomy.items() if kind != "persistent_wrong"}
    selected: list[dict[str, int | str]] = []
    for class_id in range(10):
        target_class = sorted(
            (item for item in target if classes[item] == class_id),
            key=lambda item: hashlib.sha256(f"ffnr-blind-v2:target:{label}:{class_id}:{item}".encode()).digest(),
        )
        control_class = sorted(
            (item for item in controls if classes[item] == class_id),
            key=lambda item: hashlib.sha256(f"ffnr-blind-v2:control:{label}:{class_id}:{item}".encode()).digest(),
        )
        pair_count = min(len(target_class), len(control_class), MAX_BLIND_PAIRS_PER_CLASS)
        for target_id, control_id in zip(target_class[:pair_count], control_class[:pair_count], strict=True):
            selected.extend(
                (
                    {"sample_id": target_id, "class_id": class_id},
                    {"sample_id": control_id, "class_id": class_id},
                )
            )
    if not selected:
        raise StrongDiagnosticsError("dense persistent-wrong teacher-wrong blind panel is empty")
    # Pair construction must not leak target/control membership through row order.
    # The public manifest contains neither role, so order it with a separate,
    # role-independent stable-ID hash after the class-matched sample is fixed.
    return sorted(
        selected,
        key=lambda row: hashlib.sha256(
            f"ffnr-blind-order-v2:{label}:{row['class_id']}:{row['sample_id']}".encode()
        ).digest(),
    )


def _oof_scores(ids: Sequence[int], labels: Mapping[int, int], features: Mapping[str, Mapping[int, float]], classes: Mapping[int, int]) -> dict[str, Any]:
    """Five-fold class-stratified OOF logistic comparison using fold-local empirical ranks."""
    specs = {"M": ("M",), "M+D": ("M", "D"), "H": ("H",), "H+D": ("H", "D"), "M+H": ("M", "H"), "M+H+D": ("M", "H", "D")}
    if not ids or len(set(ids)) != len(ids) or any(item not in labels or labels[item] not in {0, 1} for item in ids):
        raise StrongDiagnosticsError("D3 OOF labels must cover every requested eligible ID")
    subset_labels = {item: int(labels[item]) for item in ids}
    result: dict[str, Any] = {}
    assignment = _class_stratified_folds(ids, classes)
    for name, columns in specs.items():
        prediction: dict[int, float] = {}
        for fold in range(5):
            train = [item for item in ids if assignment[item] != fold]
            test = [item for item in ids if assignment[item] == fold]
            if not train or not test or len({subset_labels[item] for item in train}) < 2:
                continue
            train_vectors, test_vectors = _fold_rank_vectors(train=train, test=test, columns=columns, features=features)
            try:
                probabilities = _vectorized_logistic_predict(train_vectors, [subset_labels[item] for item in train], test_vectors)
            except Exception as exc:
                raise StrongDiagnosticsError("D3 fold-local OOF logistic fit failed") from exc
            prediction.update(dict(zip(test, probabilities, strict=True)))
        if set(prediction) != set(ids):
            raise StrongDiagnosticsError("class-stratified OOF folds lack complete coverage")
        rank = deterministic_midranks(prediction)
        metric = _metric(rank, subset_labels)
        logloss = -sum(subset_labels[item] * math.log(max(prediction[item], 1e-12)) + (1 - subset_labels[item]) * math.log(max(1 - prediction[item], 1e-12)) for item in ids) / len(ids)
        brier = sum((prediction[item] - subset_labels[item]) ** 2 for item in ids) / len(ids)
        result[name] = {"folds": 5, "auroc": metric["auroc"], "auprc": metric["auprc"], "logloss": logloss, "brier": brier}
    for base, expanded in (("M", "M+D"), ("H", "H+D"), ("M+H", "M+H+D")):
        result[expanded]["delta_vs_" + base] = {key: result[expanded][key] - result[base][key] for key in ("auroc", "auprc", "logloss", "brier") if result[expanded][key] is not None and result[base][key] is not None}
    return result


def _student_measures(row: Mapping[str, Any]) -> dict[str, float | bool]:
    """Expose frozen replay primitives without inferring an additional cohort."""
    raw_values = {
        "clean_probability_margin": row.get("student_clean_probability_margin"),
        "adversarial_probability_margin": row.get("student_adversarial_probability_margin"),
        "clean_logit_margin": row.get("student_clean_logit_margin"),
        "adversarial_logit_margin": row.get("student_adversarial_logit_margin"),
        "adversarial_ce": row.get("student_adversarial_ce"),
        "probability_margin_delta": row.get("student_clean_to_adversarial_probability_margin_delta"),
        "logit_margin_delta": row.get("student_clean_to_adversarial_logit_margin_delta"),
    }
    values: dict[str, float] = {}
    for name, value in raw_values.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise StrongDiagnosticsError("student replay primitive is non-finite")
        values[name] = float(value)
    values["probability_margin_drop"] = -values.pop("probability_margin_delta")
    values["logit_margin_drop"] = -values.pop("logit_margin_delta")
    clean_correct, robust_correct, prediction_flip = (
        row.get("student_clean_correct"),
        row.get("student_robust_correct"),
        row.get("student_clean_to_adversarial_prediction_flip"),
    )
    if not isinstance(clean_correct, bool) or not isinstance(robust_correct, bool) or not isinstance(prediction_flip, bool):
        raise StrongDiagnosticsError("student replay correctness primitive is invalid")
    return {**values, "clean_correct": clean_correct, "robust_correct": robust_correct, "prediction_flip": prediction_flip}


def _taxonomy_distributions(
    taxonomy: Mapping[int, str],
    *,
    raw_rows: Mapping[int, Mapping[str, Any]],
    online_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for kind in sorted(set(taxonomy.values())):
        members = [item for item, value in taxonomy.items() if value == kind]
        teacher = [_teacher(raw_rows[item]) for item in members]
        student = [_student_measures(raw_rows[item]) for item in members]
        summary[kind] = {
            "count": len(members),
            "teacher_adversarial_correct_rate": sum(bool(value["correct"]) for value in teacher) / len(members) if members else None,
            "teacher_signed_dominance": _quantiles([float(value["dominance"]) for value in teacher]),
            "teacher_entropy": _quantiles([float(value["entropy"]) for value in teacher]),
            "student_adversarial_logit_margin": _quantiles([float(value["adversarial_logit_margin"]) for value in student]),
            "student_adversarial_ce": _quantiles([float(value["adversarial_ce"]) for value in student]),
            "online_margin_ema_risk": _quantiles([float(online_rows[item]["margin_risk"]) for item in members]),
            "online_correctness_frequency_risk": _quantiles([float(online_rows[item]["frequency_risk"]) for item in members]),
        }
    return summary


def _dense_nr_subtype_tables(
    taxonomy: Mapping[int, str],
    *,
    online_current_wrong: set[int],
    raw_rows: Mapping[int, Mapping[str, Any]],
    strong_rows: Mapping[int, Mapping[str, Any]],
    outcome: Mapping[int, Mapping[int, Mapping[str, Any]]],
    teacher: Mapping[int, Mapping[str, float | bool]],
    clean_teacher: Mapping[int, Mapping[str, float | bool]],
) -> dict[str, Any]:
    """D5 table over the dense strong-domain NR partition only."""
    expected = ("persistent_wrong", "recovered_relapsed", "recovered_stable")
    if set(taxonomy.values()) - set(expected):
        raise StrongDiagnosticsError("dense NR taxonomy contains an unknown subtype")
    outcome_epochs = tuple(sorted(outcome))[-3:]
    if len(outcome_epochs) != 3:
        raise StrongDiagnosticsError("D5 plateau table requires exactly three outcome epochs")
    tables: dict[str, Any] = {}
    for kind in expected:
        members = sorted(item for item, value in taxonomy.items() if value == kind)
        measures = {item: _student_measures(raw_rows[item]) for item in members}
        failures = {
            item: sum(not bool(outcome[epoch][item]["correct"]) for epoch in outcome_epochs)
            for item in members
        }
        tables[kind] = {
            "count": len(members),
            "class_counts": dict(Counter(int(strong_rows[item]["class_id"]) for item in members)),
            "online_current_wrong_overlap": {
                "count": len(set(members) & online_current_wrong),
                "fraction": len(set(members) & online_current_wrong) / len(members) if members else None,
            },
            "student": {
                "clean_wrong": sum(not bool(measures[item]["clean_correct"]) for item in members),
                "clean_correct_strong_wrong": sum(bool(measures[item]["clean_correct"]) and not bool(measures[item]["robust_correct"]) for item in members),
                "clean_probability_margin": _quantiles([float(measures[item]["clean_probability_margin"]) for item in members]),
                "adversarial_probability_margin": _quantiles([float(measures[item]["adversarial_probability_margin"]) for item in members]),
                "clean_logit_margin": _quantiles([float(measures[item]["clean_logit_margin"]) for item in members]),
                "adversarial_logit_margin": _quantiles([float(measures[item]["adversarial_logit_margin"]) for item in members]),
                "adversarial_ce": _quantiles([float(measures[item]["adversarial_ce"]) for item in members]),
                "probability_margin_drop": _quantiles([float(measures[item]["probability_margin_drop"]) for item in members]),
                "logit_margin_drop": _quantiles([float(measures[item]["logit_margin_drop"]) for item in members]),
                "prediction_flip": sum(bool(measures[item]["prediction_flip"]) for item in members),
            },
            "teacher": {
                "clean": {
                    "correct": sum(bool(clean_teacher[item]["correct"]) for item in members),
                    "wrong": sum(not bool(clean_teacher[item]["correct"]) for item in members),
                    "true_probability": _quantiles([float(clean_teacher[item]["true_probability"]) for item in members]),
                    "max_wrong_probability": _quantiles([float(clean_teacher[item]["max_wrong_probability"]) for item in members]),
                    "signed_dominance": _quantiles([float(clean_teacher[item]["dominance"]) for item in members]),
                    "entropy": _quantiles([float(clean_teacher[item]["entropy"]) for item in members]),
                },
                "adversarial": {
                    "correct": sum(bool(teacher[item]["correct"]) for item in members),
                    "wrong": sum(not bool(teacher[item]["correct"]) for item in members),
                    "true_probability": _quantiles([float(teacher[item]["true_probability"]) for item in members]),
                    "max_wrong_probability": _quantiles([float(teacher[item]["max_wrong_probability"]) for item in members]),
                    "signed_dominance": _quantiles([float(teacher[item]["dominance"]) for item in members]),
                    "entropy": _quantiles([float(teacher[item]["entropy"]) for item in members]),
                },
                "clean_adversarial_js": _quantiles([float(strong_rows[item]["teacher_js"]) for item in members]),
                "correctness_flip": sum(bool(teacher[item]["correct"]) != bool(clean_teacher[item]["correct"]) for item in members),
            },
            "plateau_pattern_counts": {
                "all_wrong": sum(failures[item] == 3 for item in members),
                "majority_wrong_mixed": sum(failures[item] == 2 for item in members),
                "majority_correct_mixed": sum(failures[item] == 1 for item in members),
                "all_correct": sum(failures[item] == 0 for item in members),
            },
        }
    if sum(int(table["count"]) for table in tables.values()) != len(taxonomy):
        raise StrongDiagnosticsError("dense NR subtype partition count drifted")
    return tables


def _set_overlap(left: set[int], right: set[int]) -> dict[str, int | float | None]:
    union = left | right
    return {
        "left_count": len(left),
        "right_count": len(right),
        "intersection_count": len(left & right),
        "union_count": len(union),
        "jaccard": len(left & right) / len(union) if union else None,
    }


def _tracked_clean_provenance() -> dict[str, Any]:
    """Bind a formal diagnostic report to one tracked, clean analysis revision."""
    root = repository_root_from_source()
    paths = {
        "ffnr_strong_diagnostics": Path(__file__).resolve(),
        "ffnr_strong_diagnostics_cli": root / "src/ard/cli/ffnr_strong_diagnostics.py",
        "ffnr_forecasting": root / "src/ard/analysis/ffnr_forecasting.py",
        "ffnr_strong_replay": root / "src/ard/analysis/ffnr_strong_replay.py",
    }
    try:
        relative = [str(path.relative_to(root)) for path in paths.values()]
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", *relative],
            check=True,
            capture_output=True,
        )
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise StrongDiagnosticsError("strong diagnostics requires tracked source and Git identity") from exc
    if len(sha) != 40 or dirty:
        raise StrongDiagnosticsError("strong diagnostics requires a tracked-clean analysis revision")
    return {"git": {"sha": sha, "dirty": False}, "source_files": {name: sha256_file(path) for name, path in paths.items()}}


def analyze_run(*, label: str, feature_observations: Path, feature_lineage: Path, outcome_observations: Path, outcome_lineage: Path, online_states: Path, online_lineage: Path, validation_history: Path, validation_manifest: Path, dense_chunks: Sequence[Mapping[str, Path]], expected_count: int = 45000, expected_universe_sha256: str = EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256) -> dict[str, Any]:
    if label not in {"L2", "L4"}:
        raise StrongDiagnosticsError("diagnostics are frozen to L2/L4")
    feature_meta = _strong_lineage(path=feature_lineage, observations=feature_observations, role="feature", expected_count=expected_count, expected_universe_sha256=expected_universe_sha256)
    outcome_meta = _strong_lineage(path=outcome_lineage, observations=outcome_observations, role="outcome", expected_count=expected_count, expected_universe_sha256=expected_universe_sha256)
    online, online_meta = _online_panel(online_states, online_lineage, expected_count)
    identity = _normalized_identity(feature=feature_meta, outcome=outcome_meta, online=online_meta, validation_manifest=validation_manifest, validation_history=validation_history)
    feature_rows = _read_parquet(feature_observations)
    feature = _strong_panel(feature_rows, epochs=feature_meta["requested_epochs"], expected_count=expected_count, expected_universe_sha256=expected_universe_sha256)
    raw_feature = _raw_strong_panel(feature_rows, epochs=feature_meta["requested_epochs"], reference=feature)
    outcome = _strong_panel(_read_parquet(outcome_observations), epochs=outcome_meta["requested_epochs"], expected_count=expected_count, expected_universe_sha256=expected_universe_sha256)
    dense = _merge_dense_chunks(dense_chunks, reference=feature_meta, expected_count=expected_count, expected_universe_sha256=expected_universe_sha256)
    for epoch, rows in {**feature, **outcome}.items():
        if epoch in dense:
            raise StrongDiagnosticsError("dense chunks duplicate an existing sparse epoch")
        dense[epoch] = rows
    required_dense_epochs = tuple(range(39, 200, 5))
    if tuple(sorted(dense)) != required_dense_epochs:
        raise StrongDiagnosticsError("dense CE-PGD20 epoch coverage is incomplete")
    ids = set(feature[39])
    if ids != set(outcome[next(iter(outcome))]) or ids != set(online[39]):
        raise StrongDiagnosticsError("sparse panel stable-ID join drifted")
    endpoints = {name: _endpoint(outcome, name) for name in (PRIMARY_ENDPOINT, SECONDARY_ENDPOINT)}
    d3: dict[str, Any] = {}
    d4: dict[str, Any] = {}
    d5: dict[str, Any] = {}
    d4_taxonomies: dict[str, dict[str, dict[str, set[int]]]] = {}
    d5_taxonomies: dict[str, dict[str, set[int]]] = {}
    blind_rows: list[dict[str, int | str]] = []
    for anchor in ANCHORS:
        eligible_ff = {item for item in ids if online[anchor][item]["current_correct"]}
        eligible_cw = ids - eligible_ff
        teacher = {item: _teacher(raw_feature[anchor][item]) for item in ids}
        clean_teacher = {item: _teacher(raw_feature[anchor][item], clean=True) for item in ids}
        for endpoint, target in endpoints.items():
            key = f"e{anchor}:{endpoint}"
            correct, wrong = {item for item in eligible_ff if teacher[item]["correct"]}, {item for item in eligible_ff if not teacher[item]["correct"]}
            rates: dict[str, Any] = {
                "teacher_adv_correct": _wilson(sum(target[item] for item in correct), len(correct)),
                "teacher_adv_wrong": _wilson(sum(target[item] for item in wrong), len(wrong)),
            }
            rc, rw = rates["teacher_adv_correct"]["rate"], rates["teacher_adv_wrong"]["rate"]
            rates["risk_difference_wrong_minus_correct"] = None if rc is None or rw is None else rw - rc
            rates["risk_ratio_wrong_over_correct"] = None if rc in {None, 0} or rw is None else rw / rc
            features = {"M": {item: -feature[anchor][item]["adv_logit_margin"] for item in eligible_ff}, "H": {item: online[anchor][item]["margin_risk"] for item in eligible_ff}, "D": {item: teacher[item]["dominance"] for item in eligible_ff}}
            ordered_margin = sorted(eligible_ff, key=lambda item: (features["M"][item], item))
            deciles = []
            for decile in range(10):
                group = ordered_margin[len(ordered_margin) * decile // 10 : len(ordered_margin) * (decile + 1) // 10]
                deciles.append({"decile": decile, "count": len(group), "teacher_correct": sum(bool(teacher[item]["correct"]) for item in group), "future_failure_rate": sum(target[item] for item in group) / len(group) if group else None})
            d_strata = {
                f"teacher_{'correct' if teacher_correct else 'wrong'}:outcome_{'failure' if outcome else 'nonfailure'}": _quantiles(
                    [float(teacher[item]["dominance"]) for item in eligible_ff if bool(teacher[item]["correct"]) == teacher_correct and target[item] == outcome]
                )
                for teacher_correct in (True, False)
                for outcome in (0, 1)
            }
            d3[key] = {
                "eligibility": {"FF": len(eligible_ff), "current_wrong": len(eligible_cw)},
                "teacher_failure_rates": rates,
                "signed_dominance": {"teacher_correct": _quantiles([float(teacher[item]["dominance"]) for item in correct]), "teacher_wrong": _quantiles([float(teacher[item]["dominance"]) for item in wrong]), "teacher_correctness_x_outcome": d_strata},
                "spearman": {"D_vs_logit_margin_risk": _spearman(features["D"], features["M"]), "D_vs_online_margin_ema_risk": _spearman(features["D"], features["H"]), "D_vs_online_correctness_frequency_risk": _spearman(features["D"], {item: float(online[anchor][item]["frequency_risk"]) for item in eligible_ff})},
                "strong_margin_deciles": deciles,
                "teacher_clean_adv_response": {"js": _quantiles([float(feature[anchor][item]["teacher_js"]) for item in eligible_ff]), "dominance_delta": _quantiles([float(teacher[item]["dominance"] - clean_teacher[item]["dominance"]) for item in eligible_ff]), "correctness_flip": sum(bool(teacher[item]["correct"]) != bool(clean_teacher[item]["correct"]) for item in eligible_ff)},
                "oof": _oof_scores(sorted(eligible_ff), target, features, {item: int(feature[anchor][item]["class_id"]) for item in eligible_ff}),
            }
        ff_primary = {item for item in eligible_ff if endpoints[PRIMARY_ENDPOINT][item]}
        online_sequence = {item: [bool(online[epoch][item]["current_correct"]) for epoch in ANCHORS if epoch <= anchor] for item in ff_primary}
        strong_sequence = {item: [bool(feature[epoch][item]["correct"]) for epoch in ANCHORS if epoch <= anchor] for item in ff_primary}
        transitions = {f"{left}->{right}:{a}{b}": sum(bool(online[left][item]["current_correct"]) == a and bool(online[right][item]["current_correct"]) == b for item in ids) for left, right in zip(ANCHORS, ANCHORS[1:]) for a, b in ((True, True), (True, False), (False, True), (False, False)) if right <= anchor}
        online_taxonomy = {item: _snapshot_taxonomy(sequence) for item, sequence in online_sequence.items()}
        strong_taxonomy = {item: _snapshot_taxonomy(sequence) for item, sequence in strong_sequence.items()}
        strong_current_wrong = {item for item in ids if not bool(dense[anchor][item]["correct"])}
        nr_taxonomy = {
            item: _dense_non_recovery([bool(dense[epoch][item]["correct"]) for epoch in sorted(dense) if epoch >= anchor])
            for item in strong_current_wrong
        }
        nr = Counter(nr_taxonomy.values())
        d4_taxonomies[str(anchor)] = {
            "online": {kind: {item for item, value in online_taxonomy.items() if value == kind} for kind in sorted(set(online_taxonomy.values()))},
            "strong": {kind: {item for item, value in strong_taxonomy.items() if value == kind} for kind in sorted(set(strong_taxonomy.values()))},
        }
        d4[str(anchor)] = {
            "online_snapshot_taxonomy": dict(Counter(online_taxonomy.values())),
            "strong_snapshot_taxonomy": dict(Counter(strong_taxonomy.values())),
            "online_subtype_distributions": _taxonomy_distributions(online_taxonomy, raw_rows=raw_feature[anchor], online_rows=online[anchor]),
            "strong_subtype_distributions": _taxonomy_distributions(strong_taxonomy, raw_rows=raw_feature[anchor], online_rows=online[anchor]),
            "eligibility_transitions": transitions,
            "current_state_discordance": sum(bool(online[anchor][item]["current_correct"]) != bool(feature[anchor][item]["correct"]) for item in ids),
            "eligibility": {"FF": len(eligible_ff), "CW": len(eligible_cw)},
        }
        patterns = Counter(sum(not bool(outcome[epoch][item]["correct"]) for epoch in sorted(outcome)[-3:]) for item in eligible_cw)
        measures = {item: _student_measures(raw_feature[anchor][item]) for item in eligible_cw}
        pattern_names = {3: "all_wrong", 2: "majority_wrong_mixed", 1: "majority_correct_mixed", 0: "all_correct"}
        pattern_members = {
            name: [
                item
                for item in eligible_cw
                if sum(not bool(outcome[epoch][item]["correct"]) for epoch in sorted(outcome)[-3:]) == failures
            ]
            for failures, name in pattern_names.items()
        }
        d5_taxonomies[str(anchor)] = {kind: {item for item, value in nr_taxonomy.items() if value == kind} for kind in sorted(set(nr_taxonomy.values()))}
        d5[str(anchor)] = {
            "current_wrong": len(eligible_cw),
            "dense_strong_current_wrong": len(strong_current_wrong),
            "online_vs_dense_strong_current_wrong": {
                "both_wrong": len(eligible_cw & strong_current_wrong),
                "online_wrong_strong_correct": len(eligible_cw - strong_current_wrong),
                "online_correct_strong_wrong": len(strong_current_wrong - eligible_cw),
                "both_correct": len(ids - (eligible_cw | strong_current_wrong)),
                "jaccard_wrong": _set_overlap(eligible_cw, strong_current_wrong)["jaccard"],
            },
            "clean_wrong": sum(not bool(measures[item]["clean_correct"]) for item in eligible_cw),
            "robustness_specific": sum(bool(measures[item]["clean_correct"]) and not bool(measures[item]["robust_correct"]) for item in eligible_cw),
            "student": {name: _quantiles([float(measures[item][name]) for item in eligible_cw]) for name in ("clean_probability_margin", "adversarial_probability_margin", "clean_logit_margin", "adversarial_logit_margin", "adversarial_ce", "probability_margin_drop", "logit_margin_drop")},
            "student_prediction_flip": sum(bool(measures[item]["prediction_flip"]) for item in eligible_cw),
            "online_vs_strong_current_discordance": sum(
                bool(online[anchor][item]["current_correct"]) != bool(feature[anchor][item]["correct"])
                for item in eligible_cw
            ),
            "teacher": {"clean_correct": sum(bool(clean_teacher[item]["correct"]) for item in eligible_cw), "adversarial_correct": sum(bool(teacher[item]["correct"]) for item in eligible_cw), "clean_true_probability": _quantiles([float(clean_teacher[item]["true_probability"]) for item in eligible_cw]), "adversarial_true_probability": _quantiles([float(teacher[item]["true_probability"]) for item in eligible_cw]), "clean_max_wrong_probability": _quantiles([float(clean_teacher[item]["max_wrong_probability"]) for item in eligible_cw]), "adversarial_max_wrong_probability": _quantiles([float(teacher[item]["max_wrong_probability"]) for item in eligible_cw]), "clean_signed_dominance": _quantiles([float(clean_teacher[item]["dominance"]) for item in eligible_cw]), "adversarial_signed_dominance": _quantiles([float(teacher[item]["dominance"]) for item in eligible_cw]), "clean_entropy": _quantiles([float(clean_teacher[item]["entropy"]) for item in eligible_cw]), "adversarial_entropy": _quantiles([float(teacher[item]["entropy"]) for item in eligible_cw]), "clean_adversarial_js": _quantiles([float(feature[anchor][item]["teacher_js"]) for item in eligible_cw]), "prediction_flip": sum(bool(teacher[item]["correct"]) != bool(clean_teacher[item]["correct"]) for item in eligible_cw)},
            "plateau_patterns": {
                "counts": {name: patterns[failures] for failures, name in pattern_names.items()},
                "by_pattern": {
                    name: {
                        "class_counts": dict(Counter(int(feature[anchor][item]["class_id"]) for item in members)),
                        "student_adversarial_ce": _quantiles([float(measures[item]["adversarial_ce"]) for item in members]),
                        "student_adversarial_logit_margin": _quantiles([float(measures[item]["adversarial_logit_margin"]) for item in members]),
                        "teacher_adversarial_dominance": _quantiles([float(teacher[item]["dominance"]) for item in members]),
                        "teacher_adversarial_correctness": {
                            "correct": sum(bool(teacher[item]["correct"]) for item in members),
                            "wrong": sum(not bool(teacher[item]["correct"]) for item in members),
                        },
                        "student_clean_correctness": {
                            "correct": sum(bool(measures[item]["clean_correct"]) for item in members),
                            "wrong": sum(not bool(measures[item]["clean_correct"]) for item in members),
                        },
                    }
                    for name, members in pattern_members.items()
                },
            },
            "dense_strong_domain_non_recovery": dict(nr),
            "dense_non_recovery_class_counts": {kind: dict(Counter(int(feature[anchor][item]["class_id"]) for item, value in nr_taxonomy.items() if value == kind)) for kind in sorted(set(nr_taxonomy.values()))},
            "dense_strong_domain_non_recovery_subtypes": _dense_nr_subtype_tables(
                nr_taxonomy,
                online_current_wrong=eligible_cw,
                raw_rows=raw_feature[anchor],
                strong_rows=feature[anchor],
                outcome=outcome,
                teacher=teacher,
                clean_teacher=clean_teacher,
            ),
        }
        if anchor == 79:
            blind_rows = _blinded_candidate_rows(
                label,
                taxonomy=nr_taxonomy,
                classes={item: int(feature[anchor][item]["class_id"]) for item in strong_current_wrong},
                teacher=teacher,
            )
    return {"schema_version": 1, "contract": CONTRACT, "diagnostic_only": True, "no_intervention_or_official_test": True, "label": label, "input_identity": {**identity, "stable_id_class_universe_sha256": expected_universe_sha256, "files": {name: _file_identity(path) for name, path in {"feature_observations": feature_observations, "feature_lineage": feature_lineage, "outcome_observations": outcome_observations, "outcome_lineage": outcome_lineage, "online_states": online_states, "online_lineage": online_lineage, "validation_history": validation_history, "validation_manifest": validation_manifest}.items()}}, "D3_teacher_decomposition": d3, "D4_snapshot_taxonomy": d4, "D5_current_wrong": d5, "dense_chunks_declared": [{name: _file_identity(path) for name, path in chunk.items()} for chunk in dense_chunks], "_d4_taxonomies": d4_taxonomies, "_d5_taxonomies": d5_taxonomies, "_blinded_candidate_rows": blind_rows}


def _cross_seed_tables(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    left, right = reports["L2"], reports["L4"]
    if left["input_identity"]["stable_id_class_universe_sha256"] != right["input_identity"]["stable_id_class_universe_sha256"]:
        return {"available": False, "reason": "stable_universe_differs"}
    tables: dict[str, Any] = {"available": True, "D4": {}, "D5": {}}
    for anchor in ANCHORS:
        key = str(anchor)
        for domain in ("online", "strong"):
            left_kinds = left["_d4_taxonomies"][key][domain]
            right_kinds = right["_d4_taxonomies"][key][domain]
            for kind in sorted(set(left_kinds) & set(right_kinds)):
                tables["D4"][f"e{anchor}:{domain}:{kind}"] = _set_overlap(left_kinds[kind], right_kinds[kind])
        left_kinds = left["_d5_taxonomies"][key]
        right_kinds = right["_d5_taxonomies"][key]
        for kind in sorted(set(left_kinds) & set(right_kinds)):
            tables["D5"][f"e{anchor}:{kind}"] = _set_overlap(left_kinds[kind], right_kinds[kind])
    return tables


def _render_blinded_cifar10_panel(*, output_dir: Path, rows: Sequence[Mapping[str, Any]], dataset_root: Path) -> list[dict[str, Any]]:
    """Render only exact stable CIFAR-10 train IDs; keep selection causes out of public output."""
    try:
        from torchvision.datasets import CIFAR10
    except ImportError as exc:  # pragma: no cover - optional runtime environment
        raise StrongDiagnosticsError("CIFAR-10 blinded panel requires torchvision") from exc
    if not dataset_root.is_dir():
        raise StrongDiagnosticsError("configured CIFAR-10 train root is unavailable")
    try:
        dataset = CIFAR10(root=str(dataset_root), train=True, download=False)
    except Exception as exc:
        raise StrongDiagnosticsError("configured CIFAR-10 train dataset is unavailable") from exc
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    public: list[dict[str, Any]] = []
    for panel_index, row in enumerate(rows, start=1):
        sample_id, class_id, label = row.get("sample_id"), row.get("class_id"), row.get("run_label")
        if not isinstance(sample_id, int) or not isinstance(class_id, int) or not isinstance(label, str) or not 0 <= sample_id < len(dataset):
            raise StrongDiagnosticsError("blinded candidate stable-ID/class contract is invalid")
        image, observed_class = dataset[sample_id]
        if int(observed_class) != class_id:
            raise StrongDiagnosticsError("CIFAR-10 rendered label mismatches replay stable ID")
        path = image_dir / f"{panel_index:04d}-{label}-id{sample_id}.png"
        image.save(path, format="PNG")
        public.append({"run_label": label, "panel_index": panel_index, "sample_id": sample_id, "class_id": class_id, "image_path": path.relative_to(output_dir).as_posix()})
    return public


def write_outputs(*, output_dir: Path, reports: Mapping[str, Mapping[str, Any]], config_path: Path, cifar10_train_root: Path) -> dict[str, Path]:
    paths = {"report": output_dir / "ffnr-strong-diagnostics.json", "blinded_manifest": output_dir / "ffnr-strong-blinded-candidates.json", "points": output_dir / "ffnr-strong-diagnostic-points.parquet"}
    if output_dir.exists() or any(path.exists() for path in paths.values()):
        raise StrongDiagnosticsError("refusing to overwrite strong diagnostics output")
    if set(reports) != {"L2", "L4"}:
        raise StrongDiagnosticsError("diagnostics require exactly L2/L4")
    provenance = _tracked_clean_provenance()
    requested_panel = [
        {"run_label": label, "sample_id": int(row["sample_id"]), "class_id": int(row["class_id"])}
        for label, report in sorted(reports.items())
        for row in report.get("_blinded_candidate_rows", [])
    ]
    if not requested_panel:
        raise StrongDiagnosticsError("blinded candidate panel is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = _render_blinded_cifar10_panel(output_dir=output_dir, rows=requested_panel, dataset_root=cifar10_train_root)
    write_sample_parquet(panel, paths["points"])
    paths["blinded_manifest"].write_bytes(canonical_json({"contract": "ffnr_strong_blinded_candidates_v2", "diagnostic_only": True, "contains_images": True, "contains_outcome_or_score_or_teacher_state": False, "rows": panel}) + b"\n")
    public = {label: {key: value for key, value in report.items() if not key.startswith("_")} for label, report in reports.items()}
    paths["report"].write_bytes(canonical_json({"schema_version": 1, "contract": CONTRACT, "analysis_provenance": provenance, "config_sha256": sha256_file(config_path), "reports": public, "cross_seed": _cross_seed_tables(reports), "blinded_manifest_sha256": sha256_file(paths["blinded_manifest"]), "points_sha256": sha256_file(paths["points"])} ) + b"\n")
    return paths
