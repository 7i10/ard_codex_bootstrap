#!/usr/bin/env python3
# ruff: noqa: E501
"""Run the CPU-only FFNR next-evidence analyses from the frozen CE-PGD20 data.

This command does not launch attacks or training.  It computes chance-adjusted
seed agreement, cross-seed teacher incremental prediction, and teacher-correct
subset diagnostics.  The future-failure endpoint and anchor are explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ard.analysis.signal_audit import binary_metrics

RUNS = ("L2", "L4")
ANCHORS = (39, 59, 79)
TERMINAL_EPOCHS = (189, 194, 199)
ENDPOINTS = {"majority": 2, "all": 3}
EXPECTED_COUNT = 45_000
DEFAULT_OUT = Path(".cache/analysis/ffnr-next-evidence-6a90011-v1")
DEFAULT_OUTCOME = {
    "L2": Path(".cache/analysis/ffnr-strong-replay/l2-outcome-e5cb442/strong-observations.parquet"),
    "L4": Path(".cache/analysis/ffnr-strong-replay/l4-outcome-e5cb442/strong-observations.parquet"),
}
DEFAULT_FEATURE = {
    "L2": Path(".cache/analysis/ffnr-strong-replay/l2-feature-e5cb442/strong-observations.parquet"),
    "L4": Path(".cache/analysis/ffnr-strong-replay/l4-feature-e5cb442/strong-observations.parquet"),
}
DEFAULT_ONLINE = {
    "L2": Path(".cache/analysis/h5-online-cd56b72/L2/online-anchor-states.parquet"),
    "L4": Path(".cache/analysis/h5-online-cd56b72/L4/online-anchor-states.parquet"),
}


class NextEvidenceError(ValueError):
    """Raised when a frozen FFNR input contract cannot be established."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        rows = pq.read_table(path).to_pylist()
    except Exception as exc:
        raise NextEvidenceError(f"cannot read Parquet: {path}") from exc
    if not rows:
        raise NextEvidenceError(f"empty Parquet: {path}")
    return rows


def _argmax(values: Sequence[float]) -> int:
    if len(values) != 10:
        raise NextEvidenceError("teacher probability vector must contain ten classes")
    maximum = max(values)
    candidates = [index for index, value in enumerate(values) if value == maximum]
    if len(candidates) != 1:
        raise NextEvidenceError("teacher probability argmax tie is outside the contract")
    return candidates[0]


def _validate_universe(rows: Sequence[Mapping[str, Any]], *, name: str) -> tuple[int, ...]:
    ids = tuple(sorted(int(row["sample_id"]) for row in rows))
    if len(ids) != EXPECTED_COUNT or len(set(ids)) != EXPECTED_COUNT:
        raise NextEvidenceError(f"{name} does not contain the expected unique 45k IDs")
    labels = {int(row["sample_id"]): int(row["class_id"]) for row in rows}
    if any(label < 0 or label >= 10 for label in labels.values()):
        raise NextEvidenceError(f"{name} has an invalid CIFAR-10 class")
    return ids


def _outcome(path: Path) -> tuple[dict[int, dict[int, bool]], dict[int, int], tuple[int, ...]]:
    rows = _read(path)
    epochs = {int(row["epoch"]) for row in rows}
    if not set(TERMINAL_EPOCHS).issubset(epochs):
        raise NextEvidenceError(f"{path} lacks all terminal epochs {TERMINAL_EPOCHS}")
    by_epoch: dict[int, dict[int, bool]] = {epoch: {} for epoch in TERMINAL_EPOCHS}
    labels: dict[int, int] = {}
    for row in rows:
        epoch, sample_id = int(row["epoch"]), int(row["sample_id"])
        if epoch not in by_epoch:
            continue
        if sample_id in by_epoch[epoch] or not isinstance(row.get("student_robust_correct"), bool):
            raise NextEvidenceError(f"{path} has duplicate or invalid terminal observation")
        label = int(row["class_id"])
        if sample_id in labels and labels[sample_id] != label:
            raise NextEvidenceError(f"{path} class identity drifted")
        labels[sample_id] = label
        by_epoch[epoch][sample_id] = bool(row["student_robust_correct"])
    universe = _validate_universe(
        [{"sample_id": sample_id, "class_id": label} for sample_id, label in labels.items()], name=str(path)
    )
    if any(set(by_epoch[epoch]) != set(universe) for epoch in TERMINAL_EPOCHS):
        raise NextEvidenceError(f"{path} terminal stable-ID universe drifted")
    return by_epoch, labels, universe


def _feature(path: Path, anchors: Sequence[int]) -> dict[int, dict[int, dict[str, float | int | bool]]]:
    rows = _read(path)
    result: dict[int, dict[int, dict[str, float | int | bool]]] = {anchor: {} for anchor in anchors}
    for row in rows:
        epoch = int(row["epoch"])
        if epoch not in result:
            continue
        sample_id, label = int(row["sample_id"]), int(row["class_id"])
        probabilities = [float(value) for value in row["teacher_adversarial_probabilities"]]
        if sample_id in result[epoch]:
            raise NextEvidenceError(f"duplicate feature ID in {path}")
        teacher_pred = _argmax(probabilities)
        wrong_probability = max(value for index, value in enumerate(probabilities) if index != label)
        result[epoch][sample_id] = {
            "class_id": label,
            "strong_margin_risk": -float(row["student_adversarial_logit_margin"]),
            "teacher_dominance": wrong_probability - probabilities[label],
            "teacher_correct": teacher_pred == label,
        }
    if any(len(result[anchor]) != EXPECTED_COUNT for anchor in anchors):
        raise NextEvidenceError(f"{path} lacks exact feature coverage for {anchors}")
    return result


def _online(path: Path, anchors: Sequence[int]) -> dict[int, dict[int, dict[str, float | int | bool]]]:
    rows = _read(path)
    result: dict[int, dict[int, dict[str, float | int | bool]]] = {anchor: {} for anchor in anchors}
    for row in rows:
        anchor = int(row["anchor_epoch"])
        if anchor not in result:
            continue
        sample_id, label = int(row["sample_id"]), int(row["true_label"])
        if sample_id in result[anchor]:
            raise NextEvidenceError(f"duplicate online ID in {path}")
        seen = int(row["robust_correct_count"])
        frequency = float(row["robust_correct_frequency_inclusive"])
        if seen < 0 or seen > anchor + 1 or not math.isclose(frequency, seen / (anchor + 1), abs_tol=1e-7):
            raise NextEvidenceError(f"online frequency contract drifted in {path}")
        result[anchor][sample_id] = {
            "class_id": label,
            "current_correct": bool(row["previous_robust_correct"]),
            "margin_ema_risk": (1.0 - float(row["margin_ema"])) / 2.0,
        }
    if any(len(result[anchor]) != EXPECTED_COUNT for anchor in anchors):
        raise NextEvidenceError(f"{path} lacks exact online coverage for {anchors}")
    return result


def _frequencies(outcome: Mapping[int, Mapping[int, bool]], universe: Sequence[int]) -> dict[int, float]:
    return {sample_id: sum(not outcome[epoch][sample_id] for epoch in TERMINAL_EPOCHS) / 3.0 for sample_id in universe}


def _binary_metrics(targets: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    result = binary_metrics(targets, scores)
    result["brier"] = sum((float(score) - target) ** 2 for target, score in zip(targets, scores, strict=True)) / len(targets)
    return result


def _fit_logistic_fast(features: Sequence[Sequence[float]], targets: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not features or len(features) != len(targets) or len(set(targets)) != 2:
        raise NextEvidenceError("cross-seed logistic fit requires both outcome classes")
    matrix = np.asarray(features, dtype=float)
    means, scales = matrix.mean(axis=0), matrix.std(axis=0)
    scales = np.maximum(scales, 1e-12)
    normalized = (matrix - means) / scales
    design = np.column_stack((np.ones(len(normalized)), normalized))
    labels = np.asarray(targets, dtype=float)
    weights = np.zeros(design.shape[1], dtype=float)
    for _ in range(180):
        logits = np.clip(design @ weights, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - labels) / len(labels)
        gradient[1:] += 0.001 * weights[1:]
        weights -= 0.15 * gradient
    return weights, means, scales


def _predict_logistic_fast(fit: tuple[np.ndarray, np.ndarray, np.ndarray], features: Sequence[Sequence[float]]) -> list[float]:
    weights, means, scales = fit
    matrix = (np.asarray(features, dtype=float) - means) / scales
    logits = np.clip(np.column_stack((np.ones(len(matrix)), matrix)) @ weights, -35.0, 35.0)
    return (1.0 / (1.0 + np.exp(-logits))).tolist()


def _agreement(left: np.ndarray, right: np.ndarray, left_freq: np.ndarray, right_freq: np.ndarray) -> dict[str, Any]:
    n = len(left)
    both = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    p11, p10, p01, p00 = both / n, int(np.logical_and(left, ~right).sum()) / n, int(np.logical_and(~left, right).sum()) / n, int(np.logical_and(~left, ~right).sum()) / n
    observed = p11 + p00
    expected = (p11 + p10) * (p11 + p01) + (p01 + p00) * (p10 + p00)
    return {
        "left_positive_count": int(left.sum()),
        "right_positive_count": int(right.sum()),
        "left_prevalence": float(left.mean()),
        "right_prevalence": float(right.mean()),
        "intersection": both,
        "union": union,
        "raw_jaccard": both / union if union else None,
        "raw_agreement_rate": observed,
        "cohen_kappa": (observed - expected) / (1.0 - expected) if expected < 1 else None,
        "frequency_spearman": _correlation(left_freq, right_freq, rank=True),
        "frequency_pearson": _correlation(left_freq, right_freq, rank=False),
        "frequency_agreement_matrix": [
            [int(np.sum((left_freq == left_level) & (right_freq == right_level))) for right_level in (0.0, 1 / 3, 2 / 3, 1.0)]
            for left_level in (0.0, 1 / 3, 2 / 3, 1.0)
        ],
    }


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float | None:
    if rank:
        left = _midranks(left)
        right = _midranks(right)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return None if denominator == 0 else float(np.dot(left, right) / denominator)


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return result


def _permutation_null(left: np.ndarray, right: np.ndarray, *, permutations: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    left_count, right_count = int(left.sum()), int(right.sum())
    # Sampling the intersection from this hypergeometric law is exactly the
    # same fixed-count null as uniformly permuting the right-hand mask over
    # the stable-ID universe, without materializing 10,000 45k-ID arrays.
    intersections = rng.hypergeometric(left_count, len(left) - left_count, right_count, size=permutations)
    unions = left_count + right_count - intersections
    values = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions != 0)
    observed_intersection = int(np.logical_and(left, right).sum())
    observed_union = int(np.logical_or(left, right).sum())
    observed = observed_intersection / observed_union
    return {
        "permutations": permutations,
        "null_sampler": "hypergeometric_equivalent_to_fixed_count_ID_permutation",
        "seed": seed,
        "null_mean": float(values.mean()),
        "null_p02_5": float(np.quantile(values, 0.025)),
        "null_p97_5": float(np.quantile(values, 0.975)),
        "observed": observed,
        "observed_minus_null_mean": observed - float(values.mean()),
        "observed_over_null_mean": observed / float(values.mean()) if values.mean() else None,
        "empirical_tail_probability": float((1 + np.count_nonzero(values >= observed)) / (permutations + 1)),
    }


def _bootstrap_agreement(left: np.ndarray, right: np.ndarray, left_freq: np.ndarray, right_freq: np.ndarray, *, replicates: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("jaccard", "kappa", "spearman", "pearson")}
    n = len(left)
    left_rank, right_rank = _midranks(left_freq), _midranks(right_freq)
    for _ in range(replicates):
        indices = rng.integers(0, n, size=n)
        a, b = left[indices], right[indices]
        intersection = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        p11 = intersection / n
        p10 = int(np.logical_and(a, ~b).sum()) / n
        p01 = int(np.logical_and(~a, b).sum()) / n
        p00 = int(np.logical_and(~a, ~b).sum()) / n
        observed = p11 + p00
        expected = (p11 + p10) * (p11 + p01) + (p01 + p00) * (p10 + p00)
        values["jaccard"].append(intersection / union if union else 0.0)
        values["kappa"].append((observed - expected) / (1.0 - expected) if expected < 1 else 0.0)
        values["spearman"].append(_correlation(left_rank[indices], right_rank[indices], rank=False))
        values["pearson"].append(_correlation(left_freq[indices], right_freq[indices], rank=False))
    return {
        "replicates": replicates,
        "seed": seed,
        "ci95": {
            name: {"lower": float(np.quantile([value for value in series if value is not None], 0.025)), "upper": float(np.quantile([value for value in series if value is not None], 0.975))}
            for name, series in values.items()
        },
    }


def _cross_seed_metrics(source: list[dict[str, Any]], target: list[dict[str, Any]], *, feature_names: tuple[str, ...]) -> dict[str, Any]:
    source = sorted(source, key=lambda row: int(row["sample_id"]))
    target = sorted(target, key=lambda row: int(row["sample_id"]))
    features_source = [[float(row[name]) for name in feature_names] for row in source]
    features_target = [[float(row[name]) for name in feature_names] for row in target]
    targets_source = [int(row["outcome"]) for row in source]
    targets_target = [int(row["outcome"]) for row in target]
    fit = _fit_logistic_fast(features_source, targets_source)
    scores = _predict_logistic_fast(fit, features_target)
    metrics = _binary_metrics(targets_target, scores)
    return {**metrics, "feature_names": feature_names, "train_count": len(source), "eval_count": len(target)}


def _teacher_correct_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [row for row in rows if bool(row["teacher_correct"])]
    ordered = sorted(correct, key=lambda row: float(row["D"]))
    bins: list[dict[str, Any]] = []
    for index in range(4):
        start, stop = len(ordered) * index // 4, len(ordered) * (index + 1) // 4
        group = ordered[start:stop]
        bins.append({"quartile": index, "count": len(group), "future_failure_rate": sum(int(row["outcome"]) for row in group) / len(group) if group else None, "dominance_mean": sum(float(row["D"]) for row in group) / len(group) if group else None})
    if len({int(row["outcome"]) for row in correct}) < 2:
        metrics = None
    else:
        raw = np.array([float(row["D"]) for row in correct], dtype=float)
        low, high = float(raw.min()), float(raw.max())
        scores = [float((value - low) / (high - low)) if high > low else 0.5 for value in raw]
        metrics = _binary_metrics([int(row["outcome"]) for row in correct], scores)
    return {"count": len(correct), "future_failure_count": sum(int(row["outcome"]) for row in correct), "metrics": metrics, "dominance_quartiles": bins}


def analyze(*, outcome_paths: Mapping[str, Path], feature_paths: Mapping[str, Path], online_paths: Mapping[str, Path], permutations: int, bootstrap_replicates: int) -> dict[str, Any]:
    outcomes, features, online = {}, {}, {}
    for run in RUNS:
        outcomes[run] = _outcome(outcome_paths[run])
        features[run] = _feature(feature_paths[run], ANCHORS)
        online[run] = _online(online_paths[run], ANCHORS)
        universe = outcomes[run][2]
        for anchor in ANCHORS:
            if set(features[run][anchor]) != set(universe) or set(online[run][anchor]) != set(universe):
                raise NextEvidenceError(f"{run} feature/online stable-ID universe drifted")
    agreement: dict[str, Any] = {}
    cross_seed: dict[str, Any] = {}
    teacher_correct: dict[str, Any] = {}
    for endpoint, threshold in ENDPOINTS.items():
        masks, frequencies = {}, {}
        for run in RUNS:
            outcome, _, universe = outcomes[run]
            frequency = _frequencies(outcome, universe)
            frequencies[run] = np.array([frequency[sample_id] for sample_id in universe], dtype=float)
            masks[run] = frequencies[run] >= threshold / 3.0
        agreement[endpoint] = {
            "cohort": "all_stable_id_universe",
            "point": _agreement(masks["L2"], masks["L4"], frequencies["L2"], frequencies["L4"]),
            "permutation_null": _permutation_null(masks["L2"], masks["L4"], permutations=permutations, seed=3100 + threshold),
            "bootstrap": _bootstrap_agreement(masks["L2"], masks["L4"], frequencies["L2"], frequencies["L4"], replicates=bootstrap_replicates, seed=4100 + threshold),
        }
        for anchor in ANCHORS:
            for source, target in (("L2", "L4"), ("L4", "L2")):
                source_rows, target_rows = [], []
                for run, destination in ((source, source_rows), (target, target_rows)):
                    outcome, labels, universe = outcomes[run]
                    future = {sample_id: int(frequencies[run][index] >= threshold / 3.0) for index, sample_id in enumerate(universe)}
                    for sample_id in universe:
                        current = bool(online[run][anchor][sample_id]["current_correct"])
                        if not current:
                            continue
                        strong = features[run][anchor][sample_id]
                        destination.append({
                            "sample_id": sample_id,
                            "class_id": labels[sample_id],
                            "outcome": future[sample_id],
                            "M": float(strong["strong_margin_risk"]),
                            "H": float(online[run][anchor][sample_id]["margin_ema_risk"]),
                            "D": float(strong["teacher_dominance"]),
                            "teacher_correct": bool(strong["teacher_correct"]),
                        })
                key = f"{endpoint}:anchor{anchor}:{source}_fit_{target}_eval"
                models = {
                    "M": _cross_seed_metrics(source_rows, target_rows, feature_names=("M",)),
                    "H": _cross_seed_metrics(source_rows, target_rows, feature_names=("H",)),
                    "M+D": _cross_seed_metrics(source_rows, target_rows, feature_names=("M", "D")),
                    "H+D": _cross_seed_metrics(source_rows, target_rows, feature_names=("H", "D")),
                    "M+H": _cross_seed_metrics(source_rows, target_rows, feature_names=("M", "H")),
                    "M+H+D": _cross_seed_metrics(source_rows, target_rows, feature_names=("M", "H", "D")),
                }
                models["delta_M+D_vs_M"] = _metric_delta(models["M+D"], models["M"])
                models["delta_H+D_vs_H"] = _metric_delta(models["H+D"], models["H"])
                models["delta_M+H+D_vs_M+H"] = _metric_delta(models["M+H+D"], models["M+H"])
                cross_seed[key] = models
                teacher_correct[f"{endpoint}:anchor{anchor}:{source}"] = _teacher_correct_subset(source_rows)
    return {
        "contract": "ffnr_next_evidence_v1",
        "endpoint_epochs": list(TERMINAL_EPOCHS),
        "anchors": list(ANCHORS),
        "permutation_count": permutations,
        "bootstrap_replicates": bootstrap_replicates,
        "source_sha256": {"outcome": {run: _sha256(outcome_paths[run]) for run in RUNS}, "feature": {run: _sha256(feature_paths[run]) for run in RUNS}, "online": {run: _sha256(online_paths[run]) for run in RUNS}},
        "agreement": agreement,
        "cross_seed": cross_seed,
        "teacher_correct_subset": teacher_correct,
        "irt_development_sensitivity": {"status": "blocked_missing_bartoldson_ce_pgd20_artifact", "required": "existing Bartoldson CE-PGD20 replay at [104,109,114] or equivalent frozen input", "official_test_or_autoattack": False},
    }


def _metric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        "delta_auroc": float(candidate["auroc"] - baseline["auroc"]),
        "delta_auprc": float(candidate["auprc"] - baseline["auprc"]),
        "delta_log_loss": float(candidate["log_loss"] - baseline["log_loss"]),
        "delta_brier": float(candidate["brier"] - baseline["brier"]),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# FF/NR 次段階エビデンス結果", "", "この報告はtrain splitの既存CE-PGD20 replayとonline stateだけを使ったCPU解析です。official CIFAR-10 test、AutoAttack、新規trainingは実行していません。", "", "## A. Future Failure seed agreement", "", "primaryは全45,000 stable-ID universe、majority/allは別endpointです。", "", "| endpoint | L2 count | L4 count | raw Jaccard | null mean | observed-null | kappa | Spearman | Pearson |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for endpoint in ENDPOINTS:
        point = report["agreement"][endpoint]["point"]
        null = report["agreement"][endpoint]["permutation_null"]
        lines.append(f"| {endpoint} | {point['left_positive_count']} | {point['right_positive_count']} | {point['raw_jaccard']:.4f} | {null['null_mean']:.4f} | {null['observed_minus_null_mean']:.4f} | {point['cohen_kappa']:.4f} | {point['frequency_spearman']:.4f} | {point['frequency_pearson']:.4f} |")
    majority_bootstrap = report["agreement"]["majority"]["bootstrap"]["ci95"]
    lines += ["", "Permutation null固定: seed=3102/3103相当、10,000 fixed-count null samples（ID permutationと超幾何分布が同値）。paired bootstrapは2,000回で、これはtraining-seed CIではありません。", f"majorityのpaired 95% CI: Jaccard [{majority_bootstrap['jaccard']['lower']:.4f}, {majority_bootstrap['jaccard']['upper']:.4f}]、kappa [{majority_bootstrap['kappa']['lower']:.4f}, {majority_bootstrap['kappa']['upper']:.4f}]。", "", "## B. Cross-seed teacher incremental information", "", "各cellは片方のseedでfitし、もう片方でevaluateしたFF（anchor時点current-correct）です。standardizationはfit seedだけで計算しました。", "", "| endpoint/anchor/direction | M AUROC | H AUROC | M+D AUROC | H+D AUROC | M+H AUROC | M+H+D AUROC | Δ(M+D−M) | Δ(H+D−H) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, value in report["cross_seed"].items():
        lines.append(f"| {key} | {value['M']['auroc']:.4f} | {value['H']['auroc']:.4f} | {value['M+D']['auroc']:.4f} | {value['H+D']['auroc']:.4f} | {value['M+H']['auroc']:.4f} | {value['M+H+D']['auroc']:.4f} | {value['delta_M+D_vs_M']['delta_auroc']:+.4f} | {value['delta_H+D_vs_H']['delta_auroc']:+.4f} |")
    lines += ["", "## C. Teacher-correct subset", "", "Teacher-correct subsetのD連続値は、Teacher wrong/correctのbinary splitと分離して解釈します。", "", "| endpoint/anchor/run | n | FF count | D AUROC | quartile FF rates |", "| --- | ---: | ---: | ---: | --- |"]
    for key, value in report["teacher_correct_subset"].items():
        metrics = value["metrics"]
        rates = ", ".join(f"Q{x['quartile']}={x['future_failure_rate']:.3f}" for x in value["dominance_quartiles"] if x["future_failure_rate"] is not None)
        lines.append(f"| {key} | {value['count']} | {value['future_failure_count']} | {metrics['auroc']:.4f} | {rates} |" if metrics else f"| {key} | {value['count']} | {value['future_failure_count']} | n/a | {rates} |")
    lines += ["", "## D. IRT", "", "Bartoldsonの同一contract CE-PGD20 replayはローカルに存在しないため、[104,109,114] sensitivityを推測・再構成していません。新規GPU replayはこの段階では自動起動していません。", "", "## 判定", "", "この結果だけでTeacher dominanceを介入へ採用しません。cross-seed delta、Teacher-correct subset、chance-adjusted agreement、IRT artifact availabilityを確認後、初めてRoute A/Bの係数dry-runとepoch-79 short pilotへ進みます。"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    args = parser.parse_args()
    if args.permutations < 1 or args.bootstrap_replicates < 1:
        parser.error("permutations and bootstrap-replicates must be positive")
    try:
        report = analyze(outcome_paths=DEFAULT_OUTCOME, feature_paths=DEFAULT_FEATURE, online_paths=DEFAULT_ONLINE, permutations=args.permutations, bootstrap_replicates=args.bootstrap_replicates)
    except NextEvidenceError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "ffnr-next-evidence-report.json"
    markdown_path = args.output / "ffnr-next-evidence-report.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "markdown": str(markdown_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
