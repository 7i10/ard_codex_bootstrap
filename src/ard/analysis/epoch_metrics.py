"""Atomic canonical epoch trajectories for tracking-only RO analysis."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ard.analysis.sample_stats import write_sample_parquet


class EpochMetricsError(ValueError):
    """Epoch metric rows cannot establish one unambiguous trajectory."""


CANONICAL_EPOCHS = 200
LATE_WINDOW = 50


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    epoch = value.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise EpochMetricsError("epoch metrics require a non-negative integer epoch")
    for key, candidate in value.items():
        if isinstance(candidate, float) and not math.isfinite(candidate):
            raise EpochMetricsError(f"epoch metric {key} is non-finite")
    return value


def load_epoch_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise EpochMetricsError("canonical epoch metric store is unreadable") from exc
    if any(not isinstance(row, Mapping) for row in rows):
        raise EpochMetricsError("canonical epoch metric store rows must be mappings")
    return merge_epoch_rows((), (dict(row) for row in rows))


def merge_epoch_rows(prior: Iterable[Mapping[str, Any]], incoming: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge exact rows; duplicate epochs must be byte-equivalent logically."""
    merged: dict[int, dict[str, Any]] = {}
    for raw in (*tuple(prior), *tuple(incoming)):
        row = _normalize_row(raw)
        epoch = int(row["epoch"])
        existing = merged.get(epoch)
        if existing is not None and _canonical_json(existing) != _canonical_json(row):
            raise EpochMetricsError(f"conflicting duplicate epoch metric row: {epoch}")
        merged[epoch] = row
    return [merged[epoch] for epoch in sorted(merged)]


class EpochMetricStore:
    """Atomic JSONL store whose rows remain canonical across epoch resumes."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def rows(self) -> list[dict[str, Any]]:
        return load_epoch_metrics(self.path)

    def merge(self, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        merged = merge_epoch_rows(self.rows(), rows)
        payload = "".join(_canonical_json(row) + "\n" for row in merged)
        if self.path.is_file() and self.path.read_text(encoding="utf-8") == payload:
            return merged
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, self.path)
        return merged

    def merge_tracker_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return self.rows()
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        except (OSError, json.JSONDecodeError) as exc:
            raise EpochMetricsError("tracker metrics JSONL is unreadable") from exc
        epoch_rows = [row for row in rows if isinstance(row, Mapping) and "epoch" in row]
        return self.merge(epoch_rows)


def write_epoch_metrics_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    if not rows:
        raise EpochMetricsError("cannot publish an empty epoch metric artifact")
    return write_sample_parquet(rows, path)


def _normalized_auc(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise EpochMetricsError("normalized AUC requires at least two epoch values")
    return (values[0] / 2.0 + sum(values[1:-1]) + values[-1] / 2.0) / (len(values) - 1)


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise EpochMetricsError("trajectory slope requires at least two epoch values")
    mean_epoch = (len(values) - 1) / 2.0
    denominator = sum((index - mean_epoch) ** 2 for index in range(len(values)))
    return (
        sum((index - mean_epoch) * (value - sum(values) / len(values)) for index, value in enumerate(values))
        / denominator
    )


def _metric_values(rows: Sequence[Mapping[str, Any]], *, metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise EpochMetricsError(f"complete canonical trajectory lacks finite {metric}")
        values.append(float(value))
    return values


def epoch_range_summary(
    rows: Sequence[Mapping[str, Any]], *, epoch_start: int, epoch_end: int
) -> dict[str, int | float | bool]:
    """Summarize one exact inclusive epoch range for read-only trajectory analysis.

    This is intentionally separate from ``epoch_trajectory_summary``: the latter
    is the stable training-run summary contract, while continuation analysis may
    start from a checkpoint after epoch zero.
    """
    if isinstance(epoch_start, bool) or not isinstance(epoch_start, int) or epoch_start < 0:
        raise EpochMetricsError("epoch range start must be a non-negative integer")
    if isinstance(epoch_end, bool) or not isinstance(epoch_end, int) or epoch_end < epoch_start:
        raise EpochMetricsError("epoch range end must be an integer no smaller than start")
    normalized = merge_epoch_rows((), rows)
    epochs = [int(row["epoch"]) for row in normalized]
    expected = list(range(epoch_start, epoch_end + 1))
    if epochs != expected:
        raise EpochMetricsError(
            f"epoch metrics lack exact requested coverage {epoch_start}..{epoch_end}: "
            f"count={len(epochs)} expected={len(expected)}"
        )

    pgd = _metric_values(normalized, metric="val_pgd_accuracy")
    summary: dict[str, int | float | bool] = {
        "trajectory_requested_epoch_start": epoch_start,
        "trajectory_requested_epoch_end": epoch_end,
        "trajectory_first_epoch": epochs[0],
        "trajectory_last_epoch": epochs[-1],
        "trajectory_epoch_count": len(epochs),
        "trajectory_complete": True,
        "val_pgd_best_epoch": epoch_start + max(range(len(pgd)), key=pgd.__getitem__),
        "val_pgd_best_accuracy": max(pgd),
        "val_pgd_last_accuracy": pgd[-1],
        "val_pgd_robust_overfit_gap": max(pgd) - pgd[-1],
        "val_pgd_normalized_auc_requested_range": _normalized_auc(pgd),
    }
    for start, end in ((100, 199), (120, 199), (150, 199)):
        if epoch_start <= start and end <= epoch_end:
            values = pgd[start - epoch_start : end - epoch_start + 1]
            summary[f"val_pgd_mean_epoch_{start}_{end}"] = sum(values) / len(values)
            if (start, end) == (100, 199):
                summary["val_pgd_normalized_auc_epoch_100_199"] = _normalized_auc(values)
            if (start, end) == (120, 199):
                summary["val_pgd_slope_epoch_120_199"] = _slope(values)
    return summary


def epoch_trajectory_summary(rows: Sequence[Mapping[str, Any]], *, expected_epochs: int) -> dict[str, Any]:
    if expected_epochs < 1:
        raise EpochMetricsError("expected epoch count must be positive")
    normalized = merge_epoch_rows((), rows)
    epochs = [int(row["epoch"]) for row in normalized]
    expected = list(range(expected_epochs))
    complete = epochs == expected
    summary: dict[str, Any] = {
        "epoch_metrics_recorded_epochs": len(epochs),
        "epoch_metrics_expected_epochs": expected_epochs,
        "epoch_metrics_complete": complete,
        "epoch_metrics_source": "local_canonical_epoch_rows",
    }
    if expected_epochs != CANONICAL_EPOCHS or not complete:
        return summary

    for metric, prefix in (("val_pgd_accuracy", "val_pgd"), ("val_clean_accuracy", "val_clean")):
        values = _metric_values(normalized, metric=metric)
        late = values[-LATE_WINDOW:]
        summary.update(
            {
                f"{prefix}_late_mean_epoch_150_199": sum(late) / len(late),
                f"{prefix}_normalized_auc": _normalized_auc(values),
                f"{prefix}_slope_per_epoch": _slope(values),
            }
        )
        if prefix == "val_pgd":
            summary.update(
                {
                    "val_pgd_mean_epoch_100_199": sum(values[100:]) / 100,
                    "val_pgd_mean_epoch_120_199": sum(values[120:]) / 80,
                    "val_pgd_mean_epoch_150_199": sum(values[150:]) / 50,
                    "val_pgd_normalized_auc_epoch_100_199": _normalized_auc(values[100:]),
                    "val_pgd_slope_epoch_120_199": _slope(values[120:]),
                }
            )
    return summary
