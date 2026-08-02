"""Explicit-cohort W&B robust-overfitting history access with a disk cache."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml

from ard.analysis import epoch_metrics
from ard.analysis.epoch_metrics import EpochMetricsError, epoch_range_summary


class WandbHistoryError(ValueError):
    """A W&B cohort or its epoch trajectory cannot prove exact coverage."""


class _Artifact(Protocol):
    type: str

    def download(self, root: str | None = None) -> str: ...


class _Run(Protocol):
    id: str
    state: str
    summary: Mapping[str, Any]

    def logged_artifacts(self) -> Iterable[_Artifact]: ...

    def history(self, **kwargs: Any) -> Any: ...


SUMMARY_KEYS = (
    "_step",
    "best_metric",
    "best_epoch",
    "best_clean_accuracy",
    "best_pgd_accuracy",
    "last_clean_accuracy",
    "last_pgd_accuracy",
    "robust_overfit_gap",
    "epoch_metrics_complete",
    "val_pgd_mean_epoch_100_199",
    "val_pgd_mean_epoch_120_199",
    "val_pgd_mean_epoch_150_199",
    "val_pgd_normalized_auc_epoch_100_199",
    "val_pgd_slope_epoch_120_199",
)
TRAJECTORY_SOURCES = frozenset({"auto", "legacy_history", "epoch_metrics_artifact"})


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary_snapshot(summary: Mapping[str, Any]) -> dict[str, int | float | str | bool | None]:
    """Keep only bounded JSON-native scalar summary evidence.

    W&B may expose nested ``SummarySubDict`` values that cannot be encoded by
    JSON.  They are neither trajectory inputs nor a stable cache fingerprint.
    """
    snapshot: dict[str, int | float | str | bool | None] = {}
    for key in SUMMARY_KEYS:
        if key not in summary:
            continue
        value = summary[key]
        if value is None or isinstance(value, (bool, str, int)):
            snapshot[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            snapshot[key] = value
    return snapshot


def load_cohort(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WandbHistoryError("cohort registry is unreadable") from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("runs"), list):
        raise WandbHistoryError("cohort registry must contain a top-level runs list")
    if value.get("metric") != "val_pgd_accuracy":
        raise WandbHistoryError("cohort metric must be exactly val_pgd_accuracy")
    configured_epochs = value.get("expected_epochs")
    if isinstance(configured_epochs, bool) or not isinstance(configured_epochs, int) or configured_epochs < 2:
        raise WandbHistoryError("cohort expected_epochs must be an integer of at least two")
    top_level_source = value.get("trajectory_source", "auto")
    if not isinstance(top_level_source, str) or top_level_source not in TRAJECTORY_SOURCES:
        raise WandbHistoryError("cohort trajectory_source must be auto, legacy_history, or epoch_metrics_artifact")
    ids: list[str] = []
    for entry in value["runs"]:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("run_id"), str) or not entry["run_id"]:
            raise WandbHistoryError("each cohort run requires an explicit run_id")
        for field in ("epoch_start", "epoch_end"):
            candidate = entry.get(field)
            if candidate is not None and (
                isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 0
            ):
                raise WandbHistoryError(f"run {entry['run_id']} {field} must be a non-negative integer")
        source = entry.get("trajectory_source", top_level_source)
        if not isinstance(source, str) or source not in TRAJECTORY_SOURCES:
            raise WandbHistoryError(
                f"run {entry['run_id']} trajectory_source must be auto, legacy_history, or epoch_metrics_artifact"
            )
        ids.append(entry["run_id"])
    if not ids or len(ids) != len(set(ids)):
        raise WandbHistoryError("cohort must contain non-empty unique explicit run IDs")
    return dict(value), tuple(ids)


def _cache_path(cache_dir: Path, *, cohort_bytes: bytes, expected_epochs: int) -> Path:
    source_hash = sha256_file(Path(__file__))
    fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "cohort_sha256": hashlib.sha256(cohort_bytes).hexdigest(),
                "source_sha256": source_hash,
                "epoch_metrics_contract_sha256": _epoch_metrics_contract_fingerprint(),
                "expected_epochs": expected_epochs,
            }
        )
    ).hexdigest()
    return cache_dir / f"wandb-ro-{fingerprint}.json"


def _epoch_metrics_contract_fingerprint() -> str:
    module_path = getattr(epoch_metrics, "__file__", None)
    if not isinstance(module_path, str):  # pragma: no cover - import machinery contract
        raise WandbHistoryError("epoch metrics contract source is unavailable")
    return sha256_file(Path(module_path).resolve())


def _rows_from_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise WandbHistoryError("W&B epoch artifact analysis requires pyarrow") from exc
    try:
        return [dict(row) for row in pq.read_table(path).to_pylist()]
    except Exception as exc:  # pragma: no cover - pyarrow version-specific errors
        raise WandbHistoryError("W&B epoch artifact Parquet is unreadable") from exc


def _artifact_rows(run: _Run, *, cache_dir: Path) -> list[dict[str, Any]] | None:
    for artifact in run.logged_artifacts():
        if getattr(artifact, "type", None) != "epoch-metrics":
            continue
        root = Path(artifact.download(root=str(cache_dir / "artifacts" / run.id))).resolve()
        candidates = sorted(root.rglob("epoch-metrics.parquet"))
        if len(candidates) != 1:
            raise WandbHistoryError("epoch-metrics artifact must contain exactly one epoch-metrics.parquet")
        return _rows_from_parquet(candidates[0])
    return None


def _history_rows(run: _Run, *, run_id: str, epoch_start: int, epoch_end: int) -> list[dict[str, Any]]:
    """Legacy fallback; reject sampled/duplicate/incomplete history before use."""
    history = run.history(
        keys=["epoch", "val_pgd_accuracy", "val_clean_accuracy"],
        x_axis="epoch",
        samples=max((epoch_end - epoch_start + 1) * 4, 1000),
        pandas=False,
    )
    if hasattr(history, "to_dict"):
        history = history.to_dict("records")
    if not isinstance(history, Sequence):
        raise WandbHistoryError("legacy W&B history is not a row sequence")
    by_epoch: dict[int, dict[str, Any]] = {}
    for raw in history:
        if not isinstance(raw, Mapping) or "epoch" not in raw:
            continue
        epoch = raw["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise WandbHistoryError(f"run {run_id} range {epoch_start}..{epoch_end}: epoch must be an integer")
        row = dict(raw)
        if epoch in by_epoch:
            raise WandbHistoryError(
                f"run {run_id} range {epoch_start}..{epoch_end}: duplicate epoch coverage (count={len(by_epoch)})"
            )
        by_epoch[epoch] = row
    expected = list(range(epoch_start, epoch_end + 1))
    if sorted(by_epoch) != expected:
        raise WandbHistoryError(
            f"run {run_id} range {epoch_start}..{epoch_end}: exact unique coverage failed "
            f"(count={len(by_epoch)}, expected={len(expected)})"
        )
    return [by_epoch[epoch] for epoch in expected]


def _requested_epoch_range(entry: Mapping[str, Any], *, expected_epochs: int) -> tuple[int, int]:
    run_id = entry["run_id"]
    assert isinstance(run_id, str)
    epoch_start = entry.get("epoch_start", 0)
    epoch_end = entry.get("epoch_end", expected_epochs - 1)
    assert isinstance(epoch_start, int) and not isinstance(epoch_start, bool)
    assert isinstance(epoch_end, int) and not isinstance(epoch_end, bool)
    if epoch_end < epoch_start or epoch_end >= expected_epochs:
        raise WandbHistoryError(
            f"run {run_id} requested range {epoch_start}..{epoch_end} is outside 0..{expected_epochs - 1}"
        )
    return epoch_start, epoch_end


def _trajectory_source(entry: Mapping[str, Any], *, cohort: Mapping[str, Any]) -> str:
    source = entry.get("trajectory_source", cohort.get("trajectory_source", "auto"))
    assert isinstance(source, str) and source in TRAJECTORY_SOURCES
    return source


def _trajectory_summary(
    rows: Sequence[Mapping[str, Any]], *, run_id: str, epoch_start: int, epoch_end: int
) -> dict[str, int | float | bool]:
    try:
        return epoch_range_summary(rows, epoch_start=epoch_start, epoch_end=epoch_end)
    except EpochMetricsError as exc:
        raise WandbHistoryError(f"run {run_id} range {epoch_start}..{epoch_end} count={len(rows)}: {exc}") from exc


def analyze_cohort(
    *,
    cohort_path: Path,
    cache_dir: Path,
    fetch_run: Callable[[str], _Run],
    expected_epochs: int = 200,
    force: bool = False,
) -> dict[str, Any]:
    """Analyze only an explicit cohort; a valid cache performs no API requests."""
    cohort, run_ids = load_cohort(cohort_path)
    cache = _cache_path(cache_dir, cohort_bytes=cohort_path.read_bytes(), expected_epochs=expected_epochs)
    if cache.is_file() and not force:
        try:
            result = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WandbHistoryError("history cache is unreadable") from exc
        if isinstance(result, Mapping):
            fingerprints = result.get("run_fingerprints")
            if (
                not isinstance(fingerprints, list)
                or len(fingerprints) != len(run_ids)
                or any(not isinstance(item, Mapping) or item.get("state") != "finished" for item in fingerprints)
            ):
                raise WandbHistoryError("history cache is not a finished-run success cache")
            return {**dict(result), "cached": True}
        raise WandbHistoryError("history cache must be a mapping")
    reports: list[dict[str, Any]] = []
    run_fingerprints: list[dict[str, Any]] = []
    entries = cohort["runs"]
    assert isinstance(entries, list)
    for entry, run_id in zip(entries, run_ids, strict=True):
        assert isinstance(entry, Mapping)
        epoch_start, epoch_end = _requested_epoch_range(entry, expected_epochs=expected_epochs)
        requested_source = _trajectory_source(entry, cohort=cohort)
        run = fetch_run(run_id)
        if run.id != run_id:
            raise WandbHistoryError("W&B client returned a run with the wrong explicit ID")
        if getattr(run, "state", None) != "finished":
            raise WandbHistoryError("unfinished W&B runs cannot enter or reuse the success history cache")
        summary = summary_snapshot(run.summary)  # summary-first, before an artifact/history request.
        run_fingerprints.append(
            {
                "run_id": run_id,
                "state": "finished",
                "summary_sha256": hashlib.sha256(_canonical_json(summary)).hexdigest(),
                "summary_last_step": summary.get("_step"),
            }
        )
        rows = None
        source = "epoch_metrics_artifact"
        if requested_source != "legacy_history":
            rows = _artifact_rows(run, cache_dir=cache_dir)
        if rows is None:
            if requested_source == "epoch_metrics_artifact":
                raise WandbHistoryError(
                    f"run {run_id} range {epoch_start}..{epoch_end} requires an epoch-metrics artifact"
                )
            rows = _history_rows(run, run_id=run_id, epoch_start=epoch_start, epoch_end=epoch_end)
            source = "legacy_history_exact_coverage"
        reports.append(
            {
                "run_id": run_id,
                "summary_first": summary,
                "trajectory_source": source,
                "trajectory": _trajectory_summary(rows, run_id=run_id, epoch_start=epoch_start, epoch_end=epoch_end),
            }
        )
    result = {
        "schema_version": 1,
        "cohort": cohort,
        "expected_epochs": expected_epochs,
        "run_fingerprints": run_fingerprints,
        "runs": reports,
        "cached": False,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(result))
    temporary.replace(cache)
    return result
