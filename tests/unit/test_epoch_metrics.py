from __future__ import annotations

import pytest

from ard.analysis.epoch_metrics import EpochMetricsError, EpochMetricStore, epoch_trajectory_summary, merge_epoch_rows

pytestmark = pytest.mark.unit


def _row(epoch: int, *, pgd: float = 0.5, clean: float = 0.7) -> dict[str, float | int]:
    return {"epoch": epoch, "val_pgd_accuracy": pgd, "val_clean_accuracy": clean, "global_step": epoch + 1}


def test_atomic_store_is_idempotent_and_rejects_conflicting_resume_rows(tmp_path) -> None:
    store = EpochMetricStore(tmp_path / "epoch-metrics.jsonl")
    first = store.merge((_row(0), _row(1)))
    before = store.path.read_bytes()
    assert store.merge((_row(0),)) == first
    assert store.path.read_bytes() == before
    with pytest.raises(EpochMetricsError, match="conflicting"):
        store.merge((_row(1, pgd=0.4),))


def test_canonical_200_epoch_summary_has_late_mean_normalized_auc_and_slope() -> None:
    rows = [_row(epoch, pgd=epoch / 199, clean=1 - epoch / 398) for epoch in range(200)]
    summary = epoch_trajectory_summary(rows, expected_epochs=200)
    assert summary["epoch_metrics_complete"] is True
    assert summary["val_pgd_late_mean_epoch_150_199"] == pytest.approx(sum(range(150, 200)) / 50 / 199)
    assert summary["val_pgd_normalized_auc"] == pytest.approx(0.5)
    assert summary["val_pgd_slope_per_epoch"] == pytest.approx(1 / 199)
    assert summary["val_clean_slope_per_epoch"] == pytest.approx(-1 / 398)
    assert summary["val_pgd_mean_epoch_100_199"] == pytest.approx(sum(range(100, 200)) / 100 / 199)
    assert summary["val_pgd_mean_epoch_120_199"] == pytest.approx(sum(range(120, 200)) / 80 / 199)
    assert summary["val_pgd_mean_epoch_150_199"] == pytest.approx(sum(range(150, 200)) / 50 / 199)
    assert summary["val_pgd_normalized_auc_epoch_100_199"] == pytest.approx(149.5 / 199)
    assert summary["val_pgd_slope_epoch_120_199"] == pytest.approx(1 / 199)


def test_partial_trajectory_is_explicitly_not_canonical_complete() -> None:
    summary = epoch_trajectory_summary((_row(0), _row(2)), expected_epochs=200)
    assert summary["epoch_metrics_complete"] is False
    assert "val_pgd_normalized_auc" not in summary


def test_merge_function_requires_exact_duplicate_semantics() -> None:
    assert merge_epoch_rows((_row(0),), (_row(0),)) == [_row(0)]
