from __future__ import annotations

# ruff: noqa: E501
import numpy as np
import pytest

from scripts.analyze_ffnr_next_evidence import _agreement, _metric_delta, _permutation_null


def test_agreement_reports_fixed_count_statistics() -> None:
    left = np.array([True, True, False, False])
    right = np.array([True, False, True, False])
    frequencies = np.array([1.0, 2 / 3, 1 / 3, 0.0])
    report = _agreement(left, right, frequencies, frequencies[::-1])
    assert report["intersection"] == 1
    assert report["union"] == 3
    assert report["raw_agreement_rate"] == 0.5


def test_hypergeometric_null_is_reproducible() -> None:
    left = np.array([True, True, False, False, False, False])
    right = np.array([True, False, True, False, False, False])
    first = _permutation_null(left, right, permutations=100, seed=17)
    second = _permutation_null(left, right, permutations=100, seed=17)
    assert first == second
    assert first["null_sampler"].startswith("hypergeometric_equivalent")


def test_metric_delta_direction_is_explicit() -> None:
    candidate = {"auroc": 0.8, "auprc": 0.7, "log_loss": 0.2, "brier": 0.1}
    baseline = {"auroc": 0.75, "auprc": 0.65, "log_loss": 0.25, "brier": 0.12}
    assert _metric_delta(candidate, baseline) == pytest.approx(
        {"delta_auroc": 0.05, "delta_auprc": 0.05, "delta_log_loss": -0.05, "delta_brier": -0.02}
    )
