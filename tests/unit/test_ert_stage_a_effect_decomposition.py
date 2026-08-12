from __future__ import annotations

import numpy as np

from ard.analysis.ert_stage_a_effect_decomposition import _bootstrap_ci, _paired_metrics


def test_weighted_identity_is_exact_for_paired_deltas() -> None:
    arrays = {
        "labels": np.asarray([0, 0, 1, 1]),
        "robust": np.asarray([1, 0, -1, 0]),
        "clean": np.asarray([0, 1, 0, -1]),
        "adv_margin": np.zeros(4),
        "clean_margin": np.zeros(4),
    }
    result = _paired_metrics(arrays)
    assert result["robust_accuracy_delta"] == 0.0
    assert result["clean_accuracy_delta"] == 0.0


def test_class_stratified_bootstrap_is_deterministic() -> None:
    arrays = {
        "labels": np.asarray([0, 0, 1, 1]),
        "robust": np.asarray([1, 0, -1, 0]),
        "clean": np.asarray([0, 1, 0, -1]),
        "adv_margin": np.zeros(4),
        "clean_margin": np.zeros(4),
    }
    first = _bootstrap_ci(arrays, seed=3)
    second = _bootstrap_ci(arrays, seed=3)
    assert first == second
