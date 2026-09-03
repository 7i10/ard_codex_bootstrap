from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[2] / "scripts/analysis/ert_i100_s2_boundary_geometry.py"
    spec = importlib.util.spec_from_file_location("ert_i100_s2_boundary_geometry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load geometry audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auc_and_average_precision_handle_ties_deterministically() -> None:
    module = _module()
    scores = np.asarray([0.5, 0.5, 0.1, 0.9])
    labels = np.asarray([1, 0, 0, 1])
    assert module._auc(scores, labels) == 0.875
    assert module._average_precision(scores, labels) == 1.0


def test_cells_use_pre_treatment_medians_and_partition_ids() -> None:
    module = _module()
    rows = {sid: {"normal_cosine": float(sid), "student_distance_inf": float(sid)} for sid in range(4)}
    cells = module._cells(rows, {0: False, 1: True, 2: False, 3: True})
    assert cells["median_normal_cosine"] == 1.5
    assert cells["median_student_distance_inf"] == 1.5
    assert sum(value["n"] for value in cells["cells"].values()) == 4


def test_ridge_cross_seed_prediction_is_probability_bounded() -> None:
    module = _module()
    train_x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    train_y = np.asarray([0, 0, 1, 1])
    test_x = np.asarray([[-1.0], [4.0]])
    probabilities = module._ridge_logistic(train_x, train_y, test_x)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
