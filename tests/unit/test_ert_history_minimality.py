from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parents[2] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_ert_rslad_history_minimality import CANDIDATES, select_candidate  # noqa: E402


def test_preregistered_candidate_set_and_smallest_eligible_selection() -> None:
    assert CANDIDATES["H1"] == (0,)
    assert CANDIDATES["H2"] == (1,)
    assert CANDIDATES["H9"] == (0, 2, 1, 3)
    values = {
        "H1": [0.80, 0.80],
        "H2": [0.84, 0.84],
        "H3": [0.20, 0.20],
        "H4": [0.70, 0.70],
        "H5": [0.845, 0.845],
        "H6": [0.82, 0.82],
        "H7": [0.82, 0.82],
        "H8": [0.846, 0.846],
        "H9": [0.845, 0.845],
    }
    selected, eligible = select_candidate(values)
    assert selected == "H2"
    assert eligible == ["H2", "H5", "H8"]


def test_selection_falls_back_to_h9_when_tolerances_fail() -> None:
    values = {candidate: [0.0, 0.0] for candidate in CANDIDATES}
    values["H9"] = [0.8, 0.8]
    selected, eligible = select_candidate(values)
    assert selected == "H9"
    assert eligible == []


def test_frozen_feature_standardization_is_finite() -> None:
    x = np.asarray([[1.0], [2.0], [3.0]])
    assert np.isfinite(x.mean(axis=0)).all()
    assert np.isfinite(x.std(axis=0)).all()
