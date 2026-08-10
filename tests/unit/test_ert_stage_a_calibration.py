from __future__ import annotations

import json
from pathlib import Path

import pytest

from ard.analysis.ert_stage_a_calibration import (
    StageACalibrationError,
    _class_stratified_ids,
    _state_masks,
)


def test_stage_a_masks_require_registered_anchor_and_explicit_cohorts(tmp_path: Path) -> None:
    path = tmp_path / "mask.json"
    path.write_text(
        json.dumps(
            {
                "contract": "ert_state_overlay_v1",
                "anchor_epoch": 79,
                "masks": {
                    "s3_t1_q10": {"selected_ids": [0, 2]},
                    "s3_t2_q10": {"selected_ids": [1]},
                    "s3_t3_q10": {"selected_ids": [3]},
                    "student_clean_wrong": {"selected_ids": [4]},
                },
            }
        ),
        encoding="utf-8",
    )
    assert _state_masks(path) == {"s3_t1": {0, 2}, "s3_t2": {1}, "s3_t3": {3}, "student_clean_wrong": {4}}

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["anchor_epoch"] = 80
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StageACalibrationError, match="epoch-79"):
        _state_masks(path)


def test_stage_a_calibration_ids_are_deterministic_and_class_stratified() -> None:
    ids = set(range(12))
    labels = {sample_id: sample_id % 3 for sample_id in ids}
    first = _class_stratified_ids(ids, labels, limit=6, seed=17)
    second = _class_stratified_ids(ids, labels, limit=6, seed=17)
    assert first == second
    assert len(first) == 6
    assert {labels[sample_id] for sample_id in first} == {0, 1, 2}
