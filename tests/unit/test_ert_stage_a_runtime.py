from __future__ import annotations

import json
from pathlib import Path

import pytest

from ard.analysis.ert_stage_a_runtime import StageARuntimeError, StageATreatment, _mask_from_overlay


def test_stage_a_treatment_requires_explicit_clean_wrong_mode() -> None:
    with pytest.raises(StageARuntimeError, match="clean-wrong"):
        StageATreatment(arm="CW1", mask_key="student_clean_wrong", kind="clean_wrong")
    assert StageATreatment(
        arm="CW1",
        mask_key="student_clean_wrong",
        kind="clean_wrong",
        beta_cleance=0.2,
        clean_wrong_mode="clean_ce_only",
    ).clean_wrong_mode == "clean_ce_only"


def test_stage_a_overlay_mask_keeps_stable_ids_and_class_counts(tmp_path: Path) -> None:
    path = tmp_path / "masks.json"
    path.write_text(
        json.dumps(
            {
                "anchor_epoch": 79,
                "masks": {
                    "s3_t1_q10": {"selected_ids": [2, 4], "selected_class_counts": {"0": 1, "1": 1}},
                },
            }
        ),
        encoding="utf-8",
    )
    mask = _mask_from_overlay(path, "s3_t1_q10")
    assert mask.selected_ids == frozenset({2, 4})
    assert mask.class_counts == {0: 1, 1: 1}
