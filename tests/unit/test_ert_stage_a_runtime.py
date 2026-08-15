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


def test_confirmatory_treatment_coefficients_are_explicit() -> None:
    t1 = StageATreatment(arm="T1WCONF", mask_key="s3_t1_q10", kind="advce", beta_advce=0.075)
    assert t1.beta_advce == 0.075
    t3 = StageATreatment(
        arm="T3LP05CONF",
        mask_key="s3_t3_q10",
        kind="advkd_advce",
        beta_advce=0.075,
        advkd_multiplier=0.5,
    )
    assert t3.advkd_multiplier == 0.5


def test_horizon_contract_rejects_duplicate_or_pre_parent_epochs() -> None:
    from ard.analysis.ert_stage_a_runtime import StageARuntimeError, _validate_horizons

    with pytest.raises(StageARuntimeError, match="horizon"):
        _validate_horizons((79, 84), 94)
    with pytest.raises(StageARuntimeError, match="unique"):
        _validate_horizons((84, 84), 94)
    _validate_horizons((84, 89, 94), 94)
