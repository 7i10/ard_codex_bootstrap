from __future__ import annotations

import json
from pathlib import Path

import pytest

from ard.analysis.ert_confirmatory_calibration import (
    ConfirmatoryCalibrationError,
    build_calibration_sanity,
)


def _calibration(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "contract": "ert_stage_a_calibration_v1",
                "measurements": [
                    {"base_adv_norm": 2.0, "adv_ce_norm": 4.0, "adv_ce_cosine": 0.5},
                    {"base_adv_norm": 1.0, "adv_ce_norm": 2.0, "adv_ce_cosine": 0.75},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_confirmatory_sanity_freezes_rounded_beta_without_update(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    _calibration(calibration)
    result = build_calibration_sanity(calibration)
    assert result["beta_advce"] == 0.075
    assert result["median_scaled_advce_to_base_advkd_gradient_ratio"] == pytest.approx(0.15)
    assert result["optimizer_step"] is False
    assert result["sample_state_mutation"] is False


def test_confirmatory_sanity_rejects_coefficient_tuning(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    _calibration(calibration)
    with pytest.raises(ConfirmatoryCalibrationError, match="0.075"):
        build_calibration_sanity(calibration, beta_advce=0.07)
