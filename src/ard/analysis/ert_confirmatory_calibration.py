"""No-update sanity check for the rounded ERT confirmatory coefficient."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


class ConfirmatoryCalibrationError(RuntimeError):
    """Raised when the frozen calibration artifact cannot support the check."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha_and_dirty() -> tuple[str, bool]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout)
    return sha, dirty


def build_calibration_sanity(
    calibration_path: Path,
    *,
    beta_advce: float = 0.075,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if beta_advce != 0.075:
        raise ConfirmatoryCalibrationError("confirmatory beta_advce is preregistered exactly as 0.075")
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != "ert_stage_a_calibration_v1":
        raise ConfirmatoryCalibrationError("unexpected Stage A calibration contract")
    measurements = payload.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise ConfirmatoryCalibrationError("calibration measurements are missing")
    ratios: list[float] = []
    cosines: list[float] = []
    for measurement in measurements:
        if not isinstance(measurement, dict):
            raise ConfirmatoryCalibrationError("calibration measurement is not an object")
        base = float(measurement["base_adv_norm"])
        advce = float(measurement["adv_ce_norm"])
        cosine = float(measurement["adv_ce_cosine"])
        if not all(math.isfinite(value) for value in (base, advce, cosine)) or base <= 0 or advce < 0:
            raise ConfirmatoryCalibrationError("non-finite or zero calibration gradient")
        ratios.append(beta_advce * advce / base)
        cosines.append(cosine)
    ratios.sort()
    cosines.sort()
    mid = len(ratios) // 2
    median_ratio = ratios[mid] if len(ratios) % 2 else (ratios[mid - 1] + ratios[mid]) / 2
    median_cosine = cosines[mid] if len(cosines) % 2 else (cosines[mid - 1] + cosines[mid]) / 2
    source_sha, dirty = _git_sha_and_dirty()
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_confirmatory_t123_calibration_sanity_v1",
        "status": "complete_no_update",
        "beta_advce": beta_advce,
        "advkd_multiplier_t3": 0.5,
        "measurement_count": len(ratios),
        "median_scaled_advce_to_base_advkd_gradient_ratio": median_ratio,
        "median_advce_cosine": median_cosine,
        "ratio_min": ratios[0],
        "ratio_max": ratios[-1],
        "calibration_sha256": _sha256(calibration_path),
        "source_git_sha": source_sha,
        "source_dirty": dirty,
        "optimizer_step": False,
        "scheduler_step": False,
        "sample_state_mutation": False,
        "validation_metric_used": False,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
