#!/usr/bin/env python3
"""Freeze the corrected S-BDD coefficient without recalibrating other arms.

This recovery-only calibration reuses the registered e99 parents, fixed S2xT1
masks, epoch-100 augmentation view, and sample-keyed KL-PGD10 attack.  It
performs no optimizer, scheduler, state-store, or checkpoint mutation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.prepare_ert_i100_s2_dynamic_bdd import BOUNDARY_EPSILON, _run
    from scripts.prepare_ert_i100_s2_rbp import sha256
except ModuleNotFoundError:  # direct ``python scripts/...`` invocation
    from prepare_ert_i100_s2_dynamic_bdd import BOUNDARY_EPSILON, _run
    from prepare_ert_i100_s2_rbp import sha256


MODE = "secant_boundary_distance"
TARGET_RATIO = 0.25
EXPECTED_BATCHES_PER_SEED = 8
EXPECTED_MEASUREMENT_COUNT = 2 * EXPECTED_BATCHES_PER_SEED


def _median(values: torch.Tensor) -> torch.Tensor:
    """Use one explicit median convention for freeze and achieved checks.

    ``torch.median`` selects the lower middle value for an even number of
    observations, whereas the calibration target is a median of the achieved
    *ratio* distribution.  A linear 0.5 quantile is unambiguous for both
    operations and makes the frozen target reproducible.
    """
    return torch.quantile(values, 0.5)


def _freeze_coefficient(base_norms: torch.Tensor, secant_norms: torch.Tensor) -> float:
    """Freeze alpha so the same pooled median estimator reaches the target."""
    ratios_at_one = secant_norms / base_norms
    return float(TARGET_RATIO / _median(ratios_at_one).item())


def _summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "min": float(tensor.min().item()),
        "median": float(_median(tensor).item()),
        "max": float(tensor.max().item()),
        "iqr": float((torch.quantile(tensor, 0.75) - torch.quantile(tensor, 0.25)).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for run in ("dev1", "dev2"):
        parser.add_argument(f"--{run}-config", type=Path, required=True)
        parser.add_argument(f"--{run}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{run}-mask", type=Path, required=True)
        parser.add_argument(f"--{run}-replay", type=Path, required=True)
    parser.add_argument("--reference-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    reference = json.loads(args.reference_calibration.read_text(encoding="utf-8"))
    if reference.get("contract") != "ert_rslad_i100_s2_dynamic_bdd_calibration_v1":
        raise ValueError("reference calibration is not the registered v1 dynamic-BDD artifact")
    specs = {
        "dev-1": (args.dev1_config, args.dev1_checkpoint, args.dev1_mask, args.dev1_replay),
        "dev-2": (args.dev2_config, args.dev2_checkpoint, args.dev2_mask, args.dev2_replay),
    }
    measurements: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    for run, values in specs.items():
        run_measurements, metadata = _run(
            run,
            *values,
            device=torch.device(args.device),
            modes=(MODE,),
        )
        measurements.extend(run_measurements)
        inputs[run] = metadata
        if len(run_measurements) != EXPECTED_BATCHES_PER_SEED:
            raise ValueError(
                f"{run}: expected {EXPECTED_BATCHES_PER_SEED} deterministic calibration batches, "
                f"got {len(run_measurements)}"
            )
    if len(measurements) != EXPECTED_MEASUREMENT_COUNT:
        raise ValueError(
            f"expected {EXPECTED_MEASUREMENT_COUNT} pooled calibration measurements, got {len(measurements)}"
        )
    base = torch.tensor([row["base_advkd_norm"] for row in measurements], dtype=torch.float64)
    secant = torch.tensor([row[f"{MODE}_norm"] for row in measurements], dtype=torch.float64)
    if not bool(torch.isfinite(base).all()) or not bool(torch.isfinite(secant).all()) or bool((secant <= 0).any()):
        raise ValueError("corrected S-BDD calibration produced non-finite or zero norms")
    coefficient = _freeze_coefficient(base, secant)
    achieved = [float(coefficient * row[f"{MODE}_norm"] / row["base_advkd_norm"]) for row in measurements]
    achieved_median = float(_median(torch.tensor(achieved, dtype=torch.float64)).item())
    if not math.isclose(achieved_median, TARGET_RATIO, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(
            f"pooled achieved median gradient ratio must equal {TARGET_RATIO}, got {achieved_median}"
        )
    result = {
        "schema_version": 2,
        "contract": "ert_rslad_i100_s2_secant_boundary_distance_calibration_v2",
        "status": "complete_no_update",
        "parent_epoch": 99,
        "calibration_view_epoch": 100,
        "target_gradient_ratio": TARGET_RATIO,
        "median_estimator": "torch.quantile(q=0.5, interpolation=linear)",
        "boundary_epsilon": BOUNDARY_EPSILON,
        "formula_version": "student_parameter_graph_v2",
        "formula": (
            "0.5 * relu(dT_sec - dS_sec)^2; "
            "dS_sec=mS_adv/(abs(mS_adv-mS_clean)/(rho+eps)+eps); "
            "Student qS is not detached"
        ),
        "reference_v1_calibration_sha256": sha256(args.reference_calibration),
        "inputs": inputs,
        "coefficients": {MODE: coefficient},
        "measurements": measurements,
        "achieved_ratios": {MODE: achieved},
        "achieved_ratio_summary": {MODE: _summary(achieved)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    digest = sha256(args.output)
    args.output.with_name(args.output.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    result["artifact_sha256"] = digest
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
