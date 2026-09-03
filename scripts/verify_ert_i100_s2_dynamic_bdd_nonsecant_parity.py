#!/usr/bin/env python3
"""Verify the S-BDD-only recovery delta on one registered batch per seed.

The frozen v1 calibration contains exact no-update measurements for the
Control RSLAD AdvKD component plus DPM and D-BDD on deterministic epoch-100
views.  This tool recreates only batch zero for each seed and refuses source
reuse if any non-secant measurement differs from that immutable reference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.prepare_ert_i100_s2_dynamic_bdd import _run
    from scripts.prepare_ert_i100_s2_rbp import sha256
except ModuleNotFoundError:  # direct ``python scripts/...`` invocation
    from prepare_ert_i100_s2_dynamic_bdd import _run
    from prepare_ert_i100_s2_rbp import sha256


MODES = ("pair_margin", "detached_boundary_distance")
FIELDS = (
    "base_advkd_norm",
    "pair_margin_norm",
    "pair_margin_active_count",
    "detached_boundary_distance_norm",
    "detached_boundary_distance_active_count",
)


def _reference_row(payload: dict[str, Any], run: str) -> dict[str, Any]:
    matches = [row for row in payload.get("measurements", []) if row.get("run") == run and row.get("batch") == 0]
    if len(matches) != 1:
        raise ValueError(f"{run}: frozen calibration lacks exactly one batch-zero reference")
    return matches[0]


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
        raise ValueError("reference is not the registered v1 dynamic-BDD calibration")
    runs = {
        "dev-1": (args.dev1_config, args.dev1_checkpoint, args.dev1_mask, args.dev1_replay),
        "dev-2": (args.dev2_config, args.dev2_checkpoint, args.dev2_mask, args.dev2_replay),
    }
    results: dict[str, Any] = {}
    for run, paths in runs.items():
        measured, inputs = _run(
            run,
            *paths,
            device=torch.device(args.device),
            modes=MODES,
            max_batches=1,
        )
        if len(measured) != 1:
            raise ValueError(f"{run}: expected exactly one fresh parity batch")
        observed = measured[0]
        expected = _reference_row(reference, run)
        checks: dict[str, dict[str, float | bool]] = {}
        for field in FIELDS:
            left, right = float(observed[field]), float(expected[field])
            equal = math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)
            checks[field] = {"observed": left, "reference": right, "match": equal}
            if not equal:
                raise ValueError(f"{run}: non-secant parity mismatch for {field}: {left} != {right}")
        results[run] = {"inputs": inputs, "checks": checks}
    output = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_dynamic_bdd_nonsecant_parity_v1",
        "status": "pass",
        "scope": "one deterministic epoch-100 batch per seed; Control AdvKD, DPM, and D-BDD only",
        "reference_calibration_sha256": sha256(args.reference_calibration),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_name(args.output.name + ".sha256").write_text(sha256(args.output) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
