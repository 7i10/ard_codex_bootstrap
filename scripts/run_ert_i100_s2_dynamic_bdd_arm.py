#!/usr/bin/env python3
"""Run one I100 S2×T1 dynamic boundary-distance continuation."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from ard.cli.ert_stage_a_runtime import main as runtime_main

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(
    os.environ.get(
        "ARD_STAGEWISE_RUN_ROOT",
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1",
    )
).expanduser()
CALIBRATIONS = {
    "dpm": ROOT / "docs/experiments/ert_rslad_i100_s2_dynamic_bdd_calibration_v1.json",
    "dbdd": ROOT / "docs/experiments/ert_rslad_i100_s2_dynamic_bdd_calibration_v1.json",
    "sbdd": ROOT / "docs/experiments/ert_rslad_i100_s2_secant_boundary_distance_calibration_v2.json",
}
PARENTS = {
    "dev-1": {
        "config": RUN_ROOT / "idbh-s100-s1/resolved_config.yaml",
        "checkpoint": RUN_ROOT / "seed1/s100/epoch-100.pt",
        "sha": "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
        "mask": ROOT / "docs/experiments/ert_rslad_i100_s2_rbp_masks_dev1_v1.json",
    },
    "dev-2": {
        "config": RUN_ROOT / "idbh-s100-s2/resolved_config.yaml",
        "checkpoint": RUN_ROOT / "seed2/s100/epoch-100.pt",
        "sha": "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
        "mask": ROOT / "docs/experiments/ert_rslad_i100_s2_rbp_masks_dev2_v1.json",
    },
}
ARMS = {
    "control": ("I100_CONTROL", None),
    "dpm": ("I100_S2T1_DYNAMIC_PAIR_MARGIN", "pair_margin"),
    "dbdd": ("I100_S2T1_DETACHED_BOUNDARY_DISTANCE", "detached_boundary_distance"),
    "sbdd": ("I100_S2T1_SECANT_BOUNDARY_DISTANCE", "secant_boundary_distance"),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", choices=tuple(PARENTS), required=True)
    p.add_argument("--arm", choices=tuple(ARMS), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--expected-source-sha")
    args = p.parse_args()
    if args.expected_source_sha is not None:
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if actual != args.expected_source_sha:
            raise SystemExit(f"source SHA mismatch: expected {args.expected_source_sha}, got {actual}")
    parent = PARENTS[args.seed]
    arm, mode = ARMS[args.arm]
    calibration_path = CALIBRATIONS.get(args.arm, CALIBRATIONS["dpm"])
    values = [
        "--parent-config",
        str(parent["config"]),
        "--parent-checkpoint",
        str(parent["checkpoint"]),
        "--calibration",
        str(calibration_path),
        "--output",
        str(args.output),
        "--epochs",
        "115",
        "--horizon-epochs",
        "104",
        "109",
        "114",
        "--run-namespace",
        "ert-i100-s2-dynamic-bdd-v1",
        "--resume-epoch",
        "99",
        "--mask-anchor-epoch",
        "99",
        "--expected-parent-sha256",
        parent["sha"],
        "--force-sample-keyed-attack",
        "--device",
        args.device,
    ]
    if args.arm == "control":
        values += ["--arm", arm, "--kind", "baseline"]
    else:
        values += [
            "--arm",
            arm,
            "--kind",
            "broad",
            "--mask",
            str(parent["mask"]),
            "--mask-key",
            "s2_t1",
            "--boundary-intervention",
            mode,
            "--boundary-coefficient",
            "__CALIBRATION__",
            "--boundary-epsilon",
            "1e-12",
        ]
        import json

        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        values[values.index("__CALIBRATION__")] = str(calibration["coefficients"][mode])
    return runtime_main(values)


if __name__ == "__main__":
    raise SystemExit(main())
