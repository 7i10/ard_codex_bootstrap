#!/usr/bin/env python3
"""Launch one immutable I100 S2 boundary-preservation continuation arm."""

from __future__ import annotations

import argparse
from pathlib import Path

from ard.cli.ert_stage_a_runtime import main as runtime_main

ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = Path("/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1")
CALIBRATION = ROOT / "docs/experiments/ert_rslad_i100_s2_rbp_calibration_v1.json"
PARENTS = {
    "dev-1": (
        PARENT_ROOT / "idbh-s100-s1/resolved_config.yaml",
        PARENT_ROOT / "seed1/s100/epoch-100.pt",
        "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
        ROOT / "docs/experiments/ert_rslad_i100_s2_rbp_masks_dev1_v1.json",
        0.04177670180797577,
    ),
    "dev-2": (
        PARENT_ROOT / "idbh-s100-s2/resolved_config.yaml",
        PARENT_ROOT / "seed2/s100/epoch-100.pt",
        "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
        ROOT / "docs/experiments/ert_rslad_i100_s2_rbp_masks_dev2_v1.json",
        0.03347739577293396,
    ),
}
SBF_COEFFICIENT = "0.23594490117507805"
TPFM_COEFFICIENT = "0.16676844691071563"
TPFM_FLOOR = "0.16590790450572968"
TPFM_CAP = "0.32364362478256226"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", choices=tuple(PARENTS), required=True)
    parser.add_argument("--arm", choices=("control", "sbf", "tpfm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, checkpoint, parent_sha, mask, student_floor = PARENTS[args.seed]
    runtime_args = [
        "--parent-config", str(config), "--parent-checkpoint", str(checkpoint),
        "--calibration", str(CALIBRATION), "--output", str(args.output),
        "--epochs", "115", "--horizon-epochs", "104", "109", "114",
        "--run-namespace", "ert-i100-s2-rbp-v1", "--resume-epoch", "99",
        "--mask-anchor-epoch", "99", "--expected-parent-sha256", parent_sha,
        "--force-sample-keyed-attack", "--device", "cuda",
    ]
    if args.arm == "control":
        runtime_args += ["--arm", "I100_CONTROL", "--kind", "baseline"]
    else:
        runtime_args += [
            "--mask", str(mask), "--mask-key", "s2_t1", "--kind", "broad",
        ]
        if args.arm == "sbf":
            runtime_args += [
                "--arm", "I100_S2T1_STUDENT_MARGIN_FLOOR",
                "--margin-target-mode", "fixed", "--margin-coefficient", SBF_COEFFICIENT,
                "--margin-gamma", str(student_floor),
            ]
        else:
            runtime_args += [
                "--arm", "I100_S2T1_TEACHER_POSITIVE_FLOOR_MARGIN",
                "--margin-target-mode", "teacher_floor", "--margin-coefficient", TPFM_COEFFICIENT,
                "--margin-floor", TPFM_FLOOR, "--margin-cap", TPFM_CAP,
            ]
    return runtime_main(runtime_args)


if __name__ == "__main__":
    raise SystemExit(main())
