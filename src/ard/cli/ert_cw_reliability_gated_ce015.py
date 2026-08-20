"""Prepare fixed reliability masks or run one gated CleanCE continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ard.analysis.ert_cw_reliability_gated import prepare_selector_bundle
from ard.analysis.ert_stage_a_runtime import StageATreatment, run_stage_a_arm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", choices=("L2", "L4"))
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--ce-meta", type=Path)
    parser.add_argument("--ce-rows", type=Path)
    parser.add_argument("--kl-meta", type=Path)
    parser.add_argument("--kl-rows", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--arm", choices=("G0_BASE", "G1_CW_ALL_CE015", "G2_CW_R_CE20_CE015", "G3_CW_R_KL10_CE015"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run-namespace", default="ert-cw-reliability-gated-ce015-v1")
    args = parser.parse_args(argv)
    if args.prepare:
        required = (args.run, args.mask, args.ce_meta, args.ce_rows, args.kl_meta, args.kl_rows)
        if any(item is None for item in required):
            parser.error("--prepare requires --run, --mask, --ce-meta, --ce-rows, --kl-meta, and --kl-rows")
        result = prepare_selector_bundle(
            run=args.run,
            mask_path=args.mask,
            ce_meta=args.ce_meta,
            ce_rows=args.ce_rows,
            kl_meta=args.kl_meta,
            kl_rows=args.kl_rows,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    required = (args.run, args.parent_config, args.parent_checkpoint, args.calibration, args.arm)
    if any(item is None for item in required):
        parser.error("training requires --run, --parent-config, --parent-checkpoint, --calibration, and --arm")
    if args.arm == "G0_BASE":
        mask_path = None
        treatment = StageATreatment(arm=args.arm, mask_key=None, kind="baseline")
    else:
        if args.mask is None:
            parser.error("G1-G3 require --mask")
        mask_path = args.mask
        treatment = StageATreatment(
            arm=args.arm,
            mask_key="student_clean_wrong",
            kind="broad",
            extra_clean_ce=0.15,
        )
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration["artifact_sha256"] = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    result = run_stage_a_arm(
        parent_config_path=args.parent_config,
        parent_checkpoint=args.parent_checkpoint,
        mask_path=mask_path,
        output_dir=args.output_dir,
        treatment=treatment,
        calibration=calibration,
        device=torch.device(args.device),
        end_epoch=94,
        horizon_epochs=(84, 89, 94),
        run_namespace=args.run_namespace,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
