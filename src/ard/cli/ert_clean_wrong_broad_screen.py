"""Run one frozen arm of the ERT Clean-Wrong broad mechanism screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ard.analysis.ert_clean_wrong_broad_screen import (
    arm_by_name,
    calibrate_bce_beta,
    validate_arm,
)
from ard.analysis.ert_stage_a_runtime import StageATreatment, run_stage_a_arm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-config", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", choices=tuple(f"C{i}" for i in range(16)))
    parser.add_argument("--run-namespace", default="ert-clean-wrong-broad")
    parser.add_argument("--epochs", type=int, default=85)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--calibrate-bce", action="store_true")
    parser.add_argument("--calibration-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.calibrate_bce:
        calibration_required = (args.parent_config, args.parent_checkpoint, args.mask, args.calibration_output)
        if any(path is None for path in calibration_required):
            raise ValueError("BCE calibration requires parent config/checkpoint, mask, and output")
        result = calibrate_bce_beta(
            config_path=args.parent_config,
            checkpoint_path=args.parent_checkpoint,
            mask_path=args.mask,
            output_path=args.calibration_output,
            device=args.device,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    required: tuple[object, ...] = (
        args.parent_config,
        args.parent_checkpoint,
        args.mask,
        args.calibration,
        args.output,
        args.arm,
    )
    if any(value is None for value in required):
        raise ValueError("training requires parent config/checkpoint, mask, calibration, output, and arm")
    arm = arm_by_name(args.arm)
    validate_arm(arm)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if not isinstance(calibration, dict):
        raise ValueError("calibration artifact must be a JSON object")
    calibration["artifact_sha256"] = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    if arm.bce:
        beta_bce = calibration.get("beta_bce")
        if not isinstance(beta_bce, (float, int)) or beta_bce <= 0:
            raise ValueError("C12 requires a frozen positive beta_bce calibration")
    else:
        beta_bce = None
    if arm.name == "C0":
        treatment = StageATreatment(arm=arm.name, mask_key=None, kind="baseline")
    else:
        treatment = StageATreatment(
            arm=arm.name,
            mask_key="student_clean_wrong",
            kind="broad",
            advkd_multiplier=arm.advkd_multiplier,
            selected_attack_epsilon=None if arm.selected_epsilon == "8/255" else _fraction(arm.selected_epsilon),
            selected_attack_step_size=None if arm.selected_epsilon == "8/255" else _fraction(arm.selected_step),
            extra_clean_ce=arm.clean_ce or None,
            bce_adv=float(beta_bce) if beta_bce is not None else None,
            adaptive_advkd_gamma=0.5 if arm.adaptive_pressure else None,
            teacher_reliability_gate=arm.teacher_gate,
            iad_inspired=arm.iad_inspired,
        )
    result = run_stage_a_arm(
        parent_config_path=args.parent_config,
        parent_checkpoint=args.parent_checkpoint,
        mask_path=None if args.arm == "C0" else args.mask,
        output_dir=args.output,
        treatment=treatment,
        calibration=calibration,
        device=torch.device(args.device),
        end_epoch=args.epochs,
        horizon_epochs=(84,),
        run_namespace=args.run_namespace,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _fraction(raw: str) -> float:
    numerator, denominator = raw.split("/", 1)
    return float(numerator) / float(denominator)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
