"""Run one fixed-parent ERT Stage A continuation arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ard.analysis.ert_stage_a_runtime import StageATreatment, run_stage_a_arm


def _parse_budget(raw: str | None) -> float | None:
    if raw is None:
        return None
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        return float(numerator) / float(denominator)
    return float(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one ERT Stage A treatment from an epoch-79 parent.")
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--mask-key")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--kind", choices=("baseline", "advce", "soft_advkd", "advkd_advce", "clean_wrong", "broad"), required=True
    )
    parser.add_argument("--beta-advce", type=float)
    parser.add_argument("--advkd-multiplier", type=float)
    parser.add_argument("--beta-cleance", type=float)
    parser.add_argument("--clean-wrong-mode", choices=("clean_ce_only", "teacher_clean_gate", "clean_kd"))
    parser.add_argument("--tau", type=float)
    parser.add_argument("--selected-attack-epsilon")
    parser.add_argument("--selected-attack-step-size")
    parser.add_argument("--extra-clean-ce", type=float)
    parser.add_argument("--bce-adv", type=float)
    parser.add_argument("--adaptive-advkd-gamma", type=float)
    parser.add_argument("--teacher-reliability-gate", action="store_true")
    parser.add_argument("--iad-inspired", action="store_true")
    parser.add_argument("--epochs", type=int, default=85)
    parser.add_argument(
        "--horizon-epochs",
        type=int,
        nargs="+",
        default=(84,),
        help="Epochs for immutable post-epoch checkpoint copies (all must be <= --epochs).",
    )
    parser.add_argument("--run-namespace", default="stage-a")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if not isinstance(calibration, dict):
        raise ValueError("calibration artifact must be a JSON object")
    calibration["artifact_sha256"] = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    treatment = StageATreatment(
        arm=args.arm,
        mask_key=args.mask_key,
        kind=args.kind,
        beta_advce=args.beta_advce,
        advkd_multiplier=args.advkd_multiplier,
        beta_cleance=args.beta_cleance,
        clean_wrong_mode=args.clean_wrong_mode,
        tau=args.tau,
        selected_attack_epsilon=_parse_budget(args.selected_attack_epsilon),
        selected_attack_step_size=_parse_budget(args.selected_attack_step_size),
        extra_clean_ce=args.extra_clean_ce,
        bce_adv=args.bce_adv,
        adaptive_advkd_gamma=args.adaptive_advkd_gamma,
        teacher_reliability_gate=args.teacher_reliability_gate,
        iad_inspired=args.iad_inspired,
    )
    if treatment.mask_key is not None and args.mask is None:
        raise ValueError("selected treatment requires --mask")
    result = run_stage_a_arm(
        parent_config_path=args.parent_config,
        parent_checkpoint=args.parent_checkpoint,
        mask_path=args.mask,
        output_dir=args.output,
        treatment=treatment,
        calibration=calibration,
        device=torch.device(args.device),
        end_epoch=args.epochs,
        horizon_epochs=tuple(args.horizon_epochs),
        run_namespace=args.run_namespace,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
