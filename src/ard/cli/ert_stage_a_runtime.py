"""Run one fixed-parent ERT Stage A continuation arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ard.analysis.ert_rslad_rng_sources import RNGSourceSeeds, ShuffleAugmentationSeeds
from ard.analysis.ert_stage_a_runtime import StageATreatment, run_stage_a_arm


def _load_calibration_artifact(path: Path, *, require_hash_sidecar: bool) -> dict[str, object]:
    """Load an immutable calibration and bind its byte hash into lineage.

    A JSON document cannot safely contain the hash of its own final bytes, so
    boundary-intervention artifacts carry an adjacent ``.sha256`` sidecar.
    The in-memory ``artifact_sha256`` is then written into the continuation
    lineage without mutating the frozen calibration file.
    """
    raw = path.read_bytes()
    calibration = json.loads(raw)
    if not isinstance(calibration, dict):
        raise ValueError("calibration artifact must be a JSON object")
    digest = hashlib.sha256(raw).hexdigest()
    if require_hash_sidecar:
        sidecar = path.with_name(path.name + ".sha256")
        if not sidecar.is_file():
            raise ValueError(f"boundary calibration hash sidecar is missing: {sidecar}")
        declared = sidecar.read_text(encoding="utf-8").strip()
        if declared != digest:
            raise ValueError(f"boundary calibration hash sidecar mismatch: {sidecar}")
    return {**calibration, "artifact_sha256": digest}


def _parse_budget(raw: str | None) -> float | None:
    if raw is None:
        return None
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        return float(numerator) / float(denominator)
    return float(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one ERT fixed-parent treatment continuation.")
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
    parser.add_argument(
        "--margin-target-mode",
        choices=("fixed", "teacher_zero", "teacher_floor", "teacher_abstain"),
    )
    parser.add_argument("--margin-coefficient", type=float)
    parser.add_argument("--margin-gamma", type=float)
    parser.add_argument("--margin-floor", type=float)
    parser.add_argument("--margin-cap", type=float)
    parser.add_argument("--teacher-reliability-gate", action="store_true")
    parser.add_argument("--iad-inspired", action="store_true")
    parser.add_argument(
        "--boundary-intervention",
        choices=("pair_margin", "detached_boundary_distance", "secant_boundary_distance"),
    )
    parser.add_argument("--boundary-coefficient", type=float)
    parser.add_argument("--boundary-epsilon", type=float, default=1e-12)
    parser.add_argument("--epochs", type=int, default=85)
    parser.add_argument(
        "--horizon-epochs",
        type=int,
        nargs="+",
        default=(84,),
        help="Epochs for immutable post-epoch checkpoint copies (all must be <= --epochs).",
    )
    parser.add_argument("--run-namespace", default="stage-a")
    parser.add_argument(
        "--resume-epoch",
        type=int,
        default=79,
        help="Serialized epoch at the end boundary of the supplied parent checkpoint.",
    )
    parser.add_argument(
        "--mask-anchor-epoch",
        type=int,
        help="Anchor epoch encoded by the fixed intervention mask (defaults to --resume-epoch).",
    )
    parser.add_argument(
        "--continuation-seed",
        type=int,
        help="Post-resume attack RNG seed for an independent matched continuation replicate.",
    )
    parser.add_argument("--data-seed", type=int, help="Post-resume data order/augmentation/worker seed.")
    parser.add_argument("--attack-seed", type=int, help="Post-resume PGD random-start seed.")
    parser.add_argument("--other-seed", type=int, help="Post-resume Python/NumPy/global Torch seed.")
    parser.add_argument("--shuffle-seed", type=int, help="Post-resume sampler-order seed.")
    parser.add_argument("--augmentation-seed", type=int, help="Post-resume source-keyed augmentation seed.")
    parser.add_argument(
        "--expected-parent-sha256",
        help="Fail closed unless the supplied parent has this exact SHA-256.",
    )
    parser.add_argument(
        "--force-sample-keyed-attack",
        action="store_true",
        help="Use the registered sample-keyed KL-PGD10 continuation contract.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        margin_coefficient=args.margin_coefficient,
        margin_target_mode=args.margin_target_mode,
        margin_gamma=args.margin_gamma,
        margin_floor=args.margin_floor,
        margin_cap=args.margin_cap,
        teacher_reliability_gate=args.teacher_reliability_gate,
        iad_inspired=args.iad_inspired,
        boundary_intervention=args.boundary_intervention,
        boundary_coefficient=args.boundary_coefficient,
        boundary_epsilon=args.boundary_epsilon,
    )
    calibration = _load_calibration_artifact(
        args.calibration,
        require_hash_sidecar=treatment.boundary_intervention is not None,
    )
    if treatment.mask_key is not None and args.mask is None:
        raise ValueError("selected treatment requires --mask")
    # ``attack`` and ``other`` are shared by both contracts.  Do not reject a
    # split invocation merely because those two flags are present without the
    # legacy ``data-seed``; validate the old triplet only when that entrypoint
    # is actually selected below.
    if args.data_seed is not None and (args.attack_seed is None or args.other_seed is None):
        raise ValueError("--data-seed, --attack-seed, and --other-seed must be supplied together")
    rng_source_seeds = (
        None
        if args.data_seed is None
        else RNGSourceSeeds(
            data_seed=args.data_seed,
            attack_seed=args.attack_seed,
            other_seed=args.other_seed,
        )
    )
    split_values = (args.shuffle_seed, args.augmentation_seed, args.attack_seed, args.other_seed)
    if any(value is not None for value in split_values) and not all(value is not None for value in split_values):
        raise ValueError(
            "--shuffle-seed, --augmentation-seed, --attack-seed, and --other-seed must be supplied together"
        )
    if any(value is not None for value in split_values) and rng_source_seeds is not None:
        raise ValueError("split shuffle/augmentation seeds cannot be combined with --data-seed")
    shuffle_augmentation_seeds = (
        None
        if args.shuffle_seed is None
        else ShuffleAugmentationSeeds(
            shuffle_seed=args.shuffle_seed,
            augmentation_seed=args.augmentation_seed,
            attack_seed=args.attack_seed,
            other_seed=args.other_seed,
        )
    )
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
        continuation_seed=args.continuation_seed,
        rng_source_seeds=rng_source_seeds,
        shuffle_augmentation_seeds=shuffle_augmentation_seeds,
        expected_parent_checkpoint_sha256=args.expected_parent_sha256,
        resume_epoch=args.resume_epoch,
        mask_anchor_epoch=args.mask_anchor_epoch,
        force_sample_keyed_attack=args.force_sample_keyed_attack,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
