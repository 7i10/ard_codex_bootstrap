"""Run one Chen ERT same-step dynamic-S3 recovery continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ard.analysis.ert_dynamic_s3_recovery import run_dynamic_s3_arm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one ERT dynamic S3 recovery arm from epoch-79.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", choices=("L2", "L4"), required=True)
    parser.add_argument("--arm", choices=("DYNBASE", "S3FIX075", "S3DYN075"), required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--peer-epoch80-state",
        type=Path,
        help="Required for S3FIX075/S3DYN075: paired arm's epoch80-routing-state.json.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_dynamic_s3_arm(
        config_path=args.config,
        run_key=args.run,
        arm=args.arm,
        output_dir=args.output,
        calibration_path=args.calibration,
        device=torch.device(args.device),
        peer_epoch80_state=args.peer_epoch80_state,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
