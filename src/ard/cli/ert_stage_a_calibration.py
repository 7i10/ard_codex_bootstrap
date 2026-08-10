"""CLI for the no-update ERT Stage A gradient calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from ard.analysis.ert_stage_a_calibration import calibrate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate frozen ERT Stage A treatment coefficients without updates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = calibrate(config_path=args.config, output=args.output, device=args.device)
    print(
        {
            "status": result["status"],
            "artifact": str(args.output.resolve()),
            "artifact_sha256": result["artifact_sha256"],
            "alpha_soft": result["alpha_soft"],
            "beta_advce_weak": result["beta_advce_weak"],
            "beta_advce_moderate": result["beta_advce_moderate"],
            "beta_cleance_weak": result["beta_cleance_weak"],
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
