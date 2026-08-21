"""CLI for the frozen Clean-Wrong margin calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from ard.analysis.ert_cw_margin_calibration import calibrate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate Clean-Wrong AdvCE and margin coefficients without updates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    result = calibrate(config_path=Path(args.config), output=Path(args.output), device=args.device)
    print({k: result[k] for k in ("status", "artifact_sha256", "beta_advce", "margin_coefficient")})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
