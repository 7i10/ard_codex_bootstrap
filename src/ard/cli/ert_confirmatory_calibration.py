"""Run the frozen-coefficient no-update sanity check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.ert_confirmatory_calibration import build_calibration_sanity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check rounded ERT confirmatory AdvCE calibration without updating a model."
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_calibration_sanity(args.calibration, output_path=args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
