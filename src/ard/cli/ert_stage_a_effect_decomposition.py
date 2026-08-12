"""Build the frozen Stage A direct/spillover/held-out report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.ert_stage_a_effect_decomposition import build_effect_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--train-output-root", type=Path, required=True)
    parser.add_argument("--mask-l2", type=Path, required=True)
    parser.add_argument("--mask-l4", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--stage-a-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_effect_report(
        train_root=args.train_root,
        validation_root=args.validation_root,
        train_output_root=args.train_output_root,
        mask_paths={"L2": args.mask_l2, "L4": args.mask_l4},
        calibration_path=args.calibration,
        stage_a_report_path=args.stage_a_report,
        output=args.output,
    )
    print(json.dumps({"output": str(args.output.resolve()), "output_sha256": result["output_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
