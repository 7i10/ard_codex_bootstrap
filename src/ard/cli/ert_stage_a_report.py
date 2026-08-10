"""Aggregate Stage A common CE-PGD20 endpoint artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.ert_stage_a_report import build_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--mask-l2", type=Path, required=True)
    parser.add_argument("--mask-l4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_report(
        endpoint_root=args.endpoint_root,
        mask_paths={"L2": args.mask_l2, "L4": args.mask_l4},
        output=args.output,
    )
    print(json.dumps({"output": str(args.output.resolve()), "output_sha256": result["output_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
