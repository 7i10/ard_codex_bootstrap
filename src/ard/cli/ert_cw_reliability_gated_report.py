"""Aggregate reliability-gated CleanCE endpoint artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.ert_cw_reliability_gated_report import build_report


def _paths(values: list[str]) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {"L2": {}, "L4": {}}
    for value in values:
        run, arm, path = value.split("=", 2)
        result[run][arm] = Path(path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--selector-l2", type=Path, required=True)
    parser.add_argument("--selector-l4", type=Path, required=True)
    parser.add_argument("--training-dir", action="append", required=True, help="RUN=ARM=DIR, repeated 8 times")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_report(
        endpoint_root=args.endpoint_root,
        training_dirs=_paths(args.training_dir),
        selector_dirs={"L2": args.selector_l2, "L4": args.selector_l4},
        output=args.output,
    )
    print(json.dumps({"output": str(args.output.resolve()), "output_sha256": result["output_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
