"""Aggregate ERT history-smoothed S3 training and CE-PGD20 endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.ert_s3_history_production_report import build_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_report(
        config_path=args.config.resolve(),
        training_root=args.training_root.resolve(),
        endpoint_root=args.endpoint_root.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps({"output": str(args.output.resolve()), "output_sha256": result["output_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
