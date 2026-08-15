"""Build the frozen dynamic-S3 transition and CE-PGD20 endpoint report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.ert_dynamic_s3_recovery import build_dynamic_s3_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_dynamic_s3_report(
        endpoint_root=args.endpoint_root,
        training_root=args.training_root,
        config_path=args.config,
        output=args.output,
    )
    print(json.dumps({"output": str(args.output.resolve()), "output_sha256": result["output_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
