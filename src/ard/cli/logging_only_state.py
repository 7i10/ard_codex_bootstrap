"""Export hash-bound epoch-99/199 logging-only sample-state primitives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.logging_only_state import logging_only_state_analysis


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected count must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-checkpoint", required=True, type=Path)
    parser.add_argument("--final-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=_positive_int)
    parser.add_argument(
        "--run-bundle-manifest",
        required=True,
        type=Path,
        help="Canonical run-bundle manifest that inventories both checkpoint bytes.",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="JSON matrix artifact for the frozen predictor boundary."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = logging_only_state_analysis(
        anchor_checkpoint=args.anchor_checkpoint.resolve(),
        final_checkpoint=args.final_checkpoint.resolve(),
        expected_count=args.expected_count,
        run_bundle_manifest=args.run_bundle_manifest.resolve(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"identity": analysis.identity, "rows": analysis.rows}, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
