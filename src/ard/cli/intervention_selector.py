"""Build a frozen post-H2 intervention selector bundle from six input files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.intervention_selector import SelectorFiles, build_selector_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-feature-panel", required=True, type=Path)
    parser.add_argument("--seed0-outcome-panel", required=True, type=Path)
    parser.add_argument("--seed0-report", required=True, type=Path)
    parser.add_argument("--seed0-lineage", required=True, type=Path)
    parser.add_argument("--l3-feature-panel", required=True, type=Path)
    parser.add_argument("--l3-lineage", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = SelectorFiles(
        seed0_feature_panel=args.seed0_feature_panel.resolve(),
        seed0_outcome_panel=args.seed0_outcome_panel.resolve(),
        seed0_report=args.seed0_report.resolve(),
        seed0_lineage=args.seed0_lineage.resolve(),
        l3_feature_panel=args.l3_feature_panel.resolve(),
        l3_lineage=args.l3_lineage.resolve(),
    )
    created = build_selector_bundle(files=files, output_dir=args.output_dir.resolve())
    print(json.dumps({key: str(path.resolve()) for key, path in sorted(created.items())}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
