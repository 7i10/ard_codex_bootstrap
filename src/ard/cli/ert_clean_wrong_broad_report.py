"""Build the direct/spillover/held-out Clean-Wrong screen report."""

from __future__ import annotations

import argparse
from pathlib import Path

from ard.analysis.ert_clean_wrong_broad_report import build_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--l2-mask", type=Path, required=True)
    parser.add_argument("--l4-mask", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args(argv)
    result = build_report(
        root=args.root,
        masks={"L2": args.l2_mask, "L4": args.l4_mask},
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        bootstrap=not args.no_bootstrap,
    )
    print(result["source_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
