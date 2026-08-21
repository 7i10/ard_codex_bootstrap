"""Write the frozen A5--A8 no-update mechanism diagnostic report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.ert_cw_a7_mechanism_report import build_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--ce-feature-root", type=Path, required=True)
    parser.add_argument("--gradient-root", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_report(
        replay_root=args.replay_root,
        endpoint_root=args.endpoint_root,
        ce_feature_root=args.ce_feature_root,
        gradient_root=args.gradient_root,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(json.dumps({"contract": result["contract"], "source_git_sha": result["source_git_sha"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
