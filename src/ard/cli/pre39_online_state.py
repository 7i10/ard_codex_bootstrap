"""Export one CPU-only exact-online PRE39 candidate anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.pre39_online_state import Pre39OnlineStateError, export_pre39_online_state, write_pre39_online_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--feature-observations", required=True, type=Path)
    parser.add_argument("--feature-lineage", required=True, type=Path)
    parser.add_argument("--anchor", required=True, type=int)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    export = export_pre39_online_state(
        checkpoint=args.checkpoint.resolve(),
        feature_observations=args.feature_observations.resolve(),
        feature_lineage=args.feature_lineage.resolve(),
        anchor=args.anchor,
        expected_count=args.expected_count,
    )
    paths = write_pre39_online_state(output_dir=args.output_dir.resolve(), export=export)
    print(json.dumps({name: str(path) for name, path in sorted(paths.items())}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except Pre39OnlineStateError as exc:
        raise SystemExit(str(exc)) from exc
