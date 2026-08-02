"""Run the separate resumable paired-bootstrap stage for an H5 point report."""

from __future__ import annotations

import argparse
from pathlib import Path

from ard.analysis.history_bootstrap import HistoryBootstrapError, run_bootstrap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    run_bootstrap(
        point_report=args.point_report.resolve(),
        output=args.output.resolve(),
        progress_dir=args.progress_dir.resolve(),
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HistoryBootstrapError as exc:
        raise SystemExit(str(exc)) from exc
