"""Confirm the exact-online PRE39 epoch-34 candidate against replay outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.pre39_online_confirm import (
    Pre39OnlineConfirmError,
    analyze_pre39_online_confirm,
    run_pre39_online_bootstrap,
    write_pre39_online_confirm,
)


def _inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--online-observations", required=True, type=Path)
    parser.add_argument("--online-lineage", required=True, type=Path)
    parser.add_argument("--feature-observations", required=True, type=Path)
    parser.add_argument("--feature-lineage", required=True, type=Path)
    parser.add_argument("--outcome-observations", required=True, type=Path)
    parser.add_argument("--outcome-lineage", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    point = commands.add_parser("point")
    _inputs(point)
    point.add_argument("--output", required=True, type=Path)
    point.add_argument("--overwrite", action="store_true")
    bootstrap = commands.add_parser("bootstrap")
    _inputs(bootstrap)
    bootstrap.add_argument("--stratum", choices=("PF", "NR"), required=True)
    bootstrap.add_argument("--baseline", choices=("instantaneous_margin", "teacher_entropy"), required=True)
    bootstrap.add_argument("--output", required=True, type=Path)
    bootstrap.add_argument("--progress", required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze_pre39_online_confirm(
        online_observations=args.online_observations.resolve(),
        online_lineage=args.online_lineage.resolve(),
        feature_observations=args.feature_observations.resolve(),
        feature_lineage=args.feature_lineage.resolve(),
        outcome_observations=args.outcome_observations.resolve(),
        outcome_lineage=args.outcome_lineage.resolve(),
        expected_count=args.expected_count,
    )
    if args.command == "point":
        output = write_pre39_online_confirm(output=args.output.resolve(), report=report, overwrite=args.overwrite)
        print(json.dumps({"report": str(output)}, sort_keys=True))
    else:
        result = run_pre39_online_bootstrap(
            report=report,
            stratum=args.stratum,
            baseline=args.baseline,
            output=args.output.resolve(),
            progress=args.progress.resolve(),
        )
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except Pre39OnlineConfirmError as exc:
        raise SystemExit(str(exc)) from exc
