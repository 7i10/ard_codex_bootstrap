"""Run the hash-bound CPU-only PRE39 replay point screen or one bootstrap task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.pre39_prescriptive import (
    MODEL_NAMES,
    Pre39PrescriptiveError,
    analyze_pre39_prescriptive,
    collect_pre39_reports,
    run_pre39_bootstrap,
    write_pre39_report,
)


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if separator != "=" or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("--feature-observations", "--outcome-observations", "--feature-lineage", "--outcome-lineage"):
        parser.add_argument(option, required=True, action="append", type=_labeled_path, metavar="LABEL=PATH")
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--bootstrap-anchor", type=int)
    parser.add_argument("--bootstrap-stratum", choices=("PF", "NR"))
    parser.add_argument("--bootstrap-baseline", choices=MODEL_NAMES, default="instantaneous_margin")
    parser.add_argument("--bootstrap-candidate", choices=MODEL_NAMES, default="student_history")
    parser.add_argument("--bootstrap-progress", type=Path)
    args = parser.parse_args(argv)
    raw_groups = (args.feature_observations, args.outcome_observations, args.feature_lineage, args.outcome_lineage)
    groups = [dict(value) for value in raw_groups]
    if any(len(group) != len(raw) for group, raw in zip(groups, raw_groups, strict=True)):
        raise Pre39PrescriptiveError("PRE39 input labels must be unique")
    labels = set(groups[0])
    if any(set(group) != labels for group in groups[1:]) or (len(labels) != 1 and labels != {"L1", "L2", "L3", "L4"}):
        raise Pre39PrescriptiveError("provide either one run or exactly L1/L2/L3/L4 matching labels")
    reports = {
        label: analyze_pre39_prescriptive(
            feature_observations=groups[0][label].resolve(),
            outcome_observations=groups[1][label].resolve(),
            feature_lineage=groups[2][label].resolve(),
            outcome_lineage=groups[3][label].resolve(),
            expected_count=args.expected_count,
        )
        for label in sorted(labels)
    }
    if args.bootstrap:
        if len(reports) != 1:
            raise Pre39PrescriptiveError("bootstrap accepts exactly one selected run")
        if args.bootstrap_anchor is None or args.bootstrap_stratum is None or args.bootstrap_progress is None:
            raise Pre39PrescriptiveError(
                "bootstrap requires --bootstrap-anchor, --bootstrap-stratum, and --bootstrap-progress"
            )
        result = run_pre39_bootstrap(
            report=next(iter(reports.values())),
            anchor=args.bootstrap_anchor,
            stratum=args.bootstrap_stratum,
            baseline=args.bootstrap_baseline,
            candidate=args.bootstrap_candidate,
            output=args.output.resolve(),
            progress=args.bootstrap_progress.resolve(),
        )
        print(json.dumps(result, sort_keys=True))
    else:
        report = next(iter(reports.values())) if len(reports) == 1 else collect_pre39_reports(reports)
        path = write_pre39_report(output=args.output.resolve(), report=report, overwrite=args.overwrite)
        print(json.dumps({"report": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except Pre39PrescriptiveError as exc:
        raise SystemExit(str(exc)) from exc
