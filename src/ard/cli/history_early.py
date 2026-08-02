"""Write a non-overwriting H5-Early JSON collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.history_early import HistoryEarlyError, analyze_history_early, collection_gate


def _lp(x: str) -> tuple[str, Path]:
    a, b, c = x.partition("=")
    if b != "=" or not a or not c:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return a, Path(c)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for x in ("--feature-observations", "--outcome-observations", "--feature-lineage", "--outcome-lineage"):
        p.add_argument(x, required=True, action="append", type=_lp, metavar="LABEL=PATH")
    p.add_argument("--expected-count", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    groups = [
        dict(a.feature_observations),
        dict(a.outcome_observations),
        dict(a.feature_lineage),
        dict(a.outcome_lineage),
    ]
    if a.expected_count < 1 or any(
        len(g) != len(raw)
        for g, raw in zip(
            groups, (a.feature_observations, a.outcome_observations, a.feature_lineage, a.outcome_lineage), strict=True
        )
    ):
        raise HistoryEarlyError("invalid expected count or duplicate labels")
    labels = set(groups[0])
    if any(set(g) != labels for g in groups[1:]):
        raise HistoryEarlyError("input labels must match")
    reports = {
        label: analyze_history_early(
            feature_observations=groups[0][label].resolve(),
            outcome_observations=groups[1][label].resolve(),
            feature_lineage=groups[2][label].resolve(),
            outcome_lineage=groups[3][label].resolve(),
            expected_count=a.expected_count,
        )
        for label in sorted(labels)
    }
    gate = collection_gate(reports)
    public_reports = {
        label: {key: value for key, value in report.items() if key != "_post_peak_gate_rows"}
        for label, report in reports.items()
    }
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError("refusing to overwrite H5-Early output")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "h5_early_collection_v1",
                "reports": public_reports,
                "primary_selection_gate": gate,
            },
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    return 0
