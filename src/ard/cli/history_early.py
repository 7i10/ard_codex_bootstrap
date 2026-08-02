"""Write a non-overwriting H5-Early JSON collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.history_early import (
    HistoryEarlyError,
    analyze_history_early,
    analyze_history_early_online,
    bind_early_collection_to_cohort,
    build_online_bootstrap_tasks,
    collection_gate,
)


def _lp(x: str) -> tuple[str, Path]:
    a, b, c = x.partition("=")
    if b != "=" or not a or not c:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return a, Path(c)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for x in ("--feature-observations", "--outcome-observations", "--feature-lineage", "--outcome-lineage"):
        p.add_argument(x, required=False, action="append", type=_lp, metavar="LABEL=PATH")
    p.add_argument("--online-states", required=False, action="append", type=_lp, metavar="LABEL=PATH")
    p.add_argument("--online-lineage", required=False, action="append", type=_lp, metavar="LABEL=PATH")
    p.add_argument("--expected-count", required=True, type=int)
    p.add_argument("--cohort-inventory", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if bool(a.online_states) != bool(a.online_lineage):
        raise HistoryEarlyError("online states and online lineage must be supplied together")
    if a.online_states:
        if not all((a.feature_observations, a.feature_lineage, a.outcome_observations, a.outcome_lineage)):
            raise HistoryEarlyError(
                "online primary report requires replay feature and outcome observations with lineage"
            )
        states, lineages = dict(a.online_states), dict(a.online_lineage)
        features, feature_lineages = dict(a.feature_observations), dict(a.feature_lineage)
        outcomes, outcome_lineages = dict(a.outcome_observations), dict(a.outcome_lineage)
        if (
            len(states) != len(a.online_states)
            or len(lineages) != len(a.online_lineage)
            or set(states) != set(lineages)
            or set(states) != set(features)
            or set(states) != set(feature_lineages)
            or set(states) != set(outcomes)
            or set(states) != set(outcome_lineages)
        ):
            raise HistoryEarlyError("online H5-Early labels must match uniquely")
        reports = {
            label: analyze_history_early_online(
                online_states=states[label].resolve(),
                online_lineage=lineages[label].resolve(),
                feature_observations=features[label].resolve(),
                feature_lineage=feature_lineages[label].resolve(),
                outcome_observations=outcomes[label].resolve(),
                outcome_lineage=outcome_lineages[label].resolve(),
                expected_count=a.expected_count,
            )
            for label in sorted(states)
        }
        tasks, primary_gate = build_online_bootstrap_tasks(reports)
        cohort_sha256 = bind_early_collection_to_cohort(cohort_inventory=a.cohort_inventory.resolve(), reports=reports)
        out = a.output.resolve()
        if out.exists():
            raise FileExistsError("refusing to overwrite H5-Early output")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "contract": "h5_early_online_collection_v1",
                    "reports": {
                        label: {key: value for key, value in report.items() if key != "_bootstrap_inputs"}
                        for label, report in reports.items()
                    },
                    "bootstrap_tasks": tasks,
                    "primary_bootstrap_gate": primary_gate,
                    "status": "point_gate_pass_bootstrap_pending" if tasks else "no_go_point_gate",
                    "cohort_inventory_sha256": cohort_sha256,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        return 0
    if not all((a.feature_observations, a.outcome_observations, a.feature_lineage, a.outcome_lineage)):
        raise HistoryEarlyError("either replay panels or online state panels are required")
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
    cohort_sha256 = bind_early_collection_to_cohort(cohort_inventory=a.cohort_inventory.resolve(), reports=reports)
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
                "cohort_inventory_sha256": cohort_sha256,
            },
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except HistoryEarlyError as exc:
        raise SystemExit(str(exc)) from exc
