"""Write a read-only, schema-v2 H4a taxonomy and blinded ID manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.h4a_taxonomy import H4aTaxonomyError, analyze_h4a_collection, analyze_h4a_taxonomy, write_h4a_outputs


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if separator != "=" or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("--feature-observations", "--outcome-observations", "--feature-lineage", "--outcome-lineage"):
        parser.add_argument(option, required=True, action="append", type=_labeled_path, metavar="LABEL=PATH")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    groups = [
        dict(args.feature_observations),
        dict(args.outcome_observations),
        dict(args.feature_lineage),
        dict(args.outcome_lineage),
    ]
    if args.expected_count != 45000 or any(
        len(group) != len(raw)
        for group, raw in zip(
            groups,
            (args.feature_observations, args.outcome_observations, args.feature_lineage, args.outcome_lineage),
            strict=True,
        )
    ):
        raise H4aTaxonomyError("H4a requires exactly 45,000 train IDs and unique labels")
    labels = set(groups[0])
    if any(set(group) != labels for group in groups[1:]):
        raise H4aTaxonomyError("all H4a input labels must match exactly")
    reports = {
        label: analyze_h4a_taxonomy(
            feature_observations=groups[0][label].resolve(),
            outcome_observations=groups[1][label].resolve(),
            feature_lineage=groups[2][label].resolve(),
            outcome_lineage=groups[3][label].resolve(),
            expected_count=args.expected_count,
        )
        for label in sorted(labels)
    }
    paths = write_h4a_outputs(output_dir=args.output_dir.resolve(), collection=analyze_h4a_collection(reports))
    print(json.dumps({name: str(path) for name, path in sorted(paths.items())}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except H4aTaxonomyError as exc:
        raise SystemExit(str(exc)) from exc
