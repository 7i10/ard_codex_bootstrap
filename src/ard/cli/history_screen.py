"""Emit canonical H5-Late reports from hash-bound common-PGD replay inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.history_screen import HistoryScreenError, analyze_history_screen


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if separator != "=" or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for option, help_text in (
        ("--feature-panel", "epoch-99 common-PGD feature panel"),
        ("--online-state-export", "hash-bound epoch-99/199 online-forgetting state export"),
        ("--replay-lineage", "lineage binding both panel bytes"),
        ("--frozen-fit", "H2 frozen predictor fit bundle"),
    ):
        parser.add_argument(
            option, action="append", required=True, type=_labeled_path, metavar="LABEL=PATH", help=help_text
        )
    parser.add_argument("--expected-count", required=True, type=_positive_int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = [
        dict(args.feature_panel),
        dict(args.online_state_export),
        dict(args.replay_lineage),
        dict(args.frozen_fit),
    ]
    if any(
        len(mapping) != len(source)
        for mapping, source in zip(
            inputs, (args.feature_panel, args.online_state_export, args.replay_lineage, args.frozen_fit), strict=True
        )
    ):
        raise HistoryScreenError("duplicate H5-Late run label")
    labels = set(inputs[0])
    if any(set(mapping) != labels for mapping in inputs[1:]):
        raise HistoryScreenError("all H5-Late input labels must match exactly")
    reports = {
        label: analyze_history_screen(
            feature_panel=inputs[0][label].resolve(),
            online_state_export=inputs[1][label].resolve(),
            replay_lineage=inputs[2][label].resolve(),
            frozen_fit=inputs[3][label].resolve(),
            expected_count=args.expected_count,
        )
        for label in sorted(labels)
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing H5-Late report")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "h5_late_history_screen_collection_v2",
                "expected_count": args.expected_count,
                "runs": reports,
            },
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except HistoryScreenError as exc:
        raise SystemExit(str(exc)) from exc
