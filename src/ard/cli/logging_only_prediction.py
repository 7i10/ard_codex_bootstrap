"""Run the frozen H2 prediction analysis on logging-only state exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.logging_only_prediction import (
    LoggingOnlyPredictionError,
    analyze_logging_only_exports,
    load_state_export_with_provenance,
)


def _run_export(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if separator != "=" or label not in {"L1", "L2", "L3", "L4"} or not raw_path:
        raise argparse.ArgumentTypeError("--run must be one of L1=PATH, L2=PATH, L3=PATH, or L4=PATH")
    return label, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", required=True, action="append", type=_run_export, help="L1=state-export.json (repeat for L1--L4)."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New H2 JSON report; never overwrites an existing report."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exports: dict[str, dict[str, object]] = {}
    source_inputs: dict[str, dict[str, str]] = {}
    for label, path in args.run:
        if label in exports:
            raise LoggingOnlyPredictionError("duplicate H2 run label")
        export, source_input = load_state_export_with_provenance(path)
        exports[label] = export
        source_inputs[label] = source_input
    report = analyze_logging_only_exports(exports)
    report["source_inputs"] = source_inputs
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing H2 report")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except LoggingOnlyPredictionError as exc:
        raise SystemExit(str(exc)) from exc
