"""Run the frozen exploratory M2a treatment-utility point audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.treatment_utility import TreatmentUtilityError, run_treatment_utility


def _panel(value: str) -> tuple[str, tuple[Path, Path, Path]]:
    label, separator, payload = value.partition("=")
    parts = payload.split(",")
    if separator != "=" or not label or len(parts) != 3 or any(not part for part in parts):
        raise argparse.ArgumentTypeError("expected SEED:ROUTE:ARM=OBSERVATIONS,LINEAGE,PARENT")
    return label, tuple(Path(part) for part in parts)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", action="append", required=True, type=_panel)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-seed", type=int, default=20260806)
    args = parser.parse_args(argv)
    panels = dict(args.panel)
    if len(panels) != len(args.panel):
        raise TreatmentUtilityError("panel labels must be unique")
    value = run_treatment_utility(
        panels={label: tuple(path.resolve() for path in paths) for label, paths in panels.items()},
        output=args.output.resolve(),
        split_seed=args.split_seed,
    )
    print(json.dumps({"contract": value["contract"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except TreatmentUtilityError as exc:
        raise SystemExit(str(exc)) from exc
