#!/usr/bin/env python3
"""Fail-closed validator for dense epoch telemetry.

This is deliberately execution-layer generic: it knows only that one row is
required for every integer epoch, selected identity fields must be stable, and
numeric fields must be finite.  It does not know a scientific metric's meaning.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class DenseMetricsError(ValueError):
    """Raised when a telemetry stream cannot be trusted as dense."""


def validate_dense_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_start: int = 0,
    expected_end: int,
    required_fields: Iterable[str] = ("epoch",),
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    expected_epochs = list(range(expected_start, expected_end + 1))
    if len(materialized) != len(expected_epochs):
        raise DenseMetricsError(f"expected {len(expected_epochs)} rows, got {len(materialized)}")
    required = tuple(dict.fromkeys(required_fields))
    for index, row in enumerate(materialized):
        missing = [field for field in required if field not in row]
        if missing:
            raise DenseMetricsError(f"row {index} missing required fields: {missing}")
        try:
            epoch = int(row["epoch"])
        except (TypeError, ValueError) as exc:
            raise DenseMetricsError(f"row {index} has invalid epoch") from exc
        if epoch != expected_epochs[index]:
            raise DenseMetricsError(f"expected epoch {expected_epochs[index]} at row {index}, got {epoch}")
        for field in required:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise DenseMetricsError(f"required field {field} is non-finite/non-numeric at epoch {epoch}")
    if identity:
        for key, expected in identity.items():
            values = {row.get(key) for row in materialized}
            if values != {expected}:
                raise DenseMetricsError(f"identity field {key} is not constant/equal to expected value")
    return {
        "valid": True,
        "row_count": len(materialized),
        "epoch_start": expected_start,
        "epoch_end": expected_end,
        "required_fields": list(required),
        "identity": dict(identity or {}),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DenseMetricsError(f"cannot read metrics: {path}") from exc
    try:
        return [json.loads(line) for line in lines if line.strip()]
    except json.JSONDecodeError as exc:
        raise DenseMetricsError(f"invalid JSONL: {path}:{exc.lineno}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-start", type=int, default=0)
    parser.add_argument("--expected-end", type=int, required=True)
    parser.add_argument("--required-field", action="append", default=["epoch"])
    parser.add_argument("--identity", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    identity: dict[str, Any] = {}
    for item in args.identity:
        if "=" not in item:
            parser.error("--identity requires KEY=VALUE")
        key, value = item.split("=", 1)
        identity[key] = value
    try:
        result = validate_dense_rows(
            load_jsonl(args.path),
            expected_start=args.expected_start,
            expected_end=args.expected_end,
            required_fields=args.required_field,
            identity=identity,
        )
    except DenseMetricsError as exc:
        parser.exit(2, f"dense metrics invalid: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
