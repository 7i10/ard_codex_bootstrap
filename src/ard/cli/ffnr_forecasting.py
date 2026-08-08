"""Run the read-only FF/current-wrong-future-failure development analysis."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ard.analysis.ffnr_forecasting import FFNRForecastingError, analyze_ffnr_run, write_ffnr_report


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FFNRForecastingError(f"FF/NR config {name} must be a mapping")
    return value


def _path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FFNRForecastingError(f"FF/NR config {name} must be a non-empty path string")
    return Path(value).expanduser().resolve()


def load_frozen_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FFNRForecastingError("FF/NR config is unreadable") from exc
    config = _mapping(value, "root")
    required = {
        "schema_version",
        "contract",
        "expected_count",
        "cohort_inventory",
        "scheduler_stages",
        "anchors",
        "candidate_grid",
        "runs",
    }
    if set(config) != required or config.get("schema_version") != 1 or config.get("contract") != "ffnr_forecasting_v1":
        raise FFNRForecastingError("FF/NR config schema/contract drifted")
    if (
        isinstance(config["expected_count"], bool)
        or not isinstance(config["expected_count"], int)
        or config["expected_count"] < 1
    ):
        raise FFNRForecastingError("FF/NR config expected_count is invalid")
    if not isinstance(config["scheduler_stages"], list) or not isinstance(config["anchors"], list):
        raise FFNRForecastingError("FF/NR config scheduler stages/anchors are invalid")
    grid = _mapping(config["candidate_grid"], "candidate_grid")
    if set(grid) != {"deltas_pp", "window_sizes", "thresholds"} or not all(isinstance(grid[key], list) for key in grid):
        raise FFNRForecastingError("FF/NR config candidate grid schema drifted")
    runs = _mapping(config["runs"], "runs")
    if set(runs) != {"L1", "L2", "L3", "L4"}:
        raise FFNRForecastingError("FF/NR config must bind exactly L1--L4")
    required_paths = {
        "feature_observations",
        "feature_lineage",
        "outcome_observations",
        "outcome_lineage",
        "online_states",
        "online_lineage",
        "validation_history",
        "validation_manifest",
    }
    parsed: dict[str, Any] = {
        "expected_count": config["expected_count"],
        "cohort_inventory": _path(config["cohort_inventory"], "cohort_inventory"),
        "scheduler_stages": config["scheduler_stages"],
        "anchors": tuple(config["anchors"]),
        "deltas_pp": tuple(grid["deltas_pp"]),
        "window_sizes": tuple(grid["window_sizes"]),
        "thresholds": tuple(grid["thresholds"]),
        "runs": {},
    }
    for label in sorted(runs):
        item = _mapping(runs[label], f"runs.{label}")
        if set(item) != required_paths:
            raise FFNRForecastingError(f"FF/NR config runs.{label} path schema drifted")
        parsed["runs"][label] = {name: _path(item[name], f"runs.{label}.{name}") for name in required_paths}
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="frozen FF/NR YAML input binding")
    parser.add_argument("--output-dir", required=True, type=Path, help="new, empty output directory")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_frozen_config(config_path)
    reports = {
        label: analyze_ffnr_run(
            label=label,
            expected_count=config["expected_count"],
            scheduler_stages=config["scheduler_stages"],
            anchors=config["anchors"],
            deltas_pp=config["deltas_pp"],
            window_sizes=config["window_sizes"],
            thresholds=config["thresholds"],
            **config["runs"][label],
        )
        for label in sorted(config["runs"])
    }
    paths = write_ffnr_report(
        output_dir=args.output_dir.resolve(),
        reports=reports,
        config_path=config_path,
        cohort_inventory=config["cohort_inventory"],
    )
    print("\n".join(f"{name}={path}" for name, path in sorted(paths.items())))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except FFNRForecastingError as exc:
        raise SystemExit(str(exc)) from exc
