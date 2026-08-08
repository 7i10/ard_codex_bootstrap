"""Run the read-only L2/L4 CE-PGD20 FF/current-wrong point analysis."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ard.analysis.ffnr_strong_point import CONTRACT, StrongPointError, analyze_strong_run, write_strong_point_report
from ard.analysis.ffnr_strong_replay import EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256


def _path(value: object, *, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StrongPointError(f"strong point config {name} must be a non-empty path")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def load_frozen_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StrongPointError("strong point config is unreadable") from exc
    if not isinstance(raw, Mapping):
        raise StrongPointError("strong point config must be a mapping")
    required = {
        "schema_version",
        "contract",
        "expected_count",
        "stable_id_class_universe_sha256",
        "scheduler_stages",
        "anchors",
        "candidate_grid",
        "runs",
    }
    if set(raw) != required or raw.get("schema_version") != 1 or raw.get("contract") != CONTRACT:
        raise StrongPointError("strong point config schema/contract drifted")
    if (
        raw.get("expected_count") != 45000
        or raw.get("stable_id_class_universe_sha256") != EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256
    ):
        raise StrongPointError("strong point config fixed CIFAR stable universe drifted")
    grid, runs = raw.get("candidate_grid"), raw.get("runs")
    if not isinstance(grid, Mapping) or set(grid) != {"deltas_pp", "window_sizes", "thresholds"}:
        raise StrongPointError("strong point config grid schema drifted")
    if not isinstance(runs, Mapping) or set(runs) != {"L2", "L4"}:
        raise StrongPointError("strong point config must bind exactly L2/L4")
    names = {
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
        "expected_count": raw["expected_count"],
        "stable_id_class_universe_sha256": raw["stable_id_class_universe_sha256"],
        "scheduler_stages": raw["scheduler_stages"],
        "anchors": tuple(raw["anchors"]),
        "deltas_pp": tuple(grid["deltas_pp"]),
        "window_sizes": tuple(grid["window_sizes"]),
        "thresholds": tuple(grid["thresholds"]),
        "runs": {},
    }
    for label in ("L2", "L4"):
        run = runs[label]
        if not isinstance(run, Mapping) or set(run) != names:
            raise StrongPointError(f"strong point config runs.{label} path schema drifted")
        parsed["runs"][label] = {
            name: _path(run[name], root=path.parent, name=f"runs.{label}.{name}") for name in names
        }
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_frozen_config(config_path)
    reports = {
        label: analyze_strong_run(
            label=label,
            expected_count=config["expected_count"],
            expected_universe_sha256=config["stable_id_class_universe_sha256"],
            scheduler_stages=config["scheduler_stages"],
            anchors=config["anchors"],
            deltas_pp=config["deltas_pp"],
            window_sizes=config["window_sizes"],
            thresholds=config["thresholds"],
            **config["runs"][label],
        )
        for label in ("L2", "L4")
    }
    paths = write_strong_point_report(output_dir=args.output_dir.resolve(), reports=reports, config_path=config_path)
    print("\n".join(f"{name}={path}" for name, path in sorted(paths.items())))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except StrongPointError as exc:
        raise SystemExit(str(exc)) from exc
