"""Write the read-only Chen FF/NR Student--Teacher mechanism report."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import multiprocessing as mp
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ard.analysis.ffnr_state_mechanism import (
    CONTRACT,
    FFNRStateMechanismError,
    analyze_run,
    cross_seed_models,
    write_outputs,
)
from ard.analysis.ffnr_strong_replay import EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256


def _path(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FFNRStateMechanismError(f"state-mechanism config {name} must be a non-empty path")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FFNRStateMechanismError("state-mechanism config is unreadable") from exc
    required = {
        "schema_version",
        "contract",
        "expected_count",
        "stable_id_class_universe_sha256",
        "anchors",
        "terminal_epochs",
        "runs",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != required
        or raw.get("schema_version") != 1
        or raw.get("contract") != CONTRACT
    ):
        raise FFNRStateMechanismError("state-mechanism config schema/contract drifted")
    if (
        raw.get("expected_count") != 45000
        or raw.get("stable_id_class_universe_sha256") != EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256
    ):
        raise FFNRStateMechanismError("state-mechanism config fixed stable universe drifted")
    if tuple(raw.get("anchors", ())) != (39, 59, 79) or tuple(raw.get("terminal_epochs", ())) != (189, 194, 199):
        raise FFNRStateMechanismError("state-mechanism config frozen epochs drifted")
    paths = {
        "feature_observations",
        "feature_lineage",
        "outcome_observations",
        "outcome_lineage",
        "online_states",
        "online_lineage",
    }
    runs = raw.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != {"L2", "L4"}:
        raise FFNRStateMechanismError("state-mechanism config requires exactly L2/L4")
    parsed: dict[str, Any] = {
        "expected_count": 45000,
        "stable_id_class_universe_sha256": EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256,
        "runs": {},
    }
    for label in ("L2", "L4"):
        value = runs[label]
        if not isinstance(value, Mapping) or set(value) != paths:
            raise FFNRStateMechanismError(f"state-mechanism config runs.{label} schema drifted")
        parsed["runs"][label] = {name: _path(path.parent, value[name], f"runs.{label}.{name}") for name in paths}
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    # Each replay bundle peaks near the host's 2-GiB worker limit.  Analyze
    # seeds in short-lived child processes so L2 and L4 are never resident at
    # the same time; this changes no rows or estimator and keeps the point
    # analysis deterministic.
    ctx = mp.get_context("spawn")
    reports: dict[str, Any] = {}
    for label in ("L2", "L4"):
        with ctx.Pool(processes=1) as pool:
            reports[label] = pool.apply(
                analyze_run,
                kwds={
                    "label": label,
                    "expected_count": config["expected_count"],
                    "expected_universe_sha256": config["stable_id_class_universe_sha256"],
                    **config["runs"][label],
                },
            )
    models = cross_seed_models(reports)
    paths = write_outputs(
        output_dir=args.output_dir.resolve(), reports=reports, cross_seed_models=models, config_path=config_path
    )
    print("\n".join(f"{name}={path}" for name, path in sorted(paths.items())))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except FFNRStateMechanismError as exc:
        raise SystemExit(str(exc)) from exc
