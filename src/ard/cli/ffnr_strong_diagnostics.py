"""Write the read-only D3/D4/D5 Chen CE-PGD20 diagnostics report."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ard.analysis.ffnr_strong_diagnostics import CONTRACT, StrongDiagnosticsError, analyze_run, write_outputs


def _path(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StrongDiagnosticsError(f"diagnostics config {name} must be a path")
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StrongDiagnosticsError("diagnostics config is unreadable") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1 or raw.get("contract") != CONTRACT:
        raise StrongDiagnosticsError("diagnostics config contract drifted")
    cifar_root = raw.get("cifar10_train_root")
    runs = raw.get("runs")
    names = {"feature_observations", "feature_lineage", "outcome_observations", "outcome_lineage", "online_states", "online_lineage", "validation_history", "validation_manifest", "dense_chunks"}
    if not isinstance(runs, Mapping) or set(runs) != {"L2", "L4"}:
        raise StrongDiagnosticsError("diagnostics config requires exactly L2/L4")
    parsed: dict[str, Any] = {"runs": {}, "cifar10_train_root": _path(path.parent, cifar_root, "cifar10_train_root")}
    for label in ("L2", "L4"):
        run = runs[label]
        if not isinstance(run, Mapping) or set(run) != names or not isinstance(run["dense_chunks"], list):
            raise StrongDiagnosticsError("diagnostics run config schema drifted")
        parsed["runs"][label] = {name: _path(path.parent, run[name], f"{label}.{name}") for name in names - {"dense_chunks"}}
        parsed["runs"][label]["dense_chunks"] = [
            {key: _path(path.parent, value, f"{label}.dense.{key}") for key, value in chunk.items()}
            for chunk in run["dense_chunks"]
            if isinstance(chunk, Mapping) and set(chunk) == {"observations", "lineage"}
        ]
        if len(parsed["runs"][label]["dense_chunks"]) != len(run["dense_chunks"]):
            raise StrongDiagnosticsError("diagnostics dense chunk schema drifted")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    reports = {label: analyze_run(label=label, **config["runs"][label]) for label in ("L2", "L4")}
    paths = write_outputs(
        output_dir=args.output_dir.resolve(),
        reports=reports,
        config_path=config_path,
        cifar10_train_root=config["cifar10_train_root"],
    )
    print("\n".join(f"{name}={path}" for name, path in sorted(paths.items())))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except StrongDiagnosticsError as exc:
        raise SystemExit(str(exc)) from exc
