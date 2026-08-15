"""Replay frozen ERT S3 history rules without training or endpoint evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from ard.analysis.ert_s3_history_replay import HistoryReplayError, analyze_run, load_trajectory, write_report
from ard.tracking.adapter import collect_git_state


def _path(value: object, *, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise HistoryReplayError(f"{name} must be a non-empty path")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def load_frozen_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or raw.get("contract") != "ert_s3_history_replay_v1":
        raise HistoryReplayError("history replay config schema/contract drifted")
    if raw.get("expected_count") != 45000 or tuple(raw.get("epochs", ())) != tuple(range(80, 95)):
        raise HistoryReplayError("history replay config must bind the 45k, epoch-80..94 trajectory")
    runs = raw.get("runs")
    if not isinstance(runs, dict) or set(runs) != {"L2", "L4"}:
        raise HistoryReplayError("history replay config must bind exactly L2 and L4")
    return {
        "expected_count": int(raw["expected_count"]),
        "epochs": tuple(int(epoch) for epoch in raw["epochs"]),
        "runs": {
            label: _path(value.get("trajectory"), root=path.parent, name=f"runs.{label}.trajectory")
            for label, value in runs.items()
            if isinstance(value, dict)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_frozen_config(config_path)
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False or not isinstance(source.get("sha"), str):
        raise HistoryReplayError("history replay requires a clean source tree")
    reports = {
        label: analyze_run(
            label=label,
            trajectory=load_trajectory(
                trajectory_path,
                expected_count=config["expected_count"],
                expected_epochs=config["epochs"],
            ),
        )
        for label, trajectory_path in config["runs"].items()
    }
    result = write_report(
        output=args.output.resolve(), reports=reports, config_path=config_path, source_git_sha=str(source["sha"])
    )
    print(json.dumps({"output": str(args.output.resolve()), "output_sha256": result["output_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (HistoryReplayError, OSError, yaml.YAMLError) as exc:
        raise SystemExit(str(exc)) from exc
