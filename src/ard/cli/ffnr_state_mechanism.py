"""Write the read-only Chen FF/NR Student--Teacher mechanism report."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ard.analysis.ffnr_state_mechanism import (
    CONTRACT,
    FFNRStateMechanismError,
    _tracked_clean_provenance,
    analyze_run,
    canonical_json,
    cross_seed_models,
    sha256_file,
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


def _bind_intermediate(report: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    # Hash the JSON-normalized representation because integer sample-ID keys
    # become strings on disk; merge must verify exactly the bytes' semantics.
    body = json.loads(json.dumps(report, sort_keys=True))
    binding = {
        "config_sha256": sha256_file(config_path),
        "analysis_provenance": _tracked_clean_provenance(),
    }
    binding["report_sha256"] = __import__("hashlib").sha256(canonical_json(body)).hexdigest()
    body["_intermediate_binding"] = binding
    return body


def _validate_intermediate(
    report: Mapping[str, Any], *, label: str, config: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    if report.get("contract") != CONTRACT or report.get("label") != label:
        raise FFNRStateMechanismError("intermediate report contract/label drifted")
    binding = report.get("_intermediate_binding")
    if not isinstance(binding, Mapping):
        raise FFNRStateMechanismError("intermediate report is not hash-bound")
    if (
        binding.get("config_sha256") != sha256_file(config_path)
        or binding.get("analysis_provenance") != _tracked_clean_provenance()
    ):
        raise FFNRStateMechanismError("intermediate source/config provenance drifted")
    body = dict(report)
    body.pop("_intermediate_binding", None)
    expected_digest = __import__("hashlib").sha256(canonical_json(body)).hexdigest()
    if binding.get("report_sha256") != expected_digest:
        raise FFNRStateMechanismError("intermediate report payload hash drifted")
    identity = report.get("input_identity")
    if not isinstance(identity, Mapping) or not isinstance(identity.get("input_sha256"), Mapping):
        raise FFNRStateMechanismError("intermediate input identity is incomplete")
    run_config = config["runs"][label]
    for name, path in run_config.items():
        if identity["input_sha256"].get(name) != sha256_file(path):
            raise FFNRStateMechanismError(f"intermediate input hash drifted: {label}.{name}")
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--single-run", choices=("L2", "L4"), help="write one compact intermediate report")
    parser.add_argument("--merge-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--l2-report", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--l4-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.merge_only:
        if not args.l2_report or not args.l4_report:
            raise FFNRStateMechanismError("merge-only requires both single-run reports")
        reports = {
            "L2": _validate_intermediate(
                json.loads(args.l2_report.resolve().read_text(encoding="utf-8")),
                label="L2",
                config=config,
                config_path=config_path,
            ),
            "L4": _validate_intermediate(
                json.loads(args.l4_report.resolve().read_text(encoding="utf-8")),
                label="L4",
                config=config,
                config_path=config_path,
            ),
        }
        models = cross_seed_models(reports)
        paths = write_outputs(
            output_dir=args.output_dir.resolve(),
            reports=reports,
            cross_seed_models=models,
            config_path=config_path,
        )
        print("\n".join(f"{name}={path}" for name, path in sorted(paths.items())))
        return 0
    if args.single_run:
        report = analyze_run(
            label=args.single_run,
            expected_count=config["expected_count"],
            expected_universe_sha256=config["stable_id_class_universe_sha256"],
            **config["runs"][args.single_run],
        )
        output_dir = args.output_dir.resolve()
        if output_dir.exists():
            raise FFNRStateMechanismError("refusing to overwrite a single-run intermediate directory")
        output_dir.mkdir(parents=True, exist_ok=False)
        report = _bind_intermediate(report, config_path=config_path)
        (output_dir / "single-report.json").write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
        print(output_dir / "single-report.json")
        return 0
    # Each replay bundle peaks near the host's 2-GiB worker limit.  Analyze
    # seeds in separate interpreter processes so their Arrow buffers are fully
    # released before the next seed starts.  This changes no rows or estimator.
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="ffnr-state-mechanism-") as temp:
        intermediate: dict[str, Path] = {}
        for label in ("L2", "L4"):
            child_dir = Path(temp) / label
            env = dict(os.environ)
            env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ard.cli.ffnr_state_mechanism",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(child_dir),
                    "--single-run",
                    label,
                ],
                check=True,
                env=env,
            )
            intermediate[label] = child_dir / "single-report.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ard.cli.ffnr_state_mechanism",
                "--config",
                str(config_path),
                "--output-dir",
                str(args.output_dir.resolve()),
                "--merge-only",
                "--l2-report",
                str(intermediate["L2"]),
                "--l4-report",
                str(intermediate["L4"]),
            ],
            check=True,
            env=env,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except FFNRStateMechanismError as exc:
        raise SystemExit(str(exc)) from exc
