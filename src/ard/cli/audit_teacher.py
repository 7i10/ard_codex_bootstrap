"""CLI for a local-only bounded RobustBench teacher accuracy audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.config.teacher_audit import load_teacher_audit_config
from ard.evaluation.teacher_audit import run_teacher_audit, write_teacher_audit_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit one hash-registered RobustBench teacher on official CIFAR-10 test data."
    )
    parser.add_argument("--config", type=Path, required=True, help="Strict local-only teacher audit YAML.")
    parser.add_argument("overrides", nargs="*", help="Dot-path YAML overrides such as run.device=cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_teacher_audit_config(args.config, args.overrides)
    report = run_teacher_audit(config, project_root=PROJECT_ROOT)
    resolved_path, result_path = write_teacher_audit_artifacts(config, report)
    print(json.dumps({"resolved_config": str(resolved_path), "result": str(result_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
