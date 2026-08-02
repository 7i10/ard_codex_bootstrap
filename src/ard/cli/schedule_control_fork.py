"""Create an epoch-79 scheduler-only counterfactual checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.schedule_control_fork import create_schedule_control_fork


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a strict delayed-MultiStepLR epoch-79 fork.")
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-resolved-config", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--artifact-inventory", type=Path, required=True)
    parser.add_argument("--artifact-attestation", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True, help="Analysis-owned hash-bound parent lineage YAML.")
    parser.add_argument("--child-config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = create_schedule_control_fork(
        parent_checkpoint=args.parent_checkpoint,
        parent_resolved_config=args.parent_resolved_config,
        parent_manifest=args.parent_manifest,
        artifact_inventory=args.artifact_inventory,
        artifact_attestation=args.artifact_attestation,
        spec_path=args.spec,
        child_config_path=args.child_config,
        root=Path.cwd(),
    )
    print(json.dumps({"checkpoint": str(checkpoint)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
