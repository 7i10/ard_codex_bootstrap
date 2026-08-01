"""Create the fixed five-arm common-state intervention continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.intervention_fork import create_intervention_forks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create strict common-state intervention arm checkpoints.")
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-resolved-config", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--arm-config", type=Path, action="append", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    created = create_intervention_forks(
        parent_checkpoint=args.parent_checkpoint,
        parent_resolved_config=args.parent_resolved_config,
        parent_manifest=args.parent_manifest,
        arm_config_paths=args.arm_config,
        root=Path.cwd(),
    )
    print(json.dumps({name: str(path) for name, path in sorted(created.items())}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
