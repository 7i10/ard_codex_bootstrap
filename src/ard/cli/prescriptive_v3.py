"""Freeze prescriptive-v3 epoch-34-online / epoch-79-state route masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.prescriptive_v3 import (
    PrescriptiveV3Error,
    build_prescriptive_v3_masks,
    create_prescriptive_v3_forks,
    write_prescriptive_v3_arm_configs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    masks = commands.add_parser("masks")
    masks.add_argument("--online-observations", type=Path, required=True)
    masks.add_argument("--online-lineage", type=Path, required=True)
    masks.add_argument("--feature-observations", type=Path, required=True)
    masks.add_argument("--feature-lineage", type=Path, required=True)
    masks.add_argument("--parent-checkpoint", type=Path, required=True)
    masks.add_argument("--output-dir", type=Path, required=True)
    forks = commands.add_parser("fork")
    forks.add_argument("--parent-checkpoint", type=Path, required=True)
    forks.add_argument("--parent-resolved-config", type=Path, required=True)
    forks.add_argument("--parent-manifest", type=Path, required=True)
    forks.add_argument("--artifact-inventory", type=Path, required=True)
    forks.add_argument("--artifact-attestation", type=Path, required=True)
    forks.add_argument("--schedule-spec", type=Path, required=True)
    forks.add_argument("--root", type=Path, required=True)
    forks.add_argument("--arm-config", type=Path, action="append", required=True)
    configs = commands.add_parser("arm-configs")
    configs.add_argument("--delayed-config", type=Path, required=True)
    configs.add_argument("--schedule-spec", type=Path, required=True)
    configs.add_argument("--parent-checkpoint", type=Path, required=True)
    configs.add_argument("--selector-bundle", type=Path, required=True)
    configs.add_argument("--masks-dir", type=Path, required=True)
    configs.add_argument("--config-dir", type=Path, required=True)
    configs.add_argument("--output-root", type=Path, required=True)
    configs.add_argument("--run-prefix", required=True)
    args = parser.parse_args(argv)
    result = (
        build_prescriptive_v3_masks(
            online_observations=args.online_observations.resolve(),
            online_lineage=args.online_lineage.resolve(),
            feature_observations=args.feature_observations.resolve(),
            feature_lineage=args.feature_lineage.resolve(),
            parent_checkpoint=args.parent_checkpoint.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        if args.mode == "masks"
        else write_prescriptive_v3_arm_configs(
            delayed_config=args.delayed_config.resolve(),
            schedule_spec=args.schedule_spec.resolve(),
            parent_checkpoint=args.parent_checkpoint.resolve(),
            selector_bundle=args.selector_bundle.resolve(),
            masks_dir=args.masks_dir.resolve(),
            config_dir=args.config_dir.resolve(),
            output_root=args.output_root.resolve(),
            run_prefix=args.run_prefix,
        )
        if args.mode == "arm-configs"
        else create_prescriptive_v3_forks(
            parent_checkpoint=args.parent_checkpoint.resolve(),
            parent_resolved_config=args.parent_resolved_config.resolve(),
            parent_manifest=args.parent_manifest.resolve(),
            artifact_inventory=args.artifact_inventory.resolve(),
            artifact_attestation=args.artifact_attestation.resolve(),
            schedule_spec=args.schedule_spec.resolve(),
            arm_config_paths=[path.resolve() for path in args.arm_config],
            root=args.root.resolve(),
        )
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except PrescriptiveV3Error as exc:
        raise SystemExit(str(exc)) from exc
