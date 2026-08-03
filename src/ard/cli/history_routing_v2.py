"""Build immutable epoch-39 online-history masks before any v2 continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.history_routing_v2 import build_history_routing_v2_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the strict epoch-39 online-history routing v2 masks.")
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-resolved-config", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--train-partition-manifest", type=Path, required=True)
    parser.add_argument("--train-partition-manifest-sha256", required=True)
    parser.add_argument("--train-partition-ids-labels-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = build_history_routing_v2_bundle(
        parent_checkpoint=args.parent_checkpoint,
        parent_resolved_config=args.parent_resolved_config,
        parent_manifest=args.parent_manifest,
        train_partition_manifest=args.train_partition_manifest,
        train_partition_manifest_sha256=args.train_partition_manifest_sha256,
        train_partition_ids_labels_sha256=args.train_partition_ids_labels_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps({name: str(path) for name, path in sorted(paths.items())}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
