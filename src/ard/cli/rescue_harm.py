"""Replay one completed-v2 arm or write its paired rescue/harm report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ard.analysis.rescue_harm import (
    ARMS,
    EPOCHS,
    RescueHarmError,
    build_checkpoint_inventory,
    merge_epoch_replays,
    replay_inventory,
    report_rescue_harm,
)


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw = value.partition("=")
    if separator != "=" or not label or not raw:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw)


def _epoch_path(value: str) -> tuple[int, Path]:
    epoch, separator, raw = value.partition("=")
    if separator != "=" or not raw:
        raise argparse.ArgumentTypeError("expected EPOCH=PATH")
    try:
        parsed = int(epoch)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("merge epoch must be an integer") from exc
    if parsed not in EPOCHS:
        raise argparse.ArgumentTypeError("merge epoch must be one of 99,104,109,199")
    return parsed, Path(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--resolved-config", required=True, type=Path)
    replay.add_argument("--inventory", required=True, type=Path)
    replay.add_argument("--observations", required=True, type=Path)
    replay.add_argument("--lineage", required=True, type=Path)
    replay.add_argument("--device", required=True)
    replay.add_argument("--batch-size", required=True, type=int)
    replay.add_argument("--analysis-seed", required=True, type=int)
    replay.add_argument(
        "--epoch", action="append", type=int, choices=EPOCHS, help="Optional one-checkpoint real smoke mode."
    )
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--manifest", required=True, type=Path)
    inventory.add_argument("--resolved-config", required=True, type=Path)
    inventory.add_argument("--arm", required=True, choices=("control", "PF_TA", "PF_R", "NR_TA", "NR_R"))
    inventory.add_argument("--seed", required=True, type=int)
    inventory.add_argument("--output", required=True, type=Path)
    merge = sub.add_parser("merge")
    merge.add_argument("--observations", required=True, action="append", type=_epoch_path, metavar="EPOCH=PATH")
    merge.add_argument("--lineage", required=True, action="append", type=_epoch_path, metavar="EPOCH=PATH")
    merge.add_argument("--output-observations", required=True, type=Path)
    merge.add_argument("--output-lineage", required=True, type=Path)
    report = sub.add_parser("report")
    for option in ("--observations", "--lineage"):
        report.add_argument(option, required=True, action="append", type=_labeled_path, metavar="ARM=PATH")
    report.add_argument("--mask-bundle", required=True, type=Path)
    report.add_argument("--feature-observations", required=True, type=Path)
    report.add_argument("--feature-lineage", required=True, type=Path)
    report.add_argument("--parent-checkpoint", required=True, type=_labeled_path, metavar="LABEL=PATH")
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--expected-count", required=True, type=int)
    args = parser.parse_args(argv)
    if args.mode == "inventory":
        value = build_checkpoint_inventory(
            manifest=args.manifest.resolve(),
            resolved_config=args.resolved_config.resolve(),
            arm=args.arm,
            seed=args.seed,
            output=args.output.resolve(),
        )
    elif args.mode == "replay":
        value = replay_inventory(
            resolved_config=args.resolved_config.resolve(),
            inventory_path=args.inventory.resolve(),
            output_parquet=args.observations.resolve(),
            output_lineage=args.lineage.resolve(),
            device=torch.device(args.device),
            batch_size=args.batch_size,
            analysis_seed=args.analysis_seed,
            epochs=EPOCHS if args.epoch is None else tuple(args.epoch),
        )
    elif args.mode == "merge":
        observations, lineages = dict(args.observations), dict(args.lineage)
        if (
            len(observations) != len(args.observations)
            or len(lineages) != len(args.lineage)
            or set(observations) != set(EPOCHS)
            or set(lineages) != set(EPOCHS)
        ):
            raise RescueHarmError("merge requires one unique observation and lineage path for each fixed epoch")
        value = merge_epoch_replays(
            inputs={epoch: (observations[epoch].resolve(), lineages[epoch].resolve()) for epoch in EPOCHS},
            output_parquet=args.output_observations.resolve(),
            output_lineage=args.output_lineage.resolve(),
        )
    else:
        observation, lineage = dict(args.observations), dict(args.lineage)
        parent_label, parent_checkpoint = args.parent_checkpoint
        if (
            set(observation) != set(ARMS)
            or set(lineage) != set(ARMS)
            or len(observation) != len(args.observations)
            or len(lineage) != len(args.lineage)
            or parent_label not in {"L1", "L3"}
        ):
            raise RescueHarmError("report requires one unique observation/lineage pair and L1/L3 parent checkpoint")
        value = report_rescue_harm(
            observations={arm: (observation[arm].resolve(), lineage[arm].resolve()) for arm in ARMS},
            mask_bundle=args.mask_bundle.resolve(),
            feature_observations=args.feature_observations.resolve(),
            feature_lineage=args.feature_lineage.resolve(),
            parent_checkpoint=parent_checkpoint.resolve(),
            output=args.output.resolve(),
            expected_count=args.expected_count,
        )
    contract = value.get("contract", "completed_v2_checkpoint_inventory_v1")
    print(json.dumps({"contract": contract}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except RescueHarmError as exc:
        raise SystemExit(str(exc)) from exc
