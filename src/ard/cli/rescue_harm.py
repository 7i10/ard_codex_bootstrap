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
    build_v3_checkpoint_inventory,
    merge_epoch_replays,
    merge_v3_epoch_replays,
    replay_inventory,
    replay_v3_inventory,
    report_rescue_harm,
    report_v3_rescue_harm,
    smoke_v3_report,
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
    if parsed not in set(EPOCHS) | set((79, 119, 129, 149)):
        raise argparse.ArgumentTypeError("merge epoch is outside the completed-v2/v3 checkpoint panels")
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
    replay.add_argument("--contract", choices=("v2", "v3"), default="v2")
    replay.add_argument(
        "--epoch", action="append", type=int, choices=EPOCHS + (79, 119, 129, 149), help="Optional replay smoke epoch."
    )
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--manifest", required=True, type=Path)
    inventory.add_argument("--resolved-config", required=True, type=Path)
    inventory.add_argument("--arm", required=True)
    inventory.add_argument("--seed", required=True, type=int)
    inventory.add_argument("--output", required=True, type=Path)
    inventory.add_argument("--contract", choices=("v2", "v3"), default="v2")
    inventory.add_argument("--epoch", action="append", type=int, choices=EPOCHS + (79, 119, 129, 149))
    inventory.add_argument("--shared-parent-checkpoint", type=Path)
    merge = sub.add_parser("merge")
    merge.add_argument("--observations", required=True, action="append", type=_epoch_path, metavar="EPOCH=PATH")
    merge.add_argument("--lineage", required=True, action="append", type=_epoch_path, metavar="EPOCH=PATH")
    merge.add_argument("--output-observations", required=True, type=Path)
    merge.add_argument("--output-lineage", required=True, type=Path)
    merge.add_argument("--contract", choices=("v2", "v3"), default="v2")
    report = sub.add_parser("report")
    for option in ("--observations", "--lineage"):
        report.add_argument(option, required=True, action="append", type=_labeled_path, metavar="ARM=PATH")
    report.add_argument("--mask-bundle", type=Path)
    report.add_argument("--feature-observations", type=Path)
    report.add_argument("--feature-lineage", type=Path)
    report.add_argument("--parent-checkpoint", type=_labeled_path, metavar="LABEL=PATH")
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--expected-count", required=True, type=int)
    report.add_argument("--contract", choices=("v2", "v3"), default="v2")
    report.add_argument("--smoke-epoch", type=int, choices=(79, 99, 119, 129, 149, 199))
    smoke = sub.add_parser("smoke-report")
    smoke.add_argument("--observations", required=True, type=Path)
    smoke.add_argument("--lineage", required=True, type=Path)
    smoke.add_argument("--arm", required=True, choices=("C", "PF-H", "PF-R", "NR-H", "NR-R"))
    smoke.add_argument("--epoch", required=True, type=int, choices=(79, 99, 119, 129, 149, 199))
    smoke.add_argument("--expected-count", required=True, type=int)
    smoke.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.mode == "inventory":
        if args.contract == "v2":
            if args.epoch is not None or args.shared_parent_checkpoint is not None:
                raise RescueHarmError("v2 inventory has a fixed completed-v2 epoch panel")
            value = build_checkpoint_inventory(
                manifest=args.manifest.resolve(),
                resolved_config=args.resolved_config.resolve(),
                arm=args.arm,
                seed=args.seed,
                output=args.output.resolve(),
            )
        else:
            arm = {"C": "C", "PF-H": "PF-H", "PF-R": "PF-R", "NR-H": "NR-H", "NR-R": "NR-R"}.get(args.arm)
            if arm is None:
                raise RescueHarmError("v3 inventory arm must be C/PF-H/PF-R/NR-H/NR-R")
            value = build_v3_checkpoint_inventory(
                manifest=args.manifest.resolve(),
                arm=arm,
                seed=args.seed,
                output=args.output.resolve(),
                epochs=(79, 99, 119, 129, 149, 199) if args.epoch is None else tuple(sorted(set(args.epoch))),
                shared_parent_checkpoint=None
                if args.shared_parent_checkpoint is None
                else args.shared_parent_checkpoint.resolve(),
            )
    elif args.mode == "replay":
        value = (
            replay_inventory(
                resolved_config=args.resolved_config.resolve(),
                inventory_path=args.inventory.resolve(),
                output_parquet=args.observations.resolve(),
                output_lineage=args.lineage.resolve(),
                device=torch.device(args.device),
                batch_size=args.batch_size,
                analysis_seed=args.analysis_seed,
                epochs=EPOCHS if args.epoch is None else tuple(args.epoch),
            )
            if args.contract == "v2"
            else replay_v3_inventory(
                resolved_config=args.resolved_config.resolve(),
                inventory_path=args.inventory.resolve(),
                output_parquet=args.observations.resolve(),
                output_lineage=args.lineage.resolve(),
                device=torch.device(args.device),
                batch_size=args.batch_size,
                analysis_seed=args.analysis_seed,
                expected_epochs=None if args.epoch is None else tuple(sorted({79, *args.epoch})),
                emit_epochs=None if args.epoch is None else tuple(sorted(set(args.epoch))),
            )
        )
    elif args.mode == "smoke-report":
        value = smoke_v3_report(
            observations=args.observations.resolve(),
            lineage=args.lineage.resolve(),
            arm=args.arm,
            epoch=args.epoch,
            expected_count=args.expected_count,
            output=args.output.resolve(),
        )
    elif args.mode == "merge":
        observations, lineages = dict(args.observations), dict(args.lineage)
        expected_epochs = EPOCHS if args.contract == "v2" else (79, 99, 119, 129, 149, 199)
        if (
            len(observations) != len(args.observations)
            or len(lineages) != len(args.lineage)
            or set(observations) != set(expected_epochs)
            or set(lineages) != set(expected_epochs)
        ):
            raise RescueHarmError("merge requires one unique observation and lineage path for every contract epoch")
        kwargs = {
            "inputs": {epoch: (observations[epoch].resolve(), lineages[epoch].resolve()) for epoch in expected_epochs},
            "output_parquet": args.output_observations.resolve(),
            "output_lineage": args.output_lineage.resolve(),
        }
        value = merge_epoch_replays(**kwargs) if args.contract == "v2" else merge_v3_epoch_replays(**kwargs)
    else:
        observation, lineage = dict(args.observations), dict(args.lineage)
        if args.contract == "v3":
            if (
                set(observation) != set({"C", "PF-H", "PF-R", "NR-H", "NR-R"})
                or set(lineage) != set(observation)
                or len(observation) != len(args.observations)
                or len(lineage) != len(args.lineage)
            ):
                raise RescueHarmError("v3 report requires one C/PF-H/PF-R/NR-H/NR-R observation/lineage pair")
            value = report_v3_rescue_harm(
                observations={arm: (observation[arm].resolve(), lineage[arm].resolve()) for arm in observation},
                output=args.output.resolve(),
                expected_count=args.expected_count,
                report_epochs=(args.smoke_epoch,) if args.smoke_epoch is not None else (79, 99, 119, 129, 149, 199),
            )
            print(json.dumps({"contract": value["contract"]}, sort_keys=True))
            return 0
        if (
            args.mask_bundle is None
            or args.feature_observations is None
            or args.feature_lineage is None
            or args.parent_checkpoint is None
        ):
            raise RescueHarmError("v2 report requires mask bundle, feature panel, and parent checkpoint")
        if args.smoke_epoch is not None:
            raise RescueHarmError("--smoke-epoch is a v3 report-only flag")
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
