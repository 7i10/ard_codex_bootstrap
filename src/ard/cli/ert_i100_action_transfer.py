"""CLI for the fixed I100 historical-action transfer screen preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.ert_i100_action_transfer import build_masks, calibrate, replay


def _seed_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dev1-config", type=Path, required=True)
    parser.add_argument("--dev1-checkpoint", type=Path, required=True)
    parser.add_argument("--dev2-config", type=Path, required=True)
    parser.add_argument("--dev2-checkpoint", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    replay_cmd = sub.add_parser("replay", help="Replay the fixed epoch-99 parent.")
    replay_cmd.add_argument("--config", type=Path, required=True)
    replay_cmd.add_argument("--checkpoint", type=Path, required=True)
    replay_cmd.add_argument("--expected-sha", required=True)
    replay_cmd.add_argument("--output", type=Path, required=True)
    replay_cmd.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    replay_cmd.add_argument("--batch-size", type=int, default=128)
    replay_cmd.add_argument("--max-batches", type=int)

    masks_cmd = sub.add_parser("masks", help="Build fixed masks from two complete replays.")
    masks_cmd.add_argument("--dev1-replay", type=Path, required=True)
    masks_cmd.add_argument("--dev2-replay", type=Path, required=True)
    masks_cmd.add_argument("--output", type=Path, required=True)

    cal_cmd = sub.add_parser("calibrate", help="Run pooled no-update gradient calibration.")
    _seed_paths(cal_cmd)
    cal_cmd.add_argument("--dev1-replay", type=Path, required=True)
    cal_cmd.add_argument("--dev2-replay", type=Path, required=True)
    cal_cmd.add_argument("--dev1-mask", type=Path, required=True)
    cal_cmd.add_argument("--dev2-mask", type=Path, required=True)
    cal_cmd.add_argument("--output", type=Path, required=True)
    cal_cmd.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    cal_cmd.add_argument("--batch-size", type=int, default=64)
    cal_cmd.add_argument("--max-per-cohort", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "replay":
        result = replay(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            expected_sha=args.expected_sha,
            output_dir=args.output,
            device=args.device,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
    elif args.command == "masks":
        result = build_masks(
            replay_dirs={"dev-1": args.dev1_replay, "dev-2": args.dev2_replay}, output_dir=args.output
        )
    else:
        result = calibrate(
            config_paths={"dev-1": args.dev1_config, "dev-2": args.dev2_config},
            checkpoint_paths={"dev-1": args.dev1_checkpoint, "dev-2": args.dev2_checkpoint},
            replay_dirs={"dev-1": args.dev1_replay, "dev-2": args.dev2_replay},
            mask_paths={"dev-1": args.dev1_mask, "dev-2": args.dev2_mask},
            output=args.output,
            device=args.device,
            batch_size=args.batch_size,
            max_per_cohort=args.max_per_cohort,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
