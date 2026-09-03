#!/usr/bin/env python3
"""Run one canonical, read-only CE-PGD20 longitudinal-state replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ard.analysis.ert_i100_s2_longitudinal import replay_canonical_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = replay_canonical_state(
        config_path=args.config,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_epoch=args.epoch,
        output_dir=args.output_dir,
        device=torch.device(args.device),
        batch_size=args.batch_size,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
