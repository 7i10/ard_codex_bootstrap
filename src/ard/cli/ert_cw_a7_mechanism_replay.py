"""Public no-update replay CLI for the frozen ERT A7 mechanism diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ard.analysis.ert_cw_a7_mechanism_replay import replay_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", choices=("L2", "L4"), required=True)
    parser.add_argument("--arm", choices=("A5", "A6", "A7", "A8"), required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    result = replay_checkpoint(
        config_path=args.config,
        checkpoint=args.checkpoint,
        mask_path=args.mask,
        output_dir=args.output,
        run=args.run,
        arm=args.arm,
        device=torch.device(args.device),
        expected_epoch=args.epoch,
    )
    print(json.dumps({"contract": result["contract"], "rows_sha256": result["rows_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
