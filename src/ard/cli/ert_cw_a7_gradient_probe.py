"""Run one deterministic no-update A5--A8 gradient probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ard.analysis.ert_cw_a7_gradient_probe import probe_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-rows", type=Path, required=True)
    parser.add_argument("--run", choices=("L2", "L4"), required=True)
    parser.add_argument("--arm", choices=("A5", "A6", "A7", "A8"), required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    result = probe_checkpoint(
        config_path=args.config,
        checkpoint=args.checkpoint,
        probe_rows=args.probe_rows,
        run=args.run,
        arm=args.arm,
        epoch=args.epoch,
        device=torch.device(args.device),
        output=args.output,
    )
    print(json.dumps({"contract": result["contract"], "probe_count": result["probe_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
