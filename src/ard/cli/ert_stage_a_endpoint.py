"""Run the independent CE-PGD20 endpoint for one Stage A checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ard.analysis.ert_stage_a_endpoint import evaluate_endpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, default=84)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    result = evaluate_endpoint(
        config_path=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output,
        device=torch.device(args.device),
        expected_epoch=args.expected_epoch,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
