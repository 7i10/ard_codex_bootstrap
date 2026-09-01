#!/usr/bin/env python3
"""Run the fixed endpoint set for one I100 transfer continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ard.analysis.ert_stage_a_endpoint import evaluate_endpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    outputs = []
    for epoch in (104, 109, 114):
        outputs.append(
            evaluate_endpoint(
                config_path=args.config,
                checkpoint=args.checkpoint_root / f"epoch-{epoch}.pt",
                output_dir=args.output_root / f"e{epoch}-validation",
                device=torch.device(args.device),
                expected_epoch=epoch,
                split="validation",
            )
        )
    outputs.append(
        evaluate_endpoint(
            config_path=args.config,
            checkpoint=args.checkpoint_root / "epoch-114.pt",
            output_dir=args.output_root / "e114-train",
            device=torch.device(args.device),
            expected_epoch=114,
            split="train",
        )
    )
    summary = {
        "contract": "ert_rslad_i100_action_transfer_endpoint_v1",
        "outputs": outputs,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
