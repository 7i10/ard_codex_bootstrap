#!/usr/bin/env python3
"""Run the fixed CE-PGD20 endpoints for one long-horizon continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ard.analysis.ert_stage_a_endpoint import evaluate_endpoint


EPOCHS = (129, 149, 169, 189, 199)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    outputs = []
    for epoch in EPOCHS:
        outputs.append(evaluate_endpoint(
            config_path=args.config,
            checkpoint=args.checkpoint_root / f"epoch-{epoch}.pt",
            output_dir=args.output_root / f"e{epoch}-validation",
            device=torch.device(args.device), expected_epoch=epoch, split="validation",
        ))
    outputs.append(evaluate_endpoint(
        config_path=args.config,
        checkpoint=args.checkpoint_root / "epoch-199.pt",
        output_dir=args.output_root / "e199-train",
        device=torch.device(args.device), expected_epoch=199, split="train",
    ))
    payload = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_cw_long_horizon_endpoint_v1",
        "validation_epochs": list(EPOCHS),
        "train_epoch": 199,
        "attack_identity_sha256": outputs[0]["attack_identity_sha256"],
        "outputs": outputs,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["summary_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
