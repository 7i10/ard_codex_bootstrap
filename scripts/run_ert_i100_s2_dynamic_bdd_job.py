#!/usr/bin/env python3
"""Run one boundary-distance continuation and its fixed CE-PGD20 endpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from ard.analysis.ert_stage_a_endpoint import evaluate_endpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", choices=("dev-1", "dev-2"), required=True)
    parser.add_argument("--arm", choices=("control", "dpm", "dbdd", "sbdd"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=115)
    args = parser.parse_args()
    if args.epochs != 115:
        raise SystemExit("dynamic-BDD screen is frozen to runtime epochs=115")
    args.output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_ert_i100_s2_dynamic_bdd_arm.py")
    command = [
        sys.executable, str(runner), "--seed", args.seed, "--arm", args.arm,
        "--output", str(args.output / "training"), "--device", args.device,
    ]
    completed = subprocess.run(command)
    if completed.returncode != 0:
        return completed.returncode
    config = args.output / "training" / "resolved_config.yaml"
    checkpoint_root = args.output / "training" / "checkpoints"
    endpoint_root = args.output / "endpoints"
    endpoint_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for epoch in (104, 109, 114):
        outputs.append(
            evaluate_endpoint(
                config_path=config,
                checkpoint=checkpoint_root / f"epoch-{epoch}.pt",
                output_dir=endpoint_root / f"e{epoch}-validation",
                device=torch.device(args.device), expected_epoch=epoch, split="validation",
            )
        )
    outputs.append(
        evaluate_endpoint(
            config_path=config,
            checkpoint=checkpoint_root / "epoch-114.pt",
            output_dir=endpoint_root / "e114-train",
            device=torch.device(args.device), expected_epoch=114, split="train",
        )
    )
    summary = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_dynamic_bdd_endpoint_v1",
        "seed": args.seed,
        "arm": args.arm,
        "outputs": outputs,
    }
    (endpoint_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "production-summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "ert_rslad_i100_s2_dynamic_bdd_job_v1",
                "seed": args.seed,
                "arm": args.arm,
                "training_result": str((args.output / "training" / "horizon-checkpoints.json").resolve()),
                "endpoint_summary": str((endpoint_root / "summary.json").resolve()),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
