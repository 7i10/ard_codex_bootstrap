#!/usr/bin/env python3
"""Finish the protected Ferret ws2 run before releasing its GPUs."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


class FinalizationError(RuntimeError):
    pass


def _manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"invalid manifest: {path}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"manifest must be an object: {path}")
    return value


def _synced_terminal(output: Path) -> bool:
    manifest = _manifest(output / "run-bundle" / "manifest.json")
    return manifest.get("status") == "completed" and manifest.get("sync_state") in {None, "synced"}


def _validate_evaluation(output: Path) -> None:
    if not _synced_terminal(output):
        raise FinalizationError("evaluation W&B lineage is not terminal and synced")
    try:
        results = json.loads((output / "evaluation-results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError("evaluation results are absent or invalid") from exc
    if not isinstance(results, list) or len(results) != 2:
        raise FinalizationError("evaluation must contain exactly best and last")
    if {row.get("checkpoint") for row in results if isinstance(row, dict)} != {"best.pt", "last.pt"}:
        raise FinalizationError("evaluation checkpoint set is not best/last")
    for row in results:
        if (
            not isinstance(row, dict)
            or row.get("count") != 10_000
            or not all(
                isinstance(row.get(key), (int, float))
                and not isinstance(row.get(key), bool)
                and math.isfinite(float(row[key]))
                for key in ("clean_accuracy", "pgd_accuracy")
            )
        ):
            raise FinalizationError("evaluation metrics are incomplete or non-finite")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    old_repo = run_dir / "repo"
    outputs = run_dir / "outputs"
    train_output = outputs / "cifar10-r18-rslad-chen2021-ltd-wrn34-10-production-s0"
    evaluation_output = train_output / "evaluation-pgd"
    exit_code = run_dir / "control" / "exit_code"
    release_marker = run_dir / "control" / "campaign-release-ready.json"
    while not exit_code.is_file():
        time.sleep(args.interval_seconds)
    if int(exit_code.read_text(encoding="utf-8").strip()) != 0:
        raise FinalizationError("protected training run did not exit successfully")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(old_repo / "src"),
            "CUDA_VISIBLE_DEVICES": "0",
            "ARD_CIFAR10_ROOT": "/home/shunsukenaito/workspace-local/datasets/ard/torchvision",
            "ARD_OUTPUT_ROOT": str(outputs),
            "ARD_SEED": "0",
            "ARD_PER_RANK_BATCH_SIZE": "64",
            "ARD_NUM_WORKERS": "4",
            "ARD_DEVICE": "cuda",
            "WANDB_ENTITY": "shunsuke-n-waseda-university",
            "WANDB_PROJECT": "single-teacher-ard",
            "WANDB_GROUP_CHEN": "chen-cifar10-r18",
            "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT": str(
                old_repo / "teacher_cache" / "robustbench" / "Chen2021LTD_WRN34_10.pt"
            ),
            "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT_SHA256": (
                "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"
            ),
        }
    )
    sync = [sys.executable, str(old_repo / "scripts" / "sync_wandb.py"), "--root", str(outputs)]
    if subprocess.run(sync, cwd=old_repo, env=environment).returncode != 0 or not _synced_terminal(train_output):
        raise FinalizationError("protected training W&B sync failed")
    if not evaluation_output.exists():
        command = [
            sys.executable,
            "-m",
            "ard.cli.evaluate",
            "--config",
            "configs/production/cifar10_r18_rslad_chen2021_ltd_wrn34_10.yaml",
            "--checkpoint-dir",
            str(train_output),
            "--output",
            str(evaluation_output),
        ]
        if subprocess.run(command, cwd=old_repo, env=environment).returncode != 0:
            raise FinalizationError("protected saved-checkpoint evaluation failed")
    if subprocess.run(sync, cwd=old_repo, env=environment).returncode != 0:
        raise FinalizationError("protected evaluation W&B sync failed")
    _validate_evaluation(evaluation_output)
    temporary = release_marker.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "chen-rslad-production-s0-0ca90ad",
                "training_git_sha": "0ca90ad3d48fe019151363b00c6da2160d64eb99",
                "execution_profile": "ws2_prb64_gb128_localbn",
                "training_sync": "completed",
                "saved_checkpoint_pgd": "completed",
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(release_marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
