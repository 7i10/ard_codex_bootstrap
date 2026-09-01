#!/usr/bin/env python3
"""Build the immutable training/endpoint DAG for the I100 transfer screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PARENTS = {
    "dev-1": ("parent-dev1.pt", "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835"),
    "dev-2": ("parent-dev2.pt", "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--dev1-config", type=Path, required=True)
    parser.add_argument("--dev2-config", type=Path, required=True)
    parser.add_argument("--dev1-mask", type=Path, required=True)
    parser.add_argument("--dev2-mask", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if not isinstance(calibration, dict) or calibration.get("status") != "complete_no_update":
        raise ValueError("calibration artifact is not complete_no_update")
    beta = float(calibration["beta_advce"])
    margin = float(calibration["margin_coefficient"])
    floor = float(calibration["margin_floor"])
    cap = float(calibration["margin_cap"])
    root = args.root.resolve()
    repo = args.repo.resolve()
    calibration = args.calibration.resolve()
    configs = {"dev-1": args.dev1_config.resolve(), "dev-2": args.dev2_config.resolve()}
    masks = {"dev-1": args.dev1_mask.resolve(), "dev-2": args.dev2_mask.resolve()}
    jobs: list[dict[str, Any]] = []
    for seed, (parent_name, parent_sha) in PARENTS.items():
        config = configs[seed]
        mask = masks[seed]
        for arm, extra in (
            ("I100_CONTROL", ["--kind", "baseline"]),
            ("PILOT_S3_T1_WEAK_ADVCE", ["--kind", "advce", "--mask", str(mask), "--mask-key", "pilot_s3_t1", "--beta-advce", str(beta)]),
            ("CLEAN_WRONG_PLAIN_ADVCE", ["--kind", "advce", "--mask", str(mask), "--mask-key", "clean_wrong", "--beta-advce", str(beta)]),
            ("CLEAN_WRONG_A7_MARGIN_ONLY", ["--kind", "broad", "--mask", str(mask), "--mask-key", "clean_wrong", "--margin-target-mode", "teacher_floor", "--margin-coefficient", str(margin), "--margin-floor", str(floor), "--margin-cap", str(cap)]),
        ):
            slug = f"{seed}/{arm}"
            train_out = root / "runs" / seed / arm
            endpoint_out = train_out / "endpoints"
            job_id = f"train-{seed}-{arm.lower()}"
            command = [
                "/home/shunsukenaito/.conda/envs/adv/bin/python", "-m", "ard.cli.ert_stage_a_runtime",
                "--parent-config", str(config), "--parent-checkpoint", str(root / "inputs" / parent_name),
                "--calibration", str(calibration), "--output", str(train_out), "--arm", arm,
                # Trainer.fit uses an exclusive upper bound.  The scientific
                # continuation is epochs 100--114 inclusive, so the runtime
                # endpoint must be 115 while horizon labels remain 104/109/114.
                "--epochs", "115", "--horizon-epochs", "104", "109", "114",
                "--run-namespace", "ert-i100-action-transfer-v1", "--resume-epoch", "99",
                "--mask-anchor-epoch", "99", "--expected-parent-sha256", parent_sha, "--device", "cuda",
                *extra,
            ]
            jobs.append({
                "job_id": job_id,
                "run_id": f"ert-i100-{seed}-{arm.lower()}",
                "host": "ferret",
                "command": command,
                "cwd": str(repo),
                "env": {"PYTHONPATH": "src", "WANDB_MODE": "online"},
                "required_paths": [str(root / "inputs" / parent_name), str(config), str(calibration), str(mask)],
                "output_dir": str(train_out), "completion_marker": "completion.json", "dependencies": [],
                "estimated_work": 15 * 45000, "scientific_identity": {
                    "method_id": "i100-action-transfer", "seed_bundle": seed, "arm": arm,
                    "parent_checkpoint_sha256": parent_sha, "calibration_sha256": sha256(calibration),
                    "source_sha": args.source_sha, "attack": "KL-PGD10-teacher-clean-sample-keyed-v1",
                }, "retry_policy": {"max_attempts": 2}, "executor": {"type": "local"},
                "gpu": 0 if seed == "dev-1" else 1,
            })
            endpoint_job_id = f"endpoint-{seed}-{arm.lower()}"
            endpoint_cmd = [
                "/home/shunsukenaito/.conda/envs/adv/bin/python", "scripts/run_i100_action_transfer_endpoints.py",
                "--config", str(config), "--checkpoint-root", str(train_out / "checkpoints"),
                "--output-root", str(endpoint_out), "--device", "cuda",
            ]
            jobs.append({
                "job_id": endpoint_job_id, "run_id": f"ert-i100-endpoint-{seed}-{arm.lower()}", "host": "ferret",
                "command": endpoint_cmd, "cwd": str(repo), "env": {"PYTHONPATH": "src", "WANDB_MODE": "online"},
                # The checkpoint is a dependency-produced path and is
                # intentionally checked by the endpoint command after the
                # training marker, not by campaign preflight.
                "required_paths": [], "output_dir": str(endpoint_out),
                "completion_marker": "completion.json", "dependencies": [job_id], "estimated_work": 4 * 45000 * 20,
                "scientific_identity": {"method_id": "i100-action-transfer-ce-pgd20", "seed_bundle": seed, "arm": arm,
                    "parent_job": job_id, "source_sha": args.source_sha, "attack": "CE-PGD20-8/255-2/255-random"},
                "retry_policy": {"max_attempts": 2}, "executor": {"type": "local"},
            })
    manifest = {
        "schema_version": 1, "campaign_id": "ert-rslad-i100-action-transfer-v1", "source": {"git_sha": args.source_sha},
        "state_path": str(root / "production.state.json"), "reservation_root": str(root / "locks"),
        "hosts": {"ferret": {"backend": "local", "python": "/home/shunsukenaito/.conda/envs/adv/bin/python",
            "required_paths": [str(repo), str(root / "inputs"), "/home/shunsukenaito/workspace-local/datasets/ard/torchvision", str(repo / "teacher_cache")],
            "gpus": [{"index": 0, "uuid": "GPU-d6b53af2-a086-be46-30db-10976ef3b989", "throughput": 600.0},
                     {"index": 1, "uuid": "GPU-cf338b20-ad89-5bfd-bc79-74610aebc333", "throughput": 607.0}] }},
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output.resolve()), "jobs": len(jobs), "calibration_sha256": sha256(calibration)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
