#!/usr/bin/env python3
"""Build the immutable I100 Clean-Wrong long-horizon train/endpoint DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


E114 = {
    "dev-1": {
        "I100_CONTROL": "b4bc74fb1d92b1ad27a7aae305c112f74d63785e9d5d119d76080d281a051c8d",
        "CLEAN_WRONG_PLAIN_ADVCE": "52cda658d826ccf9d0a1d0a5cbda2d7f66c8ec87f00505270d221aabbffbf903",
        "CLEAN_WRONG_A7_MARGIN_ONLY": "9e09227da2c4df648c7342e4c3ddb1c971b5976e3c71513a013c3f8d4b93f11a",
    },
    "dev-2": {
        "I100_CONTROL": "7c8d35f07955820b67d4935cbdc69a37988acacc2bfd3e309b5c1df68a9c4cff",
        "CLEAN_WRONG_PLAIN_ADVCE": "43c8f154963dd5a60caa5abb3d8f161d26cfcd3c43aa4cc0cb3bbb0a876da8e5",
        "CLEAN_WRONG_A7_MARGIN_ONLY": "da24e9395b95b68a14bcfeaf9458a11addbc7988551c9101d51d697de2b4205a6",
    },
}
E99 = {
    "dev-1": "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    "dev-2": "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
}
GPU_UUIDS = {0: "GPU-d6b53af2-a086-be46-30db-10976ef3b989", 1: "GPU-cf338b20-ad89-5bfd-bc79-74610aebc333"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--dev1-config", type=Path, required=True)
    p.add_argument("--dev2-config", type=Path, required=True)
    p.add_argument("--dev1-mask", type=Path, required=True)
    p.add_argument("--dev2-mask", type=Path, required=True)
    args = p.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration.get("status") != "complete_no_update" or float(calibration.get("tau", 2.0)) != 2.0:
        raise ValueError("invalid frozen calibration")
    beta = float(calibration["beta_advce"])
    margin, floor, cap = (float(calibration[k]) for k in ("margin_coefficient", "margin_floor", "margin_cap"))
    root, repo = args.root.resolve(), args.repo.resolve()
    configs = {"dev-1": args.dev1_config.resolve(), "dev-2": args.dev2_config.resolve()}
    masks = {"dev-1": args.dev1_mask.resolve(), "dev-2": args.dev2_mask.resolve()}
    jobs: list[dict[str, Any]] = []
    for seed in ("dev-1", "dev-2"):
        for arm in ("I100_CONTROL", "CLEAN_WRONG_PLAIN_ADVCE", "CLEAN_WRONG_TPFM"):
            source_arm = "CLEAN_WRONG_A7_MARGIN_ONLY" if arm == "CLEAN_WRONG_TPFM" else arm
            # TPFM is the stable long-horizon name for the historical A7
            # checkpoint; its bytes remain the already accepted A7 e114 parent.
            parent = root / "inputs" / f"{seed}-{source_arm}.epoch-114.pt"
            train_out = root / "runs" / seed / arm
            if arm == "I100_CONTROL":
                extra = ["--kind", "baseline"]
            elif arm == "CLEAN_WRONG_PLAIN_ADVCE":
                extra = ["--kind", "advce", "--mask", str(masks[seed]), "--mask-key", "clean_wrong", "--beta-advce", str(beta)]
            else:
                extra = ["--kind", "broad", "--mask", str(masks[seed]), "--mask-key", "clean_wrong",
                         "--margin-target-mode", "teacher_floor", "--margin-coefficient", str(margin),
                         "--margin-floor", str(floor), "--margin-cap", str(cap)]
            jid = f"train-{seed}-{arm.lower()}"
            cmd = ["/home/shunsukenaito/.conda/envs/adv/bin/python", "-m", "ard.cli.ert_stage_a_runtime",
                   "--parent-config", str(configs[seed]), "--parent-checkpoint", str(parent),
                   "--calibration", str(args.calibration.resolve()), "--output", str(train_out), "--arm", arm,
                   "--epochs", "200", "--horizon-epochs", "129", "149", "169", "189", "199",
                   "--run-namespace", "ert-i100-cw-long-horizon-v1", "--resume-epoch", "114", "--mask-anchor-epoch", "99",
                   "--expected-parent-sha256", E114[seed][source_arm], "--device", "cuda", *extra]
            jobs.append({"job_id": jid, "run_id": f"ert-i100-cw-long-{seed}-{arm.lower()}", "host": "ferret", "gpu": 0 if seed == "dev-1" else 1,
                         "command": cmd, "cwd": str(repo), "env": {"PYTHONPATH": "src", "WANDB_MODE": "online"},
                         "required_paths": [str(parent), str(configs[seed]), str(args.calibration.resolve()), str(masks[seed])],
                         "output_dir": str(train_out), "completion_marker": "orchestration/completion.json", "dependencies": [],
                         "estimated_work": 85 * 45000, "scientific_identity": {"method_id": "i100-cw-long-horizon", "seed_bundle": seed, "arm": arm,
                             "source_arm": source_arm, "parent_e114_sha256": E114[seed][source_arm], "parent_e99_sha256": E99[seed],
                             "mask_sha256": sha256(masks[seed]), "calibration_sha256": sha256(args.calibration), "attack": "KL-PGD10-teacher-clean-sample-keyed-v1"},
                         "retry_policy": {"max_attempts": 2}, "executor": {"type": "local"}})
            endpoint_id = f"endpoint-{seed}-{arm.lower()}"
            endpoint_out = train_out / "endpoints"
            endpoint_cmd = ["/home/shunsukenaito/.conda/envs/adv/bin/python", "scripts/run_i100_cw_long_horizon_endpoints.py",
                            "--config", str(configs[seed]), "--checkpoint-root", str(train_out / "checkpoints"),
                            "--output-root", str(endpoint_out), "--device", "cuda"]
            jobs.append({"job_id": endpoint_id, "run_id": f"ert-i100-cw-long-endpoint-{seed}-{arm.lower()}", "host": "ferret",
                         "command": endpoint_cmd, "cwd": str(repo), "env": {"PYTHONPATH": "src", "WANDB_MODE": "online"},
                         "required_paths": [], "output_dir": str(endpoint_out), "completion_marker": "orchestration/completion.json",
                         "dependencies": [jid], "estimated_work": 6 * 50000 * 20, "scientific_identity": {"method_id": "i100-cw-long-endpoint", "seed_bundle": seed,
                             "arm": arm, "parent_job": jid, "source_sha": args.source_sha, "attack": "CE-PGD20-8/255-2/255-random"},
                         "retry_policy": {"max_attempts": 2}, "executor": {"type": "local"}})
    manifest = {"schema_version": 1, "campaign_id": "ert-rslad-i100-cw-long-horizon-v1", "source": {"git_sha": args.source_sha},
                "state_path": str(root / "production.state.json"), "reservation_root": str(root / "locks"), "jobs": jobs,
                "hosts": {"ferret": {"backend": "local", "python": "/home/shunsukenaito/.conda/envs/adv/bin/python",
                    "required_paths": [str(repo), str(root / "inputs"), "/home/shunsukenaito/workspace-local/datasets/ard/torchvision", str(repo / "teacher_cache")],
                    "gpus": [{"index": i, "uuid": GPU_UUIDS[i], "throughput": 600.0 if i == 0 else 607.0} for i in (0, 1)]}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output.resolve()), "jobs": len(jobs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
