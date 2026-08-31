#!/usr/bin/env python3
"""Build the immutable four-suffix history-ordering campaign manifest."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-history-ordering-v2-final")
PYTHON = "/home/shunsukenaito/.conda/envs/adv/bin/python"
SOURCE = "aafc5b7b18a557a027d9dcd4b0064bfcaf843404"
ATTACK = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
PARENTS = {1: "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835", 2: "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7"}
ENV = {"PYTHONPATH": str(REPO / "src"), "PYTHONUNBUFFERED": "1", "WANDB_MODE": "online"}


def train_job(seed: int, history: bool) -> dict[str, object]:
    arm = "history" if history else "control"
    name = f"new-{arm}-s{seed}"
    root = RUN_ROOT / name
    config = root / "resolved_config.yaml"
    command = [PYTHON, "-m", "ard.cli.train", "--config", str(config), "--resume", str(root / "last.pt"), "--output", str(root), "--ordering-policy", "history_balanced_v1" if history else "none"]
    return {
        "job_id": f"train-{arm}-s{seed}", "run_id": f"ert-rslad-history-ordering-new-{arm}-s{seed}", "host": "hamster", "command": command, "cwd": str(REPO), "env": ENV, "required_paths": [str(root / "last.pt")], "output_dir": str(root), "completion_marker": "orchestration/completion.json", "dependencies": [], "estimated_work": 100, "retry_policy": {"max_attempts": 2},
        "scientific_identity": {"method_id": "I100_sample_keyed_rng", "arm": "NEW_HISTORY" if history else "NEW_CONTROL", "seed": seed, "ordering_policy": "history_balanced_v1" if history else "epoch_shuffle_control", "parent_checkpoint_sha256": PARENTS[seed], "attack_random_start_keying": "sample_keyed_v1"},
    }


def endpoint_job(seed: int, history: bool, epoch: int) -> dict[str, object]:
    arm = "history" if history else "control"
    name = f"new-{arm}-s{seed}"
    root = RUN_ROOT / name
    endpoint_root = root / "endpoints" / f"epoch-{epoch}" / "validation"
    command = [PYTHON, "-m", "ard.cli.ert_stage_a_endpoint", "--config", str(root / "resolved_config.yaml"), "--checkpoint", str(root / f"epoch-{epoch}.pt"), "--output", str(endpoint_root), "--expected-epoch", str(epoch - 1), "--split", "validation", "--device", "cuda"]
    return {
        "job_id": f"endpoint-{arm}-s{seed}-e{epoch}", "run_id": f"ert-rslad-history-ordering-endpoint-{arm}-s{seed}-e{epoch}", "host": "hamster", "command": command, "cwd": str(REPO), "env": ENV, "required_paths": [], "output_dir": str(endpoint_root), "completion_marker": "orchestration/completion.json", "dependencies": [f"train-{arm}-s{seed}"], "estimated_work": 10, "retry_policy": {"max_attempts": 2},
        "scientific_identity": {"method_id": "CE-PGD20", "arm": "NEW_HISTORY" if history else "NEW_CONTROL", "seed": seed, "endpoint_epoch": epoch, "attack_identity": ATTACK},
    }


def main() -> None:
    jobs: list[dict[str, object]] = []
    for seed in (1, 2):
        for history in (False, True):
            jobs.append(train_job(seed, history))
    for seed in (1, 2):
        for history in (False, True):
            jobs.extend(endpoint_job(seed, history, epoch) for epoch in (149, 199))
    jobs.append({
        "job_id": "aggregate-report", "run_id": "ert-rslad-history-ordering-aggregate-report", "host": "hamster", "gpu_count": 0, "command": [PYTHON, "scripts/aggregate_history_ordering_campaign.py"], "cwd": str(REPO), "env": {"PYTHONPATH": str(REPO / "src"), "PYTHONUNBUFFERED": "1"}, "output_dir": str(RUN_ROOT / "aggregate"), "completion_marker": "orchestration/completion.json", "dependencies": [job["job_id"] for job in jobs if str(job["job_id"]).startswith("endpoint-")], "estimated_work": 1, "scientific_identity": {"method_id": "aggregate_report", "campaign": "ert-rslad-history-ordering-dev-v2"},
    })
    manifest = {
        "schema_version": 1, "campaign_id": "ert-rslad-history-ordering-dev-v2", "source": {"git_sha": SOURCE, "scientific_source_note": "Frozen implementation source commit; fork lineage records the exact clean source used for materialization."}, "state_path": str(RUN_ROOT / "orchestration.state.json"), "reservation_root": "/home/shunsukenaito/.cache/ard-experiment-orchestrator/reservations", "hosts": {"hamster": {"backend": "local", "python": PYTHON, "required_paths": [str(REPO), "/home/shunsukenaito/workspace-local/datasets/ard/torchvision", "/home/shunsukenaito/workspace-local/ard_codex_bootstrap/teacher_cache/robustbench/Chen2021LTD_WRN34_10.pt"], "preflight_command": ["nvidia-smi", "-L"], "gpus": [{"index": 0, "uuid": "GPU-7ce112db-1ab9-ff86-9810-5bb92e222c2a", "throughput": 679.0}, {"index": 1, "uuid": "GPU-9bb01a06-72d2-cc68-eeca-9c2c9daf37ce", "throughput": 679.0}]}}, "jobs": jobs,
    }
    path = REPO / "docs/experiments/ert_rslad_history_balanced_ordering_dev_v2_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "jobs": len(jobs)}, sort_keys=True))


if __name__ == "__main__":
    main()
