#!/usr/bin/env python3
"""Build the detached, GPU-aware manifest for the 16 pure-order probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PYTHON = "/home/shunsukenaito/.conda/envs/adv/bin/python"
REPO = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-ordering-probes-v1")
REGISTRY = REPO / "docs/experiments/ert_rslad_pure_order_probe_registry_v1.json"
PARENTS = {
    1: "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    2: "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
}
GPU_UUIDS = {
    0: "GPU-7ce112db-1ab9-ff86-9810-5bb92e222c2a",
    1: "GPU-9bb01a06-72d2-cc68-eeca-9c2c9daf37ce",
}
ENV = {"PYTHONPATH": str(REPO / "src"), "PYTHONUNBUFFERED": "1", "WANDB_MODE": "online"}


def train_job(
    schedule_id: str,
    offset: int,
    seed: int,
    *,
    run_root: Path,
    run_id_prefix: str,
) -> dict[str, object]:
    run = run_root / f"{schedule_id.lower()}-s{seed}"
    run_id = f"{run_id_prefix}-{schedule_id.lower()}-s{seed}"
    return {
        "job_id": f"train-{schedule_id.lower()}-s{seed}",
        "run_id": run_id,
        "host": "hamster",
        "command": [
            PYTHON,
            "-m",
            "ard.cli.train",
            "--config",
            str(run / "resolved_config.yaml"),
            "--resume",
            str(run / "last.pt"),
            "--output",
            str(run),
            "--ordering-policy",
            "epoch_shuffle_offset",
            "--ordering-seed-offset",
            str(offset),
        ],
        "cwd": str(REPO),
        "env": ENV,
        "required_paths": [str(run / "last.pt"), str(run / "resolved_config.yaml")],
        "output_dir": str(run),
        "completion_marker": "orchestration/completion.json",
        "dependencies": [],
        "estimated_work": 15,
        "retry_policy": {"max_attempts": 1},
        "scientific_identity": {
            "method_id": "I100_sample_keyed_rng_pure_order_probe",
            "schedule_id": schedule_id,
            "order_policy": "epoch_shuffle_offset",
            "order_seed_offset": offset,
            "seed": seed,
            "parent_checkpoint_sha256": PARENTS[seed],
            "probe_epochs": "100-114",
            "attack_random_start_keying": "sample_keyed_v1",
            "attack_identity": "KL-PGD10|eps=8/255|step=2/255|random_start|teacher_clean_target",
            "telemetry_contract": "ert_rslad_pure_order_probe_telemetry_v1",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--campaign-id", default="ert-rslad-ordering-mechanism-probe-v1")
    parser.add_argument("--run-id-prefix", default="ert-rslad-pure-order")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    registry_path = args.registry.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    jobs: list[dict[str, object]] = []
    for seed in (1, 2):
        for item in registry["schedules"]:
            jobs.append(
                train_job(
                    str(item["schedule_id"]),
                    int(item["order_seed_offset"]),
                    seed,
                    run_root=run_root,
                    run_id_prefix=args.run_id_prefix,
                )
            )
    aggregate_id = "aggregate-pure-order-probes"
    jobs.append(
        {
            "job_id": aggregate_id,
            "run_id": "ert-rslad-pure-order-probe-aggregate",
            "host": "hamster",
            "gpu_count": 0,
            "command": [
                PYTHON,
                "scripts/analysis/aggregate_ert_rslad_pure_order_probes.py",
                "--registry",
                str(registry_path),
                "--root",
                str(run_root),
                "--output",
                str(REPO / "docs/experiments/ert_rslad_pure_order_probe_results_v1.json"),
            ],
            "cwd": str(REPO),
            "env": {"PYTHONPATH": str(REPO / "src"), "PYTHONUNBUFFERED": "1"},
            "required_paths": [str(REGISTRY)],
            "output_dir": str(run_root / "aggregate"),
            "completion_marker": "orchestration/completion.json",
            "dependencies": [job["job_id"] for job in jobs],
            "estimated_work": 1,
            "retry_policy": {"max_attempts": 1},
            "scientific_identity": {
                "method_id": "ert_rslad_pure_order_probe_aggregate_v1",
                "registry_sha256": __import__("hashlib").sha256(registry_path.read_bytes()).hexdigest(),
                "probe_count": 16,
            },
        }
    )
    manifest = {
        "schema_version": 1,
        "campaign_id": args.campaign_id,
        "source": {"git_sha": args.source_sha},
        "state_path": str(run_root / "orchestration.state.json"),
        "reservation_root": "/home/shunsukenaito/.cache/ard-experiment-orchestrator/reservations",
        "hosts": {
            "hamster": {
                "backend": "local",
                "python": PYTHON,
                "required_paths": [
                    str(REPO),
                    "/home/shunsukenaito/workspace-local/datasets/ard/torchvision",
                    "/home/shunsukenaito/workspace-local/ard_codex_bootstrap/teacher_cache/robustbench/Chen2021LTD_WRN34_10.pt",
                ],
                "preflight_command": ["nvidia-smi", "-L"],
                "gpus": [{"index": index, "uuid": uuid, "throughput": 679.0} for index, uuid in GPU_UUIDS.items()],
            }
        },
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(args.output), "jobs": len(jobs), "train_jobs": 16}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
