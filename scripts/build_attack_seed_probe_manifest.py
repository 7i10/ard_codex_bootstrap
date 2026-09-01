#!/usr/bin/env python3
"""Build the immutable DAG manifest for attack-seed characterization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PYTHON = "/home/shunsukenaito/.conda/envs/adv/bin/python"
REPO = Path(__file__).resolve().parents[1]
PARENT = {
    1: REPO.parent / "ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1/seed1/s100/epoch-100.pt",
    2: REPO.parent / "ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1/seed2/s100/epoch-100.pt",
}
BASE_CONFIG = {
    1: REPO.parent / "ard-runs/ard_codex_bootstrap/ert-rslad-ordering-probes-v1/shuffle_plus_0-s1/resolved_config.yaml",
    2: REPO.parent / "ard-runs/ard_codex_bootstrap/ert-rslad-ordering-probes-v1/shuffle_plus_0-s2/resolved_config.yaml",
}
GPU_UUIDS = {
    0: "GPU-7ce112db-1ab9-ff86-9810-5bb92e222c2a",
    1: "GPU-9bb01a06-72d2-cc68-eeca-9c2c9daf37ce",
}
DATA_ROOT = "/home/islab/workspace-local/shunsuke.naito/datasets/ard/torchvision"
TEACHER_PATH = (
    "/home/islab/workspace-local/shunsuke.naito/ard_codex_bootstrap/teacher_cache/robustbench/Chen2021LTD_WRN34_10.pt"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job(
    *,
    job_id: str,
    run_id: str,
    command: list[str],
    output: Path,
    deps: list[str],
    gpu: int,
    estimated_work: float,
    identity: dict[str, object],
    required_paths: list[str],
    source_root: Path,
    env: dict[str, str] | None = None,
    gpu_count: int = 1,
) -> dict[str, object]:
    return {
        "job_id": job_id,
        "run_id": run_id,
        "host": "hamster",
        "gpu": gpu if gpu_count else None,
        "gpu_count": gpu_count,
        "command": command,
        "cwd": str(source_root),
        "env": env or {"PYTHONPATH": str(source_root / "src"), "PYTHONUNBUFFERED": "1", "WANDB_MODE": "online"},
        "required_paths": required_paths,
        "output_dir": str(output),
        "completion_marker": "orchestration/completion.json",
        "dependencies": deps,
        "estimated_work": estimated_work,
        "retry_policy": {"max_attempts": 1},
        "scientific_identity": identity,
        "executor": {"type": "local"},
    }


def build_manifest(
    *, source_sha: str, registry: Path, root: Path, fixed_root: Path, output: Path, source_root: Path
) -> dict[str, object]:
    registry_sha = sha256(registry)
    jobs: list[dict[str, object]] = []
    for seed in (1, 2):
        fixed_out = fixed_root / f"seed{seed}"
        jobs.append(
            _job(
                job_id=f"fixed-model-s{seed}",
                run_id=f"ert-rslad-attack-fixed-model-s{seed}",
                command=[
                    PYTHON,
                    # Fixed-model replay is read-only and uses the current
                    # bug-fixed analysis helper.  Training jobs below still
                    # execute from the immutable fork source_root so their
                    # fork_git_sha contract remains exact.
                    str(REPO / "scripts/analysis/ert_rslad_attack_randomness.py"),
                    "fixed-model",
                    "--config",
                    str(BASE_CONFIG[seed]),
                    "--parent",
                    str(PARENT[seed]),
                    "--registry",
                    str(registry),
                    "--output",
                    str(fixed_out),
                    "--seed",
                    str(seed),
                    "--limit",
                    "8192",
                ],
                output=fixed_out,
                deps=[],
                gpu=seed - 1,
                estimated_work=120,
                identity={
                    "method_id": "ert_rslad_attack_seed_fixed_model_v1",
                    "seed": seed,
                    "parent_checkpoint_sha256": sha256(PARENT[seed]),
                    "registry_sha256": registry_sha,
                },
                required_paths=[str(BASE_CONFIG[seed]), str(PARENT[seed]), str(registry)],
                source_root=source_root,
            )
        )
    train_job_ids: list[str] = []
    for seed in (1, 2):
        for index in range(8):
            run = root / f"attack-seed-{index}-s{seed}"
            job_id = f"train-attack-{index}-s{seed}"
            train_job_ids.append(job_id)
            jobs.append(
                _job(
                    job_id=job_id,
                    run_id=f"ert-rslad-attack-seed-{index}-s{seed}",
                    command=[
                        PYTHON,
                        "-m",
                        "ard.cli.train",
                        "--config",
                        str(run / "resolved_config.yaml"),
                        "--resume",
                        str(run / "last.pt"),
                        "--output",
                        str(run),
                    ],
                    output=run,
                    deps=[f"fixed-model-s{seed}"],
                    gpu=seed - 1,
                    estimated_work=15,
                    identity={
                        "method_id": "ert_rslad_attack_seed_probe_v1",
                        "seed": seed,
                        "attack_index": index,
                        "parent_checkpoint_sha256": sha256(PARENT[seed]),
                        "registry_sha256": registry_sha,
                        "epochs": "100-114",
                        "attack_random_start_keying": "sample_keyed_v1",
                    },
                    required_paths=[str(run / "resolved_config.yaml"), str(run / "last.pt")],
                    source_root=source_root,
                )
            )
            endpoint_out = run / "endpoint" / "validation"
            jobs.append(
                _job(
                    job_id=f"endpoint-attack-{index}-s{seed}",
                    run_id=f"ert-rslad-attack-endpoint-{index}-s{seed}",
                    command=[
                        PYTHON,
                        "-m",
                        "ard.cli.ert_stage_a_endpoint",
                        "--config",
                        str(run / "resolved_config.yaml"),
                        "--checkpoint",
                        str(run / "epoch-114.pt"),
                        "--output",
                        str(endpoint_out),
                        "--expected-epoch",
                        "113",
                        "--split",
                        "validation",
                        "--device",
                        "cuda",
                    ],
                    output=endpoint_out,
                    deps=[job_id],
                    gpu=seed - 1,
                    estimated_work=25,
                    identity={
                        "method_id": "ert_rslad_attack_seed_endpoint_v1",
                        "seed": seed,
                        "attack_index": index,
                        "source_training_job": job_id,
                        "attack_identity_sha256": "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2",
                    },
                    required_paths=[str(run / "epoch-114.pt"), str(run / "resolved_config.yaml")],
                    source_root=source_root,
                )
            )
    endpoint_ids = [job["job_id"] for job in jobs if str(job["job_id"]).startswith("endpoint-")]
    fixed_ids = [f"fixed-model-s{seed}" for seed in (1, 2)]
    aggregate_out = root / "aggregate"
    jobs.append(
        _job(
            job_id="aggregate-attack-seed-probe",
            run_id="ert-rslad-attack-seed-probe-aggregate",
            command=[
                PYTHON,
                str(source_root / "scripts/analysis/aggregate_ert_rslad_attack_seed_probe.py"),
                "--registry",
                str(registry),
                "--root",
                str(root),
                "--fixed-root",
                str(fixed_root),
                "--pure-order-results",
                str(REPO / "docs/experiments/ert_rslad_pure_order_probe_results_v5.json"),
                "--output",
                str(output),
                "--report",
                str(REPO / "docs/ERT_RSLAD_ATTACK_RANDOMNESS_CHARACTERIZATION.md"),
            ],
            output=aggregate_out,
            deps=fixed_ids + endpoint_ids,
            gpu=0,
            gpu_count=0,
            estimated_work=1,
            identity={
                "method_id": "ert_rslad_attack_seed_probe_aggregate_v1",
                "registry_sha256": registry_sha,
                "probe_count": 16,
                "endpoint_count": 16,
            },
            required_paths=[str(registry), str(REPO / "docs/experiments/ert_rslad_pure_order_probe_results_v5.json")],
            source_root=source_root,
            env={"PYTHONPATH": str(source_root / "src"), "PYTHONUNBUFFERED": "1"},
        )
    )
    return {
        "schema_version": 1,
        "campaign_id": "ert-rslad-attack-seed-randomness-v1",
        "source": {"git_sha": source_sha},
        "state_path": str(root / "orchestration.state.json"),
        "reservation_root": "/home/shunsukenaito/.cache/ard-experiment-orchestrator/reservations",
        "hosts": {
            "hamster": {
                "backend": "local",
                "python": PYTHON,
                "required_paths": [str(REPO), str(REPO / "src"), DATA_ROOT, TEACHER_PATH],
                "gpus": [{"index": i, "uuid": GPU_UUIDS[i], "throughput": 679.0} for i in (0, 1)],
            }
        },
        "jobs": jobs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=REPO)
    args = parser.parse_args(argv)
    manifest = build_manifest(
        source_sha=args.source_sha,
        registry=args.registry.resolve(),
        root=args.root.resolve(),
        fixed_root=args.fixed_root.resolve(),
        output=args.output.resolve(),
        source_root=args.source_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"path": str(args.output.resolve()), "jobs": len(manifest["jobs"]), "sha256": sha256(args.output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
