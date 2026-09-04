#!/usr/bin/env python3
"""Build the immutable local DAG for the Online-State S2×T1 screen.

This is deliberately a manifest builder, not a launcher.  The production
launch gate performs all source/input/GPU validation and hands the resulting
immutable manifest to the detached generic orchestrator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/shunsukenaito/.conda/envs/adv/bin/python")
RUN_ROOT = Path("/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1")
RUNTIME_ROOT = Path("/home/shunsukenaito/workspace-local/ard-runtime/ard_codex_bootstrap")
DATASET_ROOT = Path("/home/shunsukenaito/workspace-local/datasets/ard/torchvision")
TEACHER = Path(
    "/home/shunsukenaito/workspace-local/ard_codex_bootstrap/teacher_cache/robustbench/Chen2021LTD_WRN34_10.pt"
)
WORKSPACE_REGISTRY = ROOT / "configs/workspace/ard_workspace_v1.json"
CALIBRATION = ROOT / "docs/experiments/ert_rslad_i100_s2_dynamic_bdd_calibration_v1.json"

PARENTS = {
    "dev-1": {
        "config": RUN_ROOT / "idbh-s100-s1/resolved_config.yaml",
        "checkpoint": RUN_ROOT / "seed1/s100/epoch-100.pt",
        "sha256": "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
        "config_sha256": "e87905b549f741de04576bc36c00c51c0ac464f832dd89d7b5c855349926c5a6",
        "gpu": 0,
        "gpu_uuid": "GPU-7ce112db-1ab9-ff86-9810-5bb92e222c2a",
    },
    "dev-2": {
        "config": RUN_ROOT / "idbh-s100-s2/resolved_config.yaml",
        "checkpoint": RUN_ROOT / "seed2/s100/epoch-100.pt",
        "sha256": "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
        "config_sha256": "1ce45685583518a66872b92f40a23c62129b85c6a07c01f50c134b36481dcd5c",
        "gpu": 1,
        "gpu_uuid": "GPU-9bb01a06-72d2-cc68-eeca-9c2c9daf37ce",
    },
}
ARMS = ("control", "pmp", "dbdp")

TEACHER_SHA256 = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"
CALIBRATION_SHA256 = "37bf0a0e1aa6ff12951f1c05f59f6df55700be0e28291c6925670d7b6cb56840"
TRAIN_ATTACK_SHA256 = "97a41870008f5946af3b10dd0d7f145324fe5265b12d3c523bf3f8d099623d4d"
ENDPOINT_ATTACK_SHA256 = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
VALIDATION_SPLIT_SHA256 = "16ec66fbcdeae0b70261589b1ba5f1e7fd4128743ce0194eabc5bea53a0cc6c4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_clean_source(expected: str) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if head != expected:
        raise SystemExit(f"source SHA mismatch: expected {expected}, got {head}")
    if dirty:
        raise SystemExit("manifest builder requires the committed, clean production source")


def _require_registered_inputs() -> None:
    for path, expected, label in (
        (CALIBRATION, CALIBRATION_SHA256, "calibration"),
        (TEACHER, TEACHER_SHA256, "Teacher"),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise SystemExit(f"registered {label} bytes are unavailable or changed: {path}")
    for seed, parent in PARENTS.items():
        for field, expected in (("checkpoint", parent["sha256"]), ("config", parent["config_sha256"])):
            path = Path(str(parent[field]))
            if not path.is_file() or _sha256(path) != expected:
                raise SystemExit(f"registered {seed} {field} bytes are unavailable or changed: {path}")
    for path in (PYTHON, DATASET_ROOT, WORKSPACE_REGISTRY):
        if not path.exists():
            raise SystemExit(f"required local production input is unavailable: {path}")


def _command(*args: str) -> list[str]:
    return [str(PYTHON), *args]


def _identity(*, seed: str, arm: str, parent_sha256: str, role: str) -> dict[str, Any]:
    return {
        "method_id": "ert_rslad_i100_online_state_s2_t1_preservation_v1",
        "role": role,
        "seed_bundle": seed,
        "arm": arm,
        "parent_checkpoint_sha256": parent_sha256,
        "training_attack_identity_sha256": TRAIN_ATTACK_SHA256,
        "endpoint_attack_identity_sha256": ENDPOINT_ATTACK_SHA256,
        "validation_split_sha256": VALIDATION_SPLIT_SHA256,
        "calibration_sha256": CALIBRATION_SHA256,
        "teacher_sha256": TEACHER_SHA256,
    }


def _base_job(
    *,
    job_id: str,
    seed: str,
    arm: str,
    role: str,
    output_dir: Path,
    command: list[str],
    dependencies: list[str],
    estimated_work: float,
    job_type: str,
    expected_outputs: list[dict[str, Any]],
    inputs: list[dict[str, Any]] | None = None,
    parent: dict[str, Any] | None = None,
    calibration: list[dict[str, Any]] | None = None,
    gpu_count: int = 1,
    pin_to_seed_gpu: bool = True,
    epoch_binding: dict[str, int] | None = None,
) -> dict[str, Any]:
    parent_record = PARENTS[seed]
    return {
        "job_id": job_id,
        "run_id": f"ert-i100-online-state-s2-v1-{seed}-{arm}",
        "seed": seed,
        "arm": arm,
        "host": "hamster",
        # State-mutating prefix/arm jobs remain pinned to their seed's GPU.
        # Immutable read-only endpoint/replay jobs can use either Hamster GPU
        # after their dependency is complete, avoiding artificial idle time.
        "gpu": parent_record["gpu"] if gpu_count and pin_to_seed_gpu else None,
        "gpu_count": gpu_count,
        "cwd": str(ROOT),
        "command": command,
        "env": {
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": "online",
        },
        "output_dir": str(output_dir),
        "dependencies": dependencies,
        "estimated_work": estimated_work,
        "work_unit": "measured_training_image_equivalents_v1",
        "retry_policy": {"max_attempts": 1},
        "attempt_scoped_output": {"enabled": True},
        "job_type": job_type,
        "scientific_config_path": str(parent_record["config"]),
        "config_sha256": parent_record["config_sha256"],
        "parent": parent
        or {
            "path": str(parent_record["checkpoint"]),
            "sha256": parent_record["sha256"],
        },
        "attack": "train_kl10" if role in {"prefix", "threshold", "arm"} else "endpoint_ce20",
        "inputs": inputs or [],
        "calibration": calibration or [],
        "expected_outputs": expected_outputs,
        "epoch_binding": epoch_binding,
        "scientific_identity": _identity(seed=seed, arm=arm, parent_sha256=parent_record["sha256"], role=role),
    }


def build_manifest(*, source_sha: str, campaign_root: Path, requested_at: str) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    script = ROOT / "scripts/run_ert_i100_online_state_s2.py"
    endpoint_script = ROOT / "scripts/run_ert_i100_online_state_s2_endpoints.py"
    replay_script = ROOT / "scripts/replay_ert_i100_online_state_s2_train.py"
    aggregate_script = ROOT / "scripts/aggregate_ert_i100_online_state_s2.py"
    for seed in PARENTS:
        prefix_root = campaign_root / "prefix" / seed
        threshold_root = campaign_root / "thresholds" / seed
        prefix_id = f"prefix-{seed}"
        threshold_id = f"threshold-{seed}"
        jobs.append(
            _base_job(
                job_id=prefix_id,
                seed=seed,
                arm="prefix",
                role="prefix",
                output_dir=prefix_root,
                command=_command(
                    str(script),
                    "--expected-source-sha",
                    source_sha,
                    "prefix",
                    "--seed",
                    seed,
                    "--output",
                    "{attempt_output_dir}",
                    "--device",
                    "cuda",
                    "--epochs",
                    "101",
                ),
                dependencies=[],
                estimated_work=45_000.0,
                job_type="training",
                epoch_binding={"scientific_start_epoch": 100, "scientific_final_epoch": 100},
                expected_outputs=[
                    {"path": "prefix-summary.json"},
                    {"path": "training/checkpoints/epoch-100.pt"},
                ],
            )
        )
        jobs.append(
            _base_job(
                job_id=threshold_id,
                seed=seed,
                arm="threshold",
                role="threshold",
                output_dir=threshold_root,
                command=_command(
                    str(script),
                    "--expected-source-sha",
                    source_sha,
                    "freeze",
                    "--seed",
                    seed,
                    "--prefix-output",
                    str(prefix_root),
                    "--output",
                    "{attempt_output_dir}/frozen-thresholds.json",
                ),
                dependencies=[prefix_id],
                estimated_work=30.0,
                job_type="collection",
                inputs=[
                    {
                        "kind": "dependency_output",
                        "producer_job_id": prefix_id,
                        "path": str(prefix_root / "training/checkpoints/epoch-100.pt"),
                    }
                ],
                expected_outputs=[
                    {"path": "frozen-thresholds.json"},
                    {"path": "frozen-thresholds.json.summary.json"},
                ],
            )
        )
        for arm in ARMS:
            arm_root = campaign_root / "arms" / seed / arm
            arm_id = f"arm-{seed}-{arm}"
            jobs.append(
                _base_job(
                    job_id=arm_id,
                    seed=seed,
                    arm=arm,
                    role="arm",
                    output_dir=arm_root,
                    command=_command(
                        str(script),
                        "--expected-source-sha",
                        source_sha,
                        "arm",
                        "--seed",
                        seed,
                        "--arm",
                        arm,
                        "--prefix-checkpoint",
                        str(prefix_root / "training/checkpoints/epoch-100.pt"),
                        "--thresholds",
                        str(threshold_root / "frozen-thresholds.json"),
                        "--output",
                        "{attempt_output_dir}",
                        "--device",
                        "cuda",
                        "--epochs",
                        "115",
                    ),
                    dependencies=[prefix_id, threshold_id],
                    estimated_work={"control": 630_000.0, "pmp": 630_000.0, "dbdp": 1_260_000.0}[arm],
                    job_type="training",
                    epoch_binding={"scientific_start_epoch": 101, "scientific_final_epoch": 114},
                    inputs=[
                        {
                            "kind": "dependency_output",
                            "producer_job_id": prefix_id,
                            "path": str(prefix_root / "training/checkpoints/epoch-100.pt"),
                        },
                        {
                            "kind": "dependency_output",
                            "producer_job_id": threshold_id,
                            "path": str(threshold_root / "frozen-thresholds.json"),
                        },
                    ],
                    calibration=[
                        {
                            "path": str(CALIBRATION),
                            "sha256": CALIBRATION_SHA256,
                            "contract": "ert_rslad_i100_s2_dynamic_bdd_calibration_v1",
                        }
                    ],
                    expected_outputs=[
                        {"path": "arm-summary.json"},
                        {"path": "training/checkpoints/epoch-114.pt"},
                        {"path": "training/online-state-manifest.json"},
                    ],
                )
            )
            endpoint_root = campaign_root / "endpoints" / seed / arm
            endpoint_id = f"endpoint-{seed}-{arm}"
            jobs.append(
                _base_job(
                    job_id=endpoint_id,
                    seed=seed,
                    arm=arm,
                    role="endpoint",
                    output_dir=endpoint_root,
                    command=_command(
                        str(endpoint_script),
                        "--seed",
                        seed,
                        "--arm",
                        arm,
                        "--training-root",
                        str(arm_root),
                        "--output",
                        "{attempt_output_dir}",
                        "--expected-source-sha",
                        source_sha,
                        "--device",
                        "cuda",
                    ),
                    dependencies=[arm_id],
                    estimated_work=30_000.0,
                    job_type="collection",
                    pin_to_seed_gpu=False,
                    inputs=[
                        {
                            "kind": "dependency_output",
                            "producer_job_id": arm_id,
                            "path": str(arm_root / "arm-summary.json"),
                        }
                    ],
                    expected_outputs=[{"path": "summary.json"}],
                )
            )
            canonical_root = campaign_root / "canonical" / seed / arm
            canonical_id = f"canonical-{seed}-{arm}"
            jobs.append(
                _base_job(
                    job_id=canonical_id,
                    seed=seed,
                    arm=arm,
                    role="canonical_train_replay",
                    output_dir=canonical_root,
                    command=_command(
                        str(replay_script),
                        "--seed",
                        seed,
                        "--arm",
                        arm,
                        "--training-root",
                        str(arm_root),
                        "--output",
                        "{attempt_output_dir}",
                        "--expected-source-sha",
                        source_sha,
                        "--device",
                        "cuda",
                    ),
                    dependencies=[arm_id],
                    estimated_work=90_000.0,
                    job_type="collection",
                    pin_to_seed_gpu=False,
                    inputs=[
                        {
                            "kind": "dependency_output",
                            "producer_job_id": arm_id,
                            "path": str(arm_root / "arm-summary.json"),
                        }
                    ],
                    expected_outputs=[
                        {"path": "state-replay.json"},
                        {"path": "state-rows.parquet"},
                    ],
                )
            )

    terminal_dependencies = [job["job_id"] for job in jobs if job["job_id"].startswith(("endpoint-", "canonical-"))]
    jobs.append(
        {
            "job_id": "aggregate",
            "run_id": "ert-i100-online-state-s2-v1-aggregate",
            "seed": "pooled-dev-1-dev-2",
            "arm": "aggregate",
            "host": "hamster",
            "gpu_count": 0,
            "cwd": str(ROOT),
            "command": _command(
                str(aggregate_script),
                "--campaign",
                str(campaign_root),
                "--expected-source-sha",
                source_sha,
            ),
            "env": {"PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1", "WANDB_MODE": "online"},
            "output_dir": str(campaign_root / "aggregate"),
            "dependencies": terminal_dependencies,
            "estimated_work": 240.0,
            "work_unit": "estimated_cpu_seconds",
            "retry_policy": {"max_attempts": 1},
            "job_type": "aggregation",
            "scientific_config_path": str(PARENTS["dev-1"]["config"]),
            "config_sha256": PARENTS["dev-1"]["config_sha256"],
            "parent": {"path": str(PARENTS["dev-1"]["checkpoint"]), "sha256": PARENTS["dev-1"]["sha256"]},
            "attack": "endpoint_ce20",
            "calibration": [{"path": str(CALIBRATION), "sha256": CALIBRATION_SHA256}],
            "scientific_identity": {
                "method_id": "ert_rslad_i100_online_state_s2_t1_preservation_v1",
                "role": "aggregation",
                "seeds": list(PARENTS),
                "arms": list(ARMS),
                "training_attack_identity_sha256": TRAIN_ATTACK_SHA256,
                "endpoint_attack_identity_sha256": ENDPOINT_ATTACK_SHA256,
            },
        }
    )
    return {
        "schema_version": 1,
        "campaign_id": "ert-i100-online-state-s2-v1",
        "operational_profile": "FULL_NEW_INTEGRATION",
        "integration_changes": {
            "new_trainer_execution_path": True,
            "new_checkpoint_serialization": True,
            "new_artifact_schema": True,
        },
        "timing": {
            "request_received": requested_at,
            "request_precision": "recorded_after_reconciliation; original user-turn timestamp unavailable to builder",
        },
        "source": {"git_sha": source_sha, "repo_path": str(ROOT)},
        "workspace_contract": {"registry": str(WORKSPACE_REGISTRY), "enforce_future_writes": True},
        "state_path": str(campaign_root / "orchestration" / "state.json"),
        "reservation_root": str(RUNTIME_ROOT / "locks"),
        "orchestration_root": str(campaign_root / "orchestration"),
        "dataset": {
            "identity": "cifar10_train_validation_45k_5k_v1",
            "split_identity": VALIDATION_SPLIT_SHA256,
            "host_paths": {"hamster": str(DATASET_ROOT)},
        },
        "teacher": {
            "identity": "Chen2021LTD_WRN34_10",
            "checkpoint": {"path": str(TEACHER), "sha256": TEACHER_SHA256},
        },
        "training": {"scientific_start_epoch": 100, "scientific_final_epoch": 114},
        "attacks": {
            "train_kl10": {
                "loss": "kl_teacher_clean",
                "epsilon": "8/255",
                "step_size": "2/255",
                "steps": 10,
                "random_start": True,
                "target": "teacher_clean_target",
                "identity_sha256": TRAIN_ATTACK_SHA256,
            },
            "endpoint_ce20": {
                "loss": "hard_label_ce",
                "epsilon": "8/255",
                "step_size": "2/255",
                "steps": 20,
                "random_start": True,
                "target": "ground_truth",
                "identity_sha256": ENDPOINT_ATTACK_SHA256,
            },
        },
        "rng_contract": "sample_keyed_kl10_v1; fixed parent restore; no reset at online branch",
        "augmentation_identity": "I100_CropShift_prefix_then_IDBH_WEAK; unchanged after e100",
        "hosts": {
            "hamster": {
                "backend": "local",
                "repo_path": str(ROOT),
                "python": str(PYTHON),
                "required_paths": [str(ROOT), str(DATASET_ROOT), str(TEACHER), str(RUN_ROOT)],
                "dataset_paths": {"cifar10_train_validation_45k_5k_v1": str(DATASET_ROOT)},
                "teacher_paths": {"Chen2021LTD_WRN34_10": str(TEACHER)},
                "gpus": [
                    {"index": 0, "uuid": PARENTS["dev-1"]["gpu_uuid"], "throughput": 679.1},
                    {"index": 1, "uuid": PARENTS["dev-2"]["gpu_uuid"], "throughput": 679.1},
                ],
                "preflight_command": ["nvidia-smi", "-L"],
            }
        },
        "canary": {
            "require_exact_smoke": False,
            "static_cli": [
                {
                    "job_id": "prefix-dev-1",
                    # Static CLI checks run outside a job's environment by
                    # design.  Bind the public script to the same import root
                    # as its production argv instead of relying on Codex's
                    # ambient shell environment.
                    "commands": [["/usr/bin/env", f"PYTHONPATH={ROOT / 'src'}", *_command(str(script), "--help")]],
                    "timeout_seconds": 30,
                    "parallel_safe": False,
                }
            ],
            "jobs": [
                {
                    "job_id": "prefix-dev-1",
                    "kind": "bounded_canary",
                    "command": _command(
                        str(script),
                        "--expected-source-sha",
                        source_sha,
                        "canary",
                        "--seed",
                        "dev-1",
                        "--output",
                        str(campaign_root / "gate-canary"),
                        "--device",
                        "cuda",
                    ),
                    "timeout_seconds": 120,
                }
            ],
        },
        "jobs": jobs,
        "scientific_contract": {
            "shared_prefix": "one treatment-free e100 prefix per development seed",
            "selector": "pre-update current Online-S2×T1 with e100 frozen q10 thresholds",
            "arms": ["I100_CONTROL", "OS_PMP", "OS_DBDP"],
            "excluded": [
                "S-BDP",
                "Clean-Wrong",
                "S3",
                "new seeds",
                "e115_to_e199",
                "official_test",
                "AutoAttack",
            ],
            "ferret": (
                "preflight passed but excluded: fresh cross-host output collection cannot be SHA-bound "
                "before the output exists under the current immutable gate contract"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=RUNTIME_ROOT / "runs" / "ert-i100-online-state-s2-v1",
    )
    parser.add_argument("--output", type=Path, required=True, help="new JSON manifest path under the runtime root")
    parser.add_argument(
        "--request-received",
        default=datetime.now(UTC).isoformat(),
        help="ISO-8601 timestamp recorded after reconciliation when the original turn timestamp is unavailable",
    )
    args = parser.parse_args()
    _require_clean_source(args.expected_source_sha)
    _require_registered_inputs()
    output = args.output.resolve()
    campaign_root = args.campaign_root.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {output}")
    if campaign_root.exists():
        raise SystemExit(f"refusing to reuse existing campaign root: {campaign_root}")
    payload = build_manifest(
        source_sha=args.expected_source_sha,
        campaign_root=campaign_root,
        requested_at=args.request_received,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "campaign_root": str(campaign_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
