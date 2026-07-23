#!/usr/bin/env python3
"""Create production-gate evidence from the three completed pilot outputs.

This command is intentionally the only supported producer of pilot acceptance
evidence.  It derives every check from durable campaign/training/evaluation
artifacts and binds those artifacts by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from ard.campaign.schema import JobSpec, bind_git_sha, load_campaign  # noqa: E402
from ard.campaign.worker import default_phase_success_validator  # noqa: E402
from ard.config import load_config  # noqa: E402
from ard.config.loader import resolved_config_dict  # noqa: E402
from ard.engine import config_digest  # noqa: E402


class PilotAcceptanceError(RuntimeError):
    """A required pilot observation is absent, stale, or scientifically invalid."""


@dataclass(frozen=True)
class RequiredPilot:
    job_id: str
    host: str
    teacher: str
    method: str


REQUIRED_PILOTS = (
    RequiredPilot("pilot-h-chen-rslad-s0", "hamster", "chen2021_ltd_wrn34_10", "rslad"),
    RequiredPilot("pilot-h-chen-joint-s0", "hamster", "chen2021_ltd_wrn34_10", "rslad_joint"),
    RequiredPilot("pilot-f-bart-rslad-s0", "ferret", "bartoldson2024_adversarial_wrn94_16", "rslad"),
)
PROFILE_ID = "ws1_prb128_gb128_localbn_v1"
EXPECTED_JOINT_TARGET_POLICY: dict[str, object] = {
    "id": "teacher_target_uniform_mix",
    "version": 1,
    "risk_transform": "identity",
    "mixing": "uniform",
    "apply_to": "adversarial_student_kd",
    "rho_max": 0.5,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotAcceptanceError(f"invalid JSON artifact: {path}") from exc


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_metrics(path: Path) -> tuple[list[dict[str, Any]], float]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotAcceptanceError(f"training metrics are invalid: {path}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise PilotAcceptanceError(f"training metrics are empty: {path}")
    required = (
        "train_loss",
        "train_clean_accuracy",
        "train_robust_accuracy",
        "val_clean_accuracy",
        "val_pgd_accuracy",
        "train_cuda_peak_reserved_bytes",
    )
    for row in rows:
        for key in required:
            if key not in row or not _finite(row[key]):
                raise PilotAcceptanceError(f"training metric {key} is absent or non-finite: {path}")
    peak_bytes = max(float(row["train_cuda_peak_reserved_bytes"]) for row in rows)
    if peak_bytes <= 0:
        raise PilotAcceptanceError("CUDA pilot did not report positive peak reserved memory")
    return rows, math.ceil(peak_bytes / 1024**2)


def _validate_joint_target_policy(policy: Mapping[str, object] | None) -> None:
    if policy is None or dict(policy) != EXPECTED_JOINT_TARGET_POLICY:
        raise PilotAcceptanceError("joint pilot must use the canonical teacher_target_uniform_mix@1 target policy")


def _validate_joint_signal(path: Path, *, warmup_epochs: int) -> None:
    if warmup_epochs < 1:
        raise PilotAcceptanceError("joint pilot must declare at least one warmup epoch")
    try:
        table = pq.read_table(path, columns=["epoch", "joint_risk", "kd_weight"])
    except Exception as exc:
        raise PilotAcceptanceError(f"joint pilot sample statistics are invalid: {path}") from exc
    epochs = table.column("epoch").to_pylist()
    risks = table.column("joint_risk").to_pylist()
    weights = table.column("kd_weight").to_pylist()
    post_warmup: list[tuple[float, float]] = []
    for epoch, risk, weight in zip(epochs, risks, weights, strict=True):
        if not _finite(epoch):
            raise PilotAcceptanceError("joint pilot contains a non-finite epoch")
        if int(epoch) < warmup_epochs:
            continue
        if not _finite(risk) or not _finite(weight):
            raise PilotAcceptanceError("joint pilot contains non-finite post-warmup signal values")
        post_warmup.append((float(risk), float(weight)))
    if not post_warmup:
        raise PilotAcceptanceError("joint pilot contains no finite post-warmup signal rows")
    post_warmup_risks = [risk for risk, _ in post_warmup]
    if any(risk < 0.0 or risk > 1.0 for risk in post_warmup_risks):
        raise PilotAcceptanceError("joint pilot post-warmup risk is outside [0, 1]")
    if max(post_warmup_risks) <= 0.0 or min(post_warmup_risks) == max(post_warmup_risks):
        raise PilotAcceptanceError("joint pilot risk never became positive and nonconstant after warmup")
    if any(weight != 1.0 for _, weight in post_warmup):
        raise PilotAcceptanceError("teacher_target_uniform_mix requires uniform post-warmup KD weight 1.0")


def _job_for(spec_path: Path, *, sha: str, required: RequiredPilot) -> JobSpec:
    spec = bind_git_sha(load_campaign(spec_path), sha)
    matches = [job for job in spec.jobs if job.id == required.job_id]
    if len(matches) != 1:
        raise PilotAcceptanceError(f"campaign template lacks required pilot: {required.job_id}")
    job = matches[0]
    if (job.host, job.teacher, job.method) != (required.host, required.teacher, required.method):
        raise PilotAcceptanceError(f"required pilot identity drift: {required.job_id}")
    return job


def _accept_one(
    *,
    run_dir: Path,
    job: JobSpec,
    required: RequiredPilot,
    sha: str,
) -> dict[str, Any]:
    run_metadata = _json(run_dir / "control" / "campaign-run.json")
    if not isinstance(run_metadata, dict) or run_metadata.get("git_sha") != sha:
        raise PilotAcceptanceError(f"campaign run SHA mismatch: {run_dir}")
    state_path = run_dir / "state" / "jobs" / f"{job.id}.json"
    state = _json(state_path)
    expected_wandb_id = f"{job.wandb.run_id}-{sha[:7]}"
    if (
        not isinstance(state, dict)
        or state.get("job_id") != job.id
        or state.get("state") != "completed"
        or state.get("identity", {}).get("effective_wandb_run_id") != expected_wandb_id
        or state.get("identity", {}).get("git_sha") != sha
    ):
        raise PilotAcceptanceError(f"pilot did not complete with the expected identity: {job.id}")
    evidence = state.get("evidence")
    if not isinstance(evidence, list) or not any(
        isinstance(item, dict) and item.get("kind") == "process_adopted" for item in evidence
    ):
        raise PilotAcceptanceError(f"pilot lacks detached-process adoption evidence: {job.id}")

    output = run_dir / "outputs" / job.output
    manifest_path = output / "run-bundle" / "manifest.json"
    manifest = _json(manifest_path)
    config_path = output / "resolved_config.yaml"
    config = load_config(config_path)
    expected_config_hash = config_digest(resolved_config_dict(config))
    execution = config.training
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "completed"
        or manifest.get("tracking_mode") != "online"
        or manifest.get("wandb_initialized") is not True
        or manifest.get("run_id") != expected_wandb_id
        or manifest.get("config_hash") != expected_config_hash
        or config.method.id != required.method
        or config.teacher is None
        or config.teacher.registry_id != required.teacher
        or execution.per_rank_batch_size != 128
        or execution.global_batch_size != 128
        or execution.batchnorm_mode != "local_per_rank"
    ):
        raise PilotAcceptanceError(f"pilot config or W&B terminal lineage is invalid: {job.id}")

    _, peak_reserved_mib = _load_metrics(output / "run-bundle" / "metrics.jsonl")
    evaluation_path = output / "evaluation-pgd" / "evaluation-results.json"
    failure = default_phase_success_validator(job, "pgd", run_dir / "outputs")
    if failure is not None:
        raise PilotAcceptanceError(f"pilot PGD evaluation is invalid ({job.id}): {failure}")
    results = _json(evaluation_path)
    assert isinstance(results, list)
    threat_hashes = {item.get("threat_hash") for item in results if isinstance(item, dict)}
    if len(threat_hashes) != 1:
        raise PilotAcceptanceError(f"pilot PGD threat identity is inconsistent: {job.id}")
    threat_hash = next(iter(threat_hashes))
    if not isinstance(threat_hash, str) or len(threat_hash) != 64:
        raise PilotAcceptanceError(f"pilot PGD threat hash is invalid: {job.id}")

    checks = {
        "finite_train_metrics": True,
        "best_last_pgd_10000": True,
        "terminal_lineage": True,
        "wandb_completed": True,
        "process_adoption": True,
        "execution_profile_match": True,
    }
    if required.method == "rslad_joint":
        target_policy = config.method.target_policy
        _validate_joint_target_policy(None if target_policy is None else target_policy.model_dump(mode="json"))
        _validate_joint_signal(
            output / "sample-stats-train.parquet",
            warmup_epochs=config.method.student_policy_warmup_epochs,
        )
        checks["joint_post_warmup_signal_active"] = True
    return {
        "job_id": job.id,
        "state": "completed",
        "output": job.output,
        "wandb_run_id": expected_wandb_id,
        "peak_reserved_mib": peak_reserved_mib,
        "training_manifest_sha256": _sha256(manifest_path),
        "evaluation_results_sha256": _sha256(evaluation_path),
        "job_state_sha256": _sha256(state_path),
        "training_config_hash": expected_config_hash,
        "threat_hash": threat_hash,
        "checks": checks,
    }


def create_acceptance(
    *,
    sha: str,
    hamster_run_dir: Path,
    ferret_run_dir: Path,
    campaign: Path,
) -> dict[str, Any]:
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise PilotAcceptanceError("pilot acceptance requires a lowercase full Git SHA")
    roots = {"hamster": hamster_run_dir.resolve(), "ferret": ferret_run_dir.resolve()}
    pilots: dict[str, dict[str, Any]] = {}
    for required in REQUIRED_PILOTS:
        job = _job_for(campaign, sha=sha, required=required)
        pilots[job.id] = _accept_one(run_dir=roots[required.host], job=job, required=required, sha=sha)
    return {
        "version": 1,
        "status": "accepted",
        "git_sha": sha,
        "execution_profile": PROFILE_ID,
        "pilots": pilots,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--hamster-run-dir", type=Path, required=True)
    parser.add_argument("--ferret-run-dir", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("configs/campaigns/five_gpu_single_process_pilots_v1.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    acceptance = create_acceptance(
        sha=args.sha,
        hamster_run_dir=args.hamster_run_dir,
        ferret_run_dir=args.ferret_run_dir,
        campaign=args.campaign.resolve(),
    )
    _atomic_json(args.output.resolve(), acceptance)
    print(json.dumps({"status": "accepted", "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
