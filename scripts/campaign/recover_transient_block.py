#!/usr/bin/env python3
"""Recover one of the explicitly supported pre-science control failures.

The command is intentionally dry-run by default.  It is a control-plane tool:
it neither edits campaign YAML nor touches a scientific output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ard.campaign.gpu import GPUInspectionError, inventory
from ard.campaign.schema import CampaignError, bind_git_sha, load_campaign
from ard.campaign.state import NVIDIA_SMI_ADMISSION_ERROR, CampaignStateStore, StateError


class RecoveryError(RuntimeError):
    pass


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"{label} is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must be a JSON object: {path}")
    return value


def _proc_start_time(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", maxsplit=1)[1].split()
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _proc_cwd(pid: int) -> str | None:
    try:
        return str(Path(f"/proc/{pid}/cwd").resolve())
    except OSError:
        return None


def _proc_argv(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    argv = [part.decode("utf-8", errors="surrogateescape") for part in raw.split(b"\0") if part]
    return argv or None


def _argv_digest(argv: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(argv), separators=(",", ":")).encode()).hexdigest()


def _controller_live(record: dict[str, Any]) -> bool:
    try:
        pid = int(record["pid"])
        start = int(record["start_time_ticks"])
        cwd = str(record["cwd"])
        expected_argv = record["argv"]
        digest = str(record["argv_digest"])
    except (KeyError, TypeError, ValueError):
        return False
    if not isinstance(expected_argv, list) or not all(isinstance(item, str) for item in expected_argv):
        return False
    argv = _proc_argv(pid)
    return (
        _proc_start_time(pid) == start
        and _proc_cwd(pid) == cwd
        and argv == expected_argv
        and argv is not None
        and _argv_digest(argv) == digest
    )


def _fixed_sha(repository: Path, sha: str) -> None:
    try:
        observed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RecoveryError("unable to verify fixed worktree Git SHA") from exc
    if observed != sha:
        raise RecoveryError("fixed worktree HEAD does not match --sha")


def _contained(base: Path, candidate: Path, label: str) -> Path:
    resolved_base = base.resolve()
    resolved = candidate.resolve()
    if resolved == resolved_base or resolved_base not in resolved.parents:
        raise RecoveryError(f"{label} escapes its configured root")
    return resolved


def _release_marker_is_exact(job: Any, spec: Any) -> None:
    reservations = [item for item in spec.reservations if item.active and item.host == job.host and item.gpu == job.gpu]
    for reservation in reservations:
        if reservation.release_marker is None:
            raise RecoveryError(f"active protected reservation has no release marker: {job.id}")
        payload = _read_object(reservation.release_marker, "protected reservation release marker")
        expected = {
            "status": "completed",
            "run_id": reservation.run_id,
            "training_git_sha": reservation.protected_git_sha,
            "execution_profile": reservation.execution_profile,
            "training_sync": "completed",
            "saved_checkpoint_pgd": "completed",
        }
        if payload != expected:
            raise RecoveryError(f"protected reservation release marker is not exact: {job.id}")


def validate(args: argparse.Namespace) -> tuple[Any, CampaignStateStore, list[Any]]:
    repository = args.repository.resolve()
    if not repository.is_dir():
        raise RecoveryError("fixed worktree repository is absent")
    campaign_path = _contained(repository, args.campaign, "campaign path")
    try:
        spec = bind_git_sha(load_campaign(campaign_path), args.sha)
    except CampaignError as exc:
        raise RecoveryError(str(exc)) from exc
    _fixed_sha(repository, spec.git_sha)
    state = CampaignStateStore(args.state_root.resolve())
    state.assert_campaign_identity(spec)
    campaign_state = state.campaign().get("state")
    allowed_campaign_states = (
        {"awaiting_scientific_review"}
        if args.failure_kind == "gpu_inventory"
        else {"armed", "awaiting_scientific_review"}
    )
    if campaign_state not in allowed_campaign_states:
        raise RecoveryError(f"recovery refuses campaign state: {campaign_state!r}")

    selected = []
    by_id = {job.id: job for job in spec.jobs}
    for job_id in args.job:
        job = by_id.get(job_id)
        if job is None:
            raise RecoveryError(f"explicit recovery job is absent from campaign: {job_id}")
        if job.host != args.host:
            raise RecoveryError(f"recovery job belongs to a different host: {job_id}")
        record = state.job(job.id)
        phase = record.get("phase")
        phase_name = phase.get("name") if isinstance(phase, dict) else None
        output = _contained(args.output_root, args.output_root / job.output, "job output")
        if args.failure_kind == "missing_runtime_environment" and phase_name == "autoattack":
            output /= "evaluation-autoattack"
        if output.exists():
            raise RecoveryError(f"scientific output already exists; refusing recovery: {output}")
        _release_marker_is_exact(job, spec)
        selected.append(job)
    if len({job.id for job in selected}) != len(selected):
        raise RecoveryError("recovery job IDs must be unique")

    controller_path = args.controller_record or (args.state_root.resolve().parent / "control" / "controller.json")
    if controller_path.exists():
        record = _read_object(controller_path, "controller record")
        if _controller_live(record):
            raise RecoveryError("matching controller is live; refusing concurrent recovery")

    try:
        snapshots = inventory()
    except GPUInspectionError as exc:
        raise RecoveryError(f"current GPU inventory is unavailable: {exc}") from exc
    present = {snapshot.index for snapshot in snapshots}
    missing = [job.id for job in selected if job.gpu not in present]
    if missing:
        raise RecoveryError(f"assigned GPU is absent from current inventory: {', '.join(missing)}")
    job_ids = [job.id for job in selected]
    if args.failure_kind == "gpu_inventory":
        state.validate_transient_gpu_blocks(job_ids)
    else:
        state.validate_missing_runtime_environment_failures(job_ids)
    return spec, state, selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", choices=("hamster", "ferret"), required=True)
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--controller-record", type=Path)
    parser.add_argument(
        "--failure-kind",
        choices=("gpu_inventory", "missing_runtime_environment"),
        default="gpu_inventory",
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        spec, state, selected = validate(args)
        result: dict[str, Any] = {
            "campaign_id": spec.campaign_id,
            "git_sha": spec.git_sha,
            "host": args.host,
            "jobs": [job.id for job in selected],
            "failure_kind": args.failure_kind,
            "applied": False,
        }
        if args.failure_kind == "gpu_inventory":
            result["inventory_error"] = NVIDIA_SMI_ADMISSION_ERROR
        if args.apply:
            job_ids = [job.id for job in selected]
            if args.failure_kind == "gpu_inventory":
                state.recover_transient_gpu_blocks(job_ids)
            else:
                state.recover_missing_runtime_environment_failures(job_ids)
            if state.campaign().get("state") == "awaiting_scientific_review":
                campaign = state.rearm_after_transient_gpu_recovery()
            else:
                campaign = state.campaign()
            result.update({"applied": True, "campaign_state": campaign["state"]})
        print(json.dumps(result, sort_keys=True))
        return 0
    except (RecoveryError, StateError) as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
