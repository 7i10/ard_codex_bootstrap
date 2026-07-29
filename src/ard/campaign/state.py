"""Atomic JSON state, append-only events, and advisory campaign locks."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .schema import CampaignError, CampaignSpec, campaign_identity, campaign_identity_sha256, effective_wandb_run_id


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class JobState(StrEnum):
    PENDING = "pending"
    PREFLIGHT = "preflight"
    WAITING_DEPENDENCY = "waiting_dependency"
    WAITING_GPU = "waiting_gpu"
    WAITING_FOR_MEMORY = "waiting_for_memory"
    LAUNCHING = "launching"
    TRAINING = "training"
    TRAINING_COMPLETED = "training_completed"
    PGD_EVALUATION = "pgd_evaluation"
    PGD_COMPLETED = "pgd_completed"
    AUTOATTACK = "autoattack"
    COMPLETED = "completed"
    PGD_COMPLETED_AUTOATTACK_FAILED = "pgd_completed_autoattack_failed"
    FAILED = "failed"
    BLOCKED = "blocked"


TERMINAL_JOB_STATES = frozenset(
    {JobState.COMPLETED, JobState.PGD_COMPLETED_AUTOATTACK_FAILED, JobState.FAILED, JobState.BLOCKED}
)

_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.PENDING: frozenset({JobState.PREFLIGHT, JobState.BLOCKED}),
    JobState.PREFLIGHT: frozenset(
        {
            JobState.WAITING_DEPENDENCY,
            JobState.WAITING_GPU,
            JobState.WAITING_FOR_MEMORY,
            JobState.LAUNCHING,
            JobState.BLOCKED,
        }
    ),
    JobState.WAITING_DEPENDENCY: frozenset({JobState.PREFLIGHT, JobState.BLOCKED}),
    JobState.WAITING_GPU: frozenset({JobState.PREFLIGHT, JobState.BLOCKED}),
    JobState.WAITING_FOR_MEMORY: frozenset({JobState.PREFLIGHT, JobState.BLOCKED}),
    JobState.LAUNCHING: frozenset(
        {JobState.TRAINING, JobState.PGD_EVALUATION, JobState.AUTOATTACK, JobState.FAILED, JobState.BLOCKED}
    ),
    JobState.TRAINING: frozenset({JobState.TRAINING_COMPLETED, JobState.FAILED, JobState.BLOCKED}),
    JobState.TRAINING_COMPLETED: frozenset(
        {
            JobState.WAITING_DEPENDENCY,
            JobState.WAITING_GPU,
            JobState.WAITING_FOR_MEMORY,
            JobState.LAUNCHING,
            JobState.PGD_EVALUATION,
            JobState.FAILED,
            JobState.BLOCKED,
        }
    ),
    JobState.PGD_EVALUATION: frozenset({JobState.PGD_COMPLETED, JobState.FAILED, JobState.BLOCKED}),
    JobState.PGD_COMPLETED: frozenset(
        {
            JobState.WAITING_DEPENDENCY,
            JobState.WAITING_GPU,
            JobState.WAITING_FOR_MEMORY,
            JobState.LAUNCHING,
            JobState.AUTOATTACK,
            JobState.COMPLETED,
            JobState.BLOCKED,
        }
    ),
    JobState.AUTOATTACK: frozenset({JobState.COMPLETED, JobState.PGD_COMPLETED_AUTOATTACK_FAILED, JobState.BLOCKED}),
    JobState.COMPLETED: frozenset(),
    JobState.PGD_COMPLETED_AUTOATTACK_FAILED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.BLOCKED: frozenset(),
}


class StateError(CampaignError):
    pass


# This is deliberately the exact ``repr`` produced by the conservative
# nvidia-smi admission failure in ``campaign.gpu``.  Recovery may not turn a
# generic controller or launch failure into a new training attempt.
NVIDIA_SMI_ADMISSION_ERROR = "GPUInspectionError('nvidia-smi inventory failed; refusing GPU admission')"
MISSING_RUNTIME_ENVIRONMENT_ERROR = "ValueError: missing environment variables: ARD_CIFAR10_ROOT"


class FileLock:
    """A local advisory lock with a timeout-free nonblocking mode."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def acquire(self, *, blocking: bool = True) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(self._handle.fileno(), flags)
        except BlockingIOError:
            self._handle.close()
            self._handle = None
            return False
        return True

    def release(self) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None

    def __enter__(self) -> FileLock:
        if not self.acquire():  # pragma: no cover - blocking acquire cannot fail normally
            raise StateError(f"unable to acquire lock: {self.path}")
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, sort_keys=True, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"invalid durable state file: {path}") from exc
    if not isinstance(value, dict):
        raise StateError(f"state file must be a JSON object: {path}")
    return value


_SAFE_LOCK = re.compile(r"^[A-Za-z0-9_.-]+$")


class CampaignStateStore:
    """One state root; every state mutation is serialized by the host lock."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.campaign_path = root / "campaign.json"
        self.events_path = root / "events.jsonl"
        self.jobs_path = root / "jobs"
        self.locks_path = root / "locks"

    @property
    def host_lock(self) -> FileLock:
        return FileLock(self.locks_path / "host.lock")

    def gpu_lock(self, gpu_uuid: str) -> FileLock:
        if not _SAFE_LOCK.fullmatch(gpu_uuid):
            raise StateError("unsafe GPU UUID for lock path")
        return FileLock(self.locks_path / f"gpu-{gpu_uuid}.lock")

    def initialize(self, spec: CampaignSpec) -> None:
        identity = campaign_identity(spec)
        with self.host_lock:
            if self.campaign_path.exists():
                self.assert_campaign_identity(spec)
                return
            self.jobs_path.mkdir(parents=True, exist_ok=True)
            self.locks_path.mkdir(parents=True, exist_ok=True)
            now = utc_now()
            campaign = {
                "version": 1,
                "identity": identity,
                "identity_sha256": campaign_identity_sha256(spec),
                "state": "unarmed",
                "created_at": now,
                "updated_at": now,
            }
            _atomic_json(self.campaign_path, campaign)
            for job in spec.jobs:
                _atomic_json(
                    self.jobs_path / f"{job.id}.json",
                    {
                        "version": 1,
                        "job_id": job.id,
                        "state": JobState.PENDING.value,
                        "identity": {
                            "campaign_id": spec.campaign_id,
                            "git_sha": spec.git_sha,
                            "execution_profile": spec.execution_profile.model_dump(mode="json"),
                            "host": job.host,
                            "gpu": job.gpu,
                            "output": job.output,
                            "wandb": job.wandb.model_dump(mode="json"),
                            "effective_wandb_run_id": effective_wandb_run_id(spec, job),
                        },
                        "created_at": now,
                        "updated_at": now,
                        "revision": 0,
                    },
                )
            self._append_event_locked({"kind": "campaign_initialized", "campaign_id": spec.campaign_id})

    def assert_campaign_identity(self, spec: CampaignSpec) -> None:
        existing = _read_json(self.campaign_path)
        identity_matches = existing.get("identity") == campaign_identity(spec)
        digest_matches = existing.get("identity_sha256") == campaign_identity_sha256(spec)
        if not identity_matches or not digest_matches:
            raise StateError("campaign identity or execution profile drift is forbidden")

    def campaign(self) -> dict[str, Any]:
        return _read_json(self.campaign_path)

    def set_campaign_state(self, state: str) -> None:
        if state not in {"unarmed", "armed", "awaiting_scientific_review"}:
            raise StateError(f"unknown campaign state: {state}")
        with self.host_lock:
            campaign = _read_json(self.campaign_path)
            old = campaign.get("state")
            if not isinstance(old, str):
                raise StateError("campaign state is invalid")
            valid = {"unarmed": {"armed"}, "armed": {"awaiting_scientific_review"}, "awaiting_scientific_review": set()}
            if old == state:
                return
            if state not in valid.get(old, set()):
                raise StateError(f"invalid campaign transition {old!r} -> {state!r}")
            campaign["state"] = state
            campaign["updated_at"] = utc_now()
            _atomic_json(self.campaign_path, campaign)
            self._append_event_locked({"kind": "campaign_transition", "from": old, "to": state})

    def job(self, job_id: str) -> dict[str, Any]:
        return _read_json(self.jobs_path / f"{job_id}.json")

    def jobs(self) -> dict[str, dict[str, Any]]:
        if not self.jobs_path.exists():
            return {}
        return {path.stem: _read_json(path) for path in sorted(self.jobs_path.glob("*.json"))}

    def transition_job(self, job_id: str, target: JobState, **updates: Any) -> dict[str, Any]:
        with self.host_lock:
            return self._transition_job_locked(job_id, target, **updates)

    def _transition_job_locked(self, job_id: str, target: JobState, **updates: Any) -> dict[str, Any]:
        path = self.jobs_path / f"{job_id}.json"
        job = _read_json(path)
        try:
            current = JobState(job["state"])
        except (KeyError, ValueError) as exc:
            raise StateError(f"job {job_id} has an invalid state") from exc
        if current != target and target not in _TRANSITIONS[current]:
            raise StateError(f"invalid job transition {current.value} -> {target.value}")
        if current == target and not updates:
            return job
        job.update(updates)
        job["state"] = target.value
        job["updated_at"] = utc_now()
        job["revision"] = int(job.get("revision", 0)) + 1
        _atomic_json(path, job)
        self._append_event_locked(
            {
                "kind": "job_transition",
                "job_id": job_id,
                "from": current.value,
                "to": target.value,
                "revision": job["revision"],
            }
        )
        return job

    def append_evidence(self, job_id: str, kind: str, value: dict[str, Any]) -> None:
        with self.host_lock:
            path = self.jobs_path / f"{job_id}.json"
            job = _read_json(path)
            evidence = list(job.get("evidence", []))
            evidence.append({"at": utc_now(), "kind": kind, "value": value})
            job["evidence"] = evidence
            job["updated_at"] = utc_now()
            job["revision"] = int(job.get("revision", 0)) + 1
            _atomic_json(path, job)
            self._append_event_locked({"kind": "evidence", "job_id": job_id, "evidence_kind": kind})

    def recover_transient_gpu_blocks(self, job_ids: tuple[str, ...] | list[str]) -> dict[str, dict[str, Any]]:
        """Atomically requeue explicitly named, never-launched inventory blocks.

        This is intentionally narrower than a general terminal-state reset:
        only the exact nvidia-smi admission error may be recovered, and a job
        that has any phase/launch evidence is never eligible.  All records are
        validated while holding the host lock before writing any job file.
        """
        requested = tuple(job_ids)
        with self.host_lock:
            records = self._validate_transient_gpu_blocks_locked(requested)

            recovered: dict[str, dict[str, Any]] = {}
            now = utc_now()
            for job_id, job in records.items():
                prior = {
                    "state": job["state"],
                    "failure": job["failure"],
                    "inventory_error": job["inventory_error"],
                    "revision": job.get("revision", 0),
                    "recovered_at": now,
                }
                history = list(job.get("recovery_history", []))
                history.append(prior)
                job["recovery_history"] = history
                # Keep the failed admission available in immutable history,
                # but do not let it masquerade as the reason for a later
                # state transition.
                job.pop("failure", None)
                job.pop("inventory_error", None)
                job["state"] = JobState.PREFLIGHT.value
                job["updated_at"] = now
                job["revision"] = int(job.get("revision", 0)) + 1
                _atomic_json(self.jobs_path / f"{job_id}.json", job)
                self._append_event_locked(
                    {
                        "kind": "transient_gpu_block_recovered",
                        "job_id": job_id,
                        "from": JobState.BLOCKED.value,
                        "to": JobState.PREFLIGHT.value,
                        "revision": job["revision"],
                    }
                )
                recovered[job_id] = job
            return recovered

    def validate_transient_gpu_blocks(self, job_ids: tuple[str, ...] | list[str]) -> dict[str, dict[str, Any]]:
        """Validate a recovery dry run under the same lock as its mutation."""
        with self.host_lock:
            return self._validate_transient_gpu_blocks_locked(tuple(job_ids))

    def _validate_transient_gpu_blocks_locked(self, requested: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        if not requested or len(set(requested)) != len(requested):
            raise StateError("recovery requires one or more unique explicit job IDs")
        records: dict[str, dict[str, Any]] = {}
        for job_id in requested:
            path = self.jobs_path / f"{job_id}.json"
            if not path.is_file():
                raise StateError(f"recovery job is absent: {job_id}")
            job = _read_json(path)
            if job.get("state") != JobState.BLOCKED.value:
                raise StateError(f"recovery job is not blocked: {job_id}")
            if job.get("failure") != "GPU inventory unavailable":
                raise StateError(f"recovery job has a non-transient failure: {job_id}")
            if job.get("inventory_error") != NVIDIA_SMI_ADMISSION_ERROR:
                raise StateError(f"recovery job has an unexpected inventory error: {job_id}")
            # A blocked preflight has no launch intent.  Any of these fields
            # proves recovery could duplicate a phase or obscure output
            # lineage.
            if any(key in job for key in ("phase", "launch_intent", "launch_record", "exit_record", "gpu_uuid")):
                raise StateError(f"recovery job has phase or launch evidence: {job_id}")
            records[job_id] = job
        return records

    def rearm_after_transient_gpu_recovery(self) -> dict[str, Any]:
        """Arm only from the explicit human scientific-review boundary."""
        with self.host_lock:
            campaign = _read_json(self.campaign_path)
            if campaign.get("state") != "awaiting_scientific_review":
                raise StateError("transient recovery may re-arm only from awaiting_scientific_review")
            campaign["state"] = "armed"
            campaign["updated_at"] = utc_now()
            _atomic_json(self.campaign_path, campaign)
            self._append_event_locked(
                {
                    "kind": "campaign_rearmed_after_transient_gpu_recovery",
                    "from": "awaiting_scientific_review",
                    "to": "armed",
                }
            )
            return campaign

    def validate_missing_runtime_environment_failures(
        self, job_ids: tuple[str, ...] | list[str]
    ) -> dict[str, dict[str, Any]]:
        with self.host_lock:
            return self._validate_missing_runtime_environment_failures_locked(tuple(job_ids))

    def recover_missing_runtime_environment_failures(
        self, job_ids: tuple[str, ...] | list[str]
    ) -> dict[str, dict[str, Any]]:
        """Archive and requeue exact config-before-output controller failures."""
        requested = tuple(job_ids)
        with self.host_lock:
            records = self._validate_missing_runtime_environment_failures_locked(requested)
            recovered: dict[str, dict[str, Any]] = {}
            now = utc_now()
            for job_id, job in records.items():
                phase_name = str(job["phase"]["name"])
                phase_dir = self.root / "phases" / job_id / phase_name
                archive = (
                    self.root / "recovery-archive" / job_id / f"{phase_name}-missing-runtime-environment"
                )
                archive.parent.mkdir(parents=True, exist_ok=True)
                if archive.exists():
                    raise StateError(f"runtime-environment recovery archive already exists: {job_id}")
                os.replace(phase_dir, archive)
                history = list(job.get("recovery_history", []))
                history.append(
                    {
                        "state": job["state"],
                        "failure": job.get("failure"),
                        "phase_exit": job["phase_exit"],
                        "phase_archive": str(archive),
                        "recovered_at": now,
                    }
                )
                job["recovery_history"] = history
                for key in (
                    "failure",
                    "phase",
                    "phase_exit",
                    "launch_intent",
                    "pending_successor_phase",
                    "gpu_snapshot",
                    "gpu_uuid",
                    "admission",
                    "shared_gpu_at_launch",
                    "live_phase_digest_evidenced",
                    "autoattack_status",
                ):
                    job.pop(key, None)
                target = JobState.PREFLIGHT if phase_name == "train" else JobState.PGD_COMPLETED
                job["state"] = target.value
                job["updated_at"] = now
                job["revision"] = int(job.get("revision", 0)) + 1
                _atomic_json(self.jobs_path / f"{job_id}.json", job)
                self._append_event_locked(
                    {
                        "kind": "missing_runtime_environment_failure_recovered",
                        "job_id": job_id,
                        "from": (
                            JobState.FAILED.value
                            if phase_name == "train"
                            else JobState.PGD_COMPLETED_AUTOATTACK_FAILED.value
                        ),
                        "to": target.value,
                        "revision": job["revision"],
                        "phase_archive": str(archive),
                    }
                )
                recovered[job_id] = job
            return recovered

    def _validate_missing_runtime_environment_failures_locked(
        self, requested: tuple[str, ...]
    ) -> dict[str, dict[str, Any]]:
        if not requested or len(set(requested)) != len(requested):
            raise StateError("recovery requires one or more unique explicit job IDs")
        records: dict[str, dict[str, Any]] = {}
        for job_id in requested:
            job = _read_json(self.jobs_path / f"{job_id}.json")
            phase = job.get("phase")
            phase_exit = job.get("phase_exit")
            phase_name = phase.get("name") if isinstance(phase, dict) else None
            if phase_name not in {"train", "autoattack"}:
                raise StateError(f"job is not an exact pre-output runtime-environment failure: {job_id}")
            expected_state = (
                JobState.FAILED.value
                if phase_name == "train"
                else JobState.PGD_COMPLETED_AUTOATTACK_FAILED.value
            )
            expected_failure = phase_name == "train"
            phase_dir = self.root / "phases" / job_id / phase_name
            expected_exit = phase_dir / "exit.json"
            expected_launch = phase_dir / "launch.json"
            stderr = phase_dir / "stderr.log"
            if (
                job.get("state") != expected_state
                or (expected_failure and job.get("failure") != "phase returned nonzero")
                or (not expected_failure and job.get("autoattack_status") != "failed")
                or not isinstance(phase, dict)
                or phase.get("exit_record") != str(expected_exit)
                or phase.get("launch_record") != str(expected_launch)
                or not isinstance(phase_exit, dict)
                or phase_exit.get("exit_code") != 1
                or phase_exit.get("error") is not None
                or not expected_exit.is_file()
                or not expected_launch.is_file()
                or not stderr.is_file()
            ):
                raise StateError(f"job is not an exact pre-output runtime-environment failure: {job_id}")
            stderr_lines = stderr.read_text(encoding="utf-8").rstrip().splitlines()
            if not stderr_lines or stderr_lines[-1] != MISSING_RUNTIME_ENVIRONMENT_ERROR:
                raise StateError(f"job has a different runtime failure: {job_id}")
            archive = (
                self.root / "recovery-archive" / job_id / f"{phase_name}-missing-runtime-environment"
            )
            if archive.exists():
                raise StateError(f"runtime-environment recovery archive already exists: {job_id}")
            records[job_id] = job
        return records

    def _append_event_locked(self, event: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"at": utc_now(), **event}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self.host_lock:
            yield
