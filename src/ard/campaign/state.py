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
        {JobState.LAUNCHING, JobState.PGD_EVALUATION, JobState.FAILED, JobState.BLOCKED}
    ),
    JobState.PGD_EVALUATION: frozenset({JobState.PGD_COMPLETED, JobState.FAILED, JobState.BLOCKED}),
    JobState.PGD_COMPLETED: frozenset({JobState.LAUNCHING, JobState.AUTOATTACK, JobState.COMPLETED, JobState.BLOCKED}),
    JobState.AUTOATTACK: frozenset({JobState.COMPLETED, JobState.PGD_COMPLETED_AUTOATTACK_FAILED, JobState.BLOCKED}),
    JobState.COMPLETED: frozenset(),
    JobState.PGD_COMPLETED_AUTOATTACK_FAILED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.BLOCKED: frozenset(),
}


class StateError(CampaignError):
    pass


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
