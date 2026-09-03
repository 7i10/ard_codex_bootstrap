#!/usr/bin/env python3
"""Manifest-driven, resumable campaign controller.

The controller is deliberately scientific-method agnostic.  A job is an argv
plus immutable identity metadata; this tool only schedules it, records
lineage, and advances validated dependencies.  Long-running work belongs to a
detached controller/worker process, not to the Codex session.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
TERMINAL = {"completed", "failed", "blocked", "orphaned"}
ACTIVE = {"running", "retrying"}
RESERVATION_HANDLES: dict[tuple[str, int], Any] = {}


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical(value))
    os.replace(temporary, path)


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def abs_path(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(path: Path) -> tuple[dict[str, Any], str, Path]:
    path = path.resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("manifest schema_version must be 1")
    if not isinstance(manifest.get("campaign_id"), str) or not manifest["campaign_id"]:
        raise ValueError("campaign_id is required")
    source = manifest.get("source")
    if not isinstance(source, dict) or not _is_sha(source.get("git_sha")):
        raise ValueError("source.git_sha must be a full 40-hex Git SHA")
    hosts = manifest.get("hosts")
    if not isinstance(hosts, dict) or not hosts:
        raise ValueError("hosts must be a non-empty mapping")
    for name, profile in hosts.items():
        if not isinstance(name, str) or not isinstance(profile, dict):
            raise ValueError("host profiles must be mappings")
        _normalize_gpus(profile)
        for field in ("required_paths", "required_env"):
            values = profile.get(field, [])
            if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
                raise ValueError(f"{name}: {field} must be a list of non-empty strings")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a non-empty list")
    ids: set[str] = set()
    for job in jobs:
        _validate_job(job, ids, hosts, path.parent)
    _check_dependencies(jobs)
    state = abs_path(manifest.get("state_path"), path.parent)
    if state is None:
        state = (path.parent / ".orchestration" / f"{manifest['campaign_id']}.state.json").resolve()
    manifest["_manifest_path"] = str(path)
    manifest["_state_path"] = str(state)
    manifest["_manifest_sha256"] = file_digest(path)
    return manifest, manifest["_manifest_sha256"], path


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdefABCDEF" for c in value)


def _normalize_gpus(profile: dict[str, Any]) -> list[dict[str, Any]]:
    raw = profile.get("gpus", [])
    if not isinstance(raw, list):
        raise ValueError("host.gpus must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in raw:
        if isinstance(item, int):
            gpu = {"index": item}
        elif isinstance(item, dict) and isinstance(item.get("index"), int):
            gpu = dict(item)
        else:
            raise ValueError("each GPU must be an integer or {index: int}")
        if gpu["index"] in seen or gpu["index"] < 0:
            raise ValueError("GPU indices must be unique non-negative integers")
        seen.add(gpu["index"])
        normalized.append(gpu)
    profile["gpus"] = normalized
    return normalized


def _validate_job(job: Any, ids: set[str], hosts: dict[str, Any], base: Path) -> None:
    if not isinstance(job, dict) or not isinstance(job.get("job_id"), str) or not job["job_id"]:
        raise ValueError("each job requires a unique job_id")
    job_id = job["job_id"]
    if job_id in ids:
        raise ValueError(f"duplicate job_id: {job_id}")
    ids.add(job_id)
    command = job.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        raise ValueError(f"{job_id}: command must be a non-empty argv list")
    command_cwd = abs_path(job.get("cwd"), base) or base
    _validate_local_shell_argv(command, cwd=command_cwd, job_id=job_id, field="command")
    identity = job.get("scientific_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{job_id}: scientific_identity mapping is required")
    host = job.get("host")
    if host is not None and host not in hosts:
        raise ValueError(f"{job_id}: unknown host {host}")
    host_constraints = job.get("host_constraints", [])
    if not isinstance(host_constraints, list) or not all(item in hosts for item in host_constraints):
        raise ValueError(f"{job_id}: host_constraints must contain known hosts")
    for field in ("required_paths", "required_env"):
        values = job.get(field, [])
        if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
            raise ValueError(f"{job_id}: {field} must be a list of non-empty strings")
    deps = job.get("dependencies", [])
    if not isinstance(deps, list) or not all(isinstance(x, str) for x in deps):
        raise ValueError(f"{job_id}: dependencies must be a list of job IDs")
    output = abs_path(job.get("output_dir"), base)
    if output is None:
        raise ValueError(f"{job_id}: output_dir is required")
    job["_output_dir"] = str(output)
    marker = abs_path(job.get("completion_marker"), output)
    job["_completion_marker"] = str(marker or output / "completion.json")
    failure = abs_path(job.get("technical_failure_marker"), output)
    job["_technical_failure_marker"] = str(failure or output / "technical-failure.json")
    job["estimated_work"] = float(job.get("estimated_work", 1.0))
    if job["estimated_work"] < 0:
        raise ValueError(f"{job_id}: estimated_work must be non-negative")
    job["gpu_count"] = int(job.get("gpu_count", 1))
    if job["gpu_count"] not in (0, 1):
        raise ValueError(f"{job_id}: only zero or one GPU per job is supported")
    fixed_gpu = job.get("gpu")
    if fixed_gpu is not None and (not isinstance(fixed_gpu, int) or fixed_gpu < 0):
        raise ValueError(f"{job_id}: gpu must be a non-negative integer")
    retries = job.get("retry_policy", {})
    if not isinstance(retries, dict):
        raise ValueError(f"{job_id}: retry_policy must be a mapping")
    max_attempts = int(retries.get("max_attempts", 1))
    if max_attempts < 1:
        raise ValueError(f"{job_id}: max_attempts must be positive")
    retries["max_attempts"] = max_attempts
    job["retry_policy"] = retries
    executor = job.get("executor", {"type": "local"})
    if not isinstance(executor, dict) or executor.get("type", "local") not in {"local", "external_probe"}:
        raise ValueError(f"{job_id}: executor.type must be local or external_probe")
    if executor.get("type") == "external_probe":
        probe = job.get("completion_probe")
        if not isinstance(probe, list) or not probe or not all(isinstance(x, str) and x for x in probe):
            raise ValueError(f"{job_id}: external_probe requires completion_probe argv")
        _validate_local_shell_argv(probe, cwd=command_cwd, job_id=job_id, field="completion_probe")


def _validate_local_shell_argv(argv: list[str], *, cwd: Path, job_id: str, field: str) -> None:
    """Reject a known non-executable shell wrapper before reserving a GPU.

    The orchestrator executes argv without a shell.  A tracked/local `*.sh`
    wrapper therefore needs either its executable bit or an explicit `bash`
    interpreter.  Unknown remote-only paths remain the remote executor's
    responsibility, so this only rejects a path that is present locally.
    """
    executable = Path(argv[0])
    if executable.suffix != ".sh":
        return
    candidate = executable if executable.is_absolute() else cwd / executable
    if candidate.is_file() and not os.access(candidate, os.X_OK):
        raise ValueError(
            f"{job_id}: {field} directly invokes non-executable shell wrapper {executable}; "
            "use ['bash', '<script>.sh', ...] or set its executable bit"
        )


def _check_dependencies(jobs: list[dict[str, Any]]) -> None:
    ids = {job["job_id"] for job in jobs}
    for job in jobs:
        missing = set(job.get("dependencies", [])) - ids
        if missing:
            raise ValueError(f"{job['job_id']}: missing dependencies {sorted(missing)}")
    visiting: set[str] = set()
    visited: set[str] = set()
    by_id = {job["job_id"]: job for job in jobs}

    def visit(job_id: str) -> None:
        if job_id in visiting:
            raise ValueError("dependency cycle detected")
        if job_id in visited:
            return
        visiting.add(job_id)
        for dep in by_id[job_id].get("dependencies", []):
            visit(dep)
        visiting.remove(job_id)
        visited.add(job_id)

    for job_id in by_id:
        visit(job_id)


def scientific_identity(manifest: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    return {"source_sha": manifest["source"]["git_sha"].lower(), **job["scientific_identity"]}


def job_identity(manifest: dict[str, Any], job: dict[str, Any]) -> str:
    return digest(scientific_identity(manifest, job))


def initial_state(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["_manifest_sha256"],
        "source_sha": manifest["source"]["git_sha"].lower(),
        "status": "pending",
        "created_at": now(),
        "updated_at": now(),
        "jobs": {
            job["job_id"]: {
                "status": "pending",
                "identity_hash": job_identity(manifest, job),
                "attempts": [],
            }
            for job in manifest["jobs"]
        },
        "events": [],
    }


def event(state: dict[str, Any], event_type: str, job_id: str | None = None, **detail: Any) -> None:
    state["events"].append({"timestamp": now(), "event_type": event_type, "job_id": job_id, **detail})


def read_state(manifest: dict[str, Any]) -> dict[str, Any]:
    path = Path(manifest["_state_path"])
    if not path.exists():
        return initial_state(manifest)
    state = json.loads(path.read_text(encoding="utf-8"))
    if (
        state.get("campaign_id") != manifest["campaign_id"]
        or state.get("manifest_sha256") != manifest["_manifest_sha256"]
    ):
        raise ValueError("existing state belongs to a different campaign or manifest")
    return state


def save_state(manifest: dict[str, Any], state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    atomic_json(Path(manifest["_state_path"]), state)


def marker_payload(manifest: dict[str, Any], job: dict[str, Any], attempt: int, attempt_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "campaign_id": manifest["campaign_id"],
        "job_id": job["job_id"],
        "attempt": attempt,
        "attempt_id": attempt_id,
        "identity_hash": job_identity(manifest, job),
        "source_sha": manifest["source"]["git_sha"].lower(),
        "output_dir": job["_output_dir"],
        "completed_at": now(),
    }


def valid_marker(manifest: dict[str, Any], job: dict[str, Any], path: Path) -> bool:
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("status") == "completed"
        and marker.get("campaign_id") == manifest["campaign_id"]
        and marker.get("job_id") == job["job_id"]
        and marker.get("source_sha", "").lower() == manifest["source"]["git_sha"].lower()
        and marker.get("identity_hash") == job_identity(manifest, job)
    )


def failure_info(job: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(job["_technical_failure_marker"])
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("failure_class") != "technical":
        return None
    return value


def valid_failure_marker(manifest: dict[str, Any], job: dict[str, Any]) -> dict[str, Any] | None:
    value = failure_info(job)
    if value is None:
        return None
    if value.get("campaign_id") not in (None, manifest["campaign_id"]):
        return None
    if value.get("job_id") not in (None, job["job_id"]):
        return None
    source_sha = value.get("source_sha")
    if source_sha is not None and (
        not isinstance(source_sha, str) or source_sha.lower() != manifest["source"]["git_sha"].lower()
    ):
        return None
    identity = value.get("identity_hash")
    if identity is not None and identity != job_identity(manifest, job):
        return None
    return value


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def reservation_path(manifest: dict[str, Any], host: str, gpu: int) -> Path:
    root = abs_path(manifest.get("reservation_root"), Path(manifest["_manifest_path"]).parent)
    if root is None:
        root = Path.home() / ".cache" / "ard-experiment-orchestrator" / "reservations"
    safe_host = "".join(char if char.isalnum() or char in "-_" else "_" for char in host)
    return root / f"{safe_host}-gpu{gpu}.lock"


def claim_slot(manifest: dict[str, Any], slot: tuple[str, int, str | None, float]) -> bool:
    host, gpu, uuid, _ = slot
    if gpu < 0:
        return True
    key = (host, gpu)
    if key in RESERVATION_HANDLES:
        return True
    path = reservation_path(manifest, host, gpu)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return False
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"campaign_id": manifest["campaign_id"], "host": host, "gpu": gpu, "uuid": uuid}))
    handle.flush()
    RESERVATION_HANDLES[key] = handle
    return True


def release_reservations() -> None:
    for handle in RESERVATION_HANDLES.values():
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            handle.close()
    RESERVATION_HANDLES.clear()


def release_slot(slot: tuple[str, int, str | None, float] | tuple[str, int]) -> None:
    host, gpu = slot[:2]
    handle = RESERVATION_HANDLES.pop((host, gpu), None)
    if handle is None:
        return
    with contextlib.suppress(OSError):
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        handle.close()


def host_slots(manifest: dict[str, Any]) -> list[tuple[str, int, str | None, float]]:
    slots: list[tuple[str, int, str | None, float]] = []
    for host, profile in manifest["hosts"].items():
        default_throughput = float(profile.get("throughput", 1.0)) or 1.0
        for gpu in _normalize_gpus(profile):
            throughput = float(gpu.get("throughput", default_throughput)) or 1.0
            slots.append((host, gpu["index"], gpu.get("uuid"), throughput))
    return slots


def assign_slot(
    manifest: dict[str, Any], job: dict[str, Any], occupied: set[tuple[str, int]]
) -> tuple[str, int, str | None, float] | None:
    if job["gpu_count"] == 0:
        host = job.get("host") or next(iter(manifest["hosts"]))
        throughput = float(manifest["hosts"][host].get("throughput", 1.0)) or 1.0
        return host, -1, None, throughput
    candidates = []
    allowed_hosts = set(job.get("host_constraints", [])) if job.get("host_constraints") else None
    for host, index, uuid, throughput in host_slots(manifest):
        if allowed_hosts is not None and host not in allowed_hosts:
            continue
        if job.get("host") is not None and host != job["host"]:
            continue
        if job.get("gpu") is not None and index != job["gpu"]:
            continue
        if (host, index) in occupied:
            continue
        transfer = float(job.get("transfer_seconds", 0.0))
        score = transfer + job["estimated_work"] / throughput
        candidates.append((score, host, index, uuid, throughput))
    if not candidates:
        return None
    _, host, index, uuid, throughput = min(candidates)
    return host, index, uuid, throughput


def has_candidate_slot(manifest: dict[str, Any], job: dict[str, Any]) -> bool:
    """Return whether a job could ever fit, ignoring current wave occupancy."""
    if job["gpu_count"] == 0:
        return True
    allowed_hosts = set(job.get("host_constraints", [])) if job.get("host_constraints") else None
    for host, index, _, _ in host_slots(manifest):
        if allowed_hosts is not None and host not in allowed_hosts:
            continue
        if job.get("host") is not None and host != job["host"]:
            continue
        if job.get("gpu") is not None and index != job["gpu"]:
            continue
        return True
    return False


def ready_jobs(manifest: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    ready: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        record = state["jobs"][job["job_id"]]
        if record["status"] != "pending":
            continue
        dependencies = [state["jobs"][dep]["status"] for dep in job.get("dependencies", [])]
        if any(status in {"failed", "blocked", "orphaned"} for status in dependencies):
            record["status"] = "blocked"
            event(state, "dependency_blocked", job["job_id"], dependencies=job.get("dependencies", []))
            continue
        if all(status == "completed" for status in dependencies):
            ready.append(job)
    return sorted(ready, key=lambda item: (-item["estimated_work"], item["job_id"]))


def start_worker(
    manifest: dict[str, Any], job: dict[str, Any], attempt: int, slot: tuple[str, int, str | None, float]
) -> dict[str, Any]:
    output = Path(job["_output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    attempt_id = f"{job.get('run_id', job['job_id'])}-attempt-{attempt}"
    log = output / "orchestration" / f"{job['job_id']}.attempt-{attempt}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    Path(job["_technical_failure_marker"]).unlink(missing_ok=True)
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in job.get("env", {}).items()})
    env.update(
        {
            "ARD_ORCH_CAMPAIGN_ID": manifest["campaign_id"],
            "ARD_ORCH_JOB_ID": job["job_id"],
            "ARD_ORCH_ATTEMPT": str(attempt),
            "ARD_ORCH_ATTEMPT_ID": attempt_id,
            "ARD_ORCH_IDENTITY_HASH": job_identity(manifest, job),
            "ARD_ORCH_SOURCE_SHA": manifest["source"]["git_sha"].lower(),
            "ARD_ORCH_HOST": slot[0],
            "ARD_ORCH_GPU_INDEX": str(slot[1]),
        }
    )
    # A launch gate may provide an attempt-aware W&B template.  Keeping this
    # in the execution layer means technical retries get distinct execution
    # IDs while `job_identity` remains unchanged.
    wandb_template = job.get("wandb_run_id_template")
    if isinstance(wandb_template, str) and wandb_template:
        env["WANDB_RUN_ID"] = wandb_template.format(attempt=attempt, attempt_id=attempt_id)
        env.setdefault("WANDB_RESUME", "never")
    if slot[1] >= 0:
        env.setdefault("CUDA_VISIBLE_DEVICES", str(slot[1]))
    worker_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--manifest",
        manifest["_manifest_path"],
        "--job-id",
        job["job_id"],
        "--attempt",
        str(attempt),
        "--attempt-id",
        attempt_id,
    ]
    with log.open("ab") as handle:
        process = subprocess.Popen(
            worker_command,
            cwd=Path(manifest["_manifest_path"]).parent,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {
        "attempt": attempt,
        "attempt_id": attempt_id,
        "identity_hash": job_identity(manifest, job),
        "pid": process.pid,
        "host": slot[0],
        "gpu": slot[1],
        "gpu_uuid": slot[2],
        "launched_at": now(),
        "log": str(log),
        "status": "running",
    }


def reconcile_running(manifest: dict[str, Any], state: dict[str, Any]) -> None:
    for job in manifest["jobs"]:
        record = state["jobs"][job["job_id"]]
        if record["status"] != "running" or not record["attempts"]:
            continue
        attempt = record["attempts"][-1]
        result_path = (
            Path(job["_output_dir"]) / "orchestration" / f"{job['job_id']}.attempt-{attempt['attempt']}.result.json"
        )
        marker = Path(job["_completion_marker"])
        if valid_marker(manifest, job, marker):
            attempt["status"] = "completed"
            record["status"] = "completed"
            release_slot((attempt["host"], attempt["gpu"]))
            event(state, "completion_marker_seen", job["job_id"], attempt=attempt["attempt"])
            continue
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                result = {}
            code = result.get("exit_code")
            info = valid_failure_marker(manifest, job)
            retryable = bool(info and info.get("retryable") is True)
            if code == 0 and result.get("status") == "completed":
                record["status"] = "failed"
                attempt["status"] = "failed"
                release_slot((attempt["host"], attempt["gpu"]))
                event(state, "marker_invalid", job["job_id"], attempt=attempt["attempt"])
            elif retryable and len(record["attempts"]) < job["retry_policy"]["max_attempts"]:
                attempt["status"] = "technical_failed"
                record["status"] = "pending"
                event(
                    state, "technical_retry", job["job_id"], attempt=attempt["attempt"], retry_of=attempt["attempt_id"]
                )
            else:
                attempt["status"] = "failed"
                record["status"] = "failed"
                release_slot((attempt["host"], attempt["gpu"]))
                event(
                    state,
                    "technical_failure" if retryable else "run_failed",
                    job["job_id"],
                    attempt=attempt["attempt"],
                    exit_code=code,
                )
        elif not pid_alive(attempt.get("pid")):
            attempt["status"] = "orphaned"
            record["status"] = "orphaned"
            release_slot((attempt["host"], attempt["gpu"]))
            event(state, "orphaned", job["job_id"], attempt=attempt["attempt"])


def controller_tick(manifest: dict[str, Any], state: dict[str, Any]) -> bool:
    reconcile_running(manifest, state)
    for job in manifest["jobs"]:
        record = state["jobs"][job["job_id"]]
        if record["status"] == "pending" and valid_marker(manifest, job, Path(job["_completion_marker"])):
            record["status"] = "completed"
            event(state, "completion_marker_recovered", job["job_id"])
    occupied = {
        (record["attempts"][-1]["host"], record["attempts"][-1]["gpu"])
        for record in state["jobs"].values()
        if record["status"] == "running" and record["attempts"]
    }
    for job in ready_jobs(manifest, state):
        slot = assign_slot(manifest, job, occupied)
        if slot is None:
            continue
        if not claim_slot(manifest, slot):
            occupied.add((slot[0], slot[1]))
            continue
        record = state["jobs"][job["job_id"]]
        attempt = len(record["attempts"]) + 1
        started = start_worker(manifest, job, attempt, slot)
        record["attempts"].append(started)
        record["status"] = "running"
        occupied.add((slot[0], slot[1]))
        event(state, "launch_success", job["job_id"], attempt=attempt, host=slot[0], gpu=slot[1], gpu_uuid=slot[2])
        if pid_alive(started["pid"]):
            event(state, "stable_confirmed", job["job_id"], attempt=attempt)
    statuses = [record["status"] for record in state["jobs"].values()]
    if all(status in TERMINAL for status in statuses):
        state["status"] = "completed" if all(status == "completed" for status in statuses) else "failed"
        if not state.get("finished_at"):
            state["finished_at"] = now()
            event(state, "campaign_complete", status=state["status"])
        return True
    state["status"] = "running" if any(status == "running" for status in statuses) else "pending"
    return False


def run_controller(manifest: dict[str, Any], *, foreground: bool, once: bool, poll_interval: float) -> int:
    state_path = Path(manifest["_state_path"])
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    if not foreground:
        log_path = state_path.with_suffix(state_path.suffix + ".controller.log")
        with locked(lock_path):
            state = read_state(manifest)
            existing_pid = state.get("controller_pid")
            if state.get("status") in {"completed", "failed"}:
                print(
                    json.dumps(
                        {"campaign_id": manifest["campaign_id"], "status": state["status"], "state": str(state_path)}
                    )
                )
                return 0
            if isinstance(existing_pid, int) and pid_alive(existing_pid):
                print(
                    json.dumps(
                        {
                            "campaign_id": manifest["campaign_id"],
                            "controller_pid": existing_pid,
                            "state": str(state_path),
                            "already_running": True,
                        }
                    )
                )
                return 0
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "controller",
            "--manifest",
            manifest["_manifest_path"],
            "--poll-interval",
            str(poll_interval),
        ]
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        with locked(lock_path):
            state = read_state(manifest)
            state["controller_pid"] = process.pid
            event(state, "controller_started", pid=process.pid, log=str(log_path))
            save_state(manifest, state)
        print(
            json.dumps(
                {"campaign_id": manifest["campaign_id"], "controller_pid": process.pid, "state": str(state_path)}
            )
        )
        return 0
    try:
        while True:
            with locked(lock_path):
                state = read_state(manifest)
                done = controller_tick(manifest, state)
                save_state(manifest, state)
            if done or once:
                return 0
            time.sleep(max(0.01, poll_interval))
    finally:
        release_reservations()


def worker(manifest: dict[str, Any], job: dict[str, Any], attempt: int, attempt_id: str) -> int:
    output = Path(job["_output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in job.get("env", {}).items()})
    env.update({"ARD_ORCH_ATTEMPT": str(attempt), "ARD_ORCH_ATTEMPT_ID": attempt_id})
    cwd = abs_path(job.get("cwd"), Path(manifest["_manifest_path"]).parent) or Path(manifest["_manifest_path"]).parent
    executor = job.get("executor", {"type": "local"}).get("type", "local")
    command = [str(x) for x in job["command"]]
    code = subprocess.run(command, cwd=cwd, env=env).returncode
    probe_timed_out = False
    if code == 0 and executor == "external_probe":
        probe = [str(x) for x in job["completion_probe"]]
        interval = max(0.01, float(job.get("probe_interval_seconds", 30)))
        timeout = job.get("probe_timeout_seconds")
        deadline = time.monotonic() + float(timeout) if timeout is not None else None
        while subprocess.run(probe, cwd=cwd, env=env).returncode != 0:
            if deadline is not None and time.monotonic() >= deadline:
                probe_timed_out = True
                break
            delay = interval if deadline is None else min(interval, max(0.01, deadline - time.monotonic()))
            time.sleep(delay)
    if code == 0 and not probe_timed_out:
        atomic_json(Path(job["_completion_marker"]), marker_payload(manifest, job, attempt, attempt_id))
        status = "completed"
    else:
        status = "failed"
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "campaign_id": manifest["campaign_id"],
        "job_id": job["job_id"],
        "attempt": attempt,
        "attempt_id": attempt_id,
        "identity_hash": job_identity(manifest, job),
        "exit_code": code,
        "probe_timed_out": probe_timed_out,
        "finished_at": now(),
    }
    atomic_json(output / "orchestration" / f"{job['job_id']}.attempt-{attempt}.result.json", result)
    return code


def status(manifest: dict[str, Any]) -> int:
    state = read_state(manifest)
    summary = {
        "campaign_id": state["campaign_id"],
        "status": state["status"],
        "jobs": {k: v["status"] for k, v in state["jobs"].items()},
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def plan(manifest: dict[str, Any]) -> int:
    by_id = {job["job_id"]: job for job in manifest["jobs"]}
    remaining = set(by_id)
    completed: set[str] = set()
    unavailable: set[str] = set()
    rows = []
    while remaining:
        wave = sorted(
            (
                by_id[job_id]
                for job_id in remaining
                if set(by_id[job_id].get("dependencies", [])) <= completed
                or set(by_id[job_id].get("dependencies", [])) & unavailable
            ),
            key=lambda item: (-item["estimated_work"], item["job_id"]),
        )
        if not wave:
            for job_id in sorted(remaining):
                rows.append({"job_id": job_id, "status": "dependency_unavailable"})
                unavailable.add(job_id)
                remaining.remove(job_id)
            break
        occupied: set[tuple[str, int]] = set()
        assigned = 0
        for job in wave:
            if set(job.get("dependencies", [])) & unavailable:
                rows.append({"job_id": job["job_id"], "status": "dependency_unavailable"})
                unavailable.add(job["job_id"])
                remaining.remove(job["job_id"])
                continue
            slot = assign_slot(manifest, job, occupied)
            if slot is None:
                if has_candidate_slot(manifest, job):
                    continue
                rows.append({"job_id": job["job_id"], "status": "resource_conflict"})
                unavailable.add(job["job_id"])
                remaining.remove(job["job_id"])
                continue
            occupied.add((slot[0], slot[1]))
            rows.append(
                {
                    "job_id": job["job_id"],
                    "host": slot[0],
                    "gpu": slot[1],
                    "estimated_finish": job["estimated_work"] / slot[3],
                    "dependencies": job.get("dependencies", []),
                }
            )
            completed.add(job["job_id"])
            remaining.remove(job["job_id"])
            assigned += 1
        if assigned == 0 and wave:
            # No slot can ever satisfy this wave (for example GPU 4 on a
            # two-GPU host).  Mark unresolved entries explicitly to avoid a
            # read-only dry-run loop.
            for job in wave:
                if job["job_id"] not in remaining:
                    continue
                rows.append({"job_id": job["job_id"], "status": "resource_conflict"})
                unavailable.add(job["job_id"])
                remaining.remove(job["job_id"])
    print(
        json.dumps(
            {"campaign_id": manifest["campaign_id"], "manifest_sha256": manifest["_manifest_sha256"], "jobs": rows},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def preflight(manifest: dict[str, Any]) -> int:
    errors: list[str] = []
    base = Path(manifest["_manifest_path"]).parent
    for name, profile in manifest["hosts"].items():
        if profile.get("backend", "local") == "local":
            python_path = profile.get("python")
            resolved_python = abs_path(python_path, base) if python_path else None
            if resolved_python and not resolved_python.exists():
                errors.append(f"{name}: python does not exist: {python_path}")
            for raw in profile.get("required_paths", []):
                path = abs_path(raw, base)
                if path is None or not path.exists():
                    errors.append(f"{name}: required path missing: {raw}")
        command = profile.get("preflight_command")
        if command is not None:
            if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
                errors.append(f"{name}: preflight_command must be an argv list")
            else:
                result = subprocess.run(command, text=True, capture_output=True)
                if result.returncode != 0:
                    errors.append(f"{name}: preflight_command failed ({result.returncode})")
        for variable in profile.get("required_env", []):
            if variable not in os.environ:
                errors.append(f"{name}: required environment variable missing: {variable}")
    for job in manifest["jobs"]:
        host = job.get("host") or next(iter(manifest["hosts"]))
        profile = manifest["hosts"][host]
        if profile.get("backend", "local") != "local":
            continue
        available_env = set(os.environ) | set(job.get("env", {}))
        for variable in job.get("required_env", []):
            if variable not in available_env:
                errors.append(f"{job['job_id']}: required environment variable missing: {variable}")
        for raw in job.get("required_paths", []):
            path = abs_path(raw, Path(manifest["_manifest_path"]).parent)
            if path is None or not path.exists():
                errors.append(f"{job['job_id']}: required path missing: {raw}")
        cwd = abs_path(job.get("cwd"), Path(manifest["_manifest_path"]).parent)
        if cwd is not None and not cwd.is_dir():
            errors.append(f"{job['job_id']}: cwd missing or not a directory: {cwd}")
    result = {
        "ready": not errors,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["_manifest_sha256"],
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "preflight", "status"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        if name == "plan":
            p.add_argument("--dry-run", action="store_true", help="explicitly document that no state is written")
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--foreground", action="store_true")
    run.add_argument("--once", action="store_true")
    run.add_argument("--poll-interval", type=float, default=2.0)
    controller = sub.add_parser("controller")
    controller.add_argument("--manifest", type=Path, required=True)
    controller.add_argument("--poll-interval", type=float, default=2.0)
    worker_parser = sub.add_parser("_worker")
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--job-id", required=True)
    worker_parser.add_argument("--attempt", type=int, required=True)
    worker_parser.add_argument("--attempt-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, _, _ = load_manifest(args.manifest)
    if args.command == "validate":
        print(
            json.dumps(
                {"valid": True, "campaign_id": manifest["campaign_id"], "manifest_sha256": manifest["_manifest_sha256"]}
            )
        )
        return 0
    if args.command == "plan":
        return plan(manifest)
    if args.command == "preflight":
        return preflight(manifest)
    if args.command == "status":
        return status(manifest)
    if args.command == "_worker":
        by_id = {job["job_id"]: job for job in manifest["jobs"]}
        return worker(manifest, by_id[args.job_id], args.attempt, args.attempt_id)
    if args.command == "controller":
        return run_controller(manifest, foreground=True, once=False, poll_interval=args.poll_interval)
    return run_controller(manifest, foreground=args.foreground, once=args.once, poll_interval=args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
