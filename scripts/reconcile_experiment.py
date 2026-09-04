#!/usr/bin/env python3
"""Reconcile one detached experiment and hand off post-processing once.

This command is intentionally a small control-plane helper.  It reads one
runtime-bound ``experiment-state.json`` and existing terminal evidence; it
never scans the repository, changes scientific configuration, or polls a
long-running job.  Evaluation/aggregation/reporting are delegated to the
existing postprocess command recorded by the experiment state.
"""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "workspace" / "ard_workspace_v1.json"
SCHEMA_VERSION = 2
COMPATIBLE_SCHEMA_VERSIONS = {1, 2}
TERMINAL_STATES = {"PUSHED", "AWAITING_RESEARCH_REVIEW", "NEEDS_RESEARCH_DECISION", "NEEDS_TECHNICAL_RECOVERY"}
POSTPROCESS_STATES = {"EVALUATING", "SUMMARIZING", "PUSHED", "AWAITING_RESEARCH_REVIEW"}
KNOWN_STATES = {
    "PLANNED",
    "IMPLEMENTING",
    "VALIDATING",
    "LAUNCHING",
    "TRAINING",
    "LAUNCH_FAILED",
    "NEEDS_TECHNICAL_RECOVERY",
    "TRAINING_SUCCESS",
    "TRAINING_FAILED",
    "EVALUATING",
    "SUMMARIZING",
    "PUSHED",
    "AWAITING_RESEARCH_REVIEW",
    "NEEDS_RESEARCH_DECISION",
}
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
ORCHESTRATOR_MODES = {"orchestrator_campaign", "single_process"}


class ReconcileError(RuntimeError):
    """A fail-closed state or evidence error."""


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def iso(value: dt.datetime | None = None) -> str:
    return (value or now()).isoformat(timespec="microseconds")


def parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=dt.UTC) if parsed.tzinfo is None else parsed.astimezone(dt.UTC)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconcileError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ReconcileError(f"JSON evidence must be an object: {path}")
    return value


class NonBlockingLock:
    """Short-lived OS lock used together with the durable state lease."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            self.handle = None
            return False
        return True

    def release(self) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

    def __enter__(self) -> NonBlockingLock:
        if not self.acquire():
            raise ReconcileError(f"state is concurrently reconciled: {self.path}")
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _nested(state: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    if key in state:
        return state[key]
    child = state.get(section)
    return child.get(key, default) if isinstance(child, dict) else default


def state_path(args: argparse.Namespace) -> Path:
    if args.state:
        path = args.state.resolve(strict=False)
    else:
        if not SAFE_ID.fullmatch(args.experiment_id):
            raise ReconcileError("experiment-id contains unsafe path characters")
        runtime = args.runtime_root
        if runtime is None:
            registry = args.registry or DEFAULT_REGISTRY
            try:
                values = read_json(registry)
                runtime = Path(str(values["runtime_root"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ReconcileError(f"cannot resolve runtime_root from {registry}") from exc
        path = (runtime / "runs" / args.experiment_id / "experiment-state.json").resolve(strict=False)
    if not path.exists():
        raise ReconcileError(f"experiment state does not exist: {path}")
    return path


def path_from_state(value: Any, *, state_file: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = state_file.parent / candidate
    return candidate.resolve(strict=False)


def process_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def identity_matches(marker: dict[str, Any], state: dict[str, Any]) -> bool:
    pairs = (("experiment_id", "experiment_id"), ("source_sha", "source_sha"))
    for marker_key, state_key in pairs:
        expected = state.get(state_key)
        observed = marker.get(marker_key)
        if expected is not None:
            if observed is None or str(expected) != str(observed):
                return False
    expected_identity = state.get("scientific_identity_hash")
    if expected_identity is not None:
        observed_identity = marker.get("scientific_identity_hash", marker.get("identity_hash"))
        if observed_identity is None or str(expected_identity) != str(observed_identity):
            return False
    return True


def marker_ok(path: Path | None, state: dict[str, Any], *, failure: bool = False) -> tuple[bool, dict[str, Any] | None]:
    if path is None or not path.is_file():
        return False, None
    marker = read_json(path)
    if not identity_matches(marker, state):
        raise ReconcileError(f"terminal marker identity mismatch: {path}")
    if failure:
        failure_class = marker.get("failure_class")
        if failure_class not in {"technical", "scientific", "unknown"}:
            raise ReconcileError(f"failure marker lacks a valid failure_class: {path}")
        return True, marker
    status = str(marker.get("status", "")).lower()
    successful = bool(marker.get("success")) or status in {"ok", "success", "succeeded", "complete", "completed"}
    return successful, marker if successful else None


def exit_ok(state: dict[str, Any], state_file: Path) -> bool:
    evidence = _nested(state, "training", "exit_evidence")
    if isinstance(evidence, str):
        path = path_from_state(evidence, state_file=state_file)
        evidence = read_json(path) if path else None
    if not isinstance(evidence, dict):
        return False
    if not identity_matches(evidence, state):
        raise ReconcileError("launcher exit evidence identity mismatch")
    return evidence.get("exit_code") == 0 and evidence.get("status", "success") not in {"failed", "failure"}


def expected_outputs_exist(state: dict[str, Any], state_file: Path) -> bool:
    outputs = state.get("expected_outputs")
    if outputs is None:
        outputs = _nested(state, "training", "expected_outputs", [])
    if not isinstance(outputs, list) or not outputs:
        return False
    paths = [path_from_state(item, state_file=state_file) for item in outputs]
    return all(path is not None and path.exists() for path in paths)


def completion_evidence(state: dict[str, Any], state_file: Path) -> tuple[bool, str]:
    completion = path_from_state(_nested(state, "training", "completion_marker"), state_file=state_file)
    failure = path_from_state(_nested(state, "training", "failure_marker"), state_file=state_file)
    failed, failure_marker = marker_ok(failure, state, failure=True)
    if failed:
        return False, f"failure_marker:{failure_marker.get('failure_class', 'unknown')}"
    complete, _ = marker_ok(completion, state)
    if not complete:
        return False, "missing_or_invalid_completion_marker"
    if not exit_ok(state, state_file):
        return False, "missing_or_nonzero_launcher_exit"
    if not expected_outputs_exist(state, state_file):
        return False, "missing_expected_output"
    return True, "valid_completion"


def state_mode(state: dict[str, Any]) -> str:
    """Return the explicit v2 mode, or preserve v1 as a local process."""
    mode = state.get("mode", "single_process")
    if mode not in ORCHESTRATOR_MODES:
        raise ReconcileError(f"unsupported experiment-state mode: {mode}")
    return str(mode)


def orchestrator_state(state: dict[str, Any], state_file: Path) -> dict[str, Any]:
    raw = state.get("orchestrator_state_path")
    path = path_from_state(raw, state_file=state_file)
    if path is None or not path.is_file():
        raise ReconcileError("orchestrator campaign state is not registered")
    value = read_json(path)
    if value.get("campaign_id") != state.get("campaign_id"):
        raise ReconcileError("orchestrator campaign identity mismatch")
    if value.get("source_sha") and str(value["source_sha"]).lower() != str(state["source_sha"]).lower():
        raise ReconcileError("orchestrator source identity mismatch")
    expected_manifest = state.get("manifest_sha256")
    if expected_manifest and value.get("manifest_sha256") != expected_manifest:
        raise ReconcileError("orchestrator manifest identity mismatch")
    if not isinstance(value.get("jobs"), dict):
        raise ReconcileError("orchestrator state has no job records")
    return value


def required_job_ids(state: dict[str, Any], orch: dict[str, Any]) -> list[str]:
    declared = state.get("required_training_jobs")
    if isinstance(declared, list) and declared:
        return [str(item) for item in declared]
    return list(orch.get("jobs", {}).keys())


def terminal_result_job_ids(state: dict[str, Any]) -> list[str]:
    declared = state.get("terminal_result_jobs")
    return [str(item) for item in declared] if isinstance(declared, list) else []


def campaign_failure_class(state: dict[str, Any], orch: dict[str, Any]) -> tuple[str, str]:
    """Aggregate all failed required jobs before selecting the safest class.

    Scientific evidence dominates unknown/non-retryable technical evidence,
    which dominates retryable technical evidence. A campaign is retryable
    only when every observed failure is explicitly technical and retryable.
    """
    failures: list[tuple[str, str, bool]] = []
    job_ids = required_job_ids(state, orch) + terminal_result_job_ids(state)
    for job_id in job_ids:
        record = orch.get("jobs", {}).get(job_id)
        if not isinstance(record, dict) or record.get("status") not in {"failed", "orphaned", "blocked"}:
            continue
        observed = False
        attempts = record.get("attempts")
        if isinstance(attempts, list):
            for attempt in reversed(attempts):
                if not isinstance(attempt, dict):
                    continue
                failure_class = attempt.get("failure_class")
                if failure_class:
                    observed = True
                    failures.append((job_id, str(failure_class), attempt.get("retryable") is True))
                    break
        if not observed:
            failures.append((job_id, "unknown", False))
    if not failures:
        return "unknown", "no-registered-failure-evidence"
    ids = ",".join(job_id for job_id, _, _ in failures)
    if any(kind == "scientific" for _, kind, _ in failures):
        return "scientific", f"jobs:{ids}"
    if all(kind == "technical" and retryable for _, kind, retryable in failures):
        return "technical_retryable", f"jobs:{ids}"
    return "unknown", f"jobs:{ids}"


def campaign_training_status(state: dict[str, Any], state_file: Path) -> tuple[str, str, dict[str, Any]]:
    orch = orchestrator_state(state, state_file)
    required = required_job_ids(state, orch)
    missing = [job_id for job_id in required if job_id not in orch["jobs"]]
    if missing:
        raise ReconcileError(f"orchestrator state missing required jobs: {missing}")
    statuses = {job_id: orch["jobs"][job_id].get("status") for job_id in required}
    active = {"pending", "running", "retrying"}
    if any(status in active for status in statuses.values()):
        return "running", "required_training_jobs_active", orch
    failed = {"failed", "orphaned", "blocked"}
    if any(status in failed for status in statuses.values()):
        failure_class, reason = campaign_failure_class(state, orch)
        return "failed", f"{failure_class}:{reason}", orch
    if not all(status == "completed" for status in statuses.values()):
        return "failed", "unknown:training_status_incomplete", orch
    return "success", "required_training_jobs_completed", orch


def orchestrator_result_status(state: dict[str, Any], orch: dict[str, Any]) -> tuple[str, str]:
    result_jobs = terminal_result_job_ids(state)
    if not result_jobs:
        return "none", "no-terminal-result-jobs"
    statuses = {job_id: orch.get("jobs", {}).get(job_id, {}).get("status") for job_id in result_jobs}
    if any(status in {"pending", "running", "retrying"} for status in statuses.values()):
        return "active", "downstream_jobs_active"
    if any(status in {"failed", "orphaned", "blocked"} for status in statuses.values()):
        failure_class, reason = campaign_failure_class(state, orch)
        return "failed", f"{failure_class}:{reason}"
    if all(status == "completed" for status in statuses.values()):
        return "complete", "downstream_jobs_completed"
    return "failed", "downstream_status_incomplete"


def recovery_command(state: dict[str, Any]) -> list[str] | None:
    value = state.get("recovery_command")
    if value is None:
        recovery = state.get("recovery")
        value = recovery.get("command") if isinstance(recovery, dict) else None
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ReconcileError("recovery command must be a non-empty argv list")
    return list(value)


def recovery_limit(state: dict[str, Any], default: int) -> int:
    recovery = state.get("recovery")
    value = recovery.get("max_attempts", default) if isinstance(recovery, dict) else default
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ReconcileError("recovery.max_attempts must be an integer") from exc
    if limit < 1:
        raise ReconcileError("recovery.max_attempts must be >= 1")
    return limit


def recovery_lease_active(state: dict[str, Any]) -> bool:
    expiry = parse_time(state.get("recovery_lease_expires_at"))
    return expiry is not None and expiry > now()


def clear_recovery_lease(state: dict[str, Any]) -> None:
    for key in (
        "recovery_owner",
        "recovery_lease_id",
        "recovery_lease_started_at",
        "recovery_lease_expires_at",
        "recovery_pid",
    ):
        state.pop(key, None)


def argv_digest(command: list[str] | None) -> str | None:
    if command is None:
        return None
    payload = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def delegate_recovery(
    state: dict[str, Any], state_file: Path, *, reason: str, args: argparse.Namespace
) -> dict[str, Any]:
    """Delegate a pre-registered technical recovery command exactly once."""
    if recovery_lease_active(state):
        atomic_json(state_file, state)
        return {"status": "NO_OP", "reason": "recovery_lease_active", "state": state.get("state")}
    command = recovery_command(state)
    failure_class = reason.split(":", 1)[0]
    if failure_class != "technical_retryable":
        state["state"] = "NEEDS_RESEARCH_DECISION"
        state["needs_research_decision_reason"] = f"training failure is not auto-retryable: {reason}"
        clear_recovery_lease(state)
        atomic_json(state_file, state)
        return {"status": "NEEDS_RESEARCH_DECISION", "state": state["state"], "reason": reason}
    attempts = int(state.get("recovery_attempts", 0))
    if attempts >= recovery_limit(state, args.max_recovery_attempts):
        state["state"] = "NEEDS_RESEARCH_DECISION"
        state["needs_research_decision_reason"] = "technical recovery attempt bound reached"
        clear_recovery_lease(state)
        atomic_json(state_file, state)
        return {"status": "NEEDS_RESEARCH_DECISION", "state": state["state"], "reason": "recovery_attempt_bound"}
    if command is None:
        state["state"] = "NEEDS_RESEARCH_DECISION"
        state["needs_research_decision_reason"] = "technical failure has no registered recovery command"
        clear_recovery_lease(state)
        atomic_json(state_file, state)
        return {"status": "NEEDS_RESEARCH_DECISION", "state": state["state"], "reason": "recovery_not_registered"}
    expected_command_digest = state.get("recovery_command_sha256")
    if expected_command_digest is not None and expected_command_digest != argv_digest(command):
        state["state"] = "NEEDS_RESEARCH_DECISION"
        state["needs_research_decision_reason"] = "registered recovery command changed after launch"
        clear_recovery_lease(state)
        atomic_json(state_file, state)
        return {"status": "NEEDS_RESEARCH_DECISION", "state": state["state"], "reason": "recovery_command_drift"}
    lease_id = uuid.uuid4().hex
    started = now()
    state.update(
        {
            "recovery_owner": owner_string(),
            "recovery_lease_id": lease_id,
            "recovery_lease_started_at": iso(started),
            "recovery_lease_expires_at": iso(started + dt.timedelta(seconds=args.lease_seconds)),
            "recovery_attempts": attempts + 1,
            "recovery_reason": reason,
        }
    )
    env = os.environ.copy()
    env.update(
        {
            "ERT_EXPERIMENT_ID": str(state.get("experiment_id", "")),
            "ERT_EXPERIMENT_STATE": str(state_file),
            "ERT_RECOVERY_LEASE_ID": lease_id,
            "ERT_RECOVERY_ATTEMPT": str(attempts + 1),
        }
    )
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        state["recovery_failure_reason"] = str(exc)
        state["recovery_failure_class"] = "technical"
        atomic_json(state_file, state)
        return {"status": "TECHNICAL_FAILURE", "state": state.get("state"), "reason": "recovery_launch_failed"}
    state["recovery_pid"] = process.pid
    state["state"] = "TRAINING"
    atomic_json(state_file, state)
    return {
        "status": "RECOVERY_HANDOFF",
        "state": state["state"],
        "pid": process.pid,
        "recovery_lease_id": lease_id,
    }


def reconcile_campaign_state(state: dict[str, Any], path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Reconcile a multi-job campaign using orchestrator evidence only."""
    training_status, reason, orch = campaign_training_status(state, path)
    owner_kind = _nested(state, "postprocess", "owner_kind", "orchestrator_dag")
    if owner_kind not in {"orchestrator_dag", "external_registered_command"}:
        raise ReconcileError(f"unsupported postprocess owner_kind: {owner_kind}")
    if training_status == "running":
        if state.get("state") != "LAUNCHING":
            state["state"] = "TRAINING"
        atomic_json(path, state)
        return {"status": "NO_OP", "reason": "training_campaign_active", "state": state["state"]}
    if training_status == "failed":
        state["state"] = "TRAINING_FAILED"
        state["failure_reason"] = reason
        result = delegate_recovery(state, path, reason=reason, args=args)
        if result.get("status") == "RECOVERY_HANDOFF":
            return result
        if result.get("status") == "NO_OP":
            return result
        return result
    state["state"] = "TRAINING_SUCCESS"
    state.setdefault("training_completed_at", iso())
    clear_recovery_lease(state)
    if owner_kind == "orchestrator_dag":
        downstream, downstream_reason = orchestrator_result_status(state, orch)
        if downstream == "active":
            state["state"] = "EVALUATING"
            state["postprocess_state"] = "owned_by_orchestrator_dag"
            atomic_json(path, state)
            return {"status": "NO_OP", "reason": downstream_reason, "state": state["state"]}
        if downstream == "failed":
            failure_class = downstream_reason.split(":", 1)[0]
            state["state"] = "NEEDS_TECHNICAL_RECOVERY" if failure_class == "technical_retryable" else "NEEDS_RESEARCH_DECISION"
            state["postprocess_state"] = "failed"
            key = "technical_recovery_reason" if failure_class == "technical_retryable" else "needs_research_decision_reason"
            state[key] = downstream_reason
            clear_recovery_lease(state)
            atomic_json(path, state)
            return {"status": state["state"], "state": state["state"], "reason": downstream_reason}
        complete_marker, failure_marker = postprocess_marker_paths(state, path)
        if marker_ok(failure_marker, state, failure=True)[0]:
            state["state"] = "NEEDS_RESEARCH_DECISION"
            state["postprocess_state"] = "failed"
            clear_lease(state)
            atomic_json(path, state)
            return {
                "status": "NEEDS_RESEARCH_DECISION",
                "state": state["state"],
                "reason": "postprocess_failure_marker",
            }
        if marker_ok(complete_marker, state)[0]:
            marker = read_json(complete_marker) if complete_marker else {}
            state["state"] = completed_final_state(marker)
            state["postprocess_state"] = "complete"
            clear_lease(state)
            atomic_json(path, state)
            return {"status": "COMPLETE", "state": state["state"], "reason": "postprocess_marker_seen"}
        if complete_marker is None or failure_marker is None:
            state["state"] = "NEEDS_RESEARCH_DECISION"
            state["postprocess_state"] = "not_registered"
            state["needs_research_decision_reason"] = "orchestrator terminal markers are not registered"
            atomic_json(path, state)
            return {
                "status": "NEEDS_RESEARCH_DECISION",
                "state": state["state"],
                "reason": "orchestrator_terminal_markers_not_registered",
            }
        state["state"] = "EVALUATING"
        state["postprocess_state"] = "awaiting_orchestrator_terminal_marker"
        atomic_json(path, state)
        return {"status": "NO_OP", "reason": "orchestrator_dag_terminal_outputs_pending", "state": state["state"]}
    # External registered postprocessing follows the existing lease/handoff
    # path below.  The caller re-enters the existing single-owner handoff
    # implementation without duplicating it here.
    atomic_json(path, state)
    return {"status": "DELEGATE_POSTPROCESS", "state": state["state"]}


def active_lease(state: dict[str, Any]) -> bool:
    expiry = parse_time(state.get("lease_expires_at"))
    # The durable expiry is authoritative.  A crashed owner must not be
    # replaced while its lease is still live; recovery becomes eligible only
    # after expiry, at which point the bounded retry path can reclaim it.
    return expiry is not None and expiry > now()


def owner_string() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def acquire_lease(state: dict[str, Any], *, lease_seconds: float) -> str | None:
    if active_lease(state):
        return None
    lease_id = uuid.uuid4().hex
    started = now()
    owner = owner_string()
    state.update(
        {
            "owner": owner,
            "postprocess_owner": owner,
            "lease_id": lease_id,
            "acquired_at": iso(started),
            "lease_started_at": iso(started),
            "lease_expires_at": iso(started + dt.timedelta(seconds=lease_seconds)),
        }
    )
    return lease_id


def postprocess_command(state: dict[str, Any]) -> list[str] | None:
    value = state.get("postprocess_command")
    if value is None:
        value = _nested(state, "postprocess", "command")
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ReconcileError("postprocess command must be a non-empty argv list")
    return list(value)


def postprocess_marker_paths(state: dict[str, Any], state_file: Path) -> tuple[Path | None, Path | None]:
    complete = _nested(state, "postprocess", "completion_marker")
    failure = _nested(state, "postprocess", "failure_marker")
    return path_from_state(complete, state_file=state_file), path_from_state(failure, state_file=state_file)


def completed_final_state(marker: dict[str, Any]) -> str:
    final_state = str(marker.get("final_state", "AWAITING_RESEARCH_REVIEW"))
    if final_state not in {"PUSHED", "AWAITING_RESEARCH_REVIEW"}:
        raise ReconcileError(f"postprocess marker has invalid final_state: {final_state}")
    return final_state


def clear_lease(state: dict[str, Any]) -> None:
    for key in ("owner", "postprocess_owner", "lease_id", "acquired_at", "lease_started_at", "lease_expires_at"):
        state.pop(key, None)


def reconcile(args: argparse.Namespace) -> dict[str, Any]:
    path = state_path(args)
    lock = NonBlockingLock(path.with_suffix(path.suffix + ".lock"))
    if not lock.acquire():
        return {"status": "NO_OP", "reason": "concurrent_reconciler", "state": str(path)}
    try:
        state = read_json(path)
        if state.get("schema_version") not in COMPATIBLE_SCHEMA_VERSIONS:
            raise ReconcileError("unsupported experiment-state schema")
        for field in ("experiment_id", "source_sha", "scientific_identity_hash"):
            if not isinstance(state.get(field), str) or not state[field]:
                raise ReconcileError(f"experiment-state missing required identity field: {field}")
        if state.get("experiment_id") and args.experiment_id and state["experiment_id"] != args.experiment_id:
            raise ReconcileError("experiment-id does not match experiment-state")
        state["last_reconciled_at"] = iso()
        current = state.get("state")
        if current not in KNOWN_STATES:
            raise ReconcileError(f"unknown experiment-state lifecycle state: {current}")
        if current in TERMINAL_STATES:
            atomic_json(path, state)
            return {"status": "NO_OP", "reason": "terminal_state", "state": current}
        if state_mode(state) == "orchestrator_campaign":
            campaign_result = reconcile_campaign_state(state, path, args)
            if campaign_result.get("status") != "DELEGATE_POSTPROCESS":
                return campaign_result
            # The campaign reconciler has proven all required training jobs
            # complete and selected the external registered owner.  Continue
            # through the existing lease/marker handoff below in this same
            # locked invocation; no second command or PID is consulted.
            current = "TRAINING_SUCCESS"
        if current == "TRAINING":
            pid = _nested(state, "training", "pid")
            if process_alive(pid):
                atomic_json(path, state)
                return {"status": "NO_OP", "reason": "training_running", "state": current, "pid": pid}
            complete, reason = completion_evidence(state, path)
            if not complete:
                # A dead PID without a valid completion marker is never a success.
                state["state"] = "TRAINING_FAILED"
                state["failure_reason"] = reason
                state["failure_class"] = (
                    "unknown_terminal_failure" if reason.startswith("missing") else reason.split(":", 1)[-1]
                )
                atomic_json(path, state)
                return {"status": "FAILED", "state": state["state"], "reason": reason}
            state["state"] = "TRAINING_SUCCESS"
            state["training_completed_at"] = iso()
            current = "TRAINING_SUCCESS"
        if current == "TRAINING_FAILED":
            atomic_json(path, state)
            return {"status": "NO_OP", "reason": "training_failed_requires_existing_retry_policy", "state": current}
        if current == "TRAINING_SUCCESS":
            complete_marker, failure_marker = postprocess_marker_paths(state, path)
            if marker_ok(complete_marker, state)[0]:
                state["state"] = completed_final_state(read_json(complete_marker))
                state["postprocess_state"] = "complete"
                clear_lease(state)
                atomic_json(path, state)
                return {"status": "COMPLETE", "state": state["state"], "reason": "postprocess_marker_seen"}
            if marker_ok(failure_marker, state, failure=True)[0]:
                state["state"] = "NEEDS_RESEARCH_DECISION"
                state["postprocess_state"] = "failed"
                clear_lease(state)
                atomic_json(path, state)
                return {
                    "status": "NEEDS_RESEARCH_DECISION",
                    "state": state["state"],
                    "reason": "postprocess_failure_marker",
                }
            if active_lease(state):
                atomic_json(path, state)
                return {"status": "NO_OP", "reason": "postprocess_lease_active", "state": "EVALUATING"}
            lease_id = acquire_lease(state, lease_seconds=args.lease_seconds)
            if lease_id is None:  # defensive race check
                atomic_json(path, state)
                return {"status": "NO_OP", "reason": "postprocess_lease_active", "state": "EVALUATING"}
            if complete_marker is None or failure_marker is None:
                state["state"] = "NEEDS_RESEARCH_DECISION"
                state["postprocess_state"] = "not_registered"
                state["needs_research_decision_reason"] = "postprocess terminal markers are not registered"
                clear_lease(state)
                atomic_json(path, state)
                return {
                    "status": "NEEDS_RESEARCH_DECISION",
                    "state": state["state"],
                    "reason": "postprocess_markers_not_registered",
                }
            command = postprocess_command(state)
            if command is None:
                state["state"] = "NEEDS_RESEARCH_DECISION"
                state["postprocess_state"] = "not_registered"
                state["needs_research_decision_reason"] = "no existing postprocess command registered"
                clear_lease(state)
                atomic_json(path, state)
                return {
                    "status": "NEEDS_RESEARCH_DECISION",
                    "state": state["state"],
                    "reason": "postprocess_not_registered",
                }
            expected_command_digest = state.get("postprocess_command_sha256")
            if expected_command_digest is not None and expected_command_digest != argv_digest(command):
                state["state"] = "NEEDS_RESEARCH_DECISION"
                state["postprocess_state"] = "command_drift"
                state["needs_research_decision_reason"] = "registered postprocess command changed after launch"
                clear_lease(state)
                atomic_json(path, state)
                return {
                    "status": "NEEDS_RESEARCH_DECISION",
                    "state": state["state"],
                    "reason": "postprocess_command_drift",
                }
            env = os.environ.copy()
            env.update(
                {
                    "ERT_EXPERIMENT_ID": str(state.get("experiment_id", args.experiment_id)),
                    "ERT_EXPERIMENT_STATE": str(path),
                    "ERT_POSTPROCESS_LEASE_ID": lease_id,
                    "ERT_POSTPROCESS_COMPLETION_MARKER": str(complete_marker or ""),
                    "ERT_POSTPROCESS_FAILURE_MARKER": str(failure_marker or ""),
                }
            )
            attempts = int(state.get("postprocess_attempts", 0)) + 1
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as exc:
                state["state"] = "EVALUATING"
                state["postprocess_state"] = "launch_failed"
                state["postprocess_attempts"] = attempts
                state["postprocess_failure_class"] = "technical"
                state["postprocess_failure_reason"] = str(exc)
                atomic_json(path, state)
                return {
                    "status": "TECHNICAL_FAILURE",
                    "state": state["state"],
                    "reason": "postprocess_launch_failed",
                }
            state["state"] = "EVALUATING"
            state["postprocess_state"] = "running"
            state["postprocess_pid"] = process.pid
            state["postprocess_started_at"] = iso()
            state["postprocess_attempts"] = attempts
            atomic_json(path, state)
            return {"status": "HANDOFF", "state": state["state"], "pid": process.pid, "lease_id": lease_id}
        if current in POSTPROCESS_STATES:
            complete_marker, failure_marker = postprocess_marker_paths(state, path)
            if marker_ok(complete_marker, state)[0]:
                marker = read_json(complete_marker)
                state["state"] = completed_final_state(marker)
                state["postprocess_state"] = "complete"
                clear_lease(state)
                atomic_json(path, state)
                return {"status": "COMPLETE", "state": state["state"], "reason": "postprocess_marker_seen"}
            if marker_ok(failure_marker, state, failure=True)[0]:
                state["state"] = "NEEDS_RESEARCH_DECISION"
                state["postprocess_state"] = "failed"
                clear_lease(state)
                atomic_json(path, state)
                return {
                    "status": "NEEDS_RESEARCH_DECISION",
                    "state": state["state"],
                    "reason": "postprocess_failure_marker",
                }
            if process_alive(state.get("postprocess_pid")) or active_lease(state):
                atomic_json(path, state)
                return {"status": "NO_OP", "reason": "postprocess_running_or_leased", "state": current}
            # A crashed postprocessor may be recovered only by a bounded retry
            # of the same registered command and identity.
            if int(state.get("postprocess_attempts", 0)) >= args.max_postprocess_attempts:
                state["state"] = "NEEDS_RESEARCH_DECISION"
                state["needs_research_decision_reason"] = "postprocess lease expired after bounded attempts"
                clear_lease(state)
                atomic_json(path, state)
                return {
                    "status": "NEEDS_RESEARCH_DECISION",
                    "state": state["state"],
                    "reason": "postprocess_attempt_bound",
                }
            state["state"] = "TRAINING_SUCCESS"
            atomic_json(path, state)
            # Re-enter the same handoff path on this invocation without polling.
            lock.release()
            return reconcile(args)
        atomic_json(path, state)
        return {"status": "NO_OP", "reason": "state_not_actionable", "state": current}
    finally:
        lock.release()


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=False, default="")
    parser.add_argument("--state", type=Path, help="explicit experiment-state.json (useful for tests)")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--scheduled", action="store_true", help="bounded wake mode; never performs repository scans")
    parser.add_argument("--lease-seconds", type=float, default=900.0)
    parser.add_argument("--max-postprocess-attempts", type=int, default=2)
    parser.add_argument("--max-recovery-attempts", type=int, default=2)
    return parser


def main() -> int:
    args = parser().parse_args()
    if args.lease_seconds <= 0 or args.max_postprocess_attempts < 1 or args.max_recovery_attempts < 1:
        raise SystemExit("lease-seconds and retry limits must be >= 1")
    try:
        result = reconcile(args)
    except ReconcileError as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
