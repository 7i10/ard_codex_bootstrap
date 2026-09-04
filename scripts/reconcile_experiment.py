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
SCHEMA_VERSION = 1
TERMINAL_STATES = {"PUSHED", "AWAITING_RESEARCH_REVIEW", "NEEDS_RESEARCH_DECISION"}
POSTPROCESS_STATES = {"EVALUATING", "SUMMARIZING", "PUSHED", "AWAITING_RESEARCH_REVIEW"}
KNOWN_STATES = {
    "PLANNED",
    "IMPLEMENTING",
    "VALIDATING",
    "TRAINING",
    "TRAINING_SUCCESS",
    "TRAINING_FAILED",
    "EVALUATING",
    "SUMMARIZING",
    "PUSHED",
    "AWAITING_RESEARCH_REVIEW",
    "NEEDS_RESEARCH_DECISION",
}
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


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
        if state.get("schema_version") != SCHEMA_VERSION:
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
    return parser


def main() -> int:
    args = parser().parse_args()
    if args.lease_seconds <= 0 or args.max_postprocess_attempts < 1:
        raise SystemExit("lease-seconds must be positive and max-postprocess-attempts >= 1")
    try:
        result = reconcile(args)
    except ReconcileError as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
