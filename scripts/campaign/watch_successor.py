#!/usr/bin/env python3
"""Wait for one recorded phase, then launch one exact successor command."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ard.campaign.gpu import GPUInspectionError, inventory
from ard.campaign.launcher import launch_phase
from ard.campaign.state import FileLock, _atomic_json, utc_now


class SuccessorWatchError(RuntimeError):
    pass


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuccessorWatchError(f"{label} is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise SuccessorWatchError(f"{label} must be an object: {path}")
    return value


def _append_event(path: Path, event: str, **values: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": utc_now(), "event": event, **values}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_predecessor(spec: dict[str, Any]) -> None:
    exit_record = _read_object(Path(spec["predecessor_exit"]), "predecessor exit")
    expected = {
        "exit_code": 0,
        "run_id": spec["predecessor_run_id"],
        "git_sha": spec["scientific_git_sha"],
    }
    drift = {
        key: (expected_value, exit_record.get(key))
        for key, expected_value in expected.items()
        if exit_record.get(key) != expected_value
    }
    if drift:
        raise SuccessorWatchError(f"predecessor did not complete with the expected identity: {drift}")
    completion = _read_object(Path(spec["predecessor_completion"]), "predecessor completion")
    if completion.get("status") != "completed":
        raise SuccessorWatchError("predecessor completion marker is not completed")
    output = Path(spec["predecessor_output"])
    required = (
        output / "resolved_config.yaml",
        output / "best.pt",
        output / "last.pt",
        output / "run-bundle" / "manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SuccessorWatchError("predecessor output is incomplete: " + ", ".join(missing))


def archive_owned_lease(spec: dict[str, Any], state_dir: Path) -> None:
    lock_root = Path(spec["gpu_lock_root"])
    gpu_uuid = str(spec["gpu_uuid"])
    lease = lock_root / f"gpu-{gpu_uuid}.lease.json"
    with FileLock(lock_root / f"gpu-{gpu_uuid}.lock"):
        if not lease.exists():
            return
        record = _read_object(lease, "predecessor GPU lease")
        expected = {
            "job_id": spec["predecessor_run_id"],
            "phase": spec["predecessor_phase"],
            "gpu_uuid": gpu_uuid,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise SuccessorWatchError("GPU lease is not owned by the exact predecessor")
        os.replace(lease, state_dir / "predecessor-lease.json")


def _gpu_idle(spec: dict[str, Any]) -> bool:
    snapshot = next((item for item in inventory() if item.index == int(spec["gpu"])), None)
    if snapshot is None or snapshot.uuid != spec["gpu_uuid"]:
        raise SuccessorWatchError("GPU index/UUID identity drift")
    utilization = snapshot.utilization_percent
    if utilization is None:
        return False
    return (
        not snapshot.processes
        and utilization <= 5
        and snapshot.memory_free_mib >= int(spec["required_free_memory_mib"])
    )


def execute_spec(path: Path) -> int:
    spec = _read_object(path, "successor watch spec")
    state_dir = path.parent
    events = state_dir / "watch-events.jsonl"
    deadline = time.monotonic() + float(spec["timeout_seconds"])
    _append_event(events, "watch_started")
    while not Path(spec["predecessor_exit"]).is_file():
        if time.monotonic() >= deadline:
            raise SuccessorWatchError("timed out waiting for predecessor exit")
        _atomic_json(state_dir / "watch-status.json", {"status": "waiting_predecessor", "at": utc_now()})
        time.sleep(float(spec["poll_seconds"]))
    validate_predecessor(spec)
    _append_event(events, "predecessor_validated")
    repository = Path(spec["repository"])
    if _git(repository, "rev-parse", "HEAD") != spec["runtime_git_sha"] or _git(repository, "status", "--porcelain"):
        raise SuccessorWatchError("watcher runtime repository identity drift")
    while True:
        if time.monotonic() >= deadline:
            raise SuccessorWatchError("timed out waiting for predecessor GPU release")
        try:
            idle = _gpu_idle(spec)
        except GPUInspectionError:
            idle = False
        if idle:
            break
        _atomic_json(state_dir / "watch-status.json", {"status": "waiting_gpu_idle", "at": utc_now()})
        time.sleep(float(spec["poll_seconds"]))
    archive_owned_lease(spec, state_dir)
    _append_event(events, "predecessor_lease_archived")
    completed = subprocess.run(spec["successor_argv"], check=False)
    _append_event(events, "successor_launcher_finished", exit_code=completed.returncode)
    if completed.returncode != 0:
        raise SuccessorWatchError(f"successor launcher exited {completed.returncode}")
    _atomic_json(state_dir / "watch-status.json", {"status": "successor_launched", "at": utc_now()})
    return 0


def launch(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    runtime_sha = _git(repository, "rev-parse", "HEAD")
    if _git(repository, "status", "--porcelain"):
        raise SuccessorWatchError("watcher repository must be clean")
    state_dir = args.state_dir.resolve()
    if state_dir.exists() and any(state_dir.iterdir()):
        raise SuccessorWatchError("watcher state directory is not empty")
    state_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": 1,
        "repository": str(repository),
        "runtime_git_sha": runtime_sha,
        "scientific_git_sha": args.scientific_git_sha,
        "predecessor_exit": str(args.predecessor_exit.resolve()),
        "predecessor_completion": str(args.predecessor_completion.resolve()),
        "predecessor_output": str(args.predecessor_output.resolve()),
        "predecessor_run_id": args.predecessor_run_id,
        "predecessor_phase": args.predecessor_phase,
        "gpu": args.gpu,
        "gpu_uuid": args.gpu_uuid,
        "gpu_lock_root": str(args.gpu_lock_root.resolve()),
        "required_free_memory_mib": args.required_free_memory_mib,
        "poll_seconds": args.poll_seconds,
        "timeout_seconds": args.timeout_seconds,
        "successor_argv": args.successor_argv,
        "created_at": utc_now(),
    }
    spec_path = state_dir / "watch-spec.json"
    _atomic_json(spec_path, spec)
    record = launch_phase(
        [sys.executable, str(Path(__file__).resolve()), "--execute-spec", str(spec_path)],
        cwd=repository,
        stdout_path=state_dir / "stdout.log",
        stderr_path=state_dir / "stderr.log",
        exit_record=state_dir / "exit.json",
        launch_record=state_dir / "launch.json",
        run_id=f"watch-{args.predecessor_run_id}",
        git_sha=runtime_sha,
        environment={"PYTHONPATH": str(repository / "src")},
    )
    return {"status": "watching", "state_dir": str(state_dir), "launch": record}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-spec", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--scientific-git-sha")
    parser.add_argument("--predecessor-exit", type=Path)
    parser.add_argument("--predecessor-completion", type=Path)
    parser.add_argument("--predecessor-output", type=Path)
    parser.add_argument("--predecessor-run-id")
    parser.add_argument("--predecessor-phase", default="train")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--gpu-lock-root", type=Path)
    parser.add_argument("--required-free-memory-mib", type=int, default=7500)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=86400.0)
    parser.add_argument("successor_argv", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.execute_spec is not None:
        try:
            return execute_spec(args.execute_spec.resolve())
        except (OSError, subprocess.SubprocessError, SuccessorWatchError, ValueError) as exc:
            parser.error(str(exc))
    required = (
        "repository",
        "state_dir",
        "scientific_git_sha",
        "predecessor_exit",
        "predecessor_completion",
        "predecessor_output",
        "predecessor_run_id",
        "gpu",
        "gpu_uuid",
        "gpu_lock_root",
    )
    if args.successor_argv and args.successor_argv[0] == "--":
        args.successor_argv = args.successor_argv[1:]
    missing = [name for name in required if getattr(args, name) is None]
    if missing or not args.successor_argv:
        parser.error("missing launch arguments: " + ", ".join(missing or ["successor argv"]))
    try:
        result = launch(args)
    except (OSError, subprocess.SubprocessError, SuccessorWatchError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
