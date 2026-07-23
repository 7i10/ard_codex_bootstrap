"""Detached argv-only phase launcher with durable exit records."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .state import _atomic_json, utc_now


class LaunchError(RuntimeError):
    pass


def argv_digest(argv: Sequence[str]) -> str:
    if not argv or not all(isinstance(item, str) and item and "\x00" not in item for item in argv):
        raise LaunchError("a phase command must be a non-empty argv array without NUL")
    return hashlib.sha256(json.dumps(list(argv), separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _proc_start_time_ticks(pid: int) -> int | None:
    try:
        # The executable name may contain spaces in parentheses; starttime is
        # field 22 and everything after the final ')' is whitespace-delimited.
        contents = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = contents.rsplit(")", maxsplit=1)[1].split()
        return int(fields[19])
    except (FileNotFoundError, IndexError, OSError, ValueError):
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
    values = [part.decode("utf-8", errors="surrogateescape") for part in raw.split(b"\0") if part]
    return values or None


def process_identity(pid: int, *, cwd: Path, argv: Sequence[str]) -> dict[str, Any]:
    start_time = _proc_start_time_ticks(pid)
    if start_time is None:
        raise LaunchError(f"process vanished before identity could be recorded: {pid}")
    return {
        "pid": pid,
        "start_time_ticks": start_time,
        "cwd": str(cwd.resolve()),
        "argv_digest": argv_digest(argv),
    }


def process_matches(record: dict[str, Any]) -> bool:
    """Require PID, immutable start time, cwd, and wrapper argv to agree."""
    try:
        pid = int(record["pid"])
        expected_start = int(record["start_time_ticks"])
        expected_cwd = str(record["cwd"])
        expected_digest = str(record["argv_digest"])
    except (KeyError, TypeError, ValueError):
        return False
    current_start = _proc_start_time_ticks(pid)
    current_argv = _proc_argv(pid)
    return (
        current_start == expected_start
        and _proc_cwd(pid) == expected_cwd
        and current_argv is not None
        and argv_digest(current_argv) == expected_digest
    )


def read_exit_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("exit_code"), int):
            raise ValueError("invalid exit record")
        return payload
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise LaunchError(f"invalid phase exit record: {path}") from exc


def read_launch_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("wrapper"), dict):
            raise ValueError("invalid launch record")
        return payload
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise LaunchError(f"invalid phase launch record: {path}") from exc


def launch_phase(
    argv: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    exit_record: Path,
    launch_record: Path | None = None,
    gpu_lease_path: Path | None = None,
    lease_handshake: Path | None = None,
    run_id: str,
    git_sha: str,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Start an independent wrapper process without a shell or inherited tty."""
    phase_digest = argv_digest(argv)
    launch_record = launch_record or exit_record.with_name("launch.json")
    if exit_record.exists():
        raise LaunchError(f"refusing to overwrite an existing exit record: {exit_record}")
    if launch_record.exists():
        raise LaunchError(f"refusing to overwrite an existing launch record: {launch_record}")
    if (gpu_lease_path is None) != (lease_handshake is None):
        raise LaunchError("GPU lease path and handshake path must be supplied together")
    if not cwd.is_dir():
        raise LaunchError(f"phase cwd does not exist: {cwd}")
    for path in (stdout_path, stderr_path, exit_record, launch_record, lease_handshake):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_argv = [
        sys.executable,
        "-m",
        "ard.campaign.launcher",
        "--phase-wrapper",
        "--exit-record",
        str(exit_record.resolve()),
        "--launch-record",
        str(launch_record.resolve()),
        "--stdout",
        str(stdout_path.resolve()),
        "--stderr",
        str(stderr_path.resolve()),
        "--run-id",
        run_id,
        "--git-sha",
        git_sha,
        "--phase-argv-digest",
        phase_digest,
    ]
    if gpu_lease_path is not None and lease_handshake is not None:
        wrapper_argv.extend(
            ["--gpu-lease", str(gpu_lease_path.resolve()), "--lease-handshake", str(lease_handshake.resolve())]
        )
    wrapper_argv.extend(["--", *argv])
    env = os.environ.copy()
    if environment:
        env.update(environment)
    # A detached wrapper changes cwd.  Relative PYTHONPATH entries would then
    # silently point at the output directory instead of the immutable source.
    if env.get("PYTHONPATH"):
        entries: list[str] = []
        for entry in env["PYTHONPATH"].split(os.pathsep):
            if not entry:
                entries.append(entry)
            else:
                candidate = Path(entry)
                entries.append(str(candidate if candidate.is_absolute() else (Path.cwd() / candidate).resolve()))
        env["PYTHONPATH"] = os.pathsep.join(entries)
    try:
        process = subprocess.Popen(
            wrapper_argv,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        raise LaunchError("unable to start detached phase wrapper") from exc
    # The detached wrapper writes its own launch record before starting the
    # scientific child.  If this caller dies immediately after Popen, a new
    # controller can still adopt the wrapper from that durable handshake.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        record = read_launch_record(launch_record)
        if record is not None:
            wrapper = record.get("wrapper")
            if not isinstance(wrapper, dict) or wrapper.get("pid") != process.pid:
                raise LaunchError("wrapper launch record PID drift")
            return record
        if process.poll() is not None:
            raise LaunchError("phase wrapper exited before writing its launch record")
        time.sleep(0.01)
    raise LaunchError("phase wrapper did not write its launch record")


def phase_is_live(record: dict[str, Any]) -> bool:
    wrapper = record.get("wrapper")
    return isinstance(wrapper, dict) and process_matches(wrapper)


def _phase_wrapper(arguments: argparse.Namespace) -> int:
    argv = list(arguments.argv)
    if argv and argv[0] == "--":  # argparse keeps this on some invocations.
        argv = argv[1:]
    if not argv:
        raise LaunchError("phase wrapper requires argv after --")
    if argv_digest(argv) != arguments.phase_argv_digest:
        raise LaunchError("phase argv digest drift in wrapper")
    stdout = Path(arguments.stdout)
    stderr = Path(arguments.stderr)
    exit_record = Path(arguments.exit_record)
    launch_record = Path(arguments.launch_record)
    if exit_record.exists():
        raise LaunchError("phase wrapper will not overwrite an exit record")
    if launch_record.exists():
        raise LaunchError("phase wrapper will not overwrite a launch record")
    wrapper_argv = _proc_argv(os.getpid())
    if wrapper_argv is None:
        raise LaunchError("phase wrapper cannot inspect its own argv")
    _atomic_json(
        launch_record,
        {
            "wrapper": process_identity(os.getpid(), cwd=Path.cwd(), argv=wrapper_argv),
            "phase_argv_digest": arguments.phase_argv_digest,
            "run_id": arguments.run_id,
            "git_sha": arguments.git_sha,
            "exit_record": str(exit_record.resolve()),
            "launch_record": str(launch_record.resolve()),
            "gpu_lease_path": None if arguments.gpu_lease is None else str(arguments.gpu_lease.resolve()),
            "lease_handshake": (
                None if arguments.lease_handshake is None else str(arguments.lease_handshake.resolve())
            ),
            "launched_at": utc_now(),
        },
    )
    lease_handle: Any | None = None
    if arguments.gpu_lease is not None:
        assert arguments.lease_handshake is not None
        lease_path = Path(arguments.gpu_lease)
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        lease_handle = lease_path.open("a+", encoding="utf-8")
        fcntl.flock(lease_handle.fileno(), fcntl.LOCK_EX)
        _atomic_json(
            Path(arguments.lease_handshake),
            {
                "version": 1,
                "wrapper_pid": os.getpid(),
                "run_id": arguments.run_id,
                "git_sha": arguments.git_sha,
                "phase_argv_digest": arguments.phase_argv_digest,
                "acquired_at": utc_now(),
            },
        )
    error: str | None
    try:
        with stdout.open("ab") as out, stderr.open("ab") as err:
            process = subprocess.Popen(
                argv,
                cwd=os.getcwd(),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=False,
                close_fds=True,
            )
            code = process.wait()
    except OSError as exc:
        code = 127
        error = repr(exc)
    else:
        error = None
    try:
        _atomic_json(
            exit_record,
            {
                "version": 1,
                "exit_code": code,
                "finished_at": utc_now(),
                "run_id": arguments.run_id,
                "git_sha": arguments.git_sha,
                "phase_argv_digest": arguments.phase_argv_digest,
                "wrapper_pid": os.getpid(),
                "error": error,
            },
        )
    finally:
        if lease_handle is not None:
            fcntl.flock(lease_handle.fileno(), fcntl.LOCK_UN)
            lease_handle.close()
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-wrapper", action="store_true")
    parser.add_argument("--exit-record", type=Path)
    parser.add_argument("--launch-record", type=Path)
    parser.add_argument("--stdout", type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--git-sha")
    parser.add_argument("--phase-argv-digest")
    parser.add_argument("--gpu-lease", type=Path)
    parser.add_argument("--lease-handshake", type=Path)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    if not arguments.phase_wrapper:
        parser.error("this module is only the detached phase wrapper")
    required = (
        arguments.exit_record,
        arguments.launch_record,
        arguments.stdout,
        arguments.stderr,
        arguments.run_id,
        arguments.git_sha,
        arguments.phase_argv_digest,
    )
    if any(value is None for value in required):
        parser.error("phase wrapper metadata is required")
    if (arguments.gpu_lease is None) != (arguments.lease_handshake is None):
        parser.error("GPU lease path and handshake path must be supplied together")
    try:
        return _phase_wrapper(arguments)
    except LaunchError as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
