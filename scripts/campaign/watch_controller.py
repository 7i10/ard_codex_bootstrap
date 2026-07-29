#!/usr/bin/env python3
"""Host-local singleton watchdog for an armed immutable campaign controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ard.campaign.schema import CampaignError, bind_git_sha, load_campaign
from ard.campaign.state import CampaignStateStore, FileLock, StateError


class WatchError(RuntimeError):
    pass


_CHEN_SHA256 = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"
_BARTOLDSON_SHA256 = "56bbad8ad748df86e67c24dba4f59a9e7d285e583251460b2ed154017a18cb0b"


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WatchError(f"{label} is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise WatchError(f"{label} must be a JSON object: {path}")
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


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def controller_live(record: dict[str, Any], *, repository: Path, host: str, sha: str) -> bool:
    try:
        pid = int(record["pid"])
        start = int(record["start_time_ticks"])
        cwd = str(record["cwd"])
        expected_argv = record["argv"]
        digest = str(record["argv_digest"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        record.get("git_sha") != sha
        or record.get("host") != host
        or cwd != str(repository)
        or not isinstance(expected_argv, list)
        or not all(isinstance(item, str) for item in expected_argv)
    ):
        return False
    argv = _proc_argv(pid)
    return (
        _proc_start_time(pid) == start
        and _proc_cwd(pid) == cwd
        and argv == expected_argv
        and argv is not None
        and _argv_digest(argv) == digest
    )


def _fixed_campaign(args: argparse.Namespace) -> tuple[Any, Path, Path]:
    run_dir = args.run_dir.resolve()
    metadata = _read_object(run_dir / "control" / "campaign-run.json", "campaign run metadata")
    repository = Path(str(metadata.get("repository", ""))).resolve()
    sha = str(metadata.get("git_sha", ""))
    if not repository.is_dir() or not (repository / "scripts" / "campaign" / "manage.py").is_file():
        raise WatchError("fixed worktree or manage.py is absent")
    try:
        observed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise WatchError("unable to verify fixed worktree SHA") from exc
    if observed != sha:
        raise WatchError("fixed worktree HEAD drift")
    campaign = (repository / args.campaign).resolve()
    if repository not in campaign.parents or not campaign.is_file():
        raise WatchError("campaign path is missing or escapes fixed worktree")
    try:
        spec = bind_git_sha(load_campaign(campaign), sha)
    except CampaignError as exc:
        raise WatchError(str(exc)) from exc
    return spec, repository, campaign


def _control_identity() -> tuple[Path, str]:
    repository = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise WatchError("unable to verify controller source identity") from exc
    if len(sha) != 40 or dirty:
        raise WatchError("controller source must be a clean committed worktree")
    return repository, sha


def _gpu_lock_root(args: argparse.Namespace, record_path: Path) -> Path:
    if args.gpu_lock_root is not None:
        return args.gpu_lock_root.resolve()
    if record_path.is_file():
        prior = _read_object(record_path, "controller record")
        value = prior.get("gpu_lock_root")
        if isinstance(value, str) and Path(value).is_absolute():
            return Path(value).resolve()
    return Path(
        os.environ.get(
            "ARD_CAMPAIGN_GPU_LOCK_ROOT",
            "/home/shunsukenaito/workspace-local/.ard-campaign-gpu-locks",
        )
    ).resolve()


def _controller_environment(scientific_repository: Path, control_repository: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(control_repository / "src"),
            "ARD_CIFAR10_ROOT": os.environ.get(
                "ARD_CIFAR10_ROOT",
                "/home/shunsukenaito/workspace-local/datasets/ard/torchvision",
            ),
            "ARD_NUM_WORKERS": os.environ.get("ARD_NUM_WORKERS", "4"),
            "ARD_CAMPAIGN_ALLOW_EXTERNAL_GPU_PROCESSES": "1",
            "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT": str(
                scientific_repository / "teacher_cache" / "robustbench" / "Chen2021LTD_WRN34_10.pt"
            ),
            "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT_SHA256": _CHEN_SHA256,
            "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT": str(
                scientific_repository
                / "teacher_cache"
                / "robustbench"
                / "Bartoldson2024Adversarial_WRN-94-16.pt"
            ),
            "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT_SHA256": _BARTOLDSON_SHA256,
        }
    )
    return environment


def _launch_controller(
    args: argparse.Namespace,
    *,
    spec: Any,
    scientific_repository: Path,
    campaign: Path,
    control_repository: Path,
    control_sha: str,
    record_path: Path,
) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    gpu_lock_root = _gpu_lock_root(args, record_path)
    gpu_lock_root.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        "-m",
        "ard.campaign.cli",
        "run-loop",
        "--campaign",
        str(campaign),
        "--sha",
        spec.git_sha,
        "--state-root",
        str(run_dir / "state"),
        "--host",
        args.host,
        "--repository",
        str(scientific_repository),
        "--output-root",
        str(run_dir / "outputs"),
        "--gpu-lock-root",
        str(gpu_lock_root),
        "--allow-external-gpu-processes",
        "--interval-seconds",
        str(args.interval_seconds),
    ]
    environment = _controller_environment(scientific_repository, control_repository)
    stdout = (run_dir / "control" / "controller.stdout.log").open("ab")
    stderr = (run_dir / "control" / "controller.stderr.log").open("ab")
    try:
        process = subprocess.Popen(
            argv,
            cwd=scientific_repository,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        raise WatchError("unable to launch campaign controller") from exc
    finally:
        stdout.close()
        stderr.close()
    start_time = _proc_start_time(process.pid)
    if start_time is None:
        raise WatchError("controller vanished before its identity was recorded")
    record = {
        "version": 2,
        "pid": process.pid,
        "pgid": process.pid,
        "start_time_ticks": start_time,
        "cwd": str(scientific_repository),
        "argv": argv,
        "argv_digest": _argv_digest(argv),
        "git_sha": spec.git_sha,
        "host": args.host,
        "campaign": str(campaign),
        "gpu_lock_root": str(gpu_lock_root),
        "control_repository": str(control_repository),
        "control_git_sha": control_sha,
    }
    _atomic_json(record_path, record)
    return record


def watch(args: argparse.Namespace) -> int:
    spec, repository, campaign = _fixed_campaign(args)
    control_repository, control_sha = _control_identity()
    state = CampaignStateStore(args.run_dir.resolve() / "state")
    state.assert_campaign_identity(spec)
    lock = FileLock(args.run_dir.resolve() / "control" / f"watch-controller-{args.host}.lock")
    if not lock.acquire(blocking=False):
        raise WatchError("host-local controller watchdog is already running")
    try:
        backoff = args.interval_seconds
        record_path = args.run_dir.resolve() / "control" / "controller.json"
        while True:
            campaign_state = state.campaign().get("state")
            if campaign_state == "awaiting_scientific_review":
                return 0
            if campaign_state != "armed":
                raise WatchError(f"watchdog refuses campaign state: {campaign_state!r}")
            live = False
            if record_path.exists():
                record = _read_object(record_path, "controller record")
                live = controller_live(record, repository=repository, host=args.host, sha=spec.git_sha)
            if live:
                if (
                    record.get("control_repository") != str(control_repository)
                    or record.get("control_git_sha") != control_sha
                ):
                    raise WatchError("a live controller uses a different or unrecorded control-plane revision")
                time.sleep(args.interval_seconds)
                backoff = args.interval_seconds
                continue
            try:
                _launch_controller(
                    args,
                    spec=spec,
                    scientific_repository=repository,
                    campaign=campaign,
                    control_repository=control_repository,
                    control_sha=control_sha,
                    record_path=record_path,
                )
                backoff = args.interval_seconds
            except WatchError:
                backoff = min(args.max_backoff_seconds, max(args.interval_seconds, backoff * 2))
            time.sleep(backoff)
    finally:
        lock.release()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--host", choices=("hamster", "ferret"), required=True)
    parser.add_argument("--gpu-lock-root", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=20.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not 1.0 <= args.interval_seconds <= args.max_backoff_seconds <= 300.0:
        parser.error("interval/backoff must satisfy 1 <= interval <= max-backoff <= 300")
    try:
        return watch(args)
    except (WatchError, StateError) as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
