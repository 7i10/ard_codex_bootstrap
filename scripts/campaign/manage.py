#!/usr/bin/env python3
"""Prepare and control one immutable host-local ARD campaign worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

_SHA = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_MIN_FREE_BYTES = 50 * 1024**3
_TEACHERS = {
    "Chen2021LTD_WRN34_10.pt": "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983",
    "Bartoldson2024Adversarial_WRN-94-16.pt": "56bbad8ad748df86e67c24dba4f59a9e7d285e583251460b2ed154017a18cb0b",
}
_PILOT_BASE_CHECKS = {
    "finite_train_metrics",
    "best_last_pgd_10000",
    "terminal_lineage",
    "wandb_completed",
    "process_adoption",
    "execution_profile_match",
}


class ManagementError(RuntimeError):
    pass


def _run(command: Sequence[str], *, cwd: Path | None = None, environment: dict[str, str] | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "")
        raise ManagementError(f"command failed: {list(command)!r}: {stderr}") from exc
    return result.stdout.strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _validate_run_id(value: str) -> str:
    if not _RUN_ID.fullmatch(value) or ".." in value:
        raise ManagementError("unsafe run ID")
    return value


def _validate_sha(value: str) -> str:
    if not _SHA.fullmatch(value):
        raise ManagementError("a full lowercase 40-hex Git SHA is required")
    return value


def _metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "control" / "campaign-run.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManagementError(f"campaign run metadata is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ManagementError("campaign run metadata must be an object")
    return value


def _metadata_or_adopt(run_dir: Path, sha_argument: str | None) -> dict[str, Any]:
    if (run_dir / "control" / "campaign-run.json").is_file():
        return _metadata(run_dir)
    if sha_argument is None:
        raise ManagementError("campaign run metadata is absent; --sha is required to adopt a prepared worktree")
    sha = _validate_sha(sha_argument)
    repository = run_dir / "repo"
    if not repository.is_dir() or _run(["git", "rev-parse", "HEAD"], cwd=repository) != sha:
        raise ManagementError("prepared fixed worktree does not match --sha")
    for name in (".external", "teacher_cache"):
        if not (repository / name).exists():
            raise ManagementError(f"prepared fixed worktree lacks runtime asset: {name}")
    checkpoint_hashes: dict[str, str] = {}
    for filename, expected in _TEACHERS.items():
        checkpoint = repository / "teacher_cache" / "robustbench" / filename
        if not checkpoint.is_file() or _sha256(checkpoint) != expected:
            raise ManagementError(f"prepared teacher checkpoint hash mismatch: {filename}")
        checkpoint_hashes[filename] = expected
    for name in ("outputs", "state", "control"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": 1,
        "run_id": run_dir.name,
        "git_sha": sha,
        "source_repo": None,
        "repository": str(repository.resolve()),
        "output_root": str((run_dir / "outputs").resolve()),
        "state_root": str((run_dir / "state").resolve()),
        "adopted_fixed_worktree": True,
        "checkpoint_sha256": checkpoint_hashes,
    }
    _atomic_json(run_dir / "control" / "campaign-run.json", metadata)
    return metadata


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
    result = [part.decode("utf-8", errors="surrogateescape") for part in raw.split(b"\0") if part]
    return result or None


def _argv_digest(argv: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(argv), separators=(",", ":")).encode()).hexdigest()


def _controller_live(record: dict[str, Any]) -> bool:
    try:
        pid = int(record["pid"])
        start = int(record["start_time_ticks"])
        cwd = str(record["cwd"])
        digest = str(record["argv_digest"])
    except (KeyError, TypeError, ValueError):
        return False
    argv = _proc_argv(pid)
    return (
        _proc_start_time(pid) == start and _proc_cwd(pid) == cwd and argv is not None and _argv_digest(argv) == digest
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_repo.resolve()
    run_root = args.run_root.resolve()
    run_id = _validate_run_id(args.run_id)
    sha = _validate_sha(args.sha)
    run_dir = run_root / run_id
    if run_dir.exists():
        raise ManagementError(f"refusing to reuse campaign run directory: {run_dir}")
    if shutil.disk_usage(_nearest_existing(run_root)).free < _MIN_FREE_BYTES:
        raise ManagementError("campaign run root has less than 50 GiB free")
    observed = _run(["git", "rev-parse", f"{sha}^{{commit}}"], cwd=source)
    if observed != sha:
        raise ManagementError("requested Git object did not resolve to the exact commit")
    run_dir.mkdir(parents=True)
    repository = run_dir / "repo"
    try:
        _run(["git", "worktree", "add", "--detach", str(repository), sha], cwd=source)
        for name in (".external", "teacher_cache"):
            target = source / name
            if not target.exists():
                raise ManagementError(f"required runtime asset is absent: {target}")
            (repository / name).symlink_to(target.resolve(), target_is_directory=True)
        for name in ("outputs", "state", "control"):
            (run_dir / name).mkdir(exist_ok=True)
        checkpoint_hashes: dict[str, str] = {}
        for filename, expected in _TEACHERS.items():
            checkpoint = repository / "teacher_cache" / "robustbench" / filename
            if not checkpoint.is_file():
                raise ManagementError(f"teacher checkpoint is absent: {checkpoint}")
            actual = _sha256(checkpoint)
            if actual != expected:
                raise ManagementError(f"teacher checkpoint hash mismatch: {filename}")
            checkpoint_hashes[filename] = actual
        metadata = {
            "version": 1,
            "run_id": run_id,
            "git_sha": sha,
            "source_repo": str(source),
            "repository": str(repository),
            "output_root": str(run_dir / "outputs"),
            "state_root": str(run_dir / "state"),
            "checkpoint_sha256": checkpoint_hashes,
        }
        _atomic_json(run_dir / "control" / "campaign-run.json", metadata)
        return metadata
    except Exception:
        # Keep a failed preparation for inspection.  It is never treated as a
        # successful run and is not automatically deleted or reused.
        (run_dir / "PREPARE_FAILED").write_text("campaign preparation failed; inspect before cleanup\n")
        raise


def _runtime_environment(run_dir: Path, repository: Path) -> dict[str, str]:
    dataset = Path(
        os.environ.get("ARD_CIFAR10_ROOT", "/home/shunsukenaito/workspace-local/datasets/ard/torchvision")
    ).resolve()
    if not dataset.is_dir():
        raise ManagementError(f"CIFAR-10 root is absent: {dataset}")
    if not (os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").is_file()):
        raise ManagementError("W&B online credentials are not discoverable")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(repository / "src"),
            "ARD_CIFAR10_ROOT": str(dataset),
            "ARD_NUM_WORKERS": os.environ.get("ARD_NUM_WORKERS", "4"),
            "ARD_CAMPAIGN_ALLOW_EXTERNAL_GPU_PROCESSES": "1",
            "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT": str(
                repository / "teacher_cache" / "robustbench" / "Chen2021LTD_WRN34_10.pt"
            ),
            "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT_SHA256": _TEACHERS["Chen2021LTD_WRN34_10.pt"],
            "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT": str(
                repository / "teacher_cache" / "robustbench" / "Bartoldson2024Adversarial_WRN-94-16.pt"
            ),
            "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT_SHA256": _TEACHERS[
                "Bartoldson2024Adversarial_WRN-94-16.pt"
            ],
        }
    )
    return environment


def _validate_pilot_evidence(path: Path, *, sha: str) -> dict[str, Any]:
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManagementError(f"pilot acceptance evidence is invalid: {path}") from exc
    if not isinstance(evidence, dict):
        raise ManagementError("pilot acceptance evidence must be an object")
    if (
        evidence.get("version") != 1
        or evidence.get("status") != "accepted"
        or evidence.get("git_sha") != sha
        or evidence.get("execution_profile") != "ws1_prb128_gb128_localbn_v1"
    ):
        raise ManagementError("pilot acceptance evidence is absent, stale, or not accepted")
    pilots = evidence.get("pilots")
    expected = {
        "pilot-h-chen-rslad-s0",
        "pilot-h-chen-joint-s0",
        "pilot-f-bart-rslad-s0",
    }
    if not isinstance(pilots, dict) or set(pilots) != expected:
        raise ManagementError("pilot acceptance evidence does not contain the exact required pilot set")
    for pilot_id, value in pilots.items():
        if not isinstance(value, dict):
            raise ManagementError(f"pilot evidence is invalid: {pilot_id}")
        peak = value.get("peak_reserved_mib")
        checks = value.get("checks")
        expected_checks = _PILOT_BASE_CHECKS | (
            {"joint_post_warmup_signal_active"} if pilot_id == "pilot-h-chen-joint-s0" else set()
        )
        hashes = (
            value.get("training_manifest_sha256"),
            value.get("evaluation_results_sha256"),
            value.get("job_state_sha256"),
            value.get("training_config_hash"),
            value.get("threat_hash"),
        )
        if (
            value.get("job_id") != pilot_id
            or value.get("state") != "completed"
            or value.get("wandb_run_id") != f"{pilot_id}-{sha[:7]}"
            or not isinstance(value.get("output"), str)
            or not value["output"]
            or any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item) for item in hashes)
            or isinstance(peak, bool)
            or not isinstance(peak, (int, float))
            or not math.isfinite(float(peak))
            or float(peak) <= 0
            or not isinstance(checks, dict)
            or set(checks) != expected_checks
            or not all(value is True for value in checks.values())
        ):
            raise ManagementError(f"pilot evidence is incomplete or failed: {pilot_id}")
    return evidence


def start(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    metadata = _metadata_or_adopt(run_dir, args.sha)
    repository = Path(metadata["repository"]).resolve()
    sha = _validate_sha(str(metadata["git_sha"]))
    if _run(["git", "rev-parse", "HEAD"], cwd=repository) != sha:
        raise ManagementError("fixed worktree HEAD drift")
    if _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repository):
        raise ManagementError("fixed worktree is dirty")
    remote_contains = _run(["git", "branch", "-r", "--contains", sha], cwd=repository).splitlines()
    if not any(line.strip().startswith("origin/") for line in remote_contains):
        raise ManagementError("campaign SHA is not contained by an origin remote-tracking branch")
    if shutil.disk_usage(run_dir).free < _MIN_FREE_BYTES:
        raise ManagementError("campaign run root has less than 50 GiB free")
    campaign = (repository / args.campaign).resolve()
    if repository not in campaign.parents or not campaign.is_file():
        raise ManagementError("campaign path is missing or escapes the fixed worktree")
    campaign_data = yaml.safe_load(campaign.read_text(encoding="utf-8"))
    if not isinstance(campaign_data, dict):
        raise ManagementError("campaign structure is invalid")
    pilot_evidence: dict[str, Any] | None = None
    if campaign_data.get("campaign_id") == "c10-r18-ws1-b128-core-s0-v1":
        if args.pilot_evidence is None:
            raise ManagementError("production campaign requires --pilot-evidence")
        pilot_evidence = _validate_pilot_evidence(args.pilot_evidence.resolve(), sha=sha)
        if not isinstance(campaign_data.get("jobs"), list):
            raise ManagementError("production campaign structure is invalid")
        pilot_rows = pilot_evidence["pilots"]
        observed_by_teacher = {
            "chen2021_ltd_wrn34_10": max(
                float(pilot_rows["pilot-h-chen-rslad-s0"]["peak_reserved_mib"]),
                float(pilot_rows["pilot-h-chen-joint-s0"]["peak_reserved_mib"]),
            ),
            "bartoldson2024_adversarial_wrn94_16": float(pilot_rows["pilot-f-bart-rslad-s0"]["peak_reserved_mib"]),
        }
        for job in campaign_data["jobs"]:
            if not isinstance(job, dict) or job.get("teacher") not in observed_by_teacher:
                raise ManagementError("production campaign has an unknown teacher")
            declared = job.get("pilot_peak_reserved_mib")
            if (
                isinstance(declared, bool)
                or not isinstance(declared, (int, float))
                or float(declared) < observed_by_teacher[str(job["teacher"])]
            ):
                raise ManagementError("production memory gate is below accepted pilot peak")
        durable_evidence = run_dir / "control" / "pilot-acceptance.json"
        if durable_evidence.exists() and durable_evidence.read_bytes() != args.pilot_evidence.resolve().read_bytes():
            raise ManagementError("durable pilot acceptance evidence drift")
        if not durable_evidence.exists():
            shutil.copy2(args.pilot_evidence.resolve(), durable_evidence)
    control = run_dir / "control"
    gpu_lock_root = Path(
        os.environ.get(
            "ARD_CAMPAIGN_GPU_LOCK_ROOT",
            "/home/shunsukenaito/workspace-local/.ard-campaign-gpu-locks",
        )
    ).resolve()
    gpu_lock_root.mkdir(parents=True, exist_ok=True)
    record_path = control / "controller.json"
    if record_path.exists():
        prior = json.loads(record_path.read_text(encoding="utf-8"))
        if isinstance(prior, dict) and _controller_live(prior):
            raise ManagementError("campaign controller is already live")
    environment = _runtime_environment(run_dir, repository)
    common = [
        sys.executable,
        "-m",
        "ard.campaign.cli",
    ]
    identity_args = [
        "--campaign",
        str(campaign),
        "--sha",
        sha,
        "--state-root",
        str(run_dir / "state"),
        "--host",
        args.host,
        "--repository",
        str(repository),
        "--output-root",
        str(run_dir / "outputs"),
        "--gpu-lock-root",
        str(gpu_lock_root),
    ]
    _run(
        [*common, "init", "--campaign", str(campaign), "--sha", sha, "--state-root", str(run_dir / "state")],
        cwd=repository,
        environment=environment,
    )
    _run([*common, "arm", *identity_args], cwd=repository, environment=environment)
    argv = [
        *common,
        "run-loop",
        *identity_args,
        "--allow-external-gpu-processes",
        "--interval-seconds",
        str(args.interval_seconds),
    ]
    stdout = (control / "controller.stdout.log").open("ab")
    stderr = (control / "controller.stderr.log").open("ab")
    try:
        process = subprocess.Popen(
            argv,
            cwd=repository,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        stdout.close()
        stderr.close()
    start_time = _proc_start_time(process.pid)
    if start_time is None:
        raise ManagementError("controller vanished before its identity was recorded")
    record = {
        "version": 1,
        "pid": process.pid,
        "pgid": process.pid,
        "start_time_ticks": start_time,
        "cwd": str(repository),
        "argv": argv,
        "argv_digest": _argv_digest(argv),
        "git_sha": sha,
        "host": args.host,
        "campaign": args.campaign,
        "gpu_lock_root": str(gpu_lock_root),
        "pilot_evidence_sha256": (
            None if pilot_evidence is None else _sha256(run_dir / "control" / "pilot-acceptance.json")
        ),
    }
    _atomic_json(record_path, record)
    return {**record, "run_dir": str(run_dir)}


def status(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    metadata = _metadata(run_dir)
    controller_path = run_dir / "control" / "controller.json"
    controller: dict[str, Any] | None = None
    if controller_path.is_file():
        raw = json.loads(controller_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            controller = {**raw, "live": _controller_live(raw)}
    campaign_path = run_dir / "state" / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8")) if campaign_path.is_file() else None
    jobs: dict[str, Any] = {}
    for path in sorted((run_dir / "state" / "jobs").glob("*.json")):
        jobs[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return {"metadata": metadata, "controller": controller, "campaign": campaign, "jobs": jobs}


def stop(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    record_path = run_dir / "control" / "controller.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManagementError("controller record is absent or invalid") from exc
    if not isinstance(record, dict) or not _controller_live(record):
        raise ManagementError("controller identity is not live; refusing to signal a bare PID")
    os.killpg(int(record["pgid"]), signal.SIGTERM)
    return {
        "stopped_controller_pgid": int(record["pgid"]),
        "training_phases_untouched": True,
        "run_dir": str(run_dir),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source-repo", type=Path, required=True)
    prepare_parser.add_argument("--run-root", type=Path, required=True)
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.add_argument("--sha", required=True)
    start_parser = commands.add_parser("start")
    start_parser.add_argument("--run-dir", type=Path, required=True)
    start_parser.add_argument("--campaign", required=True)
    start_parser.add_argument("--host", choices=("hamster", "ferret"), required=True)
    start_parser.add_argument("--sha")
    start_parser.add_argument("--pilot-evidence", type=Path)
    start_parser.add_argument("--interval-seconds", type=float, default=20.0)
    for name in ("status", "stop"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = {"prepare": prepare, "start": start, "status": status, "stop": stop}[args.command](args)
    except ManagementError as exc:
        _parser().error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
