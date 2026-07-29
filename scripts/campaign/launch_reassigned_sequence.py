#!/usr/bin/env python3
"""Launch a recorded phase sequence on an idle replacement GPU.

This is an operational escape hatch for a phase that has not started. It does
not mutate the immutable source campaign state; the caller must keep the source
queue from launching the same phase until terminal evidence is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ard.campaign.gpu import inventory
from ard.campaign.launcher import launch_phase
from ard.campaign.state import FileLock, _atomic_json, utc_now

CHEN_SHA256 = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"
BARTOLDSON_SHA256 = "56bbad8ad748df86e67c24dba4f59a9e7d285e583251460b2ed154017a18cb0b"


class ReassignmentError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    config = (args.repository / args.config).resolve()
    output = args.output.resolve()
    commands: list[dict[str, Any]] = []
    for phase in args.phases:
        if phase == "train":
            argv = [sys.executable, "-m", "ard.cli.train", "--config", str(config), "--output", str(output)]
        elif phase == "pgd":
            argv = [
                sys.executable,
                "-m",
                "ard.cli.evaluate",
                "--config",
                str(config),
                "--checkpoint-dir",
                str(output),
                "--output",
                str(output / "evaluation-pgd"),
            ]
        else:
            argv = [
                sys.executable,
                "-m",
                "ard.cli.evaluate",
                "--config",
                str(config),
                "--checkpoint-dir",
                str(output),
                "--output",
                str(output / "evaluation-autoattack"),
                "--allow-autoattack",
                "evaluation.autoattack=true",
            ]
        commands.append({"phase": phase, "argv": argv})
    return commands


def _input_hashes(output: Path, phases: list[str]) -> dict[str, str]:
    if phases == ["train", "pgd"]:
        if output.exists():
            raise ReassignmentError("train destination output already exists")
        return {}
    if phases != ["autoattack"]:
        raise ReassignmentError("only train+pgd or autoattack sequences are allowed")
    required = (
        output / "resolved_config.yaml",
        output / "best.pt",
        output / "last.pt",
        output / "run-bundle" / "manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ReassignmentError("AutoAttack inputs are missing: " + ", ".join(missing))
    return {str(path): _sha256(path) for path in required}


def _environment(args: argparse.Namespace) -> dict[str, str]:
    repository = args.repository.resolve()
    runtime_source = (args.runtime_source or (repository / "src")).resolve()
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": str(runtime_source),
            "ARD_CIFAR10_ROOT": args.dataset_root,
            "ARD_NUM_WORKERS": str(args.num_workers),
            "ARD_JOB_OUTPUT_DIR": str(args.output.resolve()),
            "ARD_RUN_ID": args.wandb_run_id,
            "ARD_SEED": str(args.seed),
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_GROUP": args.wandb_group,
            "WANDB_GROUP_CHEN": args.wandb_group,
            "WANDB_GROUP_BARTOLDSON": args.wandb_group,
            "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT": str(
                repository / "teacher_cache" / "robustbench" / "Chen2021LTD_WRN34_10.pt"
            ),
            "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT_SHA256": CHEN_SHA256,
            "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT": str(
                repository / "teacher_cache" / "robustbench" / "Bartoldson2024Adversarial_WRN-94-16.pt"
            ),
            "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT_SHA256": BARTOLDSON_SHA256,
        }
    )
    return environment


def _execute_spec(path: Path) -> int:
    spec = json.loads(path.read_text(encoding="utf-8"))
    repository = Path(spec["repository"])
    runtime_repository = Path(spec["runtime_source"]).parent
    lease = Path(spec["lease_path"])
    events = path.with_name("sequence-events.jsonl")
    try:
        if _git(repository, "rev-parse", "HEAD") != spec["git_sha"] or _git(repository, "status", "--porcelain"):
            raise ReassignmentError("scientific repository identity changed after launch")
        if (
            _git(runtime_repository, "rev-parse", "HEAD") != spec["runtime_git_sha"]
            or _git(runtime_repository, "status", "--porcelain")
        ):
            raise ReassignmentError("runtime source identity changed after launch")
        for raw_path, expected in spec["input_sha256"].items():
            artifact = Path(raw_path)
            if not artifact.is_file() or _sha256(artifact) != expected:
                raise ReassignmentError(f"reassigned input hash drift: {artifact}")
        for command in spec["commands"]:
            with events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"at": utc_now(), "event": "started", "phase": command["phase"]}) + "\n")
            completed = subprocess.run(command["argv"], cwd=repository, env=os.environ.copy(), check=False)
            with events.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "at": utc_now(),
                            "event": "finished",
                            "phase": command["phase"],
                            "exit_code": completed.returncode,
                        }
                    )
                    + "\n"
                )
            if completed.returncode != 0:
                return completed.returncode
        _atomic_json(path.with_name("sequence-completion.json"), {"status": "completed", "at": utc_now()})
        return 0
    finally:
        lease.unlink(missing_ok=True)


def launch(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    if _git(repository, "rev-parse", "HEAD") != args.git_sha or _git(repository, "status", "--porcelain"):
        raise ReassignmentError("scientific repository must be clean at the exact requested SHA")
    if not (repository / args.config).is_file():
        raise ReassignmentError("config is absent from the scientific repository")
    input_hashes = _input_hashes(args.output.resolve(), args.phases)
    commands = _commands(args)
    runtime_source = (args.runtime_source or (repository / "src")).resolve()
    if not runtime_source.is_dir():
        raise ReassignmentError("runtime source directory is absent")
    runtime_repository = runtime_source.parent
    runtime_git_sha = _git(runtime_repository, "rev-parse", "HEAD")
    if _git(runtime_repository, "status", "--porcelain"):
        raise ReassignmentError("runtime source repository must be clean")
    if args.dry_run:
        return {
            "status": "dry-run",
            "commands": commands,
            "input_sha256": input_hashes,
            "runtime_git_sha": runtime_git_sha,
        }

    snapshots = inventory()
    snapshot = next((item for item in snapshots if item.index == args.gpu), None)
    if snapshot is None or snapshot.uuid != args.gpu_uuid:
        raise ReassignmentError("destination GPU index/UUID mismatch")
    if snapshot.processes or snapshot.utilization_percent > 5:
        raise ReassignmentError("destination GPU is not idle")
    if snapshot.memory_free_mib < args.required_free_memory_mib:
        raise ReassignmentError("destination GPU lacks the required free memory")

    state_dir = args.state_dir.resolve()
    if state_dir.exists() and any(state_dir.iterdir()):
        raise ReassignmentError("reassignment state directory is not empty")
    state_dir.mkdir(parents=True, exist_ok=True)
    lease_root = args.gpu_lock_root.resolve()
    lease = lease_root / f"gpu-{snapshot.uuid}.lease.json"
    lock = FileLock(lease_root / f"gpu-{snapshot.uuid}.lock")
    with lock:
        if lease.exists():
            raise ReassignmentError("destination GPU already has a campaign lease")
        spec_path = state_dir / "sequence-spec.json"
        spec = {
            "version": 1,
            "source_job_id": args.source_job_id,
            "source_host": args.source_host,
            "destination_host": args.destination_host,
            "destination_gpu": args.gpu,
            "destination_gpu_uuid": snapshot.uuid,
            "repository": str(repository),
            "git_sha": args.git_sha,
            "runtime_source": str(runtime_source),
            "runtime_git_sha": runtime_git_sha,
            "output": str(args.output.resolve()),
            "commands": commands,
            "input_sha256": input_hashes,
            "lease_path": str(lease),
            "created_at": utc_now(),
        }
        _atomic_json(spec_path, spec)
        _atomic_json(
            lease,
            {
                "version": 1,
                "kind": "cross-host-phase-reassignment",
                "source_job_id": args.source_job_id,
                "destination_host": args.destination_host,
                "gpu_uuid": snapshot.uuid,
                "state_dir": str(state_dir),
            },
        )
        try:
            record = launch_phase(
                [sys.executable, str(Path(__file__).resolve()), "--execute-spec", str(spec_path)],
                cwd=repository,
                stdout_path=state_dir / "stdout.log",
                stderr_path=state_dir / "stderr.log",
                exit_record=state_dir / "exit.json",
                launch_record=state_dir / "launch.json",
                gpu_lease_path=lease,
                lease_handshake=state_dir / "lease-handshake.json",
                run_id=f"reassigned-{args.source_job_id}",
                git_sha=args.git_sha,
                environment=_environment(args),
            )
        except Exception:
            lease.unlink(missing_ok=True)
            raise
    return {"status": "launched", "state_dir": str(state_dir), "launch": record}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-spec", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--runtime-source", type=Path)
    parser.add_argument("--git-sha")
    parser.add_argument("--config")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--source-job-id")
    parser.add_argument("--source-host", choices=("hamster", "ferret"))
    parser.add_argument("--destination-host", choices=("hamster", "ferret"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--gpu-lock-root", type=Path)
    parser.add_argument("--phases", nargs="+", choices=("train", "pgd", "autoattack"))
    parser.add_argument("--required-free-memory-mib", type=int, default=7500)
    parser.add_argument("--dataset-root", default="/home/shunsukenaito/workspace-local/datasets/ard/torchvision")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-entity", default="shunsuke-n-waseda-university")
    parser.add_argument("--wandb-project", default="single-teacher-ard")
    parser.add_argument("--wandb-group", required=False)
    parser.add_argument("--wandb-run-id", required=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.execute_spec is not None:
        return _execute_spec(args.execute_spec.resolve())
    required = (
        "repository",
        "git_sha",
        "config",
        "output",
        "state_dir",
        "source_job_id",
        "source_host",
        "destination_host",
        "gpu",
        "gpu_uuid",
        "gpu_lock_root",
        "phases",
        "wandb_group",
        "wandb_run_id",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing launch arguments: " + ", ".join(missing))
    try:
        result = launch(args)
    except (OSError, subprocess.SubprocessError, ReassignmentError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
