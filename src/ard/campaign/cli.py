"""Small command-line API for static campaign wrappers.

This module runs at most one reconciliation pass.  Host-specific shell wrappers
may invoke it repeatedly after arranging their immutable worktree environment.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from .schema import CampaignError, CampaignSpec, bind_git_sha, load_campaign
from .state import CampaignStateStore
from .worker import CampaignWorker


def _fixed_spec(path: Path, sha: str | None) -> CampaignSpec:
    spec = load_campaign(path)
    if sha is not None:
        return bind_git_sha(spec, sha)
    if spec.git_sha is None:
        raise CampaignError("a template campaign requires --sha before state initialization")
    return spec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--campaign", type=Path, required=True)
    validate.add_argument("--sha")

    for name in ("init", "arm", "run-once", "run-loop"):
        command = commands.add_parser(name)
        command.add_argument("--campaign", type=Path, required=True)
        command.add_argument("--sha")
        command.add_argument("--state-root", type=Path, required=True)
    arm = commands.choices["arm"]
    arm.add_argument("--host", choices=("hamster", "ferret"), required=True)
    arm.add_argument("--repository", type=Path, required=True)
    arm.add_argument("--output-root", type=Path, required=True)
    arm.add_argument("--gpu-lock-root", type=Path, required=True)
    run_once = commands.choices["run-once"]
    run_once.add_argument("--host", choices=("hamster", "ferret"), required=True)
    run_once.add_argument("--repository", type=Path, required=True)
    run_once.add_argument("--output-root", type=Path, required=True)
    run_once.add_argument("--gpu-lock-root", type=Path, required=True)
    run_once.add_argument("--allow-external-gpu-processes", action="store_true")
    run_loop = commands.choices["run-loop"]
    run_loop.add_argument("--host", choices=("hamster", "ferret"), required=True)
    run_loop.add_argument("--repository", type=Path, required=True)
    run_loop.add_argument("--output-root", type=Path, required=True)
    run_loop.add_argument("--gpu-lock-root", type=Path, required=True)
    run_loop.add_argument("--allow-external-gpu-processes", action="store_true")
    run_loop.add_argument("--interval-seconds", type=float, default=20.0)

    status = commands.add_parser("status")
    status.add_argument("--state-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "status":
            state = CampaignStateStore(args.state_root)
            print(json.dumps({"campaign": state.campaign(), "jobs": state.jobs()}, sort_keys=True))
            return 0
        spec = _fixed_spec(args.campaign, args.sha)
        if args.command == "validate":
            print(json.dumps({"campaign_id": spec.campaign_id, "git_sha": spec.git_sha}, sort_keys=True))
            return 0
        state = CampaignStateStore(args.state_root)
        state.initialize(spec)
        if args.command == "init":
            print(json.dumps(state.campaign(), sort_keys=True))
            return 0
        worker = CampaignWorker(
            spec,
            state,
            host=args.host,
            repository=args.repository,
            output_root=args.output_root,
            gpu_lock_root=args.gpu_lock_root,
            external_processes_enabled=True if getattr(args, "allow_external_gpu_processes", False) else None,
        )
        if args.command == "arm":
            worker.arm()
            print(json.dumps(state.campaign(), sort_keys=True))
            return 0
        if args.command == "run-loop":
            if not 1.0 <= args.interval_seconds <= 300.0:
                raise CampaignError("--interval-seconds must be within [1, 300]")
            while True:
                worker.run_once()
                if state.campaign()["state"] == "awaiting_scientific_review":
                    return 0
                time.sleep(args.interval_seconds)
        print(json.dumps(worker.run_once(), sort_keys=True))
        return 0
    except CampaignError as exc:
        _parser().error(str(exc))
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
