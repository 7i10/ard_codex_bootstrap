#!/usr/bin/env python3
"""Maintain a compact, runtime-only navigation record for one ARD task.

Task context shortens recovery after a Codex/session interruption.  It is not
scientific evidence: reports, registered artifacts, and exact source/checkpoint
identities remain authoritative.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ard.workspace import load_workspace_contract  # noqa: E402

SCHEMA_VERSION = 1
LIST_FIELDS = (
    "authoritative_files",
    "decisions",
    "completed_milestones",
    "pending_milestones",
    "blockers",
    "active_jobs",
    "stop_rules",
)


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def context_path(args: argparse.Namespace) -> tuple[Path, Any]:
    contract = load_workspace_contract(args.registry)
    default = contract.path("task_context_root") / f"{args.task_id}.json"
    path = args.output.resolve() if getattr(args, "output", None) else default
    contract.require_runtime_write(path)
    return path, contract


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported task context schema")
    return value


def init(args: argparse.Namespace) -> int:
    path, contract = context_path(args)
    if path.exists():
        raise ValueError(f"refusing to overwrite task context: {path}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": args.task_id,
        "goal": args.goal,
        "source_sha": args.source_sha,
        "workspace_contract": str(contract.registry_path),
        "workspace_contract_sha256": contract.registry_sha256,
        "authoritative_files": args.authoritative_file or [],
        "decisions": [],
        "completed_milestones": [],
        "pending_milestones": args.pending_milestone or [],
        "blockers": [],
        "active_jobs": [],
        "stop_rules": args.stop_rule or [],
        "created_at": now(),
        "updated_at": now(),
    }
    atomic_json(path, payload)
    print(json.dumps({"status": "created", "context": str(path)}, sort_keys=True))
    return 0


def append(args: argparse.Namespace) -> int:
    path, _ = context_path(args)
    payload = load(path)
    field = args.field
    if field not in LIST_FIELDS:
        raise ValueError(f"unknown task-context list field: {field}")
    value: Any = json.loads(args.value) if args.json_value else args.value
    if value not in payload[field]:
        payload[field].append(value)
    payload["updated_at"] = now()
    atomic_json(path, payload)
    print(json.dumps({"status": "updated", "context": str(path), "field": field}, sort_keys=True))
    return 0


def replace(args: argparse.Namespace) -> int:
    path, _ = context_path(args)
    payload = load(path)
    field = args.field
    if field not in LIST_FIELDS:
        raise ValueError(f"unknown task-context list field: {field}")
    value = json.loads(args.value)
    if not isinstance(value, list):
        raise ValueError("replacement task-context value must be a JSON list")
    payload[field] = value
    payload["updated_at"] = now()
    atomic_json(path, payload)
    print(json.dumps({"status": "replaced", "context": str(path), "field": field}, sort_keys=True))
    return 0


def show(args: argparse.Namespace) -> int:
    path, _ = context_path(args)
    print(json.dumps(load(path), indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--registry", type=Path)
    commands = root.add_subparsers(dest="command", required=True)
    create = commands.add_parser("init")
    create.add_argument("--task-id", required=True)
    create.add_argument("--goal", required=True)
    create.add_argument("--source-sha", required=True)
    create.add_argument("--output", type=Path)
    create.add_argument("--authoritative-file", action="append")
    create.add_argument("--pending-milestone", action="append")
    create.add_argument("--stop-rule", action="append")
    create.set_defaults(handler=init)
    update = commands.add_parser("append")
    update.add_argument("--task-id", required=True)
    update.add_argument("--output", type=Path)
    update.add_argument("--field", required=True)
    update.add_argument("--value", required=True)
    update.add_argument("--json-value", action="store_true")
    update.set_defaults(handler=append)
    replace_list = commands.add_parser("replace")
    replace_list.add_argument("--task-id", required=True)
    replace_list.add_argument("--output", type=Path)
    replace_list.add_argument("--field", required=True)
    replace_list.add_argument("--value", required=True)
    replace_list.set_defaults(handler=replace)
    inspect = commands.add_parser("show")
    inspect.add_argument("--task-id", required=True)
    inspect.add_argument("--output", type=Path)
    inspect.set_defaults(handler=show)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
