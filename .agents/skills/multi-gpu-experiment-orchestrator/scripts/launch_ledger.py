#!/usr/bin/env python3
"""Maintain a compact, evidence-linked experiment launch timing ledger.

This script is intentionally operational: it never starts, stops, or polls a
job.  Its sole purpose is to make request-to-launch delay and its blockers
visible before a long campaign becomes a serial debugging session.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EVENTS = frozenset(
    {
        "input_inventory_complete",
        "host_config_matrix_complete",
        "source_frozen",
        "manifest_frozen",
        "controller_launched",
        "host_confirmed",
        "launch_blocker",
        "campaign_complete",
    }
)
READY_EVENTS = frozenset(
    {
        "input_inventory_complete",
        "host_config_matrix_complete",
        "source_frozen",
        "manifest_frozen",
    }
)


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_ledger(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or not isinstance(payload.get("events"), list):
        raise ValueError(f"{path}: incompatible launch ledger")
    return payload


def event_times(payload: dict[str, Any], event: str) -> list[datetime]:
    return [parse_time(str(row["at"])) for row in payload["events"] if row.get("event") == event]


def summary(payload: dict[str, Any], *, as_of: datetime | None = None) -> dict[str, Any]:
    requested = parse_time(str(payload["requested_at"]))
    launched = event_times(payload, "controller_launched")
    host_confirmed = event_times(payload, "host_confirmed")
    present = {str(row.get("event")) for row in payload["events"]}
    target_seconds = int(payload["target_controller_launch_minutes"]) * 60
    result: dict[str, Any] = {
        "campaign_id": payload["campaign_id"],
        "requested_at": requested.isoformat(),
        "requested_at_precision": payload["requested_at_precision"],
        "target_controller_launch_minutes": payload["target_controller_launch_minutes"],
        "required_prelaunch_events_missing": sorted(READY_EVENTS - present),
        "launch_blocker_count": sum(row.get("event") == "launch_blocker" for row in payload["events"]),
    }
    if launched:
        seconds = (launched[0] - requested).total_seconds()
        result.update(
            controller_launched_at=launched[0].isoformat(),
            request_to_controller_seconds=seconds,
            request_to_controller_minutes=seconds / 60.0,
            controller_launch_within_target=seconds <= target_seconds,
        )
    else:
        now = as_of or datetime.now(timezone.utc)
        elapsed = (now - requested).total_seconds()
        result.update(
            controller_launched_at=None,
            elapsed_without_controller_seconds=elapsed,
            launch_target_overdue=elapsed > target_seconds,
        )
    if host_confirmed:
        result["first_host_confirmed_at"] = host_confirmed[0].isoformat()
        if launched:
            result["controller_to_host_confirm_seconds"] = (host_confirmed[0] - launched[0]).total_seconds()
    return result


def cmd_init(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing launch ledger: {args.output}")
    requested = parse_time(args.requested_at)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": args.campaign_id,
        "requested_at": requested.isoformat(),
        "requested_at_precision": args.requested_at_precision,
        "target_controller_launch_minutes": args.target_controller_launch_minutes,
        "events": [
            {
                "event": "request_received",
                "at": requested.isoformat(),
                "evidence": args.request_evidence,
            }
        ],
    }
    atomic_json(args.output, payload)
    print(json.dumps(summary(payload), sort_keys=True))
    return 0


def cmd_mark(args: argparse.Namespace) -> int:
    if args.event not in EVENTS:
        raise ValueError(f"unknown launch-ledger event: {args.event}")
    if not args.evidence:
        raise ValueError("an evidence path, hash, or log reference is required")
    payload = read_ledger(args.output)
    at = parse_time(args.at) if args.at else datetime.now(timezone.utc)
    prior = [parse_time(str(row["at"])) for row in payload["events"]]
    if prior and at < prior[-1]:
        raise ValueError("ledger events must be appended in nondecreasing time order")
    if args.event != "launch_blocker" and any(row.get("event") == args.event for row in payload["events"]):
        raise ValueError(f"event already recorded: {args.event}")
    payload["events"].append({"event": args.event, "at": at.isoformat(), "evidence": args.evidence})
    atomic_json(args.output, payload)
    print(json.dumps(summary(payload), sort_keys=True))
    return 0


def cmd_ready(args: argparse.Namespace) -> int:
    payload = read_ledger(args.output)
    result = summary(payload)
    ready = not result["required_prelaunch_events_missing"]
    result["ready_to_launch"] = ready
    print(json.dumps(result, sort_keys=True))
    return 0 if ready else 2


def cmd_summary(args: argparse.Namespace) -> int:
    print(json.dumps(summary(read_ledger(args.output)), indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--campaign-id", required=True)
    init.add_argument("--requested-at", required=True)
    init.add_argument("--requested-at-precision", choices=("exact", "user-reported", "approximate"), required=True)
    init.add_argument("--request-evidence", required=True)
    init.add_argument("--target-controller-launch-minutes", type=int, default=30)
    init.set_defaults(handler=cmd_init)
    mark = commands.add_parser("mark")
    mark.add_argument("--output", type=Path, required=True)
    mark.add_argument("--event", required=True)
    mark.add_argument("--evidence", required=True)
    mark.add_argument("--at")
    mark.set_defaults(handler=cmd_mark)
    ready = commands.add_parser("ready")
    ready.add_argument("--output", type=Path, required=True)
    ready.set_defaults(handler=cmd_ready)
    show = commands.add_parser("summary")
    show.add_argument("--output", type=Path, required=True)
    show.set_defaults(handler=cmd_summary)
    return root


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "target_controller_launch_minutes", 30) <= 0:
        raise ValueError("target controller launch minutes must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
