#!/usr/bin/env python3
"""Emit one JSON line per campaign/job/hand-run state transition.

The completion contract lives in ``ardx_common``; this file only turns it into
a cursor-backed transition stream.  Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ardx_common import (  # noqa: E402
    SCHEMA_VERSION,
    TERMINAL_JOB_STATUSES,
    add_registry_argument,
    ardx_root,
    atomic_write_json,
    classify_campaign,
    classify_run,
    default_bundle_roots,
    default_watch_roots,
    find_campaign_states,
    find_run_manifests,
    is_campaign_state,
    last_failure_evidence,
    load_registry,
    now_iso,
    read_json,
)

_STOP = False


def _request_stop(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def warn(message: str) -> None:
    print(f"campaign_watch: {message}", file=sys.stderr, flush=True)


def load_cursor(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        warn(f"cursor unreadable, starting from an empty snapshot: {path}")
        return {}
    sources = payload.get("sources") if isinstance(payload, dict) else None
    return sources if isinstance(sources, dict) else {}


def save_cursor(path: Path, sources: dict[str, Any]) -> None:
    atomic_write_json(path, {"schema_version": SCHEMA_VERSION, "updated_at": now_iso(), "sources": sources})


def event(
    *,
    kind: str,
    ident: str,
    status: str,
    terminal: bool,
    success: bool | None,
    failure_class: str | None,
    path: Path,
    campaign_id: str | None = None,
    job_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ts": now_iso(),
        "kind": kind,
        "id": ident,
        "campaign_id": campaign_id,
        "job_id": job_id,
        "status": status,
        "terminal": terminal,
        "success": success,
        "failure_class": failure_class,
        "path": str(path),
        "detail": detail or {},
    }


def job_detail(state: dict[str, Any], job_id: str) -> dict[str, Any]:
    record = state.get("jobs", {}).get(job_id)
    if not isinstance(record, dict):
        return {}
    attempts = record.get("attempts") if isinstance(record.get("attempts"), list) else []
    last = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}
    evidence = last_failure_evidence(record)
    return {
        "attempts": len(attempts),
        "attempt": last.get("attempt"),
        "host": last.get("host"),
        "gpu": last.get("gpu"),
        "finished_at": last.get("finished_at"),
        "failure_class": evidence[0] if evidence else None,
        "retryable": evidence[1] if evidence else None,
    }


def scan_campaign(path: Path, previous: dict[str, Any] | None, emit_existing: bool) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    try:
        state = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"skipping unreadable state file {path}: {exc}")
        return [], previous
    if not is_campaign_state(state):
        return [], None
    summary = classify_campaign(state)
    campaign_id = summary["campaign_id"] or path.parent.parent.name
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    snapshot = {
        "kind": "campaign",
        "mtime": mtime,
        "campaign_id": campaign_id,
        "jobs": summary["job_statuses"],
        "status": summary["status"],
        "terminal": summary["terminal"],
        "success": summary["success"],
        "failure_class": summary["failure_class"],
    }
    detail = {
        "counts": summary["counts"],
        "declared_status": summary["declared_status"],
        "finished_at": summary["finished_at"],
        "updated_at": summary["updated_at"],
        "jobs_total": len(summary["job_statuses"]),
        "run_dir": str(path.parent.parent),
    }
    first_seen = previous is None or previous.get("kind") != "campaign"
    if first_seen and not emit_existing:
        return [], snapshot

    events: list[dict[str, Any]] = []
    seen_jobs = {} if first_seen else (previous.get("jobs") if isinstance(previous.get("jobs"), dict) else {})
    for job_id, status in sorted(summary["job_statuses"].items()):
        if seen_jobs.get(job_id) == status:
            continue
        events.append(
            event(
                kind="job",
                ident=job_id,
                status=status,
                terminal=status in TERMINAL_JOB_STATUSES,
                success=(status == "completed") if status in TERMINAL_JOB_STATUSES else None,
                failure_class=None,
                path=path,
                campaign_id=campaign_id,
                job_id=job_id,
                detail=job_detail(state, job_id),
            )
        )
    changed = first_seen or any(
        previous.get(key) != snapshot[key] for key in ("status", "terminal", "success", "failure_class")
    )
    if changed:
        events.append(
            event(
                kind="campaign",
                ident=campaign_id,
                status=summary["status"],
                terminal=summary["terminal"],
                success=summary["success"],
                failure_class=summary["failure_class"],
                path=path,
                campaign_id=campaign_id,
                detail=detail,
            )
        )
    return events, snapshot


def scan_run(path: Path, previous: dict[str, Any] | None, emit_existing: bool, stale_seconds: float) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    try:
        manifest = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"skipping unreadable manifest {path}: {exc}")
        return [], previous
    if not isinstance(manifest, dict):
        warn(f"skipping manifest that is not a JSON object: {path}")
        return [], previous
    summary = classify_run(path, manifest, stale_seconds=stale_seconds)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    snapshot = {
        "kind": "run",
        "mtime": mtime,
        "run_id": summary["run_id"],
        "status": summary["status"],
        "terminal": summary["terminal"],
        "success": summary["success"],
    }
    first_seen = previous is None or previous.get("kind") != "run"
    if first_seen and not emit_existing:
        return [], snapshot
    changed = first_seen or any(previous.get(key) != snapshot[key] for key in ("status", "terminal", "success"))
    if not changed:
        return [], snapshot
    detail = {
        key: summary[key]
        for key in (
            "manifest_status",
            "completion_json",
            "error_marker",
            "epoch",
            "global_step",
            "progress_timestamp",
            "age_seconds",
            "output_dir",
            "wandb_url",
        )
    }
    return [
        event(
            kind="run",
            ident=summary["run_id"],
            status=summary["status"],
            terminal=summary["terminal"],
            success=summary["success"],
            failure_class=summary["failure_class"],
            path=path,
            detail=detail,
        )
    ], snapshot


def fire_hook(command: str, evt: dict[str, Any]) -> None:
    if not command or evt["kind"] not in {"campaign", "run"} or not evt["terminal"]:
        return
    env = dict(os.environ)
    env["ARDX_EVENT_JSON"] = json.dumps(evt, sort_keys=True)
    argv = [command, evt["kind"], evt["id"], evt["status"]]
    try:
        subprocess.Popen(  # noqa: S603 - operator-supplied hook command
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:  # never let a hook kill the watcher
        warn(f"on-event hook failed for {evt['kind']} {evt['id']}: {exc}")


def scan_once(
    args: argparse.Namespace,
    roots: list[Path],
    cursor: dict[str, Any],
    emit_existing: bool,
    bundle_roots: list[Path] | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in find_campaign_states(roots):
        key = str(path)
        new_events, snapshot = scan_campaign(path, cursor.get(key), emit_existing)
        events.extend(new_events)
        if snapshot is None:
            cursor.pop(key, None)
        else:
            cursor[key] = snapshot
    if args.include_hand_run:
        for path in find_run_manifests(list(bundle_roots) if bundle_roots else roots):
            key = str(path)
            new_events, snapshot = scan_run(path, cursor.get(key), emit_existing, args.stale_seconds)
            events.extend(new_events)
            if snapshot is not None:
                cursor[key] = snapshot
    return events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch orchestrator campaigns and hand-run bundles for state transitions.")
    add_registry_argument(parser)
    parser.add_argument("--roots", type=Path, nargs="+", default=None, help="directories to scan (default: run_root and orchestration_root)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="scan once and exit (default)")
    mode.add_argument("--follow", action="store_true", help="scan forever every --interval seconds")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between scans in --follow (default: 30)")
    parser.add_argument("--state", type=Path, default=None, help="cursor file (default: <runtime>/orchestration/ardx/watch-state.json)")
    parser.add_argument("--on-event", default=None, help="command run as CMD KIND ID STATUS for terminal campaign/run events")
    parser.add_argument("--emit-existing", action="store_true", help="emit the first snapshot instead of only recording it")
    parser.add_argument("--stale-seconds", type=float, default=3600.0, help="hand-run progress age before 'stale' (default: 3600)")
    parser.add_argument(
        "--include-hand-run",
        action="store_true",
        help="also scan **/run-bundle/manifest.json (default roots add <repo>/outputs, where hand-runs live)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The registry is only needed for the defaults it supplies.
    registry: dict[str, Any] = {}
    if not args.roots or not args.state:
        registry = load_registry(args.registry)
    roots = [Path(root) for root in args.roots] if args.roots else default_watch_roots(registry)
    # Hand-run bundles live under <repo>/outputs, not under the campaign roots.
    bundle_roots = list(roots) if args.roots else default_bundle_roots(registry)
    state_path = Path(args.state) if args.state else ardx_root(registry) / "watch-state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    cold_start = not state_path.exists()
    cursor = load_cursor(state_path)
    if not cursor:
        cold_start = True
    # A cold start snapshots history silently; once a cursor exists, a source
    # that appears later is a real transition and is emitted.
    emit_first_seen = args.emit_existing or not cold_start
    while True:
        events = scan_once(args, roots, cursor, emit_first_seen, bundle_roots)
        try:
            for evt in events:
                print(json.dumps(evt, sort_keys=True), flush=True)
                if args.on_event:
                    fire_hook(args.on_event, evt)
        except BrokenPipeError:
            # A reader such as `| head` went away: leave the cursor untouched so
            # nothing is silently marked as delivered, and exit quietly.
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            return 0
        save_cursor(state_path, cursor)
        emit_first_seen = True
        if not args.follow or _STOP:
            return 0
        deadline = time.monotonic() + max(1.0, args.interval)
        while not _STOP and time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        if _STOP:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
