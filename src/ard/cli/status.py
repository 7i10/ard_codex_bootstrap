"""Read-only local status view derived from run-bundle manifests.

This intentionally does not inspect GPU/PID state or contact W&B.  W&B is the
cross-host live view; this command makes the durable local evidence legible.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _state(manifest: dict[str, Any], *, now: datetime, stale_seconds: float) -> tuple[str, str | None]:
    declared = manifest.get("status")
    if declared == "failed":
        return "failed", manifest.get("finished_at") if isinstance(manifest.get("finished_at"), str) else None
    if declared == "completed":
        return "completed", manifest.get("finished_at") if isinstance(manifest.get("finished_at"), str) else None
    if declared == "sync_pending":
        return "sync-pending", manifest.get("finished_at") if isinstance(manifest.get("finished_at"), str) else None
    progress = manifest.get("latest_progress")
    timestamps = [
        _parse_timestamp(progress.get("timestamp") if isinstance(progress, dict) else manifest.get("created_at")),
        _parse_timestamp(manifest.get("resumed_at")),
    ]
    events = manifest.get("resume_events")
    if isinstance(events, list):
        timestamps.extend(_parse_timestamp(event.get("at")) for event in events if isinstance(event, dict))
    observed = max((timestamp for timestamp in timestamps if timestamp is not None), default=None)
    if observed is None:
        return "unknown", None
    age_seconds = max(0.0, (now - observed).total_seconds())
    if age_seconds > stale_seconds:
        return "stale", observed.isoformat()
    return "running", observed.isoformat()


def collect_statuses(*, roots: list[Path], stale_seconds: float, now: datetime | None = None) -> list[dict[str, Any]]:
    """Return deterministic, read-only summaries for manifests beneath roots."""
    if stale_seconds < 0:
        raise ValueError("stale_seconds must be non-negative")
    observed_now = datetime.now(UTC) if now is None else now.astimezone(UTC)
    manifest_paths: set[Path] = set()
    for root in roots:
        if root.is_file() and root.name == "manifest.json":
            manifest_paths.add(root.resolve())
        elif root.is_dir():
            manifest_paths.update(path.resolve() for path in root.rglob("run-bundle/manifest.json"))
    rows: list[dict[str, Any]] = []
    for path in sorted(manifest_paths, key=lambda candidate: candidate.as_posix()):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rows.append({"manifest": str(path), "state": "invalid"})
            continue
        if not isinstance(payload, dict):
            rows.append({"manifest": str(path), "state": "invalid"})
            continue
        progress = payload.get("latest_progress")
        progress = progress if isinstance(progress, dict) else {}
        state, timestamp = _state(payload, now=observed_now, stale_seconds=stale_seconds)
        rows.append(
            {
                "run_id": payload.get("run_id"),
                "state": state,
                "epoch": progress.get("epoch"),
                "global_step": progress.get("global_step"),
                "updated_at": timestamp
                or progress.get("timestamp")
                or payload.get("finished_at")
                or payload.get("created_at"),
                "wandb_url": payload.get("wandb_url"),
                "output_dir": str(path.parent.parent),
                "manifest": str(path),
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    columns = ("run_id", "state", "epoch", "global_step", "updated_at", "wandb_url", "output_dir")
    lines = [
        "# Local experiment status",
        "",
        "Read-only snapshot from local run-bundle manifests; W&B is the cross-host live view.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "") if row.get(column) is not None else "") for column in columns]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a read-only local ARD run status view.")
    parser.add_argument(
        "--root", type=Path, action="append", default=[], help="Output root or manifest path (repeatable)"
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--stale-seconds", type=float, default=3600.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = args.root or [Path("outputs")]
    rows = collect_statuses(roots=roots, stale_seconds=args.stale_seconds)
    if args.format == "json":
        print(json.dumps(rows, sort_keys=True, indent=2))
    else:
        print(render_markdown(rows), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
