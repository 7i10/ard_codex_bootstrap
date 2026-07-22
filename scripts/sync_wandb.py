#!/usr/bin/env python3
"""Synchronize only durable local runs still marked as offline pending."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def pending_manifests(root: Path) -> tuple[Path, ...]:
    found = []
    for manifest in root.glob("**/run-bundle/manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("tracking_mode") == "offline_sync" and data.get("sync_state") == "sync_pending":
            found.append(manifest)
    return tuple(sorted(found))


def _valid_segment(segment: object, run_id: str) -> Path | None:
    if not isinstance(segment, dict) or segment.get("run_id") != run_id:
        return None
    value = segment.get("path")
    if not isinstance(value, str):
        return None
    directory = Path(value)
    if not directory.is_dir():
        return None
    markers = tuple(directory.glob("*.wandb"))
    return directory if markers and any(run_id in marker.name for marker in markers) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifests = pending_manifests(args.root)
    for manifest in manifests:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        segments = data.get("wandb_segments", [])
        if not segments:
            return 1
        cursor = int(data.get("sync_cursor", 0))
        for index, segment in enumerate(segments[cursor:], start=cursor):
            run_dir = _valid_segment(segment, data["run_id"])
            if run_dir is None:
                return 1
            command = ["wandb", "sync", "--id", data["run_id"]]
            if index:
                command.append("--append")
            command.append(str(run_dir))
            print("would sync: " + " ".join(command) if args.dry_run else "syncing: " + " ".join(command))
            if not args.dry_run and subprocess.run(command).returncode != 0:
                return 1
            if not args.dry_run:
                data["sync_cursor"] = index + 1
                temporary = manifest.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                temporary.replace(manifest)
        if args.dry_run:
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["sync_state"] = "synced"
        if data.get("status") != "failed":
            data["status"] = "completed"
        marker = manifest.parent / "sync-complete.json"
        marker.write_text(json.dumps({"run_id": data["run_id"], "synced": True}) + "\n", encoding="utf-8")
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
