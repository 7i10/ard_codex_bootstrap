#!/usr/bin/env python3
"""One-screen operational status: hosts, watcher, campaigns, bundles, decisions.

Read-only and time-bounded (< 15 s even when Ferret is down).  Set
``ARDX_SKIP_REMOTE=1`` (or pass ``--no-remote``) to skip both GPU probes; the
tests and the SessionStart hook use it to stay offline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ardx_common import (  # noqa: E402
    add_registry_argument,
    ardx_root,
    classify_campaign,
    classify_run,
    default_watch_roots,
    find_campaign_states,
    find_run_manifests,
    is_campaign_state,
    load_registry,
    newest_attempt_finished_at,
    parse_timestamp,
    read_json,
    registry_path,
)

GPU_QUERY = ["--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader"]
NON_TERMINAL_FIRST = {"running": 0, "pending": 1, "failed": 2, "completed": 3}


def skip_remote(args: argparse.Namespace) -> bool:
    return bool(args.no_remote or os.environ.get("ARDX_SKIP_REMOTE") == "1")


def run_command(argv: list[str], timeout: float) -> list[str]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def host_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if skip_remote(args):
        return [
            {"host": "hamster", "gpus": [], "note": "skipped (ARDX_SKIP_REMOTE)"},
            {"host": "ferret", "gpus": [], "note": "skipped (ARDX_SKIP_REMOTE)"},
        ]
    local = run_command(["nvidia-smi", *GPU_QUERY], timeout=6.0)
    rows.append({"host": "hamster", "gpus": local, "note": None if local else "unreachable"})
    remote = run_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "Ferret",
            "nvidia-smi " + " ".join(GPU_QUERY),
        ],
        timeout=10.0,
    )
    rows.append({"host": "ferret", "gpus": remote, "note": None if remote else "unreachable"})
    return rows


def watcher_row(registry: dict[str, Any], state_path: Path) -> dict[str, Any]:
    active = run_command(["systemctl", "--user", "is-active", "ardx-watch.service"], timeout=5.0)
    cursor_mtime = None
    sources = 0
    if state_path.exists():
        try:
            cursor_mtime = dt.datetime.fromtimestamp(state_path.stat().st_mtime, dt.timezone.utc).isoformat()
        except OSError:
            cursor_mtime = None
        try:
            payload = read_json(state_path)
            sources = len(payload.get("sources", {})) if isinstance(payload, dict) else 0
        except (OSError, json.JSONDecodeError):
            sources = 0
    return {
        "service": active[0] if active else "inactive",
        "state_path": str(state_path),
        "last_scan": cursor_mtime,
        "tracked_sources": sources,
    }


def campaign_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in find_campaign_states(roots):
        try:
            state = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not is_campaign_state(state):
            continue
        summary = classify_campaign(state)
        rows.append(
            {
                "id": summary["campaign_id"] or path.parent.parent.name,
                "status": summary["status"],
                "terminal": summary["terminal"],
                "success": summary["success"],
                "failure_class": summary["failure_class"],
                "counts": summary["counts"],
                "declared_status": summary["declared_status"],
                "newest_attempt_finished_at": newest_attempt_finished_at(state),
                "path": str(path),
            }
        )
    rows.sort(
        key=lambda row: (
            NON_TERMINAL_FIRST.get(row["status"], 4),
            row["newest_attempt_finished_at"] or "",
        )
    )
    return rows


def bundle_rows(roots: list[Path], stale_seconds: float, limit: int = 20) -> list[dict[str, Any]]:
    paths = find_run_manifests(roots)
    paths.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
    rows: list[dict[str, Any]] = []
    for path in paths[:limit]:
        try:
            manifest = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        summary = classify_run(path, manifest, stale_seconds=stale_seconds)
        rows.append(
            {
                "run_id": summary["run_id"],
                "status": summary["status"],
                "terminal": summary["terminal"],
                "success": summary["success"],
                "manifest_status": summary["manifest_status"],
                "progress_timestamp": summary["progress_timestamp"],
                "path": str(path),
            }
        )
    return rows


def parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if not line or line.startswith((" ", "\t", "#")) or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip().strip("'\"")
    return fields


def decision_rows(repo_root: Path) -> list[dict[str, Any]]:
    directory = repo_root / "docs" / "decisions"
    rows: list[dict[str, Any]] = []
    if not directory.is_dir():
        return rows
    for path in sorted(directory.glob("*.md")):
        try:
            fields = parse_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if fields.get("status") != "pending":
            continue
        rows.append(
            {
                "id": fields.get("id", path.stem),
                "question": fields.get("question", ""),
                "created": fields.get("created", ""),
                "campaign": fields.get("campaign", ""),
                "path": str(path),
            }
        )
    return rows


def postrun_rows(registry: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    directory = ardx_root(registry) / "claude-runs"
    if not directory.is_dir():
        return []
    files = sorted(
        (path for path in directory.iterdir() if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [{"name": path.name, "path": str(path), "bytes": path.stat().st_size} for path in files[:limit]]


def collect(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.registry)
    roots = [Path(root) for root in args.roots] if args.roots else default_watch_roots(registry)
    repo_root = registry_path(registry, "repo_root")
    bundle_roots = list(roots) + [repo_root / "outputs"]
    include_bundles = args.json or not args.brief
    return {
        "hosts": host_rows(args),
        "watcher": watcher_row(registry, ardx_root(registry) / "watch-state.json"),
        "campaigns": campaign_rows(roots),
        "bundles": bundle_rows(bundle_roots, args.stale_seconds) if include_bundles else [],
        "decisions": decision_rows(repo_root),
        "postruns": postrun_rows(registry),
    }


def render_brief(data: dict[str, Any]) -> str:
    lines = ["# ARD status (brief)"]
    for host in data["hosts"]:
        gpus = host["gpus"]
        summary = host["note"] or "; ".join(gpus) or "no gpus reported"
        lines.append(f"- {host['host']}: {summary}")
    watcher = data["watcher"]
    lines.append(f"- watcher: {watcher['service']} (last scan {watcher['last_scan'] or 'never'})")
    active = [row for row in data["campaigns"] if not row["terminal"]]
    if active:
        for row in active[:4]:
            lines.append(f"- campaign {row['id']}: {row['status']} {row['counts']}")
    else:
        lines.append(f"- campaigns: none active ({len(data['campaigns'])} known)")
    stale = [row for row in data["bundles"] if row["status"] == "stale"]
    if stale:
        lines.append(f"- hand-run bundles stale: {len(stale)} (newest {stale[0]['run_id']})")
    lines.append(f"- pending decisions: {len(data['decisions'])}")
    newest = data["postruns"][0]["name"] if data["postruns"] else "none"
    lines.append(f"- newest postrun: {newest}")
    return "\n".join(lines[:15])


def render_markdown(data: dict[str, Any], *, brief: bool) -> str:
    if brief:
        return render_brief(data)
    out: list[str] = ["# ARD status", "", "## Hosts"]
    for host in data["hosts"]:
        if host["note"]:
            out.append(f"- **{host['host']}**: {host['note']}")
        else:
            out.append(f"- **{host['host']}**:")
            out.extend(f"  - {line}" for line in host["gpus"])
    watcher = data["watcher"]
    out += [
        "",
        "## Watcher",
        f"- ardx-watch.service: {watcher['service']}",
        f"- cursor: {watcher['state_path']} ({watcher['tracked_sources']} sources, last scan {watcher['last_scan'] or 'never'})",
        "",
        "## Campaigns",
    ]
    if not data["campaigns"]:
        out.append("- none found")
    for row in data["campaigns"]:
        detail = f"status={row['status']}"
        if row["failure_class"]:
            detail += f" failure_class={row['failure_class']}"
        counts = ", ".join(f"{key}={value}" for key, value in sorted(row["counts"].items()))
        out.append(
            f"- **{row['id']}** {detail} (declared={row['declared_status']}; {counts}; "
            f"newest attempt finished {row['newest_attempt_finished_at'] or 'n/a'})"
        )
        out.append(f"  - {row['path']}")
    out += ["", "## Hand-run bundles"]
    if not data["bundles"]:
        out.append("- none found")
    for row in data["bundles"]:
        out.append(
            f"- **{row['run_id']}** {row['status']} (manifest={row['manifest_status']}, progress={row['progress_timestamp']})"
        )
    out += ["", "## Pending decisions"]
    if not data["decisions"]:
        out.append("- none")
    for row in data["decisions"]:
        out.append(f"- **{row['id']}** ({row['created']}) {row['question']}")
    out += ["", "## Recent postruns"]
    if not data["postruns"]:
        out.append("- none")
    for row in data["postruns"]:
        out.append(f"- {row['name']} ({row['bytes']} bytes)")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the ARD operational status view.")
    add_registry_argument(parser)
    parser.add_argument("--roots", type=Path, nargs="+", default=None, help="campaign roots (default: run_root and orchestration_root)")
    parser.add_argument("--brief", action="store_true", help="<= 15 lines, for the SessionStart hook")
    parser.add_argument("--json", action="store_true", help="emit the structured payload instead of Markdown")
    parser.add_argument("--stale-seconds", type=float, default=3600.0, help="hand-run staleness threshold (default: 3600)")
    parser.add_argument("--no-remote", action="store_true", help="skip nvidia-smi and the Ferret ssh probe (also ARDX_SKIP_REMOTE=1)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = collect(args)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(render_markdown(data, brief=args.brief))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
