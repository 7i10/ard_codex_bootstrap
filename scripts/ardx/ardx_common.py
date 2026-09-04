"""Shared stdlib helpers for the ``scripts/ardx`` execution-plane tools.

Standard library only: these run under ``/usr/bin/python3`` (no PyYAML) as well
as the project Python.  The completion contract implemented here is the single
signal consumed by the watcher, the status view and the postrun hook.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# Mirrors orchestrate.py TERMINAL/ACTIVE.
TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "blocked", "orphaned"})
ACTIVE_JOB_STATUSES = frozenset({"running", "retrying"})
FAILED_JOB_STATUSES = frozenset({"failed", "blocked", "orphaned"})

HAND_RUN_OK_MARKER = "no application error recorded"
# Launch-gate canary bundles are gate artifacts, not science: never a hand-run.
GATE_ARTIFACT_DIRS = frozenset({"gate-canary", "launch-gate", "gate-canary-prior-canary"})
HAND_RUN_SUCCESS_STATUSES = frozenset({"completed", "sync_pending"})

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "workspace" / "ard_workspace_v1.json"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def add_registry_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"workspace registry JSON (default: {DEFAULT_REGISTRY})",
    )


def load_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"registry is not a JSON object: {path}")
    return payload


def registry_path(registry: dict[str, Any], key: str, default: str | None = None) -> Path:
    value = registry.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"registry key {key!r} is missing or not a path")
    return Path(value)


def runtime_root(registry: dict[str, Any]) -> Path:
    return registry_path(registry, "runtime_root")


def ardx_root(registry: dict[str, Any]) -> Path:
    """``<runtime>/orchestration/ardx`` -- watcher cursor, log and claude runs."""
    orchestration = registry.get("orchestration_root")
    base = Path(orchestration) if isinstance(orchestration, str) and orchestration else runtime_root(registry) / "orchestration"
    return base / "ardx"


def default_watch_roots(registry: dict[str, Any]) -> list[Path]:
    run_root = registry.get("run_root")
    orchestration_root = registry.get("orchestration_root")
    roots = [
        Path(run_root) if isinstance(run_root, str) and run_root else runtime_root(registry) / "runs",
        Path(orchestration_root)
        if isinstance(orchestration_root, str) and orchestration_root
        else runtime_root(registry) / "orchestration",
    ]
    return roots


def default_bundle_roots(registry: dict[str, Any]) -> list[Path]:
    """Where hand-run bundles actually live: the watch roots plus ``<repo>/outputs``.

    Every real hand-run bundle is written under the repo's ``outputs/`` tree; the
    bundles under ``run_root`` are campaign-owned or gate canaries and are
    excluded by ``find_run_manifests``.  Campaign scanning keeps the two frozen
    default roots.
    """
    roots = default_watch_roots(registry)
    try:
        outputs = registry_path(registry, "repo_root") / "outputs"
    except ValueError:
        return roots
    if outputs.is_dir():
        roots.append(outputs)
    return roots


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(dt.timezone.utc)


# --------------------------------------------------------------------------
# Campaign completion contract (orchestrator state.json)
# --------------------------------------------------------------------------


def is_campaign_state(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("jobs"), dict) and "campaign_id" in payload


def job_statuses(state: dict[str, Any]) -> dict[str, str]:
    jobs = state.get("jobs")
    if not isinstance(jobs, dict):
        return {}
    out: dict[str, str] = {}
    for job_id, record in jobs.items():
        status = record.get("status") if isinstance(record, dict) else None
        out[str(job_id)] = str(status) if isinstance(status, str) else "unknown"
    return out


def last_failure_evidence(record: dict[str, Any]) -> tuple[str, bool] | None:
    """Newest attempt carrying a ``failure_class`` (reconcile_experiment.py:293-327)."""
    attempts = record.get("attempts")
    if not isinstance(attempts, list):
        return None
    for attempt in reversed(attempts):
        if not isinstance(attempt, dict):
            continue
        failure_class = attempt.get("failure_class")
        if failure_class:
            return str(failure_class), attempt.get("retryable") is True
    return None


def aggregate_failure_class(state: dict[str, Any]) -> str:
    """scientific > unknown/non-retryable technical > technical_retryable."""
    jobs = state.get("jobs")
    if not isinstance(jobs, dict):
        return "unknown"
    failures: list[tuple[str, bool]] = []
    for _job_id, record in sorted(jobs.items()):
        if not isinstance(record, dict) or record.get("status") not in FAILED_JOB_STATUSES:
            continue
        evidence = last_failure_evidence(record)
        failures.append(evidence if evidence is not None else ("unknown", False))
    if not failures:
        return "unknown"
    if any(kind == "scientific" for kind, _ in failures):
        return "scientific"
    if all(kind == "technical" and retryable for kind, retryable in failures):
        return "technical_retryable"
    return "unknown"


def classify_campaign(state: dict[str, Any]) -> dict[str, Any]:
    """Derive the single completion signal from ``jobs[*].status`` only."""
    statuses = job_statuses(state)
    counts: dict[str, int] = {}
    for status in statuses.values():
        counts[status] = counts.get(status, 0) + 1
    terminal = bool(statuses) and all(status in TERMINAL_JOB_STATUSES for status in statuses.values())
    success = terminal and all(status == "completed" for status in statuses.values())
    if terminal:
        derived = "completed" if success else "failed"
    elif any(status in ACTIVE_JOB_STATUSES for status in statuses.values()):
        derived = "running"
    else:
        derived = "pending"
    failure_class = aggregate_failure_class(state) if terminal and not success else None
    return {
        "campaign_id": str(state.get("campaign_id") or ""),
        "status": derived,
        "terminal": terminal,
        "success": success if terminal else None,
        "failure_class": failure_class,
        "counts": counts,
        "job_statuses": statuses,
        "declared_status": state.get("status"),
        "finished_at": state.get("finished_at"),
        "updated_at": state.get("updated_at"),
    }


def newest_attempt_finished_at(state: dict[str, Any]) -> str | None:
    jobs = state.get("jobs")
    if not isinstance(jobs, dict):
        return None
    newest: dt.datetime | None = None
    newest_raw: str | None = None
    for record in jobs.values():
        attempts = record.get("attempts") if isinstance(record, dict) else None
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            parsed = parse_timestamp(attempt.get("finished_at"))
            if parsed is not None and (newest is None or parsed > newest):
                newest = parsed
                newest_raw = str(attempt.get("finished_at"))
    return newest_raw


def find_campaign_states(roots: list[Path]) -> list[Path]:
    """Orchestrator state files under the watch roots.

    ``<root>/*/orchestration/state.json`` is the orchestrator layout;
    ``<root>/<campaign>/production.state.json`` and ``<root>/<campaign>/orchestration.state.json``
    are the manifest-builder layouts; flat ``<root>/*.json`` files are accepted
    when they parse as campaign state (that is how
    ``operational-foundation-hamster-state-v1.json`` is stored).
    """
    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            found.add(root.resolve())
            continue
        if not root.is_dir():
            continue
        for pattern in (
            "*/orchestration/state.json",
            "*/orchestration*.state.json",
            "*/*.state.json",
            "*.state.json",
            "*.json",
        ):
            for path in root.glob(pattern):
                if path.is_file():
                    found.add(path.resolve())
    return sorted(found)


# --------------------------------------------------------------------------
# Hand-run bundle contract (run-bundle/manifest.json)
# --------------------------------------------------------------------------


def campaign_owned_dirs(roots: list[Path]) -> set[Path]:
    """Directories whose run bundles belong to an orchestrator campaign.

    Those bundles are reported through their campaign's completion contract;
    only bundles outside them are hand-runs.
    """
    owned: set[Path] = set()
    for path in find_campaign_states(roots):
        if path.name == "state.json" and path.parent.name == "orchestration":
            owned.add(path.parent.parent)
            continue
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not is_campaign_state(payload):
            continue
        campaign_id = str(payload.get("campaign_id") or "")
        candidate = path.parent / campaign_id
        if campaign_id and candidate.is_dir():
            owned.add(candidate.resolve())
    return owned


def find_run_manifests(roots: list[Path], *, exclude_campaign_bundles: bool = True) -> list[Path]:
    """``**/run-bundle/manifest.json`` under the roots, minus campaign-owned ones."""
    owned = campaign_owned_dirs(roots) if exclude_campaign_bundles else set()
    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file() and root.name == "manifest.json":
            found.add(root.resolve())
        elif root.is_dir():
            found.update(path.resolve() for path in root.rglob("run-bundle/manifest.json") if path.is_file())
    return sorted(
        path
        for path in found
        if not any(parent in owned for parent in path.parents)
        and not GATE_ARTIFACT_DIRS.intersection(path.parts)
    )


def classify_run(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    stale_seconds: float,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """The triple rule; ``stale`` is neither terminal nor failed."""
    observed_now = dt.datetime.now(dt.timezone.utc) if now is None else now.astimezone(dt.timezone.utc)
    bundle = manifest_path.parent
    declared = manifest.get("status")
    completion = (bundle / "completion.json").exists()
    marker_path = bundle / "error-marker.txt"
    marker: str | None = None
    if marker_path.exists():
        try:
            marker = marker_path.read_text(encoding="utf-8").strip()
        except OSError:
            marker = None
    triple = completion and declared in HAND_RUN_SUCCESS_STATUSES and marker == HAND_RUN_OK_MARKER
    progress = manifest.get("latest_progress")
    progress = progress if isinstance(progress, dict) else {}
    observed = parse_timestamp(progress.get("timestamp"))
    age_seconds = None if observed is None else max(0.0, (observed_now - observed).total_seconds())

    if declared == "failed":
        status, terminal, success = "failed", True, False
    elif triple:
        status, terminal, success = "completed", True, True
    elif declared in HAND_RUN_SUCCESS_STATUSES:
        # Declared terminal, but the completion triple is not satisfied: an
        # anomaly to surface, not a completion signal, so it stays non-terminal
        # and never fires the postrun hook.
        status, terminal, success = "incomplete", False, None
    elif declared == "running" and age_seconds is not None and age_seconds > stale_seconds:
        status, terminal, success = "stale", False, None
    elif declared == "running":
        status, terminal, success = "running", False, None
    else:
        status, terminal, success = "unknown", False, None

    return {
        "run_id": str(manifest.get("run_id") or bundle.parent.name),
        "status": status,
        "terminal": terminal,
        "success": success,
        "failure_class": "unknown" if terminal and success is False else None,
        "manifest_status": declared,
        "completion_json": completion,
        "error_marker": marker,
        "epoch": progress.get("epoch"),
        "global_step": progress.get("global_step"),
        "progress_timestamp": progress.get("timestamp"),
        "age_seconds": age_seconds,
        "output_dir": str(bundle.parent),
        "wandb_url": manifest.get("wandb_url"),
    }
