from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ard.cli.status import collect_statuses, render_markdown

pytestmark = pytest.mark.t1


def _manifest(root: Path, name: str, payload: dict[str, object]) -> Path:
    path = root / name / "run-bundle" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_status_classifies_manifest_lifecycle_without_network_or_processes(tmp_path: Path) -> None:
    now = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    recent = "2026-07-31T23:59:30+00:00"
    old = "2026-07-31T20:00:00+00:00"
    _manifest(
        tmp_path,
        "running",
        {"run_id": "run", "status": "running", "latest_progress": {"epoch": 4, "global_step": 25, "timestamp": recent}},
    )
    _manifest(tmp_path, "stale", {"run_id": "stale", "status": "running", "latest_progress": {"timestamp": old}})
    finished = "2026-07-31T23:58:00+00:00"
    _manifest(
        tmp_path,
        "done",
        {"run_id": "done", "status": "completed", "finished_at": finished, "latest_progress": {"timestamp": old}},
    )
    _manifest(
        tmp_path,
        "failed",
        {"run_id": "failed", "status": "failed", "finished_at": finished, "latest_progress": {"timestamp": old}},
    )
    _manifest(tmp_path, "pending", {"run_id": "pending", "status": "sync_pending"})

    rows = collect_statuses(roots=[tmp_path], stale_seconds=60.0, now=now)

    assert [(row["run_id"], row["state"]) for row in rows] == [
        ("done", "completed"),
        ("failed", "failed"),
        ("pending", "sync-pending"),
        ("run", "running"),
        ("stale", "stale"),
    ]
    running = next(row for row in rows if row["run_id"] == "run")
    assert running["epoch"] == 4 and running["global_step"] == 25
    assert next(row for row in rows if row["run_id"] == "done")["updated_at"] == finished
    assert next(row for row in rows if row["run_id"] == "failed")["updated_at"] == finished
    rendered = render_markdown(rows)
    assert "# Local experiment status" in rendered and "sync-pending" in rendered


def test_status_reports_invalid_manifest_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "broken" / "run-bundle" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("not-json", encoding="utf-8")
    rows = collect_statuses(roots=[tmp_path], stale_seconds=0.0, now=datetime.now(UTC))
    assert rows == [{"manifest": str(path.resolve()), "state": "invalid"}]


def test_status_resume_event_refreshes_running_state_without_rewriting_measurement(tmp_path: Path) -> None:
    now = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    measured = "2026-07-30T00:00:00+00:00"
    resumed = "2026-07-31T23:59:30+00:00"
    _manifest(
        tmp_path,
        "resumed",
        {
            "run_id": "resumed",
            "status": "running",
            "latest_progress": {"epoch": 7, "global_step": 42, "timestamp": measured},
            "resume_events": [{"at": resumed, "git": "abc"}],
        },
    )

    [row] = collect_statuses(roots=[tmp_path], stale_seconds=60.0, now=now)

    assert row["state"] == "running"
    assert row["epoch"] == 7 and row["global_step"] == 42
    assert row["updated_at"] == resumed
