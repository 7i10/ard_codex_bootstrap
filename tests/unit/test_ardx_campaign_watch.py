"""Completion contract and transition stream of scripts/ardx/campaign_watch.py."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARDX = REPO_ROOT / "scripts" / "ardx"
WATCH = ARDX / "campaign_watch.py"
RUNTIME_ROOT = Path("/home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap")

REAL_STATES = {
    "attempt11": RUNTIME_ROOT / "runs/ert-i100-online-state-s2-v1-attempt11/orchestration/state.json",
    "recovery17": RUNTIME_ROOT / "runs/ert-i100-online-state-s2-v1-recovery17/orchestration/state.json",
    "operational": RUNTIME_ROOT / "orchestration/operational-foundation-hamster-state-v1.json",
}


@pytest.fixture(scope="module")
def ardx_common():
    sys.path.insert(0, str(ARDX))
    try:
        import ardx_common  # noqa: PLC0415

        return ardx_common
    finally:
        sys.path.remove(str(ARDX))


def job(status: str, *, failure_class: str | None = None, retryable: bool | None = None) -> dict:
    attempt: dict = {"attempt": 1, "status": status, "finished_at": "2026-09-04T05:31:23+00:00"}
    if failure_class is not None:
        attempt["failure_class"] = failure_class
    if retryable is not None:
        attempt["retryable"] = retryable
    return {"status": status, "attempts": [] if status == "blocked" else [attempt]}


def campaign(campaign_id: str, jobs: dict[str, dict], status: str = "running") -> dict:
    return {"schema_version": 1, "campaign_id": campaign_id, "status": status, "jobs": jobs}


def write_state(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run_watch(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(WATCH), *args], capture_output=True, text=True, check=False, timeout=120
    )


def events(result: subprocess.CompletedProcess) -> list[dict]:
    assert result.returncode == 0, result.stderr
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Completion contract
# --------------------------------------------------------------------------


def test_real_state_files_classified(tmp_path, ardx_common):
    missing = [name for name, path in REAL_STATES.items() if not path.exists()]
    if missing:
        pytest.skip(f"runtime fixtures unavailable: {missing}")
    copies = {}
    for name, path in REAL_STATES.items():
        target = tmp_path / name / "orchestration" / "state.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copies[name] = json.loads(target.read_text(encoding="utf-8"))

    attempt11 = ardx_common.classify_campaign(copies["attempt11"])
    assert len(attempt11["job_statuses"]) == 23
    assert attempt11["terminal"] is True
    assert attempt11["success"] is False
    assert attempt11["status"] == "failed"
    # No attempt in the real file carries failure_class evidence.
    assert attempt11["failure_class"] == "unknown"

    for name in ("recovery17", "operational"):
        summary = ardx_common.classify_campaign(copies[name])
        assert summary["terminal"] is True, name
        assert summary["success"] is True, name
        assert summary["failure_class"] is None, name


@pytest.mark.parametrize(
    ("jobs", "expected"),
    [
        ({"a": job("failed", failure_class="technical", retryable=True)}, "technical_retryable"),
        (
            {
                "a": job("failed", failure_class="technical", retryable=True),
                "b": job("failed", failure_class="scientific", retryable=False),
            },
            "scientific",
        ),
        (
            {
                "a": job("failed", failure_class="technical", retryable=True),
                "b": job("failed", failure_class="technical", retryable=False),
            },
            "unknown",
        ),
        ({"a": job("failed"), "b": job("blocked")}, "unknown"),
    ],
)
def test_failure_class_precedence(ardx_common, jobs, expected):
    summary = ardx_common.classify_campaign(campaign("c", jobs))
    assert summary["terminal"] is True
    assert summary["success"] is False
    assert summary["failure_class"] == expected


def test_declared_status_is_not_trusted(ardx_common):
    state = campaign("c", {"a": job("running"), "b": job("completed")}, status="completed")
    summary = ardx_common.classify_campaign(state)
    assert summary["terminal"] is False
    assert summary["success"] is None
    assert summary["status"] == "running"
    assert summary["declared_status"] == "completed"


# --------------------------------------------------------------------------
# Transition stream
# --------------------------------------------------------------------------


def test_first_scan_emits_nothing(tmp_path):
    root = tmp_path / "runs"
    write_state(root / "camp" / "orchestration" / "state.json", campaign("camp", {"a": job("completed")}))
    cursor = tmp_path / "watch-state.json"
    result = run_watch(["--once", "--roots", str(root), "--state", str(cursor)])
    assert events(result) == []
    assert json.loads(cursor.read_text(encoding="utf-8"))["sources"]


def test_emit_existing_snapshots_history(tmp_path):
    root = tmp_path / "runs"
    write_state(root / "camp" / "orchestration" / "state.json", campaign("camp", {"a": job("completed")}))
    cursor = tmp_path / "watch-state.json"
    result = run_watch(["--once", "--emit-existing", "--roots", str(root), "--state", str(cursor)])
    emitted = events(result)
    kinds = [evt["kind"] for evt in emitted]
    assert kinds == ["job", "campaign"]
    assert emitted[-1]["terminal"] is True
    assert emitted[-1]["success"] is True
    assert emitted[-1]["id"] == "camp"


def test_job_flip_emits_exactly_one_campaign_event(tmp_path):
    root = tmp_path / "runs"
    state = root / "camp" / "orchestration" / "state.json"
    write_state(state, campaign("camp", {"a": job("completed"), "b": job("running")}))
    cursor = tmp_path / "watch-state.json"
    assert events(run_watch(["--once", "--roots", str(root), "--state", str(cursor)])) == []

    write_state(state, campaign("camp", {"a": job("completed"), "b": job("completed")}, status="completed"))
    emitted = events(run_watch(["--once", "--roots", str(root), "--state", str(cursor)]))
    campaigns = [evt for evt in emitted if evt["kind"] == "campaign"]
    jobs = [evt for evt in emitted if evt["kind"] == "job"]
    assert len(campaigns) == 1
    assert [evt["job_id"] for evt in jobs] == ["b"]
    assert campaigns[0]["status"] == "completed"
    assert campaigns[0]["terminal"] is True
    assert campaigns[0]["success"] is True
    assert campaigns[0]["failure_class"] is None
    assert campaigns[0]["path"] == str(state.resolve())
    assert campaigns[0]["detail"]["counts"] == {"completed": 2}


def test_on_event_receives_argv_and_event_json(tmp_path):
    root = tmp_path / "runs"
    state = root / "camp" / "orchestration" / "state.json"
    write_state(state, campaign("camp", {"a": job("running")}))
    cursor = tmp_path / "watch-state.json"
    record = tmp_path / "hook-record.json"
    hook = tmp_path / "recorder.py"
    hook.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"open({str(record)!r}, 'w').write(json.dumps("
        "{'argv': sys.argv[1:], 'event': os.environ.get('ARDX_EVENT_JSON')}))\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    assert events(run_watch(["--once", "--roots", str(root), "--state", str(cursor)])) == []

    write_state(state, campaign("camp", {"a": job("failed", failure_class="scientific")}, status="failed"))
    emitted = events(run_watch(["--once", "--roots", str(root), "--state", str(cursor), "--on-event", str(hook)]))
    assert [evt["kind"] for evt in emitted if evt["kind"] == "campaign"] == ["campaign"]
    for _ in range(100):
        if record.exists():
            break
        os.sched_yield()
        import time  # noqa: PLC0415

        time.sleep(0.05)
    assert record.exists(), "on-event hook never ran"
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["argv"] == ["campaign", "camp", "failed"]
    event = json.loads(payload["event"])
    assert event["kind"] == "campaign"
    assert event["terminal"] is True
    assert event["success"] is False
    assert event["failure_class"] == "scientific"


def test_on_event_not_fired_for_non_terminal_or_job(tmp_path):
    root = tmp_path / "runs"
    state = root / "camp" / "orchestration" / "state.json"
    write_state(state, campaign("camp", {"a": job("running"), "b": job("completed")}))
    cursor = tmp_path / "watch-state.json"
    record = tmp_path / "hook-record.json"
    hook = tmp_path / "recorder.py"
    hook.write_text(
        "#!/usr/bin/env python3\nimport sys\n" f"open({str(record)!r}, 'a').write(' '.join(sys.argv[1:]) + chr(10))\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    emitted = events(
        run_watch(
            ["--once", "--emit-existing", "--roots", str(root), "--state", str(cursor), "--on-event", str(hook)]
        )
    )
    assert any(evt["kind"] == "job" for evt in emitted)
    import time

    time.sleep(0.5)
    assert not record.exists()


def test_malformed_json_is_skipped(tmp_path):
    root = tmp_path / "runs"
    bad = root / "broken" / "orchestration" / "state.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not json", encoding="utf-8")
    write_state(root / "camp" / "orchestration" / "state.json", campaign("camp", {"a": job("completed")}))
    cursor = tmp_path / "watch-state.json"
    result = run_watch(["--once", "--emit-existing", "--roots", str(root), "--state", str(cursor)])
    emitted = events(result)
    assert [evt["id"] for evt in emitted if evt["kind"] == "campaign"] == ["camp"]
    assert "skipping unreadable state file" in result.stderr


# --------------------------------------------------------------------------
# Hand-run bundles
# --------------------------------------------------------------------------


def make_bundle(base: Path, run_id: str, *, status: str, completion: bool, marker: str | None, progress: str | None) -> Path:
    bundle = base / run_id / "run-bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "run_id": run_id, "status": status}
    if progress is not None:
        manifest["latest_progress"] = {"epoch": 1, "global_step": 10, "timestamp": progress}
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if completion:
        (bundle / "completion.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    if marker is not None:
        (bundle / "error-marker.txt").write_text(marker + "\n", encoding="utf-8")
    return bundle / "manifest.json"


@pytest.mark.parametrize("status", ["completed", "sync_pending"])
def test_hand_run_triple_is_the_only_success(tmp_path, ardx_common, status):
    path = make_bundle(
        tmp_path, f"run-{status}", status=status, completion=True, marker="no application error recorded", progress=None
    )
    summary = ardx_common.classify_run(path, json.loads(path.read_text()), stale_seconds=3600)
    assert (summary["status"], summary["terminal"], summary["success"]) == ("completed", True, True)


@pytest.mark.parametrize(
    ("completion", "marker"),
    [(False, "no application error recorded"), (True, "application failure recorded"), (True, None)],
)
def test_hand_run_without_full_triple_is_not_terminal(tmp_path, ardx_common, completion, marker):
    path = make_bundle(
        tmp_path, f"run-{completion}-{marker}", status="completed", completion=completion, marker=marker, progress=None
    )
    summary = ardx_common.classify_run(path, json.loads(path.read_text()), stale_seconds=3600)
    assert summary["status"] == "incomplete"
    assert summary["terminal"] is False


def test_hand_run_failed_is_terminal(tmp_path, ardx_common):
    path = make_bundle(tmp_path, "run-failed", status="failed", completion=False, marker="application failure recorded", progress=None)
    summary = ardx_common.classify_run(path, json.loads(path.read_text()), stale_seconds=3600)
    assert (summary["status"], summary["terminal"], summary["success"]) == ("failed", True, False)


def test_hand_run_stale_is_not_terminal_and_not_failed(tmp_path, ardx_common):
    import datetime as dt

    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=5)).isoformat()
    fresh = dt.datetime.now(dt.timezone.utc).isoformat()
    stale_path = make_bundle(tmp_path, "run-stale", status="running", completion=False, marker=None, progress=old)
    live_path = make_bundle(tmp_path, "run-live", status="running", completion=False, marker=None, progress=fresh)
    stale = ardx_common.classify_run(stale_path, json.loads(stale_path.read_text()), stale_seconds=3600)
    live = ardx_common.classify_run(live_path, json.loads(live_path.read_text()), stale_seconds=3600)
    assert (stale["status"], stale["terminal"], stale["success"]) == ("stale", False, None)
    assert (live["status"], live["terminal"], live["success"]) == ("running", False, None)


def test_include_hand_run_emits_run_events(tmp_path):
    root = tmp_path / "runs"
    make_bundle(root, "hand-1", status="completed", completion=True, marker="no application error recorded", progress=None)
    cursor = tmp_path / "watch-state.json"
    emitted = events(
        run_watch(["--once", "--emit-existing", "--include-hand-run", "--roots", str(root), "--state", str(cursor)])
    )
    runs = [evt for evt in emitted if evt["kind"] == "run"]
    assert len(runs) == 1
    assert runs[0]["id"] == "hand-1"
    assert runs[0]["terminal"] is True
    assert runs[0]["success"] is True


def test_campaign_owned_bundles_are_not_hand_runs(tmp_path):
    root = tmp_path / "runs"
    write_state(root / "camp" / "orchestration" / "state.json", campaign("camp", {"a": job("completed")}))
    make_bundle(root / "camp" / "arms", "job-a", status="completed", completion=True, marker="no application error recorded", progress=None)
    cursor = tmp_path / "watch-state.json"
    emitted = events(
        run_watch(["--once", "--emit-existing", "--include-hand-run", "--roots", str(root), "--state", str(cursor)])
    )
    assert [evt["kind"] for evt in emitted if evt["kind"] == "run"] == []


def test_cursor_is_written_atomically_and_reused(tmp_path):
    root = tmp_path / "runs"
    write_state(root / "camp" / "orchestration" / "state.json", campaign("camp", {"a": job("completed")}))
    cursor = tmp_path / "nested" / "watch-state.json"
    run_watch(["--once", "--roots", str(root), "--state", str(cursor)])
    payload = json.loads(cursor.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert list(payload["sources"].values())[0]["status"] == "completed"
    assert not list(cursor.parent.glob("*.tmp-*"))


def test_launch_gate_canary_bundles_are_not_hand_runs(tmp_path):
    root = tmp_path / "runs"
    make_bundle(root / "attempt1" / "gate-canary" / "arms", "canary-1", status="failed", completion=False, marker=None, progress=None)
    make_bundle(root / "attempt1" / "outputs", "science-1", status="completed", completion=True, marker="no application error recorded", progress=None)
    cursor = tmp_path / "watch-state.json"
    emitted = events(
        run_watch(["--once", "--emit-existing", "--include-hand-run", "--roots", str(root), "--state", str(cursor)])
    )
    assert [evt["id"] for evt in emitted if evt["kind"] == "run"] == ["science-1"]
