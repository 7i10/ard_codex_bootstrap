"""scripts/ardx/status.py, postrun_hook.sh and install_units.sh (no GPU, no network, no claude)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARDX = REPO_ROOT / "scripts" / "ardx"
STATUS = ARDX / "status.py"
POSTRUN = ARDX / "postrun_hook.sh"
INSTALL = ARDX / "install_units.sh"
UNIT = ARDX / "systemd" / "ardx-watch.service"


@pytest.fixture
def workspace(tmp_path):
    """A registry pointing at tmp roots holding one live and one failed campaign."""
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    (repo / "docs" / "decisions").mkdir(parents=True)
    (repo / "outputs").mkdir()
    (runtime / "runs").mkdir(parents=True)
    (runtime / "orchestration" / "ardx" / "claude-runs").mkdir(parents=True)

    def state(name, jobs, status):
        path = runtime / "runs" / name / "orchestration" / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"schema_version": 1, "campaign_id": name, "status": status, "jobs": jobs}), encoding="utf-8"
        )

    state(
        "camp-failed",
        {
            "a": {
                "status": "failed",
                "attempts": [
                    {
                        "attempt": 1,
                        "failure_class": "scientific",
                        "retryable": False,
                        "finished_at": "2026-09-04T05:00:00+00:00",
                    }
                ],
            },
            "b": {"status": "blocked", "attempts": []},
        },
        "failed",
    )
    state("camp-live", {"a": {"status": "running", "attempts": [{"attempt": 1}]}}, "running")

    bundle = repo / "outputs" / "hand-1" / "run-bundle"
    bundle.mkdir(parents=True)
    bundle.joinpath("manifest.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "hand-1", "status": "completed"}), encoding="utf-8"
    )
    bundle.joinpath("completion.json").write_text("{}", encoding="utf-8")
    bundle.joinpath("error-marker.txt").write_text("no application error recorded\n", encoding="utf-8")

    (repo / "docs" / "decisions" / "0001-next-arm.md").write_text(
        "---\nid: 0001\nstatus: pending\ncreated: 2026-09-05\ncampaign: camp-failed\n"
        "question: which arm next\noptions:\n  A: rerun\nrecommendation: A\nchosen: null\n---\n\nbody\n",
        encoding="utf-8",
    )
    (repo / "docs" / "decisions" / "0000-done.md").write_text(
        "---\nid: 0000\nstatus: decided\ncreated: 2026-09-01\nchosen: A\n---\n", encoding="utf-8"
    )
    (runtime / "orchestration" / "ardx" / "claude-runs" / "20260905T000000Z-camp-failed.json").write_text(
        json.dumps({"result": "recorded"}), encoding="utf-8"
    )

    # Inventory roots: two present (one local mirror, one historical), one absent.
    (repo / "outputs" / "scientific" / "0087").mkdir(parents=True)
    (runtime / "staging" / "ferret-results" / "run-a").mkdir(parents=True)
    (tmp_path / "legacy-runs" / "old-1").mkdir(parents=True)
    (tmp_path / "legacy-runs" / "old-2").mkdir(parents=True)

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_root": str(repo),
                "runtime_root": str(runtime),
                "run_root": str(runtime / "runs"),
                "orchestration_root": str(runtime / "orchestration"),
                "staging_root": str(runtime / "staging"),
                "worktree_root": str(runtime / "worktrees"),
                "lock_root": str(runtime / "locks"),
                "python": sys.executable,
                "historical_roots": {
                    "run_root": str(tmp_path / "legacy-runs"),
                    "campaign_run_root": str(tmp_path / "legacy-campaign-runs"),
                    "analysis_root": str(tmp_path / "legacy-analysis"),
                    "ferret_result_root": str(tmp_path / "ferret-results"),
                },
            }
        ),
        encoding="utf-8",
    )
    return registry


def run_status(registry: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, ARDX_SKIP_REMOTE="1")
    return subprocess.run(
        [sys.executable, str(STATUS), "--registry", str(registry), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=env,
    )


def test_status_json_reports_every_section(workspace):
    result = run_status(workspace, "--json")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert {row["host"] for row in data["hosts"]} == {"hamster", "ferret"}
    assert all(row["note"] == "skipped (ARDX_SKIP_REMOTE)" for row in data["hosts"])
    campaigns = {row["id"]: row for row in data["campaigns"]}
    assert campaigns["camp-live"]["terminal"] is False
    assert campaigns["camp-failed"]["terminal"] is True
    assert campaigns["camp-failed"]["success"] is False
    assert campaigns["camp-failed"]["failure_class"] == "scientific"
    # non-terminal campaigns sort first
    assert data["campaigns"][0]["id"] == "camp-live"
    assert [row["run_id"] for row in data["bundles"]] == ["hand-1"]
    assert data["bundles"][0]["status"] == "completed"
    assert [row["id"] for row in data["decisions"]] == ["0001"]
    assert data["decisions"][0]["question"] == "which arm next"
    assert data["postruns"][0]["name"].endswith("camp-failed.json")
    assert data["watcher"]["state_path"].endswith("orchestration/ardx/watch-state.json")


def test_status_brief_is_at_most_fifteen_lines(workspace):
    result = run_status(workspace, "--brief")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert 0 < len(lines) <= 15
    assert any("camp-live" in line for line in lines)
    assert any("pending decisions: 1" in line for line in lines)


def test_status_markdown_lists_bundles_and_decisions(workspace):
    result = run_status(workspace)
    assert result.returncode == 0, result.stderr
    for heading in (
        "## Hosts",
        "## Watcher",
        "## Campaigns",
        "## Hand-run bundles",
        "## Pending decisions",
        "## Recent postruns",
    ):
        assert heading in result.stdout
    assert "hand-1" in result.stdout
    assert "which arm next" in result.stdout


def test_status_inventory_lists_every_root_per_host(workspace, tmp_path):
    result = run_status(workspace, "--inventory")
    assert result.returncode == 0, result.stderr
    assert str(tmp_path / "legacy-runs") in result.stdout
    assert "missing" in result.stdout  # legacy-campaign-runs was never created

    payload = run_status(workspace, "--json")
    assert payload.returncode == 0, payload.stderr
    rows = {(row["host"], row["label"]): row for row in json.loads(payload.stdout)["inventory"]}
    assert rows[("hamster", "run_root")]["entries"] == 2  # camp-failed, camp-live
    assert rows[("hamster", "run_root")]["newest_mtime"] is not None
    assert rows[("hamster", "repo outputs/scientific")]["entries"] == 1
    assert rows[("hamster", "historical run_root")]["entries"] == 2
    assert rows[("hamster", "historical campaign_run_root")]["note"] == "missing"
    assert rows[("hamster", "historical campaign_run_root")]["entries"] is None
    assert rows[("ferret", "staging mirror")]["entries"] == 1
    remote = rows[("ferret", "remote run_root")]
    assert remote["location"] == "remote"
    assert remote["note"] == "skipped (ARDX_SKIP_REMOTE)"


def test_status_brief_reports_incomplete_bundles(workspace):
    registry = json.loads(workspace.read_text(encoding="utf-8"))
    bundle = Path(registry["repo_root"]) / "outputs" / "hand-2" / "run-bundle"
    bundle.mkdir(parents=True)
    # Declared terminal without the completion triple: the `incomplete` anomaly.
    bundle.joinpath("manifest.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "hand-2", "status": "completed"}), encoding="utf-8"
    )
    result = run_status(workspace, "--brief")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) <= 15
    assert "- bundles: 0 stale, 1 incomplete" in lines


def test_postrun_fallback_allowlist_matches_settings_json():
    """The fallback only runs when settings.json is gone; keep it a literal copy."""
    settings = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    text = POSTRUN.read_text(encoding="utf-8")
    snippet = text[text.index("DEFAULT_TOOLS=(") : text.index("allowed_tools() {")] + "\ndefault_tools\n"
    result = subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split(",") == settings["permissions"]["allow"]


def test_status_survives_a_malformed_state_file(workspace, tmp_path):
    registry = json.loads(workspace.read_text(encoding="utf-8"))
    broken = Path(registry["run_root"]) / "camp-broken" / "orchestration" / "state.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("{oops", encoding="utf-8")
    result = run_status(workspace, "--json")
    assert result.returncode == 0, result.stderr
    ids = {row["id"] for row in json.loads(result.stdout)["campaigns"]}
    assert ids == {"camp-live", "camp-failed"}


def test_postrun_hook_dry_run_prints_the_resolved_command():
    result = subprocess.run(
        ["bash", str(POSTRUN), "campaign", "demo-campaign", "completed"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=dict(os.environ, ARDX_DRY_RUN="1"),
    )
    assert result.returncode == 0, result.stderr
    assert "/experiment-postrun demo-campaign --status completed --kind campaign" in result.stdout
    assert "--output-format json" in result.stdout
    assert "--permission-mode acceptEdits" in result.stdout
    assert "--allowedTools" in result.stdout


def test_postrun_hook_dry_run_honours_model_and_turn_overrides():
    result = subprocess.run(
        ["bash", str(POSTRUN), "run", "hand-1", "completed"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=dict(os.environ, ARDX_DRY_RUN="1", ARDX_MODEL="sonnet", ARDX_MAX_TURNS="7"),
    )
    assert result.returncode == 0, result.stderr
    assert "--model sonnet" in result.stdout
    assert "--max-turns 7" in result.stdout


def test_postrun_hook_never_fails_on_bad_arguments():
    result = subprocess.run(["bash", str(POSTRUN)], capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0
    assert "usage" in result.stderr


def test_install_units_dry_run_prints_the_unit_path():
    result = subprocess.run(
        ["bash", str(INSTALL), "--dry-run"], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert str(UNIT) in result.stdout
    assert "systemd/user/ardx-watch.service" in result.stdout
    assert "would run: systemctl --user enable --now ardx-watch.service" in result.stdout


def test_unit_file_runs_the_watcher_with_the_hook():
    text = UNIT.read_text(encoding="utf-8")
    assert "ExecStart=/usr/bin/python3 " in text
    assert "campaign_watch.py --follow --include-hand-run --on-event " in text
    assert "postrun_hook.sh" in text
    assert "Restart=always" in text
    assert "RestartSec=30" in text
