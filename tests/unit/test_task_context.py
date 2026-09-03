from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "task_context.py"
REGISTRY = REPO_ROOT / "configs" / "workspace" / "ard_workspace_v1.json"


def test_task_context_is_runtime_bound_non_overwriting_and_appendable(tmp_path: Path) -> None:
    values = json.loads(REGISTRY.read_text())
    runtime = tmp_path / "runtime"
    values["runtime_root"] = str(runtime)
    for field in (
        "run_root",
        "analysis_root",
        "staging_root",
        "worktree_root",
        "orchestration_root",
        "task_context_root",
        "lock_root",
        "temp_root",
    ):
        values[field] = str(runtime / field.removesuffix("_root"))
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(values), encoding="utf-8")
    common = ["--registry", str(registry)]
    created = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *common,
            "init",
            "--task-id",
            "fixture",
            "--goal",
            "verify",
            "--source-sha",
            "a" * 40,
        ],
        text=True,
        capture_output=True,
    )
    assert created.returncode == 0, created.stderr
    updated = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *common,
            "append",
            "--task-id",
            "fixture",
            "--field",
            "completed_milestones",
            "--value",
            "M0",
        ],
        text=True,
        capture_output=True,
    )
    assert updated.returncode == 0, updated.stderr
    shown = subprocess.run(
        [sys.executable, str(SCRIPT), *common, "show", "--task-id", "fixture"], text=True, capture_output=True
    )
    assert shown.returncode == 0, shown.stderr
    assert json.loads(shown.stdout)["completed_milestones"] == ["M0"]
    duplicate = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *common,
            "init",
            "--task-id",
            "fixture",
            "--goal",
            "verify",
            "--source-sha",
            "a" * 40,
        ],
        text=True,
        capture_output=True,
    )
    assert duplicate.returncode != 0
