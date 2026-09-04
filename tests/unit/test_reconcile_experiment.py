from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "reconcile_experiment.py"


def write_state(
    tmp_path: Path,
    *,
    state: str = "TRAINING",
    pid: int | None = None,
    command: list[str] | None = None,
) -> Path:
    root = tmp_path / "exp"
    root.mkdir()
    (root / "training").mkdir()
    (root / "postprocess").mkdir()
    (root / "training" / "last.pt").write_text("checkpoint", encoding="utf-8")
    completion = root / "training" / "completion.json"
    completion.write_text(
        json.dumps(
            {
                "status": "completed",
                "experiment_id": "fixture",
                "source_sha": "a" * 40,
                "scientific_identity_hash": "identity",
            }
        ),
        encoding="utf-8",
    )
    exit_evidence = root / "training" / "exit.json"
    exit_evidence.write_text(
        json.dumps(
            {
                "exit_code": 0,
                "status": "success",
                "experiment_id": "fixture",
                "source_sha": "a" * 40,
                "scientific_identity_hash": "identity",
            }
        ),
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "fixture",
        "scientific_identity_hash": "identity",
        "source_sha": "a" * 40,
        "state": state,
        "training": {
            "pid": pid if pid is not None else 99999999,
            "completion_marker": "training/completion.json",
            "exit_evidence": "training/exit.json",
            "expected_outputs": ["training/last.pt"],
        },
        "postprocess": {
            "command": command or [sys.executable, "-c", "pass"],
            "completion_marker": "postprocess/completion.json",
            "failure_marker": "postprocess/failure.json",
        },
    }
    path = root / "experiment-state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run(state: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--state", str(state), *extra],
        text=True,
        capture_output=True,
        check=False,
    )


def read(state: Path) -> dict:
    return json.loads(state.read_text(encoding="utf-8"))


def test_running_pid_is_fast_noop(tmp_path: Path) -> None:
    state = write_state(tmp_path, pid=os.getpid())
    result = run(state, "--scheduled")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["reason"] == "training_running"
    assert read(state)["state"] == "TRAINING"


def test_dead_pid_without_completion_is_not_success(tmp_path: Path) -> None:
    state = write_state(tmp_path)
    data = read(state)
    data["training"]["completion_marker"] = "training/missing.json"
    state.write_text(json.dumps(data), encoding="utf-8")
    result = run(state)
    assert result.returncode == 0, result.stderr
    assert read(state)["state"] == "TRAINING_FAILED"
    assert read(state)["failure_class"] == "unknown_terminal_failure"


def test_valid_success_claims_one_postprocess_owner(tmp_path: Path) -> None:
    marker = tmp_path / "marker.py"
    marker.write_text(
        "import json, os, pathlib, time; "
        "p = pathlib.Path(os.environ['ERT_POSTPROCESS_COMPLETION_MARKER']); "
        "p.parent.mkdir(exist_ok=True); time.sleep(.4); "
        "json.dump({"
        "'status': 'completed', 'experiment_id': 'fixture', "
        "'source_sha': '"
        + "a" * 40
        + "', 'scientific_identity_hash': 'identity', "
        "'final_state': 'AWAITING_RESEARCH_REVIEW'}, p.open('w'))",
        encoding="utf-8",
    )
    state = write_state(tmp_path, command=[sys.executable, str(marker)])
    first = run(state)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["status"] == "HANDOFF"
    second = run(state, "--scheduled")
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["status"] == "NO_OP"
    assert json.loads(second.stdout)["reason"] in {"postprocess_running_or_leased", "postprocess_lease_active"}
    time.sleep(.55)
    third = run(state, "--scheduled")
    assert third.returncode == 0, third.stderr
    assert read(state)["state"] == "AWAITING_RESEARCH_REVIEW"


def test_two_simultaneous_reconcilers_have_one_owner(tmp_path: Path) -> None:
    marker = tmp_path / "slow.py"
    marker.write_text(
        "import os,time; time.sleep(1.0)",
        encoding="utf-8",
    )
    state = write_state(tmp_path, command=[sys.executable, str(marker)])
    first = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--state", str(state), "--scheduled"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    second = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--state", str(state), "--scheduled"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    first_out, first_err = first.communicate(timeout=5)
    second_out, second_err = second.communicate(timeout=5)
    assert first.returncode == 0, first_err
    assert second.returncode == 0, second_err
    statuses = {json.loads(first_out)["status"], json.loads(second_out)["status"]}
    assert statuses <= {"HANDOFF", "NO_OP"}
    assert "HANDOFF" in statuses
    assert read(state)["postprocess_attempts"] == 1


def test_active_lease_is_noop_even_without_a_live_pid(tmp_path: Path) -> None:
    state = write_state(tmp_path, state="TRAINING_SUCCESS")
    data = read(state)
    data["postprocess_owner"] = "other-host:123"
    data["lease_id"] = "already-owned"
    data["lease_expires_at"] = "2999-01-01T00:00:00+00:00"
    state.write_text(json.dumps(data), encoding="utf-8")

    result = run(state, "--scheduled")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["reason"] == "postprocess_lease_active"
    assert read(state)["lease_id"] == "already-owned"


def test_expired_lease_can_recover_same_identity(tmp_path: Path) -> None:
    state = write_state(tmp_path)
    data = read(state)
    data["state"] = "EVALUATING"
    data["postprocess_pid"] = 99999999
    data["postprocess_attempts"] = 1
    data["lease_expires_at"] = "2000-01-01T00:00:00+00:00"
    state.write_text(json.dumps(data), encoding="utf-8")
    result = run(state, "--max-postprocess-attempts", "2")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "HANDOFF"
    updated = read(state)
    assert updated["scientific_identity_hash"] == "identity"
    assert updated["source_sha"] == "a" * 40
    assert updated["postprocess_attempts"] == 2


def test_already_pushed_is_noop(tmp_path: Path) -> None:
    state = write_state(tmp_path, state="PUSHED")
    result = run(state, "--scheduled")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "NO_OP", "reason": "terminal_state", "state": "PUSHED"}


@pytest.mark.parametrize("failure_class", ["technical", "scientific"])
def test_failure_marker_never_changes_scientific_identity(tmp_path: Path, failure_class: str) -> None:
    state = write_state(tmp_path)
    data = read(state)
    failure = state.parent / "training" / "failure.json"
    failure.write_text(
        json.dumps(
            {
                "failure_class": failure_class,
                "experiment_id": "fixture",
                "source_sha": "a" * 40,
                "scientific_identity_hash": "identity",
            }
        ),
        encoding="utf-8",
    )
    data["training"]["failure_marker"] = "training/failure.json"
    state.write_text(json.dumps(data), encoding="utf-8")
    result = run(state)
    assert result.returncode == 0, result.stderr
    updated = read(state)
    assert updated["state"] == "TRAINING_FAILED"
    assert updated["scientific_identity_hash"] == "identity"
    assert updated["source_sha"] == "a" * 40
