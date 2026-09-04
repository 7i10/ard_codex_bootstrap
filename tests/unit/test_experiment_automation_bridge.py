from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECONCILER = ROOT / "scripts" / "reconcile_experiment.py"
GATE = ROOT / ".agents/skills/production-launch-gate/scripts/launch_gate.py"
PUBLISHER = ROOT / "scripts/publish_experiment_terminal_event.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_reconciler(state: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RECONCILER), "--state", str(state), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def write_campaign(tmp_path: Path, *, statuses: dict[str, str], owner: str = "orchestrator_dag") -> Path:
    root = tmp_path / "campaign"
    root.mkdir()
    manifest_sha = "m" * 64
    orch = {
        "schema_version": 1,
        "campaign_id": "campaign",
        "manifest_sha256": manifest_sha,
        "source_sha": "a" * 40,
        "status": "running",
        "jobs": {
            job_id: {
                "status": status,
                "attempts": [
                    {
                        "failure_class": "technical",
                        "retryable": True,
                        "failure_reason": "fixture",
                    }
                ]
                if status == "failed"
                else [],
            }
            for job_id, status in statuses.items()
        },
    }
    orch_path = root / "orchestrator-state.json"
    orch_path.write_text(json.dumps(orch), encoding="utf-8")
    state = {
        "schema_version": 2,
        "experiment_id": "campaign",
        "mode": "orchestrator_campaign",
        "campaign_id": "campaign",
        "manifest_sha256": manifest_sha,
        "source_sha": "a" * 40,
        "scientific_identity_hash": "identity",
        "orchestrator_state_path": str(orch_path),
        "required_training_jobs": ["train"],
        "terminal_result_jobs": ["endpoint"],
        "state": "TRAINING",
        "postprocess": {"owner_kind": owner},
    }
    state_path = root / "experiment-state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state_path


def test_campaign_reads_orchestrator_as_authority_and_ignores_pid(tmp_path: Path) -> None:
    state = write_campaign(tmp_path, statuses={"train": "running", "endpoint": "pending"})
    data = json.loads(state.read_text())
    data["training"] = {"pid": 99999999}
    state.write_text(json.dumps(data), encoding="utf-8")
    result = run_reconciler(state, "--scheduled")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["reason"] == "training_campaign_active"


def test_campaign_dag_owner_never_duplicates_downstream(tmp_path: Path) -> None:
    state = write_campaign(tmp_path, statuses={"train": "completed", "endpoint": "running"})
    result = run_reconciler(state, "--scheduled")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "NO_OP"
    assert payload["reason"] == "downstream_jobs_active"
    assert json.loads(state.read_text())["state"] == "EVALUATING"


def test_external_campaign_reuses_single_owner_handoff(tmp_path: Path) -> None:
    state = write_campaign(
        tmp_path,
        statuses={"train": "completed", "endpoint": "pending"},
        owner="external_registered_command",
    )
    data = json.loads(state.read_text())
    data["postprocess"].update(
        {
            "command": [sys.executable, "-c", "pass"],
            "completion_marker": "postprocess/completion.json",
            "failure_marker": "postprocess/failure.json",
        }
    )
    state.write_text(json.dumps(data), encoding="utf-8")
    result = run_reconciler(state)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "HANDOFF"
    assert json.loads(state.read_text())["postprocess_attempts"] == 1


def test_technical_failure_delegates_bounded_retry_without_identity_change(tmp_path: Path) -> None:
    state = write_campaign(
        tmp_path,
        statuses={"train": "failed", "endpoint": "pending"},
        owner="external_registered_command",
    )
    data = json.loads(state.read_text())
    data["recovery"] = {"command": [sys.executable, "-c", "pass"], "max_attempts": 1}
    state.write_text(json.dumps(data), encoding="utf-8")
    result = run_reconciler(state, "--lease-seconds", "30")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "RECOVERY_HANDOFF"
    updated = json.loads(state.read_text())
    assert updated["state"] == "TRAINING"
    assert updated["source_sha"] == "a" * 40
    assert updated["scientific_identity_hash"] == "identity"
    assert updated["recovery_attempts"] == 1


def test_scientific_failure_is_not_auto_retried(tmp_path: Path) -> None:
    state = write_campaign(tmp_path, statuses={"train": "failed", "endpoint": "pending"})
    orch_path = Path(json.loads(state.read_text())["orchestrator_state_path"])
    orch = json.loads(orch_path.read_text())
    orch["jobs"]["train"]["attempts"][0]["failure_class"] = "scientific"
    orch["jobs"]["train"]["attempts"][0]["retryable"] = False
    orch_path.write_text(json.dumps(orch), encoding="utf-8")
    result = run_reconciler(state)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "NEEDS_RESEARCH_DECISION"


def test_runtime_signature_requires_registry_match(tmp_path: Path) -> None:
    gate = load_module(GATE, "bridge_gate_signature")
    spec = {"operational_profile": "FAST_EXISTING_RUNTIME", "tier": "production", "runtime_signature": "unknown"}
    assert gate.runtime_signature_report(spec, tmp_path)["status"] == "registry_missing"
    registry = tmp_path / "signatures.json"
    registry.write_text(
        json.dumps({"schema_version": 1, "signatures": [{"id": "known", "executor_class": "local"}]}),
        encoding="utf-8",
    )
    spec["runtime_signature_registry"] = str(registry)
    spec["runtime_signature"] = {"id": "known", "executor_class": "local"}
    report = gate.runtime_signature_report(spec, tmp_path)
    assert report["required"] is True
    assert report["status"] == "validated"


def test_experiment_state_bridge_uses_frozen_manifest_digest(tmp_path: Path) -> None:
    gate = load_module(GATE, "bridge_gate_state")
    frozen = tmp_path / "resolved-manifest.json"
    frozen.write_text("{}\n", encoding="utf-8")
    manifest = {
        "campaign_id": "campaign",
        "source": {"git_sha": "a" * 40},
        "state_path": str(tmp_path / "orch-state.json"),
        "launch_gate": {"scientific_identity_hashes": {"train": "b" * 64}},
        "jobs": [{"job_id": "train", "job_type": "training"}],
    }
    spec = {"experiment_state_path": str(tmp_path / "experiment-state.json")}
    path = gate.write_experiment_state(spec, manifest, frozen)
    assert path and path.is_file()
    state = json.loads(path.read_text())
    assert state["manifest_sha256"] == gate.file_digest(frozen)
    assert state["required_training_jobs"] == ["train"]


def test_terminal_event_publisher_is_idempotent(tmp_path: Path) -> None:
    publisher = load_module(PUBLISHER, "bridge_publisher")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "experiment-results"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    report = tmp_path / "report.md"
    result_manifest = tmp_path / "result.json"
    report.write_text("done\n", encoding="utf-8")
    result_manifest.write_text("{}\n", encoding="utf-8")
    payload = tmp_path / "terminal.json"
    payload.write_text(
        json.dumps(
            {
                "terminal_state": "AWAITING_RESEARCH_REVIEW",
                "result_id": "campaign",
                "result_revision": "r1",
                "canonical_commit_sha": "a" * 40,
                "source_sha": "b" * 40,
                "result_manifest": str(result_manifest),
                "report": str(report),
                "artifact_digest": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    first = publisher.publish(payload, worktree=repo, push=False, ensure_pr=False)
    second = publisher.publish(payload, worktree=repo, push=False, ensure_pr=False)
    assert first["status"] == "PUBLISHED"
    assert second["status"] == "NO_OP"


def test_publisher_rejects_nonterminal_manifest(tmp_path: Path) -> None:
    publisher = load_module(PUBLISHER, "bridge_publisher_invalid")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"terminal_state": "TRAINING", "result_id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not publishable"):
        publisher.load_result(path)


def test_publisher_rejects_path_traversal_revision(tmp_path: Path) -> None:
    publisher = load_module(PUBLISHER, "bridge_publisher_path")
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "terminal_state": "SUCCESS",
                "result_id": "../escape",
                "result_revision": "r1",
                "canonical_commit_sha": "a" * 40,
                "source_sha": "b" * 40,
                "result_manifest": "manifest.json",
                "report": "report.md",
                "artifact_digest": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="safe path component"):
        publisher.load_result(path)
