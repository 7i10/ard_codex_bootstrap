from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.campaign.watch_successor import (
    SuccessorWatchError,
    archive_owned_lease,
    validate_predecessor,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _spec(tmp_path: Path) -> dict[str, object]:
    output = tmp_path / "output"
    for relative in ("resolved_config.yaml", "best.pt", "last.pt", "run-bundle/manifest.json"):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence", encoding="utf-8")
    predecessor_exit = tmp_path / "train-exit.json"
    completion = output / "run-bundle" / "completion.json"
    _write_json(
        predecessor_exit,
        {"exit_code": 0, "run_id": "job-1", "git_sha": "a" * 40},
    )
    _write_json(completion, {"status": "completed"})
    return {
        "predecessor_exit": str(predecessor_exit),
        "predecessor_completion": str(completion),
        "predecessor_output": str(output),
        "predecessor_run_id": "job-1",
        "predecessor_phase": "train",
        "scientific_git_sha": "a" * 40,
        "gpu_uuid": "GPU-test",
        "gpu_lock_root": str(tmp_path / "locks"),
    }


def test_validate_predecessor_requires_exact_success_identity(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    validate_predecessor(spec)
    _write_json(Path(str(spec["predecessor_exit"])), {"exit_code": 1, "run_id": "job-1", "git_sha": "a" * 40})
    with pytest.raises(SuccessorWatchError, match="expected identity"):
        validate_predecessor(spec)


def test_archive_owned_lease_preserves_evidence(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    state_dir = tmp_path / "watch"
    state_dir.mkdir()
    lease = Path(str(spec["gpu_lock_root"])) / "gpu-GPU-test.lease.json"
    _write_json(lease, {"job_id": "job-1", "phase": "train", "gpu_uuid": "GPU-test"})
    archive_owned_lease(spec, state_dir)
    assert not lease.exists()
    assert json.loads((state_dir / "predecessor-lease.json").read_text(encoding="utf-8"))["job_id"] == "job-1"


def test_archive_refuses_a_different_lease_owner(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    state_dir = tmp_path / "watch"
    state_dir.mkdir()
    lease = Path(str(spec["gpu_lock_root"])) / "gpu-GPU-test.lease.json"
    _write_json(lease, {"job_id": "other", "phase": "train", "gpu_uuid": "GPU-test"})
    with pytest.raises(SuccessorWatchError, match="not owned"):
        archive_owned_lease(spec, state_dir)
    assert lease.exists()
