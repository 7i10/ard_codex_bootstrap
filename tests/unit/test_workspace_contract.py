from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ard.workspace import RUNTIME_FIELDS, load_workspace_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "configs" / "workspace" / "ard_workspace_v1.json"


def test_workspace_registry_has_all_runtime_paths_under_one_root() -> None:
    contract = load_workspace_contract(REGISTRY)

    assert contract.values["schema_version"] == 1
    assert set(contract.runtime_paths) == set(RUNTIME_FIELDS)
    for path in contract.runtime_paths.values():
        assert path == contract.runtime_root or contract.runtime_root in path.parents


def test_workspace_contract_rejects_unregistered_runtime_write() -> None:
    contract = load_workspace_contract(REGISTRY)

    with pytest.raises(ValueError, match="future ARD runtime write"):
        contract.require_runtime_write("/tmp/not-an-ard-runtime-write")


def test_workspace_doctor_reports_compact_json_and_can_create_runtime_layout(tmp_path: Path) -> None:
    values = json.loads(REGISTRY.read_text(encoding="utf-8"))
    runtime_root = tmp_path / "runtime"
    values["runtime_root"] = str(runtime_root)
    for field in RUNTIME_FIELDS:
        if field == "runtime_root":
            continue
        values[field] = str(runtime_root / field.removesuffix("_root"))
    registry = tmp_path / "workspace.json"
    registry.write_text(json.dumps(values), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "workspace_doctor.py"),
            "--registry",
            str(registry),
            "--json",
            "--ensure-runtime",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["workspace_registry"] == str(registry.resolve())
    assert all(row["exists"] for row in report["runtime_layout"].values())
