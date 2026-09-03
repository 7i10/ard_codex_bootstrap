from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / ".agents/skills/production-launch-gate/scripts/artifact_inventory.py"
SOURCE_SHA = "a" * 40


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(*, epoch: int = 114, campaign_id: str = "campaign-v1", source_sha: str = SOURCE_SHA) -> dict:
    return {
        "campaign_id": campaign_id,
        "source_sha": source_sha,
        "job_id": "endpoint-dev1",
        "seed": "dev-1",
        "arm": "CONTROL",
        "epoch": epoch,
        "split": "held-out",
        "attack_identity": "attack-v1",
    }


def inventory(tmp_path: Path, *, required: list[dict] | None = None) -> tuple[Path, dict, Path, Path]:
    staging = tmp_path / "staging" / "remote-rows.json"
    staging.parent.mkdir(parents=True)
    staging.write_bytes(b"remote artifact bytes\n")
    collected = tmp_path / "canonical" / "endpoint-dev1-e114.json"
    artifact = {
        "origin": {"host": "ferret", "path": "/remote/outputs/endpoint-dev1.json"},
        "staging_path": str(staging),
        "collected_path": str(collected),
        "sha256": sha(staging),
        "identity": identity(),
    }
    payload = {
        "schema_version": 1,
        "campaign_id": "campaign-v1",
        "source_sha": SOURCE_SHA,
        "artifacts": [artifact],
        "required_cells": required if required is not None else [identity()],
    }
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload, staging, collected


def invoke(command: str, manifest: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), command, "--manifest", str(manifest)], text=True, capture_output=True
    )


def test_collection_stages_remote_metadata_to_canonical_local_path(tmp_path: Path) -> None:
    manifest, _, _, collected = inventory(tmp_path)
    result = invoke("stage", manifest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert collected.read_bytes() == b"remote artifact bytes\n"
    report = json.loads(result.stdout)
    assert report["artifacts"][0]["origin_path"] == "/remote/outputs/endpoint-dev1.json"
    assert report["artifacts"][0]["collected_path"] == str(collected)


def test_inventory_rejects_missing_required_endpoint_cell(tmp_path: Path) -> None:
    manifest, _, _, _ = inventory(tmp_path, required=[identity(), identity(epoch=109)])
    result = invoke("inspect", manifest)
    assert result.returncode == 2
    assert "required_cells" in result.stdout


def test_inventory_rejects_collected_hash_mismatch(tmp_path: Path) -> None:
    manifest, _, _, collected = inventory(tmp_path)
    assert invoke("stage", manifest).returncode == 0
    collected.write_bytes(b"one byte changed\n")
    result = invoke("validate", manifest)
    assert result.returncode == 2
    assert "collected_path.sha256" in result.stdout


def test_inventory_rejects_foreign_campaign_or_source_identity(tmp_path: Path) -> None:
    manifest, payload, _, _ = inventory(tmp_path)
    payload["artifacts"][0]["identity"] = identity(campaign_id="other-campaign")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result = invoke("inspect", manifest)
    assert result.returncode == 2
    assert "identity.campaign_id" in result.stdout
