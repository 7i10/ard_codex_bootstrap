from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "operational_non_overwrite_dummy.py"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_public_dummy_cli_creates_only_its_own_new_output(tmp_path: Path) -> None:
    output = tmp_path / "scientific-like-output"
    result = run("--output-dir", str(output), "--payload", "ownership")
    assert result.returncode == 0, result.stderr
    payload = json.loads((output / "artifact.json").read_text())
    assert payload["kind"] == "operational_non_overwrite_dummy"
    assert payload["payload"] == "ownership"
    assert sorted(path.name for path in output.iterdir()) == ["artifact.json"]


def test_public_dummy_cli_refuses_precreated_output(tmp_path: Path) -> None:
    output = tmp_path / "already-exists"
    output.mkdir()
    result = run("--output-dir", str(output), "--payload", "ownership")
    assert result.returncode != 0
    assert "refusing to overwrite existing output directory" in result.stderr
