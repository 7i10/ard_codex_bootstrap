from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[2] / ".agents/skills/run-on-ferret/scripts"


@pytest.mark.unit
@pytest.mark.parametrize("name", ["ferret-common", "ferret-preflight", "ferret-prepare", "ferret-launch", "ferret-status", "ferret-logs", "ferret-collect", "ferret-cancel", "ferret-cleanup"])
def test_shell_syntax(name: str) -> None:
    subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)


@pytest.mark.unit
@pytest.mark.parametrize("run_id", ["../x", "a..b", "has space", "a/b", "$(x)"])
def test_cleanup_rejects_unsafe_run_id(run_id: str) -> None:
    result = subprocess.run([str(SCRIPTS / "ferret-cleanup"), "--run-id", run_id], text=True, capture_output=True)
    assert result.returncode != 0
    assert "invalid run-id" in result.stderr


@pytest.mark.unit
def test_cleanup_is_dry_run_by_default(tmp_path: Path) -> None:
    result = subprocess.run([str(SCRIPTS / "ferret-cleanup"), "--run-id", "safe-run"], env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "HAMSTER_RESULT_ROOT": str(tmp_path)}, text=True, capture_output=True, check=True)
    assert '"dry_run":true' in result.stdout


@pytest.mark.unit
def test_launch_rejects_duplicate_gpu_before_ssh() -> None:
    result = subprocess.run([str(SCRIPTS / "ferret-launch"), "--run-id", "safe", "--gpus", "0,0", "--", "true"], text=True, capture_output=True)
    assert result.returncode != 0
    assert "duplicate GPU" in result.stderr


@pytest.mark.unit
def test_nohup_setsid_launch_detaches_ssh_standard_streams() -> None:
    launch = (SCRIPTS / "ferret-launch").read_text()
    assert 'nohup setsid bash "$RUN/control/launch.sh"' in launch
    assert '</dev/null >"$RUN/control/supervisor.log" 2>&1 &' in launch
    assert "screen -" not in launch
