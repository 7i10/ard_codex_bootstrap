from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[2] / ".agents/skills/run-on-ferret/scripts"


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        "ferret-common",
        "ferret-preflight",
        "ferret-prepare",
        "ferret-launch",
        "ferret-status",
        "ferret-logs",
        "ferret-collect",
        "ferret-cancel",
        "ferret-cleanup",
    ],
)
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
    result = subprocess.run(
        [str(SCRIPTS / "ferret-cleanup"), "--run-id", "safe-run"],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "HAMSTER_RESULT_ROOT": str(tmp_path)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"dry_run":true' in result.stdout


@pytest.mark.unit
def test_launch_rejects_duplicate_gpu_before_ssh() -> None:
    result = subprocess.run(
        [str(SCRIPTS / "ferret-launch"), "--run-id", "safe", "--gpus", "0,0", "--", "true"],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "duplicate GPU" in result.stderr


@pytest.mark.unit
def test_nohup_setsid_launch_detaches_ssh_standard_streams() -> None:
    launch = (SCRIPTS / "ferret-launch").read_text()
    assert 'nohup setsid bash "$RUN/control/launch.sh"' in launch
    assert '</dev/null >"$RUN/control/supervisor.log" 2>&1 &' in launch
    assert "screen -" not in launch


@pytest.mark.unit
def test_prepare_links_only_named_shared_runtime_assets() -> None:
    prepare = (SCRIPTS / "ferret-prepare").read_text()
    assert "for name in .external teacher_cache" in prepare
    assert "required shared runtime asset is missing" in prepare
    assert 'ln -s "$REPO/$name" "$RUN/repo/$name"' in prepare
    assert "shared_runtime_assets" in prepare


@pytest.mark.unit
def test_shared_runtime_asset_symlinks_are_gitignored(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    repository = tmp_path / "repository"
    assets = tmp_path / "assets"
    repository.mkdir()
    (assets / ".external").mkdir(parents=True)
    (assets / "teacher_cache").mkdir()
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
    shutil.copy2(root / ".gitignore", repository / ".gitignore")
    (repository / ".external").symlink_to(assets / ".external", target_is_directory=True)
    (repository / "teacher_cache").symlink_to(assets / "teacher_cache", target_is_directory=True)

    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--", ".external", "teacher_cache"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert status.stdout == ""


@pytest.mark.unit
def test_preflight_requires_all_locked_external_checkouts_and_teacher_cache() -> None:
    preflight = (SCRIPTS / "ferret-preflight").read_text()
    assert "('saad','trades','robustbench')" in preflight
    assert "all(report['external_checkouts_available'].values())" in preflight
    assert "report['teacher_cache_available']" in preflight
