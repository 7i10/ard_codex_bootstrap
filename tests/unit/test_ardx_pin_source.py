"""scripts/ardx/pin_source.py against a throwaway git repository in tmp_path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIN_SOURCE = REPO_ROOT / "scripts" / "ardx" / "pin_source.py"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=ardx-test", "-c", "user.email=ardx@test", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def workspace(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / ".gitignore").write_text(".external\nteacher_cache\n", encoding="utf-8")
    (repo / ".external").mkdir()
    (repo / ".external" / "marker.txt").write_text("upstream\n", encoding="utf-8")
    (repo / "teacher_cache").mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "initial")
    sha = git(repo, "rev-parse", "HEAD")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_root": str(repo),
                "runtime_root": str(tmp_path / "runtime"),
                "run_root": str(tmp_path / "runtime" / "runs"),
                "orchestration_root": str(tmp_path / "runtime" / "orchestration"),
                "worktree_root": str(tmp_path / "runtime" / "worktrees"),
                "lock_root": str(tmp_path / "runtime" / "locks"),
                "python": sys.executable,
            }
        ),
        encoding="utf-8",
    )
    return repo, registry, sha


def pin(registry: Path, sha: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PIN_SOURCE), sha, "--registry", str(registry)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_creates_then_reuses_the_pinned_worktree(workspace):
    repo, registry, sha = workspace
    first = pin(registry, sha)
    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout)
    assert payload["sha"] == sha
    assert payload["status"] == "created"
    worktree = Path(payload["path"])
    assert worktree.name == f"source-{sha[:12]}"
    assert git(worktree, "rev-parse", "HEAD") == sha
    assert git(worktree, "status", "--porcelain") == ""
    for name in (".external", "teacher_cache"):
        link = worktree / name
        assert link.is_symlink()
        assert link.resolve() == (repo / name).resolve()

    second = pin(registry, sha)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout) == {"path": str(worktree), "sha": sha, "status": "reused"}


def test_short_revision_resolves_to_the_same_worktree(workspace):
    _repo, registry, sha = workspace
    created = json.loads(pin(registry, sha).stdout)
    reused = pin(registry, sha[:8])
    assert reused.returncode == 0, reused.stderr
    payload = json.loads(reused.stdout)
    assert payload["path"] == created["path"]
    assert payload["sha"] == sha
    assert payload["status"] == "reused"


def test_unknown_sha_exits_two(workspace):
    _repo, registry, _sha = workspace
    result = pin(registry, "0" * 40)
    assert result.returncode == 2
    assert "unknown sha" in result.stderr


def test_dirty_worktree_exits_two(workspace):
    _repo, registry, sha = workspace
    worktree = Path(json.loads(pin(registry, sha).stdout)["path"])
    (worktree / "src" / "mod.py").write_text("VALUE = 2\n", encoding="utf-8")
    result = pin(registry, sha)
    assert result.returncode == 2
    assert "is not clean" in result.stderr


def test_head_mismatch_exits_two(workspace):
    repo, registry, sha = workspace
    worktree = Path(json.loads(pin(registry, sha).stdout)["path"])
    (repo / "src" / "mod.py").write_text("VALUE = 3\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "second")
    second = git(repo, "rev-parse", "HEAD")
    git(worktree, "checkout", "-q", "--detach", second)
    result = pin(registry, sha)
    assert result.returncode == 2
    assert "expected" in result.stderr


def test_missing_shared_asset_exits_two(workspace):
    repo, registry, sha = workspace
    (repo / "teacher_cache").rmdir()
    result = pin(registry, sha)
    assert result.returncode == 2
    assert "shared runtime asset is missing" in result.stderr
