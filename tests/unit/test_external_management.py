from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.t0

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_external import bootstrap  # noqa: E402
from external_common import ExternalError, load_lock, load_repositories  # noqa: E402
from verify_external import verify  # noqa: E402


def run(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def make_source(tmp_path: Path, *, with_license: bool = True) -> tuple[Path, str]:
    source = tmp_path / "upstream"
    source.mkdir()
    run("init", cwd=source)
    run("config", "user.email", "test@example.invalid", cwd=source)
    run("config", "user.name", "Test", cwd=source)
    if with_license:
        (source / "LICENSE").write_text("test license\n", encoding="utf-8")
    (source / "module.py").write_text("x = 1\n", encoding="utf-8")
    run("add", ".", cwd=source)
    run("commit", "-m", "fixture", cwd=source)
    return source, run("rev-parse", "HEAD", cwd=source)


def write_lock(root: Path, source: Path, commit: str) -> None:
    (root / "external.lock.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "repositories": {
                    "saad": {
                        "url": str(source),
                        "commit": commit,
                        "fetched_at": None,
                        "license_file": None,
                        "license_status": "unclear",
                        "license_evidence": None,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def write_multi_lock(root: Path, sources: dict[str, tuple[Path, str]]) -> None:
    (root / "external.lock.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                # Deliberately reverse insertion order: --all must be stable.
                "repositories": {
                    name: {
                        "url": str(source),
                        "commit": commit,
                        "fetched_at": None,
                        "license_file": None,
                        "license_status": "unclear",
                        "license_evidence": None,
                    }
                    for name, (source, commit) in reversed(tuple(sources.items()))
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_repository_lock_pins_the_approved_saad_revision() -> None:
    _, locked = load_lock(ROOT / "external.lock.yaml")
    assert locked.url == "https://github.com/HongsinLee/saad.git"
    assert locked.commit == "295121c5d2eed827b5b2d6aa42307de809bdfada"
    _, trades = load_lock(ROOT / "external.lock.yaml", repository="trades")
    assert trades.url == "https://github.com/yaodongyu/TRADES.git"
    assert trades.commit == "6e8e11b7c281371c2f027ffadfbaea80361f09de"
    assert trades.license_file == "LICENSE"
    assert trades.license_status == "verified"
    assert trades.license_evidence == {"sha256": "4b42e38a6899d82801eb6782fe161cccb5d3d685c8bcddc2b877ac9f87161a30"}


def test_bootstrap_is_atomic_and_verifiable_with_local_git_fixture(tmp_path: Path) -> None:
    source, commit = make_source(tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    write_lock(root, source, commit)

    checkout = bootstrap(root=root, update_lock=True)

    assert checkout == root / ".external" / "saad"
    assert run("rev-parse", "HEAD", cwd=checkout) == commit
    assert verify(root=root)["license_file"] == "LICENSE"
    assert not list((root / ".external").glob(".saad.clone-*"))


def test_verified_lock_requires_file_and_digest(tmp_path: Path) -> None:
    source, commit = make_source(tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    write_lock(root, source, commit)
    raw = yaml.safe_load((root / "external.lock.yaml").read_text(encoding="utf-8"))
    raw["repositories"]["saad"]["license_status"] = "verified"
    (root / "external.lock.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ExternalError, match="verified license status"):
        load_lock(root / "external.lock.yaml")


@pytest.mark.parametrize("status", ("absent", "unclear"))
def test_newly_appearing_license_must_match_absent_or_unclear_lock(tmp_path: Path, status: str) -> None:
    source, commit = make_source(tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    write_lock(root, source, commit)
    raw = yaml.safe_load((root / "external.lock.yaml").read_text(encoding="utf-8"))
    raw["repositories"]["saad"]["license_status"] = status
    (root / "external.lock.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    bootstrap(root=root)
    with pytest.raises(ExternalError, match="license evidence does not match"):
        verify(root=root)


def test_disappearing_verified_license_is_rejected(tmp_path: Path) -> None:
    source, commit = make_source(tmp_path, with_license=False)
    root = tmp_path / "project"
    root.mkdir()
    write_lock(root, source, commit)
    raw = yaml.safe_load((root / "external.lock.yaml").read_text(encoding="utf-8"))
    entry = raw["repositories"]["saad"]
    entry["license_status"] = "verified"
    entry["license_file"] = "LICENSE"
    entry["license_evidence"] = {"sha256": "0" * 64}
    (root / "external.lock.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    bootstrap(root=root)
    with pytest.raises(ExternalError, match="license evidence does not match"):
        verify(root=root)


def test_existing_remote_mismatch_and_dirty_tree_are_preserved(tmp_path: Path) -> None:
    source, commit = make_source(tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    write_lock(root, source, commit)
    checkout = bootstrap(root=root)
    run("remote", "set-url", "origin", "different://remote", cwd=checkout)
    with pytest.raises(ExternalError, match="origin mismatch"):
        bootstrap(root=root)
    assert run("remote", "get-url", "origin", cwd=checkout) == "different://remote"
    run("remote", "set-url", "origin", str(source), cwd=checkout)
    (checkout / "local-change").write_text("do not erase", encoding="utf-8")
    with pytest.raises(ExternalError, match="dirty"):
        bootstrap(root=root)
    assert (checkout / "local-change").read_text(encoding="utf-8") == "do not erase"


def test_existing_head_mismatch_is_rejected_without_checkout(tmp_path: Path) -> None:
    source, commit = make_source(tmp_path)
    (source / "module.py").write_text("x = 2\n", encoding="utf-8")
    run("add", "module.py", cwd=source)
    run("commit", "-m", "later", cwd=source)
    later = run("rev-parse", "HEAD", cwd=source)
    root = tmp_path / "project"
    root.mkdir()
    write_lock(root, source, commit)
    checkout = bootstrap(root=root)
    run("checkout", "--detach", later, cwd=checkout)
    observed = run("rev-parse", "HEAD", cwd=checkout)
    with pytest.raises(ExternalError, match="HEAD mismatch"):
        bootstrap(root=root)
    assert run("rev-parse", "HEAD", cwd=checkout) == observed


def test_failed_clone_never_creates_destination(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    write_lock(root, tmp_path / "does-not-exist", "295121c5d2eed827b5b2d6aa42307de809bdfada")

    with pytest.raises(ExternalError):
        bootstrap(root=root)

    assert not (root / ".external" / "saad").exists()


def test_lock_evidence_changes_only_with_explicit_update(tmp_path: Path) -> None:
    source, commit = make_source(tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    write_lock(root, source, commit)
    before = (root / "external.lock.yaml").read_text(encoding="utf-8")

    bootstrap(root=root)
    assert (root / "external.lock.yaml").read_text(encoding="utf-8") == before
    bootstrap(root=root, update_lock=True)
    _, locked = load_lock(root / "external.lock.yaml")
    assert locked.license_file == "LICENSE"
    assert locked.license_status == "unclear"
    assert locked.license_evidence and len(locked.license_evidence["sha256"]) == 64


def test_update_lock_preserves_matching_verified_license_evidence(tmp_path: Path) -> None:
    source, commit = make_source(tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    write_lock(root, source, commit)
    bootstrap(root=root, update_lock=True)
    raw = yaml.safe_load((root / "external.lock.yaml").read_text(encoding="utf-8"))
    raw["repositories"]["saad"]["license_status"] = "verified"
    (root / "external.lock.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    bootstrap(root=root, update_lock=True)

    _, locked = load_lock(root / "external.lock.yaml")
    assert locked.license_status == "verified"
    assert locked.license_file == "LICENSE"
    assert locked.license_evidence and len(locked.license_evidence["sha256"]) == 64


def test_named_and_all_repository_selection_is_sorted_and_preserves_default(tmp_path: Path) -> None:
    saad_parent = tmp_path / "saad"
    trades_parent = tmp_path / "trades"
    saad_parent.mkdir()
    trades_parent.mkdir()
    saad_source, saad_commit = make_source(saad_parent)
    trades_source, trades_commit = make_source(trades_parent)
    root = tmp_path / "project"
    root.mkdir()
    write_multi_lock(root, {"saad": (saad_source, saad_commit), "trades": (trades_source, trades_commit)})

    named = bootstrap(root=root, repository="trades", update_lock=True)
    assert named == root / ".external" / "trades"
    assert not (root / ".external" / "saad").exists()
    all_paths = bootstrap(root=root, all_repositories=True, update_lock=True)
    assert all_paths == (root / ".external" / "saad", root / ".external" / "trades")
    assert bootstrap(root=root) == root / ".external" / "saad"
    reports = verify(root=root, all_repositories=True)
    assert tuple(report["name"] for report in reports) == ("saad", "trades")
    _, repositories = load_repositories(root / "external.lock.yaml")
    assert tuple(locked.name for locked in repositories) == ("saad", "trades")


def test_unknown_or_unsafe_named_repository_fails_closed(tmp_path: Path) -> None:
    source, commit = make_source(tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    write_lock(root, source, commit)
    with pytest.raises(ExternalError, match="no repository"):
        bootstrap(root=root, repository="trades")
    raw = yaml.safe_load((root / "external.lock.yaml").read_text(encoding="utf-8"))
    raw["repositories"]["../unsafe"] = raw["repositories"].pop("saad")
    (root / "external.lock.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ExternalError, match="unsafe"):
        load_repositories(root / "external.lock.yaml")
