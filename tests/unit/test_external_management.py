from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.t0

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import bootstrap_teacher as bootstrap_teacher_script  # noqa: E402
from bootstrap_external import bootstrap  # noqa: E402
from bootstrap_teacher import bootstrap as bootstrap_teacher  # noqa: E402
from external_common import ExternalError, load_lock, load_repositories  # noqa: E402
from verify_external import verify  # noqa: E402
from verify_teacher import verify as verify_teacher  # noqa: E402

from ard.models.teacher_registry import TeacherRegistry, TeacherRegistryError  # noqa: E402


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


def test_teacher_bootstrap_is_explicit_atomic_and_never_overwrites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import shutil

    shutil.copyfile(ROOT / "teachers.lock.yaml", tmp_path / "teachers.lock.yaml")
    real_load = TeacherRegistry.load
    registry = replace(real_load(ROOT), root=tmp_path)
    source = tmp_path / "supplied.pt"
    source.write_bytes(b"local-only teacher bytes")
    monkeypatch.setattr("bootstrap_teacher.TeacherRegistry.load", lambda _root: registry)
    monkeypatch.setattr(TeacherRegistry, "validate_external", lambda _self: None)

    destination = bootstrap_teacher(
        root=tmp_path, registry_id="chen2021_ltd_wrn34_10", source=source, update_lock=True
    )
    assert destination.read_bytes() == source.read_bytes()
    raw = yaml.safe_load((tmp_path / "teachers.lock.yaml").read_text(encoding="utf-8"))
    entry = raw["teachers"]["chen2021_ltd_wrn34_10"]
    assert entry["checkpoint_status"] == "verified" and len(entry["checkpoint_sha256"]) == 64
    monkeypatch.setattr(TeacherRegistry, "load", staticmethod(real_load))
    report = verify_teacher(root=tmp_path, registry_id="chen2021_ltd_wrn34_10")
    assert report["external_commit"] == registry.repository_commit
    assert report["checkpoint_sha256"] == entry["checkpoint_sha256"]
    with pytest.raises(TeacherRegistryError, match="refusing to overwrite"):
        bootstrap_teacher(root=tmp_path, registry_id="chen2021_ltd_wrn34_10", source=source, update_lock=True)
    with pytest.raises(TeacherRegistryError, match="no download will be attempted"):
        verify_teacher(root=tmp_path, registry_id="bartoldson2024_adversarial_wrn94_16")


def test_teacher_bootstrap_requires_lock_update_and_rolls_back_lock_publish_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import shutil

    shutil.copyfile(ROOT / "teachers.lock.yaml", tmp_path / "teachers.lock.yaml")
    registry = replace(TeacherRegistry.load(ROOT), root=tmp_path)
    source = tmp_path / "supplied.pt"
    source.write_bytes(b"local-only teacher bytes")
    destination = tmp_path / registry.spec("chen2021_ltd_wrn34_10").checkpoint_path
    monkeypatch.setattr("bootstrap_teacher.TeacherRegistry.load", lambda _root: registry)
    monkeypatch.setattr(TeacherRegistry, "validate_external", lambda _self: None)
    with pytest.raises(TeacherRegistryError, match="requires --update-lock"):
        bootstrap_teacher(root=tmp_path, registry_id="chen2021_ltd_wrn34_10", source=source)
    assert not destination.exists()

    original_publish_lock = bootstrap_teacher_script._publish_lock
    monkeypatch.setattr(
        bootstrap_teacher_script,
        "_publish_lock",
        lambda _temporary, _lock_path: (_ for _ in ()).throw(OSError("injected lock write failure")),
    )
    with pytest.raises(OSError, match="injected lock write failure"):
        bootstrap_teacher(root=tmp_path, registry_id="chen2021_ltd_wrn34_10", source=source, update_lock=True)
    assert not destination.exists()
    raw = yaml.safe_load((tmp_path / "teachers.lock.yaml").read_text(encoding="utf-8"))
    assert raw["teachers"]["chen2021_ltd_wrn34_10"]["checkpoint_status"] == "missing"

    monkeypatch.setattr(bootstrap_teacher_script, "_publish_lock", original_publish_lock)
    retried = bootstrap_teacher(
        root=tmp_path, registry_id="chen2021_ltd_wrn34_10", source=source, update_lock=True
    )
    assert retried == destination


def test_teacher_bootstrap_lock_is_project_global_across_registry_ids(tmp_path: Path) -> None:
    expected = tmp_path / ".cache" / "teacher-bootstrap.lock"
    assert bootstrap_teacher_script._project_lock_path(tmp_path) == expected
    # The lock deliberately has no registry ID: Chen and Bartoldson updates
    # serialize their shared teachers.lock.yaml replacement.
    assert bootstrap_teacher_script._project_lock_path(tmp_path) == expected


def test_teacher_bootstrap_preserves_destination_race_and_allows_retry_after_removal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import shutil

    shutil.copyfile(ROOT / "teachers.lock.yaml", tmp_path / "teachers.lock.yaml")
    registry = replace(TeacherRegistry.load(ROOT), root=tmp_path)
    source = tmp_path / "supplied.pt"
    source.write_bytes(b"local-only teacher bytes")
    destination = tmp_path / registry.spec("chen2021_ltd_wrn34_10").checkpoint_path
    monkeypatch.setattr("bootstrap_teacher.TeacherRegistry.load", lambda _root: registry)
    monkeypatch.setattr(TeacherRegistry, "validate_external", lambda _self: None)
    original_publish_checkpoint = bootstrap_teacher_script._publish_checkpoint

    def inject_race(_temporary: Path, target: Path) -> None:
        target.write_bytes(b"other process")
        raise TeacherRegistryError("refusing to overwrite existing teacher cache file")

    monkeypatch.setattr(bootstrap_teacher_script, "_publish_checkpoint", inject_race)
    with pytest.raises(TeacherRegistryError, match="refusing to overwrite"):
        bootstrap_teacher(root=tmp_path, registry_id="chen2021_ltd_wrn34_10", source=source, update_lock=True)
    assert destination.read_bytes() == b"other process"
    raw = yaml.safe_load((tmp_path / "teachers.lock.yaml").read_text(encoding="utf-8"))
    assert raw["teachers"]["chen2021_ltd_wrn34_10"]["checkpoint_status"] == "missing"
    destination.unlink()
    monkeypatch.setattr(bootstrap_teacher_script, "_publish_checkpoint", original_publish_checkpoint)
    retried = bootstrap_teacher(
        root=tmp_path, registry_id="chen2021_ltd_wrn34_10", source=source, update_lock=True
    )
    assert retried == destination
