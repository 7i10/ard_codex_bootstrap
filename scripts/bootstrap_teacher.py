#!/usr/bin/env python3
"""Register an explicitly supplied local RobustBench checkpoint; never download weights."""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ard.models.teacher_registry import TeacherRegistry, TeacherRegistryError, sha256_file  # noqa: E402


def bootstrap(
    *,
    root: Path,
    registry_id: str,
    source: Path,
    update_lock: bool = False,
    install_locked: bool = False,
) -> Path:
    """Atomically copy a user-provided file into its pinned local cache location.

    Exactly one explicit mode is required. ``--update-lock`` establishes the
    first project-owned SHA from a missing lock entry. ``--install-locked``
    restores bytes already identified by a verified lock and never rewrites
    the lock. Existing files are never overwritten; a failed lock publication
    removes only the inode created by this invocation.
    """
    if update_lock == install_locked:
        raise TeacherRegistryError(
            "teacher bootstrap requires --update-lock or --install-locked, but not both; "
            "unregistered cache files are unusable"
        )
    with _project_lock(root):
        registry = TeacherRegistry.load(root)
        spec = registry.spec(registry_id)
        registry.validate_external()
        if not source.is_file():
            raise TeacherRegistryError(f"explicit teacher source file is missing: {source}")
        destination = registry.checkpoint_path(spec)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lock_path = root / "teachers.lock.yaml"
        fd, temporary_name = tempfile.mkstemp(prefix=f".{spec.checkpoint_filename}.", dir=destination.parent)
        temporary = Path(temporary_name)
        lock_temporary: Path | None = None
        published = False
        try:
            if destination.exists():
                raise TeacherRegistryError(f"refusing to overwrite existing teacher cache file: {destination}")
            with source.open("rb") as input_handle, os.fdopen(fd, "wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            observed = sha256_file(temporary)
            if install_locked:
                expected = _require_verified_checkpoint_lock(lock_path, registry_id, spec.checkpoint_sha256)
                if observed != expected:
                    raise TeacherRegistryError(f"teacher checkpoint hash mismatch: expected {expected}, got {observed}")
            else:
                lock_temporary = _prepare_checkpoint_lock(lock_path, registry_id, observed)
            _publish_checkpoint(temporary, destination)
            published = True
            if lock_temporary is not None:
                _publish_lock(lock_temporary, lock_path)
        except Exception:
            if published:
                _remove_our_published_checkpoint(destination, temporary)
            raise
        finally:
            temporary.unlink(missing_ok=True)
            if lock_temporary is not None:
                lock_temporary.unlink(missing_ok=True)
    return destination


def _project_lock_path(root: Path) -> Path:
    return root / ".cache" / "teacher-bootstrap.lock"


@contextmanager
def _project_lock(root: Path):
    path = _project_lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _prepare_checkpoint_lock(lock_path: Path, registry_id: str, digest: str) -> Path:
    raw = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("teachers"), dict):
        raise TeacherRegistryError("teacher lock cannot be updated because its schema is invalid")
    entry = raw["teachers"].get(registry_id)
    if not isinstance(entry, dict):
        raise TeacherRegistryError(f"teacher lock has no registry ID {registry_id!r}")
    if entry.get("checkpoint_status") != "missing" or entry.get("checkpoint_sha256") is not None:
        raise TeacherRegistryError("teacher lock checkpoint SHA is already registered; refusing to advance it")
    entry["checkpoint_sha256"] = digest
    entry["checkpoint_status"] = "verified"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{lock_path.name}.", dir=lock_path.parent)
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _require_verified_checkpoint_lock(lock_path: Path, registry_id: str, registry_sha: str | None) -> str:
    """Return the immutable SHA for an install-only restore.

    The raw lock is checked in addition to the loaded registry so a stale or
    inconsistent registry object cannot turn a restore into an identity
    change. This function is deliberately read-only: install mode must retain
    ``teachers.lock.yaml`` byte-for-byte.
    """
    raw = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("teachers"), dict):
        raise TeacherRegistryError("teacher lock cannot be used for install because its schema is invalid")
    entry = raw["teachers"].get(registry_id)
    if not isinstance(entry, dict):
        raise TeacherRegistryError(f"teacher lock has no registry ID {registry_id!r}")
    digest = entry.get("checkpoint_sha256")
    if (
        entry.get("checkpoint_status") != "verified"
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or registry_sha != digest
    ):
        raise TeacherRegistryError(
            "teacher lock is not a consistent verified checkpoint identity; --install-locked is unavailable"
        )
    return digest


def _publish_checkpoint(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise TeacherRegistryError(f"refusing to overwrite existing teacher cache file: {destination}") from exc


def _publish_lock(temporary: Path, lock_path: Path) -> None:
    os.replace(temporary, lock_path)


def _remove_our_published_checkpoint(destination: Path, temporary: Path) -> None:
    try:
        destination_stat = destination.stat()
        temporary_stat = temporary.stat()
    except FileNotFoundError:
        return
    if (destination_stat.st_dev, destination_stat.st_ino) == (temporary_stat.st_dev, temporary_stat.st_ino):
        destination.unlink()


def _record_checkpoint_sha(lock_path: Path, registry_id: str, digest: str) -> None:
    """Compatibility helper for isolated tests; bootstrap uses the transactional path."""
    temporary = _prepare_checkpoint_lock(lock_path, registry_id, digest)
    try:
        _publish_lock(temporary, lock_path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registry-id", required=True)
    parser.add_argument("--source", type=Path, required=True, help="existing local checkpoint file; no URL is accepted")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--update-lock",
        action="store_true",
        help="establish a missing checkpoint SHA and atomically publish the file",
    )
    mode.add_argument(
        "--install-locked",
        action="store_true",
        help="restore bytes matching an existing verified SHA without changing teachers.lock.yaml",
    )
    args = parser.parse_args()
    try:
        destination = bootstrap(
            root=args.root.resolve(),
            registry_id=args.registry_id,
            source=args.source.resolve(),
            update_lock=args.update_lock,
            install_locked=args.install_locked,
        )
        print(destination)
    except TeacherRegistryError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
