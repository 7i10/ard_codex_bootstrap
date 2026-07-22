#!/usr/bin/env python3
"""Clone one or all locked external revisions atomically without touching existing checkouts."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from external_common import (
    ExternalError,
    LockedRepository,
    git,
    license_evidence,
    repository_root,
    select_repositories,
    validate_checkout,
    write_lock,
)


def bootstrap(
    *, root: Path, update_lock: bool = False, repository: str | None = None, all_repositories: bool = False
) -> Path | tuple[Path, ...]:
    lock_path = root / "external.lock.yaml"
    raw, locked_repositories = select_repositories(lock_path, repository=repository, all_repositories=all_repositories)
    destinations = tuple(_bootstrap_one(root=root, locked=locked) for locked in locked_repositories)
    if update_lock:
        for locked, destination in zip(locked_repositories, destinations, strict=True):
            license_path, evidence = license_evidence(destination)
            _record_evidence(raw, locked.name, license_path, evidence)
        write_lock(lock_path, raw)
    return destinations if all_repositories else destinations[0]


def _bootstrap_one(*, root: Path, locked: LockedRepository) -> Path:
    # Kept separate from selector handling so every repository follows exactly
    # the established clone/remote/SHA/dirty safety path.
    destination = root / ".external" / locked.name
    if destination.exists():
        validate_checkout(destination, locked)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{locked.name}.clone-", dir=destination.parent))
    checkout = staging / locked.name
    try:
        git(["clone", "--no-checkout", locked.url, str(checkout)])
        # Fetch the lock explicitly: a default branch clone need not contain an old locked commit.
        git(["fetch", "--depth", "1", "origin", locked.commit], cwd=checkout)
        git(["checkout", "--detach", locked.commit], cwd=checkout)
        validate_checkout(checkout, locked)
        checkout.rename(destination)
    except Exception:
        # The destination was never created, so a partial clone cannot look successful.
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def _record_evidence(raw: dict, repository: str, license_path: str | None, evidence: dict | None) -> None:
    entry = raw["repositories"][repository]
    preserve_verified = (
        entry.get("license_status") == "verified"
        and entry.get("license_file") == license_path
        and entry.get("license_evidence") == evidence
    )
    entry["fetched_at"] = datetime.now(UTC).isoformat()
    entry["license_file"] = license_path
    # Preserve a manually reviewed license identification only while the
    # observed file and digest still match. New evidence remains unclassified.
    entry["license_status"] = "verified" if preserve_verified else "unclear" if license_path else "absent"
    entry["license_evidence"] = evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root())
    parser.add_argument("--update-lock", action="store_true", help="explicitly record observed license evidence")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--repository", help="locked repository name (default: saad)")
    selection.add_argument("--all", action="store_true", help="bootstrap every locked repository by name")
    args = parser.parse_args()
    try:
        paths = bootstrap(
            root=args.root.resolve(),
            update_lock=args.update_lock,
            repository=args.repository,
            all_repositories=args.all,
        )
    except (ExternalError, ValueError) as exc:
        parser.error(str(exc))
    if isinstance(paths, tuple):
        print("\n".join(str(path) for path in paths))
    else:
        print(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
