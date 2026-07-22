#!/usr/bin/env python3
"""Clone the locked SAAD revision atomically, without touching an existing checkout."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from external_common import (
    ExternalError,
    git,
    license_evidence,
    load_lock,
    repository_root,
    validate_checkout,
    write_lock,
)


def bootstrap(*, root: Path, update_lock: bool = False) -> Path:
    lock_path = root / "external.lock.yaml"
    raw, locked = load_lock(lock_path)
    destination = root / ".external" / locked.name
    if destination.exists():
        validate_checkout(destination, locked)
        license_path, evidence = license_evidence(destination)
        _record_evidence(raw, lock_path, license_path, evidence, update_lock)
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
        license_path, evidence = license_evidence(checkout)
        checkout.rename(destination)
        _record_evidence(raw, lock_path, license_path, evidence, update_lock)
    except Exception:
        # The destination was never created, so a partial clone cannot look successful.
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def _record_evidence(
    raw: dict, lock_path: Path, license_path: str | None, evidence: dict | None, update_lock: bool
) -> None:
    if not update_lock:
        return
    entry = raw["repositories"]["saad"]
    entry["fetched_at"] = datetime.now(UTC).isoformat()
    entry["license_file"] = license_path
    # Presence is evidence, not legal verification.  Do not overstate it in the lock.
    entry["license_status"] = "unclear" if license_path else "absent"
    entry["license_evidence"] = evidence
    write_lock(lock_path, raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root())
    parser.add_argument("--update-lock", action="store_true", help="explicitly record observed license evidence")
    args = parser.parse_args()
    try:
        path = bootstrap(root=args.root.resolve(), update_lock=args.update_lock)
    except ExternalError as exc:
        parser.error(str(exc))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
