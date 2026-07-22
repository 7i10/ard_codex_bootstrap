#!/usr/bin/env python3
"""Verify that the local SAAD checkout exactly matches external.lock.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

from external_common import ExternalError, license_evidence, load_lock, repository_root, validate_checkout


def verify(*, root: Path) -> dict[str, object]:
    _, locked = load_lock(root / "external.lock.yaml")
    checkout = root / ".external" / locked.name
    validate_checkout(checkout, locked)
    observed_path, observed_evidence = license_evidence(checkout)
    if observed_path != locked.license_file or observed_evidence != locked.license_evidence:
        raise ExternalError(
            f"{locked.license_status} license evidence does not match checkout: "
            f"locked file={locked.license_file!r}, observed file={observed_path!r}"
        )
    return {
        "name": locked.name,
        "commit": locked.commit,
        "license_file": observed_path,
        "license_evidence": observed_evidence,
        "license_status": locked.license_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root())
    args = parser.parse_args()
    try:
        report = verify(root=args.root.resolve())
    except ExternalError as exc:
        parser.error(str(exc))
    print(" ".join(f"{key}={value}" for key, value in report.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
