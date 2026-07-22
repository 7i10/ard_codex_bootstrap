#!/usr/bin/env python3
"""Verify one or all local external checkouts against external.lock.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

from external_common import (
    ExternalError,
    LockedRepository,
    license_evidence,
    repository_root,
    select_repositories,
    validate_checkout,
)


def verify(
    *, root: Path, repository: str | None = None, all_repositories: bool = False
) -> dict[str, object] | tuple[dict[str, object], ...]:
    _, repositories = select_repositories(
        root / "external.lock.yaml", repository=repository, all_repositories=all_repositories
    )
    reports = tuple(_verify_one(root=root, locked=locked) for locked in repositories)
    return reports if all_repositories else reports[0]


def _verify_one(*, root: Path, locked: LockedRepository) -> dict[str, object]:
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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--repository", help="locked repository name (default: saad)")
    selection.add_argument("--all", action="store_true", help="verify every locked repository by name")
    args = parser.parse_args()
    try:
        reports = verify(root=args.root.resolve(), repository=args.repository, all_repositories=args.all)
    except (ExternalError, ValueError) as exc:
        parser.error(str(exc))
    if isinstance(reports, tuple):
        print("\n".join(" ".join(f"{key}={value}" for key, value in report.items()) for report in reports))
    else:
        print(" ".join(f"{key}={value}" for key, value in reports.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
