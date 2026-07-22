#!/usr/bin/env python3
"""Launch pinned full SAAD only as an isolated upstream subprocess.

This is deliberately not a clean-room method implementation: it never imports
from ``.external`` and it refuses an unverified or dirty clone.  AutoAttack in
the upstream process remains upstream behavior; it is not part of this test or
training path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _git(directory: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=directory,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "failed to inspect upstream clone")
    return completed.stdout.strip()


def verified_saad_clone(root: Path = ROOT) -> Path:
    lock = yaml.safe_load((root / "external.lock.yaml").read_text(encoding="utf-8"))
    expected = lock["repositories"]["saad"]
    clone = root / ".external" / "saad"
    if not clone.is_dir():
        raise FileNotFoundError("pinned SAAD clone is absent; run scripts/bootstrap_external.py first")
    if _git(clone, "remote", "get-url", "origin") != expected["url"]:
        raise RuntimeError("SAAD origin differs from external.lock.yaml")
    if _git(clone, "rev-parse", "HEAD") != expected["commit"]:
        raise RuntimeError("SAAD commit differs from external.lock.yaml")
    if _git(clone, "status", "--porcelain"):
        raise RuntimeError("refusing to launch a dirty SAAD clone")
    return clone


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="verify clone and print, without execution")
    parser.add_argument("--execute", action="store_true", help="run upstream saad.py as a subprocess")
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER, help="arguments forwarded unchanged to saad.py")
    args = parser.parse_args(argv)
    if args.dry_run == args.execute:
        parser.error("select exactly one of --dry-run or --execute")
    clone = verified_saad_clone()
    command = [sys.executable, "saad.py", *args.upstream_args]
    print("verified_saad_commit=" + _git(clone, "rev-parse", "HEAD"))
    print("subprocess=" + " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=clone).returncode


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
