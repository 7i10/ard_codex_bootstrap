#!/usr/bin/env python3
"""Create or reuse the pinned worktree a GPU job must run from.

``<worktree_root>/source-<sha12>`` is a detached worktree of the repository at
SHA with ``.external`` and ``teacher_cache`` symlinked to the live checkout
(both are gitignored, so the worktree stays clean).  Idempotent.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ardx_common import add_registry_argument, load_registry, registry_path  # noqa: E402

SHARED_ASSETS = (".external", "teacher_cache")


class PinError(Exception):
    """Operator-visible failure: dirty tree, HEAD mismatch or unknown sha."""


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PinError(f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def resolve_sha(repo: Path, sha: str) -> str:
    try:
        resolved = git(repo, "rev-parse", "--verify", f"{sha}^{{commit}}")
    except PinError as exc:
        raise PinError(f"unknown sha {sha!r} in {repo}: {exc}") from exc
    if len(resolved) != 40:
        raise PinError(f"unexpected rev-parse output for {sha!r}: {resolved!r}")
    return resolved.lower()


def link_shared_assets(repo: Path, worktree: Path) -> list[str]:
    linked: list[str] = []
    for name in SHARED_ASSETS:
        target = repo / name
        link = worktree / name
        if link.is_symlink() or link.exists():
            continue
        if not target.exists():
            raise PinError(f"required shared runtime asset is missing: {target}")
        os.symlink(target, link)
        linked.append(name)
    return linked


def pin(sha: str, registry_file: Path) -> dict[str, object]:
    registry = load_registry(registry_file)
    repo = registry_path(registry, "repo_root")
    worktree_root = registry_path(registry, "worktree_root")
    if not (repo / ".git").exists():
        raise PinError(f"repo_root is not a git checkout: {repo}")
    resolved = resolve_sha(repo, sha)
    worktree = worktree_root / f"source-{resolved[:12]}"

    if worktree.exists():
        status = "reused"
    else:
        worktree_root.mkdir(parents=True, exist_ok=True)
        git(repo, "worktree", "add", "--detach", str(worktree), resolved)
        status = "created"

    head = git(worktree, "rev-parse", "HEAD").lower()
    if head != resolved:
        raise PinError(f"worktree {worktree} is at {head}, expected {resolved}")
    linked = link_shared_assets(repo, worktree)
    dirty = git(worktree, "status", "--porcelain")
    if dirty:
        raise PinError(f"worktree {worktree} is not clean:\n{dirty}")
    return {"path": str(worktree), "sha": resolved, "status": status, "linked": linked}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ensure a pinned detached worktree exists for SHA.")
    parser.add_argument("sha", metavar="SHA", help="commit to pin (any git revision that names a commit)")
    add_registry_argument(parser)
    parser.add_argument("--json", action="store_true", help="explicit JSON output (JSON is the default)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = pin(args.sha, args.registry)
    except (PinError, OSError, ValueError) as exc:
        print(f"pin_source: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in ("path", "sha", "status")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
