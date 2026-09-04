#!/usr/bin/env python3
"""Publish one idempotent terminal-result pointer on the event branch.

The publisher is deliberately result-agnostic.  It validates a canonical
terminal manifest, writes only a compact pointer under
``automation/experiment-events/``, and optionally pushes the dedicated
``experiment-results`` branch.  It never edits the master worktree or merges a
pull request.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ALLOWED_TERMINAL_STATES = {"SUCCESS", "AWAITING_RESEARCH_REVIEW", "NEEDS_RESEARCH_DECISION", "NEEDS_TECHNICAL_RECOVERY"}
SHA40 = re.compile(r"^[0-9a-fA-F]{40}$")
SHA64 = re.compile(r"^[0-9a-fA-F]{64}$")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
EVENT_BRANCH = "experiment-results"
EVENT_TITLE = "Automated Experiment Results"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git(cwd: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def load_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid terminal result manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("terminal result manifest must be a JSON object")
    state = value.get("terminal_state")
    if state not in ALLOWED_TERMINAL_STATES:
        raise ValueError(f"terminal_state is not publishable: {state}")
    for field in ("result_id", "result_revision", "canonical_commit_sha", "source_sha", "result_manifest", "report"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"terminal result missing {field}")
    if not SHA40.fullmatch(value["canonical_commit_sha"]) or not SHA40.fullmatch(value["source_sha"]):
        raise ValueError("canonical_commit_sha and source_sha must be full 40-hex SHAs")
    for field in ("result_id", "result_revision"):
        if not SAFE_COMPONENT.fullmatch(value[field]):
            raise ValueError(f"{field} must be a single safe path component")
    digest = value.get("artifact_digest")
    if not isinstance(digest, str) or not SHA64.fullmatch(digest):
        raise ValueError("artifact_digest must be a 64-hex SHA-256")
    return value


class PublisherLock:
    """Serialize check/write/commit so concurrent scheduled wakes are idempotent."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any | None = None

    def __enter__(self) -> PublisherLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: object) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def validate_references(result: dict[str, Any], base: Path) -> None:
    for key in ("result_manifest", "report"):
        path = Path(result[key]).expanduser()
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            raise ValueError(f"{key} does not exist: {path}")
        expected = result.get(f"{key}_sha256")
        if expected is not None:
            if not SHA64.fullmatch(str(expected)) or sha256_bytes(path.read_bytes()) != str(expected).lower():
                raise ValueError(f"{key}_sha256 does not match referenced bytes")


def validate_canonical_fields(result: dict[str, Any], *, required: bool) -> None:
    if not required:
        return
    required_fields = {
        "result_manifest_path": "result_manifest_sha256",
        "report_path": "report_sha256",
    }
    for path_key, digest_key in required_fields.items():
        value = result.get(path_key) or result.get(f"repository_relative_{path_key[:-5]}")
        digest_value = result.get(digest_key)
        if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(f"{path_key} must be a repository-relative path for remote publication")
        if not isinstance(digest_value, str) or not SHA64.fullmatch(digest_value):
            raise ValueError(f"{digest_key} must be a SHA-256 for remote publication")
    failure_path = result.get("failure_report_path") or result.get("repository_relative_failure_report")
    if failure_path is not None and (Path(str(failure_path)).is_absolute() or ".." in Path(str(failure_path)).parts):
        raise ValueError("failure_report_path must be repository-relative")


def _remote_commit_reachable(worktree: Path, commit: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, f"origin/{EVENT_BRANCH}"],
        cwd=worktree,
        capture_output=True,
    ).returncode == 0


def _verify_canonical_remote(result: dict[str, Any], worktree: Path, *, required: bool) -> None:
    """Verify canonical result commit and declared repository-relative blobs.

    Local unit fixtures may publish without a remote. Production pushes must
    have an origin/master containing the canonical commit and matching blobs.
    """
    canonical_commit = result["canonical_commit_sha"]
    if not required:
        return
    if subprocess.run(["git", "cat-file", "-e", f"{canonical_commit}^{{commit}}"], cwd=worktree, capture_output=True).returncode != 0:
        raise ValueError("canonical commit is not present locally")
    git(worktree, "fetch", "origin", "--prune")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", canonical_commit, "origin/master"], cwd=worktree, capture_output=True
    ).returncode != 0:
        raise ValueError("canonical commit is not reachable from origin/master")
    for key in ("result_manifest_path", "report_path", "failure_report_path"):
        relative = result.get(key) or result.get(f"repository_relative_{key[:-5]}")
        if relative is None:
            continue
        candidate = Path(str(relative))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"{key} must be repository-relative")
        if subprocess.run(
            ["git", "cat-file", "-e", f"{canonical_commit}:{candidate.as_posix()}"], cwd=worktree, capture_output=True
        ).returncode != 0:
            raise ValueError(f"{key} is missing from canonical commit")
        blob = git(worktree, "show", f"{canonical_commit}:{candidate.as_posix()}", check=False)
        if blob == "":
            raise ValueError(f"{key} is missing from canonical commit")
        expected = result.get(f"{key}_sha256")
        if expected and sha256_bytes(blob.encode()) != str(expected).lower():
            raise ValueError(f"{key} digest does not match canonical commit")


def event_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "result_id": result["result_id"],
        "result_revision": result["result_revision"],
        "terminal_state": result["terminal_state"],
        "canonical_commit_sha": result["canonical_commit_sha"],
        "source_sha": result["source_sha"],
        "scientific_identity_hash": result.get("scientific_identity_hash"),
        "result_manifest": result["result_manifest"],
        "report": result["report"],
        "failure_report": result.get("failure_report"),
        "artifact_digest": result["artifact_digest"],
        "result_manifest_path": result.get("result_manifest_path") or result.get("repository_relative_result_manifest"),
        "report_path": result.get("report_path") or result.get("repository_relative_report"),
        "failure_report_path": result.get("failure_report_path") or result.get("repository_relative_failure_report"),
        "result_manifest_sha256": result.get("result_manifest_sha256"),
        "report_sha256": result.get("report_sha256"),
        "failure_report_sha256": result.get("failure_report_sha256"),
        "published_at": result.get("published_at") or dt.datetime.now(dt.UTC).isoformat(),
    }
    return payload


def _payload_identity(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "published_at"}


def _ensure_remote_branch(worktree: Path, commit: str, *, push: bool) -> None:
    if not push:
        return
    if _remote_commit_reachable(worktree, commit):
        return
    git(worktree, "push", "origin", f"{commit}:refs/heads/{EVENT_BRANCH}")
    git(worktree, "fetch", "origin", "--prune")
    if not _remote_commit_reachable(worktree, commit):
        raise ValueError("event commit push did not make the remote branch reachable")


def publish(result_path: Path, *, worktree: Path, push: bool, ensure_pr: bool) -> dict[str, Any]:
    result = load_result(result_path)
    validate_references(result, result_path.parent)
    validate_canonical_fields(result, required=push)
    worktree = worktree.resolve()
    if not (worktree / ".git").exists() and git(worktree, "rev-parse", "--is-inside-work-tree", check=False) != "true":
        raise ValueError(f"publisher worktree is not a Git worktree: {worktree}")
    branch = git(worktree, "branch", "--show-current")
    if branch != EVENT_BRANCH:
        raise ValueError(f"publisher worktree must be on {EVENT_BRANCH}, observed {branch or 'detached'}")
    git_dir = Path(git(worktree, "rev-parse", "--git-dir"))
    if not git_dir.is_absolute():
        git_dir = (worktree / git_dir).resolve()
    with PublisherLock(git_dir / "experiment-results.publisher.lock"):
        if git(worktree, "status", "--porcelain"):
            raise ValueError("publisher worktree is dirty before event publication")
        relative_result = Path(result["result_id"]) / f"{result['result_revision']}.json"
        event_path = worktree / "automation" / "experiment-events" / relative_result
        payload = event_payload(result)
        _verify_canonical_remote(result, worktree, required=push)
        if event_path.exists():
            existing = json.loads(event_path.read_text(encoding="utf-8"))
            if _payload_identity(existing) != _payload_identity(payload):
                raise ValueError(f"RESULT_REVISION_COLLISION: event path has different payload: {event_path}")
            commit = git(worktree, "log", "-1", "--format=%H", "--", str(event_path.relative_to(worktree)), check=False)
            if not commit:
                raise ValueError("published event exists but no local commit contains it")
            _ensure_remote_branch(worktree, commit, push=push)
            return {"status": "NO_OP" if _remote_commit_reachable(worktree, commit) or not push else "RESUMED", "reason": "event_already_published", "event_path": str(event_path), "commit": commit}
        atomic_json(event_path, payload)
        git(worktree, "add", str(event_path.relative_to(worktree)))
        git(worktree, "commit", "-m", f"Publish terminal result event {result['result_id']}@{result['result_revision']}")
        commit = git(worktree, "rev-parse", "HEAD")
        _ensure_remote_branch(worktree, commit, push=push)
        pr = None
        if ensure_pr:
            existing = subprocess.run(
                ["gh", "pr", "list", "--base", "master", "--head", EVENT_BRANCH, "--state", "open", "--json", "number"],
                cwd=worktree,
                text=True,
                capture_output=True,
            )
            if existing.returncode != 0:
                raise RuntimeError(existing.stderr.strip() or "gh pr list failed")
            rows = json.loads(existing.stdout or "[]")
            if rows:
                pr = rows[0].get("number")
            else:
                created = subprocess.run(
                    [
                        "gh",
                        "pr",
                        "create",
                        "--base",
                        "master",
                        "--head",
                        EVENT_BRANCH,
                        "--title",
                        EVENT_TITLE,
                        "--body",
                        "Terminal experiment result notification bus; canonical results remain in registered commits.",
                    ],
                    cwd=worktree,
                    text=True,
                    capture_output=True,
                )
                if created.returncode != 0:
                    raise RuntimeError(created.stderr.strip() or "gh pr create failed")
                pr = created.stdout.strip()
        return {"status": "PUBLISHED", "event_path": str(event_path), "commit": commit, "pull_request": pr}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-manifest", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--ensure-pr", action="store_true")
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                publish(
                    args.result_manifest.resolve(),
                    worktree=args.worktree,
                    push=args.push,
                    ensure_pr=args.ensure_pr,
                ),
                sort_keys=True,
            )
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
